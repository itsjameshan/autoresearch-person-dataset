"""
bilevel_runner.py — 双层自主优化智能体 (Bilevel Autoresearch for YOLO)

借鉴 Bilevel-Autoresearch (https://github.com/EdwardOptimization/Bilevel-Autoresearch)
的双层优化思想，应用于 YOLO 密集人群检测模型训练。

三层架构：
  Level 1   (内循环)    : LLM 提出超参修改 → 训练 → 评估 CDS → keep/discard
  Level 1.5 (外循环配置) : 分析 trace → 冻结无效参数 → 调整搜索策略
  Level 2   (机制发现)   : 内置多种搜索机制 (Tabu/ElitePool/SA/Crossover 等)

所有机制代码化实现，无需 Level 2 LLM 写代码（避免本地小模型写代码不稳定）。
"""

import os
import re
import sys
import json
import time
import math
import random
import shutil
import subprocess
import threading
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

try:
    from skopt import Optimizer
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# ============================================================
# 配置
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
USE_LLM = os.environ.get("BILEVEL_USE_LLM", "auto").lower()  # auto, true, false

TRAIN_SCRIPT = "train.py"
RUN_LOG = "bilevel_run.log"
RESULTS_TSV = "bilevel_results.tsv"
STATUS_MD = "bilevel_status.md"
BEST_MODEL_DIR = "bilevel_best_model"
ARTIFACTS_DIR = "bilevel_artifacts"

TRAIN_TIMEOUT = 7200
PYTHON = sys.executable

# 停止条件
MAX_TOTAL_EXPERIMENTS = 100
MAX_NO_IMPROVE_INNER = 8
MAX_OUTER_CYCLES = 10
CDS_IMPROVEMENT_TARGET = 0.05

# 搜索参数定义
SEARCH_PARAMS = {
    "LR0": {"type": "float", "min": 0.00001, "max": 0.01, "default": 0.001},
    "LRF": {"type": "float", "min": 0.0001, "max": 0.5, "default": 0.01},
    "MOSAIC": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
    "MIXUP": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.1},
    "COPY_PASTE": {"type": "float", "min": 0.0, "max": 0.3, "default": 0.1},
    "DEGREES": {"type": "float", "min": 0.0, "max": 20.0, "default": 10.0},
    "TRANSLATE": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.3},
    "SCALE": {"type": "float", "min": 0.2, "max": 0.9, "default": 0.5},
    "ERASING": {"type": "float", "min": 0.0, "max": 0.7, "default": 0.4},
    "BOX": {"type": "float", "min": 5.0, "max": 30.0, "default": 15.0},
    "CLS": {"type": "float", "min": 0.3, "max": 3.0, "default": 1.0},
    "HSV_S": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.7},
    "HSV_V": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.4},
    "FLIPUD": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    "CLOSE_MOSAIC": {"type": "int", "min": 5, "max": 25, "default": 15},
    "PATIENCE": {"type": "int", "min": 5, "max": 50, "default": 30},
}

BASELINE_PARAMS = {k: v["default"] for k, v in SEARCH_PARAMS.items()}


# ============================================================
# Level 2: 搜索机制 (自主发现的机制，代码化实现)
# ============================================================

@dataclass
class ElitePool:
    """精英池: 维护 Top-K 最优配置，帮助识别模式"""
    pool: list = field(default_factory=list)
    k: int = 5

    def add(self, params: dict, cds: float):
        self.pool.append({"params": params.copy(), "cds": cds})
        self.pool.sort(key=lambda x: -x["cds"])
        self.pool = self.pool[:self.k]

    def best(self) -> Optional[dict]:
        return self.pool[0]["params"] if self.pool else None

    def best_cds(self) -> float:
        return self.pool[0]["cds"] if self.pool else 0.0

    def get_crossover_candidate(self) -> Optional[dict]:
        """基因交叉: 从精英池中随机选两个，混合它们的参数"""
        if len(self.pool) < 2:
            return None
        a, b = random.sample(self.pool[:min(3, len(self.pool))], 2)
        child = {}
        for key in SEARCH_PARAMS:
            child[key] = random.choice([a["params"][key], b["params"][key]])
        return child

    def get_param_patterns(self) -> dict:
        """分析精英池中的参数模式，返回每个参数的均值和趋势"""
        patterns = {}
        if not self.pool:
            return patterns
        for key in SEARCH_PARAMS:
            vals = [entry["params"][key] for entry in self.pool]
            patterns[key] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5
            }
        return patterns


@dataclass
class TabuManager:
    """禁忌搜索: 防止重复尝试相似的参数组合"""
    tabu_list: list = field(default_factory=list)
    max_tabu: int = 15
    similarity_threshold: float = 0.9

    def _normalize(self, params: dict) -> dict:
        norm = {}
        for key, spec in SEARCH_PARAMS.items():
            val = params.get(key, spec["default"])
            norm[key] = (val - spec["min"]) / (spec["max"] - spec["min"])
        return norm

    def _similarity(self, p1: dict, p2: dict) -> float:
        n1, n2 = self._normalize(p1), self._normalize(p2)
        diff = sum(abs(n1[k] - n2[k]) for k in SEARCH_PARAMS)
        return 1.0 - diff / len(SEARCH_PARAMS)

    def is_tabu(self, params: dict) -> bool:
        for tabu_entry in self.tabu_list:
            if self._similarity(params, tabu_entry) > self.similarity_threshold:
                return True
        return False

    def add(self, params: dict):
        self.tabu_list.append(params.copy())
        if len(self.tabu_list) > self.max_tabu:
            self.tabu_list.pop(0)


@dataclass
class SimulatedAnnealing:
    """模拟退火: 以一定概率接受较差解，跳出局部最优"""
    temperature: float = 0.05
    cooling_rate: float = 0.95
    min_temperature: float = 0.005

    def should_accept(self, current_cds: float, new_cds: float) -> bool:
        if new_cds >= current_cds:
            return True
        delta = new_cds - current_cds
        if self.temperature <= 0:
            return False
        prob = math.exp(delta / self.temperature)
        return random.random() < prob

    def cool(self):
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)


@dataclass
class MomentumTracker:
    """动量追踪: 跟踪每个参数的变化方向与效果的相关性"""
    param_deltas: dict = field(default_factory=lambda: {})  # key -> list of (delta_param, delta_cds)
    window_size: int = 20

    def record(self, param_name: str, delta_param: float, delta_cds: float):
        if param_name not in self.param_deltas:
            self.param_deltas[param_name] = []
        self.param_deltas[param_name].append((delta_param, delta_cds))
        if len(self.param_deltas[param_name]) > self.window_size:
            self.param_deltas[param_name].pop(0)

    def get_direction_hint(self, param_name: str) -> str:
        """返回参数调整方向建议: 'up', 'down', 或 'unknown'"""
        deltas = self.param_deltas.get(param_name, [])
        if len(deltas) < 3:
            return "unknown"
        up_good = sum(1 for dp, dc in deltas if dp > 0 and dc > 0)
        down_good = sum(1 for dp, dc in deltas if dp < 0 and dc > 0)
        if up_good > down_good * 1.5:
            return "up"
        if down_good > up_good * 1.5:
            return "down"
        return "unknown"

    def get_all_hints(self) -> dict:
        return {k: self.get_direction_hint(k) for k in SEARCH_PARAMS}


@dataclass
class PlateauDetector:
    """平台检测: 检测搜索是否停滞，强制多样化探索"""
    consecutive_same_best: int = 0
    last_best_cds: float = 0.0
    stagnation_threshold: int = 5

    def update(self, best_cds: float):
        if abs(best_cds - self.last_best_cds) < 0.001:
            self.consecutive_same_best += 1
        else:
            self.consecutive_same_best = 0
            self.last_best_cds = best_cds

    def is_stagnant(self) -> bool:
        return self.consecutive_same_best >= self.stagnation_threshold

    def reset(self):
        self.consecutive_same_best = 0


@dataclass
class ExplorationBudget:
    """探索预算: 定期强制探索轮次，避免陷入局部最优"""
    explore_every: int = 4
    counter: int = 0

    def is_exploration_turn(self) -> bool:
        self.counter += 1
        return self.counter % self.explore_every == 0

    def reset(self):
        self.counter = 0


# ============================================================
# Level 1.5: 外循环配置管理
# ============================================================

@dataclass
class SearchConfig:
    """搜索配置: 外循环调整这个来改变内循环的搜索方式"""
    active_params: list = field(default_factory=lambda: list(SEARCH_PARAMS.keys()))
    frozen_params: list = field(default_factory=list)
    strategy: str = "explore"  # explore, exploit, focused
    guidance: str = ""
    inner_budget: int = 8  # 每次外循环内的内循环轮数

    def freeze(self, param: str):
        if param in self.active_params and param not in self.frozen_params:
            self.active_params.remove(param)
            self.frozen_params.append(param)

    def unfreeze(self, param: str):
        if param in self.frozen_params:
            self.frozen_params.remove(param)
            self.active_params.append(param)


# ============================================================
# Ollama LLM 交互
# ============================================================

def query_ollama(prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode()).get("response", "")
    except Exception as e:
        print(f"[ERROR] Ollama query failed: {e}")
        return None


def unload_ollama():
    try:
        payload = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0}).encode()
        urllib.request.urlopen(
            urllib.request.Request(OLLAMA_URL, data=payload), timeout=10
        ).read()
    except:
        pass


# ============================================================
# Git 操作 (状态机)
# ============================================================

def git(*args):
    try:
        r = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, timeout=20,
            encoding="utf-8", errors="replace"
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except:
        return -1, "", ""


def git_commit(msg: str):
    git("add", TRAIN_SCRIPT)
    return git("commit", "-m", msg)


def git_reset_hard():
    return git("reset", "--hard", "HEAD~1")


def git_short_hash() -> str:
    _, out, _ = git("rev-parse", "--short", "HEAD")
    return out


# ============================================================
# 参数读写
# ============================================================

def read_current_params() -> dict:
    params = {}
    try:
        with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()
        for key in SEARCH_PARAMS:
            m = re.search(rf"^{key}\s*=\s*([^\n]+)", content, re.MULTILINE)
            if m:
                try:
                    params[key] = eval(m.group(1).strip())
                except:
                    pass
    except:
        pass
    return params


def apply_params(params: dict):
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        for key, val in params.items():
            if re.match(rf"^{key}\s*=.*", line):
                if isinstance(val, bool):
                    lines[i] = f"{key} = {repr(val)}\n"
                elif isinstance(val, int):
                    lines[i] = f"{key} = {val}\n"
                elif isinstance(val, float):
                    lines[i] = f"{key} = {val}\n"
                else:
                    lines[i] = f"{key} = {repr(val)}\n"

    with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ============================================================
# 训练与评估
# ============================================================

def run_training() -> bool:
    """运行训练，返回是否成功"""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if os.path.exists(RUN_LOG):
        try:
            os.remove(RUN_LOG)
        except:
            pass

    proc = subprocess.Popen(
        [PYTHON, "-u", TRAIN_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True, encoding="utf-8", errors="replace"
    )

    def watchdog():
        time.sleep(TRAIN_TIMEOUT)
        try:
            proc.kill()
        except:
            pass
    threading.Thread(target=watchdog, daemon=True).start()

    with open(RUN_LOG, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
    proc.wait()
    return proc.returncode == 0


def parse_cds_from_log() -> tuple[float, dict]:
    """从训练日志中解析 CDS 和所有指标"""
    try:
        with open(RUN_LOG, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        metrics = {}

        # 优先解析 evaluate.py 的结构化输出 (cds: 0.xxxx)
        for key in ["cds", "mAP50", "mAP50_95", "precision", "recall",
                     "f1_optimal", "small_obj_recall", "counting_acc",
                     "counting_mae", "mean_confidence", "latency_score",
                     "inference_ms", "peak_memory_mb", "epochs_completed"]:
            pattern = rf"^{key}:\s+([\d.eE+-]+)"
            m = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if m:
                try:
                    val_str = m.group(1).strip()
                    if "." in val_str or "e" in val_str.lower():
                        metrics[key] = float(val_str)
                    else:
                        metrics[key] = int(val_str)
                except:
                    pass

        # 备用：从 ultralytics 输出中解析 mAP
        if "mAP50" not in metrics or "recall" not in metrics:
            pattern = re.compile(r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
            matches = pattern.findall(content)
            if matches:
                p, r, m50, m5095 = matches[-1]
                metrics.setdefault("precision", float(p))
                metrics.setdefault("recall", float(r))
                metrics.setdefault("mAP50", float(m50))
                metrics.setdefault("mAP50_95", float(m5095))

        # 如果 CDS 没直接输出，用加权近似
        if "cds" not in metrics or metrics["cds"] == 0.0:
            if metrics.get("mAP50", 0) > 0:
                approx = (0.25 * metrics.get("mAP50", 0)
                        + 0.25 * metrics.get("mAP50_95", 0)
                        + 0.10 * metrics.get("f1_optimal", 0)
                        + 0.15 * metrics.get("counting_acc", 0)
                        + 0.15 * metrics.get("small_obj_recall", 0)
                        + 0.10 * metrics.get("latency_score", 0))
                if approx > 0:
                    metrics["cds"] = approx

        return metrics.get("cds", 0.0), metrics
    except Exception as e:
        print(f"[WARN] parse_cds_from_log failed: {e}")
        return 0.0, {}


# ============================================================
# LLM 可用性检测
# ============================================================

_ollama_available = None

def is_ollama_available() -> bool:
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    if USE_LLM == "false":
        _ollama_available = False
        return False
    if USE_LLM == "true":
        _ollama_available = True
        return True
    try:
        req = urllib.request.Request(OLLAMA_URL.replace("/generate", "/tags"))
        with urllib.request.urlopen(req, timeout=3) as resp:
            _ollama_available = resp.status == 200
    except:
        _ollama_available = False
    return _ollama_available


# ============================================================
# 贝叶斯优化器 (Fallback 方案)
# ============================================================

class BayesianOptimizer:
    """贝叶斯优化 fallback — 当 Ollama 不可用时使用"""

    def __init__(self):
        self.optimizer = None
        self.space = []
        self.param_names = []

    def build_space(self, active_params: list):
        self.param_names = [p for p in active_params if p in SEARCH_PARAMS]
        self.space = []
        for name in self.param_names:
            spec = SEARCH_PARAMS[name]
            if spec["type"] == "float":
                self.space.append(Real(spec["min"], spec["max"], name=name))
            elif spec["type"] == "int":
                self.space.append(Integer(spec["min"], spec["max"], name=name))
        if SKOPT_AVAILABLE and self.space:
            self.optimizer = Optimizer(
                dimensions=self.space,
                base_estimator="GP",
                acq_func="gp_hedge",
                random_state=42,
            )
        else:
            self.optimizer = None

    def ask(self, current_params: dict) -> Optional[dict]:
        if not self.optimizer:
            return None
        try:
            next_point = self.optimizer.ask()
            params = dict(zip(self.param_names, next_point))
            result = {}
            for k, v in params.items():
                spec = SEARCH_PARAMS[k]
                if spec["type"] == "float":
                    result[k] = float(v)
                elif spec["type"] == "int":
                    result[k] = int(v)
            # 只返回与当前值有差异的参数
            changed = {k: v for k, v in result.items()
                      if abs(current_params.get(k, 0) - v) > 1e-8}
            return changed if changed else None
        except Exception as e:
            print(f"[WARN] Bayesian optimizer ask failed: {e}")
            return None

    def tell(self, params: dict, cds: float):
        if not self.optimizer:
            return
        try:
            point = [params.get(name, SEARCH_PARAMS[name]["default"])
                     for name in self.param_names]
            self.optimizer.tell(point, -cds)  # 最小化负CDS = 最大化CDS
        except Exception as e:
            print(f"[WARN] Bayesian optimizer tell failed: {e}")


# ============================================================
# LLM 提案生成 (Level 1)
# ============================================================

def generate_proposal(
    current_params: dict,
    best_params: dict,
    best_cds: float,
    config: SearchConfig,
    elite_pool: ElitePool,
    momentum: MomentumTracker,
    is_exploration: bool = False,
) -> Optional[dict]:
    """让 LLM 生成下一个超参提案"""

    active_params_str = ", ".join(config.active_params)
    frozen_params_str = ", ".join(config.frozen_params) if config.frozen_params else "none"
    direction_hints = momentum.get_all_hints()
    hints_str = "\n".join(
        f"  - {p}: {d}" for p, d in direction_hints.items() if d != "unknown"
    ) or "  (not enough data yet)"

    elite_patterns = elite_pool.get_param_patterns()
    elite_summary = ""
    if elite_patterns:
        elite_summary = "Elite pool parameter patterns (top performers):\n"
        for p, info in list(elite_patterns.items())[:8]:
            elite_summary += f"  - {p}: mean={info['mean']:.4f}, range=[{info['min']:.4f}, {info['max']:.4f}]\n"

    exploration_note = ""
    if is_exploration:
        exploration_note = """
⚠️ EXPLORATION ROUND — Be BOLD and CREATIVE!
Try a completely different parameter region or parameter combination.
Don't just tweak — make a significant change. Explore under-explored areas.
"""

    prompt = f"""You are a YOLO hyperparameter optimization expert.
Your goal is to maximize CDS (Crowd Detection Score) for dense crowd detection.

Current best CDS: {best_cds:.4f}
Current strategy: {config.strategy}

Active parameters you can modify: {active_params_str}
Frozen parameters (DO NOT touch): {frozen_params_str}

Current parameters:
{json.dumps(current_params, indent=2, default=str)}

Best parameters so far:
{json.dumps(best_params, indent=2, default=str)}

Momentum direction hints (which parameter directions have been working):
{hints_str}

{elite_summary}
{exploration_note}
{config.guidance}

## Task
Propose a NEW set of hyperparameters to try next.
Focus on the ACTIVE parameters only. Do NOT change frozen parameters.

Return ONLY a JSON object with parameter names as keys and new values as values.
Only include parameters you are CHANGING (not all of them).

Example:
{{
  "LR0": 0.0015,
  "MOSAIC": 0.6,
  "BOX": 12.0
}}

Be strategic. Think about what parameter changes are most likely to improve CDS.
"""

    response = query_ollama(prompt, temperature=0.8 if is_exploration else 0.6)
    if not response:
        return None

    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            params = json.loads(json_match.group())
            validated = {}
            for key, val in params.items():
                if key not in config.active_params:
                    continue
                if key not in SEARCH_PARAMS:
                    continue
                spec = SEARCH_PARAMS[key]
                if spec["type"] == "float":
                    val = float(val)
                    val = max(spec["min"], min(spec["max"], val))
                elif spec["type"] == "int":
                    val = int(val)
                    val = max(spec["min"], min(spec["max"], val))
                validated[key] = val
            return validated if validated else None
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response: {e}")
    return None


# ============================================================
# Level 1.5: 外循环分析
# ============================================================

def run_outer_analysis(
    history: list,
    config: SearchConfig,
    elite_pool: ElitePool,
) -> SearchConfig:
    """外循环: 分析内循环历史，调整搜索配置"""

    if len(history) < 5:
        return config

    # 统计每个参数被修改时的平均CDS变化
    param_effect = {}  # param -> list of delta_cds
    for i in range(1, len(history)):
        prev = history[i-1]["params"]
        curr = history[i]["params"]
        delta_cds = history[i]["cds"] - history[i-1]["cds"]
        for key in SEARCH_PARAMS:
            if key not in config.active_params:
                continue
            if abs(curr.get(key, 0) - prev.get(key, 0)) > 1e-8:
                if key not in param_effect:
                    param_effect[key] = []
                param_effect[key].append(delta_cds)

    # 找出无效参数 (平均效果接近0或负面)
    ineffective = []
    for param, deltas in param_effect.items():
        if len(deltas) >= 3:
            avg_effect = sum(deltas) / len(deltas)
            if avg_effect <= 0.001:  # 几乎没帮助
                ineffective.append(param)

    # 找出高效参数
    effective = []
    for param, deltas in param_effect.items():
        if len(deltas) >= 3:
            avg_effect = sum(deltas) / len(deltas)
            if avg_effect > 0.005:
                effective.append(param)

    # 更新配置
    new_config = SearchConfig(
        active_params=config.active_params.copy(),
        frozen_params=config.frozen_params.copy(),
        strategy=config.strategy,
        guidance=config.guidance,
        inner_budget=config.inner_budget,
    )

    # 冻结最多 3 个无效参数（但至少保留 8 个活跃参数）
    frozen_count = 0
    for param in ineffective:
        if len(new_config.active_params) <= 8:
            break
        if frozen_count >= 3:
            break
        new_config.freeze(param)
        frozen_count += 1

    # 调整策略
    if elite_pool.best_cds() > 0 and len(history) > 15:
        recent_best = max(h["cds"] for h in history[-10:])
        overall_best = elite_pool.best_cds()
        if recent_best < overall_best - 0.005:
            new_config.strategy = "explore"
            new_config.guidance = "Recent performance is declining. Be more exploratory."
        else:
            new_config.strategy = "exploit"
            new_config.guidance = "Good progress! Focus on fine-tuning the most effective parameters."

    if effective:
        new_config.guidance += f"\nMost effective parameters: {', '.join(effective)}"

    return new_config


# ============================================================
# 状态与日志
# ============================================================

def init_results_file():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", encoding="utf-8") as f:
            f.write("exp\touter_cycle\tinner_iter\tcds\tmAP50\tmAP50_95\trecall\tprecision\tstatus\tparams\n")


def append_result(exp_num, outer_cycle, inner_iter, cds, metrics, status, params):
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        m50 = metrics.get("mAP50", 0)
        m5095 = metrics.get("mAP50_95", 0)
        r = metrics.get("recall", 0)
        p = metrics.get("precision", 0)
        params_str = json.dumps(params, ensure_ascii=False)
        f.write(f"{exp_num}\t{outer_cycle}\t{inner_iter}\t{cds:.4f}\t{m50:.4f}\t{m5095:.4f}\t{r:.4f}\t{p:.4f}\t{status}\t{params_str}\n")


def update_status(
    outer_cycle: int,
    inner_iter: int,
    total_exp: int,
    best_cds: float,
    best_params: dict,
    config: SearchConfig,
    elite_pool: ElitePool,
    status: str = "running",
):
    lines = [
        "# Bilevel Autoresearch — 双层自主优化状态面板",
        "",
        f"**状态**: {status}",
        f"**外循环周期**: {outer_cycle}",
        f"**当前内循环迭代**: {inner_iter}",
        f"**总实验数**: {total_exp}",
        "",
        "## 当前最佳",
        f"- **CDS**: {best_cds:.4f}",
        f"- **策略**: {config.strategy}",
        f"- **活跃参数**: {len(config.active_params)} 个",
        f"- **冻结参数**: {len(config.frozen_params)} 个: {', '.join(config.frozen_params) if config.frozen_params else 'none'}",
        "",
        "## 最佳参数",
        "```json",
        json.dumps(best_params, indent=2, default=str),
        "```",
        "",
        "## 精英池 (Top-5)",
    ]
    for i, entry in enumerate(elite_pool.pool):
        lines.append(f"{i+1}. CDS={entry['cds']:.4f}")
    lines.append("")
    lines.append(f"_最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}_")

    with open(STATUS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_best_model():
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    src = "autoresearch_runs/current/weights/best.pt"
    if os.path.exists(src):
        shutil.copy(src, os.path.join(BEST_MODEL_DIR, "best.pt"))
    last = "autoresearch_runs/current/weights/last.pt"
    if os.path.exists(last):
        shutil.copy(last, os.path.join(BEST_MODEL_DIR, "last.pt"))


# ============================================================
# 主循环: 双层自主优化
# ============================================================

def main():
    print("=" * 70)
    print("  Bilevel Autoresearch — YOLO 双层自主优化智能体")
    print("  Level 1: 超参优化  |  Level 1.5: 配置调整  |  Level 2: 搜索机制")
    print("=" * 70)

    # 检测可用的优化器
    use_llm = is_ollama_available()
    use_bayesian = SKOPT_AVAILABLE
    print(f"LLM (Ollama): {'✅ 可用' if use_llm else '❌ 不可用'}")
    print(f"Bayesian (skopt): {'✅ 可用' if use_bayesian else '❌ 不可用'}")
    if use_llm:
        print(f"LLM Model: {OLLAMA_MODEL}")
    print(f"Max experiments: {MAX_TOTAL_EXPERIMENTS}")
    print(f"Max outer cycles: {MAX_OUTER_CYCLES}")
    print(f"Target CDS improvement: +{CDS_IMPROVEMENT_TARGET:.1%}")
    print()

    if not use_llm and not use_bayesian:
        print("[ERROR] No optimizer available! Need Ollama or scikit-optimize.")
        return

    # 初始化
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    init_results_file()

    # 初始化机制
    elite_pool = ElitePool(k=5)
    tabu_manager = TabuManager(max_tabu=15)
    sa = SimulatedAnnealing(temperature=0.05, cooling_rate=0.95)
    momentum = MomentumTracker()
    plateau = PlateauDetector(stagnation_threshold=5)
    exploration = ExplorationBudget(explore_every=4)

    # 贝叶斯优化器 (fallback)
    bayesian_opt = BayesianOptimizer() if use_bayesian else None

    # 初始配置
    config = SearchConfig(
        active_params=list(SEARCH_PARAMS.keys()),
        strategy="explore",
        guidance="Start with broad exploration. Try diverse parameter combinations.",
        inner_budget=8,
    )

    # 初始化贝叶斯优化空间
    if bayesian_opt:
        bayesian_opt.build_space(config.active_params)

    baseline_params = read_current_params()
    print(f"Baseline params loaded: {len(baseline_params)} parameters")

    # ======================================================
    # Phase 0: 基线实验
    # ======================================================
    print("\n" + "=" * 50)
    print("Phase 0: 基线实验 (Baseline)")
    print("=" * 50)

    apply_params(baseline_params)
    success = run_training()
    baseline_cds, baseline_metrics = parse_cds_from_log()

    if baseline_cds == 0.0 or not success:
        print("[ERROR] Baseline training failed or CDS=0. Cannot proceed.")
        print("Check run.log for details.")
        return

    print(f"Baseline CDS: {baseline_cds:.4f}")
    print(f"  mAP50: {baseline_metrics.get('mAP50', 0):.4f}")
    print(f"  mAP50-95: {baseline_metrics.get('mAP50_95', 0):.4f}")
    print(f"  Recall: {baseline_metrics.get('recall', 0):.4f}")

    elite_pool.add(baseline_params, baseline_cds)
    best_params = baseline_params.copy()
    best_cds = baseline_cds
    target_cds = baseline_cds * (1 + CDS_IMPROVEMENT_TARGET)

    append_result(0, 0, 0, baseline_cds, baseline_metrics, "BASELINE", baseline_params)
    save_best_model()

    history = [{"params": baseline_params.copy(), "cds": baseline_cds, "metrics": baseline_metrics}]

    update_status(0, 0, 0, best_cds, best_params, config, elite_pool, "baseline done")

    print(f"\nTarget CDS (improve by {CDS_IMPROVEMENT_TARGET:.0%}): {target_cds:.4f}")

    total_exp = 0
    no_improve_overall = 0

    # ======================================================
    # 双层主循环
    # ======================================================
    for outer_cycle in range(1, MAX_OUTER_CYCLES + 1):
        print(f"\n{'='*60}")
        print(f"OUTER CYCLE {outer_cycle}/{MAX_OUTER_CYCLES}")
        print(f"  Strategy: {config.strategy}")
        print(f"  Active params: {len(config.active_params)}")
        print(f"  Frozen params: {len(config.frozen_params)}")
        print(f"  Best CDS so far: {best_cds:.4f}")
        print(f"{'='*60}")

        # 保存外循环配置
        cycle_dir = os.path.join(ARTIFACTS_DIR, f"outer_cycle_{outer_cycle:02d}")
        os.makedirs(cycle_dir, exist_ok=True)
        with open(os.path.join(cycle_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, default=str)

        inner_no_improve = 0

        for inner_iter in range(1, config.inner_budget + 1):
            total_exp += 1
            if total_exp > MAX_TOTAL_EXPERIMENTS:
                print(f"\n[STOP] Reached max total experiments ({MAX_TOTAL_EXPERIMENTS})")
                break

            print(f"\n  --- Inner Iter {inner_iter}/{config.inner_budget} (Exp #{total_exp}) ---")

            current_params = read_current_params()

            # 检查是否是探索轮
            is_exploration = exploration.is_exploration_turn()
            if is_exploration:
                print("  [EXPLORATION] Bold exploration round!")

            # 生成提案
            proposal = None
            proposal_source = "unknown"

            if use_llm:
                proposal = generate_proposal(
                    current_params, best_params, best_cds,
                    config, elite_pool, momentum, is_exploration
                )
                if proposal:
                    proposal_source = "LLM"

            # 如果 LLM 提案失败或不可用，尝试贝叶斯优化
            if not proposal and bayesian_opt:
                proposal = bayesian_opt.ask(current_params)
                if proposal:
                    proposal_source = "Bayesian"
                    print("  [FALLBACK] Using Bayesian optimization proposal")

            # 如果还是没有，尝试交叉候选
            if not proposal:
                crossover = elite_pool.get_crossover_candidate()
                if crossover:
                    proposal = {k: v for k, v in crossover.items()
                              if abs(current_params.get(k, 0) - v) > 1e-8
                              and k in config.active_params}
                    if proposal:
                        proposal_source = "Crossover"
                        print("  [FALLBACK] Using crossover candidate from elite pool")

            # 还是没有，随机扰动
            if not proposal:
                proposal = {}
                for _ in range(3):
                    param = random.choice(config.active_params)
                    spec = SEARCH_PARAMS[param]
                    cur = current_params.get(param, spec["default"])
                    if spec["type"] == "float":
                        delta = random.uniform(-0.2, 0.2) * (spec["max"] - spec["min"])
                        new_val = cur + delta
                        new_val = max(spec["min"], min(spec["max"], new_val))
                    else:
                        delta = random.randint(-3, 3)
                        new_val = max(spec["min"], min(spec["max"], cur + delta))
                    proposal[param] = new_val
                proposal_source = "Random"
                print("  [FALLBACK] Using random perturbation")

            if not proposal:
                print("  [SKIP] No valid proposal generated")
                continue

            print(f"  Proposed changes: {json.dumps(proposal, default=str)}")

            # 检查 Tabu
            new_params = current_params.copy()
            new_params.update(proposal)
            if tabu_manager.is_tabu(new_params):
                print("  [TABU] Skipping — too similar to recent attempts")
                # 轻微扰动一下
                for key in proposal:
                    spec = SEARCH_PARAMS[key]
                    if spec["type"] == "float":
                        jitter = random.uniform(-0.05, 0.05) * (spec["max"] - spec["min"])
                        proposal[key] = max(spec["min"], min(spec["max"], proposal[key] + jitter))
                new_params.update(proposal)
                print("  [TABU FIX] Applied jitter to avoid tabu")

            # 应用参数
            prev_params = current_params.copy()
            apply_params(proposal)

            # 记录动量
            delta_cds_before = best_cds - history[-1]["cds"] if history else 0

            # 运行训练
            print("  Training...")
            success = run_training()
            cds, metrics = parse_cds_from_log()

            if not success or cds == 0.0:
                print("  [CRASH] Training failed or CDS=0")
                # 回滚
                apply_params(prev_params)
                tabu_manager.add(new_params)
                append_result(total_exp, outer_cycle, inner_iter, 0.0, metrics, "CRASH", new_params)
                inner_no_improve += 1
                continue

            print(f"  Result: CDS={cds:.4f} (best={best_cds:.4f})")

            # 模拟退火决策
            is_improvement = cds > best_cds + 0.001
            sa_accept = sa.should_accept(best_cds, cds)

            if is_improvement:
                status = "BEST"
                best_cds = cds
                best_params = new_params.copy()
                elite_pool.add(new_params, cds)
                inner_no_improve = 0
                no_improve_overall = 0
                save_best_model()
                print(f"  ✅ NEW BEST! CDS improved by {cds - history[-1]['cds']:.4f}")

                # 记录动量
                for key in proposal:
                    delta_param = new_params[key] - prev_params.get(key, 0)
                    delta_cds = cds - history[-1]["cds"]
                    momentum.record(key, delta_param, delta_cds)

            elif sa_accept and not is_improvement:
                status = "SA_ACCEPT"
                print(f"  🔄 SA accepted (T={sa.temperature:.4f})")
                inner_no_improve += 1
                no_improve_overall += 1
            else:
                status = "DISCARD"
                # 回滚
                apply_params(prev_params)
                inner_no_improve += 1
                no_improve_overall += 1
                print(f"  ❌ Discarded")

            # 冷却 SA
            sa.cool()

            # 添加到 Tabu
            tabu_manager.add(new_params)

            # 更新历史
            history.append({"params": new_params.copy(), "cds": cds, "metrics": metrics})

            # 告诉贝叶斯优化器结果
            if bayesian_opt and cds > 0:
                bayesian_opt.tell(new_params, cds)

            # 记录结果
            append_result(total_exp, outer_cycle, inner_iter, cds, metrics, status, new_params)

            # 更新状态面板
            update_status(outer_cycle, inner_iter, total_exp, best_cds, best_params, config, elite_pool)

            # 平台检测
            plateau.update(best_cds)
            if plateau.is_stagnant():
                print(f"  [PLATEAU] Stagnation detected! Forcing exploration.")
                config.strategy = "explore"
                plateau.reset()

            # 检查停止条件
            if best_cds >= target_cds:
                print(f"\n[SUCCESS] Target CDS achieved! {best_cds:.4f} >= {target_cds:.4f}")
                update_status(outer_cycle, inner_iter, total_exp, best_cds, best_params, config, elite_pool, "SUCCESS")
                break

            if inner_no_improve >= MAX_NO_IMPROVE_INNER:
                print(f"  [INNER STOP] {MAX_NO_IMPROVE_INNER} consecutive no-improve")
                break

        if best_cds >= target_cds:
            break

        if no_improve_overall >= MAX_NO_IMPROVE_INNER * 2:
            print(f"\n[STOP] Overall stagnation — no improvement for too long")
            break

        # ==================================================
        # 外循环分析 & 配置调整
        # ==================================================
        print(f"\n[Outer Analysis] Adjusting search configuration...")
        old_active = len(config.active_params)
        config = run_outer_analysis(history, config, elite_pool)
        new_active = len(config.active_params)
        print(f"  Active params: {old_active} → {new_active}")
        print(f"  Strategy: {config.strategy}")
        if config.frozen_params:
            print(f"  Frozen: {', '.join(config.frozen_params)}")

        # 更新贝叶斯优化器的搜索空间
        if bayesian_opt and old_active != new_active:
            bayesian_opt.build_space(config.active_params)
            print("  [Bayesian] Rebuilt search space for new active params")

        # 保存外循环分析结果
        with open(os.path.join(cycle_dir, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump({
                "new_config": asdict(config),
                "best_cds": best_cds,
                "history_length": len(history),
            }, f, indent=2, default=str)

        update_status(outer_cycle, 0, total_exp, best_cds, best_params, config, elite_pool)

    # ======================================================
    # 最终总结
    # ======================================================
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"Total experiments: {total_exp}")
    print(f"Baseline CDS: {baseline_cds:.4f}")
    print(f"Best CDS: {best_cds:.4f}")
    print(f"Improvement: {(best_cds - baseline_cds):.4f} ({(best_cds/baseline_cds - 1)*100:.1f}%)")
    print()
    print("Best parameters:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print()
    print(f"Best model saved to: {BEST_MODEL_DIR}/best.pt")
    print(f"Results log: {RESULTS_TSV}")
    print(f"Status dashboard: {STATUS_MD}")

    update_status(MAX_OUTER_CYCLES, 0, total_exp, best_cds, best_params, config, elite_pool, "COMPLETED")

    # 保存最终报告
    final_report = {
        "baseline_cds": baseline_cds,
        "best_cds": best_cds,
        "improvement_abs": best_cds - baseline_cds,
        "improvement_pct": (best_cds / baseline_cds - 1) * 100,
        "total_experiments": total_exp,
        "best_params": best_params,
        "elite_pool": [{"cds": e["cds"], "params": e["params"]} for e in elite_pool.pool],
        "final_config": asdict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(ARTIFACTS_DIR, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, default=str)

    unload_ollama()
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
