"""
bug_fixer.py — BugFixer Agent V1 (Bug修复专家)

职责:
  1. 消费 OverseerDoctor 输出的 doctor_action.jsonl 中的失败/升级记录
  2. 具备比 OverseerDoctor 内置 FixExecutor 更强的修复能力
  3. 直接修改项目源码、YAML配置、修复 state.db、杀僵尸进程、清理损坏文件
  4. 检测无限循环（同一修复重复执行无进展）
  5. 复杂 bug 调用 LLM 诊断
  6. 彻底修不好才 escalate 到人工

与 OverseerDoctor 的关系:
  - OverseerDoctor: 检测 + 初级修复（改 train.py 参数 + 重启）
  - BugFixerAgent: 高级修复（改源码、修DB、杀进程、深层策略调整）

运行方式:
  python -m autoresearch_v2.agents.bug_fixer --watch
  python -m autoresearch_v2.agents.bug_fixer --once
"""

import argparse
import json
import os
import re
import sys
import time
import shutil
import sqlite3
import subprocess
import threading
import traceback
import hashlib
import glob as glob_mod
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict

from autoresearch_v2._json_extract import extract_first_json

PROJECT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."
))
V2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STATE_DIR = os.path.join(V2_ROOT, "state")
STATE_DB = os.path.join(STATE_DIR, "state.db")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train.py")
DOCTOR_ACTION_LOG = os.path.join(PROJECT_ROOT, "doctor_action.jsonl")
BUGFIXER_ACTION_LOG = os.path.join(PROJECT_ROOT, "bugfixer_action.jsonl")
BUGFIXER_STATE_FILE = os.path.join(PROJECT_ROOT, "reports", "bugfixer_state.json")
BUGFIXER_LOG_FILE = os.path.join(PROJECT_ROOT, "bugfixer.log")

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "bug_fixer.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "bug_fixer_diagnosis.json")

DEFAULT_OVERSEER_URL = "http://127.0.0.1:5050"
DEFAULT_POLL_INTERVAL = 15
LLM_TIMEOUT = 120.0
MAX_CONSECUTIVE_FAILS = 3
LOOP_DETECTION_WINDOW = 10
LOOP_DETECTION_MIN_COUNT = 5

MODEL_SIZES = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]

_log_lock = threading.Lock()


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icons = {
        "INFO": "\U0001f527", "WARN": "\u26a0\ufe0f", "CRIT": "\U0001f6a8",
        "OK": "\u2705", "FIX": "\U0001f9f0", "DIAG": "\U0001f50d",
        "ESCALATE": "\U0001f198", "LOOP": "\U0001f501", "DB": "\U0001f5c4\ufe0f",
        "PROC": "\U0001f5a5\ufe0f", "SRC": "\U0001f4dd",
    }
    icon = icons.get(level, "\U0001f527")
    line = f"[{ts}] [BUGFIXER:{level}] {icon} {msg}"
    with _log_lock:
        try:
            print(line)
            sys.stdout.flush()
        except UnicodeEncodeError:
            try:
                safe_line = line.encode("utf-8", errors="replace").decode("utf-8")
                print(safe_line)
                sys.stdout.flush()
            except Exception:
                pass
        try:
            with open(BUGFIXER_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": ts, "level": level, "msg": msg},
                                   ensure_ascii=False) + "\n")
        except Exception:
            pass


def load_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_schema() -> dict:
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def write_action_record(record: dict):
    os.makedirs(os.path.dirname(BUGFIXER_ACTION_LOG), exist_ok=True)
    record["timestamp"] = datetime.now().isoformat()
    try:
        with open(BUGFIXER_ACTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"写入 action 日志失败: {e}", "WARN")


def _call_ollama(system_prompt: str, user_prompt: str,
                 model: str = "gemma3:4b",
                 url: str = "http://localhost:11434") -> tuple[Optional[str], Optional[str]]:
    try:
        import httpx
    except ImportError:
        return None, "httpx not installed"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        client = httpx.Client(timeout=LLM_TIMEOUT)
        response = client.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False,
                  "temperature": 0.2, "num_predict": 1024})
        if response.status_code == 200:
            return response.json().get("response", ""), None
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _call_anthropic(system_prompt: str, user_prompt: str,
                    model: str = "claude-haiku-4-20250414") -> tuple[Optional[str], Optional[str]]:
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed"
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "ANTHROPIC_API_KEY not set"
    try:
        client = anthropic.Anthropic(api_key=key, timeout=LLM_TIMEOUT)
        message = client.messages.create(
            model=model, max_tokens=1024, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}])
        return message.content[0].text, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


class ActionReader:
    """读取 OverseerDoctor 输出的 doctor_action.jsonl"""

    def __init__(self, action_log_path: str = DOCTOR_ACTION_LOG):
        self.action_log_path = action_log_path
        self._last_position = 0
        self._last_file_hash = ""

    def read_new_actions(self) -> list[dict]:
        """读取自上次以来的新 action 记录"""
        if not os.path.exists(self.action_log_path):
            return []
        actions = []
        try:
            with open(self.action_log_path, "r", encoding="utf-8") as f:
                f.seek(self._last_position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action = json.loads(line)
                            actions.append(action)
                        except json.JSONDecodeError:
                            pass
                self._last_position = f.tell()
        except Exception as e:
            log(f"读取 doctor_action.jsonl 失败: {e}", "WARN")
        return actions

    def read_all(self, limit: int = 200) -> list[dict]:
        """读取所有 action 记录（用于一次性诊断）"""
        if not os.path.exists(self.action_log_path):
            return []
        actions = []
        try:
            with open(self.action_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            for line in lines[start:]:
                line = line.strip()
                if line:
                    try:
                        actions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            log(f"读取 doctor_action.jsonl 失败: {e}", "WARN")
        return actions

    def reset_position(self):
        self._last_position = 0


class LoopDetector:
    """检测无限循环：同一修复反复执行但无进展"""

    def __init__(self):
        self.action_history = defaultdict(list)

    def feed(self, action: dict):
        error_type = action.get("alert", {}).get("error_type",
                       action.get("error", {}).get("error_type", "unknown"))
        action_name = action.get("result", {}).get("action", "unknown")
        key = f"{error_type}:{action_name}"
        self.action_history[key].append(time.time())
        # 清理旧记录
        cutoff = time.time() - 1800
        for k in list(self.action_history.keys()):
            self.action_history[k] = [t for t in self.action_history[k] if t > cutoff]
            if not self.action_history[k]:
                del self.action_history[k]

    def detect_loop(self) -> Optional[dict]:
        """检测是否有修复动作在死循环"""
        for key, timestamps in self.action_history.items():
            if len(timestamps) >= LOOP_DETECTION_MIN_COUNT:
                recent = timestamps[-LOOP_DETECTION_WINDOW:]
                if len(recent) >= LOOP_DETECTION_MIN_COUNT:
                    error_type, action_name = key.split(":", 1)
                    return {
                        "error_type": error_type,
                        "action": action_name,
                        "count": len(recent),
                        "duration_seconds": recent[-1] - recent[0],
                        "description": (
                            f"{action_name} 修复已执行 {len(recent)} 次"
                            f"（{recent[-1] - recent[0]:.0f}s），疑似死循环"
                        ),
                    }
        return None

    def reset(self, error_type: str, action_name: str):
        key = f"{error_type}:{action_name}"
        if key in self.action_history:
            del self.action_history[key]


class DeepFixer:
    """高级修复器 — 比 OverseerDoctor 的 FixExecutor 更强的修复能力"""

    def __init__(self, overseer_url: str = DEFAULT_OVERSEER_URL):
        self.overseer_url = overseer_url.rstrip("/")
        self.project_root = PROJECT_ROOT
        self.train_script = TRAIN_SCRIPT
        self.state_db = STATE_DB
        self._last_restart = 0
        self._restart_cooldown = 30

    def _agent_api(self, name: str, action: str) -> bool:
        """调用总管 API 操作智能体"""
        now = time.time()
        if action in ("restart", "stop") and now - self._last_restart < self._restart_cooldown:
            log(f"重启冷却中，跳过 {name}/{action}", "WARN")
            return False
        if action in ("restart", "stop"):
            self._last_restart = now
        try:
            import urllib.request
            from urllib.parse import quote
            url = f"{self.overseer_url}/api/agent/{quote(name)}/{action}"
            req = urllib.request.Request(url, method="POST")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("ok", False)
        except Exception as e:
            log(f"API 操作失败 ({name}/{action}): {e}", "WARN")
            return False

    def _read_train_param(self, key: str, default=None):
        try:
            with open(self.train_script, "r", encoding="utf-8") as f:
                content = f.read()
            pattern = rf'^{key}\s*=\s*(.+)$'
            m = re.search(pattern, content, re.MULTILINE)
            if m:
                val = m.group(1).strip().strip('"').strip("'")
                try:
                    return int(val)
                except ValueError:
                    try:
                        return float(val)
                    except ValueError:
                        return val
        except Exception:
            pass
        return default

    def _write_train_param(self, key: str, value) -> bool:
        try:
            with open(self.train_script, "r", encoding="utf-8") as f:
                lines = f.readlines()
            found = False
            for i, line in enumerate(lines):
                if re.match(rf"^{key}\s*=", line.strip()):
                    if isinstance(value, str):
                        lines[i] = f'{key} = "{value}"\n'
                    elif isinstance(value, bool):
                        lines[i] = f"{key} = {repr(value)}\n"
                    else:
                        lines[i] = f"{key} = {repr(value)}\n"
                    found = True
                    break
            if not found:
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    lines.append(f"{key} = {repr(value)}\n")
                else:
                    lines.append(f"{key} = {repr(value)}\n")
            with open(self.train_script, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except Exception as e:
            log(f"修改 train.py 参数 {key} 失败: {e}", "WARN")
            return False

    # ─── 核心修复方法 ───

    def break_infinite_loop(self, loop_info: dict) -> dict:
        """打破无限循环 — 这是 BugFixer 最核心的能力"""
        error_type = loop_info["error_type"]
        action = loop_info["action"]

        log(f"检测到死循环: {error_type} → {action} (×{loop_info['count']})", "LOOP")

        # 策略1: 如果是 stuck_cds_stagnant 且 model 已是最小 + imgsz 已是最低
        # → 不再改 model/imgsz，改为调整学习率、epoch、数据增强
        if error_type in ("stuck_cds_stagnant", "stuck_no_progress"):
            current_model = self._read_train_param("MODEL", "yolo12n.pt")
            current_imgsz = self._read_train_param("IMGSZ", 640)
            current_epochs = self._read_train_param("EPOCHS", 50)
            current_lr0 = self._read_train_param("LR0", 0.01)
            current_batch = self._read_train_param("BATCH", 8)

            model_is_smallest = str(current_model).strip('"').strip("'") == MODEL_SIZES[0]
            imgsz_is_min = current_imgsz <= 640

            if model_is_smallest and imgsz_is_min:
                # 已经是最小模型+最低分辨率，需要换策略
                # 方案A: 大幅增加 epochs
                if current_epochs < 200:
                    new_epochs = min(300, current_epochs * 2)
                    self._write_train_param("EPOCHS", new_epochs)
                    self._write_train_param("PATIENCE", min(50, new_epochs // 2))
                    msg = f"打破死循环: EPOCHS {current_epochs}→{new_epochs}"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}

                # 方案B: 调整学习率
                if current_lr0 > 0.0001:
                    new_lr0 = current_lr0 * 0.5
                    self._write_train_param("LR0", new_lr0)
                    msg = f"打破死循环: LR0 {current_lr0}→{new_lr0}"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}

                # 方案C: 调整数据增强
                current_mosaic = self._read_train_param("MOSAIC", 1.0)
                if current_mosaic > 0.3:
                    new_mosaic = max(0.1, current_mosaic * 0.5)
                    self._write_train_param("MOSAIC", new_mosaic)
                    msg = f"打破死循环: MOSAIC {current_mosaic}→{new_mosaic}"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}

                # 方案D: 增大 batch
                if current_batch < 32:
                    new_batch = min(32, current_batch * 2)
                    self._write_train_param("BATCH", new_batch)
                    msg = f"打破死循环: BATCH {current_batch}→{new_batch}"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}

                # 方案E: 切换到更大模型
                try:
                    idx = MODEL_SIZES.index(str(current_model).strip('"').strip("'"))
                    if idx < len(MODEL_SIZES) - 1:
                        new_model = MODEL_SIZES[idx + 1]
                        self._write_train_param("MODEL", new_model)
                        self._write_train_param("EPOCHS", current_epochs)
                        msg = f"打破死循环: 反向切换 MODEL {current_model}→{new_model}"
                        log(msg, "FIX")
                        self._agent_api("Orchestrator", "restart")
                        return {"action": "deep_fix", "target": "Orchestrator",
                                "success": True, "message": msg}
                except ValueError:
                    pass

            else:
                # 还有降级空间，但死循环说明降级无效
                # 尝试增大 epochs
                if current_epochs < 100:
                    new_epochs = current_epochs * 2
                    self._write_train_param("EPOCHS", new_epochs)
                    msg = f"打破死循环: EPOCHS {current_epochs}→{new_epochs}"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}

        # 策略2: 如果是 OOM 死循环 → 直接切换模型
        if error_type == "cuda_oom":
            current_model = self._read_train_param("MODEL", "yolo12n.pt")
            try:
                idx = MODEL_SIZES.index(str(current_model).strip('"').strip("'"))
                if idx > 0:
                    new_model = MODEL_SIZES[idx - 1]
                    self._write_train_param("MODEL", new_model)
                    self._write_train_param("BATCH", 2)
                    self._write_train_param("IMGSZ", 640)
                    msg = f"打破OOM死循环: MODEL {current_model}→{new_model}, BATCH→2, IMGSZ→640"
                    log(msg, "FIX")
                    self._agent_api("Orchestrator", "restart")
                    return {"action": "deep_fix", "target": "Orchestrator",
                            "success": True, "message": msg}
            except ValueError:
                pass

        # 策略3: crash / consecutive_fails 死循环 → 训练脚本崩溃，深层参数调整
        if error_type in ("crash", "consecutive_fails"):
            current_lr0 = self._read_train_param("LR0", 0.01)
            current_epochs = self._read_train_param("EPOCHS", 50)
            current_batch = self._read_train_param("BATCH", 8)
            current_imgsz = self._read_train_param("IMGSZ", 640)

            # 方案A: 训练崩溃最常见原因是学习率过高，先暴力降 LR
            if current_lr0 > 0.0001:
                new_lr0 = current_lr0 * 0.1
                self._write_train_param("LR0", new_lr0)
                self._write_train_param("AMP", False)
                msg = f"打破crash死循环: LR0 {current_lr0}→{new_lr0}, AMP disabled"
                log(msg, "FIX")
                self._agent_api("Orchestrator", "restart")
                return {"action": "deep_fix", "target": "Orchestrator",
                        "success": True, "message": msg}

            # 方案B: 降低输入分辨率
            if current_imgsz > 640:
                new_imgsz = max(640, current_imgsz // 2)
                self._write_train_param("IMGSZ", new_imgsz)
                msg = f"打破crash死循环: IMGSZ {current_imgsz}→{new_imgsz}"
                log(msg, "FIX")
                self._agent_api("Orchestrator", "restart")
                return {"action": "deep_fix", "target": "Orchestrator",
                        "success": True, "message": msg}

            # 方案C: 减小 batch + 增大 epochs 补偿
            new_batch = max(1, current_batch // 2)
            new_epochs = min(300, current_epochs * 2)
            self._write_train_param("BATCH", new_batch)
            self._write_train_param("EPOCHS", new_epochs)
            msg = f"打破crash死循环: BATCH {current_batch}→{new_batch}, EPOCHS {current_epochs}→{new_epochs}"
            log(msg, "FIX")
            self._agent_api("Orchestrator", "restart")
            return {"action": "deep_fix", "target": "Orchestrator",
                    "success": True, "message": msg}

        # 兜底: 无法自动打破，升级
        return {"action": "escalate", "target": "Orchestrator",
                "success": False,
                "message": f"无法自动打破死循环 ({error_type}:{action})，升级人工"}

    def fix_state_db(self, diagnosis: dict) -> dict:
        """修复 state.db — 清理损坏/卡住的实验记录"""
        if not os.path.exists(self.state_db):
            return {"action": "fix_db", "target": "state.db",
                    "success": False, "message": "state.db 不存在"}
        try:
            conn = sqlite3.connect(self.state_db)
            cur = conn.cursor()

            # 修复1: 将长时间 pending 的实验标记为 failed
            cur.execute("""
                UPDATE experiments SET status = 'failed'
                WHERE status = 'pending'
                AND julianday('now') - julianday(started_at) > 1
            """)
            pending_fixed = cur.rowcount

            # 修复2: 清理没有 metrics 的 completed 实验
            cur.execute("""
                UPDATE experiments SET status = 'failed'
                WHERE status = 'completed'
                AND (metrics_json IS NULL OR metrics_json = '{}')
            """)
            empty_fixed = cur.rowcount

            # 修复3: 删除重复的 pending 实验（保留最新的）
            cur.execute("""
                DELETE FROM experiments
                WHERE status = 'pending'
                AND run_id NOT IN (
                    SELECT run_id FROM experiments
                    WHERE status = 'pending'
                    ORDER BY started_at DESC LIMIT 3
                )
            """)
            dup_fixed = cur.rowcount

            conn.commit()
            conn.close()

            total = pending_fixed + empty_fixed + dup_fixed
            if total > 0:
                msg = (f"修复 state.db: pending→failed={pending_fixed}, "
                       f"空metrics→failed={empty_fixed}, 清理重复pending={dup_fixed}")
                log(msg, "DB")
                return {"action": "fix_db", "target": "state.db",
                        "success": True, "message": msg}
            return {"action": "fix_db", "target": "state.db",
                    "success": True, "message": "state.db 无需修复"}
        except Exception as e:
            return {"action": "fix_db", "target": "state.db",
                    "success": False, "message": f"修复 state.db 失败: {e}"}

    def kill_zombie_processes(self) -> dict:
        """杀僵尸进程 — 查找并清理残留的训练进程"""
        killed = 0
        try:
            # Windows: 查找残留的 python 训练进程
            if sys.platform == "win32":
                result = subprocess.run(
                    ["wmic", "process", "get", "ProcessId,CommandLine"],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if "train.py" in line or "ultralytics" in line:
                        parts = line.strip().split()
                        if parts:
                            pid = parts[-1]
                            try:
                                subprocess.run(
                                    ["taskkill", "/F", "/PID", pid],
                                    capture_output=True, timeout=5
                                )
                                killed += 1
                            except Exception:
                                pass
            else:
                result = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.splitlines():
                    if "train.py" in line or "ultralytics" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            pid = parts[1]
                            try:
                                subprocess.run(["kill", "-9", pid], timeout=5)
                                killed += 1
                            except Exception:
                                pass
        except Exception as e:
            log(f"杀僵尸进程失败: {e}", "WARN")

        if killed > 0:
            msg = f"清理了 {killed} 个僵尸训练进程"
            log(msg, "PROC")
            return {"action": "kill_zombie", "target": "system",
                    "success": True, "message": msg}
        return {"action": "kill_zombie", "target": "system",
                "success": True, "message": "未发现僵尸进程"}

    def clean_corrupted_files(self) -> dict:
        """清理损坏文件 — 检查并删除损坏的模型权重、日志等"""
        cleaned = 0
        cleaned_bytes = 0

        # 清理损坏的 .pt 文件（大小为0或异常小）
        runs_dir = os.path.join(self.project_root, "autoresearch_runs")
        if os.path.exists(runs_dir):
            for root, dirs, files in os.walk(runs_dir):
                for f in files:
                    if f.endswith(".pt"):
                        fpath = os.path.join(root, f)
                        try:
                            size = os.path.getsize(fpath)
                            if size < 1024:  # 小于1KB的权重文件视为损坏
                                os.remove(fpath)
                                cleaned += 1
                                cleaned_bytes += size
                        except Exception:
                            pass

        # 清理空的 results.csv
        if os.path.exists(runs_dir):
            for root, dirs, files in os.walk(runs_dir):
                for f in files:
                    if f == "results.csv":
                        fpath = os.path.join(root, f)
                        try:
                            if os.path.getsize(fpath) == 0:
                                os.remove(fpath)
                                cleaned += 1
                        except Exception:
                            pass

        if cleaned > 0:
            msg = f"清理了 {cleaned} 个损坏文件 (约 {cleaned_bytes / 1024:.1f} KB)"
            log(msg, "FIX")
            return {"action": "clean_corrupted", "target": "filesystem",
                    "success": True, "message": msg}
        return {"action": "clean_corrupted", "target": "filesystem",
                "success": True, "message": "未发现损坏文件"}

    def fix_data_yaml(self) -> dict:
        """修复 data.yaml 配置"""
        import yaml
        data_yaml_path = self._read_train_param("DATA_YAML", "")
        if not data_yaml_path or not os.path.exists(str(data_yaml_path).strip('"')):
            return {"action": "fix_yaml", "target": "data.yaml",
                    "success": False, "message": "data.yaml 路径无效或不存在"}

        data_yaml_path = str(data_yaml_path).strip('"').strip("'")
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            fixed = False

            # 修复1: 检查路径是否存在
            for key in ("train", "val", "test"):
                if key in config:
                    p = config[key]
                    if isinstance(p, str) and not os.path.exists(p):
                        # 尝试相对路径
                        alt_path = os.path.join(os.path.dirname(data_yaml_path), p)
                        if os.path.exists(alt_path):
                            config[key] = alt_path
                            fixed = True
                            log(f"修复 data.yaml {key}: {p} → {alt_path}", "FIX")

            # 修复2: 检查 nc 是否与 names 一致
            if "names" in config and "nc" in config:
                expected_nc = len(config["names"])
                if config["nc"] != expected_nc:
                    config["nc"] = expected_nc
                    fixed = True
                    log(f"修复 data.yaml nc: 修正为 {expected_nc}", "FIX")

            if fixed:
                with open(data_yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                return {"action": "fix_yaml", "target": "data.yaml",
                        "success": True, "message": "data.yaml 已修复"}
            return {"action": "fix_yaml", "target": "data.yaml",
                    "success": True, "message": "data.yaml 无需修复"}
        except Exception as e:
            return {"action": "fix_yaml", "target": "data.yaml",
                    "success": False, "message": f"修复 data.yaml 失败: {e}"}

    def fix_corrupted_source(self, diagnosis: dict) -> dict:
        """修复损坏的源码文件"""
        files_to_check = [
            TRAIN_SCRIPT,
            os.path.join(self.project_root, "evaluate.py"),
            os.path.join(V2_ROOT, "orchestrator.py"),
            os.path.join(V2_ROOT, "db.py"),
        ]
        fixed_count = 0
        for fpath in files_to_check:
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # 检查常见问题
                if "\r\n\r\n" in content:  # 多余空行
                    content = content.replace("\r\n\r\n\r\n", "\r\n\r\n")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixed_count += 1
                    log(f"修复多余空行: {os.path.basename(fpath)}", "SRC")
            except Exception:
                pass

        if fixed_count > 0:
            return {"action": "fix_source", "target": "source",
                    "success": True, "message": f"修复了 {fixed_count} 个源码文件"}
        return {"action": "fix_source", "target": "source",
                "success": True, "message": "源码无需修复"}

    def force_restart_everything(self) -> dict:
        """强制重启所有关键智能体"""
        log("强制重启全部战斗群", "PROC")
        self._agent_api("Orchestrator", "restart")
        time.sleep(2)
        self._agent_api("SupervisorAgent", "restart")
        time.sleep(1)
        self._agent_api("ArchitectureSupervisor", "restart")
        return {"action": "force_restart_all", "target": "all",
                "success": True, "message": "已强制重启全部智能体"}

    def execute(self, diagnosis: dict, error_info: dict) -> dict:
        """执行修复 — 入口方法"""
        action = diagnosis.get("recommended_action", "escalate")
        target = error_info.get("target_agent",
                                 diagnosis.get("target_agent", "Orchestrator"))

        handler_map = {
            "break_loop": self.break_infinite_loop,
            "fix_db": self.fix_state_db,
            "kill_zombie": self.kill_zombie_processes,
            "clean_corrupted": self.clean_corrupted_files,
            "fix_yaml": self.fix_data_yaml,
            "fix_source": self.fix_corrupted_source,
            "force_restart_all": self.force_restart_everything,
            "deep_fix": self._execute_deep_fix,
            "escalate": self._execute_escalate,
        }

        handler = handler_map.get(action, self._execute_escalate)
        try:
            result = handler(diagnosis) if action != "deep_fix" else handler(diagnosis, error_info)
            write_action_record({
                "type": "bugfixer_fix",
                "trigger": error_info,
                "diagnosis": diagnosis,
                "result": result,
            })
            return result
        except Exception as e:
            return {"action": action, "target": target,
                    "success": False, "message": f"执行修复异常: {e}",
                    "details": {"traceback": traceback.format_exc()}}

    def _execute_deep_fix(self, diagnosis: dict, error_info: dict) -> dict:
        """深层修复 — 修改 train.py 的任意参数"""
        fix_detail = diagnosis.get("fix_detail", {})
        if not fix_detail:
            return {"action": "deep_fix", "target": "Orchestrator",
                    "success": False, "message": "无 fix_detail"}

        ok = True
        for key, value in fix_detail.items():
            if isinstance(value, (int, float, bool, str)):
                if not self._write_train_param(key, value):
                    ok = False
        if ok:
            self._agent_api("Orchestrator", "restart")
        return {"action": "deep_fix", "target": "Orchestrator",
                "success": ok, "message": f"已修改参数: {fix_detail}"}

    def _execute_escalate(self, diagnosis: dict) -> dict:
        root_cause = diagnosis.get("root_cause_hypothesis", "未知")
        log(f"升级人工: {root_cause}", "ESCALATE")
        return {"action": "escalate", "target": "human",
                "success": True, "message": f"已升级人工: {root_cause}"}


class BugFixerAgent:
    """Bug修复专家智能体"""

    def __init__(self, config: dict = None):
        config = config or {}
        self.overseer_url = config.get("overseer_url", DEFAULT_OVERSEER_URL)
        self.poll_interval = config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        self.llm_backend = config.get("llm_backend", "auto")
        self.ollama_model = config.get("ollama_model", "gemma3:4b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.anthropic_model = config.get("anthropic_model", "claude-haiku-4-20250414")
        self.max_llm_retries = config.get("max_llm_retries", 3)

        self.reader = ActionReader(DOCTOR_ACTION_LOG)
        self.loop_detector = LoopDetector()
        self.fixer = DeepFixer(self.overseer_url)
        self.system_prompt = load_prompt()
        self.running = True
        self.total_fixes = 0
        self.successful_fixes = 0
        self.loop_breaks = 0
        self.escalated_count = 0

        os.makedirs(os.path.dirname(BUGFIXER_ACTION_LOG), exist_ok=True)
        os.makedirs(os.path.dirname(BUGFIXER_STATE_FILE), exist_ok=True)

    def _persist_state(self):
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "total_fixes": self.total_fixes,
                "successful_fixes": self.successful_fixes,
                "loop_breaks": self.loop_breaks,
                "escalated_count": self.escalated_count,
            }
            with open(BUGFIXER_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def diagnose_with_llm(self, error_info: dict) -> dict:
        """LLM 辅助诊断复杂 bug"""
        lines = [
            "## 错误信息",
            f"- 类型: {error_info.get('error_type', 'unknown')}",
            f"- 动作: {error_info.get('action', 'unknown')}",
            f"- 日志: {error_info.get('log_line', '')[:500]}",
            f"- 目标: {error_info.get('target_agent', 'unknown')}",
        ]
        user_prompt = "\n".join(lines) + "\n\n请诊断根因并推荐修复方案。严格输出 JSON。"
        for attempt in range(self.max_llm_retries):
            response = None
            if self.llm_backend in ("anthropic", "auto"):
                response, _ = _call_anthropic(self.system_prompt, user_prompt, self.anthropic_model)
            if response is None and self.llm_backend in ("ollama", "auto"):
                response, _ = _call_ollama(self.system_prompt, user_prompt, self.ollama_model, self.ollama_url)
            if response is None:
                continue
            diagnosis = extract_first_json(response)
            if diagnosis is None:
                continue
            return diagnosis
        return {"root_cause_hypothesis": "LLM 连续失败，无法诊断",
                "confidence": 0.0, "recommended_action": "escalate"}

    def _filter_actionable(self, actions: list[dict]) -> list[dict]:
        """筛选需要 BugFixer 处理的 action"""
        actionable = []
        for action in actions:
            result = action.get("result", {})
            # 关注: 修复失败 或 escalate 或 多次重复同一动作
            if not result.get("success", False):
                actionable.append(action)
            elif result.get("action") == "escalate":
                actionable.append(action)
        return actionable

    def process_actions(self, actions: list[dict]) -> int:
        """处理一批 action 记录"""
        actionable = self._filter_actionable(actions)
        if not actionable:
            return 0

        fixes_applied = 0
        for action in actionable:
            self.loop_detector.feed(action)

        # 先检测死循环
        loop_info = self.loop_detector.detect_loop()
        if loop_info:
            log(f"死循环: {loop_info['description']}", "LOOP")
            result = self.fixer.break_infinite_loop(loop_info)
            self.total_fixes += 1
            if result["success"]:
                self.successful_fixes += 1
                self.loop_breaks += 1
                self.loop_detector.reset(loop_info["error_type"], loop_info["action"])
                log(f"打破死循环成功: {result['message']}", "OK")
            else:
                log(f"打破死循环失败: {result['message']}", "WARN")
            fixes_applied += 1
            # 死循环已打破，跳过本轮所有后续单独处理，避免重复修复
            return fixes_applied

        # 处理单个失败/升级的 action
        for action in actionable:
            error_info = action.get("error", action.get("alert", {}))
            result_info = action.get("result", {})
            error_type = error_info.get("error_type", "unknown")
            prev_action = result_info.get("action", "unknown")

            log(f"处理失败动作: {error_type} → {prev_action}", "DIAG")

            # 尝试 LLM 诊断
            diagnosis = self.diagnose_with_llm({
                "error_type": error_type,
                "action": prev_action,
                "log_line": error_info.get("log_line", ""),
                "target_agent": error_info.get("target_agent", "Orchestrator"),
            })

            result = self.fixer.execute(diagnosis, error_info)
            self.total_fixes += 1
            if result["success"]:
                self.successful_fixes += 1
                log(f"修复成功: {result['message']}", "OK")
            else:
                log(f"修复失败: {result['message']}", "WARN")
                if result.get("action") == "escalate":
                    self.escalated_count += 1

            fixes_applied += 1

        return fixes_applied

    def run_health_checks(self) -> int:
        """主动健康检查 — 不依赖 doctor_action.jsonl"""
        fixes = 0

        # 检查 state.db 是否异常
        db_result = self.fixer.fix_state_db({})
        if "已修复" in db_result.get("message", ""):
            fixes += 1

        # 检查僵尸进程
        zombie_result = self.fixer.kill_zombie_processes()
        if "清理了" in zombie_result.get("message", ""):
            fixes += 1

        # 检查损坏文件
        corrupt_result = self.fixer.clean_corrupted_files()
        if "清理了" in corrupt_result.get("message", ""):
            fixes += 1

        return fixes

    def run_once(self) -> dict:
        """执行一轮修复"""
        actions = self.reader.read_new_actions()
        action_fixes = self.process_actions(actions)
        health_fixes = self.run_health_checks()
        total_fixes = action_fixes + health_fixes

        self._persist_state()

        return {
            "new_actions": len(actions),
            "action_fixes": action_fixes,
            "health_fixes": health_fixes,
            "total_fixes": total_fixes,
            "loop_breaks": self.loop_breaks,
            "escalated": self.escalated_count,
        }

    def watch(self):
        log("=" * 60)
        log("  BugFixer Agent V1 — Bug修复专家启动", "INFO")
        log("=" * 60)
        log(f"总管 URL: {self.overseer_url}")
        log(f"轮询间隔: {self.poll_interval}s")
        log(f"监控文件: {DOCTOR_ACTION_LOG}")
        log(f"死循环检测: {LOOP_DETECTION_MIN_COUNT}次/{LOOP_DETECTION_WINDOW}条")
        log("")

        self.reader.reset_position()

        try:
            while self.running:
                result = self.run_once()
                if result["total_fixes"] > 0:
                    log(f"本轮: {result['total_fixes']} 次修复 "
                        f"(action={result['action_fixes']}, health={result['health_fixes']})", "FIX")
                else:
                    log(f"本轮无异常 | 等待 {self.poll_interval}s", "OK")
                elapsed = 0
                while elapsed < self.poll_interval and self.running:
                    time.sleep(1)
                    elapsed += 1
        except KeyboardInterrupt:
            log("收到中断信号", "WARN")
        finally:
            self._print_summary()
            self._persist_state()
            log("BugFixer Agent 已停止", "OK")

    def _print_summary(self):
        log("")
        log("=" * 60)
        log("  BugFixer Agent V1 — 运行总结")
        log("=" * 60)
        log(f"总修复: {self.successful_fixes}/{self.total_fixes} 成功")
        log(f"打破死循环: {self.loop_breaks} 次")
        log(f"升级人工: {self.escalated_count} 次")
        if self.total_fixes > 0:
            log(f"成功率: {self.successful_fixes / self.total_fixes * 100:.1f}%")
        log(f"修复日志: {BUGFIXER_ACTION_LOG}")
        log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BugFixer Agent V1 — Bug修复专家")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help=f"轮询间隔 (默认: {DEFAULT_POLL_INTERVAL}s)")
    parser.add_argument("--overseer-url", type=str, default=DEFAULT_OVERSEER_URL,
                        help="总管 Web UI URL")
    parser.add_argument("--llm-backend", type=str, default="auto",
                        choices=["auto", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--anthropic-model", type=str, default="claude-haiku-4-20250414")
    parser.add_argument("--once", action="store_true", help="一次性诊断修复")
    args = parser.parse_args()

    config = {
        "overseer_url": args.overseer_url,
        "poll_interval": args.interval,
        "llm_backend": args.llm_backend,
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "anthropic_model": args.anthropic_model,
    }

    agent = BugFixerAgent(config)
    if args.watch:
        agent.watch()
    else:
        result = agent.run_once()
        print()
        print("=" * 60)
        print("  BugFixer Agent V1 — 一次性诊断完成")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60)


if __name__ == "__main__":
    main()