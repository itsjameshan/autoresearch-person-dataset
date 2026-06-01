"""
overseer_doctor.py — Overseer Doctor Agent V2 (总管诊断医生)

职责:
  1. 实时监控总管智能体 Web UI 日志 + API 状态
  2. 卡死检测 — 实验停滞/无日志输出/Agent 假死 → 自动重启
  3. 分级 OOM — batch_half → reduce_imgsz → switch_smaller_model
  4. 策略调整 — 连续同类错误 → 自动切换整体方案
  5. 跨智能体协调 — 修复后联动重启相关智能体
  6. LLM 诊断兜底 — 未知模式调用 LLM

运行方式:
  python -m autoresearch_v2.agents.overseer_doctor --watch
  python -m autoresearch_v2.agents.overseer_doctor
"""

import argparse
import json
import os
import re
import sys
import time
import shutil
import subprocess
import threading
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from autoresearch_v2._json_extract import extract_first_json

PROJECT_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."
))
V2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STATE_DIR = os.path.join(V2_ROOT, "state")
STATE_DB = os.path.join(STATE_DIR, "state.db")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "train.py")

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "overseer_doctor.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "overseer_doctor_diagnosis.json")

DOCTOR_ACTION_LOG = os.path.join(PROJECT_ROOT, "doctor_action.jsonl")
DOCTOR_STATE_FILE = os.path.join(PROJECT_ROOT, "reports", "doctor_state.json")

DEFAULT_OVERSEER_URL = "http://127.0.0.1:5050"
DEFAULT_POLL_INTERVAL = 10
MAX_CONSECUTIVE_FAILS = 3
LLM_TIMEOUT = 120.0
DOCTOR_LOG_FILE = os.path.join(PROJECT_ROOT, "doctor.log")

STUCK_NO_EXPERIMENT_MINUTES = 20
STUCK_NO_LOG_MINUTES = 8
STUCK_CDS_STAGNANT_ROUNDS = 6
OOM_TIER1_BATCH_MIN = 1
MODEL_SIZES = ["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"]

_log_lock = threading.Lock()


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icons = {
        "INFO": "\U0001f48a", "WARN": "\u26a0\ufe0f", "CRIT": "\U0001f691", "OK": "\u2705",
        "FIX": "\U0001f527", "DIAG": "\U0001f50d", "ESCALATE": "\U0001f198", "WATCH": "\U0001f441\U0000fe0f",
        "STUCK": "\U0001f6a8", "STRATEGY": "\U0001f9e0",
    }
    icon = icons.get(level, "\U0001f48a")
    line = f"[{ts}] [DOCTOR:{level}] {icon} {msg}"
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
            with open(DOCTOR_LOG_FILE, "a", encoding="utf-8") as f:
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
    os.makedirs(os.path.dirname(DOCTOR_ACTION_LOG), exist_ok=True)
    record["timestamp"] = datetime.now().isoformat()
    try:
        with open(DOCTOR_ACTION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"写入 action 日志失败: {e}", "WARN")


class LogFetcher:
    def __init__(self, overseer_url: str = DEFAULT_OVERSEER_URL):
        self.overseer_url = overseer_url.rstrip("/")
        self._last_log_count = 0
        self._last_log_hash = ""

    def fetch_logs(self, limit: int = 500) -> list[dict]:
        try:
            import urllib.request
            url = f"{self.overseer_url}/api/logs?limit={limit}"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("logs", [])
        except Exception as e:
            log(f"获取日志失败 ({self.overseer_url}): {e}", "WARN")
            return []

    def fetch_status(self) -> dict:
        try:
            import urllib.request
            url = f"{self.overseer_url}/api/status"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log(f"获取状态失败 ({self.overseer_url}): {e}", "WARN")
            return {}

    def has_new_content(self) -> bool:
        logs = self.fetch_logs(300)
        if not logs:
            return False
        raw = json.dumps(logs, sort_keys=True, ensure_ascii=False)
        new_hash = hashlib.md5(raw.encode()).hexdigest()
        if new_hash != self._last_log_hash:
            self._last_log_hash = new_hash
            self._last_log_count = len(logs)
            return True
        return False


ERROR_PATTERNS = [
    (re.compile(r"CUDA\s+out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|out of memory", re.I),
     "cuda_oom", "batch_half", "CUDA OOM 显存不足"),
    (re.compile(r"NaN\s+loss|nan.*loss|loss.*nan", re.I),
     "nan_loss", "lr_half_and_disable_amp", "NaN Loss 学习率过大"),
    (re.compile(r"TrainingFailed", re.I),
     "training_failed", "retry_train", "训练脚本崩溃"),
    (re.compile(r"ModuleNotFoundError|ImportError", re.I),
     "import_error", "install_package", "依赖缺失"),
    (re.compile(r"OSError.*No space|disk.*full", re.I),
     "disk_full", "clean_disk", "磁盘空间不足"),
    (re.compile(r"Connection\s+refused|ConnectionError|requests\.exceptions", re.I),
     "connection_refused", "restart_service", "服务不可达"),
    (re.compile(r"database\s+is\s+locked|sqlite3\.OperationalError", re.I),
     "db_locked", "retry_with_backoff", "数据库锁"),
    (re.compile(r"Permission\s+denied|PermissionError", re.I),
     "permission_denied", "fix_permissions", "权限不足"),
    (re.compile(r"ANTHROPIC_API_KEY\s+not\s+set", re.I),
     "api_key_missing", "fallback_ollama", "API密钥缺失"),
    (re.compile(r"git.*失败|git.*failed|git push 失败", re.I),
     "git_failed", "ignore", "Git操作失败 (非致命)"),
    (re.compile(r"JSON\s+decode|JSONDecodeError|JSON\s+parse\s+failed", re.I),
     "json_parse_error", "retry_or_fallback", "JSON解析失败"),
    (re.compile(r"researcher.*LLM.*failed|LLM\s+call\s+failed", re.I),
     "llm_failed", "restart_agent", "LLM调用失败"),
    (re.compile(r"Ollama\s+call\s+failed|Ollama\s+调用失败|Ollama\s+连接失败|Ollama.*不可达", re.I),
     "ollama_failed", "restart_service", "Ollama服务异常"),
    (re.compile(r"连续.*CDS.*(完全相同|停滞|卡死|波动)|疑似卡死|平台期", re.I),
     "stagnation", "force_strategy_change", "训练停滞/平台期"),
    (re.compile(r"连续\s*\d+\s*次.*失败|consecutive.*fail", re.I),
     "consecutive_fails", "escalate", "连续失败"),
    (re.compile(r"Dataset\s+Gate\s+拦截", re.I),
     "dataset_blocked", "escalate", "数据质量问题"),
    (re.compile(r"CRASH|FATAL|TRACEBACK|Traceback", re.I),
     "crash", "retry_train", "程序崩溃"),
    (re.compile(r"unreachable|Unreachable", re.I),
     "unreachable", "restart_agent", "LLM不可达"),
    (re.compile(r"httpx.*[Tt]imeout|HTTP.*[Tt]imeout|request.*[Tt]imeout|Connection.*[Tt]imeout|timed?\s*out|: Timeout", re.I),
     "timeout", "retry_with_longer_timeout", "网络超时"),
    (re.compile(r"STOP\s+signal|收到.*停止|KeyboardInterrupt", re.I),
     "stop_signal", "acknowledge", "人工停止信号"),
    (re.compile(r"磁盘剩余|RAM使用率.*超过|VRAM使用率.*超过", re.I),
     "resource_warning", "clean_disk", "资源警告"),
    (re.compile(r"violation|违规|架构.*违反", re.I),
     "architecture_violation", "fix_violation", "架构违规"),
]


class ErrorDetector:
    def __init__(self):
        self.seen_errors = {}
        self.fix_attempts = {}
        self.oom_tier = 0

    def detect_errors(self, logs: list[dict]) -> list[dict]:
        errors = []
        for entry in logs:
            line = entry.get("line", entry.get("msg", ""))
            level = entry.get("level", "")

            if level in ("CRIT", "WARN") or self._is_error_line(line):
                detected = self._classify_error(line, level)
                if detected:
                    error_key = detected["error_type"] + ":" + hashlib.md5(
                        line.encode()).hexdigest()[:12]
                    if error_key not in self.seen_errors:
                        self.seen_errors[error_key] = {
                            "first_seen": datetime.now().isoformat(), "count": 0}
                    self.seen_errors[error_key]["count"] += 1
                    self.seen_errors[error_key]["last_seen"] = datetime.now().isoformat()

                    count = self.seen_errors[error_key]["count"]
                    attempt_count = self.fix_attempts.get(error_key, 0)

                    if count > 1 and attempt_count >= MAX_CONSECUTIVE_FAILS:
                        detected["recommended_action"] = "escalate"
                        detected["confidence"] = 1.0
                        detected["root_cause"] = (
                            f"连续 {count} 次相同错误，"
                            f"已尝试修复 {attempt_count} 次失败，升级人工")

                    errors.append(detected)
        return errors

    def _is_error_line(self, line: str) -> bool:
        if not line:
            return False
        error_keywords = [
            "error", "crash", "fail", "traceback", "fatal",
            "exception", "unreachable", "violation", "oom",
            "nan", "timeout", "denied", "locked", "卡死", "停滞",
        ]
        line_lower = line.lower()
        return any(kw in line_lower for kw in error_keywords)

    def _classify_error(self, line: str, level: str) -> Optional[dict]:
        for pattern, error_type, action, description in ERROR_PATTERNS:
            if pattern.search(line):
                final_action = action
                needs_restart = action in (
                    "restart_agent", "batch_half", "lr_half_and_disable_amp",
                    "fix_config", "restart_service", "reduce_imgsz",
                    "switch_smaller_model", "force_strategy_change",
                )
                if error_type == "cuda_oom":
                    self.oom_tier += 1
                    if self.oom_tier >= 2:
                        final_action = "reduce_imgsz"
                        description = "CUDA OOM (已降 batch 仍不足)—降低输入分辨率"
                    if self.oom_tier >= 4:
                        final_action = "switch_smaller_model"
                        description = "CUDA OOM (降 batch+分辨率仍不足)—切换更小模型"
                    if self.oom_tier >= 6:
                        final_action = "escalate"
                        description = "CUDA OOM (所有方案无效)—升级人工"

                return {
                    "error_type": error_type,
                    "recommended_action": final_action,
                    "root_cause": description,
                    "confidence": 0.9,
                    "log_line": line[:300],
                    "log_level": level,
                    "needs_agent_restart": needs_restart,
                    "target_agent": self._infer_target_agent(line),
                }
        return None

    def _infer_target_agent(self, line: str) -> str:
        if "[Orchestrator]" in line or "[ORCHESTRATOR]" in line:
            return "Orchestrator"
        if "[SupervisorAgent]" in line or "[SUPERVISOR" in line:
            return "SupervisorAgent"
        if "[ArchitectureSupervisor]" in line or "[ARCH-SUPERVISOR" in line:
            return "ArchitectureSupervisor"
        if "[Dashboard" in line:
            return "Dashboard-v2"
        if "[总管]" in line:
            return "Overseer"
        return "Orchestrator"

    def record_fix_attempt(self, error_type: str, log_line: str):
        key = error_type + ":" + hashlib.md5(log_line.encode()).hexdigest()[:12]
        self.fix_attempts[key] = self.fix_attempts.get(key, 0) + 1

    def record_fix_success(self, error_type: str, log_line: str):
        key = error_type + ":" + hashlib.md5(log_line.encode()).hexdigest()[:12]
        self.fix_attempts[key] = 0
        if key in self.seen_errors:
            self.seen_errors[key]["count"] = 0
        if error_type == "cuda_oom":
            self.oom_tier = 0


class StuckDetector:
    """检测智能体卡死/停滞/假死

    Bug修复 (v3):
    1. _cds_snapshots 每轮都记录CDS快照 (原来只记录变化值，停滞时history不增长 → 永远检测不到)
    2. stuck_alerts_sent → 三个独立计数器 (_no_exp_alerts/_no_log_alerts/_cds_stagnant_alerts)
       避免不同类型互相抢占配额
    3. CDS停滞需 has_real_result (CDS>0.001) 才触发，避免0.0时误报
    4. _cds_improved_at 追踪CDS最后提升时间，报警带时间信息
    5. reset_after_stuck_fix(): 修复成功后重置CDS历史，避免修完立刻又误报
    """

    def __init__(self):
        self.last_experiment_count = -1
        self.last_cds = -1.0
        self.last_log_activity = time.time()
        self.last_experiment_change = time.time()
        self._cds_snapshots = []
        self._cds_round_count = 0
        self._cds_improved_at = time.time()
        self._no_exp_alerts = 0
        self._no_log_alerts = 0
        self._cds_stagnant_alerts = 0

    def update(self, status: dict, logs: list[dict]):
        health = status.get("health", {})
        exp_count = health.get("experiment_count", 0)
        best_cds = health.get("best_cds", 0)
        now = time.time()

        if logs:
            latest_ts = logs[-1].get("ts", "")
            if latest_ts:
                self.last_log_activity = now

        if exp_count > self.last_experiment_count and self.last_experiment_count >= 0:
            self.last_experiment_change = now
        self.last_experiment_count = exp_count

        self._cds_round_count += 1
        self._cds_snapshots.append(best_cds)
        if len(self._cds_snapshots) > 30:
            self._cds_snapshots = self._cds_snapshots[-30:]

        if best_cds > self.last_cds + 0.0005:
            self._cds_improved_at = now

        self.last_cds = best_cds

    def reset_after_stuck_fix(self, action: str):
        if action in ("restart_orchestrator", "force_strategy_change"):
            self._cds_snapshots = []
            self._cds_round_count = 0
            self._cds_improved_at = time.time()
            self._cds_stagnant_alerts = 0
            self._no_exp_alerts = 0
            self._no_log_alerts = 0
            self.last_experiment_change = time.time()
            self.last_log_activity = time.time()

    def detect(self, agents: list[dict]) -> list[dict]:
        now = time.time()
        alerts = []

        minutes_no_exp = (now - self.last_experiment_change) / 60
        if (self.last_experiment_count >= 0
                and minutes_no_exp >= STUCK_NO_EXPERIMENT_MINUTES
                and self._no_exp_alerts < 3):
            alerts.append({
                "error_type": "stuck_no_progress",
                "recommended_action": "restart_orchestrator",
                "root_cause": (
                    f"实验数 {minutes_no_exp:.0f}分钟未增加 "
                    f"(当前 {self.last_experiment_count} 次)，疑似卡死"
                ),
                "confidence": 0.85,
                "log_line": f"no new experiments for {minutes_no_exp:.0f} min",
                "log_level": "CRIT",
                "needs_agent_restart": True,
                "target_agent": "Orchestrator",
            })
            self._no_exp_alerts += 1

        minutes_no_log = (now - self.last_log_activity) / 60
        if minutes_no_log >= STUCK_NO_LOG_MINUTES and self._no_log_alerts < 3:
            alerts.append({
                "error_type": "stuck_no_log",
                "recommended_action": "restart_orchestrator",
                "root_cause": (
                    f"无日志输出 {minutes_no_log:.0f}分钟，"
                    f"Orchestrator 可能假死"
                ),
                "confidence": 0.8,
                "log_line": f"no log output for {minutes_no_log:.0f} min",
                "log_level": "CRIT",
                "needs_agent_restart": True,
                "target_agent": "Orchestrator",
            })
            self._no_log_alerts += 1

        if self._cds_round_count >= STUCK_CDS_STAGNANT_ROUNDS:
            recent = self._cds_snapshots[-STUCK_CDS_STAGNANT_ROUNDS:]
            cds_range = max(recent) - min(recent)
            has_real_result = max(recent) > 0.001
            if cds_range < 0.003 and has_real_result and self._cds_stagnant_alerts < 2:
                minutes_stagnant = (now - self._cds_improved_at) / 60
                alerts.append({
                    "error_type": "stuck_cds_stagnant",
                    "recommended_action": "force_strategy_change",
                    "root_cause": (
                        f"CDS 连续 {STUCK_CDS_STAGNANT_ROUNDS} 轮停滞 "
                        f"({min(recent):.4f}~{max(recent):.4f})，"
                        f"已 {minutes_stagnant:.0f} 分钟无提升，需切换策略"
                    ),
                    "confidence": 0.8,
                    "log_line": f"CDS stagnant for {minutes_stagnant:.0f} min: {recent[:3]}",
                    "log_level": "WARN",
                    "needs_agent_restart": True,
                    "target_agent": "Orchestrator",
                })
                self._cds_stagnant_alerts += 1

        for agent in agents:
            if agent.get("status") == "crashed":
                alerts.append({
                    "error_type": "agent_crashed",
                    "recommended_action": "restart_agent",
                    "root_cause": (
                        f"{agent.get('name', 'Unknown')} 已崩溃 "
                        f"(重启 {agent.get('restart_count', 0)} 次)"
                    ),
                    "confidence": 1.0,
                    "log_line": f"Agent {agent.get('name')} crashed",
                    "log_level": "CRIT",
                    "needs_agent_restart": True,
                    "target_agent": agent.get("name", "Orchestrator"),
                })

        return alerts


class StrategyAdjuster:
    """根据错误趋势调整整体方案"""

    def __init__(self):
        self.error_type_history = []
        self.strategy_changes = 0

    def record_error(self, error_type: str):
        self.error_type_history.append((time.time(), error_type))
        if len(self.error_type_history) > 50:
            self.error_type_history = self.error_type_history[-50:]

    def should_adjust(self) -> Optional[dict]:
        now = time.time()
        recent = [(ts, et) for ts, et in self.error_type_history
                   if now - ts < 600]

        oom_count = sum(1 for _, et in recent if et == "cuda_oom")
        crash_count = sum(1 for _, et in recent if et in ("crash", "training_failed"))
        nan_count = sum(1 for _, et in recent if et == "nan_loss")

        if oom_count >= 3:
            return {
                "trigger": "repeated_oom",
                "action": "switch_smaller_model",
                "reason": f"10分钟内 {oom_count} 次 OOM，切换更小模型",
            }

        if crash_count >= 4:
            return {
                "trigger": "repeated_crash",
                "action": "force_strategy_change",
                "reason": f"10分钟内 {crash_count} 次崩溃，强制切换策略",
            }

        if nan_count >= 3:
            return {
                "trigger": "repeated_nan",
                "action": "force_strategy_change",
                "reason": f"10分钟内 {nan_count} 次 NaN Loss，切换优化方案",
            }

        return None


class FixExecutor:
    def __init__(self, overseer_url: str = DEFAULT_OVERSEER_URL):
        self.overseer_url = overseer_url.rstrip("/")
        self.train_script = TRAIN_SCRIPT
        self.project_root = PROJECT_ROOT

    def execute(self, diagnosis: dict, error_info: dict) -> dict:
        action = diagnosis.get("recommended_action", "escalate")
        target = error_info.get("target_agent",
                                 diagnosis.get("target_agent", "Orchestrator"))
        result = {"action": action, "target": target,
                   "success": False, "message": "", "details": {}}

        executor_map = {
            "batch_half": self._fix_batch_half,
            "reduce_imgsz": self._fix_reduce_imgsz,
            "switch_smaller_model": self._fix_switch_smaller_model,
            "lr_half_and_disable_amp": self._fix_lr_half_and_disable_amp,
            "retry_train": self._fix_retry_train,
            "restart_agent": self._fix_restart_agent,
            "restart_orchestrator": self._fix_restart_orchestrator,
            "clean_disk": self._fix_clean_disk,
            "install_package": self._fix_install_package,
            "fix_config": self._fix_config,
            "restart_service": self._fix_restart_service,
            "retry_with_backoff": self._fix_retry_with_backoff,
            "retry_with_longer_timeout": self._fix_retry_with_longer_timeout,
            "retry_or_fallback": self._fix_retry_or_fallback,
            "fallback_ollama": self._fix_fallback_ollama,
            "fix_permissions": self._fix_permissions,
            "fix_violation": self._fix_violation,
            "force_strategy_change": self._fix_force_strategy_change,
            "escalate": self._fix_escalate,
            "ignore": self._fix_ignore,
            "acknowledge": self._fix_acknowledge,
        }

        handler = executor_map.get(action, self._fix_escalate)
        try:
            result = handler(diagnosis, error_info, target)
        except Exception as e:
            result["success"] = False
            result["message"] = f"执行修复时异常: {e}"
            result["details"]["traceback"] = traceback.format_exc()
        return result

    def _read_param(self, key: str, default=None):
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

    def _write_train_param(self, key: str, value):
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
            log(f"修改 {key} 失败: {e}", "WARN")
            return False

    def _agent_api(self, name: str, action: str) -> bool:
        try:
            import urllib.request
            url = f"{self.overseer_url}/api/agent/{urllib.request.quote(name)}/{action}"
            req = urllib.request.Request(url, method="POST")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("ok", False)
        except Exception as e:
            log(f"API 操作失败 ({name}/{action}): {e}", "WARN")
            return False

    def _get_current_model(self) -> str:
        val = self._read_param("MODEL", "yolo12s.pt")
        return str(val).strip('"').strip("'")

    def _get_current_imgsz(self) -> int:
        return self._read_param("IMGSZ", 1280) or 1280

    def _get_current_batch(self) -> int:
        return self._read_param("BATCH", 8) or 8

    def _fix_batch_half(self, diagnosis, error_info, target) -> dict:
        current_batch = self._get_current_batch()
        new_batch = max(OOM_TIER1_BATCH_MIN, current_batch // 2)
        ok = self._write_train_param("BATCH", new_batch)
        log(f"batch_half: BATCH {current_batch} → {new_batch}", "FIX")
        if ok:
            self._agent_api(target, "restart")
        return {"action": "batch_half", "target": target, "success": ok,
                "message": f"BATCH {current_batch} → {new_batch}",
                "details": {"old_batch": current_batch, "new_batch": new_batch}}

    def _fix_reduce_imgsz(self, diagnosis, error_info, target) -> dict:
        current_imgsz = self._get_current_imgsz()
        new_imgsz = max(640, current_imgsz - 320)
        ok = self._write_train_param("IMGSZ", new_imgsz)
        log(f"reduce_imgsz: IMGSZ {current_imgsz} → {new_imgsz}", "FIX")
        if ok:
            self._agent_api(target, "restart")
        return {"action": "reduce_imgsz", "target": target, "success": ok,
                "message": f"IMGSZ {current_imgsz} → {new_imgsz}",
                "details": {"old_imgsz": current_imgsz, "new_imgsz": new_imgsz}}

    def _fix_switch_smaller_model(self, diagnosis, error_info, target) -> dict:
        current_model = self._get_current_model()
        try:
            idx = MODEL_SIZES.index(current_model)
            new_idx = max(0, idx - 1)
        except ValueError:
            new_idx = 0
        new_model = MODEL_SIZES[new_idx]
        if new_model == current_model:
            return {"action": "switch_smaller_model", "target": target,
                    "success": False, "message": f"已是最小模型 {current_model}，无法继续缩小"}
        ok = self._write_train_param("MODEL", new_model)
        self._write_train_param("BATCH", max(1, self._get_current_batch()))
        self._write_train_param("IMGSZ", self._get_current_imgsz())
        log(f"switch_smaller_model: {current_model} → {new_model}", "STRATEGY")
        if ok:
            self._agent_api(target, "restart")
        return {"action": "switch_smaller_model", "target": target, "success": ok,
                "message": f"模型 {current_model} → {new_model}",
                "details": {"old_model": current_model, "new_model": new_model}}

    def _fix_force_strategy_change(self, diagnosis, error_info, target) -> dict:
        current_model = self._get_current_model()
        current_imgsz = self._get_current_imgsz()
        current_batch = self._get_current_batch()

        if current_imgsz > 960:
            new_imgsz = max(640, current_imgsz - 320)
            self._write_train_param("IMGSZ", new_imgsz)
            msg = f"强制策略调整: IMGSZ {current_imgsz}→{new_imgsz}"
            new_batch = max(2, current_batch)
            self._write_train_param("BATCH", new_batch)
            msg += f", BATCH→{new_batch}"
        else:
            try:
                idx = MODEL_SIZES.index(current_model)
                new_idx = max(0, idx - 1)
            except ValueError:
                new_idx = 0
            new_model = MODEL_SIZES[new_idx]
            self._write_train_param("MODEL", new_model)
            msg = f"强制策略调整: MODEL {current_model}→{new_model}"

        log(f"force_strategy_change: {msg}", "STRATEGY")
        ok = self._agent_api(target, "restart")
        return {"action": "force_strategy_change", "target": target,
                "success": ok, "message": msg,
                "details": {"model": self._get_current_model(),
                             "imgsz": self._get_current_imgsz(),
                             "batch": self._get_current_batch()}}

    def _fix_restart_orchestrator(self, diagnosis, error_info, target) -> dict:
        log(f"restart_orchestrator: 重启整个战斗群", "STUCK")
        self._agent_api("Orchestrator", "restart")
        time.sleep(2)
        self._agent_api("SupervisorAgent", "restart")
        return {"action": "restart_orchestrator", "target": "Orchestrator+",
                "success": True, "message": "已重启 Orchestrator + 监督智能体"}

    def _fix_lr_half_and_disable_amp(self, diagnosis, error_info, target) -> dict:
        current_lr = self._read_param("LR0", 0.01) or 0.01
        new_lr = max(1e-5, current_lr * 0.5)
        ok_lr = self._write_train_param("LR0", new_lr)
        ok_amp = self._write_train_param("AMP", False)
        ok = ok_lr and ok_amp
        log(f"lr_half_and_disable_amp: LR0 {current_lr} → {new_lr}, AMP→False", "FIX")
        if ok:
            self._agent_api(target, "restart")
        return {"action": "lr_half_and_disable_amp", "target": target, "success": ok,
                "message": f"LR0 {current_lr} → {new_lr}, AMP disabled"}

    def _fix_retry_train(self, diagnosis, error_info, target) -> dict:
        log(f"retry_train: 重启 {target}", "FIX")
        ok = self._agent_api(target, "restart")
        return {"action": "retry_train", "target": target, "success": ok,
                "message": f"已请求重启 {target}"}

    def _fix_restart_agent(self, diagnosis, error_info, target) -> dict:
        log(f"restart_agent: {target}", "FIX")
        ok = self._agent_api(target, "restart")
        return {"action": "restart_agent", "target": target, "success": ok,
                "message": f"{target} 重启{' 成功 ' if ok else ' 失败 '}"}

    def _fix_clean_disk(self, diagnosis, error_info, target) -> dict:
        cleaned_bytes = 0
        dirs_to_clean = [os.path.join(self.project_root, "autoresearch_runs")]
        for d in dirs_to_clean:
            if os.path.exists(d):
                try:
                    size_before = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, files in os.walk(d) for f in files)
                    shutil.rmtree(d)
                    cleaned_bytes += size_before
                except Exception as e:
                    log(f"清理 {d} 失败: {e}", "WARN")
        for d in [os.path.join(self.project_root, "__pycache__"),
                   os.path.join(V2_ROOT, "__pycache__")]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
        log(f"clean_disk: 清理了约 {cleaned_bytes / 1024 / 1024:.1f} MB", "FIX")
        return {"action": "clean_disk", "target": target, "success": True,
                "message": f"清理了约 {cleaned_bytes / 1024 / 1024:.1f} MB"}

    def _fix_install_package(self, diagnosis, error_info, target) -> dict:
        fix_detail = diagnosis.get("fix_detail", {})
        package = fix_detail.get("package", "")
        log_line = error_info.get("log_line", "")
        if not package:
            m = re.search(r"No module named '(\w+)'", log_line)
            if m:
                package = m.group(1)
        if not package:
            return {"action": "install_package", "target": target,
                    "success": False, "message": "无法从错误信息中提取缺失的包名"}
        log(f"install_package: pip install {package}", "FIX")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True, text=True, timeout=120, cwd=self.project_root)
            ok = result.returncode == 0
            return {"action": "install_package", "target": target, "success": ok,
                    "message": f"pip install {package} {' 成功 ' if ok else ' 失败 '}"}
        except Exception as e:
            return {"action": "install_package", "target": target,
                    "success": False, "message": f"pip install 异常: {e}"}

    def _fix_config(self, diagnosis, error_info, target) -> dict:
        fix_detail = diagnosis.get("fix_detail", {})
        if not fix_detail:
            return {"action": "fix_config", "target": target,
                    "success": False, "message": "未提供 fix_detail"}
        ok_all = True
        for key, value in fix_detail.items():
            if isinstance(value, (int, float, bool, str)):
                if not self._write_train_param(key, value):
                    ok_all = False
        log(f"fix_config: {fix_detail}", "FIX")
        if ok_all:
            self._agent_api(target, "restart")
        return {"action": "fix_config", "target": target, "success": ok_all,
                "message": f"已修改参数: {fix_detail}", "details": fix_detail}

    def _fix_restart_service(self, diagnosis, error_info, target) -> dict:
        log("restart_service: 检查 Ollama 状态...", "FIX")
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return {"action": "restart_service", "target": target, "success": True,
                        "message": f"Ollama 已在线, 可用模型: {models}"}
        except Exception:
            log("restart_service: Ollama 不可达, 尝试重启...", "WARN")
            try:
                subprocess.run(["ollama", "serve"], capture_output=True, timeout=10)
            except Exception:
                pass
            time.sleep(3)
            try:
                import urllib.request
                req = urllib.request.Request("http://localhost:11434/api/tags")
                req.add_header("Accept", "application/json")
                with urllib.request.urlopen(req, timeout=5):
                    return {"action": "restart_service", "target": target, "success": True,
                            "message": "Ollama 已恢复"}
            except Exception:
                return {"action": "restart_service", "target": target, "success": False,
                        "message": "Ollama 无法恢复，请手动启动"}

    def _fix_retry_with_backoff(self, diagnosis, error_info, target) -> dict:
        wait_sec = 5
        log(f"retry_with_backoff: 等待 {wait_sec}s...", "FIX")
        time.sleep(wait_sec)
        ok = self._agent_api(target, "restart")
        return {"action": "retry_with_backoff", "target": target, "success": ok,
                "message": f"等待 {wait_sec}s 后重启 {target}"}

    def _fix_retry_with_longer_timeout(self, diagnosis, error_info, target) -> dict:
        log(f"retry_with_longer_timeout: 重启 {target}", "FIX")
        ok = self._agent_api(target, "restart")
        return {"action": "retry_with_longer_timeout", "target": target,
                "success": ok, "message": f"已重启 {target}"}

    def _fix_retry_or_fallback(self, diagnosis, error_info, target) -> dict:
        log("retry_or_fallback: 切换 LLM 后端为 Ollama", "FIX")
        self._write_train_param("LLM_BACKEND", "ollama")
        ok = self._agent_api(target, "restart")
        return {"action": "retry_or_fallback", "target": target, "success": ok,
                "message": "已切换到 Ollama 后端并重启"}

    def _fix_fallback_ollama(self, diagnosis, error_info, target) -> dict:
        log("fallback_ollama: 切换到 Ollama", "FIX")
        self._write_train_param("LLM_BACKEND", "ollama")
        ok = self._agent_api(target, "restart")
        return {"action": "fallback_ollama", "target": target, "success": ok,
                "message": "已切换到 Ollama 后端"}

    def _fix_permissions(self, diagnosis, error_info, target) -> dict:
        fixed = 0
        if os.path.exists(STATE_DIR):
            for root, dirs, files in os.walk(STATE_DIR):
                for f in files:
                    try:
                        os.chmod(os.path.join(root, f), 0o666)
                        fixed += 1
                    except Exception:
                        pass
        for f in [TRAIN_SCRIPT, os.path.join(PROJECT_ROOT, "results.tsv")]:
            if os.path.exists(f):
                try:
                    os.chmod(f, 0o666)
                    fixed += 1
                except Exception:
                    pass
        return {"action": "fix_permissions", "target": target, "success": True,
                "message": f"已尝试修复 {fixed} 个文件的权限"}

    def _fix_violation(self, diagnosis, error_info, target) -> dict:
        fix_detail = diagnosis.get("fix_detail", {})
        if fix_detail:
            return self._fix_config(diagnosis, error_info, target)
        return {"action": "fix_violation", "target": target, "success": True,
                "message": "无具体修复方案，已记录"}

    def _fix_escalate(self, diagnosis, error_info, target) -> dict:
        root_cause = diagnosis.get("root_cause_hypothesis",
                                    error_info.get("root_cause", "未知"))
        log(f"escalate: 升级人工 — {root_cause}", "ESCALATE")
        return {"action": "escalate", "target": target, "success": True,
                "message": f"已升级人工: {root_cause}"}

    def _fix_ignore(self, diagnosis, error_info, target) -> dict:
        return {"action": "ignore", "target": target, "success": True,
                "message": "非致命错误，已忽略"}

    def _fix_acknowledge(self, diagnosis, error_info, target) -> dict:
        return {"action": "acknowledge", "target": target, "success": True,
                "message": "预期行为，已确认"}


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


class OverseerDoctor:
    def __init__(self, config: dict = None):
        config = config or {}
        self.overseer_url = config.get("overseer_url", DEFAULT_OVERSEER_URL)
        self.poll_interval = config.get("poll_interval", DEFAULT_POLL_INTERVAL)
        self.llm_backend = config.get("llm_backend", "auto")
        self.ollama_model = config.get("ollama_model", "gemma3:4b")
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.anthropic_model = config.get("anthropic_model", "claude-haiku-4-20250414")
        self.max_llm_retries = config.get("max_llm_retries", 3)

        self.fetcher = LogFetcher(self.overseer_url)
        self.detector = ErrorDetector()
        self.stuck_detector = StuckDetector()
        self.strategy_adjuster = StrategyAdjuster()
        self.executor = FixExecutor(self.overseer_url)
        self.system_prompt = load_prompt()
        self.running = True
        self.total_fixes = 0
        self.successful_fixes = 0
        self.escalated_count = 0
        self.stuck_fixes = 0
        self.strategy_changes = 0
        self.last_health_snapshot = {}
        self._consecutive_timeouts = 0

        os.makedirs(os.path.dirname(DOCTOR_ACTION_LOG), exist_ok=True)
        os.makedirs(os.path.dirname(DOCTOR_STATE_FILE), exist_ok=True)

    def _persist_state(self):
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "total_fixes": self.total_fixes,
                "successful_fixes": self.successful_fixes,
                "escalated_count": self.escalated_count,
                "stuck_fixes": self.stuck_fixes,
                "strategy_changes": self.strategy_changes,
                "oom_tier": self.detector.oom_tier,
                "last_experiment_count": self.stuck_detector.last_experiment_count,
                "last_cds": self.stuck_detector.last_cds,
            }
            with open(DOCTOR_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def diagnose_with_llm(self, error_info: dict) -> dict:
        lines = [
            "## 错误信息",
            f"- 类型: {error_info.get('error_type', 'unknown')}",
            f"- 级别: {error_info.get('log_level', 'UNKNOWN')}",
            f"- 日志: {error_info.get('log_line', '')[:500]}",
            f"- 目标智能体: {error_info.get('target_agent', 'unknown')}",
        ]
        user_prompt = "\n".join(lines) + "\n\n请诊断并推荐修复方案。严格输出 JSON。"
        for attempt in range(self.max_llm_retries):
            response = None
            if self.llm_backend in ("anthropic", "auto"):
                response, _ = _call_anthropic(self.system_prompt, user_prompt, self.anthropic_model)
            if response is None and self.llm_backend in ("ollama", "auto"):
                response, _ = _call_ollama(self.system_prompt, user_prompt, self.ollama_model, self.ollama_url)
            if response is None:
                log(f"LLM 调用失败 (attempt {attempt + 1})", "WARN")
                continue
            diagnosis = extract_first_json(response)
            if diagnosis is None:
                continue
            return diagnosis
        return {"root_cause_hypothesis": "LLM 连续失败，无法诊断",
                "confidence": 0.0, "recommended_action": "escalate"}

    def process_log_errors(self, logs: list[dict]) -> int:
        errors = self.detector.detect_errors(logs)
        if not errors:
            return 0
        fixes_applied = 0
        for error in errors:
            action = error.get("recommended_action", "escalate")
            self.strategy_adjuster.record_error(error["error_type"])

            if action in ("ignore", "acknowledge"):
                log(f"[{error['error_type']}]: {error['root_cause'][:80]} → {action}", "WATCH")
                continue

            log(f"[{error['error_type']}]: {error['root_cause'][:80]} → {action}",
                "WARN" if action != "escalate" else "CRIT")

            diagnosis = {
                "recommended_action": action,
                "root_cause_hypothesis": error["root_cause"],
                "confidence": error.get("confidence", 0.9),
                "needs_agent_restart": error.get("needs_agent_restart", False),
                "target_agent": error.get("target_agent", "Orchestrator"),
            }
            if action == "escalate":
                diagnosis = self.diagnose_with_llm(error)
                action = diagnosis.get("recommended_action", "escalate")

            result = self.executor.execute(diagnosis, error)
            self.total_fixes += 1
            if result["success"]:
                self.successful_fixes += 1
                self.detector.record_fix_success(error["error_type"], error.get("log_line", ""))
                log(f"修复成功: {result['message']}", "OK")
            else:
                self.detector.record_fix_attempt(error["error_type"], error.get("log_line", ""))
                log(f"修复失败: {result['message']}", "WARN")

            if action == "escalate":
                self.escalated_count += 1

            write_action_record({"error": error, "diagnosis": diagnosis, "result": result})
            fixes_applied += 1
        return fixes_applied

    def process_stuck(self, status: dict) -> int:
        agents = status.get("agents", [])
        stuck_alerts = self.stuck_detector.detect(agents)
        if not stuck_alerts:
            return 0
        fixes = 0
        for alert in stuck_alerts:
            log(f"[卡死检测] {alert['root_cause']}", "STUCK")
            result = self.executor.execute(
                {"recommended_action": alert["recommended_action"],
                 "root_cause_hypothesis": alert["root_cause"],
                 "confidence": alert["confidence"]},
                alert)
            if result["success"]:
                self.stuck_fixes += 1
                log(f"卡死修复成功: {result['message']}", "OK")
                self.stuck_detector.reset_after_stuck_fix(alert["recommended_action"])
            else:
                log(f"卡死修复失败: {result['message']}", "WARN")
            write_action_record({"type": "stuck", "alert": alert, "result": result})
            fixes += 1
        return fixes

    def process_strategy(self) -> int:
        adjustment = self.strategy_adjuster.should_adjust()
        if not adjustment:
            return 0
        log(f"策略调整: {adjustment['trigger']} — {adjustment['reason']}", "STRATEGY")
        result = self.executor.execute(
            {"recommended_action": adjustment["action"],
             "root_cause_hypothesis": adjustment["reason"],
             "confidence": 0.85},
            {"target_agent": "Orchestrator",
             "error_type": adjustment["trigger"],
             "log_line": adjustment["reason"],
             "log_level": "WARN"})
        if result["success"]:
            self.strategy_changes += 1
        write_action_record({"type": "strategy", "adjustment": adjustment, "result": result})
        return 1 if result["success"] else 0

    def run_once(self) -> dict:
        logs = self.fetcher.fetch_logs(500)
        status = self.fetcher.fetch_status()

        if not status and not logs:
            self._consecutive_timeouts += 1
            log(f"无日志可读取（总管 Web UI 可能未运行）[{self._consecutive_timeouts}/5]", "WARN")
            if self._consecutive_timeouts >= 5:
                log("Web UI 连续 5 次超时，强制重启 Orchestrator", "STUCK")
                result = self.executor.execute(
                    {"recommended_action": "restart_orchestrator",
                     "root_cause_hypothesis": "Web UI 连续超时，Orchestrator 假死",
                     "confidence": 0.9},
                    {"target_agent": "Orchestrator",
                     "error_type": "overseer_timeout",
                     "log_line": "Web UI timeout x5",
                     "log_level": "CRIT"})
                if result["success"]:
                    self.stuck_fixes += 1
                    self._consecutive_timeouts = 0
                    time.sleep(5)
                    self._wait_for_overseer(max_wait=30)
                write_action_record({"type": "stuck", "alert": {"error_type": "overseer_timeout"}, "result": result})
            return {"logs_count": 0, "errors_found": 0, "fixes_applied": 0,
                    "stuck_fixes": 0, "strategy_changes": 0,
                    "overseer_status": "unreachable"}

        self._consecutive_timeouts = 0

        agents = status.get("agents", [])
        health = status.get("health", {})

        self.stuck_detector.update(status, logs)

        log_fix_count = self.process_log_errors(logs)
        stuck_fix_count = self.process_stuck(status)
        strategy_fix_count = self.process_strategy()

        total_fixes = log_fix_count + stuck_fix_count + strategy_fix_count
        if total_fixes > 0:
            log(f"本轮: 日志修复={log_fix_count} "
                f"卡死修复={stuck_fix_count} "
                f"策略调整={strategy_fix_count}", "FIX")
        else:
            exp_count = health.get("experiment_count", 0)
            best_cds = health.get("best_cds", 0)
            log(f"本轮无异常 | 实验: {exp_count} | CDS: {best_cds:.4f}", "OK")

        self._persist_state()

        agents_alive = sum(1 for a in agents if a.get("status") == "running")
        agents_total = len(agents)
        return {
            "logs_count": len(logs),
            "errors_found": log_fix_count,
            "fixes_applied": log_fix_count,
            "stuck_fixes": stuck_fix_count,
            "strategy_changes": strategy_fix_count,
            "overseer_status": "ok" if status else "unreachable",
            "experiments": health.get("experiment_count", 0),
            "best_cds": health.get("best_cds", 0),
            "agents_alive": f"{agents_alive}/{agents_total}",
        }

    def _wait_for_overseer(self, max_wait: int = 60) -> bool:
        start = time.time()
        while time.time() - start < max_wait:
            try:
                import urllib.request
                url = f"{self.overseer_url}/api/health"
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/json")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("ok"):
                        log(f"已连接到总管 Web UI ({self.overseer_url})", "OK")
                        return True
            except Exception:
                pass
            elapsed = time.time() - start
            log(f"等待总管 Web UI 就绪... ({elapsed:.0f}s/{max_wait}s)", "WARN")
            time.sleep(3)
        log(f"总管 Web UI 在 {max_wait}s 内未就绪，将轮询重试", "WARN")
        return False

    def watch(self):
        log("=" * 60)
        log("  Overseer Doctor V2 — 总管诊断医生启动", "INFO")
        log("=" * 60)
        log(f"总管 URL: {self.overseer_url}")
        log(f"轮询间隔: {self.poll_interval}s")
        log(f"卡死检测: 无实验>{STUCK_NO_EXPERIMENT_MINUTES}min | 无日志>{STUCK_NO_LOG_MINUTES}min")
        log(f"OOM分级: batch_half → reduce_imgsz → switch_smaller_model")
        log("")

        self._wait_for_overseer(max_wait=45)

        try:
            while self.running:
                result = self.run_once()
                elapsed = 0
                while elapsed < self.poll_interval and self.running:
                    time.sleep(1)
                    elapsed += 1
        except KeyboardInterrupt:
            log("收到中断信号", "WARN")
        finally:
            self._print_summary()
            self._persist_state()
            log("Overseer Doctor 已停止", "OK")

    def _print_summary(self):
        log("")
        log("=" * 60)
        log("  Overseer Doctor V2 — 运行总结")
        log("=" * 60)
        log(f"日志修复: {self.successful_fixes}/{self.total_fixes} 成功")
        log(f"卡死修复: {self.stuck_fixes} 次")
        log(f"策略调整: {self.strategy_changes} 次")
        log(f"升级人工: {self.escalated_count} 次")
        total_all = self.total_fixes + self.stuck_fixes + self.strategy_changes
        success_all = self.successful_fixes + self.stuck_fixes + self.strategy_changes
        if total_all > 0:
            log(f"整体成功率: {success_all / total_all * 100:.1f}%")
        log(f"修复日志: {DOCTOR_ACTION_LOG}")
        log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Overseer Doctor V2")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--overseer-url", type=str, default=DEFAULT_OVERSEER_URL)
    parser.add_argument("--llm-backend", type=str, default="auto",
                        choices=["auto", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", type=str, default="gemma3:4b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--anthropic-model", type=str, default="claude-haiku-4-20250414")
    parser.add_argument("--once", action="store_true", help="一次性诊断")
    args = parser.parse_args()

    config = {
        "overseer_url": args.overseer_url,
        "poll_interval": args.interval,
        "llm_backend": args.llm_backend,
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "anthropic_model": args.anthropic_model,
    }

    doctor = OverseerDoctor(config)
    if args.watch:
        doctor.watch()
    else:
        result = doctor.run_once()
        print()
        print("=" * 60)
        print("  Overseer Doctor V2 — 一次性诊断完成")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("=" * 60)


if __name__ == "__main__":
    main()