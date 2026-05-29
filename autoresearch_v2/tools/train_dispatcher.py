"""
train_dispatcher.py — 包装 train.py 的调度器

职责:
  - 接收 Orchestrator 的决策指令
  - 修改 train.py 配置
  - 启动训练进程
  - 记录结果到状态库
"""

import json
import os
import re
import sys
import time
import subprocess
import threading
from datetime import datetime
from typing import Optional

TRAIN_SCRIPT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "train.py"
))
LOG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "run.log"
))
TRAIN_TIMEOUT = 7200


class TrainingFailed(Exception):
    def __init__(self, message: str, return_code: int = -1, log_tail: str = ""):
        super().__init__(message)
        self.return_code = return_code
        self.log_tail = log_tail


def _sanitize_python_value(value):
    """Convert any config value into a valid Python literal string.

    Handles numpy scalars, plain floats / ints / bools, and strings,
    always producing syntactically-valid Python source.
    """
    try:
        float_val = float(value)
        if float_val != float_val:
            return "float('nan')"
        if float_val == float("inf"):
            return "float('inf')"
        if float_val == float("-inf"):
            return "float('-inf')"
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return repr(value)

    if isinstance(value, (int, float)):
        return repr(value)

    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    if hasattr(value, "item"):
        return _sanitize_python_value(value.item())

    try:
        return repr(float(value))
    except (TypeError, ValueError):
        pass

    return repr(str(value))


def apply_config_diff(config_diff: dict, train_script: str = TRAIN_SCRIPT) -> bool:
    if not config_diff:
        return True

    with open(train_script, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for key, value in config_diff.items():
            if re.match(rf"^{key}\s*=", stripped):
                safe_val = _sanitize_python_value(value)
                lines[i] = f"{key} = {safe_val}\n"
                modified = True
                break

    if modified:
        with open(train_script, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return modified


def read_current_config(train_script: str = TRAIN_SCRIPT) -> dict:
    config = {}
    try:
        with open(train_script, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r'^(\w+)\s*=\s*(.+)$', line.strip())
                if m:
                    key = m.group(1)
                    val_str = m.group(2).strip()
                    if key.isupper() and len(key) > 1:
                        try:
                            val = eval(val_str)
                        except Exception:
                            val = val_str
                        config[key] = val
    except FileNotFoundError:
        pass
    return config


def run_training(train_script: str = TRAIN_SCRIPT,
                 log_file: str = LOG_FILE,
                 timeout: int = TRAIN_TIMEOUT,
                 events_logger=None,
                 quiet: bool = False) -> tuple[bool, str]:
    """Run train.py as a subprocess, streaming stdout LIVE to the terminal.

    The Windows GPU operator running the orchestrator from a single
    terminal must see training progress in real time — they should NOT
    have to open run.log in another window.

    Each subprocess line is:
      - written to log_file (unchanged from before; tooling expects it)
      - printed to sys.stdout immediately, flushed (NEW)
      - if it looks 'interesting' (epoch summary, error, metric line) AND
        an events_logger is passed, also emitted as a train_progress
        event into activity_events.jsonl.

    Pass quiet=True to suppress the live print (e.g. for unit tests).
    """
    python = sys.executable
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    proc = subprocess.Popen(
        [python, "-u", train_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,                  # line-buffered
    )

    timed_out = {"v": False}

    def watchdog():
        time.sleep(timeout)
        if proc.poll() is None:
            timed_out["v"] = True
            try:
                proc.kill()
                if not quiet:
                    print(
                        f"\n[train_dispatcher] training exceeded {timeout}s — killed.\n",
                        flush=True,
                    )
            except Exception:
                pass

    threading.Thread(target=watchdog, daemon=True).start()

    # Tee subprocess output: log file + (live) terminal + (selective) events
    interesting_re = re.compile(
        r"(epoch|cds:|map|precision:|recall:|inference_ms|pipeline_|"
        r"target_met|fatal|error|traceback)",
        re.IGNORECASE,
    )
    with open(log_file, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if not quiet:
                # Live to terminal — flushed every line so Windows
                # operators see real-time progress in their cmd/PowerShell.
                try:
                    print(line.rstrip(), flush=True)
                except UnicodeEncodeError:
                    print(line.rstrip().encode('ascii', errors='replace').decode('ascii'), flush=True)
            if events_logger is not None and interesting_re.search(line):
                try:
                    events_logger.emit("train_progress", line=line.rstrip())
                except Exception:
                    pass  # never let events writer break training stream

    proc.wait()

    log_tail = ""
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            log_tail = "".join(all_lines[-200:])
        except Exception:
            pass

    success = proc.returncode == 0 and not timed_out["v"]
    return success, log_tail


class TrainDispatcher:
    def __init__(self, state=None, train_script: str = TRAIN_SCRIPT,
                 log_file: str = LOG_FILE, events_logger=None):
        self.state = state
        self.train_script = train_script
        self.log_file = log_file
        self.events_logger = events_logger

    def dispatch(self, run_id: str, decision: dict) -> str:
        config_diff = decision.get("config_diff", {})
        if config_diff:
            apply_config_diff(config_diff, self.train_script)

        current_config = read_current_config(self.train_script)

        if self.state:
            self.state.insert_experiment(
                run_id=run_id,
                config_json=current_config,
                status="running",
            )

        if self.events_logger:
            try:
                self.events_logger.emit(
                    "train_start",
                    run_id=run_id,
                    config_patch=config_diff,
                    log_file=self.log_file,
                )
            except Exception:
                pass

        start_time = time.time()
        success, log_tail = run_training(
            self.train_script, self.log_file,
            events_logger=self.events_logger,
        )
        elapsed_min = (time.time() - start_time) / 60.0

        if self.events_logger:
            try:
                self.events_logger.emit(
                    "train_complete",
                    run_id=run_id,
                    success=success,
                    duration_min=round(elapsed_min, 2),
                )
            except Exception:
                pass

        if not success:
            if self.state:
                self.state.update_experiment_status(
                    run_id, "failed",
                    gpu_minutes=elapsed_min,
                    cost_usd=elapsed_min * 0.01,
                )
            raise TrainingFailed(
                f"Training failed (rc={-1})",
                return_code=-1,
                log_tail=log_tail,
            )

        if self.state:
            self.state.update_experiment_status(
                run_id, "completed",
                gpu_minutes=elapsed_min,
                cost_usd=elapsed_min * 0.01,
            )

        best_pt = self._find_best_weights()
        if best_pt and self.state:
            self.state.insert_artifact(run_id, "weights", best_pt)

        return best_pt or ""

    def _find_best_weights(self) -> Optional[str]:
        runs_dir = os.path.normpath(os.path.join(
            os.path.dirname(self.train_script), "autoresearch_runs"
        ))
        if not os.path.exists(runs_dir):
            return None

        candidates = []
        for root, dirs, files in os.walk(runs_dir):
            for f in files:
                if f == "best.pt":
                    candidates.append(os.path.join(root, f))

        if not candidates:
            return None

        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return candidates[0]
