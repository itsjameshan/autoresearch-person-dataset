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


def apply_config_diff(config_diff: dict, train_script: str = TRAIN_SCRIPT) -> bool:
    if not config_diff:
        return True

    with open(train_script, "r", encoding="utf-8") as f:
        lines = f.readlines()

    modified = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in config_diff.items():
            if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                if isinstance(value, str):
                    lines[i] = f'{key} = "{value}"\n'
                elif isinstance(value, bool):
                    lines[i] = f"{key} = {value}\n"
                else:
                    lines[i] = f"{key} = {value}\n"
                modified = True

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
                 timeout: int = TRAIN_TIMEOUT) -> tuple[bool, str]:
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
    )

    def watchdog():
        time.sleep(timeout)
        try:
            proc.kill()
        except Exception:
            pass

    threading.Thread(target=watchdog, daemon=True).start()

    with open(log_file, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)

    proc.wait()

    log_tail = ""
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            log_tail = "".join(all_lines[-200:])
        except Exception:
            pass

    return proc.returncode == 0, log_tail


class TrainDispatcher:
    def __init__(self, state=None, train_script: str = TRAIN_SCRIPT,
                 log_file: str = LOG_FILE):
        self.state = state
        self.train_script = train_script
        self.log_file = log_file

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

        start_time = time.time()
        success, log_tail = run_training(self.train_script, self.log_file)
        elapsed_min = (time.time() - start_time) / 60.0

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
