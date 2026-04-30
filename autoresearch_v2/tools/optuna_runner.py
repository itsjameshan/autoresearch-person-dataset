"""
optuna_runner.py — Optuna HPO 超参搜索模块

替代 Researcher 手动试参数，用 TPE 搜索高效探索参数空间。
"""

import json
import os
import sys
import time
import subprocess
import threading
from typing import Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


DEFAULT_SEARCH_SPACE = {
    "LR0": {"type": "float", "low": 0.001, "high": 0.02, "log": True},
    "LRF": {"type": "float", "low": 0.0005, "high": 0.02, "log": True},
    "BOX": {"type": "float", "low": 5.0, "high": 20.0},
    "CLS": {"type": "float", "low": 0.3, "high": 1.5},
    "MOSAIC": {"type": "float", "low": 0.3, "high": 1.0},
    "MIXUP": {"type": "float", "low": 0.0, "high": 0.3},
    "COPY_PASTE": {"type": "float", "low": 0.0, "high": 0.3},
    "DEGREES": {"type": "float", "low": 0.0, "high": 20.0},
    "HSV_H": {"type": "float", "low": 0.0, "high": 0.1},
    "HSV_S": {"type": "float", "low": 0.0, "high": 0.9},
    "ERASING": {"type": "float", "low": 0.0, "high": 0.5},
}


TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "train.py")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "run.log")
METRICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "last_metrics.json")
TRAIN_TIMEOUT = 7200


def _apply_params_to_train_py(params: dict, train_script: str = TRAIN_SCRIPT):
    with open(train_script, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in params.items():
            if stripped.startswith(f"{key} =") or stripped.startswith(f"{key}="):
                if isinstance(value, str):
                    lines[i] = f'{key} = "{value}"\n'
                else:
                    lines[i] = f"{key} = {value}\n"

    with open(train_script, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _run_training(train_script: str = TRAIN_SCRIPT,
                  log_file: str = LOG_FILE) -> bool:
    python = sys.executable
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    if os.path.exists(log_file):
        os.remove(log_file)

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
        time.sleep(TRAIN_TIMEOUT)
        try:
            proc.kill()
        except Exception:
            pass

    threading.Thread(target=watchdog, daemon=True).start()

    with open(log_file, "w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)

    proc.wait()
    return proc.returncode == 0


def _read_metrics(metrics_file: str = METRICS_FILE) -> Optional[dict]:
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class OptunaRunner:
    def __init__(self, search_space: dict = None, n_trials: int = 20,
                 train_script: str = TRAIN_SCRIPT,
                 log_file: str = LOG_FILE,
                 metrics_file: str = METRICS_FILE):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna not installed. pip install optuna")

        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.train_script = train_script
        self.log_file = log_file
        self.metrics_file = metrics_file
        self.study = None

    def _create_objective(self, state=None):
        search_space = self.search_space
        train_script = self.train_script
        log_file = self.log_file
        metrics_file = self.metrics_file

        def objective(trial):
            params = {}
            for key, spec in search_space.items():
                if spec["type"] == "float":
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"],
                        log=spec.get("log", False)
                    )
                elif spec["type"] == "int":
                    params[key] = trial.suggest_int(
                        key, spec["low"], spec["high"]
                    )
                elif spec["type"] == "categorical":
                    params[key] = trial.suggest_categorical(
                        key, spec["choices"]
                    )

            _apply_params_to_train_py(params, train_script)

            success = _run_training(train_script, log_file)
            if not success:
                return -1.0

            metrics = _read_metrics(metrics_file)
            if metrics is None:
                return -1.0

            cds = metrics.get("cds", 0)

            if state is not None:
                run_id = f"optuna_trial_{trial.number}"
                state.insert_experiment(
                    run_id=run_id,
                    config_json=params,
                    metrics_json=metrics,
                    status="completed",
                )
                state.log_decision(
                    run_id=run_id,
                    made_by_agent="optuna",
                    action_type="hpo_sweep",
                    rationale=f"Optuna trial {trial.number}: {json.dumps(params)}",
                )

            return -cds

        return objective

    def run(self, state=None, direction: str = "maximize") -> dict:
        self.study = optuna.create_study(direction=direction)
        objective = self._create_objective(state)
        self.study.optimize(objective, n_trials=self.n_trials)

        best = self.study.best_trial
        return {
            "best_params": best.params,
            "best_cds": -best.value,
            "best_trial_number": best.number,
            "n_trials": len(self.study.trials),
        }

    def get_best_params(self) -> dict:
        if self.study is None:
            return {}
        return self.study.best_params

    def suggest_next(self) -> dict:
        if self.study is None:
            self.study = optuna.create_study(direction="maximize")
        trial = self.study.ask()
        params = {}
        for key, spec in self.search_space.items():
            if spec["type"] == "float":
                params[key] = trial.suggest_float(
                    key, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, spec["choices"])
        return params
