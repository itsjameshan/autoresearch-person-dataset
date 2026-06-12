"""
optuna_runner.py — Optuna HPO 超参搜索模块

替代 Researcher 手动试参数，用 TPE 搜索高效探索参数空间。

P5 rewrite — fixes vs the original:
  - Reuses train_dispatcher.run_training so HPO trials get the SAME live
    terminal output + activity events as a normal training run (the
    Windows operator sees every Epoch line as it streams).
  - Fixes a sign bug: previously returned -cds with direction='maximize',
    which actually MINIMIZED CDS. Now returns cds and maximizes.
  - Persists the Optuna study to SQLite so HPO can resume after a crash
    or planned stop.
  - Emits structured events (hpo_start / hpo_trial_start /
    hpo_trial_complete / hpo_complete / hpo_pruned) — the orchestrator
    EventLogger writes them to activity_events.jsonl AND prints them
    live so the operator can see HPO progress alongside training.
  - Each trial inserts an experiment row with agent_made_decision='optuna'
    so HPO trials and regular Researcher experiments live in one table.
  - Optional per-trial budget check via budget guard.
  - Reuses train_dispatcher.apply_config_diff (no duplicate regex).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from . import train_dispatcher as _td


# Default search space targets the main knobs that LLMs are bad at picking
# values for (continuous, log-scale, multimodal). MODEL / EPOCHS / BATCH
# are deliberately excluded — those are strategic decisions for Researcher.
DEFAULT_SEARCH_SPACE = {
    "LR0":        {"type": "float", "low": 0.001,  "high": 0.02,  "log": True},
    "LRF":        {"type": "float", "low": 0.0005, "high": 0.02,  "log": True},
    "BOX":        {"type": "float", "low": 5.0,    "high": 20.0},
    "CLS":        {"type": "float", "low": 0.3,    "high": 1.5},
    "MOSAIC":     {"type": "float", "low": 0.3,    "high": 1.0},
    "MIXUP":      {"type": "float", "low": 0.0,    "high": 0.3},
    "COPY_PASTE": {"type": "float", "low": 0.0,    "high": 0.3},
    "DEGREES":    {"type": "float", "low": 0.0,    "high": 20.0},
    "HSV_H":      {"type": "float", "low": 0.0,    "high": 0.1},
    "HSV_S":      {"type": "float", "low": 0.0,    "high": 0.9},
    "ERASING":    {"type": "float", "low": 0.0,    "high": 0.5},
}


DEFAULT_STUDY_DB = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "state", "optuna.db",
    )
)


def _suggest(trial, search_space: dict) -> dict:
    """Sample one config from the search space."""
    params: dict[str, Any] = {}
    for key, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            params[key] = trial.suggest_float(
                key, spec["low"], spec["high"], log=spec.get("log", False),
            )
        elif t == "int":
            params[key] = trial.suggest_int(key, spec["low"], spec["high"])
        elif t == "categorical":
            params[key] = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"unknown search_space type: {t!r} for {key}")
    return params


class OptunaSweepResult:
    """Plain dict-like result returned by OptunaRunner.run()."""
    def __init__(self, **kw):
        self.best_params: dict = kw.get("best_params", {})
        self.best_cds: float = kw.get("best_cds", 0.0)
        self.best_trial_number: int = kw.get("best_trial_number", -1)
        self.n_trials: int = kw.get("n_trials", 0)
        self.n_completed: int = kw.get("n_completed", 0)
        # n_with_metrics: 真正产出了非空 metrics 的 trial 数。failed/stale
        # trial 可能混入 n_completed (历史 bug: CDS=0.0000 也算 completed),
        # 判断 sweep 成败必须看这个数。
        self.n_with_metrics: int = kw.get("n_with_metrics", 0)
        self.n_failed: int = kw.get("n_failed", 0)
        self.n_pruned: int = kw.get("n_pruned", 0)
        self.study_name: str = kw.get("study_name", "")

    def as_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_cds": self.best_cds,
            "best_trial_number": self.best_trial_number,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_with_metrics": self.n_with_metrics,
            "n_failed": self.n_failed,
            "n_pruned": self.n_pruned,
            "study_name": self.study_name,
        }


class OptunaRunner:
    """Serial TPE HPO over the train.py config knobs.

    Single-GPU constraint forces serial trials — Optuna's `n_jobs=1` is
    fine here. Each trial calls the same train_dispatcher.run_training
    that a normal experiment uses, so live output + log file behave
    identically.
    """

    def __init__(
        self,
        search_space: dict | None = None,
        n_trials: int = 20,
        train_script: str = _td.TRAIN_SCRIPT,
        log_file: str = _td.LOG_FILE,
        metrics_file: str = "last_metrics.json",
        study_name: str | None = None,
        study_storage: str | None = None,
        events_logger=None,
        state=None,
        budget=None,
        budget_period: str | None = None,
        per_trial_timeout_sec: int = _td.TRAIN_TIMEOUT,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna not installed. pip install optuna")

        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.n_trials = n_trials
        self.train_script = train_script
        self.log_file = log_file
        self.metrics_file = metrics_file
        self.events = events_logger
        self.state = state
        self.budget = budget
        self.budget_period = budget_period
        self.per_trial_timeout_sec = per_trial_timeout_sec

        self.study_name = (
            study_name or f"hpo_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.study_storage = study_storage or f"sqlite:///{DEFAULT_STUDY_DB}"
        os.makedirs(os.path.dirname(DEFAULT_STUDY_DB), exist_ok=True)
        self.study: Optional["optuna.Study"] = None
        self._n_failed = 0
        self._n_pruned = 0
        self._n_with_metrics = 0

    # ── Internal helpers ────────────────────────────────────────────

    def _emit(self, event_type: str, **data) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(event_type, **data)
        except Exception:
            pass

    def _read_metrics(self, fresh_after: Optional[float] = None) -> Optional[dict]:
        """读 last_metrics.json。fresh_after 给定时, 文件 mtime 早于它即视为
        上一轮残留的陈旧文件, 返回 None — 否则失败 trial 会捡到旧指标,
        被当成 completed 记录 (CDS=0.0000 假完成行的来源之一)。"""
        if not os.path.exists(self.metrics_file):
            return None
        try:
            if fresh_after is not None and os.path.getmtime(self.metrics_file) < fresh_after:
                return None
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _record_trial_in_state(
        self,
        trial_number: int,
        run_id: str,
        params: dict,
        metrics: Optional[dict],
        status: str,
        gpu_minutes: float,
    ) -> None:
        if self.state is None:
            return
        if status == "completed" and not metrics:
            # 空 metrics 绝不允许以 completed 入库 — 那就是results里
            # CDS=0.0000 "completed" 假行的来源。
            status = "failed"
        try:
            self.state.insert_experiment(
                run_id=run_id,
                config_json=params,
                metrics_json=metrics or {},
                status=status,
            )
            # Tag this row as Optuna's so reports can separate HPO from
            # Researcher decisions.
            try:
                self.state.update_experiment_status(
                    run_id, status,
                    gpu_minutes=gpu_minutes,
                    cost_usd=gpu_minutes * 0.01,
                )
            except Exception:
                pass
            self.state.log_decision(
                run_id=run_id,
                made_by_agent="optuna",
                action_type="hpo_sweep",
                rationale=f"trial={trial_number} study={self.study_name} params={json.dumps(params, ensure_ascii=False)}",
            )
        except Exception as e:
            print(f"[optuna] state write failed: {e}", file=sys.stderr, flush=True)

    # ── Objective ────────────────────────────────────────────────────

    def _objective(self, trial) -> float:
        params = _suggest(trial, self.search_space)
        run_id = f"optuna_{self.study_name}_t{trial.number:03d}"

        # Per-trial budget pre-check (advisory — sweep may terminate).
        if self.budget is not None and self.budget_period is not None:
            try:
                rem = self.budget.check_remaining()
                gpu_rem = rem.get("gpu_minutes_remaining")
                if gpu_rem is not None and gpu_rem <= 0:
                    self._emit(
                        "hpo_trial_complete",
                        trial=trial.number, run_id=run_id,
                        status="aborted_no_budget",
                    )
                    raise optuna.TrialPruned("budget exhausted")
            except optuna.TrialPruned:
                raise
            except Exception:
                pass

        self._emit(
            "hpo_trial_start",
            trial=trial.number,
            run_id=run_id,
            params=params,
            study=self.study_name,
        )

        # Apply params -> train.py and run training (live output via
        # train_dispatcher's stream-to-stdout-and-log path).
        _td.apply_config_diff(params, self.train_script)
        t0 = time.time()
        success, _ = _td.run_training(
            train_script=self.train_script,
            log_file=self.log_file,
            timeout=self.per_trial_timeout_sec,
            events_logger=self.events,
        )
        gpu_min = round((time.time() - t0) / 60.0, 2)

        if self.budget is not None and self.budget_period is not None:
            try:
                self.budget.commit_actual(run_id, gpu_min, gpu_min * 0.01)
            except Exception:
                pass

        if not success:
            self._n_failed += 1
            self._record_trial_in_state(
                trial.number, run_id, params, None, "failed", gpu_min,
            )
            self._emit(
                "hpo_trial_complete",
                trial=trial.number, run_id=run_id,
                status="failed", gpu_minutes=gpu_min,
            )
            raise optuna.TrialPruned("training failed")

        metrics = self._read_metrics(fresh_after=t0)
        if not metrics or "cds" not in metrics:
            self._n_failed += 1
            self._record_trial_in_state(
                trial.number, run_id, params, None, "failed", gpu_min,
            )
            self._emit(
                "hpo_trial_complete",
                trial=trial.number, run_id=run_id,
                status="no_metrics", gpu_minutes=gpu_min,
            )
            raise optuna.TrialPruned("no metrics produced")

        cds = float(metrics.get("cds", 0.0))
        self._n_with_metrics += 1
        self._record_trial_in_state(
            trial.number, run_id, params, metrics, "completed", gpu_min,
        )
        self._emit(
            "hpo_trial_complete",
            trial=trial.number, run_id=run_id,
            status="completed",
            cds=cds,
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            gpu_minutes=gpu_min,
        )
        return cds  # MAXIMIZE — fixed sign bug

    # ── Public API ───────────────────────────────────────────────────

    def run(self) -> OptunaSweepResult:
        """Run the sweep. Returns a result object even when all trials fail."""
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
        # MedianPruner skips trials whose intermediate values fall below the
        # running median — but our objective is one-shot (no intermediate
        # report), so the pruner only catches explicit TrialPruned raises.
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.study_storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        self._emit(
            "hpo_start",
            study=self.study_name,
            n_trials=self.n_trials,
            search_space_keys=list(self.search_space.keys()),
            storage=self.study_storage,
        )

        try:
            self.study.optimize(
                self._objective,
                n_trials=self.n_trials,
                catch=(),
            )
        except KeyboardInterrupt:
            self._emit("hpo_interrupted", study=self.study_name)

        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]
        self._n_pruned = len(pruned)

        try:
            best = self.study.best_trial
            best_params = best.params
            best_cds = float(best.value) if best.value is not None else 0.0
            best_trial_number = best.number
        except (ValueError, KeyError):
            best_params = {}
            best_cds = 0.0
            best_trial_number = -1

        result = OptunaSweepResult(
            best_params=best_params,
            best_cds=best_cds,
            best_trial_number=best_trial_number,
            n_trials=len(self.study.trials),
            n_completed=len(completed),
            n_with_metrics=self._n_with_metrics,
            n_failed=self._n_failed,
            n_pruned=self._n_pruned,
            study_name=self.study_name,
        )

        self._emit(
            "hpo_complete",
            **result.as_dict(),
        )
        return result

    def get_best_params(self) -> dict:
        if self.study is None:
            return {}
        try:
            return dict(self.study.best_params)
        except (ValueError, KeyError):
            return {}


# ── Backwards-compat free helpers (used by some legacy callers) ────────

def apply_params_to_train_py(params: dict, train_script: str = _td.TRAIN_SCRIPT) -> None:
    """Thin wrapper around train_dispatcher.apply_config_diff."""
    _td.apply_config_diff(params, train_script)
