"""Activity events stream for the v2 autoresearch loop.

Why: The Windows GPU operator wants to see what the orchestrator is doing
in real time — not just the raw training stdout (that goes to run.log)
but also high-level decisions (Researcher proposed X, training started,
keep/discard, HITL waiting, etc.).

This module gives the orchestrator a single sink that:
  1. Prints a human-readable line to the terminal *immediately* (flushed).
  2. Appends a structured JSON record to activity_events.jsonl so other
     tools (dashboard, monitor, supervisor agent) can replay the session.

Usage:
    events = EventLogger()
    events.emit("iteration_start", iteration=3, best_cds=0.62)
    events.emit("decision", agent="researcher", action="patch_train_config",
                rationale="bump LR to test convergence")

Both kinds of consumers — humans watching the terminal AND machines
parsing the JSONL — get the same source of truth.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_EVENTS_PATH = "activity_events.jsonl"


# ── Pretty-printer per event type ─────────────────────────────────────

_LEVEL_ICONS = {
    "INFO": "📋", "WARN": "⚠️", "CRIT": "🚨",
    "OK": "✅", "FAIL": "❌", "AGENT": "🤖",
    "TRAIN": "🏋️ ", "EVAL": "🔍", "DECIDE": "🎯",
    "BUDGET": "💰", "HITL": "🙋", "EVENT": "▸",
}


def _level_for_event(event_type: str) -> str:
    if event_type in ("iteration_start", "iteration_end"):
        return "EVENT"
    if event_type in ("train_start", "train_progress", "train_complete"):
        return "TRAIN"
    if event_type in ("eval_start", "eval_complete"):
        return "EVAL"
    if event_type in ("decision",):
        return "DECIDE"
    if event_type in ("budget_check", "budget_commit"):
        return "BUDGET"
    if event_type in ("hitl_request", "hitl_decision"):
        return "HITL"
    if event_type in ("crash", "validation_failed", "halt"):
        return "FAIL"
    if event_type in ("keep", "experiment_complete", "target_met"):
        return "OK"
    if event_type in ("warn", "drift"):
        return "WARN"
    return "INFO"


def _format_terminal(event_type: str, data: dict) -> str:
    """One-line human summary for the terminal."""
    if event_type == "iteration_start":
        return (
            f"───── iteration #{data.get('iteration', '?')}/{data.get('max_iters', '?')} "
            f"(best_cds={data.get('best_cds', 0):.4f}, "
            f"streak={data.get('target_met_streak', 0)}, "
            f"discards={data.get('consecutive_discards', 0)})"
        )
    if event_type == "iteration_end":
        return (
            f"───── iteration #{data.get('iteration', '?')} → "
            f"{data.get('status', '?').upper()}  "
            f"({data.get('duration_sec', 0):.1f}s)"
        )
    if event_type == "decision":
        return (
            f"{data.get('agent', '?')} → {data.get('action_type', '?')}  "
            f"\"{(data.get('rationale') or '')[:80]}\""
        )
    if event_type == "train_start":
        bits = []
        for k in ("MODEL", "BATCH", "LR0", "EPOCHS"):
            if k in data.get("config_patch", {}):
                bits.append(f"{k}={data['config_patch'][k]}")
        cfg = " ".join(bits) or "no patch"
        return f"training START  sha={data.get('git_sha', '?')[:8]}  ({cfg})"
    if event_type == "train_progress":
        return data.get("line", "")
    if event_type == "train_complete":
        return (
            f"training DONE   "
            f"success={data.get('success', '?')}  "
            f"{data.get('duration_min', 0):.1f}min  "
            f"epochs={data.get('epochs_completed', '?')}"
        )
    if event_type == "eval_complete":
        m = data.get("metrics", {}) or {}
        return (
            f"eval DONE  cds={m.get('cds', 0):.4f}  "
            f"P={m.get('precision', 0):.3f}  "
            f"R={m.get('recall', 0):.3f}  "
            f"target_met={m.get('target_met', False)}"
        )
    if event_type == "keep":
        return (f"KEEP  cds={data.get('cds', 0):.4f}  "
                f"Δ={data.get('improvement', 0):+.4f}  "
                f"sha={data.get('git_sha', '?')[:8]}")
    if event_type == "discard":
        return (f"DISCARD  cds={data.get('cds', 0):.4f}  "
                f"Δ={data.get('improvement', 0):+.4f}")
    if event_type == "crash":
        return f"CRASH  {data.get('reason', '?')}"
    if event_type == "halt":
        return f"HALT  {data.get('reason', '?')}"
    if event_type == "budget_check":
        if data.get("ok"):
            return (f"budget ok  remaining gpu_min="
                    f"{data.get('gpu_minutes_remaining', '?')}")
        return f"budget BLOCK  {data.get('reason', '?')}"
    if event_type == "budget_commit":
        return (f"budget commit  +{data.get('gpu_minutes', 0):.1f} gpu_min  "
                f"+${data.get('dollars', 0):.4f}")
    if event_type == "hitl_request":
        return (f"HITL waiting  gate={data.get('gate_id', '?')}  "
                f"reason={data.get('reason', '?')[:60]}")
    if event_type == "hitl_decision":
        return (f"HITL {data.get('decision', '?')} "
                f"by {data.get('decided_by', '?')}")
    if event_type == "drift":
        return (f"drift  tile-vs-pipeline={data.get('drift', 0):.3f}  "
                f"streak={data.get('drift_streak', 0)}")
    if event_type == "validation_failed":
        return f"validation FAILED  {data.get('error', '?')}"
    if event_type == "session_start":
        return (f"=== session start: llm={data.get('llm_model', '?')}, "
                f"max_iters={data.get('max_iters', '?')} ===")
    if event_type == "session_end":
        return (f"=== session end: status={data.get('status', '?')}, "
                f"iterations={data.get('iterations', 0)}, "
                f"best_cds={data.get('best_cds', 0):.4f} ===")
    # Generic fallback
    short = ", ".join(f"{k}={v}" for k, v in data.items() if k != "ts")
    return f"{event_type}  {short[:120]}"


class EventLogger:
    """Thread-safe sink for activity events.

    Writes to:
      - sys.stdout (formatted, colored when icons render)
      - jsonl file (one JSON object per line, append-only)
    """

    def __init__(
        self,
        events_path: str | os.PathLike = DEFAULT_EVENTS_PATH,
        also_stdout: bool = True,
    ):
        self.path = Path(events_path)
        self.also_stdout = also_stdout
        self._lock = threading.Lock()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **data: Any) -> None:
        """Record an event."""
        ts = time.time()
        record = {
            "ts": ts,
            "iso": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "session": self.session_id,
            "event": event_type,
            **data,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)

        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                # Don't let the events file break the loop
                print(f"[events] failed to write {self.path}: {e}",
                      file=sys.stderr, flush=True)

            if self.also_stdout:
                level = _level_for_event(event_type)
                icon = _LEVEL_ICONS.get(level, "▸")
                hms = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                summary = _format_terminal(event_type, data)
                print(f"[{hms}] {icon} {summary}", flush=True)


# ── Convenience: pass-through for libraries that want direct prints ────

def stream_train_line(line: str, logger: EventLogger | None = None) -> None:
    """Emit a single training subprocess line.

    Always prints the line to stdout for live visibility. If a logger
    is provided AND the line looks 'interesting' (epoch summary, error,
    metrics), also records it as an event so the JSONL has a coarse
    timeline of the training (without the 1000s of progress-bar lines).
    """
    # Ensure the line gets flushed to terminal in real time
    print(line.rstrip(), flush=True)

    if logger is None:
        return

    s = line.rstrip().lower()
    interesting = any(k in s for k in (
        "epoch", "cds:", "map", "precision:", "recall:", "inference_ms",
        "pipeline_", "target_met", "fatal", "error", "traceback",
    ))
    if interesting:
        logger.emit("train_progress", line=line.rstrip())
