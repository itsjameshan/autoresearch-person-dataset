"""Train-side dispatcher: applies the Researcher's config_patch to
train.py, runs training, captures the resulting metrics, and provides
git keep/discard helpers.

Reuses the proven logic from ollama_runner.py (`apply_train_changes`,
`run_training`, `git_*`, `parse_cds`) but isolates it from the global
state of that script (no module-level side effects).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

from . import eval_dispatcher
from .. import db as state_db


# Same allowlist used in agents/researcher.py — kept here as a duplicate
# defensive list because we are about to write to a file.
_VALID_VARS_FOR_PATCH = (
    "MODEL", "IMGSZ", "EPOCHS", "BATCH", "PATIENCE", "DEVICE",
    "LR0", "LRF", "COS_LR", "HSV_H", "HSV_S", "HSV_V",
    "DEGREES", "TRANSLATE", "SCALE", "FLIPUD", "FLIPLR",
    "MOSAIC", "MIXUP", "COPY_PASTE", "ERASING", "CLOSE_MOSAIC",
    "BOX", "CLS", "AMP", "CACHE", "WORKERS", "SINGLE_CLS",
)

DEFAULT_TRAIN_TIMEOUT_SEC = 3600


def _format_python_value(v) -> str:
    """Serialize a Python value the way train.py expects to see it."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        # strings -> "..."
        return '"' + v.replace('"', '\\"') + '"'
    if v is False:  # CACHE=False edge case
        return "False"
    return repr(v)


def apply_config_patch(config_patch: dict, train_path: str | Path = "train.py") -> list[str]:
    """Edit train.py in place, replacing each variable in config_patch.

    Returns the list of lines that were actually rewritten (for logging).
    Raises ValueError if a variable name is not in the allowlist or if
    the variable is not present in train.py at all.
    """
    p = Path(train_path)
    content = p.read_text(encoding="utf-8")
    rewritten: list[str] = []

    for name, value in config_patch.items():
        if name not in _VALID_VARS_FOR_PATCH:
            raise ValueError(f"{name!r} not in train.py patch allowlist")
        rendered = f"{name} = {_format_python_value(value)}"
        # Match a top-of-line assignment (any current value) to that name
        pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", flags=re.MULTILINE)
        new_content, n = pattern.subn(rendered, content, count=1)
        if n == 0:
            raise ValueError(f"{name!r} not found in {p} — refusing to add")
        content = new_content
        rewritten.append(rendered)

    p.write_text(content, encoding="utf-8")
    return rewritten


def read_train_config(train_path: str | Path = "train.py") -> str:
    """Return the EXPERIMENT CONFIG section as plain text."""
    p = Path(train_path)
    text = p.read_text(encoding="utf-8")
    # Heuristic: pull from "# EXPERIMENT CONFIG" to the next "# ───" or "TRAINING — do not"
    m = re.search(r"#\s*EXPERIMENT CONFIG.*?(?=#\s*TRAINING|$)", text, re.DOTALL)
    if m:
        return m.group(0)
    # Fallback: return all top-level UPPER_CASE assignments
    lines = [
        ln for ln in text.splitlines()
        if re.match(r"^[A-Z_]+\s*=", ln)
    ]
    return "\n".join(lines)


# ── Git helpers ────────────────────────────────────────────────────────

def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def git_short_hash() -> str:
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


def git_full_hash() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def git_commit(message: str) -> str:
    """Commit currently-staged-or-modified train.py (and adjacent files).

    Returns the new HEAD sha. Mirrors ollama_runner.py:git_commit but
    without the global RESULTS_TSV scoping (orchestrator commits
    train.py as the experiment artifact).
    """
    _git("add", "train.py")
    _git("commit", "-m", message, "--allow-empty")
    return git_full_hash()


def git_reset_hard(target: str = "HEAD") -> None:
    _git("reset", "--hard", target)


def git_push(remote: str = "origin", branch: str | None = None) -> None:
    cur = branch or _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    _git("push", remote, cur)


# ── Training subprocess ────────────────────────────────────────────────

INTERESTING_RE = re.compile(
    r"(epoch|cds|mAP|precision|recall|inference_ms|pipeline_|"
    r"target_met|FATAL|error|traceback|warning)",
    re.IGNORECASE,
)


def _train_line_is_interesting(s: str) -> bool:
    return bool(INTERESTING_RE.search(s)) or bool(s.startswith("---"))


def run_training(
    *,
    timeout_sec: int = DEFAULT_TRAIN_TIMEOUT_SEC,
    python: str | None = None,
    train_script: str = "train.py",
    log_path: str = "run.log",
    state_db_path: str | None = None,
    propagate_env: dict | None = None,
) -> tuple[bool, dict | None, dict | None, float]:
    """Run train.py as a subprocess. Returns
    (success, tile_metrics, pipeline_metrics, duration_sec).

    If state_db_path is provided, exports AUTORESEARCH_STATE_DB so
    train.py + evaluate.py write their P1/P2 hooks into it.
    """
    eval_dispatcher.cleanup_stale_metrics()
    py = python or sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if state_db_path:
        env["AUTORESEARCH_STATE_DB"] = str(state_db_path)
    if propagate_env:
        env.update(propagate_env)

    started = time.time()
    proc = subprocess.Popen(
        [py, train_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    killer_done = {"v": False}

    def _kill_after_timeout():
        time.sleep(timeout_sec)
        if not killer_done["v"] and proc.poll() is None:
            proc.kill()
            print(f"\n[orchestrator] train timeout ({timeout_sec}s) — killed.\n")

    threading.Thread(target=_kill_after_timeout, daemon=True).start()

    try:
        with open(log_path, "w", encoding="utf-8") as log:
            if proc.stdout:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    s = line.rstrip()
                    if _train_line_is_interesting(s):
                        print(f"  │ {s}")
        rc = proc.wait()
    finally:
        killer_done["v"] = True

    duration = time.time() - started

    tile = eval_dispatcher.parse_tile_metrics()
    pipeline = eval_dispatcher.parse_pipeline_metrics()

    success = rc == 0 and tile is not None and "cds" in tile
    return success, tile, pipeline, duration


# ── Decision summary helpers ───────────────────────────────────────────

def summarize_recent_experiments(
    db_path: str,
    limit: int = 10,
) -> str:
    """Return a compact text summary of the last N experiments for the
    Researcher prompt. Mirrors ollama_runner.py's results.tsv view but
    sourced from state.db (which is the v2 source of truth)."""
    rows = state_db.read_recent_experiments(db_path, limit=limit)
    if not rows:
        return "(no prior experiments)"
    lines = []
    for r in rows:
        sha = (r.get("git_sha") or "?")[:8]
        status = r.get("status") or "?"
        desc = (r.get("description") or "").strip()[:60]
        cds = "?"
        try:
            import json as _json
            tm = r.get("tile_metrics_json")
            if tm:
                cds = f"{float(_json.loads(tm).get('cds', 0)):.4f}"
        except Exception:
            pass
        lines.append(f"  {sha}  cds={cds}  status={status:<20}  {desc}")
    return "\n".join(lines)
