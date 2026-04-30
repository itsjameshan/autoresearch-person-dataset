"""State database for the v2 autoresearch loop.

Stores experiments (keyed by git SHA), agent decisions, data issues, budget,
artifacts, and HITL gate requests. Coexists with the legacy results.tsv +
last_metrics.json + status.md outputs — those remain the canonical view for
ollama_runner.py; this database is the canonical view for the v2 orchestrator.

Design notes:
- git_sha is the natural primary key for an experiment (an experiment IS a
  commit). We never duplicate config in the DB; train.py at that SHA is the
  config of record.
- All writes are upserts so partial-state rows from train.py can be enriched
  later by evaluate.py and the orchestrator.
- WAL mode + 30s busy timeout makes concurrent reads safe even while the
  orchestrator is writing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    git_sha TEXT PRIMARY KEY,
    parent_sha TEXT,
    status TEXT NOT NULL,
    started_at REAL,
    ended_at REAL,
    tile_metrics_json TEXT,
    pipeline_metrics_json TEXT,
    gpu_minutes REAL,
    cost_usd REAL,
    action_type TEXT,
    agent_made_decision TEXT,
    description TEXT,
    epochs_completed INTEGER
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_sha TEXT,
    made_by_agent TEXT NOT NULL,
    action_type TEXT NOT NULL,
    rationale TEXT,
    expected_delta REAL,
    estimated_cost REAL,
    needs_hitl INTEGER DEFAULT 0,
    config_patch_json TEXT,
    schema_version INTEGER DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS data_issues (
    issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    severity REAL,
    discovered_in_sha TEXT,
    resolved_in_sha TEXT,
    confidence REAL,
    notes TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS budget (
    period TEXT PRIMARY KEY,
    gpu_minutes_used REAL DEFAULT 0,
    gpu_minutes_limit REAL,
    dollars_used REAL DEFAULT 0,
    dollars_limit REAL,
    period_start REAL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    git_sha TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS hitl_gates (
    gate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_sha TEXT,
    requestor_agent TEXT,
    requested_at REAL NOT NULL,
    prompt_text TEXT,
    decision TEXT NOT NULL DEFAULT 'pending',
    decided_by TEXT,
    decided_at REAL,
    comment TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_started ON experiments(started_at);
CREATE INDEX IF NOT EXISTS idx_decisions_parent ON decisions(parent_sha);
CREATE INDEX IF NOT EXISTS idx_data_issues_sha ON data_issues(discovered_in_sha);
CREATE INDEX IF NOT EXISTS idx_artifacts_sha ON artifacts(git_sha);
CREATE INDEX IF NOT EXISTS idx_hitl_decision ON hitl_gates(decision);
"""

# Columns that callers may set on the experiments table (allowlist guards
# upsert against typos).
_EXPERIMENT_COLUMNS = {
    "parent_sha", "status", "started_at", "ended_at",
    "tile_metrics_json", "pipeline_metrics_json",
    "gpu_minutes", "cost_usd", "action_type",
    "agent_made_decision", "description", "epochs_completed",
}


def init_db(path: str | os.PathLike) -> Path:
    """Create the database file (if missing) and apply schema. Idempotent."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with connect(p) as conn:
        conn.executescript(_SCHEMA_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
    return p


@contextmanager
def connect(path: str | os.PathLike):
    """Open a connection with WAL + busy timeout + foreign keys + Row factory."""
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
    finally:
        conn.close()


# ── Experiments ──────────────────────────────────────────────────────────

def upsert_experiment(db_path: str | os.PathLike, git_sha: str, **fields: Any) -> None:
    """Insert or update an experiment row keyed by git_sha.

    Only columns in _EXPERIMENT_COLUMNS are accepted (raises KeyError otherwise).
    Existing values are preserved for columns not provided.
    """
    if not git_sha:
        raise ValueError("git_sha is required")
    bad = set(fields) - _EXPERIMENT_COLUMNS
    if bad:
        raise KeyError(f"unknown experiment columns: {sorted(bad)}")

    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM experiments WHERE git_sha=?", (git_sha,)
        ).fetchone()

        if existing is None:
            cols = ["git_sha"] + list(fields.keys())
            placeholders = ",".join("?" for _ in cols)
            values = [git_sha] + [fields[c] for c in fields]
            # status is required; default to 'started' if caller didn't specify
            if "status" not in fields:
                cols.append("status")
                placeholders += ",?"
                values.append("started")
            conn.execute(
                f"INSERT INTO experiments({','.join(cols)}) VALUES ({placeholders})",
                values,
            )
        else:
            if not fields:
                return
            sets = ",".join(f"{c}=?" for c in fields)
            conn.execute(
                f"UPDATE experiments SET {sets} WHERE git_sha=?",
                list(fields.values()) + [git_sha],
            )


def record_metrics(
    db_path: str | os.PathLike,
    git_sha: str,
    tile_metrics: dict | None = None,
    pipeline_metrics: dict | None = None,
    epochs_completed: int | None = None,
) -> None:
    """Record evaluation metrics for an experiment. Convenience wrapper."""
    fields: dict[str, Any] = {"status": "evaluated", "ended_at": time.time()}
    if tile_metrics is not None:
        fields["tile_metrics_json"] = json.dumps(tile_metrics)
    if pipeline_metrics is not None:
        fields["pipeline_metrics_json"] = json.dumps(pipeline_metrics)
    if epochs_completed is not None:
        fields["epochs_completed"] = epochs_completed
    upsert_experiment(db_path, git_sha, **fields)


def read_recent_experiments(db_path: str | os.PathLike, limit: int = 10) -> list[dict]:
    """Return last N experiments by started_at desc."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY COALESCE(started_at, 0) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment(db_path: str | os.PathLike, git_sha: str) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE git_sha=?", (git_sha,)
        ).fetchone()
    return dict(row) if row else None


def get_best_experiment(
    db_path: str | os.PathLike, metric: str = "cds"
) -> dict | None:
    """Return experiment with the highest value of metric (extracted from JSON)."""
    with connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT *,
                   CAST(json_extract(tile_metrics_json, '$."{metric}"') AS REAL) AS m
            FROM experiments
            WHERE tile_metrics_json IS NOT NULL
            ORDER BY m DESC
            LIMIT 1
            """
        ).fetchone()
    return dict(row) if row else None


# ── Decisions ────────────────────────────────────────────────────────────

def record_decision(
    db_path: str | os.PathLike,
    *,
    parent_sha: str | None,
    made_by_agent: str,
    action_type: str,
    rationale: str | None = None,
    expected_delta: float | None = None,
    estimated_cost: float | None = None,
    needs_hitl: bool = False,
    config_patch: dict | None = None,
) -> int:
    """Insert a new decision row, return decision_id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO decisions(
                parent_sha, made_by_agent, action_type, rationale,
                expected_delta, estimated_cost, needs_hitl,
                config_patch_json, schema_version, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                parent_sha, made_by_agent, action_type, rationale,
                expected_delta, estimated_cost, 1 if needs_hitl else 0,
                json.dumps(config_patch) if config_patch is not None else None,
                SCHEMA_VERSION, time.time(),
            ),
        )
        return int(cur.lastrowid)


# ── Artifacts ────────────────────────────────────────────────────────────

def record_artifact(
    db_path: str | os.PathLike,
    git_sha: str,
    kind: str,
    path: str,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO artifacts(git_sha, kind, path, sha256, size_bytes, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (git_sha, kind, path, sha256, size_bytes, time.time()),
        )
        return int(cur.lastrowid)


# ── Data issues (Curator agent output) ───────────────────────────────────

def record_data_issue(
    db_path: str | os.PathLike,
    *,
    image_path: str,
    issue_type: str,
    severity: float | None = None,
    discovered_in_sha: str | None = None,
    confidence: float | None = None,
    notes: str | None = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO data_issues(
                image_path, issue_type, severity,
                discovered_in_sha, confidence, notes, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                image_path, issue_type, severity,
                discovered_in_sha, confidence, notes, time.time(),
            ),
        )
        return int(cur.lastrowid)


def resolve_data_issue(
    db_path: str | os.PathLike, issue_id: int, resolved_in_sha: str
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE data_issues SET resolved_in_sha=? WHERE issue_id=?",
            (resolved_in_sha, issue_id),
        )


# ── Budget ───────────────────────────────────────────────────────────────

def init_budget(
    db_path: str | os.PathLike,
    period: str,
    gpu_minutes_limit: float | None = None,
    dollars_limit: float | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO budget(
                period, gpu_minutes_used, gpu_minutes_limit,
                dollars_used, dollars_limit, period_start
            )
            VALUES (?, COALESCE((SELECT gpu_minutes_used FROM budget WHERE period=?), 0),
                    ?,
                    COALESCE((SELECT dollars_used FROM budget WHERE period=?), 0),
                    ?, ?)
            """,
            (period, period, gpu_minutes_limit, period, dollars_limit, time.time()),
        )


def consume_budget(
    db_path: str | os.PathLike,
    period: str,
    gpu_minutes: float = 0.0,
    dollars: float = 0.0,
) -> dict:
    """Add consumption to a period and return the updated row."""
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE budget
            SET gpu_minutes_used = gpu_minutes_used + ?,
                dollars_used = dollars_used + ?
            WHERE period = ?
            """,
            (gpu_minutes, dollars, period),
        )
        row = conn.execute(
            "SELECT * FROM budget WHERE period=?", (period,)
        ).fetchone()
    return dict(row) if row else {}


def budget_remaining(db_path: str | os.PathLike, period: str) -> dict:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM budget WHERE period=?", (period,)
        ).fetchone()
    if not row:
        return {"period": period, "exists": False}
    d = dict(row)
    d["gpu_minutes_remaining"] = (
        (d["gpu_minutes_limit"] or 0) - (d["gpu_minutes_used"] or 0)
        if d["gpu_minutes_limit"] is not None else None
    )
    d["dollars_remaining"] = (
        (d["dollars_limit"] or 0) - (d["dollars_used"] or 0)
        if d["dollars_limit"] is not None else None
    )
    d["exists"] = True
    return d


# ── HITL gates ───────────────────────────────────────────────────────────

def request_hitl(
    db_path: str | os.PathLike,
    *,
    target_sha: str | None,
    requestor_agent: str,
    prompt_text: str,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO hitl_gates(
                target_sha, requestor_agent, requested_at, prompt_text, decision
            ) VALUES (?,?,?,?,'pending')
            """,
            (target_sha, requestor_agent, time.time(), prompt_text),
        )
        return int(cur.lastrowid)


def decide_hitl(
    db_path: str | os.PathLike,
    gate_id: int,
    decision: str,
    decided_by: str,
    comment: str | None = None,
) -> None:
    if decision not in {"approved", "denied"}:
        raise ValueError(f"decision must be 'approved' or 'denied', got {decision!r}")
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE hitl_gates
            SET decision=?, decided_by=?, decided_at=?, comment=?
            WHERE gate_id=?
            """,
            (decision, decided_by, time.time(), comment, gate_id),
        )


def list_pending_hitl(db_path: str | os.PathLike) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hitl_gates WHERE decision='pending' ORDER BY requested_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Helpers for evaluate.py / orchestrator integration ───────────────────

def get_current_git_sha(repo_dir: str | os.PathLike | None = None) -> str | None:
    """Best-effort lookup of HEAD's git SHA. Returns None on any failure."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def state_db_from_env() -> str | None:
    """Resolve the configured state.db path from env, or None if unconfigured."""
    raw = os.environ.get("AUTORESEARCH_STATE_DB", "").strip()
    return raw or None
