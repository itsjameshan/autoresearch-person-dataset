"""
db.py — SQLite 状态层：6张核心表 + DAO

表:
  experiments  — 每次训练+评估的完整记录
  decisions    — agent 每次决策的不可变审计日志
  data_issues  — Curator 输出，跨实验追踪
  budget       — 预算守门读这张表
  artifacts    — 文件指针，避免大文件入库
  hitl_gates   — 人工审批记录
"""

import json
import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
DB_PATH = os.path.join(DB_DIR, "state.db")

SCHEMA_VERSION = 1


def _ensure_dir():
    os.makedirs(DB_DIR, exist_ok=True)


def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None):
    if conn is None:
        conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id          TEXT PRIMARY KEY,
    parent_run_id   TEXT,
    git_sha         TEXT,
    dataset_version TEXT,
    config_json     TEXT NOT NULL DEFAULT '{}',
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TEXT,
    ended_at        TEXT,
    gpu_minutes     REAL NOT NULL DEFAULT 0.0,
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    root_cause      TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT,
    made_by_agent       TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    rationale           TEXT,
    expected_improvement TEXT,
    estimated_cost      REAL,
    schema_version      INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

CREATE TABLE IF NOT EXISTS data_issues (
    issue_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path          TEXT,
    issue_type          TEXT NOT NULL,
    severity            TEXT NOT NULL DEFAULT 'medium',
    description         TEXT,
    discovered_in_run   TEXT,
    resolved_in_run     TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (discovered_in_run) REFERENCES experiments(run_id),
    FOREIGN KEY (resolved_in_run) REFERENCES experiments(run_id)
);

CREATE TABLE IF NOT EXISTS budget (
    period              TEXT PRIMARY KEY,
    gpu_minutes_used    REAL NOT NULL DEFAULT 0.0,
    gpu_minutes_limit   REAL NOT NULL DEFAULT 0.0,
    dollars_used        REAL NOT NULL DEFAULT 0.0,
    dollars_limit       REAL NOT NULL DEFAULT 0.0,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT,
    size_bytes  INTEGER,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, kind),
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

CREATE TABLE IF NOT EXISTS hitl_gates (
    gate_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    requested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    requestor_agent TEXT NOT NULL,
    prompt_text     TEXT,
    decision        TEXT,
    decided_by      TEXT,
    decided_at      TEXT,
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(made_by_agent);
CREATE INDEX IF NOT EXISTS idx_data_issues_type ON data_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_data_issues_resolved ON data_issues(resolved_in_run);
"""


class StateDB:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = get_conn(db_path)
        init_db(self.conn)

    def close(self):
        self.conn.close()

    def _now(self) -> str:
        return datetime.now().isoformat()

    def insert_experiment(self, run_id: str, parent_run_id: str = "",
                          git_sha: str = "", dataset_version: str = "",
                          config_json: dict = None, metrics_json: dict = None,
                          status: str = "pending") -> str:
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments
               (run_id, parent_run_id, git_sha, dataset_version,
                config_json, metrics_json, status, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, parent_run_id, git_sha, dataset_version,
             json.dumps(config_json or {}, ensure_ascii=False),
             json.dumps(metrics_json or {}, ensure_ascii=False),
             status, self._now())
        )
        self.conn.commit()
        return run_id

    def update_experiment_metrics(self, run_id: str, metrics_json: dict):
        self.conn.execute(
            """UPDATE experiments SET metrics_json=?, ended_at=?, status='completed'
               WHERE run_id=?""",
            (json.dumps(metrics_json, ensure_ascii=False), self._now(), run_id)
        )
        self.conn.commit()

    def update_experiment_status(self, run_id: str, status: str,
                                  gpu_minutes: float = 0.0, cost_usd: float = 0.0,
                                  root_cause: str = ""):
        # Ensure root_cause column exists (migration for old DBs)
        try:
            self.conn.execute(
                "ALTER TABLE experiments ADD COLUMN root_cause TEXT NOT NULL DEFAULT ''"
            )
        except Exception:
            pass  # column already exists
        self.conn.execute(
            """UPDATE experiments SET status=?, gpu_minutes=?, cost_usd=?, ended_at=?, root_cause=?
               WHERE run_id=?""",
            (status, gpu_minutes, cost_usd, self._now(), root_cause, run_id)
        )
        self.conn.commit()

    def get_experiment(self, run_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config_json"] = json.loads(d.get("config_json", "{}"))
        d["metrics_json"] = json.loads(d.get("metrics_json", "{}"))
        return d

    def get_recent_experiments(self, n: int = 10) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM experiments
               ORDER BY started_at DESC LIMIT ?""",
            (n,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["config_json"] = json.loads(d.get("config_json", "{}"))
            d["metrics_json"] = json.loads(d.get("metrics_json", "{}"))
            results.append(d)
        return results

    def get_best_metrics(self) -> Optional[dict]:
        row = self.conn.execute(
            """SELECT metrics_json FROM experiments
               WHERE status='completed' AND metrics_json != '{}'
               ORDER BY json_extract(metrics_json, '$.cds') DESC
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metrics_json"])

    def log_decision(self, run_id: str, made_by_agent: str, action_type: str,
                     rationale: str = "", expected_improvement: str = "",
                     estimated_cost: float = 0.0) -> int:
        run_ref = run_id if run_id else None
        cur = self.conn.execute(
            """INSERT INTO decisions
               (run_id, made_by_agent, action_type, rationale,
                expected_improvement, estimated_cost)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_ref, made_by_agent, action_type, rationale,
             expected_improvement, estimated_cost)
        )
        self.conn.commit()
        return cur.lastrowid

    def get_decisions(self, run_id: str = None, agent: str = None,
                      limit: int = 50) -> list[dict]:
        q = "SELECT * FROM decisions WHERE 1=1"
        params = []
        if run_id:
            q += " AND run_id=?"
            params.append(run_id)
        if agent:
            q += " AND made_by_agent=?"
            params.append(agent)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def insert_data_issue(self, image_path: str, issue_type: str,
                          severity: str = "medium", description: str = "",
                          discovered_in_run: str = "") -> int:
        run_ref = discovered_in_run if discovered_in_run else None
        cur = self.conn.execute(
            """INSERT INTO data_issues
               (image_path, issue_type, severity, description, discovered_in_run)
               VALUES (?, ?, ?, ?, ?)""",
            (image_path, issue_type, severity, description, run_ref)
        )
        self.conn.commit()
        return cur.lastrowid

    def resolve_data_issue(self, issue_id: int, resolved_in_run: str):
        run_ref = resolved_in_run if resolved_in_run else None
        self.conn.execute(
            "UPDATE data_issues SET resolved_in_run=? WHERE issue_id=?",
            (run_ref, issue_id)
        )
        self.conn.commit()

    def get_unresolved_issues(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM data_issues
               WHERE resolved_in_run IS NULL
               ORDER BY severity DESC, created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_data_issues(self, n: int = 3) -> list[dict]:
        return self.get_unresolved_issues(limit=n)

    def upsert_budget(self, period: str, gpu_minutes_limit: float = 0.0,
                      dollars_limit: float = 0.0):
        self.conn.execute(
            """INSERT INTO budget (period, gpu_minutes_limit, dollars_limit)
               VALUES (?, ?, ?)
               ON CONFLICT(period) DO UPDATE SET
                 gpu_minutes_limit=excluded.gpu_minutes_limit,
                 dollars_limit=excluded.dollars_limit""",
            (period, gpu_minutes_limit, dollars_limit)
        )
        self.conn.commit()

    def add_budget_usage(self, period: str, gpu_minutes: float = 0.0,
                         dollars: float = 0.0):
        self.conn.execute(
            """UPDATE budget SET
                 gpu_minutes_used = gpu_minutes_used + ?,
                 dollars_used = dollars_used + ?,
                 updated_at = ?
               WHERE period=?""",
            (gpu_minutes, dollars, self._now(), period)
        )
        self.conn.commit()

    def get_budget(self, period: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM budget WHERE period=?", (period,)
        ).fetchone()
        return dict(row) if row else None

    def get_remaining_budget(self, period: str) -> dict:
        b = self.get_budget(period)
        if b is None:
            return {"gpu_minutes_remaining": float("inf"),
                    "dollars_remaining": float("inf")}
        gpu_rem = b["gpu_minutes_limit"] - b["gpu_minutes_used"]
        usd_rem = b["dollars_limit"] - b["dollars_used"]
        return {"gpu_minutes_remaining": max(0, gpu_rem),
                "dollars_remaining": max(0, usd_rem)}

    def insert_artifact(self, run_id: str, kind: str, path: str):
        sha256 = ""
        size_bytes = 0
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()
        self.conn.execute(
            """INSERT OR REPLACE INTO artifacts (run_id, kind, path, sha256, size_bytes)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, kind, path, sha256, size_bytes)
        )
        self.conn.commit()

    def get_artifacts(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE run_id=?", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_hitl_gate(self, run_id: str, requestor_agent: str,
                         prompt_text: str = "") -> int:
        run_ref = run_id if run_id else None
        cur = self.conn.execute(
            """INSERT INTO hitl_gates (run_id, requestor_agent, prompt_text)
               VALUES (?, ?, ?)""",
            (run_ref, requestor_agent, prompt_text)
        )
        self.conn.commit()
        return cur.lastrowid

    def resolve_hitl_gate(self, gate_id: int, decision: str, decided_by: str = "human"):
        self.conn.execute(
            """UPDATE hitl_gates SET decision=?, decided_by=?, decided_at=?
               WHERE gate_id=?""",
            (decision, decided_by, self._now(), gate_id)
        )
        self.conn.commit()

    def get_pending_hitl_gates(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM hitl_gates WHERE decision IS NULL
               ORDER BY requested_at ASC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM experiments").fetchone()
        return row["cnt"] if row else 0

    def get_consecutive_no_improve(self) -> int:
        rows = self.conn.execute(
            """SELECT metrics_json FROM experiments
               WHERE status='completed'
               ORDER BY started_at DESC LIMIT 20"""
        ).fetchall()
        if not rows:
            return 0
        best_cds = 0.0
        count = 0
        for row in rows:
            m = json.loads(row["metrics_json"])
            cds = m.get("cds", 0)
            if cds > best_cds:
                best_cds = cds
                count = 0
            else:
                count += 1
        return count
