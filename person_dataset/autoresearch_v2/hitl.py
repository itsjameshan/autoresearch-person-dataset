"""Human-in-the-loop (HITL) approval gate for the v2 autoresearch loop.

When the Researcher agent estimates that an action will cost more than
30% of remaining budget — or proposes a destructive change like backbone
swap — it must request approval before the orchestrator dispatches it.

This module provides:
- request_approval(): blocking call that polls state.db until a human
  records a decision (or timeout fires).
- decide(): record an approval/denial (used by the Flask UI and CLI).
- list_pending(): surface what's waiting on a human.
- A CLI for command-line approve/deny (the bare-minimum interface that
  works without the Flask UI).

Pair with hitl_app.py for a tiny Flask review page.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import db as state_db


class HITLTimeout(RuntimeError):
    """Raised when request_approval() times out and on_timeout='raise'."""


def request_approval(
    db_path: str | Path,
    *,
    target_sha: str | None,
    prompt_text: str,
    requestor_agent: str = "orchestrator",
    timeout_seconds: float | None = None,
    poll_interval: float = 2.0,
    on_timeout: str = "deny",
) -> dict:
    """Block the calling process until a human records a decision.

    Returns a dict with keys: gate_id, decision, decided_by, decided_at,
    comment.

    on_timeout controls behavior when timeout_seconds elapses without a
    decision:
        'deny'    — auto-deny in the DB and return that record (default).
        'approve' — auto-approve.
        'raise'   — raise HITLTimeout (the gate stays 'pending' in DB).
    """
    if on_timeout not in {"deny", "approve", "raise"}:
        raise ValueError(f"on_timeout must be deny/approve/raise, got {on_timeout!r}")
    state_db.init_db(db_path)
    gate_id = state_db.request_hitl(
        db_path,
        target_sha=target_sha,
        requestor_agent=requestor_agent,
        prompt_text=prompt_text,
    )

    started = time.time()
    while True:
        with state_db.connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM hitl_gates WHERE gate_id=?", (gate_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"hitl gate {gate_id} disappeared from DB")
        rec = dict(row)
        if rec["decision"] in {"approved", "denied"}:
            return rec

        if timeout_seconds is not None and time.time() - started >= timeout_seconds:
            if on_timeout == "raise":
                raise HITLTimeout(
                    f"gate {gate_id} pending after {timeout_seconds}s"
                )
            auto_decision = "approved" if on_timeout == "approve" else "denied"
            state_db.decide_hitl(
                db_path, gate_id, auto_decision,
                decided_by="auto-timeout",
                comment=f"auto-{auto_decision} after {timeout_seconds}s",
            )
            with state_db.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM hitl_gates WHERE gate_id=?", (gate_id,)
                ).fetchone()
            return dict(row)

        time.sleep(poll_interval)


def decide(
    db_path: str | Path,
    gate_id: int,
    decision: str,
    decided_by: str,
    comment: str | None = None,
) -> dict:
    """Record an approval/denial. Returns the updated row."""
    state_db.decide_hitl(db_path, gate_id, decision, decided_by, comment)
    with state_db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM hitl_gates WHERE gate_id=?", (gate_id,)
        ).fetchone()
    return dict(row) if row else {}


def list_pending(db_path: str | Path) -> list[dict]:
    state_db.init_db(db_path)
    return state_db.list_pending_hitl(db_path)


def list_recent(db_path: str | Path, limit: int = 20) -> list[dict]:
    state_db.init_db(db_path)
    with state_db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hitl_gates ORDER BY requested_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── CLI ─────────────────────────────────────────────────────────────────

def _format_row(d: dict) -> str:
    sha = (d.get("target_sha") or "?")[:8]
    decision = d.get("decision", "?")
    requestor = d.get("requestor_agent", "?")
    age_s = time.time() - (d.get("requested_at") or time.time())
    decided = ""
    if d.get("decided_by"):
        decided = f" by={d['decided_by']}"
    return (
        f"#{d.get('gate_id')}  sha={sha}  {decision:<8}  "
        f"requestor={requestor}{decided}  age={age_s:.0f}s"
    )


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HITL approval gate CLI")
    p.add_argument(
        "--db",
        default="person_dataset/autoresearch_v2/state.db",
        help="Path to state.db",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List pending gates")
    p_list.add_argument("--all", action="store_true", help="Include decided gates")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="Show full prompt for a gate")
    p_show.add_argument("gate_id", type=int)

    p_approve = sub.add_parser("approve")
    p_approve.add_argument("gate_id", type=int)
    p_approve.add_argument("--by", default="cli", help="Identifier of decider")
    p_approve.add_argument("--comment", default=None)

    p_deny = sub.add_parser("deny")
    p_deny.add_argument("gate_id", type=int)
    p_deny.add_argument("--by", default="cli")
    p_deny.add_argument("--comment", default=None)

    args = p.parse_args(argv)

    if args.cmd == "list":
        rows = list_recent(args.db, args.limit) if args.all else list_pending(args.db)
        if args.json:
            print(json.dumps(rows, indent=2, default=float))
            return 0
        if not rows:
            print("hitl: no gates" + ("" if args.all else " pending"))
            return 0
        for r in rows:
            print(_format_row(r))
        return 0

    if args.cmd == "show":
        with state_db.connect(args.db) as conn:
            row = conn.execute(
                "SELECT * FROM hitl_gates WHERE gate_id=?", (args.gate_id,)
            ).fetchone()
        if not row:
            print(f"hitl: gate {args.gate_id} not found", file=sys.stderr)
            return 1
        d = dict(row)
        print(_format_row(d))
        print()
        print("--- prompt ---")
        print(d.get("prompt_text") or "(empty)")
        if d.get("comment"):
            print()
            print("--- comment ---")
            print(d["comment"])
        return 0

    if args.cmd in {"approve", "deny"}:
        rec = decide(
            args.db, args.gate_id,
            "approved" if args.cmd == "approve" else "denied",
            args.by, args.comment,
        )
        if not rec:
            print(f"hitl: gate {args.gate_id} not found", file=sys.stderr)
            return 1
        print(_format_row(rec))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
