"""Budget guardrails for the v2 autoresearch loop.

Wraps the `budget` table in state.db with a high-level API and a small CLI.
The orchestrator uses this to (a) refuse to dispatch a training run that
would blow the configured limit, and (b) record the actual GPU-minutes /
dollars spent after each run completes.

Two supported periods are intentional: 'weekly' for soft caps that reset,
'total' for hard project-lifetime caps. Other period names are accepted
but interpreted as opaque buckets — no automatic reset logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db as state_db


class BudgetExceeded(RuntimeError):
    """Raised when an attempted spend would exceed the configured limit."""


def init(
    db_path: str | Path,
    period: str,
    gpu_minutes_limit: float | None = None,
    dollars_limit: float | None = None,
) -> None:
    """Create or reset the limit configuration for a period."""
    state_db.init_db(db_path)
    state_db.init_budget(
        db_path, period,
        gpu_minutes_limit=gpu_minutes_limit,
        dollars_limit=dollars_limit,
    )


def show(db_path: str | Path, period: str | None = None) -> list[dict]:
    """Return budget rows (one per period) with remaining computed."""
    state_db.init_db(db_path)
    if period:
        d = state_db.budget_remaining(db_path, period)
        return [d] if d.get("exists") else []
    # No period filter — list all
    with state_db.connect(db_path) as conn:
        rows = conn.execute("SELECT period FROM budget ORDER BY period").fetchall()
    return [state_db.budget_remaining(db_path, r["period"]) for r in rows]


def check_can_spend(
    db_path: str | Path,
    period: str,
    gpu_minutes: float = 0.0,
    dollars: float = 0.0,
    *,
    hard: bool = True,
) -> dict:
    """Test whether spending the proposed amount would stay within budget.

    Returns a dict with keys:
        ok: bool
        reason: str | None — explanation when ok is False
        gpu_minutes_remaining_after: float | None
        dollars_remaining_after: float | None

    When hard=True (default) and ok is False, raises BudgetExceeded instead
    of returning. Use hard=False for advisory checks (e.g., warning before
    a HITL gate).
    """
    state_db.init_db(db_path)
    rem = state_db.budget_remaining(db_path, period)
    if not rem.get("exists"):
        result = {
            "ok": False,
            "reason": f"budget period {period!r} not initialized",
            "gpu_minutes_remaining_after": None,
            "dollars_remaining_after": None,
        }
        if hard:
            raise BudgetExceeded(result["reason"])
        return result

    gpu_after = None
    if rem["gpu_minutes_limit"] is not None:
        gpu_after = rem["gpu_minutes_remaining"] - gpu_minutes
    dol_after = None
    if rem["dollars_limit"] is not None:
        dol_after = rem["dollars_remaining"] - dollars

    failures = []
    if gpu_after is not None and gpu_after < 0:
        failures.append(
            f"would exceed gpu_minutes (need {gpu_minutes:.1f}, have "
            f"{rem['gpu_minutes_remaining']:.1f})"
        )
    if dol_after is not None and dol_after < 0:
        failures.append(
            f"would exceed dollars (need {dollars:.4f}, have "
            f"{rem['dollars_remaining']:.4f})"
        )

    result = {
        "ok": not failures,
        "reason": "; ".join(failures) if failures else None,
        "gpu_minutes_remaining_after": gpu_after,
        "dollars_remaining_after": dol_after,
    }
    if hard and not result["ok"]:
        raise BudgetExceeded(result["reason"])
    return result


def commit_spend(
    db_path: str | Path,
    period: str,
    gpu_minutes: float = 0.0,
    dollars: float = 0.0,
) -> dict:
    """Record actual consumption. Does NOT enforce the limit — call this
    after the work is done, even if it overran (so logs are accurate)."""
    state_db.init_db(db_path)
    return state_db.consume_budget(db_path, period, gpu_minutes=gpu_minutes, dollars=dollars)


def sync_from_experiments(db_path: str | Path, period: str) -> dict:
    """Re-derive gpu_minutes_used by summing experiments.gpu_minutes.

    Use this if the run-by-run commit_spend calls might have drifted from
    the actual experiment log (e.g., crashed mid-write). Always preserves
    dollars_used (LLM costs aren't tracked in experiments table).
    """
    state_db.init_db(db_path)
    with state_db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(gpu_minutes), 0) AS total FROM experiments"
        ).fetchone()
        total = float(row["total"] or 0.0)
        conn.execute(
            "UPDATE budget SET gpu_minutes_used = ? WHERE period = ?",
            (total, period),
        )
    return state_db.budget_remaining(db_path, period)


# ── CLI ─────────────────────────────────────────────────────────────────

def _format_row(d: dict) -> str:
    period = d.get("period", "?")
    gpu_used = d.get("gpu_minutes_used") or 0.0
    gpu_lim = d.get("gpu_minutes_limit")
    dol_used = d.get("dollars_used") or 0.0
    dol_lim = d.get("dollars_limit")
    parts = [f"period={period!r}"]
    if gpu_lim is not None:
        parts.append(f"gpu_min={gpu_used:.1f}/{gpu_lim:.1f}")
    else:
        parts.append(f"gpu_min={gpu_used:.1f}/∞")
    if dol_lim is not None:
        parts.append(f"$={dol_used:.4f}/{dol_lim:.4f}")
    else:
        parts.append(f"$={dol_used:.4f}/∞")
    return "  ".join(parts)


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Budget guardrails for v2 autoresearch")
    p.add_argument(
        "--db",
        default="person_dataset/autoresearch_v2/state.db",
        help="Path to state.db",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Set or reset budget limits for a period")
    p_init.add_argument("period", help="e.g. weekly, total")
    p_init.add_argument("--gpu-minutes", type=float, default=None)
    p_init.add_argument("--dollars", type=float, default=None)

    p_show = sub.add_parser("show", help="Show budget remaining")
    p_show.add_argument("period", nargs="?", default=None)
    p_show.add_argument("--json", action="store_true")

    p_check = sub.add_parser("check", help="Test if a spend would fit (advisory)")
    p_check.add_argument("period")
    p_check.add_argument("--gpu-minutes", type=float, default=0.0)
    p_check.add_argument("--dollars", type=float, default=0.0)

    p_commit = sub.add_parser("commit", help="Record actual consumption")
    p_commit.add_argument("period")
    p_commit.add_argument("--gpu-minutes", type=float, default=0.0)
    p_commit.add_argument("--dollars", type=float, default=0.0)

    p_sync = sub.add_parser("sync", help="Re-derive gpu_minutes_used from experiments table")
    p_sync.add_argument("period")

    args = p.parse_args(argv)

    if args.cmd == "init":
        init(args.db, args.period,
             gpu_minutes_limit=args.gpu_minutes, dollars_limit=args.dollars)
        print(f"budget: {args.period} initialized "
              f"(gpu_minutes_limit={args.gpu_minutes}, dollars_limit={args.dollars})")
        return 0

    if args.cmd == "show":
        rows = show(args.db, args.period)
        if args.json:
            print(json.dumps(rows, indent=2, default=float))
            return 0
        if not rows:
            print(f"budget: no period {args.period or '*'} found")
            return 1
        for d in rows:
            print(_format_row(d))
        return 0

    if args.cmd == "check":
        result = check_can_spend(
            args.db, args.period,
            gpu_minutes=args.gpu_minutes, dollars=args.dollars,
            hard=False,
        )
        print(json.dumps(result, indent=2, default=float))
        return 0 if result["ok"] else 2

    if args.cmd == "commit":
        d = commit_spend(args.db, args.period,
                         gpu_minutes=args.gpu_minutes, dollars=args.dollars)
        print(_format_row(d))
        return 0

    if args.cmd == "sync":
        d = sync_from_experiments(args.db, args.period)
        print(_format_row(d))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
