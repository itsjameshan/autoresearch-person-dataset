"""v2 autoresearch orchestrator.

This is the Researcher-driven main loop that coexists with (and is
intended to eventually replace) ollama_runner.py. Key differences vs
the legacy loop:

- Decisions arrive as validated JSON (not free-form Python).
- All state lives in state.db (P1) instead of results.tsv. The legacy
  TSV is still updated for backward-compat with old tooling.
- Optional HITL (P3) gates expensive or risky decisions.
- Optional budget guardrails (P3) refuse to dispatch when the period
  is exhausted.
- Pipeline (large-image) eval runs every iteration (P2), not just on
  manual demand.

Usage:
    # Setup once
    export AUTORESEARCH_STATE_DB=person_dataset/autoresearch_v2/state.db
    python -m person_dataset.autoresearch_v2.budget init weekly --gpu-minutes 600
    python -m person_dataset.autoresearch_v2.migrate_results_tsv  # optional

    # Run
    python -m person_dataset.autoresearch_v2.orchestrator \\
        --max-iters 20 \\
        --budget-period weekly \\
        --llm-model qwen2.5:14b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

from . import budget as _budget
from . import db as state_db
from . import hitl
from .agents import researcher as _researcher
from .llm_client import LLMClient, OllamaError, resolve_ollama_model_name
from .tools import train_dispatcher as _trainer
from .tools import eval_dispatcher as _evaler


# ── Orchestrator-level constants (mirror ollama_runner.py) ─────────────

MAX_EXPERIMENTS = 20
TARGET_MET_STREAK_NEEDED = 3
MAX_CONSECUTIVE_DISCARDS = 5
CDS_KEEP_THRESHOLD = 0.001
COOLDOWN_SECONDS = 30
TRAIN_TIMEOUT_SEC = 3600
DRIFT_THRESHOLD = 0.10           # tile-vs-pipeline P/R drift
DRIFT_STREAK_FOR_HITL = 3
LLM_FAILURE_RATE_HALT = 0.30
LLM_FAILURE_MIN_CALLS = 5

# Default budget consumption for HITL "30% of remaining" rule
HITL_BUDGET_FRACTION = 0.30


# ── Legacy TSV append (compat with ollama_runner.py viewers) ───────────

RESULTS_TSV = "results.tsv"
TSV_HEADER = (
    "commit\tcds\tmAP50\tmAP50_95\tsmall_obj_recall\tprecision\trecall\t"
    "counting_mae\tinference_ms\tlatency_score\tmemory_gb\tepochs\tstatus\tdescription\n"
)


def _append_tsv(commit: str, metrics: dict | None, status: str, description: str) -> None:
    """Mirror of ollama_runner.py:append_result — keeps results.tsv usable."""
    metrics = metrics or {}
    mem_mb = metrics.get("peak_memory_mb", 0.0) or 0.0
    try:
        mem_gb = f"{float(mem_mb) / 1024:.1f}"
    except (TypeError, ValueError):
        mem_gb = "0.0"
    row = (
        f"{commit}\t"
        f"{metrics.get('cds', '0.0000')}\t"
        f"{metrics.get('mAP50', '0.0000')}\t"
        f"{metrics.get('mAP50_95', '0.0000')}\t"
        f"{metrics.get('small_obj_recall', '0.0000')}\t"
        f"{metrics.get('precision', '0.0000')}\t"
        f"{metrics.get('recall', '0.0000')}\t"
        f"{metrics.get('counting_mae', '0.0')}\t"
        f"{metrics.get('inference_ms', '0.0')}\t"
        f"{metrics.get('latency_score', '0.0000')}\t"
        f"{mem_gb}\t"
        f"{metrics.get('epochs_completed', '0')}\t"
        f"{status}\t{description}\n"
    )
    p = Path(RESULTS_TSV)
    if not p.exists():
        p.write_text(TSV_HEADER, encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write(row)


# ── Stop-condition state ───────────────────────────────────────────────

class LoopState:
    """Mutable state tracked across iterations. Mirrors ollama_runner main()."""

    def __init__(self):
        self.best_cds: float = 0.0
        self.best_sha: str | None = None
        self.best_target_met: bool = False
        self.consecutive_discards: int = 0
        self.target_met_streak: int = 0
        self.drift_streak: int = 0
        self.iteration: int = 0
        self.rolling_train_min: deque[float] = deque(maxlen=5)

    def restore_from_db(self, db_path: str) -> None:
        """Hydrate state from prior experiments so the loop is resumable."""
        best = state_db.get_best_experiment(db_path, metric="cds")
        if best:
            try:
                tm = json.loads(best.get("tile_metrics_json") or "{}")
                self.best_cds = float(tm.get("cds", 0.0))
                self.best_sha = best.get("git_sha")
                self.best_target_met = bool(tm.get("target_met", False))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        # Iteration count = number of prior experiments
        prior = state_db.read_recent_experiments(db_path, limit=1000)
        # Only count rows touched by v2 (skip ollama_runner_legacy migration rows)
        v2_rows = [r for r in prior if r.get("agent_made_decision") not in (None, "ollama_runner_legacy")]
        self.iteration = len(v2_rows)


# ── Single-iteration body ──────────────────────────────────────────────

def _read_or_default(path: str, default: str = "") -> str:
    p = Path(path)
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return default
    return default


def _build_context(
    db_path: str,
    state: LoopState,
    last_pipeline_metrics: dict | None,
) -> dict:
    pipeline_summary = "(pending — first iteration)"
    if last_pipeline_metrics:
        pipeline_summary = (
            f"precision={last_pipeline_metrics.get('pipeline_precision', '?')}  "
            f"recall={last_pipeline_metrics.get('pipeline_recall', '?')}  "
            f"f1={last_pipeline_metrics.get('pipeline_f1', '?')}  "
            f"target_met={last_pipeline_metrics.get('pipeline_target_met', '?')}"
        )
    return {
        "iteration": state.iteration + 1,
        "best_cds": state.best_cds,
        "best_target_met": state.best_target_met,
        "consecutive_discards": state.consecutive_discards,
        "target_met_streak": state.target_met_streak,
        "recent_experiments": _trainer.summarize_recent_experiments(db_path, limit=10),
        "current_config": _trainer.read_train_config(),
        "pipeline_summary": pipeline_summary,
        "suggestions": _read_or_default("suggestions.md", "(none)"),
        "current_imgsz": 1280,
    }


def _maybe_force_hitl(
    decision: _researcher.Decision,
    db_path: str,
    budget_period: str | None,
) -> bool:
    """Force needs_hitl=True if estimated cost exceeds 30% of remaining
    budget. Returns the (possibly-updated) needs_hitl value."""
    if decision.needs_hitl:
        return True
    if budget_period is None or decision.estimated_gpu_minutes is None:
        return False
    rem = state_db.budget_remaining(db_path, budget_period)
    if not rem.get("exists") or rem.get("gpu_minutes_remaining") is None:
        return False
    remaining = rem["gpu_minutes_remaining"]
    if remaining <= 0:
        return True
    if decision.estimated_gpu_minutes >= HITL_BUDGET_FRACTION * remaining:
        return True
    return False


def run_one_iteration(
    args,
    state: LoopState,
    client: LLMClient,
    db_path: str,
    last_pipeline_metrics: dict | None,
) -> tuple[str, dict | None]:
    """Returns (status, pipeline_metrics_or_none).

    status is one of: keep, discard, crash, stop, halt.
    """
    state.iteration += 1
    print()
    print("=" * 60)
    print(f"v2 orchestrator — iteration #{state.iteration}/{args.max_iters}")
    print(
        f"  best_cds={state.best_cds:.4f}  "
        f"streak={state.target_met_streak}  "
        f"discards={state.consecutive_discards}"
    )
    print("=" * 60)

    # 1. Stop conditions BEFORE consulting LLM
    if state.target_met_streak >= TARGET_MET_STREAK_NEEDED:
        print(f"[stop] {TARGET_MET_STREAK_NEEDED} consecutive target_met keeps.")
        return ("stop", last_pipeline_metrics)
    if state.consecutive_discards >= MAX_CONSECUTIVE_DISCARDS:
        print(f"[stop] {MAX_CONSECUTIVE_DISCARDS} consecutive discards — plateau.")
        # Generate plateau analysis as a side effect
        history = _trainer.summarize_recent_experiments(db_path, limit=20)
        analysis = _researcher.plateau_analysis(client, history)
        Path("suggestions.md").write_text(
            f"# Plateau analysis (after iteration {state.iteration})\n\n{analysis}\n",
            encoding="utf-8",
        )
        return ("stop", last_pipeline_metrics)
    if (
        client.call_count >= LLM_FAILURE_MIN_CALLS
        and client.failure_rate > LLM_FAILURE_RATE_HALT
    ):
        print(f"[halt] LLM failure rate {client.failure_rate:.0%} > "
              f"{LLM_FAILURE_RATE_HALT:.0%} — Ollama probably down.")
        return ("halt", last_pipeline_metrics)

    # 2. Get a decision from the Researcher
    context = _build_context(db_path, state, last_pipeline_metrics)
    try:
        decision = _researcher.propose(client, context, max_retries=3)
    except _researcher.ValidationError as e:
        print(f"[halt] researcher validation exhausted: {e}")
        return ("halt", last_pipeline_metrics)
    except OllamaError as e:
        print(f"[skip] LLM error: {e}")
        state.consecutive_discards += 1
        return ("crash", last_pipeline_metrics)

    print(f"[researcher] action={decision.action_type}")
    print(f"             rationale: {decision.rationale}")
    if decision.config_patch:
        print(f"             patch: {decision.config_patch}")

    # 3. Persist the decision
    state_db.record_decision(
        db_path,
        parent_sha=state.best_sha,
        made_by_agent="researcher",
        action_type=decision.action_type,
        rationale=decision.rationale,
        expected_delta=decision.expected_cds_delta,
        estimated_cost=decision.estimated_gpu_minutes,
        needs_hitl=decision.needs_hitl,
        config_patch=decision.config_patch or None,
    )

    # 4. Stop?
    if decision.action_type == "stop":
        print("[stop] researcher requested stop.")
        return ("stop", last_pipeline_metrics)

    # 5. Pre-dispatch checks (budget + HITL)
    if args.budget_period and decision.estimated_gpu_minutes:
        check = _budget.check_can_spend(
            db_path,
            args.budget_period,
            gpu_minutes=decision.estimated_gpu_minutes,
            hard=False,
        )
        if not check["ok"]:
            print(f"[stop] budget guard: {check['reason']}")
            return ("stop", last_pipeline_metrics)

    needs_hitl = _maybe_force_hitl(decision, db_path, args.budget_period)
    if needs_hitl and not args.no_hitl:
        print("[hitl] this decision needs human approval. Waiting...")
        prompt_text = (
            f"Iteration {state.iteration} — proposed action_type="
            f"{decision.action_type}\n\n"
            f"rationale: {decision.rationale}\n\n"
            f"config_patch: {json.dumps(decision.config_patch, indent=2)}\n\n"
            f"estimated_gpu_minutes: {decision.estimated_gpu_minutes}\n"
            f"expected_cds_delta: {decision.expected_cds_delta}\n"
        )
        record = hitl.request_approval(
            db_path,
            target_sha=state.best_sha,
            prompt_text=prompt_text,
            requestor_agent="orchestrator",
            timeout_seconds=args.hitl_timeout if args.hitl_timeout > 0 else None,
            on_timeout=args.hitl_on_timeout,
        )
        if record["decision"] != "approved":
            print(f"[skip] HITL {record['decision']} (by {record.get('decided_by')}): "
                  f"{record.get('comment')}")
            state.consecutive_discards += 1
            return ("discard", last_pipeline_metrics)

    # 6. Action types other than patch_train_config — non-blocking advisory
    if decision.action_type != "patch_train_config":
        print(f"[note] action_type={decision.action_type} not yet implemented; "
              f"recording decision and skipping training this round.")
        return ("discard", last_pipeline_metrics)

    # 7. Apply config patch + git commit
    try:
        rewritten = _trainer.apply_config_patch(decision.config_patch)
        for ln in rewritten:
            print(f"  patched: {ln}")
    except ValueError as e:
        print(f"[skip] apply_config_patch failed: {e}")
        state.consecutive_discards += 1
        return ("discard", last_pipeline_metrics)

    if args.dry_run:
        print("[dry-run] would commit + train. exiting iteration here.")
        return ("discard", last_pipeline_metrics)

    new_sha = _trainer.git_commit(f"experiment: {decision.rationale}")
    sha_short = new_sha[:7]

    # 8. Run training (this hooks state.db via env var)
    print(f"[train] starting (timeout={args.train_timeout}s, sha={sha_short})...")
    success, tile_metrics, pipeline_metrics, duration = _trainer.run_training(
        timeout_sec=args.train_timeout,
        state_db_path=db_path,
    )
    state.rolling_train_min.append(duration / 60.0)

    # 9. Budget consumption (always — even on crash, to keep accounting honest)
    if args.budget_period:
        try:
            _budget.commit_spend(db_path, args.budget_period,
                                 gpu_minutes=duration / 60.0)
        except Exception as e:
            print(f"[warn] budget commit failed: {e}")

    # 10. Outcome handling
    if not success:
        print("[crash] training/eval failed. Rolling back commit.")
        _append_tsv(sha_short, {}, "crash", decision.rationale)
        state_db.upsert_experiment(
            db_path, new_sha,
            status="crashed",
            ended_at=time.time(),
            description=decision.rationale,
            agent_made_decision="researcher",
            action_type=decision.action_type,
        )
        try:
            _trainer.git_reset_hard("HEAD~1")
        except Exception as e:
            print(f"[warn] git reset failed: {e}")
        state.consecutive_discards += 1
        return ("crash", last_pipeline_metrics)

    cds = _evaler.parse_cds(tile_metrics)
    target_met = _evaler.metrics_target_met(tile_metrics)
    improvement = cds - state.best_cds
    print(f"[result] cds={cds:.4f}  best={state.best_cds:.4f}  Δ={improvement:+.4f}  "
          f"target_met={target_met}  duration={duration/60:.1f}min")

    # Drift check: tile vs pipeline precision/recall
    if pipeline_metrics:
        d_p = _evaler.pipeline_tile_drift(tile_metrics, pipeline_metrics, metric="precision")
        d_r = _evaler.pipeline_tile_drift(tile_metrics, pipeline_metrics, metric="recall")
        drift = max(d_p or 0.0, d_r or 0.0)
        if drift > DRIFT_THRESHOLD:
            state.drift_streak += 1
            print(f"[drift] tile-vs-pipeline = {drift:.3f} (streak={state.drift_streak})")
        else:
            state.drift_streak = 0

    # 11. Keep/discard decision
    if improvement > CDS_KEEP_THRESHOLD:
        status = "keep" + (" [TARGET_MET]" if target_met else "")
        print(f"[keep] sha={sha_short}")
        state.best_cds = cds
        state.best_sha = new_sha
        state.best_target_met = target_met
        if target_met:
            state.target_met_streak += 1
        else:
            state.target_met_streak = 0
        state.consecutive_discards = 0
        if not args.no_push:
            try:
                _trainer.git_push()
            except Exception as e:
                print(f"[warn] git push failed: {e}")
    else:
        status = "discard"
        print(f"[discard] sha={sha_short} — rolling back.")
        try:
            _trainer.git_reset_hard("HEAD~1")
        except Exception as e:
            print(f"[warn] git reset failed: {e}")
        state.consecutive_discards += 1
        state.target_met_streak = 0

    # Always log to legacy TSV + state.db
    _append_tsv(sha_short, tile_metrics, status, decision.rationale)
    state_db.upsert_experiment(
        db_path, new_sha,
        status=status.split()[0],  # 'keep' or 'discard'
        ended_at=time.time(),
        gpu_minutes=round(duration / 60.0, 2),
        epochs_completed=(tile_metrics or {}).get("epochs_completed"),
        description=decision.rationale,
        agent_made_decision="researcher",
        action_type=decision.action_type,
    )

    # Drift -> HITL
    if state.drift_streak >= DRIFT_STREAK_FOR_HITL and not args.no_hitl:
        print("[drift-hitl] tile-vs-pipeline drift sustained — requesting human review.")
        hitl.request_approval(
            db_path,
            target_sha=new_sha,
            prompt_text=(
                f"Tile vs pipeline metrics have drifted by >10% for "
                f"{state.drift_streak} iterations. The model may be "
                f"overfitting to tile statistics. Investigate."
            ),
            requestor_agent="orchestrator",
            timeout_seconds=None,
            on_timeout="raise",
        )
        state.drift_streak = 0  # reset after escalation

    return (status.split()[0], pipeline_metrics)


# ── Main entry point ───────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-db",
        default=os.environ.get(
            "AUTORESEARCH_STATE_DB",
            "person_dataset/autoresearch_v2/state.db",
        ),
    )
    p.add_argument("--max-iters", type=int, default=MAX_EXPERIMENTS)
    p.add_argument("--budget-period", default=None,
                   help="Period name to enforce, e.g. 'weekly'. None = no budget guard.")
    p.add_argument("--llm-model", default=None,
                   help="Override OLLAMA_MODEL env var")
    p.add_argument("--llm-url", default=None,
                   help="Override OLLAMA_URL env var")
    p.add_argument("--train-timeout", type=int, default=TRAIN_TIMEOUT_SEC)
    p.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS,
                   help="Seconds between iterations (thermal management)")
    p.add_argument("--no-push", action="store_true",
                   help="Skip git push after keep")
    p.add_argument("--no-hitl", action="store_true",
                   help="Skip HITL gate even when needs_hitl=True (USE WITH CARE)")
    p.add_argument("--hitl-timeout", type=int, default=0,
                   help="Seconds to wait for HITL decision; 0 = forever")
    p.add_argument("--hitl-on-timeout", choices=["deny", "approve", "raise"],
                   default="deny")
    p.add_argument("--dry-run", action="store_true",
                   help="Patch train.py and stop (don't commit, don't train)")
    args = p.parse_args(argv)

    state_db.init_db(args.state_db)
    client = LLMClient(model=args.llm_model, url=args.llm_url)
    client.model = resolve_ollama_model_name(client, args.llm_model or client.model)

    state = LoopState()
    state.restore_from_db(args.state_db)

    print("=" * 60)
    print("v2 autoresearch orchestrator")
    print(f"  state.db:      {args.state_db}")
    print(f"  llm:           {client.model} @ {client.url}")
    print(f"  max iters:     {args.max_iters} (already done: {state.iteration})")
    print(f"  budget period: {args.budget_period or '(none)'}")
    print(f"  resume best:   sha={(state.best_sha or '?')[:8]} cds={state.best_cds:.4f}")
    print("=" * 60)

    last_pipeline_metrics: dict | None = _evaler.parse_pipeline_metrics()
    final_status = "incomplete"

    while state.iteration < args.max_iters:
        try:
            status, last_pipeline_metrics = run_one_iteration(
                args, state, client, args.state_db, last_pipeline_metrics,
            )
        except KeyboardInterrupt:
            print("\n[orchestrator] interrupted by user.")
            final_status = "interrupted"
            break
        if status in {"stop", "halt"}:
            final_status = status
            break
        if state.iteration < args.max_iters:
            print(f"[cooldown] sleeping {args.cooldown}s before next iteration...")
            time.sleep(args.cooldown)
    else:
        final_status = "max_iters"

    print()
    print("=" * 60)
    print(f"final status: {final_status}")
    print(f"  iterations:    {state.iteration}")
    print(f"  best CDS:      {state.best_cds:.4f}")
    print(f"  best sha:      {(state.best_sha or '?')[:12]}")
    print(f"  llm calls:     {client.call_count}  failures={client.failure_count} "
          f"({client.failure_rate:.0%})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
