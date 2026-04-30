"""One-time migration: import legacy results.tsv history into state.db.

Existing rows in `experiments` are not overwritten — re-running this script
is safe and will only add experiments not already present. Use this once,
when first switching a project from ollama_runner.py to the v2 system.

Usage:
    python -m person_dataset.autoresearch_v2.migrate_results_tsv \\
        --tsv results.tsv \\
        --db person_dataset/autoresearch_v2/state.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db as state_db


# results.tsv schema (14 columns, see ollama_runner.py:append_result())
TSV_COLS = [
    "commit", "cds", "mAP50", "mAP50_95", "small_obj_recall", "precision",
    "recall", "counting_mae", "inference_ms", "latency_score", "memory_gb",
    "epochs", "status", "description",
]
# Columns mapped into tile_metrics_json (everything except commit/epochs/status/description)
_METRIC_COLS = {
    "cds", "mAP50", "mAP50_95", "small_obj_recall", "precision", "recall",
    "counting_mae", "inference_ms", "latency_score",
}


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_row(row: list[str]) -> dict | None:
    """Parse one TSV row. Returns None if malformed."""
    if len(row) < len(TSV_COLS):
        return None
    rec = dict(zip(TSV_COLS, row[: len(TSV_COLS)]))
    if not rec["commit"]:
        return None

    metrics = {}
    for k in _METRIC_COLS:
        v = _to_float(rec.get(k, ""))
        if v is not None:
            metrics[k] = v
    # memory_gb in TSV -> peak_memory_mb in metrics dict
    mem_gb = _to_float(rec.get("memory_gb", ""))
    if mem_gb is not None:
        metrics["peak_memory_mb"] = round(mem_gb * 1024, 1)

    epochs = None
    try:
        epochs = int(rec.get("epochs", "") or 0)
    except (TypeError, ValueError):
        epochs = None

    return {
        "git_sha": rec["commit"].strip(),
        "status": rec["status"].strip() or "unknown",
        "description": rec["description"].strip(),
        "epochs_completed": epochs,
        "tile_metrics_json": json.dumps(metrics) if metrics else None,
    }


def migrate(tsv_path: Path, db_path: Path) -> tuple[int, int]:
    """Import TSV rows into state.db. Returns (imported, skipped) counts."""
    state_db.init_db(db_path)

    if not tsv_path.exists():
        return (0, 0)

    imported = 0
    skipped = 0
    with tsv_path.open("r", encoding="utf-8") as f:
        header = f.readline()  # discard header
        for line_no, raw in enumerate(f, start=2):
            row = raw.rstrip("\n").split("\t")
            parsed = _parse_row(row)
            if parsed is None:
                skipped += 1
                continue

            existing = state_db.get_experiment(db_path, parsed["git_sha"])
            if existing is not None:
                skipped += 1
                continue

            try:
                state_db.upsert_experiment(
                    db_path,
                    parsed["git_sha"],
                    status=parsed["status"],
                    description=parsed["description"],
                    epochs_completed=parsed["epochs_completed"],
                    tile_metrics_json=parsed["tile_metrics_json"],
                    agent_made_decision="ollama_runner_legacy",
                )
                imported += 1
            except Exception as e:
                print(
                    f"migrate: line {line_no} skipped: {e}",
                    file=sys.stderr,
                )
                skipped += 1

    return imported, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="results.tsv", help="Path to legacy results.tsv")
    parser.add_argument(
        "--db",
        default="person_dataset/autoresearch_v2/state.db",
        help="Path to state.db (created if missing)",
    )
    args = parser.parse_args()

    tsv = Path(args.tsv)
    db = Path(args.db)
    imported, skipped = migrate(tsv, db)
    print(f"migrate: imported={imported} skipped={skipped} from {tsv} -> {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
