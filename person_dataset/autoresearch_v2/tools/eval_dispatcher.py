"""Eval-side helpers for the v2 orchestrator.

This module owns the "did the experiment succeed?" judgement. It does
NOT run evaluate.py itself — that happens via train.py at the end of
each training run. Here we just parse the JSON outputs and apply the
quality gates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_TILE_METRICS = "last_metrics.json"
DEFAULT_PIPELINE_METRICS = "last_pipeline_metrics.json"


def load_json(path: str | Path) -> dict | None:
    """Read a JSON file. Return None on missing or unparseable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def parse_tile_metrics(path: str | Path = DEFAULT_TILE_METRICS) -> dict | None:
    return load_json(path)


def parse_pipeline_metrics(path: str | Path = DEFAULT_PIPELINE_METRICS) -> dict | None:
    return load_json(path)


def metrics_target_met(metrics: dict | None) -> bool:
    """Mirror of ollama_runner.py:metrics_target_met. Tile-level gate."""
    if not metrics:
        return False
    return bool(metrics.get("target_met", False))


def parse_cds(metrics: dict | None) -> float:
    """Extract CDS from a metrics dict, defaulting to 0.0 on missing/bad."""
    if not metrics:
        return 0.0
    cds = metrics.get("cds")
    if cds is None:
        return 0.0
    try:
        return float(cds)
    except (TypeError, ValueError):
        return 0.0


def pipeline_tile_drift(
    tile: dict | None,
    pipeline: dict | None,
    *,
    metric: str = "precision",
) -> float | None:
    """Compare tile vs pipeline metrics; return abs delta or None.

    Used by the orchestrator's drift stop condition: if tile P and
    pipeline P disagree by >10% for several rounds, the model has
    overfit to tiles and we should HITL-pause.
    """
    if not tile or not pipeline:
        return None
    tile_v = tile.get(metric)
    pipe_v = pipeline.get(f"pipeline_{metric}")
    if tile_v is None or pipe_v is None:
        return None
    try:
        return abs(float(tile_v) - float(pipe_v))
    except (TypeError, ValueError):
        return None


def cleanup_stale_metrics(
    tile_path: str | Path = DEFAULT_TILE_METRICS,
    pipeline_path: str | Path = DEFAULT_PIPELINE_METRICS,
) -> None:
    """Delete metric JSONs from a previous run so we never read stale data."""
    for p in (tile_path, pipeline_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
