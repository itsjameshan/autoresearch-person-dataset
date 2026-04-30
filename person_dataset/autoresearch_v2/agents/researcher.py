"""Researcher agent — proposes the next experiment.

This is the v2 replacement for the Ollama-driven prompt loop in
ollama_runner.py:559-626. Differences:
- Output is structured JSON (researcher_decision.json schema), not free
  text + Python assignments.
- Output is validated against an explicit allowlist (MODEL path
  pattern, BATCH bounds, the 28 train.py variable names) before being
  acted on. Validation failure triggers a re-prompt, up to N retries.
- Agent does NOT touch the filesystem or git — it just returns a
  Decision object. The orchestrator does the dispatch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient, OllamaError


# ── Validation rules ───────────────────────────────────────────────────

ALLOWED_VARS: dict[str, dict] = {
    "MODEL":        {"type": str, "match": re.compile(r"^person_dataset/[A-Za-z0-9_.\-]+\.pt$")},
    "IMGSZ":        {"type": int, "min": 320, "max": 2560},
    "EPOCHS":       {"type": int, "min": 1, "max": 1000},
    "BATCH":        {"type": int, "min": 1, "max": 64},
    "PATIENCE":     {"type": int, "min": 0, "max": 1000},
    "DEVICE":       {"type": str},
    "LR0":          {"type": float, "min": 1e-7, "max": 1.0},
    "LRF":          {"type": float, "min": 1e-7, "max": 1.0},
    "COS_LR":       {"type": bool},
    "HSV_H":        {"type": float, "min": 0.0, "max": 1.0},
    "HSV_S":        {"type": float, "min": 0.0, "max": 1.0},
    "HSV_V":        {"type": float, "min": 0.0, "max": 1.0},
    "DEGREES":      {"type": float, "min": 0.0, "max": 180.0},
    "TRANSLATE":    {"type": float, "min": 0.0, "max": 1.0},
    "SCALE":        {"type": float, "min": 0.0, "max": 1.0},
    "FLIPUD":       {"type": float, "min": 0.0, "max": 1.0},
    "FLIPLR":       {"type": float, "min": 0.0, "max": 1.0},
    "MOSAIC":       {"type": float, "min": 0.0, "max": 1.0},
    "MIXUP":        {"type": float, "min": 0.0, "max": 1.0},
    "COPY_PASTE":   {"type": float, "min": 0.0, "max": 1.0},
    "ERASING":      {"type": float, "min": 0.0, "max": 1.0},
    "CLOSE_MOSAIC": {"type": int, "min": 0, "max": 100},
    "BOX":          {"type": float, "min": 0.0, "max": 100.0},
    "CLS":          {"type": float, "min": 0.0, "max": 100.0},
    "AMP":          {"type": bool},
    # CACHE accepts "ram", "disk", "False", or False
    "CACHE":        {"type": "cache_value"},
    "WORKERS":      {"type": int, "min": 0, "max": 64},
    "SINGLE_CLS":   {"type": bool},
}

ALLOWED_ACTION_TYPES = {
    "patch_train_config",
    "hpo_sweep",
    "request_pipeline_eval",
    "request_data_collection",
    "stop",
}

# Hardware constraint: BATCH * (IMGSZ/1280)^2 should fit in 12 GB VRAM.
# Plan caveat: BATCH <= 16 at imgsz=1280.
MAX_BATCH_AT_1280 = 16


class ValidationError(ValueError):
    """The Researcher's JSON failed schema validation."""


@dataclass
class Decision:
    """Validated Researcher output ready for the orchestrator to dispatch."""
    action_type: str
    rationale: str
    config_patch: dict = field(default_factory=dict)
    expected_cds_delta: float | None = None
    estimated_gpu_minutes: float | None = None
    needs_hitl: bool = False
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "rationale": self.rationale,
            "config_patch": self.config_patch,
            "expected_cds_delta": self.expected_cds_delta,
            "estimated_gpu_minutes": self.estimated_gpu_minutes,
            "needs_hitl": self.needs_hitl,
        }


def _coerce_type(name: str, value: Any) -> Any:
    """Coerce common LLM-output type sloppiness (e.g. "12" -> 12)."""
    spec = ALLOWED_VARS[name]
    t = spec["type"]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes"}:
                return True
            if low in {"false", "0", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValidationError(f"{name}: cannot coerce {value!r} to bool")
    if t is int:
        if isinstance(value, bool):
            raise ValidationError(f"{name}: bool not allowed where int expected")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as e:
                raise ValidationError(f"{name}: {e}") from None
        raise ValidationError(f"{name}: not an int: {value!r}")
    if t is float:
        if isinstance(value, bool):
            raise ValidationError(f"{name}: bool not allowed where float expected")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise ValidationError(f"{name}: not a float: {value!r}")
        raise ValidationError(f"{name}: not a number: {value!r}")
    if t is str:
        if isinstance(value, str):
            return value
        raise ValidationError(f"{name}: not a string: {value!r}")
    if t == "cache_value":
        if value is False or value == "False":
            return False
        if isinstance(value, str) and value in {"ram", "disk"}:
            return value
        raise ValidationError(f"CACHE: must be 'ram'|'disk'|False, got {value!r}")
    raise ValidationError(f"{name}: unknown type spec {t}")


def _validate_var(name: str, value: Any) -> Any:
    spec = ALLOWED_VARS[name]
    coerced = _coerce_type(name, value)
    if "match" in spec:
        if not spec["match"].match(coerced):
            raise ValidationError(
                f"{name}: value {coerced!r} does not match required pattern"
            )
    if "min" in spec and coerced < spec["min"]:
        raise ValidationError(f"{name}: {coerced} < min {spec['min']}")
    if "max" in spec and coerced > spec["max"]:
        raise ValidationError(f"{name}: {coerced} > max {spec['max']}")
    return coerced


def validate_decision(raw: dict, current_imgsz: int = 1280) -> Decision:
    """Validate a parsed JSON dict against the Researcher contract.

    Raises ValidationError with a concise message on any violation.
    """
    if not isinstance(raw, dict):
        raise ValidationError(f"top-level must be object, got {type(raw).__name__}")
    if "action_type" not in raw:
        raise ValidationError("missing required field: action_type")
    if "rationale" not in raw:
        raise ValidationError("missing required field: rationale")

    action_type = raw["action_type"]
    if action_type not in ALLOWED_ACTION_TYPES:
        raise ValidationError(
            f"action_type {action_type!r} not in {sorted(ALLOWED_ACTION_TYPES)}"
        )

    rationale = raw["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValidationError("rationale must be non-empty string")
    if len(rationale) > 1000:
        raise ValidationError(f"rationale too long: {len(rationale)} > 1000")

    config_patch_raw = raw.get("config_patch", {}) or {}
    if not isinstance(config_patch_raw, dict):
        raise ValidationError("config_patch must be object")

    config_patch: dict[str, Any] = {}
    if action_type == "patch_train_config":
        if not config_patch_raw:
            raise ValidationError(
                "patch_train_config requires non-empty config_patch"
            )
        for k, v in config_patch_raw.items():
            if k not in ALLOWED_VARS:
                raise ValidationError(
                    f"config_patch[{k!r}] is not in 28-variable allowlist"
                )
            config_patch[k] = _validate_var(k, v)
        # Hardware cross-check: BATCH cap kicks in at imgsz>=1280 (more pixels
        # per sample => less batch room in 12GB VRAM). Smaller imgsz allows
        # larger batches.
        imgsz = config_patch.get("IMGSZ", current_imgsz)
        batch = config_patch.get("BATCH")
        if batch is not None and imgsz >= 1280 and batch > MAX_BATCH_AT_1280:
            raise ValidationError(
                f"BATCH={batch} exceeds {MAX_BATCH_AT_1280} at imgsz>=1280"
            )
    elif config_patch_raw:
        # config_patch on non-patch action types is suspicious but tolerated.
        # Whitelist values anyway so we don't store nonsense.
        for k, v in config_patch_raw.items():
            if k in ALLOWED_VARS:
                config_patch[k] = _validate_var(k, v)

    return Decision(
        action_type=action_type,
        rationale=rationale.strip(),
        config_patch=config_patch,
        expected_cds_delta=raw.get("expected_cds_delta"),
        estimated_gpu_minutes=raw.get("estimated_gpu_minutes"),
        needs_hitl=bool(raw.get("needs_hitl", False)),
        raw=raw,
    )


# ── Prompt rendering ───────────────────────────────────────────────────

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "researcher.md"
)


def render_prompt(context: dict) -> str:
    """Render the researcher prompt with iteration context.

    Required keys in context: iteration, best_cds, best_target_met,
    consecutive_discards, target_met_streak, recent_experiments,
    current_config, pipeline_summary, suggestions.
    """
    template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(**context)


# ── Main entry point ──────────────────────────────────────────────────

def propose(
    client: LLMClient,
    context: dict,
    *,
    max_retries: int = 3,
) -> Decision:
    """Ask the LLM for a decision, validate, retry up to max_retries times.

    Raises ValidationError if all retries fail (orchestrator should halt).
    Raises OllamaError if the LLM is unreachable.
    """
    base_prompt = render_prompt(context)
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            prompt = base_prompt
        else:
            prompt = (
                base_prompt
                + f"\n\n# Previous attempt {attempt - 1} was rejected\n"
                + f"Validation error: {last_error}\n"
                + "Fix the JSON and respond again. JSON ONLY.\n"
            )
        try:
            raw = client.query_json(prompt, temperature=0.5)
        except OllamaError:
            raise
        try:
            return validate_decision(
                raw, current_imgsz=context.get("current_imgsz", 1280)
            )
        except ValidationError as e:
            last_error = str(e)
            continue

    raise ValidationError(
        f"researcher: {max_retries} consecutive validation failures; last: {last_error}"
    )


def plateau_analysis(
    client: LLMClient,
    history_text: str,
    *,
    max_tokens: int = 1200,
) -> str:
    """Mirror of ollama_runner.py:744-751 plateau analysis.

    Free-text output suitable for writing into suggestions.md. NOT
    parsed by the orchestrator — the next propose() call reads it as
    context. Returns empty string on LLM failure (so the loop continues).
    """
    prompt = (
        "You are an ML research advisor. The autoresearch loop has stalled "
        "(many consecutive experiments without improvement). Analyze the "
        "history below and suggest 2-3 NEW directions to try next. Be "
        "specific and concrete (mention variable names, value ranges, "
        "datasets to add, etc.). 200-400 words.\n\n"
        f"History:\n{history_text}\n"
    )
    try:
        return client.query(prompt, temperature=0.5, max_tokens=max_tokens)
    except OllamaError as e:
        return f"(plateau analysis failed: {e})"
