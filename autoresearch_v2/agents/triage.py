"""
triage.py — Failure Triage Agent (故障定级)

职责:
  - 训练崩溃时诊断根因
  - 指标回退 > 5% 时分析原因
  - Loss NaN 时分析原因
  - 输出修复建议或升级人工

触发: 训练崩溃 OR 指标回退 > 5% OR loss NaN
模型: Claude Haiku (廉价就够) 或 Ollama 兜底
"""

import json
import os
import re
import sys
from typing import Optional

from autoresearch_v2._json_extract import extract_first_json

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "triage.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "triage_diagnosis.json")

# Bounded reduction of LR0 when triage suggests a NaN/divergence fix:
# halve the current value (down to a floor) instead of clobbering it
# with a fixed 0.001, which can ACCIDENTALLY INCREASE the LR.
_NAN_LR_FLOOR = 1e-5
_NAN_LR_HALVE_FACTOR = 0.5


def _load_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _load_schema() -> dict:
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _build_context(error_info: dict, log_tail: str = "",
                   config_diff: dict = None) -> str:
    lines = []

    lines.append("## 错误信息")
    for k, v in error_info.items():
        lines.append(f"- {k}: {v}")

    if log_tail:
        lines.append("\n## 训练日志 (最后200行)")
        lines.append("```")
        lines.append(log_tail[-3000:])
        lines.append("```")

    if config_diff:
        lines.append("\n## 与上次成功配置的差异")
        lines.append(json.dumps(config_diff, indent=2, ensure_ascii=False))

    return "\n".join(lines)


def _call_anthropic(system_prompt: str, user_prompt: str,
                    model: str = "claude-haiku-4-20250414",
                    api_key: str = None,
                    timeout_sec: float = 60.0) -> tuple[Optional[str], Optional[str]]:
    """Returns (response_text, error_message)."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed"

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "ANTHROPIC_API_KEY not set"

    try:
        client = anthropic.Anthropic(api_key=key, timeout=timeout_sec)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _call_ollama(system_prompt: str, user_prompt: str,
                 model: str = "gemma3:4b",
                 url: str = "http://localhost:11434") -> tuple[Optional[str], Optional[str]]:
    """Returns (response_text, error_message)."""
    try:
        import httpx
    except ImportError:
        return None, "httpx not installed"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    try:
        client = httpx.Client(timeout=120.0)
        response = client.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.2,
                "num_predict": 1024,
            },
        )
        if response.status_code == 200:
            return response.json().get("response", ""), None
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _validate_diagnosis(diagnosis: dict) -> tuple[bool, list[str]]:
    schema = _load_schema()
    errors = []

    valid_actions = schema.get("properties", {}).get("recommended_action", {}).get("enum", [])
    if valid_actions and diagnosis.get("recommended_action") not in valid_actions:
        errors.append(f"invalid recommended_action: {diagnosis.get('recommended_action')}")

    confidence = diagnosis.get("confidence", 0)
    if not (0.0 <= confidence <= 1.0):
        errors.append(f"confidence out of range: {confidence}")

    required = schema.get("required", [])
    for field in required:
        if field not in diagnosis:
            errors.append(f"missing required field: {field}")

    return len(errors) == 0, errors


def _quick_rule_based_diagnosis(error_info: dict, log_tail: str = "",
                                 current_config: dict = None) -> Optional[dict]:
    """Rule-based first pass before LLM. Returns None when no rule matches.

    `current_config` is optional but when provided enables relative fixes
    (e.g. halving LR0) instead of clobbering with absolute defaults that
    might be HIGHER than the failing run's value.
    """
    if not log_tail:
        return None

    log_lower = log_tail.lower()
    cfg = current_config or {}

    # OOM — match the actual torch error message strings, not just any
    # mention of "cuda" + "memory" together (which appears in routine
    # log lines like "CUDA initialized, allocating memory pool").
    oom_match = (
        "out of memory" in log_lower
        or "cuda out of memory" in log_lower
        or "torch.cuda.outofmemoryerror" in log_lower
    )
    if oom_match:
        # Halve BATCH (down to floor 1) instead of jumping to a fixed 8 —
        # if BATCH was already 4, jumping to 8 makes things worse.
        # Do NOT change IMGSZ — the project is built around 1280 tile
        # size; reducing imgsz silently changes the entire training
        # regime and confounds Researcher's next experiment.
        cur_batch = int(cfg.get("BATCH", 8))
        new_batch = max(1, cur_batch // 2)
        return {
            "root_cause_hypothesis": "CUDA OOM: 显存不足，建议减小 batch 至一半",
            "confidence": 0.9,
            "recommended_action": "retry_with_fix",
            "fix_diff": {"BATCH": new_batch},
        }

    if "nan" in log_lower and "loss" in log_lower:
        # Halve current LR0 with a floor; only fall back to 0.001 when we
        # truly don't know the current value AND can't infer it.
        cur_lr = cfg.get("LR0")
        try:
            cur_lr = float(cur_lr) if cur_lr is not None else None
        except (TypeError, ValueError):
            cur_lr = None
        if cur_lr is not None and cur_lr > 0:
            new_lr = max(_NAN_LR_FLOOR, cur_lr * _NAN_LR_HALVE_FACTOR)
        else:
            new_lr = 0.001
        return {
            "root_cause_hypothesis": "NaN Loss: 学习率过大或 AMP 问题",
            "confidence": 0.85,
            "recommended_action": "retry_with_fix",
            "fix_diff": {"LR0": new_lr, "AMP": False},
        }

    if "timeout" in log_lower or "timed out" in log_lower:
        return {
            "root_cause_hypothesis": "训练超时: epoch过多或数据量过大",
            "confidence": 0.8,
            "recommended_action": "retry_with_fix",
            "fix_diff": {"EPOCHS": 50, "PATIENCE": 20},
        }

    return None


class TriageAgent:
    def __init__(self, state=None, events_logger=None, llm_backend: str = "auto",
                 anthropic_model: str = "claude-haiku-4-20250414",
                 ollama_model: str = "gemma3:4b",
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3):
        self.state = state
        self.events = events_logger
        self.llm_backend = llm_backend
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.system_prompt = _load_prompt()

    def _emit(self, event_type: str, **data) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(event_type, **data)
        except Exception:
            pass

    def diagnose(self, error_info: dict, log_tail: str = "",
                 config_diff: dict = None) -> dict:
        run_id = error_info.get("run_id", "")
        self._emit(
            "triage_start",
            run_id=run_id,
            error=str(error_info.get("error", ""))[:200],
            log_tail_len=len(log_tail),
        )

        quick = _quick_rule_based_diagnosis(error_info, log_tail, config_diff)
        if quick and quick["confidence"] >= 0.8:
            if self.state:
                self.state.log_decision(
                    run_id=run_id,
                    made_by_agent="triage",
                    action_type="quick_diagnosis",
                    rationale=quick["root_cause_hypothesis"],
                )
            self._emit(
                "triage_quick_match",
                run_id=run_id,
                root_cause=quick["root_cause_hypothesis"],
                action=quick["recommended_action"],
                fix_diff=quick.get("fix_diff", {}),
                confidence=quick["confidence"],
            )
            return quick

        context = _build_context(error_info, log_tail, config_diff)
        user_prompt = (
            f"{context}\n\n"
            "请诊断故障根因并推荐修复方案。严格输出 JSON。"
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            response: Optional[str] = None
            err: Optional[str] = None

            if self.llm_backend in ("anthropic", "auto"):
                response, err = _call_anthropic(
                    self.system_prompt, user_prompt, self.anthropic_model
                )

            if response is None and self.llm_backend in ("ollama", "auto"):
                response, ollama_err = _call_ollama(
                    self.system_prompt, user_prompt,
                    self.ollama_model, self.ollama_url
                )
                if response is None:
                    err = ollama_err if ollama_err else err

            if response is None:
                last_error = err or "LLM unreachable"
                print(f"triage: LLM call failed (attempt {attempt+1}): {last_error}",
                      file=sys.stderr, flush=True)
                continue

            diagnosis = extract_first_json(response)
            if diagnosis is None:
                last_error = f"JSON parse failed; first 200 chars: {response[:200]!r}"
                print(f"triage: JSON parse failed (attempt {attempt+1})",
                      file=sys.stderr, flush=True)
                continue

            valid, errors = _validate_diagnosis(diagnosis)
            if valid:
                # When auto-escalating due to low confidence, PRESERVE
                # the LLM's original recommendation so a human reviewer
                # can see what it actually wanted to do. The orchestrator
                # acts on `recommended_action`, but logs / dashboards
                # surface `recommended_action_original` for debugging.
                conf = diagnosis.get("confidence", 0)
                if conf < 0.7:
                    original_action = diagnosis.get("recommended_action")
                    diagnosis["recommended_action_original"] = original_action
                    diagnosis["recommended_action"] = "escalate_human"
                    diagnosis["auto_escalated_reason"] = (
                        f"confidence {conf} < 0.7"
                    )

                if self.state:
                    self.state.log_decision(
                        run_id=run_id,
                        made_by_agent="triage",
                        action_type=diagnosis.get("recommended_action", "unknown"),
                        rationale=diagnosis.get("root_cause_hypothesis", ""),
                    )
                self._emit(
                    "triage_complete",
                    run_id=run_id,
                    action=diagnosis.get("recommended_action"),
                    action_original=diagnosis.get("recommended_action_original"),
                    confidence=conf,
                    root_cause=diagnosis.get("root_cause_hypothesis", "")[:200],
                    attempts=attempt + 1,
                )
                return diagnosis

            last_error = "; ".join(errors)
            print(f"triage: validation errors: {errors} (attempt {attempt+1})",
                  file=sys.stderr, flush=True)

        self._emit(
            "triage_failed",
            run_id=run_id,
            attempts=self.max_retries,
            last_error=last_error or "unknown",
        )
        return {
            "root_cause_hypothesis": "Triage LLM 连续失败，无法诊断",
            "confidence": 0.0,
            "recommended_action": "escalate_human",
            "_failed": True,
            "_last_error": last_error or "unknown",
        }
