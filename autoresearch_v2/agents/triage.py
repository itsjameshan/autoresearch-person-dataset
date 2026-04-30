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

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "triage.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "triage_diagnosis.json")


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
                    api_key: str = None) -> Optional[str]:
    try:
        import anthropic
    except ImportError:
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _call_ollama(system_prompt: str, user_prompt: str,
                 model: str = "gemma3:4b",
                 url: str = "http://localhost:11434") -> Optional[str]:
    try:
        import httpx
    except ImportError:
        return None

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
            return response.json().get("response", "")
    except Exception as e:
        print(f"triage: Ollama call failed: {e}", file=sys.stderr)
    return None


def _extract_json(text: str) -> Optional[dict]:
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    return None


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


def _quick_rule_based_diagnosis(error_info: dict, log_tail: str = "") -> Optional[dict]:
    if not log_tail:
        return None

    log_lower = log_tail.lower()

    if "out of memory" in log_lower or "cuda" in log_lower and "memory" in log_lower:
        return {
            "root_cause_hypothesis": "CUDA OOM: 显存不足",
            "confidence": 0.9,
            "recommended_action": "retry_with_fix",
            "fix_diff": {"BATCH": 8, "IMGSZ": 640},
        }

    if "nan" in log_lower and "loss" in log_lower:
        return {
            "root_cause_hypothesis": "NaN Loss: 学习率过大或数据问题",
            "confidence": 0.85,
            "recommended_action": "retry_with_fix",
            "fix_diff": {"LR0": 0.001, "AMP": False},
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
    def __init__(self, state=None, llm_backend: str = "auto",
                 anthropic_model: str = "claude-haiku-4-20250414",
                 ollama_model: str = "gemma3:4b",
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3):
        self.state = state
        self.llm_backend = llm_backend
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.system_prompt = _load_prompt()

    def diagnose(self, error_info: dict, log_tail: str = "",
                 config_diff: dict = None) -> dict:
        quick = _quick_rule_based_diagnosis(error_info, log_tail)
        if quick and quick["confidence"] >= 0.8:
            if self.state:
                self.state.log_decision(
                    run_id=error_info.get("run_id", ""),
                    made_by_agent="triage",
                    action_type="quick_diagnosis",
                    rationale=quick["root_cause_hypothesis"],
                )
            return quick

        context = _build_context(error_info, log_tail, config_diff)
        user_prompt = (
            f"{context}\n\n"
            "请诊断故障根因并推荐修复方案。严格输出 JSON。"
        )

        for attempt in range(self.max_retries):
            response = None

            if self.llm_backend in ("anthropic", "auto"):
                response = _call_anthropic(
                    self.system_prompt, user_prompt, self.anthropic_model
                )

            if response is None and self.llm_backend in ("ollama", "auto"):
                response = _call_ollama(
                    self.system_prompt, user_prompt,
                    self.ollama_model, self.ollama_url
                )

            if response is None:
                print(f"triage: LLM call failed (attempt {attempt+1})",
                      file=sys.stderr)
                continue

            diagnosis = _extract_json(response)
            if diagnosis is None:
                print(f"triage: JSON parse failed (attempt {attempt+1})",
                      file=sys.stderr)
                continue

            valid, errors = _validate_diagnosis(diagnosis)
            if valid:
                if diagnosis.get("confidence", 0) < 0.7:
                    diagnosis["recommended_action"] = "escalate_human"
                    diagnosis["rationale"] = "confidence < 0.7, auto-escalated"

                if self.state:
                    self.state.log_decision(
                        run_id=error_info.get("run_id", ""),
                        made_by_agent="triage",
                        action_type=diagnosis.get("recommended_action", "unknown"),
                        rationale=diagnosis.get("root_cause_hypothesis", ""),
                    )
                return diagnosis

            print(f"triage: validation errors: {errors} (attempt {attempt+1})",
                  file=sys.stderr)

        return {
            "root_cause_hypothesis": "Triage LLM 连续失败，无法诊断",
            "confidence": 0.0,
            "recommended_action": "escalate_human",
        }
