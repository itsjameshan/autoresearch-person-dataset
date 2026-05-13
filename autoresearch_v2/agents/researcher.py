"""
researcher.py — Researcher Agent (主决策)

职责:
  - 读取实验历史、预算、数据问题
  - 决定下一步策略 (action_type)
  - 输出结构化 JSON 指令给 Orchestrator

模型: Claude Sonnet (通过 Anthropic API) 或 Ollama 本地兜底
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "researcher.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "researcher_decision.json")


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


def _build_context(recent_experiments: list[dict], best_metrics: dict,
                   remaining_budget: dict, top_data_issues: list[dict],
                   recent_failures: list[dict]) -> str:
    lines = []

    lines.append("## 最近实验 (最近10次)")
    for exp in recent_experiments[:10]:
        m = exp.get("metrics_json", {})
        c = exp.get("config_json", {})
        lines.append(f"- run={exp.get('run_id','?')} "
                      f"status={exp.get('status','?')} "
                      f"CDS={m.get('cds',0):.4f} "
                      f"mAP50={m.get('mAP50',0):.4f} "
                      f"precision={m.get('precision',0):.4f} "
                      f"recall={m.get('recall',0):.4f} "
                      f"small_obj_recall={m.get('small_obj_recall',0):.4f} "
                      f"counting_mae={m.get('counting_mae',0):.2f} "
                      f"inference_ms={m.get('inference_ms',0):.1f}")

    lines.append("\n## 当前最佳指标")
    if best_metrics:
        for k, v in best_metrics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
    else:
        lines.append("- 无（首次迭代）")

    lines.append("\n## 剩余预算")
    lines.append(f"- GPU分钟: {remaining_budget.get('gpu_minutes_remaining', 'unlimited')}")
    lines.append(f"- 美元: {remaining_budget.get('dollars_remaining', 'unlimited')}")

    lines.append("\n## Curator 发现的数据问题 (top-3)")
    if top_data_issues:
        for issue in top_data_issues:
            lines.append(f"- [{issue.get('severity','?')}] "
                          f"{issue.get('issue_type','?')}: "
                          f"{issue.get('description','')}")
    else:
        lines.append("- 无")

    lines.append("\n## 最近失败列表 (Triage 提供)")
    if recent_failures:
        for f in recent_failures[:5]:
            lines.append(f"- run={f.get('run_id','?')} "
                          f"reason={f.get('root_cause','?')}")
    else:
        lines.append("- 无")

    return "\n".join(lines)


def _call_anthropic(system_prompt: str, user_prompt: str,
                    model: str = "claude-sonnet-4-20250514",
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
        max_tokens=2048,
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
        client = httpx.Client(timeout=180.0)
        response = client.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.3,
                "num_predict": 2048,
            },
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        print(f"researcher: Ollama call failed: {e}", file=sys.stderr)
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


def _validate_decision(decision: dict) -> tuple[bool, list[str]]:
    schema = _load_schema()
    errors = []

    valid_actions = schema.get("properties", {}).get("action_type", {}).get("enum", [])
    if valid_actions and decision.get("action_type") not in valid_actions:
        errors.append(f"invalid action_type: {decision.get('action_type')}")

    required = schema.get("required", [])
    for field in required:
        if field not in decision:
            errors.append(f"missing required field: {field}")

    return len(errors) == 0, errors


def _normalize_decision(decision: dict) -> dict:
    """
    Normalize model output into the orchestrator-compatible decision shape.
    This keeps the loop running even when the LLM omits optional fields.
    """
    if not isinstance(decision, dict):
        return {}

    normalized = dict(decision)
    action = str(normalized.get("action_type", "")).strip()

    # Backward-compatible alias used by older prompts/examples.
    if action == "patch_train_config":
        normalized["action_type"] = "patch_train_config"

    # Keep schema-required numeric/bool fields safe by default.
    normalized.setdefault("config_diff", {})
    if not isinstance(normalized.get("config_diff"), dict):
        normalized["config_diff"] = {}

    normalized.setdefault("expected_metric_delta", {})
    if not isinstance(normalized.get("expected_metric_delta"), dict):
        normalized["expected_metric_delta"] = {}

    try:
        normalized["estimated_gpu_minutes"] = float(
            normalized.get("estimated_gpu_minutes", 0) or 0
        )
    except (TypeError, ValueError):
        normalized["estimated_gpu_minutes"] = 0.0

    try:
        normalized["estimated_cost_usd"] = float(
            normalized.get("estimated_cost_usd", 0) or 0
        )
    except (TypeError, ValueError):
        normalized["estimated_cost_usd"] = 0.0

    normalized["needs_hitl"] = bool(normalized.get("needs_hitl", False))
    normalized.setdefault("rationale", "")
    return normalized


class ResearcherAgent:
    def __init__(self, llm_backend: str = "auto",
                 anthropic_model: str = "claude-sonnet-4-20250514",
                 ollama_model: str = "gemma3:4b",
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3):
        self.llm_backend = llm_backend
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.system_prompt = _load_prompt()

    def _call_llm(self, user_prompt: str) -> Optional[str]:
        if self.llm_backend in ("anthropic", "auto"):
            result = _call_anthropic(
                self.system_prompt, user_prompt, self.anthropic_model
            )
            if result is not None:
                return result

        if self.llm_backend in ("ollama", "auto"):
            result = _call_ollama(
                self.system_prompt, user_prompt,
                self.ollama_model, self.ollama_url
            )
            if result is not None:
                return result

        return None

    def decide(self, recent_experiments: list[dict],
               best_metrics: dict, remaining_budget: dict,
               top_data_issues: list[dict] = None,
               recent_failures: list[dict] = None) -> dict:
        context = _build_context(
            recent_experiments or [],
            best_metrics or {},
            remaining_budget or {},
            top_data_issues or [],
            recent_failures or [],
        )

        user_prompt = (
            f"{context}\n\n"
            "请基于以上信息，决定下一步策略。严格输出 JSON，不要输出其他内容。"
        )

        for attempt in range(self.max_retries):
            response = self._call_llm(user_prompt)
            if response is None:
                print(f"researcher: LLM call failed (attempt {attempt+1})",
                      file=sys.stderr)
                continue

            decision = _extract_json(response)
            if decision is None:
                print(f"researcher: JSON parse failed (attempt {attempt+1})",
                      file=sys.stderr)
                continue

            decision = _normalize_decision(decision)
            valid, errors = _validate_decision(decision)
            if valid:
                return decision

            print(f"researcher: validation errors: {errors} (attempt {attempt+1})",
                  file=sys.stderr)

        return {
            "action_type": "stop",
            "config_diff": {},
            "rationale": "LLM连续失败，自动停止",
            "expected_metric_delta": {},
            "estimated_gpu_minutes": 0,
            "estimated_cost_usd": 0,
            "needs_hitl": True,
        }
