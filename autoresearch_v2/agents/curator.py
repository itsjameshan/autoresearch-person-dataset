"""
curator.py — Data Curator Agent (数据策展)

职责:
  - 分析验证集上的 FP/FN 图片裁剪
  - 发现数据质量问题和模型薄弱环节
  - 输出结构化报告写入 data_issues 表

触发: 每次 evaluate.py 跑完且产生 FP/FN 分析后
模型: Claude Sonnet (多模态版本，能看图) 或 Ollama 兜底
"""

import base64
import json
import os
import sys
import traceback
import glob
from pathlib import Path
from typing import Optional

from autoresearch_v2._json_extract import extract_first_json

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "prompts", "curator.md")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "schemas", "curator_report.json")
FP_FN_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "state", "fp_fn_crops"
))


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


def _encode_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _collect_fp_fn_images(fp_fn_dir: str = FP_FN_DIR,
                          max_images: int = 20) -> list[dict]:
    images = []
    fp_files = sorted(glob.glob(os.path.join(fp_fn_dir, "fp_*.jpg")))
    fn_files = sorted(glob.glob(os.path.join(fp_fn_dir, "fn_*.jpg")))

    for f in fp_files[:max_images // 2]:
        images.append({"path": f, "type": "FP", "name": os.path.basename(f)})
    for f in fn_files[:max_images // 2]:
        images.append({"path": f, "type": "FN", "name": os.path.basename(f)})

    return images


def _build_context(fp_fn_data: dict, dataset_stats: dict = None) -> str:
    lines = []

    fp_count = len(fp_fn_data.get("fp_images", []))
    fn_count = len(fp_fn_data.get("fn_images", []))
    lines.append(f"## FP/FN 概览")
    lines.append(f"- False Positives: {fp_count}")
    lines.append(f"- False Negatives: {fn_count}")

    if fp_fn_data.get("fp_images"):
        lines.append("\n### FP 详情 (top-10)")
        for rec in fp_fn_data["fp_images"][:10]:
            lines.append(f"- {rec.get('image', '?')}: "
                          f"pred_count={rec.get('pred_count', '?')} "
                          f"gt_count={rec.get('gt_count', '?')}")

    if fp_fn_data.get("fn_images"):
        lines.append("\n### FN 详情 (top-10)")
        for rec in fp_fn_data["fn_images"][:10]:
            lines.append(f"- {rec.get('image', '?')}: "
                          f"gt_area={rec.get('gt_area', 0):.6f} "
                          f"gt_count={rec.get('gt_count', '?')} "
                          f"pred_count={rec.get('pred_count', '?')}")

    if dataset_stats:
        lines.append("\n### 数据集分布统计")
        for k, v in dataset_stats.items():
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def _call_anthropic_vision(system_prompt: str, text_prompt: str,
                           images: list[dict],
                           model: str = "claude-sonnet-4-20250514",
                           api_key: str = None,
                           timeout_sec: float = 120.0) -> tuple[Optional[str], Optional[str]]:
    """Returns (response_text, error_message). Either is None when the other
    is set. Errors carry detail (auth / rate-limit / network) instead of
    being swallowed silently."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed"

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "ANTHROPIC_API_KEY not set"

    content = [{"type": "text", "text": text_prompt}]

    for img_info in images:
        b64 = _encode_image_base64(img_info["path"])
        if b64:
            ext = Path(img_info["path"]).suffix.lstrip(".")
            media_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            })
            content.append({
                "type": "text",
                "text": f"[这是 {img_info['type']} 样本: {img_info['name']}]",
            })

    try:
        client = anthropic.Anthropic(api_key=key, timeout=timeout_sec)
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
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
            return response.json().get("response", ""), None
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _validate_report(report: dict) -> tuple[bool, list[str]]:
    """Strict validation against curator_report.json schema.

    Checks top-level required, item-level required fields in
    worst_failure_modes and suspected_label_errors arrays, and types of
    underrepresented_scenarios / dedup_candidates / recommended_next_collection.
    """
    schema = _load_schema()
    errors: list[str] = []

    if not isinstance(report, dict):
        return False, [f"top-level not object: {type(report).__name__}"]

    required = schema.get("required", [])
    for field in required:
        if field not in report:
            errors.append(f"missing required field: {field}")

    wfm = report.get("worst_failure_modes")
    if wfm is not None:
        if not isinstance(wfm, list):
            errors.append("worst_failure_modes must be array")
        else:
            for i, item in enumerate(wfm):
                if not isinstance(item, dict):
                    errors.append(f"worst_failure_modes[{i}] not object")
                    continue
                if "pattern" not in item:
                    errors.append(f"worst_failure_modes[{i}].pattern missing")
                elif not isinstance(item["pattern"], str):
                    errors.append(f"worst_failure_modes[{i}].pattern not string")
                if "estimated_impact" not in item:
                    errors.append(f"worst_failure_modes[{i}].estimated_impact missing")
                elif not isinstance(item["estimated_impact"], str):
                    errors.append(f"worst_failure_modes[{i}].estimated_impact not string")
                if "evidence_image_ids" in item and not isinstance(item["evidence_image_ids"], list):
                    errors.append(f"worst_failure_modes[{i}].evidence_image_ids not array")

    sle = report.get("suspected_label_errors")
    if sle is not None:
        if not isinstance(sle, list):
            errors.append("suspected_label_errors must be array")
        else:
            for i, item in enumerate(sle):
                if not isinstance(item, dict):
                    errors.append(f"suspected_label_errors[{i}] not object")
                    continue
                for fld in ("image_path", "reason"):
                    if fld not in item:
                        errors.append(f"suspected_label_errors[{i}].{fld} missing")
                    elif not isinstance(item[fld], str):
                        errors.append(f"suspected_label_errors[{i}].{fld} not string")

    us = report.get("underrepresented_scenarios")
    if us is not None and not isinstance(us, list):
        errors.append("underrepresented_scenarios must be array")
    elif isinstance(us, list):
        for i, item in enumerate(us):
            if not isinstance(item, str):
                errors.append(f"underrepresented_scenarios[{i}] not string")

    rnc = report.get("recommended_next_collection")
    if rnc is not None and not isinstance(rnc, str):
        errors.append("recommended_next_collection must be string")

    return len(errors) == 0, errors


class CuratorAgent:
    def __init__(self, state=None, events_logger=None, llm_backend: str = "auto",
                 anthropic_model: str = "claude-sonnet-4-20250514",
                 ollama_model: str = "gemma3:4b",
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 3,
                 max_images_per_call: int = 20):
        self.state = state
        self.events = events_logger
        self.llm_backend = llm_backend
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.max_images_per_call = max_images_per_call
        self.system_prompt = _load_prompt()

    def _emit(self, event_type: str, **data) -> None:
        if self.events is None:
            return
        try:
            self.events.emit(event_type, **data)
        except Exception:
            pass

    def analyze(self, fp_fn_data: dict, run_id: str = "",
                dataset_stats: dict = None) -> dict:
        context = _build_context(fp_fn_data, dataset_stats)
        images = _collect_fp_fn_images(max_images=self.max_images_per_call)

        self._emit(
            "curator_start",
            run_id=run_id,
            n_fp=len(fp_fn_data.get("fp_images", [])),
            n_fn=len(fp_fn_data.get("fn_images", [])),
            n_images_loaded=len(images),
            backend=self.llm_backend,
        )

        user_prompt = (
            f"{context}\n\n"
            "请基于以上 FP/FN 分析和图片，输出你的策展报告。严格输出 JSON。"
        )

        last_error: str | None = None
        for attempt in range(self.max_retries):
            response: Optional[str] = None
            err: Optional[str] = None

            if self.llm_backend in ("anthropic", "auto") and images:
                response, err = _call_anthropic_vision(
                    self.system_prompt, user_prompt, images,
                    self.anthropic_model
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
                print(f"curator: LLM call failed (attempt {attempt+1}): {last_error}",
                      file=sys.stderr, flush=True)
                continue

            report = extract_first_json(response)
            if report is None:
                last_error = f"JSON parse failed; first 200 chars: {response[:200]!r}"
                print(f"curator: JSON parse failed (attempt {attempt+1})",
                      file=sys.stderr, flush=True)
                continue

            valid, errors = _validate_report(report)
            if valid:
                self._save_issues(report, run_id)
                self._emit(
                    "curator_complete",
                    run_id=run_id,
                    n_failure_modes=len(report.get("worst_failure_modes", [])),
                    n_label_errors=len(report.get("suspected_label_errors", [])),
                    n_underrepresented=len(report.get("underrepresented_scenarios", [])),
                    attempts=attempt + 1,
                )
                return report

            last_error = "; ".join(errors)
            print(f"curator: validation errors: {errors} (attempt {attempt+1})",
                  file=sys.stderr, flush=True)

        # Distinguish "broken curator" from "no issues found" — orchestrator
        # logs / dashboards must be able to tell them apart so the operator
        # doesn't get false reassurance.
        self._emit(
            "curator_failed",
            run_id=run_id,
            attempts=self.max_retries,
            last_error=last_error or "unknown",
        )
        return {
            "_failed": True,
            "_last_error": last_error or "unknown",
            "worst_failure_modes": [],
            "suspected_label_errors": [],
            "underrepresented_scenarios": [],
            "dedup_candidates": [],
            "recommended_next_collection": "",
        }

    def _save_issues(self, report: dict, run_id: str = ""):
        if not self.state:
            return

        for fm in report.get("worst_failure_modes", []):
            self.state.insert_data_issue(
                image_path=",".join(fm.get("evidence_image_ids", [])),
                issue_type="failure_mode",
                severity="high",
                description=f"{fm.get('pattern', '')}: {fm.get('estimated_impact', '')}",
                discovered_in_run=run_id,
            )

        for le in report.get("suspected_label_errors", []):
            self.state.insert_data_issue(
                image_path=le.get("image_path", ""),
                issue_type="label_error",
                severity="medium",
                description=le.get("reason", ""),
                discovered_in_run=run_id,
            )
