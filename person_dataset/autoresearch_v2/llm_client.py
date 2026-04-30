"""Ollama HTTP client for v2 agents.

Mirrors the proven pattern from ollama_runner.py:query_ollama() but adds:
- Per-process call/failure counters so the orchestrator can halt when
  the LLM is dead (>30% failure rate, P1 stop condition from the plan).
- Optional JSON-only post-processing (extract first {...} block) for
  agents that demand structured output.
- Configurable model name + URL (env-overridable for CI / mocking).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_TIMEOUT_SEC = 120


class OllamaError(RuntimeError):
    """Generic failure to talk to Ollama (network, HTTP, decode, etc.)."""


class LLMClient:
    """Stateful client. Tracks call_count and failure_count."""

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SEC,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
        self.url = url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.timeout = timeout
        self.call_count = 0
        self.failure_count = 0

    @property
    def failure_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.failure_count / self.call_count

    def query(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> str:
        """Send a prompt, return response text. Raises OllamaError on failure."""
        self.call_count += 1
        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload_dict["system"] = system

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.HTTPError as e:
            self.failure_count += 1
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            raise OllamaError(
                f"Ollama HTTP {e.code} at {self.url}"
                + (f" body={body!r}" if body else "")
            ) from e
        except urllib.error.URLError as e:
            self.failure_count += 1
            raise OllamaError(f"cannot reach Ollama at {self.url}: {e.reason}") from e
        except Exception as e:
            self.failure_count += 1
            raise OllamaError(f"Ollama query failed: {e}") from e

    def query_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> dict:
        """Like query() but extract the first JSON object from the response.

        Local LLMs often wrap JSON in prose or fence blocks even when told
        not to. We accept the first balanced top-level {...} regardless.
        Raises OllamaError if no parseable JSON found.
        """
        raw = self.query(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )
        return self._extract_first_json(raw)

    @staticmethod
    def _extract_first_json(text: str) -> dict:
        """Pull the first balanced {...} block out of a possibly-noisy string."""
        if not text:
            raise OllamaError("empty LLM response")

        # Quick path — clean JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strip markdown fences if present
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Walk braces to find the first balanced top-level object
        depth = 0
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = -1  # keep looking
        raise OllamaError(f"no parseable JSON object in response: {text[:200]!r}")


def resolve_ollama_model_name(client: LLMClient, wanted: str | None = None) -> str:
    """Best-effort: query /api/tags to confirm the model exists.

    Returns the wanted name if available, the first listed model otherwise,
    or the wanted name unchanged when /api/tags is unreachable (don't fail
    silently — let the actual generate call surface the real error).
    """
    target = wanted or client.model
    tags_url = client.url.rsplit("/", 2)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", [])]
        if target in names:
            return target
        if names:
            return names[0]
        return target
    except Exception:
        return target
