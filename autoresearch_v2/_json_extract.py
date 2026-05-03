"""Robust JSON extraction shared by all v2 agents.

Replaces the per-agent regex `\\{[\\s\\S]*\\}` which is greedy and matches
from the first `{` to the LAST `}` in a possibly-noisy LLM response —
breaks badly when the response contains multiple JSON-ish snippets,
prose between, or unbalanced braces in error messages quoted by the
model.

The implementation here:
  1. Strips surrounding markdown fences if present (```json ... ```).
  2. Tries a direct json.loads on the stripped text.
  3. Walks the text counting `{`/`}` (string-aware so braces inside
     "..." don't move the depth) and returns the first balanced
     top-level object that parses.

Returns None on no match — callers should treat that as a parse failure
distinct from "LLM returned an empty response" (which is OllamaError /
network) so they can attach a useful error message instead of silently
returning a stub report.
"""

from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)


def extract_first_json(text: str) -> dict | None:
    """Return the first balanced JSON object in text, or None."""
    if not text or not text.strip():
        return None

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

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
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Keep scanning — maybe a later {...} parses
                        start = -1
    return None
