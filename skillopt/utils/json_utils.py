"""JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import re


def _balanced_json_candidates(text: str, open_char: str, close_char: str) -> list[str]:
    candidates: list[str] = []
    start = text.find(open_char)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
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
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
        start = text.find(open_char, start + 1)
    return candidates


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text.

    Tries ```json fences first, then bare {...} patterns.
    """
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for candidate in _balanced_json_candidates(text, "{", "}"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_json_array(text: str) -> list | None:
    """Extract a JSON array from LLM response text."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for candidate in _balanced_json_candidates(text, "[", "]"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
