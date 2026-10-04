"""Deterministic parser for unconstrained (no JSON-schema) planner output.

Maps raw text into the same production-step representation as the Full
condition. Does not repair missing/wrong capabilities or parameters.
Does not use an LLM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.adapters.production_plan_adapter import parse_production_plan


def strip_markdown_fence(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_value(text: str) -> Optional[str]:
    """First complete JSON object or array, respecting strings/escapes."""
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch in "{[":
                start = i
                depth = 1
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_unstructured_plan(raw: Any) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Deterministic adapter used when structured generation is off.

    After JSON is recovered, interpretation is identical to the Full parser.
    """
    if not isinstance(raw, str):
        return parse_production_plan(raw)
    text = strip_markdown_fence(raw)
    if not text.strip():
        return None, ["empty raw output"]
    production, errors = parse_production_plan(text)
    if production is not None:
        return production, errors
    extracted = extract_first_json_value(text)
    if extracted is None:
        return None, ["raw output is not deterministically parseable as JSON"]
    return parse_production_plan(extracted)
