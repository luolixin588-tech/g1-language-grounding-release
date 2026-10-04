from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_production_plan(raw: Any) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Parse runtime/planner JSON into a list of production steps.

    Accepts a raw string, a dict with `steps`, or a list of steps.
    """
    errors: List[str] = []
    obj = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, ["empty raw output"]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            maybe = extract_first_json_object(text)
            if not maybe:
                return None, ["raw output is not JSON"]
            try:
                obj = json.loads(maybe)
            except json.JSONDecodeError as e:
                return None, [f"raw output JSON parse failed: {e}"]

    steps = None
    if isinstance(obj, dict):
        if "steps" in obj:
            steps = obj.get("steps")
        elif obj.get("type") in {"speak", "tool"}:
            steps = [obj]
        else:
            errors.append("JSON object has no steps field")
            return None, errors
    elif isinstance(obj, list):
        steps = obj
    else:
        return None, ["parsed value is not an object or list"]

    if not isinstance(steps, list):
        return None, ["steps is not a list"]
    if len(steps) == 0:
        return None, ["steps is empty"]
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return None, [f"step[{i}] is not an object"]
    return steps, []
