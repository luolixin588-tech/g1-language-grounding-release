from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.catalog import (
    LOCOMOTION_DIRECTIONS,
    STATE_VALUES,
    arm_action_map,
    capability_by_name,
    load_catalog,
)

_CATALOG = load_catalog()
_CAPS = capability_by_name(_CATALOG)
_ARM_MAP = arm_action_map(_CATALOG)
_ARM_IDS = set(_ARM_MAP.keys())


def _duration_ok(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) >= 0.1


def validate_semantic_plan(steps: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    """Plan Validity: executable under the registered catalog, independent of the user task."""
    errors: List[str] = []
    if steps is None:
        return {"valid": False, "errors": ["plan is not parseable"]}
    if not isinstance(steps, list) or len(steps) == 0:
        return {"valid": False, "errors": ["plan has no steps"]}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step[{i}] is not an object")
            continue
        cap = step.get("capability")
        params = step.get("parameters") if isinstance(step.get("parameters"), dict) else {}
        if cap not in _CAPS:
            errors.append(f"step[{i}] unrecognized capability: {cap}")
            continue
        if cap == "speak":
            content = params.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"step[{i}] speak.content must be a non-empty string")
        elif cap == "arm_action":
            action_id = params.get("action_id")
            action_name = params.get("action_name")
            if action_id not in _ARM_IDS:
                errors.append(f"step[{i}] invalid action_id: {action_id}")
            if not isinstance(action_name, str) or not action_name:
                errors.append(f"step[{i}] action_name missing")
            elif action_id in _ARM_MAP and _ARM_MAP[action_id] != action_name:
                errors.append(
                    f"step[{i}] action_name {action_name!r} does not match action_id {action_id}"
                )
        elif cap == "locomotion":
            direction = params.get("direction")
            duration = params.get("duration")
            if direction not in LOCOMOTION_DIRECTIONS:
                errors.append(f"step[{i}] invalid locomotion direction: {direction}")
            if not _duration_ok(duration):
                errors.append(f"step[{i}] locomotion.duration must be a number >= 0.1")
        elif cap == "state_control":
            state = params.get("state")
            if state not in STATE_VALUES:
                errors.append(f"step[{i}] invalid state: {state}")
        else:
            errors.append(f"step[{i}] capability not evaluable: {cap}")

    return {"valid": len(errors) == 0, "errors": errors}


def validate_production_then_semantic(
    production_steps: List[Dict[str, Any]] | None,
    semantic_steps: List[Dict[str, Any]] | None,
    normalize_errors: List[str],
) -> Dict[str, Any]:
    errors = list(normalize_errors)
    if production_steps is None:
        errors.append("production plan is not parseable")
        return {"valid": False, "errors": errors}
    semantic = validate_semantic_plan(semantic_steps)
    errors.extend(semantic["errors"])
    return {"valid": len(errors) == 0, "errors": errors}
