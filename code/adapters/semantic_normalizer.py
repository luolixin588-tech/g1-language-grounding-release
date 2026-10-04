from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.catalog import LOCOMOTION_DIRECTIONS, STATE_VALUES, arm_action_map, load_catalog

_CATALOG = load_catalog()
_ARM_MAP = arm_action_map(_CATALOG)


def _as_int(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _as_float(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def normalize_steps(production_steps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Map production speak/tool steps into paper-facing semantic classes."""
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, step in enumerate(production_steps):
        step_type = step.get("type")
        if step_type == "speak":
            content = step.get("content")
            out.append(
                {
                    "capability": "speak",
                    "parameters": {"content": "" if content is None else str(content)},
                }
            )
            continue
        if step_type == "tool":
            name = step.get("name")
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            if name == "execute_arm_action":
                action_id = _as_int(args.get("action_id"))
                action_name = args.get("action_name")
                if action_name is not None:
                    action_name = str(action_name)
                elif action_id in _ARM_MAP:
                    action_name = _ARM_MAP[action_id]
                out.append(
                    {
                        "capability": "arm_action",
                        "parameters": {
                            "action_id": action_id,
                            "action_name": action_name,
                        },
                    }
                )
                continue
            if name == "move_robot_logic":
                direction = args.get("direction")
                if isinstance(direction, str):
                    direction = direction.strip()
                if direction in STATE_VALUES:
                    out.append(
                        {
                            "capability": "state_control",
                            "parameters": {"state": direction},
                        }
                    )
                    continue
                duration = _as_float(args.get("duration"))
                out.append(
                    {
                        "capability": "locomotion",
                        "parameters": {
                            "direction": direction,
                            "duration": duration,
                        },
                    }
                )
                continue
            errors.append(f"step[{i}] unknown tool name: {name}")
            out.append(
                {
                    "capability": "unknown",
                    "parameters": {"name": name, "args": args},
                    "unrecognized": True,
                }
            )
            continue
        errors.append(f"step[{i}] unknown type: {step_type}")
        out.append(
            {
                "capability": "unknown",
                "parameters": dict(step),
                "unrecognized": True,
            }
        )
    return out, errors
