"""Deterministic planner-facing semantic aliases for v2 ablations.

Hidden mappings convert anonymous planner tokens back to the frozen v1.1
execution identities before semantic normalization / evaluation.
Mappings are never included in Ollama request payloads.
"""
from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from src.catalog import (
    _ARM_ACTION_DESCRIPTION,
    _MOVE_ROBOT_LOGIC_DESCRIPTION_FULL,
    arm_action_map,
    build_planner_json_schema,
    build_planner_tool_catalog,
    capability_by_name,
    planner_prompt,
)

# Substrings that must not appear in planner-visible system/catalog/schema text
# under no_capability_semantics (case-insensitive).
CAPABILITY_LEAK_TERMS = (
    "handshake",
    "wave",
    "clap",
    "hug",
    "flying_kiss",
    "flying kiss",
    "heart",
    "arms_horizontal",
    "high_five",
    "high five",
    "kamehameha",
    "double_x",
    "reset_arm_pose",
    "chest_wave",
    "overhead_wave",
    "must_match_action_map",
)

# Readable parameter values that must not appear outside the user message
# under no_parameter_semantics.
PARAMETER_LEAK_TERMS = (
    "forward",
    "backward",
    "left_strafe",
    "right_strafe",
    "turn_left",
    "turn_right",
    "stand_up",
    "damp",
    "start",
)

DURATION_LEAK_PATTERNS = (
    "requires duration",
    "required_for_locomotion",
    "minimum",
    "0.2",
    "radians",
    "distance",
    "compute duration",
    "calculation",
)

# Fixed seeds for deterministic alias permutations (never tuned on outcomes).
CAPABILITY_PERMUTATION_SEEDS = {1: 101, 2: 202, 3: 303}
PARAMETER_PERMUTATION_SEEDS = {1: 1101, 2: 1202, 3: 1303}


def _alias_permutation_id(condition: Dict[str, Any]) -> int:
    pid = int(condition.get("alias_permutation_id", 0))
    if pid not in (0, 1, 2, 3):
        raise ValueError(f"unsupported alias_permutation_id: {pid}")
    return pid


def _permuted_order(items: List[Any], permutation_id: int, seed_map: Dict[int, int]) -> List[Any]:
    ordered = list(items)
    if permutation_id == 0:
        return ordered
    rng = random.Random(seed_map[permutation_id])
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    return shuffled


def build_capability_alias_maps(
    catalog: Dict[str, Any], alias_permutation_id: int = 0
) -> Dict[str, Any]:
    """Map opaque A-codes to action IDs. Permutation 0 = sorted action_id order."""
    if alias_permutation_id not in (0, 1, 2, 3):
        raise ValueError(f"unsupported alias_permutation_id: {alias_permutation_id}")
    id_to_name = arm_action_map(catalog)
    ids = _permuted_order(
        sorted(int(x) for x in id_to_name.keys()),
        alias_permutation_id,
        CAPABILITY_PERMUTATION_SEEDS,
    )
    code_to_id: Dict[str, int] = {}
    id_to_code: Dict[int, str] = {}
    code_to_name: Dict[str, str] = {}
    for i, action_id in enumerate(ids, start=1):
        code = f"A{i:02d}"
        code_to_id[code] = action_id
        id_to_code[action_id] = code
        code_to_name[code] = id_to_name[action_id]
    return {
        "alias_permutation_id": alias_permutation_id,
        "code_to_id": code_to_id,
        "id_to_code": id_to_code,
        "code_to_name": code_to_name,
        "codes": [f"A{i:02d}" for i in range(1, len(ids) + 1)],
    }


def build_parameter_alias_maps(
    catalog: Dict[str, Any], alias_permutation_id: int = 0
) -> Dict[str, Any]:
    """Frozen locomotion+state enum order → D-codes. Permutation 0 = catalog order."""
    if alias_permutation_id not in (0, 1, 2, 3):
        raise ValueError(f"unsupported alias_permutation_id: {alias_permutation_id}")
    caps = capability_by_name(catalog)
    values = (
        list(caps["locomotion"]["parameters"]["direction"]["enum"])
        + list(caps["state_control"]["parameters"]["state"]["enum"])
    )
    permuted = _permuted_order(values, alias_permutation_id, PARAMETER_PERMUTATION_SEEDS)
    code_to_value: Dict[str, str] = {}
    value_to_code: Dict[str, str] = {}
    for i, value in enumerate(permuted, start=1):
        code = f"D{i:02d}"
        code_to_value[code] = value
        value_to_code[value] = code
    return {
        "alias_permutation_id": alias_permutation_id,
        "code_to_value": code_to_value,
        "value_to_code": value_to_code,
        "codes": [f"D{i:02d}" for i in range(1, len(permuted) + 1)],
        "values_in_order": permuted,
    }


def planner_prompt_for_condition(
    anonymize_capability_semantics: bool = False,
    anonymize_parameter_semantics: bool = False,
) -> str:
    if not anonymize_capability_semantics and not anonymize_parameter_semantics:
        return planner_prompt()

    base_head = (
        "You are the task planner for a physical Unitree G1 humanoid. "
        "Output a strict JSON plan and nothing else. "
        "Top-level JSON has one field: steps. "
        "Each step is one of: "
        '(1) {"type":"speak","content":"<exact text to say>"} '
    )

    if anonymize_capability_semantics:
        arm_ex = (
            '(2) {"type":"tool","name":"execute_arm_action",'
            '"args":{"action_code":"<A01-A15>"}} '
        )
    else:
        arm_ex = (
            '(2) {"type":"tool","name":"execute_arm_action",'
            '"args":{"action_id":<int>,"action_name":"<name>"}} '
        )

    if anonymize_parameter_semantics:
        move_ex = (
            '(3) {"type":"tool","name":"move_robot_logic",'
            '"args":{"direction":"<D01-D09>","duration":<number>}} '
            "when a duration argument is used, or "
            '{"type":"tool","name":"move_robot_logic",'
            '"args":{"direction":"<D01-D09>"}} '
            "when duration is omitted. "
        )
        param_policy = (
            "Use only registered direction codes from the capability catalog. "
            "Do not invent direction codes. "
        )
    else:
        move_ex = (
            '(3) {"type":"tool","name":"move_robot_logic",'
            '"args":{"direction":"<enum>","duration":<number>}} '
            "for locomotion, or "
            '{"type":"tool","name":"move_robot_logic",'
            '"args":{"direction":"<stand_up|damp|start>"}} '
            "for robot state control. "
        )
        param_policy = (
            "stand_up means rise to the standing posture (ready-standing mode). "
            "start means enter start mode (locomotion-ready / motion mode). "
            "Do not confuse them. "
        )

    if anonymize_capability_semantics:
        arm_policy = (
            "Select arm action_code values only from the registered catalog codes. "
            "The catalog does not explain what each code means. "
        )
    else:
        arm_policy = ""

    return (
        base_head
        + arm_ex
        + move_ex
        + "Do not use any other tool names. "
        "If the user asks for a capability the robot does not have, or a dangerous/safety-violating request, "
        "do not dispatch any physical action; output a speak step instead. The exact wording of that speak step is not scored. "
        "Physical-only requests do not require a speak step; a brief acknowledgement speak is allowed. "
        "Do not add extra physical actions. "
        + arm_policy
        + param_policy
        + "Preserve the requested order of required speech and physical steps. "
        "Quoted speech in the user instruction must be reproduced exactly in speak.content. "
        "No markdown, comments, or extra text."
    )


def build_anonymized_tool_catalog(
    catalog: Dict[str, Any],
    *,
    anonymize_capability_semantics: bool,
    anonymize_parameter_semantics: bool,
    cap_maps: Dict[str, Any],
    param_maps: Dict[str, Any],
) -> List[Dict[str, Any]]:
    caps = capability_by_name(catalog)
    arm = caps["arm_action"]

    if anonymize_capability_semantics:
        arm_tool: Dict[str, Any] = {
            "name": "execute_arm_action",
            "description": "Perform a registered predefined arm action selected by opaque action_code.",
            "parameters": {
                "action_code": {
                    "type": "string",
                    "enum": list(cap_maps["codes"]),
                    "required": True,
                }
            },
        }
    else:
        arm_tool = {
            "name": "execute_arm_action",
            "description": _ARM_ACTION_DESCRIPTION,
            "parameters": {
                "action_id": {
                    "type": "integer",
                    "enum": arm["parameters"]["action_id"]["enum"],
                },
                "action_name": {
                    "type": "string",
                    "required": True,
                    "must_match_action_map": arm["action_map"],
                },
            },
        }

    if anonymize_parameter_semantics:
        move_tool: Dict[str, Any] = {
            "name": "move_robot_logic",
            "description": "Locomotion and robot state control.",
            "parameters": {
                "direction": {
                    "type": "string",
                    "enum": list(param_maps["codes"]),
                },
                "duration": {"type": "number"},
            },
        }
    else:
        loco = caps["locomotion"]
        state = caps["state_control"]
        move_tool = {
            "name": "move_robot_logic",
            "description": _MOVE_ROBOT_LOGIC_DESCRIPTION_FULL,
            "parameters": {
                "direction": {
                    "type": "string",
                    "enum": loco["parameters"]["direction"]["enum"]
                    + state["parameters"]["state"]["enum"],
                },
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "required_for_locomotion": True,
                },
            },
        }

    return [arm_tool, move_tool]


def build_anonymized_json_schema(
    catalog: Dict[str, Any],
    *,
    anonymize_capability_semantics: bool,
    anonymize_parameter_semantics: bool,
    cap_maps: Dict[str, Any],
    param_maps: Dict[str, Any],
) -> Dict[str, Any]:
    if not anonymize_capability_semantics and not anonymize_parameter_semantics:
        return build_planner_json_schema(catalog)

    caps = capability_by_name(catalog)
    if anonymize_capability_semantics:
        action_props: Dict[str, Any] = {
            "action_code": {"type": "string", "enum": list(cap_maps["codes"])},
        }
    else:
        action_props = {
            "action_id": {
                "type": "integer",
                "enum": caps["arm_action"]["parameters"]["action_id"]["enum"],
            },
            "action_name": {"type": "string"},
        }

    if anonymize_parameter_semantics:
        direction_enum = list(param_maps["codes"])
    else:
        direction_enum = (
            caps["locomotion"]["parameters"]["direction"]["enum"]
            + caps["state_control"]["parameters"]["state"]["enum"]
        )

    args_properties: Dict[str, Any] = {
        **action_props,
        "direction": {"type": "string", "enum": direction_enum},
        "duration": {"type": "number"},
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["type", "content"],
                            "properties": {
                                "type": {"type": "string", "enum": ["speak"]},
                                "content": {"type": "string", "minLength": 1},
                            },
                        },
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["type", "name", "args"],
                            "properties": {
                                "type": {"type": "string", "enum": ["tool"]},
                                "name": {
                                    "type": "string",
                                    "enum": [
                                        "execute_arm_action",
                                        "move_robot_logic",
                                    ],
                                },
                                "args": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "properties": args_properties,
                                },
                            },
                        },
                    ]
                },
            }
        },
    }


def decode_production_plan(
    production_steps: Optional[List[Dict[str, Any]]],
    *,
    anonymize_capability_semantics: bool,
    anonymize_parameter_semantics: bool,
    cap_maps: Dict[str, Any],
    param_maps: Dict[str, Any],
) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Map planner-visible aliases to frozen production identities."""
    if production_steps is None:
        return None, []
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, step in enumerate(production_steps):
        step = deepcopy(step)
        if step.get("type") != "tool":
            out.append(step)
            continue
        name = step.get("name")
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        args = dict(args)

        if name == "execute_arm_action" and anonymize_capability_semantics:
            code = args.get("action_code")
            if code not in cap_maps["code_to_id"]:
                errors.append(f"step[{i}] unknown or missing action_code: {code!r}")
                out.append(step)
                continue
            action_id = cap_maps["code_to_id"][code]
            step["args"] = {
                "action_id": action_id,
                "action_name": cap_maps["code_to_name"][code],
            }
            out.append(step)
            continue

        if name == "move_robot_logic" and anonymize_parameter_semantics:
            direction = args.get("direction")
            if direction not in param_maps["code_to_value"]:
                errors.append(f"step[{i}] unknown or missing direction code: {direction!r}")
                out.append(step)
                continue
            args["direction"] = param_maps["code_to_value"][direction]
            step["args"] = args
            out.append(step)
            continue

        out.append(step)
    return out, errors


def scrub_feedback_for_aliases(
    feedback: str,
    *,
    anonymize_capability_semantics: bool,
    anonymize_parameter_semantics: bool,
    cap_maps: Dict[str, Any],
    param_maps: Dict[str, Any],
) -> str:
    """Prevent validation-retry feedback from leaking hidden mappings."""
    if not feedback:
        return feedback
    text = feedback
    if anonymize_capability_semantics:
        # Replace original names/ids with codes where possible; else generic.
        for action_id, code in cap_maps["id_to_code"].items():
            text = re.sub(rf"\b{action_id}\b", code, text)
            name = cap_maps["code_to_name"][code]
            text = text.replace(name, code)
        for term in CAPABILITY_LEAK_TERMS:
            if term.lower() in text.lower():
                return (
                    "The previous plan was not executable under the registered "
                    "catalog. Fix action_code / tool structure."
                )
    if anonymize_parameter_semantics:
        for value, code in param_maps["value_to_code"].items():
            text = text.replace(value, code)
        for term in PARAMETER_LEAK_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                return (
                    "The previous plan was not executable under the registered "
                    "catalog. Fix direction codes / duration / tool structure."
                )
    return text


def build_condition_planner_bundle(
    catalog: Dict[str, Any], condition: Dict[str, Any]
) -> Dict[str, Any]:
    """Assemble planner-facing artifacts + hidden maps for a condition config."""
    anon_cap = bool(condition.get("anonymize_capability_semantics", False))
    anon_param = bool(condition.get("anonymize_parameter_semantics", False))
    perm_id = _alias_permutation_id(condition)
    cap_maps = build_capability_alias_maps(catalog, alias_permutation_id=perm_id)
    param_maps = build_parameter_alias_maps(catalog, alias_permutation_id=perm_id)

    if not anon_cap and not anon_param:
        # Preserve exact Full / legacy path constructors.
        return {
            "tool_catalog": build_planner_tool_catalog(
                catalog,
                reduced=bool(condition.get("reduced_capability_info")),
                include_action_descriptions=condition.get(
                    "include_action_descriptions", True
                ),
                include_parameter_definitions=condition.get(
                    "include_parameter_definitions", True
                ),
            ),
            "schema": (
                build_planner_json_schema(catalog)
                if condition.get("use_json_schema")
                else None
            ),
            "prompt": planner_prompt(),
            "anonymize_capability_semantics": False,
            "anonymize_parameter_semantics": False,
            "cap_maps": cap_maps,
            "param_maps": param_maps,
            "alias_meta": None,
        }

    tool_catalog = build_anonymized_tool_catalog(
        catalog,
        anonymize_capability_semantics=anon_cap,
        anonymize_parameter_semantics=anon_param,
        cap_maps=cap_maps,
        param_maps=param_maps,
    )
    schema = None
    if condition.get("use_json_schema"):
        schema = build_anonymized_json_schema(
            catalog,
            anonymize_capability_semantics=anon_cap,
            anonymize_parameter_semantics=anon_param,
            cap_maps=cap_maps,
            param_maps=param_maps,
        )
    return {
        "tool_catalog": tool_catalog,
        "schema": schema,
        "prompt": planner_prompt_for_condition(
            anonymize_capability_semantics=anon_cap,
            anonymize_parameter_semantics=anon_param,
        ),
        "anonymize_capability_semantics": anon_cap,
        "anonymize_parameter_semantics": anon_param,
        "cap_maps": cap_maps,
        "param_maps": param_maps,
        "alias_meta": {
            "alias_permutation_id": perm_id,
            "capability": {
                code: {
                    "action_id": cap_maps["code_to_id"][code],
                    "action_name": cap_maps["code_to_name"][code],
                }
                for code in cap_maps["codes"]
            }
            if anon_cap
            else None,
            "parameter": {
                code: param_maps["code_to_value"][code] for code in param_maps["codes"]
            }
            if anon_param
            else None,
        },
    }


def planner_visible_system_texts(
    prompt: str, tool_catalog: List[Dict[str, Any]], schema: Optional[Dict[str, Any]]
) -> str:
    """Concatenate planner-visible non-user text for leakage scans."""
    import json

    parts = [
        prompt,
        "Registered robot capabilities (name/description/parameters). "
        "Use only these tools and parameter values:\n"
        + json.dumps(tool_catalog, ensure_ascii=False),
    ]
    if schema is not None:
        parts.append(json.dumps(schema, ensure_ascii=False))
    return "\n".join(parts)


def find_leak_terms(text: str, terms: Tuple[str, ...]) -> List[str]:
    found = []
    lower = text.lower()
    for term in terms:
        if term.lower() in lower:
            found.append(term)
    return found
