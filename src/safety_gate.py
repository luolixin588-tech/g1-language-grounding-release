from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.catalog import LOCOMOTION_DIRECTIONS, PHYSICAL_CAPABILITIES, STATE_VALUES
from src.paths import CONFIG_DIR

SAFETY_GATE_CONFIG_PATH = CONFIG_DIR / "safety_gate.json"


def load_safety_gate_config(path: Path = SAFETY_GATE_CONFIG_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compile_alias_patterns(config: Dict[str, Any]) -> List[Tuple[str, re.Pattern[str]]]:
    flags = re.IGNORECASE if config.get("matching", {}).get("case_insensitive", True) else 0
    use_wb = bool(config.get("matching", {}).get("word_boundary", True))
    patterns: List[Tuple[str, re.Pattern[str]]] = []
    for capability_id, aliases in config.get("capability_aliases", {}).items():
        for alias in aliases:
            body = re.escape(alias)
            if use_wb:
                pat = re.compile(rf"\b{body}\b", flags)
            else:
                pat = re.compile(body, flags)
            patterns.append((capability_id, pat))
    return patterns


def match_registered_physical_capabilities(
    instruction: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    patterns: Optional[Sequence[Tuple[str, re.Pattern[str]]]] = None,
) -> List[str]:
    """Return registered capability ids matched in the user instruction."""
    cfg = config if config is not None else load_safety_gate_config()
    pats = list(patterns) if patterns is not None else _compile_alias_patterns(cfg)
    text = instruction or ""
    matched: List[str] = []
    for capability_id, pat in pats:
        if pat.search(text):
            matched.append(capability_id)
    return sorted(set(matched))


def plan_has_physical_steps(
    normalized_semantic_plan: Optional[List[Dict[str, Any]]],
) -> bool:
    if not normalized_semantic_plan:
        return False
    return any(s.get("capability") in PHYSICAL_CAPABILITIES for s in normalized_semantic_plan)


def _operational_bound_violations(
    normalized_semantic_plan: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[str]:
    """Optional bounds only when values are present in the frozen plan representation."""
    bounds = config.get("operational_bounds") or {}
    violations: List[str] = []
    if not normalized_semantic_plan:
        return violations
    min_dur = bounds.get("locomotion_duration_minimum")
    for i, step in enumerate(normalized_semantic_plan):
        cap = step.get("capability")
        params = step.get("parameters") or {}
        if cap == "locomotion":
            direction = params.get("direction")
            if direction not in LOCOMOTION_DIRECTIONS:
                violations.append(f"step[{i}] locomotion direction out of allowed set")
            duration = params.get("duration")
            if duration is not None and min_dur is not None:
                try:
                    if float(duration) < float(min_dur):
                        violations.append(f"step[{i}] locomotion duration below minimum")
                except (TypeError, ValueError):
                    violations.append(f"step[{i}] locomotion duration not numeric")
        elif cap == "state_control":
            state = params.get("state")
            allowed = set(bounds.get("allowed_states") or STATE_VALUES)
            if state not in allowed:
                violations.append(f"step[{i}] state out of allowed set")
    return violations


def evaluate_safety_gate(
    instruction: str,
    normalized_semantic_plan: Optional[List[Dict[str, Any]]],
    *,
    config: Optional[Dict[str, Any]] = None,
    patterns: Optional[Sequence[Tuple[str, re.Pattern[str]]]] = None,
) -> Dict[str, Any]:
    """
    Execution-time Safety Gate (capability admissibility).

    Inputs allowed: user instruction, frozen planner plan, fixed safety config.
    Must NOT receive task family, gold labels, or evaluator outputs.
    """
    cfg = config if config is not None else load_safety_gate_config()
    has_physical = plan_has_physical_steps(normalized_semantic_plan)

    if not has_physical:
        return {
            "decision": "pass",
            "reason": "no_physical_steps_in_plan",
            "physical_dispatch_allowed": False,
            "matched_capabilities": [],
        }

    matched = match_registered_physical_capabilities(
        instruction, config=cfg, patterns=patterns
    )
    if not matched:
        return {
            "decision": "block",
            "reason": "unregistered_physical_request",
            "physical_dispatch_allowed": False,
            "matched_capabilities": [],
        }

    bound_violations = _operational_bound_violations(normalized_semantic_plan, cfg)
    if bound_violations:
        return {
            "decision": "block",
            "reason": "operational_bound_violation:" + bound_violations[0],
            "physical_dispatch_allowed": False,
            "matched_capabilities": matched,
            "operational_bound_violations": bound_violations,
        }

    return {
        "decision": "pass",
        "reason": "capability_admissible",
        "physical_dispatch_allowed": True,
        "matched_capabilities": matched,
    }


def physical_dispatch_after_gate(
    normalized_semantic_plan: Optional[List[Dict[str, Any]]],
    gate: Dict[str, Any],
) -> bool:
    return bool(gate.get("physical_dispatch_allowed")) and plan_has_physical_steps(
        normalized_semantic_plan
    )
