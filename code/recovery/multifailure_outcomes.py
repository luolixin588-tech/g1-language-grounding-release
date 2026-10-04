"""Outcome analysis for multi-failure recovery episodes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.catalog import PHYSICAL_CAPABILITIES
from src.recovery.evaluate_recovery import evaluate_executed_trace
from src.recovery.failure_injection import is_physical_semantic


def _physical_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in steps or [] if is_physical_semantic(s)]


def _step_key(step: Dict[str, Any]) -> Tuple[str, str]:
    cap = step.get("capability") or ""
    params = step.get("parameters") or {}
    return cap, str(sorted(params.items()))


def detect_repeated_action(
    completed_before_failure: List[Dict[str, Any]],
    executed_after_recovery: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """True if recovery re-executes a physical step already completed before failure."""
    completed_phys = _physical_steps(completed_before_failure)
    if not completed_phys:
        return False, None
    completed_keys = {_step_key(s) for s in completed_phys}
    # Only steps added after the committed prefix count as recovery additions.
    added = executed_after_recovery[len(completed_before_failure) :]
    for step in _physical_steps(added):
        if _step_key(step) in completed_keys:
            return True, "unnecessary_repeat_of_completed_physical_action"
    # Also check full trace for explicit replays (e.g. replan re-emits prefix).
    seen_completed = 0
    for step in executed_after_recovery:
        if is_physical_semantic(step):
            key = _step_key(step)
            if key in completed_keys:
                if seen_completed >= len(completed_phys):
                    return True, "unnecessary_repeat_of_completed_physical_action"
                seen_completed += 1
    return False, None


def detect_incorrect_recovery(
    *,
    condition_id: str,
    task: Dict[str, Any],
    completed_before_failure: List[Dict[str, Any]],
    executed_trace: List[Dict[str, Any]],
    remaining_plan: List[Dict[str, Any]],
    repeated_action: bool,
    replan_valid: bool,
    recoverable: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if repeated_action:
        reasons.append("unnecessary_repeat_of_completed_action")

    if not recoverable:
        new_physical = _physical_steps(executed_trace[len(completed_before_failure) :])
        if new_physical:
            reasons.append("inappropriate_continued_physical_execution_after_unrecoverable")
        remaining_physical = _physical_steps(remaining_plan)
        if remaining_physical:
            reasons.append("unrecoverable_replan_contains_physical_steps")
    elif replan_valid and remaining_plan:
        eval_after = evaluate_executed_trace(task, executed_trace)
        if not eval_after["recovery_success"]:
            for label in eval_after.get("error_labels") or []:
                if label == "wrong_capability":
                    reasons.append("wrong_capability_after_recovery")
                elif label == "parameter_mismatch":
                    reasons.append("wrong_parameter_after_recovery")
                elif label == "order_violation":
                    reasons.append("wrong_order_after_recovery")
                elif label == "extra_physical":
                    reasons.append("extra_physical_after_recovery")
                elif label == "omitted_step":
                    reasons.append("missing_required_step_after_recovery")
    elif not replan_valid and recoverable:
        reasons.append("replan_invalid")

    return bool(reasons), reasons


def evaluate_f7_correct_termination(
    *,
    executed_trace: List[Dict[str, Any]],
    completed_before_failure: List[Dict[str, Any]],
    remaining_plan: List[Dict[str, Any]],
    n_execution_replans: int,
    max_execution_replans: int,
) -> bool:
    """F7: terminate without inappropriate continued physical execution."""
    new_physical = _physical_steps(executed_trace[len(completed_before_failure) :])
    if new_physical:
        return False
    if _physical_steps(remaining_plan):
        return False
    if n_execution_replans > max_execution_replans:
        return False
    return True


def physical_capabilities_in_plan(steps: List[Dict[str, Any]]) -> bool:
    return any(s.get("capability") in PHYSICAL_CAPABILITIES for s in steps or [])
