"""Deterministic multi-failure execution injection for recovery evaluation.

Software-level only. Does not modify production planner, catalog, or schema.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.recovery.failure_injection import is_physical_semantic

FAILURE_CONDITIONS = (
    "F1_dispatch_rejected",
    "F2_execution_timeout",
    "F3_partial_completion",
    "F4_invalid_robot_state",
    "F5_infeasible_parameter",
    "F6_second_step_failure",
    "F7_unrecoverable_failure",
)

RECOVERABLE_CONDITIONS = FAILURE_CONDITIONS[:6]
UNRECOVERABLE_CONDITIONS = ("F7_unrecoverable_failure",)

_DIAGNOSTIC_TEXT = {
    "F1_dispatch_rejected": "dispatch rejected",
    "F2_execution_timeout": "execution timeout",
    "F3_partial_completion": (
        "partial execution / requested action not fully completed"
    ),
    "F4_invalid_robot_state": "robot state does not permit requested execution",
    "F5_infeasible_parameter": "requested execution parameter is not feasible",
    "F6_second_step_failure": (
        "execution failure on second eligible physical step after first step completed"
    ),
    "F7_unrecoverable_failure": (
        "unrecoverable execution failure; physical recovery is not permitted"
    ),
}


def nth_physical_index(semantic_steps: List[Dict[str, Any]], n: int = 0) -> int:
    """Return semantic-plan index of the n-th physical step (0-based)."""
    if n < 0:
        return -1
    seen = 0
    for i, step in enumerate(semantic_steps or []):
        if is_physical_semantic(step):
            if seen == n:
                return i
            seen += 1
    return -1


def second_physical_index(semantic_steps: List[Dict[str, Any]]) -> int:
    return nth_physical_index(semantic_steps, 1)


def resolve_injection_step_index(
    condition_id: str,
    semantic_plan: List[Dict[str, Any]],
) -> int:
    if condition_id == "F6_second_step_failure":
        return second_physical_index(semantic_plan)
    return nth_physical_index(semantic_plan, 0)


def condition_is_recoverable(condition_id: str) -> bool:
    return condition_id in RECOVERABLE_CONDITIONS


def inject_execution_failure(
    condition_id: str,
    task: Dict[str, Any],
    plan: Dict[str, Any],
    execution_state: Dict[str, Any],
    target_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Common experimental injection interface.

    Parameters
    ----------
    condition_id : one of FAILURE_CONDITIONS
    task : benchmark task row (instruction, gold, family, ...)
    plan : dict with ``production`` and ``semantic`` plan step lists
    execution_state : optional pre-existing completed indices (for F6 chaining)
    target_step : optional override of auto-resolved injection index

    Returns structured injection metadata and planner diagnostic.
    """
    if condition_id not in FAILURE_CONDITIONS:
        raise ValueError(f"unknown failure condition: {condition_id}")

    semantic: List[Dict[str, Any]] = list(plan.get("semantic") or [])
    production: List[Dict[str, Any]] = list(plan.get("production") or [])

    fail_idx = (
        int(target_step)
        if target_step is not None
        else resolve_injection_step_index(condition_id, semantic)
    )
    if fail_idx < 0 or fail_idx >= len(semantic):
        return {
            "injected": False,
            "skip_reason": "no_eligible_injection_step",
            "failure_condition": condition_id,
            "injected_step_index": fail_idx,
            "failure_diagnostic": None,
            "completed_step_indices": [],
            "partial": False,
            "recoverable": condition_is_recoverable(condition_id),
            "diagnostic": None,
            "completed_semantic_steps": [],
            "completed_production_steps": [],
            "failed_semantic_step": None,
            "failed_production_step": None,
        }

    completed_indices: List[int] = list(execution_state.get("completed_step_indices") or [])
    if not completed_indices:
        completed_indices = list(range(fail_idx))

    completed_sem = [semantic[i] for i in completed_indices if i < len(semantic)]
    completed_prod = [production[i] for i in completed_indices if i < len(production)]
    failed_sem = semantic[fail_idx]
    failed_prod = production[fail_idx] if fail_idx < len(production) else failed_sem

    partial = condition_id == "F3_partial_completion"
    recoverable = condition_is_recoverable(condition_id)
    diagnostic_text = _DIAGNOSTIC_TEXT[condition_id]

    diagnostic = {
        "event": "execution_failure",
        "failure_condition": condition_id,
        "recoverable": recoverable,
        "completed": False,
        "partial": partial,
        "failed_step_index": fail_idx,
        "failed_step": failed_prod,
        "completed_steps": completed_prod,
        "completed_step_indices": completed_indices,
        "failure_diagnostic": diagnostic_text,
        "message": diagnostic_text,
        "planner_instruction": (
            "Output a JSON plan for the REMAINING task only, including the failed "
            "step if it is still required. Do not undo completed steps. Do not "
            "dispatch zero-torque or destabilizing behavior."
        ),
    }

    if condition_id == "F7_unrecoverable_failure":
        diagnostic["error_class"] = "unrecoverable_execution_failure"
        diagnostic["planner_instruction"] = (
            "Do not generate any further physical action. "
            "The physical task must terminate. "
            "A spoken explanation is permitted."
        )
    elif condition_id == "F1_dispatch_rejected":
        diagnostic["error_class"] = "transient_dispatch_failure"
    elif condition_id == "F2_execution_timeout":
        diagnostic["error_class"] = "transient_dispatch_failure"
        diagnostic["simulated_timeout"] = True
    elif condition_id == "F3_partial_completion":
        diagnostic["error_class"] = "recoverable_controller_rejection"
    elif condition_id == "F4_invalid_robot_state":
        diagnostic["error_class"] = "temporary_capability_unavailable"
    elif condition_id == "F5_infeasible_parameter":
        diagnostic["error_class"] = "recoverable_controller_rejection"
    elif condition_id == "F6_second_step_failure":
        diagnostic["error_class"] = "recoverable_controller_rejection"

    return {
        "injected": True,
        "skip_reason": None,
        "failure_condition": condition_id,
        "injected_step_index": fail_idx,
        "failure_diagnostic": diagnostic_text,
        "completed_step_indices": completed_indices,
        "partial": partial,
        "recoverable": recoverable,
        "diagnostic": diagnostic,
        "completed_semantic_steps": completed_sem,
        "completed_production_steps": completed_prod,
        "failed_semantic_step": failed_sem,
        "failed_production_step": failed_prod,
        "task_id": task.get("semantic_task_id"),
        "instruction": task.get("instruction"),
    }


def validate_injection_preconditions(
    condition_id: str,
    semantic_plan: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Return (ok, skip_reason) before running an episode."""
    if condition_id == "F6_second_step_failure":
        if second_physical_index(semantic_plan) < 0:
            return False, "f6_requires_second_physical_step"
    elif nth_physical_index(semantic_plan, 0) < 0:
        return False, "no_eligible_physical_action"
    return True, None
