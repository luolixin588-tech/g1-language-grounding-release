from __future__ import annotations

import json
from typing import Any, Dict, List

from src.evaluator.task_success import evaluate_task_success


def evaluate_executed_trace(
    task: Dict[str, Any],
    executed_semantic_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recovery success = original v1.1 task semantics satisfied by executed steps."""
    result = evaluate_task_success(task, executed_semantic_steps, plan_valid=True)
    return {
        "recovery_success": bool(result.get("task_success")),
        "error_labels": result.get("error_labels") or [],
        "reason": result.get("reason"),
        "executed_semantic_steps": executed_semantic_steps,
    }
