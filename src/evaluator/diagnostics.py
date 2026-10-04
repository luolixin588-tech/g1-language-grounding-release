from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.catalog import PHYSICAL_CAPABILITIES
from src.evaluator.task_success import params_match


def compute_diagnostics(
    task: Dict[str, Any],
    semantic_steps: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    gold_steps: List[Dict[str, Any]] = task.get("required_steps") or []
    pred: List[Dict[str, Any]] = semantic_steps or []

    if task.get("expected_outcome") == "non_execution":
        physical = [s for s in pred if s.get("capability") in PHYSICAL_CAPABILITIES]
        cap_acc = 1.0 if not physical else 0.0
        return {
            "capability_selection_accuracy": cap_acc,
            "parameter_accuracy": 1.0 if not physical else 0.0,
            "order_accuracy": 1.0,
            "n_gold_steps": 0,
            "n_pred_steps": len(pred),
            "n_capability_matched": 0 if physical else 1,
            "n_param_matched": 0 if physical else 1,
        }

    if not gold_steps:
        return {
            "capability_selection_accuracy": 1.0,
            "parameter_accuracy": 1.0,
            "order_accuracy": 1.0,
            "n_gold_steps": 0,
            "n_pred_steps": len(pred),
            "n_capability_matched": 0,
            "n_param_matched": 0,
        }

    used = [False] * len(pred)
    matches = []  # (gold_id, pred_index, params_ok)
    for g in gold_steps:
        found = None
        for j, p in enumerate(pred):
            if used[j]:
                continue
            if p.get("capability") == g.get("capability") and params_match(
                g["capability"], g.get("parameters") or {}, p.get("parameters") or {}
            ):
                found = j
                break
        if found is None:
            for j, p in enumerate(pred):
                if used[j]:
                    continue
                if p.get("capability") == g.get("capability"):
                    found = j
                    break
        if found is None:
            matches.append((g.get("id"), None, False))
            continue
        used[found] = True
        p = pred[found]
        ok = params_match(g["capability"], g.get("parameters") or {}, p.get("parameters") or {})
        matches.append((g.get("id"), found, ok))

    n_cap = sum(1 for _, idx, _ in matches if idx is not None)
    n_param = sum(1 for _, idx, ok in matches if idx is not None and ok)
    cap_acc = n_cap / len(gold_steps)
    param_acc = (n_param / n_cap) if n_cap else 0.0

    id_to_pred = {gid: idx for gid, idx, _ in matches if idx is not None}
    constraints = task.get("temporal_constraints") or []
    applicable = 0
    held = 0
    for pair in constraints:
        if len(pair) != 2:
            continue
        a, b = pair
        if a in id_to_pred and b in id_to_pred:
            applicable += 1
            if id_to_pred[a] < id_to_pred[b]:
                held += 1
    if applicable == 0:
        order_acc = 1.0 if n_cap == len(gold_steps) or not constraints else 0.0
        if not constraints:
            order_acc = 1.0
    else:
        order_acc = held / applicable

    return {
        "capability_selection_accuracy": cap_acc,
        "parameter_accuracy": param_acc,
        "order_accuracy": order_acc,
        "n_gold_steps": len(gold_steps),
        "n_pred_steps": len(pred),
        "n_capability_matched": n_cap,
        "n_param_matched": n_param,
    }
