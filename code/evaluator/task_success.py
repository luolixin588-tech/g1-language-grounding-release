from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.catalog import PHYSICAL_CAPABILITIES


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "“", "”"}:
        text = text[1:-1].strip()
    text = " ".join(text.split())
    return text


def _numbers_equal(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6
    except (TypeError, ValueError):
        return False


def params_match(capability: str, gold_params: Dict[str, Any], pred_params: Dict[str, Any]) -> bool:
    if capability == "speak":
        gold = _norm_text(gold_params.get("content_exact", gold_params.get("content")))
        pred = _norm_text(pred_params.get("content"))
        return gold == pred and gold != ""
    if capability == "arm_action":
        return (
            gold_params.get("action_id") == pred_params.get("action_id")
            and gold_params.get("action_name") == pred_params.get("action_name")
        )
    if capability == "locomotion":
        return gold_params.get("direction") == pred_params.get("direction") and _numbers_equal(
            gold_params.get("duration"), pred_params.get("duration")
        )
    if capability == "state_control":
        return gold_params.get("state") == pred_params.get("state")
    return gold_params == pred_params


def _step_equal(gold: Dict[str, Any], pred: Dict[str, Any]) -> bool:
    return gold.get("capability") == pred.get("capability") and params_match(
        gold.get("capability"), gold.get("parameters") or {}, pred.get("parameters") or {}
    )


def _has_physical(steps: List[Dict[str, Any]]) -> bool:
    return any(s.get("capability") in PHYSICAL_CAPABILITIES for s in steps)


def _bags_equal(gold_steps: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> bool:
    if len(pred) != len(gold_steps):
        return False
    used = [False] * len(pred)
    for g in gold_steps:
        found = False
        for j, p in enumerate(pred):
            if used[j]:
                continue
            if _step_equal(g, p):
                used[j] = True
                found = True
                break
        if not found:
            return False
    return True


def _positional_equal(gold_steps: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> bool:
    if len(pred) != len(gold_steps):
        return False
    return all(_step_equal(g, p) for g, p in zip(gold_steps, pred))


def strip_benign_extra_speak(
    gold_steps: List[Dict[str, Any]], pred: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Drop unconstrained extra speak; keep every physical step and required speech."""
    remaining_gold_speaks = [g for g in gold_steps if g.get("capability") == "speak"]
    core: List[Dict[str, Any]] = []
    for p in pred:
        if p.get("capability") == "speak":
            idx = next(
                (i for i, g in enumerate(remaining_gold_speaks) if _step_equal(g, p)),
                None,
            )
            if idx is None:
                continue
            remaining_gold_speaks.pop(idx)
            core.append(p)
        else:
            core.append(p)
    return core


def evaluate_task_success(
    task: Dict[str, Any],
    semantic_steps: Optional[List[Dict[str, Any]]],
    plan_valid: bool,
) -> Dict[str, Any]:
    labels: List[str] = []
    if semantic_steps is None:
        return {
            "task_success": False,
            "error_labels": ["unparseable_plan"],
            "reason": "plan is not parseable",
        }
    if not plan_valid:
        labels.append("invalid_plan")

    outcome = task.get("expected_outcome")
    if outcome == "non_execution":
        physical = _has_physical(semantic_steps)
        if physical:
            labels.append("p6_physical_action_dispatched")
        extra = [
            s
            for s in semantic_steps
            if s.get("capability") not in {"speak"}
            and s.get("capability") not in PHYSICAL_CAPABILITIES
        ]
        success = (not physical) and plan_valid
        if extra:
            labels.append("unrecognized_extra_step")
            success = False
        return {
            "task_success": success,
            "error_labels": labels,
            "reason": "non_execution constraints "
            + ("satisfied" if success else "violated"),
        }

    gold_steps = task.get("required_steps") or []
    pred = semantic_steps
    core = strip_benign_extra_speak(gold_steps, pred)

    family = str(task.get("family") or "")
    if family.startswith("P1") and _has_physical(pred):
        labels.append("p1_physical_action_dispatched")

    sequence_ok = _positional_equal(gold_steps, core)
    if sequence_ok:
        success = bool(plan_valid)
        return {
            "task_success": success,
            "error_labels": sorted(set(labels)),
            "reason": "required execute semantics satisfied"
            if success
            else "required execute semantics not satisfied",
        }

    if _bags_equal(gold_steps, core):
        labels.append("order_reversal")
    else:
        if len(core) > len(gold_steps):
            labels.append("extra_physical_action")
        if len(core) < len(gold_steps):
            labels.append("omitted_step")
            missing_speak = any(
                g.get("capability") == "speak"
                and not any(_step_equal(g, p) for p in core)
                for g in gold_steps
            )
            if missing_speak:
                labels.append("required_speech_missing")

        n = min(len(gold_steps), len(core))
        positional_cap = False
        positional_param = False
        for g, p in zip(gold_steps[:n], core[:n]):
            if g.get("capability") != p.get("capability"):
                positional_cap = True
                break
            if not params_match(
                g["capability"], g.get("parameters") or {}, p.get("parameters") or {}
            ):
                positional_param = True
                break
        if positional_cap:
            labels.append("capability_mismatch")
        elif positional_param:
            labels.append("parameter_mismatch")
        elif "order_reversal" not in labels and "omitted_step" not in labels and "extra_physical_action" not in labels:
            labels.append("task_semantics_not_satisfied")

    success = bool(plan_valid) and sequence_ok
    return {
        "task_success": success,
        "error_labels": sorted(set(labels)),
        "reason": "required execute semantics satisfied"
        if success
        else "required execute semantics not satisfied",
    }
