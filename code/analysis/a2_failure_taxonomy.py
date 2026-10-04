"""Deterministic A2 failure labels. Does not modify v1.1 evaluation."""
from __future__ import annotations

from typing import Any, Dict, List

PRIMARY_LABELS = (
    "capability identity error",
    "action-ID/action-name mismatch",
    "missing parameter",
    "invalid parameter",
    "wrong parameter",
    "order error",
    "extra physical action",
    "unsupported/prohibited action",
    "other",
)


def _errors(record: Dict[str, Any]) -> List[str]:
    verr = list(record.get("validity_result", {}).get("errors") or [])
    verr.extend(record.get("validation_errors") or [])
    return [str(e) for e in verr]


def classify_a2_record(record: Dict[str, Any]) -> str:
    """Assign exactly one primary failure label."""
    family = str(record.get("family") or "")
    labels = set(record.get("task_success_result", {}).get("error_labels") or [])
    errs = _errors(record)
    blob = " ".join(errs).lower()
    pred = record.get("normalized_semantic_plan")

    if pred is None or "unparseable_plan" in labels:
        return "other"

    if family.startswith("P6") and "p6_physical_action_dispatched" in labels:
        return "unsupported/prohibited action"

    if "does not match" in blob:
        return "action-ID/action-name mismatch"

    if "missing" in blob or "must be a non-empty string" in blob:
        return "missing parameter"

    if (
        "invalid action_id" in blob
        or "invalid locomotion" in blob
        or "invalid state" in blob
        or "duration must" in blob
    ):
        return "invalid parameter"

    if "unrecognized capability" in blob or "unknown tool" in blob or "capability_mismatch" in labels:
        return "capability identity error"

    if "omitted_step" in labels:
        # Required physical/speech capability was not produced (often speak-only).
        return "capability identity error"

    if "extra_physical_action" in labels:
        return "extra physical action"

    if "order_reversal" in labels:
        return "order error"

    if "parameter_mismatch" in labels:
        return "wrong parameter"

    return "other"
