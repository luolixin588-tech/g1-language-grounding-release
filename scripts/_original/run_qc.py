from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from src.catalog import (
    LOCOMOTION_DIRECTIONS,
    STATE_VALUES,
    arm_action_map,
    capability_by_name,
    load_catalog,
)
from src.io_utils import read_jsonl, write_json
from src.paths import INSTRUCTIONS_PATH, REPORTS_DIR, SEMANTIC_TASKS_PATH


SEMANTIC_GOLD_KEYS = (
    "expected_outcome",
    "required_steps",
    "temporal_constraints",
    "allow_extra_steps",
    "requires_refusal_speech",
)


def _gold(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: row.get(k) for k in SEMANTIC_GOLD_KEYS}


def _instruction_identity_issues(task_id: str, task: Dict[str, Any], text: str) -> List[str]:
    """Paraphrase may change form, but must keep capability-identity cues."""
    issues: List[str] = []
    lowered = text.lower()
    for step in task.get("required_steps") or []:
        cap = step.get("capability")
        params = step.get("parameters") or {}
        if cap == "arm_action" and params.get("action_name") == "chest_wave":
            if "chest" not in lowered:
                issues.append(f"{task_id} chest_wave instruction must mention chest: {text!r}")
        if cap == "arm_action" and params.get("action_name") == "overhead_wave":
            if "overhead" not in lowered and "above your head" not in lowered and "above the head" not in lowered:
                issues.append(f"{task_id} overhead_wave instruction must mention overhead/above the head: {text!r}")
        if cap == "state_control" and params.get("state") == "start":
            if "start mode" not in lowered:
                issues.append(f"{task_id} start instruction must say 'start mode': {text!r}")
            if "standing" in lowered or "stand up" in lowered:
                issues.append(f"{task_id} start instruction must not overlap stand_up language: {text!r}")
        if cap == "state_control" and params.get("state") == "stand_up":
            if "standing posture" not in lowered:
                issues.append(f"{task_id} stand_up instruction must say 'standing posture': {text!r}")
            if "start mode" in lowered:
                issues.append(f"{task_id} stand_up instruction must not mention start mode: {text!r}")
    return issues


def validate_required_step(step: Dict[str, Any], catalog: Dict[str, Any]) -> List[str]:
    errors = []
    caps = capability_by_name(catalog)
    arm_map = arm_action_map(catalog)
    cap = step.get("capability")
    params = step.get("parameters") or {}
    if cap not in caps:
        errors.append(f"unknown capability {cap}")
        return errors
    if cap == "speak":
        if "content_exact" not in params:
            errors.append("speak missing content_exact")
    elif cap == "arm_action":
        aid = params.get("action_id")
        name = params.get("action_name")
        if aid not in arm_map:
            errors.append(f"action_id {aid} not in catalog")
        elif arm_map[aid] != name:
            errors.append(f"action_name {name} != catalog {arm_map.get(aid)}")
    elif cap == "locomotion":
        if params.get("direction") not in LOCOMOTION_DIRECTIONS:
            errors.append(f"bad locomotion direction {params.get('direction')}")
        dur = params.get("duration")
        if not isinstance(dur, (int, float)) or isinstance(dur, bool) or dur < 0.1:
            errors.append(f"bad duration {dur}")
    elif cap == "state_control":
        if params.get("state") not in STATE_VALUES:
            errors.append(f"bad state {params.get('state')}")
    return errors


def run_qc() -> Dict[str, Any]:
    catalog = load_catalog()
    tasks = read_jsonl(SEMANTIC_TASKS_PATH)
    inst = read_jsonl(INSTRUCTIONS_PATH)
    issues: List[str] = []

    n_tasks = len(tasks)
    task_ids = [t["task_id"] for t in tasks]
    if n_tasks != 60:
        issues.append(f"expected 60 semantic tasks, got {n_tasks}")
    if len(set(task_ids)) != len(task_ids):
        issues.append("duplicate semantic task_id")

    fam = Counter(t["family"] for t in tasks)
    if sorted(fam) != sorted(
        {
            "P1_spoken_response",
            "P2_single_physical_action",
            "P3_parameterized_control",
            "P4_multistep_physical_sequence",
            "P5_mixed_speech_action_sequence",
            "P6_unsupported_or_constrained",
        }
    ):
        issues.append(f"unexpected families: {dict(fam)}")
    for k, n in fam.items():
        if n != 10:
            issues.append(f"family {k} has {n} tasks, expected 10")

    by_id = {t["task_id"]: t for t in tasks}
    for t in tasks:
        if t["canonical_instruction"] == t["paraphrase_instruction"]:
            issues.append(f"{t['task_id']} canonical equals paraphrase")
        for step in t.get("required_steps") or []:
            for err in validate_required_step(step, catalog):
                issues.append(f"{t['task_id']} {step.get('id')}: {err}")
        ids = [s["id"] for s in t.get("required_steps") or []]
        if len(ids) != len(set(ids)):
            issues.append(f"{t['task_id']} duplicate required step ids")
        for pair in t.get("temporal_constraints") or []:
            if len(pair) != 2 or pair[0] not in ids or pair[1] not in ids:
                issues.append(f"{t['task_id']} bad temporal constraint {pair}")
        if t["family"].startswith("P6"):
            if t.get("expected_outcome") != "non_execution":
                issues.append(f"{t['task_id']} P6 must be non_execution")
            if t.get("required_steps"):
                issues.append(f"{t['task_id']} P6 should have empty required_steps")
            if t.get("requires_refusal_speech"):
                issues.append(f"{t['task_id']} P6 v1.1 must not require refusal-speech keyword matching")
        else:
            if t.get("expected_outcome") != "execute":
                issues.append(f"{t['task_id']} non-P6 should execute")
            if t.get("requires_refusal_speech"):
                issues.append(f"{t['task_id']} execute task should not require refusal")
            if t.get("allow_extra_steps"):
                issues.append(f"{t['task_id']} v1.1 forbids extra physical steps")

        for text in (t.get("canonical_instruction", ""), t.get("paraphrase_instruction", "")):
            issues.extend(_instruction_identity_issues(t["task_id"], t, text))

    n_inst = len(inst)
    if n_inst != 120:
        issues.append(f"expected 120 instructions, got {n_inst}")
    sample_ids = [r["sample_id"] for r in inst]
    if len(set(sample_ids)) != len(sample_ids):
        issues.append("duplicate sample_id")

    pairs = Counter((r["semantic_task_id"], r["formulation"]) for r in inst)
    for tid, t in by_id.items():
        if pairs.get((tid, "canonical"), 0) != 1:
            issues.append(f"{tid} missing canonical instruction")
        if pairs.get((tid, "paraphrase"), 0) != 1:
            issues.append(f"{tid} missing paraphrase instruction")

    gold_mismatch = 0
    instruction_mismatch = 0
    for r in inst:
        t = by_id.get(r["semantic_task_id"])
        if t is None:
            issues.append(f"instruction {r.get('sample_id')} unknown task {r.get('semantic_task_id')}")
            continue
        if _gold(r) != _gold(t):
            gold_mismatch += 1
            issues.append(f"{r['sample_id']} gold semantics differ from semantic task")
        expected_text = (
            t["canonical_instruction"] if r["formulation"] == "canonical" else t["paraphrase_instruction"]
        )
        if r.get("instruction") != expected_text:
            instruction_mismatch += 1
            issues.append(f"{r['sample_id']} instruction text differs from semantic task")

    neuro = []
    blob = str(catalog)
    for banned in ("ask_lab_assistant", "normalize_neuro_query"):
        if banned in blob:
            neuro.append(banned)

    report = {
        "n_semantic_tasks": n_tasks,
        "n_unique_task_ids": len(set(task_ids)),
        "n_instructions": n_inst,
        "n_unique_sample_ids": len(set(sample_ids)),
        "family_counts": dict(fam),
        "gold_mismatch_pairs": gold_mismatch,
        "instruction_text_mismatch": instruction_mismatch,
        "banned_tools_in_catalog": neuro,
        "n_issues": len(issues),
        "issues": issues,
        "ok": len(issues) == 0 and not neuro,
    }
    return report


def write_qc_markdown(report: Dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "data_qc.md"
    lines = [
        "# Data QC",
        "",
        f"- Semantic tasks: {report['n_semantic_tasks']} (unique IDs: {report['n_unique_task_ids']})",
        f"- Instructions: {report['n_instructions']} (unique sample IDs: {report['n_unique_sample_ids']})",
        f"- Family counts: `{report['family_counts']}`",
        f"- Canonical/paraphrase gold mismatches: {report['gold_mismatch_pairs']}",
        f"- Instruction text mismatches: {report['instruction_text_mismatch']}",
        f"- Banned tools in catalog: {report['banned_tools_in_catalog'] or 'none'}",
        f"- Status: {'PASS' if report['ok'] else 'FAIL'}",
        "",
    ]
    if report["issues"]:
        lines.append("## Issues")
        lines.append("")
        for item in report["issues"]:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("No issues found. Frozen v1.1 60/120 dataset is internally consistent with the capability catalog.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORTS_DIR / "data_qc.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run_qc()
    write_qc_markdown(report)
    print("ok" if report["ok"] else "FAIL", "n_issues=", report["n_issues"])
    for item in report["issues"][:20]:
        print(" -", item)


if __name__ == "__main__":
    main()
