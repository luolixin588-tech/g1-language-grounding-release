#!/usr/bin/env python3
"""Analyze Full vs Full+Safety Gate offline replay."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.catalog import PHYSICAL_CAPABILITIES
from src.io_utils import read_jsonl
from src.paths import REPORTS_DIR, RUNS_DIR

FULL_RUN = RUNS_DIR / "full" / "full_20260821T051022Z_d81f78bf"
FAMILIES = {
    "P1_spoken_response": "P1",
    "P2_single_physical_action": "P2",
    "P3_parameterized_control": "P3",
    "P4_multistep_physical_sequence": "P4",
    "P5_mixed_speech_action_sequence": "P5",
    "P6_unsupported_or_constrained": "P6",
}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _family_rates(records: List[Dict[str, Any]], key: str = "gated_task_success_result") -> Dict[str, float]:
    by: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        by[r["family"]].append(bool(r[key]["task_success"]))
    return {fam: sum(v) / len(v) for fam, v in by.items()}


def _has_physical(plan) -> bool:
    return bool(plan) and any(s.get("capability") in PHYSICAL_CAPABILITIES for s in plan)


def analyze(gate_off: Path, gate_on: Path) -> Dict[str, Any]:
    off = read_jsonl(gate_off / "records.jsonl")
    on = read_jsonl(gate_on / "records.jsonl")
    off_sum = json.loads((gate_off / "summary.json").read_text(encoding="utf-8"))
    on_sum = json.loads((gate_on / "summary.json").read_text(encoding="utf-8"))

    blocked = [r for r in on if r["safety_gate"]["decision"] == "block"]
    false_blocks = [
        r
        for r in blocked
        if not r["family"].startswith("P6_")
    ]
    p6_on = [r for r in on if r["family"].startswith("P6_")]
    p6_off = [r for r in off if r["family"].startswith("P6_")]

    prevented_dispatches = 0
    for r in on:
        orig = r.get("normalized_semantic_plan")
        gated = r.get("gated_semantic_plan")
        if _has_physical(orig) and not _has_physical(gated):
            prevented_dispatches += 1

    p6_err_before = sum(
        1
        for r in p6_off
        if "p6_physical_action_dispatched" in (r["task_success_result"].get("error_labels") or [])
    )
    p6_err_after = sum(
        1
        for r in p6_on
        if "p6_physical_action_dispatched" in (r["gated_task_success_result"].get("error_labels") or [])
    )

    block_reasons = Counter(r["safety_gate"]["reason"] for r in blocked)

    return {
        "source_run": str(FULL_RUN),
        "gate_off_run": str(gate_off),
        "gate_on_run": str(gate_on),
        "headline": {
            "full_gate_off": {
                "task_success_rate": off_sum["task_success_rate"],
                "n_task_success": off_sum["n_task_success"],
                "n_runs": off_sum["n_runs"],
                "plan_validity_rate": off_sum["plan_validity_rate"],
            },
            "full_safety_gate": {
                "task_success_rate": on_sum["task_success_rate"],
                "n_task_success": on_sum["n_task_success"],
                "n_runs": on_sum["n_runs"],
                "plan_validity_rate": on_sum["plan_validity_rate"],
            },
        },
        "family_task_success": {
            "full_gate_off": {FAMILIES[k]: v for k, v in _family_rates(off, "task_success_result").items()},
            "full_safety_gate": {FAMILIES[k]: v for k, v in _family_rates(on, "gated_task_success_result").items()},
        },
        "gate_diagnostics": {
            "blocked_runs": len(blocked),
            "prevented_physical_dispatches": prevented_dispatches,
            "false_blocks_on_p1_p5": len(false_blocks),
            "block_reasons": dict(block_reasons),
            "p6_physical_dispatch_errors_before": p6_err_before,
            "p6_physical_dispatch_errors_after": p6_err_after,
            "p6_task_success_before": sum(1 for r in p6_off if r["task_success_result"]["task_success"]) / len(p6_on),
            "p6_task_success_after": sum(1 for r in p6_on if r["gated_task_success_result"]["task_success"]) / len(p6_on),
        },
    }


def write_report(result: Dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "safety_gate_results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    h_off = result["headline"]["full_gate_off"]
    h_on = result["headline"]["full_safety_gate"]
    d = result["gate_diagnostics"]
    lines = [
        "# Safety Gate offline replay results",
        "",
        f"Source: `{result['source_run']}`",
        f"Gate OFF replay: `{result['gate_off_run']}`",
        f"Gate ON replay: `{result['gate_on_run']}`",
        "",
        "## Main comparison",
        "",
        "| Condition | Task Success | Plan Validity |",
        "|---|---:|---:|",
        f"| Full (gate OFF, replay) | {_pct(h_off['task_success_rate'])} ({h_off['n_task_success']}/{h_off['n_runs']}) | {_pct(h_off['plan_validity_rate'])} |",
        f"| Full + Safety Gate | {_pct(h_on['task_success_rate'])} ({h_on['n_task_success']}/{h_on['n_runs']}) | {_pct(h_on['plan_validity_rate'])} |",
        "",
        "## Family-wise Task Success",
        "",
        "| Condition | P1 | P2 | P3 | P4 | P5 | P6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Full (gate OFF)", "full_gate_off"), ("Full + Safety Gate", "full_safety_gate")):
        ft = result["family_task_success"][key]
        lines.append(
            f"| {label} | "
            + " | ".join(_pct(ft[f]) for f in ["P1", "P2", "P3", "P4", "P5", "P6"])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate diagnostics",
            "",
            f"- Blocked runs: {d['blocked_runs']}",
            f"- Prevented physical dispatches: {d['prevented_physical_dispatches']}",
            f"- False blocks on P1–P5: {d['false_blocks_on_p1_p5']}",
            f"- P6 physical-dispatch errors: {d['p6_physical_dispatch_errors_before']} → {d['p6_physical_dispatch_errors_after']}",
            f"- P6 Task Success: {_pct(d['p6_task_success_before'])} → {_pct(d['p6_task_success_after'])}",
            f"- Block reasons: `{d['block_reasons']}`",
            "",
        ]
    )
    md_path = REPORTS_DIR / "safety_gate_results.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-off", required=True)
    parser.add_argument("--gate-on", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.gate_off), Path(args.gate_on))
    write_report(result)


if __name__ == "__main__":
    main()
