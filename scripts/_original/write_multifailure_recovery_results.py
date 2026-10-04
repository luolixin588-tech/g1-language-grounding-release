#!/usr/bin/env python3
"""Write multi-failure recovery formal experiment reports."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.io_utils import read_jsonl, write_json
from src.paths import REPORTS_DIR
from src.recovery.multifailure_injection import FAILURE_CONDITIONS, _DIAGNOSTIC_TEXT
from src.recovery.multifailure_metrics import compute_multifailure_metrics, run_sanity_checks


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    records = read_jsonl(run_dir / "records.jsonl")
    metrics = compute_multifailure_metrics(records)
    sanity = run_sanity_checks(records)

    cfg_path = run_dir / "config.json"
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    by_cond: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        tid = r["semantic_task_id"]
        if tid not in by_cond[r["failure_condition"]]:
            by_cond[r["failure_condition"]].append(tid)

    m1 = metrics["m1_recovery_success_rate"]
    m2 = metrics["m2_correct_termination_rate"]
    m3 = metrics["m3_repeated_action_rate"]
    m4 = metrics["m4_incorrect_recovery_rate"]
    m5 = metrics["m5_added_recovery_latency_ms"]
    m6 = metrics["m6_replanning_attempts"]

    lines = [
        "# Multi-failure recovery formal results",
        "",
        f"Run directory: `{run_dir}`",
        f"Episodes: **{len(records)}** (target 112).",
        "",
        "Protocol: `reports/multifailure_recovery_audit.md`.",
        "Baseline R0/R1 recovery benchmark unchanged.",
        "",
        "## Failure-condition definitions",
        "",
        "| ID | Diagnostic semantics | Recoverable |",
        "|---|---|---|",
    ]
    for cond in FAILURE_CONDITIONS:
        rec = cond not in ("F7_unrecoverable_failure",)
        lines.append(
            f"| `{cond}` | {_DIAGNOSTIC_TEXT[cond]} | "
            f"{'yes' if rec else 'no (Correct Termination only)'} |"
        )

    lines += [
        "",
        "## Task coverage by condition",
        "",
        "| Failure condition | Task IDs |",
        "|---|---|",
    ]
    for cond in FAILURE_CONDITIONS:
        lines.append(f"| `{cond}` | {', '.join(sorted(by_cond.get(cond, [])))} |")

    lines += [
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| M1 Recovery Success (F1-F6) | **{_pct(m1['overall']['rate_percent'])}** "
        f"({m1['overall']['numerator']}/{m1['overall']['denominator']}) |",
        f"| M2 Correct Termination (F7) | **{_pct(m2['rate_percent'])}** "
        f"({m2['numerator']}/{m2['denominator']}) |",
        f"| M3 Repeated Action Rate | **{_pct(m3['rate_percent'])}** "
        f"({m3['numerator']}/{m3['denominator']}) |",
        f"| M4 Incorrect Recovery Rate | **{_pct(m4['rate_percent'])}** "
        f"({m4['numerator']}/{m4['denominator']}) |",
        f"| M5 Added latency median / P95 | "
        f"{m5['median']:.0f} / {m5['p95']:.0f} ms |"
        if m5["median"] is not None
        else "| M5 Added latency median / P95 | n/a |",
        f"| M6 Mean replan attempts | {m6['mean']:.2f} |"
        if m6["mean"] is not None
        else "| M6 Mean replan attempts | n/a |",
        "",
        "## Per-condition results",
        "",
        "| Condition | Episodes | Recovery success | Correct termination | Repeated action | Incorrect recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cond in FAILURE_CONDITIONS:
        subset = [r for r in records if r["failure_condition"] == cond]
        n_inj = sum(1 for r in subset if r.get("injected"))
        n_rec = sum(1 for r in subset if r.get("recovery_success"))
        n_ct = sum(1 for r in subset if r.get("correct_termination"))
        n_rep = sum(1 for r in subset if r.get("repeated_action"))
        n_inc = sum(1 for r in subset if r.get("incorrect_recovery"))
        lines.append(
            f"| `{cond}` | {len(subset)} ({n_inj} injected) | {n_rec}/{len(subset)} | "
            f"{n_ct}/{len(subset)} | {n_rep}/{len(subset)} | {n_inc}/{len(subset)} |"
        )

    lines += [
        "",
        "## M1 Recovery Success by condition (F1-F6)",
        "",
    ]
    for cond, stat in sorted(m1["by_condition"].items()):
        lines.append(
            f"- `{cond}`: {_pct(stat['rate'])} ({stat['numerator']}/{stat['denominator']})"
        )

    lines += [
        "",
        "## M4 Incorrect recovery reasons",
        "",
    ]
    if m4.get("reason_counts"):
        for k, v in sorted(m4["reason_counts"].items()):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## M6 Replanning attempt distribution",
        "",
        f"- recovered within 1 attempt: {m6['recovered_within_1']}",
        f"- recovered within 2 attempts: {m6['recovered_within_2']}",
        f"- recovered within 3 attempts: {m6['recovered_within_3']}",
        f"- failed after allowed attempts: {m6['failed_after_allowed_attempts']}",
        f"- histogram: `{m6.get('histogram')}`",
        "",
        "## Sanity checks (summary)",
        "",
    ]
    n_pass = sum(1 for c in sanity if c["passed"])
    lines.append(f"- {n_pass}/{len(sanity)} checks passed")
    failed = [c for c in sanity if not c["passed"]]
    if failed:
        lines.append("- Failed checks:")
        for c in failed[:20]:
            lines.append(f"  - `{c['check_id']}`: {c['detail']}")
        if len(failed) > 20:
            lines.append(f"  - ... and {len(failed) - 20} more")

    lines += [
        "",
        "Full episode records: `records.jsonl`. Raw planner strings: `raw_outputs.jsonl`.",
        "",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "multifailure_recovery_results.md"
    json_path = REPORTS_DIR / "multifailure_recovery_results.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "run_dir": str(run_dir),
        "config": cfg,
        "metrics": metrics,
        "sanity_checks": sanity,
        "n_episodes": len(records),
        "sanity_pass_rate": n_pass / len(sanity) if sanity else 1.0,
    }
    write_json(json_path, payload)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
