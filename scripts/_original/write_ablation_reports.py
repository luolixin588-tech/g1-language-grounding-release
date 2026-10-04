#!/usr/bin/env python3
"""Write A1/A2 ablation reports against the frozen v1.1 Full run."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.analysis.statistics import clustered_bootstrap, holm_correction, paired_mcnemar_all3
from src.analysis.summarize_results import summarize_records
from src.io_utils import read_jsonl, write_json
from src.paths import REPORTS_DIR


FAMILY_LABELS = {
    "P1_spoken_response": "P1 Spoken response",
    "P2_single_physical_action": "P2 Single physical action",
    "P3_parameterized_control": "P3 Parameterized control",
    "P4_multistep_physical_sequence": "P4 Multi-step physical sequence",
    "P5_mixed_speech_action_sequence": "P5 Mixed speech-action",
    "P6_unsupported_or_constrained": "P6 Unsupported / constrained",
}

BOOT_FIELDS = (
    "success",
    "valid",
    "latency_median",
    "diag.capability_selection_accuracy",
    "diag.parameter_accuracy",
    "diag.order_accuracy",
)


def _pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def _ms(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0f} ms"


def _ci_pct(pair) -> str:
    return f"[{100.0 * pair[0]:.1f}%, {100.0 * pair[1]:.1f}%]"


def _ci_ms(pair) -> str:
    return f"[{pair[0]:.0f}, {pair[1]:.0f}] ms"


def _ci_num(pair) -> str:
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def load_records(run_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(run_dir / "records.jsonl")


def parse_fail_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter = Counter()
    for r in records:
        if r.get("task_success_result", {}).get("task_success"):
            continue
        labels = tuple(r.get("task_success_result", {}).get("error_labels") or [])
        if "unparseable_plan" in labels or not r.get("validity_result", {}).get("valid"):
            if r.get("normalized_semantic_plan") is None:
                c["unparseable"] += 1
            else:
                c["invalid_plan"] += 1
        else:
            c["valid_but_task_fail"] += 1
        for lab in labels:
            c[f"label:{lab}"] += 1
    return dict(c)


def condition_table(summary: Dict[str, Any]) -> List[str]:
    fam = summary["success_by_family"]
    form = summary["success_by_formulation"]
    lat = summary["planning_latency_ms"]
    diag = summary["diagnostics"]
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Task Success Rate | **{_pct(summary['task_success_rate'])}** ({summary['n_task_success']} / {summary['n_runs']}) |",
        f"| Plan Validity | {_pct(summary['plan_validity_rate'])} ({summary['n_plan_valid']} / {summary['n_runs']}) |",
        f"| Latency median | {_ms(lat['median'])} |",
        f"| Latency P95 | {_ms(lat['p95'])} |",
        f"| Instructions succeeding in all 3 runs | {summary['n_instructions_success_all_repeats']} / {summary['n_instructions']} |",
        "",
        "### By family",
        "",
        "| Family | n | Task Success | Plan Validity | Cap. sel. | Param. | Order |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in FAMILY_LABELS.items():
        row = fam.get(key, {})
        lines.append(
            f"| {label} | {row.get('n', 0)} | {_pct(row.get('task_success_rate'))} | "
            f"{_pct(row.get('plan_validity_rate'))} | "
            f"{row.get('capability_selection_accuracy', float('nan')):.3f} | "
            f"{row.get('parameter_accuracy', float('nan')):.3f} | "
            f"{row.get('order_accuracy', float('nan')):.3f} |"
        )
    lines += [
        "",
        "### Canonical vs paraphrase",
        "",
        "| Formulation | n | Task Success |",
        "|---|---:|---:|",
        f"| Canonical | {form.get('canonical', {}).get('n', 0)} | {_pct(form.get('canonical', {}).get('task_success_rate'))} |",
        f"| Paraphrase | {form.get('paraphrase', {}).get('n', 0)} | {_pct(form.get('paraphrase', {}).get('task_success_rate'))} |",
        "",
        "### Diagnostics (mean over runs)",
        "",
        f"- Capability selection accuracy: {diag['capability_selection_accuracy']:.3f}",
        f"- Parameter accuracy: {diag['parameter_accuracy']:.3f}",
        f"- Order accuracy: {diag['order_accuracy']:.3f}",
        "",
    ]
    return lines


def comparison_section(
    full_sum: Dict[str, Any],
    abl_sum: Dict[str, Any],
    boot: Dict[str, Any],
    mcnemar: Dict[str, Any],
    holm_row: Dict[str, Any],
    full_name: str,
    abl_name: str,
) -> List[str]:
    f = boot["fields"]
    lines = [
        "## Comparison vs Full (matched instructions)",
        "",
        "Sampling unit: instruction (`sample_id`). The three repeats of an instruction",
        "are resampled together. 360 runs are not treated as 360 independent tasks.",
        "",
        f"McNemar aggregation (pre-declared): instruction Task Success = 1 iff all 3 repeats succeed.",
        "",
        "| Outcome | Full | Ablation | Difference (ablation − Full) | Clustered bootstrap 95% CI |",
        "|---|---:|---:|---:|---|",
        f"| Task Success | {_pct(full_sum['task_success_rate'])} | {_pct(abl_sum['task_success_rate'])} | "
        f"{_pct(f['success']['observed']['diff_b_minus_a'])} | {_ci_pct(f['success']['diff_ci95'])} |",
        f"| Plan Validity | {_pct(full_sum['plan_validity_rate'])} | {_pct(abl_sum['plan_validity_rate'])} | "
        f"{_pct(f['valid']['observed']['diff_b_minus_a'])} | {_ci_pct(f['valid']['diff_ci95'])} |",
        f"| Latency median | {_ms(full_sum['planning_latency_ms']['median'])} | {_ms(abl_sum['planning_latency_ms']['median'])} | "
        f"{f['latency_median']['observed']['diff_b_minus_a']:.0f} ms | {_ci_ms(f['latency_median']['diff_ci95'])} |",
        f"| Cap. selection | {full_sum['diagnostics']['capability_selection_accuracy']:.3f} | "
        f"{abl_sum['diagnostics']['capability_selection_accuracy']:.3f} | "
        f"{f['diag.capability_selection_accuracy']['observed']['diff_b_minus_a']:+.3f} | "
        f"{_ci_num(f['diag.capability_selection_accuracy']['diff_ci95'])} |",
        f"| Parameter acc. | {full_sum['diagnostics']['parameter_accuracy']:.3f} | "
        f"{abl_sum['diagnostics']['parameter_accuracy']:.3f} | "
        f"{f['diag.parameter_accuracy']['observed']['diff_b_minus_a']:+.3f} | "
        f"{_ci_num(f['diag.parameter_accuracy']['diff_ci95'])} |",
        f"| Order acc. | {full_sum['diagnostics']['order_accuracy']:.3f} | "
        f"{abl_sum['diagnostics']['order_accuracy']:.3f} | "
        f"{f['diag.order_accuracy']['observed']['diff_b_minus_a']:+.3f} | "
        f"{_ci_num(f['diag.order_accuracy']['diff_ci95'])} |",
        "",
        f"McNemar (instruction-level all-3 success), {full_name} vs {abl_name}:",
        f"both succeed {mcnemar['n_both_success']}, {full_name} only {mcnemar['n_a_only']},",
        f"{abl_name} only {mcnemar['n_b_only']}, both fail {mcnemar['n_both_fail']}.",
        f"Discordant n={mcnemar['mcnemar']['n_discordant']}, exact p={mcnemar['mcnemar']['p_value']:.4g}.",
        f"Holm-adjusted p={holm_row['p_holm']:.4g} ({'significant' if holm_row['significant_0.05'] else 'not significant'} at 0.05;",
        "Holm over the two primary ablation McNemar tests).",
        "",
    ]
    return lines


def write_condition_report(
    path: Path,
    title: str,
    intro: List[str],
    abl_dir: Path,
    abl_records: List[Dict[str, Any]],
    abl_sum: Dict[str, Any],
    full_sum: Dict[str, Any],
    boot: Dict[str, Any],
    mcnemar: Dict[str, Any],
    holm_row: Dict[str, Any],
    abl_name: str,
) -> None:
    br = parse_fail_breakdown(abl_records)
    lines = [
        f"# {title}",
        "",
        f"Ablation run: `{abl_dir}`",
        f"Full reference: `runs/full/full_20260821T051022Z_d81f78bf`",
        "",
        "Benchmark v1.1 is frozen. Gold, evaluator, and planner prompt were not modified.",
        "",
        *intro,
        "",
        "## Ablation headline metrics",
        "",
        *condition_table(abl_sum),
        "## Failure breakdown (unsuccessful runs)",
        "",
        f"- Unparseable: {br.get('unparseable', 0)}",
        f"- Parsed but invalid under catalog: {br.get('invalid_plan', 0)}",
        f"- Valid plan, task fail: {br.get('valid_but_task_fail', 0)}",
        "",
        *comparison_section(full_sum, abl_sum, boot, mcnemar, holm_row, "Full", abl_name),
        "Raw model outputs are stored in `records.jsonl` (`raw_model_output`) and `raw_outputs.jsonl`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--no-structure", required=True)
    parser.add_argument("--reduced", required=True)
    args = parser.parse_args()

    full_dir = Path(args.full)
    a1_dir = Path(args.no_structure)
    a2_dir = Path(args.reduced)

    full_recs = load_records(full_dir)
    a1_recs = load_records(a1_dir)
    a2_recs = load_records(a2_dir)

    full_sum = summarize_records(full_recs, condition="full")
    a1_sum = summarize_records(a1_recs, condition="no_structure")
    a2_sum = summarize_records(a2_recs, condition="reduced_capability_info")

    boot1 = clustered_bootstrap(full_recs, a1_recs, BOOT_FIELDS, n_boot=10000, seed=0)
    boot2 = clustered_bootstrap(full_recs, a2_recs, BOOT_FIELDS, n_boot=10000, seed=0)
    mc1 = paired_mcnemar_all3(full_recs, a1_recs)
    mc2 = paired_mcnemar_all3(full_recs, a2_recs)
    holm = holm_correction(
        [
            ("A1_no_structure", mc1["mcnemar"]["p_value"]),
            ("A2_reduced_capability_info", mc2["mcnemar"]["p_value"]),
        ]
    )
    holm_by = {row["name"]: row for row in holm}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_condition_report(
        REPORTS_DIR / "ablation_structured_generation.md",
        "A1 — Remove structured generation",
        [
            "Intervention: JSON-schema / constrained decoding is **off**.",
            "Held fixed: Qwen2.5-14B, temperature=0, v1.1 planner prompt, full capability catalog,",
            "frozen 120 instructions × 3, no dialogue history, v1.1 evaluator.",
            "Raw output is mapped by a deterministic unstructured parser into the same semantic",
            "representation as Full. Unparseable output is Plan Validity = 0. No LLM parser/judge,",
            "and no silent semantic repair.",
        ],
        a1_dir,
        a1_recs,
        a1_sum,
        full_sum,
        boot1,
        mc1,
        holm_by["A1_no_structure"],
        "A1",
    )
    write_condition_report(
        REPORTS_DIR / "ablation_capability_information.md",
        "A2 — Reduced capability information",
        [
            "Intervention: planner-facing catalog is reduced to capability **names** plus coarse",
            "parameter types. Detailed descriptions, semantic guidance (`stand_up` vs `start`,",
            "action maps, direction enums in the catalog text) are removed.",
            "Held fixed: structured generation **on** (same output JSON schema as Full),",
            "Qwen2.5-14B, temperature=0, v1.1 planner prompt, frozen 120 × 3, v1.1 evaluator.",
            "This is not an action-space ablation: the same tool names remain available.",
        ],
        a2_dir,
        a2_recs,
        a2_sum,
        full_sum,
        boot2,
        mc2,
        holm_by["A2_reduced_capability_info"],
        "A2",
    )

    summary_lines = [
        "# Ablation summary (v1.1 Full vs A1 / A2)",
        "",
        "v1.1 is frozen. No gold, paraphrase, evaluator, or planner-prompt changes were made",
        "after seeing these results. `no_validation` was **not** run. Execution-time replanning",
        "is out of scope for this offline benchmark.",
        "",
        f"- Full: `{full_dir}`",
        f"- A1 no_structure: `{a1_dir}`",
        f"- A2 reduced_capability_info: `{a2_dir}`",
        "",
        "## Headline",
        "",
        "| Condition | Task Success | Plan Validity | Latency median / P95 | All-3 / 120 |",
        "|---|---:|---:|---:|---:|",
        f"| Full | {_pct(full_sum['task_success_rate'])} | {_pct(full_sum['plan_validity_rate'])} | "
        f"{_ms(full_sum['planning_latency_ms']['median'])} / {_ms(full_sum['planning_latency_ms']['p95'])} | "
        f"{full_sum['n_instructions_success_all_repeats']} / {full_sum['n_instructions']} |",
        f"| A1 no structured generation | {_pct(a1_sum['task_success_rate'])} | {_pct(a1_sum['plan_validity_rate'])} | "
        f"{_ms(a1_sum['planning_latency_ms']['median'])} / {_ms(a1_sum['planning_latency_ms']['p95'])} | "
        f"{a1_sum['n_instructions_success_all_repeats']} / {a1_sum['n_instructions']} |",
        f"| A2 reduced capability info | {_pct(a2_sum['task_success_rate'])} | {_pct(a2_sum['plan_validity_rate'])} | "
        f"{_ms(a2_sum['planning_latency_ms']['median'])} / {_ms(a2_sum['planning_latency_ms']['p95'])} | "
        f"{a2_sum['n_instructions_success_all_repeats']} / {a2_sum['n_instructions']} |",
        "",
        "## Primary paired tests (McNemar, instruction all-3 success, Holm over 2 tests)",
        "",
        "| Comparison | Full only | Ablation only | Both success | Both fail | p raw | p Holm | sig. 0.05 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| Full vs A1 | {mc1['n_a_only']} | {mc1['n_b_only']} | {mc1['n_both_success']} | {mc1['n_both_fail']} | "
        f"{mc1['mcnemar']['p_value']:.4g} | {holm_by['A1_no_structure']['p_holm']:.4g} | "
        f"{'yes' if holm_by['A1_no_structure']['significant_0.05'] else 'no'} |",
        f"| Full vs A2 | {mc2['n_a_only']} | {mc2['n_b_only']} | {mc2['n_both_success']} | {mc2['n_both_fail']} | "
        f"{mc2['mcnemar']['p_value']:.4g} | {holm_by['A2_reduced_capability_info']['p_holm']:.4g} | "
        f"{'yes' if holm_by['A2_reduced_capability_info']['significant_0.05'] else 'no'} |",
        "",
        "Clustered bootstrap 95% CIs (instruction resampling, 10000 replicates, seed 0):",
        "",
        f"- A1 − Full Task Success: {_ci_pct(boot1['fields']['success']['diff_ci95'])}",
        f"- A2 − Full Task Success: {_ci_pct(boot2['fields']['success']['diff_ci95'])}",
        f"- A1 − Full Plan Validity: {_ci_pct(boot1['fields']['valid']['diff_ci95'])}",
        f"- A2 − Full Plan Validity: {_ci_pct(boot2['fields']['valid']['diff_ci95'])}",
        "",
        "See `reports/ablation_structured_generation.md` and",
        "`reports/ablation_capability_information.md` for family splits, diagnostics, and",
        "canonical vs paraphrase.",
        "",
        "Stop: no further ablations and no v1.1 edits.",
        "",
    ]
    (REPORTS_DIR / "ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    write_json(
        REPORTS_DIR / "ablation_stats.json",
        {
            "full": str(full_dir),
            "a1": str(a1_dir),
            "a2": str(a2_dir),
            "bootstrap_a1": boot1,
            "bootstrap_a2": boot2,
            "mcnemar_a1": mc1,
            "mcnemar_a2": mc2,
            "holm": holm,
        },
    )
    print("wrote", REPORTS_DIR / "ablation_structured_generation.md")
    print("wrote", REPORTS_DIR / "ablation_capability_information.md")
    print("wrote", REPORTS_DIR / "ablation_summary.md")


if __name__ == "__main__":
    main()
