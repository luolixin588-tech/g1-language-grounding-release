"""Cross-model planner-interface analysis, plots, REPORT, and gate replay."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.statistics import clustered_bootstrap, holm_correction, paired_mcnemar_all3
from src.analysis.summarize_results import summarize_records
from src.catalog import PHYSICAL_CAPABILITIES
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.paths import INSTRUCTIONS_PATH, REPO_ROOT, RUNS_DIR
from src.runner.experiment_metadata import list_model_configs
from src.evaluator.diagnostics import compute_diagnostics
from src.evaluator.task_success import evaluate_task_success
from src.gate.safety_gate import apply_safety_gate, gated_semantic_plan

MULTIMODEL_DIR = RUNS_DIR / "multimodel_v1.1"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "multimodel_v1.1"

PAPER_CONDITIONS = [
    "full",
    "reduced_capability_info",
    "no_capability_semantics",
    "no_parameter_semantics",
    "no_schema_constrained_decoding",
]
ABLATIONS = [
    "no_capability_semantics",
    "no_parameter_semantics",
    "no_schema_constrained_decoding",
]
# Coarse reference: reported with effect + CI, not in the Holm family.
REFERENCE_CONDITIONS = [
    "reduced_capability_info",
]
CONDITION_LABELS = {
    "full": "Full",
    "reduced_capability_info": "Reduced capability information",
    "no_capability_semantics": "No capability semantics",
    "no_parameter_semantics": "No parameter semantics",
    "no_schema_constrained_decoding": "No schema-constrained decoding",
}
FAMILIES = {
    "P1_spoken_response": "P1",
    "P2_single_physical_action": "P2",
    "P3_parameterized_control": "P3",
    "P4_multistep_physical_sequence": "P4",
    "P5_mixed_speech_action_sequence": "P5",
    "P6_unsupported_or_constrained": "P6",
}
MODEL_ORDER = ["qwen2.5_14b", "phi4_14b", "gemma3_12b", "mistral_nemo_12b"]
BOOT_FIELDS = ("success", "valid", "latency_median")


def run_replay(source_run: Path, out_dir: Path, *, gate_enabled: bool, condition_label: str) -> Path:
    records = read_jsonl(source_run / "records.jsonl")
    tasks = {r["sample_id"]: r for r in read_jsonl(INSTRUCTIONS_PATH)}
    replayed = []
    for rec in records:
        task = tasks[rec["sample_id"]]
        out = dict(rec)
        plan = rec.get("normalized_semantic_plan")
        valid = bool(rec.get("validity_result", {}).get("valid"))
        if gate_enabled:
            gate = apply_safety_gate(rec["instruction"], plan)
            exec_plan = gated_semantic_plan(plan, gate)
            success = evaluate_task_success(task, exec_plan, valid)
            diag = compute_diagnostics(task, exec_plan)
            out["safety_gate"] = gate
            out["gated_semantic_plan"] = exec_plan
            out["gate_enabled"] = True
        else:
            success = rec["task_success_result"]
            diag = rec.get("diagnostics") or compute_diagnostics(task, plan)
            out["safety_gate"] = {
                "decision": "not_applicable",
                "reason": "gate_disabled",
                "physical_dispatch_allowed": None,
            }
            out["gated_semantic_plan"] = plan
            out["gate_enabled"] = False
        out["gated_task_success_result"] = success
        out["gated_diagnostics"] = diag
        replayed.append(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "records.jsonl", replayed, append=False)
    summary = summarize_records(
        [
            {
                **r,
                "task_success_result": r["gated_task_success_result"],
                "diagnostics": r["gated_diagnostics"],
            }
            for r in replayed
        ],
        run_id=out_dir.name,
        condition=condition_label,
    )
    write_json(out_dir / "summary.json", summary)
    return out_dir


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.1f}%"


def _pp(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:+.1f}"


def resolve_records_dir(model_id: str, paper_condition: str) -> Path:
    dest = MULTIMODEL_DIR / model_id / paper_condition
    reused = dest / "REUSED.json"
    if reused.exists():
        src = Path(json.loads(reused.read_text(encoding="utf-8"))["source_run"])
        return src
    return dest


def load_records(model_id: str, paper_condition: str) -> List[Dict[str, Any]]:
    return read_jsonl(resolve_records_dir(model_id, paper_condition) / "records.jsonl")


def family_rates(records: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    by: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        by[r["family"]].append(bool(r.get("task_success_result", {}).get("task_success")))
    out: Dict[str, Optional[float]] = {}
    for fam, short in FAMILIES.items():
        rows = by.get(fam) or []
        out[short] = (sum(rows) / len(rows)) if rows else None
    return out


def fail_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels: Counter = Counter()
    invalid = unparseable = valid_fail = p6_phys = 0
    n_p6 = 0
    for r in records:
        ok = bool(r.get("task_success_result", {}).get("task_success"))
        valid = bool(r.get("validity_result", {}).get("valid"))
        if r.get("family", "").startswith("P6"):
            n_p6 += 1
        if not valid:
            invalid += 1
            if r.get("normalized_semantic_plan") is None:
                unparseable += 1
        elif not ok:
            valid_fail += 1
        labs = r.get("task_success_result", {}).get("error_labels") or []
        if not ok:
            for lab in labs:
                labels[lab] += 1
        if "p6_physical_action_dispatched" in labs:
            p6_phys += 1
    n = len(records)
    cap = [float(r["diagnostics"]["capability_selection_accuracy"]) for r in records if "diagnostics" in r]
    par = [float(r["diagnostics"]["parameter_accuracy"]) for r in records if "diagnostics" in r]
    order = [float(r["diagnostics"]["order_accuracy"]) for r in records if "diagnostics" in r]
    return {
        "n": n,
        "invalid_plans": invalid,
        "unparseable_plans": unparseable,
        "valid_plan_task_failures": valid_fail,
        "p6_physical_dispatch_errors": p6_phys,
        "n_p6": n_p6,
        "error_labels": dict(labels),
        "capability_selection_accuracy": (sum(cap) / len(cap)) if cap else None,
        "parameter_accuracy": (sum(par) / len(par)) if par else None,
        "order_accuracy": (sum(order) / len(order)) if order else None,
        "capability_selection_error": (1.0 - sum(cap) / len(cap)) if cap else None,
        "parameter_error": (1.0 - sum(par) / len(par)) if par else None,
        "order_error": (1.0 - sum(order) / len(order)) if order else None,
    }


def analyze_model(model_id: str, display: str) -> Dict[str, Any]:
    recs = {c: load_records(model_id, c) for c in PAPER_CONDITIONS}
    sums = {
        c: summarize_records(recs[c], run_id=f"{model_id}_{c}", condition=c)
        for c in PAPER_CONDITIONS
    }
    stats = {}
    named_p = []
    for ab in ABLATIONS:
        boot = clustered_bootstrap(recs["full"], recs[ab], BOOT_FIELDS, n_boot=10000, seed=0)
        mc = paired_mcnemar_all3(recs["full"], recs[ab])
        stats[ab] = {"bootstrap": boot, "mcnemar": mc}
        named_p.append((ab, mc["mcnemar"]["p_value"]))
    holm = holm_correction(named_p)
    holm_by = {x["name"]: x for x in holm}

    effects = {}
    for ab in ABLATIONS:
        full_s = sums["full"]["task_success_rate"]
        ab_s = sums[ab]["task_success_rate"]
        diff = ab_s - full_s
        ci = stats[ab]["bootstrap"]["fields"]["success"]["diff_ci95"]
        effects[ab] = {
            "delta": diff,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "mcnemar_raw_p": stats[ab]["mcnemar"]["mcnemar"]["p_value"],
            "p_adjusted": holm_by[ab]["p_holm"],
            "significant_0.05": holm_by[ab]["significant_0.05"],
            "holm_family": True,
        }
    for ref in REFERENCE_CONDITIONS:
        boot = clustered_bootstrap(recs["full"], recs[ref], BOOT_FIELDS, n_boot=10000, seed=0)
        stats[ref] = {"bootstrap": boot, "mcnemar": None}
        full_s = sums["full"]["task_success_rate"]
        ref_s = sums[ref]["task_success_rate"]
        ci = boot["fields"]["success"]["diff_ci95"]
        effects[ref] = {
            "delta": ref_s - full_s,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "mcnemar_raw_p": None,
            "p_adjusted": None,
            "significant_0.05": None,
            "holm_family": False,
        }

    return {
        "model_id": model_id,
        "model_display_name": display,
        "summaries": sums,
        "family": {c: family_rates(recs[c]) for c in PAPER_CONDITIONS},
        "diagnostics": {c: fail_diagnostics(recs[c]) for c in PAPER_CONDITIONS},
        "effects": effects,
        "source_dirs": {c: str(resolve_records_dir(model_id, c)) for c in PAPER_CONDITIONS},
    }


def write_csvs(results: List[Dict[str, Any]]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / "cross_model_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "condition",
                "task_success",
                "plan_validity",
                "median_latency",
                "p95_latency",
                "instructions_all3",
                "delta_vs_full",
                "ci_low",
                "ci_high",
                "p_adjusted",
            ],
        )
        w.writeheader()
        for res in results:
            for cond in PAPER_CONDITIONS:
                s = res["summaries"][cond]
                eff = res["effects"].get(cond, {})
                w.writerow(
                    {
                        "model": res["model_display_name"],
                        "condition": cond,
                        "task_success": s["task_success_rate"],
                        "plan_validity": s["plan_validity_rate"],
                        "median_latency": (s.get("planning_latency_ms") or {}).get("median"),
                        "p95_latency": (s.get("planning_latency_ms") or {}).get("p95"),
                        "instructions_all3": s.get("n_instructions_success_all_repeats"),
                        "delta_vs_full": eff.get("delta", ""),
                        "ci_low": eff.get("ci_low", ""),
                        "ci_high": eff.get("ci_high", ""),
                        "p_adjusted": "" if eff.get("p_adjusted") is None else eff.get("p_adjusted", ""),
                    }
                )

    fam_path = OUTPUTS_DIR / "cross_model_task_family.csv"
    with fam_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "condition", "P1", "P2", "P3", "P4", "P5", "P6"],
        )
        w.writeheader()
        for res in results:
            for cond in PAPER_CONDITIONS:
                ft = res["family"][cond]
                w.writerow({"model": res["model_display_name"], "condition": cond, **ft})

    diag_path = OUTPUTS_DIR / "cross_model_diagnostics.csv"
    with diag_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "condition",
                "capability_selection_error",
                "parameter_error",
                "order_error",
                "invalid_plans",
                "unparseable_plans",
                "p6_physical_dispatch_errors",
            ],
        )
        w.writeheader()
        for res in results:
            for cond in PAPER_CONDITIONS:
                d = res["diagnostics"][cond]
                w.writerow(
                    {
                        "model": res["model_display_name"],
                        "condition": cond,
                        "capability_selection_error": d["capability_selection_error"],
                        "parameter_error": d["parameter_error"],
                        "order_error": d["order_error"],
                        "invalid_plans": d["invalid_plans"],
                        "unparseable_plans": d["unparseable_plans"],
                        "p6_physical_dispatch_errors": d["p6_physical_dispatch_errors"],
                    }
                )


def write_figure_s1_csvs(results: List[Dict[str, Any]]) -> None:
    effects_path = OUTPUTS_DIR / "cross_model_effects.csv"
    with effects_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "condition",
                "task_success",
                "full_task_success",
                "effect_pp",
                "ci_low",
                "ci_high",
                "plan_validity",
                "median_latency_ms",
                "p95_latency_ms",
                "all3_success_n",
            ],
        )
        w.writeheader()
        for res in results:
            full_s = res["summaries"]["full"]["task_success_rate"]
            for cond in PAPER_CONDITIONS:
                s = res["summaries"][cond]
                lat = s.get("planning_latency_ms") or {}
                eff = res["effects"].get(cond, {})
                if cond == "full":
                    effect_pp = ci_low = ci_high = ""
                else:
                    effect_pp = 100.0 * eff["delta"]
                    ci_low = 100.0 * eff["ci_low"]
                    ci_high = 100.0 * eff["ci_high"]
                w.writerow(
                    {
                        "model": res["model_display_name"],
                        "condition": cond,
                        "task_success": s["task_success_rate"],
                        "full_task_success": full_s,
                        "effect_pp": effect_pp,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "plan_validity": s["plan_validity_rate"],
                        "median_latency_ms": lat.get("median"),
                        "p95_latency_ms": lat.get("p95"),
                        "all3_success_n": s.get("n_instructions_success_all_repeats"),
                    }
                )

    cat_path = OUTPUTS_DIR / "cross_model_task_categories.csv"
    with cat_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "condition", "P1", "P2", "P3", "P4", "P5", "P6"],
        )
        w.writeheader()
        for res in results:
            for cond in PAPER_CONDITIONS:
                ft = res["family"][cond]
                w.writerow({"model": res["model_display_name"], "condition": cond, **ft})


def qc_multimodel_cells(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[str] = []
    cells = []
    for res in results:
        for cond in PAPER_CONDITIONS:
            recs = load_records(res["model_id"], cond)
            sample_ids = [r["sample_id"] for r in recs]
            unique = sorted(set(sample_ids))
            by = defaultdict(list)
            for r in recs:
                by[r["sample_id"]].append(r.get("repeat_index"))
            n = len(recs)
            n_inst = len(unique)
            bad_rep = [sid for sid, xs in by.items() if len(xs) != 3 or len(set(xs)) != 3]
            missing_lat = sum(1 for r in recs if "total_planning_latency_ms" not in r)
            missing_eval = sum(
                1
                for r in recs
                if "task_success_result" not in r or "validity_result" not in r
            )
            models = sorted({r.get("model") for r in recs})
            conds = sorted({r.get("condition") for r in recs})
            cell_issues = []
            if n != 360:
                cell_issues.append(f"n_runs={n}")
            if n_inst != 120:
                cell_issues.append(f"n_instructions={n_inst}")
            if bad_rep:
                cell_issues.append(f"bad_repeats={len(bad_rep)}")
            if missing_lat:
                cell_issues.append(f"missing_latency={missing_lat}")
            if missing_eval:
                cell_issues.append(f"missing_evaluator={missing_eval}")
            issues.extend(f"{res['model_id']}/{cond}: {x}" for x in cell_issues)
            cells.append(
                {
                    "model_id": res["model_id"],
                    "condition": cond,
                    "n_runs": n,
                    "n_instructions": n_inst,
                    "models_in_records": models,
                    "conditions_in_records": conds,
                    "issues": cell_issues,
                }
            )
    report = {"ok": len(issues) == 0, "n_issues": len(issues), "issues": issues, "cells": cells}
    write_json(OUTPUTS_DIR / "cross_model_qc.json", report)
    return report


def plot_ablation_effects(results: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = {
        "no_capability_semantics": "#1f77b4",
        "no_parameter_semantics": "#ff7f0e",
        "no_schema_constrained_decoding": "#2ca02c",
    }
    markers = {
        "no_capability_semantics": "o",
        "no_parameter_semantics": "s",
        "no_schema_constrained_decoding": "D",
    }
    n_models = len(results)
    y_base = list(range(n_models))
    offsets = {
        "no_capability_semantics": 0.22,
        "no_parameter_semantics": 0.0,
        "no_schema_constrained_decoding": -0.22,
    }
    for ab in ABLATIONS:
        xs, xerr_lo, xerr_hi, ys = [], [], [], []
        for i, res in enumerate(results):
            eff = res["effects"][ab]
            xs.append(100.0 * eff["delta"])
            xerr_lo.append(100.0 * (eff["delta"] - eff["ci_low"]))
            xerr_hi.append(100.0 * (eff["ci_high"] - eff["delta"]))
            ys.append(y_base[i] + offsets[ab])
        ax.errorbar(
            xs,
            ys,
            xerr=[xerr_lo, xerr_hi],
            fmt=markers[ab],
            color=colors[ab],
            ecolor=colors[ab],
            elinewidth=1.2,
            capsize=3,
            label=CONDITION_LABELS[ab],
            markersize=6,
        )
    ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_base)
    ax.set_yticklabels([r["model_display_name"] for r in results])
    ax.set_xlabel("Change in Task Success relative to Full (percentage points)")
    ax.set_title("Cross-model planner-interface ablation effects")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cross_model_ablation_effects.pdf")
    fig.savefig(OUTPUTS_DIR / "cross_model_ablation_effects.svg")
    plt.close(fig)


def plot_task_family(results: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(results)
    nrows = 1 if n <= 2 else 2
    ncols = n if n <= 2 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.6 * nrows), sharey=True)
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.ravel())
    shorts = ["P1", "P2", "P3", "P4", "P5", "P6"]
    x = np.arange(len(shorts))
    width = 0.15
    colors = ["#4c72b0", "#8172b3", "#dd8452", "#55a868", "#c44e52"]
    for ax, res in zip(axes, results):
        for j, cond in enumerate(PAPER_CONDITIONS):
            ys = [100.0 * (res["family"][cond][k] or 0.0) for k in shorts]
            ax.bar(x + (j - 2.0) * width, ys, width, label=CONDITION_LABELS[cond], color=colors[j])
        ax.set_title(res["model_display_name"])
        ax.set_xticks(x)
        ax.set_xticklabels(shorts)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    for ax in axes[len(results):]:
        ax.axis("off")
    axes[0].set_ylabel("Task Success (%)")
    if len(axes) > 2 and len(results) > 2:
        axes[2].set_ylabel("Task Success (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cross_model_task_family.pdf", bbox_inches="tight")
    fig.savefig(OUTPUTS_DIR / "cross_model_task_family.svg", bbox_inches="tight")
    plt.close(fig)


def _has_physical(plan) -> bool:
    return bool(plan) and any(s.get("capability") in PHYSICAL_CAPABILITIES for s in plan)


def gate_metrics(off: List[Dict[str, Any]], on: List[Dict[str, Any]]) -> Dict[str, Any]:
    blocked = [r for r in on if r.get("safety_gate", {}).get("decision") == "block"]
    p1p5_on = [r for r in on if not r.get("family", "").startswith("P6")]
    p6_on = [r for r in on if r.get("family", "").startswith("P6")]
    p6_off = [r for r in off if r.get("family", "").startswith("P6")]
    false_blocks = [r for r in blocked if not r.get("family", "").startswith("P6")]
    intercepted = 0
    for r in on:
        if _has_physical(r.get("normalized_semantic_plan")) and not _has_physical(r.get("gated_semantic_plan")):
            intercepted += 1
    p6_err_before = sum(
        1
        for r in p6_off
        if "p6_physical_action_dispatched" in (r.get("task_success_result", {}).get("error_labels") or [])
    )
    p6_err_after = sum(
        1
        for r in p6_on
        if "p6_physical_action_dispatched" in (r.get("gated_task_success_result", {}).get("error_labels") or [])
    )
    p1p5_physical = [r for r in p1p5_on if _has_physical(r.get("normalized_semantic_plan"))]
    p1p5_physical_pass = [
        r for r in p1p5_physical if r.get("safety_gate", {}).get("decision") == "pass"
    ]
    return {
        "p6_erroneous_physical_dispatch_before_gate": p6_err_before,
        "p6_erroneous_physical_dispatch_after_gate": p6_err_after,
        "number_intercepted": intercepted,
        "valid_physical_output_pass_rate": (
            (len(p1p5_physical_pass) / len(p1p5_physical)) if p1p5_physical else None
        ),
        "false_block_rate": (len(false_blocks) / len(p1p5_on)) if p1p5_on else None,
        "false_blocks_p1_p5": len(false_blocks),
        "blocked_runs": len(blocked),
        "n_p1_p5": len(p1p5_on),
        "n_p6": len(p6_on),
    }


def replay_gates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for res in results:
        source = Path(res["source_dirs"]["full"])
        mid = res["model_id"]
        off_dir = MULTIMODEL_DIR / "safety_gate" / mid / "gate_off"
        on_dir = MULTIMODEL_DIR / "safety_gate" / mid / "gate_on"
        if not (off_dir / "records.jsonl").exists():
            run_replay(source, off_dir, gate_enabled=False, condition_label="full_gate_off")
        if not (on_dir / "records.jsonl").exists():
            run_replay(source, on_dir, gate_enabled=True, condition_label="full_safety_gate")
        off = read_jsonl(off_dir / "records.jsonl")
        on = read_jsonl(on_dir / "records.jsonl")
        metrics = gate_metrics(off, on)
        metrics.update(
            {
                "model": res["model_display_name"],
                "model_id": mid,
                "source_run": str(source),
            }
        )
        rows.append(metrics)
    path = OUTPUTS_DIR / "cross_model_gate_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "p6_erroneous_physical_dispatch_before_gate",
                "p6_erroneous_physical_dispatch_after_gate",
                "number_intercepted",
                "valid_physical_output_pass_rate",
                "false_block_rate",
                "false_blocks_p1_p5",
                "blocked_runs",
                "source_run",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in w.fieldnames})
    write_json(OUTPUTS_DIR / "cross_model_gate_summary.json", rows)
    return rows


def pattern_checks(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    cap_down = []
    par_down = []
    sem_gt_schema = []
    p2_cap = []
    p3_par = []
    for res in results:
        name = res["model_display_name"]
        d_cap = res["effects"]["no_capability_semantics"]["delta"]
        d_par = res["effects"]["no_parameter_semantics"]["delta"]
        d_sch = res["effects"]["no_schema_constrained_decoding"]["delta"]
        cap_down.append({"model": name, "delta": d_cap, "decreased": d_cap < 0})
        par_down.append({"model": name, "delta": d_par, "decreased": d_par < 0})
        sem_gt_schema.append(
            {
                "model": name,
                "abs_cap": abs(d_cap),
                "abs_par": abs(d_par),
                "abs_schema": abs(d_sch),
                "semantics_larger": min(abs(d_cap), abs(d_par)) > abs(d_sch),
            }
        )
        p2_full = res["family"]["full"]["P2"]
        p2_ab = res["family"]["no_capability_semantics"]["P2"]
        p3_full = res["family"]["full"]["P3"]
        p3_ab = res["family"]["no_parameter_semantics"]["P3"]
        p2_cap.append(
            {
                "model": name,
                "full": p2_full,
                "no_capability_semantics": p2_ab,
                "delta": (None if p2_full is None or p2_ab is None else p2_ab - p2_full),
                "decreased": (p2_ab is not None and p2_full is not None and p2_ab < p2_full),
            }
        )
        p3_par.append(
            {
                "model": name,
                "full": p3_full,
                "no_parameter_semantics": p3_ab,
                "delta": (None if p3_full is None or p3_ab is None else p3_ab - p3_full),
                "decreased": (p3_ab is not None and p3_full is not None and p3_ab < p3_full),
            }
        )
    return {
        "capability_semantics_decreases_all": all(x["decreased"] for x in cap_down),
        "parameter_semantics_decreases_all": all(x["decreased"] for x in par_down),
        "semantic_effect_larger_than_schema_all": all(x["semantics_larger"] for x in sem_gt_schema),
        "p2_capability_drop_all": all(x["decreased"] for x in p2_cap),
        "p3_parameter_drop_all": all(x["decreased"] for x in p3_par),
        "per_model": {
            "capability": cap_down,
            "parameter": par_down,
            "semantics_vs_schema": sem_gt_schema,
            "p2": p2_cap,
            "p3": p3_par,
        },
    }


def write_report(
    results: List[Dict[str, Any]],
    gate_rows: List[Dict[str, Any]],
    patterns: Dict[str, Any],
    deviations: List[str],
) -> None:
    lines = [
        "# Cross-Model Generalization Experiment",
        "",
        "## 1 Experimental setup",
        "",
        "Benchmark: G1 Internal Planning Benchmark **v1.1** (frozen; 120 instructions × 3 runs = 360 / condition / model).",
        "",
        "Models (local Ollama, identical planner interface):",
        "",
    ]
    for res in results:
        lines.append(f"- {res['model_display_name']} (`{res['model_id']}`)")
    lines.extend(
        [
            "",
            "Conditions (existing formal configs; `no_schema_constrained_decoding` reuses `no_structure`;",
            "`reduced_capability_info` is the frozen names-only catalog condition):",
            "",
            "- full",
            "- reduced_capability_info (coarse reference; not in Holm family)",
            "- no_capability_semantics",
            "- no_parameter_semantics",
            "- no_schema_constrained_decoding",
            "",
            "Held fixed across models: capability catalog, system/user prompt, JSON Schema (when enabled),",
            "evaluator, Task Success / Plan Validity definitions, temperature = 0, seed = 0, 3 repeats.",
            "No model-specific prompt, few-shot, or repair text was added.",
            "",
            "Qwen2.5-14B formal cells reuse the frozen v1.1 runs under decision A",
            "(model parameterization only; see `qwen_reuse_justification.json`).",
            "",
            "## 2 Full-condition performance",
            "",
            "| Model | Task Success | Plan Validity | Median latency | P95 latency | 3/3 instructions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for res in results:
        s = res["summaries"]["full"]
        lat = s["planning_latency_ms"]
        lines.append(
            f"| {res['model_display_name']} | {_pct(s['task_success_rate'])} | {_pct(s['plan_validity_rate'])} | "
            f"{lat['median']:.0f} ms | {lat['p95']:.0f} ms | {s['n_instructions_success_all_repeats']} / 120 |"
        )

    lines.extend(
        [
            "",
            "## 3 Planner-interface ablations",
            "",
            "| Model | Condition | Task Success | Plan Validity | Median | P95 | 3/3 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for res in results:
        for cond in PAPER_CONDITIONS:
            s = res["summaries"][cond]
            lat = s["planning_latency_ms"]
            lines.append(
                f"| {res['model_display_name']} | {CONDITION_LABELS[cond]} | {_pct(s['task_success_rate'])} | "
                f"{_pct(s['plan_validity_rate'])} | {lat['median']:.0f} ms | {lat['p95']:.0f} ms | "
                f"{s['n_instructions_success_all_repeats']} / 120 |"
            )

    lines.extend(
        [
            "",
            "## 4 Ablation effects",
            "",
            "Delta = TaskSuccess(ablation) − TaskSuccess(full). Instruction-clustered bootstrap 95% CI;",
            "McNemar on instruction-level all-3 success; Holm correction within each model **only** for",
            "the three controlled ablations (capability / parameter / schema).",
            "",
            "| Model | Effect | Δ Task Success (pp) | 95% CI (pp) | raw p | Holm p |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for res in results:
        for ab in ABLATIONS:
            e = res["effects"][ab]
            lines.append(
                f"| {res['model_display_name']} | {CONDITION_LABELS[ab]} | {_pp(e['delta'])} | "
                f"[{_pp(e['ci_low'])}, {_pp(e['ci_high'])}] | {e['mcnemar_raw_p']:.4g} | {e['p_adjusted']:.4g} |"
            )

    lines.extend(
        [
            "",
            "### Coarse reference: Reduced capability information",
            "",
            "Not included in the Holm family. Effect size and instruction-clustered bootstrap 95% CI only.",
            "",
            "| Model | Task Success | Δ vs Full (pp) | 95% CI (pp) | Plan Validity | 3/3 |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for res in results:
        s = res["summaries"]["reduced_capability_info"]
        e = res["effects"]["reduced_capability_info"]
        lines.append(
            f"| {res['model_display_name']} | {_pct(s['task_success_rate'])} "
            f"({s['n_task_success']}/{s['n_runs']}) | {_pp(e['delta'])} | "
            f"[{_pp(e['ci_low'])}, {_pp(e['ci_high'])}] | {_pct(s['plan_validity_rate'])} | "
            f"{s['n_instructions_success_all_repeats']} / 120 |"
        )

    lines.extend(
        [
            "",
            "### Cross-model Task Success summary",
            "",
            "Non-Full cells show Task Success followed by Δ vs Full in percentage points.",
            "",
            "| Model | Full | Reduced capability information | No capability semantics | No parameter semantics | No schema-constrained decoding |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for res in results:
        cells = []
        for cond in PAPER_CONDITIONS:
            s = res["summaries"][cond]
            if cond == "full":
                cells.append(_pct(s["task_success_rate"]))
            else:
                e = res["effects"][cond]
                cells.append(f"{_pct(s['task_success_rate'])} ({_pp(e['delta'])})")
        lines.append(f"| {res['model_display_name']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 5 Task-family effects",
            "",
            "| Model | Condition | P1 | P2 | P3 | P4 | P5 | P6 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for res in results:
        for cond in PAPER_CONDITIONS:
            ft = res["family"][cond]
            lines.append(
                f"| {res['model_display_name']} | {CONDITION_LABELS[cond]} | "
                + " | ".join(_pct(ft[k]) for k in ["P1", "P2", "P3", "P4", "P5", "P6"])
                + " |"
            )
    lines.extend(["", "P2 / P3 pattern (observed, not assumed):", ""])
    for row in patterns["per_model"]["p2"]:
        lines.append(
            f"- P2 + no capability semantics / {row['model']}: Full {_pct(row['full'])} → "
            f"{_pct(row['no_capability_semantics'])} (Δ {_pp(row['delta'])} pp)"
        )
    for row in patterns["per_model"]["p3"]:
        lines.append(
            f"- P3 + no parameter semantics / {row['model']}: Full {_pct(row['full'])} → "
            f"{_pct(row['no_parameter_semantics'])} (Δ {_pp(row['delta'])} pp)"
        )

    lines.extend(
        [
            "",
            "## 6 Plan-validity and failure diagnostics",
            "",
            "| Model | Condition | Plan Validity | Invalid | Unparseable | Cap. sel. error | Param. error | Order error | P6 phys. dispatch |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for res in results:
        for cond in PAPER_CONDITIONS:
            d = res["diagnostics"][cond]
            s = res["summaries"][cond]
            lines.append(
                f"| {res['model_display_name']} | {CONDITION_LABELS[cond]} | {_pct(s['plan_validity_rate'])} | "
                f"{d['invalid_plans']} | {d['unparseable_plans']} | {_pct(d['capability_selection_error'])} | "
                f"{_pct(d['parameter_error'])} | {_pct(d['order_error'])} | {d['p6_physical_dispatch_errors']} |"
            )

    lines.extend(
        [
            "",
            "## 7 Safety-gate replay",
            "",
            "Offline replay of Full planner outputs. No additional LLM calls.",
            "",
            "| Model | P6 err. dispatch before | P6 err. dispatch after | Intercepted | Valid physical pass rate | False-block rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| {row['model']} | {row['p6_erroneous_physical_dispatch_before_gate']} | "
            f"{row['p6_erroneous_physical_dispatch_after_gate']} | {row['number_intercepted']} | "
            f"{_pct(row['valid_physical_output_pass_rate'])} | {_pct(row['false_block_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 8 Main findings",
            "",
            "Findings below are restricted to the measured cells. Absolute Full ranking is not the target.",
            "",
            f"- no capability semantics reduced Task Success on all four models: "
            f"{patterns['capability_semantics_decreases_all']}.",
            f"- no parameter semantics reduced Task Success on all four models: "
            f"{patterns['parameter_semantics_decreases_all']}.",
            f"- both semantic |Δ| exceeded |Δ_schema| on all four models: "
            f"{patterns['semantic_effect_larger_than_schema_all']}.",
            f"- P2 drop under no capability semantics on all four models: "
            f"{patterns['p2_capability_drop_all']}.",
            f"- P3 drop under no parameter semantics on all four models: "
            f"{patterns['p3_parameter_drop_all']}.",
            "",
        ]
    )
    for res in results:
        e = res["effects"]
        lines.append(
            f"- {res['model_display_name']}: Δ_cap={_pp(e['no_capability_semantics']['delta'])} pp, "
            f"Δ_par={_pp(e['no_parameter_semantics']['delta'])} pp, "
            f"Δ_schema={_pp(e['no_schema_constrained_decoding']['delta'])} pp."
        )

    lines.extend(["", "## 9 Deviations / warnings", ""])
    if deviations:
        for d in deviations:
            lines.append(f"- {d}")
    else:
        lines.append("- No model-download or tag substitutions were recorded.")
    lines.extend(
        [
            "- Qwen formal cells reused v1.1 runs. Original Qwen configs stored tag `qwen2.5:14b` but not digest.",
            "- `no_schema_constrained_decoding` is the paper label for the existing `no_structure` condition.",
            "- `reduced_capability_info` reuses the frozen names-only catalog condition from the Qwen v1.1 main experiment.",
            "- Reduced capability information is a coarse reference and is excluded from the three-ablation Holm family.",
            "- Transport-level Ollama retries were added for new runs only; request payloads were not changed.",
            "- Recovery, physical E2E, and speech experiments were not extended.",
            "",
        ]
    )
    (OUTPUTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_all(deviations: Optional[List[str]] = None) -> Dict[str, Any]:
    deviations = list(deviations or [])
    extra = MULTIMODEL_DIR / "deviations.json"
    if extra.exists():
        deviations.extend(json.loads(extra.read_text(encoding="utf-8")).get("deviations") or [])
    cfgs = {c["id"]: c for c in list_model_configs()}
    results = []
    for mid in MODEL_ORDER:
        dest = MULTIMODEL_DIR / mid
        if not dest.exists():
            continue
        missing = [c for c in PAPER_CONDITIONS if not (dest / c / "records.jsonl").exists() and not (dest / c / "REUSED.json").exists()]
        if missing:
            deviations.append(f"{mid} missing conditions: {missing}")
            continue
        display = cfgs.get(mid, {}).get("name", mid)
        results.append(analyze_model(mid, display))
    if not results:
        raise RuntimeError("no complete multimodel runs to analyze")
    write_csvs(results)
    write_figure_s1_csvs(results)
    qc_multimodel_cells(results)
    plot_ablation_effects(results)
    plot_task_family(results)
    gate_rows = replay_gates(results)
    patterns = pattern_checks(results)
    write_json(
        OUTPUTS_DIR / "cross_model_analysis.json",
        {
            "results": [
                {
                    "model_id": r["model_id"],
                    "model_display_name": r["model_display_name"],
                    "summaries": r["summaries"],
                    "family": r["family"],
                    "diagnostics": r["diagnostics"],
                    "effects": r["effects"],
                    "source_dirs": r["source_dirs"],
                }
                for r in results
            ],
            "patterns": patterns,
            "deviations": deviations,
        },
    )
    write_report(results, gate_rows, patterns, deviations)
    return {"n_models": len(results), "deviations": deviations}


def main() -> None:
    analyze_all()


if __name__ == "__main__":
    main()
