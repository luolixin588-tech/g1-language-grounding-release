#!/usr/bin/env python3
"""Recompute Fig.3a inference with semantic-task clustered units.

Does not rerun planners, rewrite records, or overwrite the original
instruction-clustered reports. Point estimates stay the 360-run rates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analysis.statistics import (
    clustered_bootstrap,
    holm_correction,
    mcnemar_exact,
    paired_mcnemar_all3,
)
from src.io_utils import read_jsonl, write_json
from src.paths import REPORTS_DIR, RUNS_DIR

OFFICIAL_RUNS = {
    "full": RUNS_DIR / "full" / "full_20260821T051022Z_d81f78bf",
    "no_capability_semantics": (
        RUNS_DIR / "no_capability_semantics" / "no_capability_semantics_20260824T111546Z_368b8574"
    ),
    "no_parameter_semantics": (
        RUNS_DIR / "no_parameter_semantics" / "no_parameter_semantics_20260824T112027Z_d8471060"
    ),
    "no_json_schema": RUNS_DIR / "no_structure" / "no_structure_20260821T052009Z_6bf8d7c2",
    "reduced_capability_info": (
        RUNS_DIR / "reduced_capability_info" / "reduced_capability_info_20260821T052504Z_6e0dd3ea"
    ),
}

COMPARISONS = (
    ("no_capability_semantics", "No capability semantics"),
    ("no_parameter_semantics", "No parameter semantics"),
    ("no_json_schema", "No JSON Schema"),
)
REFERENCE = ("reduced_capability_info", "Reduced capability information")

# Instruction-clustered CIs / McNemar from the frozen Fig.3 reports.
# Recomputed below and asserted equal to these published values.
PUBLISHED_OLD = {
    "no_capability_semantics": {
        "delta_pp": -37.8,
        "ci95_pp": [-46.9, -28.6],
        "mcnemar_raw_p": 4.355626970209414e-12,
        "holm_p": 1.3066880910628242e-11,
    },
    "no_parameter_semantics": {
        "delta_pp": -29.7,
        "ci95_pp": [-38.9, -20.6],
        "mcnemar_raw_p": 5.631591193377972e-09,
        "holm_p": 1.1263182386755943e-08,
    },
    "no_json_schema": {
        "delta_pp": -1.4,
        "ci95_pp": [-3.6, 0.0],
        "mcnemar_raw_p": 0.5,
        "holm_p": 0.5,
    },
    "reduced_capability_info": {
        "delta_pp": -45.8,
        "ci95_pp": [-55.0, -36.7],
        "mcnemar_raw_p": 8.049116928532385e-16,
        "holm_p": None,
        "note": "Old Holm family was A1/A2 only; Reduced is reference-only here.",
    },
}

N_BOOT = 10000
SEED = 0
EXPECTED_N_RUNS = 360
EXPECTED_N_TASKS = 60
EXPECTED_N_INSTRUCTIONS = 120
EXPECTED_RUNS_PER_TASK = 6


def _success(row: Dict[str, Any]) -> bool:
    return bool(row.get("task_success_result", {}).get("task_success"))


def _rate(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("empty rate")
    return sum(1 for r in rows if _success(r)) / len(rows)


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    import math

    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _index_by_task(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_task[r["semantic_task_id"]].append(r)
    return dict(by_task)


def completeness_check(label: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[str] = []
    n = len(records)
    if n != EXPECTED_N_RUNS:
        issues.append(f"{label}: n_runs={n} (expected {EXPECTED_N_RUNS})")

    missing_fields = [
        i
        for i, r in enumerate(records)
        if not r.get("semantic_task_id") or not r.get("sample_id") or not r.get("formulation")
    ]
    if missing_fields:
        issues.append(f"{label}: missing id/formulation on {len(missing_fields)} rows")

    tasks = sorted({r["semantic_task_id"] for r in records})
    if len(tasks) != EXPECTED_N_TASKS:
        issues.append(f"{label}: n_semantic_task_id={len(tasks)} (expected {EXPECTED_N_TASKS})")

    samples = sorted({r["sample_id"] for r in records})
    if len(samples) != EXPECTED_N_INSTRUCTIONS:
        issues.append(f"{label}: n_sample_id={len(samples)} (expected {EXPECTED_N_INSTRUCTIONS})")

    forms = Counter(r.get("formulation") for r in records)
    if forms.get("canonical") != 180 or forms.get("paraphrase") != 180:
        issues.append(f"{label}: formulation counts {dict(forms)}")

    keys = [(r["semantic_task_id"], r["formulation"], int(r["repeat_index"])) for r in records]
    key_counts = Counter(keys)
    dups = {k: c for k, c in key_counts.items() if c > 1}
    if dups:
        issues.append(f"{label}: duplicate (task, formulation, repeat) keys: {len(dups)}")

    per_task: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        per_task[r["semantic_task_id"]][r["formulation"]].append(int(r["repeat_index"]))

    incomplete_tasks = []
    for tid in tasks:
        forms_here = per_task[tid]
        ok = (
            set(forms_here) == {"canonical", "paraphrase"}
            and sorted(forms_here["canonical"]) == [0, 1, 2]
            and sorted(forms_here["paraphrase"]) == [0, 1, 2]
            and len(forms_here["canonical"]) == 3
            and len(forms_here["paraphrase"]) == 3
        )
        if not ok:
            incomplete_tasks.append(
                {
                    "semantic_task_id": tid,
                    "canonical_repeats": sorted(forms_here.get("canonical") or []),
                    "paraphrase_repeats": sorted(forms_here.get("paraphrase") or []),
                }
            )
    if incomplete_tasks:
        issues.append(f"{label}: {len(incomplete_tasks)} tasks missing canonical/paraphrase×3")

    n_success = sum(1 for r in records if _success(r))
    return {
        "label": label,
        "n_runs": n,
        "n_success": n_success,
        "task_success_rate": n_success / n if n else None,
        "n_semantic_task_id": len(tasks),
        "n_sample_id": len(samples),
        "formulation_counts": dict(forms),
        "n_duplicate_keys": len(dups),
        "n_incomplete_tasks": len(incomplete_tasks),
        "incomplete_tasks": incomplete_tasks[:10],
        "issues": issues,
        "ok": not issues,
    }


def semantic_task_all6_success(records: List[Dict[str, Any]]) -> Dict[str, bool]:
    by_task = _index_by_task(records)
    out: Dict[str, bool] = {}
    for tid, rows in by_task.items():
        if len(rows) != EXPECTED_RUNS_PER_TASK:
            raise ValueError(f"{tid}: expected {EXPECTED_RUNS_PER_TASK} runs, got {len(rows)}")
        by_form: Dict[str, List[bool]] = defaultdict(list)
        for r in rows:
            by_form[r["formulation"]].append(_success(r))
        canon = by_form.get("canonical") or []
        para = by_form.get("paraphrase") or []
        out[tid] = (
            len(canon) == 3
            and len(para) == 3
            and all(canon)
            and all(para)
        )
    return out


def paired_mcnemar_semantic_task_all6(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    a = semantic_task_all6_success(records_a)
    b = semantic_task_all6_success(records_b)
    ids = sorted(set(a) & set(b))
    n11 = n10 = n01 = n00 = 0
    for tid in ids:
        sa, sb = a[tid], b[tid]
        if sa and sb:
            n11 += 1
        elif sa and not sb:
            n10 += 1
        elif (not sa) and sb:
            n01 += 1
        else:
            n00 += 1
    test = mcnemar_exact(n10, n01)
    return {
        "aggregation": (
            "semantic-task success = 1 iff canonical 3/3 AND paraphrase 3/3 "
            "(6/6 runs Task Success)"
        ),
        "n_semantic_tasks": len(ids),
        "n_both_success": n11,
        "n_a_only": n10,
        "n_b_only": n01,
        "n_both_fail": n00,
        "mcnemar": test,
    }


def semantic_task_clustered_bootstrap(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Paired cluster bootstrap over semantic_task_id (canonical+paraphrase × 3)."""
    import random

    idx_a = _index_by_task(records_a)
    idx_b = _index_by_task(records_b)
    ids = sorted(set(idx_a) & set(idx_b))
    if len(ids) != EXPECTED_N_TASKS:
        raise ValueError(f"matched semantic tasks={len(ids)} (expected {EXPECTED_N_TASKS})")
    for tid in ids:
        if len(idx_a[tid]) != EXPECTED_RUNS_PER_TASK or len(idx_b[tid]) != EXPECTED_RUNS_PER_TASK:
            raise ValueError(
                f"{tid}: cluster sizes Full={len(idx_a[tid])} ablation={len(idx_b[tid])}"
            )

    all_a = [r for tid in ids for r in idx_a[tid]]
    all_b = [r for tid in ids for r in idx_b[tid]]
    ra = _rate(all_a)
    rb = _rate(all_b)
    observed = {"a": ra, "b": rb, "diff_b_minus_a": rb - ra}

    rng = random.Random(seed)
    diffs: List[float] = []
    n = len(ids)
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        rows_a = [r for tid in sample for r in idx_a[tid]]
        rows_b = [r for tid in sample for r in idx_b[tid]]
        diffs.append(_rate(rows_b) - _rate(rows_a))

    diffs_sorted = sorted(diffs)
    return {
        "cluster": "semantic_task_id",
        "n_clusters": n,
        "runs_per_cluster": EXPECTED_RUNS_PER_TASK,
        "n_boot": n_boot,
        "seed": seed,
        "observed": observed,
        "diff_ci95": [_percentile(diffs_sorted, 0.025), _percentile(diffs_sorted, 0.975)],
    }


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt_pp(x: float) -> str:
    return f"{100.0 * x:+.1f}"


def _fmt_ci_pp(pair: Sequence[float]) -> str:
    return f"[{100.0 * pair[0]:+.1f}, {100.0 * pair[1]:+.1f}]"


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "n/a (reference; not in Holm family)"
    if p == 0:
        return "0"
    if p >= 0.001:
        return f"{p:.4g}"
    return f"{p:.4g}"


def _round_pp1(x: float) -> float:
    return round(100.0 * x, 1)


def main() -> None:
    records = {k: read_jsonl(p / "records.jsonl") for k, p in OFFICIAL_RUNS.items()}
    qc = {k: completeness_check(k, rows) for k, rows in records.items()}
    qc_issues = [iss for q in qc.values() for iss in q["issues"]]
    if qc_issues:
        raise SystemExit("completeness check failed:\n" + "\n".join(qc_issues))

    full = records["full"]
    full_rate = _rate(full)

    old_instruction = {}
    for key, _label in COMPARISONS + (REFERENCE,):
        boot = clustered_bootstrap(full, records[key], ("success",), n_boot=N_BOOT, seed=SEED)
        mc = paired_mcnemar_all3(full, records[key])
        old_instruction[key] = {"bootstrap": boot, "mcnemar": mc}
        pub = PUBLISHED_OLD[key]
        obs_pp = _round_pp1(boot["fields"]["success"]["observed"]["diff_b_minus_a"])
        ci = [_round_pp1(x) for x in boot["fields"]["success"]["diff_ci95"]]
        if obs_pp != pub["delta_pp"] or ci != pub["ci95_pp"]:
            raise SystemExit(
                f"{key}: recomputed instruction-clustered stats {obs_pp} {ci} "
                f"do not match published {pub['delta_pp']} {pub['ci95_pp']}"
            )
        if abs(mc["mcnemar"]["p_value"] - pub["mcnemar_raw_p"]) > 1e-18:
            raise SystemExit(f"{key}: McNemar p mismatch")

    old_holm = holm_correction(
        [
            (key, old_instruction[key]["mcnemar"]["mcnemar"]["p_value"])
            for key, _ in COMPARISONS
        ]
    )
    old_holm_by = {x["name"]: x for x in old_holm}

    new_stats = {}
    for key, _label in COMPARISONS + (REFERENCE,):
        boot = semantic_task_clustered_bootstrap(full, records[key])
        mc = paired_mcnemar_semantic_task_all6(full, records[key])
        new_stats[key] = {"bootstrap": boot, "mcnemar": mc}

    new_holm = holm_correction(
        [(key, new_stats[key]["mcnemar"]["mcnemar"]["p_value"]) for key, _ in COMPARISONS]
    )
    new_holm_by = {x["name"]: x for x in new_holm}

    rows = []
    mcnemar_rows = []
    for key, label in COMPARISONS + (REFERENCE,):
        ab_rate = _rate(records[key])
        delta = ab_rate - full_rate
        old_ci = PUBLISHED_OLD[key]["ci95_pp"]
        new_ci = [100.0 * x for x in new_stats[key]["bootstrap"]["diff_ci95"]]
        raw_p = new_stats[key]["mcnemar"]["mcnemar"]["p_value"]
        holm_p = None if key == REFERENCE[0] else new_holm_by[key]["p_holm"]
        old_sig = PUBLISHED_OLD[key]["holm_p"] is not None and PUBLISHED_OLD[key]["holm_p"] < 0.05
        if key == REFERENCE[0]:
            # Old Reduced Holm was vs A1 in a 2-test family; still significant.
            old_sig = True
        new_sig = None if holm_p is None else holm_p < 0.05
        rows.append(
            {
                "comparison": f"Full vs {label}",
                "key": key,
                "reference_only": key == REFERENCE[0],
                "full_rate": full_rate,
                "ablation_rate": ab_rate,
                "delta_pp": 100.0 * delta,
                "old_instruction_clustered_95ci_pp": old_ci,
                "new_semantic_task_clustered_95ci_pp": new_ci,
                "ci_width_old_pp": old_ci[1] - old_ci[0],
                "ci_width_new_pp": new_ci[1] - new_ci[0],
                "mcnemar_raw_p": raw_p,
                "holm_adjusted_p": holm_p,
                "old_mcnemar_raw_p": PUBLISHED_OLD[key]["mcnemar_raw_p"],
                "old_holm_adjusted_p": PUBLISHED_OLD[key]["holm_p"],
                "old_significant_0.05": old_sig if key != REFERENCE[0] else None,
                "new_significant_0.05": new_sig,
                "significance_changed": (
                    None
                    if key == REFERENCE[0]
                    else bool(old_sig) != bool(new_sig)
                ),
            }
        )
        mc = new_stats[key]["mcnemar"]
        mcnemar_rows.append(
            {
                "comparison": f"Full vs {label}",
                "key": key,
                "both_success": mc["n_both_success"],
                "full_only": mc["n_a_only"],
                "ablation_only": mc["n_b_only"],
                "both_fail": mc["n_both_fail"],
                "n_semantic_tasks": mc["n_semantic_tasks"],
            }
        )

    payload = {
        "analysis": "fig3_semantic_task_clustered_stats",
        "note": (
            "Point estimates are unchanged 360-run Task Success rates. "
            "Only the inferential unit changes from instruction/sample_id (N=120, 3 runs) "
            "to semantic_task_id (N=60, 6 runs: canonical×3 + paraphrase×3)."
        ),
        "official_runs": {k: str(p) for k, p in OFFICIAL_RUNS.items()},
        "unused_candidate_runs": [
            "runs/full/full_20260907T030550Z_8cfddfbd (task success 80.0%, not the paper Full)",
            "runs/full/full_20260907T025510Z_929f83bc",
            "runs/no_capability_semantics/no_capability_semantics_20260907T031047Z_f1f5bca2",
            "runs/no_parameter_semantics/no_parameter_semantics_20260907T031313Z_41c305ff",
            "runs/no_schema_constrained_decoding/*20260907*",
            "runs/multimodel_v1.1/* (other models; Qwen Full is a reuse of the official Full run)",
            "runs/archive/*",
            "runs/smoke/*",
        ],
        "completeness": qc,
        "old_method": {
            "cluster": "instruction / sample_id",
            "n_clusters": 120,
            "runs_per_cluster": 3,
            "bootstrap": "resample 120 instructions with replacement; 3 repeats stay together; paired Full/ablation; B=10000; seed=0; percentile 95% CI",
            "mcnemar": "instruction success = 1 iff all 3 repeats succeed; exact two-sided McNemar, p=0.5",
            "holm_family": "Fig.3a primary: Full vs no_capability_semantics, no_parameter_semantics, no_json_schema (3 tests). Reduced was previously Holm-corrected only with A1 in reports/ablation_summary.md.",
        },
        "new_method": {
            "cluster": "semantic_task_id",
            "n_clusters": 60,
            "runs_per_cluster": 6,
            "bootstrap": "resample 60 semantic tasks with replacement; each draw keeps canonical×3 and paraphrase×3; paired Full/ablation; B=10000; seed=0; percentile 95% CI",
            "mcnemar": "semantic-task success = 1 iff canonical 3/3 AND paraphrase 3/3 (6/6); exact two-sided McNemar, p=0.5",
            "holm_family": "Full vs no_capability_semantics, no_parameter_semantics, no_json_schema. Reduced is reference-only and is not Holm-corrected.",
        },
        "point_estimates": {
            k: {
                "n_runs": len(records[k]),
                "n_success": sum(1 for r in records[k] if _success(r)),
                "task_success_rate": _rate(records[k]),
            }
            for k in OFFICIAL_RUNS
        },
        "comparisons": rows,
        "mcnemar_2x2_semantic_task": mcnemar_rows,
        "new_mcnemar_detail": {k: v["mcnemar"] for k, v in new_stats.items()},
        "new_bootstrap_detail": {k: v["bootstrap"] for k, v in new_stats.items()},
        "old_instruction_mcnemar_recomputed": {
            k: v["mcnemar"] for k, v in old_instruction.items()
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "fig3_semantic_task_clustered_stats.json"
    write_json(json_path, payload)

    md = [
        "# Fig.3a semantic-task-clustered statistics",
        "",
        "Point estimates are the original 360-run Task Success rates. "
        "This file only updates clustered bootstrap CIs and McNemar units.",
        "",
        "## Official runs (paper Fig.3a)",
        "",
    ]
    for k, p in OFFICIAL_RUNS.items():
        md.append(f"- `{k}`: `{p}`")
    md += [
        "",
        "Later 20260907 runs, multimodel non-Qwen runs, archive, and smoke runs were **not** used.",
        "",
        "## Completeness",
        "",
        "Each official condition: 360 runs, 60 `semantic_task_id`, canonical+paraphrase, 3 repeats each, no missing/duplicate keys.",
        "",
        "## Method change",
        "",
        "| | Old | New |",
        "|---|---|---|",
        "| Cluster | instruction / `sample_id` | `semantic_task_id` |",
        "| N | 120 | 60 |",
        "| Runs kept together | 3 repeats of one instruction | canonical×3 + paraphrase×3 (6 runs) |",
        "| Bootstrap | paired resample, B=10000, seed=0, percentile 95% CI | same, on 60 tasks |",
        "| McNemar unit | instruction all-3 success | semantic task 6/6 success |",
        "| Holm family | 3 Fig.3a ablations | same 3; Reduced excluded |",
        "",
        "## Comparison table",
        "",
        "| Comparison | Full rate | Ablation rate | Delta pp | Old 95% CI | New semantic-task-clustered 95% CI | McNemar raw p | Holm-adjusted p |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        holm = _fmt_p(row["holm_adjusted_p"])
        old_ci = f"[{row['old_instruction_clustered_95ci_pp'][0]:+.1f}, {row['old_instruction_clustered_95ci_pp'][1]:+.1f}]"
        new_ci = f"[{row['new_semantic_task_clustered_95ci_pp'][0]:+.1f}, {row['new_semantic_task_clustered_95ci_pp'][1]:+.1f}]"
        md.append(
            f"| {row['comparison']} | {_fmt_pct(row['full_rate'])} | {_fmt_pct(row['ablation_rate'])} | "
            f"{row['delta_pp']:+.1f} | {old_ci} | {new_ci} | "
            f"{_fmt_p(row['mcnemar_raw_p'])} | {holm} |"
        )
    md += [
        "",
        "Reduced capability information is a reference comparison and is **not** included in the Holm family.",
        "",
        "## Semantic-task-level McNemar 2×2",
        "",
        "Success = canonical 3/3 **and** paraphrase 3/3 (6/6 runs).",
        "",
        "| Comparison | Both success | Full only | Ablation only | Both fail |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in mcnemar_rows:
        md.append(
            f"| {row['comparison']} | {row['both_success']} | {row['full_only']} | "
            f"{row['ablation_only']} | {row['both_fail']} |"
        )
    md += [
        "",
        "## Significance",
        "",
    ]
    for row in rows:
        if row["reference_only"]:
            md.append(
                f"- {row['comparison']}: reference only. CI "
                f"{row['old_instruction_clustered_95ci_pp']} → "
                f"[{row['new_semantic_task_clustered_95ci_pp'][0]:+.1f}, "
                f"{row['new_semantic_task_clustered_95ci_pp'][1]:+.1f}]. "
                f"McNemar raw p={_fmt_p(row['mcnemar_raw_p'])} (not Holm-adjusted)."
            )
            continue
        changed = "changed" if row["significance_changed"] else "unchanged"
        md.append(
            f"- {row['comparison']}: Holm significance {changed} "
            f"(old p_holm={_fmt_p(row['old_holm_adjusted_p'])}, "
            f"new p_holm={_fmt_p(row['holm_adjusted_p'])}). "
            f"CI width {row['ci_width_old_pp']:.1f} pp → {row['ci_width_new_pp']:.1f} pp."
        )
    md += [
        "",
        "## Files",
        "",
        "- Script: `scripts/analyze_fig3_semantic_task_clustered.py`",
        "- JSON: `reports/fig3_semantic_task_clustered_stats.json`",
        "- This note: `reports/fig3_semantic_task_clustered_stats.md`",
        "",
        "Original instruction-clustered reports were not overwritten.",
        "",
    ]
    md_path = REPORTS_DIR / "fig3_semantic_task_clustered_stats.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
