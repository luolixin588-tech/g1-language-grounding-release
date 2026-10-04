#!/usr/bin/env python3
"""Cluster-aware verification of the frozen Fig. 4 held-plan gate replay.

Does not alter src/safety_gate.py or configs/safety_gate.json.
Does not overwrite reports/safety_gate_results.json.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from src.analysis.statistics import mcnemar_exact
from src.io_utils import read_jsonl, write_json
from src.paths import INSTRUCTIONS_PATH, REPORTS_DIR, RUNS_DIR
from src.safety_gate import (
    evaluate_safety_gate,
    load_safety_gate_config,
    physical_dispatch_after_gate,
    plan_has_physical_steps,
)

FULL_RUN = RUNS_DIR / "full" / "full_20260821T051022Z_d81f78bf"
N_BOOT = 10000
SEED = 0
PHYSICAL = {"arm_action", "locomotion", "state_control"}


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _is_p6(family: str) -> bool:
    return str(family).startswith("P6_")


def _gold_requires_physical(task: Dict[str, Any]) -> bool:
    return any(s.get("capability") in PHYSICAL for s in (task.get("required_steps") or []))


def _rate(flags: Sequence[bool]) -> float:
    if not flags:
        raise ValueError("empty rate")
    return sum(1 for x in flags if x) / len(flags)


def official_run_level_mcnemar(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproduce scripts/run_safety_gate_evaluation.py:_mcnemar_erroneous."""
    b = sum(1 for r in rows if r["gate_off_dispatch"] and r["gate_on_dispatch"])
    c = sum(1 for r in rows if r["gate_off_dispatch"] and not r["gate_on_dispatch"])
    on_only = sum(1 for r in rows if (not r["gate_off_dispatch"]) and r["gate_on_dispatch"])
    both_safe = sum(1 for r in rows if (not r["gate_off_dispatch"]) and (not r["gate_on_dispatch"]))
    if b + c == 0:
        p_official = None
    else:
        n = b + c
        obs = min(b, c)
        p = sum(math.comb(n, k) * (0.5**n) for k in range(0, obs + 1))
        p_official = min(1.0, 2.0 * p)
    library = mcnemar_exact(c, b)
    return {
        "n_runs": len(rows),
        "both_still_dispatch_b": b,
        "gate_off_only_blocked_c": c,
        "gate_on_only": on_only,
        "both_no_dispatch": both_safe,
        "n_discordant": b + c + on_only,
        "official_exact_two_sided_p": p_official,
        "library_mcnemar_exact": library,
        "note": (
            "Official Fig.4 script counts discordant P6 pairs among originally "
            "erroneous runs: b=still dispatch after gate, c=blocked by gate. "
            "p = 2 * Binom(n=b+c, p=0.5) tail at k<=min(b,c)."
        ),
    }


def clustered_paired_reduction(
    rows: List[Dict[str, Any]],
    cluster_key: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r[cluster_key]].append(r)
    ids = sorted(by)
    off_obs = _rate([r["gate_off_dispatch"] for r in rows])
    on_obs = _rate([r["gate_on_dispatch"] for r in rows])
    reduction_obs = off_obs - on_obs

    rng = random.Random(seed)
    n = len(ids)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        drawn = [r for cid in sample for r in by[cid]]
        boots.append(
            _rate([r["gate_off_dispatch"] for r in drawn])
            - _rate([r["gate_on_dispatch"] for r in drawn])
        )
    boots_s = sorted(boots)
    return {
        "cluster": cluster_key,
        "n_clusters": n,
        "runs_per_cluster": {cid: len(by[cid]) for cid in ids},
        "n_boot": n_boot,
        "seed": seed,
        "observed_gate_off_rate": off_obs,
        "observed_gate_on_rate": on_obs,
        "observed_reduction_off_minus_on": reduction_obs,
        "reduction_ci95": [_percentile(boots_s, 0.025), _percentile(boots_s, 0.975)],
    }


def semantic_task_sign_flip(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["semantic_task_id"]].append(r)
    ids = sorted(by)
    diffs: List[float] = []
    detail = []
    for tid in ids:
        group = by[tid]
        off = _rate([r["gate_off_dispatch"] for r in group])
        on = _rate([r["gate_on_dispatch"] for r in group])
        d = off - on
        diffs.append(d)
        detail.append(
            {
                "semantic_task_id": tid,
                "n_runs": len(group),
                "gate_off_erroneous": sum(1 for r in group if r["gate_off_dispatch"]),
                "gate_on_erroneous": sum(1 for r in group if r["gate_on_dispatch"]),
                "difference_off_minus_on": d,
            }
        )
    observed = sum(diffs)
    n = len(diffs)
    n_extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        t = sum(s * d for s, d in zip(signs, diffs))
        if abs(t) + 1e-15 >= abs(observed):
            n_extreme += 1
    n_assignments = 2**n
    nonzero = [x for x in detail if abs(x["difference_off_minus_on"]) > 0]
    return {
        "n_semantic_tasks": n,
        "n_assignments": n_assignments,
        "observed_sum_of_differences": observed,
        "n_as_or_more_extreme": n_extreme,
        "exact_two_sided_p": n_extreme / n_assignments,
        "n_informative_nonzero_clusters": len(nonzero),
        "informative_tasks": nonzero,
        "all_task_differences": detail,
    }


def build_replay_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = load_safety_gate_config()
    if cfg.get("version") != "v1.0":
        raise SystemExit(f"expected safety_gate.json v1.0, got {cfg.get('version')}")
    tasks = {t["sample_id"]: t for t in read_jsonl(INSTRUCTIONS_PATH)}
    records = read_jsonl(FULL_RUN / "records.jsonl")
    rows: List[Dict[str, Any]] = []
    for rec in records:
        plan = rec.get("normalized_semantic_plan")
        gate = evaluate_safety_gate(rec["instruction"], plan)
        off = plan_has_physical_steps(plan)
        on = physical_dispatch_after_gate(plan, gate)
        task = tasks[rec["sample_id"]]
        rows.append(
            {
                "semantic_task_id": rec["semantic_task_id"],
                "instruction_id": rec["sample_id"],
                "formulation": rec["formulation"],
                "repeat_id": int(rec["repeat_index"]),
                "family": rec["family"],
                "instruction": rec["instruction"],
                "gate_off_dispatch": bool(off),
                "gate_on_dispatch": bool(on),
                "block_reason": gate.get("reason") if gate.get("decision") == "block" else "",
                "gate_decision": gate.get("decision"),
                "gold_requires_physical": _gold_requires_physical(task),
            }
        )
    return rows, cfg


def main() -> None:
    rows, cfg = build_replay_rows()
    p6 = [r for r in rows if _is_p6(r["family"])]
    if len(p6) != 60:
        raise SystemExit(f"expected 60 P6 rows, got {len(p6)}")
    n_instr = len({r["instruction_id"] for r in p6})
    n_task = len({r["semantic_task_id"] for r in p6})
    if n_instr != 20 or n_task != 10:
        raise SystemExit(f"P6 clusters unexpected: instructions={n_instr} tasks={n_task}")

    mcnemar = official_run_level_mcnemar(p6)
    boot_instr = clustered_paired_reduction(p6, "instruction_id")
    boot_task = clustered_paired_reduction(p6, "semantic_task_id")
    signflip = semantic_task_sign_flip(p6)

    control = [
        r
        for r in rows
        if (not _is_p6(r["family"]))
        and r["gold_requires_physical"]
        and r["gate_off_dispatch"]
    ]
    false_blocks = [r for r in control if r["gate_decision"] == "block"]
    fb_keys = sorted({(r["instruction_id"], r["repeat_id"]) for r in false_blocks})
    expected = {(f"P3_05_paraphrase", i) for i in range(3)} | {
        (f"P3_06_paraphrase", i) for i in range(3)
    }
    fb_ok = set(fb_keys) == expected
    payload = {
        "source_run": str(FULL_RUN),
        "safety_gate": {
            "module": "src.safety_gate",
            "config": "configs/safety_gate.json",
            "config_version": cfg.get("version"),
            "rules_altered": False,
        },
        "p6": {
            "n_runs": len(p6),
            "n_instruction_clusters": n_instr,
            "n_semantic_task_clusters": n_task,
            "gate_off_erroneous": sum(1 for r in p6 if r["gate_off_dispatch"]),
            "gate_on_erroneous": sum(1 for r in p6 if r["gate_on_dispatch"]),
            "mcnemar_run_level": mcnemar,
            "instruction_clustered_bootstrap": {
                k: v
                for k, v in boot_instr.items()
                if k != "runs_per_cluster"
            },
            "semantic_task_clustered_bootstrap": {
                k: v
                for k, v in boot_task.items()
                if k != "runs_per_cluster"
            },
            "semantic_task_sign_flip": signflip,
        },
        "control_p2_p5": {
            "valid_physical_outputs": len(control),
            "false_blocks": len(false_blocks),
            "false_block_matches_p3_05_p3_06_paraphrase_repeats": fb_ok,
            "false_block_instruction_ids": sorted({r["instruction_id"] for r in false_blocks}),
            "false_block_semantic_task_ids": sorted({r["semantic_task_id"] for r in false_blocks}),
            "n_distinct_instruction_clusters": len({r["instruction_id"] for r in false_blocks}),
            "n_distinct_semantic_task_clusters": len({r["semantic_task_id"] for r in false_blocks}),
            "false_block_rows": [
                {
                    "instruction_id": r["instruction_id"],
                    "semantic_task_id": r["semantic_task_id"],
                    "repeat_id": r["repeat_id"],
                    "instruction": r["instruction"],
                    "block_reason": r["block_reason"],
                }
                for r in sorted(false_blocks, key=lambda x: (x["instruction_id"], x["repeat_id"]))
            ],
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "fig4_clustered_stats.json"
    write_json(json_path, payload)

    csv_path = REPORTS_DIR / "fig4_gate_replay_records.csv"
    fieldnames = [
        "semantic_task_id",
        "instruction_id",
        "formulation",
        "repeat_id",
        "family",
        "instruction",
        "gate_off_dispatch",
        "gate_on_dispatch",
        "block_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    def pct(x: float) -> str:
        return f"{100.0 * x:.1f}%"

    def ci(pair: Sequence[float]) -> str:
        return f"[{100.0 * pair[0]:.1f}%, {100.0 * pair[1]:.1f}%]"

    md = [
        "# Fig. 4 cluster-aware gate-replay statistics",
        "",
        f"Source: `{FULL_RUN}`",
        "Gate: `src/safety_gate.py` + `configs/safety_gate.json` v1.0 (not altered).",
        "",
        "## 1. Run-level exact McNemar (60 P6 runs)",
        "",
        f"- Both still dispatch (b): {mcnemar['both_still_dispatch_b']}",
        f"- Gate-off only / blocked (c): {mcnemar['gate_off_only_blocked_c']}",
        f"- Gate-on only: {mcnemar['gate_on_only']}",
        f"- Both no dispatch: {mcnemar['both_no_dispatch']}",
        f"- Official exact two-sided p: {mcnemar['official_exact_two_sided_p']:.6g}",
        "",
        "## 2–3. Paired reduction (gate-off − gate-on), B=10000, seed=0",
        "",
        f"- Observed: {pct(boot_instr['observed_gate_off_rate'])} → {pct(boot_instr['observed_gate_on_rate'])} "
        f"(Δ = {pct(boot_instr['observed_reduction_off_minus_on'])})",
        f"- Instruction-clustered (N=20) 95% CI: {ci(boot_instr['reduction_ci95'])}",
        f"- Semantic-task-clustered (N=10) 95% CI: {ci(boot_task['reduction_ci95'])}",
        "",
        "## 4. Semantic-task sign-flip test (2^10 assignments)",
        "",
        f"- Informative/nonzero tasks: {signflip['n_informative_nonzero_clusters']} / 10",
        f"- Exact two-sided p: {signflip['exact_two_sided_p']:.6g}",
        "",
        "## 5. False blocks among 240 P2–P5 valid physical outputs",
        "",
        f"- False blocks: {len(false_blocks)}",
        f"- Exactly P3_05 paraphrase ×3 + P3_06 paraphrase ×3: {fb_ok}",
        f"- Distinct instruction clusters: {payload['control_p2_p5']['n_distinct_instruction_clusters']}",
        f"- Distinct semantic-task clusters: {payload['control_p2_p5']['n_distinct_semantic_task_clusters']}",
        "",
        "## Files",
        "",
        f"- `{json_path.name}`",
        f"- `{csv_path.name}`",
        "",
    ]
    md_path = REPORTS_DIR / "fig4_clustered_stats.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if not fb_ok:
        raise SystemExit("false-block identity check failed")


if __name__ == "__main__":
    main()
