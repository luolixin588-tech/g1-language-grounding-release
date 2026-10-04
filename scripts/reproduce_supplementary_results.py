#!/usr/bin/env python3
"""Recompute supplementary cross-model, gate, recovery, and AQ18 numbers.

Does not call Ollama.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "data" / "planning_runs" / "cross_model"
QWEN = ROOT / "data" / "planning_runs" / "qwen2.5_14b"
GATE = ROOT / "data" / "gate_replay"
REC = ROOT / "data" / "recovery"

QWEN_POINTER = {
    "full": QWEN / "full" / "full_20260821T051022Z_d81f78bf",
    "no_capability_semantics": QWEN
    / "no_capability_semantics"
    / "no_capability_semantics_20260824T111546Z_368b8574",
    "no_parameter_semantics": QWEN
    / "no_parameter_semantics"
    / "no_parameter_semantics_20260824T112027Z_d8471060",
    "no_schema_constrained_decoding": QWEN / "no_structure" / "no_structure_20260821T052009Z_6bf8d7c2",
    "reduced_capability_info": QWEN
    / "reduced_capability_info"
    / "reduced_capability_info_20260821T052504Z_6e0dd3ea",
}

MODELS = ("qwen2.5_14b", "phi4_14b", "gemma3_12b", "mistral_nemo_12b")
CONDS = (
    "full",
    "reduced_capability_info",
    "no_capability_semantics",
    "no_parameter_semantics",
    "no_schema_constrained_decoding",
)
DISPLAY = {
    "qwen2.5_14b": "Qwen2.5-14B",
    "phi4_14b": "Phi-4 14B",
    "gemma3_12b": "Gemma 3 12B",
    "mistral_nemo_12b": "Mistral NeMo 12B",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ok(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("task_success_result", {}).get("task_success"))


def records_for(model: str, cond: str) -> List[Dict[str, Any]]:
    if model == "qwen2.5_14b":
        return _read_jsonl(QWEN_POINTER[cond] / "records.jsonl")
    return _read_jsonl(CROSS / model / cond / "records.jsonl")


def family_rate(records: List[Dict[str, Any]], prefix: str) -> Tuple[int, int]:
    n = s = 0
    for r in records:
        if str(r.get("family", "")).startswith(prefix):
            n += 1
            s += int(_ok(r))
    return s, n


def section_cross_model() -> None:
    print("== Cross-model Task Success and P1–P6 ==")
    for model in MODELS:
        print(f"\n{DISPLAY[model]}")
        for cond in CONDS:
            recs = records_for(model, cond)
            s = sum(int(_ok(r)) for r in recs)
            n = len(recs)
            fams = []
            for fam in ("P1", "P2", "P3", "P4", "P5", "P6"):
                fs, fn = family_rate(recs, fam)
                fams.append(f"{fam} {100 * fs / fn:5.1f}%" if fn else f"{fam}  n/a")
            print(f"  {cond:32s} {s:3d}/{n} = {100 * s / n:5.1f}%   " + "  ".join(fams))


def section_aq18() -> None:
    print("\n== AQ18 Mistral NeMo P2 ==")
    full = records_for("mistral_nemo_12b", "full")
    nop = records_for("mistral_nemo_12b", "no_parameter_semantics")
    f_s, f_n = family_rate(full, "P2")
    p_s, p_n = family_rate(nop, "P2")
    print(f"  Full P2                 {f_s}/{f_n} = {100 * f_s / f_n:.1f}%")
    print(f"  No parameter semantics  {p_s}/{p_n} = {100 * p_s / p_n:.1f}%")
    csv_path = ROOT / "figures" / "source_data" / "cross_model_task_categories.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    full_csv = next(
        r for r in rows if r["model"] == "Mistral NeMo 12B" and r["condition"] == "full"
    )
    nop_csv = next(
        r
        for r in rows
        if r["model"] == "Mistral NeMo 12B" and r["condition"] == "no_parameter_semantics"
    )
    print(f"  CSV Full P2             {float(full_csv['P2']) * 100:.1f}%")
    print(f"  CSV No-param P2         {float(nop_csv['P2']) * 100:.1f}%")
    ok = f_s == 54 and f_n == 60 and p_s == 60 and p_n == 60
    ok = ok and abs(float(full_csv["P2"]) - 0.9) < 1e-12
    ok = ok and abs(float(nop_csv["P2"]) - 1.0) < 1e-12
    print("  Old Supplementary Fig. S1 plotting source P2_param = 90.0 was incorrect.")
    print("  Corrected No-parameter-semantics P2 = 100.0%.")
    print("  AQ18 verification:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def section_gate() -> None:
    print("\n== Fig. 4 gate replay ==")
    path = GATE / "fig4_gate_replay_records.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  replay rows: {len(rows)}")
    p6_err = [
        r
        for r in rows
        if r["family"].startswith("P6") and r["gate_off_dispatch"] == "True"
    ]
    p6_blocked = [r for r in p6_err if r["gate_on_dispatch"] == "False"]
    fb = [
        r
        for r in rows
        if r["gate_off_dispatch"] == "True"
        and r["gate_on_dispatch"] == "False"
        and not r["family"].startswith("P6")
    ]
    print(f"  P6 erroneous dispatch (gate off): {len(p6_err)}")
    print(f"  P6 intercepted (gate on):         {len(p6_blocked)}")
    print(f"  false blocks (P2–P5):             {len(fb)}")
    print("  false-block instructions:")
    for key, n in Counter(
        (r["semantic_task_id"], r["formulation"], r["instruction"]) for r in fb
    ).items():
        print(f"    {key[0]} {key[1]} ×{n}: {key[2]}")
    summary = json.loads((GATE / "safety_gate_results.json").read_text(encoding="utf-8"))
    print("  safety_gate_results.json keys:", sorted(summary.keys())[:12])


def section_recovery() -> None:
    print("\n== Fig. 5 / ED Table 3 recovery ==")
    formal = _read_jsonl(REC / "formal_F1-F7_20260824T155611Z_9fde5231" / "records.jsonl")
    f6 = _read_jsonl(REC / "f6_rerun_20260825T021403Z_306314eb" / "records.jsonl")
    f7c = _read_jsonl(REC / "f7_corrected_20260930T051720Z_883abfac" / "records.jsonl")

    def by_cond(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            out[r["failure_condition"]].append(r)
        return out

    formal_c = by_cond(formal)
    print(f"  formal episodes: {len(formal)}  conditions: { {k: len(v) for k, v in formal_c.items()} }")
    pool = []
    for cond in (
        "F1_dispatch_rejected",
        "F2_execution_timeout",
        "F3_partial_completion",
        "F4_invalid_robot_state",
        "F5_infeasible_parameter",
    ):
        inj = [r for r in formal_c[cond] if r.get("injected")]
        pool.extend(inj)
        ok = sum(1 for r in inj if r.get("recovery_success"))
        print(f"  {cond:28s} applicable {len(inj):2d}  recovery {ok}/{len(inj)}")
    f6_inj = [r for r in f6 if r.get("injected")]
    pool.extend(f6_inj)
    ok6 = sum(1 for r in f6_inj if r.get("recovery_success"))
    print(f"  {'F6_second_step_failure':28s} applicable {len(f6_inj):2d}  recovery {ok6}/{len(f6_inj)}  (rerun)")
    print(f"  recoverable F1–F6 pool: {sum(1 for r in pool if r.get('recovery_success'))}/{len(pool)}")
    f7 = [r for r in formal_c["F7_unrecoverable_failure"] if r.get("injected")]
    term = sum(1 for r in f7 if r.get("correct_termination"))
    term_c = sum(1 for r in f7c if r.get("correct_termination"))
    print(f"  F7 manuscript cell:     {term_c}/{len(f7c)} correct termination  (f7_corrected)")
    print(f"  F7 superseded formal:   {term}/{len(f7)} (prompt-conflict run; not the manuscript cell)")
    print("  Keep 92 F1–F6 episodes distinct from 16 F7 episodes.")


def main() -> None:
    section_cross_model()
    section_aq18()
    section_gate()
    section_recovery()
    print("\nSupplementary recompute finished.")


if __name__ == "__main__":
    main()
