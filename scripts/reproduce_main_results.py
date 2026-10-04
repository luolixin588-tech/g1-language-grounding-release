#!/usr/bin/env python3
"""Recompute main-text Qwen2.5 planning statistics from released records.

Does not call Ollama. Compares recomputed rates to frozen summary.json files.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.summarize_results import summarize_records  # noqa: E402
from src.io_utils import read_jsonl  # noqa: E402

QWEN = ROOT / "data" / "planning_runs" / "qwen2.5_14b"
RUNS = {
    "full": QWEN / "full" / "full_20260821T051022Z_d81f78bf",
    "no_capability_semantics": QWEN
    / "no_capability_semantics"
    / "no_capability_semantics_20260824T111546Z_368b8574",
    "no_parameter_semantics": QWEN
    / "no_parameter_semantics"
    / "no_parameter_semantics_20260824T112027Z_d8471060",
    "no_structure": QWEN / "no_structure" / "no_structure_20260821T052009Z_6bf8d7c2",
    "reduced_capability_info": QWEN
    / "reduced_capability_info"
    / "reduced_capability_info_20260821T052504Z_6e0dd3ea",
}

EXPECTED = {
    "full": (340, 360, 0.9444444444444444),
    "no_capability_semantics": (204, 360, 0.5666666666666667),
    "no_parameter_semantics": (233, 360, 0.6472222222222223),
    "no_structure": (335, 360, 0.9305555555555556),
    "reduced_capability_info": (175, 360, 0.4861111111111111),
}


def _ok(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("task_success_result", {}).get("task_success"))


def family_counts(records: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        fam = str(r.get("family", ""))[:2]
        out[fam][1] += 1
        out[fam][0] += int(_ok(r))
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def main() -> None:
    print("Qwen2.5-14B formal v1.1 planning statistics (recomputed from records)")
    print()
    mismatches = 0
    for cond, path in RUNS.items():
        records = read_jsonl(path / "records.jsonl")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        recomputed = summarize_records(records, run_id=path.name, condition=cond)
        n_s = recomputed["n_task_success"]
        n = recomputed["n_runs"]
        rate = recomputed["task_success_rate"]
        exp_s, exp_n, exp_rate = EXPECTED[cond]
        match = n_s == exp_s and n == exp_n and abs(rate - exp_rate) < 1e-12
        if (
            abs(recomputed["task_success_rate"] - summary["task_success_rate"]) > 1e-12
            or recomputed["n_task_success"] != summary["n_task_success"]
        ):
            match = False
        mismatches += int(not match)
        print(
            f"{cond:28s}  {n_s}/{n} = {100 * rate:5.1f}%   "
            f"validity={100 * recomputed['plan_validity_rate']:5.1f}%   "
            f"all3={recomputed['n_instructions_success_all_repeats']}/120   "
            f"{'OK' if match else 'MISMATCH'}"
        )
        fam = family_counts(records)
        print(
            "   families "
            + "  ".join(f"{k} {a}/{b}" for k, (a, b) in fam.items())
        )
    print()
    if mismatches:
        print(f"FAILED: {mismatches} condition(s) did not match frozen summaries.")
        sys.exit(1)
    print("All five formal Qwen conditions match frozen summary.json files.")
    print("Headline Full Task Success: 340/360 = 94.4%.")


if __name__ == "__main__":
    main()
