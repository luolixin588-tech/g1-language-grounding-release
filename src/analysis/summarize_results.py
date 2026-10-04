from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.io_utils import read_jsonl, write_json


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_records(
    records: List[Dict[str, Any]],
    run_id: str = "",
    condition: str = "",
) -> Dict[str, Any]:
    n = len(records)
    n_success = sum(1 for r in records if r.get("task_success_result", {}).get("task_success"))
    n_valid = sum(1 for r in records if r.get("validity_result", {}).get("valid"))
    lat = [float(r["total_planning_latency_ms"]) for r in records if "total_planning_latency_ms" in r]

    by_family = defaultdict(list)
    by_form = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)
        by_form[r["formulation"]].append(r)

    def rate(rows, key_path):
        if not rows:
            return None
        if key_path == "success":
            return sum(1 for x in rows if x.get("task_success_result", {}).get("task_success")) / len(rows)
        return sum(1 for x in rows if x.get("validity_result", {}).get("valid")) / len(rows)

    family_stats = {}
    for fam, rows in sorted(by_family.items()):
        family_stats[fam] = {
            "n": len(rows),
            "task_success_rate": rate(rows, "success"),
            "plan_validity_rate": rate(rows, "valid"),
            "capability_selection_accuracy": _mean(
                [x["diagnostics"]["capability_selection_accuracy"] for x in rows]
            ),
            "parameter_accuracy": _mean([x["diagnostics"]["parameter_accuracy"] for x in rows]),
            "order_accuracy": _mean([x["diagnostics"]["order_accuracy"] for x in rows]),
        }

    # all-3 independent runs success, by sample_id
    by_sample = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(bool(r.get("task_success_result", {}).get("task_success")))
    n_all3 = sum(1 for sid, flags in by_sample.items() if flags and all(flags) and len(flags) >= 3)
    n_samples = len(by_sample)

    return {
        "run_id": run_id,
        "condition": condition or (records[0]["condition"] if records else ""),
        "benchmark_version": "v1.1",
        "n_runs": n,
        "n_instructions": n_samples,
        "task_success_rate": (n_success / n) if n else None,
        "n_task_success": n_success,
        "plan_validity_rate": (n_valid / n) if n else None,
        "n_plan_valid": n_valid,
        "planning_latency_ms": {
            "median": _percentile(lat, 0.5),
            "p95": _percentile(lat, 0.95),
            "mean": _mean(lat),
        },
        "success_by_family": family_stats,
        "success_by_formulation": {
            k: {"n": len(v), "task_success_rate": rate(v, "success")}
            for k, v in sorted(by_form.items())
        },
        "n_instructions_success_all_repeats": n_all3,
        "n_instructions_with_three_repeats": sum(
            1 for flags in by_sample.values() if len(flags) >= 3
        ),
        "diagnostics": {
            "capability_selection_accuracy": _mean(
                [r["diagnostics"]["capability_selection_accuracy"] for r in records]
            ),
            "parameter_accuracy": _mean(
                [r["diagnostics"]["parameter_accuracy"] for r in records]
            ),
            "order_accuracy": _mean([r["diagnostics"]["order_accuracy"] for r in records]),
        },
    }


def summarize_path(records_path: Path, out_path: Optional[Path] = None) -> Dict[str, Any]:
    records = read_jsonl(records_path)
    summary = summarize_records(records)
    if out_path is None:
        out_path = records_path.parent / "summary.json"
    write_json(out_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summarize_path(Path(args.records_jsonl), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
