"""Instruction-clustered statistics for ablation comparisons.

The sampling unit is an instruction (sample_id). The three repeated runs of
one instruction are always kept together. 360 planning runs are not treated
as 360 independent task specifications.

Pre-declared McNemar aggregation:
    instruction-level Task Success = 1 iff all three repeats succeed.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def instruction_all3_success(records: List[Dict[str, Any]]) -> Dict[str, bool]:
    by_sample: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(bool(r.get("task_success_result", {}).get("task_success")))
    return {sid: (bool(flags) and all(flags) and len(flags) >= 3) for sid, flags in by_sample.items()}


def _index_by_sample(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(r)
    return dict(by_sample)


def _rate(rows: Sequence[Dict[str, Any]], field: str) -> Optional[float]:
    if not rows:
        return None
    if field == "success":
        return sum(1 for r in rows if r.get("task_success_result", {}).get("task_success")) / len(rows)
    if field == "valid":
        return sum(1 for r in rows if r.get("validity_result", {}).get("valid")) / len(rows)
    if field == "latency_median":
        xs = sorted(float(r["total_planning_latency_ms"]) for r in rows)
        return _percentile(xs, 0.5)
    if field.startswith("diag."):
        key = field.split(".", 1)[1]
        vals = [float(r["diagnostics"][key]) for r in rows]
        return sum(vals) / len(vals)
    raise ValueError(field)


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


def clustered_bootstrap(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    fields: Sequence[str],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster bootstrap over matched instructions.

    records_a is the reference (Full); records_b is the ablation.
    """
    idx_a = _index_by_sample(records_a)
    idx_b = _index_by_sample(records_b)
    ids = sorted(set(idx_a) & set(idx_b))
    if not ids:
        raise ValueError("no matched instructions")

    observed = {}
    all_a = [r for sid in ids for r in idx_a[sid]]
    all_b = [r for sid in ids for r in idx_b[sid]]
    for field in fields:
        ra = _rate(all_a, field)
        rb = _rate(all_b, field)
        observed[field] = {
            "a": ra,
            "b": rb,
            "diff_b_minus_a": (None if ra is None or rb is None else rb - ra),
        }

    rng = random.Random(seed)
    boots: Dict[str, List[Tuple[float, float, float]]] = {f: [] for f in fields}
    n = len(ids)
    for _ in range(n_boot):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        rows_a = [r for sid in sample for r in idx_a[sid]]
        rows_b = [r for sid in sample for r in idx_b[sid]]
        for field in fields:
            ra = _rate(rows_a, field)
            rb = _rate(rows_b, field)
            if ra is None or rb is None:
                continue
            boots[field].append((ra, rb, rb - ra))

    out: Dict[str, Any] = {
        "n_instructions": n,
        "n_boot": n_boot,
        "seed": seed,
        "fields": {},
    }
    for field in fields:
        vals = boots[field]
        if not vals:
            continue
        a_s = sorted(v[0] for v in vals)
        b_s = sorted(v[1] for v in vals)
        d_s = sorted(v[2] for v in vals)
        out["fields"][field] = {
            "observed": observed[field],
            "a_ci95": [_percentile(a_s, 0.025), _percentile(a_s, 0.975)],
            "b_ci95": [_percentile(b_s, 0.025), _percentile(b_s, 0.975)],
            "diff_ci95": [_percentile(d_s, 0.025), _percentile(d_s, 0.975)],
        }
    return out


def mcnemar_exact(n12: int, n21: int) -> Dict[str, Any]:
    """Exact two-sided McNemar test on discordant pairs, p = 0.5.

    n12: A success, B failure
    n21: A failure, B success
    """
    n = n12 + n21
    if n == 0:
        return {"n12": n12, "n21": n21, "n_discordant": 0, "p_value": 1.0}
    k = max(n12, n21)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"n12": n12, "n21": n21, "n_discordant": n, "p_value": p}


def holm_correction(named_pvalues: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    m = len(named_pvalues)
    ordered = sorted(named_pvalues, key=lambda x: x[1])
    adj_raw = []
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        raw = min(1.0, p * (m - i))
        running = max(running, raw)
        adj_raw.append((name, p, running))
    by_name = {name: (p, adj) for name, p, adj in adj_raw}
    return [
        {
            "name": name,
            "p_raw": by_name[name][0],
            "p_holm": by_name[name][1],
            "significant_0.05": by_name[name][1] < 0.05,
        }
        for name, _ in named_pvalues
    ]


def paired_mcnemar_all3(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    a = instruction_all3_success(records_a)
    b = instruction_all3_success(records_b)
    ids = sorted(set(a) & set(b))
    n11 = n10 = n01 = n00 = 0
    for sid in ids:
        sa, sb = a[sid], b[sid]
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
        "aggregation": "instruction succeeds iff all 3 repeats succeed",
        "n_instructions": len(ids),
        "n_both_success": n11,
        "n_a_only": n10,
        "n_b_only": n01,
        "n_both_fail": n00,
        "mcnemar": test,
    }


def formulation_rates(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_form: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_form[r["formulation"]].append(r)
    return {
        k: {
            "n": len(v),
            "task_success_rate": _rate(v, "success"),
            "plan_validity_rate": _rate(v, "valid"),
        }
        for k, v in sorted(by_form.items())
    }
