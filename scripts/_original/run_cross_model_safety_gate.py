#!/usr/bin/env python3
"""Offline Supplementary safety-gate replay using the official Figure 4 protocol.

Does not call LLMs. Does not overwrite reports/safety_gate_results.json.
Qwen2.5 row is written from the main-text formal result, not recomputed.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.io_utils import read_jsonl, write_json
from src.paths import REPO_ROOT, RUNS_DIR

OUT_DIR = REPO_ROOT / "outputs" / "multimodel_v1.1"
TRACE_DIR = OUT_DIR / "safety_gate_traces"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_safety_gate_evaluation.py"

DISPLAY = {
    "qwen2.5": "Qwen2.5",
    "phi4": "Phi-4",
    "gemma3": "Gemma 3",
    "mistral_nemo": "Mistral NeMo",
}

# Main-text Figure 4 / Section 4.3 confirmed numbers. Do not overwrite with replay.
# Rates are stored as the exact paper values rather than recomputed floats.
QWEN_MAIN_TEXT = {
    "model": "Qwen2.5",
    "p6_total": 60,
    "erroneous_dispatch_before_n": 20,
    "erroneous_dispatch_before_rate": 0.333333,
    "erroneous_dispatch_after_n": 0,
    "erroneous_dispatch_after_rate": 0.0,
    "intercepted_n": 20,
    "interception_rate": 1.0,
    "mcnemar_p": 1.9e-6,
    "valid_control_n": 240,
    "valid_pass_n": 234,
    "valid_pass_rate": 0.975,
    "false_block_n": 6,
    "false_block_rate": 0.025,
    "source": "main_text_formal_result",
}

NEW_MODELS = [
    {
        "key": "phi4",
        "display": "Phi-4",
        "run": RUNS_DIR / "multimodel_v1.1" / "phi4_14b" / "full",
        "expected_model": "phi4:14b",
    },
    {
        "key": "gemma3",
        "display": "Gemma 3",
        "run": RUNS_DIR / "multimodel_v1.1" / "gemma3_12b" / "full",
        "expected_model": "gemma3:12b",
    },
    {
        "key": "mistral_nemo",
        "display": "Mistral NeMo",
        "run": RUNS_DIR / "multimodel_v1.1" / "mistral_nemo_12b" / "full",
        "expected_model": "mistral-nemo:12b",
    },
]


def _load_official_eval_module():
    spec = importlib.util.spec_from_file_location("run_safety_gate_evaluation", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official gate evaluation script: {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def qc_full_run(run_dir: Path, expected_model: str) -> Dict[str, Any]:
    warnings: List[str] = []
    rec_path = run_dir / "records.jsonl"
    if not rec_path.exists():
        return {
            "ok": False,
            "warnings": [f"missing records.jsonl at {rec_path}"],
            "n_records": 0,
            "n_p6": 0,
        }
    records = read_jsonl(rec_path)
    n = len(records)
    if n != 360:
        warnings.append(f"Full output count is {n}, expected 360")

    keys = [(r.get("sample_id"), r.get("repeat_index")) for r in records]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        warnings.append(f"duplicate (sample_id, repeat_index): {dup[:10]}")

    missing_meta = [
        r.get("sample_id")
        for r in records
        if r.get("sample_id") is None
        or r.get("repeat_index") is None
        or r.get("instruction") is None
        or "normalized_semantic_plan" not in r
        or r.get("family") is None
    ]
    if missing_meta:
        warnings.append(f"missing metadata on {len(missing_meta)} records")

    models = sorted({r.get("model") for r in records})
    if models != [expected_model]:
        warnings.append(f"model field {models}, expected [{expected_model}]")

    conditions = sorted({r.get("condition") for r in records})
    if conditions and conditions != ["full"]:
        warnings.append(f"condition field {conditions}, expected [full]")

    fam = Counter(str(r.get("family", ""))[:2] for r in records)
    p6 = [r for r in records if str(r.get("family", "")).startswith("P6_")]
    if len(p6) != 60:
        warnings.append(f"P6 runs = {len(p6)}, expected 60")

    empty_plan = sum(1 for r in records if not r.get("normalized_semantic_plan"))
    p6_empty = sum(1 for r in p6 if not r.get("normalized_semantic_plan"))
    validity_fail = sum(
        1 for r in records if (r.get("validity_result") or {}).get("valid") is False
    )

    families = Counter(r.get("family") for r in records)
    expected_fams = {
        "P1_spoken_response",
        "P2_single_physical_action",
        "P3_parameterized_control",
        "P4_multistep_physical_sequence",
        "P5_mixed_speech_action_sequence",
        "P6_unsupported_or_constrained",
    }
    if set(families) != expected_fams:
        warnings.append(f"unexpected families: {sorted(families)}")
    for fam_name, count in families.items():
        if count != 60:
            warnings.append(f"{fam_name} has {count} runs, expected 60")

    return {
        "ok": not warnings,
        "warnings": warnings,
        "n_records": n,
        "n_p6": len(p6),
        "families": dict(families),
        "models": models,
        "empty_normalized_plan": empty_plan,
        "p6_empty_normalized_plan": p6_empty,
        "validity_invalid": validity_fail,
        "duplicates": len(dup),
        "family_prefix_counts": dict(fam),
    }


def _rate(n: int, d: int) -> Optional[float]:
    if d == 0:
        return None
    return n / d


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return ""
    return f"{p:.10g}"


def result_to_row(display: str, result: Dict[str, Any], source: str) -> Dict[str, Any]:
    p6_off = result["p6_metrics"]["gate_off"]["erroneous_physical_dispatch"]
    p6_on = result["p6_metrics"]["gate_on"]["erroneous_physical_dispatch"]
    n_p6 = len(result["p6_cases"])
    intercepted = result["interception"]["blocked"]
    original_err = result["interception"]["original_erroneous_dispatches"]
    ctrl = result["legitimate_control"]
    mcnemar = result["mcnemar_erroneous_dispatch"]
    p_value = mcnemar.get("p_value")
    if original_err == 0:
        p_value = None
        intercept_rate = None
    else:
        intercept_rate = intercepted / original_err
    return {
        "model": display,
        "p6_total": n_p6,
        "erroneous_dispatch_before_n": p6_off,
        "erroneous_dispatch_before_rate": _rate(p6_off, n_p6),
        "erroneous_dispatch_after_n": p6_on,
        "erroneous_dispatch_after_rate": _rate(p6_on, n_p6),
        "intercepted_n": intercepted,
        "interception_rate": intercept_rate,
        "mcnemar_p": p_value,
        "valid_control_n": ctrl["valid_physical_outputs"],
        "valid_pass_n": ctrl["passed"],
        "valid_pass_rate": _rate(ctrl["passed"], ctrl["valid_physical_outputs"]),
        "false_block_n": ctrl["blocked"],
        "false_block_rate": _rate(ctrl["blocked"], ctrl["valid_physical_outputs"]),
        "source": source,
    }


def slim_cases(cases: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    out = []
    for r in cases:
        row = {
            "sample_id": r["sample_id"],
            "repeat_index": r["repeat_index"],
            "instruction": r["instruction"],
            "original_physical_dispatch": r["original_physical_dispatch"],
            "physical_dispatch_after_gate": r["physical_dispatch_after_gate"],
            "gate_decision": (r.get("safety_gate") or {}).get("decision"),
            "gate_reason": (r.get("safety_gate") or {}).get("reason"),
        }
        if kind == "p6":
            row["blocked_status"] = r.get("blocked_status")
        else:
            row["family"] = r.get("family")
            row["false_block"] = r.get("false_block")
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            written = {}
            for k in fieldnames:
                v = row.get(k)
                if v is None:
                    written[k] = ""
                elif isinstance(v, float):
                    written[k] = f"{v:.10g}"
                else:
                    written[k] = v
            w.writerow(written)


def main() -> None:
    official = _load_official_eval_module()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qc_all: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_script": str(SCRIPT_PATH),
        "official_gate_module": "src.safety_gate",
        "official_config": "configs/safety_gate.json",
        "protocol": "scripts/run_safety_gate_evaluation.py::run_evaluation",
        "qwen_source": "main_text_formal_result",
        "models": {},
        "warnings": [],
    }

    rows: List[Dict[str, Any]] = [dict(QWEN_MAIN_TEXT)]

    for spec in NEW_MODELS:
        qc = qc_full_run(spec["run"], spec["expected_model"])
        result = official.run_evaluation(spec["run"])
        row = result_to_row(spec["display"], result, "cross_model_replay")
        rows.append(row)

        slim = {
            "model": spec["display"],
            "source_run": str(spec["run"]),
            "qc": qc,
            "p6_metrics": result["p6_metrics"],
            "interception": result["interception"],
            "legitimate_control": result["legitimate_control"],
            "mcnemar_erroneous_dispatch": result["mcnemar_erroneous_dispatch"],
            "sanity_checks": result["sanity_checks"],
            "safety_gate_config": result["safety_gate_config"],
            "p6_cases": slim_cases(result["p6_cases"], "p6"),
            "control_cases": slim_cases(result["control_cases"], "control"),
        }
        write_json(TRACE_DIR / f"{spec['key']}_safety_gate.json", slim)

        if qc["warnings"]:
            qc_all["warnings"].extend([f"{spec['display']}: {w}" for w in qc["warnings"]])
        if row["p6_total"] != 60:
            qc_all["warnings"].append(f"{spec['display']}: p6_total={row['p6_total']}")
        if row["valid_control_n"] == 0:
            qc_all["warnings"].append(f"{spec['display']}: valid-control set is empty")

        qc_all["models"][spec["key"]] = {
            "display": spec["display"],
            "qc": qc,
            "row": row,
            "gate_reasons_p6_blocked": dict(
                Counter(
                    c["gate_reason"]
                    for c in slim["p6_cases"]
                    if c["gate_decision"] == "block"
                )
            ),
            "gate_reasons_false_block": dict(
                Counter(
                    c["gate_reason"]
                    for c in slim["control_cases"]
                    if c["false_block"]
                )
            ),
        }

    summary_fields = [
        "model",
        "p6_total",
        "erroneous_dispatch_before_n",
        "erroneous_dispatch_before_rate",
        "erroneous_dispatch_after_n",
        "erroneous_dispatch_after_rate",
        "intercepted_n",
        "interception_rate",
        "mcnemar_p",
        "valid_control_n",
        "valid_pass_n",
        "valid_pass_rate",
        "false_block_n",
        "false_block_rate",
        "source",
    ]
    write_csv(OUT_DIR / "cross_model_safety_gate.csv", rows, summary_fields)

    fig_rows = []
    for row in rows:
        fig_rows.append(
            {
                "model": row["model"],
                "before_erroneous_dispatch_rate": row["erroneous_dispatch_before_rate"],
                "after_erroneous_dispatch_rate": row["erroneous_dispatch_after_rate"],
                "valid_pass_rate": row["valid_pass_rate"],
                "false_block_rate": row["false_block_rate"],
            }
        )
    write_csv(
        OUT_DIR / "figure_s2_gate.csv",
        fig_rows,
        [
            "model",
            "before_erroneous_dispatch_rate",
            "after_erroneous_dispatch_rate",
            "valid_pass_rate",
            "false_block_rate",
        ],
    )

    write_json(OUT_DIR / "cross_model_safety_gate_qc.json", qc_all)
    print(json.dumps({"rows": rows, "warnings": qc_all["warnings"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
