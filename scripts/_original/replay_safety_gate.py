#!/usr/bin/env python3
"""Offline replay: frozen Full planner outputs with/without Safety Gate."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.analysis.summarize_results import summarize_records
from src.evaluator.diagnostics import compute_diagnostics
from src.evaluator.task_success import evaluate_task_success
from src.gate.safety_gate import apply_safety_gate, gated_semantic_plan
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.paths import INSTRUCTIONS_PATH, REPORTS_DIR, RUNS_DIR

FULL_RUN = RUNS_DIR / "full" / "full_20260821T051022Z_d81f78bf"


def _tasks_by_sample() -> Dict[str, Dict[str, Any]]:
    return {r["sample_id"]: r for r in read_jsonl(INSTRUCTIONS_PATH)}


def replay_record(
    record: Dict[str, Any],
    task: Dict[str, Any],
    *,
    gate_enabled: bool,
) -> Dict[str, Any]:
    out = dict(record)
    plan = record.get("normalized_semantic_plan")
    valid = bool(record.get("validity_result", {}).get("valid"))

    if gate_enabled:
        gate = apply_safety_gate(record["instruction"], plan)
        exec_plan = gated_semantic_plan(plan, gate)
        success = evaluate_task_success(task, exec_plan, valid)
        diag = compute_diagnostics(task, exec_plan)
        out["safety_gate"] = gate
        out["gated_semantic_plan"] = exec_plan
        out["gate_enabled"] = True
    else:
        success = record["task_success_result"]
        diag = record.get("diagnostics") or compute_diagnostics(task, plan)
        out["safety_gate"] = {
            "decision": "not_applicable",
            "reason": "gate_disabled",
            "physical_dispatch_allowed": None,
        }
        out["gated_semantic_plan"] = plan
        out["gate_enabled"] = False

    out["gated_task_success_result"] = success
    out["gated_diagnostic_error_labels"] = success.get("error_labels", [])
    out["gated_diagnostics"] = diag
    return out


def run_replay(
    source_run: Path,
    out_dir: Path,
    *,
    gate_enabled: bool,
    condition_label: str,
) -> Path:
    records = read_jsonl(source_run / "records.jsonl")
    tasks = _tasks_by_sample()
    replayed: List[Dict[str, Any]] = []
    for rec in records:
        task = tasks[rec["sample_id"]]
        replayed.append(replay_record(rec, task, gate_enabled=gate_enabled))

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
    cfg = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    cfg["replay"] = {
        "source_run": str(source_run),
        "gate_enabled": gate_enabled,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "config.json", cfg)
    if (source_run / "raw_outputs.jsonl").exists():
        shutil.copy2(source_run / "raw_outputs.jsonl", out_dir / "raw_outputs.jsonl")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=FULL_RUN)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = RUNS_DIR / "safety_gate"
    gate_off = root / f"full_gate_off_replay_{stamp}"
    gate_on = root / f"full_gate_on_{stamp}"

    run_replay(args.source_run, gate_off, gate_enabled=False, condition_label="full_gate_off")
    run_replay(args.source_run, gate_on, gate_enabled=True, condition_label="full_safety_gate")
    manifest = {
        "source_run": str(args.source_run),
        "gate_off": str(gate_off),
        "gate_on": str(gate_on),
    }
    write_json(root / f"manifest_{stamp}.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
