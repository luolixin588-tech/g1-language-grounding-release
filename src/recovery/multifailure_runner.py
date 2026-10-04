"""Multi-failure recovery experiment runner (separate from frozen R0/R1 benchmark)."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.catalog import (
    BENCHMARK_VERSION,
    build_planner_json_schema,
    build_planner_tool_catalog,
    load_catalog,
    planner_prompt,
)
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.paths import CONFIG_DIR, INSTRUCTIONS_PATH, RUNS_DIR
from src.recovery.evaluate_recovery import evaluate_executed_trace
from src.recovery.multifailure_injection import (
    FAILURE_CONDITIONS,
    inject_execution_failure,
    validate_injection_preconditions,
)
from src.recovery.multifailure_outcomes import (
    detect_incorrect_recovery,
    detect_repeated_action,
    evaluate_f7_correct_termination,
)
from src.recovery.recovery_runner import _initial_plan, _replan_remaining
from src.runner.planner_runner import PlannerRunner


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_multifailure_config(name: str = "multifailure_smoke") -> Dict[str, Any]:
    path = CONFIG_DIR / "recovery" / f"{name}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def expand_formal_episodes(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build episode list from condition_task_applicability x n_repeats."""
    applicability = cfg.get("condition_task_applicability") or {}
    n_repeats = int(cfg.get("n_repeats", 2))
    episodes: List[Dict[str, Any]] = []
    for cond in FAILURE_CONDITIONS:
        task_ids = applicability.get(cond)
        if not task_ids:
            continue
        for tid in task_ids:
            for repeat_id in range(n_repeats):
                episodes.append(
                    {
                        "failure_condition": cond,
                        "semantic_task_id": tid,
                        "repeat_id": repeat_id,
                    }
                )
    return episodes


def load_episode_tasks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    formulation = cfg.get("formulation", "canonical")
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(INSTRUCTIONS_PATH):
        if row.get("formulation") == formulation:
            by_id[row["semantic_task_id"]] = row

    raw_episodes = cfg.get("episodes")
    if not raw_episodes and cfg.get("condition_task_applicability"):
        raw_episodes = expand_formal_episodes(cfg)

    episodes: List[Dict[str, Any]] = []
    for ep in raw_episodes or []:
        tid = ep["semantic_task_id"]
        if tid not in by_id:
            raise RuntimeError(f"missing task {tid}")
        task = dict(by_id[tid])
        task["_episode_meta"] = ep
        episodes.append(task)
    return episodes


def run_multifailure_episode(
    task: Dict[str, Any],
    runner: PlannerRunner,
    tool_catalog,
    schema,
    max_execution_replans: int,
) -> Dict[str, Any]:
    meta = task["_episode_meta"]
    condition_id: str = meta["failure_condition"]
    repeat_id: int = int(meta.get("repeat_id", 0))
    instruction = task["instruction"]

    initial = _initial_plan(runner, instruction, tool_catalog, schema)
    semantic = initial.get("normalized_semantic_plan") or []
    production = initial.get("parsed_production_plan") or []
    valid = bool(initial["validity"]["valid"])

    base = {
        "sample_id": task["sample_id"],
        "semantic_task_id": task["semantic_task_id"],
        "family": task["family"],
        "formulation": task["formulation"],
        "instruction": instruction,
        "failure_condition": condition_id,
        "repeat_id": repeat_id,
        "initial_plan": production,
        "initial_semantic_plan": semantic,
        "initial_plan_valid": valid,
        "initial_raw_model_output": initial.get("raw_model_output"),
        "initial_plan_latency_ms": initial.get("total_planning_latency_ms"),
        "benchmark_version": BENCHMARK_VERSION,
        "model": "qwen2.5:14b",
        "temperature": 0,
        "timestamp": _iso_now(),
    }

    if not valid:
        out = evaluate_executed_trace(task, [])
        return {
            **base,
            "injected": False,
            "skip_reason": "initial_plan_invalid",
            "injected_step_index": -1,
            "failure_diagnostic": None,
            "completed_steps_before_failure": [],
            "partial_completion": False,
            "recoverable": condition_id != "F7_unrecoverable_failure",
            "recovery_attempts": 0,
            "recovery_plans": [],
            "executed_steps_after_recovery": [],
            "executed_semantic_steps": [],
            "repeated_action": False,
            "incorrect_recovery": False,
            "incorrect_recovery_reasons": [],
            "final_task_success": False,
            "correct_termination": False,
            "failure_detected_timestamp": None,
            "recovery_plan_ready_timestamp": None,
            "added_recovery_latency_ms": 0.0,
            "n_execution_replans": 0,
            "recovery_success": False,
            "recovery_error_labels": out["error_labels"],
        }

    ok, skip = validate_injection_preconditions(condition_id, semantic)
    if not ok:
        out = evaluate_executed_trace(task, [])
        return {
            **base,
            "injected": False,
            "skip_reason": skip,
            "injected_step_index": -1,
            "failure_diagnostic": None,
            "completed_steps_before_failure": [],
            "partial_completion": False,
            "recoverable": condition_id != "F7_unrecoverable_failure",
            "recovery_attempts": 0,
            "recovery_plans": [],
            "executed_steps_after_recovery": [],
            "executed_semantic_steps": [],
            "repeated_action": False,
            "incorrect_recovery": False,
            "incorrect_recovery_reasons": [],
            "final_task_success": False,
            "correct_termination": False,
            "failure_detected_timestamp": None,
            "recovery_plan_ready_timestamp": None,
            "added_recovery_latency_ms": 0.0,
            "n_execution_replans": 0,
            "recovery_success": False,
            "recovery_error_labels": out["error_labels"],
        }

    failure_detected_ts = time.perf_counter()
    failure_detected_iso = _iso_now()

    injection = inject_execution_failure(
        condition_id=condition_id,
        task=task,
        plan={"semantic": semantic, "production": production},
        execution_state={},
        target_step=None,
    )
    completed_sem = list(injection["completed_semantic_steps"])
    diagnostic = injection["diagnostic"]
    recoverable = bool(injection["recoverable"])

    recovery_plans: List[Dict[str, Any]] = []
    replan_attempts: List[Dict[str, Any]] = []
    replan_raw: Optional[str] = None
    remaining_sem: List[Dict[str, Any]] = []
    remaining_valid = False
    n_replans = 0
    replan_latency = 0.0
    recovery_plan_ready_iso: Optional[str] = None
    added_latency_ms = 0.0

    if recoverable:
        replan = _replan_remaining(
            runner,
            instruction,
            diagnostic,
            tool_catalog,
            schema,
            max_execution_replans,
        )
        recovery_plan_ready_ts = time.perf_counter()
        recovery_plan_ready_iso = _iso_now()
        added_latency_ms = (recovery_plan_ready_ts - failure_detected_ts) * 1000.0
        replan_attempts = list(replan.get("execution_replan_attempts") or [])
        n_replans = int(replan.get("n_execution_replans") or 0)
        replan_latency = float(replan.get("replan_latency_ms") or 0.0)
        replan_raw = replan.get("raw_model_output")
        remaining_sem = list(replan.get("normalized_semantic_plan") or [])
        remaining_valid = bool(replan["validity"]["valid"])
        for att in replan_attempts:
            recovery_plans.append(
                {
                    "execution_replan_index": att.get("execution_replan_index"),
                    "valid": att.get("valid"),
                    "normalized_semantic_plan": att.get("normalized_semantic_plan"),
                    "errors": att.get("errors"),
                }
            )
        executed = list(completed_sem)
        if remaining_valid:
            executed.extend(remaining_sem)
    else:
        # F7: same recovery pathway entry, but do not simulate remaining success.
        replan = _replan_remaining(
            runner,
            instruction,
            diagnostic,
            tool_catalog,
            schema,
            max_execution_replans,
        )
        recovery_plan_ready_ts = time.perf_counter()
        recovery_plan_ready_iso = _iso_now()
        added_latency_ms = (recovery_plan_ready_ts - failure_detected_ts) * 1000.0
        replan_attempts = list(replan.get("execution_replan_attempts") or [])
        n_replans = int(replan.get("n_execution_replans") or 0)
        replan_latency = float(replan.get("replan_latency_ms") or 0.0)
        replan_raw = replan.get("raw_model_output")
        remaining_sem = list(replan.get("normalized_semantic_plan") or [])
        remaining_valid = bool(replan["validity"]["valid"])
        for att in replan_attempts:
            recovery_plans.append(
                {
                    "execution_replan_index": att.get("execution_replan_index"),
                    "valid": att.get("valid"),
                    "normalized_semantic_plan": att.get("normalized_semantic_plan"),
                    "errors": att.get("errors"),
                }
            )
        executed = list(completed_sem)

    repeated, _ = detect_repeated_action(completed_sem, executed)
    incorrect, incorrect_reasons = detect_incorrect_recovery(
        condition_id=condition_id,
        task=task,
        completed_before_failure=completed_sem,
        executed_trace=executed,
        remaining_plan=remaining_sem,
        repeated_action=repeated,
        replan_valid=remaining_valid,
        recoverable=recoverable,
    )

    eval_out = evaluate_executed_trace(task, executed)
    final_task_success = bool(eval_out["recovery_success"])

    if recoverable:
        recovery_success = final_task_success and remaining_valid
        correct_termination = False
    else:
        recovery_success = False
        correct_termination = evaluate_f7_correct_termination(
            executed_trace=executed,
            completed_before_failure=completed_sem,
            remaining_plan=remaining_sem,
            n_execution_replans=n_replans,
            max_execution_replans=max_execution_replans,
        )

    return {
        **base,
        "injected": True,
        "skip_reason": None,
        "injected_step_index": injection["injected_step_index"],
        "failure_diagnostic": injection["failure_diagnostic"],
        "diagnostic": diagnostic,
        "completed_steps_before_failure": completed_sem,
        "partial_completion": bool(injection["partial"]),
        "recoverable": recoverable,
        "recovery_attempts": n_replans,
        "recovery_plans": recovery_plans,
        "replan_attempts": replan_attempts,
        "replan_raw_model_output": replan_raw,
        "replan_valid": remaining_valid,
        "executed_steps_after_recovery": executed[len(completed_sem) :],
        "executed_semantic_steps": executed,
        "repeated_action": repeated,
        "incorrect_recovery": incorrect,
        "incorrect_recovery_reasons": incorrect_reasons,
        "final_task_success": final_task_success,
        "correct_termination": correct_termination,
        "failure_detected_timestamp": failure_detected_iso,
        "recovery_plan_ready_timestamp": recovery_plan_ready_iso,
        "added_recovery_latency_ms": added_latency_ms,
        "replan_latency_ms": replan_latency,
        "n_execution_replans": n_replans,
        "recovery_system_instruction": replan.get("recovery_system_instruction"),
        "recovery_prompt_recoverable": replan.get("recovery_prompt_recoverable"),
        "recovery_success": recovery_success,
        "recovery_error_labels": eval_out.get("error_labels") or [],
        "recovery_reason": eval_out.get("reason"),
        "injection_metadata": {
            "failure_condition": injection["failure_condition"],
            "injected_step_index": injection["injected_step_index"],
            "failure_diagnostic": injection["failure_diagnostic"],
            "completed_step_indices": injection["completed_step_indices"],
            "partial": injection["partial"],
            "recoverable": injection["recoverable"],
        },
    }


def run_multifailure_experiment(
    config_name: str = "multifailure_smoke",
    ollama_url: str = "http://127.0.0.1:11434/api/chat",
) -> Path:
    cfg = load_multifailure_config(config_name)
    episodes = load_episode_tasks(cfg)
    catalog = load_catalog()
    tool_catalog = build_planner_tool_catalog(catalog, reduced=False)
    schema = build_planner_json_schema(catalog)
    runner = PlannerRunner(
        ollama_url=ollama_url,
        model=cfg.get("model", "qwen2.5:14b"),
        planner_prompt=planner_prompt(),
    )
    max_replans = int(cfg.get("max_execution_replans", 2))

    run_id = f"multifailure_{config_name}_{_stamp()}_{uuid.uuid4().hex[:8]}"
    out_dir = RUNS_DIR / "multifailure_recovery" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    print(
        f"run_id={run_id} episodes={len(episodes)} config={config_name}",
        flush=True,
    )
    records: List[Dict[str, Any]] = []
    for i, task in enumerate(episodes, 1):
        rec = run_multifailure_episode(task, runner, tool_catalog, schema, max_replans)
        rec["run_id"] = run_id
        write_jsonl(records_path, [rec])
        records.append(rec)
        cond = rec["failure_condition"]
        flag = "OK" if rec.get("recovery_success") or rec.get("correct_termination") else "FAIL"
        print(
            f"[{i:02d}/{len(episodes)}] {cond} {flag} "
            f"{rec['semantic_task_id']} r{rec['repeat_id']} "
            f"inj={rec.get('injected')} idx={rec.get('injected_step_index')} "
            f"partial={rec.get('partial_completion')} "
            f"replans={rec.get('n_execution_replans')} "
            f"+{rec.get('added_recovery_latency_ms', 0):.0f}ms",
            flush=True,
        )

    write_jsonl(
        out_dir / "raw_outputs.jsonl",
        [
            {
                "failure_condition": r["failure_condition"],
                "semantic_task_id": r["semantic_task_id"],
                "repeat_id": r["repeat_id"],
                "initial_raw_model_output": r.get("initial_raw_model_output"),
                "replan_raw_model_output": r.get("replan_raw_model_output"),
                "recovery_system_instruction": r.get("recovery_system_instruction"),
                "recovery_prompt_recoverable": r.get("recovery_prompt_recoverable"),
            }
            for r in records
        ],
        append=False,
    )
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "config_name": config_name,
            "benchmark_version_frozen": BENCHMARK_VERSION,
            "multifailure_config": cfg,
            "n_episodes": len(records),
            "failure_conditions": list(FAILURE_CONDITIONS),
        },
    )
    print(f"wrote {out_dir}", flush=True)
    return out_dir
