"""Offline execution-time replanning recovery experiment.

Separate from the frozen v1.1 planning benchmark. Does not move the robot.
"""
from __future__ import annotations

import argparse
import json
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
from src.recovery.failure_injection import (
    failure_class_for,
    first_physical_index,
    make_diagnostic,
)
from src.runner.planner_runner import PlannerRunner


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_recovery_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "recovery" / "selected_tasks.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_selected_tasks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    wanted = set(cfg["selected_semantic_task_ids"])
    formulation = cfg.get("formulation", "canonical")
    rows = []
    for row in read_jsonl(INSTRUCTIONS_PATH):
        if row.get("semantic_task_id") in wanted and row.get("formulation") == formulation:
            rows.append(row)
    by_id = {r["semantic_task_id"]: r for r in rows}
    missing = [tid for tid in cfg["selected_semantic_task_ids"] if tid not in by_id]
    if missing:
        raise RuntimeError(f"missing selected tasks: {missing}")
    return [by_id[tid] for tid in cfg["selected_semantic_task_ids"]]


def _initial_plan(runner: PlannerRunner, instruction: str, tool_catalog, schema) -> Dict[str, Any]:
    return runner.plan(
        user_prompt=instruction,
        tool_catalog=tool_catalog,
        json_schema=schema,
        max_attempts=3,
        use_validation_retries=True,
    )


# F1–F6 keep the frozen recoverable wrapper. Do not change wording unless
# diagnostic["recoverable"] is false.
_RECOVERABLE_REPLAN_PREFIX = (
    "Execution-time diagnostic from the robot runtime. "
    "The failure is recoverable and software-level. "
    "Plan only the remaining task.\n"
    "During recovery, completed_steps are immutable and must never appear "
    "again in the recovery plan. Generate only the unfinished suffix "
    "required to complete the original request.\n"
)

_UNRECOVERABLE_REPLAN_PREFIX = (
    "Execution-time diagnostic from the robot runtime.\n\n"
    "The runtime has reported an unrecoverable execution failure. "
    "No further physical recovery is permitted for this request.\n\n"
    "Do not generate any physical action. "
    "Do not retry the failed physical step. "
    "Do not repeat any completed physical step. "
    "Do not attempt to complete the remaining physical task.\n\n"
    "If a response is appropriate, you may generate only a spoken response "
    "that briefly explains that the task cannot continue. "
    "A speak-only plan is the required terminal output. "
    "Do not return an empty plan.\n\n"
    "Any plan containing a physical action after this diagnostic is invalid.\n"
)

_UNRECOVERABLE_RETRY_PREFIX = (
    "The previous plan was not catalog-legal. "
    "Do not generate any physical action. "
    "Provide a speak-only plan. "
)


def _recovery_system_content(diagnostic: Dict[str, Any]) -> str:
    if diagnostic.get("recoverable", True):
        prefix = _RECOVERABLE_REPLAN_PREFIX
    else:
        prefix = _UNRECOVERABLE_REPLAN_PREFIX
    return prefix + json.dumps(diagnostic, ensure_ascii=False)


def _replan_remaining(
    runner: PlannerRunner,
    instruction: str,
    diagnostic: Dict[str, Any],
    tool_catalog,
    schema,
    max_execution_replans: int,
) -> Dict[str, Any]:
    recoverable = bool(diagnostic.get("recoverable", True))
    recovery_system = _recovery_system_content(diagnostic)
    extra = [
        {
            "role": "system",
            "content": recovery_system,
        }
    ]
    attempts: List[Dict[str, Any]] = []
    last: Optional[Dict[str, Any]] = None
    for i in range(1, max_execution_replans + 1):
        plan = runner.plan(
            user_prompt=instruction,
            tool_catalog=tool_catalog,
            json_schema=schema,
            max_attempts=3,
            use_validation_retries=True,
            extra_messages=extra,
        )
        last = plan
        attempts.append(
            {
                "execution_replan_index": i,
                "valid": bool(plan["validity"]["valid"]),
                "errors": plan["validity"].get("errors") or [],
                "latency_ms": plan["total_planning_latency_ms"],
                "raw_model_output": plan["raw_model_output"],
                "parsed_production_plan": plan["parsed_production_plan"],
                "normalized_semantic_plan": plan["normalized_semantic_plan"],
            }
        )
        if plan["validity"]["valid"] and plan.get("normalized_semantic_plan"):
            break
        if recoverable:
            retry_text = (
                "The remaining-task plan was not catalog-legal. Fix it. "
                f"Error: {'; '.join(plan['validity'].get('errors') or [])}"
            )
        else:
            retry_text = (
                _UNRECOVERABLE_RETRY_PREFIX
                + f"Error: {'; '.join(plan['validity'].get('errors') or [])}"
            )
        extra = extra + [
            {
                "role": "system",
                "content": retry_text,
            }
        ]
    assert last is not None
    last["execution_replan_attempts"] = attempts
    last["n_execution_replans"] = len(attempts)
    last["replan_latency_ms"] = sum(a["latency_ms"] for a in attempts)
    last["recovery_system_instruction"] = recovery_system
    last["recovery_prompt_recoverable"] = recoverable
    return last


def run_matched_pair(
    task: Dict[str, Any],
    repeat_index: int,
    runner: PlannerRunner,
    tool_catalog,
    schema,
    max_execution_replans: int,
) -> List[Dict[str, Any]]:
    instruction = task["instruction"]
    initial = _initial_plan(runner, instruction, tool_catalog, schema)
    semantic = initial.get("normalized_semantic_plan")
    production = initial.get("parsed_production_plan")
    valid = bool(initial["validity"]["valid"])
    fail_idx = first_physical_index(semantic or [])
    error_class = failure_class_for(task["semantic_task_id"], repeat_index)

    base_meta = {
        "semantic_task_id": task["semantic_task_id"],
        "sample_id": task["sample_id"],
        "family": task["family"],
        "formulation": task["formulation"],
        "instruction": instruction,
        "repeat_index": repeat_index,
        "injected_failure_class": error_class,
        "initial_plan_valid": valid,
        "initial_raw_model_output": initial["raw_model_output"],
        "initial_production_plan": production,
        "initial_semantic_plan": semantic,
        "initial_plan_latency_ms": initial["total_planning_latency_ms"],
        "initial_attempt_count": initial["attempt_count"],
        "benchmark_version": BENCHMARK_VERSION,
        "model": "qwen2.5:14b",
        "temperature": 0,
    }

    def pack(condition: str, **kwargs) -> Dict[str, Any]:
        rec = dict(base_meta)
        rec["condition"] = condition
        rec.update(kwargs)
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        return rec

    if not valid or fail_idx < 0 or production is None or semantic is None:
        outcome = evaluate_executed_trace(task, [])
        reason = "initial_plan_invalid" if not valid else "no_eligible_physical_action"
        recs = []
        for cond in ("R0", "R1"):
            recs.append(
                pack(
                    cond,
                    injected=False,
                    skip_reason=reason,
                    failed_step_index=fail_idx,
                    completed_semantic_steps=[],
                    executed_semantic_steps=[],
                    n_execution_replans=0,
                    replan_latency_ms=0.0,
                    added_recovery_latency_ms=0.0,
                    recovery_success=False,
                    recovery_error_labels=outcome["error_labels"],
                    recovery_reason=reason,
                    replan_attempts=[],
                )
            )
        return recs

    completed_prod = production[:fail_idx]
    completed_sem = semantic[:fail_idx]
    failed_prod = production[fail_idx]
    diagnostic = make_diagnostic(error_class, failed_prod, completed_prod, fail_idx)

    # R0: terminate after the injected failure.
    r0_eval = evaluate_executed_trace(task, list(completed_sem))
    r0 = pack(
        "R0",
        injected=True,
        skip_reason=None,
        failed_step_index=fail_idx,
        failed_production_step=failed_prod,
        diagnostic=diagnostic,
        completed_semantic_steps=completed_sem,
        executed_semantic_steps=completed_sem,
        n_execution_replans=0,
        replan_latency_ms=0.0,
        added_recovery_latency_ms=0.0,
        recovery_success=r0_eval["recovery_success"],
        recovery_error_labels=r0_eval["error_labels"],
        recovery_reason=r0_eval["reason"],
        replan_attempts=[],
        replan_raw_model_output=None,
    )

    # R1: bounded execution-time replan of the remaining task, then simulate success.
    replan = _replan_remaining(
        runner, instruction, diagnostic, tool_catalog, schema, max_execution_replans
    )
    remaining_sem = replan.get("normalized_semantic_plan") or []
    remaining_valid = bool(replan["validity"]["valid"])
    executed = list(completed_sem)
    if remaining_valid:
        executed.extend(remaining_sem)
        r1_eval = evaluate_executed_trace(task, executed)
        reason = r1_eval["reason"]
        success = r1_eval["recovery_success"]
        labels = r1_eval["error_labels"]
    else:
        r1_eval = evaluate_executed_trace(task, executed)
        success = False
        labels = list(r1_eval["error_labels"]) + ["replan_invalid"]
        reason = "replan_invalid"

    r1 = pack(
        "R1",
        injected=True,
        skip_reason=None,
        failed_step_index=fail_idx,
        failed_production_step=failed_prod,
        diagnostic=diagnostic,
        completed_semantic_steps=completed_sem,
        executed_semantic_steps=executed,
        n_execution_replans=replan.get("n_execution_replans", 0),
        replan_latency_ms=replan.get("replan_latency_ms", 0.0),
        added_recovery_latency_ms=replan.get("replan_latency_ms", 0.0),
        recovery_success=success,
        recovery_error_labels=labels,
        recovery_reason=reason,
        replan_attempts=replan.get("execution_replan_attempts") or [],
        replan_raw_model_output=replan.get("raw_model_output"),
        replan_valid=remaining_valid,
    )
    return [r0, r1]


def run_experiment(
    ollama_url: str = "http://127.0.0.1:11434/api/chat",
) -> Path:
    cfg = load_recovery_config()
    tasks = load_selected_tasks(cfg)
    catalog = load_catalog()
    tool_catalog = build_planner_tool_catalog(catalog, reduced=False)
    schema = build_planner_json_schema(catalog)
    runner = PlannerRunner(
        ollama_url=ollama_url,
        model=cfg.get("model", "qwen2.5:14b"),
        planner_prompt=planner_prompt(),
    )
    n_repeats = int(cfg.get("n_repeats", 3))
    max_replans = int(cfg.get("max_execution_replans", 2))

    run_id = f"recovery_{_stamp()}_{uuid.uuid4().hex[:8]}"
    out_dir = RUNS_DIR / "recovery" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    records: List[Dict[str, Any]] = []

    jobs = [(task, r) for task in tasks for r in range(n_repeats)]
    print(f"run_id={run_id} n_pairs={len(jobs)} n_episodes={len(jobs) * 2}", flush=True)
    for i, (task, repeat_idx) in enumerate(jobs, 1):
        pair = run_matched_pair(
            task, repeat_idx, runner, tool_catalog, schema, max_replans
        )
        for rec in pair:
            rec["run_id"] = run_id
            write_jsonl(records_path, [rec])
            records.append(rec)
            flag = "OK" if rec["recovery_success"] else "FAIL"
            print(
                f"[{i:02d}/{len(jobs)}] {rec['condition']} {flag} "
                f"{rec['semantic_task_id']} r{repeat_idx} "
                f"replans={rec['n_execution_replans']} "
                f"+{rec['added_recovery_latency_ms']:.0f}ms "
                f"{rec.get('recovery_error_labels')}",
                flush=True,
            )

    write_jsonl(
        out_dir / "raw_outputs.jsonl",
        [
            {
                "condition": r["condition"],
                "semantic_task_id": r["semantic_task_id"],
                "repeat_index": r["repeat_index"],
                "initial_raw_model_output": r.get("initial_raw_model_output"),
                "replan_raw_model_output": r.get("replan_raw_model_output"),
            }
            for r in records
        ],
        append=False,
    )
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "benchmark_version_frozen": BENCHMARK_VERSION,
            "recovery_config": cfg,
            "n_episodes": len(records),
            "n_tasks": len(tasks),
            "n_repeats": n_repeats,
        },
    )
    print(f"wrote {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    args = parser.parse_args()
    run_experiment(ollama_url=args.ollama_url)


if __name__ == "__main__":
    main()
