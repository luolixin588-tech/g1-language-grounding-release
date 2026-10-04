from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ablations.semantic_aliases import (
    build_condition_planner_bundle,
    decode_production_plan,
    scrub_feedback_for_aliases,
)
from src.adapters.production_plan_adapter import parse_production_plan
from src.adapters.semantic_normalizer import normalize_steps
from src.catalog import BENCHMARK_VERSION, load_catalog
from src.evaluator.diagnostics import compute_diagnostics
from src.evaluator.plan_validator import validate_production_then_semantic
from src.evaluator.task_success import evaluate_task_success
from src.io_utils import read_jsonl, write_json, write_jsonl
from src.paths import CONFIG_DIR, INSTRUCTIONS_PATH, RUNS_DIR
from src.runner.experiment_metadata import collect_run_metadata, load_model_config
from src.runner.planner_runner import PlannerRunner

# Paper-facing name -> existing frozen condition file. Do not reimplement.
CONDITION_FILE_ALIASES = {
    "no_schema_constrained_decoding": "no_structure",
}


def resolve_condition_file(name: str) -> str:
    return CONDITION_FILE_ALIASES.get(name, name)


def load_condition(name: str) -> Dict[str, Any]:
    file_stem = resolve_condition_file(name)
    path = CONFIG_DIR / "conditions" / f"{file_stem}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def select_smoke_samples(instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Two semantic tasks per family, both formulations when present (~24)."""
    by_family: Dict[str, List[str]] = {}
    for row in instructions:
        fam = row["family"]
        tid = row["semantic_task_id"]
        by_family.setdefault(fam, [])
        if tid not in by_family[fam]:
            by_family[fam].append(tid)
    wanted = set()
    for fam, tids in by_family.items():
        for tid in tids[:2]:
            wanted.add(tid)
    return [row for row in instructions if row["semantic_task_id"] in wanted]


def select_semantic_smoke_samples(
    instructions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Smoke subset emphasizing P2–P5 (~24), with P1 control and light P6.

    Selection only; does not modify dataset files.
    """
    # Prefer first 3 semantic tasks for P2–P5 (both formulations), first 1 for P1/P6.
    per_family = {
        "P1_spoken_response": 1,
        "P2_single_physical_action": 3,
        "P3_parameterized_control": 3,
        "P4_multistep_physical_sequence": 3,
        "P5_mixed_speech_action_sequence": 3,
        "P6_unsupported_or_constrained": 1,
    }
    by_family: Dict[str, List[str]] = {}
    for row in instructions:
        fam = row["family"]
        tid = row["semantic_task_id"]
        by_family.setdefault(fam, [])
        if tid not in by_family[fam]:
            by_family[fam].append(tid)
    wanted = set()
    for fam, n in per_family.items():
        for tid in by_family.get(fam, [])[:n]:
            wanted.add(tid)
    return [row for row in instructions if row["semantic_task_id"] in wanted]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def evaluate_offline_plan(task: Dict[str, Any], raw: Any) -> Dict[str, Any]:
    production, parse_errors = parse_production_plan(raw)
    semantic, norm_errors = (None, [])
    if production is not None:
        semantic, norm_errors = normalize_steps(production)
    validity = validate_production_then_semantic(
        production, semantic, parse_errors + norm_errors
    )
    success = evaluate_task_success(task, semantic, validity["valid"])
    diag = compute_diagnostics(task, semantic)
    return {
        "parsed_production_plan": production,
        "normalized_semantic_plan": semantic,
        "validity": validity,
        "task_success": success,
        "diagnostics": diag,
    }


def run_batch(
    condition_name: str = "full",
    n_repeats: int = 3,
    smoke: bool = False,
    seed: int = 0,
    ollama_url: str = "http://127.0.0.1:11434/api/chat",
    limit: Optional[int] = None,
    alias_permutation_id: Optional[int] = None,
    out_dir_override: Optional[Path] = None,
    sample_ids: Optional[List[str]] = None,
    model: Optional[str] = None,
    resume: bool = False,
) -> Path:
    condition = load_condition(condition_name)
    if alias_permutation_id is not None:
        condition = dict(condition)
        condition["alias_permutation_id"] = alias_permutation_id
    model_name = model or condition["model"]
    model_display = model_name
    try:
        model_cfg = load_model_config(model_name)
        model_display = str(model_cfg.get("name") or model_name)
        if not model:
            model_name = str(model_cfg.get("ollama_tag") or model_name)
        elif model_cfg.get("ollama_tag"):
            model_name = str(model_cfg["ollama_tag"])
            model_display = str(model_cfg.get("name") or model_name)
    except FileNotFoundError:
        model_cfg = {}
    catalog = load_catalog()
    bundle = build_condition_planner_bundle(catalog, condition)
    tool_catalog = bundle["tool_catalog"]
    schema = bundle["schema"]
    prompt = bundle["prompt"]
    anon_cap = bundle["anonymize_capability_semantics"]
    anon_param = bundle["anonymize_parameter_semantics"]
    cap_maps = bundle["cap_maps"]
    param_maps = bundle["param_maps"]

    production_decoder = None
    feedback_scrubber = None
    if anon_cap or anon_param:

        def production_decoder(visible):
            return decode_production_plan(
                visible,
                anonymize_capability_semantics=anon_cap,
                anonymize_parameter_semantics=anon_param,
                cap_maps=cap_maps,
                param_maps=param_maps,
            )

        def feedback_scrubber(text: str) -> str:
            return scrub_feedback_for_aliases(
                text,
                anonymize_capability_semantics=anon_cap,
                anonymize_parameter_semantics=anon_param,
                cap_maps=cap_maps,
                param_maps=param_maps,
            )

    instructions = read_jsonl(INSTRUCTIONS_PATH)
    if sample_ids is not None:
        wanted = set(sample_ids)
        instructions = [row for row in instructions if row["sample_id"] in wanted]
    elif smoke:
        if anon_cap or anon_param:
            instructions = select_semantic_smoke_samples(instructions)
        else:
            instructions = select_smoke_samples(instructions)
        n_repeats = 1
    if limit is not None:
        instructions = instructions[:limit]

    jobs = []
    for row in instructions:
        for r in range(n_repeats):
            jobs.append((row, r))
    rng = random.Random(seed)
    rng.shuffle(jobs)

    if out_dir_override is not None:
        out_dir = out_dir_override
        run_id = out_dir.name
    else:
        run_id = f"{condition_name}_{_stamp()}_{uuid.uuid4().hex[:8]}"
        out_dir = RUNS_DIR / ("smoke" if smoke else condition_name) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    records: List[Dict[str, Any]] = []
    done_keys = set()
    if resume and records_path.exists():
        records = read_jsonl(records_path)
        done_keys = {(r["sample_id"], r["repeat_index"]) for r in records}
        jobs = [j for j in jobs if (j[0]["sample_id"], j[1]) not in done_keys]

    runner = PlannerRunner(
        ollama_url=ollama_url,
        model=model_name,
        planner_prompt=prompt,
    )
    print(
        f"run_id={run_id} condition={condition_name} n_jobs={len(jobs)} "
        f"already_done={len(done_keys)} smoke={smoke} model={model_name}",
        flush=True,
    )
    n_total = len(jobs) + len(done_keys)
    for i, (sample, repeat_idx) in enumerate(jobs, 1):
        plan = runner.plan(
            user_prompt=sample["instruction"],
            tool_catalog=tool_catalog,
            json_schema=schema,
            max_attempts=int(condition.get("max_attempts", 1)),
            use_validation_retries=bool(condition.get("use_validation_retries")),
            production_decoder=production_decoder,
            feedback_scrubber=feedback_scrubber,
        )
        success = evaluate_task_success(
            sample,
            plan["normalized_semantic_plan"],
            plan["validity"]["valid"],
        )
        diag = compute_diagnostics(sample, plan["normalized_semantic_plan"])
        rec = {
            "run_id": run_id,
            "sample_id": sample["sample_id"],
            "semantic_task_id": sample["semantic_task_id"],
            "formulation": sample["formulation"],
            "family": sample["family"],
            "model": model_name,
            "model_display_name": model_display,
            "condition": condition_name,
            "repeat_index": repeat_idx,
            "instruction": sample["instruction"],
            "raw_model_output": plan["raw_model_output"],
            "parsed_production_plan": plan["parsed_production_plan"],
            "decoded_production_plan": plan.get("decoded_production_plan"),
            "normalized_semantic_plan": plan["normalized_semantic_plan"],
            "validity_result": plan["validity"],
            "task_success_result": success,
            "diagnostic_error_labels": success.get("error_labels", []),
            "diagnostics": diag,
            "attempt_count": plan["attempt_count"],
            "validation_errors": plan["validation_errors"],
            "latency_ms_per_attempt": plan["latency_ms_per_attempt"],
            "total_planning_latency_ms": plan["total_planning_latency_ms"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "parser": "unstructured" if schema is None else "production",
            "ollama_timings": plan.get("ollama_timings"),
            "transport_retries": plan.get("transport_retries", 0),
        }
        write_jsonl(records_path, [rec])
        records.append(rec)
        ok = "OK" if success["task_success"] else "FAIL"
        valid = "V" if plan["validity"]["valid"] else "I"
        print(
            f"[{len(done_keys) + i:03d}/{n_total}] {ok}/{valid} {sample['sample_id']} r{repeat_idx} "
            f"{sample['family']} {plan['total_planning_latency_ms']:.0f}ms "
            f"{success.get('error_labels')}",
            flush=True,
        )

    from src.analysis.summarize_results import summarize_records

    summary = summarize_records(records, run_id=run_id, condition=condition_name)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(
        out_dir / "raw_outputs.jsonl",
        [
            {
                "sample_id": r["sample_id"],
                "repeat_index": r["repeat_index"],
                "instruction": r["instruction"],
                "raw_model_output": r["raw_model_output"],
            }
            for r in records
        ],
        append=False,
    )
    config_out: Dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "condition": condition,
        "model_override": model_name,
        "n_jobs": len(done_keys) + len(jobs) if resume else len(jobs),
        "n_instructions": len(instructions),
        "n_repeats": n_repeats,
        "seed": seed,
        "smoke": smoke,
        "experiment_metadata": collect_run_metadata(
            model_display_name=model_display,
            ollama_tag=model_name,
            condition_name=condition_name,
            condition=condition,
            prompt=prompt,
            schema=schema,
            temperature=float(condition.get("temperature", 0)),
            seed=seed,
            repeats=n_repeats,
            smoke=smoke,
            endpoint=ollama_url,
        ),
    }
    if bundle.get("alias_meta") is not None:
        # Hidden mapping stored for reproducibility; never sent to the planner.
        config_out["alias_meta"] = bundle["alias_meta"]
    write_json(out_dir / "config.json", config_out)
    print(f"wrote {out_dir}", flush=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="full")
    parser.add_argument("--model", default=None, help="Ollama tag or configs/models id")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--alias-permutation-id", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_batch(
        condition_name=args.condition,
        n_repeats=args.repeats,
        smoke=args.smoke,
        seed=args.seed,
        ollama_url=args.ollama_url,
        limit=args.limit,
        alias_permutation_id=args.alias_permutation_id,
        out_dir_override=Path(args.out_dir) if args.out_dir else None,
        model=args.model,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
