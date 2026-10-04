# Repository manifest

Every released file is a **copy**. Original experimental files were not moved
or edited. `Original source path` is relative to the local working tree
`g1-planning-benchmark` unless noted.

`paths.py` in this release is the only algorithmic file whose *path constants*
were remapped to the public layout. All other copied code, configs, and
records are byte-for-byte except: (1) `REUSED.json` `source_run` rewritten to
a release-relative path; (2) some Markdown reports had a local prefix replaced
by `<original-repo>`.

## Benchmark

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `data/semantic_tasks.jsonl` | `data/benchmark/semantic_tasks.jsonl` | 60 gold semantic tasks | Methods; all planning figures |
| `data/instructions_120.jsonl` | `data/benchmark/instructions_120.jsonl` | 120 canonical/paraphrase instructions | Methods |
| `schemas/task_spec.schema.json` | `data/benchmark/task_spec.schema.json` | Task-spec schema | Methods; AQ5 |
| `reports/data_qc.{json,md}` | `data/benchmark/data_qc.{json,md}` | Frozen QC | Methods |
| `docs/benchmark_protocol.md` | `docs/benchmark_protocol.md` | Protocol v1.1 | Methods |
| *(new; retrospective capture)* | `docs/retained_environment_manifest.md` | Retained OS / Ollama / Q4_K_M / SHA256 | AQ5 |
| `configs/capability_catalog.json` | `configs/capability_catalog.json` | Catalogue v1.1 | Fig. 3; AQ5 |

## Qwen2.5 formal planning runs (main text)

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `runs/full/full_20260821T051022Z_d81f78bf/{config,records,summary}` | `data/planning_runs/qwen2.5_14b/full/full_20260821T051022Z_d81f78bf/` | Full 360 runs, 94.4% | Fig. 3; Table 1 |
| `runs/no_capability_semantics/…368b8574/` | `data/planning_runs/qwen2.5_14b/no_capability_semantics/…368b8574/` | 56.7% | Fig. 3 |
| `runs/no_parameter_semantics/…d8471060/` | `data/planning_runs/qwen2.5_14b/no_parameter_semantics/…d8471060/` | 64.7% | Fig. 3 |
| `runs/no_structure/…6bf8d7c2/` | `data/planning_runs/qwen2.5_14b/no_structure/…6bf8d7c2/` | 93.1% (No JSON Schema) | Fig. 3 |
| `runs/reduced_capability_info/…6e0dd3ea/` | `data/planning_runs/qwen2.5_14b/reduced_capability_info/…6e0dd3ea/` | 48.6% (reference) | Fig. 3 |
| `configs/conditions/*.json` | `configs/conditions/` | Condition flags | Fig. 3; AQ5 |
| `reports/fig3_semantic_task_clustered_stats.{json,md}` | `figures/source_data/` | Semantic-task clustered inference | Fig. 3 |
| `reports/ablation_stats.json` | `figures/source_data/ablation_stats.json` | Instruction-clustered ablation stats | Fig. 3 |

## Cross-model supplementary

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `runs/multimodel_v1.1/{phi4_14b,gemma3_12b,mistral_nemo_12b}/<condition>/` | `data/planning_runs/cross_model/<model>/<condition>/` | Formal 360-run cells | Supp. Fig. S1; AQ18 |
| `runs/multimodel_v1.1/qwen2.5_14b/<condition>/REUSED.json` | `data/planning_runs/cross_model/qwen2.5_14b/<condition>/` | Documents Qwen reuse | Supp. Fig. S1 |
| `outputs/multimodel_v1.1/cross_model_*.csv` | `figures/source_data/` | Family and effect tables | Supp. Fig. S1 |
| `outputs/multimodel_v1.1/REPORT.md` | `figures/source_data/REPORT.md` | Cross-model write-up | Supplementary |
| `configs/models/*.yaml` | `configs/models/` | Model tags | AQ5; Supplementary |

## Gate replay (Fig. 4; AQ7)

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `configs/safety_gate.json` | `configs/safety_gate.json` | Frozen gate v1.0 | Fig. 4; AQ7 |
| `src/safety_gate.py` | `code/gate/safety_gate.py` and `src/safety_gate.py` | Official gate | Fig. 4 |
| `runs/safety_gate/full_gate_off_replay_20260824T113519Z/` | `data/gate_replay/full_gate_off_replay_20260824T113519Z/` | Gate-off decisions | Fig. 4 |
| `runs/safety_gate/full_gate_on_20260824T113519Z/` | `data/gate_replay/full_gate_on_20260824T113519Z/` | Gate-on decisions | Fig. 4 |
| `reports/fig4_gate_replay_records.csv` | `data/gate_replay/` and `figures/source_data/` | 360-row replay table; 6 false blocks; 20 P6 | Fig. 4 |
| `reports/safety_gate_results.json` | `data/gate_replay/` and `figures/source_data/` | 20/20 intercept; 6/240 false block | Fig. 4 |
| `reports/fig4_clustered_stats.{json,md}` | `data/gate_replay/` and `figures/source_data/` | Cluster-aware tests | Fig. 4 |
| `outputs/multimodel_v1.1/cross_model_safety_gate.csv` | `data/gate_replay/cross_model/` and `figures/source_data/figure_s2_gate.csv` | Official S2 gate | Supp. Fig. S2 |
| `outputs/multimodel_v1.1/safety_gate_traces/` | `data/gate_replay/cross_model/safety_gate_traces/` | Per-model traces | Supp. Fig. S2 |

## Recovery (Fig. 5 / Extended Data Table 3)

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `runs/multifailure_recovery/…formal_20260824T155611Z_9fde5231/` | `data/recovery/formal_F1-F7_20260824T155611Z_9fde5231/` | F1–F5 + original F7 | Fig. 5; ED Table 3 |
| `runs/multifailure_recovery/…f6_rerun_20260825T021403Z_306314eb/` | `data/recovery/f6_rerun_20260825T021403Z_306314eb/` | Paper F6 (4/12) | Fig. 5; ED Table 3 |
| `runs/multifailure_recovery/…f7_corrected_20260930T051720Z_883abfac/` | `data/recovery/f7_corrected_20260930T051720Z_883abfac/` | Later F7 10/16 | AQ / correction note |
| `reports/multifailure_recovery_final_summary.{json,md}` | `data/recovery/` and `figures/source_data/` | Frozen 74/92 F1–F6 report (F7 line in that file is superseded) | Fig. 5 F1–F6 |
| `runs/multifailure_recovery/…f7_corrected_…/` | `data/recovery/f7_corrected_20260930T051720Z_883abfac/` | Manuscript F7 = 10/16 | ED Table 3 F7 |
| `configs/recovery/*.json` | `configs/recovery/` | Injection configs | Fig. 5 |
| `src/recovery/*` | `code/recovery/` and `src/recovery/` | Injection + replan | Fig. 5 |

## Code (exact copies)

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `src/catalog.py` | `code/planner/catalog.py` | Prompt, catalogue, JSON Schema | Methods; AQ5 |
| `src/runner/*` | `code/planner/runner/` | Planner runner | Methods |
| `src/adapters/production_plan_adapter.py` | `code/adapters/` | Runtime plan adapter | Methods |
| `src/adapters/semantic_normalizer.py` | `code/adapters/` | Semantic normalizer | Methods |
| `src/evaluator/plan_validator.py` | `code/evaluator/` | Deterministic validator | Methods |
| `src/evaluator/task_success.py` | `code/evaluator/` | Task-success evaluator | Methods |
| `src/safety_gate.py` | `code/gate/safety_gate.py` | Official gate | Fig. 4 |
| `src/recovery/*` | `code/recovery/` | Failure injection | Fig. 5 |
| `src/analysis/statistics.py` | `code/analysis/statistics.py` | Clustered bootstrap / McNemar / Holm | Fig. 3–4 |
| `src/analysis/multimodel.py` | `code/analysis/multimodel.py` | Cross-model tables | Supp. Fig. S1 |
| `scripts/analyze_fig3_semantic_task_clustered.py` | `scripts/_original/` | Fig. 3 clustered recompute | Fig. 3 |
| `scripts/analyze_fig4_clustered_stats.py` | `scripts/_original/` | Fig. 4 clustered recompute | Fig. 4 |
| `scripts/run_safety_gate_evaluation.py` | `scripts/_original/` | Fig. 4 replay | Fig. 4 |
| `scripts/run_qc.py` | `scripts/_original/run_qc.py` | QC implementation | Methods |
| `tests/test_*.py` | `tests/` | Evaluator / gate / recovery tests | Reproducibility |

## Speech and physical trials

| Original source path | Release path | Purpose | Manuscript target |
|---|---|---|---|
| `~/.ros/log/python_3417980_1787580797685.log` | `data/speech/raw_log_01.txt` | STT latency log, 2026-08-24 (75 `stt_ms`/`e2e_ms` lines) | AQ11; physical-window speech |
| `~/.ros/log/python_4155467_1787644803083.log` | `data/speech/raw_log_02.txt` | STT latency log, 2026-08-25 (92 `stt_ms`/`e2e_ms` lines) | AQ11; physical-window speech |
| `humanoid-g1-natural-language-paper/e2erecord/G1_physical_e2e_20x5_results.csv` | `data/physical_trials/` and `figures/source_data/` | Scored 20×5 physical E2E table (79/100 Success) | 100-trial analysis; AQ12 |
| `…/e2erecord/rawdata.jsonl` | `data/physical_trials/rawdata.jsonl` | On-robot plan/exec traces | 100-trial supporting log |
| `…/e2erecord/rawdate.log` | `data/physical_trials/rawdate.log` | Controller log; local paths rewritten in the copy | 100-trial supporting log |

## Intentionally not released (see RELEASE_AUDIT.md)

- `runs/smoke/`, `runs/multimodel_v1.1/_smoke/`
- Unofficial 2026-09-07 Qwen reruns
- `runs/archive/v1.0/`
- `raw_outputs.jsonl`
- `src/gate/safety_gate.py` (non-Fig. 4 prototype)
- `outputs/multimodel_v1.1/cross_model_gate_summary.csv` (prototype-gate table; not Fig. S2)
- Model weights, virtualenvs, `__pycache__`, ROS workspaces
- Duplicate tree `g1-hri-open-source/g1-planning-benchmark/`
- Informal `~/g1_rag` conversation logs and `~/.ros/log`
