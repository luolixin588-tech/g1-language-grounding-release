# Release audit

Companion release for
“Action and parameter meanings shape language grounding in a physical humanoid”.

Original working tree `g1-planning-benchmark` was **copied, not moved or
edited**.

This file reports inclusions, exclusions, missing manuscript-supporting
files, conflicting result sources, and whether headline numbers can be
traced to released records. Numeric records were not silently corrected.

## What was included

- Frozen v1.1 benchmark: 60 semantic tasks, 120 instructions, gold specs,
  `task_spec.schema.json`, QC protocol and PASS report.
- Formal Qwen2.5-14B Full and four ablation runs (2026-08-21 / 08-24),
  `config.json` + `records.jsonl` + `summary.json` only.
- Formal cross-model runs for Phi-4, Gemma 3, and Mistral NeMo (five
  conditions each). Qwen cells documented via `REUSED.json` pointing at the
  main-text runs.
- Fig. 4 held-plan gate-off / gate-on records, frozen `safety_gate.json` v1.0,
  360-row replay CSV, six false-block cases, 20 P6 erroneous-dispatch cases,
  official cross-model gate traces (`src.safety_gate`).
- Recovery: formal 112-episode file (F1–F5 for the paper pool), F6 rerun,
  and the prompt-consistent F7 rerun used as the manuscript F7 cell
  (**10/16**). F1–F7 labels preserved.
- Exact planner / catalogue / schema / adapter / normalizer / validator /
  evaluator / official gate / recovery / statistics / figure-source scripts.
- Tests for evaluator, gate, catalogue ablations, and recovery.
- Figure source CSVs/JSON for Fig. 3, Fig. 4, Fig. 5, Supp. Fig. S1/S2.
- Physical 20×5 (100-trial) scored CSV and supporting on-robot traces.

## What was excluded

- Model weights (Ollama GGUF, Faster-Whisper, Silero ONNX).
- Virtual environments, `__pycache__`, build / install / devel trees.
- `raw_outputs.jsonl` (duplicate raw dumps; `records.jsonl` already stores
  `raw_model_output`).
- Smoke tests: `runs/smoke/`, `runs/multimodel_v1.1/_smoke/`, recovery smoke.
- Obsolete / unofficial planning runs:
  - 2026-09-07 Qwen reruns (including the ~80% Full cells)
  - `runs/archive/v1.0/`
  - `runs/no_action_description/`, `runs/no_parameter_definition/`
  - `runs/robustness/`, `runs/ablations/`
  - early single-failure `runs/recovery/recovery_20260821T070000Z_3b9f2763`
    (not the Fig. 5 multifailure table)
- Prototype gate `src/gate/safety_gate.py` (not Fig. 4).
- Prototype-gate table `outputs/multimodel_v1.1/cross_model_gate_summary.csv`
  (and `.json`); kept in the internal tree only, not in public
  `figures/source_data/`.
- Credentials / API keys (none were present in formal configs).
- Duplicate graduation / open-source trees (`g1-hri-open-source/…`).
- Informal `~/g1_rag` conversation logs and the remaining `~/.ros/log`
  files (local identifiers). Two physical-window STT logs were copied to
  `data/speech/`.

## Missing manuscript-supporting files

| Expected for | Status |
|---|---|
| Extended Data Table 2 anonymized speech outputs | **No saved Table 2 file.** Raw STT latency logs from the 24–25 August physical window are in `data/speech/`. |
| 100-trial physical-trial records | **Added** from `e2erecord/` (operator workstation): scored CSV + raw jsonl/log. Headline from CSV `Final result`: **79/100** Success. |
| Old Supplementary Fig. S1 plotting source with `P2_param = 90.0` | **Missing as a file.** Current records and `cross_model_task_categories.csv` already store the corrected 100.0% cell. |
| Formal-run Ollama version / GGUF digest / Q4_K_M in 21 August 2026 `config.json` | **Not run-embedded.** Reported instead from the retained workstation and local model in `docs/retained_environment_manifest.md`. |
| Publisher Author Query document (AQ5/7/11/12/18) | **Not in the working tree.** Support is via records + this audit. |

Physical-trial records are in `data/physical_trials/`. Speech Table 2 is
still not a saved aggregation; raw STT logs are in `data/speech/`.

## Duplicated or conflicting result sources

1. **Qwen formal vs later 2026-09-07 reruns.** The later reruns were excluded.
   Only 2026-08-21/24 formal runs are released. Do not mix them.

2. **Qwen main-text runs vs `runs/multimodel_v1.1/qwen2.5_14b/`.** Cross-model
   Qwen cells reuse the main-text runs (`REUSED.json`). Records are released
   once under `data/planning_runs/qwen2.5_14b/`.

3. **Fig. 4 official gate vs prototype gate.**
   - Official: `src/safety_gate.py` + `configs/safety_gate.json` v1.0
     → `data/gate_replay/`, `figure_s2_gate.csv`, `cross_model_safety_gate.csv`.
   - Prototype: `src/gate/safety_gate.py` produced
     `outputs/multimodel_v1.1/cross_model_gate_summary.csv` in the internal
     tree. That file is **not** in the public `figures/source_data/` directory.
     It was not numerically altered.

4. **F6 original vs F6 rerun.** Formal F6 recovery_success = 2/12 applicable.
   Paper pool uses the F6 rerun = 4/12. Both runs are released. The 92-episode
   aggregate 74/92 uses F1–F5 from the formal run + F6 from the rerun.

5. **F7.** Manuscript / public-release F7 is the prompt-consistent rerun
   **10/16**. The formal 112-episode file still contains the earlier F7 rows
   (6/16, prompt conflict). Those rows were not rewritten; they are not the
   manuscript F7 cell. The copied `multifailure_recovery_final_summary.md`
   still prints 6/16 and was left unchanged as a frozen report.

6. **Instruction-clustered vs semantic-task-clustered Fig. 3 stats.** Both
   source files are released. Point estimates (360-run rates) agree; CIs
   differ by clustering unit. Neither file was rewritten.

## AQ18 (explicit verification)

From `data/planning_runs/cross_model/mistral_nemo_12b/`:

| Cell | Count | Rate |
|---|---:|---:|
| Mistral NeMo Full P2 | 54/60 | 90.0% |
| Mistral NeMo No parameter semantics P2 | 60/60 | 100.0% |

`figures/source_data/cross_model_task_categories.csv` matches these rates
(0.9 and 1.0). The old Supplementary Fig. S1 plotting source that used
`P2_param = 90.0` for the no-parameter-semantics cell was incorrect; the
corrected value is **100.0%**. Raw records were left unchanged.

## Can headline numbers be traced to released records?

| Headline | Trace | Result |
|---|---|---|
| Qwen Full 340/360 = 94.4% | `data/planning_runs/qwen2.5_14b/full/…/records.jsonl` | Yes |
| Qwen no-cap 204/360 = 56.7% | corresponding records | Yes |
| Qwen no-param 233/360 = 64.7% | corresponding records | Yes |
| Qwen no-structure / No JSON Schema 335/360 = 93.1% | corresponding records | Yes |
| Qwen reduced 175/360 = 48.6% | corresponding records | Yes |
| Fig. 4: 20 P6 intercepts, 6/240 false blocks | `fig4_gate_replay_records.csv` | Yes |
| Fig. 5: 74/92 F1–F6 | formal F1–F5 + F6 rerun | Yes |
| ED Table 3 F7 = 10/16 | `f7_corrected_…` records | Yes |
| Superseded formal F7 = 6/16 | still inside the formal 112-episode file | Retained; not the manuscript cell |
| AQ18 Mistral P2 90.0% / 100.0% | Mistral records + CSV | Yes |
| Cross-model Full rates 94.4 / 97.5 / 95.0 / 70.0% | records | Yes |
| ED Table 2 speech latencies | `data/speech/raw_log_0{1,2}.txt` | Raw logs only; no saved Table 2 means |
| 100-trial physical Success 79/100 | `data/physical_trials/G1_physical_e2e_20x5_results.csv` | Yes |

## Release-local edits (not original files)

- `src/paths.py` / `code/planner/paths.py`: directory roots only.
- Cross-model `REUSED.json`: `source_run` remapped to a release-relative path.
- Some Markdown reports: the original local working-tree prefix
  was replaced by `<original-repo>`.
- New files: README, LICENSE, docs (including
  `docs/retained_environment_manifest.md`), reproduce scripts, this audit.
- Removed from public `figures/source_data/` only:
  `cross_model_gate_summary.csv` and `.json` (prototype gate). Internal
  originals were not modified.

`records.jsonl`, `summary.json`, and condition/gate JSON were not numerically
edited.

## Residual machine-local strings

After packaging, 17 report JSON/CSV/MD copies had the original local
working-tree prefix rewritten to `<original-repo>` (provenance only; numbers
untouched). `records.jsonl` had no home paths.

No API keys, tokens, or passwords were found. The remaining string
`<original-repo>` is a placeholder, not a machine path.

Physical-trial copies: `rawdate.log` had 19 `/home/luolixin/g1_rag/…` paths
rewritten to `<local-log-dir>/`. In the public copies only, `Valid records`
was restored from `5月5日` to `5/5` (20 rows; Success 79/100 unchanged), and
five 2026-05-25 name-lookup lines were removed from `rawdata.jsonl` (119 →
114). Incoming `/tmp` copies and the operator-workstation originals were
not modified.

Speech copies: `data/speech/raw_log_01.txt` and `raw_log_02.txt` each had
one `/home/luolixin/unitree_ros2_ws/install/my_g1_audio` path rewritten to
`<local-install-dir>`. The original `~/.ros/log` files were not modified.
