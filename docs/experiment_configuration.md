# Experiment configuration

Manuscript: “Action and parameter meanings shape language grounding in a
physical humanoid”.

This file records the software and planner settings that support Author
Queries AQ5, AQ7, AQ11, AQ12 and AQ18.

Run-embedded metadata (present in released `config.json` / records) is
distinguished from the retained-workstation capture in
`docs/retained_environment_manifest.md`. The 21 August 2026 Full
`config.json` stores `condition.model = "qwen2.5:14b"` only. That does not
prevent reporting the retained OS, Python, ROS 2, Ollama, quantization, and
hashes; those values were not written into the run file.

## Software environment

| Component | Version / setting | Source |
|---|---|---|
| OS | Ubuntu 20.04.6 LTS | Retained workstation (see environment manifest) |
| Python | 3.8.10 | Retained workstation |
| Robot middleware | ROS 2 Foxy | Retained workstation |
| Local LLM server | Ollama 0.13.2 | Retained workstation client |
| Primary planner | Qwen2.5-14B Instruct, Q4_K_M, tag `qwen2.5:14b` | Tag in `config.json`; Q4_K_M and SHA256 from retained local GGUF |
| Planner temperature | 0 (hard-coded) | Condition configs / runner |
| Repeats / seed | 3 repeats, seed 0 | Condition configs / runner |
| Endpoint | `http://127.0.0.1:11434/api/chat` | Model YAML / runner |

## Planner interface (held fixed)

- Capability catalogue: `configs/capability_catalog.json` **v1.1**
- Prompt construction and runtime JSON Schema: `code/planner/catalog.py`
  (`planner_prompt`, `build_planner_json_schema`)
- Production-plan adapter: `code/adapters/production_plan_adapter.py`
- Semantic normalizer: `code/adapters/semantic_normalizer.py`
- Deterministic validator: `code/evaluator/plan_validator.py`
- Task-success evaluator: `code/evaluator/task_success.py`
- Conditions: `configs/conditions/*.json`

Full-plan normalization used for scoring: JSON strip, direction strip, numeric
coerce. The Full evaluator does **not** case-fold, punctuation-canonicalize, or
alias-canonicalize physical capabilities.

## Cross-model (supplementary)

Same catalogue, prompts, schema (when enabled), evaluator, temperature 0, seed
0, 360 runs / cell.

| Model | Ollama tag | Formal records |
|---|---|---|
| Qwen2.5-14B | `qwen2.5:14b` | Reused main-text runs (`REUSED.json`) |
| Phi-4 14B | `phi4:14b` | `data/planning_runs/cross_model/phi4_14b/` |
| Gemma 3 12B | `gemma3:12b` | `data/planning_runs/cross_model/gemma3_12b/` |
| Mistral NeMo 12B | `mistral-nemo:12b` | `data/planning_runs/cross_model/mistral_nemo_12b/` |

## Safety / dispatch gate (Fig. 4; AQ7)

- Implementation: `code/gate/safety_gate.py` (original `src/safety_gate.py`)
- Frozen config: `configs/safety_gate.json` **v1.0**
- Rules: whitelist aliases from the v1.1 catalogue; locomotion `duration ≥ 0.1`
  when duration is present; direction and state enums
- Held-plan replay of Qwen Full outputs: `data/gate_replay/`
- Not used for Fig. 4: the prototype `src/gate/safety_gate.py` (excluded)

## Recovery / failure injection (Fig. 5)

- Logic: `code/recovery/`
- Configs: `configs/recovery/multifailure_formal.json`,
  `multifailure_f6_rerun.json`, `multifailure_f7_corrected.json`
- Labels F1–F7 preserved; 92 applicable F1–F6 episodes vs 16 F7 episodes
- Manuscript F7 cell: **10/16** correct termination
  (`data/recovery/f7_corrected_20260930T051720Z_883abfac`)
- See `data/recovery/README.md`

## Audio / VAD (AQ5, AQ11) — source-code settings plus raw STT logs

Recorded from the on-robot ear node (`audio_bridge_node.py`):

- Input: Unitree UDP PCM, **16 kHz**, **mono**, **PCM16**
- VAD: local Silero ONNX
- Speech recognition: Faster-Whisper (weights not redistributed)
- Raw STT latency logs from the 24–25 August physical window:
  `data/speech/raw_log_01.txt`, `data/speech/raw_log_02.txt`
- No saved Extended Data Table 2 aggregation was found
  (see `data/speech/README.md`)

## Controller (AQ5, AQ12) — source-code settings, not 100-trial data

- Predefined arm actions: ROS 2 topic `/g1_arm_action_cmd` (`Int32` action id)
- Locomotion: `g1_loco_client --set_velocity` with linear speed **v = 0.2**
- State: `--damp`, `--stand_up`, `--start` (`start` → FSM 501)
- Structured 100-trial records: `data/physical_trials/G1_physical_e2e_20x5_results.csv`
  (20 × 5; **79/100** Success from the `Final result` column)

## What is not redistributed

- Model weights (Qwen / Phi / Gemma / Mistral / Whisper / Silero)
- Virtual environments
- Ollama blobs
- API keys (none are required for statistic reproduction)
