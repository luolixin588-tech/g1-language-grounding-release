# Action and parameter meanings shape language grounding in a physical humanoid

Public companion repository for the manuscript
**“Action and parameter meanings shape language grounding in a physical humanoid”**.

It contains the frozen v1.1 G1 planning benchmark, the formal Qwen2.5 and
cross-model planning records, the Fig. 4 safety-gate replay, the Fig. 5
injected-failure records, the evaluator / gate / recovery code, and scripts
that recompute the published statistics from those records.

Model weights are **not** redistributed.

## Repository structure

```
g1-language-grounding-release/
├── code/                    # planner, evaluator, adapters, gate, recovery, analysis
├── src/                     # same modules under the original `src.*` import paths
├── configs/                 # catalogue v1.1, conditions, safety_gate.json v1.0
├── data/
│   ├── benchmark/           # 60 tasks, 120 instructions, schema, QC
│   ├── planning_runs/       # Qwen formal + cross-model formal runs
│   ├── gate_replay/         # Fig. 4 held-plan decisions
│   ├── recovery/            # F1–F7 injected-failure records
│   ├── speech/              # raw STT latency logs (24–25 Aug physical window)
│   └── physical_trials/     # 20×5 = 100 on-robot trials (CSV + raw traces)
├── scripts/                 # supported reproduce_* / data_qc; _original/ is provenance only
├── figures/source_data/     # CSVs / JSON used for figures and AQ18
├── docs/                    # protocol, configuration, retained environment, manifest
├── tests/
├── RELEASE_AUDIT.md
└── requirements.txt
```

See `docs/repository_manifest.md` for the original → release path map.

## Software environment

The formal 21 August 2026 Qwen `config.json` stores the Ollama tag
`qwen2.5:14b` only. The workstation and local model were kept after data
collection; the following values were queried from that retained environment
and are **not** run-embedded fields. Full capture:
`docs/retained_environment_manifest.md`.

| Component | Value | Provenance |
|---|---|---|
| OS | Ubuntu 20.04.6 LTS | Retained workstation |
| Python | 3.8.10 | Retained workstation (`/usr/bin/python3.8`) |
| Robot middleware | ROS 2 Foxy | Retained workstation |
| Local LLM server | Ollama 0.13.2 | Retained workstation client |
| Primary planner | `qwen2.5:14b`, Q4_K_M | Tag in `config.json`; quantization and hashes from the retained local model |

Statistic reproduction from the released records needs only Python 3.8 and
the standard library. `requests` is required only if you re-run the planner
against a local Ollama server.

```bash
python3 -m pip install -r requirements.txt
```

## Inspect the benchmark

```bash
# 60 semantic tasks and 120 instructions
wc -l data/benchmark/semantic_tasks.jsonl data/benchmark/instructions_120.jsonl

# schema + family notes
less data/benchmark/README.md
less docs/benchmark_protocol.md

# QC (expected PASS: 60/120, 10 tasks per family P1–P6)
PYTHONPATH=. python3 scripts/data_qc.py
```

Task families: P1 spoken response, P2 single physical action, P3 parameterized
control, P4 multi-step physical sequence, P5 mixed speech–action, P6
unsupported / constrained request.

## Reproduce the main planning statistics (Fig. 3)

No Ollama call. Recomputes Task Success, Plan Validity, family rates, and
all-3 instruction counts from the five formal Qwen2.5 runs and checks them
against frozen `summary.json` files.

```bash
PYTHONPATH=. python3 scripts/reproduce_main_results.py
```

Expected Full headline: **340/360 = 94.4%** Task Success.

## Reproduce gate and recovery summaries (Fig. 4–5)

```bash
PYTHONPATH=. python3 scripts/reproduce_supplementary_results.py
```

This reprints:

- Fig. 4: 20/20 P6 intercepts; 6/240 false blocks (`P3_05` / `P3_06` paraphrase)
- Fig. 5 / ED Table 3: 74/92 recoverable F1–F6; F7 **10/16** correct termination
  (prompt-consistent rerun)

Gate config is `configs/safety_gate.json` v1.0. Records are under
`data/gate_replay/` and `data/recovery/`.

`scripts/reproduce_*.py` are the supported reproduction entry points.
`scripts/_original/` contains archived analysis scripts retained for
provenance and may reference the original internal directory layout.

## Reproduce supplementary cross-model results (Fig. S1; AQ18)

The same command also recomputes four-model × five-condition family rates
from `data/planning_runs/cross_model/` (Qwen cells reuse
`data/planning_runs/qwen2.5_14b/`).

AQ18, verified from Mistral NeMo records:

- Full P2 = **54/60 = 90.0%**
- No parameter semantics P2 = **60/60 = 100.0%**

An older Supplementary Fig. S1 plotting source that showed P2_param = 90.0
for Mistral NeMo was incorrect. The corrected value is 100.0%. See
`figures/source_data/AQ18_mistral_p2_verification.md`.

## Model weights

This repository does **not** include Ollama blobs, GGUF files, Faster-Whisper
weights, or Silero ONNX. Re-running planners requires a local Ollama install
and the tags listed in `configs/models/`. Recomputing published numbers does
not.

## Speech and physical trials

Structured 100-trial physical records are in `data/physical_trials/`.
The scored table is `G1_physical_e2e_20x5_results.csv` (20 tasks × 5
repeats). Success counted from `Final result` is **79/100**. See that
directory’s README.

Raw Faster-Whisper latency logs from the 24–25 August physical-trial
window are in `data/speech/` (`raw_log_01.txt`, `raw_log_02.txt`). They
are runtime `g1_audio_bridge` lines, not a saved Table 2 aggregation.

## Citation

Please cite the manuscript
“Action and parameter meanings shape language grounding in a physical humanoid”.

## License

MIT. See `LICENSE`.
