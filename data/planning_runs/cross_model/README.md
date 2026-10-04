# Formal cross-model planning runs

Four local Ollama models, identical planner interface, frozen v1.1 benchmark.

| Model id | Ollama tag | Paper name |
|---|---|---|
| `qwen2.5_14b` | `qwen2.5:14b` | Qwen2.5-14B |
| `phi4_14b` | (see `configs/models/phi4_14b.yaml`) | Phi-4 14B |
| `gemma3_12b` | (see `configs/models/gemma3_12b.yaml`) | Gemma 3 12B |
| `mistral_nemo_12b` | (see `configs/models/mistral_nemo_12b.yaml`) | Mistral NeMo 12B |

Conditions per model: Full, reduced capability information, no capability
semantics, no parameter semantics, no schema-constrained decoding
(`no_structure` in the original tree).

Qwen2.5 cells **reuse** the formal main-text runs. See each
`qwen2.5_14b/<condition>/REUSED.json`. Records are not duplicated here; they
live under `../qwen2.5_14b/`.

Smoke directories (`_smoke/`) and unofficial 2026-09-07 Qwen reruns are
excluded.

## AQ18 (Mistral NeMo P2)

Verified from `mistral_nemo_12b/full/records.jsonl` and
`mistral_nemo_12b/no_parameter_semantics/records.jsonl`:

- Full P2 = 54/60 = 90.0%
- No parameter semantics P2 = 60/60 = 100.0%

The old Supplementary Fig. S1 plotting source that showed P2_param = 90.0 for
Mistral NeMo was incorrect. The corrected cell is 100.0%. See
`figures/source_data/AQ18_mistral_p2_verification.md` and `RELEASE_AUDIT.md`.
