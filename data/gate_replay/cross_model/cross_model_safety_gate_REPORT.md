# Supplementary cross-model execution-time safety-gate experiment

Protocol: official Figure 4 / Section 4.3 implementation
`scripts/run_safety_gate_evaluation.py` + `src.safety_gate.evaluate_safety_gate` + `configs/safety_gate.json`.

No LLM calls. Saved Full-condition planner outputs only. Gate configuration is identical across models.

Qwen2.5-14B is **not** re-run. Its row is the main-text formal result.

## Summary table

| Model | P6 | Before | After | Intercepted | Valid controls | Pass | False-block |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 | 60 | 20/60 (33.3%) | 0/60 (0.0%) | 20/20 | 240 | 234/240 (97.5%) | 6/240 (2.5%) |
| Phi-4 | 60 | 0/60 (0.0%) | 0/60 (0.0%) | n/a | 240 | 234/240 (97.5%) | 6/240 (2.5%) |
| Gemma 3 | 60 | 0/60 (0.0%) | 0/60 (0.0%) | n/a | 237 | 231/237 (97.5%) | 6/237 (2.5%) |
| Mistral NeMo | 60 | 48/60 (80.0%) | 0/60 (0.0%) | 48/48 | 240 | 234/240 (97.5%) | 6/240 (2.5%) |

- Qwen McNemar p = 1.9e-6 (main text).
- Mistral NeMo McNemar p = 7.1e-15 (exact binomial two-sided, official script).
- Phi-4 / Gemma 3: no erroneous physical dispatch in Full-condition P6 outputs; McNemar not computed.

## Protocol notes

- Control-set rule is the official script rule: non-P6, gold `required_steps` contain a physical capability, and the saved plan contains a physical step. Task-success is **not** an extra filter (this is how Figure 4 obtained n = 240 for Qwen).
- Gemma 3 valid-control n = 237 because `P3_02_paraphrase` (all 3 repetitions) saved a speak-only plan (`Okay.`) with no physical step, so those 3 runs are excluded by the official rule. This is a planner-output difference, not a protocol change.
- The 6 false blocks are the same 6 runs on all four models: `P3_05_paraphrase` × 3 and `P3_06_paraphrase` × 3 (`Rotate to the left/right...`), reason `unregistered_physical_request`. This is expected under the fixed alias list.
- Previous `cross_model_gate_summary.csv` used `src.gate.safety_gate` (capability-regex only) and must **not** be used for Supplementary Figure S2.

## Output files

- `outputs/multimodel_v1.1/cross_model_safety_gate.csv`
- `outputs/multimodel_v1.1/figure_s2_gate.csv`
- `outputs/multimodel_v1.1/safety_gate_traces/{phi4,gemma3,mistral_nemo}_safety_gate.json`
