# Cross-Model Generalization Experiment

## 1 Experimental setup

Benchmark: G1 Internal Planning Benchmark **v1.1** (frozen; 120 instructions × 3 runs = 360 / condition / model).

Models (local Ollama, identical planner interface):

- Qwen2.5-14B (`qwen2.5_14b`)
- Phi-4 14B (`phi4_14b`)
- Gemma 3 12B (`gemma3_12b`)
- Mistral NeMo 12B (`mistral_nemo_12b`)

Conditions (existing formal configs; `no_schema_constrained_decoding` reuses `no_structure`;
`reduced_capability_info` is the frozen names-only catalog condition):

- full
- reduced_capability_info (coarse reference; not in Holm family)
- no_capability_semantics
- no_parameter_semantics
- no_schema_constrained_decoding

Held fixed across models: capability catalog, system/user prompt, JSON Schema (when enabled),
evaluator, Task Success / Plan Validity definitions, temperature = 0, seed = 0, 3 repeats.
No model-specific prompt, few-shot, or repair text was added.

Qwen2.5-14B formal cells reuse the frozen v1.1 runs under decision A
(model parameterization only; see `qwen_reuse_justification.json`).

## 2 Full-condition performance

| Model | Task Success | Plan Validity | Median latency | P95 latency | 3/3 instructions |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 94.4% | 100.0% | 642 ms | 1448 ms | 113 / 120 |
| Phi-4 14B | 97.5% | 100.0% | 715 ms | 1608 ms | 117 / 120 |
| Gemma 3 12B | 95.0% | 100.0% | 962 ms | 1934 ms | 114 / 120 |
| Mistral NeMo 12B | 70.0% | 100.0% | 509 ms | 1887 ms | 83 / 120 |

## 3 Planner-interface ablations

| Model | Condition | Task Success | Plan Validity | Median | P95 | 3/3 |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-14B | Full | 94.4% | 100.0% | 642 ms | 1448 ms | 113 / 120 |
| Qwen2.5-14B | Reduced capability information | 48.6% | 59.2% | 1022 ms | 4536 ms | 58 / 120 |
| Qwen2.5-14B | No capability semantics | 56.7% | 100.0% | 615 ms | 1440 ms | 68 / 120 |
| Qwen2.5-14B | No parameter semantics | 64.7% | 88.3% | 862 ms | 2449 ms | 77 / 120 |
| Qwen2.5-14B | No schema-constrained decoding | 93.1% | 100.0% | 627 ms | 1689 ms | 111 / 120 |
| Phi-4 14B | Full | 97.5% | 100.0% | 715 ms | 1608 ms | 117 / 120 |
| Phi-4 14B | Reduced capability information | 50.8% | 67.5% | 914 ms | 4429 ms | 61 / 120 |
| Phi-4 14B | No capability semantics | 60.0% | 100.0% | 712 ms | 1386 ms | 72 / 120 |
| Phi-4 14B | No parameter semantics | 72.5% | 99.2% | 708 ms | 2010 ms | 87 / 120 |
| Phi-4 14B | No schema-constrained decoding | 96.7% | 100.0% | 748 ms | 1784 ms | 116 / 120 |
| Gemma 3 12B | Full | 95.0% | 100.0% | 962 ms | 1934 ms | 114 / 120 |
| Gemma 3 12B | Reduced capability information | 48.3% | 68.1% | 1084 ms | 6292 ms | 58 / 120 |
| Gemma 3 12B | No capability semantics | 60.0% | 100.0% | 923 ms | 1719 ms | 72 / 120 |
| Gemma 3 12B | No parameter semantics | 68.3% | 100.0% | 956 ms | 1986 ms | 82 / 120 |
| Gemma 3 12B | No schema-constrained decoding | 94.4% | 99.2% | 1119 ms | 2974 ms | 113 / 120 |
| Mistral NeMo 12B | Full | 70.0% | 100.0% | 509 ms | 1887 ms | 83 / 120 |
| Mistral NeMo 12B | Reduced capability information | 35.0% | 59.2% | 1022 ms | 3775 ms | 41 / 120 |
| Mistral NeMo 12B | No capability semantics | 53.6% | 100.0% | 477 ms | 1359 ms | 64 / 120 |
| Mistral NeMo 12B | No parameter semantics | 52.5% | 85.0% | 503 ms | 2394 ms | 63 / 120 |
| Mistral NeMo 12B | No schema-constrained decoding | 68.1% | 100.0% | 502 ms | 1566 ms | 81 / 120 |

## 4 Ablation effects

Delta = TaskSuccess(ablation) − TaskSuccess(full). Instruction-clustered bootstrap 95% CI;
McNemar on instruction-level all-3 success; Holm correction within each model **only** for
the three controlled ablations (capability / parameter / schema).

| Model | Effect | Δ Task Success (pp) | 95% CI (pp) | raw p | Holm p |
|---|---|---:|---|---:|---:|
| Qwen2.5-14B | No capability semantics | -37.8 | [-46.9, -28.6] | 4.356e-12 | 1.307e-11 |
| Qwen2.5-14B | No parameter semantics | -29.7 | [-38.9, -20.6] | 5.632e-09 | 1.126e-08 |
| Qwen2.5-14B | No schema-constrained decoding | -1.4 | [-3.6, +0.0] | 0.5 | 0.5 |
| Phi-4 14B | No capability semantics | -37.5 | [-46.7, -29.2] | 6.821e-13 | 2.046e-12 |
| Phi-4 14B | No parameter semantics | -25.0 | [-34.2, -16.7] | 2.272e-07 | 4.544e-07 |
| Phi-4 14B | No schema-constrained decoding | -0.8 | [-2.5, +0.0] | 1 | 1 |
| Gemma 3 12B | No capability semantics | -35.0 | [-43.3, -26.7] | 4.547e-13 | 1.364e-12 |
| Gemma 3 12B | No parameter semantics | -26.7 | [-35.0, -19.2] | 4.657e-10 | 9.313e-10 |
| Gemma 3 12B | No schema-constrained decoding | -0.6 | [-4.2, +3.1] | 1 | 1 |
| Mistral NeMo 12B | No capability semantics | -16.4 | [-25.6, -7.5] | 0.001878 | 0.005635 |
| Mistral NeMo 12B | No parameter semantics | -17.5 | [-27.5, -7.5] | 0.002887 | 0.005774 |
| Mistral NeMo 12B | No schema-constrained decoding | -1.9 | [-4.7, +0.0] | 0.5 | 0.5 |

### Coarse reference: Reduced capability information

Not included in the Holm family. Effect size and instruction-clustered bootstrap 95% CI only.

| Model | Task Success | Δ vs Full (pp) | 95% CI (pp) | Plan Validity | 3/3 |
|---|---:|---:|---|---:|---:|
| Qwen2.5-14B | 48.6% (175/360) | -45.8 | [-55.0, -36.7] | 59.2% | 58 / 120 |
| Phi-4 14B | 50.8% (183/360) | -46.7 | [-55.8, -37.5] | 67.5% | 61 / 120 |
| Gemma 3 12B | 48.3% (174/360) | -46.7 | [-55.8, -37.5] | 68.1% | 58 / 120 |
| Mistral NeMo 12B | 35.0% (126/360) | -35.0 | [-45.0, -25.0] | 59.2% | 41 / 120 |

### Cross-model Task Success summary

Non-Full cells show Task Success followed by Δ vs Full in percentage points.

| Model | Full | Reduced capability information | No capability semantics | No parameter semantics | No schema-constrained decoding |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 94.4% | 48.6% (-45.8) | 56.7% (-37.8) | 64.7% (-29.7) | 93.1% (-1.4) |
| Phi-4 14B | 97.5% | 50.8% (-46.7) | 60.0% (-37.5) | 72.5% (-25.0) | 96.7% (-0.8) |
| Gemma 3 12B | 95.0% | 48.3% (-46.7) | 60.0% (-35.0) | 68.3% (-26.7) | 94.4% (-0.6) |
| Mistral NeMo 12B | 70.0% | 35.0% (-35.0) | 53.6% (-16.4) | 52.5% (-17.5) | 68.1% (-1.9) |

## 5 Task-family effects

| Model | Condition | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | Full | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 66.7% |
| Qwen2.5-14B | Reduced capability information | 100.0% | 0.0% | 100.0% | 0.0% | 35.0% | 56.7% |
| Qwen2.5-14B | No capability semantics | 100.0% | 5.0% | 100.0% | 40.0% | 40.0% | 55.0% |
| Qwen2.5-14B | No parameter semantics | 100.0% | 100.0% | 25.0% | 40.0% | 65.0% | 58.3% |
| Qwen2.5-14B | No schema-constrained decoding | 95.0% | 100.0% | 100.0% | 100.0% | 96.7% | 66.7% |
| Phi-4 14B | Full | 95.0% | 100.0% | 100.0% | 100.0% | 90.0% | 100.0% |
| Phi-4 14B | Reduced capability information | 100.0% | 0.0% | 80.0% | 0.0% | 30.0% | 95.0% |
| Phi-4 14B | No capability semantics | 100.0% | 0.0% | 100.0% | 40.0% | 40.0% | 80.0% |
| Phi-4 14B | No parameter semantics | 100.0% | 100.0% | 25.0% | 40.0% | 70.0% | 100.0% |
| Phi-4 14B | No schema-constrained decoding | 95.0% | 100.0% | 100.0% | 100.0% | 85.0% | 100.0% |
| Gemma 3 12B | Full | 100.0% | 100.0% | 90.0% | 100.0% | 80.0% | 100.0% |
| Gemma 3 12B | Reduced capability information | 100.0% | 0.0% | 75.0% | 0.0% | 30.0% | 85.0% |
| Gemma 3 12B | No capability semantics | 100.0% | 0.0% | 90.0% | 40.0% | 40.0% | 90.0% |
| Gemma 3 12B | No parameter semantics | 100.0% | 95.0% | 20.0% | 35.0% | 60.0% | 100.0% |
| Gemma 3 12B | No schema-constrained decoding | 100.0% | 100.0% | 91.7% | 95.0% | 85.0% | 95.0% |
| Mistral NeMo 12B | Full | 80.0% | 90.0% | 100.0% | 75.0% | 55.0% | 20.0% |
| Mistral NeMo 12B | Reduced capability information | 80.0% | 0.0% | 80.0% | 0.0% | 23.3% | 26.7% |
| Mistral NeMo 12B | No capability semantics | 90.0% | 15.0% | 100.0% | 40.0% | 36.7% | 40.0% |
| Mistral NeMo 12B | No parameter semantics | 85.0% | 100.0% | 25.0% | 30.0% | 45.0% | 30.0% |
| Mistral NeMo 12B | No schema-constrained decoding | 75.0% | 90.0% | 100.0% | 70.0% | 55.0% | 18.3% |

P2 / P3 pattern (observed, not assumed):

- P2 + no capability semantics / Qwen2.5-14B: Full 100.0% → 5.0% (Δ -95.0 pp)
- P2 + no capability semantics / Phi-4 14B: Full 100.0% → 0.0% (Δ -100.0 pp)
- P2 + no capability semantics / Gemma 3 12B: Full 100.0% → 0.0% (Δ -100.0 pp)
- P2 + no capability semantics / Mistral NeMo 12B: Full 90.0% → 15.0% (Δ -75.0 pp)
- P3 + no parameter semantics / Qwen2.5-14B: Full 100.0% → 25.0% (Δ -75.0 pp)
- P3 + no parameter semantics / Phi-4 14B: Full 100.0% → 25.0% (Δ -75.0 pp)
- P3 + no parameter semantics / Gemma 3 12B: Full 90.0% → 20.0% (Δ -70.0 pp)
- P3 + no parameter semantics / Mistral NeMo 12B: Full 100.0% → 25.0% (Δ -75.0 pp)

## 6 Plan-validity and failure diagnostics

| Model | Condition | Plan Validity | Invalid | Unparseable | Cap. sel. error | Param. error | Order error | P6 phys. dispatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-14B | Full | 100.0% | 0 | 0 | 5.6% | 5.6% | 0.0% | 20 |
| Qwen2.5-14B | Reduced capability information | 59.2% | 147 | 0 | 8.1% | 40.4% | 0.0% | 26 |
| Qwen2.5-14B | No capability semantics | 100.0% | 0 | 0 | 7.5% | 35.4% | 2.5% | 27 |
| Qwen2.5-14B | No parameter semantics | 88.3% | 42 | 0 | 12.3% | 25.1% | 5.8% | 25 |
| Qwen2.5-14B | No schema-constrained decoding | 100.0% | 0 | 0 | 5.6% | 5.7% | 0.0% | 20 |
| Phi-4 14B | Full | 100.0% | 0 | 0 | 0.0% | 0.4% | 0.4% | 0 |
| Phi-4 14B | Reduced capability information | 67.5% | 117 | 0 | 5.0% | 37.6% | 0.0% | 3 |
| Phi-4 14B | No capability semantics | 100.0% | 0 | 0 | 4.2% | 32.4% | 0.0% | 12 |
| Phi-4 14B | No parameter semantics | 99.2% | 3 | 0 | 7.5% | 19.2% | 4.2% | 0 |
| Phi-4 14B | No schema-constrained decoding | 100.0% | 0 | 0 | 0.0% | 0.4% | 0.8% | 0 |
| Gemma 3 12B | Full | 100.0% | 0 | 0 | 0.8% | 1.7% | 1.4% | 0 |
| Gemma 3 12B | Reduced capability information | 68.1% | 115 | 0 | 9.7% | 40.1% | 2.6% | 9 |
| Gemma 3 12B | No capability semantics | 100.0% | 0 | 0 | 5.0% | 32.4% | 3.1% | 6 |
| Gemma 3 12B | No parameter semantics | 100.0% | 0 | 0 | 15.2% | 16.8% | 10.8% | 0 |
| Gemma 3 12B | No schema-constrained decoding | 99.2% | 3 | 0 | 2.5% | 3.1% | 2.1% | 3 |
| Mistral NeMo 12B | Full | 100.0% | 0 | 0 | 13.3% | 22.4% | 1.1% | 48 |
| Mistral NeMo 12B | Reduced capability information | 59.2% | 147 | 0 | 14.7% | 53.4% | 0.4% | 44 |
| Mistral NeMo 12B | No capability semantics | 100.0% | 0 | 0 | 11.1% | 37.5% | 1.5% | 36 |
| Mistral NeMo 12B | No parameter semantics | 85.0% | 54 | 0 | 16.7% | 36.7% | 6.9% | 42 |
| Mistral NeMo 12B | No schema-constrained decoding | 100.0% | 0 | 0 | 13.6% | 23.5% | 1.4% | 49 |

## 7 Safety-gate replay

Offline replay of Full planner outputs. No additional LLM calls.

| Model | P6 err. dispatch before | P6 err. dispatch after | Intercepted | Valid physical pass rate | False-block rate |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-14B | 20 | 0 | 20 | 100.0% | 0.0% |
| Phi-4 14B | 0 | 0 | 0 | 100.0% | 0.0% |
| Gemma 3 12B | 0 | 0 | 0 | 100.0% | 0.0% |
| Mistral NeMo 12B | 48 | 0 | 48 | 100.0% | 0.0% |

## 8 Main findings

Findings below are restricted to the measured cells. Absolute Full ranking is not the target.

- no capability semantics reduced Task Success on all four models: True.
- no parameter semantics reduced Task Success on all four models: True.
- both semantic |Δ| exceeded |Δ_schema| on all four models: True.
- P2 drop under no capability semantics on all four models: True.
- P3 drop under no parameter semantics on all four models: True.

- Qwen2.5-14B: Δ_cap=-37.8 pp, Δ_par=-29.7 pp, Δ_schema=-1.4 pp.
- Phi-4 14B: Δ_cap=-37.5 pp, Δ_par=-25.0 pp, Δ_schema=-0.8 pp.
- Gemma 3 12B: Δ_cap=-35.0 pp, Δ_par=-26.7 pp, Δ_schema=-0.6 pp.
- Mistral NeMo 12B: Δ_cap=-16.4 pp, Δ_par=-17.5 pp, Δ_schema=-1.9 pp.

## 9 Deviations / warnings

- No model-download or tag substitutions were recorded.
- Qwen formal cells reused v1.1 runs. Original Qwen configs stored tag `qwen2.5:14b` but not digest.
- `no_schema_constrained_decoding` is the paper label for the existing `no_structure` condition.
- `reduced_capability_info` reuses the frozen names-only catalog condition from the Qwen v1.1 main experiment.
- Reduced capability information is a coarse reference and is excluded from the three-ablation Holm family.
- Transport-level Ollama retries were added for new runs only; request payloads were not changed.
- Recovery, physical E2E, and speech experiments were not extended.

