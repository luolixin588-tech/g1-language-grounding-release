# Qwen2.5-14B formal planning runs (manuscript main text)

These are the frozen v1.1 Qwen2.5-14B runs used for Fig. 3 and the main
planner-interface ablations. Each directory contains `config.json`,
`records.jsonl` and `summary.json` only.

| Condition | Run id | Task success | Paper label |
|---|---|---:|---|
| `full/full_20260821T051022Z_d81f78bf` | full_20260821T051022Z_d81f78bf | 340/360 = 94.4% | Full |
| `no_capability_semantics/no_capability_semantics_20260824T111546Z_368b8574` | …368b8574 | 204/360 = 56.7% | No capability semantics |
| `no_parameter_semantics/no_parameter_semantics_20260824T112027Z_d8471060` | …d8471060 | 233/360 = 64.7% | No parameter semantics |
| `no_structure/no_structure_20260821T052009Z_6bf8d7c2` | …6bf8d7c2 | 335/360 = 93.1% | No JSON Schema |
| `reduced_capability_info/reduced_capability_info_20260821T052504Z_6e0dd3ea` | …6e0dd3ea | 175/360 = 48.6% | Reduced capability information |

Protocol: 120 instructions × 3 repeats, seed 0, temperature 0, Ollama tag
`qwen2.5:14b` (the only model field stored in these `config.json` files).
Ollama version, weight digest, and Q4_K_M were captured later from the
retained workstation and local model; see
`docs/retained_environment_manifest.md`.

Do not mix these with later unofficial 2026-09-07 reruns (excluded from this
release).
