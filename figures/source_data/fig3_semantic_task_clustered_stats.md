# Fig.3a semantic-task-clustered statistics

Point estimates are the original 360-run Task Success rates. This file only updates clustered bootstrap CIs and McNemar units.

## Official runs (paper Fig.3a)

- `full`: `<original-repo>/runs/full/full_20260821T051022Z_d81f78bf`
- `no_capability_semantics`: `<original-repo>/runs/no_capability_semantics/no_capability_semantics_20260824T111546Z_368b8574`
- `no_parameter_semantics`: `<original-repo>/runs/no_parameter_semantics/no_parameter_semantics_20260824T112027Z_d8471060`
- `no_json_schema`: `<original-repo>/runs/no_structure/no_structure_20260821T052009Z_6bf8d7c2`
- `reduced_capability_info`: `<original-repo>/runs/reduced_capability_info/reduced_capability_info_20260821T052504Z_6e0dd3ea`

Later 20260907 runs, multimodel non-Qwen runs, archive, and smoke runs were **not** used.

## Completeness

Each official condition: 360 runs, 60 `semantic_task_id`, canonical+paraphrase, 3 repeats each, no missing/duplicate keys.

## Method change

| | Old | New |
|---|---|---|
| Cluster | instruction / `sample_id` | `semantic_task_id` |
| N | 120 | 60 |
| Runs kept together | 3 repeats of one instruction | canonical×3 + paraphrase×3 (6 runs) |
| Bootstrap | paired resample, B=10000, seed=0, percentile 95% CI | same, on 60 tasks |
| McNemar unit | instruction all-3 success | semantic task 6/6 success |
| Holm family | 3 Fig.3a ablations | same 3; Reduced excluded |

## Comparison table

| Comparison | Full rate | Ablation rate | Delta pp | Old 95% CI | New semantic-task-clustered 95% CI | McNemar raw p | Holm-adjusted p |
|---|---:|---:|---:|---|---|---:|---:|
| Full vs No capability semantics | 94.4% | 56.7% | -37.8 | [-46.9, -28.6] | [-50.3, -25.3] | 1.55e-06 | 4.649e-06 |
| Full vs No parameter semantics | 94.4% | 64.7% | -29.7 | [-38.9, -20.6] | [-41.7, -18.1] | 5.722e-06 | 1.144e-05 |
| Full vs No JSON Schema | 94.4% | 93.1% | -1.4 | [-3.6, +0.0] | [-3.6, +0.0] | 0.5 | 0.5 |
| Full vs Reduced capability information | 94.4% | 48.6% | -45.8 | [-55.0, -36.7] | [-58.3, -33.3] | 3.725e-09 | n/a (reference; not in Holm family) |

Reduced capability information is a reference comparison and is **not** included in the Holm family.

## Semantic-task-level McNemar 2×2

Success = canonical 3/3 **and** paraphrase 3/3 (6/6 runs).

| Comparison | Both success | Full only | Ablation only | Both fail |
|---|---:|---:|---:|---:|
| Full vs No capability semantics | 32 | 24 | 1 | 3 |
| Full vs No parameter semantics | 34 | 22 | 1 | 3 |
| Full vs No JSON Schema | 54 | 2 | 0 | 4 |
| Full vs Reduced capability information | 27 | 29 | 0 | 4 |

## Significance

- Full vs No capability semantics: Holm significance unchanged (old p_holm=1.307e-11, new p_holm=4.649e-06). CI width 18.3 pp → 25.0 pp.
- Full vs No parameter semantics: Holm significance unchanged (old p_holm=1.126e-08, new p_holm=1.144e-05). CI width 18.3 pp → 23.6 pp.
- Full vs No JSON Schema: Holm significance unchanged (old p_holm=0.5, new p_holm=0.5). CI width 3.6 pp → 3.6 pp.
- Full vs Reduced capability information: reference only. CI [-55.0, -36.7] → [-58.3, -33.3]. McNemar raw p=3.725e-09 (not Holm-adjusted).

## Files

- Script: `scripts/analyze_fig3_semantic_task_clustered.py`
- JSON: `reports/fig3_semantic_task_clustered_stats.json`
- This note: `reports/fig3_semantic_task_clustered_stats.md`

Original instruction-clustered reports were not overwritten.
