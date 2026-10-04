# AQ18 — Mistral NeMo P2 verification

Verified from released formal records (not from a plotting spreadsheet):

| Condition | File | P2 successes | P2 rate |
|---|---|---:|---:|
| Full | `data/planning_runs/cross_model/mistral_nemo_12b/full/records.jsonl` | 54/60 | 90.0% |
| No parameter semantics | `data/planning_runs/cross_model/mistral_nemo_12b/no_parameter_semantics/records.jsonl` | 60/60 | 100.0% |

The same cells appear in `figures/source_data/cross_model_task_categories.csv`:

- `Mistral NeMo 12B,full,P2 = 0.9`
- `Mistral NeMo 12B,no_parameter_semantics,P2 = 1.0`

The old Supplementary Fig. S1 plotting source that used `P2_param = 90.0` for
Mistral NeMo under no-parameter-semantics was **incorrect**. The corrected
value is **100.0%**. That old plotting file was not found in the working tree
and is not included here. Raw records were not altered.
