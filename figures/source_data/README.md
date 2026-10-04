# Figure source data

Numeric tables copied from the original analysis outputs. Records themselves
were not rewritten.

| File | Figure / table |
|---|---|
| `fig3_semantic_task_clustered_stats.json` | Fig. 3 clustered inference |
| `ablation_stats.json` | Fig. 3 instruction-clustered stats |
| `fig4_gate_replay_records.csv` | Fig. 4 decision table |
| `fig4_clustered_stats.json` | Fig. 4 McNemar / bootstrap |
| `safety_gate_results.json` | Fig. 4 headline gate metrics |
| `multifailure_recovery_final_summary.json` | Fig. 5 / ED Table 3 |
| `cross_model_task_categories.csv` | Supp. Fig. S1 family rates (includes AQ18) |
| `cross_model_effects.csv` | Supp. Fig. S1 ablation effects |
| `cross_model_summary.csv` | Supp. cross-model summary |
| `figure_s2_gate.csv` | Supp. Fig. S2 (official `src.safety_gate` replay) |
| `AQ18_mistral_p2_verification.md` | AQ18 |
| `G1_physical_e2e_20x5_results.csv` | 100-trial physical E2E (20×5; 79/100 Success) |

Supplementary Fig. S2 source data in this directory is `figure_s2_gate.csv`.
The same official table is also at
`data/gate_replay/cross_model/cross_model_safety_gate.csv`.
The earlier prototype-gate file `cross_model_gate_summary.csv` is **not**
included here; it remains only in the internal working tree.
