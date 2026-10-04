# Multi-failure recovery final summary

Generated: `2026-08-25T03:06:22.977246+00:00`

## Recoverable pool (N=92)

Formal F1–F5 (original) + F6 (new recovery prompt rerun). Excludes old F6, F7, and non-injected F6.

### Episode count verification

| Condition | Count |
|---|---:|
| `F1_dispatch_rejected` | 16 |
| `F2_execution_timeout` | 16 |
| `F3_partial_completion` | 16 |
| `F4_invalid_robot_state` | 16 |
| `F5_infeasible_parameter` | 16 |
| `F6_second_step_failure_new_prompt` | 12 |
| `total_recoverable` | 92 |

## Final aggregate metrics (recoverable pool)

- **Recovery success**: 74/92 (80.4%)
- **Incorrect recovery**: 18/92 (19.6%)
- **Repeated action**: 8/12 (66.7%)

## Added recovery latency (92 episodes)

- **N**: 92
- **Median**: 838 ms
- **P95**: 1433 ms
- **Mean** (diagnostic): 874 ms
- **Min / Max** (QC): 502 / 1756 ms

- **Mean replanning attempts**: 1.00

## F7 correct termination (separate; formal run)

- 6/16 (37.5%)

## Per-condition table (final)

| Condition | Applicable | Recovery / handling | Repeated action | Incorrect recovery |
|---|---:|---|---|---|
| Dispatch rejected | 16 | 16/16 (100.0%) | 0/0 | 0/16 |
| Execution timeout | 16 | 16/16 (100.0%) | 0/0 | 0/16 |
| Partial completion | 16 | 16/16 (100.0%) | 0/0 | 0/16 |
| Invalid robot state | 16 | 12/16 (75.0%) | 0/0 | 4/16 |
| Infeasible parameter | 16 | 10/16 (62.5%) | 0/0 | 6/16 |
| Second-step failure | 12 | 4/12 (33.3%) | 8/12 | 8/12 |
| Unrecoverable | 16 | 6/16 (37.5% termination) | — | 10/16 |

## Sources

- F1–F5, F7: `runs/multifailure_recovery/multifailure_multifailure_formal_20260824T155611Z_9fde5231`
- F6 (final): `runs/multifailure_recovery/multifailure_multifailure_f6_rerun_20260825T021403Z_306314eb`

