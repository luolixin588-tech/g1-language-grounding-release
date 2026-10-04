# F7 corrected-prompt rerun

- run_id: `multifailure_multifailure_f7_corrected_20260930T051720Z_883abfac`
- model: qwen2.5:14b, temperature=0
- episodes: 8 tasks × 2 repeats = 16
- does not overwrite formal run `multifailure_multifailure_formal_20260824T155611Z_9fde5231`

## Aggregate

| metric | old conflicting prompt | corrected prompt |
|---|---|---|
| correct termination | 6/16 | 10/16 |
| continued physical behavior | 10/16 | 6/16 |
| n_execution_replans = 1 | 16/16 | 16/16 |
| arm_action in remaining plan | 6 | 4 |
| locomotion in remaining plan | 6 | 2 |
| state_control in remaining plan | 0 | 0 |
| repeated completed physical action | 0 | 0 |

Legal F7 terminal form is speak-only (schema `minItems=1`; empty plan invalid).
