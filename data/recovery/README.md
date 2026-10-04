# Injected-failure recovery (Fig. 5 / Extended Data Table 3)

Labels F1–F7 are preserved. Manuscript aggregates **must** keep the 92
applicable F1–F6 episodes separate from the 16 F7 episodes.

## Released runs

| Directory | What it is | Use in the manuscript |
|---|---|---|
| `formal_F1-F7_20260824T155611Z_9fde5231` | 112 episodes: 16 per F1–F7 | Source of F1–F5 only |
| `f6_rerun_20260825T021403Z_306314eb` | 16 F6 episodes (12 applicable) | Paper F6 (new recovery prompt) |
| `f7_corrected_20260930T051720Z_883abfac` | 16 F7 episodes, prompt-consistent rerun | **Paper F7: 10/16** correct termination |

## 92 applicable F1–F6 episodes (recoverable pool)

| Condition | Applicable | Recovery success in the paper pool |
|---|---:|---|
| F1 dispatch rejected | 16 | 16/16 |
| F2 execution timeout | 16 | 16/16 |
| F3 partial completion | 16 | 16/16 |
| F4 invalid robot state | 16 | 12/16 |
| F5 infeasible parameter | 16 | 10/16 |
| F6 second-step failure (rerun) | 12 | 4/12 |
| **Total F1–F6** | **92** | **74/92 (80.4%)** |

F6 in the original formal run had 16 rows but only 12 injected/applicable
episodes. The paper uses the F6 rerun (4/12), not the original formal F6 (2/12).

## 16 F7 episodes (unrecoverable; separate)

Manuscript / public-release F7 is the prompt-consistent rerun:

- **10/16** correct termination
- Records: `f7_corrected_20260930T051720Z_883abfac`

The formal 112-episode file still contains the earlier F7 rows (6/16). Those
rows are superseded: the first F7 prompt conflicted with the recoverable-failure
wrapper. They are kept only so F1–F5 stay in their original file. Do not use
6/16 as the manuscript F7 cell.

The copied report `multifailure_recovery_final_summary.md` still prints the
superseded 6/16 F7 line because that report was not rewritten. Use this README
and `scripts/reproduce_supplementary_results.py` for the manuscript F7 number.
