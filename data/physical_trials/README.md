# Physical end-to-end trials (20 × 5 = 100)

Source (copied, not moved):
`humanoid-g1-natural-language-paper/e2erecord/` on the operator workstation.

These files support the 100-trial on-robot analysis. They were not in the
planning-benchmark working tree at pack time.

## Files

| File | Role |
|---|---|
| `G1_physical_e2e_20x5_results.csv` | Scored 20-task × 5-repeat table (manuscript 100-trial summary) |
| `rawdata.jsonl` | On-robot planner/exec traces (119 lines; mixed dates) |
| `rawdate.log` | Controller log (original filename kept; 19 local paths rewritten to `<local-log-dir>/` in this copy only) |

## Headline counts from the CSV (not rewritten)

Each of E01–E20 has **5/5** valid records (`Valid records` was restored from
a spreadsheet-locale `5月5日`). Totals below use the `Final result` column:

| Outcome | Trials |
|---|---:|
| Success | **79/100** |
| Partial (E07 only) | 3/100 |
| Not counted as Success | 21/100 |

E07 is scored `Partial 3/5; Failure 2/5` and contributes 0 to the 79 successes.

## `rawdata.jsonl` coverage

Date counts in the released file:

- 2026-05-25: 12 lines (five lab-member name-lookup rows were removed from this copy)
- 2026-08-24: 58 lines
- 2026-08-25: 44 lines

The scored 100-trial table is the CSV. Do not treat every jsonl line as one
of the 100 formal trials.
