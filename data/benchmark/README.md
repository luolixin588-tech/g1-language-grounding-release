# G1 internal planning benchmark (frozen v1.1)

This directory is the authoritative public copy of the 60-task / 120-instruction
benchmark used for the manuscript
“Action and parameter meanings shape language grounding in a physical humanoid”.

## Files

| File | Contents |
|---|---|
| `semantic_tasks.jsonl` | 60 semantic tasks with gold task specifications |
| `instructions_120.jsonl` | 120 canonical + paraphrased instructions (same gold spec per pair) |
| `task_spec.schema.json` | Benchmark task-specification schema |
| `data_qc.json` / `data_qc.md` | Frozen QC report (60/120, 0 gold mismatches, PASS) |

## Scale

- 6 task families × 10 semantic tasks = 60 tasks
- Each task has one canonical and one natural paraphrase instruction
- 120 instructions × 3 independent planner repeats = 360 runs / condition
- Canonical and paraphrase share the same gold specification

## Task families P1–P6

| Family | Name | What is tested |
|---|---|---|
| P1 | Spoken response | Speech-only requests; any physical action is a failure |
| P2 | Single physical action | One registered arm / locomotion / state action |
| P3 | Parameterized control | Direction, duration, or state parameters must match |
| P4 | Multi-step physical sequence | 2–4-step physical composition and order |
| P5 | Mixed speech–action sequence | Required speech content and speech/action order |
| P6 | Unsupported / constrained request | No prohibited or unregistered physical dispatch |

## Gold fields

Each semantic task records:

- `expected_outcome`: `execute` or `non_execution`
- `required_steps` with capability + parameters
- `temporal_constraints`
- `allow_extra_steps`
- `requires_refusal_speech` (schema-compatible; v1.1 P6 success does **not** use refusal-keyword matching)

## QC

```bash
PYTHONPATH=. python3 scripts/data_qc.py
```

Expected: 60 tasks, 120 instructions, 10 tasks per family, 0 gold mismatches, PASS.
