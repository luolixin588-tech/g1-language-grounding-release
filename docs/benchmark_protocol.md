# G1 Internal Planning Benchmark — Protocol v1.1

Release note: this is the frozen protocol used for the manuscript
“Action and parameter meanings shape language grounding in a physical humanoid”.
Paper results use **v1.1 only**. The v1.0 Full run is not included in this
release.

## Purpose

This benchmark is **not** proposed as a new general-purpose robotics benchmark.
It is an internal, G1-specific evaluation aligned with the paper's three
planning questions:

1. grounding;
2. temporal composition;
3. failure/constraint handling.

This internal benchmark tests the exact capability space used by the physical
Unitree G1 system.

## Scale

- 60 semantic tasks.
- 6 task families × 10 semantic tasks.
- Each semantic task has two paired formulations:
  - canonical;
  - natural paraphrase.
- 120 instructions total.
- 3 independent runs per instruction.
- 360 planning runs per experimental condition.

Canonical and paraphrased instructions share exactly the same gold task
specification. Paraphrase may change linguistic form, but must not drop cues
that determine capability identity (for example `chest` vs overhead wave,
`stand_up` vs `start`).

## Families

- P1 Spoken response
- P2 Single physical action
- P3 Parameterized control
- P4 Multi-step physical sequence
- P5 Mixed speech-action sequence
- P6 Unsupported / constrained request

P4 and P5 contain 2-, 3-, and 4-step tasks.

## Main metrics

### Task Success Rate

A run succeeds when the generated plan satisfies the requested robot task,
not when it is an exact plan-string match.

v1.1 success rules:

- required physical capabilities and parameters must match;
- required speech content must be present when the task constrains speech;
- required speech/action order must hold for mixed tasks (P5);
- extra physical action → failure;
- wrong physical capability → failure;
- wrong parameter → failure;
- required speech missing → failure;
- P1 speech-only task with any physical action → failure;
- P5 required speech/action order wrong → failure;
- benign extra speech is allowed unless speech content/order is explicitly
  constrained by the unmatched required steps.

P6 Task Success is deterministic and does **not** use refusal-keyword matching
or an LLM judge:

- No prohibited or unsupported physical action is generated/dispatched.

A speak step may be recorded, but its wording does not decide planning
success. Refusal-language quality is a separate HRI question and is out of
scope for this planning benchmark.

### Plan Validity

Whether the generated plan is syntactically and semantically executable under
the registered capability catalog. Plan validity is independent of whether the
plan satisfies the user's requested task.

### Planning latency

Report median and P95.

## Diagnostics

Use only to localize failure:

- capability selection accuracy;
- parameter accuracy;
- order accuracy.

Do not promote diagnostic metrics to headline outcomes unless needed.

## Repeated-run reporting

Report per-run Task Success Rate over 360 runs.

Also report the transparent count:
"X of 120 instructions succeeded in all three independent runs."

## Evaluation principle

Gold annotations specify task semantics, not a unique JSON string.

The evaluator should normalize runtime/tool-level output into paper-facing
semantic steps:

- speak;
- arm_action;
- locomotion;
- state_control.

For P1/P5, quoted speech content is intentionally used so *required* speech
steps can be evaluated deterministically without BLEU/ROUGE or an LLM judge.
Unconstrained extra acknowledgement speech is ignored.

System meanings used in gold language:

- `stand_up`: rise to the standing posture (ready-standing / preparatory standing mode);
- `start`: enter start mode (locomotion-ready / motion mode);
- `damp`: enter damping mode.

## Statistics

Primary paired comparisons should resample or test at the semantic-task /
instruction level, not individual output tokens.

For ablations:

- identical instructions and model;
- paired task-level comparison;
- bootstrap 95% CIs over instructions;
- McNemar tests for paired binary success when inferential testing is used;
- Holm correction if multiple pairwise hypothesis tests are reported.

Latency: median / P95.

## QC protocol

See `data/benchmark/README.md` and `scripts/data_qc.py`. The frozen QC report
is `data/benchmark/data_qc.md` (PASS: 60/120, 0 gold mismatches).
