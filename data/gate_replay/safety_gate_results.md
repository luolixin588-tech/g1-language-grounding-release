# Safety Gate results (capability admissibility)

Source: `<original-repo>/runs/full/full_20260821T051022Z_d81f78bf`
Config: `configs/safety_gate.json`

## P6 metrics

| Metric | Gate OFF | Gate ON |
|---|---:|---:|
| Erroneous physical dispatch | 20 (33.3%) | 0 (0.0%) |
| Safe non-execution | 40 (66.7%) | 60 (100.0%) |

## Interception (original 20 erroneous dispatches)

- Blocked: 20
- Not blocked: 0
- Interception rate: 100.0%

## Legitimate P2–P5 control

- Valid physical outputs tested: 240
- Passed: 234
- Blocked (false blocks): 6
- False Block Rate: 2.5%

## Operational bounds

- Implemented: locomotion duration >= configured minimum when duration is present in plan
- Implemented: locomotion direction must be in registered enum
- Implemented: state_control state must be in registered enum
- Not implemented: velocity (vmax) — not represented in planner benchmark
- Not implemented: global time horizon (tmax) — not represented in planner benchmark

