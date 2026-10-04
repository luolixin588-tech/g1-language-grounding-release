# Fig. 4 cluster-aware gate-replay statistics

Source: `<original-repo>/runs/full/full_20260821T051022Z_d81f78bf`
Gate: `src/safety_gate.py` + `configs/safety_gate.json` v1.0 (not altered).

## 1. Run-level exact McNemar (60 P6 runs)

- Both still dispatch (b): 0
- Gate-off only / blocked (c): 20
- Gate-on only: 0
- Both no dispatch: 40
- Official exact two-sided p: 1.90735e-06

## 2–3. Paired reduction (gate-off − gate-on), B=10000, seed=0

- Observed: 33.3% → 0.0% (Δ = 33.3%)
- Instruction-clustered (N=20) 95% CI: [13.3%, 55.0%]
- Semantic-task-clustered (N=10) 95% CI: [8.3%, 61.7%]

## 4. Semantic-task sign-flip test (2^10 assignments)

- Informative/nonzero tasks: 4 / 10
- Exact two-sided p: 0.125

## 5. False blocks among 240 P2–P5 valid physical outputs

- False blocks: 6
- Exactly P3_05 paraphrase ×3 + P3_06 paraphrase ×3: True
- Distinct instruction clusters: 2
- Distinct semantic-task clusters: 2

## Files

- `fig4_clustered_stats.json`
- `fig4_gate_replay_records.csv`
