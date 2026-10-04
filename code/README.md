# Code map

Exact copies of the manuscript analysis / planner modules. The runnable
import package remains `src.*` at the repository root (original import paths).

| Release path | Original path | Role |
|---|---|---|
| `code/planner/catalog.py` | `src/catalog.py` | Planner prompt, catalogue load, runtime JSON Schema |
| `code/planner/runner/` | `src/runner/` | Batch planner runner |
| `code/adapters/` | `src/adapters/` | Production-plan adapter and semantic normalizer |
| `code/evaluator/` | `src/evaluator/` | Deterministic validator and task-success evaluator |
| `code/gate/safety_gate.py` | `src/safety_gate.py` | Frozen Fig. 4 safety / dispatch gate |
| `code/recovery/` | `src/recovery/` | Failure injection and recovery |
| `code/analysis/` | `src/analysis/` | Statistics and figure source-data writers |
