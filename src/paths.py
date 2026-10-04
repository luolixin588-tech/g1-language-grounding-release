from pathlib import Path

# Release layout. Algorithmic modules are unchanged; only repository roots differ
# from the original working tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "benchmark"
CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "data" / "planning_runs"
REPORTS_DIR = REPO_ROOT / "figures" / "source_data"
SCHEMA_PATH = DATA_DIR / "task_spec.schema.json"
CATALOG_PATH = CONFIG_DIR / "capability_catalog.json"
SEMANTIC_TASKS_PATH = DATA_DIR / "semantic_tasks.jsonl"
INSTRUCTIONS_PATH = DATA_DIR / "instructions_120.jsonl"
