#!/usr/bin/env python3
"""QC the released 60/120 benchmark (no planner calls)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_qc_impl", ROOT / "scripts" / "_original" / "run_qc.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = mod.run_qc()
    mod.write_qc_markdown(report)
    print("ok" if report["ok"] else "FAIL", "n_issues=", report["n_issues"])
    print(
        "tasks={n_semantic_tasks} instructions={n_instructions} families={family_counts}".format(
            **report
        )
    )
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
