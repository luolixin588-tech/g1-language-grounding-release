from __future__ import annotations

import unittest

from src.adapters.unstructured_plan_adapter import parse_unstructured_plan
from src.analysis.statistics import holm_correction, mcnemar_exact


class UnstructuredParserTests(unittest.TestCase):
    def test_markdown_fence(self):
        raw = """```json
{"steps": [{"type": "speak", "content": "Hello."}]}
```"""
        steps, errors = parse_unstructured_plan(raw)
        self.assertEqual(errors, [])
        self.assertEqual(steps[0]["type"], "speak")
        self.assertEqual(steps[0]["content"], "Hello.")

    def test_prose_then_json(self):
        raw = 'Here is the plan:\n{"steps": [{"type": "speak", "content": "Okay."}]}\nThanks.'
        steps, errors = parse_unstructured_plan(raw)
        self.assertEqual(errors, [])
        self.assertEqual(steps[0]["content"], "Okay.")

    def test_unparseable(self):
        steps, errors = parse_unstructured_plan("I will wave at you now.")
        self.assertIsNone(steps)
        self.assertTrue(errors)

    def test_no_semantic_repair_of_unknown_tool(self):
        raw = '{"steps": [{"type": "tool", "name": "pick_up", "args": {}}]}'
        steps, errors = parse_unstructured_plan(raw)
        self.assertIsNotNone(steps)
        self.assertEqual(steps[0]["name"], "pick_up")


class StatsTests(unittest.TestCase):
    def test_mcnemar_no_discordant(self):
        self.assertEqual(mcnemar_exact(0, 0)["p_value"], 1.0)

    def test_holm_monotonic(self):
        rows = holm_correction([("a", 0.04), ("b", 0.03)])
        by = {r["name"]: r for r in rows}
        self.assertLessEqual(by["b"]["p_holm"], by["a"]["p_holm"] + 1e-12)
        self.assertAlmostEqual(by["b"]["p_holm"], 0.06, places=10)


if __name__ == "__main__":
    unittest.main()
