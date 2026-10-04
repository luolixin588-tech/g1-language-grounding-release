from __future__ import annotations

import json
import unittest

from src.catalog import (
    _ARM_ACTION_DESCRIPTION,
    _MOVE_ROBOT_LOGIC_DESCRIPTION_FULL,
    build_planner_json_schema,
    build_planner_tool_catalog,
    load_catalog,
    planner_prompt,
)
from src.runner.batch_runner import load_condition


def _tool_by_name(tools):
    return {t["name"]: t for t in tools}


def _reference_full_catalog(catalog):
    """Frozen v1.1 Full planner-facing catalog (pre-ablation string)."""
    caps = {c["name"]: c for c in catalog["capabilities"]}
    arm = caps["arm_action"]
    loco = caps["locomotion"]
    state = caps["state_control"]
    return [
        {
            "name": "execute_arm_action",
            "description": _ARM_ACTION_DESCRIPTION,
            "parameters": {
                "action_id": {
                    "type": "integer",
                    "enum": arm["parameters"]["action_id"]["enum"],
                },
                "action_name": {
                    "type": "string",
                    "required": True,
                    "must_match_action_map": arm["action_map"],
                },
            },
        },
        {
            "name": "move_robot_logic",
            "description": _MOVE_ROBOT_LOGIC_DESCRIPTION_FULL,
            "parameters": {
                "direction": {
                    "type": "string",
                    "enum": loco["parameters"]["direction"]["enum"]
                    + state["parameters"]["state"]["enum"],
                },
                "duration": {
                    "type": "number",
                    "minimum": 0.1,
                    "required_for_locomotion": True,
                },
            },
        },
    ]


class CatalogAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.full_tools = build_planner_tool_catalog(cls.catalog)
        cls.full_schema = build_planner_json_schema(cls.catalog)
        cls.reference_full = _reference_full_catalog(cls.catalog)

    def test_full_catalog_unchanged(self):
        self.assertEqual(self.full_tools, self.reference_full)
        self.assertEqual(
            build_planner_tool_catalog(self.catalog, reduced=False),
            self.reference_full,
        )

    def test_no_action_description_preserves_identity_and_parameters(self):
        tools = build_planner_tool_catalog(
            self.catalog,
            include_action_descriptions=False,
            include_parameter_definitions=True,
        )
        full_by = _tool_by_name(self.full_tools)
        abl_by = _tool_by_name(tools)
        self.assertEqual(set(abl_by), {"execute_arm_action", "move_robot_logic"})
        self.assertNotIn("description", abl_by["execute_arm_action"])
        self.assertNotIn("description", abl_by["move_robot_logic"])
        self.assertEqual(
            abl_by["execute_arm_action"]["parameters"],
            full_by["execute_arm_action"]["parameters"],
        )
        self.assertEqual(
            abl_by["move_robot_logic"]["parameters"],
            full_by["move_robot_logic"]["parameters"],
        )

    def test_no_parameter_definition_preserves_fields_and_enums(self):
        tools = build_planner_tool_catalog(
            self.catalog,
            include_action_descriptions=True,
            include_parameter_definitions=False,
        )
        full_by = _tool_by_name(self.full_tools)
        abl_by = _tool_by_name(tools)
        self.assertEqual(abl_by["execute_arm_action"]["description"], _ARM_ACTION_DESCRIPTION)
        self.assertIn("description", abl_by["move_robot_logic"])
        self.assertNotEqual(
            abl_by["move_robot_logic"]["description"],
            full_by["move_robot_logic"]["description"],
        )
        arm_params = abl_by["execute_arm_action"]["parameters"]
        self.assertEqual(arm_params["action_id"], full_by["execute_arm_action"]["parameters"]["action_id"])
        self.assertEqual(
            arm_params["action_name"]["must_match_action_map"],
            full_by["execute_arm_action"]["parameters"]["action_name"]["must_match_action_map"],
        )
        move_params = abl_by["move_robot_logic"]["parameters"]
        self.assertEqual(
            move_params["direction"]["enum"],
            full_by["move_robot_logic"]["parameters"]["direction"]["enum"],
        )
        self.assertEqual(move_params["duration"], {"type": "number"})
        self.assertNotIn("minimum", move_params["duration"])
        self.assertNotIn("required_for_locomotion", move_params["duration"])

    def test_new_conditions_use_json_schema(self):
        for name in ("no_action_description", "no_parameter_definition"):
            cond = load_condition(name)
            self.assertTrue(cond["use_json_schema"])
            tools = build_planner_tool_catalog(
                self.catalog,
                include_action_descriptions=cond.get(
                    "include_action_descriptions", True
                ),
                include_parameter_definitions=cond.get(
                    "include_parameter_definitions", True
                ),
            )
            self.assertEqual(len(tools), 2)

    def test_action_mapping_identical_across_conditions(self):
        variants = [
            self.full_tools,
            build_planner_tool_catalog(
                self.catalog, include_action_descriptions=False
            ),
            build_planner_tool_catalog(
                self.catalog, include_parameter_definitions=False
            ),
        ]
        maps = [
            t["parameters"]["action_name"]["must_match_action_map"]
            for tools in variants
            for t in tools
            if t["name"] == "execute_arm_action"
        ]
        self.assertEqual(len(set(json.dumps(m, sort_keys=True) for m in maps)), 1)

    def test_json_schema_identical_across_full_and_new_ablations(self):
        schema = build_planner_json_schema(self.catalog)
        self.assertEqual(schema, self.full_schema)
        for kwargs in (
            {"include_action_descriptions": False},
            {"include_parameter_definitions": False},
        ):
            build_planner_tool_catalog(self.catalog, **kwargs)
            self.assertEqual(build_planner_json_schema(self.catalog), self.full_schema)

    def test_planner_prompt_unchanged(self):
        prompt = planner_prompt()
        self.assertIn(
            "If the user asks for a capability the robot does not have",
            prompt,
        )
        self.assertIn("Do not add extra physical actions.", prompt)

    def test_reduced_mode_unchanged(self):
        reduced = build_planner_tool_catalog(self.catalog, reduced=True)
        self.assertEqual(
            reduced[0]["description"],
            "Perform a predefined arm gesture.",
        )

    def test_catalog_condition_diff_audit(self):
        from scripts.audit_catalog_conditions import audit_catalog_conditions

        lines, ok = audit_catalog_conditions()
        self.assertTrue(ok, "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
