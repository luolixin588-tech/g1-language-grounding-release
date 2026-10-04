from __future__ import annotations

import json
import re
import unittest

from src.ablations.semantic_aliases import (
    CAPABILITY_LEAK_TERMS,
    PARAMETER_LEAK_TERMS,
    build_capability_alias_maps,
    build_condition_planner_bundle,
    build_parameter_alias_maps,
    decode_production_plan,
    find_leak_terms,
    planner_prompt_for_condition,
    planner_visible_system_texts,
)
from src.catalog import (
    build_planner_json_schema,
    build_planner_tool_catalog,
    load_catalog,
    planner_prompt,
)
from src.runner.batch_runner import load_condition


class SemanticAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()
        cls.cap_maps = build_capability_alias_maps(cls.catalog)
        cls.param_maps = build_parameter_alias_maps(cls.catalog)

    def test_full_artifacts_unchanged(self):
        cond = load_condition("full")
        bundle = build_condition_planner_bundle(self.catalog, cond)
        self.assertEqual(bundle["prompt"], planner_prompt())
        self.assertEqual(
            bundle["tool_catalog"],
            build_planner_tool_catalog(self.catalog),
        )
        self.assertEqual(bundle["schema"], build_planner_json_schema(self.catalog))
        self.assertIsNone(bundle["alias_meta"])

    def test_capability_alias_map_stable_sorted_ids(self):
        codes = self.cap_maps["codes"]
        self.assertEqual(codes[0], "A01")
        self.assertEqual(len(codes), 15)
        # Sorted action_id order: 11 -> A01
        self.assertEqual(self.cap_maps["code_to_id"]["A01"], 11)
        self.assertEqual(self.cap_maps["code_to_name"]["A01"], "double_hand_flying_kiss")
        self.assertEqual(self.cap_maps["code_to_id"]["A15"], 99)
        self.assertEqual(self.cap_maps["code_to_name"]["A15"], "reset_arm_pose")
        # handshake is 27
        handshake_code = self.cap_maps["id_to_code"][27]
        self.assertEqual(self.cap_maps["code_to_name"][handshake_code], "handshake")

    def test_parameter_alias_map_frozen_order(self):
        self.assertEqual(self.param_maps["codes"][0], "D01")
        self.assertEqual(self.param_maps["code_to_value"]["D01"], "forward")
        self.assertEqual(self.param_maps["code_to_value"]["D06"], "turn_right")
        self.assertEqual(self.param_maps["code_to_value"]["D07"], "stand_up")
        self.assertEqual(self.param_maps["code_to_value"]["D09"], "start")
        self.assertEqual(len(self.param_maps["codes"]), 9)

    def test_no_capability_semantics_planner_visible_has_no_action_leaks(self):
        cond = load_condition("no_capability_semantics")
        bundle = build_condition_planner_bundle(self.catalog, cond)
        text = planner_visible_system_texts(
            bundle["prompt"], bundle["tool_catalog"], bundle["schema"]
        )
        leaks = find_leak_terms(text, CAPABILITY_LEAK_TERMS)
        self.assertEqual(leaks, [], msg=f"capability leaks: {leaks}")
        self.assertNotIn("must_match_action_map", text)
        self.assertNotIn("handshake", text.lower())
        self.assertIn("A01", text)
        self.assertIn("action_code", text)

    def test_no_parameter_semantics_planner_visible_has_no_direction_leaks(self):
        cond = load_condition("no_parameter_semantics")
        bundle = build_condition_planner_bundle(self.catalog, cond)
        text = planner_visible_system_texts(
            bundle["prompt"], bundle["tool_catalog"], bundle["schema"]
        )
        # Word-boundary check for parameter terms
        found = []
        for term in PARAMETER_LEAK_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                found.append(term)
        self.assertEqual(found, [], msg=f"parameter leaks: {found}")
        self.assertNotIn("required_for_locomotion", text)
        self.assertNotIn('"minimum"', text)
        self.assertIn("D01", text)
        # Capability semantics remain available
        self.assertIn("handshake", text)
        self.assertIn("must_match_action_map", text)

    def test_decode_capability_aliases(self):
        visible = [
            {
                "type": "tool",
                "name": "execute_arm_action",
                "args": {"action_code": self.cap_maps["id_to_code"][27]},
            }
        ]
        decoded, errors = decode_production_plan(
            visible,
            anonymize_capability_semantics=True,
            anonymize_parameter_semantics=False,
            cap_maps=self.cap_maps,
            param_maps=self.param_maps,
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            decoded[0]["args"],
            {"action_id": 27, "action_name": "handshake"},
        )

    def test_decode_parameter_aliases(self):
        visible = [
            {
                "type": "tool",
                "name": "move_robot_logic",
                "args": {"direction": "D01", "duration": 2},
            }
        ]
        decoded, errors = decode_production_plan(
            visible,
            anonymize_capability_semantics=False,
            anonymize_parameter_semantics=True,
            cap_maps=self.cap_maps,
            param_maps=self.param_maps,
        )
        self.assertEqual(errors, [])
        self.assertEqual(decoded[0]["args"]["direction"], "forward")
        self.assertEqual(decoded[0]["args"]["duration"], 2)

    def test_prompts_keep_p6_policy(self):
        for anon_cap, anon_param in ((True, False), (False, True)):
            p = planner_prompt_for_condition(anon_cap, anon_param)
            self.assertIn(
                "If the user asks for a capability the robot does not have",
                p,
            )

    def test_new_conditions_use_json_schema(self):
        for name in ("no_capability_semantics", "no_parameter_semantics"):
            cond = load_condition(name)
            self.assertTrue(cond["use_json_schema"])
            bundle = build_condition_planner_bundle(self.catalog, cond)
            self.assertIsNotNone(bundle["schema"])

    def test_decode_with_permuted_capability_map(self):
        cap1 = build_capability_alias_maps(self.catalog, 1)
        code = cap1["id_to_code"][27]
        visible = [
            {"type": "tool", "name": "execute_arm_action", "args": {"action_code": code}}
        ]
        decoded, errors = decode_production_plan(
            visible,
            anonymize_capability_semantics=True,
            anonymize_parameter_semantics=False,
            cap_maps=cap1,
            param_maps=self.param_maps,
        )
        self.assertEqual(errors, [])
        self.assertEqual(decoded[0]["args"]["action_id"], 27)

    def test_permutation_zero_matches_legacy_maps(self):
        legacy_cap = build_capability_alias_maps(self.catalog, 0)
        legacy_par = build_parameter_alias_maps(self.catalog, 0)
        self.assertEqual(legacy_cap["code_to_id"]["A01"], 11)
        self.assertEqual(legacy_par["code_to_value"]["D01"], "forward")

    def test_permutations_are_nontrivial(self):
        cap0 = build_capability_alias_maps(self.catalog, 0)
        cap1 = build_capability_alias_maps(self.catalog, 1)
        self.assertNotEqual(cap0["code_to_id"], cap1["code_to_id"])
        par0 = build_parameter_alias_maps(self.catalog, 0)
        par1 = build_parameter_alias_maps(self.catalog, 1)
        self.assertNotEqual(par0["code_to_value"], par1["code_to_value"])

    def test_leakage_audit_all_permutations(self):
        from scripts.audit_semantic_ablations import audit_all

        _, ok = audit_all()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
