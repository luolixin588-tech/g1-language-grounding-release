from __future__ import annotations

import unittest

from src.analysis.a2_failure_taxonomy import classify_a2_record
from src.recovery.evaluate_recovery import evaluate_executed_trace
from src.recovery.failure_injection import first_physical_index, failure_class_for


class TaxonomyTests(unittest.TestCase):
    def test_name_mismatch(self):
        rec = {
            "family": "P2_single_physical_action",
            "normalized_semantic_plan": [
                {
                    "capability": "arm_action",
                    "parameters": {"action_id": 20, "action_name": "right_one_handed_heart"},
                }
            ],
            "task_success_result": {"task_success": False, "error_labels": ["invalid_plan", "parameter_mismatch"]},
            "validity_result": {
                "valid": False,
                "errors": ["step[0] action_name 'right_one_handed_heart' does not match action_id 20"],
            },
        }
        self.assertEqual(classify_a2_record(rec), "action-ID/action-name mismatch")

    def test_p6_physical(self):
        rec = {
            "family": "P6_unsupported_or_constrained",
            "normalized_semantic_plan": [{"capability": "state_control", "parameters": {"state": "start"}}],
            "task_success_result": {"task_success": False, "error_labels": ["p6_physical_action_dispatched"]},
            "validity_result": {"valid": True, "errors": []},
        }
        self.assertEqual(classify_a2_record(rec), "unsupported/prohibited action")

    def test_wrong_parameter_valid_plan(self):
        rec = {
            "family": "P4_multistep_physical_sequence",
            "normalized_semantic_plan": [
                {"capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}},
                {"capability": "locomotion", "parameters": {"direction": "right_strafe", "duration": 1.0}},
            ],
            "task_success_result": {"task_success": False, "error_labels": ["parameter_mismatch"]},
            "validity_result": {"valid": True, "errors": []},
        }
        self.assertEqual(classify_a2_record(rec), "wrong parameter")

    def test_omitted_physical_is_capability_identity(self):
        rec = {
            "family": "P2_single_physical_action",
            "normalized_semantic_plan": [{"capability": "speak", "parameters": {"content": "I cannot."}}],
            "task_success_result": {"task_success": False, "error_labels": ["omitted_step"]},
            "validity_result": {"valid": True, "errors": []},
        }
        self.assertEqual(classify_a2_record(rec), "capability identity error")


class RecoveryEvalTests(unittest.TestCase):
    def test_r0_prefix_fails(self):
        task = {
            "family": "P4_multistep_physical_sequence",
            "expected_outcome": "execute",
            "required_steps": [
                {"id": "s1", "capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}},
                {"id": "s2", "capability": "locomotion", "parameters": {"direction": "turn_right", "duration": 1.0}},
            ],
            "temporal_constraints": [["s1", "s2"]],
        }
        out = evaluate_executed_trace(task, [])
        self.assertFalse(out["recovery_success"])

    def test_r1_retry_satisfies_gold(self):
        task = {
            "family": "P2_single_physical_action",
            "expected_outcome": "execute",
            "required_steps": [
                {
                    "id": "s1",
                    "capability": "arm_action",
                    "parameters": {"action_id": 27, "action_name": "handshake"},
                }
            ],
            "temporal_constraints": [],
        }
        out = evaluate_executed_trace(
            task,
            [{"capability": "arm_action", "parameters": {"action_id": 27, "action_name": "handshake"}}],
        )
        self.assertTrue(out["recovery_success"])

    def test_first_physical_skips_speak(self):
        steps = [
            {"capability": "speak", "parameters": {"content": "Hello."}},
            {"capability": "arm_action", "parameters": {"action_id": 25, "action_name": "chest_wave"}},
        ]
        self.assertEqual(first_physical_index(steps), 1)

    def test_failure_class_deterministic(self):
        self.assertEqual(failure_class_for("P2_01", 0), failure_class_for("P2_01", 0))


if __name__ == "__main__":
    unittest.main()
