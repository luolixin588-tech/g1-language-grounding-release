from __future__ import annotations

import unittest

from src.safety_gate import (
    evaluate_safety_gate,
    match_registered_physical_capabilities,
    physical_dispatch_after_gate,
    plan_has_physical_steps,
)


class SafetyGateCapabilityTests(unittest.TestCase):
    def test_no_physical_steps_always_pass(self):
        gate = evaluate_safety_gate("Open the door.", [{"capability": "speak", "parameters": {"content": "no"}}])
        self.assertEqual(gate["decision"], "pass")
        self.assertFalse(gate["physical_dispatch_allowed"])

    def test_supported_handshake_passes(self):
        plan = [{"capability": "arm_action", "parameters": {"action_id": 27, "action_name": "handshake"}}]
        gate = evaluate_safety_gate("Shake my hand.", plan)
        self.assertEqual(gate["decision"], "pass")
        self.assertTrue(gate["physical_dispatch_allowed"])

    def test_unsupported_door_request_blocks(self):
        plan = [
            {
                "capability": "arm_action",
                "parameters": {"action_id": 23, "action_name": "right_arm_horizontal"},
            }
        ]
        gate = evaluate_safety_gate("Open the door.", plan)
        self.assertEqual(gate["decision"], "block")
        self.assertEqual(gate["reason"], "unregistered_physical_request")
        self.assertFalse(physical_dispatch_after_gate(plan, gate))

    def test_match_does_not_use_task_family(self):
        matched = match_registered_physical_capabilities("Could you open that door for me?")
        self.assertEqual(matched, [])

    def test_locomotion_with_duration_passes(self):
        plan = [{"capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}}]
        gate = evaluate_safety_gate("Move forward for 2 seconds.", plan)
        self.assertEqual(gate["decision"], "pass")
        self.assertTrue(gate["physical_dispatch_allowed"])

    def test_operational_duration_bound_blocks(self):
        plan = [{"capability": "locomotion", "parameters": {"direction": "forward", "duration": 0.05}}]
        gate = evaluate_safety_gate("Move forward for 2 seconds.", plan)
        self.assertEqual(gate["decision"], "block")
        self.assertIn("operational_bound_violation", gate["reason"])


if __name__ == "__main__":
    unittest.main()
