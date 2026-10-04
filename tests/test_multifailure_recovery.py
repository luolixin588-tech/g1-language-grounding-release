from __future__ import annotations

import unittest

from src.recovery.multifailure_injection import (
    FAILURE_CONDITIONS,
    inject_execution_failure,
    nth_physical_index,
    resolve_injection_step_index,
    second_physical_index,
    validate_injection_preconditions,
)
from src.recovery.multifailure_outcomes import (
    detect_repeated_action,
    evaluate_f7_correct_termination,
)
from src.recovery.recovery_runner import (
    _RECOVERABLE_REPLAN_PREFIX,
    _UNRECOVERABLE_REPLAN_PREFIX,
    _recovery_system_content,
)


class MultifailureInjectionTests(unittest.TestCase):
    def _plan(self):
        semantic = [
            {"capability": "speak", "parameters": {"content": "Hi."}},
            {"capability": "locomotion", "parameters": {"direction": "forward", "duration": 1.0}},
            {"capability": "arm_action", "parameters": {"action_id": 27, "action_name": "handshake"}},
        ]
        production = list(semantic)
        return semantic, production

    def test_f6_targets_second_physical(self):
        semantic, _ = self._plan()
        self.assertEqual(nth_physical_index(semantic, 0), 1)
        self.assertEqual(second_physical_index(semantic), 2)
        self.assertEqual(
            resolve_injection_step_index("F6_second_step_failure", semantic),
            2,
        )

    def test_f3_partial_metadata(self):
        semantic, production = self._plan()
        task = {"semantic_task_id": "P5_09", "instruction": "test"}
        out = inject_execution_failure(
            "F3_partial_completion",
            task,
            {"semantic": semantic, "production": production},
            {},
        )
        self.assertTrue(out["injected"])
        self.assertTrue(out["partial"])
        self.assertFalse(out["diagnostic"]["completed"])
        self.assertIn("partial execution", out["failure_diagnostic"])

    def test_f7_not_recoverable(self):
        semantic, production = self._plan()
        task = {"semantic_task_id": "P2_01", "instruction": "test"}
        out = inject_execution_failure(
            "F7_unrecoverable_failure",
            task,
            {"semantic": semantic, "production": production},
            {},
        )
        self.assertFalse(out["recoverable"])
        self.assertEqual(out["diagnostic"]["error_class"], "unrecoverable_execution_failure")
        self.assertNotIn("REMAINING task", out["diagnostic"]["planner_instruction"])
        self.assertNotIn("failed step if it is still required", out["diagnostic"]["planner_instruction"])
        self.assertIn("must terminate", out["diagnostic"]["planner_instruction"])

    def test_f6_precondition(self):
        semantic = [{"capability": "arm_action", "parameters": {"action_id": 27, "action_name": "handshake"}}]
        ok, reason = validate_injection_preconditions("F6_second_step_failure", semantic)
        self.assertFalse(ok)
        self.assertEqual(reason, "f6_requires_second_physical_step")

    def test_all_conditions_defined(self):
        self.assertEqual(len(FAILURE_CONDITIONS), 7)

    def test_f1_keeps_remaining_task_instruction(self):
        semantic, production = self._plan()
        task = {"semantic_task_id": "P2_01", "instruction": "test"}
        out = inject_execution_failure(
            "F1_dispatch_rejected",
            task,
            {"semantic": semantic, "production": production},
            {},
        )
        self.assertTrue(out["recoverable"])
        self.assertIn("REMAINING task only", out["diagnostic"]["planner_instruction"])
        self.assertIn("failed step if it is still required", out["diagnostic"]["planner_instruction"])

    def test_recovery_prompt_branches_on_recoverable(self):
        rec_prompt = _recovery_system_content({"recoverable": True, "planner_instruction": "remain"})
        unrec_prompt = _recovery_system_content({"recoverable": False, "planner_instruction": "terminate"})
        self.assertTrue(rec_prompt.startswith(_RECOVERABLE_REPLAN_PREFIX))
        self.assertTrue(unrec_prompt.startswith(_UNRECOVERABLE_REPLAN_PREFIX))
        self.assertIn("The failure is recoverable and software-level.", rec_prompt)
        self.assertIn("Plan only the remaining task.", rec_prompt)
        self.assertNotIn("The failure is recoverable", unrec_prompt)
        self.assertNotIn("Plan only the remaining task", unrec_prompt)
        self.assertIn("unrecoverable execution failure", unrec_prompt)
        self.assertIn("Do not generate any physical action", unrec_prompt)


class MultifailureOutcomeTests(unittest.TestCase):
    def test_repeated_action_detected(self):
        completed = [
            {"capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}},
        ]
        executed = completed + [
            {"capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}},
            {"capability": "locomotion", "parameters": {"direction": "turn_right", "duration": 1.0}},
        ]
        repeated, reason = detect_repeated_action(completed, executed)
        self.assertTrue(repeated)
        self.assertIsNotNone(reason)

    def test_f7_correct_termination_no_new_physical(self):
        completed = [{"capability": "arm_action", "parameters": {"action_id": 27, "action_name": "handshake"}}]
        executed = list(completed)
        ok = evaluate_f7_correct_termination(
            executed_trace=executed,
            completed_before_failure=completed,
            remaining_plan=[{"capability": "speak", "parameters": {"content": "stop"}}],
            n_execution_replans=1,
            max_execution_replans=2,
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
