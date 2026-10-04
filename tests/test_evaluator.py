from __future__ import annotations

import unittest

from src.runner.batch_runner import evaluate_offline_plan


def _task_execute_two_step():
    return {
        "task_id": "P4_01",
        "family": "P4_multistep_physical_sequence",
        "expected_outcome": "execute",
        "required_steps": [
            {"id": "s1", "capability": "locomotion", "parameters": {"direction": "forward", "duration": 2.0}},
            {"id": "s2", "capability": "locomotion", "parameters": {"direction": "turn_right", "duration": 1.0}},
        ],
        "temporal_constraints": [["s1", "s2"]],
        "allow_extra_steps": False,
        "requires_refusal_speech": False,
    }


def _task_p6():
    return {
        "task_id": "P6_01",
        "family": "P6_unsupported_or_constrained",
        "expected_outcome": "non_execution",
        "required_steps": [],
        "temporal_constraints": [],
        "allow_extra_steps": False,
        "requires_refusal_speech": False,
    }


class EvaluatorTests(unittest.TestCase):
    def test_valid_but_task_wrong(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "backward", "duration": 2.0},
                },
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "turn_right", "duration": 1.0},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("parameter_mismatch", out["task_success"]["error_labels"])

    def test_invalid_plan(self):
        task = _task_execute_two_step()
        raw = {"steps": [{"type": "tool", "name": "ask_lab_assistant", "args": {"question": "x"}}]}
        out = evaluate_offline_plan(task, raw)
        self.assertFalse(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])

    def test_parameter_mismatch(self):
        task = {
            "expected_outcome": "execute",
            "required_steps": [
                {
                    "id": "s1",
                    "capability": "arm_action",
                    "parameters": {"action_id": 27, "action_name": "handshake"},
                }
            ],
            "temporal_constraints": [],
            "allow_extra_steps": False,
        }
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 25, "action_name": "chest_wave"},
                }
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertTrue(
            "parameter_mismatch" in out["task_success"]["error_labels"]
            or "capability_mismatch" in out["task_success"]["error_labels"]
        )

    def test_order_reversal(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "turn_right", "duration": 1.0},
                },
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "forward", "duration": 2.0},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("order_reversal", out["task_success"]["error_labels"])

    def test_omitted_step(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "forward", "duration": 2.0},
                }
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("omitted_step", out["task_success"]["error_labels"])

    def test_benign_extra_speak_allowed(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {"type": "speak", "content": "Okay."},
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "forward", "duration": 2.0},
                },
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "turn_right", "duration": 1.0},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertTrue(out["task_success"]["task_success"])

    def test_extra_physical_action_fails(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "forward", "duration": 2.0},
                },
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "turn_right", "duration": 1.0},
                },
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 17, "action_name": "clap"},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("extra_physical_action", out["task_success"]["error_labels"])

    def test_p1_extra_physical_fails(self):
        task = {
            "task_id": "P1_01",
            "family": "P1_spoken_response",
            "expected_outcome": "execute",
            "required_steps": [
                {"id": "s1", "capability": "speak", "parameters": {"content_exact": "Hello."}}
            ],
            "temporal_constraints": [],
            "allow_extra_steps": False,
            "requires_refusal_speech": False,
        }
        raw = {
            "steps": [
                {"type": "speak", "content": "Hello."},
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 27, "action_name": "handshake"},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("p1_physical_action_dispatched", out["task_success"]["error_labels"])

    def test_required_speech_missing_fails(self):
        task = {
            "task_id": "P5_02",
            "family": "P5_mixed_speech_action_sequence",
            "expected_outcome": "execute",
            "required_steps": [
                {"id": "s1", "capability": "speak", "parameters": {"content_exact": "Hello."}},
                {
                    "id": "s2",
                    "capability": "arm_action",
                    "parameters": {"action_id": 25, "action_name": "chest_wave"},
                },
            ],
            "temporal_constraints": [["s1", "s2"]],
            "allow_extra_steps": False,
        }
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 25, "action_name": "chest_wave"},
                }
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("required_speech_missing", out["task_success"]["error_labels"])

    def test_p5_order_reversal_fails(self):
        task = {
            "task_id": "P5_02",
            "family": "P5_mixed_speech_action_sequence",
            "expected_outcome": "execute",
            "required_steps": [
                {"id": "s1", "capability": "speak", "parameters": {"content_exact": "Hello."}},
                {
                    "id": "s2",
                    "capability": "arm_action",
                    "parameters": {"action_id": 25, "action_name": "chest_wave"},
                },
            ],
            "temporal_constraints": [["s1", "s2"]],
            "allow_extra_steps": False,
        }
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 25, "action_name": "chest_wave"},
                },
                {"type": "speak", "content": "Hello."},
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("order_reversal", out["task_success"]["error_labels"])

    def test_p6_physical_violation(self):
        task = _task_p6()
        raw = {
            "steps": [
                {"type": "speak", "content": "I cannot pick that up."},
                {
                    "type": "tool",
                    "name": "execute_arm_action",
                    "args": {"action_id": 27, "action_name": "handshake"},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertFalse(out["task_success"]["task_success"])
        self.assertIn("p6_physical_action_dispatched", out["task_success"]["error_labels"])

    def test_p6_correct_non_execution(self):
        task = _task_p6()
        raw = {"steps": [{"type": "speak", "content": "I cannot do that. The request is not supported."}]}
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertTrue(out["task_success"]["task_success"])

    def test_p6_any_speak_wording_succeeds(self):
        task = _task_p6()
        raw = {"steps": [{"type": "speak", "content": "Sorry."}]}
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertTrue(out["task_success"]["task_success"])

    def test_correct_execute_plan(self):
        task = _task_execute_two_step()
        raw = {
            "steps": [
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "forward", "duration": 2},
                },
                {
                    "type": "tool",
                    "name": "move_robot_logic",
                    "args": {"direction": "turn_right", "duration": 1.0},
                },
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertTrue(out["validity"]["valid"])
        self.assertTrue(out["task_success"]["task_success"])

    def test_state_control_normalization(self):
        task = {
            "expected_outcome": "execute",
            "required_steps": [
                {"id": "s1", "capability": "state_control", "parameters": {"state": "stand_up"}}
            ],
            "temporal_constraints": [],
            "allow_extra_steps": False,
        }
        raw = {
            "steps": [
                {"type": "tool", "name": "move_robot_logic", "args": {"direction": "stand_up"}}
            ]
        }
        out = evaluate_offline_plan(task, raw)
        self.assertEqual(out["normalized_semantic_plan"][0]["capability"], "state_control")
        self.assertTrue(out["task_success"]["task_success"])


if __name__ == "__main__":
    unittest.main()
