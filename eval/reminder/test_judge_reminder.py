#!/usr/bin/env python3
"""Regression tests for hidden/reminder judge parity."""

from __future__ import annotations

import json
import unittest

import judge_reminder


class ReminderJudgeBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        judge_reminder._install_reminder_behavior()

    @staticmethod
    def reminder_record() -> dict:
        return {
            "run_id": "test_run",
            "user_id": "user_001",
            "task_id": "task_easy",
            "pass_n": 1,
            "pass_index": 1,
            "task_title": "测试任务",
            "preference_injection_mode": "reminder",
            "preference_injection_label": "用户交互历史 + 生成前偏好提醒",
            "provided_eval_context": {
                "mode": "reminder",
                "context": {
                    "task": {"task_id": "task_easy", "title": "测试任务"},
                    "user_interaction_history": {"interactions": []},
                    "available_api_names": [],
                },
            },
        }

    def test_record_is_blinded_to_hidden_before_judging(self) -> None:
        reminder_record = self.reminder_record()
        normalized = judge_reminder._as_hidden_judge_record(reminder_record)

        self.assertEqual(normalized["preference_injection_mode"], "hidden")
        self.assertEqual(normalized["preference_injection_label"], "用户交互历史")
        self.assertEqual(normalized["provided_eval_context"]["mode"], "hidden")
        self.assertEqual(reminder_record["preference_injection_mode"], "reminder")

    def test_hidden_and_reminder_generate_identical_judge_prompts(self) -> None:
        hidden_record = judge_reminder._as_hidden_judge_record(self.reminder_record())
        reminder_for_judge = judge_reminder._as_hidden_judge_record(self.reminder_record())
        plan = {"steps": []}

        hidden_messages = (
            judge_reminder.judge.plan_judge_messages(hidden_record, plan)
            + judge_reminder.judge.trajectory_judge_messages(hidden_record, plan, [])
            + judge_reminder.judge.update_plan_judge_messages(hidden_record, [])
        )
        reminder_messages = (
            judge_reminder.judge.plan_judge_messages(reminder_for_judge, plan)
            + judge_reminder.judge.trajectory_judge_messages(reminder_for_judge, plan, [])
            + judge_reminder.judge.update_plan_judge_messages(reminder_for_judge, [])
        )

        self.assertEqual(hidden_messages, reminder_messages)
        self.assertNotIn("reminder", json.dumps(reminder_messages, ensure_ascii=False).lower())

    def test_reminder_uses_separate_result_directory(self) -> None:
        judge = judge_reminder.judge
        old_difficulty = judge.TASK_DIFFICULTY
        old_long_context = judge.ENABLE_LONG_CONTEXT
        try:
            judge.TASK_DIFFICULTY = "easy"
            judge.ENABLE_LONG_CONTEXT = False
            result_dir = judge.eval_result_dir_for_model("test-model", "reminder", 1)
        finally:
            judge.TASK_DIFFICULTY = old_difficulty
            judge.ENABLE_LONG_CONTEXT = old_long_context

        self.assertTrue(str(result_dir).endswith("test-model/easy/reminder/pass_1"))


if __name__ == "__main__":
    unittest.main()
