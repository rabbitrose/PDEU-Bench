#!/usr/bin/env python3
"""Regression tests for hidden/reminder eval parity."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_hidden
import eval_reminder


class ReminderEvalBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_reminder._install_reminder_behavior()
        eval_reminder.eval_core.PREFERENCE_INJECTION_MODE = eval_reminder.REMINDER_MODE

    def test_hidden_and_reminder_configs_match(self) -> None:
        names = (
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
            "PASS_N",
            "MAX_TOOL_CALLS_PER_TASK",
            "HIDDEN_CONTEXT_MAX_CHARS",
            "FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT",
            "SKIP_TASKS_ALREADY_IN_RESULT_FILE",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
            "LONG_CONTEXT_TURNS_LIST",
            "EVAL_CONCURRENCY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(eval_hidden, name), getattr(eval_reminder, name))

    def test_update_prompt_filters_metadata_without_mutating_observation(self) -> None:
        core = eval_reminder.eval_core
        context = {
            "task": {},
            "available_api_names": ["test.tool"],
            "available_api_contracts": [],
            "tool_catalog": {},
        }
        plan = {
            "phase": "initial_plan",
            "task_status": "in_progress",
            "plan_version": 1,
            "overall_goal": "测试",
            "completion_criteria": [],
            "steps": [],
            "next_step_id": None,
            "monitoring": {},
        }
        action = {
            "phase": "reason_action",
            "iteration": 1,
            "task_status": "in_progress",
            "selected_step_id": None,
            "reason": "测试",
            "action_type": "tool_call",
            "tool_call": {"name": "test.tool", "arguments": {}},
            "completion_assessment": {},
        }
        observation = {
            "sandbox_status": "success",
            "api_name": "test.tool",
            "transport": {
                "latency": "DROP_LATENCY",
                "latency_ms": "DROP_LATENCY_MS",
                "rate_limit": "DROP_RATE_LIMIT",
                "request_id": "DROP_NESTED_REQUEST_ID",
            },
            "request_id": "DROP_REQUEST_ID",
            "request_echo": "DROP_REQUEST_ECHO",
            "selected_inputs": "DROP_SELECTED_INPUTS",
            "context_effects": ["DROP_CONTEXT_EFFECTS"],
            "raw_response_shape": "DROP_RAW_RESPONSE_SHAPE",
            "warnings": ["KEEP_WARNING"],
            "response_body": {
                "summary": "KEEP_SUMMARY",
                "selected_inputs": "DROP_NESTED_SELECTED_INPUTS",
                "items": [{"id": "KEEP_CANDIDATE"}],
                "source_info": {"hits": [{"doc_id": "KEEP_SOURCE"}]},
                "hard_constraint_filter": {"output_record_count": 1},
            },
            "feedback": {"planner_relevance": "KEEP_FEEDBACK"},
        }
        original_observation = copy.deepcopy(observation)
        old_filter = core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
        try:
            core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = True
            messages = core.update_plan_messages(
                context=context,
                previous_plan=plan,
                action_json=action,
                sandbox_observation=observation,
                iteration=1,
            )
        finally:
            core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = old_filter

        user_prompt = next(message["content"] for message in messages if message["role"] == "user")
        prompt_observation = json.loads(user_prompt.split("SandBoxDB observation:\n", 1)[1])

        def collect_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {
                    nested_key
                    for item in value.values()
                    for nested_key in collect_keys(item)
                }
            if isinstance(value, list):
                return {
                    nested_key
                    for item in value
                    for nested_key in collect_keys(item)
                }
            return set()

        self.assertTrue(
            core.SANDBOX_OBSERVATION_PROMPT_EXCLUDED_FIELDS.isdisjoint(
                collect_keys(prompt_observation)
            )
        )
        for marker in (
            "DROP_LATENCY",
            "DROP_LATENCY_MS",
            "DROP_RATE_LIMIT",
            "DROP_NESTED_REQUEST_ID",
            "DROP_REQUEST_ID",
            "DROP_REQUEST_ECHO",
            "DROP_SELECTED_INPUTS",
            "DROP_NESTED_SELECTED_INPUTS",
            "DROP_CONTEXT_EFFECTS",
            "DROP_RAW_RESPONSE_SHAPE",
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, user_prompt)
        for marker in (
            "KEEP_WARNING",
            "KEEP_SUMMARY",
            "KEEP_CANDIDATE",
            "KEEP_SOURCE",
            "hard_constraint_filter",
            "KEEP_FEEDBACK",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, user_prompt)
        self.assertEqual(observation, original_observation)

    def test_each_generation_phase_contains_one_reminder(self) -> None:
        context = {
            "task": {},
            "available_api_names": [],
            "available_api_contracts": [],
            "tool_catalog": {},
        }
        plan = {
            "phase": "initial_plan",
            "task_status": "in_progress",
            "plan_version": 1,
            "overall_goal": "测试",
            "completion_criteria": [],
            "steps": [],
            "next_step_id": None,
            "monitoring": {},
        }
        action = {
            "phase": "reason_action",
            "iteration": 1,
            "task_status": "in_progress",
            "selected_step_id": None,
            "reason": "测试",
            "action_type": "tool_call",
            "tool_call": {"name": "test.tool", "arguments": {}},
            "completion_assessment": {},
        }
        observation = {"sandbox_status": "success"}
        prompt_groups = (
            eval_reminder.eval_core.initial_plan_messages(context),
            eval_reminder.eval_core.action_messages(
                context=context,
                current_plan=plan,
                trajectory_summary=[],
                iteration=1,
            ),
            eval_reminder.eval_core.update_plan_messages(
                context=context,
                previous_plan=plan,
                action_json=action,
                sandbox_observation=observation,
                iteration=1,
            ),
        )

        for messages in prompt_groups:
            user_prompt = next(message["content"] for message in messages if message["role"] == "user")
            self.assertTrue(user_prompt.startswith(eval_reminder.REMINDER_PROMPT + "\n\n"))
            self.assertEqual(user_prompt.count(eval_reminder.REMINDER_PROMPT), 1)

    def test_reminder_uses_separate_result_directory(self) -> None:
        core = eval_reminder.eval_core
        old_difficulty = core.TASK_DIFFICULTY
        old_long_context = core.ENABLE_LONG_CONTEXT
        try:
            core.TASK_DIFFICULTY = "easy"
            core.ENABLE_LONG_CONTEXT = False
            result_dir = core.result_dir_for_model("test-model", "reminder", 1)
        finally:
            core.TASK_DIFFICULTY = old_difficulty
            core.ENABLE_LONG_CONTEXT = old_long_context

        self.assertTrue(str(result_dir).endswith("test-model/easy/reminder/pass_1"))


if __name__ == "__main__":
    unittest.main()
