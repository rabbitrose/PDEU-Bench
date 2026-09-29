#!/usr/bin/env python3
"""Regression tests for RAG prompt isolation and observation filtering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

import eval_rag


class RagEvalBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_rag._install_rag_behavior()
        eval_rag._configure_evaluation("hard", 0)

    @staticmethod
    def _context() -> dict:
        return {
            "task": {},
            "available_api_names": ["test.tool"],
            "available_api_contracts": [],
            "tool_catalog": {},
            "rag_retrieved_user_interaction_history": {
                "retrieval_method": "LEAK_METHOD",
                "records": [
                    {
                        "rank": 1,
                        "score": 9.9,
                        "interaction_id": "LEAK_ID",
                        "kind": "LEAK_KIND",
                        "retrieved_topic": "LEAK_TOPIC",
                        "implicit_signal": "LEAK_SIGNAL",
                        "turns": [
                            {"role": "user", "content": "KEEP_QUESTION"},
                            {"role": "assistant", "content": "KEEP_ANSWER"},
                        ],
                    }
                ],
            },
        }

    @staticmethod
    def _plan() -> dict:
        return {
            "phase": "initial_plan",
            "task_status": "in_progress",
            "plan_version": 1,
            "overall_goal": "测试",
            "completion_criteria": [],
            "steps": [],
            "next_step_id": None,
            "monitoring": {},
        }

    @staticmethod
    def _action() -> dict:
        return {
            "phase": "reason_action",
            "iteration": 1,
            "task_status": "in_progress",
            "selected_step_id": None,
            "reason": "测试",
            "action_type": "tool_call",
            "tool_call": {"name": "test.tool", "arguments": {}},
            "completion_assessment": {},
        }

    def test_all_generation_phases_receive_only_raw_qa_turns(self) -> None:
        context = self._context()
        prompt_groups = (
            eval_rag.eval_core.initial_plan_messages(context),
            eval_rag.eval_core.action_messages(
                context=context,
                current_plan=self._plan(),
                trajectory_summary=[],
                iteration=1,
            ),
            eval_rag.eval_core.update_plan_messages(
                context=context,
                previous_plan=self._plan(),
                action_json=self._action(),
                sandbox_observation={"sandbox_status": "success"},
                iteration=1,
            ),
        )

        for messages in prompt_groups:
            prompt = next(message["content"] for message in messages if message["role"] == "user")
            self.assertIn("KEEP_QUESTION", prompt)
            self.assertIn("KEEP_ANSWER", prompt)
            for marker in ("LEAK_METHOD", "LEAK_ID", "LEAK_KIND", "LEAK_TOPIC", "LEAK_SIGNAL"):
                with self.subTest(marker=marker):
                    self.assertNotIn(marker, prompt)

    def test_update_prompt_filters_observation_metadata(self) -> None:
        messages = eval_rag.eval_core.update_plan_messages(
            context=self._context(),
            previous_plan=self._plan(),
            action_json=self._action(),
            sandbox_observation={
                "sandbox_status": "success",
                "transport": {"latency": "DROP_LATENCY"},
                "request_id": "DROP_REQUEST_ID",
                "response_body": {"summary": "KEEP_SUMMARY"},
            },
            iteration=1,
        )
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        self.assertNotIn("DROP_LATENCY", prompt)
        self.assertNotIn("DROP_REQUEST_ID", prompt)
        self.assertIn("KEEP_SUMMARY", prompt)


if __name__ == "__main__":
    unittest.main()
