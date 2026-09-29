#!/usr/bin/env python3
"""Regression tests for RAG-aligned PAG evaluation behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PAG_DIR = Path(__file__).resolve().parent
if str(PAG_DIR) not in sys.path:
    sys.path.insert(0, str(PAG_DIR))

import eval_pag


class PagEvalBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_pag._install_pag_behavior()
        eval_pag._configure_evaluation("hard", 0)

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
            eval_pag.PAG_ABSTRACT_CONTEXT_KEY: "KEEP_ABSTRACT",
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

    def test_runtime_configuration_matches_rag(self) -> None:
        names = (
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
            "TASK_DIFFICULTY_LIST",
            "PASS_N",
            "MAX_TOOL_CALLS_PER_TASK",
            "FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT",
            "SKIP_TASKS_ALREADY_IN_RESULT_FILE",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
            "LONG_CONTEXT_TURNS_LIST",
            "DEFAULT_EVAL_CONCURRENCY",
            "QINGYUN_EVAL_CONCURRENCY",
            "RAG_PROMPT_MAX_CHARS",
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(eval_pag, name), getattr(eval_pag.eval_rag, name))

        self.assertEqual(
            eval_pag.eval_rag.RAG_PROMPT_MAX_CHARS,
            eval_pag.RAG_PROMPT_MAX_CHARS,
        )

    def test_all_generation_phases_add_only_abstract_beyond_rag(self) -> None:
        context = self._context()
        prompt_groups = (
            eval_pag.eval_core.initial_plan_messages(context),
            eval_pag.eval_core.action_messages(
                context=context,
                current_plan=self._plan(),
                trajectory_summary=[],
                iteration=1,
            ),
            eval_pag.eval_core.update_plan_messages(
                context=context,
                previous_plan=self._plan(),
                action_json=self._action(),
                sandbox_observation={
                    "sandbox_status": "success",
                    "transport": {"latency": "DROP_LATENCY"},
                    "request_id": "DROP_REQUEST_ID",
                    "response_body": {"summary": "KEEP_SUMMARY"},
                },
                iteration=1,
            ),
        )

        for messages in prompt_groups:
            prompt_text = next(message["content"] for message in messages if message["role"] == "user")
            self.assertEqual(prompt_text.count(eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
            self.assertEqual(prompt_text.count(eval_pag.eval_rag.RAG_PROMPT_PREFIX), 1)
            self.assertIn("KEEP_ABSTRACT", prompt_text)
            self.assertIn("KEEP_QUESTION", prompt_text)
            self.assertIn("KEEP_ANSWER", prompt_text)
            for marker in (
                "LEAK_METHOD",
                "LEAK_ID",
                "LEAK_KIND",
                "LEAK_TOPIC",
                "LEAK_SIGNAL",
            ):
                with self.subTest(marker=marker):
                    self.assertNotIn(marker, prompt_text)

        update_prompt = next(
            message["content"] for message in prompt_groups[-1] if message["role"] == "user"
        )
        self.assertNotIn("DROP_LATENCY", update_prompt)
        self.assertNotIn("DROP_REQUEST_ID", update_prompt)
        self.assertIn("KEEP_SUMMARY", update_prompt)

    def test_user_abstract_profile_uses_only_abstract_string(self) -> None:
        with patch.object(
            eval_pag.eval_core,
            "pag_prompt_abstract_for_task",
            return_value={
                "task_related_user_abstract": "KEEP_PROFILE_ONLY",
                "preference_points": ["DROP_PREFERENCE_POINTS"],
            },
        ):
            profile = eval_pag._user_abstract_profile_for_task(
                user_id="user_001",
                task_id="task_hard",
            )
        self.assertEqual(profile, "KEEP_PROFILE_ONLY")

    def test_abstract_is_bounded_before_context_storage(self) -> None:
        with patch.object(
            eval_pag.eval_core,
            "pag_prompt_abstract_for_task",
            return_value={"task_related_user_abstract": "A" * 7000},
        ):
            profile = eval_pag._user_abstract_profile_for_task(
                user_id="user_001",
                task_id="task_hard",
            )
        self.assertEqual(len(profile), eval_pag.USER_ABSTRACT_PROFILE_MAX_CHARS)
        self.assertTrue(profile.endswith("..."))

    def test_rag_phrase_inside_qa_does_not_suppress_abstract(self) -> None:
        messages = [{"role": "user", "content": "QA mentions User abstract profile literally."}]
        patched = eval_pag._prepend_user_abstract_profile(messages, "KEEP_ABSTRACT")
        prompt = patched[0]["content"]
        self.assertTrue(prompt.startswith(eval_pag.PAG_ABSTRACT_PROMPT_PREFIX + ":"))
        self.assertIn("KEEP_ABSTRACT", prompt)

    def test_pag_record_uses_rag_audit_metadata_plus_abstract_delta(self) -> None:
        record = {
            "hyperparameters": {
                "pag_retrieval_top_k": 3,
                "pag_retrieval_injection_phases": ["legacy"],
            }
        }
        aligned = eval_pag._align_pag_record_metadata(record)
        hyperparameters = aligned["hyperparameters"]
        self.assertNotIn("pag_retrieval_top_k", hyperparameters)
        self.assertNotIn("pag_retrieval_injection_phases", hyperparameters)
        self.assertEqual(hyperparameters["rag_top_k"], eval_pag.PAG_RETRIEVAL_TOP_K)
        self.assertEqual(
            hyperparameters["pag_difference_from_rag"],
            "user_abstract_profile_only",
        )

    def test_install_is_idempotent(self) -> None:
        eval_pag._install_pag_behavior()
        eval_pag._install_pag_behavior()
        messages = eval_pag.eval_core.initial_plan_messages(self._context())
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        self.assertEqual(prompt.count(eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)

    def test_repair_preserves_one_rag_and_one_abstract_context(self) -> None:
        original = eval_pag.eval_core.initial_plan_messages(self._context())
        repaired = eval_pag.eval_core.build_repair_messages(
            original,
            "invalid",
            "invalid JSON",
        )
        combined = "\n".join(str(message.get("content") or "") for message in repaired)
        self.assertEqual(combined.count(eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
        self.assertEqual(combined.count(eval_pag.eval_rag.RAG_PROMPT_PREFIX), 1)

    def test_tool_discovery_is_rag_topk_plus_one_abstract(self) -> None:
        with patch.object(
            eval_pag,
            "_user_abstract_profile_for_task",
            return_value="KEEP_ABSTRACT",
        ):
            messages = eval_pag.eval_core.tool_discovery_messages(
                user_id="user_001",
                task_id="task_hard",
                task_doc={"task": {"task_id": "task_hard", "title": "KEEP_TASK"}},
                user_profile={"user_id": "user_001"},
                hidden_info={
                    "interactions": [
                        {
                            "interaction_id": "LEAK_ID",
                            "kind": "LEAK_KIND",
                            "implicit_signal": "LEAK_SIGNAL",
                            "turns": [
                                {"role": "user", "content": "KEEP_QUESTION"},
                                {"role": "assistant", "content": "KEEP_ANSWER"},
                            ],
                        }
                    ]
                },
                long_context=None,
                user_sandbox={"user_context": {}},
                api_definitions={
                    "test.tool": {
                        "category": "test",
                        "action": "tool",
                        "description": "test",
                        "interface": {"parameters": []},
                    }
                },
                preference_mode="pag",
            )
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        self.assertEqual(prompt.count(eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
        self.assertIn("KEEP_ABSTRACT", prompt)
        self.assertIn("KEEP_QUESTION", prompt)
        self.assertIn("KEEP_ANSWER", prompt)
        for marker in ("LEAK_ID", "LEAK_KIND", "LEAK_SIGNAL"):
            self.assertNotIn(marker, prompt)

    def test_pag_uses_separate_result_directory(self) -> None:
        result_dir = eval_pag.eval_core.result_dir_for_model("test-model", "pag", 1)
        self.assertTrue(str(result_dir).endswith("test-model/hard/pag/pass_1"))


if __name__ == "__main__":
    unittest.main()
