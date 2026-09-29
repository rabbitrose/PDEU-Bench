#!/usr/bin/env python3
"""Offline regression tests for PAG-aligned Agent RAG-A plus Wikipedia RAG-B."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


AGENT_DIR = Path(__file__).resolve().parent
EVAL_DIR = AGENT_DIR.parent
PAG_DIR = EVAL_DIR / "pag"
for module_dir in (EVAL_DIR, AGENT_DIR, PAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_agent
import judge_agent


class AgentPagAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_agent._install_agent_behavior()
        judge_agent._install_agent_judge_behavior()

    @staticmethod
    def _hidden_info() -> dict:
        return {
            "interactions": [
                {
                    "interaction_id": "interaction_001",
                    "kind": "preference_history",
                    "implicit_signal": "KEEP_PRIVATE_SIGNAL",
                    "turns": [
                        {"role": "user", "content": "KEEP_QUESTION"},
                        {"role": "assistant", "content": "KEEP_ANSWER"},
                    ],
                },
                {
                    "interaction_id": "interaction_002",
                    "turns": [
                        {"role": "user", "content": "UNRELATED"},
                        {"role": "assistant", "content": "OTHER"},
                    ],
                },
            ]
        }

    @staticmethod
    def _wikipedia() -> dict:
        return {
            "retrieved": True,
            "retrieval_source": "KEEP_AUDIT_SOURCE",
            "retrieval_method": "KEEP_AUDIT_METHOD",
            "query_terms": ["KEEP_PRIVATE_QUERY_TERM"],
            "candidate_count": 64,
            "article": {
                "title": "KEEP_WIKI_TITLE",
                "summary": "KEEP_WIKI_SUMMARY",
                "bm25_score": -12.3,
                "rerank_score": 45.6,
            },
            "index_metadata": {"indexed_article_count": 277477},
        }

    @classmethod
    def _context(cls, *, include_wikipedia: bool) -> dict:
        context = {
            "task": {},
            "available_api_names": ["test.tool"],
            "available_api_contracts": [],
            "tool_catalog": {},
            eval_agent.AGENT_RAG_A_CONTEXT_KEY: {
                "retrieval_method": "lightweight_bm25_over_task_text_and_hidden_qa",
                "records": [
                    {
                        "rank": 1,
                        "score": "KEEP_PRIVATE_SCORE",
                        "implicit_signal": "KEEP_PRIVATE_SIGNAL",
                        "turns": [
                            {"role": "user", "content": "KEEP_QUESTION"},
                            {"role": "assistant", "content": "KEEP_ANSWER"},
                        ],
                    }
                ],
            },
            eval_agent.eval_pag.PAG_ABSTRACT_CONTEXT_KEY: "KEEP_PAG_ABSTRACT",
        }
        if include_wikipedia:
            context[eval_agent.AGENT_RAG_B_CONTEXT_KEY] = cls._wikipedia()
        return context

    @staticmethod
    def _plan() -> dict:
        return {
            "phase": "initial_plan",
            "task_status": "in_progress",
            "plan_version": 1,
            "overall_goal": "test",
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
            "reason": "test",
            "action_type": "tool_call",
            "tool_call": {"name": "test.tool", "arguments": {}},
            "completion_assessment": {},
        }

    @staticmethod
    def _user_prompt(messages: list[dict[str, str]]) -> str:
        return next(str(item.get("content") or "") for item in messages if item.get("role") == "user")

    def test_runtime_constants_match_pag(self) -> None:
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
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
            "DEFAULT_EVAL_CONCURRENCY",
            "QINGYUN_EVAL_CONCURRENCY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(eval_agent, name), getattr(eval_agent.eval_pag, name))
        self.assertEqual(eval_agent.RAG_A_TOP_K, eval_agent.eval_pag.PAG_RETRIEVAL_TOP_K)
        self.assertEqual(eval_agent.RAG_A_PROMPT_MAX_CHARS, eval_agent.eval_pag.RAG_PROMPT_MAX_CHARS)

    def test_configured_runtime_matches_pag_except_mode(self) -> None:
        names = (
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
            "PASS_N",
            "MAX_TOOL_CALLS_PER_TASK",
            "FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT",
            "SKIP_TASKS_ALREADY_IN_RESULT_FILE",
            "OVERWRITE_SELECTED_USER_RESULTS_ON_START",
            "AUTO_TUNE_CONCURRENCY",
            "DEFAULT_EVAL_CONCURRENCY",
            "DEEPSEEK_EVAL_CONCURRENCY",
            "DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY",
            "QINGYUN_DEEPSEEK_EVAL_CONCURRENCY",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
        )
        eval_agent.eval_pag._configure_evaluation("hard", 0)
        pag_values = {name: copy.deepcopy(getattr(eval_agent.eval_core, name)) for name in names}
        eval_agent.configure_evaluation("hard", 0)
        agent_values = {name: copy.deepcopy(getattr(eval_agent.eval_core, name)) for name in names}
        self.assertEqual(agent_values, pag_values)
        self.assertEqual(eval_agent.eval_core.PREFERENCE_INJECTION_MODE, eval_agent.AGENT_MODE)

    def test_rag_a_is_exact_pag_retriever(self) -> None:
        hidden_info = self._hidden_info()
        task_doc = {"task": {"title": "KEEP_QUESTION"}}
        expected = eval_agent.eval_pag.eval_rag.retrieve_hidden_info_for_task(
            hidden_info=hidden_info,
            task_doc=task_doc,
            top_k=eval_agent.eval_pag.PAG_RETRIEVAL_TOP_K,
        )
        actual = eval_agent.retrieve_history_rag_a(hidden_info, task_doc)
        self.assertEqual(actual, expected)
        self.assertEqual(actual["records"][0]["interaction_id"], "interaction_001")

    def test_wikipedia_prompt_exposes_only_article_content(self) -> None:
        messages = eval_agent._prepend_wikipedia_context(
            [{"role": "user", "content": "BASE_PROMPT"}],
            self._wikipedia(),
        )
        prompt = self._user_prompt(messages)
        self.assertTrue(prompt.startswith(eval_agent.WIKIPEDIA_PROMPT_PREFIX))
        self.assertIn("KEEP_WIKI_TITLE", prompt)
        self.assertIn("KEEP_WIKI_SUMMARY", prompt)
        for private_value in (
            "KEEP_AUDIT_SOURCE",
            "KEEP_AUDIT_METHOD",
            "KEEP_PRIVATE_QUERY_TERM",
            "bm25_score",
            "rerank_score",
            "indexed_article_count",
        ):
            self.assertNotIn(private_value, prompt)

    def test_all_generation_prompts_have_only_wikipedia_delta(self) -> None:
        eval_agent.configure_evaluation("hard", 0)
        pag_context = self._context(include_wikipedia=False)
        agent_context = self._context(include_wikipedia=True)

        def prompt_groups(context: dict) -> tuple[list[dict[str, str]], ...]:
            return (
                eval_agent.eval_core.initial_plan_messages(context),
                eval_agent.eval_core.action_messages(
                    context=context,
                    current_plan=self._plan(),
                    trajectory_summary=[],
                    iteration=1,
                ),
                eval_agent.eval_core.update_plan_messages(
                    context=context,
                    previous_plan=self._plan(),
                    action_json=self._action(),
                    sandbox_observation={
                        "response_body": {"summary": "KEEP_OBSERVATION"},
                        "latency": "DROP_LATENCY",
                        "request_id": "DROP_REQUEST_ID",
                    },
                    iteration=1,
                ),
            )

        for pag_messages, agent_messages in zip(prompt_groups(pag_context), prompt_groups(agent_context)):
            pag_prompt = self._user_prompt(pag_messages)
            agent_prompt = self._user_prompt(agent_messages)
            self.assertEqual(agent_prompt.split("\n\n", 1)[1], pag_prompt)
            self.assertEqual(agent_prompt.count(eval_agent.WIKIPEDIA_PROMPT_PREFIX), 1)
            self.assertEqual(agent_prompt.count(eval_agent.eval_pag.eval_rag.RAG_PROMPT_PREFIX), 1)
            self.assertEqual(agent_prompt.count(eval_agent.eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
            self.assertNotIn("KEEP_PRIVATE_SIGNAL", agent_prompt)
            self.assertNotIn("KEEP_PRIVATE_SCORE", agent_prompt)

        update_prompt = self._user_prompt(prompt_groups(agent_context)[-1])
        self.assertIn("KEEP_OBSERVATION", update_prompt)
        self.assertNotIn("DROP_LATENCY", update_prompt)
        self.assertNotIn("DROP_REQUEST_ID", update_prompt)

    def test_wikipedia_cache_reuses_one_offline_retrieval(self) -> None:
        with eval_agent._WIKIPEDIA_CACHE_LOCK:
            eval_agent._WIKIPEDIA_CACHE.clear()
        with patch.object(
            eval_agent,
            "retrieve_wikipedia_summary",
            return_value=self._wikipedia(),
        ) as mocked:
            first = eval_agent._cached_wikipedia_retrieval(
                user_id="user_001",
                task_id="task_hard",
                task_doc={"task": {"title": "test"}},
            )
            second = eval_agent._cached_wikipedia_retrieval(
                user_id="user_001",
                task_id="task_hard",
                task_doc={"task": {"title": "test"}},
            )
        self.assertEqual(first, second)
        self.assertEqual(mocked.call_count, 1)

    def test_agent_metadata_records_only_rag_method_delta(self) -> None:
        record = {
            "preference_injection_mode": "pag",
            "preference_injection_label": "pag",
            "provided_eval_context": {"mode": "pag", "context": {}},
            "judge_results": {"offline_judge_entrypoint": "python eval/pag/judge_pag.py"},
            "hyperparameters": {"pag_difference_from_rag": "user_abstract_profile_only"},
        }
        aligned = eval_agent._align_agent_record_metadata(record)
        self.assertEqual(aligned["preference_injection_mode"], eval_agent.AGENT_MODE)
        self.assertEqual(aligned["provided_eval_context"]["mode"], eval_agent.AGENT_MODE)
        self.assertNotIn("pag_difference_from_rag", aligned["hyperparameters"])
        self.assertEqual(
            aligned["hyperparameters"]["agent_difference_from_pag"],
            "offline_wikipedia_rag_b_only",
        )
        self.assertEqual(
            aligned["judge_results"]["offline_judge_entrypoint"],
            "python eval/agent/judge_agent.py",
        )

    def test_judge_runtime_and_preference_context_match_pag(self) -> None:
        names = (
            "TARGET_GEN_LLM",
            "PASS_N",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
            "LONG_CONTEXT_TURNS_LIST",
            "TARGET_EVAL_RUN_ID",
            "JUDGE_LATEST_EVAL_RUN_ONLY",
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
            "DEFAULT_JUDGE_CONCURRENCY",
            "QINGYUN_JUDGE_CONCURRENCY",
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
            "TASK_DIFFICULTY_LIST",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(judge_agent, name), getattr(judge_agent.judge_pag, name))

        record = {
            "preference_injection_mode": judge_agent.AGENT_MODE,
            "provided_eval_context": {
                "mode": judge_agent.AGENT_MODE,
                "context": {
                    "rag_retrieved_user_interaction_history": {"records": ["KEEP_RAG_A"]},
                    judge_agent.PAG_ABSTRACT_CONTEXT_KEY: "KEEP_ABSTRACT",
                    judge_agent.AGENT_RAG_B_CONTEXT_KEY: self._wikipedia(),
                },
            },
        }
        core = judge_agent.judge.profile_core(record)
        self.assertEqual(
            core["rag_retrieved_user_interaction_history"],
            {"records": ["KEEP_RAG_A"]},
        )
        self.assertEqual(core[judge_agent.PAG_ABSTRACT_CONTEXT_KEY], "KEEP_ABSTRACT")
        self.assertNotIn(judge_agent.AGENT_RAG_B_CONTEXT_KEY, core)

    def test_configured_judge_runtime_matches_pag_except_mode(self) -> None:
        names = (
            "TARGET_GEN_LLM",
            "PASS_N",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
            "TARGET_EVAL_RUN_ID",
            "JUDGE_LATEST_EVAL_RUN_ONLY",
            "AUTO_TUNE_CONCURRENCY",
            "DEFAULT_JUDGE_CONCURRENCY",
            "DEEPSEEK_JUDGE_CONCURRENCY",
            "QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY",
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
        )
        judge_agent.judge_pag._configure_judge("hard", 0)
        pag_values = {name: copy.deepcopy(getattr(judge_agent.judge, name)) for name in names}
        judge_agent._configure_judge("hard", 0)
        agent_values = {name: copy.deepcopy(getattr(judge_agent.judge, name)) for name in names}
        self.assertEqual(agent_values, pag_values)
        self.assertEqual(judge_agent.judge.PREFERENCE_INJECTION_MODE, judge_agent.AGENT_MODE)

    def test_real_wikipedia_index_is_readable_offline(self) -> None:
        if not eval_agent.WIKIPEDIA_INDEX_PATH.exists():
            self.skipTest("Wikipedia index is only present in the server project.")
        self.assertTrue(eval_agent.wikipedia_index_is_ready())
        result = eval_agent.retrieve_wikipedia_summary(
            {
                "task": {
                    "title": "旅行住宿计划",
                    "daily_life_domain": "travel",
                    "user_goal": "制定旅行和酒店安排",
                }
            }
        )
        self.assertTrue(result["retrieved"])
        self.assertTrue(result["article"]["title"])
        self.assertTrue(result["article"]["summary"])


if __name__ == "__main__":
    unittest.main()
