#!/usr/bin/env python3
"""Regression tests for BANK/PAG parity without external API calls."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BANK_DIR = Path(__file__).resolve().parent
EVAL_DIR = BANK_DIR.parent
PAG_DIR = EVAL_DIR / "pag"
for module_dir in (EVAL_DIR, BANK_DIR, PAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_bank
import judge_bank


class BankPagAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_bank._install_bank_behavior()
        judge_bank._install_bank_behavior()

    @staticmethod
    def _hidden_info() -> dict:
        return {
            "interactions": [
                {
                    "interaction_id": "interaction_001",
                    "kind": "preference_history",
                    "implicit_signal": "quiet hotel",
                    "turns": [
                        {"role": "user", "content": "KEEP_QUESTION"},
                        {"role": "assistant", "content": "KEEP_ANSWER"},
                    ],
                }
            ]
        }

    @staticmethod
    def _memory(*, interaction_id: str = "interaction_001", question: str = "KEEP_QUESTION") -> dict:
        return {
            "users": {
                "user_001": {
                    "qa_memories": [
                        {
                            "interaction_id": interaction_id,
                            "question": question,
                            "answer": "KEEP_ANSWER",
                            "summary": "KEEP_SUMMARY",
                        }
                    ]
                }
            }
        }

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
    def _prompt_context(*, include_summary: bool) -> dict:
        records = [
            {
                "rank": 1,
                "score": 1.0,
                "turns": [
                    {"role": "user", "content": "KEEP_QUESTION"},
                    {"role": "assistant", "content": "KEEP_ANSWER"},
                ],
            }
        ]
        if include_summary:
            records.append(
                {
                    "rank": 2,
                    "score": 0.8,
                    "turns": [
                        {"role": eval_bank.BANK_SUMMARY_ROLE, "content": "KEEP_SUMMARY"}
                    ],
                }
            )
        return {
            "task": {},
            "available_api_names": ["test.tool"],
            "available_api_contracts": [],
            "tool_catalog": {},
            "rag_retrieved_user_interaction_history": {
                "retrieval_method": "same_pag_retriever",
                "records": records,
            },
            eval_bank.eval_pag.PAG_ABSTRACT_CONTEXT_KEY: "KEEP_ABSTRACT",
        }

    @staticmethod
    def _user_prompt(messages: list[dict[str, str]]) -> str:
        return next(message["content"] for message in messages if message.get("role") == "user")

    @staticmethod
    def _rag_histories(prompt: str) -> list:
        prefix = eval_bank.eval_pag.eval_rag.RAG_PROMPT_PREFIX + "："
        start = prompt.index(prefix) + len(prefix)
        end = prompt.index("。\n\n", start)
        return json.loads(prompt[start:end])

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
            "RAG_PROMPT_MAX_CHARS",
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
            "DEFAULT_EVAL_CONCURRENCY",
            "QINGYUN_EVAL_CONCURRENCY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(eval_bank, name), getattr(eval_bank.eval_pag, name))
        self.assertEqual(eval_bank.BANK_RETRIEVAL_TOP_K, eval_bank.eval_pag.PAG_RETRIEVAL_TOP_K)

    def test_configured_core_runtime_matches_pag_except_mode(self) -> None:
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
        eval_bank.eval_pag._configure_evaluation("hard", 0)
        pag_values = {name: copy.deepcopy(getattr(eval_bank.eval_core, name)) for name in names}
        eval_bank.configure_evaluation("hard", 0)
        bank_values = {name: copy.deepcopy(getattr(eval_bank.eval_core, name)) for name in names}
        self.assertEqual(bank_values, pag_values)
        self.assertEqual(eval_bank.eval_core.PREFERENCE_INJECTION_MODE, eval_bank.BANK_MODE)

    def test_summary_is_added_as_a_separate_memory_piece(self) -> None:
        hidden_info = self._hidden_info()
        original = copy.deepcopy(hidden_info)
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=self._memory()):
            enriched = eval_bank._hidden_info_with_separate_bank_memories(
                user_id="user_001",
                hidden_info=hidden_info,
            )
        self.assertEqual(hidden_info, original)
        self.assertEqual(enriched["interactions"][0], original["interactions"][0])
        self.assertEqual(
            enriched["interactions"][1],
            {
                "interaction_id": f"interaction_001{eval_bank.BANK_SUMMARY_ID_SUFFIX}",
                "memory_piece_type": eval_bank.BANK_SUMMARY_PIECE_TYPE,
                "source_interaction_id": "interaction_001",
                "turns": [
                    {"role": eval_bank.BANK_SUMMARY_ROLE, "content": "KEEP_SUMMARY"}
                ],
            },
        )
        self.assertEqual(len(enriched["interactions"]), 2 * len(original["interactions"]))
        summary_piece = enriched["interactions"][1]
        self.assertEqual(
            eval_bank.eval_pag.eval_rag._interaction_search_text(summary_piece),
            "KEEP_SUMMARY",
        )

    def test_missing_raw_interaction_id_is_rejected(self) -> None:
        hidden_info = self._hidden_info()
        hidden_info["interactions"][0].pop("interaction_id")
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=self._memory()):
            with self.assertRaisesRegex(RuntimeError, "非空 interaction_id"):
                eval_bank._hidden_info_with_separate_bank_memories(
                    user_id="user_001",
                    hidden_info=hidden_info,
                )

    def test_all_raw_memories_precede_summary_memories_for_stable_ties(self) -> None:
        hidden_info = self._hidden_info()
        hidden_info["interactions"].append(
            {
                "interaction_id": "interaction_002",
                "kind": "preference_history",
                "turns": [
                    {"role": "user", "content": "SECOND_QUESTION"},
                    {"role": "assistant", "content": "SECOND_ANSWER"},
                ],
            }
        )
        memory = self._memory()
        memory["users"]["user_001"]["qa_memories"].append(
            {
                "interaction_id": "interaction_002",
                "question": "SECOND_QUESTION",
                "answer": "SECOND_ANSWER",
                "summary": "SECOND_SUMMARY",
            }
        )
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=memory):
            enriched = eval_bank._hidden_info_with_separate_bank_memories(
                user_id="user_001",
                hidden_info=hidden_info,
            )
        self.assertEqual(
            [item["interaction_id"] for item in enriched["interactions"]],
            [
                "interaction_001",
                "interaction_002",
                f"interaction_001{eval_bank.BANK_SUMMARY_ID_SUFFIX}",
                f"interaction_002{eval_bank.BANK_SUMMARY_ID_SUFFIX}",
            ],
        )

    def test_stale_memory_ids_are_rejected(self) -> None:
        with patch.object(
            eval_bank.eval_core,
            "load_bank_memory_doc",
            return_value=self._memory(interaction_id="stale_interaction"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing_summary_ids"):
                eval_bank._hidden_info_with_separate_bank_memories(
                    user_id="user_001",
                    hidden_info=self._hidden_info(),
                )

    def test_stale_memory_content_is_rejected(self) -> None:
        with patch.object(
            eval_bank.eval_core,
            "load_bank_memory_doc",
            return_value=self._memory(question="STALE_QUESTION"),
        ):
            with self.assertRaisesRegex(RuntimeError, "内容不一致"):
                eval_bank._hidden_info_with_separate_bank_memories(
                    user_id="user_001",
                    hidden_info=self._hidden_info(),
                )

    def test_bank_delegates_to_pag_with_only_augmented_hidden_info(self) -> None:
        kwargs = {
            "user_id": "user_001",
            "hidden_info": self._hidden_info(),
            "preference_mode": eval_bank.BANK_MODE,
            "sentinel": "KEEP_OTHER_ARGUMENTS",
        }
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=self._memory()):
            aligned = eval_bank._pag_aligned_kwargs_for_bank(kwargs)
        self.assertEqual(aligned["preference_mode"], eval_bank.eval_pag.PAG_MODE)
        self.assertEqual(aligned["sentinel"], kwargs["sentinel"])
        self.assertEqual(
            aligned["hidden_info"]["interactions"][0],
            kwargs["hidden_info"]["interactions"][0],
        )
        self.assertEqual(
            aligned["hidden_info"]["interactions"][1]["turns"],
            [{"role": eval_bank.BANK_SUMMARY_ROLE, "content": "KEEP_SUMMARY"}],
        )

    def test_pag_retriever_can_return_raw_or_summary_memory_piece(self) -> None:
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=self._memory()):
            enriched = eval_bank._hidden_info_with_separate_bank_memories(
                user_id="user_001",
                hidden_info=self._hidden_info(),
            )
        raw_retrieval = eval_bank.eval_pag.eval_rag.retrieve_hidden_info_for_task(
            hidden_info=enriched,
            task_doc={"task": {"title": "KEEP_QUESTION"}},
            top_k=1,
        )
        summary_retrieval = eval_bank.eval_pag.eval_rag.retrieve_hidden_info_for_task(
            hidden_info=enriched,
            task_doc={"task": {"title": "KEEP_SUMMARY"}},
            top_k=1,
        )
        self.assertEqual(raw_retrieval["records"][0]["interaction_id"], "interaction_001")
        self.assertEqual(
            summary_retrieval["records"][0]["interaction_id"],
            f"interaction_001{eval_bank.BANK_SUMMARY_ID_SUFFIX}",
        )
        self.assertEqual(
            summary_retrieval["records"][0]["turns"],
            [{"role": eval_bank.BANK_SUMMARY_ROLE, "content": "KEEP_SUMMARY"}],
        )

    def test_top_k_competes_over_one_unified_pool_without_pairing(self) -> None:
        hidden_info = {
            "interactions": [
                {
                    "interaction_id": "interaction_001",
                    "turns": [
                        {"role": "user", "content": "RAW_ALPHA"},
                        {"role": "assistant", "content": "RAW_ANSWER"},
                    ],
                },
                {
                    "interaction_id": "interaction_002",
                    "turns": [
                        {"role": "user", "content": "UNRELATED_RAW"},
                        {"role": "assistant", "content": "UNRELATED_ANSWER"},
                    ],
                },
            ]
        }
        memory = {
            "users": {
                "user_001": {
                    "qa_memories": [
                        {
                            "interaction_id": "interaction_001",
                            "question": "RAW_ALPHA",
                            "answer": "RAW_ANSWER",
                            "summary": "SUMMARY_ALPHA",
                        },
                        {
                            "interaction_id": "interaction_002",
                            "question": "UNRELATED_RAW",
                            "answer": "UNRELATED_ANSWER",
                            "summary": "UNRELATED_SUMMARY",
                        },
                    ]
                }
            }
        }
        with patch.object(eval_bank.eval_core, "load_bank_memory_doc", return_value=memory):
            enriched = eval_bank._hidden_info_with_separate_bank_memories(
                user_id="user_001",
                hidden_info=hidden_info,
            )

        self.assertEqual(len(enriched["interactions"]), 4)
        retrieval = eval_bank.eval_pag.eval_rag.retrieve_hidden_info_for_task(
            hidden_info=enriched,
            task_doc={"task": {"title": "RAW_ALPHA SUMMARY_ALPHA"}},
            top_k=3,
        )
        retrieved_ids = [record["interaction_id"] for record in retrieval["records"]]
        self.assertEqual(len(retrieved_ids), 3)
        self.assertIn("interaction_001", retrieved_ids)
        self.assertIn(f"interaction_001{eval_bank.BANK_SUMMARY_ID_SUFFIX}", retrieved_ids)
        self.assertEqual(
            sum(item.endswith(eval_bank.BANK_SUMMARY_ID_SUFFIX) for item in retrieved_ids),
            1,
        )

    def test_all_generation_prompts_have_only_summary_delta(self) -> None:
        eval_bank.configure_evaluation("hard", 0)
        pag_context = self._prompt_context(include_summary=False)
        bank_context = self._prompt_context(include_summary=True)

        def prompt_groups(context: dict) -> tuple[list[dict[str, str]], ...]:
            return (
                eval_bank.eval_core.initial_plan_messages(context),
                eval_bank.eval_core.action_messages(
                    context=context,
                    current_plan=self._plan(),
                    trajectory_summary=[],
                    iteration=1,
                ),
                eval_bank.eval_core.update_plan_messages(
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

        pag_groups = prompt_groups(pag_context)
        bank_groups = prompt_groups(bank_context)
        for pag_messages, bank_messages in zip(pag_groups, bank_groups):
            pag_prompt = self._user_prompt(pag_messages)
            bank_prompt = self._user_prompt(bank_messages)
            self.assertEqual(pag_prompt.count(eval_bank.eval_pag.eval_rag.RAG_PROMPT_PREFIX), 1)
            self.assertEqual(bank_prompt.count(eval_bank.eval_pag.eval_rag.RAG_PROMPT_PREFIX), 1)
            self.assertEqual(pag_prompt.count(eval_bank.eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
            self.assertEqual(bank_prompt.count(eval_bank.eval_pag.PAG_ABSTRACT_PROMPT_PREFIX), 1)
            pag_histories = self._rag_histories(pag_prompt)
            bank_histories = self._rag_histories(bank_prompt)
            self.assertEqual(bank_histories[0], pag_histories[0])
            self.assertEqual(
                bank_histories[1],
                [{"role": eval_bank.BANK_SUMMARY_ROLE, "content": "KEEP_SUMMARY"}],
            )
            self.assertNotIn("KEEP_SUMMARY", pag_prompt)
            self.assertIn("KEEP_SUMMARY", bank_prompt)
        for prompt in (self._user_prompt(pag_groups[-1]), self._user_prompt(bank_groups[-1])):
            self.assertNotIn("DROP_LATENCY", prompt)
            self.assertNotIn("DROP_REQUEST_ID", prompt)
            self.assertIn("KEEP_OBSERVATION", prompt)

    def test_bank_metadata_records_summary_as_the_only_method_delta(self) -> None:
        record = {
            "preference_injection_mode": "pag",
            "preference_injection_label": "pag",
            "provided_eval_context": {"mode": "pag", "context": {}},
            "judge_results": {"offline_judge_entrypoint": "python eval/pag/judge_pag.py"},
            "hyperparameters": {
                "rag_top_k": 3,
                "rag_retrieval_injection_phases": ["initial_plan", "reason_action", "update_plan"],
                "pag_difference_from_rag": "user_abstract_profile_only",
            },
        }
        aligned = eval_bank._align_bank_record_metadata(record)
        self.assertEqual(aligned["preference_injection_mode"], eval_bank.BANK_MODE)
        self.assertEqual(aligned["provided_eval_context"]["mode"], eval_bank.BANK_MODE)
        self.assertEqual(
            aligned["judge_results"]["offline_judge_entrypoint"],
            "python eval/bank/judge_bank.py",
        )
        hyperparameters = aligned["hyperparameters"]
        self.assertNotIn("pag_difference_from_rag", hyperparameters)
        self.assertEqual(
            hyperparameters["bank_difference_from_pag"],
            "independent_summary_memory_pieces_added_to_pag_candidate_pool_only",
        )
        self.assertEqual(
            hyperparameters["rag_retrieval_method"],
            "pag_bm25_over_unified_raw_qa_and_independent_summary_memory_pieces",
        )

    def test_judge_runtime_and_context_match_pag_structure(self) -> None:
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
                self.assertEqual(getattr(judge_bank, name), getattr(judge_bank.judge_pag, name))

        record = {
            "preference_injection_mode": judge_bank.BANK_MODE,
            "provided_eval_context": {
                "mode": judge_bank.BANK_MODE,
                "context": {
                    "rag_retrieved_user_interaction_history": {"records": ["KEEP_RAG"]},
                    judge_bank.PAG_ABSTRACT_CONTEXT_KEY: "KEEP_ABSTRACT",
                },
            },
        }
        core = judge_bank.judge.profile_core(record)
        self.assertEqual(
            core["rag_retrieved_user_interaction_history"],
            {"records": ["KEEP_RAG"]},
        )
        self.assertEqual(core[judge_bank.PAG_ABSTRACT_CONTEXT_KEY], "KEEP_ABSTRACT")
        self.assertIn("summary", judge_bank.judge.preference_source_description(record))

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
        judge_bank.judge_pag._configure_judge("hard", 0)
        pag_values = {name: copy.deepcopy(getattr(judge_bank.judge, name)) for name in names}
        judge_bank._configure_judge("hard", 0)
        bank_values = {name: copy.deepcopy(getattr(judge_bank.judge, name)) for name in names}
        self.assertEqual(bank_values, pag_values)
        self.assertEqual(judge_bank.judge.PREFERENCE_INJECTION_MODE, judge_bank.BANK_MODE)

    def test_bank_uses_separate_result_directory(self) -> None:
        result_dir = eval_bank.eval_core.result_dir_for_model("test-model", eval_bank.BANK_MODE, 1)
        self.assertTrue(str(result_dir).endswith("test-model/hard/bank/pass_1"))


if __name__ == "__main__":
    unittest.main()
