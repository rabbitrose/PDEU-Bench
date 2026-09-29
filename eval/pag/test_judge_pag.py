#!/usr/bin/env python3
"""Regression tests for RAG-aligned PAG judge behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PAG_DIR = Path(__file__).resolve().parent
if str(PAG_DIR) not in sys.path:
    sys.path.insert(0, str(PAG_DIR))

import judge_pag


class PagJudgeBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        judge_pag._install_pag_behavior()

    def test_runtime_configuration_matches_rag(self) -> None:
        names = (
            "TARGET_GEN_LLM",
            "PASS_N",
            "ENABLE_LONG_CONTEXT",
            "LONG_CONTEXT_TURNS",
            "LONG_CONTEXT_TURNS_LIST",
            "TARGET_EVAL_RUN_ID",
            "JUDGE_LATEST_EVAL_RUN_ONLY",
            "DEFAULT_JUDGE_CONCURRENCY",
            "QINGYUN_JUDGE_CONCURRENCY",
            "USER_ID_FILTER",
            "TASK_ID_FILTER",
            "TASK_DIFFICULTY",
            "TASK_DIFFICULTY_LIST",
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(judge_pag, name), getattr(judge_pag.judge_rag, name))

    def test_pag_judge_context_is_rag_context_plus_abstract(self) -> None:
        rag_history = {"records": [{"turns": [{"role": "user", "content": "QA"}]}]}
        record = {
            "user_id": "user_001",
            "preference_injection_mode": "pag",
            "provided_eval_context": {
                "mode": "pag",
                "context": {
                    "rag_retrieved_user_interaction_history": rag_history,
                    judge_pag.PAG_ABSTRACT_CONTEXT_KEY: "ABSTRACT",
                },
            },
            "sandbox_context": {"user_context": {}},
        }
        core = judge_pag.judge.profile_core(record)
        self.assertEqual(core["rag_retrieved_user_interaction_history"], rag_history)
        self.assertEqual(core[judge_pag.PAG_ABSTRACT_CONTEXT_KEY], "ABSTRACT")
        source = judge_pag.judge.preference_source_description(record)
        self.assertIn("Top-3", source)
        self.assertIn("User abstract profile", source)

    def test_pag_judge_rejects_missing_abstract(self) -> None:
        record = {
            "user_id": "user_001",
            "preference_injection_mode": "pag",
            "provided_eval_context": {
                "mode": "pag",
                "context": {
                    "rag_retrieved_user_interaction_history": {"records": []},
                },
            },
            "sandbox_context": {"user_context": {}},
        }
        with self.assertRaisesRegex(RuntimeError, judge_pag.PAG_ABSTRACT_CONTEXT_KEY):
            judge_pag.judge.profile_core(record)


if __name__ == "__main__":
    unittest.main()
