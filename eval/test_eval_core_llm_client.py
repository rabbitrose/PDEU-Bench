#!/usr/bin/env python3
"""Regression tests for generation API response parsing."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_core


class EvalCoreLLMClientTest(unittest.TestCase):
    def test_extracts_chat_completion_content_and_usage(self) -> None:
        raw = {
            "choices": [{"message": {"content": '{"phase":"initial_plan"}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        }

        content, usage = eval_core.LLMClient._extract_content_and_usage(raw)

        self.assertEqual(content, '{"phase":"initial_plan"}')
        self.assertEqual(usage["total_tokens"], 17)

    def test_supports_legacy_text_completion(self) -> None:
        raw = {"choices": [{"text": "legacy response"}], "usage": {"total_tokens": 3}}

        content, usage = eval_core.LLMClient._extract_content_and_usage(raw)

        self.assertEqual(content, "legacy response")
        self.assertEqual(usage, {"total_tokens": 3})

    def test_preserves_unstructured_or_unknown_responses(self) -> None:
        self.assertEqual(eval_core.LLMClient._extract_content_and_usage("plain text"), ("plain text", {}))

        raw = {"unexpected": True, "usage": {"total_tokens": 1}}
        content, usage = eval_core.LLMClient._extract_content_and_usage(raw)
        self.assertEqual(json.loads(content), raw)
        self.assertEqual(usage, {"total_tokens": 1})


if __name__ == "__main__":
    unittest.main()
