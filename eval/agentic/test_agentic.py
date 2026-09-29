#!/usr/bin/env python3
"""Offline regression tests for PAG-aligned Agentic Memory evaluation."""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch


AGENTIC_DIR = Path(__file__).resolve().parent
EVAL_DIR = AGENTIC_DIR.parent
PAG_DIR = EVAL_DIR / "pag"
for module_dir in (EVAL_DIR, AGENTIC_DIR, PAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_agentic
import judge_agentic


class _FakeClient:
    def __init__(self, name: str = "shared-client") -> None:
        self.name = name
        self.model = "test-model"
        self.base_url = "https://test.invalid/v1"
        self.api_key = "test-key"


class AgenticPagAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        eval_agentic._install_agentic_behavior()
        eval_agentic.configure_evaluation("hard", 0)
        installer = getattr(
            judge_agentic,
            "_install_agentic_behavior",
            getattr(judge_agentic, "_install_agentic_judge_behavior", None),
        )
        if not callable(installer):
            raise AssertionError("judge_agentic 缺少 Agentic judge 安装函数。")
        installer()

    def setUp(self) -> None:
        self._clear_task_state()

    def tearDown(self) -> None:
        self._clear_task_state()

    @staticmethod
    def _clear_task_state() -> None:
        if hasattr(eval_agentic._TASK_STATE, "value"):
            delattr(eval_agentic._TASK_STATE, "value")

    @staticmethod
    def _state(
        *,
        user_id: str = "user_001",
        task_id: str = "task_hard",
        initial_profile: str = "PROFILE_P0",
        current_profile: str | None = None,
    ) -> eval_agentic._AgenticTaskState:
        profile = initial_profile if current_profile is None else current_profile
        return eval_agentic._AgenticTaskState(
            user_id=user_id,
            task_id=task_id,
            run_id="run_test",
            pass_n=1,
            pass_index=1,
            detail_path=None,
            initial_profile=initial_profile,
            current_profile=profile,
        )

    @staticmethod
    @contextmanager
    def _activated(
        state: eval_agentic._AgenticTaskState,
    ) -> Iterator[eval_agentic._AgenticTaskState]:
        if eval_agentic._active_state() is not None:
            raise AssertionError("测试线程中残留 Agentic task state。")
        eval_agentic._TASK_STATE.value = state
        try:
            yield state
        finally:
            if hasattr(eval_agentic._TASK_STATE, "value"):
                delattr(eval_agentic._TASK_STATE, "value")

    @staticmethod
    def _context(profile: str = "PROFILE_P0") -> dict[str, Any]:
        return {
            "task": {"title": "KEEP_TASK"},
            "available_api_names": ["test.tool"],
            "available_api_contracts": [],
            "tool_catalog": {},
            "rag_retrieved_user_interaction_history": {
                "retrieval_method": "LEAK_METHOD",
                "records": [
                    {
                        "rank": 1,
                        "score": "LEAK_SCORE",
                        "interaction_id": "LEAK_ID",
                        "turns": [
                            {"role": "user", "content": "KEEP_QUESTION"},
                            {"role": "assistant", "content": "KEEP_ANSWER"},
                        ],
                    }
                ],
            },
            eval_agentic.eval_pag.PAG_ABSTRACT_CONTEXT_KEY: profile,
        }

    @staticmethod
    def _plan() -> dict[str, Any]:
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
    def _action(*, action_type: str = "tool_call") -> dict[str, Any]:
        action: dict[str, Any] = {
            "phase": "reason_action",
            "iteration": 1,
            "task_status": "in_progress",
            "selected_step_id": None,
            "reason": "test",
            "action_type": action_type,
            "completion_assessment": {},
        }
        if action_type == "tool_call":
            action["tool_call"] = {"name": "test.tool", "arguments": {}}
        else:
            action["final_answer"] = {"summary": "finished"}
        return action

    @classmethod
    def _prompt_groups(cls, context: dict[str, Any]) -> tuple[list[dict[str, str]], ...]:
        return (
            eval_agentic.eval_core.initial_plan_messages(context),
            eval_agentic.eval_core.action_messages(
                context=context,
                current_plan=cls._plan(),
                trajectory_summary=[],
                iteration=1,
            ),
            eval_agentic.eval_core.update_plan_messages(
                context=context,
                previous_plan=cls._plan(),
                action_json=cls._action(),
                sandbox_observation={
                    "sandbox_status": "success",
                    "transport": {"latency": "DROP_LATENCY"},
                    "request_id": "DROP_REQUEST_ID",
                    "response_body": {"summary": "KEEP_OBSERVATION"},
                },
                iteration=1,
            ),
        )

    @staticmethod
    def _user_prompt(messages: list[dict[str, str]]) -> str:
        return next(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        )

    @staticmethod
    def _replace_profile(
        messages: list[dict[str, str]],
        *,
        before: str,
        after: str,
    ) -> list[dict[str, str]]:
        replaced = copy.deepcopy(messages)
        for message in replaced:
            if isinstance(message.get("content"), str):
                message["content"] = message["content"].replace(before, after)
        return replaced

    @staticmethod
    def _update_result(profile: str) -> dict[str, Any]:
        return {
            "updated_user_abstract": profile,
            "update_decision": "updated",
            "newly_supported_preferences": [f"evidence for {profile}"],
            "weakened_or_removed_preferences": [],
            "evidence_summary": [f"updated to {profile}"],
        }

    @staticmethod
    def _completed_state() -> eval_agentic._AgenticTaskState:
        state = AgenticPagAlignmentTest._state(
            initial_profile="PROFILE_P0",
            current_profile="PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE",
        )
        profiles = (
            (
                "initial_plan",
                0,
                AgenticPagAlignmentTest._plan(),
                "PROFILE_P0",
                "PROFILE_P1",
            ),
            (
                "reason_action",
                1,
                AgenticPagAlignmentTest._action(action_type="finish"),
                "PROFILE_P1",
                "PROFILE_P2",
            ),
            (
                "update_plan",
                1,
                {
                    **AgenticPagAlignmentTest._plan(),
                    "phase": "update_plan",
                    "iteration": 1,
                    "plan_version": 2,
                },
                "PROFILE_P2",
                "PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE",
            ),
        )
        state.current_version = 3
        state.main_call_count = 3
        state.updates = [
            {
                "sequence": sequence,
                "phase": phase,
                "iteration": iteration,
                "main_purpose": f"user_001/task_hard/{phase}"
                + (f"/{iteration}" if iteration else ""),
                "profile_version_used": sequence - 1,
                "profile_snapshot_id_used": f"profile_v{sequence - 1}",
                "profile_before": before,
                "main_input_message_count": 1,
                "main_input_sha256": eval_agentic._json_digest(
                    [{"role": "user", "content": f"INPUT_{phase}_{iteration}"}]
                ),
                "main_output_sha256": eval_agentic._json_digest(parsed),
                "same_llm_client_object": True,
                "profile_update_model": "test-model",
                "profile_update_base_url": "https://test.invalid/v1",
                "status": "completed",
                "profile_after": after,
                "profile_version_after": sequence,
                "profile_snapshot_id_after": f"profile_v{sequence}",
                "update_decision": "updated",
                "newly_supported_preferences": [],
                "weakened_or_removed_preferences": [],
                "evidence_summary": [],
                "profile_update_attempts": [],
            }
            for sequence, (phase, iteration, parsed, before, after) in enumerate(
                profiles,
                start=1,
            )
        ]
        return state

    def _aligned_record(self) -> dict[str, Any]:
        record = {
            "user_id": "user_001",
            "task_id": "task_hard",
            "model": {
                "name": "test-model",
                "base_url": "https://test.invalid/v1",
            },
            "preference_injection_mode": eval_agentic.eval_pag.PAG_MODE,
            "preference_injection_label": eval_agentic.eval_pag.PAG_LABEL,
            "trajectory": [
                {
                    "phase": "initial_plan",
                    "iteration": 0,
                    "parsed": self._plan(),
                    "llm_attempts": [{"ok": True}],
                },
                {
                    "phase": "reason_action",
                    "iteration": 1,
                    "parsed": self._action(action_type="finish"),
                    "llm_attempts": [{"ok": True}],
                },
                {
                    "phase": "update_plan",
                    "iteration": 1,
                    "parsed": {
                        **self._plan(),
                        "phase": "update_plan",
                        "iteration": 1,
                        "plan_version": 2,
                    },
                    "llm_attempts": [{"ok": True}],
                },
            ],
            "provided_eval_context": {
                "mode": eval_agentic.eval_pag.PAG_MODE,
                "context": {
                    "rag_retrieved_user_interaction_history": {
                        "records": [{"turns": [{"role": "user", "content": "QA"}]}]
                    },
                    eval_agentic.eval_pag.PAG_ABSTRACT_CONTEXT_KEY: (
                        "PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE"
                    ),
                },
            },
            "sandbox_context": {"user_context": {}},
            "judge_results": {
                "offline_judge_entrypoint": "python eval/pag/judge_pag.py"
            },
            "judge_summary": {
                "enabled": False,
                "reason": "python eval/pag/judge_pag.py",
            },
            "hyperparameters": {
                "pag_difference_from_rag": "user_abstract_profile_only"
            },
        }
        return eval_agentic._align_agentic_record_metadata(
            record,
            self._completed_state(),
        )

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
            "PAG_RETRIEVAL_TOP_K",
            "RAG_PROMPT_MAX_CHARS",
            "DEEPSEEK_CONCURRENCY_BY_DIFFICULTY",
            "DEFAULT_EVAL_CONCURRENCY",
            "QINGYUN_EVAL_CONCURRENCY",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(eval_agentic, name),
                    getattr(eval_agentic.eval_pag, name),
                )

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
        eval_agentic.eval_pag._configure_evaluation("hard", 0)
        pag_values = {
            name: copy.deepcopy(getattr(eval_agentic.eval_core, name)) for name in names
        }
        eval_agentic.configure_evaluation("hard", 0)
        agentic_values = {
            name: copy.deepcopy(getattr(eval_agentic.eval_core, name)) for name in names
        }
        self.assertEqual(agentic_values, pag_values)
        self.assertEqual(
            eval_agentic.eval_core.PREFERENCE_INJECTION_MODE,
            eval_agentic.AGENTIC_MODE,
        )

    def test_all_main_prompts_are_byte_aligned_with_pag_except_profile_value(self) -> None:
        context = self._context("PROFILE_P0")
        pag_groups = self._prompt_groups(context)
        state = self._state(
            initial_profile="PROFILE_P0",
            current_profile="PROFILE_CURRENT_DYNAMIC",
        )
        state.context = context
        with self._activated(state):
            agentic_groups = self._prompt_groups(context)

        self.assertEqual(
            context[eval_agentic.eval_pag.PAG_ABSTRACT_CONTEXT_KEY],
            "PROFILE_P0",
        )
        for pag_messages, agentic_messages in zip(pag_groups, agentic_groups):
            self.assertEqual(
                self._replace_profile(
                    agentic_messages,
                    before="PROFILE_CURRENT_DYNAMIC",
                    after="PROFILE_P0",
                ),
                pag_messages,
            )
            prompt = self._user_prompt(agentic_messages)
            self.assertEqual(
                prompt.count(eval_agentic.eval_pag.PAG_ABSTRACT_PROMPT_PREFIX),
                1,
            )
            self.assertEqual(
                prompt.count(eval_agentic.eval_pag.eval_rag.RAG_PROMPT_PREFIX),
                1,
            )
            self.assertIn("KEEP_QUESTION", prompt)
            self.assertIn("KEEP_ANSWER", prompt)
            self.assertNotIn("LEAK_METHOD", prompt)
            self.assertNotIn("LEAK_SCORE", prompt)
            self.assertNotIn("LEAK_ID", prompt)

        update_prompt = self._user_prompt(agentic_groups[-1])
        self.assertIn("KEEP_OBSERVATION", update_prompt)
        self.assertNotIn("DROP_LATENCY", update_prompt)
        self.assertNotIn("DROP_REQUEST_ID", update_prompt)

    def test_long_main_input_is_bounded_without_dropping_main_output(self) -> None:
        output_marker = "MAIN_OUTPUT_MUST_SURVIVE_INPUT_TRUNCATION"
        messages = eval_agentic._profile_update_messages(
            profile_before="PROFILE_P0",
            phase="reason_action",
            iteration=1,
            main_messages=[
                {
                    "role": "user",
                    "content": "INPUT_START_" + ("X" * 110_000) + "_INPUT_END",
                }
            ],
            main_output={
                "phase": "reason_action",
                "iteration": 1,
                "summary": output_marker,
            },
        )
        user_content = self._user_prompt(messages)
        interaction = user_content.split(
            "本轮主 LLM input/output:\n",
            1,
        )[1].split("\n\n必须遵守的 JSON schema:\n", 1)[0]

        self.assertLessEqual(
            len(interaction),
            eval_agentic.AGENTIC_PROFILE_UPDATE_INPUT_MAX_CHARS,
        )
        self.assertIn(output_marker, interaction)
        self.assertIn("llm_output（优先保留，过长时单独截断）", interaction)
        self.assertIn("INPUT_START_", interaction)
        self.assertNotIn("_INPUT_END", interaction)

    def test_p0_to_p3_update_chain_uses_same_client_without_recursion(self) -> None:
        state = self._state(initial_profile="PROFILE_P0")
        context = self._context("PROFILE_P0")
        state.context = context
        client = _FakeClient()
        calls: list[dict[str, Any]] = []
        update_profiles = iter(("PROFILE_P1", "PROFILE_P2", "PROFILE_P3"))
        main_outputs = {
            "user_001/task_hard/initial_plan": self._plan(),
            "user_001/task_hard/reason_action/1": self._action(action_type="finish"),
            "user_001/task_hard/update_plan/1": {
                **self._plan(),
                "phase": "update_plan",
                "iteration": 1,
                "plan_version": 2,
            },
        }

        def fake_request(
            request_client: Any,
            *,
            messages: list[dict[str, str]],
            purpose: str,
            validator: Any,
            logger: Any,
            max_tokens: int | None = None,
            temperature: float | None = None,
            timeout: int | None = None,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            calls.append(
                {
                    "client": request_client,
                    "messages": copy.deepcopy(messages),
                    "purpose": purpose,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout": timeout,
                }
            )
            if "/agentic_profile_update/" in purpose:
                return self._update_result(next(update_profiles)), []
            return copy.deepcopy(main_outputs[purpose]), []

        with (
            patch.object(
                eval_agentic,
                "_BASE_REQUEST_STRUCTURED_JSON",
                new=fake_request,
            ),
            patch.object(
                eval_agentic.eval_core,
                "persist_llm_attempt_details",
                return_value=[],
            ),
            self._activated(state),
        ):
            phase_calls = (
                (
                    "user_001/task_hard/initial_plan",
                    lambda: eval_agentic.eval_core.initial_plan_messages(context),
                ),
                (
                    "user_001/task_hard/reason_action/1",
                    lambda: eval_agentic.eval_core.action_messages(
                        context=context,
                        current_plan=self._plan(),
                        trajectory_summary=[],
                        iteration=1,
                    ),
                ),
                (
                    "user_001/task_hard/update_plan/1",
                    lambda: eval_agentic.eval_core.update_plan_messages(
                        context=context,
                        previous_plan=self._plan(),
                        action_json=self._action(),
                        sandbox_observation={"response_body": {"summary": "ok"}},
                        iteration=1,
                    ),
                ),
            )
            for purpose, build_messages in phase_calls:
                messages = build_messages()
                parsed, _ = eval_agentic.eval_core.request_structured_json(
                    client,
                    messages=messages,
                    purpose=purpose,
                    validator=lambda obj: [],
                    logger=None,
                )
                self.assertEqual(parsed, main_outputs[purpose])

        expected_purposes = [
            "user_001/task_hard/initial_plan",
            "user_001/task_hard/agentic_profile_update/001_initial_plan_0",
            "user_001/task_hard/reason_action/1",
            "user_001/task_hard/agentic_profile_update/002_reason_action_1",
            "user_001/task_hard/update_plan/1",
            "user_001/task_hard/agentic_profile_update/003_update_plan_1",
        ]
        self.assertEqual([call["purpose"] for call in calls], expected_purposes)
        self.assertTrue(all(call["client"] is client for call in calls))
        self.assertEqual(len(calls), 6)

        main_calls = [call for call in calls if "/agentic_profile_update/" not in call["purpose"]]
        for call, profile in zip(main_calls, ("PROFILE_P0", "PROFILE_P1", "PROFILE_P2")):
            prompt = self._user_prompt(call["messages"])
            self.assertIn(profile, prompt)
        self.assertNotIn("PROFILE_P1", self._user_prompt(main_calls[0]["messages"]))
        self.assertNotIn("PROFILE_P2", self._user_prompt(main_calls[1]["messages"]))
        self.assertNotIn("PROFILE_P3", self._user_prompt(main_calls[2]["messages"]))

        update_calls = [call for call in calls if "/agentic_profile_update/" in call["purpose"]]
        for call in update_calls:
            self.assertEqual(
                call["max_tokens"],
                eval_agentic.AGENTIC_PROFILE_UPDATE_MAX_TOKENS,
            )
            self.assertEqual(
                call["temperature"],
                eval_agentic.AGENTIC_PROFILE_UPDATE_TEMPERATURE,
            )
            self.assertEqual(
                call["timeout"],
                eval_agentic.AGENTIC_PROFILE_UPDATE_TIMEOUT_SECONDS,
            )

        self.assertEqual(state.current_profile, "PROFILE_P3")
        self.assertEqual(state.current_version, 3)
        self.assertEqual(state.main_call_count, 3)
        self.assertEqual(
            [event["phase"] for event in state.updates],
            ["initial_plan", "reason_action", "update_plan"],
        )
        self.assertEqual(
            [event["profile_before"] for event in state.updates],
            ["PROFILE_P0", "PROFILE_P1", "PROFILE_P2"],
        )
        self.assertEqual(
            [event["profile_after"] for event in state.updates],
            ["PROFILE_P1", "PROFILE_P2", "PROFILE_P3"],
        )
        self.assertTrue(
            all(event["same_llm_client_object"] for event in state.updates)
        )
        self.assertEqual(state.updates[1]["phase"], "reason_action")

    def test_profile_update_failure_is_fail_open_and_audited(self) -> None:
        state = self._state(initial_profile="PROFILE_P0")
        client = _FakeClient()
        purposes: list[str] = []

        def fake_request(
            request_client: Any,
            *,
            messages: list[dict[str, str]],
            purpose: str,
            validator: Any,
            logger: Any,
            **kwargs: Any,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            self.assertIs(request_client, client)
            purposes.append(purpose)
            if "/agentic_profile_update/" in purpose:
                raise RuntimeError("EXPECTED_UPDATE_FAILURE")
            return self._plan(), []

        with (
            patch.object(
                eval_agentic,
                "_BASE_REQUEST_STRUCTURED_JSON",
                new=fake_request,
            ),
            patch.object(
                eval_agentic.eval_core,
                "persist_llm_attempt_details",
                return_value=[],
            ),
            patch.object(eval_agentic, "AGENTIC_PROFILE_UPDATE_STRICT", False),
            self._activated(state),
        ):
            parsed, _ = eval_agentic.eval_core.request_structured_json(
                client,
                messages=[{"role": "user", "content": "MAIN_INPUT"}],
                purpose="user_001/task_hard/initial_plan",
                validator=lambda obj: [],
                logger=None,
            )

        self.assertEqual(parsed, self._plan())
        self.assertEqual(len(purposes), 2)
        self.assertEqual(state.current_profile, "PROFILE_P0")
        self.assertEqual(state.current_version, 0)
        self.assertEqual(state.main_call_count, 1)
        self.assertEqual(len(state.updates), 1)
        event = state.updates[0]
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["profile_before"], "PROFILE_P0")
        self.assertEqual(event["profile_after"], "PROFILE_P0")
        self.assertEqual(event["profile_version_after"], 0)
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertIn("EXPECTED_UPDATE_FAILURE", event["error"])

    def test_thread_local_profiles_do_not_cross_between_tasks(self) -> None:
        barrier = threading.Barrier(2)

        def fake_request(
            request_client: Any,
            *,
            messages: list[dict[str, str]],
            purpose: str,
            validator: Any,
            logger: Any,
            **kwargs: Any,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            user_id = purpose.split("/", 1)[0]
            if "/agentic_profile_update/" in purpose:
                return self._update_result(f"UPDATED_{user_id}"), []
            barrier.wait(timeout=5)
            return {"phase": "initial_plan", "owner": user_id}, []

        def worker(user_id: str) -> tuple[str, str, list[dict[str, Any]]]:
            state = self._state(
                user_id=user_id,
                task_id="shared_task_id",
                initial_profile=f"INITIAL_{user_id}",
            )
            client = _FakeClient(user_id)
            with self._activated(state):
                eval_agentic.eval_core.request_structured_json(
                    client,
                    messages=[{"role": "user", "content": f"INPUT_{user_id}"}],
                    purpose=f"{user_id}/shared_task_id/initial_plan",
                    validator=lambda obj: [],
                    logger=None,
                )
            return state.initial_profile, state.current_profile, state.updates

        with (
            patch.object(
                eval_agentic,
                "_BASE_REQUEST_STRUCTURED_JSON",
                new=fake_request,
            ),
            patch.object(
                eval_agentic.eval_core,
                "persist_llm_attempt_details",
                return_value=[],
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [executor.submit(worker, user_id) for user_id in ("user_A", "user_B")]
            results = [future.result(timeout=10) for future in futures]

        self.assertEqual(
            {(initial, current) for initial, current, _ in results},
            {
                ("INITIAL_user_A", "UPDATED_user_A"),
                ("INITIAL_user_B", "UPDATED_user_B"),
            },
        )
        for initial, current, updates in results:
            self.assertEqual(len(updates), 1)
            self.assertEqual(updates[0]["profile_before"], initial)
            self.assertEqual(updates[0]["profile_after"], current)
        self.assertIsNone(eval_agentic._active_state())

    def test_record_keeps_initial_profile_as_provided_context(self) -> None:
        aligned = self._aligned_record()
        provided = aligned["provided_eval_context"]
        self.assertEqual(provided["mode"], eval_agentic.AGENTIC_MODE)
        self.assertEqual(
            provided["context"][eval_agentic.eval_pag.PAG_ABSTRACT_CONTEXT_KEY],
            "PROFILE_P0",
        )
        self.assertNotIn(
            "PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE",
            json.dumps(provided, ensure_ascii=False),
        )

        memory = aligned["agentic_profile_memory"]
        self.assertEqual(memory["initial_profile"], "PROFILE_P0")
        self.assertEqual(
            memory["final_profile"],
            "PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE",
        )
        self.assertEqual(memory["final_profile_version"], 3)
        self.assertEqual(len(memory["snapshots"]), 4)
        self.assertEqual(
            [
                event["agentic_profile_memory"]["profile_used"]
                for event in aligned["trajectory"]
            ],
            ["PROFILE_P0", "PROFILE_P1", "PROFILE_P2"],
        )
        self.assertEqual(
            aligned["judge_results"]["offline_judge_entrypoint"],
            "python eval/agentic/judge_agentic.py",
        )
        self.assertIn(
            "python eval/agentic/judge_agentic.py",
            aligned["judge_summary"]["reason"],
        )

    def test_judge_uses_initial_profile_and_never_final_profile(self) -> None:
        record = self._aligned_record()
        audit = judge_agentic._validate_agentic_profile_memory(record)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["judge_reference_profile_version"], 0)
        self.assertFalse(audit["dynamic_profiles_used_for_scoring"])

        core = judge_agentic.judge.profile_core(record)
        self.assertEqual(
            core[judge_agentic.PAG_ABSTRACT_CONTEXT_KEY],
            "PROFILE_P0",
        )
        judge_context = judge_agentic.judge.build_judge_context(record)
        serialized = json.dumps(judge_context, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(
            "PROFILE_P3_FINAL_SHOULD_NOT_BE_JUDGE_EVIDENCE",
            serialized,
        )
        self.assertIn("Top-3", judge_agentic.judge.preference_source_description(record))

    def test_judge_rejects_a_broken_dynamic_profile_chain(self) -> None:
        record = self._aligned_record()
        record["agentic_profile_memory"]["updates"][1]["profile_before"] = (
            "BROKEN_CROSS_TASK_PROFILE"
        )
        with self.assertRaisesRegex(RuntimeError, "profile_before"):
            judge_agentic._validate_agentic_profile_memory(record)

    def test_judge_clears_inherited_proxy_environment(self) -> None:
        proxy_values = {name: "http://unreachable.invalid:7890" for name in judge_agentic.PROXY_ENV_NAMES}
        with patch.dict(os.environ, proxy_values, clear=False):
            removed = judge_agentic._clear_proxy_environment()
            self.assertEqual(set(removed), set(proxy_values))
            self.assertTrue(all(name not in os.environ for name in proxy_values))

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
        judge_agentic.judge_pag._configure_judge("hard", 0)
        pag_values = {
            name: copy.deepcopy(getattr(judge_agentic.judge, name)) for name in names
        }
        judge_agentic._configure_judge("hard", 0)
        agentic_values = {
            name: copy.deepcopy(getattr(judge_agentic.judge, name)) for name in names
        }
        self.assertEqual(agentic_values, pag_values)
        self.assertEqual(
            judge_agentic.judge.PREFERENCE_INJECTION_MODE,
            judge_agentic.AGENTIC_MODE,
        )

    def test_target_model_override_is_read_from_agentic_config(self) -> None:
        with patch.object(judge_agentic, "TARGET_GEN_LLM", "agentic-test-model"):
            self.assertEqual(
                judge_agentic._target_gen_model_name(),
                "agentic-test-model",
            )

    def test_judge_runtime_matches_pag(self) -> None:
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
                self.assertEqual(
                    getattr(judge_agentic, name),
                    getattr(judge_agentic.judge_pag, name),
                )


if __name__ == "__main__":
    unittest.main()
