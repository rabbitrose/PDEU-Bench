#!/usr/bin/env python3
"""Run RAG-aligned evaluation with one additional user abstract profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_core
import eval_rag


# =========================
# 用户长期可调评估超参数
# =========================

# 除 User abstract profile 外，以下配置与 eval/rag/eval_rag.py 保持一致。
USER_ID_FILTER: list[str] = list(eval_rag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(eval_rag.TASK_ID_FILTER)
TASK_DIFFICULTY = eval_rag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(eval_rag.TASK_DIFFICULTY_LIST)
PASS_N = eval_rag.PASS_N
MAX_TOOL_CALLS_PER_TASK = eval_rag.MAX_TOOL_CALLS_PER_TASK
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
    eval_rag.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
)
SKIP_TASKS_ALREADY_IN_RESULT_FILE = eval_rag.SKIP_TASKS_ALREADY_IN_RESULT_FILE
ENABLE_LONG_CONTEXT = eval_rag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = eval_rag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(eval_rag.LONG_CONTEXT_TURNS_LIST)
PAG_RETRIEVAL_TOP_K = eval_rag.RAG_TOP_K
RAG_PROMPT_MAX_CHARS = eval_rag.RAG_PROMPT_MAX_CHARS
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(eval_rag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
DEFAULT_EVAL_CONCURRENCY = eval_rag.DEFAULT_EVAL_CONCURRENCY
QINGYUN_EVAL_CONCURRENCY = eval_rag.QINGYUN_EVAL_CONCURRENCY

# PAG 唯一额外注入内容：每个任务对应的一条 User abstract profile。
USER_ABSTRACT_PROFILE_MAX_CHARS = 6000


PAG_MODE = "pag"
PAG_LABEL = "RAG Top-3 原始 QA + User abstract profile"
PAG_ABSTRACT_CONTEXT_KEY = "pag_user_abstract_profile"
PAG_ABSTRACT_PROMPT_PREFIX = "User abstract profile"


def _task_identity(kwargs: dict[str, Any]) -> tuple[str, str]:
    user_profile = kwargs.get("user_profile") if isinstance(kwargs.get("user_profile"), dict) else {}
    task_doc = kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {}
    workflow = kwargs.get("workflow") if isinstance(kwargs.get("workflow"), dict) else {}
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    user_id = str(user_profile.get("user_id") or "unknown_user")
    task_id = str(task.get("task_id") or workflow.get("task_id") or "unknown_task")
    return user_id, task_id


def _user_abstract_profile_for_task(*, user_id: str, task_id: str) -> str:
    abstract = eval_core.pag_prompt_abstract_for_task(user_id=user_id, task_id=task_id)
    if isinstance(abstract, dict):
        value = abstract.get("task_related_user_abstract") or abstract.get("summary")
        if value in (None, ""):
            value = eval_core.compact_json(abstract, USER_ABSTRACT_PROFILE_MAX_CHARS)
    else:
        value = abstract
    profile = eval_core.compact_text(
        str(value or "").strip(),
        USER_ABSTRACT_PROFILE_MAX_CHARS,
    )
    if not profile:
        raise RuntimeError(f"PAG User abstract profile 为空: user={user_id} task={task_id}")
    return profile


def _prepend_user_abstract_profile(
    messages: list[dict[str, str]],
    profile: str,
) -> list[dict[str, str]]:
    sentence = f"{PAG_ABSTRACT_PROMPT_PREFIX}: {json.dumps(profile, ensure_ascii=False)}."
    patched = [dict(message) for message in messages]
    for message in patched:
        if message.get("role") == "user":
            content = str(message.get("content") or "")
            message["content"] = f"{sentence}\n\n{content}"
            break
    return patched


def _context_without_pag_abstract(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if key != PAG_ABSTRACT_CONTEXT_KEY}


def _register_pag_mode() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(PAG_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[PAG_MODE] = PAG_LABEL


def _patch_pag_context_builders() -> None:
    original_tool_discovery_messages = eval_core.tool_discovery_messages
    original_build_common_context = eval_core.build_common_context

    def tool_discovery_messages_with_pag(**kwargs: Any) -> list[dict[str, str]]:
        if kwargs.get("preference_mode") != PAG_MODE:
            return original_tool_discovery_messages(**kwargs)
        user_id, task_id = _task_identity(kwargs)
        messages = original_tool_discovery_messages(
            **{**kwargs, "preference_mode": eval_rag.RAG_MODE}
        )
        return _prepend_user_abstract_profile(
            messages,
            _user_abstract_profile_for_task(user_id=user_id, task_id=task_id),
        )

    def build_common_context_with_pag(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != PAG_MODE:
            return original_build_common_context(**kwargs)
        user_id, task_id = _task_identity(kwargs)
        context = original_build_common_context(
            **{**kwargs, "preference_mode": eval_rag.RAG_MODE}
        )
        context[PAG_ABSTRACT_CONTEXT_KEY] = _user_abstract_profile_for_task(
            user_id=user_id,
            task_id=task_id,
        )
        return context

    eval_core.tool_discovery_messages = tool_discovery_messages_with_pag
    eval_core.build_common_context = build_common_context_with_pag


def _patch_pag_plan_prompts() -> None:
    original_initial_plan_messages = eval_core.initial_plan_messages
    original_action_messages = eval_core.action_messages
    original_update_plan_messages = eval_core.update_plan_messages

    def initial_plan_messages_with_pag(context: dict[str, Any]) -> list[dict[str, str]]:
        profile = str(context.get(PAG_ABSTRACT_CONTEXT_KEY) or "")
        messages = original_initial_plan_messages(_context_without_pag_abstract(context))
        return _prepend_user_abstract_profile(messages, profile) if profile else messages

    def action_messages_with_pag(
        *,
        context: dict[str, Any],
        current_plan: dict[str, Any],
        trajectory_summary: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, str]]:
        profile = str(context.get(PAG_ABSTRACT_CONTEXT_KEY) or "")
        messages = original_action_messages(
            context=_context_without_pag_abstract(context),
            current_plan=current_plan,
            trajectory_summary=trajectory_summary,
            iteration=iteration,
        )
        return _prepend_user_abstract_profile(messages, profile) if profile else messages

    def update_plan_messages_with_pag(
        *,
        context: dict[str, Any],
        previous_plan: dict[str, Any],
        action_json: dict[str, Any],
        sandbox_observation: dict[str, Any],
        iteration: int,
    ) -> list[dict[str, str]]:
        profile = str(context.get(PAG_ABSTRACT_CONTEXT_KEY) or "")
        messages = original_update_plan_messages(
            context=_context_without_pag_abstract(context),
            previous_plan=previous_plan,
            action_json=action_json,
            sandbox_observation=sandbox_observation,
            iteration=iteration,
        )
        return _prepend_user_abstract_profile(messages, profile) if profile else messages

    eval_core.initial_plan_messages = initial_plan_messages_with_pag
    eval_core.action_messages = action_messages_with_pag
    eval_core.update_plan_messages = update_plan_messages_with_pag


def _align_pag_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = record.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        hyperparameters.pop("pag_retrieval_top_k", None)
        hyperparameters.pop("pag_retrieval_injection_phases", None)
        hyperparameters["rag_top_k"] = PAG_RETRIEVAL_TOP_K
        hyperparameters["rag_retrieval_method"] = (
            "lightweight_bm25_over_task_text_and_hidden_qa"
        )
        hyperparameters["rag_retrieval_injection_phases"] = [
            "initial_plan",
            "reason_action",
            "update_plan",
        ]
        hyperparameters["pag_user_abstract_profile_injection_phases"] = [
            "initial_plan",
            "reason_action",
            "update_plan",
        ]
        hyperparameters["pag_difference_from_rag"] = "user_abstract_profile_only"
    return record


def _patch_pag_run_task() -> None:
    original_run_task = eval_core.run_task

    def run_task_with_pag(**kwargs: Any) -> dict[str, Any]:
        record = original_run_task(**kwargs)
        if kwargs.get("preference_mode") == PAG_MODE:
            return _align_pag_record_metadata(record)
        return record

    eval_core.run_task = run_task_with_pag


def _install_pag_behavior() -> None:
    eval_rag.RAG_TOP_K = PAG_RETRIEVAL_TOP_K
    eval_rag.RAG_PROMPT_MAX_CHARS = RAG_PROMPT_MAX_CHARS
    eval_rag.QINGYUN_EVAL_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY
    eval_rag._install_rag_behavior()
    _register_pag_mode()
    if getattr(eval_core, "_PAG_RAG_ALIGNED_BEHAVIOR_INSTALLED", False):
        return
    _patch_pag_context_builders()
    _patch_pag_plan_prompts()
    _patch_pag_run_task()
    setattr(eval_core, "_PAG_RAG_ALIGNED_BEHAVIOR_INSTALLED", True)


def _configure_evaluation(task_difficulty: str, turns: int) -> None:
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    eval_core.PREFERENCE_INJECTION_MODE = PAG_MODE
    eval_core.USER_ID_FILTER = USER_ID_FILTER
    eval_core.TASK_ID_FILTER = TASK_ID_FILTER
    eval_core.TASK_DIFFICULTY = difficulty
    eval_core.PASS_N = PASS_N
    eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
    eval_core.PAG_RETRIEVAL_TOP_K = PAG_RETRIEVAL_TOP_K
    eval_core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
        FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
    )
    eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = not SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.AUTO_TUNE_CONCURRENCY = False
    eval_core.DEFAULT_EVAL_CONCURRENCY = DEFAULT_EVAL_CONCURRENCY
    eval_core.DEEPSEEK_EVAL_CONCURRENCY = DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[difficulty]
    eval_core.DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY = dict(
        DEEPSEEK_CONCURRENCY_BY_DIFFICULTY
    )
    eval_core.QINGYUN_DEEPSEEK_EVAL_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY
    eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    eval_core.LONG_CONTEXT_TURNS = turns


def main() -> None:
    _install_pag_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            _configure_evaluation(task_difficulty, turns)
            eval_core.main()


if __name__ == "__main__":
    main()
