#!/usr/bin/env python3
"""Run the hidden-preference evaluation with a generation-time reminder."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_core


# =========================
# 用户长期可调评估超参数
# =========================

# 为空表示评估 data/ 下所有 user_xxx；例如 ["user_001", "user_019"]。
USER_ID_FILTER: list[str] = []

# 为空表示评估每个用户全部 task_xxx；例如 ["task_001"]。
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "easy"

# 每个 task_xxx 重复执行多少次；结果写入 runresult/<模型>/<难度>/reminder/pass_<PASS_N>/。
PASS_N = 1

# 每个任务最多执行多少次 action/tool_call；第 N 次 update_plan 后强制 finalization。
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK

# reminder 与 hidden 使用相同的完整评估上下文字符上限；需完整容纳全部工具 schema。
HIDDEN_CONTEXT_MAX_CHARS = 200_000

# 与 hidden 一致：仅精简 update_plan 提示词，保留完整 Sandbox 返回与轨迹。
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = True

# 补跑时设为 True：保留已有完整结果，只追加缺失任务；正式重跑设为 False 覆盖旧结果。
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False

# 是否把 data/LongContext/long_context_<N>.json 中的 LMSYS 多轮对话合并进用户交互历史。
ENABLE_LONG_CONTEXT = False

# 长上下文 QA pair 轮次；0 表示不追加日常长上下文。
LONG_CONTEXT_TURNS = 0

# 批量评测多个长上下文轮次；非空时会依次覆盖 LONG_CONTEXT_TURNS，并分别写入 longctx_<N>。
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 与 hidden 相同的离线评测固定并发。
EVAL_CONCURRENCY = 10


REMINDER_MODE = "reminder"
HIDDEN_MODE = "hidden"
REMINDER_LABEL = "隐式交互历史 + 生成前偏好提醒"
REMINDER_PROMPT = (
    "**Before you generate**,you should recall my preference indicated in the conversation history, "
    "and provide an answer that aligns with my preferences"
)


def _append_reminder(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    patched = [dict(message) for message in messages]
    for message in reversed(patched):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if REMINDER_PROMPT not in content:
            message["content"] = f"{REMINDER_PROMPT}\n\n{content}"
        break
    return patched


def _install_reminder_behavior() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(REMINDER_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[REMINDER_MODE] = REMINDER_LABEL
    if getattr(eval_core, "_REMINDER_BEHAVIOR_INSTALLED", False):
        return

    original_run_task = eval_core.run_task
    original_offline_judge_entrypoint = eval_core.offline_judge_entrypoint_for_mode
    original_initial_plan_messages = eval_core.initial_plan_messages
    original_action_messages = eval_core.action_messages
    original_update_plan_messages = eval_core.update_plan_messages

    def run_task_with_hidden_context(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != REMINDER_MODE:
            return original_run_task(**kwargs)

        record = original_run_task(**{**kwargs, "preference_mode": HIDDEN_MODE})
        record["preference_injection_mode"] = REMINDER_MODE
        record["preference_injection_label"] = REMINDER_LABEL
        provided_context = record.get("provided_eval_context")
        if isinstance(provided_context, dict):
            provided_context["mode"] = REMINDER_MODE
        judge_results = record.get("judge_results")
        if isinstance(judge_results, dict):
            judge_results["offline_judge_entrypoint"] = "python eval/reminder/judge_reminder.py"
        return record

    def offline_judge_entrypoint_with_reminder(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == REMINDER_MODE:
            return "python eval/reminder/judge_reminder.py"
        return original_offline_judge_entrypoint(preference_mode)

    def initial_plan_messages_with_reminder(context: dict[str, Any]) -> list[dict[str, str]]:
        return _append_reminder(original_initial_plan_messages(context))

    def action_messages_with_reminder(
        *,
        context: dict[str, Any],
        current_plan: dict[str, Any],
        trajectory_summary: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, str]]:
        return _append_reminder(
            original_action_messages(
                context=context,
                current_plan=current_plan,
                trajectory_summary=trajectory_summary,
                iteration=iteration,
            )
        )

    def update_plan_messages_with_reminder(
        *,
        context: dict[str, Any],
        previous_plan: dict[str, Any],
        action_json: dict[str, Any],
        sandbox_observation: dict[str, Any],
        iteration: int,
    ) -> list[dict[str, str]]:
        return _append_reminder(
            original_update_plan_messages(
                context=context,
                previous_plan=previous_plan,
                action_json=action_json,
                sandbox_observation=sandbox_observation,
                iteration=iteration,
            )
        )

    eval_core.run_task = run_task_with_hidden_context
    eval_core.offline_judge_entrypoint_for_mode = offline_judge_entrypoint_with_reminder
    eval_core.initial_plan_messages = initial_plan_messages_with_reminder
    eval_core.action_messages = action_messages_with_reminder
    eval_core.update_plan_messages = update_plan_messages_with_reminder
    setattr(eval_core, "_REMINDER_BEHAVIOR_INSTALLED", True)


def main() -> None:
    _install_reminder_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        eval_core.PREFERENCE_INJECTION_MODE = REMINDER_MODE
        eval_core.USER_ID_FILTER = USER_ID_FILTER
        eval_core.TASK_ID_FILTER = TASK_ID_FILTER
        eval_core.TASK_DIFFICULTY = os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
        eval_core.PASS_N = int(os.environ.get("PPEVAL_PASS_N", PASS_N))
        eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
        eval_core.MAX_JSON_CONTEXT_CHARS = HIDDEN_CONTEXT_MAX_CHARS
        eval_core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
            FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
        )
        skip_existing = os.environ.get("PPEVAL_SKIP_EXISTING", str(SKIP_TASKS_ALREADY_IN_RESULT_FILE)).lower() == "true"
        eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = skip_existing
        eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = not skip_existing
        eval_core.AUTO_TUNE_CONCURRENCY = False
        concurrency = int(os.environ.get("PPEVAL_CONCURRENCY", EVAL_CONCURRENCY))
        eval_core.DEFAULT_EVAL_CONCURRENCY = concurrency
        eval_core.DEEPSEEK_EVAL_CONCURRENCY = concurrency
        eval_core.QINGYUN_DEEPSEEK_EVAL_CONCURRENCY = concurrency
        difficulty = eval_core.normalize_task_difficulty(eval_core.TASK_DIFFICULTY)
        if difficulty != "all":
            eval_core.DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY[difficulty] = concurrency
        eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
        eval_core.LONG_CONTEXT_TURNS = turns
        eval_core.main()


if __name__ == "__main__":
    main()
