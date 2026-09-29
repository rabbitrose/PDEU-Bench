#!/usr/bin/env python3
"""Run the shared planning evaluation with hidden-preference injection."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_core


# =========================
# 用户长期可调评估超参数
# =========================

# 为空表示评估 data/ 下所有 user_xxx；例如 ["user_001", "user_019"]。
USER_ID_FILTER: list[str] = []

# 为空表示评估每个用户全部 task_xxx；例如 ["task_easy"]。
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "hard"

# 每个 task_xxx 重复执行多少次；结果写入 runresult/<模型>/<难度>/reason/pass_<PASS_N>/。
PASS_N = 1

# 每个任务最多执行多少次 action/tool_call；每次调用后都执行 update_plan。
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK

# 完整评估上下文字符上限；需容纳全部工具 schema。
CONTEXT_MAX_CHARS = 200_000

# 补跑时设为 True；正式重跑设为 False 覆盖旧结果。
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False

# 是否追加 LMSYS 长上下文，以及追加的 QA pair 轮次。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []

# reason 兼容入口固定并发；可依据 API 稳定性调整。
EVAL_CONCURRENCY = 10


def _install_reason_output_routing() -> None:
    def result_dir(model_name: str, pass_n: int | None, *, judge: bool) -> Path:
        resolved_pass_n = eval_core.normalize_pass_n(eval_core.PASS_N if pass_n is None else pass_n)
        base_dir = eval_core.JUDGE_RESULT_BASE_DIR if judge else eval_core.RESULT_BASE_DIR
        base = (
            base_dir
            / eval_core.safe_model_dir_name(model_name)
            / eval_core.difficulty_result_component()
            / "reason"
        )
        long_context_component = eval_core.long_context_result_component("hidden")
        if long_context_component:
            base = base / long_context_component
        return base / f"pass_{resolved_pass_n}"

    def configure_result_dir(
        model_name: str,
        preference_mode: str = "hidden",
        pass_n: int | None = None,
    ) -> tuple[Path, Path]:
        _ = preference_mode
        eval_core.RESULT_DIR = result_dir(model_name, pass_n, judge=False)
        eval_core.JUDGE_RESULT_DIR = result_dir(model_name, pass_n, judge=True)
        return eval_core.RESULT_DIR, eval_core.JUDGE_RESULT_DIR

    def offline_judge_entrypoint_for_mode(preference_mode: str) -> str:
        _ = preference_mode
        return "python eval/reason/judge_reason.py"

    eval_core.configure_result_dir = configure_result_dir
    eval_core.offline_judge_entrypoint_for_mode = offline_judge_entrypoint_for_mode


def main() -> None:
    _install_reason_output_routing()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        eval_core.PREFERENCE_INJECTION_MODE = "hidden"
        eval_core.USER_ID_FILTER = USER_ID_FILTER
        eval_core.TASK_ID_FILTER = TASK_ID_FILTER
        eval_core.TASK_DIFFICULTY = os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
        eval_core.PASS_N = int(os.environ.get("PPEVAL_PASS_N", PASS_N))
        eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
        eval_core.MAX_JSON_CONTEXT_CHARS = CONTEXT_MAX_CHARS
        skip_existing = os.environ.get(
            "PPEVAL_SKIP_EXISTING",
            str(SKIP_TASKS_ALREADY_IN_RESULT_FILE),
        ).lower() == "true"
        eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = skip_existing
        eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = not skip_existing
        eval_core.AUTO_TUNE_CONCURRENCY = False
        concurrency = int(os.environ.get("PPEVAL_CONCURRENCY", EVAL_CONCURRENCY))
        eval_core.DEFAULT_EVAL_CONCURRENCY = concurrency
        eval_core.DEEPSEEK_EVAL_CONCURRENCY = concurrency
        eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
        eval_core.LONG_CONTEXT_TURNS = turns
        eval_core.main()


if __name__ == "__main__":
    main()
