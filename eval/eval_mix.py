#!/usr/bin/env python3
"""Run planning-agent evaluation with explicit profile plus hidden preferences."""

from __future__ import annotations

import eval_core


# =========================
# 用户长期可调评估超参数
# =========================

# 为空表示评估 data/ 下所有 user_xxx；例如 ["user_001", "user_019"]。
USER_ID_FILTER: list[str] = []

# 为空表示评估每个用户全部 task_xxx；例如 ["task_001"]。
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "hard"

# 每个 task_xxx 重复执行多少次；结果写入 runresult/<模型>/mix/pass_<PASS_N>/。
PASS_N = 1

# 与 eval_core.py 保持一致：默认覆盖选中用户结果，不做断点追加。
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False
OVERWRITE_SELECTED_USER_RESULTS_ON_START = True

# mix 只比 direct 多注入 hidden_info 隐式偏好；不追加 LMSYS 长上下文干扰。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 与 eval_core.py 默认值保持一致的核心评估超参数。
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK
MODEL_TEMPERATURE = eval_core.MODEL_TEMPERATURE
MODEL_MAX_TOKENS = eval_core.MODEL_MAX_TOKENS
REQUEST_TIMEOUT_SECONDS = eval_core.REQUEST_TIMEOUT_SECONDS
API_RETRY_ATTEMPTS = eval_core.API_RETRY_ATTEMPTS
API_RETRY_BASE_SLEEP_SECONDS = eval_core.API_RETRY_BASE_SLEEP_SECONDS
JSON_REPAIR_ATTEMPTS = eval_core.JSON_REPAIR_ATTEMPTS
USE_JSON_RESPONSE_FORMAT = eval_core.USE_JSON_RESPONSE_FORMAT
TOOL_AVAILABILITY_MODE = eval_core.TOOL_AVAILABILITY_MODE

# 与 eval_core.py 默认并发保持一致。
AUTO_TUNE_CONCURRENCY = eval_core.AUTO_TUNE_CONCURRENCY
EVAL_CONCURRENCY = eval_core.DEEPSEEK_EVAL_CONCURRENCY


def main() -> None:
    eval_core.PREFERENCE_INJECTION_MODE = "mix"
    eval_core.USER_ID_FILTER = USER_ID_FILTER
    eval_core.TASK_ID_FILTER = TASK_ID_FILTER
    eval_core.TASK_DIFFICULTY = TASK_DIFFICULTY
    eval_core.PASS_N = PASS_N
    eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
    eval_core.MODEL_TEMPERATURE = MODEL_TEMPERATURE
    eval_core.MODEL_MAX_TOKENS = MODEL_MAX_TOKENS
    eval_core.REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS
    eval_core.API_RETRY_ATTEMPTS = API_RETRY_ATTEMPTS
    eval_core.API_RETRY_BASE_SLEEP_SECONDS = API_RETRY_BASE_SLEEP_SECONDS
    eval_core.JSON_REPAIR_ATTEMPTS = JSON_REPAIR_ATTEMPTS
    eval_core.USE_JSON_RESPONSE_FORMAT = USE_JSON_RESPONSE_FORMAT
    eval_core.TOOL_AVAILABILITY_MODE = TOOL_AVAILABILITY_MODE
    eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = OVERWRITE_SELECTED_USER_RESULTS_ON_START
    eval_core.AUTO_TUNE_CONCURRENCY = AUTO_TUNE_CONCURRENCY
    eval_core.DEFAULT_EVAL_CONCURRENCY = EVAL_CONCURRENCY
    eval_core.DEEPSEEK_EVAL_CONCURRENCY = EVAL_CONCURRENCY
    eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    eval_core.LONG_CONTEXT_TURNS = LONG_CONTEXT_TURNS
    eval_core.main()


if __name__ == "__main__":
    main()
