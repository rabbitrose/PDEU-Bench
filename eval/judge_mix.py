#!/usr/bin/env python3
"""Judge mixed explicit+hidden preference eval results with LLM-as-judge."""

from __future__ import annotations

import judge


# =========================
# 用户长期可调 judge 超参数
# =========================

# 为空表示使用 api.py 中的 GEN_MODEL。与 judge_hidden.py 保持一致。
TARGET_GEN_LLM = "deepseek-v4-pro"

# 选择要 judge 的 eval pass 目录；例如 1 读取 pass_1。
PASS_N = 1

# mix 只比 direct 多评估 hidden_info 隐式偏好；不读取 longctx 结果目录。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0

# 批量 judge 多个长上下文轮次；非空时会依次覆盖 LONG_CONTEXT_TURNS。
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 指定后只 judge 对应 eval run_id；为空时默认 judge 目标目录下最新一轮 eval。
TARGET_EVAL_RUN_ID = ""
JUDGE_LATEST_EVAL_RUN_ONLY = True

# 与 judge_hidden.py 默认并发保持一致。
JUDGE_CONCURRENCY = 50

# 为空表示评估全部用户/任务；例如 ["user_001"]、["task_001"]。
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "hard"


def main() -> None:
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        judge.TARGET_GEN_LLM = TARGET_GEN_LLM
        judge.PREFERENCE_INJECTION_MODE = "mix"
        judge.USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE = False
        judge.PASS_N = PASS_N
        judge.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
        judge.LONG_CONTEXT_TURNS = turns
        judge.TASK_DIFFICULTY = TASK_DIFFICULTY
        judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
        judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
        judge.AUTO_TUNE_CONCURRENCY = False
        judge.DEFAULT_JUDGE_CONCURRENCY = JUDGE_CONCURRENCY
        judge.DEEPSEEK_JUDGE_CONCURRENCY = JUDGE_CONCURRENCY
        judge.USER_ID_FILTER = USER_ID_FILTER
        judge.TASK_ID_FILTER = TASK_ID_FILTER
        judge.main()


if __name__ == "__main__":
    main()
