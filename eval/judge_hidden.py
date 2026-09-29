#!/usr/bin/env python3
"""Judge hidden-preference eval results with LLM-as-judge."""

from __future__ import annotations

import os

import judge


# =========================
# 用户长期可调 judge 超参数
# =========================

# 为空表示使用 api.py 中的 GEN_MODEL。
TARGET_GEN_LLM = "deepseek-v4-flash"

# 选择要 judge 的 eval pass 目录；例如 1 读取 pass_1。
PASS_N = 1

# 长上下文结果目录控制。hidden 默认 judge runresult/<模型>/hidden/pass_<PASS_N>/。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0

# 批量 judge 多个长上下文轮次；非空时会依次覆盖 LONG_CONTEXT_TURNS。
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 指定后只 judge 对应 eval run_id；为空时默认 judge 目标目录下最新一轮 eval。
TARGET_EVAL_RUN_ID = ""
# 断点续跑可能跨多个 run_id；按 case 去重后聚合全部运行，避免只 judge 最后一批补跑子集。
JUDGE_LATEST_EVAL_RUN_ONLY = False

# hidden 离线 judge 固定并发。正式大批量前可根据 API 稳定性调高或调低。
JUDGE_CONCURRENCY = 100

# 每次 Instantiation Judge 同时判定的工具调用数量；增大可减少重复发送任务和用户偏好参考。
INSTANTIATION_JUDGE_CHUNK_SIZE = 4

# 为空表示评估全部用户/任务；例如 ["user_001"]、["task_001"]。
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "easy"


def main() -> None:
    target_model = os.environ.get("PPEVAL_GEN_MODEL", TARGET_GEN_LLM)
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        judge.API_RETRY_ATTEMPTS = 3
        judge.API_RETRY_BASE_SLEEP_SECONDS = 5.0
        judge.TARGET_GEN_LLM = target_model
        judge.PREFERENCE_INJECTION_MODE = "hidden"
        judge.USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE = True
        judge.INSTANTIATION_JUDGE_CHUNK_SIZE = INSTANTIATION_JUDGE_CHUNK_SIZE
        judge.PASS_N = int(os.environ.get("PPEVAL_PASS_N", PASS_N))
        judge.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
        judge.LONG_CONTEXT_TURNS = turns
        judge.TASK_DIFFICULTY = os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
        judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
        judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
        judge.AUTO_TUNE_CONCURRENCY = False
        concurrency = int(os.environ.get("PPEVAL_CONCURRENCY", JUDGE_CONCURRENCY))
        judge.DEFAULT_JUDGE_CONCURRENCY = concurrency
        judge.DEEPSEEK_JUDGE_CONCURRENCY = concurrency
        judge.QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = concurrency
        judge.USER_ID_FILTER = USER_ID_FILTER
        judge.TASK_ID_FILTER = TASK_ID_FILTER
        judge.main()


if __name__ == "__main__":
    main()
