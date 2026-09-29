#!/usr/bin/env python3
"""Judge reminder results with the same principles and logic as judge_hidden."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import judge


# =========================
# 用户长期可调 judge 超参数
# =========================

# 为空表示使用 api.py 中的 GEN_MODEL。
TARGET_GEN_LLM = "deepseek-v4-flash"

# 选择要 judge 的 eval pass 目录；例如 1 读取 pass_1。
PASS_N = 1

# 长上下文结果目录控制。reminder 读取 runresult/<模型>/<难度>/reminder/pass_<PASS_N>/。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0

# 批量 judge 多个长上下文轮次；非空时会依次覆盖 LONG_CONTEXT_TURNS。
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 指定后只 judge 对应 eval run_id；为空时默认 judge 目标目录下最新一轮 eval。
TARGET_EVAL_RUN_ID = ""
# 断点续跑可能跨多个 run_id；按 case 去重后聚合全部运行，避免只 judge 最后一批补跑子集。
JUDGE_LATEST_EVAL_RUN_ONLY = False

# 与 hidden 相同的离线 judge 固定并发。
JUDGE_CONCURRENCY = 100

# 与 hidden 相同的 Instantiation Judge 分块大小。
INSTANTIATION_JUDGE_CHUNK_SIZE = 4

# 为空表示评估全部用户/任务；例如 ["user_001"]、["task_001"]。
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "easy"


REMINDER_MODE = "reminder"
HIDDEN_MODE = "hidden"
REMINDER_LABEL = "用户交互历史 + 生成前偏好提醒"


def _as_hidden_judge_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(record)
    normalized["preference_injection_mode"] = HIDDEN_MODE
    normalized["preference_injection_label"] = judge.PREFERENCE_INJECTION_LABELS[HIDDEN_MODE]
    provided = normalized.get("provided_eval_context")
    if isinstance(provided, dict):
        provided["mode"] = HIDDEN_MODE
    return normalized


def _install_reminder_behavior() -> None:
    judge.VALID_PREFERENCE_INJECTION_MODES.add(REMINDER_MODE)
    judge.PREFERENCE_INJECTION_LABELS[REMINDER_MODE] = REMINDER_LABEL
    if getattr(judge, "_REMINDER_JUDGE_INSTALLED", False):
        return

    original_judge_case = judge.judge_case

    def judge_case_as_hidden(record: dict[str, Any], client: Any, logger: Any) -> dict[str, Any]:
        return original_judge_case(_as_hidden_judge_record(record), client, logger)

    judge.judge_case = judge_case_as_hidden
    setattr(judge, "_REMINDER_JUDGE_INSTALLED", True)


def main() -> None:
    _install_reminder_behavior()
    target_model = os.environ.get("PPEVAL_GEN_MODEL", TARGET_GEN_LLM)
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        judge.API_RETRY_ATTEMPTS = 3
        judge.API_RETRY_BASE_SLEEP_SECONDS = 5.0
        judge.TARGET_GEN_LLM = target_model
        judge.PREFERENCE_INJECTION_MODE = REMINDER_MODE
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
