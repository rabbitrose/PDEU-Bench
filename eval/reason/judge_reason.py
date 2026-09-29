#!/usr/bin/env python3
"""Judge reason-directory results with the shared hidden-preference judge."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import judge


# =========================
# 用户长期可调 judge 超参数
# =========================

# 为空表示使用 api.py 中的 GEN_MODEL。
TARGET_GEN_LLM = "deepseek-v4-flash"

# 选择要 judge 的 eval pass 目录。
PASS_N = 1

# 长上下文结果目录控制。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []

# 指定 eval run_id；为空时按 case 合并并保留最新记录。
TARGET_EVAL_RUN_ID = ""
JUDGE_LATEST_EVAL_RUN_ONLY = False

# DeepSeek 官方 judge 可按任务数高并发；可依据 API 稳定性调整。
JUDGE_CONCURRENCY = 100

# 每次 Instantiation Judge 同时判定的工具调用数量。
INSTANTIATION_JUDGE_CHUNK_SIZE = 4

# 为空表示评估全部用户/任务。
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []

# 任务难度：easy -> task_easy, hard -> task_hard, all -> 不追加难度过滤。
TASK_DIFFICULTY = "hard"


def _install_reason_output_routing() -> None:
    def result_dir(
        model_name: str,
        pass_n: int | None,
        *,
        base_dir: Path,
    ) -> Path:
        resolved_pass_n = judge.normalize_pass_n(judge.PASS_N if pass_n is None else pass_n)
        base = (
            base_dir
            / judge.safe_model_dir_name(model_name)
            / judge.difficulty_result_component()
            / "reason"
        )
        long_context_component = judge.long_context_result_component("hidden")
        if long_context_component:
            base = base / long_context_component
        return base / f"pass_{resolved_pass_n}"

    def configure_model_dirs(
        model_name: str,
        preference_mode: str = "hidden",
        pass_n: int | None = None,
    ) -> tuple[Path, Path, Path]:
        _ = preference_mode
        judge.EVAL_RESULT_DIR = result_dir(
            model_name,
            pass_n,
            base_dir=judge.EVAL_RESULT_BASE_DIR,
        )
        judge.JUDGE_RESULT_DIR = result_dir(
            model_name,
            pass_n,
            base_dir=judge.JUDGE_RESULT_BASE_DIR,
        )
        judge.HTML_DIR = result_dir(
            model_name,
            pass_n,
            base_dir=judge.HTML_BASE_DIR,
        )
        return judge.EVAL_RESULT_DIR, judge.JUDGE_RESULT_DIR, judge.HTML_DIR

    judge.configure_model_dirs = configure_model_dirs


def main() -> None:
    _install_reason_output_routing()
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
        judge.USER_ID_FILTER = USER_ID_FILTER
        judge.TASK_ID_FILTER = TASK_ID_FILTER
        judge.main()


if __name__ == "__main__":
    main()
