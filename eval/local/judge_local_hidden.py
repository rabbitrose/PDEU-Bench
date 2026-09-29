#!/usr/bin/env python3
"""Judge local-model hidden-preference results through the configured API judge."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge
from local_model import USEABLE_LOCAL_MODEL_PATHS


# =========================
# 用户长期可调 judge 超参数
# =========================

TARGET_GEN_LLM = "Qwen3-4B-Instruct-2507"
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_URL = "https://api.qingyuntop.top/v1"
TASK_DIFFICULTY = "mid"
PASS_N = 1
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []
TARGET_EVAL_RUN_ID = ""
JUDGE_LATEST_EVAL_RUN_ONLY = False
JUDGE_CONCURRENCY = 20
INSTANTIATION_JUDGE_CHUNK_SIZE = 4
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []


LEGACY_DIFFICULTY_ALIASES = {"mid": "hard"}
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def normalize_local_difficulty(value: str) -> tuple[str, str]:
    requested = str(value).strip().lower()
    normalized = LEGACY_DIFFICULTY_ALIASES.get(requested, requested)
    return requested, judge.normalize_task_difficulty(normalized)


def clear_proxy_environment() -> None:
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    target_model = os.environ.get("PPEVAL_GEN_MODEL", TARGET_GEN_LLM)
    if target_model not in USEABLE_LOCAL_MODEL_PATHS:
        raise ValueError(
            f"未知本地被测模型 {target_model!r}；可用模型为 {sorted(USEABLE_LOCAL_MODEL_PATHS)}"
        )
    requested, normalized = normalize_local_difficulty(
        os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
    )
    if requested != normalized:
        print(
            f"难度兼容映射：{requested!r} -> {normalized!r}；旧 mid 结果当前存储在 hard 目录",
            flush=True,
        )
    clear_proxy_environment()
    os.environ["PPEVAL_JUDGE_MODEL"] = JUDGE_MODEL
    os.environ["PPEVAL_JUDGE_URL"] = JUDGE_URL
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
        judge.TASK_DIFFICULTY = normalized
        judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
        judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
        judge.AUTO_TUNE_CONCURRENCY = False
        concurrency = int(os.environ.get("PPEVAL_CONCURRENCY", JUDGE_CONCURRENCY))
        judge.DEFAULT_JUDGE_CONCURRENCY = concurrency
        judge.DEEPSEEK_JUDGE_CONCURRENCY = concurrency
        judge.QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = concurrency
        judge.USER_ID_FILTER = env_list("PPEVAL_USER_ID_FILTER", USER_ID_FILTER)
        judge.TASK_ID_FILTER = env_list("PPEVAL_TASK_ID_FILTER", TASK_ID_FILTER)
        judge.main()


if __name__ == "__main__":
    main()
