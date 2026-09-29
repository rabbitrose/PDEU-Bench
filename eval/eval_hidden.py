#!/usr/bin/env python3
"""Run planning-agent evaluation with user interaction history only."""

from __future__ import annotations

import os

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

# 每个 task_xxx 重复执行多少次；结果写入 runresult/<模型>/hidden/pass_<PASS_N>/。
PASS_N = 1

# 每个任务最多执行多少次 action/tool_call；第 N 次 update_plan 后强制 finalization。
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK

# hidden 标准评测的完整评估上下文字符上限；需完整容纳全部工具 schema。
HIDDEN_CONTEXT_MAX_CHARS = 200_000

# 仅精简传入 update_plan 提示词的 Sandbox observation；原始返回与轨迹仍完整保存。
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = True

# 补跑时设为 True：保留已有完整结果，只追加缺失任务；正式重跑设为 False 覆盖旧结果。
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False

# 是否把 data/LongContext/long_context_<N>.json 中的 LMSYS 多轮对话合并进用户交互历史。
ENABLE_LONG_CONTEXT = False

# 长上下文 QA pair 轮次；0 表示不追加日常长上下文。
LONG_CONTEXT_TURNS = 0

# 批量评测多个长上下文轮次；非空时会依次覆盖 LONG_CONTEXT_TURNS，并分别写入 longctx_<N>。
LONG_CONTEXT_TURNS_LIST: list[int] = []

# hidden 离线评测固定并发。正式大批量前可根据 API 稳定性调高或调低。
EVAL_CONCURRENCY = 10

# 单次生成的最大输出 token 数；不影响 hidden prompt 的输入上下文长度。
MODEL_MAX_TOKENS = 8192


def main() -> None:
    # 进程级覆盖用于并行 tmux 评测，避免多个模型竞争修改共享 api.py。
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for turns in turns_list:
        eval_core.PREFERENCE_INJECTION_MODE = "hidden"
        eval_core.USER_ID_FILTER = USER_ID_FILTER
        eval_core.TASK_ID_FILTER = TASK_ID_FILTER
        eval_core.TASK_DIFFICULTY = os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
        eval_core.PASS_N = int(os.environ.get("PPEVAL_PASS_N", PASS_N))
        eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
        eval_core.MODEL_MAX_TOKENS = int(
            os.environ.get("PPEVAL_MODEL_MAX_TOKENS", MODEL_MAX_TOKENS)
        )
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
