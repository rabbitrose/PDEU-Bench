#!/usr/bin/env python3
"""Run hidden-preference evaluation with Top-K RAG over hidden_info."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
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

# 单难度调试配置；非空列表用于依次评估多个正式难度。
TASK_DIFFICULTY = "hard"
TASK_DIFFICULTY_LIST: list[str] = ["hard"]

# 每个 task_xxx 重复执行多少次；结果写入
# runresult/<模型>/<难度>/rag/pass_<PASS_N>/。
PASS_N = 1

# 每个任务最多执行多少次 action/tool_call；第 N 次 update_plan 后强制 finalization。
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK

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

# RAG 从 hidden_info.interactions 中检索与当前 task 最相关的历史 QA 数量。
RAG_TOP_K = 3

# RAG prompt 中最多保留的检索记录字符数。
RAG_PROMPT_MAX_CHARS = 6000

# DeepSeek 官方接口按项目约定随任务难度设置；Qingyun 固定为 20。
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = {"easy": 68, "hard": 75, "all": 75}
DEFAULT_EVAL_CONCURRENCY = 8

# Qingyun 代理地址按项目约定固定使用较低并发。
QINGYUN_EVAL_CONCURRENCY = 20


RAG_MODE = "rag"
HIDDEN_CONTEXT_MODE = "hidden"
RAG_PROMPT_PREFIX = "从 hidden info 中检索到的与当前任务最相关的历史 QA 原始对话"
TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [match.group(0) for match in TOKEN_RE.finditer(lowered)]
    for segment in CJK_SEQUENCE_RE.findall(text):
        tokens.extend(segment[idx : idx + 2] for idx in range(max(0, len(segment) - 1)))
    return tokens


def _task_query_text(task_doc: dict[str, Any]) -> str:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    keys = [
        "title",
        "daily_life_domain",
        "scenario",
        "user_goal",
        "constraints",
        "required_reasoning_steps",
        "decision_points",
        "information_to_collect",
        "expected_deliverable",
        "success_criteria",
        "potential_failure_modes",
    ]
    return _textify({key: task.get(key) for key in keys if key in task})


def _interaction_search_text(interaction: dict[str, Any]) -> str:
    parts: list[str] = [
        _textify(interaction.get("kind")),
        _textify(interaction.get("implicit_signal")),
    ]
    turns = interaction.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                parts.append(_textify(turn.get("content")))
    return "\n".join(part for part in parts if part)


def _clean_turns(interaction: dict[str, Any]) -> list[dict[str, str]]:
    turns = interaction.get("turns")
    if not isinstance(turns, list):
        return []
    cleaned: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(role, str) and isinstance(content, str):
            cleaned.append({"role": role, "content": content})
    return cleaned


def retrieve_hidden_info_for_task(
    *,
    hidden_info: dict[str, Any] | None,
    task_doc: dict[str, Any],
    top_k: int = RAG_TOP_K,
) -> dict[str, Any]:
    interactions = (
        [item for item in hidden_info.get("interactions", []) if isinstance(item, dict)]
        if isinstance(hidden_info, dict)
        else []
    )
    resolved_top_k = max(1, int(top_k))
    query_text = _task_query_text(task_doc)
    query_tokens = _tokenize(query_text)
    doc_tokens = [_tokenize(_interaction_search_text(item)) for item in interactions]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    doc_count = len(doc_tokens)
    avg_doc_len = (sum(len(tokens) for tokens in doc_tokens) / doc_count) if doc_count else 1.0
    query_counts: Counter[str] = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (interaction, tokens) in enumerate(zip(interactions, doc_tokens), start=1):
        tf: Counter[str] = Counter(tokens)
        doc_len = max(1, len(tokens))
        score = 0.0
        for token, query_tf in query_counts.items():
            term_freq = tf.get(token, 0)
            if term_freq <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5)) if doc_count else 0.0
            denom = term_freq + k1 * (1.0 - b + b * doc_len / avg_doc_len)
            score += idf * (term_freq * (k1 + 1.0) / denom) * min(query_tf, 3)
        scored.append((score, idx, interaction))

    top_items = sorted(scored, key=lambda item: (-item[0], item[1]))[:resolved_top_k]
    records = [
        {
            "rank": rank,
            "score": round(score, 6),
            "interaction_id": interaction.get("interaction_id"),
            "kind": interaction.get("kind"),
            "retrieved_topic": interaction.get("implicit_signal"),
            "turns": _clean_turns(interaction),
        }
        for rank, (score, _idx, interaction) in enumerate(top_items, start=1)
    ]
    return {
        "retrieval_source": "hidden_info.interactions",
        "retrieval_method": "lightweight_bm25_over_task_text_and_hidden_qa",
        "top_k": resolved_top_k,
        "retrieved_count": len(records),
        "records": records,
    }


def _top_k_hidden_info(hidden_info: dict[str, Any] | None, task_doc: dict[str, Any]) -> dict[str, Any]:
    retrieval = retrieve_hidden_info_for_task(
        hidden_info=hidden_info,
        task_doc=task_doc,
        top_k=RAG_TOP_K,
    )
    selected_ids = {
        str(record.get("interaction_id"))
        for record in retrieval["records"]
        if record.get("interaction_id") is not None
    }
    interactions = hidden_info.get("interactions", []) if isinstance(hidden_info, dict) else []
    selected = [
        interaction
        for interaction in interactions
        if isinstance(interaction, dict) and str(interaction.get("interaction_id")) in selected_ids
    ]
    return {"interactions": selected}


def _register_rag_mode() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(RAG_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[RAG_MODE] = "隐式交互历史 RAG Top-3"


def _patch_hidden_context_builders() -> None:
    original_tool_discovery_messages = eval_core.tool_discovery_messages
    original_build_common_context = eval_core.build_common_context
    original_preference_source_description = eval_core.preference_source_description
    original_offline_judge_entrypoint = eval_core.offline_judge_entrypoint_for_mode
    original_run_task = eval_core.run_task

    def tool_discovery_messages_with_rag_context(**kwargs: Any) -> list[dict[str, str]]:
        if kwargs.get("preference_mode") == RAG_MODE:
            task_doc = kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {}
            kwargs = {
                **kwargs,
                "preference_mode": HIDDEN_CONTEXT_MODE,
                "hidden_info": _top_k_hidden_info(kwargs.get("hidden_info"), task_doc),
                "long_context": None,
            }
        return original_tool_discovery_messages(**kwargs)

    def build_common_context_with_rag(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != RAG_MODE:
            return original_build_common_context(**kwargs)
        hidden_kwargs = {
            **kwargs,
            "preference_mode": HIDDEN_CONTEXT_MODE,
            "hidden_info": {"interactions": []},
            "long_context": None,
        }
        context = original_build_common_context(**hidden_kwargs)
        context["rag_retrieved_user_interaction_history"] = retrieve_hidden_info_for_task(
            hidden_info=kwargs.get("hidden_info"),
            task_doc=kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {},
            top_k=RAG_TOP_K,
        )
        return context

    def preference_source_description_with_rag(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == RAG_MODE:
            return "从 hidden info 中检索得到的与当前任务最相关 Top-3 历史 QA 记录及其隐式偏好"
        return original_preference_source_description(preference_mode)

    def offline_judge_entrypoint_with_rag(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == RAG_MODE:
            return "python eval/rag/judge_rag.py"
        return original_offline_judge_entrypoint(preference_mode)

    def run_task_with_rag(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != RAG_MODE:
            return original_run_task(**kwargs)

        record = original_run_task(**kwargs)
        record["preference_injection_mode"] = RAG_MODE
        record["preference_injection_label"] = eval_core.PREFERENCE_INJECTION_LABELS[RAG_MODE]
        record["user_profile"] = None
        record["hidden_info"] = kwargs.get("hidden_info")
        hyperparameters = record.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            hyperparameters["rag_top_k"] = RAG_TOP_K
            hyperparameters["rag_retrieval_method"] = "lightweight_bm25_over_task_text_and_hidden_qa"
            hyperparameters["rag_retrieval_injection_phases"] = [
                "initial_plan",
                "reason_action",
                "update_plan",
            ]
        provided_context = record.get("provided_eval_context")
        if isinstance(provided_context, dict):
            provided_context["mode"] = RAG_MODE
        sandbox_context = record.get("sandbox_context")
        if isinstance(sandbox_context, dict):
            sandbox_context["user_context"] = {}
        judge_results = record.get("judge_results")
        if isinstance(judge_results, dict):
            judge_results["offline_judge_entrypoint"] = "python eval/rag/judge_rag.py"
        return record

    eval_core.tool_discovery_messages = tool_discovery_messages_with_rag_context
    eval_core.build_common_context = build_common_context_with_rag
    eval_core.preference_source_description = preference_source_description_with_rag
    eval_core.offline_judge_entrypoint_for_mode = offline_judge_entrypoint_with_rag
    eval_core.run_task = run_task_with_rag


def _rag_qa_turns_for_prompt(context: dict[str, Any]) -> list[list[dict[str, str]]]:
    rag_context = context.get("rag_retrieved_user_interaction_history")
    records = rag_context.get("records", []) if isinstance(rag_context, dict) else []
    qa_histories: list[list[dict[str, str]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        turns = record.get("turns")
        if not isinstance(turns, list):
            continue
        clean_turns = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in turns
            if isinstance(turn, dict)
            and isinstance(turn.get("role"), str)
            and isinstance(turn.get("content"), str)
        ]
        if clean_turns:
            qa_histories.append(clean_turns)
    return qa_histories


def _prepend_rag_prompt(messages: list[dict[str, str]], context: dict[str, Any]) -> list[dict[str, str]]:
    qa_histories = _rag_qa_turns_for_prompt(context)
    prompt = f"{RAG_PROMPT_PREFIX}：{eval_core.compact_json(qa_histories, RAG_PROMPT_MAX_CHARS)}。"
    patched = [dict(message) for message in messages]
    for message in patched:
        if message.get("role") == "user":
            content = str(message.get("content") or "")
            if RAG_PROMPT_PREFIX not in content:
                message["content"] = f"{prompt}\n\n{content}"
            break
    return patched


def _context_without_rag(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if key != "rag_retrieved_user_interaction_history"
    }


def _patch_plan_prompts() -> None:
    original_initial_plan_messages = eval_core.initial_plan_messages
    original_action_messages = eval_core.action_messages
    original_update_plan_messages = eval_core.update_plan_messages

    def initial_plan_messages_with_rag(context: dict[str, Any]) -> list[dict[str, str]]:
        return _prepend_rag_prompt(
            original_initial_plan_messages(_context_without_rag(context)),
            context,
        )

    def action_messages_with_rag(
        *,
        context: dict[str, Any],
        current_plan: dict[str, Any],
        trajectory_summary: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, str]]:
        return _prepend_rag_prompt(
            original_action_messages(
                context=_context_without_rag(context),
                current_plan=current_plan,
                trajectory_summary=trajectory_summary,
                iteration=iteration,
            ),
            context,
        )

    def update_plan_messages_with_rag(
        *,
        context: dict[str, Any],
        previous_plan: dict[str, Any],
        action_json: dict[str, Any],
        sandbox_observation: dict[str, Any],
        iteration: int,
    ) -> list[dict[str, str]]:
        return _prepend_rag_prompt(
            original_update_plan_messages(
                context=_context_without_rag(context),
                previous_plan=previous_plan,
                action_json=action_json,
                sandbox_observation=sandbox_observation,
                iteration=iteration,
            ),
            context,
        )

    eval_core.initial_plan_messages = initial_plan_messages_with_rag
    eval_core.action_messages = action_messages_with_rag
    eval_core.update_plan_messages = update_plan_messages_with_rag


def _patch_qingyun_concurrency() -> None:
    if getattr(eval_core, "_RAG_QINGYUN_CONCURRENCY_PATCH_INSTALLED", False):
        return

    original_calibrate_concurrency = eval_core.calibrate_concurrency

    def calibrate_concurrency_with_qingyun_limit(
        client: eval_core.LLMClient,
        logger: Any,
    ) -> int:
        if client.base_url == "https://api.qingyuntop.top/v1":
            logger.info("根据项目 API 约定，Qingyun eval 并发采用 %s。", QINGYUN_EVAL_CONCURRENCY)
            return QINGYUN_EVAL_CONCURRENCY
        return original_calibrate_concurrency(client, logger)

    eval_core.calibrate_concurrency = calibrate_concurrency_with_qingyun_limit
    setattr(eval_core, "_RAG_QINGYUN_CONCURRENCY_PATCH_INSTALLED", True)


def _install_rag_behavior() -> None:
    already_installed = getattr(eval_core, "_RAG_BEHAVIOR_INSTALLED", False)
    _register_rag_mode()
    _patch_qingyun_concurrency()
    if already_installed:
        return
    _patch_hidden_context_builders()
    _patch_plan_prompts()
    setattr(eval_core, "_RAG_BEHAVIOR_INSTALLED", True)


def _configure_evaluation(task_difficulty: str, turns: int) -> None:
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    eval_core.PREFERENCE_INJECTION_MODE = RAG_MODE
    eval_core.USER_ID_FILTER = USER_ID_FILTER
    eval_core.TASK_ID_FILTER = TASK_ID_FILTER
    eval_core.TASK_DIFFICULTY = difficulty
    eval_core.PASS_N = PASS_N
    eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
    eval_core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
        FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
    )
    eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = not SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.AUTO_TUNE_CONCURRENCY = False
    eval_core.DEFAULT_EVAL_CONCURRENCY = DEFAULT_EVAL_CONCURRENCY
    eval_core.DEEPSEEK_EVAL_CONCURRENCY = DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[difficulty]
    eval_core.DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY = dict(DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
    eval_core.QINGYUN_DEEPSEEK_EVAL_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY
    eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    eval_core.LONG_CONTEXT_TURNS = turns


def main() -> None:
    _install_rag_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            _configure_evaluation(task_difficulty, turns)
            eval_core.main()


if __name__ == "__main__":
    main()
