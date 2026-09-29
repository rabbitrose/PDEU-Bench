#!/usr/bin/env python3
"""Judge Agent results with PAG-aligned behavior plus Wikipedia RAG-B context."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
PAG_DIR = EVAL_DIR / "pag"
for module_dir in (EVAL_DIR, PAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import judge
import judge_pag


# =========================
# 用户长期可调 Agent judge 超参数
# =========================

# 除 preference mode、结果目录与 Wikipedia RAG-B 外，全部配置继承 PAG。
TARGET_GEN_LLM = judge_pag.TARGET_GEN_LLM
PASS_N = judge_pag.PASS_N
ENABLE_LONG_CONTEXT = judge_pag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = judge_pag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(judge_pag.LONG_CONTEXT_TURNS_LIST)
TARGET_EVAL_RUN_ID = judge_pag.TARGET_EVAL_RUN_ID
JUDGE_LATEST_EVAL_RUN_ONLY = judge_pag.JUDGE_LATEST_EVAL_RUN_ONLY
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(judge_pag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
DEFAULT_JUDGE_CONCURRENCY = judge_pag.DEFAULT_JUDGE_CONCURRENCY
QINGYUN_JUDGE_CONCURRENCY = judge_pag.QINGYUN_JUDGE_CONCURRENCY
USER_ID_FILTER: list[str] = list(judge_pag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(judge_pag.TASK_ID_FILTER)
TASK_DIFFICULTY = judge_pag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(judge_pag.TASK_DIFFICULTY_LIST)


AGENT_MODE = "agent"
AGENT_LABEL = "PAG RAG-A Top-3 原始 QA + User abstract profile + 离线 Wikipedia RAG-B"
PAG_ABSTRACT_CONTEXT_KEY = judge_pag.PAG_ABSTRACT_CONTEXT_KEY
AGENT_RAG_B_CONTEXT_KEY = "agent_rag_b_wikipedia"


def _register_agent_mode() -> None:
    judge.VALID_PREFERENCE_INJECTION_MODES.add(AGENT_MODE)
    judge.PREFERENCE_INJECTION_LABELS[AGENT_MODE] = AGENT_LABEL


def _patch_agent_judge_context() -> None:
    if getattr(judge, "_AGENT_PAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", False):
        return

    original_preference_source_description = judge.preference_source_description
    original_profile_core = judge.profile_core
    original_build_judge_context = judge.build_judge_context

    def preference_source_description_with_agent(record: dict) -> str:
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode == AGENT_MODE:
            return (
                "PAG 同一 User abstract profile 与 RAG-A Top-3 原始历史 QA 提供用户偏好；"
                "RAG-B 离线 Wikipedia 摘要只提供外部任务知识，不是偏好证据"
            )
        return original_preference_source_description(record)

    def profile_core_with_agent(record: dict) -> dict:
        core = original_profile_core(record)
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode != AGENT_MODE or not isinstance(core, dict):
            return core
        provided_context = judge_pag._provided_context(record)
        rag_history = provided_context.get("rag_retrieved_user_interaction_history")
        if rag_history is None:
            raise RuntimeError("Agent eval record 缺少 RAG-A rag_retrieved_user_interaction_history。")
        user_abstract = provided_context.get(PAG_ABSTRACT_CONTEXT_KEY)
        if not isinstance(user_abstract, str) or not user_abstract.strip():
            raise RuntimeError(f"Agent eval record 缺少非空 {PAG_ABSTRACT_CONTEXT_KEY}。")
        core["preference_injection_mode"] = AGENT_MODE
        core["preference_injection_label"] = AGENT_LABEL
        core["rag_retrieved_user_interaction_history"] = rag_history
        core[PAG_ABSTRACT_CONTEXT_KEY] = user_abstract
        core["preference_evidence_policy"] = (
            "只允许使用 PAG User abstract profile 与 RAG-A 历史对话判断偏好；"
            "不得把 Wikipedia 内容解释为用户偏好。"
        )
        return core

    def build_judge_context_with_agent(record: dict) -> dict:
        context = original_build_judge_context(record)
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode == AGENT_MODE:
            provided_context = judge_pag._provided_context(record)
            wikipedia = provided_context.get(AGENT_RAG_B_CONTEXT_KEY)
            if not isinstance(wikipedia, dict):
                raise RuntimeError(f"Agent eval record 缺少 {AGENT_RAG_B_CONTEXT_KEY}。")
            context["external_knowledge_context"] = {
                "rag_b_wikipedia": wikipedia,
                "policy": "该字段只用于任务背景知识与事实支撑，不属于用户偏好。",
            }
        return context

    judge.preference_source_description = preference_source_description_with_agent
    judge.profile_core = profile_core_with_agent
    judge.build_judge_context = build_judge_context_with_agent
    setattr(judge, "_AGENT_PAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", True)


def _install_agent_judge_behavior() -> None:
    judge_pag._install_pag_behavior()
    _register_agent_mode()
    _patch_agent_judge_context()


def _target_gen_model_name() -> str:
    if TARGET_GEN_LLM:
        return TARGET_GEN_LLM
    return str(getattr(judge.load_api_config(), "GEN_MODEL"))


def _rename_agent_html_and_update_summary(previous_htmls: set[Path]) -> None:
    target_model = _target_gen_model_name()
    html_dir = judge.html_dir_for_model(target_model, AGENT_MODE, PASS_N)
    judge_dir = judge.judge_result_dir_for_model(target_model, AGENT_MODE, PASS_N)
    for old_path in sorted(set(html_dir.glob("judge_report_*.html")) - previous_htmls):
        run_id = old_path.stem.removeprefix("judge_report_")
        new_path = old_path.with_name(f"智能体评测报告_{run_id}.html")
        old_path.replace(new_path)
        summary_path = judge_dir / f"judge_summary_{run_id}.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
        outputs["html_report"] = str(new_path)
        summary["outputs"] = outputs
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(summary_path)


def _configure_judge(task_difficulty: str, turns: int) -> None:
    judge_pag._configure_judge(task_difficulty, turns)
    judge.PREFERENCE_INJECTION_MODE = AGENT_MODE
    judge.PREFERENCE_INJECTION_LABELS[AGENT_MODE] = AGENT_LABEL


def main() -> None:
    _install_agent_judge_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            _configure_judge(task_difficulty, turns)
            html_dir = judge.html_dir_for_model(_target_gen_model_name(), AGENT_MODE, PASS_N)
            previous_htmls = set(html_dir.glob("judge_report_*.html")) if html_dir.exists() else set()
            judge.main()
            _rename_agent_html_and_update_summary(previous_htmls)


if __name__ == "__main__":
    main()
