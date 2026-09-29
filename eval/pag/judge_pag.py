#!/usr/bin/env python3
"""Judge PAG results with RAG-aligned LLM-as-judge behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parents[1]
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import judge
import judge_rag


# =========================
# 用户长期可调 judge 超参数
# =========================

# 除 preference mode 与结果目录外，以下配置与 eval/rag/judge_rag.py 保持一致。
TARGET_GEN_LLM = judge_rag.TARGET_GEN_LLM
PASS_N = judge_rag.PASS_N
ENABLE_LONG_CONTEXT = judge_rag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = judge_rag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(judge_rag.LONG_CONTEXT_TURNS_LIST)
TARGET_EVAL_RUN_ID = judge_rag.TARGET_EVAL_RUN_ID
JUDGE_LATEST_EVAL_RUN_ONLY = judge_rag.JUDGE_LATEST_EVAL_RUN_ONLY
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(
    judge_rag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY
)
DEFAULT_JUDGE_CONCURRENCY = judge_rag.DEFAULT_JUDGE_CONCURRENCY
QINGYUN_JUDGE_CONCURRENCY = judge_rag.QINGYUN_JUDGE_CONCURRENCY
USER_ID_FILTER: list[str] = list(judge_rag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(judge_rag.TASK_ID_FILTER)
TASK_DIFFICULTY = judge_rag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(judge_rag.TASK_DIFFICULTY_LIST)


PAG_MODE = "pag"
PAG_LABEL = "RAG Top-3 原始 QA + User abstract profile"
PAG_ABSTRACT_CONTEXT_KEY = "pag_user_abstract_profile"


def _register_pag_mode() -> None:
    judge.VALID_PREFERENCE_INJECTION_MODES.add(PAG_MODE)
    judge.PREFERENCE_INJECTION_LABELS[PAG_MODE] = PAG_LABEL


def _provided_context(record: dict) -> dict:
    provided = (
        record.get("provided_eval_context")
        if isinstance(record.get("provided_eval_context"), dict)
        else record.get("provided_preference_context")
        if isinstance(record.get("provided_preference_context"), dict)
        else {}
    )
    return provided.get("context") if isinstance(provided.get("context"), dict) else {}


def _patch_pag_judge_context() -> None:
    if getattr(judge, "_PAG_RAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", False):
        return

    original_preference_source_description = judge.preference_source_description
    original_profile_core = judge.profile_core

    def preference_source_description_with_pag(record: dict) -> str:
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode == PAG_MODE:
            return (
                "从 hidden info 中检索得到的与当前任务最相关 Top-3 原始历史 QA，"
                "以及额外提供的 User abstract profile"
            )
        return original_preference_source_description(record)

    def profile_core_with_pag(record: dict) -> dict:
        core = original_profile_core(record)
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode != PAG_MODE or not isinstance(core, dict):
            return core
        provided_context = _provided_context(record)
        rag_history = provided_context.get("rag_retrieved_user_interaction_history")
        if rag_history is None:
            raise RuntimeError("PAG eval record 缺少 rag_retrieved_user_interaction_history。")
        core["rag_retrieved_user_interaction_history"] = rag_history
        user_abstract = provided_context.get(PAG_ABSTRACT_CONTEXT_KEY)
        if not isinstance(user_abstract, str) or not user_abstract.strip():
            raise RuntimeError(f"PAG eval record 缺少非空 {PAG_ABSTRACT_CONTEXT_KEY}。")
        core[PAG_ABSTRACT_CONTEXT_KEY] = user_abstract
        return core

    judge.preference_source_description = preference_source_description_with_pag
    judge.profile_core = profile_core_with_pag
    setattr(judge, "_PAG_RAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", True)


def _install_pag_behavior() -> None:
    judge_rag.QINGYUN_JUDGE_CONCURRENCY = QINGYUN_JUDGE_CONCURRENCY
    judge_rag._install_rag_behavior()
    _register_pag_mode()
    _patch_pag_judge_context()


def _target_gen_model_name() -> str:
    if TARGET_GEN_LLM:
        return TARGET_GEN_LLM
    return str(getattr(judge.load_api_config(), "GEN_MODEL"))


def _rename_pag_html_and_update_summary(previous_htmls: set[Path]) -> None:
    target_model = _target_gen_model_name()
    html_dir = judge.html_dir_for_model(target_model, PAG_MODE, PASS_N)
    judge_dir = judge.judge_result_dir_for_model(target_model, PAG_MODE, PASS_N)
    for old_path in sorted(set(html_dir.glob("judge_report_*.html")) - previous_htmls):
        run_id = old_path.stem.removeprefix("judge_report_")
        new_path = old_path.with_name(f"PAG评测报告_{run_id}.html")
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
    difficulty = judge.normalize_task_difficulty(task_difficulty)
    judge.TARGET_GEN_LLM = TARGET_GEN_LLM
    judge.PREFERENCE_INJECTION_MODE = PAG_MODE
    judge.PASS_N = PASS_N
    judge.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    judge.LONG_CONTEXT_TURNS = turns
    judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
    judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
    judge.AUTO_TUNE_CONCURRENCY = False
    judge.DEFAULT_JUDGE_CONCURRENCY = DEFAULT_JUDGE_CONCURRENCY
    judge.DEEPSEEK_JUDGE_CONCURRENCY = DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[difficulty]
    judge.QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = QINGYUN_JUDGE_CONCURRENCY
    judge.USER_ID_FILTER = USER_ID_FILTER
    judge.TASK_ID_FILTER = TASK_ID_FILTER
    judge.TASK_DIFFICULTY = difficulty


def main() -> None:
    _install_pag_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            _configure_judge(task_difficulty, turns)
            html_dir = judge.html_dir_for_model(_target_gen_model_name(), PAG_MODE, PASS_N)
            previous_htmls = set(html_dir.glob("judge_report_*.html")) if html_dir.exists() else set()
            judge.main()
            _rename_pag_html_and_update_summary(previous_htmls)


if __name__ == "__main__":
    main()
