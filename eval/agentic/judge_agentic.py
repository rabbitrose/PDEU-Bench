#!/usr/bin/env python3
"""Judge Agentic Memory results with PAG-aligned LLM-as-judge behavior."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
PAG_DIR = EVAL_DIR / "pag"
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, PAG_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import judge
import judge_pag


# =========================
# 用户长期可调 Agentic Memory judge 超参数
# =========================

# 除 preference mode 与结果目录外，全部 judge 配置直接继承 PAG。
TARGET_GEN_LLM = judge_pag.TARGET_GEN_LLM
PASS_N = judge_pag.PASS_N
ENABLE_LONG_CONTEXT = judge_pag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = judge_pag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(judge_pag.LONG_CONTEXT_TURNS_LIST)
TARGET_EVAL_RUN_ID = judge_pag.TARGET_EVAL_RUN_ID
JUDGE_LATEST_EVAL_RUN_ONLY = judge_pag.JUDGE_LATEST_EVAL_RUN_ONLY
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(
    judge_pag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY
)
DEFAULT_JUDGE_CONCURRENCY = judge_pag.DEFAULT_JUDGE_CONCURRENCY
QINGYUN_JUDGE_CONCURRENCY = judge_pag.QINGYUN_JUDGE_CONCURRENCY
USER_ID_FILTER: list[str] = list(judge_pag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(judge_pag.TASK_ID_FILTER)
TASK_DIFFICULTY = judge_pag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(judge_pag.TASK_DIFFICULTY_LIST)

# 动态画像是被测方法的内部状态，不作为离线 judge 的偏好真值。
# judge 始终使用与 PAG 相同的 Top-3 原始历史 QA 和初始 abstract profile P0。
USE_DYNAMIC_PROFILE_AS_JUDGE_REFERENCE = False


AGENTIC_MODE = "agentic"
AGENTIC_LABEL = "PAG + per-LLM-turn dynamic User abstract profile memory"
PAG_MODE = judge_pag.PAG_MODE
PAG_ABSTRACT_CONTEXT_KEY = judge_pag.PAG_ABSTRACT_CONTEXT_KEY
AGENTIC_MAIN_PHASES = {"initial_plan", "reason_action", "update_plan"}
PROFILE_UPDATE_STATUSES = {"completed", "failed"}
PROFILE_UPDATE_DECISIONS = {"updated", "unchanged"}
AGENTIC_PROFILE_MAX_CHARS = 6000
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


def _register_agentic_mode() -> None:
    judge.VALID_PREFERENCE_INJECTION_MODES.add(AGENTIC_MODE)
    judge.PREFERENCE_INJECTION_LABELS[AGENTIC_MODE] = AGENTIC_LABEL


def _clear_proxy_environment() -> list[str]:
    """Qingyun is called directly; inherited local proxy endpoints are unreachable remotely."""

    removed: list[str] = []
    for name in PROXY_ENV_NAMES:
        if name in os.environ:
            os.environ.pop(name, None)
            removed.append(name)
    return removed


def _provided_context(record: dict[str, Any]) -> dict[str, Any]:
    provided = (
        record.get("provided_eval_context")
        if isinstance(record.get("provided_eval_context"), dict)
        else record.get("provided_preference_context")
        if isinstance(record.get("provided_preference_context"), dict)
        else {}
    )
    return provided.get("context") if isinstance(provided.get("context"), dict) else {}


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raise_memory_error(record: dict[str, Any], errors: list[str]) -> None:
    if not errors:
        return
    case = (
        f"{record.get('run_id')}::{record.get('user_id')}::{record.get('task_id')}"
        f"::pass_{record.get('pass_index') or 1}"
    )
    raise RuntimeError(
        f"Agentic profile memory 审计失败: case={case}: " + "; ".join(errors)
    )


def _validate_agentic_profile_memory(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete P0 -> Pn chain without exposing Pn to judge prompts."""

    errors: list[str] = []
    memory = record.get("agentic_profile_memory")
    if not isinstance(memory, dict):
        _raise_memory_error(record, ["缺少 agentic_profile_memory object"])
        raise AssertionError("unreachable")

    initial_profile = memory.get("initial_profile")
    if not isinstance(initial_profile, str) or not initial_profile.strip():
        errors.append("initial_profile 必须是非空字符串")
        initial_profile = ""
    elif len(initial_profile) > AGENTIC_PROFILE_MAX_CHARS:
        errors.append(f"initial_profile 超过 {AGENTIC_PROFILE_MAX_CHARS} 字符")

    authoritative_profile = _provided_context(record).get(PAG_ABSTRACT_CONTEXT_KEY)
    if not isinstance(authoritative_profile, str) or not authoritative_profile.strip():
        errors.append(f"provided context 缺少非空 {PAG_ABSTRACT_CONTEXT_KEY}")
    elif authoritative_profile != initial_profile:
        errors.append("provided context 中的权威画像不是 initial_profile P0")

    updates = memory.get("updates")
    if not isinstance(updates, list):
        errors.append("updates 必须是数组")
        updates = []

    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    trajectory_events: dict[tuple[str, int], dict[str, Any]] = {}
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "")
        iteration = _nonnegative_int(item.get("iteration"))
        if phase in AGENTIC_MAIN_PHASES and iteration is not None:
            trajectory_events[(phase, iteration)] = item

    current_profile = initial_profile
    current_version = 0
    completed_count = 0
    failed_count = 0
    seen_phase_iterations: set[tuple[str, int]] = set()
    expected_snapshots: list[dict[str, Any]] = [
        {"snapshot_id": "profile_v0", "version": 0, "profile": initial_profile}
    ]
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    eval_model_name = model.get("name")
    eval_base_url = model.get("base_url")

    for index, event in enumerate(updates, start=1):
        if not isinstance(event, dict):
            errors.append(f"updates[{index}] 必须是 object")
            continue

        if event.get("sequence") != index:
            errors.append(f"updates[{index}].sequence 必须为 {index}")
        phase = str(event.get("phase") or "")
        iteration = _nonnegative_int(event.get("iteration"))
        if phase not in AGENTIC_MAIN_PHASES:
            errors.append(f"updates[{index}].phase 非法: {phase!r}")
        if iteration is None:
            errors.append(f"updates[{index}].iteration 必须是非负整数")
            iteration = 0
        if phase == "initial_plan" and iteration != 0:
            errors.append(f"updates[{index}] initial_plan 的 iteration 必须为 0")
        if phase in {"reason_action", "update_plan"} and iteration <= 0:
            errors.append(f"updates[{index}] {phase} 的 iteration 必须为正整数")
        phase_iteration = (phase, iteration)
        if phase_iteration in seen_phase_iterations:
            errors.append(f"updates[{index}] 重复主阶段 {phase_iteration}")
        seen_phase_iterations.add(phase_iteration)

        expected_used_id = f"profile_v{current_version}"
        if event.get("profile_version_used") != current_version:
            errors.append(f"updates[{index}] profile_version_used 链断裂")
        if event.get("profile_snapshot_id_used") != expected_used_id:
            errors.append(f"updates[{index}] profile_snapshot_id_used 链断裂")
        if event.get("profile_before") != current_profile:
            errors.append(f"updates[{index}] profile_before 与上一有效画像不一致")
        if event.get("same_llm_client_object") is not True:
            errors.append(f"updates[{index}] 未证明使用同一个 LLMClient")
        if event.get("profile_update_model") != eval_model_name:
            errors.append(f"updates[{index}] profile update 模型与被测模型不一致")
        if event.get("profile_update_base_url") != eval_base_url:
            errors.append(f"updates[{index}] profile update API URL 与被测模型不一致")
        expected_purpose = f"{record.get('user_id')}/{record.get('task_id')}/{phase}"
        if phase != "initial_plan":
            expected_purpose += f"/{iteration}"
        if event.get("main_purpose") != expected_purpose:
            errors.append(f"updates[{index}].main_purpose 与主阶段不一致")

        status = event.get("status")
        if status not in PROFILE_UPDATE_STATUSES:
            errors.append(f"updates[{index}].status 非法: {status!r}")
            continue

        if status == "completed":
            completed_count += 1
            profile_after = event.get("profile_after")
            version_after = current_version + 1
            if not isinstance(profile_after, str) or not profile_after.strip():
                errors.append(f"updates[{index}].profile_after 必须是非空字符串")
                profile_after = current_profile
            elif len(profile_after) > AGENTIC_PROFILE_MAX_CHARS:
                errors.append(f"updates[{index}].profile_after 超过长度限制")
            if event.get("profile_version_after") != version_after:
                errors.append(f"updates[{index}] profile_version_after 必须为 {version_after}")
            if event.get("profile_snapshot_id_after") != f"profile_v{version_after}":
                errors.append(f"updates[{index}] profile_snapshot_id_after 链断裂")
            decision = event.get("update_decision")
            if decision not in PROFILE_UPDATE_DECISIONS:
                errors.append(f"updates[{index}].update_decision 非法: {decision!r}")
            if decision == "unchanged" and profile_after != current_profile:
                errors.append(f"updates[{index}] unchanged 却改写了画像")
            for key in (
                "newly_supported_preferences",
                "weakened_or_removed_preferences",
                "evidence_summary",
            ):
                value = event.get(key)
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    errors.append(f"updates[{index}].{key} 必须是字符串数组")
            current_profile = profile_after
            current_version = version_after
            expected_snapshots.append(
                {
                    "snapshot_id": f"profile_v{current_version}",
                    "version": current_version,
                    "profile": current_profile,
                }
            )
        else:
            failed_count += 1
            if event.get("profile_after") != current_profile:
                errors.append(f"updates[{index}] failed 后必须保留上一画像")
            if event.get("profile_version_after") != current_version:
                errors.append(f"updates[{index}] failed 后不得递增版本")
            if event.get("profile_snapshot_id_after") != expected_used_id:
                errors.append(f"updates[{index}] failed 后 snapshot id 必须保持不变")

        trajectory_event = trajectory_events.get(phase_iteration)
        if trajectory_event is not None:
            parsed = trajectory_event.get("parsed")
            if not isinstance(parsed, dict):
                errors.append(f"trajectory {phase_iteration} 缺少主 LLM parsed output")
            elif event.get("main_output_sha256") != _json_digest(parsed):
                errors.append(f"trajectory {phase_iteration} 的主输出 hash 不一致")
            attempts = trajectory_event.get("llm_attempts")
            if not isinstance(attempts, list) or not attempts:
                errors.append(f"trajectory {phase_iteration} 缺少主 LLM attempts")
            elif not isinstance(attempts[-1], dict) or attempts[-1].get("ok") is not True:
                errors.append(f"trajectory {phase_iteration} 的最终主 LLM attempt 未成功")
            attachment = trajectory_event.get("agentic_profile_memory")
            if not isinstance(attachment, dict):
                errors.append(f"trajectory {phase_iteration} 缺少画像审计附件")
            else:
                expected_attachment = {
                    "profile_version_used": event.get("profile_version_used"),
                    "profile_snapshot_id_used": event.get("profile_snapshot_id_used"),
                    "profile_used": event.get("profile_before"),
                    "profile_update_status": event.get("status"),
                    "profile_version_after": event.get("profile_version_after"),
                    "profile_snapshot_id_after": event.get("profile_snapshot_id_after"),
                    "profile_after": event.get("profile_after"),
                }
                if attachment != expected_attachment:
                    errors.append(f"trajectory {phase_iteration} 的画像附件与 update 不一致")

    if memory.get("snapshots") != expected_snapshots:
        errors.append("snapshots 与 completed update 链不一致")
    if memory.get("final_profile") != current_profile:
        errors.append("final_profile 与更新链终点不一致")
    if memory.get("final_profile_version") != current_version:
        errors.append("final_profile_version 与更新链终点不一致")
    if memory.get("main_successful_llm_turn_count") != len(updates):
        errors.append("main_successful_llm_turn_count 与 updates 数量不一致")
    if memory.get("profile_update_event_count") != len(updates):
        errors.append("profile_update_event_count 与 updates 数量不一致")
    if memory.get("completed_profile_update_count") != completed_count:
        errors.append("completed_profile_update_count 统计错误")
    if memory.get("failed_profile_update_count") != failed_count:
        errors.append("failed_profile_update_count 统计错误")

    _raise_memory_error(record, errors)
    return {
        "status": "passed",
        "initial_profile_version": 0,
        "final_profile_version": current_version,
        "profile_update_event_count": len(updates),
        "completed_profile_update_count": completed_count,
        "failed_profile_update_count": failed_count,
        "judge_reference_profile_version": 0,
        "dynamic_profiles_used_for_scoring": False,
    }


def _patch_agentic_judge_context() -> None:
    if getattr(judge, "_AGENTIC_PAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", False):
        return

    original_preference_source_description = judge.preference_source_description
    original_profile_core = judge.profile_core
    original_judge_case = judge.judge_case
    original_discover_eval_records = judge.discover_eval_records

    def preference_source_description_with_agentic(record: dict[str, Any]) -> str:
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode == AGENTIC_MODE:
            return (
                "与 PAG 完全相同的 Top-3 原始历史 QA 和初始 User abstract profile P0；"
                "动态画像 P1...Pn 仅用于方法审计，不作为偏好评分真值"
            )
        return original_preference_source_description(record)

    def profile_core_with_agentic(record: dict[str, Any]) -> dict[str, Any]:
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        if mode != AGENTIC_MODE:
            return original_profile_core(record)
        _validate_agentic_profile_memory(record)
        pag_record = copy.deepcopy(record)
        pag_record["preference_injection_mode"] = PAG_MODE
        provided = pag_record.get("provided_eval_context")
        if isinstance(provided, dict):
            provided["mode"] = PAG_MODE
        core = original_profile_core(pag_record)
        if not isinstance(core, dict):
            raise RuntimeError("PAG-aligned judge profile_core 未返回 object。")
        core["preference_injection_mode"] = AGENTIC_MODE
        core["preference_injection_label"] = AGENTIC_LABEL
        return core

    def discover_eval_records_with_agentic_audit(logger: Any) -> list[dict[str, Any]]:
        records = original_discover_eval_records(logger)
        for record in records:
            mode = str(
                record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE
            )
            if mode == AGENTIC_MODE:
                record["_agentic_profile_memory_audit"] = (
                    _validate_agentic_profile_memory(record)
                )
        return records

    def judge_case_with_agentic_audit(record: dict[str, Any], client: Any, logger: Any) -> dict[str, Any]:
        mode = str(record.get("preference_injection_mode") or judge.PREFERENCE_INJECTION_MODE)
        audit = record.pop("_agentic_profile_memory_audit", None)
        if mode == AGENTIC_MODE and not isinstance(audit, dict):
            audit = _validate_agentic_profile_memory(record)
        result = original_judge_case(record, client, logger)
        if audit is not None and isinstance(result, dict):
            result["agentic_profile_memory_audit"] = audit
        return result

    judge.preference_source_description = preference_source_description_with_agentic
    judge.profile_core = profile_core_with_agentic
    judge.discover_eval_records = discover_eval_records_with_agentic_audit
    judge.judge_case = judge_case_with_agentic_audit
    setattr(judge, "_AGENTIC_PAG_ALIGNED_JUDGE_BEHAVIOR_INSTALLED", True)


def _install_agentic_behavior() -> None:
    judge_pag._install_pag_behavior()
    _register_agentic_mode()
    _patch_agentic_judge_context()


def _target_gen_model_name() -> str:
    if TARGET_GEN_LLM:
        return TARGET_GEN_LLM
    return str(getattr(judge.load_api_config(), "GEN_MODEL"))


def _rename_agentic_html_and_update_summary(previous_htmls: set[Path]) -> None:
    target_model = _target_gen_model_name()
    html_dir = judge.html_dir_for_model(target_model, AGENTIC_MODE, PASS_N)
    judge_dir = judge.judge_result_dir_for_model(target_model, AGENTIC_MODE, PASS_N)
    for old_path in sorted(set(html_dir.glob("judge_report_*.html")) - previous_htmls):
        run_id = old_path.stem.removeprefix("judge_report_")
        new_path = old_path.with_name(f"Agentic评测报告_{run_id}.html")
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
    judge_pag._configure_judge(difficulty, turns)
    judge_pag.judge_rag.QINGYUN_JUDGE_CONCURRENCY = QINGYUN_JUDGE_CONCURRENCY
    judge.TARGET_GEN_LLM = TARGET_GEN_LLM
    judge.PREFERENCE_INJECTION_MODE = AGENTIC_MODE
    judge.PREFERENCE_INJECTION_LABELS[AGENTIC_MODE] = AGENTIC_LABEL
    judge.PASS_N = PASS_N
    judge.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    judge.LONG_CONTEXT_TURNS = turns
    judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
    judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
    judge.AUTO_TUNE_CONCURRENCY = False
    judge.DEFAULT_JUDGE_CONCURRENCY = DEFAULT_JUDGE_CONCURRENCY
    judge.DEEPSEEK_JUDGE_CONCURRENCY = DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[
        difficulty
    ]
    judge.QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = QINGYUN_JUDGE_CONCURRENCY
    judge.USER_ID_FILTER = list(USER_ID_FILTER)
    judge.TASK_ID_FILTER = list(TASK_ID_FILTER)
    judge.TASK_DIFFICULTY = difficulty


def main() -> None:
    _clear_proxy_environment()
    _install_agentic_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            _configure_judge(task_difficulty, turns)
            html_dir = judge.html_dir_for_model(
                _target_gen_model_name(), AGENTIC_MODE, PASS_N
            )
            previous_htmls = (
                set(html_dir.glob("judge_report_*.html")) if html_dir.exists() else set()
            )
            judge.main()
            _rename_agentic_html_and_update_summary(previous_htmls)


if __name__ == "__main__":
    main()
