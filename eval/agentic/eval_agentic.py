#!/usr/bin/env python3
"""Run PAG with a task-local abstract profile updated after every main LLM turn."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


EVAL_DIR = Path(__file__).resolve().parents[1]
PAG_DIR = EVAL_DIR / "pag"
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, PAG_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_core
import eval_pag


# =========================
# 用户长期可调 Agentic Memory 超参数
# =========================

# 除 abstract profile 的逐交互更新外，全部评测配置直接继承 PAG。
USER_ID_FILTER: list[str] = list(eval_pag.USER_ID_FILTER)
TASK_ID_FILTER: list[str] = list(eval_pag.TASK_ID_FILTER)
TASK_DIFFICULTY = eval_pag.TASK_DIFFICULTY
TASK_DIFFICULTY_LIST: list[str] = list(eval_pag.TASK_DIFFICULTY_LIST)
PASS_N = eval_pag.PASS_N
MAX_TOOL_CALLS_PER_TASK = eval_pag.MAX_TOOL_CALLS_PER_TASK
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
    eval_pag.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
)
SKIP_TASKS_ALREADY_IN_RESULT_FILE = eval_pag.SKIP_TASKS_ALREADY_IN_RESULT_FILE
ENABLE_LONG_CONTEXT = eval_pag.ENABLE_LONG_CONTEXT
LONG_CONTEXT_TURNS = eval_pag.LONG_CONTEXT_TURNS
LONG_CONTEXT_TURNS_LIST: list[int] = list(eval_pag.LONG_CONTEXT_TURNS_LIST)
PAG_RETRIEVAL_TOP_K = eval_pag.PAG_RETRIEVAL_TOP_K
RAG_PROMPT_MAX_CHARS = eval_pag.RAG_PROMPT_MAX_CHARS
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(eval_pag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
DEFAULT_EVAL_CONCURRENCY = eval_pag.DEFAULT_EVAL_CONCURRENCY
QINGYUN_EVAL_CONCURRENCY = eval_pag.QINGYUN_EVAL_CONCURRENCY

# Profile update 使用主任务传入的同一个 LLMClient、模型、URL 与 API key。
AGENTIC_PROFILE_UPDATE_TEMPERATURE = eval_core.MODEL_TEMPERATURE
AGENTIC_PROFILE_UPDATE_MAX_TOKENS = eval_core.MODEL_MAX_TOKENS
AGENTIC_PROFILE_UPDATE_TIMEOUT_SECONDS = eval_core.REQUEST_TIMEOUT_SECONDS

# 更新提示中保留本轮主 LLM input/output 的最大字符数；主任务 Prompt 不受该值影响。
AGENTIC_PROFILE_UPDATE_INPUT_MAX_CHARS = 48000

# 优先为本轮主 LLM output 保留额度，避免长 PAG input 把 output 从尾部截掉。
AGENTIC_PROFILE_UPDATE_OUTPUT_MAX_CHARS = 24000

# 与 PAG User abstract profile 相同的最大长度。
AGENTIC_PROFILE_MAX_CHARS = eval_pag.USER_ABSTRACT_PROFILE_MAX_CHARS

# False 表示更新 API 偶发失败时显式记录错误并保留上一有效 profile，主评测继续。
AGENTIC_PROFILE_UPDATE_STRICT = False


AGENTIC_MODE = "agentic"
AGENTIC_LABEL = "PAG + per-LLM-turn dynamic User abstract profile memory"
AGENTIC_MAIN_PHASES = ("initial_plan", "reason_action", "update_plan")
PROFILE_UPDATE_DECISIONS = {"updated", "unchanged"}
PROFILE_UPDATE_KEYS = {
    "updated_user_abstract",
    "update_decision",
    "newly_supported_preferences",
    "weakened_or_removed_preferences",
    "evidence_summary",
}


@dataclass
class _AgenticTaskState:
    user_id: str
    task_id: str
    run_id: str
    pass_n: int
    pass_index: int
    detail_path: Path | None
    initial_profile: str
    current_profile: str
    current_version: int = 0
    context: dict[str, Any] | None = None
    main_call_count: int = 0
    updates: list[dict[str, Any]] = field(default_factory=list)


_TASK_STATE = threading.local()
_BASE_REQUEST_STRUCTURED_JSON: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]] | None = None
_STATIC_PAG_PROFILE_LOADER: Callable[..., str] | None = None


def _active_state() -> _AgenticTaskState | None:
    state = getattr(_TASK_STATE, "value", None)
    return state if isinstance(state, _AgenticTaskState) else None


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _main_phase_from_purpose(
    state: _AgenticTaskState,
    purpose: str,
) -> tuple[str, int] | None:
    prefix = f"{state.user_id}/{state.task_id}/"
    if not purpose.startswith(prefix):
        return None
    suffix = purpose[len(prefix) :]
    if suffix == "initial_plan":
        return "initial_plan", 0
    match = re.fullmatch(r"(reason_action|update_plan)/(\d+)", suffix)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _validate_profile_update(
    obj: Any,
    *,
    profile_before: str | None = None,
) -> list[str]:
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    errors: list[str] = []
    unexpected_keys = sorted(set(obj) - PROFILE_UPDATE_KEYS)
    if unexpected_keys:
        errors.append(f"不允许额外字段: {unexpected_keys}。")
    profile = obj.get("updated_user_abstract")
    if not isinstance(profile, str) or not profile.strip():
        errors.append("updated_user_abstract 必须是非空字符串。")
    elif len(profile) > AGENTIC_PROFILE_MAX_CHARS:
        errors.append(
            f"updated_user_abstract 不能超过 {AGENTIC_PROFILE_MAX_CHARS} 字符，实际 {len(profile)}。"
        )
    decision = obj.get("update_decision")
    if decision not in PROFILE_UPDATE_DECISIONS:
        errors.append(f"update_decision 必须属于 {sorted(PROFILE_UPDATE_DECISIONS)}。")
    elif isinstance(profile, str) and profile_before is not None:
        if decision == "unchanged" and profile != profile_before:
            errors.append("update_decision=unchanged 时必须逐字保留旧 profile。")
        if decision == "updated" and profile == profile_before:
            errors.append("update_decision=updated 时 profile 必须发生实际变化。")
    for key in ("newly_supported_preferences", "weakened_or_removed_preferences", "evidence_summary"):
        value = obj.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            errors.append(f"{key} 必须是字符串数组。")
    return errors


def _profile_update_messages(
    *,
    profile_before: str,
    phase: str,
    iteration: int,
    main_messages: list[dict[str, str]],
    main_output: dict[str, Any],
) -> list[dict[str, str]]:
    interaction_metadata = {
        "phase": phase,
        "iteration": iteration,
    }
    output_limit = min(
        AGENTIC_PROFILE_UPDATE_OUTPUT_MAX_CHARS,
        AGENTIC_PROFILE_UPDATE_INPUT_MAX_CHARS // 2,
    )
    output_text = eval_core.compact_json(main_output, output_limit)
    interaction_prefix = (
        f"interaction_metadata: {eval_core.compact_json(interaction_metadata, 1000)}\n"
        f"llm_output（优先保留，过长时单独截断）:\n{output_text}\n\n"
        "llm_input_messages（使用剩余字符额度，过长时截断）:\n"
    )
    input_limit = max(
        1000,
        AGENTIC_PROFILE_UPDATE_INPUT_MAX_CHARS - len(interaction_prefix),
    )
    interaction_text = interaction_prefix + eval_core.compact_json(
        main_messages,
        input_limit,
    )
    schema = {
        "updated_user_abstract": "更新后的完整 User abstract profile；没有可靠新证据时必须原样复制旧 profile",
        "update_decision": "updated 或 unchanged",
        "newly_supported_preferences": ["仅列本轮新增且有直接证据支持的稳定用户偏好"],
        "weakened_or_removed_preferences": ["仅列被本轮直接证据削弱或推翻的旧偏好"],
        "evidence_summary": ["简短说明本轮 input/output 提供了什么用户偏好证据"],
    }
    return [
        {
            "role": "system",
            "content": (
                "你是 Agentic Memory 的 User abstract profile 更新器。"
                "你必须保守更新长期用户画像，只依据本轮 LLM input/output 中关于用户本人稳定偏好的直接证据。"
                "任务要求、工具 schema、sandbox observation、计划进度、模型自行提出的建议都不是用户偏好。"
                "没有可靠新证据时，updated_user_abstract 必须逐字保留旧 profile。"
                "不要添加 source、provenance 或引用字段，只输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"当前 User abstract profile:\n{json.dumps(profile_before, ensure_ascii=False)}\n\n"
                f"本轮主 LLM input/output:\n{interaction_text}\n\n"
                f"必须遵守的 JSON schema:\n{eval_core.compact_json(schema, 6000)}"
            ),
        },
    ]


def _persist_profile_update_attempts(
    *,
    state: _AgenticTaskState,
    phase: str,
    iteration: int,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return eval_core.persist_llm_attempt_details(
        detail_path=state.detail_path,
        run_id=state.run_id,
        user_id=state.user_id,
        task_id=state.task_id,
        pass_n=state.pass_n,
        pass_index=state.pass_index,
        phase=f"agentic_profile_update_after_{phase}",
        iteration=iteration,
        attempts=attempts,
    )


def _update_profile_after_main_call(
    *,
    client: eval_core.LLMClient,
    logger: Any,
    state: _AgenticTaskState,
    phase: str,
    iteration: int,
    purpose: str,
    main_messages: list[dict[str, str]],
    main_output: dict[str, Any],
) -> None:
    if _BASE_REQUEST_STRUCTURED_JSON is None:
        raise RuntimeError("Agentic profile update request function 尚未安装。")

    state.main_call_count += 1
    sequence = state.main_call_count
    profile_before = state.current_profile
    update_messages = _profile_update_messages(
        profile_before=profile_before,
        phase=phase,
        iteration=iteration,
        main_messages=main_messages,
        main_output=main_output,
    )
    event: dict[str, Any] = {
        "sequence": sequence,
        "phase": phase,
        "iteration": iteration,
        "main_purpose": purpose,
        "profile_version_used": state.current_version,
        "profile_snapshot_id_used": f"profile_v{state.current_version}",
        "profile_before": profile_before,
        "main_input_message_count": len(main_messages),
        "main_input_sha256": _json_digest(main_messages),
        "main_output_sha256": _json_digest(main_output),
        "same_llm_client_object": True,
        "profile_update_model": client.model,
        "profile_update_base_url": client.base_url,
        "status": "running",
    }
    try:
        parsed, attempts = _BASE_REQUEST_STRUCTURED_JSON(
            client,
            messages=update_messages,
            purpose=(
                f"{state.user_id}/{state.task_id}/agentic_profile_update/"
                f"{sequence:03d}_{phase}_{iteration}"
            ),
            validator=lambda obj: _validate_profile_update(
                obj,
                profile_before=profile_before,
            ),
            logger=logger,
            max_tokens=AGENTIC_PROFILE_UPDATE_MAX_TOKENS,
            temperature=AGENTIC_PROFILE_UPDATE_TEMPERATURE,
            timeout=AGENTIC_PROFILE_UPDATE_TIMEOUT_SECONDS,
        )
        light_attempts = _persist_profile_update_attempts(
            state=state,
            phase=phase,
            iteration=iteration,
            attempts=attempts,
        )
        profile_after = str(parsed["updated_user_abstract"])
        state.current_profile = profile_after
        state.current_version += 1
        event.update(
            {
                "status": "completed",
                "profile_after": profile_after,
                "profile_version_after": state.current_version,
                "profile_snapshot_id_after": f"profile_v{state.current_version}",
                "update_decision": parsed.get("update_decision"),
                "newly_supported_preferences": parsed.get("newly_supported_preferences", []),
                "weakened_or_removed_preferences": parsed.get(
                    "weakened_or_removed_preferences", []
                ),
                "evidence_summary": parsed.get("evidence_summary", []),
                "profile_update_attempts": light_attempts,
            }
        )
        state.updates.append(event)
    except Exception as exc:
        raw_attempts = list(getattr(exc, "attempts", []))
        light_attempts = _persist_profile_update_attempts(
            state=state,
            phase=phase,
            iteration=iteration,
            attempts=raw_attempts,
        )
        event.update(
            {
                "status": "failed",
                "profile_after": profile_before,
                "profile_version_after": state.current_version,
                "profile_snapshot_id_after": f"profile_v{state.current_version}",
                "profile_update_attempts": light_attempts,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        state.updates.append(event)
        warning = getattr(logger, "warning", None)
        if callable(warning):
            warning(
                "Agentic profile update 失败，保留上一有效画像：user=%s task=%s "
                "phase=%s iteration=%s error=%s",
                state.user_id,
                state.task_id,
                phase,
                iteration,
                exc,
            )
        if AGENTIC_PROFILE_UPDATE_STRICT:
            raise RuntimeError(
                f"Agentic profile update 失败: phase={phase} iteration={iteration}: {exc}"
            ) from exc


def _attach_profile_versions_to_trajectory(
    record: dict[str, Any],
    updates: list[dict[str, Any]],
) -> None:
    by_phase_iteration = {
        (str(event.get("phase")), int(event.get("iteration") or 0)): event
        for event in updates
        if isinstance(event, dict)
    }
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list):
        return
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        phase = str(event.get("phase") or "")
        if phase not in AGENTIC_MAIN_PHASES:
            continue
        iteration = int(event.get("iteration") or 0)
        update = by_phase_iteration.get((phase, iteration))
        if not isinstance(update, dict):
            continue
        event["agentic_profile_memory"] = {
            "profile_version_used": update.get("profile_version_used"),
            "profile_snapshot_id_used": update.get("profile_snapshot_id_used"),
            "profile_used": update.get("profile_before"),
            "profile_update_status": update.get("status"),
            "profile_version_after": update.get("profile_version_after"),
            "profile_snapshot_id_after": update.get("profile_snapshot_id_after"),
            "profile_after": update.get("profile_after"),
        }


def _align_agentic_record_metadata(
    record: dict[str, Any],
    state: _AgenticTaskState,
) -> dict[str, Any]:
    eval_pag._align_pag_record_metadata(record)
    record["preference_injection_mode"] = AGENTIC_MODE
    record["preference_injection_label"] = AGENTIC_LABEL
    _attach_profile_versions_to_trajectory(record, state.updates)
    snapshots = [{"snapshot_id": "profile_v0", "version": 0, "profile": state.initial_profile}]
    for event in state.updates:
        if event.get("status") != "completed":
            continue
        snapshots.append(
            {
                "snapshot_id": event.get("profile_snapshot_id_after"),
                "version": event.get("profile_version_after"),
                "profile": event.get("profile_after"),
            }
        )
    record["agentic_profile_memory"] = {
        "initial_profile": state.initial_profile,
        "final_profile": state.current_profile,
        "final_profile_version": state.current_version,
        "snapshots": snapshots,
        "main_successful_llm_turn_count": state.main_call_count,
        "profile_update_event_count": len(state.updates),
        "completed_profile_update_count": sum(
            event.get("status") == "completed" for event in state.updates
        ),
        "failed_profile_update_count": sum(event.get("status") == "failed" for event in state.updates),
        "updates": copy.deepcopy(state.updates),
        "policy": (
            "每次成功的 initial_plan、reason_action、update_plan 逻辑生成后，"
            "使用同一个被测 LLMClient 更新一次 User abstract profile；"
            "profile-update 调用自身不递归触发更新；更新失败时显式记录并保留上一有效 profile。"
        ),
    }
    hyperparameters = record.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        hyperparameters.pop("pag_difference_from_rag", None)
        hyperparameters.update(
            {
                "agentic_difference_from_pag": "dynamic_user_abstract_profile_update_only",
                "agentic_main_prompt_template": "exact_pag_template_with_current_profile_value",
                "agentic_profile_update_after_phases": list(AGENTIC_MAIN_PHASES),
                "agentic_profile_update_same_llm_client": True,
                "agentic_profile_update_strict": AGENTIC_PROFILE_UPDATE_STRICT,
                "agentic_profile_max_chars": AGENTIC_PROFILE_MAX_CHARS,
                "agentic_profile_update_input_max_chars": (
                    AGENTIC_PROFILE_UPDATE_INPUT_MAX_CHARS
                ),
                "agentic_profile_update_output_max_chars": (
                    AGENTIC_PROFILE_UPDATE_OUTPUT_MAX_CHARS
                ),
                "agentic_profile_update_temperature": AGENTIC_PROFILE_UPDATE_TEMPERATURE,
                "agentic_profile_update_max_tokens": AGENTIC_PROFILE_UPDATE_MAX_TOKENS,
                "agentic_profile_update_timeout_seconds": (
                    AGENTIC_PROFILE_UPDATE_TIMEOUT_SECONDS
                ),
            }
        )
    provided = record.get("provided_eval_context")
    if isinstance(provided, dict):
        provided["mode"] = AGENTIC_MODE
        provided_context = provided.get("context")
        if isinstance(provided_context, dict):
            provided_context[eval_pag.PAG_ABSTRACT_CONTEXT_KEY] = state.initial_profile
    judge_results = record.get("judge_results")
    if isinstance(judge_results, dict):
        judge_results["offline_judge_entrypoint"] = "python eval/agentic/judge_agentic.py"
    judge_summary = record.get("judge_summary")
    if isinstance(judge_summary, dict) and judge_summary.get("enabled") is False:
        judge_summary["reason"] = (
            "inline judge 已关闭；正式指标请运行 "
            "python eval/agentic/judge_agentic.py 进行离线 judge。"
        )
    return record


def _register_agentic_mode() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(AGENTIC_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[AGENTIC_MODE] = AGENTIC_LABEL


def _install_agentic_behavior() -> None:
    global _BASE_REQUEST_STRUCTURED_JSON, _STATIC_PAG_PROFILE_LOADER

    eval_pag._install_pag_behavior()
    _register_agentic_mode()
    if getattr(eval_core, "_AGENTIC_PAG_ALIGNED_BEHAVIOR_INSTALLED", False):
        _BASE_REQUEST_STRUCTURED_JSON = getattr(
            eval_core,
            "_AGENTIC_BASE_REQUEST_STRUCTURED_JSON",
            _BASE_REQUEST_STRUCTURED_JSON,
        )
        _STATIC_PAG_PROFILE_LOADER = getattr(
            eval_core,
            "_AGENTIC_STATIC_PAG_PROFILE_LOADER",
            _STATIC_PAG_PROFILE_LOADER,
        )
        return

    original_run_task = eval_core.run_task
    original_build_common_context = eval_core.build_common_context
    original_initial_plan_messages = eval_core.initial_plan_messages
    original_action_messages = eval_core.action_messages
    original_update_plan_messages = eval_core.update_plan_messages
    original_offline_judge_entrypoint_for_mode = eval_core.offline_judge_entrypoint_for_mode
    _BASE_REQUEST_STRUCTURED_JSON = eval_core.request_structured_json
    _STATIC_PAG_PROFILE_LOADER = eval_pag._user_abstract_profile_for_task
    setattr(
        eval_core,
        "_AGENTIC_BASE_REQUEST_STRUCTURED_JSON",
        _BASE_REQUEST_STRUCTURED_JSON,
    )
    setattr(
        eval_core,
        "_AGENTIC_STATIC_PAG_PROFILE_LOADER",
        _STATIC_PAG_PROFILE_LOADER,
    )

    def build_common_context_with_agentic_state(**kwargs: Any) -> dict[str, Any]:
        context = original_build_common_context(**kwargs)
        state = _active_state()
        if state is not None and kwargs.get("preference_mode") == eval_pag.PAG_MODE:
            state.context = context
        return context

    def context_with_current_profile(context: dict[str, Any]) -> dict[str, Any]:
        state = _active_state()
        if state is None or state.context is not context:
            return context
        return {
            **context,
            eval_pag.PAG_ABSTRACT_CONTEXT_KEY: state.current_profile,
        }

    def initial_plan_messages_with_agentic_profile(
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        return original_initial_plan_messages(context_with_current_profile(context))

    def action_messages_with_agentic_profile(
        *,
        context: dict[str, Any],
        current_plan: dict[str, Any],
        trajectory_summary: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, str]]:
        return original_action_messages(
            context=context_with_current_profile(context),
            current_plan=current_plan,
            trajectory_summary=trajectory_summary,
            iteration=iteration,
        )

    def update_plan_messages_with_agentic_profile(
        *,
        context: dict[str, Any],
        previous_plan: dict[str, Any],
        action_json: dict[str, Any],
        sandbox_observation: dict[str, Any],
        iteration: int,
    ) -> list[dict[str, str]]:
        return original_update_plan_messages(
            context=context_with_current_profile(context),
            previous_plan=previous_plan,
            action_json=action_json,
            sandbox_observation=sandbox_observation,
            iteration=iteration,
        )

    def request_structured_json_with_agentic_memory(
        client: eval_core.LLMClient,
        *,
        messages: list[dict[str, str]],
        purpose: str,
        validator: Callable[[Any], list[str]],
        logger: Any,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert _BASE_REQUEST_STRUCTURED_JSON is not None
        parsed, attempts = _BASE_REQUEST_STRUCTURED_JSON(
            client,
            messages=messages,
            purpose=purpose,
            validator=validator,
            logger=logger,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        state = _active_state()
        phase_info = _main_phase_from_purpose(state, purpose) if state is not None else None
        if state is not None and phase_info is not None:
            phase, iteration = phase_info
            _update_profile_after_main_call(
                client=client,
                logger=logger,
                state=state,
                phase=phase,
                iteration=iteration,
                purpose=purpose,
                main_messages=messages,
                main_output=parsed,
            )
        return parsed, attempts

    def offline_judge_entrypoint_for_agentic(preference_mode: str) -> str:
        if str(preference_mode).strip().lower() == AGENTIC_MODE:
            return "python eval/agentic/judge_agentic.py"
        return original_offline_judge_entrypoint_for_mode(preference_mode)

    def run_task_with_agentic_memory(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != AGENTIC_MODE:
            return original_run_task(**kwargs)
        if _active_state() is not None:
            raise RuntimeError("Agentic task state 在线程内发生嵌套，拒绝继续执行。")
        if not isinstance(kwargs.get("hidden_info"), dict):
            raise RuntimeError("agentic 模式需要每个用户目录下存在 hidden_info.json。")
        user_id = str(kwargs.get("user_id") or "unknown_user")
        task_doc = kwargs.get("task_doc") if isinstance(kwargs.get("task_doc"), dict) else {}
        task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
        workflow = kwargs.get("workflow") if isinstance(kwargs.get("workflow"), dict) else {}
        task_id = str(task.get("task_id") or workflow.get("task_id") or "unknown_task")
        assert _STATIC_PAG_PROFILE_LOADER is not None
        initial_profile = _STATIC_PAG_PROFILE_LOADER(user_id=user_id, task_id=task_id)
        state = _AgenticTaskState(
            user_id=user_id,
            task_id=task_id,
            run_id=str(kwargs.get("run_id") or ""),
            pass_n=int(kwargs.get("pass_n") or 1),
            pass_index=int(kwargs.get("pass_index") or 1),
            detail_path=kwargs.get("llm_attempt_detail_path_for_user"),
            initial_profile=initial_profile,
            current_profile=initial_profile,
        )
        _TASK_STATE.value = state
        try:
            record = original_run_task(
                **{**kwargs, "preference_mode": eval_pag.PAG_MODE}
            )
            return _align_agentic_record_metadata(record, state)
        finally:
            if hasattr(_TASK_STATE, "value"):
                delattr(_TASK_STATE, "value")

    eval_core.build_common_context = build_common_context_with_agentic_state
    eval_core.initial_plan_messages = initial_plan_messages_with_agentic_profile
    eval_core.action_messages = action_messages_with_agentic_profile
    eval_core.update_plan_messages = update_plan_messages_with_agentic_profile
    eval_core.request_structured_json = request_structured_json_with_agentic_memory
    eval_core.offline_judge_entrypoint_for_mode = offline_judge_entrypoint_for_agentic
    eval_core.run_task = run_task_with_agentic_memory
    setattr(eval_core, "_AGENTIC_PAG_ALIGNED_BEHAVIOR_INSTALLED", True)


def configure_evaluation(task_difficulty: str, turns: int) -> None:
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    eval_pag._configure_evaluation(difficulty, turns)

    # 先继承 PAG 的完整配置流程，再应用本文件顶部可调值。
    eval_pag.eval_rag.RAG_TOP_K = PAG_RETRIEVAL_TOP_K
    eval_pag.eval_rag.RAG_PROMPT_MAX_CHARS = RAG_PROMPT_MAX_CHARS
    eval_pag.eval_rag.QINGYUN_EVAL_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY
    eval_core.USER_ID_FILTER = list(USER_ID_FILTER)
    eval_core.TASK_ID_FILTER = list(TASK_ID_FILTER)
    eval_core.TASK_DIFFICULTY = difficulty
    eval_core.PASS_N = PASS_N
    eval_core.MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_PER_TASK
    eval_core.PAG_RETRIEVAL_TOP_K = PAG_RETRIEVAL_TOP_K
    eval_core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
        FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
    )
    eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = SKIP_TASKS_ALREADY_IN_RESULT_FILE
    eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = (
        not SKIP_TASKS_ALREADY_IN_RESULT_FILE
    )
    eval_core.AUTO_TUNE_CONCURRENCY = False
    eval_core.DEFAULT_EVAL_CONCURRENCY = DEFAULT_EVAL_CONCURRENCY
    eval_core.DEEPSEEK_EVAL_CONCURRENCY = DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[
        difficulty
    ]
    eval_core.DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY = dict(
        DEEPSEEK_CONCURRENCY_BY_DIFFICULTY
    )
    eval_core.QINGYUN_DEEPSEEK_EVAL_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY
    eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    eval_core.LONG_CONTEXT_TURNS = turns
    eval_core.PREFERENCE_INJECTION_MODE = AGENTIC_MODE
    eval_core.PREFERENCE_INJECTION_LABELS[AGENTIC_MODE] = AGENTIC_LABEL


def main() -> None:
    _install_agentic_behavior()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
        for turns in turns_list:
            configure_evaluation(task_difficulty, turns)
            eval_core.main()


if __name__ == "__main__":
    main()
