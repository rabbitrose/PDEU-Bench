#!/usr/bin/env python3
"""Build BANK user memories, then run planning-agent evaluation with task-aware RAG."""

from __future__ import annotations

import json
import logging
import copy
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
PAG_DIR = EVAL_DIR / "pag"
RAG_DIR = EVAL_DIR / "rag"
for module_dir in (EVAL_DIR, PAG_DIR, RAG_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import eval_core
import eval_pag


# =========================
# 用户长期可调 BANK 超参数
# =========================

# 除 BANK summary memory 的构建与检索文档内容外，全部评测配置继承 PAG。
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
BANK_RETRIEVAL_TOP_K = eval_pag.PAG_RETRIEVAL_TOP_K
RAG_PROMPT_MAX_CHARS = eval_pag.RAG_PROMPT_MAX_CHARS
DEEPSEEK_CONCURRENCY_BY_DIFFICULTY = dict(eval_pag.DEEPSEEK_CONCURRENCY_BY_DIFFICULTY)
DEFAULT_EVAL_CONCURRENCY = eval_pag.DEFAULT_EVAL_CONCURRENCY
QINGYUN_EVAL_CONCURRENCY = eval_pag.QINGYUN_EVAL_CONCURRENCY

# 一次正式运行是否先构建 BANK memory，以及是否继续运行被测 LLM eval。
BUILD_BANK_MEMORY = False
RUN_EVALUATION = True

# 构建时覆盖原有 bank_memory.json 与生成明细；不使用 V1/V2 文件迭代。
OVERWRITE_EXISTING_BANK_MEMORY = True

# BANK 记忆生成模型参数。
BANK_MODEL_TEMPERATURE = 0.1
BANK_MODEL_MAX_TOKENS = 8192
BANK_REQUEST_TIMEOUT_SECONDS = 120
BANK_JSON_REPAIR_ATTEMPTS = 1
BANK_MAX_CONTEXT_CHARS = 100000

# 被测 LLM 的生成参数。记忆构建结束后显式恢复为这些值。
EVAL_MODEL_TEMPERATURE = eval_core.MODEL_TEMPERATURE
EVAL_MODEL_MAX_TOKENS = eval_core.MODEL_MAX_TOKENS
EVAL_REQUEST_TIMEOUT_SECONDS = eval_core.REQUEST_TIMEOUT_SECONDS
EVAL_JSON_REPAIR_ATTEMPTS = eval_core.JSON_REPAIR_ATTEMPTS

# BANK summary 预生成使用独立并发；正式 eval 并发完全继承 PAG。
DEFAULT_BANK_BUILD_CONCURRENCY = 8
QINGYUN_BANK_BUILD_CONCURRENCY = QINGYUN_EVAL_CONCURRENCY


BANK_DIR = PROJECT_ROOT / "eval" / "bank"
BANK_MEMORY_PATH = BANK_DIR / "bank_memory.json"
BANK_DETAIL_PATH = BANK_DIR / "bank_memory_details.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"
DETAIL_LOCK = threading.Lock()

BANK_MODE = "bank"
BANK_LABEL = "PAG Top-3 统一检索原始 QA 与独立 summary memory pieces + User abstract profile"
BANK_SUMMARY_ROLE = "summary"
BANK_SUMMARY_PIECE_TYPE = "summary"
BANK_SUMMARY_ID_SUFFIX = "__summary"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_detail(record: dict[str, Any]) -> None:
    with DETAIL_LOCK:
        BANK_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BANK_DETAIL_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def compact_json(value: Any, limit: int = BANK_MAX_CONTEXT_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    if len(text) <= limit:
        return text
    return text[: limit - 200] + f"...<已裁剪 {len(text) - limit + 200} 字符>"


def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppeval_bank")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_DIR / f"bank_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def selected_user_dirs() -> list[Path]:
    selected = set(USER_ID_FILTER)
    selected_task_ids = set(TASK_ID_FILTER)
    difficulties = TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]
    for raw_difficulty in difficulties:
        difficulty = eval_core.normalize_task_difficulty(raw_difficulty)
        if difficulty != "all":
            selected_task_ids.add(eval_core.TASK_ID_BY_DIFFICULTY[difficulty])

    def has_selected_task(user_dir: Path) -> bool:
        if not selected_task_ids:
            return any(user_dir.glob("task_*.json"))
        for task_path in user_dir.glob("task_*.json"):
            task_doc = read_json(task_path)
            if eval_core.task_id_from_doc(task_doc, task_path) in selected_task_ids:
                return True
        return False

    user_dirs = [
        path
        for path in sorted((PROJECT_ROOT / "data").glob("user_*"))
        if path.is_dir() and (not selected or path.name in selected) and has_selected_task(path)
    ]
    if not user_dirs:
        raise RuntimeError(f"没有找到待处理用户，USER_ID_FILTER={USER_ID_FILTER}")
    return user_dirs


def qa_from_interaction(interaction: dict[str, Any], index: int) -> dict[str, str]:
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    for turn in interaction.get("turns", []) if isinstance(interaction.get("turns"), list) else []:
        if not isinstance(turn, dict) or not isinstance(turn.get("content"), str):
            continue
        if turn.get("role") == "user":
            user_turns.append(turn["content"])
        elif turn.get("role") == "assistant":
            assistant_turns.append(turn["content"])
    interaction_id = str(interaction.get("interaction_id") or f"interaction_{index:03d}")
    if not user_turns or not assistant_turns:
        raise RuntimeError(f"{interaction_id} 必须至少包含一个 user turn 和一个 assistant turn。")
    return {
        "interaction_id": interaction_id,
        "question": "\n".join(user_turns),
        "answer": "\n".join(assistant_turns),
    }


def hidden_qa_records(hidden_info: dict[str, Any]) -> list[dict[str, str]]:
    interactions = hidden_info.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        raise RuntimeError("hidden_info.interactions 必须是非空数组。")
    records = [qa_from_interaction(item, index) for index, item in enumerate(interactions, start=1) if isinstance(item, dict)]
    ids = [record["interaction_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("hidden_info 中 interaction_id 不唯一。")
    return records


def qa_summary_schema(qa_ids: list[str]) -> dict[str, Any]:
    return {
        "qa_summaries": [
            {
                "interaction_id": f"必须来自输入，完整覆盖且不重复；候选为 {qa_ids}",
                "summary": "对这一 QA 的独立简明摘要",
            }
        ]
    }


def validate_qa_summaries(expected_ids: list[str]) -> Callable[[Any], list[str]]:
    expected = set(expected_ids)

    def validator(obj: Any) -> list[str]:
        if not isinstance(obj, dict):
            return ["顶层必须是 JSON object。"]
        rows = obj.get("qa_summaries")
        if not isinstance(rows, list):
            return ["qa_summaries 必须是数组。"]
        errors: list[str] = []
        actual_ids: list[str] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                errors.append(f"qa_summaries[{index}] 必须是 object。")
                continue
            interaction_id = row.get("interaction_id")
            if not isinstance(interaction_id, str):
                errors.append(f"qa_summaries[{index}].interaction_id 必须是字符串。")
            else:
                actual_ids.append(interaction_id)
            if not isinstance(row.get("summary"), str) or not row.get("summary", "").strip():
                errors.append(f"qa_summaries[{index}].summary 必须是非空字符串。")
        if set(actual_ids) != expected or len(actual_ids) != len(expected_ids):
            errors.append(f"interaction_id 必须完整且不重复，expected={expected_ids}, actual={actual_ids}")
        return errors

    return validator


def qa_summary_messages(user_id: str, qa_records: list[dict[str, str]]) -> list[dict[str, str]]:
    qa_ids = [record["interaction_id"] for record in qa_records]
    payload = {"user_id": user_id, "qa_pairs": qa_records}
    return [
        {
            "role": "system",
            "content": (
                "你是 PPEval BANK 的 QA memory 构建器。需要为每一条输入 QA 独立生成 summary。"
                "只输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "逐条总结全部 QA，保持 interaction_id 一一对应、完整覆盖且不重复。"
                "不得合并不同 QA，不得添加 source 字段，不得编造输入未支持的信息。\n\n"
                f"JSON schema:\n{compact_json(qa_summary_schema(qa_ids), 12000)}\n\n"
                f"输入:\n{compact_json(payload)}"
            ),
        },
    ]


def request_bank_json(
    *,
    client: eval_core.LLMClient,
    logger: logging.Logger,
    messages: list[dict[str, str]],
    purpose: str,
    validator: Callable[[Any], list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed, attempts = eval_core.request_structured_json(
        client,
        messages=messages,
        purpose=purpose,
        validator=validator,
        logger=logger,
        max_tokens=BANK_MODEL_MAX_TOKENS,
        temperature=BANK_MODEL_TEMPERATURE,
        timeout=BANK_REQUEST_TIMEOUT_SECONDS,
    )
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{purpose} 未返回 JSON object。")
    return parsed, attempts


def build_user_memory(
    *,
    user_dir: Path,
    client: eval_core.LLMClient,
    logger: logging.Logger,
) -> dict[str, Any]:
    user_id = user_dir.name
    hidden_path = user_dir / "hidden_info.json"
    if not hidden_path.exists():
        raise RuntimeError(f"缺少 hidden_info.json: {hidden_path}")
    hidden_info = read_json(hidden_path)
    qa_records = hidden_qa_records(hidden_info)

    summary_prompt = qa_summary_messages(user_id, qa_records)
    summary_attempts: list[dict[str, Any]] = []
    try:
        summaries, summary_attempts = request_bank_json(
            client=client,
            logger=logger,
            messages=summary_prompt,
            purpose=f"{user_id}/bank_qa_summaries",
            validator=validate_qa_summaries([record["interaction_id"] for record in qa_records]),
        )
    except Exception as exc:
        summary_attempts = list(getattr(exc, "attempts", []))
        write_detail(
            {
                "created_at": utc_now_iso(),
                "user_id": user_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "qa_summary_generation": {"messages": summary_prompt, "attempts": summary_attempts},
            }
        )
        raise
    summary_by_id = {row["interaction_id"]: row for row in summaries["qa_summaries"]}
    qa_memories = [
        {
            **record,
            "summary": summary_by_id[record["interaction_id"]]["summary"],
        }
        for record in qa_records
    ]
    write_detail(
        {
            "created_at": utc_now_iso(),
            "user_id": user_id,
            "status": "completed",
            "qa_summary_generation": {
                "messages": summary_prompt,
                "attempts": summary_attempts,
            },
        }
    )
    return {
        "user_id": user_id,
        "qa_count": len(qa_memories),
        "qa_memories": qa_memories,
        "generation": {
            "qa_summary_attempt_count": len(summary_attempts),
            "qa_summary_usage": summary_attempts[-1].get("usage") if summary_attempts else {},
        },
    }


def concurrency_for_api(api_config: Any) -> int:
    base_url = str(getattr(api_config, "GEN_URL"))
    model = str(getattr(api_config, "GEN_MODEL"))
    if base_url == "https://api.deepseek.com" and model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        difficulties = [
            eval_core.normalize_task_difficulty(value)
            for value in (TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY])
        ]
        return min(DEEPSEEK_CONCURRENCY_BY_DIFFICULTY[value] for value in difficulties)
    if base_url == "https://api.qingyuntop.top/v1":
        return QINGYUN_BANK_BUILD_CONCURRENCY
    return DEFAULT_BANK_BUILD_CONCURRENCY


def build_bank_memory(logger: logging.Logger, run_id: str) -> None:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    if OVERWRITE_EXISTING_BANK_MEMORY:
        BANK_MEMORY_PATH.unlink(missing_ok=True)
        BANK_DETAIL_PATH.unlink(missing_ok=True)
    elif BANK_MEMORY_PATH.exists():
        logger.info("复用已有 BANK memory：%s", BANK_MEMORY_PATH)
        return

    eval_core.MODEL_TEMPERATURE = BANK_MODEL_TEMPERATURE
    eval_core.MODEL_MAX_TOKENS = BANK_MODEL_MAX_TOKENS
    eval_core.REQUEST_TIMEOUT_SECONDS = BANK_REQUEST_TIMEOUT_SECONDS
    eval_core.JSON_REPAIR_ATTEMPTS = BANK_JSON_REPAIR_ATTEMPTS
    api_config = eval_core.load_api_config()
    client = eval_core.LLMClient(
        api_key=getattr(api_config, "GEN_KEY"),
        base_url=getattr(api_config, "GEN_URL"),
        model=getattr(api_config, "GEN_MODEL"),
        logger=logger,
    )
    concurrency = concurrency_for_api(api_config)
    user_dirs = selected_user_dirs()
    logger.info(
        "开始构建 BANK memory：run_id=%s model=%s base_url=%s key=%s users=%s concurrency=%s",
        run_id,
        client.model,
        client.base_url,
        eval_core.mask_secret(client.api_key),
        len(user_dirs),
        concurrency,
    )

    started = time.perf_counter()
    users: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="bank_build") as pool:
        future_map = {
            pool.submit(build_user_memory, user_dir=user_dir, client=client, logger=logger): user_dir
            for user_dir in user_dirs
        }
        for future in as_completed(future_map):
            user_dir = future_map[future]
            try:
                entry = future.result()
                users[entry["user_id"]] = entry
            except Exception as exc:  # noqa: BLE001 - preserve all per-user failures in the artifact and log.
                failures.append({"user_id": user_dir.name, "message": str(exc)})
                logger.exception("BANK memory 构建失败：user=%s error=%s", user_dir.name, exc)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(user_dirs):
                logger.info("BANK memory 进度：%s/%s failures=%s", completed, len(user_dirs), len(failures))

    output = {
        "schema_version": "bank.memory.v1",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "elapsed_seconds": time.perf_counter() - started,
        "model": {
            "name": client.model,
            "base_url": client.base_url,
            "api_key": eval_core.mask_secret(client.api_key),
            "temperature": BANK_MODEL_TEMPERATURE,
            "max_tokens": BANK_MODEL_MAX_TOKENS,
        },
        "hyperparameters": {
            "user_id_filter": USER_ID_FILTER,
            "task_difficulty": TASK_DIFFICULTY,
            "task_difficulty_list": TASK_DIFFICULTY_LIST,
            "request_timeout_seconds": BANK_REQUEST_TIMEOUT_SECONDS,
            "json_repair_attempts": BANK_JSON_REPAIR_ATTEMPTS,
            "max_context_chars": BANK_MAX_CONTEXT_CHARS,
            "concurrency": concurrency,
        },
        "user_count": len(users),
        "failure_count": len(failures),
        "failures": failures,
        "users": {user_id: users[user_id] for user_id in sorted(users)},
    }
    write_json_atomic(BANK_MEMORY_PATH, output)
    eval_core._BANK_MEMORY_CACHE = None
    logger.info(
        "BANK memory 完成：users=%s failures=%s memory=%s details=%s",
        len(users),
        len(failures),
        BANK_MEMORY_PATH,
        BANK_DETAIL_PATH,
    )
    if failures:
        raise RuntimeError(f"BANK memory 存在 {len(failures)} 个失败用户，停止 eval；详情见 {BANK_MEMORY_PATH}")


def _bank_memory_by_interaction_id(user_id: str) -> dict[str, dict[str, Any]]:
    document = eval_core.load_bank_memory_doc()
    users = document.get("users") if isinstance(document.get("users"), dict) else {}
    user_memory = users.get(user_id) if isinstance(users.get(user_id), dict) else None
    if not isinstance(user_memory, dict):
        raise RuntimeError(f"BANK memory 缺少用户 {user_id}: {BANK_MEMORY_PATH}")
    rows = user_memory.get("qa_memories")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"BANK memory 用户 {user_id} 的 qa_memories 必须是非空数组。")

    memories: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"BANK memory {user_id}.qa_memories[{index}] 必须是 object。")
        interaction_id = row.get("interaction_id")
        summary = row.get("summary")
        if not isinstance(interaction_id, str) or not interaction_id:
            raise RuntimeError(f"BANK memory {user_id}.qa_memories[{index}] 缺少 interaction_id。")
        if interaction_id in memories:
            raise RuntimeError(f"BANK memory 用户 {user_id} 的 interaction_id 重复: {interaction_id}")
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError(f"BANK memory 用户 {user_id} 的 summary 为空: {interaction_id}")
        memories[interaction_id] = row
    return memories


def _hidden_info_with_separate_bank_memories(
    *,
    user_id: str,
    hidden_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(hidden_info, dict):
        raise RuntimeError("BANK 模式需要 hidden_info object。")
    interactions = hidden_info.get("interactions")
    if not isinstance(interactions, list) or not interactions:
        raise RuntimeError("BANK 模式需要非空 hidden_info.interactions。")

    for index, interaction in enumerate(interactions, start=1):
        if not isinstance(interaction, dict):
            raise RuntimeError(f"hidden_info 用户 {user_id} 包含非 object interaction: index={index}")
        interaction_id = interaction.get("interaction_id")
        if not isinstance(interaction_id, str) or not interaction_id.strip():
            raise RuntimeError(
                f"BANK 统一检索池要求原始 interaction 具有非空 interaction_id: "
                f"user={user_id} index={index}"
            )

    memory_by_id = _bank_memory_by_interaction_id(user_id)
    current_records = [
        qa_from_interaction(interaction, index)
        for index, interaction in enumerate(interactions, start=1)
    ]
    current_ids = [record["interaction_id"] for record in current_records]
    if len(current_ids) != len(set(current_ids)):
        raise RuntimeError(f"hidden_info 用户 {user_id} 的 interaction_id 不唯一。")

    current_id_set = set(current_ids)
    memory_id_set = set(memory_by_id)
    if current_id_set != memory_id_set:
        missing = sorted(current_id_set - memory_id_set)
        stale = sorted(memory_id_set - current_id_set)
        raise RuntimeError(
            f"BANK memory 与当前 hidden_info 不一致: user={user_id} "
            f"missing_summary_ids={missing[:10]} stale_memory_ids={stale[:10]}。"
            "请先设置 BUILD_BANK_MEMORY=True 重新生成 bank_memory.json。"
        )

    enriched = copy.deepcopy(hidden_info)
    raw_interactions = enriched.get("interactions")
    assert isinstance(raw_interactions, list)
    summary_interactions: list[dict[str, Any]] = []
    original_ids = set(current_ids)
    for index, (interaction, current_record) in enumerate(
        zip(raw_interactions, current_records),
        start=1,
    ):
        assert isinstance(interaction, dict)
        interaction_id = current_record["interaction_id"]
        memory = memory_by_id[interaction_id]
        for key in ("question", "answer"):
            stored = memory.get(key)
            if isinstance(stored, str) and stored != current_record[key]:
                raise RuntimeError(
                    f"BANK memory 与当前 hidden_info 内容不一致: "
                    f"user={user_id} interaction={interaction_id} field={key}。"
                    "请重新生成 bank_memory.json。"
                )
        turns = interaction.get("turns")
        if not isinstance(turns, list):
            raise RuntimeError(f"hidden_info interaction 缺少 turns: user={user_id} index={index}")
        summary_id = f"{interaction_id}{BANK_SUMMARY_ID_SUFFIX}"
        if summary_id in original_ids:
            raise RuntimeError(
                f"BANK summary memory interaction_id 与原始记录冲突: "
                f"user={user_id} interaction_id={summary_id}"
            )
        summary_interactions.append(
            {
                "interaction_id": summary_id,
                # 这两个审计字段不参与 PAG 的 BM25 文档构造；摘要候选只索引 summary 文本。
                "memory_piece_type": BANK_SUMMARY_PIECE_TYPE,
                "source_interaction_id": interaction_id,
                "turns": [
                    {
                        "role": BANK_SUMMARY_ROLE,
                        "content": str(memory["summary"]).strip(),
                    }
                ],
            }
        )
    enriched["interactions"] = raw_interactions + summary_interactions
    return enriched


def validate_bank_memory_for_evaluation(logger: logging.Logger) -> None:
    user_dirs = selected_user_dirs()
    for user_dir in user_dirs:
        hidden_path = user_dir / "hidden_info.json"
        if not hidden_path.exists():
            raise RuntimeError(f"缺少 hidden_info.json: {hidden_path}")
        _hidden_info_with_separate_bank_memories(
            user_id=user_dir.name,
            hidden_info=read_json(hidden_path),
        )
    logger.info("BANK memory 与当前 hidden_info 一致：users=%s", len(user_dirs))


def _align_bank_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    record["preference_injection_mode"] = BANK_MODE
    record["preference_injection_label"] = BANK_LABEL
    provided = record.get("provided_eval_context")
    if isinstance(provided, dict):
        provided["mode"] = BANK_MODE
    judge_results = record.get("judge_results")
    if isinstance(judge_results, dict):
        judge_results["offline_judge_entrypoint"] = "python eval/bank/judge_bank.py"
    hyperparameters = record.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        hyperparameters.pop("pag_difference_from_rag", None)
        hyperparameters["bank_retrieval_top_k"] = BANK_RETRIEVAL_TOP_K
        hyperparameters["rag_top_k"] = BANK_RETRIEVAL_TOP_K
        hyperparameters["rag_retrieval_method"] = (
            "pag_bm25_over_unified_raw_qa_and_independent_summary_memory_pieces"
        )
        hyperparameters["rag_retrieval_injection_phases"] = [
            "initial_plan",
            "reason_action",
            "update_plan",
        ]
        hyperparameters["bank_difference_from_pag"] = (
            "independent_summary_memory_pieces_added_to_pag_candidate_pool_only"
        )
        hyperparameters["rag_prompt_max_chars"] = RAG_PROMPT_MAX_CHARS
    return record


def _register_bank_mode() -> None:
    eval_core.VALID_PREFERENCE_INJECTION_MODES.add(BANK_MODE)
    eval_core.PREFERENCE_INJECTION_LABELS[BANK_MODE] = BANK_LABEL


def _pag_aligned_kwargs_for_bank(kwargs: dict[str, Any]) -> dict[str, Any]:
    user_id = str(kwargs.get("user_id") or "")
    return {
        **kwargs,
        "preference_mode": eval_pag.PAG_MODE,
        "hidden_info": _hidden_info_with_separate_bank_memories(
            user_id=user_id,
            hidden_info=kwargs.get("hidden_info"),
        ),
    }


def _install_bank_behavior() -> None:
    eval_pag._install_pag_behavior()
    _register_bank_mode()
    if getattr(eval_core, "_BANK_PAG_ALIGNED_BEHAVIOR_INSTALLED", False):
        return

    original_run_task = eval_core.run_task

    def run_task_with_bank_summary(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("preference_mode") != BANK_MODE:
            return original_run_task(**kwargs)
        record = original_run_task(**_pag_aligned_kwargs_for_bank(kwargs))
        return _align_bank_record_metadata(record)

    eval_core.run_task = run_task_with_bank_summary
    setattr(eval_core, "_BANK_PAG_ALIGNED_BEHAVIOR_INSTALLED", True)


def configure_evaluation(task_difficulty: str, turns: int) -> None:
    difficulty = eval_core.normalize_task_difficulty(task_difficulty)
    eval_pag._configure_evaluation(difficulty, turns)
    eval_core.PREFERENCE_INJECTION_MODE = BANK_MODE
    eval_core.PREFERENCE_INJECTION_LABELS[BANK_MODE] = BANK_LABEL
    eval_core.BANK_RETRIEVAL_TOP_K = BANK_RETRIEVAL_TOP_K
    eval_core.MODEL_TEMPERATURE = EVAL_MODEL_TEMPERATURE
    eval_core.MODEL_MAX_TOKENS = EVAL_MODEL_MAX_TOKENS
    eval_core.REQUEST_TIMEOUT_SECONDS = EVAL_REQUEST_TIMEOUT_SECONDS
    eval_core.JSON_REPAIR_ATTEMPTS = EVAL_JSON_REPAIR_ATTEMPTS


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)
    _install_bank_behavior()
    if BUILD_BANK_MEMORY:
        build_bank_memory(logger, run_id)
    elif not BANK_MEMORY_PATH.exists():
        raise RuntimeError(f"BUILD_BANK_MEMORY=False，但 BANK memory 不存在: {BANK_MEMORY_PATH}")
    if RUN_EVALUATION:
        validate_bank_memory_for_evaluation(logger)
        turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
        for task_difficulty in TASK_DIFFICULTY_LIST or [TASK_DIFFICULTY]:
            for turns in turns_list:
                configure_evaluation(task_difficulty, turns)
                logger.info(
                    "开始 BANK eval：difficulty=%s long_context_turns=%s",
                    task_difficulty,
                    turns,
                )
                eval_core.main()


if __name__ == "__main__":
    main()
