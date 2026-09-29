#!/usr/bin/env python3
"""Generate PAG user abstracts from each user's hidden_info.json with GEN_LLM."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_core


# =========================
# 用户长期可调生成超参数
# =========================

# 为空表示处理 data/ 下所有 user_xxx；例如 ["user_001", "user_019"]。
USER_ID_FILTER: list[str] = []

# 为空表示每个用户使用全部 task_xxx；当前 benchmark 为每个用户 1 个 task。
TASK_ID_FILTER: list[str] = []

# 正式生成时覆盖 eval/pag/abstract.json 和 eval/pag/abstract_details.jsonl。
OVERWRITE_EXISTING_ABSTRACT = True

# GEN_LLM 生成参数。
MODEL_TEMPERATURE = 0.1
MODEL_MAX_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 120
JSON_REPAIR_ATTEMPTS = 2

# 摘要 prompt 裁剪上限。hidden_info 当前约 20 条 QA，默认不会触发裁剪。
MAX_CONTEXT_CHARS = 60000

# 并发默认值；DeepSeek 官方 deepseek-v4-flash/deepseek-v4-pro 按项目约定采用 50。
DEFAULT_CONCURRENCY = 8
DEEPSEEK_CONCURRENCY = 50
QINGYUN_CONCURRENCY = 10

# 输出路径。
PAG_DIR = PROJECT_ROOT / "eval" / "pag"
OUTPUT_PATH = PAG_DIR / "abstract.json"
DETAIL_PATH = PAG_DIR / "abstract_details.jsonl"
LOG_DIR = PROJECT_ROOT / "logs"


DETAIL_LOCK = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with DETAIL_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def compact_json(value: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    if len(text) <= limit:
        return text
    return text[: limit - 200] + f"...<已裁剪 {len(text) - limit + 200} 字符>"


def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppeval_pag_abstract")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_DIR / f"pag_abstract_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def selected_user_dirs() -> list[Path]:
    selected = set(USER_ID_FILTER)
    dirs = [
        path
        for path in sorted((PROJECT_ROOT / "data").glob("user_*"))
        if path.is_dir() and (not selected or path.name in selected)
    ]
    if not dirs:
        raise RuntimeError(f"没有找到待生成摘要的用户目录，USER_ID_FILTER={USER_ID_FILTER}")
    return dirs


def task_id_from_doc(task_doc: dict[str, Any], task_path: Path) -> str:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    task_id = task.get("task_id") if isinstance(task, dict) else None
    return str(task_id or task_path.stem)


def selected_tasks_for_user(user_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    selected = set(TASK_ID_FILTER)
    tasks: list[tuple[str, Path, dict[str, Any]]] = []
    for task_path in sorted(user_dir.glob("task_*.json")):
        task_doc = read_json(task_path)
        task_id = task_id_from_doc(task_doc, task_path)
        if selected and task_id not in selected:
            continue
        tasks.append((task_id, task_path, task_doc))
    if not tasks:
        raise RuntimeError(f"{user_dir.name} 没有匹配 TASK_ID_FILTER={TASK_ID_FILTER} 的 task 文件。")
    return tasks


def abstract_schema() -> dict[str, Any]:
    return {
        "task_related_user_abstract": "string，150-260字；只总结 hidden_info 中能支持且与当前 task 执行相关的用户偏好",
        "preference_points": ["3-8条短句；任务执行中必须遵守的个性化偏好"],
        "constraints_to_preserve": ["2-6条短句；不能违反的边界、风险或格式约束"],
        "tool_call_implications": ["2-6条短句；这些偏好如何影响工具选择、检索、筛选、排序或 JSON 参数"],
        "risk_warnings": ["0-5条短句；从 hidden_info 推断出的潜在失败风险"],
        "language_and_format_preferences": ["0-5条短句；语言、文体、报告结构或交付格式偏好"],
    }


def build_messages(*, user_id: str, task_id: str, task_doc: dict[str, Any], hidden_info: dict[str, Any]) -> list[dict[str, str]]:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    payload = {
        "user_id": user_id,
        "task_id": task_id,
        "task": task,
        "hidden_info": {
            "interaction_count": hidden_info.get("interaction_count"),
            "composition": hidden_info.get("composition"),
            "interactions": hidden_info.get("interactions", []),
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "你是 PPEval 的 PAG 用户摘要生成器。"
                "你只能基于 hidden_info 中的历史 QA 总结用户偏好，不能使用或猜测外部用户画像。"
                "摘要必须服务于当前 task 的 planning 和工具调用参数生成。"
                "只输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请从 hidden_info 中提炼当前 task 相关的用户摘要。"
                "不要保留来源信息，不要逐条复述 QA，不要编造 hidden_info 没有支持的偏好。"
                "如果某项偏好只弱相关，请写成风险或倾向，不要写成硬约束。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(abstract_schema(), 5000)}\n\n"
                f"输入数据:\n{compact_json(payload, MAX_CONTEXT_CHARS)}"
            ),
        },
    ]


def validate_abstract_json(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    if not isinstance(obj.get("task_related_user_abstract"), str) or not obj.get("task_related_user_abstract"):
        errors.append("task_related_user_abstract 必须是非空字符串。")
    for key in [
        "preference_points",
        "constraints_to_preserve",
        "tool_call_implications",
        "risk_warnings",
        "language_and_format_preferences",
    ]:
        value = obj.get(key)
        if not isinstance(value, list):
            errors.append(f"{key} 必须是数组。")
            continue
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{key}[{idx}] 必须是非空字符串。")
    return errors


def compact_attempt_for_abstract(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair_index": attempt.get("repair_index"),
        "purpose": attempt.get("purpose"),
        "ok": attempt.get("ok"),
        "parse_or_validation_error": attempt.get("parse_or_validation_error"),
        "usage": attempt.get("usage"),
        "http_status": attempt.get("http_status"),
        "elapsed_seconds": attempt.get("elapsed_seconds"),
        "api_attempt_count": attempt.get("api_attempt_count"),
        "content": attempt.get("content"),
        "parsed": attempt.get("parsed"),
        "raw_response": attempt.get("raw_response"),
    }


def task_title(task_doc: dict[str, Any]) -> str:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    title = task.get("title") if isinstance(task, dict) else ""
    return str(title or "")


def generate_task_abstract(
    *,
    client: eval_core.LLMClient,
    logger: logging.Logger,
    user_id: str,
    task_id: str,
    task_doc: dict[str, Any],
    hidden_info: dict[str, Any],
) -> dict[str, Any]:
    parsed, attempts = eval_core.request_structured_json(
        client,
        messages=build_messages(user_id=user_id, task_id=task_id, task_doc=task_doc, hidden_info=hidden_info),
        purpose=f"{user_id}/{task_id}/pag_abstract",
        validator=validate_abstract_json,
        logger=logger,
        max_tokens=MODEL_MAX_TOKENS,
        temperature=MODEL_TEMPERATURE,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    final_attempt = attempts[-1] if attempts else {}
    record = {
        "user_id": user_id,
        "task_id": task_id,
        "task_title": task_title(task_doc),
        **parsed,
        "generation": {
            "ok": True,
            "attempt_count": len(attempts),
            "usage": final_attempt.get("usage"),
            "elapsed_seconds": final_attempt.get("elapsed_seconds"),
            "api_attempt_count": final_attempt.get("api_attempt_count"),
        },
    }
    detail_record = {
        "created_at": utc_now_iso(),
        "user_id": user_id,
        "task_id": task_id,
        "task_title": task_title(task_doc),
        "attempts": [compact_attempt_for_abstract(attempt) for attempt in attempts],
    }
    write_jsonl(DETAIL_PATH, detail_record)
    return record


def generate_user_entry(
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
    task_summaries: dict[str, Any] = {}
    for task_id, _, task_doc in selected_tasks_for_user(user_dir):
        task_summaries[task_id] = generate_task_abstract(
            client=client,
            logger=logger,
            user_id=user_id,
            task_id=task_id,
            task_doc=task_doc,
            hidden_info=hidden_info,
        )
    first_task_id = sorted(task_summaries)[0]
    first_summary = task_summaries[first_task_id]
    return {
        "user_id": user_id,
        "task_id": first_task_id,
        "task_title": first_summary.get("task_title"),
        "summary": first_summary.get("task_related_user_abstract"),
        "task_summaries": task_summaries,
    }


def concurrency_for_api(api_config: Any) -> int:
    gen_url = str(getattr(api_config, "GEN_URL"))
    gen_model = str(getattr(api_config, "GEN_MODEL"))
    if gen_url == "https://api.deepseek.com" and gen_model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        return DEEPSEEK_CONCURRENCY
    if gen_url == "https://api.qingyuntop.top/v1":
        return QINGYUN_CONCURRENCY
    return DEFAULT_CONCURRENCY


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)
    eval_core.MODEL_TEMPERATURE = MODEL_TEMPERATURE
    eval_core.MODEL_MAX_TOKENS = MODEL_MAX_TOKENS
    eval_core.REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT_SECONDS
    eval_core.JSON_REPAIR_ATTEMPTS = JSON_REPAIR_ATTEMPTS

    PAG_DIR.mkdir(parents=True, exist_ok=True)
    if OVERWRITE_EXISTING_ABSTRACT:
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        if DETAIL_PATH.exists():
            DETAIL_PATH.unlink()
    elif OUTPUT_PATH.exists():
        raise RuntimeError(f"摘要文件已存在且 OVERWRITE_EXISTING_ABSTRACT=False: {OUTPUT_PATH}")

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
        "开始生成 PAG abstract：run_id=%s model=%s base_url=%s key=%s users=%s concurrency=%s output=%s",
        run_id,
        client.model,
        client.base_url,
        eval_core.mask_secret(client.api_key),
        len(user_dirs),
        concurrency,
        OUTPUT_PATH,
    )

    started = time.perf_counter()
    users: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="pag_abs") as pool:
        future_map = {
            pool.submit(generate_user_entry, user_dir=user_dir, client=client, logger=logger): user_dir
            for user_dir in user_dirs
        }
        for future in as_completed(future_map):
            user_dir = future_map[future]
            try:
                entry = future.result()
                users[str(entry["user_id"])] = entry
            except Exception as exc:  # noqa: BLE001 - preserve per-user generation failures.
                failures.append(
                    {
                        "user_id": user_dir.name,
                        "message": str(exc),
                    }
                )
                logger.exception("用户摘要生成失败：user=%s error=%s", user_dir.name, exc)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(user_dirs):
                logger.info("PAG abstract 进度：%s/%s failures=%s", completed, len(user_dirs), len(failures))

    elapsed = time.perf_counter() - started
    output = {
        "schema_version": "pag.abstract.v1",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "model": {
            "name": client.model,
            "base_url": client.base_url,
            "api_key": eval_core.mask_secret(client.api_key),
            "temperature": MODEL_TEMPERATURE,
            "max_tokens": MODEL_MAX_TOKENS,
        },
        "hyperparameters": {
            "user_id_filter": USER_ID_FILTER,
            "task_id_filter": TASK_ID_FILTER,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "json_repair_attempts": JSON_REPAIR_ATTEMPTS,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "concurrency": concurrency,
        },
        "outputs": {
            "abstract_json": str(OUTPUT_PATH),
            "details_jsonl": str(DETAIL_PATH),
            "log": str(LOG_DIR / f"pag_abstract_{run_id}.log"),
        },
        "user_count": len(users),
        "failure_count": len(failures),
        "failures": failures,
        "users": {user_id: users[user_id] for user_id in sorted(users)},
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "PAG abstract 完成：elapsed=%.2fs users=%s failures=%s output=%s details=%s",
        elapsed,
        len(users),
        len(failures),
        OUTPUT_PATH,
        DETAIL_PATH,
    )
    if failures:
        raise RuntimeError(f"PAG abstract 生成存在失败用户 {len(failures)} 个，详情见 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
