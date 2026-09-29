#!/usr/bin/env python3
"""Shared planning-agent evaluation engine for personalized methods."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sandbox_database import (
    GlobalSandboxDatabase,
    hard_constraints_compatible,
    materialize_sandbox_response,
)


# =========================
# 用户长期可调评估超参数
# =========================

# 为空表示评估 data/ 下所有 user_xxx；例如 ["user_001", "user_019"]。
USER_ID_FILTER: list[str] = []

# 为空表示评估每个用户全部 task_xxx；例如 ["task_001"]。
TASK_ID_FILTER: list[str] = []

# 任务难度过滤：easy -> task_easy, hard -> task_hard。
# all 仅用于审计或调试；正式评测建议分别运行 easy/hard。
TASK_DIFFICULTY = "hard"
VALID_TASK_DIFFICULTIES = {"easy", "hard", "all"}
TASK_ID_BY_DIFFICULTY = {
    "easy": "task_easy",
    "hard": "task_hard",
}
MAX_TOOL_CALLS_BY_TASK_DIFFICULTY = {
    "easy": 5,
    "hard": 15,
}

# 每个 task_xxx 重复执行多少次；结果写入 runresult/<模型>/<direct|hidden|mix|pag|bank>/pass_<PASS_N>/。
PASS_N = 1

# 每个任务固定 1 次 first_plan，最多执行多少次 action/tool_call。
# 每次 tool_call 后必须执行 1 次 update_plan；第 MAX_TOOL_CALLS_PER_TASK 次 update_plan 后强制 finalization。
MAX_TOOL_CALLS_PER_TASK = 15
TASK_MAX_TOOL_CALLS_OVERRIDE: int | None = None

# 生成模型参数。
MODEL_TEMPERATURE = 0.2
MODEL_MAX_TOKENS = 8192
REQUEST_TIMEOUT_SECONDS = 120
API_RETRY_ATTEMPTS = 3
API_RETRY_BASE_SLEEP_SECONDS = 2.0
JSON_REPAIR_ATTEMPTS = 1
USE_JSON_RESPONSE_FORMAT = True
LOCAL_INFERENCE_CONFIG: dict[str, Any] | None = None
GENERATION_CLIENT_FACTORY: Callable[[Any, logging.Logger], Any] | None = None

# Inline LLM-as-judge。默认关闭，正式评测应先只生成 runresult，再运行 eval/judge_hidden.py 离线 judge。
# 手动改为 True 时，judge 只旁路记录和统计，绝不写入被测 LLM 后续 prompt。
ENABLE_INLINE_JUDGE = False
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 8192

# 工具可用性设置：
# all=向每个任务展示全局全部工具及 schema；
# oracle=使用每个 task 预设的 workflow.available_api_names；
# noisy=在 oracle 子集上加入确定性干扰 API；
# global_retrieval=先让被测 LLM 从全局 API 简表中选择候选 API，再给 schema 进入 first plan。
TOOL_AVAILABILITY_MODE = "all"
VALID_TOOL_AVAILABILITY_MODES = {"all", "oracle", "noisy", "global_retrieval"}
NOISY_SAME_CATEGORY_DISTRACTOR_COUNT = 12
NOISY_CROSS_CATEGORY_DISTRACTOR_COUNT = 12
GLOBAL_RETRIEVAL_MAX_APIS = 30
GLOBAL_RETRIEVAL_MIN_APIS = 8

# 并发控制。AUTO_TUNE_CONCURRENCY=True 时会先探测 API 可接受并发。
AUTO_TUNE_CONCURRENCY = False
DEFAULT_EVAL_CONCURRENCY = 50
DEEPSEEK_EVAL_CONCURRENCY = 60
DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY = {
    "easy": 68,
    "hard": 75,
}
QINGYUN_DEEPSEEK_EVAL_CONCURRENCY = 10
MIMO_V25_PRO_EVAL_CONCURRENCY = 4
CONCURRENCY_CANDIDATES = (1, 2, 4, 8, 12, 16, 20, 40, 80, 120, 150)
CONCURRENCY_PROBE_SUCCESS_THRESHOLD = 0.90
CONCURRENCY_PROBE_TIMEOUT_SECONDS = 60

# 结果与日志控制。
RESULT_DIR_NAME = "runresult"
JUDGE_RESULT_DIR_NAME = "judgeresult"
LOG_DIR_NAME = "logs"
DATA_DIR_NAME = "data"
SANDBOX_DIR_NAME = "SandboxDB"
SAVE_PROMPTS_IN_RESULTS = True
SAVE_RAW_RESPONSES_IN_RESULTS = True
SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR = True
KEEP_FULL_LLM_ATTEMPTS_IN_MAIN_RESULT = False
LLM_ATTEMPT_DETAIL_DIR_NAME = "api_details"
OVERWRITE_SELECTED_USER_RESULTS_ON_START = True
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False

# 提示词上下文裁剪，只影响下一轮提示词；完整轨迹仍会保存到结果。
MAX_PLAN_STEPS_IN_PROMPT = 10
MAX_RECENT_TRAJECTORY_EVENTS_IN_PROMPT = 8
MAX_JSON_CONTEXT_CHARS = 200_000
MAX_TOOL_OBSERVATION_CHARS_IN_PROMPT = 5000
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = False
SANDBOX_OBSERVATION_PROMPT_EXCLUDED_FIELDS = frozenset(
    {
        "transport",
        "latency",
        "latency_ms",
        "rate_limit",
        "request_id",
        "request_echo",
        "selected_inputs",
        "context_effects",
        "raw_response_shape",
    }
)
MAX_JUDGE_CONTEXT_CHARS = 18000
MAX_JUDGE_PLAN_CHARS = 10000
MAX_JUDGE_TRAJECTORY_CHARS = 26000
MAX_JUDGE_OBSERVATION_CHARS = 8000
MAX_JUDGE_REASON_CHARS = 700

# SandBoxDB 模拟执行参数。
SANDBOX_LATENCY_MIN_MS = 80
SANDBOX_LATENCY_MAX_MS = 240

# 用户信息注入方式：direct=用户信息，hidden=用户交互历史，mix=二者同时提供，
# pag=检索QA+用户摘要，bank=用户抽象画像+从 question/answer/summary 记忆库检索的任务相关记录。
PREFERENCE_INJECTION_MODE = "hidden"
PREFERENCE_INJECTION_LABELS = {
    "direct": "显式用户画像",
    "hidden": "隐式交互历史",
    "mix": "显式用户画像 + 隐式交互历史",
    "pag": "PAG 检索QA + 用户摘要",
    "bank": "BANK 用户抽象画像 + QA Summary RAG",
}

# PAG 方法参数。PAG 摘要由 python eval/pag/gen_absract.py 生成。
PAG_RETRIEVAL_TOP_K = 3
PAG_ABSTRACT_FILE_NAME = "abstract.json"

# BANK 方法参数。记忆文件由 python eval/bank/eval_bank.py 在正式评测前生成。
BANK_RETRIEVAL_TOP_K = 3
BANK_MEMORY_FILE_NAME = "bank_memory.json"

# 长上下文控制。0 或 ENABLE_LONG_CONTEXT=False 表示不追加。
# direct 模式只追加 LMSYS 长对话历史，不追加 hidden_info 历史。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_ALLOWED_TURNS = {0, 10, 50, 100}
LONG_CONTEXT_DIR_NAME = "LongContext"
LONG_CONTEXT_FILE_TEMPLATE = "long_context_{turns}.json"
LONG_CONTEXT_MAX_JSON_CONTEXT_CHARS = 240000


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Prompts.eval import plan_act_action, plan_act_initial_plan, plan_act_update_plan
from Prompts.eval.plan_act_system import system_prompt as plan_act_system_prompt
from Prompts.common import concurrency_probe, json_repair

DATA_DIR = PROJECT_ROOT / DATA_DIR_NAME
SANDBOX_DIR = PROJECT_ROOT / SANDBOX_DIR_NAME
LONG_CONTEXT_DIR = DATA_DIR / LONG_CONTEXT_DIR_NAME
PAG_DIR = PROJECT_ROOT / "eval" / "pag"
PAG_ABSTRACT_PATH = PAG_DIR / PAG_ABSTRACT_FILE_NAME
BANK_DIR = PROJECT_ROOT / "eval" / "bank"
BANK_MEMORY_PATH = BANK_DIR / BANK_MEMORY_FILE_NAME
RESULT_BASE_DIR = PROJECT_ROOT / RESULT_DIR_NAME
RESULT_DIR = RESULT_BASE_DIR
JUDGE_RESULT_BASE_DIR = PROJECT_ROOT / JUDGE_RESULT_DIR_NAME
JUDGE_RESULT_DIR = JUDGE_RESULT_BASE_DIR
LOG_DIR = PROJECT_ROOT / LOG_DIR_NAME
VALID_PREFERENCE_INJECTION_MODES = {"direct", "hidden", "mix", "pag", "bank"}
_LONG_CONTEXT_CACHE: dict[int, dict[str, Any]] = {}
_LONG_CONTEXT_CACHE_LOCK = threading.Lock()
_PAG_ABSTRACT_CACHE: dict[str, Any] | None = None
_PAG_ABSTRACT_CACHE_LOCK = threading.Lock()
_BANK_MEMORY_CACHE: dict[str, Any] | None = None
_BANK_MEMORY_CACHE_LOCK = threading.Lock()

STATUS_VALUES = {"pending", "in_progress", "completed", "blocked", "skipped"}
TASK_STATUS_VALUES = {"in_progress", "completed", "blocked", "failed"}
ACTION_TYPES = {"tool_call", "finish"}
SUCCESSFUL_EVAL_STATUSES = {
    "completed",
    "finished_by_model",
    "max_iterations_reached",
    "forced_finalization_after_max_tool_calls",
}

class StructuredOutputError(RuntimeError):
    """Raised when the model cannot produce valid structured JSON."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts

@dataclass
class LLMResponse:
    content: str
    raw_response: dict[str, Any] | str
    usage: dict[str, Any]
    request_payload: dict[str, Any]
    http_status: int
    elapsed_seconds: float
    attempt_count: int

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")

def project_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def compact_json(value: Any, limit: int = MAX_JSON_CONTEXT_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    if len(text) <= limit:
        return text
    return text[: limit - 200] + f"...<已裁剪 {len(text) - limit + 200} 字符>"

def compact_text(value: Any, limit: int = MAX_JUDGE_REASON_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "1"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "0"}:
            return False
    return None

def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def max_tool_calls_for_task_doc(task_doc: dict[str, Any], task_id: str | None = None) -> int:
    if TASK_MAX_TOOL_CALLS_OVERRIDE is not None:
        if TASK_MAX_TOOL_CALLS_OVERRIDE <= 0:
            raise ValueError("TASK_MAX_TOOL_CALLS_OVERRIDE 必须为正整数")
        return TASK_MAX_TOOL_CALLS_OVERRIDE
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    metadata = task.get("difficulty_metadata") if isinstance(task, dict) and isinstance(task.get("difficulty_metadata"), dict) else {}
    explicit_budget = int_value(metadata.get("eval_max_tool_calls"))
    if explicit_budget is not None and explicit_budget > 0:
        return explicit_budget

    raw_difficulty = ""
    if isinstance(task, dict):
        raw_difficulty = str(task.get("difficulty") or "")
    if not raw_difficulty and task_id:
        raw_difficulty = str(task_id).replace("task_", "")
    difficulty = raw_difficulty.strip().lower()
    if difficulty in MAX_TOOL_CALLS_BY_TASK_DIFFICULTY:
        return MAX_TOOL_CALLS_BY_TASK_DIFFICULTY[difficulty]
    return MAX_TOOL_CALLS_PER_TASK

def context_json_limit(context: dict[str, Any]) -> int:
    interaction_history = context.get("user_interaction_history")
    if isinstance(interaction_history, dict):
        if int(interaction_history.get("interaction_count") or 0) > 40:
            return LONG_CONTEXT_MAX_JSON_CONTEXT_CHARS
    return MAX_JSON_CONTEXT_CHARS

def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "***"
    return value[:4] + "***" + value[-4:]

def safe_model_dir_name(model_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_name.strip())
    return safe or "unknown_model"

def normalize_preference_injection_mode(mode: str) -> str:
    clean = str(mode).strip().lower()
    if clean not in VALID_PREFERENCE_INJECTION_MODES:
        raise ValueError(
            f"PREFERENCE_INJECTION_MODE 必须是 {sorted(VALID_PREFERENCE_INJECTION_MODES)} 之一，当前为 {mode!r}"
        )
    return clean

def normalize_task_difficulty(difficulty: str) -> str:
    clean = str(difficulty).strip().lower()
    if clean not in VALID_TASK_DIFFICULTIES:
        raise ValueError(f"TASK_DIFFICULTY 必须是 {sorted(VALID_TASK_DIFFICULTIES)} 之一，当前为 {difficulty!r}")
    return clean

def task_ids_for_active_filters() -> set[str]:
    selected = {str(task_id) for task_id in TASK_ID_FILTER}
    difficulty = normalize_task_difficulty(TASK_DIFFICULTY)
    if difficulty != "all":
        selected.add(TASK_ID_BY_DIFFICULTY[difficulty])
    return selected

def difficulty_result_component(difficulty: str | None = None) -> str:
    return normalize_task_difficulty(TASK_DIFFICULTY if difficulty is None else difficulty)

def apply_task_difficulty_runtime_config(logger: logging.Logger | None = None) -> str:
    global MAX_TOOL_CALLS_PER_TASK
    difficulty = normalize_task_difficulty(TASK_DIFFICULTY)
    if difficulty != "all":
        MAX_TOOL_CALLS_PER_TASK = MAX_TOOL_CALLS_BY_TASK_DIFFICULTY[difficulty]
        if logger is not None:
            logger.info(
                "按任务难度设置最大工具调用轮次：difficulty=%s max_tool_calls=%s task_id=%s",
                difficulty,
                MAX_TOOL_CALLS_PER_TASK,
                TASK_ID_BY_DIFFICULTY[difficulty],
            )
    return difficulty

def normalize_pass_n(pass_n: int) -> int:
    if isinstance(pass_n, bool) or not isinstance(pass_n, int) or pass_n < 1:
        raise ValueError(f"PASS_N 必须是 >= 1 的整数，当前为 {pass_n!r}")
    return pass_n

def normalize_long_context_turns(turns: int) -> int:
    if isinstance(turns, bool) or not isinstance(turns, int):
        raise ValueError(f"LONG_CONTEXT_TURNS 必须是整数，当前为 {turns!r}")
    if turns not in LONG_CONTEXT_ALLOWED_TURNS:
        raise ValueError(f"LONG_CONTEXT_TURNS 必须是 {sorted(LONG_CONTEXT_ALLOWED_TURNS)} 之一，当前为 {turns!r}")
    return turns

def long_context_enabled_for_mode(preference_mode: str) -> bool:
    mode = normalize_preference_injection_mode(preference_mode)
    return bool(ENABLE_LONG_CONTEXT and normalize_long_context_turns(LONG_CONTEXT_TURNS) > 0 and mode in VALID_PREFERENCE_INJECTION_MODES)

def long_context_result_component(preference_mode: str) -> str | None:
    if not long_context_enabled_for_mode(preference_mode):
        return None
    return f"longctx_{normalize_long_context_turns(LONG_CONTEXT_TURNS)}"

def result_dir_for_model(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> Path:
    mode = normalize_preference_injection_mode(preference_mode)
    resolved_pass_n = normalize_pass_n(PASS_N if pass_n is None else pass_n)
    base = RESULT_BASE_DIR / safe_model_dir_name(model_name) / difficulty_result_component() / mode
    long_context_component = long_context_result_component(mode)
    if long_context_component:
        base = base / long_context_component
    return base / f"pass_{resolved_pass_n}"

def judge_result_dir_for_model(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> Path:
    mode = normalize_preference_injection_mode(preference_mode)
    resolved_pass_n = normalize_pass_n(PASS_N if pass_n is None else pass_n)
    base = JUDGE_RESULT_BASE_DIR / safe_model_dir_name(model_name) / difficulty_result_component() / mode
    long_context_component = long_context_result_component(mode)
    if long_context_component:
        base = base / long_context_component
    return base / f"pass_{resolved_pass_n}"

def configure_result_dir(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> tuple[Path, Path]:
    global RESULT_DIR, JUDGE_RESULT_DIR
    RESULT_DIR = result_dir_for_model(model_name, preference_mode, pass_n)
    JUDGE_RESULT_DIR = judge_result_dir_for_model(model_name, preference_mode, pass_n)
    return RESULT_DIR, JUDGE_RESULT_DIR

def load_api_config() -> Any:
    api_path = PROJECT_ROOT / "api.py"
    spec = importlib.util.spec_from_file_location("ppeval_api_config", api_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 API 配置文件: {api_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.environ.get("PPEVAL_GEN_MODEL"):
        module.GEN_MODEL = os.environ["PPEVAL_GEN_MODEL"]
    if os.environ.get("PPEVAL_GEN_URL"):
        module.GEN_URL = os.environ["PPEVAL_GEN_URL"]
    if os.environ.get("PPEVAL_GEN_KEY"):
        module.GEN_KEY = os.environ["PPEVAL_GEN_KEY"]
    return module

def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppeval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_DIR / f"eval_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def disable_proxy_for_eval(logger: logging.Logger) -> None:
    """关闭本进程内代理，避免正式评测走 VPN/proxy。"""
    proxy_env_names = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]
    before = {name: os.environ.get(name) for name in proxy_env_names if os.environ.get(name)}
    proxyoff_result: dict[str, Any] = {"attempted": True, "ok": False, "returncode": None, "stderr": ""}
    try:
        completed = subprocess.run(
            ["bash", "-lc", "proxyoff"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        proxyoff_result.update(
            {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip()[:500],
            }
        )
    except Exception as exc:  # noqa: BLE001 - proxy cleanup must not block eval.
        proxyoff_result.update({"ok": False, "stderr": str(exc)[:500]})

    for name in proxy_env_names:
        os.environ.pop(name, None)
    after = {name: os.environ.get(name) for name in proxy_env_names if os.environ.get(name)}
    logger.info(
        "评测前代理关闭检查：proxyoff=%s cleared_env_before=%s remaining_env_after=%s",
        proxyoff_result,
        {name: "***" for name in before},
        {name: "***" for name in after},
    )

def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start_positions = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos != -1]
    if not start_positions:
        raise json.JSONDecodeError("no JSON object or array found", stripped, 0)
    start = min(start_positions)
    stack: list[str] = []
    in_string = False
    escaped = False
    for idx, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                continue
            opener = stack.pop()
            if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                raise json.JSONDecodeError("mismatched JSON brackets", stripped, idx)
            if not stack:
                return json.loads(stripped[start : idx + 1])
    raise json.JSONDecodeError("unterminated JSON object", stripped, start)

def validate_plan_json(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    if not isinstance(obj.get("plan_version"), int):
        errors.append("plan_version 必须是整数。")
    if not isinstance(obj.get("steps"), list) or not obj.get("steps"):
        errors.append("steps 必须是非空数组。")
    else:
        seen_step_ids: set[str] = set()
        for idx, step in enumerate(obj["steps"], start=1):
            if not isinstance(step, dict):
                errors.append(f"steps[{idx}] 必须是 object。")
                continue
            step_id = step.get("step_id")
            if not isinstance(step_id, str) or not step_id:
                errors.append(f"steps[{idx}].step_id 必须是非空字符串。")
            elif step_id in seen_step_ids:
                errors.append(f"重复 step_id: {step_id}。")
            else:
                seen_step_ids.add(step_id)
            if not isinstance(step.get("description"), str) or not step.get("description"):
                errors.append(f"steps[{idx}].description 必须是非空字符串。")
    return errors

def initial_plan_step_count(plan: dict[str, Any]) -> int:
    steps = plan.get("steps") if isinstance(plan, dict) else []
    if not isinstance(steps, list):
        return 0
    return sum(1 for step in steps if isinstance(step, dict))

def max_iterations_from_initial_plan(
    plan: dict[str, Any],
    task_doc: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> int:
    _ = plan
    if task_doc is not None:
        return max_tool_calls_for_task_doc(task_doc, task_id)
    return MAX_TOOL_CALLS_PER_TASK

def validate_expected_phase(
    obj: Any,
    *,
    expected_phase: str,
    expected_iteration: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return errors
    if obj.get("phase") != expected_phase:
        errors.append(f"phase 必须是 {expected_phase!r}。")
    # iteration 由外层轨迹事件记录为权威值。模型偶尔复用旧 iteration，
    # 不应触发额外 LLM 修复调用，否则会改变被测模型的原始行为并显著放大 API 成本。
    return errors

def validate_action_json(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    action_type = obj.get("action_type")
    if action_type not in ACTION_TYPES:
        errors.append(f"action_type 必须是 {sorted(ACTION_TYPES)} 之一。")
    if not isinstance(obj.get("reason"), str) or not obj.get("reason"):
        errors.append("reason 必须是非空字符串。")
    if action_type == "tool_call":
        tool_call = obj.get("tool_call")
        if not isinstance(tool_call, dict):
            errors.append("action_type=tool_call 时 tool_call 必须是 object。")
        else:
            if not isinstance(tool_call.get("name"), str) or not tool_call.get("name"):
                errors.append("tool_call.name 必须是非空字符串。")
            if not isinstance(tool_call.get("arguments"), dict):
                errors.append("tool_call.arguments 必须是 object。")
    if action_type == "finish" and "final_answer" not in obj:
        errors.append("action_type=finish 时必须包含 final_answer。")
    if not isinstance(obj.get("completion_assessment"), dict):
        errors.append("completion_assessment 必须是 object。")
    return errors

def make_action_validator(
    *,
    available_api_names: list[str],
    api_definitions: dict[str, dict[str, Any]],
    expected_iteration: int | None = None,
) -> Callable[[Any], list[str]]:
    allowed_names = set(available_api_names)

    def validator(obj: Any) -> list[str]:
        errors = validate_action_json(obj)
        errors.extend(
            validate_expected_phase(
                obj,
                expected_phase="reason_action",
                expected_iteration=expected_iteration,
            )
        )
        if not isinstance(obj, dict) or obj.get("action_type") != "tool_call":
            return errors

        tool_call = obj.get("tool_call")
        if not isinstance(tool_call, dict):
            return errors
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return errors
        if name not in allowed_names:
            errors.append(
                "tool_call.name 必须严格来自 available_api_names，"
                f"当前为 {name!r}，允许值为 {sorted(allowed_names)}。"
            )
            return errors
        api_def = api_definitions.get(name)
        if not api_def:
            errors.append(f"tool_call.name 在全局 SandboxDB 中不存在: {name}")
            return errors
        return errors

    return validator

class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        logger: logging.Logger,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = logger
        self._response_format_supported = USE_JSON_RESPONSE_FORMAT
        self._response_format_lock = threading.Lock()

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        purpose: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> LLMResponse:
        timeout = timeout or REQUEST_TIMEOUT_SECONDS
        max_tokens = max_tokens or MODEL_MAX_TOKENS
        temperature = MODEL_TEMPERATURE if temperature is None else temperature
        last_error: Exception | None = None

        for attempt in range(1, API_RETRY_ATTEMPTS + 1):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            with self._response_format_lock:
                response_format_supported = self._response_format_supported
            if response_format_supported:
                payload["response_format"] = {"type": "json_object"}

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                self.chat_url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw_body = response.read().decode("utf-8")
                    elapsed = time.perf_counter() - started
                    try:
                        raw_json: dict[str, Any] | str = json.loads(raw_body)
                    except json.JSONDecodeError:
                        raw_json = raw_body
                    content, usage = self._extract_content_and_usage(raw_json)
                    self.logger.info(
                        "API 请求成功：purpose=%s status=%s attempt=%s elapsed=%.2fs prompt_tokens=%s completion_tokens=%s total_tokens=%s content_chars=%s",
                        purpose,
                        response.status,
                        attempt,
                        elapsed,
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                        len(content),
                    )
                    return LLMResponse(
                        content=content,
                        raw_response=raw_json,
                        usage=usage,
                        request_payload=payload,
                        http_status=response.status,
                        elapsed_seconds=elapsed,
                        attempt_count=attempt,
                    )
            except urllib.error.HTTPError as exc:
                elapsed = time.perf_counter() - started
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {error_body[:1000]}")
                if exc.code == 400 and "response_format" in error_body and response_format_supported:
                    with self._response_format_lock:
                        self._response_format_supported = False
                    self.logger.info("模型接口不接受 response_format，后续自动改用纯提示词 JSON 约束。")
                    continue
                qingyun_transient_400 = (
                    exc.code == 400
                    and "qingyuntop.top" in self.base_url
                    and any(
                        marker in error_body
                        for marker in ("Arrearage", "overdue-payment", "Access denied")
                    )
                )
                if (
                    exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
                    or qingyun_transient_400
                ) and attempt < API_RETRY_ATTEMPTS:
                    sleep_seconds = self._retry_sleep_seconds(attempt, exc.headers.get("Retry-After"))
                    self.logger.info(
                        "API 请求失败后重试：purpose=%s status=%s attempt=%s/%s elapsed=%.2fs sleep=%.2fs",
                        purpose,
                        exc.code,
                        attempt,
                        API_RETRY_ATTEMPTS,
                        elapsed,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"LLM API 请求失败 purpose={purpose}: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                elapsed = time.perf_counter() - started
                last_error = exc
                if attempt < API_RETRY_ATTEMPTS:
                    sleep_seconds = self._retry_sleep_seconds(attempt)
                    self.logger.info(
                        "API 网络错误后重试：purpose=%s attempt=%s/%s elapsed=%.2fs sleep=%.2fs error=%s",
                        purpose,
                        attempt,
                        API_RETRY_ATTEMPTS,
                        elapsed,
                        sleep_seconds,
                        exc,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"LLM API 网络错误 purpose={purpose}: {exc}") from exc

        raise RuntimeError(f"LLM API 请求失败 purpose={purpose}: {last_error}")

    @staticmethod
    def _retry_sleep_seconds(attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return max(0.5, float(retry_after))
            except ValueError:
                pass
        jitter = random.uniform(0.0, 0.5)
        return API_RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1)) + jitter

    @staticmethod
    def _extract_content_and_usage(raw_json: dict[str, Any] | str) -> tuple[str, dict[str, Any]]:
        if not isinstance(raw_json, dict):
            return str(raw_json), {}
        usage = raw_json.get("usage") or {}
        choices = raw_json.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content, usage
                text = first.get("text")
                if isinstance(text, str):
                    return text, usage
        return json.dumps(raw_json, ensure_ascii=False), usage

def request_structured_json(
    client: LLMClient,
    *,
    messages: list[dict[str, str]],
    purpose: str,
    validator: Callable[[Any], list[str]],
    logger: logging.Logger,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    current_messages = messages
    last_error = ""

    for repair_idx in range(JSON_REPAIR_ATTEMPTS + 1):
        response = client.chat(
            current_messages,
            purpose=purpose,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        parsed: Any | None = None
        parse_error = ""
        validation_errors: list[str] = []
        try:
            parsed = extract_json_object(response.content)
            validation_errors = validator(parsed)
            if validation_errors:
                parse_error = "结构校验失败：" + "; ".join(validation_errors)
        except json.JSONDecodeError as exc:
            parse_error = f"JSON 解析失败：{exc}"

        attempt_record = {
            "repair_index": repair_idx,
            "purpose": purpose,
            "ok": bool(parsed is not None and not validation_errors),
            "content": response.content if SAVE_RAW_RESPONSES_IN_RESULTS else None,
            "parsed": parsed if parsed is not None else None,
            "parse_or_validation_error": parse_error,
            "usage": response.usage,
            "http_status": response.http_status,
            "elapsed_seconds": response.elapsed_seconds,
            "api_attempt_count": response.attempt_count,
            "request_payload": response.request_payload if SAVE_PROMPTS_IN_RESULTS else None,
            "raw_response": response.raw_response if SAVE_RAW_RESPONSES_IN_RESULTS else None,
        }
        attempts.append(attempt_record)

        if parsed is not None and not validation_errors:
            return parsed, attempts

        last_error = parse_error
        logger.info("结构化 JSON 生成失败，准备修复：purpose=%s repair=%s error=%s", purpose, repair_idx, parse_error)
        current_messages = build_repair_messages(messages, response.content, parse_error)

    raise StructuredOutputError(f"{purpose} 无法生成有效 JSON：{last_error}", attempts)

def llm_attempt_detail_path(*, run_id: str, user_id: str) -> Path:
    return RESULT_DIR / LLM_ATTEMPT_DETAIL_DIR_NAME / run_id / f"{user_id}.jsonl"

def compact_llm_attempt_for_main_result(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair_index": attempt.get("repair_index"),
        "purpose": attempt.get("purpose"),
        "ok": attempt.get("ok"),
        "parse_or_validation_error": attempt.get("parse_or_validation_error"),
        "usage": attempt.get("usage"),
        "http_status": attempt.get("http_status"),
        "elapsed_seconds": attempt.get("elapsed_seconds"),
        "api_attempt_count": attempt.get("api_attempt_count"),
        "content_chars": len(attempt.get("content") or "") if isinstance(attempt.get("content"), str) else None,
        "request_message_count": len((attempt.get("request_payload") or {}).get("messages", []))
        if isinstance(attempt.get("request_payload"), dict)
        else None,
        "raw_response_type": type(attempt.get("raw_response")).__name__ if attempt.get("raw_response") is not None else None,
    }

def persist_llm_attempt_details(
    *,
    detail_path: Path | None,
    run_id: str,
    user_id: str,
    task_id: str,
    pass_n: int,
    pass_index: int,
    phase: str,
    iteration: int,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not attempts:
        return []

    light_attempts: list[dict[str, Any]] = []
    for attempt_index, attempt in enumerate(attempts, start=1):
        detail_ref = {
            "detail_path": project_relative_path(detail_path) if detail_path is not None else None,
            "attempt_index": attempt_index,
            "purpose": attempt.get("purpose"),
        }
        if SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR and detail_path is not None:
            write_jsonl(
                detail_path,
                {
                    "run_id": run_id,
                    "created_at": utc_now_iso(),
                    "user_id": user_id,
                    "task_id": task_id,
                    "pass_n": pass_n,
                    "pass_index": pass_index,
                    "phase": phase,
                    "iteration": iteration,
                    "attempt_index": attempt_index,
                    "detail_ref": detail_ref,
                    "attempt": attempt,
                },
            )

        if KEEP_FULL_LLM_ATTEMPTS_IN_MAIN_RESULT:
            light_attempt = copy.deepcopy(attempt)
        else:
            light_attempt = compact_llm_attempt_for_main_result(attempt)
        light_attempt["detail_ref"] = detail_ref
        light_attempts.append(light_attempt)
    return light_attempts

def build_repair_messages(
    original_messages: list[dict[str, str]],
    bad_content: str,
    error_message: str,
) -> list[dict[str, str]]:
    return json_repair.build_messages(
        original_messages=original_messages,
        bad_content=bad_content,
        error_message=error_message,
        compact_json=compact_json,
        preserve_semantics_instruction="请在不改变语义目标的前提下，修复为满足要求的 JSON object。",
    )

class SandboxExecutor:
    def __init__(
        self,
        api_definitions: dict[str, dict[str, Any]],
    ) -> None:
        self.api_definitions = api_definitions
        self.database = GlobalSandboxDatabase(api_definitions)

    def prepare_retrieval_indexes(self, logger: logging.Logger | None = None) -> dict[str, Any]:
        return self.database.prepare_vector_indexes(logger=logger)

    def execute(
        self,
        *,
        tool_call: dict[str, Any],
        task_id: str,
        workflow: dict[str, Any],
        user_sandbox: dict[str, Any],
        call_index: int,
        used_example_counts: dict[str, int],
    ) -> dict[str, Any]:
        api_name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        available_names = set(workflow.get("available_api_names") or user_sandbox.get("accessible_api_names") or [])
        validation = self._validate_call(api_name, arguments, available_names)
        if validation["errors"]:
            return self._error_response(api_name, arguments, task_id, call_index, validation)

        assert isinstance(api_name, str)
        assert isinstance(arguments, dict)
        user_id = str(user_sandbox.get("user_id") or workflow.get("user_id") or "unknown_user")
        selected_record = self.database.select_response(
            api_name,
            arguments,
            user_id=user_id,
            task_id=task_id,
        )
        if selected_record is not None and isinstance(selected_record.get("mock_response"), dict):
            return materialize_sandbox_response(
                template=selected_record["mock_response"],
                api_name=api_name,
                arguments=arguments,
                task_id=task_id,
                action=self.api_definitions[api_name].get("action"),
                transport=self._transport(api_name, call_index, http_status=200),
                validation=validation,
                conditioning_database=self.database,
                retrieval_metadata=selected_record.get("_sandbox_retrieval"),
                user_context=user_sandbox.get("user_context", {}),
            )

        template = self._select_mock_response(workflow, api_name, arguments, used_example_counts)
        if template is not None:
            return materialize_sandbox_response(
                template=template,
                api_name=api_name,
                arguments=arguments,
                task_id=task_id,
                action=self.api_definitions[api_name].get("action"),
                transport=self._transport(api_name, call_index, http_status=200),
                validation=validation,
                conditioning_database=self.database,
                retrieval_metadata={
                    "strategy": "workflow_call_example_fallback",
                    "selected_record_id": None,
                },
                user_context=user_sandbox.get("user_context", {}),
            )

        api_def = self.api_definitions[api_name]
        response_body = self._generic_response_body(api_def, arguments, task_id, call_index)
        return materialize_sandbox_response(
            template={
                "sandbox_status": "success",
                "api_name": api_name,
                "action": api_def.get("action"),
                "task_context": task_id,
                "transport": self._transport(api_name, call_index, http_status=200),
                "request_echo": {
                    "function": api_name,
                    "arguments": copy.deepcopy(arguments),
                    "idempotency_key": f"idm_{api_name.replace('.', '_')}_{task_id}_{call_index:03d}",
                },
                "validation": validation,
                "context_effects": [
                    "调用参数已按用户信息、任务约束和可用 API 合约进行校验。",
                    "返回内容为 SandBoxDB 的确定性模拟结果，用于规划型 LLM 评估。",
                ],
                "warnings": validation["warnings"],
                "errors": [],
                "response_body": response_body,
                "feedback": {
                    "machine_readable_summary": response_body.get("summary", "SandboxDB call completed."),
                    "planner_relevance": "usable_for_next_step",
                    "requires_user_confirmation": False,
                    "missing_information": [],
                    "next_suggested_api_calls": self._suggest_next_apis(workflow, api_name),
                },
                "raw_response_shape": {
                    key: self._shape_name(value)
                    for key, value in response_body.items()
                },
            },
            api_name=api_name,
            arguments=arguments,
            task_id=task_id,
            action=api_def.get("action"),
            transport=self._transport(api_name, call_index, http_status=200),
            validation=validation,
            conditioning_database=self.database,
            retrieval_metadata={
                "strategy": "generic_fallback_after_no_compatible_cache_record",
                "selected_record_id": None,
            },
            user_context=user_sandbox.get("user_context", {}),
        )

    def _validate_call(
        self,
        api_name: Any,
        arguments: Any,
        available_names: set[str],
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        missing_required: list[str] = []
        unexpected_arguments: list[str] = []
        type_warnings: list[str] = []

        if not isinstance(api_name, str) or not api_name:
            errors.append("tool_call.name 必须是非空字符串。")
            return {
                "allowed": False,
                "errors": errors,
                "warnings": warnings,
                "missing_required": missing_required,
                "unexpected_arguments": unexpected_arguments,
                "type_warnings": type_warnings,
            }
        if api_name not in self.api_definitions:
            errors.append(f"未知 API: {api_name}")
        if available_names and api_name not in available_names:
            errors.append(f"当前任务不可调用 API: {api_name}")
        if not isinstance(arguments, dict):
            errors.append("tool_call.arguments 必须是 object。")
            arguments = {}

        api_def = self.api_definitions.get(api_name)
        if api_def:
            params = api_def.get("interface", {}).get("parameters", [])
            param_names = {param.get("name") for param in params}
            for param in params:
                name = param.get("name")
                if param.get("required") and name not in arguments:
                    missing_required.append(str(name))
                elif name in arguments:
                    warning = self._type_warning(name, arguments[name], str(param.get("type", "")))
                    if warning:
                        type_warnings.append(warning)
            unexpected_arguments = sorted(str(name) for name in arguments if name not in param_names)
            if unexpected_arguments:
                warnings.append(f"存在 API 定义之外的参数: {unexpected_arguments}")
            if missing_required:
                errors.append(f"缺少必填参数: {missing_required}")
            warnings.extend(type_warnings)

        return {
            "allowed": not errors,
            "errors": errors,
            "warnings": warnings,
            "missing_required": missing_required,
            "unexpected_arguments": unexpected_arguments,
            "type_warnings": type_warnings,
        }

    @staticmethod
    def _type_warning(name: str, value: Any, expected_type: str) -> str | None:
        if expected_type in {"string", "datetime", "date"} and not isinstance(value, str):
            return f"{name} 期望 {expected_type}，实际 {type(value).__name__}"
        if expected_type == "integer" and not isinstance(value, int):
            return f"{name} 期望 integer，实际 {type(value).__name__}"
        if expected_type == "number" and not isinstance(value, (int, float)):
            return f"{name} 期望 number，实际 {type(value).__name__}"
        if expected_type == "boolean" and not isinstance(value, bool):
            return f"{name} 期望 boolean，实际 {type(value).__name__}"
        if expected_type.startswith("array") and not isinstance(value, list):
            return f"{name} 期望 {expected_type}，实际 {type(value).__name__}"
        if expected_type == "object" and not isinstance(value, dict):
            return f"{name} 期望 object，实际 {type(value).__name__}"
        return None

    @staticmethod
    def _select_mock_response(
        workflow: dict[str, Any],
        api_name: str,
        arguments: dict[str, Any],
        used_example_counts: dict[str, int],
    ) -> dict[str, Any] | None:
        examples = [
            example
            for example in workflow.get("call_examples", [])
            if example.get("api_name") == api_name and isinstance(example.get("mock_response"), dict)
            and hard_constraints_compatible(
                arguments,
                (example.get("function_call") or {}).get("arguments", {}),
            )[0]
        ]
        if not examples:
            return None
        used = used_example_counts.get(api_name, 0)
        used_example_counts[api_name] = used + 1
        return examples[used % len(examples)]["mock_response"]

    def _error_response(
        self,
        api_name: Any,
        arguments: Any,
        task_id: str,
        call_index: int,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        safe_name = api_name if isinstance(api_name, str) and api_name else "unknown"
        return {
            "sandbox_status": "error",
            "api_name": safe_name,
            "action": None,
            "task_context": task_id,
            "transport": self._transport(safe_name, call_index, http_status=400),
            "request_echo": {
                "function": api_name,
                "arguments": copy.deepcopy(arguments),
                "idempotency_key": f"idm_invalid_{task_id}_{call_index:03d}",
            },
            "validation": validation,
            "warnings": validation.get("warnings", []),
            "errors": validation.get("errors", []),
            "response_body": {
                "summary": "工具调用未通过 SandBoxDB 校验，需要修正 API 名称或参数后重试。",
                "missing_information": validation.get("missing_required", []),
            },
            "feedback": {
                "machine_readable_summary": "invalid_tool_call",
                "planner_relevance": "retry_with_corrected_tool_call",
                "requires_user_confirmation": False,
                "missing_information": validation.get("missing_required", []),
                "next_suggested_api_calls": [],
            },
            "raw_response_shape": {
                "error": "object",
            },
        }

    @staticmethod
    def _transport(api_name: str, call_index: int, *, http_status: int) -> dict[str, Any]:
        endpoint = "/v1/" + api_name.replace(".", "/")
        return {
            "protocol": "sandbox",
            "method": "POST",
            "endpoint": endpoint,
            "http_status": http_status,
            "request_id": f"req_{api_name.replace('.', '_')}_{call_index:03d}",
            "latency_ms": random.randint(SANDBOX_LATENCY_MIN_MS, SANDBOX_LATENCY_MAX_MS),
            "rate_limit": {
                "limit": 1000,
                "remaining": max(0, 1000 - call_index),
                "reset_at": "2026-07-02T00:00:00Z",
            },
        }

    def _generic_response_body(
        self,
        api_def: dict[str, Any],
        arguments: dict[str, Any],
        task_id: str,
        call_index: int,
    ) -> dict[str, Any]:
        api_name = api_def["api_name"]
        returns = api_def.get("interface", {}).get("returns", {})
        response: dict[str, Any] = {
            "summary": f"{api_name} completed in SandboxDB for {task_id}.",
            "selected_inputs": copy.deepcopy(arguments),
            "items": [
                {
                    "id": f"{api_name.replace('.', '_')}_{task_id}_{call_index:03d}_item_1",
                    "title": arguments.get("title") or arguments.get("query") or api_def.get("description"),
                    "status": "candidate" if api_def.get("action", "").startswith(("search", "find", "retrieve")) else "created",
                    "context_match_score": 0.84,
                }
            ],
            "missing_information": [],
        }
        for key, type_name in returns.items():
            if key in response:
                continue
            response[key] = self._value_for_return_type(key, type_name, api_name, task_id, call_index)
        return response

    @staticmethod
    def _value_for_return_type(
        key: str,
        type_name: str,
        api_name: str,
        task_id: str,
        call_index: int,
    ) -> Any:
        if type_name == "array":
            return [
                {
                    "id": f"{api_name.replace('.', '_')}_{task_id}_{call_index:03d}_{key}_1",
                    "status": "available",
                    "confidence": 0.82,
                }
            ]
        if type_name == "object":
            return {
                "id": f"{api_name.replace('.', '_')}_{task_id}_{call_index:03d}_{key}",
                "status": "ready",
                "confidence": 0.82,
            }
        if type_name == "boolean":
            return True
        if type_name == "integer":
            return 1
        if type_name == "number":
            return 0.82
        if type_name.endswith("|null"):
            return None
        return f"{api_name.replace('.', '_')}_{task_id}_{call_index:03d}_{key}"

    @staticmethod
    def _shape_name(value: Any) -> str:
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        if value is None:
            return "null"
        return type(value).__name__

    @staticmethod
    def _suggest_next_apis(workflow: dict[str, Any], current_api_name: str) -> list[str]:
        names = [example.get("api_name") for example in workflow.get("call_examples", [])]
        clean_names = [name for name in names if isinstance(name, str)]
        if current_api_name in clean_names:
            idx = clean_names.index(current_api_name)
            return clean_names[idx + 1 : idx + 4]
        return clean_names[:3]

def load_global_api_definitions() -> dict[str, dict[str, Any]]:
    index_path = SANDBOX_DIR / "api_index.json"
    api_index = read_json(index_path)
    api_definitions: dict[str, dict[str, Any]] = {}
    for api_name, entry in api_index.items():
        definition_path = SANDBOX_DIR / entry["definition_path"]
        api_definitions[api_name] = read_json(definition_path)
    return api_definitions

def load_global_sandbox_database_manifest() -> dict[str, Any]:
    manifest_path = SANDBOX_DIR / "database" / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"全局 SandboxDB database 不存在: {manifest_path}。"
            "请先运行 python scripts/migrate_to_global_sandbox_database.py。"
        )
    return read_json(manifest_path)

def load_user_sandbox(user_dir: Path) -> dict[str, Any]:
    database_manifest_path = SANDBOX_DIR / "database" / "manifest.json"
    workflow_index_path = SANDBOX_DIR / "database" / "workflow_index.json"
    if database_manifest_path.exists() and workflow_index_path.exists():
        database_manifest = read_json(database_manifest_path)
        workflow_index = read_json(workflow_index_path)
        user_index_path = SANDBOX_DIR / "database" / "user_index.json"
        user_index = read_json(user_index_path) if user_index_path.exists() else {}
        user_entry = user_index.get(user_dir.name, {})
        user_context_path = SANDBOX_DIR / "database" / "users" / f"{user_dir.name}.json"
        user_doc = read_json(user_context_path) if user_context_path.exists() else {}
        workflows: list[dict[str, Any]] = []
        for key, entry in sorted(workflow_index.items()):
            if not key.startswith(f"{user_dir.name}/"):
                continue
            workflow_doc = read_json(SANDBOX_DIR / entry["path"])
            workflow = workflow_doc.get("workflow")
            if isinstance(workflow, dict):
                workflows.append(workflow)
        if not workflows:
            raise RuntimeError(f"全局 SandboxDB database 中找不到用户 workflow: {user_dir.name}")
        accessible_api_names = sorted({name for workflow in workflows for name in workflow.get("available_api_names", [])})
        return {
            "schema_version": database_manifest.get("schema_version", "SandboxDatabase.v1"),
            "user_id": user_dir.name,
            "source": {
                "generation_method": "global_sandbox_database",
                "database_manifest": "SandboxDB/database/manifest.json",
            },
            "sandbox_reference": {
                "sandbox_dir": "SandboxDB",
                "schema_version": "SandboxDB.v2",
                "database_path": "SandboxDB/database",
                "contract": "All users share one global offline sandbox database.",
            },
            "user_context": user_doc.get("user_context", {}),
            "accessible_api_names": user_entry.get("accessible_api_names", accessible_api_names),
            "task_workflows": workflows,
        }

    sandbox_dir = user_dir / "sandbox"
    manifest_path = sandbox_dir / "manifest.json"
    if not manifest_path.exists():
        return read_json(user_dir / "sandboxDB.json")

    manifest = read_json(manifest_path)
    user_context_path = sandbox_dir / manifest.get("files", {}).get("user_context", "user_context.json")
    user_context = read_json(user_context_path) if user_context_path.exists() else {}
    workflows: list[dict[str, Any]] = []
    for workflow_entry in manifest.get("task_workflows", []):
        workflow_path = workflow_entry.get("workflow_path")
        if workflow_path:
            task_workflow_path = sandbox_dir / workflow_path
            workflows.append(read_json(task_workflow_path))
        else:
            workflows.append(workflow_entry)
    return {
        **manifest,
        "user_context": user_context,
        "task_workflows": workflows,
    }

def compact_api_definition(api_def: dict[str, Any]) -> dict[str, Any]:
    interface = api_def.get("interface", {})
    return {
        "api_name": api_def.get("api_name"),
        "category": api_def.get("category"),
        "action": api_def.get("action"),
        "NoRelation": int(api_def.get("NoRelation") or 0),
        "description": compact_text(api_def.get("description"), 240),
        "parameters": [
            {
                "name": param.get("name"),
                "type": param.get("type"),
                "required": param.get("required"),
                "preference_sensitive": param.get("preference_sensitive"),
                "description": compact_text(param.get("description"), 180),
            }
            for param in interface.get("parameters", [])
        ],
        "returns": interface.get("returns", {}),
        "preference_sensitive_parameters": interface.get("preference_sensitive_parameters", []),
    }

def compact_source_hit_for_prompt(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "doc_id": hit.get("doc_id"),
            "domain": hit.get("domain"),
            "title": compact_text(hit.get("title"), 240),
            "text": compact_text(hit.get("text"), 500),
            "attributes": hit.get("attributes"),
            "multilingual_e5_score": hit.get("multilingual_e5_score"),
        }.items()
        if value not in (None, "", [], {})
    }

def compact_response_body_for_prompt(response_body: Any) -> Any:
    if not isinstance(response_body, dict):
        return response_body
    projected: dict[str, Any] = {}
    for key in (
        "summary",
        "items",
        "quality_checks",
        "missing_information",
        "selected_inputs",
        "options",
        "results",
        "data",
    ):
        value = response_body.get(key)
        if value not in (None, "", [], {}):
            projected[key] = value
    source_info = response_body.get("source_info")
    if isinstance(source_info, dict):
        hits = source_info.get("hits") if isinstance(source_info.get("hits"), list) else []
        projected["source_info"] = {
            "hits": [compact_source_hit_for_prompt(hit) for hit in hits[:5] if isinstance(hit, dict)]
        }
    if not projected:
        for key, value in response_body.items():
            if key not in {"argument_conditioning", "raw_response_shape", "pagination"}:
                projected[key] = value
    return projected

def neutralize_internal_task_ids_for_prompt(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key).replace("task_easy", "task").replace("task_hard", "task"): neutralize_internal_task_ids_for_prompt(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [neutralize_internal_task_ids_for_prompt(item) for item in value]
    if isinstance(value, str):
        return value.replace("task_easy", "task").replace("task_hard", "task")
    return value

def filter_sandbox_observation_metadata_for_prompt(value: Any) -> Any:
    """Return a prompt-only copy without transport and duplicated request metadata."""
    if isinstance(value, dict):
        return {
            key: filter_sandbox_observation_metadata_for_prompt(item)
            for key, item in value.items()
            if key not in SANDBOX_OBSERVATION_PROMPT_EXCLUDED_FIELDS
        }
    if isinstance(value, list):
        return [filter_sandbox_observation_metadata_for_prompt(item) for item in value]
    return value

def compact_sandbox_observation_for_prompt(observation: Any) -> Any:
    if not isinstance(observation, dict):
        return observation
    projected: dict[str, Any] = {}
    for key in ("sandbox_status", "api_name", "action", "validation", "warnings", "errors", "feedback"):
        value = observation.get(key)
        if value not in (None, "", [], {}):
            projected[key] = value
    if "response_body" in observation:
        projected["response_body"] = compact_response_body_for_prompt(observation.get("response_body"))
    return neutralize_internal_task_ids_for_prompt(projected)

def compact_sandbox_observation_for_ledger(observation: Any) -> Any:
    projected = compact_sandbox_observation_for_prompt(observation)
    if not isinstance(projected, dict):
        return projected
    ledger: dict[str, Any] = {
        key: projected.get(key)
        for key in ("sandbox_status", "api_name", "errors", "warnings")
        if projected.get(key) not in (None, "", [], {})
    }
    validation = projected.get("validation") if isinstance(projected.get("validation"), dict) else {}
    validation_summary = {
        key: validation.get(key)
        for key in ("allowed", "errors", "warnings", "missing_required", "unexpected_arguments")
        if validation.get(key) not in (None, "", [], {})
    }
    if validation_summary:
        ledger["validation"] = validation_summary
    feedback = projected.get("feedback") if isinstance(projected.get("feedback"), dict) else {}
    feedback_summary = {
        key: feedback.get(key)
        for key in ("machine_readable_summary", "missing_information", "next_suggested_api_calls")
        if feedback.get(key) not in (None, "", [], {})
    }
    if feedback_summary:
        ledger["feedback"] = feedback_summary
    body = projected.get("response_body") if isinstance(projected.get("response_body"), dict) else {}
    body_summary: dict[str, Any] = {}
    if body.get("summary") not in (None, ""):
        body_summary["summary"] = compact_text(body.get("summary"), 400)
    items = body.get("items") if isinstance(body.get("items"), list) else []
    if items:
        body_summary["items"] = [
            {
                key: item.get(key)
                for key in ("id", "title", "name", "status", "confidence", "price", "rating")
                if item.get(key) not in (None, "", [], {})
            }
            for item in items[:3]
            if isinstance(item, dict)
        ]
    if body.get("missing_information") not in (None, "", [], {}):
        body_summary["missing_information"] = body.get("missing_information")
    source_info = body.get("source_info") if isinstance(body.get("source_info"), dict) else {}
    hits = source_info.get("hits") if isinstance(source_info.get("hits"), list) else []
    if hits:
        body_summary["source_hits"] = [
            {
                key: hit.get(key)
                for key in ("doc_id", "domain", "title", "attributes", "multilingual_e5_score")
                if hit.get(key) not in (None, "", [], {})
            }
            for hit in hits[:3]
            if isinstance(hit, dict)
        ]
    if body_summary:
        ledger["result"] = body_summary
    return ledger

def current_action_context(context: dict[str, Any], current_plan: dict[str, Any]) -> dict[str, Any]:
    action_context = copy.deepcopy(context)
    next_step_id = current_plan.get("next_step_id")
    steps = current_plan.get("steps") if isinstance(current_plan.get("steps"), list) else []
    current_step = next(
        (
            step
            for step in steps
            if isinstance(step, dict) and step.get("step_id") == next_step_id
        ),
        None,
    )
    target_api = current_step.get("target_api") if isinstance(current_step, dict) else None
    action_context["current_action_step"] = {
        "step_id": next_step_id,
        "target_api": target_api,
    }
    return action_context

def find_workflow(user_sandbox: dict[str, Any], task_id: str) -> dict[str, Any]:
    for workflow in user_sandbox.get("task_workflows", []):
        if workflow.get("task_id") == task_id:
            return workflow
    return {
        "task_id": task_id,
        "available_api_names": user_sandbox.get("accessible_api_names", []),
        "call_examples": [],
    }

def build_reference_examples(workflow: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for example in workflow.get("call_examples", [])[:limit]:
        function_call = example.get("function_call") or {}
        examples.append(
            {
                "api_name": example.get("api_name"),
                "call_id": example.get("call_id"),
                "example_arguments": function_call.get("arguments", {}),
            }
        )
    return examples

def redacted_user_context_for_hidden(user_context: dict[str, Any]) -> dict[str, Any]:
    _ = user_context
    return {}

def task_context_for_mode(task_doc: dict[str, Any], preference_mode: str) -> dict[str, Any]:
    task = copy.deepcopy(task_doc.get("task", task_doc))
    normalize_preference_injection_mode(preference_mode)
    prompt_fields = (
        "title",
        "daily_life_domain",
        "scenario",
        "user_goal",
        "constraints",
        "expected_deliverable",
        "success_criteria",
    )
    return {
        field: task[field]
        for field in prompt_fields
        if field in task and task[field] not in (None, "", [], {})
    }

def redact_direct_preference_values(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if "preference" in key_text or key_text in {"priority_rules", "avoid_time_blocks"}:
                continue
            cleaned = redact_direct_preference_values(item)
            if cleaned not in (None, "", [], {}):
                redacted[key] = cleaned
        return redacted
    if isinstance(value, list):
        clean_list: list[Any] = []
        for item in value:
            cleaned = redact_direct_preference_values(item)
            if cleaned not in (None, "", [], {}):
                clean_list.append(cleaned)
        return clean_list
    if isinstance(value, str):
        lowered = value.lower()
        direct_markers = (
            "prefer",
            "preference",
            "dislike",
            "avoid",
            "values",
            "privacy",
            "budget",
            "willing to",
            "strong privacy",
        )
        if any(marker in lowered for marker in direct_markers):
            return None
    return value

def reference_examples_for_mode(workflow: dict[str, Any], preference_mode: str) -> list[dict[str, Any]]:
    examples = build_reference_examples(workflow)
    normalize_preference_injection_mode(preference_mode)
    redacted: list[dict[str, Any]] = []
    for example in examples:
        cleaned = copy.deepcopy(example)
        cleaned["example_arguments"] = redact_direct_preference_values(cleaned.get("example_arguments", {}))
        redacted.append(cleaned)
    return redacted

def normalize_tool_availability_mode(mode: str) -> str:
    clean = str(mode).strip().lower()
    if clean not in VALID_TOOL_AVAILABILITY_MODES:
        raise ValueError(
            f"TOOL_AVAILABILITY_MODE 必须是 {sorted(VALID_TOOL_AVAILABILITY_MODES)} 之一，当前为 {mode!r}"
        )
    return clean

def api_category(api_definitions: dict[str, dict[str, Any]], api_name: str) -> str:
    api_def = api_definitions.get(api_name, {})
    return str(api_def.get("category") or api_name.split(".", 1)[0])

def global_api_catalog(api_definitions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "api_name": name,
            "category": api_category(api_definitions, name),
            "action": api_def.get("action"),
            "description": api_def.get("description"),
            "required_parameters": [
                param.get("name")
                for param in api_def.get("interface", {}).get("parameters", [])
                if param.get("required")
            ],
            "optional_parameters": [
                param.get("name")
                for param in api_def.get("interface", {}).get("parameters", [])
                if not param.get("required")
            ][:8],
        }
        for name, api_def in sorted(api_definitions.items())
    ]

def deterministic_sample(items: list[str], *, seed: str, count: int) -> list[str]:
    if count <= 0 or not items:
        return []
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]

def noisy_available_api_names(
    *,
    oracle_names: list[str],
    api_definitions: dict[str, dict[str, Any]],
    user_id: str,
    task_id: str,
) -> tuple[list[str], dict[str, Any]]:
    oracle_set = set(oracle_names)
    oracle_categories = {api_category(api_definitions, name) for name in oracle_names if name in api_definitions}
    all_names = sorted(api_definitions)
    same_category_candidates = [
        name
        for name in all_names
        if name not in oracle_set and api_category(api_definitions, name) in oracle_categories
    ]
    cross_category_candidates = [
        name
        for name in all_names
        if name not in oracle_set and api_category(api_definitions, name) not in oracle_categories
    ]
    same = deterministic_sample(
        same_category_candidates,
        seed=f"{user_id}|{task_id}|same_category_distractors",
        count=NOISY_SAME_CATEGORY_DISTRACTOR_COUNT,
    )
    cross = deterministic_sample(
        cross_category_candidates,
        seed=f"{user_id}|{task_id}|cross_category_distractors",
        count=NOISY_CROSS_CATEGORY_DISTRACTOR_COUNT,
    )
    names = sorted(oracle_set | set(same) | set(cross))
    return names, {
        "same_category_distractors": same,
        "cross_category_distractors": cross,
        "oracle_api_count": len(oracle_names),
        "selected_api_count": len(names),
    }

def validate_tool_discovery_json(api_definitions: dict[str, dict[str, Any]]) -> Callable[[Any], list[str]]:
    allowed = set(api_definitions)

    def validator(obj: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(obj, dict):
            return ["顶层必须是 JSON object。"]
        selected = obj.get("selected_api_names")
        if not isinstance(selected, list):
            return ["selected_api_names 必须是数组。"]
        if not selected:
            errors.append("selected_api_names 不能为空。")
        if len(selected) > GLOBAL_RETRIEVAL_MAX_APIS:
            errors.append(f"selected_api_names 数量不能超过 {GLOBAL_RETRIEVAL_MAX_APIS}。")
        for idx, name in enumerate(selected, start=1):
            if not isinstance(name, str) or not name:
                errors.append(f"selected_api_names[{idx}] 必须是非空字符串。")
            elif name not in allowed:
                errors.append(f"selected_api_names[{idx}] 不在全局 API definitions 中: {name}")
        if not isinstance(obj.get("selection_rationale"), list):
            errors.append("selection_rationale 必须是数组。")
        return errors

    return validator

def tool_discovery_messages(
    *,
    user_id: str,
    task_id: str,
    task_doc: dict[str, Any],
    user_profile: dict[str, Any],
    hidden_info: dict[str, Any] | None,
    long_context: dict[str, Any] | None,
    user_sandbox: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    preference_mode: str,
) -> list[dict[str, str]]:
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    sandbox_user_context = user_sandbox.get("user_context", {})
    mode = normalize_preference_injection_mode(preference_mode)
    if mode in {"hidden", "pag", "bank"} and isinstance(sandbox_user_context, dict):
        sandbox_user_context = redacted_user_context_for_hidden(sandbox_user_context)
    discovery_context: dict[str, Any] = {}
    if mode == "bank":
        if not isinstance(hidden_info, dict):
            raise RuntimeError("bank 模式需要每个用户目录下存在 hidden_info.json。")
        discovery_context["bank_context"] = build_bank_context(
            user_id=user_id,
            task_id=task_id,
            task_doc=task_doc,
        )
    elif mode == "pag":
        if not isinstance(hidden_info, dict):
            raise RuntimeError("pag 模式需要每个用户目录下存在 hidden_info.json。")
        discovery_context["pag_context"] = build_pag_context(
            user_id=user_id,
            task_id=task_id,
            hidden_info=hidden_info,
            task_doc=task_doc,
        )
    discovery_context.update({
        "task": task_context_for_mode(task_doc, mode),
        "user_context": neutral_user_information(sandbox_user_context),
        "global_api_catalog": global_api_catalog(api_definitions),
        "selection_limits": {
            "min_api_count": GLOBAL_RETRIEVAL_MIN_APIS,
            "max_api_count": GLOBAL_RETRIEVAL_MAX_APIS,
        },
    })
    if mode in {"direct", "mix"}:
        discovery_context["user_information"] = user_information_for_prompt(user_profile)
    if mode in {"hidden", "mix"}:
        interactions = shuffled_user_interactions(
            hidden_info=hidden_info,
            long_context=long_context,
            user_id=user_id,
            task_id=task_id,
            mode=mode,
        )
        if interactions:
            discovery_context["user_interaction_history"] = {
                "interaction_count": len(interactions),
                "interactions": interactions,
            }

    schema = {
        "phase": "tool_discovery",
        "selected_api_names": ["必须来自 global_api_catalog.api_name"],
        "selection_rationale": [
            {
                "api_name": "string",
                "reason": "为什么该 API 对任务和用户偏好有用",
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "你正在为一个工具调用规划任务做 tool discovery。请只从 global_api_catalog 中选择后续 first plan "
                "可能需要的 API 子集。不要生成计划，不要调用工具，只输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请选择 {GLOBAL_RETRIEVAL_MIN_APIS} 到 {GLOBAL_RETRIEVAL_MAX_APIS} 个 API。"
                "需要覆盖任务可能需要的主要工具，也要考虑用户偏好会影响的工具类别和参数。"
                "不要选择明显无关的 API。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 3000)}\n\n"
                f"tool_discovery_context:\n{compact_json(discovery_context, LONG_CONTEXT_MAX_JSON_CONTEXT_CHARS)}"
            ),
        },
    ]

def resolve_available_apis(
    *,
    client: LLMClient,
    logger: logging.Logger,
    user_id: str,
    task_id: str,
    task_doc: dict[str, Any],
    user_profile: dict[str, Any],
    hidden_info: dict[str, Any] | None,
    long_context: dict[str, Any] | None,
    user_sandbox: dict[str, Any],
    workflow: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    preference_mode: str,
) -> dict[str, Any]:
    mode = normalize_tool_availability_mode(TOOL_AVAILABILITY_MODE)
    oracle_names = [
        name
        for name in (workflow.get("available_api_names") or user_sandbox.get("accessible_api_names", []))
        if isinstance(name, str) and name in api_definitions
    ]
    if mode == "all":
        all_names = sorted(api_definitions)
        return {
            "mode": mode,
            "available_api_names": all_names,
            "oracle_api_names": oracle_names,
            "metadata": {
                "all_global_tools_visible": True,
                "global_api_count": len(all_names),
                "oracle_api_count": len(oracle_names),
            },
            "llm_attempts": [],
        }
    if mode == "oracle":
        return {
            "mode": mode,
            "available_api_names": oracle_names,
            "oracle_api_names": oracle_names,
            "metadata": {
                "oracle_api_count": len(oracle_names),
                "selected_api_count": len(oracle_names),
            },
            "llm_attempts": [],
        }
    if mode == "noisy":
        names, metadata = noisy_available_api_names(
            oracle_names=oracle_names,
            api_definitions=api_definitions,
            user_id=user_id,
            task_id=task_id,
        )
        return {
            "mode": mode,
            "available_api_names": names,
            "oracle_api_names": oracle_names,
            "metadata": metadata,
            "llm_attempts": [],
        }

    parsed, attempts = request_structured_json(
        client,
        messages=tool_discovery_messages(
            user_id=user_id,
            task_id=task_id,
            task_doc=task_doc,
            user_profile=user_profile,
            hidden_info=hidden_info,
            long_context=long_context,
            user_sandbox=user_sandbox,
            api_definitions=api_definitions,
            preference_mode=preference_mode,
        ),
        purpose=f"{user_id}/{task_id}/tool_discovery",
        validator=validate_tool_discovery_json(api_definitions),
        logger=logger,
        max_tokens=MODEL_MAX_TOKENS,
        temperature=MODEL_TEMPERATURE,
    )
    selected_names: list[str] = []
    seen: set[str] = set()
    for name in parsed.get("selected_api_names", []):
        if isinstance(name, str) and name in api_definitions and name not in seen:
            selected_names.append(name)
            seen.add(name)
    if len(selected_names) < GLOBAL_RETRIEVAL_MIN_APIS:
        filler = [name for name in oracle_names if name not in seen]
        selected_names.extend(filler[: GLOBAL_RETRIEVAL_MIN_APIS - len(selected_names)])
    selected_names = selected_names[:GLOBAL_RETRIEVAL_MAX_APIS]
    return {
        "mode": mode,
        "available_api_names": selected_names,
        "oracle_api_names": oracle_names,
        "metadata": {
            "oracle_api_count": len(oracle_names),
            "selected_api_count": len(selected_names),
            "llm_selected_count": len(parsed.get("selected_api_names", [])),
            "fallback_from_oracle_count": max(0, GLOBAL_RETRIEVAL_MIN_APIS - len(parsed.get("selected_api_names", []))),
            "selection_rationale": parsed.get("selection_rationale", []),
        },
        "llm_attempts": attempts,
    }

def long_context_path(turns: int) -> Path:
    resolved_turns = normalize_long_context_turns(turns)
    return LONG_CONTEXT_DIR / LONG_CONTEXT_FILE_TEMPLATE.format(turns=resolved_turns)

def load_long_context_doc(turns: int) -> dict[str, Any]:
    resolved_turns = normalize_long_context_turns(turns)
    with _LONG_CONTEXT_CACHE_LOCK:
        cached = _LONG_CONTEXT_CACHE.get(resolved_turns)
        if cached is not None:
            return cached
        path = long_context_path(resolved_turns)
        if not path.exists():
            raise RuntimeError(
                f"已开启长上下文但文件不存在: {path}。"
                "请先运行 scripts/build_lmsys_long_context.py 生成对应轮次。"
            )
        doc = read_json(path)
        if not isinstance(doc, dict):
            raise RuntimeError(f"长上下文文件顶层必须是 object: {path}")
        _LONG_CONTEXT_CACHE[resolved_turns] = doc
        return doc

def qa_pairs_from_long_context_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    qa_pairs = case.get("qa_pairs")
    if not isinstance(qa_pairs, list):
        raise RuntimeError("长上下文 case 缺少 qa_pairs 数组。")
    interactions: list[dict[str, Any]] = []
    for idx, pair in enumerate(qa_pairs, start=1):
        if not isinstance(pair, dict):
            raise RuntimeError(f"长上下文 qa_pairs[{idx}] 必须是 object。")
        user_text = pair.get("user")
        assistant_text = pair.get("assistant")
        if not isinstance(user_text, str) or not isinstance(assistant_text, str):
            raise RuntimeError(f"长上下文 qa_pairs[{idx}] 必须包含 string 类型 user/assistant。")
        interactions.append(
            {
                "interaction_id": f"long_context_{idx:03d}",
                "kind": "日常长上下文多轮问答",
                "说明": "以下是用户历史中的日常长上下文交互。原始对话可能来自多语言公开数据；评估时只作为长上下文干扰或历史背景，不代表当前任务偏好。",
                "source": {
                    "dataset": "LMSYS-Chat-1M",
                    "sample_id": pair.get("sample_id"),
                    "conversation_index": pair.get("conversation_index"),
                    "pair_index": pair.get("pair_index"),
                },
                "用户发言": user_text,
                "助手回复": assistant_text,
                "turns": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
            }
        )
    return interactions

def long_context_case_for_task(
    *,
    user_id: str,
    task_id: str,
    turns: int,
) -> dict[str, Any]:
    doc = load_long_context_doc(turns)
    cases = doc.get("cases")
    if not isinstance(cases, dict):
        raise RuntimeError(f"长上下文文件缺少 cases object: {long_context_path(turns)}")
    case_key = f"{user_id}/{task_id}"
    case = cases.get(case_key)
    if not isinstance(case, dict):
        raise RuntimeError(
            f"长上下文文件中缺少 case={case_key}: {long_context_path(turns)}。"
            "请确认 LongContext 数据按当前 data/user_xxx/task_xxx 完整生成。"
        )
    if int(case.get("turn_count") or -1) != normalize_long_context_turns(turns):
        raise RuntimeError(f"长上下文 case={case_key} turn_count 与 LONG_CONTEXT_TURNS 不一致。")
    return case

def build_long_context_for_task(
    *,
    user_id: str,
    task_id: str,
    preference_mode: str,
) -> dict[str, Any] | None:
    if not long_context_enabled_for_mode(preference_mode):
        return None
    turns = normalize_long_context_turns(LONG_CONTEXT_TURNS)
    case = long_context_case_for_task(user_id=user_id, task_id=task_id, turns=turns)
    interactions = qa_pairs_from_long_context_case(case)
    if len(interactions) != turns:
        raise RuntimeError(f"长上下文 case={user_id}/{task_id} 实际 QA 数 {len(interactions)} 与配置 {turns} 不一致。")
    return {
        "enabled": True,
        "turn_count": turns,
        "source_dataset": "LMSYS-Chat-1M",
        "source_file": str(long_context_path(turns).relative_to(PROJECT_ROOT)),
        "case_key": f"{user_id}/{task_id}",
        "sampling_seed": case.get("sampling_seed"),
        "qa_pairs": case.get("qa_pairs", []),
        "interactions": interactions,
    }

def neutral_user_information(value: Any) -> Any:
    if isinstance(value, dict):
        renamed: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            lower_key = clean_key.lower()
            if lower_key in {
                "source",
                "schema_version",
                "provenance_note",
                "seed_field",
                "enrichment",
                "generation_method",
                "generation_prompt",
                "generation_metadata",
            }:
                continue
            replacements = {
                "explicit_preferences": "user_information",
                "assistant_interaction_preferences": "assistant_interaction_style",
                "privacy_preferences": "privacy_notes",
                "budget_preferences": "budget_notes",
                "technology_preferences": "technology_notes",
                "food_preferences": "food_notes",
                "travel_preferences": "travel_notes",
                "learning_preferences": "learning_notes",
                "content_preferences": "content_notes",
                "product_preferences": "product_notes",
                "service_preferences": "service_notes",
                "entertainment_preferences": "entertainment_notes",
                "communication_preferences": "communication_notes",
                "preference_conflicts": "note_conflicts",
            }
            clean_key = replacements.get(clean_key, clean_key.replace("preferences", "notes").replace("preference", "note"))
            renamed[clean_key] = neutral_user_information(item)
        return renamed
    if isinstance(value, list):
        return [neutral_user_information(item) for item in value]
    return copy.deepcopy(value)

def user_information_for_prompt(user_profile: dict[str, Any]) -> dict[str, Any]:
    explicit = user_profile.get("explicit_preferences")
    if isinstance(explicit, dict):
        return neutral_user_information(explicit)
    return neutral_user_information(user_profile)

def neutral_interaction(raw: dict[str, Any], index: int) -> dict[str, Any]:
    turns = raw.get("turns") if isinstance(raw.get("turns"), list) else []
    clean_turns: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(role, str) and isinstance(content, str):
            clean_turns.append({"role": role, "content": content})
    return {
        "interaction_id": f"interaction_{index:03d}",
        "turns": clean_turns,
    }

def shuffled_user_interactions(
    *,
    hidden_info: dict[str, Any] | None,
    long_context: dict[str, Any] | None,
    user_id: str,
    task_id: str,
    mode: str,
) -> list[dict[str, Any]]:
    history_items = list(hidden_info.get("interactions", [])) if isinstance(hidden_info, dict) else []
    daily_items = list(long_context.get("interactions", [])) if isinstance(long_context, dict) else []
    combined: list[dict[str, Any]] = []
    if history_items and daily_items:
        combined = history_items + daily_items
        seed = f"{user_id}|{task_id}|{mode}|{long_context.get('turn_count')}|{long_context.get('sampling_seed')}"
        random.Random(seed).shuffle(combined)
    else:
        combined = history_items or daily_items
    return [
        neutral_interaction(item, idx)
        for idx, item in enumerate(combined, start=1)
        if isinstance(item, dict)
    ]

def pag_tokenize(text: str) -> list[str]:
    lowered = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9][a-z0-9_.:/+-]{1,}", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    cjk_tokens: list[str] = []
    for n in (2, 3):
        cjk_tokens.extend("".join(cjk_chars[idx : idx + n]) for idx in range(0, max(0, len(cjk_chars) - n + 1)))
    return ascii_tokens + cjk_tokens

def task_text_for_pag(task_doc: dict[str, Any]) -> str:
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
    parts: list[str] = []
    for key in keys:
        value = task.get(key) if isinstance(task, dict) else None
        if value not in (None, "", [], {}):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=False))
    return "\n".join(parts)

def hidden_interaction_text_for_pag(interaction: dict[str, Any]) -> str:
    parts = [
        str(interaction.get("kind") or ""),
        str(interaction.get("implicit_signal") or ""),
    ]
    for turn in interaction.get("turns", []) if isinstance(interaction.get("turns"), list) else []:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(part for part in parts if part)

def pag_interaction_qa_record(interaction: dict[str, Any], *, rank: int, score: float) -> dict[str, Any]:
    qa_turns: list[dict[str, str]] = []
    for turn in interaction.get("turns", []) if isinstance(interaction.get("turns"), list) else []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(role, str) and isinstance(content, str):
            qa_turns.append({"role": role, "content": content})
    return {
        "rank": rank,
        "score": round(score, 6),
        "interaction_id": interaction.get("interaction_id"),
        "topic": interaction.get("implicit_signal") or interaction.get("kind"),
        "historical_qa": qa_turns,
        "implicit_signal": interaction.get("implicit_signal"),
    }

def pag_retrieve_hidden_history(
    *,
    hidden_info: dict[str, Any],
    task_doc: dict[str, Any],
    top_k: int = PAG_RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    interactions = [
        item
        for item in hidden_info.get("interactions", [])
        if isinstance(item, dict)
    ]
    if top_k <= 0 or not interactions:
        return {
            "retrieval_method": "bm25_hidden_interaction_topk",
            "top_k": max(0, top_k),
            "retrieved_records": [],
        }

    docs = [pag_tokenize(hidden_interaction_text_for_pag(item)) for item in interactions]
    query_tokens = pag_tokenize(task_text_for_pag(task_doc))
    if not query_tokens:
        selected = [
            pag_interaction_qa_record(item, rank=idx, score=0.0)
            for idx, item in enumerate(interactions[:top_k], start=1)
        ]
        return {
            "retrieval_method": "bm25_hidden_interaction_topk",
            "top_k": top_k,
            "query_token_count": 0,
            "retrieved_records": selected,
        }

    doc_freq: Counter[str] = Counter()
    doc_counters: list[Counter[str]] = []
    for tokens in docs:
        counter = Counter(tokens)
        doc_counters.append(counter)
        doc_freq.update(counter.keys())

    doc_count = len(docs)
    avg_len = sum(len(tokens) for tokens in docs) / max(1, doc_count)
    query_counter = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (interaction, counter, tokens) in enumerate(zip(interactions, doc_counters, docs)):
        doc_len = max(1, len(tokens))
        score = 0.0
        for token, query_weight in query_counter.items():
            freq = counter.get(token, 0)
            if freq <= 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1e-9))
            score += query_weight * idf * (freq * (k1 + 1) / denom)
        scored.append((score, idx, interaction))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [
        pag_interaction_qa_record(interaction, rank=rank, score=score)
        for rank, (score, _, interaction) in enumerate(scored[:top_k], start=1)
    ]
    return {
        "retrieval_method": "bm25_hidden_interaction_topk",
        "top_k": top_k,
        "query_token_count": len(query_tokens),
        "candidate_count": len(interactions),
        "retrieved_records": selected,
    }

def load_pag_abstract_doc() -> dict[str, Any]:
    global _PAG_ABSTRACT_CACHE
    with _PAG_ABSTRACT_CACHE_LOCK:
        if _PAG_ABSTRACT_CACHE is not None:
            return _PAG_ABSTRACT_CACHE
        if not PAG_ABSTRACT_PATH.exists():
            raise RuntimeError(
                f"PAG 模式缺少用户摘要文件: {PAG_ABSTRACT_PATH}。"
                "请先运行 python eval/pag/gen_absract.py 生成 abstract.json。"
            )
        doc = read_json(PAG_ABSTRACT_PATH)
        if not isinstance(doc, dict):
            raise RuntimeError(f"PAG 摘要文件顶层必须是 object: {PAG_ABSTRACT_PATH}")
        _PAG_ABSTRACT_CACHE = doc
        return doc

def pag_prompt_abstract_for_task(*, user_id: str, task_id: str) -> dict[str, Any] | str:
    doc = load_pag_abstract_doc()
    users = doc.get("users")
    if not isinstance(users, dict) or not isinstance(users.get(user_id), dict):
        raise RuntimeError(f"PAG 摘要文件缺少用户 {user_id}: {PAG_ABSTRACT_PATH}")
    user_entry = users[user_id]
    task_summaries = user_entry.get("task_summaries")
    if isinstance(task_summaries, dict) and isinstance(task_summaries.get(task_id), dict):
        raw_summary: Any = task_summaries[task_id]
    else:
        raw_summary = user_entry

    if isinstance(raw_summary, str):
        return raw_summary
    if not isinstance(raw_summary, dict):
        return str(raw_summary)

    allowed_keys = [
        "task_title",
        "task_related_user_abstract",
        "preference_points",
        "constraints_to_preserve",
        "tool_call_implications",
        "risk_warnings",
        "language_and_format_preferences",
        "summary",
    ]
    prompt_summary = {
        key: raw_summary.get(key)
        for key in allowed_keys
        if raw_summary.get(key) not in (None, "", [], {})
    }
    if not prompt_summary and "summary" in user_entry:
        prompt_summary["summary"] = user_entry.get("summary")
    return prompt_summary

def build_pag_context(
    *,
    user_id: str,
    task_id: str,
    hidden_info: dict[str, Any],
    task_doc: dict[str, Any],
) -> dict[str, Any]:
    retrieved = pag_retrieve_hidden_history(
        hidden_info=hidden_info,
        task_doc=task_doc,
        top_k=PAG_RETRIEVAL_TOP_K,
    )
    user_abstract = pag_prompt_abstract_for_task(user_id=user_id, task_id=task_id)
    return {
        "retrieved_hidden_history": retrieved,
        "user_abstract": user_abstract,
        "prompt_injection_policy": (
            "在 initial_plan、每次 reason_action 和每次 update_plan 阶段的 prompt 开头加入 PAG 上下文："
            "与任务最相关 topic 的历史 QA 记录；每个用户的任务相关摘要。"
        ),
    }

def load_bank_memory_doc() -> dict[str, Any]:
    global _BANK_MEMORY_CACHE
    with _BANK_MEMORY_CACHE_LOCK:
        if _BANK_MEMORY_CACHE is not None:
            return _BANK_MEMORY_CACHE
        if not BANK_MEMORY_PATH.exists():
            raise RuntimeError(
                f"BANK 模式缺少记忆文件: {BANK_MEMORY_PATH}。"
                "请先运行 python eval/bank/eval_bank.py，并保持 BUILD_BANK_MEMORY=True。"
            )
        doc = read_json(BANK_MEMORY_PATH)
        if not isinstance(doc, dict) or not isinstance(doc.get("users"), dict):
            raise RuntimeError(f"BANK 记忆文件必须包含 users object: {BANK_MEMORY_PATH}")
        _BANK_MEMORY_CACHE = doc
        return doc

def bank_user_memory(user_id: str) -> dict[str, Any]:
    users = load_bank_memory_doc().get("users", {})
    entry = users.get(user_id) if isinstance(users, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(f"BANK 记忆文件缺少用户 {user_id}: {BANK_MEMORY_PATH}")
    if not isinstance(entry.get("abstract_profile"), dict):
        raise RuntimeError(f"BANK 用户 {user_id} 缺少 abstract_profile。")
    if not isinstance(entry.get("qa_memories"), list):
        raise RuntimeError(f"BANK 用户 {user_id} 缺少 qa_memories。")
    return entry

def bank_memory_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(record.get(key) or "")
        for key in ("question", "answer", "summary")
    )

def bank_retrieve_qa_memories(
    *,
    user_id: str,
    task_doc: dict[str, Any],
    top_k: int = BANK_RETRIEVAL_TOP_K,
) -> dict[str, Any]:
    entry = bank_user_memory(user_id)
    memories = [item for item in entry.get("qa_memories", []) if isinstance(item, dict)]
    if top_k <= 0 or not memories:
        return {
            "retrieval_method": "bm25_question_answer_summary_topk",
            "top_k": max(0, top_k),
            "candidate_count": len(memories),
            "retrieved_records": [],
        }

    docs = [pag_tokenize(bank_memory_text(item)) for item in memories]
    query_tokens = pag_tokenize(task_text_for_pag(task_doc))
    doc_freq: Counter[str] = Counter()
    doc_counters: list[Counter[str]] = []
    for tokens in docs:
        counter = Counter(tokens)
        doc_counters.append(counter)
        doc_freq.update(counter.keys())

    doc_count = len(docs)
    avg_len = sum(len(tokens) for tokens in docs) / max(1, doc_count)
    query_counter = Counter(query_tokens)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (memory, counter, tokens) in enumerate(zip(memories, doc_counters, docs)):
        score = 0.0
        doc_len = max(1, len(tokens))
        for token, query_weight in query_counter.items():
            freq = counter.get(token, 0)
            if freq <= 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1e-9))
            score += query_weight * idf * (freq * 2.5 / denom)
        scored.append((score, idx, memory))

    scored.sort(key=lambda item: (-item[0], item[1]))
    retrieved: list[dict[str, Any]] = []
    for rank, (score, _, memory) in enumerate(scored[:top_k], start=1):
        retrieved.append(
            {
                "rank": rank,
                "score": round(score, 6),
                "interaction_id": memory.get("interaction_id"),
                "question": memory.get("question"),
                "answer": memory.get("answer"),
                "summary": memory.get("summary"),
            }
        )
    return {
        "retrieval_method": "bm25_question_answer_summary_topk",
        "top_k": top_k,
        "query_token_count": len(query_tokens),
        "candidate_count": len(memories),
        "retrieved_records": retrieved,
    }

def build_bank_context(*, user_id: str, task_id: str, task_doc: dict[str, Any]) -> dict[str, Any]:
    entry = bank_user_memory(user_id)
    return {
        "user_abstract_profile": copy.deepcopy(entry["abstract_profile"]),
        "retrieved_qa_memories": bank_retrieve_qa_memories(
            user_id=user_id,
            task_doc=task_doc,
            top_k=BANK_RETRIEVAL_TOP_K,
        ),
        "context_order": ["user_abstract_profile", "retrieved_qa_memories"],
    }

def build_common_context(
    *,
    user_profile: dict[str, Any],
    hidden_info: dict[str, Any] | None,
    long_context: dict[str, Any] | None,
    user_sandbox: dict[str, Any],
    task_doc: dict[str, Any],
    workflow: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    preference_mode: str,
    tool_availability: dict[str, Any],
) -> dict[str, Any]:
    mode = normalize_preference_injection_mode(preference_mode)
    user_id = str(user_profile.get("user_id") or "unknown_user")
    task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    task_id = str(task.get("task_id") or workflow.get("task_id") or "unknown_task")
    available_api_names = tool_availability.get("available_api_names") or workflow.get("available_api_names") or user_sandbox.get("accessible_api_names", [])
    available_apis = [
        compact_api_definition(api_definitions[name])
        for name in available_api_names
        if name in api_definitions
    ]
    sandbox_user_context = user_sandbox.get("user_context", {})
    if mode in {"hidden", "pag", "bank"} and isinstance(sandbox_user_context, dict):
        sandbox_user_context = redacted_user_context_for_hidden(sandbox_user_context)

    context: dict[str, Any] = {}
    if mode == "bank":
        context["bank_context"] = build_bank_context(
            user_id=user_id,
            task_id=task_id,
            task_doc=task_doc,
        )
    context.update({
        "user_context": neutral_user_information(sandbox_user_context),
        "task": task_context_for_mode(task_doc, mode),
        "available_api_names": available_api_names,
        "available_api_contracts": available_apis,
        "tool_catalog": {
            "visibility": "all_available_tools",
            "tool_count": len(available_api_names),
            "schemas_included": True,
        },
    })
    if mode in {"direct", "mix"}:
        context["user_information"] = user_information_for_prompt(user_profile)
    if mode in {"hidden", "mix", "pag", "bank"}:
        if not isinstance(hidden_info, dict):
            raise RuntimeError("hidden/mix/pag/bank 模式需要每个用户目录下存在 hidden_info.json。")
    # PAG only exposes its task-retrieved Top-K history through pag_context;
    # the full hidden interaction history remains exclusive to hidden/mix.
    interactions = shuffled_user_interactions(
        hidden_info=hidden_info if mode in {"hidden", "mix"} else None,
        long_context=long_context,
        user_id=user_id,
        task_id=task_id,
        mode=mode,
    )
    if interactions:
        context["user_interaction_history"] = {
            "interaction_count": len(interactions),
            "interactions": interactions,
        }
    if mode == "pag":
        if not isinstance(hidden_info, dict):
            raise RuntimeError("pag 模式需要每个用户目录下存在 hidden_info.json。")
        context["pag_context"] = build_pag_context(
            user_id=user_id,
            task_id=task_id,
            hidden_info=hidden_info,
            task_doc=task_doc,
        )
    return context

def system_prompt(preference_mode: str = PREFERENCE_INJECTION_MODE) -> str:
    normalize_preference_injection_mode(preference_mode)
    return plan_act_system_prompt(preference_mode)

def _context_without(context: dict[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {key: value for key, value in context.items() if key not in excluded}

def _prepend_pag_prompt(messages: list[dict[str, str]], context: dict[str, Any]) -> list[dict[str, str]]:
    pag_context = context.get("pag_context") if isinstance(context.get("pag_context"), dict) else None
    if not pag_context:
        return messages
    retrieved = (pag_context.get("retrieved_hidden_history") or {}).get("retrieved_records", [])
    user_abstract = pag_context.get("user_abstract")
    pag_prompt = (
        "与从 hidden info 中检索到与该任务最相关的 topic 的历史 QA 记录："
        f"{compact_json(retrieved, MAX_JSON_CONTEXT_CHARS)}\n"
        "每个用户的用户摘要："
        f"{compact_json(user_abstract, MAX_JSON_CONTEXT_CHARS)}\n\n"
    )
    for message in messages:
        if message.get("role") == "user":
            message["content"] = pag_prompt + str(message.get("content") or "")
            break
    return messages

def initial_plan_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    context_for_prompt = _context_without(context, "bank_context", "pag_context")
    messages = plan_act_initial_plan.build_messages(
        context=context_for_prompt,
        compact_json=compact_json,
        context_json_limit=context_json_limit,
        max_plan_steps=MAX_PLAN_STEPS_IN_PROMPT,
        system_content=system_prompt(),
    )
    _prepend_pag_prompt(messages, context)
    bank_context = context.get("bank_context") if isinstance(context.get("bank_context"), dict) else None
    if bank_context and len(messages) >= 2 and messages[1].get("role") == "user":
        retrieved = (bank_context.get("retrieved_qa_memories") or {}).get("retrieved_records", [])
        bank_prompt = (
            "首先提供基于该用户 hidden info 推断得到的用户抽象画像："
            f"{compact_json(bank_context.get('user_abstract_profile'), MAX_JSON_CONTEXT_CHARS)}\n"
            "然后提供从该用户 question/answer/summary 记忆库中检索到的任务相关记录："
            f"{compact_json(retrieved, MAX_JSON_CONTEXT_CHARS)}\n\n"
        )
        messages[1]["content"] = bank_prompt + messages[1]["content"]
    return messages

def action_messages(
    *,
    context: dict[str, Any],
    current_plan: dict[str, Any],
    trajectory_summary: list[dict[str, Any]],
    iteration: int,
) -> list[dict[str, str]]:
    messages = plan_act_action.build_messages(
        context=_context_without(current_action_context(context, current_plan), "pag_context"),
        current_plan=current_plan,
        trajectory_summary=trajectory_summary,
        iteration=iteration,
        compact_json=compact_json,
        context_json_limit=context_json_limit,
        system_content=system_prompt(),
    )
    return _prepend_pag_prompt(messages, context)

def update_plan_messages(
    *,
    context: dict[str, Any],
    previous_plan: dict[str, Any],
    action_json: dict[str, Any],
    sandbox_observation: dict[str, Any],
    iteration: int,
) -> list[dict[str, str]]:
    prompt_observation = (
        filter_sandbox_observation_metadata_for_prompt(sandbox_observation)
        if FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
        else sandbox_observation
    )
    messages = plan_act_update_plan.build_messages(
        context=_context_without(context, "pag_context"),
        previous_plan=previous_plan,
        action_json=action_json,
        sandbox_observation=prompt_observation,
        iteration=iteration,
        compact_json=compact_json,
        context_json_limit=context_json_limit,
        max_plan_steps=MAX_PLAN_STEPS_IN_PROMPT,
        max_tool_observation_chars=MAX_TOOL_OBSERVATION_CHARS_IN_PROMPT,
        system_content=system_prompt(),
    )
    return _prepend_pag_prompt(messages, context)

def failed_action_observation(
    *,
    user_id: str,
    task_id: str,
    iteration: int,
    error: StructuredOutputError,
) -> dict[str, Any]:
    return {
        "sandbox_status": "error",
        "api_name": "__action_json_generation__",
        "action": "reason_action_json_generation",
        "task_context": task_id,
        "transport": {
            "http_status": 422,
            "request_id": f"req_action_json_generation_{user_id}_{task_id}_{iteration:03d}",
            "endpoint": "/internal/reason_action/json_generation",
        },
        "request_echo": {
            "function": "__action_json_generation__",
            "arguments": {},
            "idempotency_key": f"idm_action_json_generation_{user_id}_{task_id}_{iteration:03d}",
        },
        "validation": {
            "schema_valid": False,
            "errors": [str(error)],
            "warnings": [],
            "missing_required": [],
            "unexpected_arguments": [],
            "type_warnings": [],
        },
        "context_effects": [
            "reason_action 未能在一次生成加一次 repair 内产生合法 JSON object。",
            "按评估规则，本轮 action 标记为执行失败，并进入 update_plan。",
        ],
        "warnings": [],
        "errors": [str(error)],
        "response_body": {
            "summary": "reason_action JSON 生成失败，未执行任何外部 SandboxDB API。",
            "attempt_count": len(error.attempts),
        },
        "feedback": {
            "machine_readable_summary": "invalid_reason_action_json",
            "planner_relevance": "update_plan_after_failed_action_generation",
            "requires_user_confirmation": False,
            "missing_information": ["valid_reason_action_json"],
            "next_suggested_api_calls": [],
        },
        "raw_response_shape": {"summary": "string", "attempt_count": "integer"},
    }

def summarize_trajectory_for_prompt(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for event in trajectory[-MAX_RECENT_TRAJECTORY_EVENTS_IN_PROMPT:]:
        phase = event.get("phase")
        if phase == "initial_plan":
            # current_plan 已单独进入 reason_action prompt，不重复携带初始计划全文。
            continue
        item: dict[str, Any] = {"phase": phase, "iteration": event.get("iteration")}
        if phase == "update_plan":
            parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
            item["update"] = {
                "plan_version": parsed.get("plan_version"),
                "task_status": parsed.get("task_status"),
                "update_reason": compact_text(parsed.get("update_reason"), 500),
                "next_step_id": parsed.get("next_step_id"),
            }
        elif phase == "reason_action":
            parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
            tool_call = parsed.get("tool_call") if isinstance(parsed.get("tool_call"), dict) else None
            item["action"] = {
                "selected_step_id": parsed.get("selected_step_id"),
                "action_type": parsed.get("action_type"),
                "reason": compact_text(parsed.get("reason"), 240),
                "tool_call": tool_call,
            }
        elif phase == "sandbox_observation":
            observation = compact_sandbox_observation_for_ledger(event.get("observation"))
            item["observation"] = observation
        else:
            continue
        summary.append(item)
    return summary

def case_id_from_record(record: dict[str, Any]) -> str:
    return (
        f"{record.get('run_id')}::{record.get('user_id')}::{record.get('task_id')}"
        f"::pass_{record.get('pass_n')}_{record.get('pass_index')}"
    )

def preference_source_description(preference_mode: str) -> str:
    mode = normalize_preference_injection_mode(preference_mode)
    if mode == "hidden":
        return "用户交互历史中呈现的隐式偏好和行为倾向"
    if mode == "pag":
        return "PAG 检索历史 QA 与任务相关用户摘要中呈现的偏好和行为倾向"
    if mode == "bank":
        return "BANK 用户抽象画像及从 question/answer/summary 记忆库检索的任务相关记录中呈现的偏好和行为倾向"
    if mode == "mix":
        return "显式用户画像与用户交互历史中共同呈现的偏好和行为倾向"
    return "显式用户画像中呈现的用户偏好和约束"

def offline_judge_entrypoint_for_mode(preference_mode: str) -> str:
    mode = normalize_preference_injection_mode(preference_mode)
    if mode == "hidden":
        return "python eval/judge_hidden.py"
    if mode == "mix":
        return "python eval/judge_mix.py"
    if mode == "pag":
        return "python eval/pag/judge_pag.py"
    if mode == "bank":
        return "python eval/bank/judge_bank.py"
    return "python eval/judge_hidden.py"

def judge_context_for_prompt(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    available_api_names = context.get("available_api_names") if isinstance(context.get("available_api_names"), list) else []
    return {
        "case_id": case_id_from_record(record),
        "user_id": record.get("user_id"),
        "task_id": record.get("task_id"),
        "pass_n": record.get("pass_n"),
        "pass_index": record.get("pass_index"),
        "task_title": record.get("task_title"),
        "preference_injection_mode": record.get("preference_injection_mode"),
        "preference_injection_label": record.get("preference_injection_label"),
        "preference_source": preference_source_description(str(record.get("preference_injection_mode") or PREFERENCE_INJECTION_MODE)),
        "task": context.get("task"),
        "user_information": context.get("user_information"),
        "user_interaction_history": context.get("user_interaction_history"),
        "bank_context": context.get("bank_context"),
        "sandbox_user_context": context.get("user_context"),
        "available_api_names": available_api_names,
        "available_api_contracts": [
            compact_api_definition(api_definitions[name])
            for name in available_api_names
            if name in api_definitions
        ],
        "judge_policy": {
            "llm_as_judge_only": True,
            "preference_relevance_rule": (
                "只在工具、步骤、参数或更新语义与用户偏好/任务约束相关时要求体现偏好；"
                "无关参数不因未体现偏好扣分，但不能违反已知偏好或编造个人信息。"
            ),
        },
    }

def plan_steps(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]

def planned_tool_actions(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in plan_steps(plan):
        target_api = step.get("target_api")
        if not isinstance(target_api, str) or not target_api.strip():
            continue
        planned_arguments = None
        for key in ("target_arguments", "planned_arguments", "tool_arguments", "arguments"):
            value = step.get(key)
            if isinstance(value, dict):
                planned_arguments = value
                break
        actions.append(
            {
                "planned_action_index": len(actions) + 1,
                "step_id": step.get("step_id"),
                "description": step.get("description"),
                "status": step.get("status"),
                "depends_on": step.get("depends_on"),
                "target_api": target_api,
                "planned_arguments": planned_arguments,
                "success_criteria": step.get("success_criteria"),
                "evidence": step.get("evidence"),
            }
        )
    return actions

def validate_planned_arguments_schema(
    *,
    action: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    api_name = action.get("target_api")
    planned_arguments = action.get("planned_arguments")
    errors: list[str] = []
    warnings: list[str] = []
    missing_required: list[str] = []
    unexpected_arguments: list[str] = []
    type_warnings: list[str] = []

    if not isinstance(api_name, str) or not api_name:
        errors.append("target_api 必须是非空字符串。")
        return {
            "schema_valid": False,
            "errors": errors,
            "warnings": warnings,
            "missing_required": missing_required,
            "unexpected_arguments": unexpected_arguments,
            "type_warnings": type_warnings,
        }

    api_def = api_definitions.get(api_name)
    if not api_def:
        errors.append(f"target_api 在 API definitions 中不存在: {api_name}")
        return {
            "schema_valid": False,
            "errors": errors,
            "warnings": warnings,
            "missing_required": missing_required,
            "unexpected_arguments": unexpected_arguments,
            "type_warnings": type_warnings,
        }

    if planned_arguments is None:
        planned_arguments = {}
    if not isinstance(planned_arguments, dict):
        errors.append("planned_arguments 必须是 object 或 null。")
        planned_arguments = {}

    params = api_def.get("interface", {}).get("parameters", [])
    param_names = {param.get("name") for param in params}
    for param in params:
        name = param.get("name")
        if not isinstance(name, str) or not name:
            continue
        if param.get("required") and name not in planned_arguments:
            missing_required.append(name)
            continue
        if name in planned_arguments and planned_arguments[name] is not None:
            warning = SandboxExecutor._type_warning(name, planned_arguments[name], str(param.get("type", "")))
            if warning:
                type_warnings.append(warning)

    unexpected_arguments = sorted(str(name) for name in planned_arguments if name not in param_names)
    if missing_required:
        errors.append(f"缺少 API schema 必填 planned_arguments: {missing_required}")
    if unexpected_arguments:
        warnings.append(f"planned_arguments 包含 API schema 未定义参数: {unexpected_arguments}")
    warnings.extend(type_warnings)

    return {
        "schema_valid": not errors and not type_warnings,
        "errors": errors,
        "warnings": warnings,
        "missing_required": missing_required,
        "unexpected_arguments": unexpected_arguments,
        "type_warnings": type_warnings,
    }

def attach_planned_argument_schema_checks(
    *,
    actions: list[dict[str, Any]],
    api_definitions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    checked_actions: list[dict[str, Any]] = []
    for action in actions:
        checked = copy.deepcopy(action)
        checked["planned_arguments_schema_check"] = validate_planned_arguments_schema(
            action=checked,
            api_definitions=api_definitions,
        )
        checked_actions.append(checked)
    return checked_actions

def reason_action_by_iteration(record: dict[str, Any], iteration: int) -> dict[str, Any]:
    for event in record.get("trajectory", []) if isinstance(record.get("trajectory"), list) else []:
        if not isinstance(event, dict) or event.get("phase") != "reason_action":
            continue
        if event.get("iteration") == iteration and isinstance(event.get("parsed"), dict):
            return event["parsed"]
    return {}

def update_events_for_final_judge(record: dict[str, Any]) -> list[dict[str, Any]]:
    events_by_iteration: dict[int, dict[str, Any]] = {}
    previous_plan: dict[str, Any] | None = None
    for plan in record.get("plan_versions", []) if isinstance(record.get("plan_versions"), list) else []:
        if isinstance(plan, dict):
            previous_plan = plan
            break

    for event in record.get("trajectory", []) if isinstance(record.get("trajectory"), list) else []:
        if not isinstance(event, dict):
            continue
        iteration = event.get("iteration")
        if not isinstance(iteration, int) or iteration <= 0:
            continue
        item = events_by_iteration.setdefault(iteration, {"iteration": iteration})
        phase = event.get("phase")
        if phase == "reason_action":
            parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
            item["reason_action"] = {
                "selected_step_id": parsed.get("selected_step_id"),
                "reason": compact_text(parsed.get("reason")),
                "action_type": parsed.get("action_type"),
                "tool_call": parsed.get("tool_call"),
                "completion_assessment": parsed.get("completion_assessment"),
            }
        elif phase == "sandbox_observation":
            observation = copy.deepcopy(event.get("observation"))
            if isinstance(observation, dict) and "response_body" in observation:
                observation["response_body"] = compact_json(observation["response_body"], MAX_JUDGE_OBSERVATION_CHARS)
            item["sandbox_observation"] = observation
        elif phase == "update_plan":
            item["updated_plan"] = event.get("parsed")

    updates: list[dict[str, Any]] = []
    for iteration in sorted(events_by_iteration):
        item = events_by_iteration[iteration]
        if not isinstance(item.get("updated_plan"), dict):
            continue
        updates.append(
            {
                "update_index": len(updates) + 1,
                "iteration": iteration,
                "previous_plan": previous_plan or {},
                "reason_action": item.get("reason_action", {}),
                "sandbox_observation": item.get("sandbox_observation", {}),
                "updated_plan": item.get("updated_plan", {}),
            }
        )
        previous_plan = item.get("updated_plan") if isinstance(item.get("updated_plan"), dict) else previous_plan
    return updates

def tool_call_inputs_for_final_judge(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, item in enumerate(record.get("tool_calls", []) if isinstance(record.get("tool_calls"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        tool_call = item.get("tool_call_input") if isinstance(item.get("tool_call_input"), dict) else {}
        iteration = item.get("iteration")
        action_json = reason_action_by_iteration(record, iteration) if isinstance(iteration, int) else {}
        calls.append(
            {
                "call_index": idx,
                "iteration": iteration,
                "selected_step_id": action_json.get("selected_step_id"),
                "reason": compact_text(action_json.get("reason")),
                "api_name": tool_call.get("name"),
                "arguments": tool_call.get("arguments"),
            }
        )
    return calls

FIRST_PLAN_CHECKLIST_KEYS = [
    "profile_preference_coverage",
    "preference_violation_avoidance",
]

def validate_inline_first_plan_judge(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    if bool_value(obj.get("first_acc")) is None:
        errors.append("first_acc 必须是 boolean。")
    checklist = obj.get("checklist")
    if not isinstance(checklist, dict):
        errors.append("checklist 必须是 object。")
        return errors
    missing = [key for key in FIRST_PLAN_CHECKLIST_KEYS if key not in checklist]
    if missing:
        errors.append(f"checklist 缺少指标: {missing}")
    for key in FIRST_PLAN_CHECKLIST_KEYS:
        item = checklist.get(key)
        if not isinstance(item, dict):
            errors.append(f"checklist.{key} 必须是 object。")
            continue
        if bool_value(item.get("pass")) is None:
            errors.append(f"checklist.{key}.pass 必须是 boolean。")
        if not isinstance(item.get("evidence"), list):
            errors.append(f"checklist.{key}.evidence 必须是数组。")
        if not isinstance(item.get("issues"), list):
            errors.append(f"checklist.{key}.issues 必须是数组。")
    return errors

def validate_inline_action_param_judge(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    for key in (
        "action_param_acc",
        "api_choice_preference_following",
        "parameter_preference_following",
        "task_constraint_alignment",
        "no_unsupported_sensitive_arguments",
    ):
        if bool_value(obj.get(key)) is None:
            errors.append(f"{key} 必须是 boolean。")
    for key in ("preference_relevant_parameters", "supporting_evidence", "issues", "violated_preferences"):
        if not isinstance(obj.get(key), list):
            errors.append(f"{key} 必须是数组。")
    return errors

def validate_inline_update_judge(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    for key in (
        "update_acc",
        "update_quality_correct",
        "update_preference_correct",
        "observation_integration",
        "status_and_evidence_update",
        "next_step_consistency",
        "no_fabrication_or_contradiction",
        "preference_retention",
        "preference_aware_replanning",
        "planned_arguments_preference_following",
        "no_preference_violation",
    ):
        if bool_value(obj.get(key)) is None:
            errors.append(f"{key} 必须是 boolean。")
    for key in ("supporting_evidence", "quality_issues", "preference_issues", "issues", "violated_preferences"):
        if not isinstance(obj.get(key), list):
            errors.append(f"{key} 必须是数组。")
    return errors

def validate_inline_trajectory_judge(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    for key in (
        "trajectory_acc",
        "task_completion",
        "personalization_consistency",
        "action_sequence_reasonable",
        "action_params_consistent",
        "updates_consistent",
        "final_answer_alignment",
    ):
        if bool_value(obj.get(key)) is None:
            errors.append(f"{key} 必须是 boolean。")
    for key in ("supporting_evidence", "issues"):
        if not isinstance(obj.get(key), list):
            errors.append(f"{key} 必须是数组。")
    return errors

def normalize_first_plan_judge(parsed: dict[str, Any]) -> dict[str, Any]:
    checklist = parsed.get("checklist") if isinstance(parsed.get("checklist"), dict) else {}
    normalized_checklist: dict[str, Any] = {}
    for key in FIRST_PLAN_CHECKLIST_KEYS:
        item = checklist.get(key) if isinstance(checklist.get(key), dict) else {}
        passed = bool_value(item.get("pass")) is True
        normalized_checklist[key] = {
            "pass": passed,
            "score": 1 if passed else 0,
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
            "issues": item.get("issues") if isinstance(item.get("issues"), list) else [],
        }
    first_acc = normalized_checklist.get("preference_violation_avoidance", {}).get("pass") is True
    return {
        "case_id": parsed.get("case_id"),
        "first_acc": first_acc,
        "first_plan_preference_following": first_acc,
        "checklist": normalized_checklist,
        "summary": parsed.get("summary", ""),
        "confidence": parsed.get("confidence"),
    }

def normalize_action_param_judge(
    *,
    parsed: dict[str, Any],
    iteration: int,
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "api_choice_preference_following": bool_value(parsed.get("api_choice_preference_following")) is True,
        "parameter_preference_following": bool_value(parsed.get("parameter_preference_following")) is True,
        "task_constraint_alignment": bool_value(parsed.get("task_constraint_alignment")) is True,
        "no_unsupported_sensitive_arguments": bool_value(parsed.get("no_unsupported_sensitive_arguments")) is True,
    }
    action_param_acc = bool_value(parsed.get("action_param_acc")) is True and all(checks.values())
    return {
        "iteration": iteration,
        "api_name": tool_call.get("name"),
        "arguments": copy.deepcopy(tool_call.get("arguments")),
        "action_param_acc": action_param_acc,
        **checks,
        "preference_relevant_parameters": parsed.get("preference_relevant_parameters")
        if isinstance(parsed.get("preference_relevant_parameters"), list)
        else [],
        "supporting_evidence": parsed.get("supporting_evidence") if isinstance(parsed.get("supporting_evidence"), list) else [],
        "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
        "violated_preferences": parsed.get("violated_preferences") if isinstance(parsed.get("violated_preferences"), list) else [],
        "summary": parsed.get("summary", ""),
        "confidence": parsed.get("confidence"),
    }

def normalize_update_judge(
    *,
    parsed: dict[str, Any],
    iteration: int,
) -> dict[str, Any]:
    quality_checks = {
        "observation_integration": bool_value(parsed.get("observation_integration")) is True,
        "status_and_evidence_update": bool_value(parsed.get("status_and_evidence_update")) is True,
        "next_step_consistency": bool_value(parsed.get("next_step_consistency")) is True,
        "no_fabrication_or_contradiction": bool_value(parsed.get("no_fabrication_or_contradiction")) is True,
    }
    preference_checks = {
        "preference_retention": bool_value(parsed.get("preference_retention")) is True,
        "preference_aware_replanning": bool_value(parsed.get("preference_aware_replanning")) is True,
        "planned_arguments_preference_following": bool_value(parsed.get("planned_arguments_preference_following")) is True,
        "no_preference_violation": bool_value(parsed.get("no_preference_violation")) is True,
    }
    update_quality_correct = all(quality_checks.values())
    update_preference_correct = all(preference_checks.values())
    update_acc = update_quality_correct and update_preference_correct
    quality_issues = parsed.get("quality_issues") if isinstance(parsed.get("quality_issues"), list) else []
    preference_issues = parsed.get("preference_issues") if isinstance(parsed.get("preference_issues"), list) else []
    issues = parsed.get("issues") if isinstance(parsed.get("issues"), list) else [*quality_issues, *preference_issues]
    return {
        "iteration": iteration,
        "update_acc": update_acc,
        "update_quality_correct": update_quality_correct,
        **quality_checks,
        "update_preference_correct": update_preference_correct,
        **preference_checks,
        # 兼容旧 inline 汇总字段。
        "plan_state_update": quality_checks["status_and_evidence_update"],
        "next_action_preference_consistency": preference_checks["preference_aware_replanning"],
        "action_parameter_guidance_preference_following": preference_checks["planned_arguments_preference_following"],
        "supporting_evidence": parsed.get("supporting_evidence") if isinstance(parsed.get("supporting_evidence"), list) else [],
        "quality_issues": quality_issues,
        "preference_issues": preference_issues,
        "issues": issues,
        "violated_preferences": parsed.get("violated_preferences") if isinstance(parsed.get("violated_preferences"), list) else [],
        "summary": parsed.get("summary", ""),
        "confidence": parsed.get("confidence"),
    }

def normalize_trajectory_judge(
    *,
    parsed: dict[str, Any],
    judge_results: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "task_completion": bool_value(parsed.get("task_completion")) is True,
        "personalization_consistency": bool_value(parsed.get("personalization_consistency")) is True,
        "action_sequence_reasonable": bool_value(parsed.get("action_sequence_reasonable")) is True,
        "action_params_consistent": bool_value(parsed.get("action_params_consistent")) is True,
        "updates_consistent": bool_value(parsed.get("updates_consistent")) is True,
        "final_answer_alignment": bool_value(parsed.get("final_answer_alignment")) is True,
    }
    action_param_case_acc = all_action_params_pass(judge_results)
    update_case_acc = all_updates_pass(judge_results)
    raw_trajectory_acc = bool_value(parsed.get("trajectory_acc")) is True and all(checks.values())
    trajectory_acc = raw_trajectory_acc and action_param_case_acc and update_case_acc
    return {
        "trajectory_acc": trajectory_acc,
        "raw_trajectory_acc": raw_trajectory_acc,
        "action_param_case_acc_required": action_param_case_acc,
        "update_case_acc_required": update_case_acc,
        **checks,
        "supporting_evidence": parsed.get("supporting_evidence") if isinstance(parsed.get("supporting_evidence"), list) else [],
        "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
        "summary": parsed.get("summary", ""),
        "confidence": parsed.get("confidence"),
    }

def first_plan_judge_messages(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    initial_plan: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "case_id": case_id_from_record(record),
        "first_acc": "boolean，主 First Acc；必须等于 preference_violation_avoidance.pass。profile_preference_coverage 只作诊断，不参与主分数",
        "checklist": {
            "profile_preference_coverage": {
                "pass": "boolean",
                "score": "0 or 1",
                "evidence": ["first_plan 中明确体现当前 task 相关用户偏好的证据；可为空"],
                "issues": ["未覆盖或弱覆盖的 task-relevant 偏好；仅作诊断，不直接扣 First Acc"],
            },
            "preference_violation_avoidance": {
                "pass": "boolean",
                "score": "0 or 1",
                "evidence": ["first_plan 没有违反隐私、预算、语言、格式、禁忌、风格等偏好的证据"],
                "issues": ["明确或隐含的偏好违背"],
            },
        },
        "summary": "简短中文总结",
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的旁路 LLM-as-judge。你只评估 first plan 是否遵循用户偏好，"
                "不评估后续执行是否成功，也不评估一般任务完整性、工具可执行性、API schema、证据质量或规划质量，"
                "除非这些内容本身是用户偏好。不要要求固定工具顺序。"
                "First Acc 只评价 first plan 是否与已知用户偏好冲突，不因未覆盖所有偏好而扣主分。"
                "必须保守打分，只输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请用两个偏好指标评估 first_plan，但 first_acc 只由 preference_violation_avoidance 决定："
                "只要 first_plan 没有明确或隐含违背已知用户偏好，first_acc 就应为 true。"
                "profile_preference_coverage 只用于诊断覆盖强弱，不参与 First Acc 主分数。"
                "profile_preference_coverage 只要求覆盖当前 task 相关偏好，不要求覆盖与当前任务无关的全部用户画像。"
                "preference_violation_avoidance 用于检查 first_plan 是否明确或隐含违反已知偏好；如有违反必须 fail。"
                "不要因为计划缺少一般任务细节、没有覆盖全部成功标准、API 选择不够完美、planned_arguments schema 不完整、"
                "没有主动加入证据验证步骤，或没有主动覆盖所有用户偏好而扣 first_acc；"
                "只有这些问题造成偏好冲突或违背时才扣 First Acc。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"judge_context:\n{compact_json(judge_context_for_prompt(record=record, context=context, api_definitions=api_definitions), MAX_JUDGE_CONTEXT_CHARS)}\n\n"
                f"first_plan:\n{compact_json(initial_plan, MAX_JUDGE_PLAN_CHARS)}\n\n"
                f"planned_tool_actions_for_reference_only:\n{compact_json(actions, MAX_JUDGE_PLAN_CHARS)}"
            ),
        },
    ]

def action_param_judge_messages(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    current_plan: dict[str, Any],
    action_json: dict[str, Any],
    tool_call: dict[str, Any],
    iteration: int,
) -> list[dict[str, str]]:
    api_name = tool_call.get("name")
    api_contract = compact_api_definition(api_definitions[api_name]) if isinstance(api_name, str) and api_name in api_definitions else {}
    schema = {
        "case_id": case_id_from_record(record),
        "iteration": iteration,
        "action_param_acc": "boolean，必须等于下面四个 boolean 的逻辑与",
        "api_choice_preference_following": "boolean",
        "parameter_preference_following": "boolean",
        "task_constraint_alignment": "boolean",
        "no_unsupported_sensitive_arguments": "boolean",
        "preference_relevant_parameters": [
            {
                "parameter": "string",
                "verdict": "pass|fail|not_relevant",
                "reason": "string",
            }
        ],
        "supporting_evidence": ["引用 action、参数、用户偏好、任务约束或 API contract"],
        "issues": ["失败时列出具体错误"],
        "violated_preferences": ["被违反或遗漏的具体偏好；无则空数组"],
        "summary": "简短中文总结",
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的旁路 LLM-as-judge。你只评估本次 reason_action 选择的工具和 tool_call.arguments "
                "是否符合用户偏好与任务约束。不要根据 sandbox observation 事后成败打分，也不要要求固定参考轨迹。"
                "只看当时给工具填写的参数是否合理、个性化、无偏好违背。只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请评估这一次工具调用参数。偏好相关性原则：只有当参数语义涉及用户的隐私、预算、时间、地点、语言、饮食、风格、"
                "证据标准、风险偏好或任务硬约束时，才要求该参数体现对应偏好；无关参数不能强行扣分。"
                "如果 API 有相关参数且上下文提供了必要偏好信息，参数应合理体现；如果参数编造敏感信息、遗漏关键偏好或与任务冲突，则判失败。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"judge_context:\n{compact_json(judge_context_for_prompt(record=record, context=context, api_definitions=api_definitions), MAX_JUDGE_CONTEXT_CHARS)}\n\n"
                f"current_plan:\n{compact_json(current_plan, MAX_JUDGE_PLAN_CHARS)}\n\n"
                f"reason_action:\n{compact_json(action_json, 9000)}\n\n"
                f"api_contract:\n{compact_json(api_contract, 6000)}\n\n"
                f"tool_call:\n{compact_json(tool_call, 8000)}"
            ),
        },
    ]

def update_plan_inline_judge_messages(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    previous_plan: dict[str, Any],
    action_json: dict[str, Any],
    sandbox_observation: dict[str, Any],
    updated_plan: dict[str, Any],
    iteration: int,
) -> list[dict[str, str]]:
    clipped_observation = copy.deepcopy(sandbox_observation)
    if isinstance(clipped_observation, dict) and "response_body" in clipped_observation:
        clipped_observation["response_body"] = compact_json(clipped_observation["response_body"], MAX_JUDGE_OBSERVATION_CHARS)
    schema = {
        "case_id": case_id_from_record(record),
        "iteration": iteration,
        "update_acc": "boolean，主 Update Acc；必须等于 update_quality_correct AND update_preference_correct",
        "update_quality_correct": "boolean，必须等于下面 4 个质量检查的逻辑与",
        "observation_integration": "boolean",
        "status_and_evidence_update": "boolean",
        "next_step_consistency": "boolean",
        "no_fabrication_or_contradiction": "boolean",
        "update_preference_correct": "boolean，必须等于下面 4 个偏好检查的逻辑与",
        "preference_retention": "boolean",
        "preference_aware_replanning": "boolean",
        "planned_arguments_preference_following": "boolean",
        "no_preference_violation": "boolean",
        "supporting_evidence": ["引用 previous_plan、action、observation、updated_plan 的具体证据"],
        "quality_issues": ["质量层失败时列出具体更新错误"],
        "preference_issues": ["偏好层失败时列出具体偏好遗漏或违背"],
        "issues": ["quality_issues 与 preference_issues 的合并摘要"],
        "violated_preferences": ["被违反或遗漏的具体偏好；无则空数组"],
        "summary": "简短中文总结",
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的旁路 LLM-as-judge。你只评估本次 update_plan 是否在看到工具 observation 后，"
                "正确更新计划，并让更新后的计划继续符合用户画像、任务约束和个性化偏好。不要要求固定参考轨迹；"
                "不要把 judge 结果反馈给被测 LLM。只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请评估这一次 update_plan。update_acc 必须同时满足更新质量和用户偏好遵守。\n\n"
                "更新质量检查：是否忠实吸收 observation；是否正确更新 step status、evidence、task_status 与风险；"
                "next_step_id 是否合理；是否没有虚构 observation 没有返回的信息或把失败当成功。\n"
                "用户偏好检查：是否保留 previous_plan 中仍适用的预算、地点、时间、语言、隐私、价值观、风险偏好、输出格式等约束；"
                "重规划或换 API 时是否优先选择符合用户偏好的替代方案；后续 planned_arguments 是否继续体现偏好约束；"
                "是否没有新增、保留或升级明显违背用户偏好的步骤、参数、假设或交付方向。"
                "如果某轮 update 没有新的偏好相关内容，只要没有丢失、削弱或违背已有偏好，可判偏好层通过。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"judge_context:\n{compact_json(judge_context_for_prompt(record=record, context=context, api_definitions=api_definitions), MAX_JUDGE_CONTEXT_CHARS)}\n\n"
                f"previous_plan:\n{compact_json(previous_plan, MAX_JUDGE_PLAN_CHARS)}\n\n"
                f"reason_action:\n{compact_json(action_json, 9000)}\n\n"
                f"sandbox_observation:\n{compact_json(clipped_observation, MAX_JUDGE_OBSERVATION_CHARS)}\n\n"
                f"updated_plan:\n{compact_json(updated_plan, MAX_JUDGE_PLAN_CHARS)}"
            ),
        },
    ]

def trajectory_inline_judge_messages(
    *,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    first_plan = record.get("plan_versions", [None])[0] if isinstance(record.get("plan_versions"), list) and record.get("plan_versions") else None
    trajectory_package = {
        "eval_status": record.get("status"),
        "first_plan": first_plan,
        "tool_call_inputs": tool_call_inputs_for_final_judge(record),
        "update_events": update_events_for_final_judge(record),
        "final_answer": record.get("final_answer"),
        "completion_assessment": record.get("completion_assessment"),
        "trajectory_pattern_check": record.get("trajectory_pattern_check"),
    }
    schema = {
        "case_id": case_id_from_record(record),
        "trajectory_acc": "boolean，必须等于下面六个 boolean 的逻辑与",
        "task_completion": "boolean",
        "personalization_consistency": "boolean",
        "action_sequence_reasonable": "boolean",
        "action_params_consistent": "boolean",
        "updates_consistent": "boolean",
        "final_delivery_quality": "boolean",
        "supporting_evidence": ["引用完整轨迹中的具体证据"],
        "issues": ["失败时列出具体问题"],
        "summary": "简短中文总结",
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的旁路 LLM-as-judge。你只评估完整规划与工具执行轨迹，"
                "不得把 judge 结论反馈给被测模型。只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请评估完整轨迹。trajectory_acc 不是只看最终回答，而是综合任务完成、个性化一致性、工具动作顺序、"
                "每次工具参数、每次 update_plan 和最终交付物。必须基于具体 action 和 update 的证据保守判定。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"judge_context:\n{compact_json(judge_context_for_prompt(record=record, context=context, api_definitions=api_definitions), MAX_JUDGE_CONTEXT_CHARS)}\n\n"
                f"trajectory_package:\n{compact_json(trajectory_package, MAX_JUDGE_TRAJECTORY_CHARS)}"
            ),
        },
    ]

def append_judge_error(judge_results: dict[str, Any], phase: str, exc: Exception) -> None:
    judge_results.setdefault("errors", []).append(
        {
            "phase": phase,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "attempts": getattr(exc, "attempts", []),
        }
    )

def judge_first_plan_inline(
    *,
    judge_client: LLMClient | None,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    initial_plan: dict[str, Any],
    logger: logging.Logger,
) -> None:
    judge_results = record.setdefault("judge_results", {"enabled": ENABLE_INLINE_JUDGE, "errors": []})
    actions = attach_planned_argument_schema_checks(
        actions=planned_tool_actions(initial_plan),
        api_definitions=api_definitions,
    )
    if not ENABLE_INLINE_JUDGE or judge_client is None:
        return
    try:
        parsed, attempts = request_structured_json(
            judge_client,
            messages=first_plan_judge_messages(
                record=record,
                context=context,
                api_definitions=api_definitions,
                initial_plan=initial_plan,
                actions=actions,
            ),
            purpose=f"{case_id_from_record(record)}/judge/first_plan",
            validator=validate_inline_first_plan_judge,
            logger=logger,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        judge_results["first_plan"] = {
            "parsed": normalize_first_plan_judge(parsed),
            "attempts": attempts,
        }
    except Exception as exc:  # noqa: BLE001 - judge must not affect eval.
        append_judge_error(judge_results, "first_plan", exc)
        judge_results["first_plan"] = {
            "parsed": {
                "case_id": case_id_from_record(record),
                "first_acc": False,
                "first_plan_preference_following": False,
                "checklist": {
                    key: {
                        "pass": False,
                        "score": 0,
                        "evidence": [],
                        "issues": [f"first plan judge 失败: {exc}"],
                    }
                    for key in FIRST_PLAN_CHECKLIST_KEYS
                },
                "summary": f"first plan judge 失败: {exc}",
                "confidence": 1.0,
            },
            "attempts": getattr(exc, "attempts", []),
        }

def judge_action_param_inline(
    *,
    judge_client: LLMClient | None,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    current_plan: dict[str, Any],
    action_json: dict[str, Any],
    tool_call: dict[str, Any],
    iteration: int,
    logger: logging.Logger,
) -> None:
    judge_results = record.setdefault("judge_results", {"enabled": ENABLE_INLINE_JUDGE, "errors": []})
    judge_results.setdefault("action_params", [])
    if not ENABLE_INLINE_JUDGE or judge_client is None:
        return
    try:
        parsed, attempts = request_structured_json(
            judge_client,
            messages=action_param_judge_messages(
                record=record,
                context=context,
                api_definitions=api_definitions,
                current_plan=current_plan,
                action_json=action_json,
                tool_call=tool_call,
                iteration=iteration,
            ),
            purpose=f"{case_id_from_record(record)}/judge/action_param/{iteration}",
            validator=validate_inline_action_param_judge,
            logger=logger,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        judge_results["action_params"].append(
            {
                "parsed": normalize_action_param_judge(parsed=parsed, iteration=iteration, tool_call=tool_call),
                "attempts": attempts,
            }
        )
    except Exception as exc:  # noqa: BLE001 - judge must not affect eval.
        append_judge_error(judge_results, f"action_param/{iteration}", exc)
        judge_results["action_params"].append(
            {
                "parsed": {
                    "iteration": iteration,
                    "api_name": tool_call.get("name"),
                    "arguments": copy.deepcopy(tool_call.get("arguments")),
                    "action_param_acc": False,
                    "api_choice_preference_following": False,
                    "parameter_preference_following": False,
                    "task_constraint_alignment": False,
                    "no_unsupported_sensitive_arguments": False,
                    "preference_relevant_parameters": [],
                    "supporting_evidence": [],
                    "issues": [f"action param judge 失败: {exc}"],
                    "violated_preferences": [],
                    "summary": f"action param judge 失败: {exc}",
                    "confidence": 1.0,
                },
                "attempts": getattr(exc, "attempts", []),
            }
        )

def judge_update_inline(
    *,
    judge_client: LLMClient | None,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    previous_plan: dict[str, Any],
    action_json: dict[str, Any],
    sandbox_observation: dict[str, Any],
    updated_plan: dict[str, Any],
    iteration: int,
    logger: logging.Logger,
) -> None:
    judge_results = record.setdefault("judge_results", {"enabled": ENABLE_INLINE_JUDGE, "errors": []})
    judge_results.setdefault("updates", [])
    if not ENABLE_INLINE_JUDGE or judge_client is None:
        return
    try:
        parsed, attempts = request_structured_json(
            judge_client,
            messages=update_plan_inline_judge_messages(
                record=record,
                context=context,
                api_definitions=api_definitions,
                previous_plan=previous_plan,
                action_json=action_json,
                sandbox_observation=sandbox_observation,
                updated_plan=updated_plan,
                iteration=iteration,
            ),
            purpose=f"{case_id_from_record(record)}/judge/update_plan/{iteration}",
            validator=validate_inline_update_judge,
            logger=logger,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        judge_results["updates"].append(
            {
                "parsed": normalize_update_judge(parsed=parsed, iteration=iteration),
                "attempts": attempts,
            }
        )
    except Exception as exc:  # noqa: BLE001 - judge must not affect eval.
        append_judge_error(judge_results, f"update_plan/{iteration}", exc)
        judge_results["updates"].append(
            {
                "parsed": {
                    "iteration": iteration,
                    "update_acc": False,
                    "update_quality_correct": False,
                    "observation_integration": False,
                    "status_and_evidence_update": False,
                    "next_step_consistency": False,
                    "no_fabrication_or_contradiction": False,
                    "update_preference_correct": False,
                    "preference_retention": False,
                    "preference_aware_replanning": False,
                    "planned_arguments_preference_following": False,
                    "no_preference_violation": False,
                    "plan_state_update": False,
                    "next_action_preference_consistency": False,
                    "action_parameter_guidance_preference_following": False,
                    "supporting_evidence": [],
                    "quality_issues": [f"update plan judge 失败: {exc}"],
                    "preference_issues": [f"update plan judge 失败: {exc}"],
                    "issues": [f"update plan judge 失败: {exc}"],
                    "violated_preferences": [],
                    "summary": f"update plan judge 失败: {exc}",
                    "confidence": 1.0,
                },
                "attempts": getattr(exc, "attempts", []),
            }
        )

def judge_trajectory_inline(
    *,
    judge_client: LLMClient | None,
    record: dict[str, Any],
    context: dict[str, Any],
    api_definitions: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> None:
    judge_results = record.setdefault("judge_results", {"enabled": ENABLE_INLINE_JUDGE, "errors": []})
    record["trajectory_pattern_check"] = validate_trajectory_pattern(record.get("trajectory", []))
    if not ENABLE_INLINE_JUDGE or judge_client is None:
        return
    try:
        parsed, attempts = request_structured_json(
            judge_client,
            messages=trajectory_inline_judge_messages(
                record=record,
                context=context,
                api_definitions=api_definitions,
            ),
            purpose=f"{case_id_from_record(record)}/judge/trajectory",
            validator=validate_inline_trajectory_judge,
            logger=logger,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        judge_results["trajectory"] = {
            "parsed": normalize_trajectory_judge(parsed=parsed, judge_results=judge_results),
            "attempts": attempts,
        }
    except Exception as exc:  # noqa: BLE001 - judge must not affect eval.
        append_judge_error(judge_results, "trajectory", exc)
        judge_results["trajectory"] = {
            "parsed": {
                "case_id": case_id_from_record(record),
                "trajectory_acc": False,
                "raw_trajectory_acc": False,
                "task_completion": False,
                "personalization_consistency": False,
                "action_sequence_reasonable": False,
                "action_params_consistent": False,
                "updates_consistent": False,
                "final_delivery_quality": False,
                "supporting_evidence": [],
                "issues": [f"trajectory judge 失败: {exc}"],
                "summary": f"trajectory judge 失败: {exc}",
                "confidence": 1.0,
            },
            "attempts": getattr(exc, "attempts", []),
        }

def first_plan_pass(judge_results: dict[str, Any]) -> bool:
    parsed = (judge_results.get("first_plan") or {}).get("parsed") if isinstance(judge_results, dict) else {}
    return bool(isinstance(parsed, dict) and parsed.get("first_acc") is True)

def all_action_params_pass(judge_results: dict[str, Any]) -> bool:
    items = judge_results.get("action_params") if isinstance(judge_results, dict) else []
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        parsed = item.get("parsed") if isinstance(item, dict) else {}
        if not isinstance(parsed, dict) or parsed.get("action_param_acc") is not True:
            return False
    return True

def all_updates_pass(judge_results: dict[str, Any]) -> bool:
    items = judge_results.get("updates") if isinstance(judge_results, dict) else []
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        parsed = item.get("parsed") if isinstance(item, dict) else {}
        if not isinstance(parsed, dict) or parsed.get("update_acc") is not True:
            return False
    return True

def all_update_quality_pass(judge_results: dict[str, Any]) -> bool:
    items = judge_results.get("updates") if isinstance(judge_results, dict) else []
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        parsed = item.get("parsed") if isinstance(item, dict) else {}
        if not isinstance(parsed, dict) or parsed.get("update_quality_correct") is not True:
            return False
    return True

def all_update_preference_pass(judge_results: dict[str, Any]) -> bool:
    items = judge_results.get("updates") if isinstance(judge_results, dict) else []
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        parsed = item.get("parsed") if isinstance(item, dict) else {}
        if not isinstance(parsed, dict) or parsed.get("update_preference_correct") is not True:
            return False
    return True

def trajectory_pass(judge_results: dict[str, Any]) -> bool:
    parsed = (judge_results.get("trajectory") or {}).get("parsed") if isinstance(judge_results, dict) else {}
    return bool(isinstance(parsed, dict) and parsed.get("trajectory_acc") is True)

def build_judge_summary(record: dict[str, Any]) -> dict[str, Any]:
    judge_results = record.get("judge_results") if isinstance(record.get("judge_results"), dict) else {}
    first_parsed = (judge_results.get("first_plan") or {}).get("parsed") if isinstance(judge_results, dict) else {}
    first_checklist = (
        first_parsed.get("checklist")
        if isinstance(first_parsed, dict) and isinstance(first_parsed.get("checklist"), dict)
        else {}
    )
    action_items = judge_results.get("action_params") if isinstance(judge_results.get("action_params"), list) else []
    update_items = judge_results.get("updates") if isinstance(judge_results.get("updates"), list) else []
    trajectory_parsed = (
        (judge_results.get("trajectory") or {}).get("parsed")
        if isinstance(judge_results.get("trajectory"), dict)
        else {}
    )

    first_total = len(FIRST_PLAN_CHECKLIST_KEYS)
    first_correct = sum(
        1
        for key in FIRST_PLAN_CHECKLIST_KEYS
        if isinstance(first_checklist.get(key), dict) and first_checklist[key].get("pass") is True
    )
    action_total = len(action_items)
    action_correct = sum(
        1
        for item in action_items
        if isinstance(item, dict)
        and isinstance(item.get("parsed"), dict)
        and item["parsed"].get("action_param_acc") is True
    )
    update_total = len(update_items)
    update_correct = sum(
        1
        for item in update_items
        if isinstance(item, dict)
        and isinstance(item.get("parsed"), dict)
        and item["parsed"].get("update_acc") is True
    )
    update_quality_correct = sum(
        1
        for item in update_items
        if isinstance(item, dict)
        and isinstance(item.get("parsed"), dict)
        and item["parsed"].get("update_quality_correct") is True
    )
    update_preference_correct = sum(
        1
        for item in update_items
        if isinstance(item, dict)
        and isinstance(item.get("parsed"), dict)
        and item["parsed"].get("update_preference_correct") is True
    )
    summary = {
        "first_acc": first_plan_pass(judge_results),
        "first_action_total": first_total,
        "first_action_correct": first_correct,
        "first_planned_arguments_schema_valid": None,
        "first_action_acc": round(first_correct / first_total, 6) if first_total else 0.0,
        "update_acc": all_updates_pass(judge_results),
        "update_quality_acc": all_update_quality_pass(judge_results),
        "update_preference_acc": all_update_preference_pass(judge_results),
        "update_total": update_total,
        "update_correct": update_correct,
        "update_item_acc": round(update_correct / update_total, 6) if update_total else 0.0,
        "update_quality_correct": update_quality_correct,
        "update_quality_item_acc": round(update_quality_correct / update_total, 6) if update_total else 0.0,
        "update_preference_correct": update_preference_correct,
        "update_preference_item_acc": round(update_preference_correct / update_total, 6) if update_total else 0.0,
        "trajectory_acc": trajectory_pass(judge_results),
        "action_param_acc": all_action_params_pass(judge_results),
        "action_param_total": action_total,
        "action_param_correct": action_correct,
        "action_param_item_acc": round(action_correct / action_total, 6) if action_total else 0.0,
        "judge_error_count": len(judge_results.get("errors", [])) if isinstance(judge_results.get("errors"), list) else 0,
        "trajectory_components": trajectory_parsed if isinstance(trajectory_parsed, dict) else {},
    }
    record["judge_summary"] = summary
    return summary

def judge_record_from_eval_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "created_at": utc_now_iso(),
        "eval_created_at": record.get("created_at"),
        "case_id": case_id_from_record(record),
        "user_id": record.get("user_id"),
        "task_id": record.get("task_id"),
        "pass_n": record.get("pass_n"),
        "pass_index": record.get("pass_index"),
        "task_title": record.get("task_title"),
        "preference_injection_mode": record.get("preference_injection_mode"),
        "preference_injection_label": record.get("preference_injection_label"),
        "eval_model": record.get("model"),
        "eval_status": record.get("status"),
        "tool_availability": record.get("tool_availability"),
        "trajectory_pattern_check": record.get("trajectory_pattern_check"),
        "judge_results": record.get("judge_results"),
        "judge_summary": record.get("judge_summary"),
        "tool_call_count": len(record.get("tool_calls", []) if isinstance(record.get("tool_calls"), list) else []),
        "plan_version_count": len(record.get("plan_versions", []) if isinstance(record.get("plan_versions"), list) else []),
        "no_interference_policy": (
            "该记录由 eval_core 旁路生成，只保存 judge 结果；被测 LLM 的 prompt 中不包含 judge 信息。"
        ),
    }

def event_from_llm_attempts(
    *,
    phase: str,
    iteration: int,
    parsed: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "iteration": iteration,
        "created_at": utc_now_iso(),
        "parsed": parsed,
        "llm_attempts": attempts,
    }

def validate_trajectory_pattern(
    trajectory: list[dict[str, Any]],
    max_tool_calls_per_task: int | None = None,
) -> dict[str, Any]:
    max_tool_calls = (
        max_tool_calls_per_task
        if isinstance(max_tool_calls_per_task, int) and not isinstance(max_tool_calls_per_task, bool) and max_tool_calls_per_task > 0
        else MAX_TOOL_CALLS_PER_TASK
    )
    errors: list[str] = []
    normalized_events: list[str] = []
    tool_call_count = 0
    update_plan_count = 0
    finished_by_model = False
    expected_pattern = (
        f"1 次 initial_plan -> 最多 {max_tool_calls} 组 "
        "(reason_action(tool_call) -> sandbox_observation -> update_plan) "
        f"-> 可选 reason_action(finish)，或第 {max_tool_calls} 次 update_plan 后追加 "
        "finalization(forced_after_max_tool_calls)"
    )

    if not trajectory:
        return {
            "ok": False,
            "expected_pattern": expected_pattern,
            "observed_pattern": [],
            "errors": ["trajectory 为空。"],
            "tool_call_count": 0,
            "update_plan_count": 0,
            "finished_by_model": False,
        }

    first_phase = trajectory[0].get("phase")
    normalized_events.append(str(first_phase))
    if first_phase != "initial_plan":
        errors.append("第一个事件必须是 initial_plan。")
    if len(trajectory) == 1:
        errors.append("initial_plan 之后至少需要一个 reason_action。")

    index = 1
    while index < len(trajectory):
        event = trajectory[index]
        phase = event.get("phase")
        normalized_events.append(str(phase))
        if phase != "reason_action":
            if phase == "finalization":
                if index != len(trajectory) - 1:
                    errors.append("finalization 之后不应继续追加 trajectory 事件。")
                finalization_type = event.get("finalization_type")
                if finalization_type != "forced_after_max_tool_calls":
                    errors.append(f"finalization_type 必须是 'forced_after_max_tool_calls'，实际为 {finalization_type!r}。")
                break
            errors.append(f"位置 {index} 必须是 reason_action，实际为 {phase!r}。")
            index += 1
            continue

        parsed = event.get("parsed")
        action_type = parsed.get("action_type") if isinstance(parsed, dict) else None
        if action_type == "finish":
            finished_by_model = True
            if index != len(trajectory) - 1:
                errors.append("finish reason_action 之后不应继续追加 trajectory 事件。")
            break
        if action_type != "tool_call":
            errors.append(f"reason_action.action_type 必须是 tool_call 或 finish，实际为 {action_type!r}。")
            index += 1
            continue

        if index + 2 >= len(trajectory):
            errors.append("tool_call reason_action 之后必须依次包含 sandbox_observation 和 update_plan。")
            break
        observation_phase = trajectory[index + 1].get("phase")
        update_phase = trajectory[index + 2].get("phase")
        normalized_events.append(str(observation_phase))
        normalized_events.append(str(update_phase))
        if observation_phase != "sandbox_observation":
            errors.append(f"tool_call 之后必须是 sandbox_observation，实际为 {observation_phase!r}。")
        if update_phase != "update_plan":
            errors.append(f"sandbox_observation 之后必须是 update_plan，实际为 {update_phase!r}。")
        tool_call_count += 1
        if update_phase == "update_plan":
            update_plan_count += 1
        index += 3

    if tool_call_count > max_tool_calls:
        errors.append(f"tool_call 次数不能超过 {max_tool_calls}，实际为 {tool_call_count}。")
    if update_plan_count != tool_call_count:
        errors.append(f"每次 tool_call 后必须有 1 次 update_plan，当前 tool_call={tool_call_count}, update_plan={update_plan_count}。")
    forced_finalization = (
        bool(trajectory)
        and isinstance(trajectory[-1], dict)
        and trajectory[-1].get("phase") == "finalization"
        and trajectory[-1].get("finalization_type") == "forced_after_max_tool_calls"
    )
    if not finished_by_model and tool_call_count != max_tool_calls:
        errors.append(
            f"未由模型 finish 时，必须在第 {max_tool_calls} 次 update_plan 后强制 finalization，"
            f"当前 tool_call={tool_call_count}。"
        )
    if not finished_by_model and tool_call_count == max_tool_calls and not forced_finalization:
        errors.append(f"第 {max_tool_calls} 次 update_plan 后必须追加 finalization 事件。")

    return {
        "ok": not errors,
        "expected_pattern": expected_pattern,
        "observed_pattern": normalized_events,
        "errors": errors,
        "tool_call_count": tool_call_count,
        "update_plan_count": update_plan_count,
        "finished_by_model": finished_by_model,
        "forced_finalization_after_max_tool_calls": forced_finalization,
    }

def finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = record.get("hyperparameters") if isinstance(record.get("hyperparameters"), dict) else {}
    max_tool_calls = int_value(hyperparameters.get("max_iterations_for_task")) or int_value(
        hyperparameters.get("max_tool_calls_per_task")
    )
    record["trajectory_pattern_check"] = validate_trajectory_pattern(record.get("trajectory", []), max_tool_calls)
    judge_results = record.get("judge_results") if isinstance(record.get("judge_results"), dict) else {}
    if isinstance(judge_results, dict) and judge_results.get("enabled") is True:
        build_judge_summary(record)
    else:
        record["judge_summary"] = {
            "enabled": False,
            "reason": (
                "inline judge 已关闭；正式指标请运行 "
                f"{offline_judge_entrypoint_for_mode(str(record.get('preference_injection_mode') or PREFERENCE_INJECTION_MODE))} "
                "进行离线 judge。"
            ),
        }
    return record

def run_task(
    *,
    client: LLMClient,
    judge_client: LLMClient | None,
    sandbox_executor: SandboxExecutor,
    api_definitions: dict[str, dict[str, Any]],
    user_id: str,
    user_profile: dict[str, Any],
    hidden_info: dict[str, Any] | None,
    long_context: dict[str, Any] | None,
    user_sandbox: dict[str, Any],
    task_doc: dict[str, Any],
    workflow: dict[str, Any],
    run_id: str,
    preference_mode: str,
    pass_n: int,
    pass_index: int,
    llm_attempt_detail_path_for_user: Path | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    task = task_doc.get("task", task_doc)
    task_id = task.get("task_id") or workflow.get("task_id") or "unknown_task"
    task_title = task.get("title", "")
    task_max_tool_calls = max_tool_calls_for_task_doc(task_doc, str(task_id))
    logger.info("开始任务：user=%s task=%s pass=%s/%s title=%s", user_id, task_id, pass_index, pass_n, task_title)

    resolved_long_context = long_context
    tool_availability = resolve_available_apis(
        client=client,
        logger=logger,
        user_id=user_id,
        task_id=task_id,
        task_doc=task_doc,
        user_profile=user_profile,
        hidden_info=hidden_info,
        long_context=resolved_long_context,
        user_sandbox=user_sandbox,
        workflow=workflow,
        api_definitions=api_definitions,
        preference_mode=preference_mode,
    )
    execution_workflow = {
        **workflow,
        "available_api_names": list(tool_availability.get("available_api_names") or []),
    }
    context = build_common_context(
        user_profile=user_profile,
        hidden_info=hidden_info,
        long_context=resolved_long_context,
        user_sandbox=user_sandbox,
        task_doc=task_doc,
        workflow=execution_workflow,
        api_definitions=api_definitions,
        preference_mode=preference_mode,
        tool_availability=tool_availability,
    )

    normalized_preference_mode = normalize_preference_injection_mode(preference_mode)
    record_sandbox_user_context = (
        {}
        if normalized_preference_mode in {"hidden", "pag", "bank"}
        else user_sandbox.get("user_context", {})
    )

    record: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "user_id": user_id,
        "task_id": task_id,
        "pass_n": pass_n,
        "pass_index": pass_index,
        "task_title": task_title,
        "preference_injection_mode": normalized_preference_mode,
        "preference_injection_label": PREFERENCE_INJECTION_LABELS[normalized_preference_mode],
        "model": {
            "name": client.model,
            "base_url": client.base_url,
            "api_key": mask_secret(client.api_key),
            "temperature": MODEL_TEMPERATURE,
            "max_tokens": MODEL_MAX_TOKENS,
        },
        "hyperparameters": {
            "pass_n": pass_n,
            "pass_index": pass_index,
            "first_plan_count": 1,
            "max_tool_calls_per_task": task_max_tool_calls,
            "update_plan_after_each_tool_call": True,
            "force_finalization_after_update_plan_number": task_max_tool_calls,
            "max_iterations_policy": "fixed_max_tool_calls_then_forced_finalization",
            "json_repair_attempts": JSON_REPAIR_ATTEMPTS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "filter_sandbox_observation_metadata_in_prompt": bool(
                FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
            ),
            "sandbox_observation_prompt_excluded_fields": (
                sorted(SANDBOX_OBSERVATION_PROMPT_EXCLUDED_FIELDS)
                if FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
                else []
            ),
            "max_tool_observation_chars_in_prompt": MAX_TOOL_OBSERVATION_CHARS_IN_PROMPT,
            "local_inference": copy.deepcopy(LOCAL_INFERENCE_CONFIG),
            "tool_availability_mode": normalize_tool_availability_mode(TOOL_AVAILABILITY_MODE),
            "pag_retrieval_top_k": PAG_RETRIEVAL_TOP_K if normalized_preference_mode == "pag" else None,
            "pag_retrieval_injection_phases": (
                ["initial_plan", "reason_action", "update_plan"]
                if normalized_preference_mode == "pag"
                else None
            ),
            "bank_retrieval_top_k": BANK_RETRIEVAL_TOP_K if normalized_preference_mode == "bank" else None,
            "enable_long_context": bool(resolved_long_context),
            "long_context_turns": resolved_long_context.get("turn_count") if isinstance(resolved_long_context, dict) else 0,
        },
        "task": task_doc,
        "user_profile": user_profile if normalized_preference_mode in {"direct", "mix"} else None,
        "hidden_info": hidden_info if normalized_preference_mode in {"hidden", "mix", "pag"} else None,
        "long_context": (
            {
                "enabled": True,
                "turn_count": resolved_long_context.get("turn_count"),
                "source_dataset": resolved_long_context.get("source_dataset"),
                "source_file": resolved_long_context.get("source_file"),
                "case_key": resolved_long_context.get("case_key"),
                "sampling_seed": resolved_long_context.get("sampling_seed"),
                "qa_pairs": resolved_long_context.get("qa_pairs", []),
            }
            if isinstance(resolved_long_context, dict)
            else {"enabled": False, "turn_count": 0}
        ),
        "provided_eval_context": {
            "mode": normalized_preference_mode,
            "context": context,
        },
        "sandbox_context": {
            "user_context": record_sandbox_user_context,
            "accessible_api_names": user_sandbox.get("accessible_api_names", []),
            "workflow": workflow,
            "execution_workflow": execution_workflow,
        },
        "tool_availability": tool_availability,
        "plan_versions": [],
        "tool_calls": [],
        "trajectory": [],
        "judge_results": {
            "enabled": ENABLE_INLINE_JUDGE,
            "mode": "inline" if ENABLE_INLINE_JUDGE else "disabled",
            "offline_judge_entrypoint": offline_judge_entrypoint_for_mode(normalized_preference_mode),
            "judge_model": (
                {
                    "name": judge_client.model,
                    "base_url": judge_client.base_url,
                    "api_key": mask_secret(judge_client.api_key),
                    "temperature": JUDGE_TEMPERATURE,
                    "max_tokens": JUDGE_MAX_TOKENS,
                }
                if judge_client is not None
                else None
            ),
            "no_interference_policy": (
                "judge_results 只写入结果文件，不会进入被测 LLM 的 initial_plan/reason_action/update_plan prompt。"
                if ENABLE_INLINE_JUDGE
                else "inline judge 默认关闭；eval_core 只生成 runresult，离线 judge 由 eval/judge_hidden.py 完成。"
            ),
            "first_plan": None,
            "action_params": [],
            "updates": [],
            "trajectory": None,
            "errors": [],
        },
        "final_answer": None,
        "status": "running",
        "error": None,
    }

    used_example_counts: dict[str, int] = {}
    current_plan: dict[str, Any]

    try:
        initial_plan, attempts = request_structured_json(
            client,
            messages=initial_plan_messages(context),
            purpose=f"{user_id}/{task_id}/initial_plan",
            validator=lambda obj: validate_plan_json(obj)
            + validate_expected_phase(obj, expected_phase="initial_plan"),
            logger=logger,
        )
        attempts = persist_llm_attempt_details(
            detail_path=llm_attempt_detail_path_for_user,
            run_id=run_id,
            user_id=user_id,
            task_id=task_id,
            pass_n=pass_n,
            pass_index=pass_index,
            phase="initial_plan",
            iteration=0,
            attempts=attempts,
        )
        current_plan = initial_plan
        initial_step_count = initial_plan_step_count(initial_plan)
        max_iterations_for_task = max_iterations_from_initial_plan(
            initial_plan,
            task_doc=task_doc,
            task_id=str(task_id),
        )
        record["hyperparameters"]["initial_plan_step_count"] = initial_step_count
        record["hyperparameters"]["max_iterations_for_task"] = max_iterations_for_task
        record["plan_versions"].append(initial_plan)
        record["trajectory"].append(
            event_from_llm_attempts(
                phase="initial_plan",
                iteration=0,
                parsed=initial_plan,
                attempts=attempts,
            )
        )
        judge_first_plan_inline(
            judge_client=judge_client,
            record=record,
            context=context,
            api_definitions=api_definitions,
            initial_plan=initial_plan,
            logger=logger,
        )

        for iteration in range(1, max_iterations_for_task + 1):
            action_validator = make_action_validator(
                available_api_names=context["available_api_names"],
                api_definitions=api_definitions,
                expected_iteration=iteration,
            )
            try:
                action_json, action_attempts = request_structured_json(
                    client,
                    messages=action_messages(
                        context=context,
                        current_plan=current_plan,
                        trajectory_summary=summarize_trajectory_for_prompt(record["trajectory"]),
                        iteration=iteration,
                    ),
                    purpose=f"{user_id}/{task_id}/reason_action/{iteration}",
                    validator=action_validator,
                    logger=logger,
                )
                action_attempts = persist_llm_attempt_details(
                    detail_path=llm_attempt_detail_path_for_user,
                    run_id=run_id,
                    user_id=user_id,
                    task_id=task_id,
                    pass_n=pass_n,
                    pass_index=pass_index,
                    phase="reason_action",
                    iteration=iteration,
                    attempts=action_attempts,
                )
            except StructuredOutputError as exc:
                action_attempts = persist_llm_attempt_details(
                    detail_path=llm_attempt_detail_path_for_user,
                    run_id=run_id,
                    user_id=user_id,
                    task_id=task_id,
                    pass_n=pass_n,
                    pass_index=pass_index,
                    phase="reason_action",
                    iteration=iteration,
                    attempts=exc.attempts,
                )
                action_json = {
                    "phase": "reason_action",
                    "iteration": iteration,
                    "task_status": "in_progress",
                    "selected_step_id": None,
                    "reason": (
                        "本轮 reason_action 未能在一次生成加一次 repair 内产生合法 JSON object；"
                        "按评估规则标记该 action 执行失败，并进入 update_plan。"
                    ),
                    "action_type": "tool_call",
                    "tool_call": {
                        "name": "__action_json_generation__",
                        "arguments": {},
                    },
                    "expected_observation": "internal_action_json_generation_failed",
                    "completion_assessment": {
                        "is_complete": False,
                        "confidence": 0.0,
                        "satisfied_criteria": [],
                        "remaining_work": ["生成合法 reason_action JSON 后继续执行工具调用。"],
                    },
                    "structured_output_error": {
                        "message": str(exc),
                        "attempt_count": len(exc.attempts),
                    },
                }
            record["trajectory"].append(
                event_from_llm_attempts(
                    phase="reason_action",
                    iteration=iteration,
                    parsed=action_json,
                    attempts=action_attempts,
                )
            )

            if action_json.get("action_type") == "finish":
                record["status"] = "completed" if action_json.get("task_status") == "completed" else "finished_by_model"
                record["final_answer"] = action_json.get("final_answer")
                record["completion_assessment"] = action_json.get("completion_assessment")
                logger.info(
                    "任务结束：user=%s task=%s pass=%s/%s status=%s iteration=%s",
                    user_id,
                    task_id,
                    pass_index,
                    pass_n,
                    record["status"],
                    iteration,
                )
                judge_trajectory_inline(
                    judge_client=judge_client,
                    record=record,
                    context=context,
                    api_definitions=api_definitions,
                    logger=logger,
                )
                return finalize_record(record)

            tool_call = action_json["tool_call"]
            if tool_call.get("name") == "__action_json_generation__":
                observation = failed_action_observation(
                    user_id=user_id,
                    task_id=task_id,
                    iteration=iteration,
                    error=StructuredOutputError(
                        action_json.get("structured_output_error", {}).get("message", "reason_action JSON 生成失败。"),
                        action_attempts,
                    ),
                )
            else:
                judge_action_param_inline(
                    judge_client=judge_client,
                    record=record,
                    context=context,
                    api_definitions=api_definitions,
                    current_plan=current_plan,
                    action_json=action_json,
                    tool_call=tool_call,
                    iteration=iteration,
                    logger=logger,
                )
                observation = sandbox_executor.execute(
                    tool_call=tool_call,
                    task_id=task_id,
                    workflow=execution_workflow,
                    user_sandbox=user_sandbox,
                    call_index=iteration,
                    used_example_counts=used_example_counts,
                )
            tool_call_record = {
                "iteration": iteration,
                "created_at": utc_now_iso(),
                "tool_call_input": copy.deepcopy(tool_call),
                "sandbox_observation": observation,
            }
            record["tool_calls"].append(tool_call_record)
            record["trajectory"].append(
                {
                    "phase": "sandbox_observation",
                    "iteration": iteration,
                    "created_at": utc_now_iso(),
                    "tool_call": copy.deepcopy(tool_call),
                    "observation": observation,
                }
            )

            updated_plan, update_attempts = request_structured_json(
                client,
                messages=update_plan_messages(
                    context=context,
                    previous_plan=current_plan,
                    action_json=action_json,
                    sandbox_observation=observation,
                    iteration=iteration,
                ),
                purpose=f"{user_id}/{task_id}/update_plan/{iteration}",
                validator=lambda obj, iteration=iteration: validate_plan_json(obj)
                + validate_expected_phase(
                    obj,
                    expected_phase="update_plan",
                    expected_iteration=iteration,
                ),
                logger=logger,
            )
            update_attempts = persist_llm_attempt_details(
                detail_path=llm_attempt_detail_path_for_user,
                run_id=run_id,
                user_id=user_id,
                task_id=task_id,
                pass_n=pass_n,
                pass_index=pass_index,
                phase="update_plan",
                iteration=iteration,
                attempts=update_attempts,
            )
            previous_plan_for_judge = current_plan
            current_plan = updated_plan
            record["plan_versions"].append(updated_plan)
            record["trajectory"].append(
                event_from_llm_attempts(
                    phase="update_plan",
                    iteration=iteration,
                    parsed=updated_plan,
                    attempts=update_attempts,
                )
            )
            judge_update_inline(
                judge_client=judge_client,
                record=record,
                context=context,
                api_definitions=api_definitions,
                previous_plan=previous_plan_for_judge,
                action_json=action_json,
                sandbox_observation=observation,
                updated_plan=updated_plan,
                iteration=iteration,
                logger=logger,
            )

        record["status"] = "forced_finalization_after_max_tool_calls"
        record["final_answer"] = {
            "summary": f"模型已完成固定上限内的工具调用与 update_plan，评测框架在第 {max_iterations_for_task} 次 update_plan 后强制进入 finalization。",
            "last_plan": current_plan,
            "forced_finalization": True,
            "max_tool_calls_per_task": max_iterations_for_task,
        }
        record["trajectory"].append(
            {
                "phase": "finalization",
                "iteration": max_iterations_for_task,
                "created_at": utc_now_iso(),
                "finalization_type": "forced_after_max_tool_calls",
                "reason": f"第 {max_iterations_for_task} 次 update_plan 已完成，按评测轨迹规则强制进入 finalization。",
                "final_answer": record["final_answer"],
            }
        )
        logger.info(
            "任务强制进入 finalization：user=%s task=%s pass=%s/%s max_tool_calls=%s",
            user_id,
            task_id,
            pass_index,
            pass_n,
            max_iterations_for_task,
        )
        judge_trajectory_inline(
            judge_client=judge_client,
            record=record,
            context=context,
            api_definitions=api_definitions,
            logger=logger,
        )
        return finalize_record(record)
    except StructuredOutputError as exc:
        failed_attempts = persist_llm_attempt_details(
            detail_path=llm_attempt_detail_path_for_user,
            run_id=run_id,
            user_id=user_id,
            task_id=task_id,
            pass_n=pass_n,
            pass_index=pass_index,
            phase="structured_output_error",
            iteration=-1,
            attempts=exc.attempts,
        )
        record["status"] = "failed_structured_output"
        record["error"] = {
            "message": str(exc),
            "attempts": failed_attempts,
            "traceback": traceback.format_exc(),
        }
        logger.info(
            "任务失败：user=%s task=%s pass=%s/%s structured_output_error=%s",
            user_id,
            task_id,
            pass_index,
            pass_n,
            exc,
        )
        return finalize_record(record)
    except Exception as exc:  # noqa: BLE001 - eval needs per-task isolation.
        record["status"] = "failed_exception"
        record["error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.info("任务异常：user=%s task=%s pass=%s/%s error=%s", user_id, task_id, pass_index, pass_n, exc)
        return finalize_record(record)

def discover_user_dirs() -> list[Path]:
    if not DATA_DIR.exists():
        raise RuntimeError(f"数据目录不存在: {DATA_DIR}")
    selected = set(USER_ID_FILTER)
    selected_task_ids = task_ids_for_active_filters()
    user_dirs = [
        path
        for path in sorted(DATA_DIR.glob("user_*"))
        if path.is_dir()
        and (not selected or path.name in selected)
        and (
            not selected_task_ids
            or any((path / f"{task_id}.json").exists() for task_id in selected_task_ids)
        )
    ]
    return user_dirs

def task_id_from_doc(task_doc: dict[str, Any], task_path: Path) -> str:
    task = task_doc.get("task", task_doc)
    return str(task.get("task_id") or task_path.stem)

def existing_pass_indexes_for_user(result_path: Path) -> dict[str, set[int]]:
    if not result_path.exists():
        return {}
    task_passes: dict[str, set[int]] = {}
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = row.get("task_id")
            if not task_id or row.get("status") not in SUCCESSFUL_EVAL_STATUSES:
                continue
            pass_index = row.get("pass_index")
            if isinstance(pass_index, bool) or not isinstance(pass_index, int) or pass_index < 1:
                pass_index = len(task_passes.get(str(task_id), set())) + 1
            task_passes.setdefault(str(task_id), set()).add(pass_index)
    return task_passes

def run_user(
    *,
    user_dir: Path,
    client: LLMClient,
    judge_client: LLMClient | None,
    sandbox_executor: SandboxExecutor,
    api_definitions: dict[str, dict[str, Any]],
    run_id: str,
    preference_mode: str,
    pass_n: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    pass_n = normalize_pass_n(pass_n)
    user_id = user_dir.name
    result_path = RESULT_DIR / f"{user_id}.jsonl"
    judge_result_path = JUDGE_RESULT_DIR / f"{user_id}.jsonl" if ENABLE_INLINE_JUDGE else None
    detail_path = llm_attempt_detail_path(run_id=run_id, user_id=user_id) if SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR else None
    user_profile = read_json(user_dir / "user_profile.json")
    hidden_path = user_dir / "hidden_info.json"
    hidden_info = read_json(hidden_path) if hidden_path.exists() else None
    if normalize_preference_injection_mode(preference_mode) in {"hidden", "mix", "pag", "bank"} and hidden_info is None:
        raise RuntimeError(f"{preference_mode} 模式缺少用户交互历史文件: {hidden_path}")
    user_sandbox = load_user_sandbox(user_dir)
    selected_task_ids = task_ids_for_active_filters()
    existing_pass_indexes = existing_pass_indexes_for_user(result_path) if SKIP_TASKS_ALREADY_IN_RESULT_FILE else {}

    task_paths = sorted(user_dir.glob("task_*.json"))
    summaries: list[dict[str, Any]] = []
    for task_path in task_paths:
        task_doc = read_json(task_path)
        task_id = task_id_from_doc(task_doc, task_path)
        if selected_task_ids and task_id not in selected_task_ids:
            continue
        workflow = find_workflow(user_sandbox, task_id)
        completed_passes = existing_pass_indexes.get(task_id, set())
        for pass_index in range(1, pass_n + 1):
            if pass_index in completed_passes:
                logger.info("跳过已存在结果：user=%s task=%s pass=%s/%s", user_id, task_id, pass_index, pass_n)
                summaries.append(
                    {
                        "task_id": task_id,
                        "pass_n": pass_n,
                        "pass_index": pass_index,
                        "status": "skipped_existing",
                    }
                )
                continue
            record = run_task(
                client=client,
                judge_client=judge_client,
                sandbox_executor=sandbox_executor,
                api_definitions=api_definitions,
                user_id=user_id,
                user_profile=user_profile,
                hidden_info=hidden_info,
                long_context=build_long_context_for_task(
                    user_id=user_id,
                    task_id=task_id,
                    preference_mode=preference_mode,
                ),
                user_sandbox=user_sandbox,
                task_doc=task_doc,
                workflow=workflow,
                run_id=run_id,
                preference_mode=preference_mode,
                pass_n=pass_n,
                pass_index=pass_index,
                llm_attempt_detail_path_for_user=detail_path,
                logger=logger,
            )
            write_jsonl(result_path, record)
            if judge_result_path is not None:
                write_jsonl(judge_result_path, judge_record_from_eval_record(record))
            summaries.append(
                {
                    "task_id": task_id,
                    "pass_n": pass_n,
                    "pass_index": pass_index,
                    "status": record.get("status"),
                    "tool_call_count": len(record.get("tool_calls", [])),
                    "plan_version_count": len(record.get("plan_versions", [])),
                    "judge_summary": record.get("judge_summary", {}),
                }
            )
    return {
        "user_id": user_id,
        "result_path": str(result_path),
        "judge_result_path": str(judge_result_path) if judge_result_path is not None else None,
        "llm_attempt_detail_path": str(detail_path) if detail_path is not None else None,
        "tasks": summaries,
    }

def maybe_overwrite_selected_results(user_dirs: list[Path], logger: logging.Logger) -> None:
    if not OVERWRITE_SELECTED_USER_RESULTS_ON_START:
        return
    for user_dir in user_dirs:
        result_path = RESULT_DIR / f"{user_dir.name}.jsonl"
        if result_path.exists():
            result_path.unlink()
            logger.info("已清空旧结果文件：%s", result_path)
        if ENABLE_INLINE_JUDGE:
            judge_result_path = JUDGE_RESULT_DIR / f"{user_dir.name}.jsonl"
            if judge_result_path.exists():
                judge_result_path.unlink()
                logger.info("已清空旧 inline judge 结果文件：%s", judge_result_path)

def pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count * 100.0 / total, 2)

def summary_bool(summary: dict[str, Any], key: str) -> bool:
    return bool(isinstance(summary, dict) and summary.get(key) is True)

def aggregate_eval_judge_summary(user_summaries: list[dict[str, Any]], pass_n: int) -> dict[str, Any]:
    if not ENABLE_INLINE_JUDGE:
        offline_entrypoint = offline_judge_entrypoint_for_mode(PREFERENCE_INJECTION_MODE)
        return {
            "enabled": False,
            "reason": f"inline judge 已关闭；eval_core 只生成 runresult，正式指标请运行 {offline_entrypoint}。",
            "offline_judge_entrypoint": offline_entrypoint,
        }

    case_rows: list[dict[str, Any]] = []
    for user_summary in user_summaries:
        user_id = str(user_summary.get("user_id") or "")
        for task in user_summary.get("tasks", []) if isinstance(user_summary.get("tasks"), list) else []:
            if not isinstance(task, dict) or task.get("status") == "skipped_existing":
                continue
            row = {
                "user_id": user_id,
                "task_id": str(task.get("task_id") or ""),
                "pass_index": int(task.get("pass_index") or 1),
                "status": task.get("status"),
                "judge_summary": task.get("judge_summary") if isinstance(task.get("judge_summary"), dict) else {},
            }
            case_rows.append(row)

def probe_once(client: LLMClient, probe_id: str) -> tuple[bool, float, str]:
    messages = concurrency_probe.build_messages(probe_id)
    started = time.perf_counter()
    try:
        response = client.chat(
            messages,
            purpose=f"concurrency_probe/{probe_id}",
            max_tokens=128,
            temperature=0.0,
            timeout=CONCURRENCY_PROBE_TIMEOUT_SECONDS,
        )
        parsed = extract_json_object(response.content)
        ok = isinstance(parsed, dict) and parsed.get("ok") is True
        return ok, time.perf_counter() - started, "" if ok else f"unexpected_probe_json={parsed}"
    except Exception as exc:  # noqa: BLE001 - probe should report failures.
        return False, time.perf_counter() - started, str(exc)

def calibrate_concurrency(client: LLMClient, logger: logging.Logger) -> int:
    if client.base_url == "https://api.qingyuntop.top/v1" and client.model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        logger.info("根据项目 API 约定，Qingyun DeepSeek eval 并发采用 %s。", QINGYUN_DEEPSEEK_EVAL_CONCURRENCY)
        return QINGYUN_DEEPSEEK_EVAL_CONCURRENCY

    if client.base_url == "https://api.deepseek.com" and client.model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        difficulty = normalize_task_difficulty(TASK_DIFFICULTY)
        if difficulty in DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY:
            concurrency = DEEPSEEK_EVAL_CONCURRENCY_BY_DIFFICULTY[difficulty]
            logger.info("根据项目 API 约定，DeepSeek eval 并发采用 %s（difficulty=%s）。", concurrency, difficulty)
            return concurrency
        logger.info("根据项目 API 约定，DeepSeek eval 并发采用 %s（difficulty=all）。", DEEPSEEK_EVAL_CONCURRENCY)
        return DEEPSEEK_EVAL_CONCURRENCY

    if client.base_url == "https://token-plan-cn.xiaomimimo.com/v1" and client.model == "mimo-v2.5-pro":
        logger.info("根据本轮 MIMO API 限流探测，mimo-v2.5-pro eval 并发采用 %s。", MIMO_V25_PRO_EVAL_CONCURRENCY)
        return MIMO_V25_PRO_EVAL_CONCURRENCY

    if not AUTO_TUNE_CONCURRENCY:
        logger.info("未启用并发探测，使用 DEFAULT_EVAL_CONCURRENCY=%s", DEFAULT_EVAL_CONCURRENCY)
        return DEFAULT_EVAL_CONCURRENCY

    best = 1
    logger.info("开始并发探测，候选值=%s", CONCURRENCY_CANDIDATES)
    for candidate in CONCURRENCY_CANDIDATES:
        started = time.perf_counter()
        results: list[tuple[bool, float, str]] = []
        with ThreadPoolExecutor(max_workers=candidate, thread_name_prefix=f"probe{candidate}") as pool:
            futures = [pool.submit(probe_once, client, f"{candidate}_{idx}") for idx in range(candidate)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.perf_counter() - started
        success_count = sum(1 for ok, _, _ in results if ok)
        success_rate = success_count / candidate
        avg_latency = sum(latency for _, latency, _ in results) / max(1, len(results))
        errors = [error for ok, _, error in results if not ok for error in [error]]
        logger.info(
            "并发探测结果：candidate=%s success=%s/%s success_rate=%.2f avg_latency=%.2fs elapsed=%.2fs errors=%s",
            candidate,
            success_count,
            candidate,
            success_rate,
            avg_latency,
            elapsed,
            errors[:2],
        )
        if success_rate >= CONCURRENCY_PROBE_SUCCESS_THRESHOLD:
            best = candidate
            continue
        logger.info("并发 %s 未达成功率阈值，停止继续升高并发。", candidate)
        break
    logger.info("最终采用评估并发：%s", best)
    return best

def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)
    disable_proxy_for_eval(logger)
    preference_mode = normalize_preference_injection_mode(PREFERENCE_INJECTION_MODE)
    task_difficulty = apply_task_difficulty_runtime_config(logger)
    pass_n = normalize_pass_n(PASS_N)

    api_config = load_api_config()
    result_dir, judge_result_dir = configure_result_dir(getattr(api_config, "GEN_MODEL"), preference_mode, pass_n)
    result_dir.mkdir(parents=True, exist_ok=True)
    if ENABLE_INLINE_JUDGE:
        judge_result_dir.mkdir(parents=True, exist_ok=True)
    client = (
        GENERATION_CLIENT_FACTORY(api_config, logger)
        if GENERATION_CLIENT_FACTORY is not None
        else LLMClient(
            api_key=getattr(api_config, "GEN_KEY"),
            base_url=getattr(api_config, "GEN_URL"),
            model=getattr(api_config, "GEN_MODEL"),
            logger=logger,
        )
    )
    judge_client = (
        LLMClient(
            api_key=getattr(api_config, "JUDGE_KEY", getattr(api_config, "GEN_KEY")),
            base_url=getattr(api_config, "JUDGE_URL", getattr(api_config, "GEN_URL")),
            model=getattr(api_config, "JUDGE_MODEL", getattr(api_config, "GEN_MODEL")),
            logger=logger,
        )
        if ENABLE_INLINE_JUDGE
        else None
    )
    logger.info(
        "开始 eval：run_id=%s model=%s base_url=%s key=%s preference_mode=%s",
        run_id,
        client.model,
        client.base_url,
        mask_secret(client.api_key),
        preference_mode,
    )
    logger.info(
        "核心配置：PASS_N=%s TASK_DIFFICULTY=%s TRAJECTORY_POLICY=fixed_first_plan_dynamic_tool_calls_update_after_each_tool_call_then_forced_finalization MAX_TOOL_CALLS_PER_TASK=%s TOOL_AVAILABILITY_MODE=%s USER_ID_FILTER=%s TASK_ID_FILTER=%s ACTIVE_TASK_FILTER=%s AUTO_TUNE_CONCURRENCY=%s RESULT_DIR=%s JUDGE_RESULT_DIR=%s PREFERENCE_INJECTION_MODE=%s BANK_RETRIEVAL_TOP_K=%s ENABLE_LONG_CONTEXT=%s LONG_CONTEXT_TURNS=%s ENABLE_INLINE_JUDGE=%s",
        pass_n,
        task_difficulty,
        MAX_TOOL_CALLS_PER_TASK,
        TOOL_AVAILABILITY_MODE,
        USER_ID_FILTER,
        TASK_ID_FILTER,
        sorted(task_ids_for_active_filters()),
        AUTO_TUNE_CONCURRENCY,
        RESULT_DIR,
        JUDGE_RESULT_DIR,
        preference_mode,
        BANK_RETRIEVAL_TOP_K if preference_mode == "bank" else None,
        ENABLE_LONG_CONTEXT,
        LONG_CONTEXT_TURNS,
        ENABLE_INLINE_JUDGE,
    )
    if judge_client is not None:
        logger.info(
            "Inline judge：model=%s base_url=%s key=%s temperature=%s max_tokens=%s no_interference=true",
            judge_client.model,
            judge_client.base_url,
            mask_secret(judge_client.api_key),
            JUDGE_TEMPERATURE,
            JUDGE_MAX_TOKENS,
        )

    api_definitions = load_global_api_definitions()
    sandbox_executor = SandboxExecutor(api_definitions)
    user_dirs = discover_user_dirs()
    maybe_overwrite_selected_results(user_dirs, logger)
    logger.info("待评估用户数：%s", len(user_dirs))

    eval_concurrency = calibrate_concurrency(client, logger)
    all_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=eval_concurrency, thread_name_prefix="eval") as pool:
        futures = [
            pool.submit(
                run_user,
                user_dir=user_dir,
                client=client,
                judge_client=judge_client,
                sandbox_executor=sandbox_executor,
                api_definitions=api_definitions,
                run_id=run_id,
                preference_mode=preference_mode,
                pass_n=pass_n,
                logger=logger,
            )
            for user_dir in user_dirs
        ]
        for future in as_completed(futures):
            summary = future.result()
            all_summaries.append(summary)
            logger.info("用户完成：user=%s tasks=%s", summary["user_id"], summary["tasks"])

    elapsed = time.perf_counter() - started
    summary_path = LOG_DIR / f"eval_summary_{run_id}.json"
    offline_judge_entrypoint = offline_judge_entrypoint_for_mode(preference_mode)
    summary_record = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "model": {
            "name": client.model,
            "base_url": client.base_url,
            "api_key": mask_secret(client.api_key),
        },
        "preference_injection_mode": preference_mode,
        "preference_injection_label": PREFERENCE_INJECTION_LABELS[preference_mode],
        "task_difficulty": task_difficulty,
        "active_task_filter": sorted(task_ids_for_active_filters()),
        "max_tool_calls_per_task": MAX_TOOL_CALLS_PER_TASK,
        "filter_sandbox_observation_metadata_in_prompt": bool(
            FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
        ),
        "sandbox_observation_prompt_excluded_fields": (
            sorted(SANDBOX_OBSERVATION_PROMPT_EXCLUDED_FIELDS)
            if FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
            else []
        ),
        "max_tool_observation_chars_in_prompt": MAX_TOOL_OBSERVATION_CHARS_IN_PROMPT,
        "local_inference": copy.deepcopy(LOCAL_INFERENCE_CONFIG),
        "tool_availability_mode": normalize_tool_availability_mode(TOOL_AVAILABILITY_MODE),
        "bank_retrieval_top_k": BANK_RETRIEVAL_TOP_K if preference_mode == "bank" else None,
        "pass_n": pass_n,
        "eval_concurrency": eval_concurrency,
        "inline_judge": (
            {
                "enabled": True,
                "name": judge_client.model,
                "base_url": judge_client.base_url,
                "api_key": mask_secret(judge_client.api_key),
                "temperature": JUDGE_TEMPERATURE,
                "max_tokens": JUDGE_MAX_TOKENS,
                "no_interference_policy": (
                    "judge 在 eval_core 内旁路执行，结果只进入 judge_results/judge_summary，不进入被测 LLM prompt。"
                ),
            }
            if judge_client is not None
            else {"enabled": False}
        ),
        "user_count": len(user_dirs),
        "outputs": {
            "result_dir": str(RESULT_DIR),
            "llm_attempt_detail_dir": str(RESULT_DIR / LLM_ATTEMPT_DETAIL_DIR_NAME / run_id)
            if SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR
            else None,
            "inline_judge_result_dir": str(JUDGE_RESULT_DIR) if ENABLE_INLINE_JUDGE else None,
            "offline_judge_entrypoint": offline_judge_entrypoint,
            "summary_json": str(summary_path),
        },
        "llm_attempt_detail_policy": {
            "save_to_sidecar": SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR,
            "keep_full_attempts_in_main_result": KEEP_FULL_LLM_ATTEMPTS_IN_MAIN_RESULT,
            "sidecar_detail_dir_name": LLM_ATTEMPT_DETAIL_DIR_NAME,
            "main_result_policy": (
                "主结果仅保留 detail_ref、usage、耗时、状态和解析摘要；完整 prompt/raw response/content/parsed 写入 sidecar。"
                if SAVE_LLM_ATTEMPT_DETAILS_TO_SIDECAR and not KEEP_FULL_LLM_ATTEMPTS_IN_MAIN_RESULT
                else "主结果保留完整 llm_attempts。"
            ),
        },
        "inline_judge_aggregate_summary": aggregate_eval_judge_summary(all_summaries, pass_n),
        "summaries": sorted(all_summaries, key=lambda item: item["user_id"]),
    }
    summary_path.write_text(json.dumps(summary_record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "eval 完成：elapsed=%.2fs summary=%s result_dir=%s inline_judge_result_dir=%s offline_judge_entrypoint=%s",
        elapsed,
        summary_path,
        RESULT_DIR,
        JUDGE_RESULT_DIR if ENABLE_INLINE_JUDGE else None,
        offline_judge_entrypoint,
    )
