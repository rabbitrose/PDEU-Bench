#!/usr/bin/env python3
"""Judge planning results with LLM-as-judge and generate an HTML report."""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# =========================
# 用户长期可调 judge 超参数
# =========================

# 为空表示使用 api.py 中的 GEN_MODEL，读取 runresult/<GEN_MODEL>/<模式>/pass_<PASS_N>/。
TARGET_GEN_LLM = "deepseek-v4-flash"

# 评估上下文方式：direct=用户信息，hidden=用户交互历史，mix=二者同时提供，
# pag=检索QA+用户摘要，bank=用户抽象画像+QA Summary RAG。
PREFERENCE_INJECTION_MODE = "hidden"

# 选择要 judge 的 eval pass 目录；例如 1 读取 pass_1，3 读取 pass_3。
PASS_N = 1

# 长上下文结果目录控制。需要与 eval_core.py/eval_hidden.py/eval_mix.py 的配置保持一致。
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_ALLOWED_TURNS = {0, 10, 50, 100}

# 指定后只 judge 对应 eval run_id；为空时默认 judge 目标模型目录下最新一轮 eval。
TARGET_EVAL_RUN_ID = ""
JUDGE_LATEST_EVAL_RUN_ONLY = True

# 当同一 user/task/pass 存在多轮 eval 记录时，保留 run_id 最新的一条用于 judge。
# 这适用于补跑缺失任务后，把旧完整 run 与新补跑 run 合并成每个 case 一条最新记录。
DEDUP_EVAL_RECORDS_BY_CASE = True

# 为空表示评估全部用户/任务；例如 ["user_001"]、["task_001"]。
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []

# 任务难度过滤：easy -> task_easy, hard -> task_hard。
TASK_DIFFICULTY = "hard"
VALID_TASK_DIFFICULTIES = {"easy", "hard", "all"}
TASK_ID_BY_DIFFICULTY = {
    "easy": "task_easy",
    "hard": "task_hard",
}

# Judge 模型参数。
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 8192
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("PPEVAL_JUDGE_REQUEST_TIMEOUT_SECONDS", "120"))
API_RETRY_ATTEMPTS = 1
API_RETRY_BASE_SLEEP_SECONDS = 2.0
JSON_REPAIR_ATTEMPTS = 1
USE_JSON_RESPONSE_FORMAT = True

# 并发控制。DeepSeek deepseek-v4-flash/deepseek-v4-pro 按项目约定可开到 150。
AUTO_TUNE_CONCURRENCY = False
DEFAULT_JUDGE_CONCURRENCY = 8
DEEPSEEK_JUDGE_CONCURRENCY = 150
QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = 10
CONCURRENCY_CANDIDATES = (1, 2, 4, 8, 12, 16, 20, 40, 80, 120, 150)
CONCURRENCY_PROBE_SUCCESS_THRESHOLD = 0.90
CONCURRENCY_PROBE_TIMEOUT_SECONDS = 60

# 输入裁剪只影响 judge prompt；eval 原始结果和 judge 原始响应会完整落盘。
MAX_CONTEXT_CHARS = 16000
MAX_PLAN_CHARS = 9000
MAX_TOOL_CALLS_PER_PROMPT = 30
MAX_TOOL_INPUT_CHARS = 22000
MAX_UPDATE_EVENTS_PER_PROMPT = 30
MAX_UPDATE_JUDGE_CHARS = 26000
# 长轨迹按小批次 judge，避免 reasoning tokens 挤占结构化输出空间。
INSTANTIATION_JUDGE_CHUNK_SIZE = 4
UPDATE_JUDGE_CHUNK_SIZE = 2
MAX_REASON_CHARS_PER_TOOL = 600
MAX_HTML_CASES = 1000
MAX_HTML_TEXT_CHARS = 900

# 路径与保存控制。
EVAL_RESULT_DIR_NAME = "runresult"
JUDGE_RESULT_DIR_NAME = "judgeresult"
HTML_DIR_NAME = "htmls"
LOG_DIR_NAME = "logs"
SANDBOX_DIR_NAME = "SandboxDB"
EVAL_RESULT_GLOB = "user_*.jsonl"
SAVE_PROMPTS_IN_JUDGE_RESULTS = True
SAVE_RAW_RESPONSES_IN_JUDGE_RESULTS = True

# hidden judge 可使用 data/user_xxx/user_profile.json 中的显式偏好作为离线评分参考。
# 该参考只进入 judge prompt，不会进入被测 LLM 的 eval prompt。
USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Prompts.judge import first_plan as first_plan_prompt
from Prompts.judge import trajectory as trajectory_prompt
from Prompts.judge import update_plan as update_plan_prompt
from Prompts.common import concurrency_probe, json_repair

EVAL_RESULT_BASE_DIR = PROJECT_ROOT / EVAL_RESULT_DIR_NAME
JUDGE_RESULT_BASE_DIR = PROJECT_ROOT / JUDGE_RESULT_DIR_NAME
HTML_BASE_DIR = PROJECT_ROOT / HTML_DIR_NAME
SANDBOX_DIR = PROJECT_ROOT / SANDBOX_DIR_NAME
DATA_DIR = PROJECT_ROOT / "data"
EVAL_RESULT_DIR = EVAL_RESULT_BASE_DIR
JUDGE_RESULT_DIR = JUDGE_RESULT_BASE_DIR
HTML_DIR = HTML_BASE_DIR
LOG_DIR = PROJECT_ROOT / LOG_DIR_NAME
VALID_PREFERENCE_INJECTION_MODES = {"direct", "hidden", "mix", "pag", "bank"}
PREFERENCE_INJECTION_LABELS = {
    "direct": "用户信息",
    "hidden": "用户交互历史",
    "mix": "用户信息 + 用户交互历史",
    "pag": "PAG 检索QA + 用户摘要",
    "bank": "BANK 用户抽象画像 + QA Summary RAG",
}
API_DEFINITIONS: dict[str, dict[str, Any]] = {}

PLAN_CHECKLIST_KEYS = [
    "profile_preference_coverage",
    "preference_violation_avoidance",
]

PLAN_CHECKLIST_LABELS = {
    "profile_preference_coverage": "画像偏好覆盖",
    "preference_violation_avoidance": "偏好违背规避",
}

PLAN_QUALITY_CHECKLIST_KEYS = list(first_plan_prompt.PLAN_QUALITY_CHECKS)

PLAN_QUALITY_CHECKLIST_LABELS = {
    "goal_and_criteria_coverage": "目标与成功标准覆盖",
    "step_actionability": "步骤可执行与可监控",
    "dependency_and_ordering": "依赖与顺序正确",
    "tool_and_information_acquisition": "工具与信息获取充分",
    "constraint_feasibility": "约束下可行",
    "global_coherence_and_completion_path": "全局连贯与完成路径",
}


class StructuredOutputError(RuntimeError):
    """Raised when the judge model cannot produce valid structured JSON."""

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"结果文件 JSONL 解析失败: {path}:{line_no}: {exc}") from exc
    return rows


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def compact_json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=False)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit + 200
    return text[: limit - 200] + f"...<已裁剪 {omitted} 字符>"


def compact_text(value: Any, limit: int = MAX_HTML_TEXT_CHARS) -> str:
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


def load_api_definitions() -> dict[str, dict[str, Any]]:
    index_path = SANDBOX_DIR / "api_index.json"
    api_index = read_json(index_path)
    definitions: dict[str, dict[str, Any]] = {}
    for api_name, entry in api_index.items():
        if not isinstance(entry, dict):
            continue
        definition_path = SANDBOX_DIR / str(entry.get("definition_path") or "")
        if definition_path.exists():
            definitions[str(api_name)] = read_json(definition_path)
    return definitions


def api_no_relation(api_name: Any) -> int:
    if not isinstance(api_name, str):
        return 0
    api_def = API_DEFINITIONS.get(api_name)
    if not isinstance(api_def, dict):
        return 0
    return 1 if int(api_def.get("NoRelation") or 0) == 1 else 0


def api_preference_sensitive_required_params(api_name: Any) -> list[str]:
    if not isinstance(api_name, str):
        return []
    api_def = API_DEFINITIONS.get(api_name)
    if not isinstance(api_def, dict):
        return []
    params = api_def.get("interface", {}).get("parameters", [])
    return [
        str(param.get("name"))
        for param in params
        if isinstance(param, dict)
        and param.get("required") is True
        and param.get("preference_sensitive") is True
        and param.get("name")
    ]


def api_nonpreference_required_params(api_name: Any) -> list[str]:
    if not isinstance(api_name, str):
        return []
    api_def = API_DEFINITIONS.get(api_name)
    if not isinstance(api_def, dict):
        return []
    params = api_def.get("interface", {}).get("parameters", [])
    return [
        str(param.get("name"))
        for param in params
        if isinstance(param, dict)
        and param.get("required") is True
        and param.get("preference_sensitive") is not True
        and param.get("name")
    ]


def long_context_enabled_for_mode(preference_mode: str) -> bool:
    mode = normalize_preference_injection_mode(preference_mode)
    return bool(ENABLE_LONG_CONTEXT and normalize_long_context_turns(LONG_CONTEXT_TURNS) > 0 and mode in VALID_PREFERENCE_INJECTION_MODES)


def long_context_result_component(preference_mode: str) -> str | None:
    if not long_context_enabled_for_mode(preference_mode):
        return None
    return f"longctx_{normalize_long_context_turns(LONG_CONTEXT_TURNS)}"


def eval_result_dir_for_model(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> Path:
    mode = normalize_preference_injection_mode(preference_mode)
    resolved_pass_n = normalize_pass_n(PASS_N if pass_n is None else pass_n)
    base = EVAL_RESULT_BASE_DIR / safe_model_dir_name(model_name) / difficulty_result_component() / mode
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
    base = base / f"pass_{resolved_pass_n}"
    return _apply_judge_tag(base)


def _apply_judge_tag(base: Path) -> Path:
    """按判分模型区分输出目录，避免同一组生成结果被不同 judge 覆盖。"""
    judge_tag = os.environ.get("PPEVAL_JUDGE_TAG", "").strip()
    if judge_tag:
        return base / f"judge_{safe_model_dir_name(judge_tag)}"
    return base



def html_dir_for_model(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> Path:
    mode = normalize_preference_injection_mode(preference_mode)
    resolved_pass_n = normalize_pass_n(PASS_N if pass_n is None else pass_n)
    base = HTML_BASE_DIR / safe_model_dir_name(model_name) / difficulty_result_component() / mode
    long_context_component = long_context_result_component(mode)
    if long_context_component:
        base = base / long_context_component
    base = base / f"pass_{resolved_pass_n}"
    return _apply_judge_tag(base)


def configure_model_dirs(
    model_name: str,
    preference_mode: str = PREFERENCE_INJECTION_MODE,
    pass_n: int | None = None,
) -> tuple[Path, Path, Path]:
    global EVAL_RESULT_DIR, JUDGE_RESULT_DIR, HTML_DIR
    EVAL_RESULT_DIR = eval_result_dir_for_model(model_name, preference_mode, pass_n)
    JUDGE_RESULT_DIR = judge_result_dir_for_model(model_name, preference_mode, pass_n)
    HTML_DIR = html_dir_for_model(model_name, preference_mode, pass_n)
    return EVAL_RESULT_DIR, JUDGE_RESULT_DIR, HTML_DIR


# 由 eval/local/judge_local_hidden.py 注入，用于把判分请求分发到多个本地 vLLM 副本。
JUDGE_CLIENT_FACTORY: Callable[[Any, logging.Logger], Any] | None = None


def load_api_config() -> Any:
    api_path = PROJECT_ROOT / "api.py"
    spec = importlib.util.spec_from_file_location("ppeval_api_config", api_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 API 配置文件: {api_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.environ.get("PPEVAL_JUDGE_URL"):
        module.JUDGE_URL = os.environ["PPEVAL_JUDGE_URL"]
    if os.environ.get("PPEVAL_JUDGE_MODEL"):
        module.JUDGE_MODEL = os.environ["PPEVAL_JUDGE_MODEL"]
    if os.environ.get("PPEVAL_JUDGE_KEY"):
        module.JUDGE_KEY = os.environ["PPEVAL_JUDGE_KEY"]
    if os.environ.get("PPEVAL_GEN_MODEL"):
        module.GEN_MODEL = os.environ["PPEVAL_GEN_MODEL"]
    return module


def setup_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppeval_judge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(LOG_DIR / f"judge_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


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
        self._url_opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if "api.deepseek.com" in self.base_url
            else urllib.request.build_opener()
        )
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
        max_tokens = max_tokens or JUDGE_MAX_TOKENS
        temperature = JUDGE_TEMPERATURE if temperature is None else temperature
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
                with self._url_opener.open(request, timeout=timeout) as response:
                    raw_body = response.read().decode("utf-8")
                    elapsed = time.perf_counter() - started
                    try:
                        raw_json: dict[str, Any] | str = json.loads(raw_body)
                    except json.JSONDecodeError:
                        raw_json = raw_body
                    content, usage = self._extract_content_and_usage(raw_json)
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
                    self.logger.info("Judge 接口不接受 response_format，后续自动改用纯提示词 JSON 约束。")
                    continue
                if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < API_RETRY_ATTEMPTS:
                    sleep_seconds = self._retry_sleep_seconds(attempt, exc.headers.get("Retry-After"))
                    self.logger.info(
                        "Judge API 请求失败后重试：purpose=%s status=%s attempt=%s/%s elapsed=%.2fs sleep=%.2fs",
                        purpose,
                        exc.code,
                        attempt,
                        API_RETRY_ATTEMPTS,
                        elapsed,
                        sleep_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"Judge API 请求失败 purpose={purpose}: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                elapsed = time.perf_counter() - started
                last_error = exc
                if attempt < API_RETRY_ATTEMPTS:
                    sleep_seconds = self._retry_sleep_seconds(attempt)
                    self.logger.info(
                        "Judge API 网络错误后重试：purpose=%s attempt=%s/%s elapsed=%.2fs sleep=%.2fs error=%s",
                        purpose,
                        attempt,
                        API_RETRY_ATTEMPTS,
                        elapsed,
                        sleep_seconds,
                        exc,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(f"Judge API 网络错误 purpose={purpose}: {exc}") from exc

        raise RuntimeError(f"Judge API 请求失败 purpose={purpose}: {last_error}")

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
        choices = raw_json.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content, raw_json.get("usage") or {}
                text = first.get("text")
                if isinstance(text, str):
                    return text, raw_json.get("usage") or {}
        return json.dumps(raw_json, ensure_ascii=False), raw_json.get("usage") or {}


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
        preserve_semantics_instruction="请修复为满足要求的 JSON object。不要改变原始评判语义。",
    )


def request_structured_json(
    client: LLMClient,
    *,
    messages: list[dict[str, str]],
    purpose: str,
    validator: Callable[[Any], list[str]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    current_messages = messages
    last_error = ""

    for repair_idx in range(JSON_REPAIR_ATTEMPTS + 1):
        response = client.chat(current_messages, purpose=purpose)
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
            "content": response.content if SAVE_RAW_RESPONSES_IN_JUDGE_RESULTS else None,
            "parsed": parsed if parsed is not None else None,
            "parse_or_validation_error": parse_error,
            "usage": response.usage,
            "http_status": response.http_status,
            "elapsed_seconds": response.elapsed_seconds,
            "api_attempt_count": response.attempt_count,
            "request_payload": response.request_payload if SAVE_PROMPTS_IN_JUDGE_RESULTS else None,
            "raw_response": response.raw_response if SAVE_RAW_RESPONSES_IN_JUDGE_RESULTS else None,
        }
        attempts.append(attempt_record)

        if parsed is not None and not validation_errors:
            return parsed, attempts

        last_error = parse_error
        logger.info("Judge 结构化 JSON 失败，准备修复：purpose=%s repair=%s error=%s", purpose, repair_idx, parse_error)
        current_messages = build_repair_messages(messages, response.content, parse_error)

    raise StructuredOutputError(f"{purpose} 无法生成有效 JSON：{last_error}", attempts)


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


UPDATE_QUALITY_KEYS = [
    "observation_integration",
    "status_and_evidence_update",
    "next_step_consistency",
    "no_fabrication_or_contradiction",
]

UPDATE_PREFERENCE_KEYS = [
    "preference_retention",
    "preference_aware_replanning",
    "planned_arguments_preference_following",
    "no_preference_violation",
]

UPDATE_PLAN_QUALITY_KEYS = list(update_plan_prompt.UPDATED_PLAN_QUALITY_CHECKS)

UPDATE_PLAN_QUALITY_LABELS = {
    "remaining_goal_and_criteria_coverage": "剩余目标与成功标准覆盖",
    "replan_decision_correctness": "重规划决策正确",
    "remaining_steps_actionability": "剩余步骤可执行",
    "information_acquisition_and_verification": "信息获取与核对充分",
    "constraint_feasibility_after_update": "更新后约束可行",
    "global_coherence_and_completion_path": "全局连贯与完成路径",
}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_plan_judge(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["顶层必须是 JSON object。"]
    checklist = obj.get("checklist")
    if not isinstance(checklist, dict):
        errors.append("checklist 必须是 object。")
    else:
        missing = [key for key in PLAN_CHECKLIST_KEYS if key not in checklist]
        if missing:
            errors.append(f"checklist 缺少指标: {missing}")
        for key in PLAN_CHECKLIST_KEYS:
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

    quality_checklist = obj.get("plan_quality_checklist")
    if not isinstance(quality_checklist, dict):
        errors.append("plan_quality_checklist 必须是 object。")
    else:
        missing = [key for key in PLAN_QUALITY_CHECKLIST_KEYS if key not in quality_checklist]
        if missing:
            errors.append(f"plan_quality_checklist 缺少指标: {missing}")
        for key in PLAN_QUALITY_CHECKLIST_KEYS:
            item = quality_checklist.get(key)
            if not isinstance(item, dict):
                errors.append(f"plan_quality_checklist.{key} 必须是 object。")
                continue
            if bool_value(item.get("pass")) is None:
                errors.append(f"plan_quality_checklist.{key}.pass 必须是 boolean。")
            if not isinstance(item.get("evidence"), list):
                errors.append(f"plan_quality_checklist.{key}.evidence 必须是数组。")
            if not isinstance(item.get("issues"), list):
                errors.append(f"plan_quality_checklist.{key}.issues 必须是数组。")
    return errors


def validate_instantiation_judge(expected_count: int) -> Callable[[Any], list[str]]:
    def validator(obj: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(obj, dict):
            return ["顶层必须是 JSON object。"]
        if bool_value(
            obj.get("overall_instantiation_preference_following", obj.get("overall_trajectory_preference_following"))
        ) is None:
            errors.append("overall_instantiation_preference_following 必须是 boolean。")
        judgements = obj.get("instantiation_judgements")
        if judgements is None:
            judgements = obj.get("tool_call_judgements")
        if not isinstance(judgements, list):
            errors.append("instantiation_judgements 必须是数组。")
            return errors
        if len(judgements) < expected_count:
            errors.append(f"instantiation_judgements 至少需要覆盖 {expected_count} 个工具调用，实际 {len(judgements)}。")
        for idx, item in enumerate(judgements[:expected_count], start=1):
            if not isinstance(item, dict):
                errors.append(f"instantiation_judgements[{idx}] 必须是 object。")
                continue
            if bool_value(item.get("json_object_preference_following", item.get("input_preference_following"))) is None:
                errors.append(f"instantiation_judgements[{idx}].json_object_preference_following 必须是 boolean。")
            if not isinstance(item.get("violated_preferences"), list):
                errors.append(f"instantiation_judgements[{idx}].violated_preferences 必须是数组。")
            if not isinstance(item.get("supporting_evidence"), list):
                errors.append(f"instantiation_judgements[{idx}].supporting_evidence 必须是数组。")
        return errors

    return validator


def validate_trajectory_judge(expected_count: int) -> Callable[[Any], list[str]]:
    return validate_instantiation_judge(expected_count)


def validate_update_judge(expected_count: int) -> Callable[[Any], list[str]]:
    def validator(obj: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(obj, dict):
            return ["顶层必须是 JSON object。"]
        judgements = obj.get("update_plan_judgements")
        if not isinstance(judgements, list):
            errors.append("update_plan_judgements 必须是数组。")
            return errors
        if len(judgements) < expected_count:
            errors.append(f"update_plan_judgements 至少需要覆盖 {expected_count} 次 update_plan，实际 {len(judgements)}。")
        for idx, item in enumerate(judgements[:expected_count], start=1):
            if not isinstance(item, dict):
                errors.append(f"update_plan_judgements[{idx}] 必须是 object。")
                continue
            for key in (*UPDATE_QUALITY_KEYS, *UPDATE_PLAN_QUALITY_KEYS, *UPDATE_PREFERENCE_KEYS):
                if bool_value(item.get(key)) is None:
                    errors.append(f"update_plan_judgements[{idx}].{key} 必须是 boolean。")
            if not isinstance(item.get("supporting_evidence"), list):
                errors.append(f"update_plan_judgements[{idx}].supporting_evidence 必须是数组。")
            if not isinstance(item.get("quality_issues"), list):
                errors.append(f"update_plan_judgements[{idx}].quality_issues 必须是数组。")
            if not isinstance(item.get("plan_quality_issues"), list):
                errors.append(f"update_plan_judgements[{idx}].plan_quality_issues 必须是数组。")
            if not isinstance(item.get("preference_issues"), list):
                errors.append(f"update_plan_judgements[{idx}].preference_issues 必须是数组。")
            if not isinstance(item.get("violated_preferences"), list):
                errors.append(f"update_plan_judgements[{idx}].violated_preferences 必须是数组。")
        return errors

    return validator


def normalize_plan_judge(judge: dict[str, Any]) -> dict[str, Any]:
    checklist = judge.get("checklist") if isinstance(judge.get("checklist"), dict) else {}
    quality_checklist = (
        judge.get("plan_quality_checklist") if isinstance(judge.get("plan_quality_checklist"), dict) else {}
    )
    normalized: dict[str, Any] = {
        "case_id": judge.get("case_id"),
        "first_plan_preference_following": False,
        "first_plan_quality_correct": False,
        "first_plan_joint_correct": False,
        "checklist": {},
        "plan_quality_checklist": {},
        "summary": judge.get("summary", ""),
        "confidence": judge.get("confidence"),
    }
    for key in PLAN_CHECKLIST_KEYS:
        item = checklist.get(key) if isinstance(checklist.get(key), dict) else {}
        passed = bool_value(item.get("pass")) is True
        normalized["checklist"][key] = {
            "pass": passed,
            "score": 1 if passed else 0,
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
            "issues": item.get("issues") if isinstance(item.get("issues"), list) else [],
        }
    for key in PLAN_QUALITY_CHECKLIST_KEYS:
        item = quality_checklist.get(key) if isinstance(quality_checklist.get(key), dict) else {}
        passed = bool_value(item.get("pass")) is True
        normalized["plan_quality_checklist"][key] = {
            "pass": passed,
            "score": 1 if passed else 0,
            "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
            "issues": item.get("issues") if isinstance(item.get("issues"), list) else [],
        }
    normalized["first_plan_preference_following"] = all(
        normalized["checklist"].get(key, {}).get("pass") is True for key in PLAN_CHECKLIST_KEYS
    )
    quality_correct_count = sum(
        int(normalized["plan_quality_checklist"].get(key, {}).get("pass") is True)
        for key in PLAN_QUALITY_CHECKLIST_KEYS
    )
    normalized["first_plan_quality_correct"] = quality_correct_count == len(PLAN_QUALITY_CHECKLIST_KEYS)
    normalized["first_plan_quality_score"] = round(
        quality_correct_count / len(PLAN_QUALITY_CHECKLIST_KEYS), 6
    )
    normalized["first_plan_joint_correct"] = bool(
        normalized["first_plan_preference_following"] and normalized["first_plan_quality_correct"]
    )
    return normalized


def normalize_instantiation_judge(judge: dict[str, Any], judged_calls: list[dict[str, Any]]) -> dict[str, Any]:
    raw_items = judge.get("instantiation_judgements")
    if not isinstance(raw_items, list):
        raw_items = judge.get("tool_call_judgements") if isinstance(judge.get("tool_call_judgements"), list) else []
    items: list[dict[str, Any]] = []
    all_pass = True
    for idx, call in enumerate(judged_calls):
        raw = raw_items[idx] if idx < len(raw_items) and isinstance(raw_items[idx], dict) else {}
        passed = bool_value(raw.get("json_object_preference_following", raw.get("input_preference_following"))) is True
        all_pass = all_pass and passed
        structured_output = call.get("structured_output") if isinstance(call.get("structured_output"), dict) else {}
        first_valid = bool(structured_output.get("first_attempt_valid"))
        final_valid = bool(structured_output.get("final_attempt_valid")) and not bool(structured_output.get("generation_failed"))
        score = 1.0 if first_valid and passed else 0.5 if final_valid and passed else 0.0
        items.append(
            {
                "call_index": call.get("call_index", raw.get("call_index", idx + 1)),
                "iteration": call.get("iteration", raw.get("iteration")),
                "api_name": call.get("api_name", raw.get("api_name")),
                "NoRelation": int(call.get("NoRelation") or 0),
                "structured_output": structured_output,
                "preference_sensitive_required_params": call.get("preference_sensitive_required_params", []),
                "nonpreference_required_params": call.get("nonpreference_required_params", []),
                "json_object_preference_following": passed,
                "input_preference_following": passed,
                "instantiation_score": score,
                "violated_preferences": raw.get("violated_preferences")
                if isinstance(raw.get("violated_preferences"), list)
                else [],
                "supporting_evidence": raw.get("supporting_evidence")
                if isinstance(raw.get("supporting_evidence"), list)
                else [],
                "argument_notes": raw.get("argument_notes") if isinstance(raw.get("argument_notes"), list) else [],
                "confidence": raw.get("confidence"),
            }
        )
    return {
        "case_id": judge.get("case_id"),
        "overall_instantiation_preference_following": all_pass,
        "overall_trajectory_preference_following": all_pass,
        "instantiation_judgements": items,
        "tool_call_judgements": items,
        "failure_summary": judge.get("failure_summary") if isinstance(judge.get("failure_summary"), list) else [],
        "confidence": judge.get("confidence"),
        "tool_call_count": len(judged_calls),
        "judged_tool_call_count": len(judged_calls),
    }


def normalize_trajectory_judge(judge: dict[str, Any], tool_count: int) -> dict[str, Any]:
    judged_calls = [{"call_index": idx + 1} for idx in range(tool_count)]
    return normalize_instantiation_judge(judge, judged_calls)


def normalize_update_judge(judge: dict[str, Any], update_count: int) -> dict[str, Any]:
    raw_items = judge.get("update_plan_judgements") if isinstance(judge.get("update_plan_judgements"), list) else []
    items: list[dict[str, Any]] = []
    all_pass = True
    all_quality_pass = True
    all_plan_quality_pass = True
    all_preference_pass = True
    for idx in range(update_count):
        raw = raw_items[idx] if idx < len(raw_items) and isinstance(raw_items[idx], dict) else {}
        quality_checks = {key: bool_value(raw.get(key)) is True for key in UPDATE_QUALITY_KEYS}
        plan_quality_checks = {key: bool_value(raw.get(key)) is True for key in UPDATE_PLAN_QUALITY_KEYS}
        preference_checks = {key: bool_value(raw.get(key)) is True for key in UPDATE_PREFERENCE_KEYS}
        quality_pass = all(quality_checks.values())
        plan_quality_pass = all(plan_quality_checks.values())
        preference_pass = all(preference_checks.values())
        passed = quality_pass and plan_quality_pass and preference_pass
        all_pass = all_pass and passed
        all_quality_pass = all_quality_pass and quality_pass
        all_plan_quality_pass = all_plan_quality_pass and plan_quality_pass
        all_preference_pass = all_preference_pass and preference_pass
        quality_issues = list_value(raw.get("quality_issues"))
        plan_quality_issues = list_value(raw.get("plan_quality_issues"))
        preference_issues = list_value(raw.get("preference_issues"))
        issues = list_value(raw.get("issues"))
        if not issues:
            issues = [*quality_issues, *plan_quality_issues, *preference_issues]
        items.append(
            {
                "update_index": raw.get("update_index", idx + 1),
                "iteration": raw.get("iteration"),
                "update_correct": passed,
                "update_joint_correct": passed,
                "update_quality_correct": quality_pass,
                **quality_checks,
                "update_plan_quality_correct": plan_quality_pass,
                "update_plan_quality_score": round(
                    sum(int(value) for value in plan_quality_checks.values()) / len(UPDATE_PLAN_QUALITY_KEYS), 6
                ),
                **plan_quality_checks,
                "update_preference_correct": preference_pass,
                **preference_checks,
                "supporting_evidence": list_value(raw.get("supporting_evidence")),
                "quality_issues": quality_issues,
                "plan_quality_issues": plan_quality_issues,
                "preference_issues": preference_issues,
                "violated_preferences": list_value(raw.get("violated_preferences")),
                "issues": issues,
                "confidence": raw.get("confidence"),
            }
        )
    return {
        "case_id": judge.get("case_id"),
        "overall_update_plan_correct": all_pass,
        "overall_update_joint_correct": all_pass,
        "overall_update_quality_correct": all_quality_pass,
        "overall_update_plan_quality_correct": all_plan_quality_pass,
        "overall_update_preference_correct": all_preference_pass,
        "update_plan_judgements": items,
        "failure_summary": judge.get("failure_summary") if isinstance(judge.get("failure_summary"), list) else [],
        "confidence": judge.get("confidence"),
        "update_plan_count": update_count,
    }


def chunked(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size 必须大于 0，当前为 {chunk_size}")
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def combine_update_judge_chunks(case_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    items = [item for chunk in chunks for item in chunk.get("update_plan_judgements", [])]
    return {
        "case_id": case_id,
        "overall_update_plan_correct": all(bool(item.get("update_correct")) for item in items),
        "overall_update_joint_correct": all(bool(item.get("update_joint_correct")) for item in items),
        "overall_update_quality_correct": all(bool(item.get("update_quality_correct")) for item in items),
        "overall_update_plan_quality_correct": all(
            bool(item.get("update_plan_quality_correct")) for item in items
        ),
        "overall_update_preference_correct": all(bool(item.get("update_preference_correct")) for item in items),
        "update_plan_judgements": items,
        "failure_summary": [
            str(summary)
            for chunk in chunks
            for summary in chunk.get("failure_summary", [])
        ],
        "confidence": min(
            (float(chunk["confidence"]) for chunk in chunks if isinstance(chunk.get("confidence"), (int, float))),
            default=None,
        ),
        "update_plan_count": len(items),
    }


def deterministic_missing_plan_judge(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "first_plan_preference_following": False,
        "first_plan_quality_correct": False,
        "first_plan_quality_score": 0.0,
        "first_plan_joint_correct": False,
        "checklist": {
            key: {
                "pass": False,
                "score": 0,
                "evidence": [],
                "issues": [reason],
            }
            for key in PLAN_CHECKLIST_KEYS
        },
        "plan_quality_checklist": {
            key: {
                "pass": False,
                "score": 0,
                "evidence": [],
                "issues": [reason],
            }
            for key in PLAN_QUALITY_CHECKLIST_KEYS
        },
        "summary": reason,
        "confidence": 1.0,
    }


def deterministic_empty_trajectory_judge(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "overall_instantiation_preference_following": True,
        "overall_trajectory_preference_following": True,
        "instantiation_judgements": [],
        "tool_call_judgements": [],
        "failure_summary": ["该 case 没有工具调用输入；按“每一次工具实例化均无违反”计为无实例化违规。"],
        "confidence": 1.0,
        "tool_call_count": 0,
        "instantiation_score_avg": 1.0,
    }


def deterministic_failed_empty_trajectory_judge(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "overall_instantiation_preference_following": False,
        "overall_trajectory_preference_following": False,
        "tool_call_input_preference_following": False,
        "update_plan_correct": False,
        "instantiation_judgements": [],
        "tool_call_judgements": [],
        "failure_summary": [reason],
        "confidence": 1.0,
        "tool_call_count": 0,
        "instantiation_score_avg": 0.0,
    }


def deterministic_instantiation_item(
    *,
    call: dict[str, Any],
    preference_following: bool,
    reason: str,
    no_relation_bypass: bool = False,
) -> dict[str, Any]:
    structured_output = call.get("structured_output") if isinstance(call.get("structured_output"), dict) else {}
    first_valid = bool(structured_output.get("first_attempt_valid"))
    final_valid = bool(structured_output.get("final_attempt_valid")) and not bool(structured_output.get("generation_failed"))
    passed = bool(preference_following and final_valid)
    score = 1.0 if first_valid and passed else 0.5 if final_valid and passed else 0.0
    return {
        "call_index": call.get("call_index"),
        "iteration": call.get("iteration"),
        "api_name": call.get("api_name"),
        "NoRelation": int(call.get("NoRelation") or 0),
        "no_relation_bypass": no_relation_bypass,
        "structured_output": structured_output,
        "preference_sensitive_required_params": call.get("preference_sensitive_required_params", []),
        "nonpreference_required_params": call.get("nonpreference_required_params", []),
        "json_object_preference_following": passed,
        "input_preference_following": passed,
        "instantiation_score": score,
        "violated_preferences": [] if passed else [reason],
        "supporting_evidence": [reason] if passed else [],
        "argument_notes": [reason],
        "confidence": 1.0,
    }


def combine_instantiation_judgements(
    *,
    case_id: str,
    calls: list[dict[str, Any]],
    llm_items: list[dict[str, Any]],
    failure_summary: list[str] | None = None,
    confidence: Any = None,
) -> dict[str, Any]:
    llm_by_call_index = {
        item.get("call_index"): item
        for item in llm_items
        if isinstance(item, dict)
    }
    items: list[dict[str, Any]] = []
    for call in calls:
        structured_output = call.get("structured_output") if isinstance(call.get("structured_output"), dict) else {}
        final_valid = bool(structured_output.get("final_attempt_valid")) and not bool(structured_output.get("generation_failed"))
        if not final_valid:
            item = deterministic_instantiation_item(
                call=call,
                preference_following=False,
                reason="reason_action JSON object 两次生成均失败，或最终结构化输出无效。",
                no_relation_bypass=bool(call.get("NoRelation") == 1),
            )
        elif int(call.get("NoRelation") or 0) == 1:
            item = deterministic_instantiation_item(
                call=call,
                preference_following=True,
                reason="该 API 标记 NoRelation=1，工具参数偏好无关，跳过 LLM-as-judge 后直接按偏好通过。",
                no_relation_bypass=True,
            )
        else:
            item = llm_by_call_index.get(call.get("call_index"))
            if not isinstance(item, dict):
                item = deterministic_instantiation_item(
                    call=call,
                    preference_following=False,
                    reason="偏好相关 API 缺少对应 LLM-as-judge 结果。",
                )
            else:
                structured_output = call.get("structured_output") if isinstance(call.get("structured_output"), dict) else {}
                passed = bool(item.get("json_object_preference_following", item.get("input_preference_following")))
                first_valid = bool(structured_output.get("first_attempt_valid"))
                final_valid = bool(structured_output.get("final_attempt_valid")) and not bool(structured_output.get("generation_failed"))
                score = item.get("instantiation_score")
                if not isinstance(score, (int, float)):
                    score = 1.0 if first_valid and passed else 0.5 if final_valid and passed else 0.0
                item = {
                    "call_index": call.get("call_index", item.get("call_index")),
                    "iteration": call.get("iteration", item.get("iteration")),
                    "api_name": call.get("api_name", item.get("api_name")),
                    "NoRelation": int(call.get("NoRelation") or 0),
                    "no_relation_bypass": False,
                    "structured_output": structured_output,
                    "preference_sensitive_required_params": call.get("preference_sensitive_required_params", []),
                    "nonpreference_required_params": call.get("nonpreference_required_params", []),
                    **item,
                    "json_object_preference_following": passed and final_valid,
                    "input_preference_following": passed and final_valid,
                    "instantiation_score": score,
                }
        items.append(item)

    overall = all(bool(item.get("json_object_preference_following")) for item in items)
    score_sum = sum(float(item.get("instantiation_score") or 0.0) for item in items)
    score_avg = round(score_sum / len(items), 6) if items else 1.0
    return {
        "case_id": case_id,
        "overall_instantiation_preference_following": overall,
        "tool_call_input_preference_following": overall,
        "overall_trajectory_preference_following": overall,
        "instantiation_judgements": items,
        "tool_call_judgements": items,
        "failure_summary": failure_summary if isinstance(failure_summary, list) else [],
        "confidence": confidence,
        "tool_call_count": len(calls),
        "judged_tool_call_count": sum(1 for call in calls if int(call.get("NoRelation") or 0) == 0),
        "no_relation_tool_call_count": sum(1 for call in calls if int(call.get("NoRelation") or 0) == 1),
        "instantiation_score_sum": score_sum,
        "instantiation_score_avg": score_avg,
    }


def deterministic_empty_update_judge(case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "overall_update_plan_correct": True,
        "overall_update_joint_correct": True,
        "overall_update_quality_correct": True,
        "overall_update_plan_quality_correct": True,
        "overall_update_preference_correct": True,
        "update_plan_judgements": [],
        "failure_summary": ["该 case 没有 update_plan 事件；按“没有可评估更新错误”计为通过。"],
        "confidence": 1.0,
        "update_plan_count": 0,
    }


def deterministic_failed_empty_update_judge(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "overall_update_plan_correct": False,
        "overall_update_joint_correct": False,
        "overall_update_quality_correct": False,
        "overall_update_plan_quality_correct": False,
        "overall_update_preference_correct": False,
        "update_plan_judgements": [],
        "failure_summary": [reason],
        "confidence": 1.0,
        "update_plan_count": 0,
    }


def discover_eval_records(logger: logging.Logger) -> list[dict[str, Any]]:
    selected_users = set(USER_ID_FILTER)
    selected_tasks = task_ids_for_active_filters()
    paths = sorted(EVAL_RESULT_DIR.glob(EVAL_RESULT_GLOB))
    records: list[dict[str, Any]] = []
    for path in paths:
        if selected_users and path.stem not in selected_users:
            continue
        for record in read_jsonl(path):
            if TARGET_EVAL_RUN_ID and record.get("run_id") != TARGET_EVAL_RUN_ID:
                continue
            if selected_tasks and record.get("task_id") not in selected_tasks:
                continue
            records.append(record)
    selected_eval_run_id = TARGET_EVAL_RUN_ID
    if records and not selected_eval_run_id and JUDGE_LATEST_EVAL_RUN_ONLY:
        selected_eval_run_id = max(str(record.get("run_id") or "") for record in records)
        records = [record for record in records if str(record.get("run_id") or "") == selected_eval_run_id]
    before_dedup_count = len(records)
    if records and DEDUP_EVAL_RECORDS_BY_CASE:
        latest_records: dict[tuple[str, str, int], dict[str, Any]] = {}
        for record in records:
            key = (
                str(record.get("user_id") or ""),
                str(record.get("task_id") or ""),
                int(record.get("pass_index") or 1),
            )
            previous = latest_records.get(key)
            if previous is None or str(record.get("run_id") or "") >= str(previous.get("run_id") or ""):
                latest_records[key] = record
        records = list(latest_records.values())
    logger.info(
        "已发现 eval 记录：dir=%s pass_n=%s files=%s records=%s before_dedup=%s dedup_by_case=%s target_run_id=%s latest_only=%s user_filter=%s task_filter=%s",
        EVAL_RESULT_DIR,
        PASS_N,
        len(paths),
        len(records),
        before_dedup_count,
        DEDUP_EVAL_RECORDS_BY_CASE,
        selected_eval_run_id or "<全部>",
        JUDGE_LATEST_EVAL_RUN_ONLY,
        USER_ID_FILTER,
        TASK_ID_FILTER,
    )
    return sorted(
        records,
        key=lambda item: (
            str(item.get("user_id")),
            str(item.get("task_id")),
            int(item.get("pass_index") or 1),
            str(item.get("run_id")),
        ),
    )


def case_id(record: dict[str, Any]) -> str:
    return (
        f"{record.get('run_id')}::{record.get('user_id')}::{record.get('task_id')}"
        f"::pass_{record.get('pass_n') or PASS_N}_{record.get('pass_index') or 1}"
    )


def task_core(record: dict[str, Any]) -> dict[str, Any]:
    provided = (
        record.get("provided_eval_context")
        if isinstance(record.get("provided_eval_context"), dict)
        else record.get("provided_preference_context")
        if isinstance(record.get("provided_preference_context"), dict)
        else {}
    )
    provided_context = provided.get("context") if isinstance(provided.get("context"), dict) else {}
    provided_task = provided_context.get("task") if isinstance(provided_context.get("task"), dict) else None
    if provided_task is not None:
        task = provided_task
    else:
        task_doc = record.get("task") if isinstance(record.get("task"), dict) else {}
        task = task_doc.get("task") if isinstance(task_doc.get("task"), dict) else task_doc
    keys = [
        "task_id",
        "title",
        "daily_life_domain",
        "scenario",
        "user_goal",
        "constraints",
        "relevant_user_preferences",
        "required_reasoning_steps",
        "decision_points",
        "information_to_collect",
        "expected_deliverable",
        "success_criteria",
        "potential_failure_modes",
    ]
    return {key: task.get(key) for key in keys if key in task}


_EXPLICIT_PROFILE_REFERENCE_CACHE: dict[str, dict[str, Any]] = {}


def explicit_profile_preference_reference(user_id: str) -> dict[str, Any]:
    cached = _EXPLICIT_PROFILE_REFERENCE_CACHE.get(user_id)
    if cached is not None:
        return cached
    path = DATA_DIR / user_id / "user_profile.json"
    if not path.exists():
        _EXPLICIT_PROFILE_REFERENCE_CACHE[user_id] = {}
        return {}
    profile = read_json(path)
    explicit = profile.get("explicit_preferences") if isinstance(profile.get("explicit_preferences"), dict) else {}
    detailed = explicit.get("detailed_user_profile") if isinstance(explicit.get("detailed_user_profile"), dict) else {}
    domain_preferences = detailed.get("explicit_preferences") if isinstance(detailed.get("explicit_preferences"), dict) else {}
    reference = {
        "preference_reference_source": "data/user_xxx/user_profile.json",
        "domain_preferences": domain_preferences,
        "dislikes_and_constraints": detailed.get("dislikes_and_constraints", []),
        "communication_style": detailed.get("communication_style", {}),
        "decision_making": detailed.get("decision_making", {}),
        "assistant_interaction_preferences": detailed.get("assistant_interaction_preferences", {}),
    }
    _EXPLICIT_PROFILE_REFERENCE_CACHE[user_id] = reference
    return reference


def profile_core(record: dict[str, Any]) -> dict[str, Any]:
    provided = (
        record.get("provided_eval_context")
        if isinstance(record.get("provided_eval_context"), dict)
        else record.get("provided_preference_context")
        if isinstance(record.get("provided_preference_context"), dict)
        else {}
    )
    provided_context = provided.get("context") if isinstance(provided.get("context"), dict) else {}
    mode = str(record.get("preference_injection_mode") or provided.get("mode") or PREFERENCE_INJECTION_MODE)
    if mode == "hidden" and USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE:
        return {
            "preference_injection_mode": mode,
            "preference_injection_label": "user profile 显式偏好对照参考",
            "explicit_user_profile_preference_reference": explicit_profile_preference_reference(
                str(record.get("user_id") or "")
            ),
        }
    if provided_context:
        core: dict[str, Any] = {
            "preference_injection_mode": mode,
            "preference_injection_label": PREFERENCE_INJECTION_LABELS.get(mode, mode),
        }
        if "user_profile_summary" in provided_context:
            core["explicit_user_profile"] = provided_context.get("user_profile_summary")
        if "implicit_interaction_history" in provided_context:
            core["implicit_interaction_history"] = provided_context.get("implicit_interaction_history")
        if "non_preference_long_context" in provided_context:
            core["non_preference_long_context"] = provided_context.get("non_preference_long_context")
        if "user_information" in provided_context:
            core["user_information"] = provided_context.get("user_information")
        if "user_interaction_history" in provided_context:
            core["user_interaction_history"] = provided_context.get("user_interaction_history")
        if "pag_context" in provided_context:
            pag_context = provided_context.get("pag_context") if isinstance(provided_context.get("pag_context"), dict) else {}
            core["pag_retrieved_history"] = pag_context.get("retrieved_hidden_history")
            core["pag_user_abstract"] = pag_context.get("user_abstract")
        if "bank_context" in provided_context:
            bank_context = provided_context.get("bank_context") if isinstance(provided_context.get("bank_context"), dict) else {}
            core["bank_user_abstract_profile"] = bank_context.get("user_abstract_profile")
            core["bank_retrieved_qa_memories"] = bank_context.get("retrieved_qa_memories")
        if core.keys() - {"preference_injection_mode", "preference_injection_label"}:
            return core

    profile = record.get("user_profile") if isinstance(record.get("user_profile"), dict) else {}
    explicit = profile.get("explicit_preferences") if isinstance(profile.get("explicit_preferences"), dict) else {}
    detailed = explicit.get("detailed_user_profile") if isinstance(explicit.get("detailed_user_profile"), dict) else {}
    sandbox_context = record.get("sandbox_context") if isinstance(record.get("sandbox_context"), dict) else {}
    return {
        "preference_injection_mode": mode,
        "preference_injection_label": PREFERENCE_INJECTION_LABELS.get(mode, mode),
        "explicit_preferences": explicit,
        "sandbox_user_context": sandbox_context.get("user_context", {}),
        "assistant_interaction_preferences": detailed.get("assistant_interaction_preferences", {}),
        "decision_making": detailed.get("decision_making", {}),
        "dislikes_and_constraints": detailed.get("dislikes_and_constraints", []),
        "communication_style": detailed.get("communication_style", {}),
    }


def sandbox_core(record: dict[str, Any]) -> dict[str, Any]:
    provided = (
        record.get("provided_eval_context")
        if isinstance(record.get("provided_eval_context"), dict)
        else record.get("provided_preference_context")
        if isinstance(record.get("provided_preference_context"), dict)
        else {}
    )
    provided_context = provided.get("context") if isinstance(provided.get("context"), dict) else {}
    if provided_context:
        return {
            "sandbox_user_context": provided_context.get("sandbox_user_context", provided_context.get("user_context", {})),
            "accessible_api_names": provided_context.get("available_api_names", []),
            "workflow_available_api_names": provided_context.get("available_api_names", []),
        }

    sandbox_context = record.get("sandbox_context") if isinstance(record.get("sandbox_context"), dict) else {}
    workflow = sandbox_context.get("workflow") if isinstance(sandbox_context.get("workflow"), dict) else {}
    return {
        "sandbox_user_context": sandbox_context.get("user_context", {}),
        "accessible_api_names": sandbox_context.get("accessible_api_names", []),
        "workflow_available_api_names": workflow.get("available_api_names", []),
    }


def build_judge_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id(record),
        "user_id": record.get("user_id"),
        "task_id": record.get("task_id"),
        "pass_n": record.get("pass_n") or PASS_N,
        "pass_index": record.get("pass_index") or 1,
        "task_title": record.get("task_title"),
        "preference_injection_mode": record.get("preference_injection_mode"),
        "preference_injection_label": record.get("preference_injection_label"),
        "eval_status": record.get("status"),
        "task": task_core(record),
        "user_preferences": profile_core(record),
        "sandbox": sandbox_core(record),
    }


def first_plan(record: dict[str, Any]) -> dict[str, Any] | None:
    plans = record.get("plan_versions")
    if isinstance(plans, list) and plans and isinstance(plans[0], dict):
        return plans[0]
    trajectory = record.get("trajectory")
    if isinstance(trajectory, list):
        for event in trajectory:
            if isinstance(event, dict) and event.get("phase") == "initial_plan" and isinstance(event.get("parsed"), dict):
                return event["parsed"]
    return None


def tool_call_inputs(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, item in enumerate(record.get("tool_calls", []) if isinstance(record.get("tool_calls"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        tool_input = item.get("tool_call_input")
        if not isinstance(tool_input, dict):
            continue
        calls.append(
            {
                "call_index": idx,
                "iteration": item.get("iteration"),
                "api_name": tool_input.get("name"),
                "arguments": tool_input.get("arguments"),
                "NoRelation": api_no_relation(tool_input.get("name")),
                "preference_sensitive_required_params": api_preference_sensitive_required_params(tool_input.get("name")),
                "nonpreference_required_params": api_nonpreference_required_params(tool_input.get("name")),
                "structured_output": structured_output_status_for_iteration(record, item.get("iteration")),
                "sandbox_status": (item.get("sandbox_observation") or {}).get("sandbox_status")
                if isinstance(item.get("sandbox_observation"), dict)
                else None,
            }
        )

    reason_by_iteration: dict[Any, dict[str, Any]] = {}
    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    for event in trajectory:
        if not isinstance(event, dict) or event.get("phase") != "reason_action":
            continue
        parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
        reason_by_iteration[event.get("iteration")] = {
            "selected_step_id": parsed.get("selected_step_id"),
            "reason": compact_text(parsed.get("reason"), MAX_REASON_CHARS_PER_TOOL),
            "expected_observation": compact_text(parsed.get("expected_observation"), MAX_REASON_CHARS_PER_TOOL),
        }

    for call in calls:
        call.update(reason_by_iteration.get(call.get("iteration"), {}))
    return calls[:MAX_TOOL_CALLS_PER_PROMPT]


def structured_output_status_for_iteration(record: dict[str, Any], iteration: Any) -> dict[str, Any]:
    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    for event in trajectory:
        if not isinstance(event, dict) or event.get("phase") != "reason_action" or event.get("iteration") != iteration:
            continue
        attempts = event.get("llm_attempts") if isinstance(event.get("llm_attempts"), list) else []
        if not attempts:
            return {
                "attempt_count": 0,
                "first_attempt_valid": False,
                "final_attempt_valid": False,
                "repair_used": False,
                "generation_failed": True,
                "errors": ["reason_action 缺少 llm_attempts 记录。"],
            }
        first_ok = bool((attempts[0] or {}).get("ok")) if isinstance(attempts[0], dict) else False
        final_ok = bool((attempts[-1] or {}).get("ok")) if isinstance(attempts[-1], dict) else False
        parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
        generation_failed = (
            bool(parsed.get("structured_output_error"))
            or parsed.get("tool_call", {}).get("name") == "__action_json_generation__"
        )
        errors = [
            str(attempt.get("parse_or_validation_error"))
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("parse_or_validation_error")
        ]
        return {
            "attempt_count": len(attempts),
            "first_attempt_valid": first_ok,
            "final_attempt_valid": final_ok and not generation_failed,
            "repair_used": not first_ok and len(attempts) > 1,
            "generation_failed": generation_failed or not final_ok,
            "errors": errors,
        }
    return {
        "attempt_count": 0,
        "first_attempt_valid": False,
        "final_attempt_valid": False,
        "repair_used": False,
        "generation_failed": True,
        "errors": [f"未找到 iteration={iteration} 的 reason_action 事件。"],
    }


def update_plan_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), list) else []
    events_by_iteration: dict[Any, dict[str, Any]] = {}
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        iteration = event.get("iteration")
        if iteration is None:
            continue
        item = events_by_iteration.setdefault(iteration, {"iteration": iteration})
        phase = event.get("phase")
        if phase == "reason_action":
            parsed = event.get("parsed") if isinstance(event.get("parsed"), dict) else {}
            item["reason_action"] = {
                "selected_step_id": parsed.get("selected_step_id"),
                "reason": compact_text(parsed.get("reason"), MAX_REASON_CHARS_PER_TOOL),
                "action_type": parsed.get("action_type"),
                "tool_call": parsed.get("tool_call"),
                "expected_observation": compact_text(parsed.get("expected_observation"), MAX_REASON_CHARS_PER_TOOL),
                "completion_assessment": parsed.get("completion_assessment"),
            }
        elif phase == "sandbox_observation":
            item["sandbox_observation"] = event.get("observation")
        elif phase == "update_plan":
            item["updated_plan"] = event.get("parsed")

    updates: list[dict[str, Any]] = []
    previous_plan: dict[str, Any] | None = first_plan(record)
    for iteration in sorted(events_by_iteration, key=lambda value: (value is None, value)):
        item = events_by_iteration[iteration]
        updated_plan = item.get("updated_plan")
        if not isinstance(updated_plan, dict):
            continue
        updates.append(
            {
                "update_index": len(updates) + 1,
                "iteration": iteration,
                "previous_plan": previous_plan or {},
                "reason_action": item.get("reason_action", {}),
                "sandbox_observation": item.get("sandbox_observation", {}),
                "updated_plan": updated_plan,
            }
        )
        previous_plan = updated_plan
        if len(updates) >= MAX_UPDATE_EVENTS_PER_PROMPT:
            break
    return updates


def preference_source_description(record: dict[str, Any]) -> str:
    mode = str(record.get("preference_injection_mode") or PREFERENCE_INJECTION_MODE)
    if mode == "hidden":
        if USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE:
            return "user profile 中记录的显式用户偏好对照参考"
        return "用户交互历史中呈现的用户偏好和行为倾向"
    if mode == "pag":
        return "PAG 检索历史 QA 与任务相关用户摘要中呈现的用户偏好和行为倾向"
    if mode == "bank":
        return "BANK 用户抽象画像及从 question/answer/summary 记忆库检索的任务相关记录中呈现的用户偏好和行为倾向"
    if mode == "mix":
        return "用户信息与用户交互历史中呈现的用户偏好和行为倾向"
    return "用户信息中呈现的用户偏好和行为倾向"


def plan_judge_messages(record: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, str]]:
    return first_plan_prompt.build_messages(
        case_id=case_id(record),
        preference_source=preference_source_description(record),
        judge_context=build_judge_context(record),
        plan=plan,
        compact_json=compact_json,
        max_context_chars=MAX_CONTEXT_CHARS,
        max_plan_chars=MAX_PLAN_CHARS,
    )


def trajectory_judge_messages(
    record: dict[str, Any],
    plan: dict[str, Any] | None,
    calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return trajectory_prompt.build_messages(
        case_id=case_id(record),
        preference_source=preference_source_description(record),
        judge_context=build_judge_context(record),
        plan=plan,
        calls=calls,
        compact_json=compact_json,
        max_context_chars=MAX_CONTEXT_CHARS,
        max_plan_chars=MAX_PLAN_CHARS,
        max_tool_input_chars=MAX_TOOL_INPUT_CHARS,
    )


def update_plan_judge_messages(record: dict[str, Any], updates: list[dict[str, Any]]) -> list[dict[str, str]]:
    return update_plan_prompt.build_messages(
        case_id=case_id(record),
        judge_context=build_judge_context(record),
        updates=updates,
        compact_json=compact_json,
        max_context_chars=MAX_CONTEXT_CHARS,
        max_update_judge_chars=MAX_UPDATE_JUDGE_CHARS,
    )


def judge_case(record: dict[str, Any], client: LLMClient, logger: logging.Logger) -> dict[str, Any]:
    cid = case_id(record)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "judge_run_created_at": utc_now_iso(),
        "case_id": cid,
        "run_id": record.get("run_id"),
        "user_id": record.get("user_id"),
        "task_id": record.get("task_id"),
        "pass_n": record.get("pass_n") or PASS_N,
        "pass_index": record.get("pass_index") or 1,
        "task_title": record.get("task_title"),
        "eval_status": record.get("status"),
        "eval_model": record.get("model", {}),
        "eval_hyperparameters": record.get("hyperparameters", {}),
        "trajectory_pattern_ok": (record.get("trajectory_pattern_check") or {}).get("ok"),
        "tool_call_count": len(record.get("tool_calls", []) if isinstance(record.get("tool_calls"), list) else []),
        "plan_judge": None,
        "trajectory_judge": None,
        "update_judge": None,
        "judge_errors": [],
        "elapsed_seconds": None,
    }

    plan = first_plan(record)
    if plan is None:
        result["plan_judge"] = {
            "parsed": deterministic_missing_plan_judge(cid, "eval 结果中缺少 first plan，无法证明计划遵循偏好。"),
            "attempts": [],
        }
    else:
        try:
            parsed, attempts = request_structured_json(
                client,
                messages=plan_judge_messages(record, plan),
                purpose=f"{cid}/first_plan",
                validator=validate_plan_judge,
                logger=logger,
            )
            result["plan_judge"] = {
                "parsed": normalize_plan_judge(parsed),
                "attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001 - judge should isolate per-case failures.
            result["judge_errors"].append(
                {
                    "phase": "first_plan",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": getattr(exc, "attempts", []),
                }
            )
            result["plan_judge"] = {
                "parsed": deterministic_missing_plan_judge(cid, f"first plan judge 失败: {exc}"),
                "attempts": getattr(exc, "attempts", []),
            }

    calls = tool_call_inputs(record)
    if not calls:
        if str(record.get("status") or "") == "failed_exception":
            reason = "eval 在生成可执行轨迹前失败，缺少工具调用输入，trajectory 不能按成功计入。"
            result["instantiation_judge"] = {
                "parsed": deterministic_failed_empty_trajectory_judge(cid, reason),
                "attempts": [],
            }
        else:
            result["instantiation_judge"] = {
                "parsed": deterministic_empty_trajectory_judge(cid),
                "attempts": [],
            }
        result["trajectory_judge"] = result["instantiation_judge"]
    else:
        llm_calls = [
            call
            for call in calls
            if int(call.get("NoRelation") or 0) == 0
            and ((call.get("structured_output") or {}).get("final_attempt_valid") is True)
            and ((call.get("structured_output") or {}).get("generation_failed") is not True)
        ]
        try:
            if llm_calls:
                attempts = []
                llm_items: list[dict[str, Any]] = []
                failure_summary: list[str] = []
                confidences: list[float] = []
                call_chunks = chunked(llm_calls, INSTANTIATION_JUDGE_CHUNK_SIZE)
                for chunk_index, call_chunk in enumerate(call_chunks, start=1):
                    parsed, chunk_attempts = request_structured_json(
                        client,
                        messages=trajectory_judge_messages(record, plan, call_chunk),
                        purpose=f"{cid}/instantiation/chunk_{chunk_index}_of_{len(call_chunks)}",
                        validator=validate_instantiation_judge(len(call_chunk)),
                        logger=logger,
                    )
                    chunk_parsed = normalize_instantiation_judge(parsed, call_chunk)
                    attempts.extend(chunk_attempts)
                    llm_items.extend(chunk_parsed.get("instantiation_judgements", []))
                    failure_summary.extend(chunk_parsed.get("failure_summary", []))
                    if isinstance(chunk_parsed.get("confidence"), (int, float)):
                        confidences.append(float(chunk_parsed["confidence"]))
                combined = combine_instantiation_judgements(
                    case_id=cid,
                    calls=calls,
                    llm_items=llm_items,
                    failure_summary=failure_summary,
                    confidence=min(confidences) if confidences else None,
                )
            else:
                attempts = []
                combined = combine_instantiation_judgements(case_id=cid, calls=calls, llm_items=[])
            result["instantiation_judge"] = {
                "parsed": combined,
                "attempts": attempts,
            }
            result["trajectory_judge"] = result["instantiation_judge"]
        except Exception as exc:  # noqa: BLE001 - judge should isolate per-case failures.
            result["judge_errors"].append(
                {
                    "phase": "instantiation",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": getattr(exc, "attempts", []),
                }
            )
            failed_items = [
                deterministic_instantiation_item(
                    call=call,
                    preference_following=(int(call.get("NoRelation") or 0) == 1),
                    reason=f"instantiation judge 失败: {exc}",
                    no_relation_bypass=bool(int(call.get("NoRelation") or 0) == 1),
                )
                for call in calls
            ]
            failed = combine_instantiation_judgements(
                case_id=cid,
                calls=calls,
                llm_items=failed_items,
                failure_summary=[f"instantiation judge 失败: {exc}"],
                confidence=1.0,
            )
            result["instantiation_judge"] = {
                "parsed": failed,
                "attempts": getattr(exc, "attempts", []),
            }
            result["trajectory_judge"] = result["instantiation_judge"]

    updates = update_plan_events(record)
    if not updates:
        if str(record.get("status") or "") == "failed_exception":
            reason = "eval 在生成 update_plan 前失败，缺少可评估更新，update plan 不能按成功计入。"
            result["update_judge"] = {
                "parsed": deterministic_failed_empty_update_judge(cid, reason),
                "attempts": [],
            }
        else:
            result["update_judge"] = {
                "parsed": deterministic_empty_update_judge(cid),
                "attempts": [],
            }
    else:
        try:
            attempts = []
            parsed_chunks: list[dict[str, Any]] = []
            update_chunks = chunked(updates, UPDATE_JUDGE_CHUNK_SIZE)
            for chunk_index, update_chunk in enumerate(update_chunks, start=1):
                parsed, chunk_attempts = request_structured_json(
                    client,
                    messages=update_plan_judge_messages(record, update_chunk),
                    purpose=f"{cid}/update_plan/chunk_{chunk_index}_of_{len(update_chunks)}",
                    validator=validate_update_judge(len(update_chunk)),
                    logger=logger,
                )
                attempts.extend(chunk_attempts)
                parsed_chunks.append(normalize_update_judge(parsed, len(update_chunk)))
            result["update_judge"] = {
                "parsed": combine_update_judge_chunks(cid, parsed_chunks),
                "attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001 - judge should isolate per-case failures.
            result["judge_errors"].append(
                {
                    "phase": "update_plan",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "attempts": getattr(exc, "attempts", []),
                }
            )
            failed = {
                "case_id": cid,
                "overall_update_plan_correct": False,
                "overall_update_joint_correct": False,
                "overall_update_quality_correct": False,
                "overall_update_plan_quality_correct": False,
                "overall_update_preference_correct": False,
                "update_plan_judgements": [
                    {
                        "update_index": update.get("update_index", idx + 1),
                        "iteration": update.get("iteration"),
                        "update_correct": False,
                        "update_joint_correct": False,
                        "update_quality_correct": False,
                        "observation_integration": False,
                        "status_and_evidence_update": False,
                        "next_step_consistency": False,
                        "no_fabrication_or_contradiction": False,
                        "update_plan_quality_correct": False,
                        "update_plan_quality_score": 0.0,
                        **{key: False for key in UPDATE_PLAN_QUALITY_KEYS},
                        "update_preference_correct": False,
                        "preference_retention": False,
                        "preference_aware_replanning": False,
                        "planned_arguments_preference_following": False,
                        "no_preference_violation": False,
                        "supporting_evidence": [],
                        "quality_issues": [f"update_plan judge 失败: {exc}"],
                        "plan_quality_issues": [f"update_plan judge 失败: {exc}"],
                        "preference_issues": [f"update_plan judge 失败: {exc}"],
                        "violated_preferences": [],
                        "issues": [f"update_plan judge 失败: {exc}"],
                        "confidence": 1.0,
                    }
                    for idx, update in enumerate(updates)
                ],
                "failure_summary": [f"update_plan judge 失败: {exc}"],
                "confidence": 1.0,
                "update_plan_count": len(updates),
            }
            result["update_judge"] = {
                "parsed": failed,
                "attempts": getattr(exc, "attempts", []),
            }

    merge_update_result_into_trajectory(result)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result

def probe_once(client: LLMClient, probe_id: str) -> tuple[bool, float, str]:
    messages = concurrency_probe.build_messages(probe_id)
    started = time.perf_counter()
    try:
        response = client.chat(
            messages,
            purpose=f"judge_concurrency_probe/{probe_id}",
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
        logger.info("根据项目 API 约定，Qingyun DeepSeek judge 并发采用 %s。", QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY)
        return QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY

    if client.base_url == "https://api.deepseek.com" and client.model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        logger.info("根据项目 API 约定，DeepSeek judge 并发采用 %s。", DEEPSEEK_JUDGE_CONCURRENCY)
        return DEEPSEEK_JUDGE_CONCURRENCY

    if not AUTO_TUNE_CONCURRENCY:
        logger.info("未启用 judge 并发探测，使用 DEFAULT_JUDGE_CONCURRENCY=%s", DEFAULT_JUDGE_CONCURRENCY)
        return DEFAULT_JUDGE_CONCURRENCY

    best = 1
    logger.info("开始 judge 并发探测，候选值=%s", CONCURRENCY_CANDIDATES)
    for candidate in CONCURRENCY_CANDIDATES:
        started = time.perf_counter()
        results: list[tuple[bool, float, str]] = []
        with ThreadPoolExecutor(max_workers=candidate, thread_name_prefix=f"judge_probe{candidate}") as pool:
            futures = [pool.submit(probe_once, client, f"{candidate}_{idx}") for idx in range(candidate)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.perf_counter() - started
        success_count = sum(1 for ok, _, _ in results if ok)
        success_rate = success_count / candidate
        avg_latency = sum(latency for _, latency, _ in results) / max(1, len(results))
        errors = [error for ok, _, error in results if not ok for error in [error]]
        logger.info(
            "judge 并发探测结果：candidate=%s success=%s/%s success_rate=%.2f avg_latency=%.2fs elapsed=%.2fs errors=%s",
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
        logger.info("judge 并发 %s 未达成功率阈值，停止继续升高并发。", candidate)
        break
    logger.info("最终采用 judge 并发：%s", best)
    return best


def plan_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("plan_judge") or {}).get("parsed") or {}
    return bool(parsed.get("first_plan_preference_following"))


def plan_quality_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("plan_judge") or {}).get("parsed") or {}
    return bool(parsed.get("first_plan_quality_correct"))


def plan_joint_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("plan_judge") or {}).get("parsed") or {}
    return bool(parsed.get("first_plan_joint_correct"))


def instantiation_parsed(result: dict[str, Any]) -> dict[str, Any]:
    parsed = (result.get("instantiation_judge") or {}).get("parsed")
    if isinstance(parsed, dict):
        return parsed
    parsed = (result.get("trajectory_judge") or {}).get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def instantiation_pass(result: dict[str, Any]) -> bool:
    parsed = instantiation_parsed(result)
    return bool(
        parsed.get(
            "overall_instantiation_preference_following",
            parsed.get("tool_call_input_preference_following", parsed.get("overall_trajectory_preference_following")),
        )
    )


def instantiation_score(result: dict[str, Any]) -> float:
    parsed = instantiation_parsed(result)
    if isinstance(parsed.get("instantiation_score_avg"), (int, float)):
        return float(parsed["instantiation_score_avg"])
    items = parsed.get("instantiation_judgements")
    if not isinstance(items, list):
        items = parsed.get("tool_call_judgements") if isinstance(parsed.get("tool_call_judgements"), list) else []
    if not items:
        return 1.0
    return sum(float(item.get("instantiation_score") or 0.0) for item in items if isinstance(item, dict)) / len(items)


def tool_input_pass(result: dict[str, Any]) -> bool:
    return instantiation_pass(result)


def update_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("update_judge") or {}).get("parsed") or {}
    return bool(parsed.get("overall_update_plan_correct"))


def update_quality_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("update_judge") or {}).get("parsed") or {}
    return bool(parsed.get("overall_update_quality_correct"))


def update_plan_quality_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("update_judge") or {}).get("parsed") or {}
    return bool(parsed.get("overall_update_plan_quality_correct"))


def update_preference_pass(result: dict[str, Any]) -> bool:
    parsed = (result.get("update_judge") or {}).get("parsed") or {}
    return bool(parsed.get("overall_update_preference_correct"))


def trajectory_pass(result: dict[str, Any]) -> bool:
    return instantiation_pass(result) and update_pass(result)


def merge_update_result_into_trajectory(result: dict[str, Any]) -> None:
    trajectory = result.get("instantiation_judge") if isinstance(result.get("instantiation_judge"), dict) else {}
    if not trajectory:
        trajectory = result.get("trajectory_judge") if isinstance(result.get("trajectory_judge"), dict) else {}
    parsed = trajectory.get("parsed") if isinstance(trajectory.get("parsed"), dict) else None
    if parsed is None:
        return

    tool_only_ok = bool(
        parsed.get(
            "overall_instantiation_preference_following",
            parsed.get("tool_call_input_preference_following", parsed.get("overall_trajectory_preference_following")),
        )
    )
    update_ok = update_pass(result)
    parsed["overall_instantiation_preference_following"] = tool_only_ok
    parsed["tool_call_input_preference_following"] = tool_only_ok
    parsed["update_plan_correct"] = update_ok
    parsed["overall_trajectory_preference_following"] = tool_only_ok and update_ok
    if not update_ok:
        failure_summary = parsed.get("failure_summary") if isinstance(parsed.get("failure_summary"), list) else []
        update_parsed = (result.get("update_judge") or {}).get("parsed") or {}
        update_summary = update_parsed.get("failure_summary") if isinstance(update_parsed.get("failure_summary"), list) else []
        failure_summary.append("update_plan 操作未通过正确性评估，因此轨迹整体按失败计入。")
        failure_summary.extend(str(item) for item in update_summary[:2])
        parsed["failure_summary"] = failure_summary


def pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def pct_float(numerator: float, denom: int) -> float:
    """macro 平均用：分子是各轨迹比例之和（浮点），分母是轨迹条数。"""
    if denom <= 0:
        return 0.0
    return round(numerator * 100.0 / denom, 2)


def _observation_status_by_iteration(eval_result_dir: Path, result: dict[str, Any]) -> dict[int, str]:
    """从 runresult 读取该 case 每一轮工具观测的 sandbox_status，按 iteration 索引。"""
    user_id = str(result.get("user_id") or "")
    task_id = str(result.get("task_id") or "")
    pass_index = result.get("pass_index")
    path = Path(eval_result_dir) / f"{user_id}.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("task_id")) != task_id:
            continue
        if pass_index is not None and rec.get("pass_index") not in (None, pass_index):
            continue
        status: dict[int, str] = {}
        for tc in rec.get("tool_calls") or []:
            obs = tc.get("sandbox_observation") or {}
            if isinstance(obs, str):
                try:
                    obs = json.loads(obs)
                except json.JSONDecodeError:
                    obs = {}
            it = tc.get("iteration")
            if isinstance(it, int):
                status[it] = str(obs.get("sandbox_status") or "")
        return status
    return {}


def compute_recovery_metrics(results: list[dict[str, Any]], eval_result_dir: Path) -> dict[str, Any]:
    """审稿意见(2)：对"观测导致当前计划失效"的轮次，专门报告恢复准确率。

    锚点（分母）= 失效观测（sandbox_status == 'error'）。对每个失效观测 iter=k：
      - replan 口径：下一轮 update_plan[k] 正确吸收观测（observation_integration=True）
      - 执行口径：在 replan 正确的前提下，基于新计划的再执行 tool_call[k+1] 也通过
    失效观测后若没有对应的有效 update 判定，计入分母、不计入分子（即“该恢复却没恢复”记为失败）。
    """
    replan_denom = 0
    replan_ok = 0
    exec_denom = 0
    exec_ok = 0
    for item in results:
        obs_status = _observation_status_by_iteration(eval_result_dir, item)
        if not obs_status:
            continue
        updates = (((item.get("update_judge") or {}).get("parsed") or {}).get("update_plan_judgements") or [])
        update_by_iter = {u.get("iteration"): u for u in updates if isinstance(u, dict)}
        inst = (((item.get("instantiation_judge") or {}).get("parsed") or {}).get("instantiation_judgements") or [])
        if not isinstance(inst, list) or not inst:
            inst = (((item.get("instantiation_judge") or {}).get("parsed") or {}).get("tool_call_judgements") or [])
        inst_by_iter = {t.get("iteration"): t for t in inst if isinstance(t, dict)}

        for it, status in obs_status.items():
            if status != "error":
                continue
            # replan 口径
            replan_denom += 1
            u = update_by_iter.get(it)
            replanned = bool(u and u.get("observation_integration"))
            if replanned:
                replan_ok += 1
            # 执行口径：需要存在下一轮再执行
            nxt = inst_by_iter.get(it + 1)
            if nxt is not None:
                exec_denom += 1
                nxt_ok = bool(nxt.get("json_object_preference_following", nxt.get("input_preference_following")))
                if replanned and nxt_ok:
                    exec_ok += 1
    return {
        "recovery_replan_accuracy_ratio": pct(replan_ok, replan_denom),
        "recovery_replan_case_count": replan_denom,
        "recovery_replan_ok_count": replan_ok,
        "recovery_execution_accuracy_ratio": pct(exec_ok, exec_denom),
        "recovery_execution_case_count": exec_denom,
        "recovery_execution_ok_count": exec_ok,
    }


def pass_k_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    pass_values = sorted({int(item.get("pass_n") or PASS_N) for item in results})
    pass_n = max(pass_values) if pass_values else normalize_pass_n(PASS_N)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        groups[(str(item.get("user_id")), str(item.get("task_id")))].append(item)

    complete_groups = {
        key: sorted(value, key=lambda item: int(item.get("pass_index") or 1))
        for key, value in groups.items()
        if len({int(item.get("pass_index") or 1) for item in value}) == pass_n
    }

    def metric_counts(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        total = len(complete_groups)
        pass_at_k_count = sum(1 for runs in complete_groups.values() if any(predicate(item) for item in runs))
        pass_power_k_count = sum(1 for runs in complete_groups.values() if all(predicate(item) for item in runs))
        return {
            "pass_at_k_count": pass_at_k_count,
            "pass_at_k_ratio": pct(pass_at_k_count, total),
            "pass_power_k_count": pass_power_k_count,
            "pass_power_k_ratio": pct(pass_power_k_count, total),
        }

    return {
        "pass_n": pass_n,
        "task_group_count": len(groups),
        "complete_task_group_count": len(complete_groups),
        "incomplete_task_group_count": len(groups) - len(complete_groups),
        "denominator": len(complete_groups),
        "metrics": {
            "first_plan": metric_counts(plan_pass),
            "first_plan_quality": metric_counts(plan_quality_pass),
            "first_plan_joint": metric_counts(plan_joint_pass),
            "instantiation": metric_counts(instantiation_pass),
            "trajectory": metric_counts(trajectory_pass),
            "update_plan": metric_counts(update_pass),
            "update_quality": metric_counts(update_quality_pass),
            "update_plan_quality": metric_counts(update_plan_quality_pass),
            "update_joint": metric_counts(update_pass),
            "update_preference": metric_counts(update_preference_pass),
        },
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    first_plan_ok = sum(1 for item in results if plan_pass(item))
    first_plan_quality_ok = sum(1 for item in results if plan_quality_pass(item))
    first_plan_joint_ok = sum(1 for item in results if plan_joint_pass(item))
    instantiation_ok = sum(1 for item in results if instantiation_pass(item))
    trajectory_ok = sum(1 for item in results if trajectory_pass(item))
    update_ok = sum(1 for item in results if update_pass(item))
    update_quality_ok = sum(1 for item in results if update_quality_pass(item))
    update_plan_quality_ok = sum(1 for item in results if update_plan_quality_pass(item))
    update_preference_ok = sum(1 for item in results if update_preference_pass(item))
    status_counts = Counter(str(item.get("eval_status")) for item in results)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    checklist_counts = {key: 0 for key in PLAN_CHECKLIST_KEYS}
    plan_quality_checklist_counts = {key: 0 for key in PLAN_QUALITY_CHECKLIST_KEYS}
    tool_total = 0
    tool_ok = 0
    instantiation_score_sum = 0.0
    no_relation_tool_total = 0
    judged_relation_tool_total = 0
    update_total = 0
    update_item_ok = 0
    update_quality_item_ok = 0
    update_plan_quality_item_ok = 0
    update_preference_item_ok = 0
    update_plan_quality_atomic_ok = 0
    update_plan_quality_check_counts = {key: 0 for key in UPDATE_PLAN_QUALITY_KEYS}
    zero_update_cases = 0
    zero_tool_cases = 0
    api_counts: dict[str, Counter[str]] = defaultdict(Counter)
    # 审稿意见(1)：按轨迹长度归一化（macro 平均）——先在每条轨迹内部算通过比例，
    # 再对所有轨迹等权平均，避免长轨迹（hard）因严格全有或全无指标被系统性压分。
    traj_update_frac_sum = 0.0
    traj_update_frac_denom = 0
    traj_inst_frac_sum = 0.0
    traj_inst_frac_denom = 0
    traj_combined_frac_sum = 0.0
    traj_combined_frac_denom = 0


    for item in results:
        task_counts[str(item.get("task_id"))]["total"] += 1
        task_counts[str(item.get("task_id"))]["plan_ok"] += int(plan_pass(item))
        task_counts[str(item.get("task_id"))]["plan_quality_ok"] += int(plan_quality_pass(item))
        task_counts[str(item.get("task_id"))]["plan_joint_ok"] += int(plan_joint_pass(item))
        task_counts[str(item.get("task_id"))]["trajectory_ok"] += int(trajectory_pass(item))
        task_counts[str(item.get("task_id"))]["instantiation_ok"] += int(instantiation_pass(item))
        task_counts[str(item.get("task_id"))]["tool_input_case_ok"] += int(instantiation_pass(item))
        task_counts[str(item.get("task_id"))]["update_ok"] += int(update_pass(item))
        task_counts[str(item.get("task_id"))]["update_quality_ok"] += int(update_quality_pass(item))
        task_counts[str(item.get("task_id"))]["update_plan_quality_ok"] += int(update_plan_quality_pass(item))
        task_counts[str(item.get("task_id"))]["update_preference_ok"] += int(update_preference_pass(item))
        plan_parsed = ((item.get("plan_judge") or {}).get("parsed") or {})
        checklist = plan_parsed.get("checklist") or {}
        for key in PLAN_CHECKLIST_KEYS:
            if isinstance(checklist.get(key), dict) and checklist[key].get("pass"):
                checklist_counts[key] += 1
        plan_quality_checklist = plan_parsed.get("plan_quality_checklist") or {}
        for key in PLAN_QUALITY_CHECKLIST_KEYS:
            if isinstance(plan_quality_checklist.get(key), dict) and plan_quality_checklist[key].get("pass"):
                plan_quality_checklist_counts[key] += 1
        inst_parsed = instantiation_parsed(item)
        tool_items = inst_parsed.get("instantiation_judgements")
        if not isinstance(tool_items, list):
            tool_items = inst_parsed.get("tool_call_judgements") or []
        if not tool_items:
            zero_tool_cases += 1
        tool_total_i = 0
        tool_ok_i = 0
        for tool in tool_items:
            if not isinstance(tool, dict):
                continue
            tool_total += 1
            tool_total_i += 1
            instantiation_score_sum += float(tool.get("instantiation_score") or 0.0)
            if int(tool.get("NoRelation") or 0) == 1:
                no_relation_tool_total += 1
            else:
                judged_relation_tool_total += 1
            api_name = str(tool.get("api_name") or "unknown")
            if tool.get("json_object_preference_following", tool.get("input_preference_following")):
                tool_ok += 1
                tool_ok_i += 1
                api_counts[api_name]["ok"] += 1
            else:
                api_counts[api_name]["fail"] += 1
        update_items = (((item.get("update_judge") or {}).get("parsed") or {}).get("update_plan_judgements") or [])
        if not update_items:
            zero_update_cases += 1
        update_total_i = 0
        update_ok_i = 0
        for update in update_items:
            update_total += 1
            update_total_i += 1
            if update.get("update_correct"):
                update_item_ok += 1
                update_ok_i += 1
            if update.get("update_quality_correct"):
                update_quality_item_ok += 1
            if update.get("update_plan_quality_correct"):
                update_plan_quality_item_ok += 1
            if update.get("update_preference_correct"):
                update_preference_item_ok += 1
            update_plan_quality_atomic_ok += sum(
                int(update.get(key) is True) for key in UPDATE_PLAN_QUALITY_KEYS
            )
            for key in UPDATE_PLAN_QUALITY_KEYS:
                update_plan_quality_check_counts[key] += int(update.get(key) is True)

        # 轨迹长度归一化：每条轨迹内部的通过比例（无对应步骤时记为 1.0，与"无违规"口径一致）。
        if update_total_i:
            traj_update_frac_sum += update_ok_i / update_total_i
            traj_update_frac_denom += 1
        if tool_total_i:
            traj_inst_frac_sum += tool_ok_i / tool_total_i
            traj_inst_frac_denom += 1
        combined_total_i = tool_total_i + update_total_i
        if combined_total_i:
            traj_combined_frac_sum += (tool_ok_i + update_ok_i) / combined_total_i
            traj_combined_frac_denom += 1


    aggregate = {
        "total_cases": total,
        "first_plan_preference_following_count": first_plan_ok,
        "first_plan_preference_following_ratio": pct(first_plan_ok, total),
        "first_plan_quality_correct_count": first_plan_quality_ok,
        "first_plan_quality_correct_ratio": pct(first_plan_quality_ok, total),
        "first_plan_joint_correct_count": first_plan_joint_ok,
        "first_plan_joint_correct_ratio": pct(first_plan_joint_ok, total),
        "first_plan_quality_score": round(
            sum(plan_quality_checklist_counts.values()) / (total * len(PLAN_QUALITY_CHECKLIST_KEYS)), 6
        ) if total else 0.0,
        "first_plan_quality_score_percent": pct(
            sum(plan_quality_checklist_counts.values()), total * len(PLAN_QUALITY_CHECKLIST_KEYS)
        ),
        "instantiation_case_correct_count": instantiation_ok,
        "instantiation_case_correct_ratio": pct(instantiation_ok, total),
        "instantiation_step_count": tool_total,
        "instantiation_step_preference_following_count": tool_ok,
        "instantiation_step_preference_following_ratio": pct(tool_ok, tool_total),
        "instantiation_score_sum": round(instantiation_score_sum, 6),
        "instantiation_acc": round(instantiation_score_sum / tool_total, 6) if tool_total else 1.0,
        "instantiation_acc_percent": round((instantiation_score_sum / tool_total) * 100.0, 2) if tool_total else 100.0,
        "instantiation_no_relation_step_count": no_relation_tool_total,
        "instantiation_llm_judged_step_count": judged_relation_tool_total,
        "trajectory_preference_following_count": trajectory_ok,
        "trajectory_preference_following_ratio": pct(trajectory_ok, total),
        # 审稿意见(1) 新增：按轨迹长度归一化的 macro 平均指标（每条轨迹等权）。
        # 与 *_step_* 的 micro 口径互补：micro 把所有步骤汇总（长轨迹权重大），
        # macro 先在轨迹内归一再跨轨迹平均（长短轨迹等权），缓解严格全有或全无指标对长轨迹的惩罚。
        "trajectory_update_lengthnorm_ratio": pct_float(
            traj_update_frac_sum, traj_update_frac_denom
        ),
        "trajectory_update_lengthnorm_denom": traj_update_frac_denom,
        "trajectory_instantiation_lengthnorm_ratio": pct_float(
            traj_inst_frac_sum, traj_inst_frac_denom
        ),
        "trajectory_instantiation_lengthnorm_denom": traj_inst_frac_denom,
        "trajectory_combined_lengthnorm_ratio": pct_float(
            traj_combined_frac_sum, traj_combined_frac_denom
        ),
        "trajectory_combined_lengthnorm_denom": traj_combined_frac_denom,

        "tool_call_input_case_following_count": instantiation_ok,
        "tool_call_input_case_following_ratio": pct(instantiation_ok, total),
        "update_plan_correct_count": update_ok,
        "update_plan_correct_ratio": pct(update_ok, total),
        "update_joint_correct_count": update_ok,
        "update_joint_correct_ratio": pct(update_ok, total),
        "update_quality_correct_count": update_quality_ok,
        "update_quality_correct_ratio": pct(update_quality_ok, total),
        "update_plan_quality_correct_count": update_plan_quality_ok,
        "update_plan_quality_correct_ratio": pct(update_plan_quality_ok, total),
        "update_preference_correct_count": update_preference_ok,
        "update_preference_correct_ratio": pct(update_preference_ok, total),
        "tool_call_input_count": tool_total,
        "tool_call_input_following_count": tool_ok,
        "tool_call_input_following_ratio": pct(tool_ok, tool_total),
        "tool_call_input_score_avg": round(instantiation_score_sum / tool_total, 6) if tool_total else 1.0,
        "update_plan_step_count": update_total,
        "update_plan_step_correct_count": update_item_ok,
        "update_plan_step_correct_ratio": pct(update_item_ok, update_total),
        "update_quality_step_correct_count": update_quality_item_ok,
        "update_quality_step_correct_ratio": pct(update_quality_item_ok, update_total),
        "update_plan_quality_step_correct_count": update_plan_quality_item_ok,
        "update_plan_quality_step_correct_ratio": pct(update_plan_quality_item_ok, update_total),
        "update_plan_quality_score": round(
            update_plan_quality_atomic_ok / (update_total * len(UPDATE_PLAN_QUALITY_KEYS)), 6
        ) if update_total else 1.0,
        "update_plan_quality_score_percent": pct(
            update_plan_quality_atomic_ok, update_total * len(UPDATE_PLAN_QUALITY_KEYS)
        ) if update_total else 100.0,
        "update_plan_quality_check_counts": update_plan_quality_check_counts,
        "update_plan_quality_check_ratios": {
            key: pct(value, update_total) if update_total else 100.0
            for key, value in update_plan_quality_check_counts.items()
        },
        "update_preference_step_correct_count": update_preference_item_ok,
        "update_preference_step_correct_ratio": pct(update_preference_item_ok, update_total),
        "zero_tool_call_case_count": zero_tool_cases,
        "zero_update_plan_case_count": zero_update_cases,
        "judge_error_case_count": sum(1 for item in results if item.get("judge_errors")),
        "status_counts": dict(status_counts),
        "checklist_counts": checklist_counts,
        "checklist_ratios": {key: pct(value, total) for key, value in checklist_counts.items()},
        "plan_quality_checklist_counts": plan_quality_checklist_counts,
        "plan_quality_checklist_ratios": {
            key: pct(value, total) for key, value in plan_quality_checklist_counts.items()
        },
        "task_counts": {
            task_id: {
                "total": counter["total"],
                "plan_ok": counter["plan_ok"],
                "plan_ratio": pct(counter["plan_ok"], counter["total"]),
                "plan_quality_ok": counter["plan_quality_ok"],
                "plan_quality_ratio": pct(counter["plan_quality_ok"], counter["total"]),
                "plan_joint_ok": counter["plan_joint_ok"],
                "plan_joint_ratio": pct(counter["plan_joint_ok"], counter["total"]),
                "trajectory_ok": counter["trajectory_ok"],
                "trajectory_ratio": pct(counter["trajectory_ok"], counter["total"]),
                "instantiation_ok": counter["instantiation_ok"],
                "instantiation_ratio": pct(counter["instantiation_ok"], counter["total"]),
                "tool_input_case_ok": counter["tool_input_case_ok"],
                "tool_input_case_ratio": pct(counter["tool_input_case_ok"], counter["total"]),
                "update_ok": counter["update_ok"],
                "update_ratio": pct(counter["update_ok"], counter["total"]),
                "update_quality_ok": counter["update_quality_ok"],
                "update_quality_ratio": pct(counter["update_quality_ok"], counter["total"]),
                "update_plan_quality_ok": counter["update_plan_quality_ok"],
                "update_plan_quality_ratio": pct(counter["update_plan_quality_ok"], counter["total"]),
                "update_preference_ok": counter["update_preference_ok"],
                "update_preference_ratio": pct(counter["update_preference_ok"], counter["total"]),
            }
            for task_id, counter in sorted(task_counts.items())
        },
        "api_counts": {
            api_name: {
                "total": counter["ok"] + counter["fail"],
                "ok": counter["ok"],
                "fail": counter["fail"],
                "ok_ratio": pct(counter["ok"], counter["ok"] + counter["fail"]),
            }
            for api_name, counter in sorted(api_counts.items())
        },
    }
    if any(int(item.get("pass_n") or PASS_N) > 1 for item in results):
        aggregate["pass_k_summary"] = pass_k_summary(results)
    return aggregate


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def badge(ok: bool, true_text: str = "通过", false_text: str = "失败") -> str:
    cls = "ok" if ok else "fail"
    text = true_text if ok else false_text
    return f'<span class="badge {cls}">{esc(text)}</span>'


def render_metric_card(title: str, value: Any, sub: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-title">{esc(title)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-sub">{esc(sub)}</div>'
        "</div>"
    )


def render_counter_table(title: str, rows: list[tuple[Any, Any]]) -> str:
    body = "\n".join(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>" for key, value in rows)
    return f"""
    <section>
      <h2>{esc(title)}</h2>
      <table>
        <thead><tr><th>项目</th><th>数量/比例</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </section>
    """


def first_issue(result: dict[str, Any]) -> str:
    plan = ((result.get("plan_judge") or {}).get("parsed") or {})
    checklist = plan.get("checklist") or {}
    issues: list[str] = []
    for key in PLAN_CHECKLIST_KEYS:
        item = checklist.get(key)
        if isinstance(item, dict) and not item.get("pass"):
            issues.extend(str(issue) for issue in item.get("issues", [])[:2])
    quality_checklist = plan.get("plan_quality_checklist") or {}
    for key in PLAN_QUALITY_CHECKLIST_KEYS:
        item = quality_checklist.get(key)
        if isinstance(item, dict) and not item.get("pass"):
            issues.extend(str(issue) for issue in item.get("issues", [])[:2])
    trajectory = instantiation_parsed(result)
    tool_items = trajectory.get("instantiation_judgements")
    if not isinstance(tool_items, list):
        tool_items = trajectory.get("tool_call_judgements") if isinstance(trajectory.get("tool_call_judgements"), list) else []
    for tool in tool_items:
        if isinstance(tool, dict) and not tool.get("json_object_preference_following", tool.get("input_preference_following")):
            issues.extend(str(issue) for issue in tool.get("violated_preferences", [])[:2])
    update = ((result.get("update_judge") or {}).get("parsed") or {})
    for update_item in update.get("update_plan_judgements", []) if isinstance(update.get("update_plan_judgements"), list) else []:
        if isinstance(update_item, dict) and not update_item.get("update_correct"):
            issues.extend(str(issue) for issue in update_item.get("issues", [])[:2])
            issues.extend(str(issue) for issue in update_item.get("plan_quality_issues", [])[:2])
            issues.extend(str(issue) for issue in update_item.get("preference_issues", [])[:2])
            issues.extend(str(issue) for issue in update_item.get("violated_preferences", [])[:2])
    if result.get("judge_errors"):
        issues.append("judge 执行存在错误，详见 JSONL 明细。")
    return compact_text("; ".join(issues), 500)


def render_case_details(result: dict[str, Any]) -> str:
    plan = ((result.get("plan_judge") or {}).get("parsed") or {})
    checklist = plan.get("checklist") or {}
    quality_checklist = plan.get("plan_quality_checklist") or {}
    trajectory = instantiation_parsed(result)
    update = ((result.get("update_judge") or {}).get("parsed") or {})
    checklist_rows = []
    for key in PLAN_CHECKLIST_KEYS:
        item = checklist.get(key) if isinstance(checklist.get(key), dict) else {}
        checklist_rows.append(
            "<tr>"
            f"<td>{esc(PLAN_CHECKLIST_LABELS[key])}</td>"
            f"<td>{badge(bool(item.get('pass')))}</td>"
            f"<td>{esc(compact_text(item.get('evidence', []), 500))}</td>"
            f"<td>{esc(compact_text(item.get('issues', []), 500))}</td>"
            "</tr>"
        )
    quality_checklist_rows = []
    for key in PLAN_QUALITY_CHECKLIST_KEYS:
        item = quality_checklist.get(key) if isinstance(quality_checklist.get(key), dict) else {}
        quality_checklist_rows.append(
            "<tr>"
            f"<td>{esc(PLAN_QUALITY_CHECKLIST_LABELS[key])}</td>"
            f"<td>{badge(bool(item.get('pass')))}</td>"
            f"<td>{esc(compact_text(item.get('evidence', []), 500))}</td>"
            f"<td>{esc(compact_text(item.get('issues', []), 500))}</td>"
            "</tr>"
        )
    tool_rows = []
    tool_items = trajectory.get("instantiation_judgements")
    if not isinstance(tool_items, list):
        tool_items = trajectory.get("tool_call_judgements") if isinstance(trajectory.get("tool_call_judgements"), list) else []
    for tool in tool_items:
        if not isinstance(tool, dict):
            continue
        tool_rows.append(
            "<tr>"
            f"<td>{esc(tool.get('call_index'))}</td>"
            f"<td>{esc(tool.get('iteration'))}</td>"
            f"<td>{esc(tool.get('api_name'))}</td>"
            f"<td>{esc(tool.get('NoRelation'))}</td>"
            f"<td>{badge(bool(tool.get('json_object_preference_following', tool.get('input_preference_following'))))}</td>"
            f"<td>{esc(tool.get('instantiation_score'))}</td>"
            f"<td>{esc(compact_text(tool.get('supporting_evidence', []), 500))}</td>"
            f"<td>{esc(compact_text(tool.get('violated_preferences', []), 500))}</td>"
            "</tr>"
        )
    if not tool_rows:
        tool_rows.append('<tr><td colspan="8">无工具调用输入。</td></tr>')

    update_rows = []
    update_items = update.get("update_plan_judgements") if isinstance(update.get("update_plan_judgements"), list) else []
    for update_item in update_items:
        if not isinstance(update_item, dict):
            continue
        update_rows.append(
            "<tr>"
            f"<td>{esc(update_item.get('update_index'))}</td>"
            f"<td>{esc(update_item.get('iteration'))}</td>"
            f"<td>{badge(bool(update_item.get('update_joint_correct', update_item.get('update_correct'))))}</td>"
            f"<td>{badge(bool(update_item.get('update_quality_correct')))}</td>"
            f"<td>{badge(bool(update_item.get('update_plan_quality_correct')))}</td>"
            f"<td>{badge(bool(update_item.get('update_preference_correct')))}</td>"
            f"<td>{esc(compact_text(update_item.get('supporting_evidence', []), 500))}</td>"
            f"<td>{esc(compact_text(update_item.get('quality_issues', update_item.get('issues', [])), 500))}</td>"
            f"<td>{esc(compact_text(update_item.get('plan_quality_issues', []), 500))}</td>"
            f"<td>{esc(compact_text(update_item.get('preference_issues', []), 500))}</td>"
            f"<td>{esc(compact_text(update_item.get('violated_preferences', []), 500))}</td>"
            "</tr>"
        )
    if not update_rows:
        update_rows.append('<tr><td colspan="11">无 update_plan 事件。</td></tr>')

    return f"""
    <details>
      <summary>查看 checklist、Instantiation 与 update_plan 判定</summary>
      <h4>First Plan Checklist</h4>
      <table class="nested">
        <thead><tr><th>指标</th><th>判定</th><th>证据</th><th>问题</th></tr></thead>
        <tbody>{''.join(checklist_rows)}</tbody>
      </table>
      <h4>First Plan Quality Checklist</h4>
      <table class="nested">
        <thead><tr><th>质量问题</th><th>判定</th><th>证据</th><th>问题</th></tr></thead>
        <tbody>{''.join(quality_checklist_rows)}</tbody>
      </table>
      <h4>Instantiation 判定</h4>
      <table class="nested">
        <thead><tr><th>#</th><th>轮次</th><th>API</th><th>NoRelation</th><th>偏好判定</th><th>分数</th><th>证据</th><th>违反偏好</th></tr></thead>
        <tbody>{''.join(tool_rows)}</tbody>
      </table>
      <h4>Update Plan 正确性判定</h4>
      <table class="nested">
        <thead><tr><th>#</th><th>轮次</th><th>Joint</th><th>状态转移</th><th>Plan Quality</th><th>偏好</th><th>证据</th><th>状态问题</th><th>计划问题</th><th>偏好问题</th><th>违反偏好</th></tr></thead>
        <tbody>{''.join(update_rows)}</tbody>
      </table>
    </details>
    """


def render_html_report(
    *,
    run_id: str,
    target_gen_model: str,
    preference_mode: str,
    pass_n: int,
    selected_eval_run_ids: list[str],
    judge_model: dict[str, Any],
    eval_result_dir: Path,
    detail_path: Path,
    summary_path: Path,
    results: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> str:
    status_rows = sorted(aggregate["status_counts"].items())
    checklist_rows = [
        (
            PLAN_CHECKLIST_LABELS[key],
            f"{aggregate['checklist_counts'][key]}/{aggregate['total_cases']} ({aggregate['checklist_ratios'][key]}%)",
        )
        for key in PLAN_CHECKLIST_KEYS
    ]
    plan_quality_checklist_rows = [
        (
            PLAN_QUALITY_CHECKLIST_LABELS[key],
            f"{aggregate['plan_quality_checklist_counts'][key]}/{aggregate['total_cases']} "
            f"({aggregate['plan_quality_checklist_ratios'][key]}%)",
        )
        for key in PLAN_QUALITY_CHECKLIST_KEYS
    ]
    update_plan_quality_check_rows = [
        (
            UPDATE_PLAN_QUALITY_LABELS[key],
            f"{aggregate['update_plan_quality_check_counts'][key]}/{aggregate['update_plan_step_count']} "
            f"({aggregate['update_plan_quality_check_ratios'][key]}%)",
        )
        for key in UPDATE_PLAN_QUALITY_KEYS
    ]
    task_rows = [
        (
            task_id,
            f"first-pref {item['plan_ok']}/{item['total']} ({item['plan_ratio']}%), "
            f"first-quality {item['plan_quality_ok']}/{item['total']} ({item['plan_quality_ratio']}%), "
            f"first-joint {item['plan_joint_ok']}/{item['total']} ({item['plan_joint_ratio']}%), "
            f"instantiation {item['instantiation_ok']}/{item['total']} ({item['instantiation_ratio']}%), "
            f"trajectory {item['trajectory_ok']}/{item['total']} ({item['trajectory_ratio']}%), "
            f"update-plan-quality {item['update_plan_quality_ok']}/{item['total']} "
            f"({item['update_plan_quality_ratio']}%), "
            f"update-joint {item['update_ok']}/{item['total']} ({item['update_ratio']}%)",
        )
        for task_id, item in aggregate["task_counts"].items()
    ]
    api_rows = [
        (
            api_name,
            f"{item['ok']}/{item['total']} ({item['ok_ratio']}%), fail={item['fail']}",
        )
        for api_name, item in sorted(
            aggregate["api_counts"].items(),
            key=lambda kv: (-kv[1]["fail"], kv[0]),
        )[:80]
    ]

    case_rows: list[str] = []
    for result in results[:MAX_HTML_CASES]:
        checklist = (((result.get("plan_judge") or {}).get("parsed") or {}).get("checklist") or {})
        checklist_badges = " ".join(
            f'<span title="{esc(PLAN_CHECKLIST_LABELS[key])}">{badge(bool((checklist.get(key) or {}).get("pass")), key[-2:], key[-2:])}</span>'
            for key in PLAN_CHECKLIST_KEYS
        )
        case_rows.append(
            "<tr>"
            f"<td>{esc(result.get('user_id'))}</td>"
            f"<td>{esc(result.get('task_id'))}</td>"
            f"<td>{esc(result.get('pass_index'))}/{esc(result.get('pass_n'))}</td>"
            f"<td>{esc(result.get('eval_status'))}</td>"
            f"<td>{badge(plan_pass(result))}</td>"
            f"<td>{badge(plan_quality_pass(result))}</td>"
            f"<td>{badge(plan_joint_pass(result))}</td>"
            f"<td>{checklist_badges}</td>"
            f"<td>{badge(trajectory_pass(result))}</td>"
            f"<td>{badge(update_pass(result))}</td>"
            f"<td>{esc(result.get('tool_call_count'))}</td>"
            f"<td>{esc(first_issue(result))}{render_case_details(result)}</td>"
            "</tr>"
        )

    generated_at = utc_now_iso()
    pass_k_cards = ""
    pass_k = aggregate.get("pass_k_summary") if isinstance(aggregate.get("pass_k_summary"), dict) else None
    if pass_k:
        k_value = pass_k.get("pass_n", pass_n)
        denominator = pass_k.get("denominator", 0)
        metrics = pass_k.get("metrics", {})
        fp = metrics.get("first_plan", {})
        fp_quality = metrics.get("first_plan_quality", {})
        fp_joint = metrics.get("first_plan_joint", {})
        inst = metrics.get("instantiation", {})
        traj = metrics.get("trajectory", {})
        upd = metrics.get("update_plan", {})
        upd_quality = metrics.get("update_plan_quality", {})
        pass_k_cards = f"""
      {render_metric_card(f"First Preference Pass@{k_value}", f"{fp.get('pass_at_k_ratio', 0)}%", f"{fp.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"First Plan Quality Pass@{k_value}", f"{fp_quality.get('pass_at_k_ratio', 0)}%", f"{fp_quality.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"First Joint Pass@{k_value}", f"{fp_joint.get('pass_at_k_ratio', 0)}%", f"{fp_joint.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"Instantiation Pass@{k_value}", f"{inst.get('pass_at_k_ratio', 0)}%", f"{inst.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"Trajectory Pass@{k_value}", f"{traj.get('pass_at_k_ratio', 0)}%", f"{traj.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"Update Plan Quality Pass@{k_value}", f"{upd_quality.get('pass_at_k_ratio', 0)}%", f"{upd_quality.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
      {render_metric_card(f"Update Joint Pass@{k_value}", f"{upd.get('pass_at_k_ratio', 0)}%", f"{upd.get('pass_at_k_count', 0)}/{denominator} 个子任务至少一次成功")}
        """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PPEval Judge Report {esc(run_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d8dee8;
      --ok: #166534;
      --ok-bg: #dcfce7;
      --fail: #991b1b;
      --fail-bg: #fee2e2;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 28px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    main {{ padding: 24px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    h3 {{ margin: 18px 0 8px; font-size: 16px; }}
    h4 {{ margin: 14px 0 8px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-title {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .metric-sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
      overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f5f9; font-weight: 650; white-space: nowrap; }}
    tr:hover td {{ background: #fafbff; }}
    .nested {{ margin-top: 8px; font-size: 13px; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
      margin: 1px;
    }}
    .badge.ok {{ color: var(--ok); background: var(--ok-bg); }}
    .badge.fail {{ color: var(--fail); background: var(--fail-bg); }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); }}
    code {{ background: #eef2f7; padding: 1px 4px; border-radius: 4px; }}
    .note {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>PPEval LLM-as-Judge 统计报告</h1>
    <div class="meta">
      target_GEN_LLM={esc(target_gen_model)} ｜ judge_run_id={esc(run_id)} ｜ 生成时间={esc(generated_at)} ｜ judge_model={esc(judge_model.get("name"))} ｜
      base_url={esc(judge_model.get("base_url"))}
    </div>
    <div class="meta">
      偏好注入方式: <code>{esc(preference_mode)}</code> ｜ {esc(PREFERENCE_INJECTION_LABELS.get(preference_mode, preference_mode))}
    </div>
    <div class="meta">
      PASS_N: <code>{esc(pass_n)}</code> ｜ 每条 eval 轨迹均单独进行 LLM-as-Judge
    </div>
    <div class="meta">
      Eval run_id: <code>{esc(", ".join(selected_eval_run_ids))}</code>
    </div>
    <div class="meta">
      Eval 结果目录: <code>{esc(eval_result_dir)}</code> ｜ 明细 JSONL: <code>{esc(detail_path)}</code> ｜ 汇总 JSON: <code>{esc(summary_path)}</code>
    </div>
  </header>
  <main>
    <div class="metrics">
      {render_metric_card("Case 总数", aggregate["total_cases"], f"来自 {eval_result_dir}/user_*.jsonl")}
      {render_metric_card("First Preference Acc", f"{aggregate['first_plan_preference_following_ratio']}%", f"{aggregate['first_plan_preference_following_count']}/{aggregate['total_cases']}")}
      {render_metric_card("First Plan Quality Acc", f"{aggregate['first_plan_quality_correct_ratio']}%", f"{aggregate['first_plan_quality_correct_count']}/{aggregate['total_cases']}，6 项全部通过")}
      {render_metric_card("First Joint Acc", f"{aggregate['first_plan_joint_correct_ratio']}%", f"{aggregate['first_plan_joint_correct_count']}/{aggregate['total_cases']}，偏好与质量均通过")}
      {render_metric_card("First Plan Quality Score", f"{aggregate['first_plan_quality_score_percent']}%", "6 个质量小问题的微平均")}
      {render_metric_card("Instantiation Acc", f"{aggregate['instantiation_acc_percent']}%", f"步骤均分={aggregate['instantiation_score_sum']}/{aggregate['instantiation_step_count']}，NoRelation={aggregate['instantiation_no_relation_step_count']}")}
      {render_metric_card("Instantiation Case 通过比例", f"{aggregate['instantiation_case_correct_ratio']}%", f"{aggregate['instantiation_case_correct_count']}/{aggregate['total_cases']}")}
      {render_metric_card("轨迹正确比例", f"{aggregate['trajectory_preference_following_ratio']}%", f"{aggregate['trajectory_preference_following_count']}/{aggregate['total_cases']}，Instantiation 与 update_plan 均通过")}
      {render_metric_card("轨迹长度归一 Update（macro）", f"{aggregate['trajectory_update_lengthnorm_ratio']}%", f"每条轨迹内 update 通过比例的均值，n={aggregate['trajectory_update_lengthnorm_denom']} 条含 update 的轨迹")}
      {render_metric_card("轨迹长度归一 Instantiation（macro）", f"{aggregate['trajectory_instantiation_lengthnorm_ratio']}%", f"每条轨迹内工具调用通过比例的均值，n={aggregate['trajectory_instantiation_lengthnorm_denom']} 条含工具调用的轨迹")}
      {render_metric_card("轨迹长度归一 综合（macro）", f"{aggregate['trajectory_combined_lengthnorm_ratio']}%", f"工具+update 合并后每条轨迹通过比例的均值，n={aggregate['trajectory_combined_lengthnorm_denom']} 条轨迹")}
      {render_metric_card("恢复准确率 Replan", f"{aggregate.get('recovery_replan_accuracy_ratio', 0)}%", f"失效观测后正确吸收并重规划：{aggregate.get('recovery_replan_ok_count', 0)}/{aggregate.get('recovery_replan_case_count', 0)} 个失效观测")}
      {render_metric_card("恢复准确率 Execution", f"{aggregate.get('recovery_execution_accuracy_ratio', 0)}%", f"重规划正确且基于新计划再执行通过：{aggregate.get('recovery_execution_ok_count', 0)}/{aggregate.get('recovery_execution_case_count', 0)} 个可观测再执行的失效观测")}
      {render_metric_card("偏好相关工具 LLM Judge 数", aggregate["instantiation_llm_judged_step_count"], "NoRelation API 不进入 LLM-as-judge")}
      {render_metric_card("Instantiation Step 偏好通过比例", f"{aggregate['instantiation_step_preference_following_ratio']}%", f"{aggregate['instantiation_step_preference_following_count']}/{aggregate['instantiation_step_count']}")}
      {render_metric_card("Update Joint Acc", f"{aggregate['update_joint_correct_ratio']}%", f"{aggregate['update_joint_correct_count']}/{aggregate['total_cases']}，状态、计划质量与偏好均通过")}
      {render_metric_card("Update State Acc", f"{aggregate['update_quality_correct_ratio']}%", f"{aggregate['update_quality_correct_count']}/{aggregate['total_cases']}，状态转移 4 项全部通过")}
      {render_metric_card("Update Plan Quality Acc", f"{aggregate['update_plan_quality_correct_ratio']}%", f"{aggregate['update_plan_quality_correct_count']}/{aggregate['total_cases']}，每轮 6 项全部通过")}
      {render_metric_card("Update Plan Quality Score", f"{aggregate['update_plan_quality_score_percent']}%", "全部 update 的 6 个质量小问题微平均")}
      {render_metric_card("Update Preference Acc", f"{aggregate['update_preference_correct_ratio']}%", f"{aggregate['update_preference_correct_count']}/{aggregate['total_cases']}，附录诊断")}
      {render_metric_card("Update 步骤正确比例", f"{aggregate['update_plan_step_correct_ratio']}%", f"{aggregate['update_plan_step_correct_count']}/{aggregate['update_plan_step_count']}")}
      {render_metric_card("Update 质量步骤通过比例", f"{aggregate['update_quality_step_correct_ratio']}%", f"{aggregate['update_quality_step_correct_count']}/{aggregate['update_plan_step_count']}")}
      {render_metric_card("Update Plan Quality 步骤通过比例", f"{aggregate['update_plan_quality_step_correct_ratio']}%", f"{aggregate['update_plan_quality_step_correct_count']}/{aggregate['update_plan_step_count']}")}
      {render_metric_card("Update 偏好步骤通过比例", f"{aggregate['update_preference_step_correct_ratio']}%", f"{aggregate['update_preference_step_correct_count']}/{aggregate['update_plan_step_count']}")}
      {render_metric_card("零工具调用 Case", aggregate["zero_tool_call_case_count"], "Instantiation 部分按无违规计入；轨迹仍受 update_plan 影响")}
      {render_metric_card("零 Update Case", aggregate["zero_update_plan_case_count"], "没有 update_plan 事件时按无更新错误计入")}
      {render_metric_card("Judge 错误 Case", aggregate["judge_error_case_count"], "错误 case 在对应指标中按失败保守计入")}
      {pass_k_cards}
    </div>

    <section>
      <h2>评估方法</h2>
      <p>
        First plan 在同一次 judge 请求中独立回答 2 个偏好小问题和 6 个 Plan Quality 小问题。
        First Preference Acc 与 First Plan Quality Acc 分开统计；First Joint Acc 仅在两层都全部通过时为 1。
        Plan Quality 覆盖目标与成功标准、可执行性、依赖顺序、工具与信息获取、约束可行性和全局完成路径。
      </p>
      <p>
        Instantiation Acc 评估每一次工具调用时基于 schema 生成 JSON object 的偏好遵循程度：
        1.0 表示首次 JSON 合法且偏好正确；0.5 表示首次 JSON 失败但一次 repair 后合法且偏好正确；
        0 表示 repair 后偏好错误，或两次 JSON 生成均失败。NoRelation=1 的 API 跳过 LLM-as-judge，
        只按结构化 JSON 合法性给分。
      </p>
      <p>
        轨迹正确性是复合指标：Instantiation 与 update_plan 均需通过。每一次 update_plan 都必须准确吸收 sandbox observation。
        任一偏好相关工具实例化或 update_plan 失败，该 case 的轨迹整体按失败计入。
      </p>
      <p>
        Update plan 在同一次 judge 请求中回答 4 个状态转移、6 个更新后 Plan Quality 和 4 个偏好小问题。
        Update Joint Acc 要求一条轨迹中的每一次 update 都同时通过三层；所有派生总判定由程序逻辑与计算，
        不额外调用 judge API，也不信任模型自行汇总。
      </p>
    </section>

    {render_counter_table("Eval 状态分布", status_rows)}
    {render_counter_table("First Plan 偏好指标通过率", checklist_rows)}
    {render_counter_table("First Plan Quality 小问题通过率", plan_quality_checklist_rows)}
    {render_counter_table("Update Plan Quality 小问题通过率", update_plan_quality_check_rows)}
    {render_counter_table("按 task_id 汇总", task_rows)}
    {render_counter_table("按 API 汇总 Instantiation 通过率（失败优先）", api_rows)}

    <section>
      <h2>Case 明细</h2>
      <p class="note">表格保留每个 case 的 checklist、Instantiation 判定摘要；完整 prompt、raw response、usage 保存在 JSONL 明细。</p>
      <table>
        <thead>
          <tr>
            <th>User</th><th>Task</th><th>Pass</th><th>Eval 状态</th><th>First Preference</th><th>First Quality</th><th>First Joint</th><th>偏好小问题</th>
            <th>Traj LLM</th><th>Update LLM</th><th>工具数</th><th>主要问题与详情</th>
          </tr>
        </thead>
        <tbody>{''.join(case_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    global API_DEFINITIONS
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)
    preference_mode = normalize_preference_injection_mode(PREFERENCE_INJECTION_MODE)
    pass_n = normalize_pass_n(PASS_N)
    API_DEFINITIONS = load_api_definitions()
    no_relation_api_count = sum(1 for api_def in API_DEFINITIONS.values() if int(api_def.get("NoRelation") or 0) == 1)

    api_config = load_api_config()
    target_gen_model = TARGET_GEN_LLM or getattr(api_config, "GEN_MODEL")
    eval_result_dir, judge_result_dir, html_dir = configure_model_dirs(target_gen_model, preference_mode, pass_n)
    judge_result_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    client = (
        JUDGE_CLIENT_FACTORY(api_config, logger)
        if JUDGE_CLIENT_FACTORY is not None
        else LLMClient(
            api_key=getattr(api_config, "JUDGE_KEY"),
            base_url=getattr(api_config, "JUDGE_URL"),
            model=getattr(api_config, "JUDGE_MODEL"),
            logger=logger,
        )
    )
    logger.info(
        "开始 judge：run_id=%s model=%s base_url=%s key=%s",
        run_id,
        client.model,
        client.base_url,
        mask_secret(client.api_key),
    )
    logger.info(
        "核心配置：TARGET_GEN_LLM=%s PREFERENCE_INJECTION_MODE=%s TASK_DIFFICULTY=%s PASS_N=%s TARGET_EVAL_RUN_ID=%s LATEST_ONLY=%s USER_ID_FILTER=%s TASK_ID_FILTER=%s ACTIVE_TASK_FILTER=%s EVAL_RESULT_DIR=%s JUDGE_RESULT_DIR=%s ENABLE_LONG_CONTEXT=%s LONG_CONTEXT_TURNS=%s MAX_TOOL_CALLS_PER_PROMPT=%s MAX_UPDATE_EVENTS_PER_PROMPT=%s NoRelation_API=%s/%s",
        target_gen_model,
        preference_mode,
        normalize_task_difficulty(TASK_DIFFICULTY),
        pass_n,
        TARGET_EVAL_RUN_ID or "<全部>",
        JUDGE_LATEST_EVAL_RUN_ONLY,
        USER_ID_FILTER,
        TASK_ID_FILTER,
        sorted(task_ids_for_active_filters()),
        eval_result_dir,
        judge_result_dir,
        ENABLE_LONG_CONTEXT,
        LONG_CONTEXT_TURNS,
        MAX_TOOL_CALLS_PER_PROMPT,
        MAX_UPDATE_EVENTS_PER_PROMPT,
        no_relation_api_count,
        len(API_DEFINITIONS),
    )

    records = discover_eval_records(logger)
    if not records:
        raise RuntimeError(f"没有发现可 judge 的 eval 记录: {EVAL_RESULT_DIR / EVAL_RESULT_GLOB}")
    selected_eval_run_ids = sorted({str(record.get("run_id") or "") for record in records})
    selected_pass_indexes = sorted({int(record.get("pass_index") or 1) for record in records})

    detail_path = JUDGE_RESULT_DIR / "judge_details.jsonl"
    summary_path = JUDGE_RESULT_DIR / "judge_summary.json"
    html_file_name = "BANK评测报告.html" if preference_mode == "bank" else "评测报告.html"
    html_path = HTML_DIR / html_file_name
    judge_concurrency = calibrate_concurrency(client, logger)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=judge_concurrency, thread_name_prefix="judge") as pool:
        futures = [pool.submit(judge_case, copy.deepcopy(record), client, logger) for record in records]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_jsonl(detail_path, result)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(records):
                logger.info(
                    "judge 进度：%s/%s latest=%s first_pref=%s first_quality=%s first_joint=%s "
                    "trajectory=%s update_joint=%s",
                    completed,
                    len(records),
                    result.get("case_id"),
                    plan_pass(result),
                    plan_quality_pass(result),
                    plan_joint_pass(result),
                    trajectory_pass(result),
                    update_pass(result),
                )

    results.sort(
        key=lambda item: (
            str(item.get("user_id")),
            str(item.get("task_id")),
            int(item.get("pass_index") or 1),
            str(item.get("run_id")),
        )
    )
    elapsed = time.perf_counter() - started
    aggregate = aggregate_results(results)
    aggregate.update(compute_recovery_metrics(results, EVAL_RESULT_DIR))
    summary_record = {
        "judge_run_id": run_id,
        "created_at": utc_now_iso(),
        "elapsed_seconds": elapsed,
        "eval_record_count": len(records),
        "target_gen_model": target_gen_model,
        "preference_injection_mode": preference_mode,
        "preference_injection_label": PREFERENCE_INJECTION_LABELS[preference_mode],
        "task_difficulty": normalize_task_difficulty(TASK_DIFFICULTY),
        "active_task_filter": sorted(task_ids_for_active_filters()),
        "pass_n": pass_n,
        "selected_pass_indexes": selected_pass_indexes,
        "selected_eval_run_ids": selected_eval_run_ids,
        "eval_result_dir": str(EVAL_RESULT_DIR),
        "judge_result_dir": str(JUDGE_RESULT_DIR),
        "judge_concurrency": judge_concurrency,
        "api_schema_audit": {
            "api_definition_count": len(API_DEFINITIONS),
            "no_relation_api_count": no_relation_api_count,
            "no_relation_policy": "NoRelation=1 的 API 调用跳过 Instantiation LLM-as-judge；仍按 JSON 结构化生成合法性计 1/0.5/0 分。",
        },
        "judge_model": {
            "name": client.model,
            "base_url": client.base_url,
            "api_key": mask_secret(client.api_key),
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS,
        },
        "hyperparameters": {
            "target_gen_llm": target_gen_model,
            "preference_injection_mode": preference_mode,
            "task_difficulty": normalize_task_difficulty(TASK_DIFFICULTY),
            "active_task_filter": sorted(task_ids_for_active_filters()),
            "pass_n": pass_n,
            "enable_long_context": bool(ENABLE_LONG_CONTEXT),
            "long_context_turns": normalize_long_context_turns(LONG_CONTEXT_TURNS),
            "target_eval_run_id": TARGET_EVAL_RUN_ID,
            "judge_latest_eval_run_only": JUDGE_LATEST_EVAL_RUN_ONLY,
            "max_tool_calls_per_prompt": MAX_TOOL_CALLS_PER_PROMPT,
            "max_update_events_per_prompt": MAX_UPDATE_EVENTS_PER_PROMPT,
            "instantiation_judge_chunk_size": INSTANTIATION_JUDGE_CHUNK_SIZE,
            "update_judge_chunk_size": UPDATE_JUDGE_CHUNK_SIZE,
            "hidden_judge_explicit_profile_reference": USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE,
            "json_repair_attempts": JSON_REPAIR_ATTEMPTS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        },
        "outputs": {
            "detail_jsonl": str(detail_path),
            "summary_json": str(summary_path),
            "html_report": str(html_path),
        },
        "aggregate": aggregate,
    }
    summary_path.write_text(json.dumps(summary_record, ensure_ascii=False, indent=2), encoding="utf-8")

    html_report = render_html_report(
        run_id=run_id,
        target_gen_model=target_gen_model,
        preference_mode=preference_mode,
        pass_n=pass_n,
        selected_eval_run_ids=selected_eval_run_ids,
        judge_model=summary_record["judge_model"],
        eval_result_dir=EVAL_RESULT_DIR,
        detail_path=detail_path,
        summary_path=summary_path,
        results=results,
        aggregate=aggregate,
    )
    html_path.write_text(html_report, encoding="utf-8")

    logger.info(
        "按任务汇总：First Preference Acc=%s/%s %.2f%%；First Plan Quality Acc=%s/%s %.2f%%；"
        "First Joint Acc=%s/%s %.2f%%；Instantiation Acc=%s/%s %.2f%%；"
        "Update Plan Quality Acc=%s/%s %.2f%%；Update Joint Acc=%s/%s %.2f%%",
        aggregate["first_plan_preference_following_count"],
        aggregate["total_cases"],
        aggregate["first_plan_preference_following_ratio"],
        aggregate["first_plan_quality_correct_count"],
        aggregate["total_cases"],
        aggregate["first_plan_quality_correct_ratio"],
        aggregate["first_plan_joint_correct_count"],
        aggregate["total_cases"],
        aggregate["first_plan_joint_correct_ratio"],
        aggregate["instantiation_case_correct_count"],
        aggregate["total_cases"],
        aggregate["instantiation_case_correct_ratio"],
        aggregate["update_plan_quality_correct_count"],
        aggregate["total_cases"],
        aggregate["update_plan_quality_correct_ratio"],
        aggregate["update_plan_correct_count"],
        aggregate["total_cases"],
        aggregate["update_plan_correct_ratio"],
    )
    logger.info(
        "按 Loop 汇总：Instantiation Case Acc=%s/%s %.2f%%；Update Plan Quality Case Acc=%s/%s %.2f%%；"
        "Update Joint Case Acc=%s/%s %.2f%%",
        aggregate["instantiation_step_preference_following_count"],
        aggregate["instantiation_step_count"],
        aggregate["instantiation_step_preference_following_ratio"],
        aggregate["update_plan_quality_step_correct_count"],
        aggregate["update_plan_step_count"],
        aggregate["update_plan_quality_step_correct_ratio"],
        aggregate["update_plan_step_correct_count"],
        aggregate["update_plan_step_count"],
        aggregate["update_plan_step_correct_ratio"],
    )
    logger.info(
        "judge 完成：elapsed=%.2fs trajectory=%s/%s %.2f%% html=%s",
        elapsed,
        aggregate["trajectory_preference_following_count"],
        aggregate["total_cases"],
        aggregate["trajectory_preference_following_ratio"],
        html_path,
    )


if __name__ == "__main__":
    main()
