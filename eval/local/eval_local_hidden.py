#!/usr/bin/env python3
"""Run hidden-preference evaluation against a local vLLM server."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_core
from local_model import USEABLE_LOCAL_MODEL_PATHS


# =========================
# 用户长期可调评估超参数
# =========================

LOCAL_MODEL_NAME = "Qwen3-4B-Instruct-2507"

# 旧 mid 数据已迁移为当前 hard；保留 mid 作为本地入口的兼容名称。
TASK_DIFFICULTY = "mid"
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []
PASS_N = 1
MAX_TOOL_CALLS_PER_TASK = eval_core.MAX_TOOL_CALLS_PER_TASK
TOOL_AVAILABILITY_MODE = "all"
SKIP_TASKS_ALREADY_IN_RESULT_FILE = False
MAX_TOOL_CALLS_OVERRIDE: int | None = None
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []
HIDDEN_CONTEXT_MAX_CHARS = 200_000
FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = True

# vLLM 服务与批处理参数。HTTP 并发可高于同时驻留 GPU 的序列数，vLLM 会连续批处理。
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8100
VLLM_API_KEY = "local-vllm"
VLLM_ENV_DIR = PROJECT_ROOT / ".venv-vllm"
VLLM_GPU_IDS = "0,1"
TENSOR_PARALLEL_SIZE = 1
DTYPE = "auto"
KV_CACHE_DTYPE = "auto"
GPU_MEMORY_UTILIZATION = 0.94
GPU_MEMORY_RESERVE_GIB = 1.0
MIN_FREE_GPU_MEMORY_GIB = 12.0
AUTO_ADJUST_FOR_FREE_GPU_MEMORY = True
MAX_MODEL_LEN = 65_536
MIN_MODEL_LEN_FOR_CURRENT_PROMPTS = 49_152
MAX_NUM_SEQS = 2
MAX_NUM_BATCHED_TOKENS = 16_384
CPU_OFFLOAD_GB = 0
SAFETENSORS_LOAD_STRATEGY = "prefetch"
ENABLE_PREFIX_CACHING = True
ENABLE_CHUNKED_PREFILL = True
EVAL_CONCURRENCY = 4
LOCAL_REQUEST_TIMEOUT_SECONDS = 900
SANDBOX_E5_DEVICE = "cpu"
VLLM_STARTUP_TIMEOUT_SECONDS = 900
VLLM_HEALTH_POLL_SECONDS = 2.0
REUSE_RUNNING_VLLM_SERVER = False


LEGACY_DIFFICULTY_ALIASES = {"mid": "hard"}
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


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_local_difficulty(value: str) -> tuple[str, str]:
    requested = str(value).strip().lower()
    normalized = LEGACY_DIFFICULTY_ALIASES.get(requested, requested)
    return requested, eval_core.normalize_task_difficulty(normalized)


def query_gpu_memory_mib(gpu_id: str) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    first_line = completed.stdout.strip().splitlines()[0]
    total_text, free_text = (part.strip() for part in first_line.split(",", 1))
    return int(total_text), int(free_text)


def resolve_runtime_config() -> dict[str, Any]:
    model_name = os.environ.get("PPEVAL_LOCAL_MODEL", LOCAL_MODEL_NAME)
    if model_name not in USEABLE_LOCAL_MODEL_PATHS:
        raise ValueError(
            f"未知本地模型 {model_name!r}；可用模型为 {sorted(USEABLE_LOCAL_MODEL_PATHS)}"
        )
    model_path = Path(USEABLE_LOCAL_MODEL_PATHS[model_name]).resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"本地模型目录不完整: {model_path}")

    gpu_ids_text = os.environ.get("PPEVAL_VLLM_GPU_IDS", VLLM_GPU_IDS)
    gpu_ids = [item.strip() for item in gpu_ids_text.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"PPEVAL_VLLM_GPU_IDS 必须是非重复 GPU 列表，当前为 {gpu_ids_text!r}")
    minimum_free = env_float("PPEVAL_VLLM_MIN_FREE_GPU_GIB", MIN_FREE_GPU_MEMORY_GIB)
    requested_utilization = env_float(
        "PPEVAL_VLLM_GPU_MEMORY_UTILIZATION", GPU_MEMORY_UTILIZATION
    )
    reserve_gib = env_float("PPEVAL_VLLM_GPU_MEMORY_RESERVE_GIB", GPU_MEMORY_RESERVE_GIB)
    auto_adjust = env_bool(
        "PPEVAL_VLLM_AUTO_ADJUST_FREE_MEMORY", AUTO_ADJUST_FOR_FREE_GPU_MEMORY
    )
    requested_max_num_seqs = env_int("PPEVAL_VLLM_MAX_NUM_SEQS", MAX_NUM_SEQS)
    port = env_int("PPEVAL_VLLM_PORT", VLLM_PORT)
    max_model_len = env_int("PPEVAL_VLLM_MAX_MODEL_LEN", MAX_MODEL_LEN)
    max_num_batched_tokens = env_int(
        "PPEVAL_VLLM_MAX_NUM_BATCHED_TOKENS", MAX_NUM_BATCHED_TOKENS
    )
    if max_num_batched_tokens <= 0 or max_model_len <= 0:
        raise ValueError("vLLM token 上限必须为正整数")
    if max_model_len < MIN_MODEL_LEN_FOR_CURRENT_PROMPTS:
        raise ValueError(
            f"max_model_len={max_model_len} 无法容纳当前约 37k token prompt 与最多 "
            f"{eval_core.MODEL_MAX_TOKENS} token 输出；不得低于 {MIN_MODEL_LEN_FOR_CURRENT_PROMPTS}"
        )
    replicas: list[dict[str, Any]] = []
    host = os.environ.get("PPEVAL_VLLM_HOST", VLLM_HOST)
    for replica_index, gpu_id in enumerate(gpu_ids):
        total_mib, free_mib = query_gpu_memory_mib(gpu_id)
        free_gib = free_mib / 1024
        if free_gib < minimum_free:
            raise RuntimeError(
                f"GPU {gpu_id} 仅剩 {free_gib:.2f} GiB，低于启动阈值 {minimum_free:.2f} GiB"
            )
        safe_utilization = max(
            0.1,
            min(0.99, (free_mib - reserve_gib * 1024) / total_mib),
        )
        resolved_utilization = (
            min(requested_utilization, safe_utilization)
            if auto_adjust
            else requested_utilization
        )
        if not 0 < resolved_utilization < 1:
            raise ValueError(
                f"GPU {gpu_id} 的 gpu_memory_utilization 必须在 (0, 1) 内，"
                f"当前为 {resolved_utilization}"
            )
        free_ratio = max(0.1, min(1.0, free_mib / total_mib))
        resolved_max_num_seqs = (
            max(1, round(requested_max_num_seqs * free_ratio))
            if auto_adjust
            else requested_max_num_seqs
        )
        replica_port = port + replica_index
        replicas.append(
            {
                "replica_index": replica_index,
                "gpu_ids": gpu_id,
                "port": replica_port,
                "base_url": f"http://{host}:{replica_port}/v1",
                "gpu_memory_total_mib": total_mib,
                "gpu_memory_free_before_start_mib": free_mib,
                "gpu_memory_utilization": round(resolved_utilization, 4),
                "max_num_seqs": resolved_max_num_seqs,
            }
        )

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "host": host,
        "port": port,
        "base_url": replicas[0]["base_url"],
        "api_key": os.environ.get("PPEVAL_VLLM_API_KEY", VLLM_API_KEY),
        "vllm_executable": str(VLLM_ENV_DIR / "bin" / "vllm"),
        "gpu_ids": gpu_ids,
        "replica_count": len(replicas),
        "replicas": replicas,
        "gpu_memory_utilization_requested": requested_utilization,
        "gpu_memory_reserve_gib": reserve_gib,
        "auto_adjust_for_free_gpu_memory": auto_adjust,
        "tensor_parallel_size": env_int(
            "PPEVAL_VLLM_TENSOR_PARALLEL_SIZE", TENSOR_PARALLEL_SIZE
        ),
        "dtype": os.environ.get("PPEVAL_VLLM_DTYPE", DTYPE),
        "kv_cache_dtype": os.environ.get("PPEVAL_VLLM_KV_CACHE_DTYPE", KV_CACHE_DTYPE),
        "max_model_len": max_model_len,
        "max_num_seqs_requested": requested_max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "cpu_offload_gb": env_int("PPEVAL_VLLM_CPU_OFFLOAD_GB", CPU_OFFLOAD_GB),
        "safetensors_load_strategy": os.environ.get(
            "PPEVAL_VLLM_SAFETENSORS_LOAD_STRATEGY", SAFETENSORS_LOAD_STRATEGY
        ),
        "enable_prefix_caching": env_bool(
            "PPEVAL_VLLM_ENABLE_PREFIX_CACHING", ENABLE_PREFIX_CACHING
        ),
        "enable_chunked_prefill": env_bool(
            "PPEVAL_VLLM_ENABLE_CHUNKED_PREFILL", ENABLE_CHUNKED_PREFILL
        ),
        "eval_concurrency": env_int("PPEVAL_CONCURRENCY", EVAL_CONCURRENCY),
    }


def build_vllm_command(config: dict[str, Any]) -> list[str]:
    executable = Path(str(config["vllm_executable"]))
    if not executable.is_file():
        raise FileNotFoundError(
            f"找不到 vLLM 可执行文件 {executable}；请先用项目 uv 环境安装 local-inference 依赖"
        )
    command = [
        str(executable),
        "serve",
        str(config["model_path"]),
        "--served-model-name",
        str(config["model_name"]),
        "--host",
        str(config["host"]),
        "--port",
        str(config["port"]),
        "--api-key",
        str(config["api_key"]),
        "--tensor-parallel-size",
        str(config["tensor_parallel_size"]),
        "--dtype",
        str(config["dtype"]),
        "--kv-cache-dtype",
        str(config["kv_cache_dtype"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(config["max_num_batched_tokens"]),
        "--cpu-offload-gb",
        str(config["cpu_offload_gb"]),
        "--generation-config",
        "vllm",
        "--safetensors-load-strategy",
        str(config["safetensors_load_strategy"]),
    ]
    if config["enable_prefix_caching"]:
        command.append("--enable-prefix-caching")
    if config["enable_chunked_prefill"]:
        command.append("--enable-chunked-prefill")
    return command


def server_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def server_is_ready(config: dict[str, Any]) -> bool:
    request = urllib.request.Request(
        f"{config['base_url']}/models",
        headers={"Authorization": f"Bearer {config['api_key']}"},
    )
    try:
        with server_opener().open(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    model_ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict)
    }
    return str(config["model_name"]) in model_ids


def wait_for_server(process: subprocess.Popen[Any], config: dict[str, Any]) -> None:
    deadline = time.monotonic() + env_int(
        "PPEVAL_VLLM_STARTUP_TIMEOUT_SECONDS", VLLM_STARTUP_TIMEOUT_SECONDS
    )
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM 服务启动失败，退出码 {return_code}")
        if server_is_ready(config):
            return
        time.sleep(VLLM_HEALTH_POLL_SECONDS)
    raise TimeoutError("等待 vLLM 服务启动超时")


def start_vllm_server(config: dict[str, Any]) -> tuple[subprocess.Popen[Any] | None, Any]:
    if server_is_ready(config):
        if env_bool("PPEVAL_VLLM_REUSE_RUNNING", REUSE_RUNNING_VLLM_SERVER):
            print(f"复用已运行的 vLLM 服务: {config['base_url']}", flush=True)
            return None, None
        raise RuntimeError(f"端口上已有同名 vLLM 服务: {config['base_url']}")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (
        f"vllm_{config['model_name']}_gpu{config['gpu_ids']}_{timestamp}.log"
    )
    log_handle = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    for name in PROXY_ENV_NAMES:
        environment.pop(name, None)
    environment.update(
        {
            "PATH": f"{VLLM_ENV_DIR / 'bin'}:{environment.get('PATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(config["gpu_ids"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    command = build_vllm_command(config)
    printable_command = list(command)
    api_key_index = printable_command.index("--api-key") + 1
    printable_command[api_key_index] = "***"
    print(
        "启动本地 vLLM 副本："
        f"replica={config['replica_index']} gpu={config['gpu_ids']} "
        f"model={config['model_name']} path={config['model_path']} "
        f"gpu_free={config['gpu_memory_free_before_start_mib']}MiB "
        f"gpu_util={config['gpu_memory_utilization']} max_model_len={config['max_model_len']} "
        f"max_num_seqs={config['max_num_seqs']} eval_concurrency={config['eval_concurrency']} "
        f"log={log_path}",
        flush=True,
    )
    print(f"vLLM command: {' '.join(printable_command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_server(process, config)
    except BaseException as exc:
        stop_vllm_server(process)
        log_handle.close()
        raise RuntimeError(f"vLLM 未成功就绪，请检查 {log_path}") from exc
    print(f"vLLM 已就绪: {config['base_url']}", flush=True)
    return process, log_handle


def stop_vllm_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


class StickyReplicaLLMClient:
    """Route all turns for one user to the same independent GPU replica."""

    def __init__(
        self,
        *,
        api_key: str,
        base_urls: list[str],
        model: str,
        logger: Any,
    ) -> None:
        if not base_urls:
            raise ValueError("本地 vLLM endpoint 列表不能为空")
        self.api_key = api_key
        self.base_url = ",".join(base_urls)
        self.model = model
        self.logger = logger
        self._clients = [
            eval_core.LLMClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                logger=logger,
            )
            for base_url in base_urls
        ]
        self._logged_users: set[str] = set()
        self._log_lock = threading.Lock()

    def client_index_for_purpose(self, purpose: str) -> int:
        user_id = str(purpose).split("/", 1)[0]
        match = re.search(r"(\d+)$", user_id)
        if match:
            return int(match.group(1)) % len(self._clients)
        return sum(user_id.encode("utf-8")) % len(self._clients)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        purpose = str(kwargs.get("purpose") or "")
        client_index = self.client_index_for_purpose(purpose)
        user_id = purpose.split("/", 1)[0]
        with self._log_lock:
            if user_id and user_id not in self._logged_users:
                self.logger.info(
                    "本地 vLLM 固定路由：user=%s replica=%s base_url=%s",
                    user_id,
                    client_index,
                    self._clients[client_index].base_url,
                )
                self._logged_users.add(user_id)
        return self._clients[client_index].chat(messages, **kwargs)


def configure_eval(config: dict[str, Any]) -> tuple[str, str]:
    requested_difficulty = os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
    requested, normalized = normalize_local_difficulty(requested_difficulty)
    if requested != normalized:
        print(
            f"难度兼容映射：{requested!r} -> {normalized!r}；旧 mid 数据当前存储为 task_hard",
            flush=True,
        )
    os.environ["PPEVAL_GEN_MODEL"] = str(config["model_name"])
    os.environ["PPEVAL_GEN_URL"] = str(config["base_url"])
    os.environ["PPEVAL_GEN_KEY"] = str(config["api_key"])

    eval_core.PREFERENCE_INJECTION_MODE = "hidden"
    eval_core.USER_ID_FILTER = env_list("PPEVAL_USER_ID_FILTER", USER_ID_FILTER)
    eval_core.TASK_ID_FILTER = env_list("PPEVAL_TASK_ID_FILTER", TASK_ID_FILTER)
    eval_core.TASK_DIFFICULTY = normalized
    eval_core.PASS_N = env_int("PPEVAL_PASS_N", PASS_N)
    max_tool_calls_override_text = os.environ.get("PPEVAL_LOCAL_MAX_TOOL_CALLS")
    max_tool_calls_override = (
        int(max_tool_calls_override_text)
        if max_tool_calls_override_text is not None
        else MAX_TOOL_CALLS_OVERRIDE
    )
    eval_core.MAX_TOOL_CALLS_PER_TASK = (
        max_tool_calls_override
        if max_tool_calls_override is not None
        else MAX_TOOL_CALLS_PER_TASK
    )
    if max_tool_calls_override is not None:
        if max_tool_calls_override <= 0:
            raise ValueError("PPEVAL_LOCAL_MAX_TOOL_CALLS 必须为正整数")
        eval_core.MAX_TOOL_CALLS_BY_TASK_DIFFICULTY[normalized] = max_tool_calls_override
    eval_core.TASK_MAX_TOOL_CALLS_OVERRIDE = max_tool_calls_override
    eval_core.MAX_JSON_CONTEXT_CHARS = HIDDEN_CONTEXT_MAX_CHARS
    eval_core.FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT = (
        FILTER_SANDBOX_OBSERVATION_METADATA_IN_PROMPT
    )
    skip_existing = env_bool("PPEVAL_SKIP_EXISTING", SKIP_TASKS_ALREADY_IN_RESULT_FILE)
    eval_core.SKIP_TASKS_ALREADY_IN_RESULT_FILE = skip_existing
    eval_core.OVERWRITE_SELECTED_USER_RESULTS_ON_START = not skip_existing
    eval_core.AUTO_TUNE_CONCURRENCY = False
    eval_core.DEFAULT_EVAL_CONCURRENCY = int(config["eval_concurrency"])
    replica_base_urls = [str(replica["base_url"]) for replica in config["replicas"]]
    eval_core.GENERATION_CLIENT_FACTORY = lambda api_config, logger: StickyReplicaLLMClient(
        api_key=getattr(api_config, "GEN_KEY"),
        base_urls=replica_base_urls,
        model=getattr(api_config, "GEN_MODEL"),
        logger=logger,
    )
    eval_core.REQUEST_TIMEOUT_SECONDS = env_int(
        "PPEVAL_LOCAL_REQUEST_TIMEOUT_SECONDS", LOCAL_REQUEST_TIMEOUT_SECONDS
    )
    eval_core.TOOL_AVAILABILITY_MODE = TOOL_AVAILABILITY_MODE
    os.environ["PPEVAL_SANDBOX_E5_DEVICE"] = os.environ.get(
        "PPEVAL_SANDBOX_E5_DEVICE", SANDBOX_E5_DEVICE
    )
    eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
    eval_core.LOCAL_INFERENCE_CONFIG = {
        **{key: value for key, value in config.items() if key != "api_key"},
        "requested_task_difficulty": requested,
        "normalized_task_difficulty": normalized,
        "offline_model_loading": True,
        "sandbox_e5_device": os.environ["PPEVAL_SANDBOX_E5_DEVICE"],
        "max_tool_calls_override": max_tool_calls_override,
    }
    return requested, normalized


def main() -> None:
    config = resolve_runtime_config()
    configure_eval(config)
    replica_configs = [{**config, **replica} for replica in config["replicas"]]
    processes: list[subprocess.Popen[Any] | None] = []
    log_handles: list[Any] = []
    previous_signal_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGHUP, signal.SIGTERM)
    }

    def stop_on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in previous_signal_handlers:
        signal.signal(signum, stop_on_signal)
    try:
        for replica_config in replica_configs:
            process, log_handle = start_vllm_server(replica_config)
            processes.append(process)
            log_handles.append(log_handle)
        turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
        for turns in turns_list:
            eval_core.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
            eval_core.LONG_CONTEXT_TURNS = turns
            eval_core.main()
    finally:
        for process in reversed(processes):
            stop_vllm_server(process)
        for log_handle in log_handles:
            if log_handle is not None:
                log_handle.close()
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
