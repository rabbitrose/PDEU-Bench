#!/usr/bin/env python3
"""Judge local-model hidden-preference results through the configured API judge."""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import threading
import sys
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge
import vllm_runtime
from local_model import USEABLE_LOCAL_MODEL_PATHS
from vllm_runtime import env_bool, env_float, env_int


# =========================
# 用户长期可调 judge 超参数
# =========================

TARGET_GEN_LLM = "Qwen3-8B"

# judge 后端：api 走下面的 JUDGE_URL/JUDGE_MODEL，local 走本地 vLLM 离线判分。
JUDGE_BACKEND = "api"
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_URL = "https://api.qingyuntop.top/v1"

# local 判分后端的 vLLM 参数。默认与生成侧端口错开，便于两者共存。
LOCAL_JUDGE_MODEL = "Qwen3-30B-A3B"
JUDGE_VLLM_HOST = "127.0.0.1"
JUDGE_VLLM_PORT = 8200
JUDGE_VLLM_API_KEY = "local-vllm-judge"
JUDGE_VLLM_GPU_IDS = "0,1,2,3,4,5,6,7"
JUDGE_VLLM_TENSOR_PARALLEL_SIZE = 1
JUDGE_VLLM_DTYPE = "auto"
JUDGE_VLLM_KV_CACHE_DTYPE = "auto"
JUDGE_VLLM_GPU_MEMORY_UTILIZATION = 0.90
JUDGE_VLLM_GPU_MEMORY_RESERVE_GIB = 1.0
JUDGE_VLLM_MIN_FREE_GPU_GIB = 12.0
JUDGE_VLLM_AUTO_ADJUST_FREE_MEMORY = True
JUDGE_VLLM_MAX_MODEL_LEN = 81_920
JUDGE_VLLM_ALLOW_ROPE_SCALING = True
JUDGE_VLLM_MAX_NUM_SEQS = 4
JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS = 16_384
JUDGE_VLLM_CPU_OFFLOAD_GB = 0
JUDGE_VLLM_SAFETENSORS_LOAD_STRATEGY = "prefetch"
JUDGE_VLLM_ENABLE_PREFIX_CACHING = True
JUDGE_VLLM_ENABLE_CHUNKED_PREFILL = True
JUDGE_VLLM_STARTUP_TIMEOUT_SECONDS = 900
JUDGE_VLLM_REUSE_RUNNING = False

# Qwen3 等思考型模型默认会把 8192 输出预算全部用于 <think>，导致结构化 JSON 被截断。
# 与生成侧同理，采用模型自带的推荐采样参数以抑制重复退化。
JUDGE_GENERATION_CONFIG_SOURCE = "auto"

JUDGE_DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


# 偏好注入模式，需与生成侧一致；hidden 或 mix。
PREFERENCE_INJECTION_MODE = "hidden"

TASK_DIFFICULTY = "mid"
PASS_N = 1
ENABLE_LONG_CONTEXT = False
LONG_CONTEXT_TURNS = 0
LONG_CONTEXT_TURNS_LIST: list[int] = []
TARGET_EVAL_RUN_ID = ""
JUDGE_LATEST_EVAL_RUN_ONLY = False
JUDGE_CONCURRENCY = 20
INSTANTIATION_JUDGE_CHUNK_SIZE = 4
USER_ID_FILTER: list[str] = []
TASK_ID_FILTER: list[str] = []


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


def normalize_local_difficulty(value: str) -> tuple[str, str]:
    requested = str(value).strip().lower()
    normalized = LEGACY_DIFFICULTY_ALIASES.get(requested, requested)
    return requested, judge.normalize_task_difficulty(normalized)


def clear_proxy_environment() -> None:
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)


def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_judge_vllm_config() -> dict[str, Any]:
    """Build a single-replica vLLM config used to serve the offline judge model."""
    model_name = os.environ.get("PPEVAL_JUDGE_LOCAL_MODEL", LOCAL_JUDGE_MODEL)
    model_path = vllm_runtime.resolve_model_path(model_name)
    gpu_ids_text = os.environ.get("PPEVAL_JUDGE_VLLM_GPU_IDS", JUDGE_VLLM_GPU_IDS)
    gpu_ids = [item.strip() for item in gpu_ids_text.split(",") if item.strip()]
    max_model_len, model_path = vllm_runtime.resolve_context_length(
        model_name,
        model_path,
        env_int("PPEVAL_JUDGE_VLLM_MAX_MODEL_LEN", JUDGE_VLLM_MAX_MODEL_LEN),
        allow_rope_scaling=env_bool(
            "PPEVAL_JUDGE_VLLM_ALLOW_ROPE_SCALING", JUDGE_VLLM_ALLOW_ROPE_SCALING
        ),
    )
    host = os.environ.get("PPEVAL_JUDGE_VLLM_HOST", JUDGE_VLLM_HOST)
    tensor_parallel_size = env_int(
        "PPEVAL_JUDGE_VLLM_TENSOR_PARALLEL_SIZE", JUDGE_VLLM_TENSOR_PARALLEL_SIZE
    )
    if tensor_parallel_size > 1:
        # 单卡放不下的大模型：一个副本用 tensor parallel 横跨 tensor_parallel_size 张卡。
        if len(gpu_ids) != tensor_parallel_size:
            raise ValueError(
                f"张量并行要求 GPU 数({len(gpu_ids)})等于 tensor_parallel_size"
                f"({tensor_parallel_size})"
            )
        replicas = vllm_runtime.plan_tensor_parallel_replica(
            gpu_ids=gpu_ids,
            host=host,
            port=env_int("PPEVAL_JUDGE_VLLM_PORT", JUDGE_VLLM_PORT),
            requested_utilization=env_float(
                "PPEVAL_JUDGE_VLLM_GPU_MEMORY_UTILIZATION",
                JUDGE_VLLM_GPU_MEMORY_UTILIZATION,
            ),
            requested_max_num_seqs=env_int(
                "PPEVAL_JUDGE_VLLM_MAX_NUM_SEQS", JUDGE_VLLM_MAX_NUM_SEQS
            ),
        )
    else:
        replicas = vllm_runtime.plan_replicas(
            gpu_ids=gpu_ids,
            host=host,
            port=env_int("PPEVAL_JUDGE_VLLM_PORT", JUDGE_VLLM_PORT),
            minimum_free_gib=env_float(
                "PPEVAL_JUDGE_VLLM_MIN_FREE_GPU_GIB", JUDGE_VLLM_MIN_FREE_GPU_GIB
            ),
            requested_utilization=env_float(
                "PPEVAL_JUDGE_VLLM_GPU_MEMORY_UTILIZATION",
                JUDGE_VLLM_GPU_MEMORY_UTILIZATION,
            ),
            reserve_gib=env_float(
                "PPEVAL_JUDGE_VLLM_GPU_MEMORY_RESERVE_GIB",
                JUDGE_VLLM_GPU_MEMORY_RESERVE_GIB,
            ),
            auto_adjust=env_bool(
                "PPEVAL_JUDGE_VLLM_AUTO_ADJUST_FREE_MEMORY",
                JUDGE_VLLM_AUTO_ADJUST_FREE_MEMORY,
            ),
            requested_max_num_seqs=env_int(
                "PPEVAL_JUDGE_VLLM_MAX_NUM_SEQS", JUDGE_VLLM_MAX_NUM_SEQS
            ),
        )

    return {
        **replicas[0],
        "replicas": replicas,
        "replica_count": len(replicas),
        "model_name": model_name,
        "model_path": str(model_path),
        "host": host,
        "api_key": os.environ.get("PPEVAL_JUDGE_VLLM_API_KEY", JUDGE_VLLM_API_KEY),
        "vllm_executable": str(vllm_runtime.VLLM_ENV_DIR / "bin" / "vllm"),
        "tensor_parallel_size": env_int(
            "PPEVAL_JUDGE_VLLM_TENSOR_PARALLEL_SIZE", JUDGE_VLLM_TENSOR_PARALLEL_SIZE
        ),
        "dtype": os.environ.get("PPEVAL_JUDGE_VLLM_DTYPE", JUDGE_VLLM_DTYPE),
        "kv_cache_dtype": os.environ.get(
            "PPEVAL_JUDGE_VLLM_KV_CACHE_DTYPE", JUDGE_VLLM_KV_CACHE_DTYPE
        ),
        "max_model_len": max_model_len,
        "generation_config": os.environ.get(
            "PPEVAL_JUDGE_VLLM_GENERATION_CONFIG", JUDGE_GENERATION_CONFIG_SOURCE
        ),
        "default_chat_template_kwargs": json.loads(
            os.environ.get("PPEVAL_JUDGE_VLLM_CHAT_TEMPLATE_KWARGS", json.dumps(JUDGE_DEFAULT_CHAT_TEMPLATE_KWARGS))
        ),
        "max_num_batched_tokens": env_int(
            "PPEVAL_JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS",
            JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS,
        ),
        "cpu_offload_gb": env_int(
            "PPEVAL_JUDGE_VLLM_CPU_OFFLOAD_GB", JUDGE_VLLM_CPU_OFFLOAD_GB
        ),
        "safetensors_load_strategy": os.environ.get(
            "PPEVAL_JUDGE_VLLM_SAFETENSORS_LOAD_STRATEGY",
            JUDGE_VLLM_SAFETENSORS_LOAD_STRATEGY,
        ),
        "enable_prefix_caching": env_bool(
            "PPEVAL_JUDGE_VLLM_ENABLE_PREFIX_CACHING",
            JUDGE_VLLM_ENABLE_PREFIX_CACHING,
        ),
        "enable_chunked_prefill": env_bool(
            "PPEVAL_JUDGE_VLLM_ENABLE_CHUNKED_PREFILL",
            JUDGE_VLLM_ENABLE_CHUNKED_PREFILL,
        ),
    }


class RoundRobinJudgeClient:
    """Spread judge requests across the independent single-GPU vLLM replicas."""

    def __init__(
        self,
        *,
        api_key: str,
        base_urls: list[str],
        model: str,
        logger: Any,
    ) -> None:
        if not base_urls:
            raise ValueError("本地 judge endpoint 列表不能为空")
        self.api_key = api_key
        self.base_url = ",".join(base_urls)
        self.model = model
        self.logger = logger
        self._clients = [
            judge.LLMClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                logger=logger,
            )
            for base_url in base_urls
        ]
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        with self._lock:
            index = next(self._counter) % len(self._clients)
        return self._clients[index].chat(messages, **kwargs)


def configure_judge_backend() -> tuple[str, list[subprocess.Popen[Any] | None], list[Any]]:
    """Point judge.py at either the configured API judge or a local vLLM judge."""
    backend = os.environ.get("PPEVAL_JUDGE_BACKEND", JUDGE_BACKEND).strip().lower()
    if backend not in {"api", "local"}:
        raise ValueError(f"PPEVAL_JUDGE_BACKEND 只能是 api 或 local，当前为 {backend!r}")
    if backend == "api":
        os.environ["PPEVAL_JUDGE_MODEL"] = os.environ.get(
            "PPEVAL_JUDGE_MODEL", JUDGE_MODEL
        )
        os.environ["PPEVAL_JUDGE_URL"] = os.environ.get("PPEVAL_JUDGE_URL", JUDGE_URL)
        print(
            f"judge 后端=api model={os.environ['PPEVAL_JUDGE_MODEL']} "
            f"url={os.environ['PPEVAL_JUDGE_URL']}",
            flush=True,
        )
        return backend, [], []

    config = resolve_judge_vllm_config()
    startup_timeout = env_int(
        "PPEVAL_JUDGE_VLLM_STARTUP_TIMEOUT_SECONDS", JUDGE_VLLM_STARTUP_TIMEOUT_SECONDS
    )
    reuse_running = env_bool(
        "PPEVAL_JUDGE_VLLM_REUSE_RUNNING", JUDGE_VLLM_REUSE_RUNNING
    )
    processes: list[subprocess.Popen[Any] | None] = []
    log_handles: list[Any] = []
    base_urls: list[str] = []
    for replica in config["replicas"]:
        replica_config = {**config, **replica}
        process, log_handle = vllm_runtime.start_server(
            replica_config,
            role="judge",
            startup_timeout_seconds=startup_timeout,
            reuse_running=reuse_running,
        )
        processes.append(process)
        log_handles.append(log_handle)
        base_urls.append(str(replica["base_url"]))
    os.environ["PPEVAL_JUDGE_MODEL"] = str(config["model_name"])
    os.environ["PPEVAL_JUDGE_URL"] = str(config["base_url"])
    os.environ["PPEVAL_JUDGE_KEY"] = str(config["api_key"])
    judge.JUDGE_CLIENT_FACTORY = lambda api_config, logger: RoundRobinJudgeClient(
        api_key=getattr(api_config, "JUDGE_KEY"),
        base_urls=base_urls,
        model=getattr(api_config, "JUDGE_MODEL"),
        logger=logger,
    )
    print(
        f"judge 后端=local model={config['model_name']} "
        f"replicas={len(base_urls)} urls={','.join(base_urls)}",
        flush=True,
    )
    return backend, processes, log_handles


def main() -> None:
    target_model = os.environ.get("PPEVAL_GEN_MODEL", TARGET_GEN_LLM)
    if target_model not in USEABLE_LOCAL_MODEL_PATHS:
        # 被测结果也可能来自 API 模型；此入口只负责判分，不限制生成侧后端。
        print(
            f"被测模型 {target_model!r} 不在本地模型白名单中，按 API 生成结果判分；"
            f"本地可用模型为 {sorted(USEABLE_LOCAL_MODEL_PATHS)}",
            flush=True,
        )
    requested, normalized = normalize_local_difficulty(
        os.environ.get("PPEVAL_TASK_DIFFICULTY", TASK_DIFFICULTY)
    )
    if requested != normalized:
        print(
            f"难度兼容映射：{requested!r} -> {normalized!r}；旧 mid 结果当前存储在 hard 目录",
            flush=True,
        )
    clear_proxy_environment()
    preference_mode = judge.normalize_preference_injection_mode(
        os.environ.get("PPEVAL_PREFERENCE_MODE", PREFERENCE_INJECTION_MODE)
    )
    print(f"偏好注入模式={preference_mode}", flush=True)
    previous_signal_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGHUP, signal.SIGTERM)
    }

    def stop_on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in previous_signal_handlers:
        signal.signal(signum, stop_on_signal)
    backend, judge_processes, judge_log_handles = configure_judge_backend()
    turns_list = LONG_CONTEXT_TURNS_LIST or [LONG_CONTEXT_TURNS]
    try:
        for turns in turns_list:
            judge.API_RETRY_ATTEMPTS = 3
            judge.API_RETRY_BASE_SLEEP_SECONDS = 5.0
            judge.TARGET_GEN_LLM = target_model
            judge.PREFERENCE_INJECTION_MODE = preference_mode
            # 只有 hidden 模式下被测模型没看过显式画像，judge 才用它作隐藏参考。
            judge.USE_EXPLICIT_USER_PROFILE_AS_HIDDEN_REFERENCE = (
                preference_mode == "hidden"
            )
            judge.INSTANTIATION_JUDGE_CHUNK_SIZE = INSTANTIATION_JUDGE_CHUNK_SIZE
            judge.PASS_N = int(os.environ.get("PPEVAL_PASS_N", PASS_N))
            judge.ENABLE_LONG_CONTEXT = ENABLE_LONG_CONTEXT
            judge.LONG_CONTEXT_TURNS = turns
            judge.TASK_DIFFICULTY = normalized
            judge.TARGET_EVAL_RUN_ID = TARGET_EVAL_RUN_ID
            judge.JUDGE_LATEST_EVAL_RUN_ONLY = JUDGE_LATEST_EVAL_RUN_ONLY
            judge.AUTO_TUNE_CONCURRENCY = False
            default_concurrency = 4 if backend == "local" else JUDGE_CONCURRENCY
            concurrency = int(os.environ.get("PPEVAL_CONCURRENCY", default_concurrency))
            judge.DEFAULT_JUDGE_CONCURRENCY = concurrency
            judge.DEEPSEEK_JUDGE_CONCURRENCY = concurrency
            judge.QINGYUN_DEEPSEEK_JUDGE_CONCURRENCY = concurrency
            judge.USER_ID_FILTER = env_list("PPEVAL_USER_ID_FILTER", USER_ID_FILTER)
            judge.TASK_ID_FILTER = env_list("PPEVAL_TASK_ID_FILTER", TASK_ID_FILTER)
            judge.main()
    finally:
        for process in reversed(judge_processes):
            vllm_runtime.stop_server(process)
        for log_handle in judge_log_handles:
            if log_handle is not None:
                log_handle.close()
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
