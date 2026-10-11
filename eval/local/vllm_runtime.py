#!/usr/bin/env python3
"""Shared local vLLM runtime helpers for offline generation and offline judging."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from local_model import (
    LOCAL_MODEL_NATIVE_MAX_MODEL_LEN,
    USEABLE_LOCAL_MODEL_PATHS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLLM_ENV_DIR = PROJECT_ROOT / ".venv-vllm"
LOG_DIR = PROJECT_ROOT / "logs"

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

HEALTH_POLL_SECONDS = 2.0


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


def resolve_model_path(model_name: str) -> Path:
    if model_name not in USEABLE_LOCAL_MODEL_PATHS:
        raise ValueError(
            f"未知本地模型 {model_name!r}；可用模型为 {sorted(USEABLE_LOCAL_MODEL_PATHS)}"
        )
    model_path = Path(USEABLE_LOCAL_MODEL_PATHS[model_name]).resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"本地模型目录不完整: {model_path}")
    return model_path


def native_max_model_len(model_name: str) -> int | None:
    return LOCAL_MODEL_NATIVE_MAX_MODEL_LEN.get(model_name)


MODEL_OVERRIDE_DIR = PROJECT_ROOT / ".model_overrides"


def _materialize_extended_context_model(
    model_name: str,
    model_path: Path,
    max_model_len: int,
    native_limit: int,
    factor: float,
) -> Path:
    """Mirror the model dir with weights symlinked and only config.json rewritten.

    vLLM 0.19 从 config.json 推导上下文上限，`--hf-overrides` 不参与该推导，
    所以必须落一份改过 max_position_embeddings 与 YaRN 参数的 config.json。
    """
    target = MODEL_OVERRIDE_DIR / f"{model_name}-ctx{max_model_len}"
    target.mkdir(parents=True, exist_ok=True)
    for entry in model_path.iterdir():
        if entry.name == "config.json":
            continue
        link = target / entry.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(entry)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    rope = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": native_limit,
    }
    config["max_position_embeddings"] = max_model_len
    config["rope_scaling"] = rope
    config["rope_parameters"] = rope
    (target / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def resolve_context_length(
    model_name: str,
    model_path: Path,
    requested_max_model_len: int,
    *,
    allow_rope_scaling: bool,
) -> tuple[int, Path]:
    """Keep the requested context length, extending it with static YaRN when needed.

    评测 prompt 可达数万 token，加上最多 8k 输出会超过 Qwen3 系列 40960 的原生上限。
    压低 max_model_len 会让请求直接 400 并截断上下文，因此默认走 YaRN 外推。
    """
    native_limit = native_max_model_len(model_name)
    if native_limit is None or requested_max_model_len <= native_limit:
        return requested_max_model_len, model_path
    if not allow_rope_scaling:
        print(
            f"max_model_len {requested_max_model_len} 超过 {model_name} 原生上下文 "
            f"{native_limit}，且已禁用 rope scaling，收敛到原生上限",
            flush=True,
        )
        return native_limit, model_path
    factor = round(requested_max_model_len / native_limit, 4)
    effective_path = _materialize_extended_context_model(
        model_name, model_path, requested_max_model_len, native_limit, factor
    )
    print(
        f"{model_name} 原生上下文 {native_limit} 不足 {requested_max_model_len}，"
        f"启用静态 YaRN 外推 factor={factor}，使用改写后的模型目录 {effective_path}",
        flush=True,
    )
    return requested_max_model_len, effective_path


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


def plan_replicas(
    *,
    gpu_ids: list[str],
    host: str,
    port: int,
    minimum_free_gib: float,
    requested_utilization: float,
    reserve_gib: float,
    auto_adjust: bool,
    requested_max_num_seqs: int,
) -> list[dict[str, Any]]:
    """Map each GPU to one independent single-GPU vLLM replica on its own port."""
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU 列表必须非空且不重复，当前为 {gpu_ids!r}")
    replicas: list[dict[str, Any]] = []
    for replica_index, gpu_id in enumerate(gpu_ids):
        total_mib, free_mib = query_gpu_memory_mib(gpu_id)
        free_gib = free_mib / 1024
        if free_gib < minimum_free_gib:
            raise RuntimeError(
                f"GPU {gpu_id} 仅剩 {free_gib:.2f} GiB，低于启动阈值 {minimum_free_gib:.2f} GiB"
            )
        safe_utilization = max(0.1, min(0.99, (free_mib - reserve_gib * 1024) / total_mib))
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
    return replicas



def plan_tensor_parallel_replica(
    *,
    gpu_ids: list[str],
    host: str,
    port: int,
    requested_utilization: float,
    requested_max_num_seqs: int,
) -> list[dict[str, Any]]:
    """Build a single replica that shards one model across all listed GPUs (TP).

    用于 122B 这类单卡放不下的模型：一个副本占用全部 GPU，tensor_parallel_size=len(gpu_ids)。
    """
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU 列表必须非空且不重复，当前为 {gpu_ids!r}")
    joined = ",".join(gpu_ids)
    total_mib, free_mib = query_gpu_memory_mib(gpu_ids[0])
    return [
        {
            "replica_index": 0,
            "gpu_ids": joined,
            "port": port,
            "base_url": f"http://{host}:{port}/v1",
            "gpu_memory_total_mib": total_mib,
            "gpu_memory_free_before_start_mib": free_mib,
            "gpu_memory_utilization": round(requested_utilization, 4),
            "max_num_seqs": requested_max_num_seqs,
        }
    ]



def plan_tp_replicas(
    *,
    gpu_ids: list[str],
    tp: int,
    host: str,
    port: int,
    requested_utilization: float,
    requested_max_num_seqs: int,
) -> list[dict[str, Any]]:
    """把 GPU 按 tp 分组，每组一个张量并行副本（用于 32B 这类单卡放不下但不必占满 8 卡的模型）。"""
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"GPU 列表必须非空且不重复，当前为 {gpu_ids!r}")
    if tp < 1 or len(gpu_ids) % tp != 0:
        raise ValueError(f"GPU 数({len(gpu_ids)})必须能被 tensor_parallel_size({tp}) 整除")
    replicas: list[dict[str, Any]] = []
    for idx in range(0, len(gpu_ids), tp):
        group = gpu_ids[idx:idx + tp]
        total_mib, free_mib = query_gpu_memory_mib(group[0])
        replica_port = port + idx // tp
        replicas.append({
            "replica_index": idx // tp,
            "gpu_ids": ",".join(group),
            "port": replica_port,
            "base_url": f"http://{host}:{replica_port}/v1",
            "gpu_memory_total_mib": total_mib,
            "gpu_memory_free_before_start_mib": free_mib,
            "gpu_memory_utilization": round(requested_utilization, 4),
            "max_num_seqs": requested_max_num_seqs,
        })
    return replicas

def build_vllm_command(config: dict[str, Any]) -> list[str]:
    executable = Path(str(config["vllm_executable"]))
    if not executable.is_file():
        raise FileNotFoundError(
            f"找不到 vLLM 可执行文件 {executable}；请先创建 .venv-vllm 并安装 vllm"
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
        str(config.get("generation_config", "auto")),
        "--safetensors-load-strategy",
        str(config["safetensors_load_strategy"]),
    ]
    if config["enable_prefix_caching"]:
        command.append("--enable-prefix-caching")
    if config["enable_chunked_prefill"]:
        command.append("--enable-chunked-prefill")
    if config.get("default_chat_template_kwargs"):
        command += [
            "--default-chat-template-kwargs",
            json.dumps(config["default_chat_template_kwargs"]),
        ]
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
        str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)
    }
    return str(config["model_name"]) in model_ids


def wait_for_server(
    process: subprocess.Popen[Any],
    config: dict[str, Any],
    startup_timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM 服务启动失败，退出码 {return_code}")
        if server_is_ready(config):
            return
        time.sleep(HEALTH_POLL_SECONDS)
    raise TimeoutError("等待 vLLM 服务启动超时")


def stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def start_server(
    config: dict[str, Any],
    *,
    role: str,
    startup_timeout_seconds: int,
    reuse_running: bool,
) -> tuple[subprocess.Popen[Any] | None, Any]:
    if server_is_ready(config):
        if reuse_running:
            print(f"复用已运行的 vLLM 服务[{role}]: {config['base_url']}", flush=True)
            return None, None
        raise RuntimeError(f"端口上已有同名 vLLM 服务: {config['base_url']}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / (
        f"vllm_{role}_{config['model_name']}_gpu{config['gpu_ids']}_{timestamp}.log"
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
    printable_command[printable_command.index("--api-key") + 1] = "***"
    print(
        f"启动本地 vLLM 副本[{role}]："
        f"replica={config['replica_index']} gpu={config['gpu_ids']} "
        f"model={config['model_name']} path={config['model_path']} "
        f"gpu_free={config['gpu_memory_free_before_start_mib']}MiB "
        f"gpu_util={config['gpu_memory_utilization']} max_model_len={config['max_model_len']} "
        f"max_num_seqs={config['max_num_seqs']} log={log_path}",
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
        wait_for_server(process, config, startup_timeout_seconds)
    except BaseException as exc:
        stop_server(process)
        log_handle.close()
        raise RuntimeError(f"vLLM 未成功就绪，请检查 {log_path}") from exc
    print(f"vLLM 已就绪[{role}]: {config['base_url']}", flush=True)
    return process, log_handle
