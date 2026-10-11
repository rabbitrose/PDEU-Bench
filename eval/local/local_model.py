#这里存储的是许多可以调用访问用于推理的本地的模型路径
#每个模型有自己的缩写或者代指叫法：
#当说到qwen的4b模型，说的就是Qwen3-4B
#当说到qwen的8b模型，说的就是Qwen3-8B
#当说到qwen的7b模型，说的就是Qwen2.5-7B-Instruct
#当说到llama的3b模型，说的就是Llama-3.2-3B-Instruct
#当说到qwen的30b模型，说的就是Qwen3-30B-A3B

LOCAL_MODEL_ROOT = "/root/paddlejob/workspace/env_run/huayi/models"

USEABLE_LOCAL_MODEL_PATHS = {
    "Qwen3-1.7B": f"{LOCAL_MODEL_ROOT}/Qwen3-1.7B",
    "Qwen3-4B": f"{LOCAL_MODEL_ROOT}/Qwen3-4B",
    "Qwen3-8B": f"{LOCAL_MODEL_ROOT}/Qwen3-8B",
    "Qwen3-14B": f"{LOCAL_MODEL_ROOT}/Qwen3-14B",
    "Qwen3-32B": f"{LOCAL_MODEL_ROOT}/Qwen3-32B",
    "Qwen3-30B-A3B": f"{LOCAL_MODEL_ROOT}/Qwen3-30B-A3B",
    "Qwen2.5-1.5B": f"{LOCAL_MODEL_ROOT}/Qwen2.5-1.5B",
    "Qwen2.5-3B": f"{LOCAL_MODEL_ROOT}/Qwen2.5-3B",
    "Qwen2.5-7B-Instruct": f"{LOCAL_MODEL_ROOT}/Qwen2.5-7B-Instruct",
    "Llama-3.2-3B-Instruct": f"{LOCAL_MODEL_ROOT}/Llama-3.2-3B-Instruct",
    "Qwen3.5-9B": f"{LOCAL_MODEL_ROOT}/Qwen3.5-9B",
    "Qwen3.8-27B": f"{LOCAL_MODEL_ROOT}/Qwen3.8-27B",
    "Qwen3.6-27B": f"{LOCAL_MODEL_ROOT}/Qwen3.6-27B",
    "Qwen3.5-122B-A10B": f"{LOCAL_MODEL_ROOT}/Qwen3.5-122B-A10B",
    "Qwen3.6-35B-A3B": f"{LOCAL_MODEL_ROOT}/Qwen3.6-35B-A3B",
    "Qwen3-30B-A3B-Thinking-2507": f"{LOCAL_MODEL_ROOT}/Qwen3-30B-A3B-Thinking-2507",
    "Llama-4-Scout-17B-16E-Instruct": f"{LOCAL_MODEL_ROOT}/Llama-4-Scout-17B-16E-Instruct",
    "Qwen3-Next-80B-A3B-Instruct": f"{LOCAL_MODEL_ROOT}/Qwen3-Next-80B-A3B-Instruct",
}

# 各本地模型 config.json 中声明的原生上下文长度上限，用于自动收敛 vLLM 的 max_model_len，
# 避免请求超过模型 max_position_embeddings 而启动失败。
LOCAL_MODEL_NATIVE_MAX_MODEL_LEN = {
    "Qwen3-1.7B": 40960,
    "Qwen3-4B": 40960,
    "Qwen3-8B": 40960,
    "Qwen3-14B": 40960,
    "Qwen3-32B": 40960,
    "Qwen3-30B-A3B": 40960,
    "Qwen2.5-1.5B": 131072,
    "Qwen2.5-3B": 32768,
    "Qwen2.5-7B-Instruct": 32768,
    "Llama-3.2-3B-Instruct": 131072,
    # Qwen3.5 系列原生上下文 262144，远超评测所需 81920，无需 YaRN 外推。
    "Qwen3.5-9B": 262144,
    "Qwen3.8-27B": 262144,
    "Qwen3.6-27B": 262144,
    "Qwen3.5-122B-A10B": 262144,
    # 新增判分模型（2507/3.6 系列原生上下文均 >= 131072）；判分脚本会按实际 config 再收敛 max_model_len。
    "Qwen3.6-35B-A3B": 262144,
    "Qwen3-30B-A3B-Thinking-2507": 262144,
    # 跨家族/新架构判分模型；判分脚本按实际 config 收敛到 <=81920。
    "Llama-4-Scout-17B-16E-Instruct": 10485760,
    "Qwen3-Next-80B-A3B-Instruct": 262144,
}
