#这里存储的是许多可以调用访问用于推理的本地的模型路径
#每个模型有自己的缩写或者代指叫法：
#当说到qwen的4b模型，说的就是Qwen3-4B-Instruct-2507
#当说到deepseek的7b模型，说的就是DeepSeek-R1-Distill-Qwen-7B
#当说到qwen的7b模型，说的就是Qwen2.5-7B-Instruct
#当说到llama的8b模型，说的就是Meta-Llama-3.1-8B-Instruct

USEABLE_LOCAL_MODEL_PATHS = {
    "Qwen3-4B-Instruct-2507":"/root/eb-public/huggingface-models/Qwen/Qwen3-4B-Instruct-2507",
    "DeepSeek-R1-Distill-Qwen-7B":"/root/eb-public/huggingface-models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen2.5-7B-Instruct":"/mnt/slow/hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "Meta-Llama-3.1-8B-Instruct":"/mnt/slow/hf_home/hub/modelscope--LLM-Research--Meta-Llama-3.1-8B-Instruct"
}
