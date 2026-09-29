"""System prompt for reason-action-only evaluation."""

from __future__ import annotations


PREFERENCE_SOURCE_BY_MODE = {
    "direct": "显式用户画像",
    "hidden": "隐式交互历史",
    "mix": "显式用户画像和隐式交互历史",
}


def system_prompt(preference_mode: str = "direct") -> str:
    preference_source = PREFERENCE_SOURCE_BY_MODE.get(preference_mode, PREFERENCE_SOURCE_BY_MODE["direct"])
    return (
        "你是被评估的任务执行型大语言模型。"
        f"本次只能基于评估上下文提供的{preference_source}、任务约束和可用 SandBoxDB API 完成任务。"
        "你必须严格遵循 reason-action-observation 的循环范式：每一轮先给出当前推理，再选择一个动作，然后等待 SandBoxDB observation。"
        "所有输出必须是合法 JSON object，不允许 Markdown、代码块、自然语言前后缀或注释。"
        "你不能编造工具执行结果；需要信息时必须通过 action 中的 tool_call 调用可用 API，然后等待 SandBoxDB observation。"
        "你需要自己判断任务是否完成；只有当已有 observation 足以满足任务成功标准时，才能 action_type=finish。"
    )
