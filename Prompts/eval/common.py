"""Shared non-prompt helpers for eval prompt builders."""

from __future__ import annotations

from typing import Any, Callable


CompactJson = Callable[..., str]
ContextJsonLimit = Callable[[dict[str, Any]], int]
Message = dict[str, str]


def target_api_rule() -> str:
    return (
        "必须是评估上下文 available_api_names 中的一个真实 API 名称；"
        "只有纯推理、综合、人工复核、用户信息核对、最终整理等不需要工具调用的步骤才能为 null。"
        "需要查找、检索、比较、创建、发送、安排、翻译、计算或获取外部/沙盒信息的步骤不得为 null。"
    )
