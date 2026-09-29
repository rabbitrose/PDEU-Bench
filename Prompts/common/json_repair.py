"""Prompt for repairing malformed JSON model outputs."""

from __future__ import annotations

from typing import Callable


CompactJson = Callable[..., str]
Message = dict[str, str]


def build_messages(
    *,
    original_messages: list[Message],
    bad_content: str,
    error_message: str,
    compact_json: CompactJson,
    preserve_semantics_instruction: str,
) -> list[Message]:
    return [
        {
            "role": "system",
            "content": "你是 JSON 修复器。只能输出修复后的 JSON object，不能输出解释、Markdown 或代码块。",
        },
        {
            "role": "user",
            "content": (
                "下面是原始任务提示、模型输出和错误信息。"
                f"{preserve_semantics_instruction}\n\n"
                f"原始提示:\n{compact_json(original_messages, 8000)}\n\n"
                f"错误信息:\n{error_message}\n\n"
                f"待修复输出:\n{bad_content}"
            ),
        },
    ]
