"""Reason-action prompt for reason-action-only evaluation."""

from __future__ import annotations

from typing import Any

from .common import CompactJson, Message
from .reason_action_system import system_prompt


def build_messages(
    *,
    context: dict[str, Any],
    trajectory_summary: list[dict[str, Any]],
    iteration: int,
    compact_json: CompactJson,
    system_content: str | None = None,
) -> list[Message]:
    schema = {
        "phase": "reason",
        "iteration": iteration,
        "task_status": "in_progress|completed|blocked|failed",
        "reason": "说明当前观察、任务目标、用户偏好、缺口信息和为什么选择这个动作",
        "preference_alignment": ["说明本轮动作如何满足或避免违反用户偏好"],
        "action_type": "tool_call|finish",
        "tool_call": {
            "name": "必须是 available_api_names 中的 API 名称；finish 时为 null",
            "arguments": {"must_match": "对应 API 参数定义"},
        },
        "expected_observation": "string or null",
        "completion_assessment": {
            "is_complete": False,
            "confidence": 0.0,
            "satisfied_criteria": ["string"],
            "remaining_work": ["string"],
        },
        "final_answer": None,
    }
    return [
        {"role": "system", "content": system_content or system_prompt()},
        {
            "role": "user",
            "content": (
                "现在执行 reason-action-observation 循环中的 reason-action 阶段。只能选择一个动作："
                "要么调用一个 SandBoxDB 工具，要么在确认任务完成后 finish。"
                "每一轮都必须基于评估上下文和最近执行轨迹重新推理，不要输出 plan 或 update_plan。"
                "如果调用工具，tool_call.arguments 必须是 object 且符合 API 参数定义。"
                "如果 finish，final_answer 必须总结已完成的交付物、关键证据、偏好满足情况和未解决风险。"
                "只输出 JSON。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 5000)}\n\n"
                f"评估上下文:\n{compact_json(context)}\n\n"
                f"最近执行轨迹:\n{compact_json(trajectory_summary, 9000)}"
            ),
        },
    ]
