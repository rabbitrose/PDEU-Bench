"""Reason-action trajectory LLM-as-judge prompt."""

from __future__ import annotations

from typing import Any, Callable


CompactJson = Callable[..., str]
Message = dict[str, str]


def build_messages(
    *,
    case_id: str,
    preference_source: str,
    judge_context: dict[str, Any],
    calls: list[dict[str, Any]],
    compact_json: CompactJson,
    max_context_chars: int,
    max_tool_trajectory_chars: int,
) -> list[Message]:
    schema = {
        "case_id": case_id,
        "trajectory_acc": "boolean，必须等于每个 instantiation_judgements.json_object_preference_following 的逻辑与",
        "overall_instantiation_preference_following": "boolean，必须等于 trajectory_acc",
        "instantiation_judgements": [
            {
                "call_index": 1,
                "iteration": 1,
                "api_name": "string",
                "json_object_preference_following": "boolean，只评价 tool_call.arguments 是否符合偏好和任务硬约束",
                "supporting_evidence": ["说明 arguments 中相关字段为什么符合偏好与任务约束"],
                "violated_preferences": ["如果失败，列出被违反或遗漏的具体偏好/约束"],
                "argument_notes": ["指出哪些参数与偏好相关、如何体现偏好；无关参数不需要强行关联偏好"],
                "confidence": "0 到 1",
            }
        ],
        "failure_summary": ["整体失败原因；如果全部通过可为空数组"],
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的 LLM-as-judge。你只评估 eval_reason 每一步工具调用生成的 JSON object "
                f"tool_call.arguments 是否符合{preference_source}和任务约束。reason 评估没有 first plan 和 update_plan，"
                "不要按 plan/update_plan 打分；也不要评价 API 选择、调用顺序或 sandbox observation 的好坏，"
                "除非 arguments 内容本身编造、遗漏或违反了用户偏好与任务硬约束。"
                "只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请逐个检查 tool_call_trajectory 中每一步的 arguments。"
                "只有每一次工具调用的 json_object_preference_following 都为 true 时，"
                "trajectory_acc 与 overall_instantiation_preference_following 才能为 true。\n\n"
                "判定准则：\n"
                "- 只评价 JSON object 的内容填充，即 tool_call.arguments；不要评价 API 顺序、停止时机或 API 是否最佳。\n"
                f"- arguments 采用偏好相关性原则：只有当参数语义涉及{preference_source}中的隐私、预算、时间、语言、饮食、风格、证据标准、风险偏好或任务硬约束时，才要求体现对应偏好。\n"
                "- 不要求一次工具调用使用全部用户偏好；与某项偏好无关的参数，不能因为未体现该偏好而扣分。\n"
                "- 如果 API 有相关参数且上下文提供了必要偏好信息，arguments 应合理体现；如果 API 没有相关参数或该偏好与本次调用无关，不应强行扣分。\n"
                "- 不能把无依据的个人信息、敏感信息、预算、位置、联系人、语言偏好等写入参数；不能与任务硬约束冲突。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 8000)}\n\n"
                f"评估上下文:\n{compact_json(judge_context, max_context_chars)}\n\n"
                f"tool_call_trajectory:\n{compact_json(calls, max_tool_trajectory_chars)}"
            ),
        },
    ]
