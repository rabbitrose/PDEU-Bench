"""Tool instantiation LLM-as-judge prompt."""

from __future__ import annotations

from typing import Any, Callable


CompactJson = Callable[..., str]
Message = dict[str, str]


def build_messages(
    *,
    case_id: str,
    preference_source: str,
    judge_context: dict[str, Any],
    plan: dict[str, Any] | None,
    calls: list[dict[str, Any]],
    compact_json: CompactJson,
    max_context_chars: int,
    max_plan_chars: int,
    max_tool_input_chars: int,
) -> list[Message]:
    schema = {
        "case_id": case_id,
        "overall_instantiation_preference_following": "boolean，且必须等于每个 instantiation_judgements.json_object_preference_following 的逻辑与",
        "instantiation_judgements": [
            {
                "call_index": 1,
                "iteration": 1,
                "api_name": "string",
                "json_object_preference_following": "boolean",
                "violated_preferences": ["如果失败，列出被违反或遗漏的具体偏好"],
                "supporting_evidence": ["说明 arguments 中相关字段为什么符合或不符合偏好与任务约束"],
                "argument_notes": ["逐项指出哪些参数与偏好相关、如何体现偏好；无关参数不需要强行关联偏好"],
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
                "你是严格的 LLM-as-judge。你只评估被测 LLM 在每一次工具调用中生成的 JSON object "
                f"tool_call.arguments 的内容填充是否符合{preference_source}和任务约束。"
                "不要评估 sandbox observation 的好坏，不要因为工具返回失败而惩罚 arguments；"
                "也不要评价 API 选择是否正确，除非 arguments 内容本身编造、遗漏或违反了偏好。"
                "不要求每次工具调用、每个 API 或每个参数都覆盖全部用户偏好。必须逐个工具调用给出判定。"
                "只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请逐个检查 tool_call_inputs 中的 arguments。只有每一次工具调用的 json_object_preference_following=true 时，"
                "overall_instantiation_preference_following 才能为 true。\n\n"
                "判定准则：\n"
                f"- 只评价 JSON object 的内容填充，即 tool_call.arguments；不要评价 API 顺序或 API 是否是最佳选择。\n"
                f"- 偏好相关性原则：只有当参数语义涉及{preference_source}中的隐私、预算、时间、地点、语言、饮食、风格、证据标准、风险偏好或任务硬约束时，才要求该参数考虑对应偏好。\n"
                "- 不要求一次工具调用使用全部用户偏好；与某项偏好无关的参数，不能因为未体现该偏好而扣分。\n"
                "- 如果 API 有相关参数且上下文提供了必要偏好信息，arguments 应合理体现；如果该偏好与本次参数无关，不应强行扣分。\n"
                "- 不能把无依据的个人信息、敏感信息或和任务冲突的假设放入参数。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"评估上下文:\n{compact_json(judge_context, max_context_chars)}\n\n"
                f"first_plan:\n{compact_json(plan or {}, max_plan_chars)}\n\n"
                f"tool_call_inputs:\n{compact_json(calls, max_tool_input_chars)}"
            ),
        },
    ]
