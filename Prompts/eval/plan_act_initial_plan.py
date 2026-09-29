"""Initial-plan prompt for plan-reason-action-update_plan evaluation."""

from __future__ import annotations

from typing import Any

from .common import CompactJson, ContextJsonLimit, Message, target_api_rule
from .plan_act_system import system_prompt


def build_messages(
    *,
    context: dict[str, Any],
    compact_json: CompactJson,
    context_json_limit: ContextJsonLimit,
    max_plan_steps: int,
    system_content: str | None = None,
) -> list[Message]:
    schema = {
        "phase": "initial_plan",
        "task_status": "in_progress",
        "plan_version": 1,
        "overall_goal": "string",
        "completion_criteria": ["string"],
        "steps": [
            {
                "step_id": "S1",
                "description": "string",
                "status": "pending|in_progress|completed|blocked|skipped",
                "depends_on": ["step_id"],
                "target_api": target_api_rule(),
                "planned_arguments": "object or null；如果 target_api 不为 null，填写已知且与用户偏好/任务约束相关的参数草案；不要编造未知信息",
                "success_criteria": ["string"],
                "evidence": [],
            }
        ],
        "next_step_id": "S1",
        "monitoring": {
            "current_risks": ["string"],
            "user_context_checks": ["string"],
        },
    }
    return [
        {"role": "system", "content": system_content or system_prompt()},
        {
            "role": "user",
            "content": (
                "请先生成初始计划。计划要结构化、步骤状态可修改可监控，并体现评估上下文中的用户相关信息如何影响任务执行。"
                f"steps 最多 {max_plan_steps} 个，只保留关键可执行步骤，不要穷举所有子步骤。"
                "每个 step 的 target_api 必须优先绑定到 available_api_names 中最合适的真实 API；"
                "只有该 step 确实不需要任何 API 执行时才允许写 null，并要在 description 中体现它是推理/综合/复核类步骤。"
                "如果 step 计划调用 API，请在 planned_arguments 中写出当前已知的关键参数草案，尤其是预算、时间、地点、语言、隐私、格式、风险偏好等会影响工具输入的约束；"
                "未知参数写 null 或省略，不要编造。"
                "不要执行动作，只输出 JSON。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 5000)}\n\n"
                f"评估上下文:\n{compact_json(context, context_json_limit(context))}"
            ),
        },
    ]
