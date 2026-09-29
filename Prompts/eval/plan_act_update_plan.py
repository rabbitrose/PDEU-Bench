"""Update-plan prompt for plan-reason-action-update_plan evaluation."""

from __future__ import annotations

from typing import Any

from .common import CompactJson, ContextJsonLimit, Message, target_api_rule
from .plan_act_system import system_prompt


def build_messages(
    *,
    context: dict[str, Any],
    previous_plan: dict[str, Any],
    action_json: dict[str, Any],
    sandbox_observation: dict[str, Any],
    iteration: int,
    compact_json: CompactJson,
    context_json_limit: ContextJsonLimit,
    max_plan_steps: int,
    max_tool_observation_chars: int,
    system_content: str | None = None,
) -> list[Message]:
    schema = {
        "phase": "update_plan",
        "iteration": iteration,
        "task_status": "in_progress|completed|blocked|failed",
        "plan_version": "previous_plan.plan_version + 1",
        "update_reason": "说明基于工具结果如何更新计划",
        "overall_goal": "string",
        "completion_criteria": ["string"],
        "steps": [
            {
                "step_id": "S1",
                "description": "string",
                "status": "pending|in_progress|completed|blocked|skipped",
                "depends_on": ["step_id"],
                "target_api": target_api_rule(),
                "planned_arguments": "object or null；如果 target_api 不为 null，填写下一次或后续工具调用应使用的偏好相关参数草案；不要编造 observation 未支持的信息",
                "success_criteria": ["string"],
                "evidence": ["引用 sandbox observation 中的关键信息"],
            }
        ],
        "next_step_id": "step_id or null",
        "monitoring": {
            "current_risks": ["string"],
            "user_context_checks": ["string"],
            "completion_readiness": "string",
        },
    }
    return [
        {"role": "system", "content": system_content or system_prompt()},
        {
            "role": "user",
            "content": (
                "现在执行 update_plan。请根据刚刚的 action 和 SandBoxDB observation 更新计划。"
                "必须修改相关 step 的 status/evidence/next_step_id，使计划可监控。"
                f"updated plan 的 steps 最多 {max_plan_steps} 个；如果原计划很长，请压缩合并为关键步骤。"
                "如果工具调用失败，计划中要反映需要重试或改用其他 API。"
                "每个 step 的 target_api 必须优先绑定到 available_api_names 中最合适的真实 API；"
                "只有该 step 确实不需要任何 API 执行时才允许写 null，并要在 description 中体现它是推理/综合/复核类步骤。"
                "如果后续 step 计划调用 API，请更新 planned_arguments，明确保留或修正与用户偏好/任务约束相关的参数，例如预算、时间、地点、语言、隐私、格式、风险偏好等；"
                "不能把 observation 没有支持的信息写成已确认参数。"
                "如果所有成功标准已经满足，可以把 task_status 标为 completed，next_step_id 写 null，并在 evidence/monitoring 中记录最终交付物已经就绪。"
                "后续如果系统继续请求 reason_action，再通过 finish action 给出最终交付物。"
                "只输出 JSON。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 5000)}\n\n"
                f"评估上下文:\n{compact_json(context, context_json_limit(context))}\n\n"
                f"上一版计划:\n{compact_json(previous_plan)}\n\n"
                f"刚刚的 action:\n{compact_json(action_json)}\n\n"
                f"SandBoxDB observation:\n{compact_json(sandbox_observation, max_tool_observation_chars)}"
            ),
        },
    ]
