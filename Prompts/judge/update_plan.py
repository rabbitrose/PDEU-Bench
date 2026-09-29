"""Update-plan LLM-as-judge prompt."""

from __future__ import annotations

from typing import Any, Callable


CompactJson = Callable[..., str]
Message = dict[str, str]


UPDATED_PLAN_QUALITY_CHECKS = {
    "remaining_goal_and_criteria_coverage": (
        "updated_plan 是否保留全部尚未满足的任务目标、交付物和成功标准，并正确移除或完成已有证据支持的部分"
    ),
    "replan_decision_correctness": (
        "基于 observation 选择继续、重试、更换 API、回退、完成或阻塞是否正确，既不无故重规划也不忽略必须重规划的反馈"
    ),
    "remaining_steps_actionability": (
        "剩余步骤是否具体、可执行、可监控，所需 API 可用且 planned_arguments 不依赖 observation 未支持的事实"
    ),
    "information_acquisition_and_verification": (
        "更新后的计划是否继续获取和核对完成任务所缺的关键信息，并对冲突、失败或信息缺口设置合理处理路径"
    ),
    "constraint_feasibility_after_update": (
        "剩余计划是否同时符合新 observation、任务硬约束和现实资源限制，不包含局部或全局不可行安排"
    ),
    "global_coherence_and_completion_path": (
        "更新后的全部剩余步骤能否形成连贯的端到端完成路径；若目标已不可达，是否正确标记 blocked/failed 而非伪造完成"
    ),
}


def build_messages(
    *,
    case_id: str,
    judge_context: dict[str, Any],
    updates: list[dict[str, Any]],
    compact_json: CompactJson,
    max_context_chars: int,
    max_update_judge_chars: int,
) -> list[Message]:
    schema = {
        "case_id": case_id,
        "update_plan_judgements": [
            {
                "update_index": 1,
                "iteration": 1,
                "observation_integration": "boolean，是否准确吸收 sandbox_observation 的真实结果、错误、缺失信息和 next_suggested_api_calls",
                "status_and_evidence_update": "boolean，是否正确更新 step status、evidence、task_status、completion_criteria 进展",
                "next_step_consistency": "boolean，next_step_id 和剩余步骤是否与 previous_plan、action、observation 一致",
                "no_fabrication_or_contradiction": "boolean，是否没有虚构 observation、错误完成未完成事项、删除关键约束或与证据矛盾",
                **{key: f"boolean，{description}" for key, description in UPDATED_PLAN_QUALITY_CHECKS.items()},
                "preference_retention": "boolean，是否保留 previous_plan 中仍然适用的用户偏好、任务硬约束和个性化执行原则",
                "preference_aware_replanning": "boolean，重规划、重试或更换后续 API/步骤时是否继续优先满足用户偏好",
                "planned_arguments_preference_following": "boolean，后续 planned_arguments 是否保留或修正预算、时间、地点、语言、隐私、格式、风险偏好等偏好相关参数",
                "no_preference_violation": "boolean，是否没有新增、保留或升级任何违背用户偏好的步骤、参数、假设或交付物方向",
                "supporting_evidence": ["说明哪些更新准确"],
                "quality_issues": ["如果状态转移质量层失败，列出具体错误、遗漏或矛盾"],
                "plan_quality_issues": ["如果更新后计划质量层失败，列出具体不可行、遗漏或不连贯之处"],
                "preference_issues": ["如果偏好层失败，列出具体偏好遗漏、偏离或违背"],
                "violated_preferences": ["被违反、遗漏或弱化的具体用户偏好；无则空数组"],
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
                "你是严格的 LLM-as-judge。你只评估被测 LLM 每一次 update_plan 操作是否准确无误。"
                "评估对象是 previous_plan + reason_action + sandbox_observation 到 updated_plan 的状态转移。"
                "在同一次请求中分别检查状态转移质量、更新后剩余计划质量和个性化偏好遵循。"
                "状态转移质量的 4 个问题不要追究 previous_plan 已有缺陷；但更新后 Plan Quality 的 6 个问题必须评价"
                "updated_plan 当前完整的剩余计划，因此任何会使当前计划不完整、不可行或无法完成目标的继承缺陷都应在"
                "Plan Quality 层失败。若新 observation 使原路径失效，还必须检查本轮是否正确重规划。"
                "不要要求与固定 golden API 序列完全一致，允许多条合理且可执行的替代路径。"
                "必须保守判定：如果 updated_plan 声称完成了 observation 没有支持的事项、遗漏关键失败/缺失信息、"
                "next_step/status/evidence 与观察结果矛盾，或在重规划中丢失/违背用户偏好，就判失败。"
                "只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请逐个检查 update_plan_events。每轮只输出 14 个原子 boolean：4 个状态转移质量问题、"
                "6 个更新后 Plan Quality 问题和 4 个偏好问题。各层 Acc、Joint Acc、update_correct 及任务级逻辑与"
                "均由评估程序确定性计算，不要在 JSON 中重复输出派生总判定。\n\n"
                "状态转移质量问题：\n"
                "- observation_integration：必须准确使用 sandbox_observation 中的成功/失败、response_body、missing_information、feedback、next_suggested_api_calls。\n"
                "- status_and_evidence_update：必须把已执行步骤、证据、阻塞、风险和 task_status 更新到合理状态，不能空泛复制旧计划。\n"
                "- next_step_consistency：next_step_id 必须指向合理的下一步；不能跳过必要步骤，也不能重复已经完成且无需重试的步骤。\n"
                "- no_fabrication_or_contradiction：不能虚构工具没返回的信息，不能把失败工具当成功，不能删除任务硬约束或用户偏好。\n\n"
                "更新后 Plan Quality 问题：\n"
                + "\n".join(
                    f"- {key}：{description}。"
                    for key, description in UPDATED_PLAN_QUALITY_CHECKS.items()
                )
                + "\n\n"
                "用户偏好判定准则：\n"
                "- preference_retention：必须保留 previous_plan 中仍然适用的用户偏好、任务硬约束和个性化执行原则，例如预算、地点、时间、语言、隐私、价值观、风险偏好、输出格式等。\n"
                "- preference_aware_replanning：如果 observation 迫使计划改变、重试或换 API，必须优先选择符合用户偏好的替代方案；若无法满足，应显式记录风险或阻塞。\n"
                "- planned_arguments_preference_following：后续 planned_arguments 必须继续体现可操作的偏好约束；未知信息可以写 null 或省略，但不能改成违背偏好的参数。\n"
                "- no_preference_violation：不能新增或保留会明显违背用户偏好的步骤、参数、假设或交付物方向；不能因为工具失败就弱化隐私、预算、价值观等硬约束。\n"
                "如果某轮 update 没有新的偏好相关内容，只要它没有丢失、削弱或违背已有偏好，可判偏好层通过。\n\n"
                "每个 supporting_evidence/quality_issues/plan_quality_issues/preference_issues/violated_preferences "
                "数组最多 2 条短句；相应层通过时问题数组应为空，以控制输出 token。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 7000)}\n\n"
                f"评估上下文:\n{compact_json(judge_context, max_context_chars)}\n\n"
                f"update_plan_events:\n{compact_json(updates, max_update_judge_chars)}"
            ),
        },
    ]
