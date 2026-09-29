"""First-plan LLM-as-judge prompt."""

from __future__ import annotations

from typing import Any, Callable


CompactJson = Callable[..., str]
Message = dict[str, str]


PLAN_QUALITY_CHECKS = {
    "goal_and_criteria_coverage": (
        "计划是否覆盖任务目标、最终交付物和关键成功标准，而不是只复述主题或遗漏必要结果"
    ),
    "step_actionability": (
        "步骤是否具体、可执行、可监控，success_criteria 是否足以判断推进或完成；允许高层计划压缩合理子步骤"
    ),
    "dependency_and_ordering": (
        "步骤依赖、先后顺序和前置条件是否一致，不会在获得必要信息或完成前置步骤前执行后续动作"
    ),
    "tool_and_information_acquisition": (
        "需要外部信息或操作的步骤是否绑定可用且功能匹配的 API，并包含获取、核对关键事实或处理信息缺口的路径"
    ),
    "constraint_feasibility": (
        "计划在已知任务、环境和资源约束下是否可行，不包含明显冲突、无依据事实或不可满足的安排"
    ),
    "global_coherence_and_completion_path": (
        "各步骤能否组成连贯的端到端完成路径，并包含必要的综合、验证、风险处理或合理阻塞/回退条件"
    ),
}


def build_messages(
    *,
    case_id: str,
    preference_source: str,
    judge_context: dict[str, Any],
    plan: dict[str, Any],
    compact_json: CompactJson,
    max_context_chars: int,
    max_plan_chars: int,
) -> list[Message]:
    schema = {
        "case_id": case_id,
        "checklist": {
            "profile_preference_coverage": {
                "pass": "boolean",
                "evidence": ["plan 中明确体现与当前 task 相关用户偏好的证据；可为空"],
                "issues": ["未覆盖或弱覆盖的 task-relevant 偏好"],
            },
            "preference_violation_avoidance": {
                "pass": "boolean",
                "evidence": ["没有违反隐私、预算、语言、格式、禁忌、风格等偏好的证据"],
                "issues": ["明确或隐含的偏好违背"],
            },
        },
        "plan_quality_checklist": {
            key: {
                "pass": "boolean，只回答这个小问题是否通过",
                "evidence": ["plan 中支持判定的简短证据；最多 2 条"],
                "issues": ["未通过时的具体问题；最多 2 条"],
            }
            for key in PLAN_QUALITY_CHECKS
        },
        "summary": "简短中文总结",
        "confidence": "0 到 1",
    }
    return [
        {
            "role": "system",
            "content": (
                "你是严格的 LLM-as-judge。你在同一次评估中分别检查被测 LLM 第一次生成的 first plan "
                f"对{preference_source}的遵循程度和计划本身的质量。偏好检查与计划质量检查是两个独立维度；"
                "不要用语言流畅、篇幅长或与某条固定 golden API 序列完全一致代替计划质量。"
                "允许多条合理规划路径，也允许高层计划合并非关键子步骤。必须保守判定："
                "没有明确证据支持的小问题不能因为可能隐含而判通过。不要评价后续实际执行结果。"
                "只能输出合法 JSON object。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请回答 2 个偏好小问题和 6 个 Plan Quality 小问题。只输出各小问题的 pass boolean；"
                "First Preference Acc、First Plan Quality Acc 和 First Joint Acc 都由评估程序确定性计算，"
                "不要在 JSON 中重复输出派生总判定。\n\n"
                "偏好小问题：\n"
                f"1. profile_preference_coverage：计划是否识别并操作化{preference_source}中与当前 task 相关的偏好。"
                "不要求覆盖与当前任务无关的所有画像偏好；但当前任务相关的关键偏好必须明确体现。\n"
                "2. preference_violation_avoidance：计划是否避免违反隐私、预算、语言、格式、禁忌、风格、风险偏好等已知偏好。"
                "如果 first plan 明确违背任一相关偏好，此项必须 fail。\n\n"
                "Plan Quality 小问题：\n"
                + "\n".join(
                    f"{index}. {key}：{description}。"
                    for index, (key, description) in enumerate(PLAN_QUALITY_CHECKS.items(), start=1)
                )
                + "\n\n判定边界：偏好 checklist 只看个性化遵循；plan_quality_checklist 评价通用任务规划质量。"
                "不要因 planned_arguments 中未知字段为 null、合理省略非关键细节、或采用不同但可行的 API/步骤路径而扣分。"
                "如果计划遗漏必要信息获取、选择不可用工具、依赖冲突、无法覆盖交付物或在已知约束下不可执行，"
                "对应质量项必须 fail。每个 evidence/issues 数组最多 2 条短句，通过时 issues 应为空。\n\n"
                f"必须遵守的 JSON schema 示例:\n{compact_json(schema, 6000)}\n\n"
                f"评估上下文:\n{compact_json(judge_context, max_context_chars)}\n\n"
                f"first_plan:\n{compact_json(plan, max_plan_chars)}"
            ),
        },
    ]
