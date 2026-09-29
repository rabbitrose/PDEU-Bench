#!/usr/bin/env python3
"""Regression tests for atomic plan-quality and joint judge metrics."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import judge


def first_plan_response(*, failed_quality_key: str | None = None) -> dict:
    return {
        "case_id": "user_001/task_easy/pass_1",
        "first_plan_preference_following": False,
        "first_plan_quality_correct": True,
        "first_plan_joint_correct": True,
        "checklist": {
            key: {"pass": True, "evidence": ["有证据"], "issues": []}
            for key in judge.PLAN_CHECKLIST_KEYS
        },
        "plan_quality_checklist": {
            key: {
                "pass": key != failed_quality_key,
                "evidence": ["有证据"] if key != failed_quality_key else [],
                "issues": [] if key != failed_quality_key else ["缺少完成路径"],
            }
            for key in judge.PLAN_QUALITY_CHECKLIST_KEYS
        },
        "summary": "测试",
        "confidence": 1.0,
    }


def update_response(*, failed_plan_quality_key: str | None = None) -> dict:
    item = {
        "update_index": 1,
        "iteration": 1,
        "update_correct": True,
        "update_joint_correct": True,
        **{key: True for key in judge.UPDATE_QUALITY_KEYS},
        **{key: key != failed_plan_quality_key for key in judge.UPDATE_PLAN_QUALITY_KEYS},
        **{key: True for key in judge.UPDATE_PREFERENCE_KEYS},
        "supporting_evidence": ["状态更新正确"],
        "quality_issues": [],
        "plan_quality_issues": [] if failed_plan_quality_key is None else ["剩余计划不可行"],
        "preference_issues": [],
        "violated_preferences": [],
        "issues": [],
        "confidence": 1.0,
    }
    return {
        "case_id": "user_001/task_easy/pass_1",
        "overall_update_plan_correct": True,
        "overall_update_plan_quality_correct": True,
        "overall_update_joint_correct": True,
        "update_plan_judgements": [item],
        "failure_summary": [],
        "confidence": 1.0,
    }


class JudgePlanQualityTest(unittest.TestCase):
    def test_first_plan_derives_quality_and_joint_from_atomic_checks(self) -> None:
        failed_key = judge.PLAN_QUALITY_CHECKLIST_KEYS[-1]
        raw = first_plan_response(failed_quality_key=failed_key)

        self.assertEqual(judge.validate_plan_judge(raw), [])
        parsed = judge.normalize_plan_judge(raw)

        self.assertTrue(parsed["first_plan_preference_following"])
        self.assertFalse(parsed["first_plan_quality_correct"])
        self.assertFalse(parsed["first_plan_joint_correct"])
        self.assertEqual(parsed["first_plan_quality_score"], round(5 / 6, 6))

    def test_update_derives_three_layers_and_joint_from_atomic_checks(self) -> None:
        failed_key = judge.UPDATE_PLAN_QUALITY_KEYS[0]
        raw = update_response(failed_plan_quality_key=failed_key)

        self.assertEqual(judge.validate_update_judge(1)(raw), [])
        parsed = judge.normalize_update_judge(raw, 1)
        item = parsed["update_plan_judgements"][0]

        self.assertTrue(item["update_quality_correct"])
        self.assertFalse(item["update_plan_quality_correct"])
        self.assertTrue(item["update_preference_correct"])
        self.assertFalse(item["update_joint_correct"])
        self.assertFalse(parsed["overall_update_plan_correct"])
        self.assertFalse(parsed["overall_update_joint_correct"])

    def test_quality_metrics_reuse_first_and_update_requests(self) -> None:
        record = {
            "run_id": "test_run",
            "user_id": "user_001",
            "task_id": "task_easy",
            "pass_n": 1,
            "pass_index": 1,
            "task_title": "测试任务",
            "status": "completed",
            "preference_injection_mode": "direct",
            "trajectory": [
                {"phase": "initial_plan", "parsed": {"steps": [], "next_step_id": None}},
                {
                    "phase": "reason_action",
                    "iteration": 1,
                    "parsed": {"action_type": "finish", "completion_assessment": "完成"},
                },
                {
                    "phase": "sandbox_observation",
                    "iteration": 1,
                    "observation": {"sandbox_status": "success", "response_body": {"done": True}},
                },
                {
                    "phase": "update_plan",
                    "iteration": 1,
                    "parsed": {"task_status": "completed", "steps": [], "next_step_id": None},
                },
            ],
            "tool_calls": [],
        }
        purposes: list[str] = []

        def fake_request(_client, *, purpose, **_kwargs):
            purposes.append(purpose)
            if purpose.endswith("/first_plan"):
                return first_plan_response(), []
            if "/update_plan/" in purpose:
                return update_response(), []
            raise AssertionError(f"出现非预期 judge 请求: {purpose}")

        with patch.object(judge, "request_structured_json", side_effect=fake_request):
            result = judge.judge_case(record, None, logging.getLogger("test"))

        self.assertEqual(len(purposes), 2)
        self.assertTrue(any(purpose.endswith("/first_plan") for purpose in purposes))
        self.assertTrue(any("/update_plan/" in purpose for purpose in purposes))
        self.assertTrue(judge.plan_joint_pass(result))
        self.assertTrue(judge.update_plan_quality_pass(result))
        self.assertTrue(judge.update_pass(result))

    def test_aggregate_and_html_expose_quality_and_joint_metrics(self) -> None:
        plan = judge.normalize_plan_judge(first_plan_response())
        update = judge.normalize_update_judge(update_response(), 1)
        result = {
            "user_id": "user_001",
            "task_id": "task_easy",
            "pass_n": 1,
            "pass_index": 1,
            "eval_status": "completed",
            "tool_call_count": 0,
            "plan_judge": {"parsed": plan},
            "instantiation_judge": {"parsed": judge.deterministic_empty_trajectory_judge("case")},
            "update_judge": {"parsed": update},
            "judge_errors": [],
        }
        judge.merge_update_result_into_trajectory(result)

        aggregate = judge.aggregate_results([result])
        self.assertEqual(aggregate["first_plan_quality_correct_ratio"], 100.0)
        self.assertEqual(aggregate["first_plan_joint_correct_ratio"], 100.0)
        self.assertEqual(aggregate["update_plan_quality_correct_ratio"], 100.0)
        self.assertEqual(aggregate["update_joint_correct_ratio"], 100.0)

        report = judge.render_html_report(
            run_id="test",
            target_gen_model="test-model",
            preference_mode="direct",
            pass_n=1,
            selected_eval_run_ids=["test_run"],
            judge_model={"name": "deepseek-v4-pro", "base_url": "https://api.deepseek.com"},
            eval_result_dir=Path("runresult/test"),
            detail_path=Path("judgeresult/details.jsonl"),
            summary_path=Path("judgeresult/summary.json"),
            results=[result],
            aggregate=aggregate,
        )
        self.assertIn("First Plan Quality Acc", report)
        self.assertIn("First Joint Acc", report)
        self.assertIn("Update Plan Quality Acc", report)
        self.assertIn("Update Joint Acc", report)


if __name__ == "__main__":
    unittest.main()
