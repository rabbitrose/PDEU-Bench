#!/usr/bin/env python3
"""Regression tests for local vLLM hidden evaluation entrypoints."""

from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

import eval_local_hidden
import judge_local_hidden


class LocalHiddenEvalTest(unittest.TestCase):
    def test_legacy_mid_maps_to_current_hard_dataset(self) -> None:
        self.assertEqual(eval_local_hidden.normalize_local_difficulty("mid"), ("mid", "hard"))
        self.assertEqual(judge_local_hidden.normalize_local_difficulty("mid"), ("mid", "hard"))

    def test_runtime_config_uses_existing_local_model_and_free_gpu_memory(self) -> None:
        with mock.patch.object(eval_local_hidden, "query_gpu_memory_mib", return_value=(24576, 24124)):
            config = eval_local_hidden.resolve_runtime_config()

        self.assertEqual(config["model_name"], "Qwen3-4B-Instruct-2507")
        self.assertTrue(config["model_path"].endswith("/Qwen/Qwen3-4B-Instruct-2507"))
        self.assertEqual(config["max_model_len"], 65_536)
        self.assertEqual(config["replica_count"], 2)
        self.assertEqual([item["gpu_ids"] for item in config["replicas"]], ["0", "1"])
        self.assertEqual([item["port"] for item in config["replicas"]], [8100, 8101])
        for replica in config["replicas"]:
            self.assertLessEqual(replica["gpu_memory_utilization"], 0.94)
            self.assertGreater(replica["gpu_memory_utilization"], 0)
            self.assertEqual(replica["max_num_seqs"], 2)
        self.assertEqual(config["eval_concurrency"], 4)

    def test_vllm_command_uses_local_path_and_batching_controls(self) -> None:
        with mock.patch.object(eval_local_hidden, "query_gpu_memory_mib", return_value=(24576, 24124)):
            config = eval_local_hidden.resolve_runtime_config()
        config["vllm_executable"] = "/bin/true"
        replica_config = {**config, **config["replicas"][0]}
        command = eval_local_hidden.build_vllm_command(replica_config)

        self.assertEqual(command[:2], ["/bin/true", "serve"])
        self.assertEqual(command[2], config["model_path"])
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-chunked-prefill", command)
        self.assertIn("--safetensors-load-strategy", command)
        self.assertIn("prefetch", command)
        self.assertEqual(command[command.index("--max-model-len") + 1], "65536")
        self.assertNotIn("huggingface.co", " ".join(command))

    def test_users_are_sticky_and_balanced_across_replicas(self) -> None:
        client = eval_local_hidden.StickyReplicaLLMClient(
            api_key="local",
            base_urls=["http://127.0.0.1:8100/v1", "http://127.0.0.1:8101/v1"],
            model="test-model",
            logger=mock.Mock(),
        )
        self.assertEqual(client.client_index_for_purpose("user_001/task_hard/initial_plan"), 1)
        self.assertEqual(client.client_index_for_purpose("user_001/task_hard/update_plan/1"), 1)
        self.assertEqual(client.client_index_for_purpose("user_002/task_hard/initial_plan"), 0)

    def test_smoke_tool_call_override_precedes_dataset_metadata(self) -> None:
        core = eval_local_hidden.eval_core
        old_override = core.TASK_MAX_TOOL_CALLS_OVERRIDE
        try:
            core.TASK_MAX_TOOL_CALLS_OVERRIDE = 2
            actual = core.max_tool_calls_for_task_doc(
                {"task": {"difficulty_metadata": {"eval_max_tool_calls": 15}}},
                "task_hard",
            )
        finally:
            core.TASK_MAX_TOOL_CALLS_OVERRIDE = old_override
        self.assertEqual(actual, 2)

    def test_configure_eval_records_local_runtime_without_api_key(self) -> None:
        with mock.patch.object(eval_local_hidden, "query_gpu_memory_mib", return_value=(24576, 24124)):
            config = eval_local_hidden.resolve_runtime_config()
        old_config = copy.deepcopy(eval_local_hidden.eval_core.LOCAL_INFERENCE_CONFIG)
        old_filter = eval_local_hidden.eval_core.USER_ID_FILTER
        old_task_filter = eval_local_hidden.eval_core.TASK_ID_FILTER
        old_timeout = eval_local_hidden.eval_core.REQUEST_TIMEOUT_SECONDS
        old_factory = eval_local_hidden.eval_core.GENERATION_CLIENT_FACTORY
        try:
            requested, normalized = eval_local_hidden.configure_eval(config)
            recorded = eval_local_hidden.eval_core.LOCAL_INFERENCE_CONFIG
            configured_timeout = eval_local_hidden.eval_core.REQUEST_TIMEOUT_SECONDS
        finally:
            eval_local_hidden.eval_core.LOCAL_INFERENCE_CONFIG = old_config
            eval_local_hidden.eval_core.USER_ID_FILTER = old_filter
            eval_local_hidden.eval_core.TASK_ID_FILTER = old_task_filter
            eval_local_hidden.eval_core.REQUEST_TIMEOUT_SECONDS = old_timeout
            eval_local_hidden.eval_core.GENERATION_CLIENT_FACTORY = old_factory

        self.assertEqual((requested, normalized), ("mid", "hard"))
        self.assertIsInstance(recorded, dict)
        self.assertNotIn("api_key", recorded)
        self.assertEqual(recorded["normalized_task_difficulty"], "hard")
        self.assertTrue(recorded["offline_model_loading"])
        self.assertEqual(recorded["sandbox_e5_device"], "cpu")
        self.assertEqual(configured_timeout, 900)

    def test_judge_proxy_cleanup_removes_both_cases(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HTTP_PROXY": "http://127.0.0.1:7890", "all_proxy": "socks5://127.0.0.1:7890"},
            clear=False,
        ):
            judge_local_hidden.clear_proxy_environment()
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("all_proxy", os.environ)

    def test_judge_uses_local_result_model_and_qingyun_api(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(judge_local_hidden.judge, "main") as judge_main,
        ):
            judge_local_hidden.main()
            self.assertEqual(judge_local_hidden.judge.TARGET_GEN_LLM, "Qwen3-4B-Instruct-2507")
            self.assertEqual(judge_local_hidden.judge.TASK_DIFFICULTY, "hard")
            self.assertEqual(os.environ["PPEVAL_JUDGE_MODEL"], "deepseek-v4-flash")
            self.assertEqual(os.environ["PPEVAL_JUDGE_URL"], "https://api.qingyuntop.top/v1")
            judge_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
