"""Prompt for lightweight API concurrency probes."""

from __future__ import annotations


def build_messages(probe_id: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "你只输出合法 JSON object，不输出解释。"},
        {"role": "user", "content": f"请返回 JSON：{{\"ok\": true, \"probe_id\": \"{probe_id}\"}}"},
    ]
