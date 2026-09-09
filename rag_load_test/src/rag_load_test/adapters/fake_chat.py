"""A deterministic ChatModelPort for tests and LLM-free load tests."""

from __future__ import annotations

import asyncio


class FakeChatModel:
    """Echo the last user message; optionally sleep to imitate generation latency.

    Example::

        chat = FakeChatModel(latency_ms=50)
        answer = await chat.generate([{"role": "user", "content": "hi"}])
    """

    model_name = "fake-llm"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms = latency_ms
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        return f"[fake-llm] {_last_user_content(messages)[:200]}"


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""
