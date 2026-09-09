from __future__ import annotations

import time

from rag_load_test.adapters.fake_chat import FakeChatModel


async def test_fake_chat_echoes_last_user_message():
    chat = FakeChatModel()
    answer = await chat.generate(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "what is the vacation policy?"},
        ]
    )
    assert answer == "[fake-llm] what is the vacation policy?"
    assert chat.calls == 1
    assert chat.model_name == "fake-llm"


async def test_fake_chat_honours_latency():
    chat = FakeChatModel(latency_ms=30)
    t0 = time.perf_counter()
    await chat.generate([{"role": "user", "content": "x"}])
    assert (time.perf_counter() - t0) * 1000 >= 25
