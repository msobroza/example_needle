"""OpenAIChatModel with a fake ``chat.completions.create`` (no network)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from rag_load_test.adapters.openai_chat import OpenAIChatModel
from rag_load_test.contracts import ChatModelPort
from rag_load_test.errors import RagDependencyError
from rag_load_test.settings import RagSettings


class _FakeCompletions:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _response(content: str | None) -> Any:
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://x")


def _chat_raising(error: Exception) -> OpenAIChatModel:
    return OpenAIChatModel(_FakeClient(_FakeCompletions(error=error)), "gpt-test")


async def _expect_dependency_error(chat: OpenAIChatModel) -> RagDependencyError:
    with pytest.raises(RagDependencyError) as info:
        await chat.generate([{"role": "user", "content": "q"}])
    return info.value


async def test_generate_returns_content_and_passes_model_and_temperature():
    completions = _FakeCompletions(result=_response("42"))
    chat = OpenAIChatModel(_FakeClient(completions), "gpt-test", temperature=0.3)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert await chat.generate(messages) == "42"
    assert chat.model_name == "gpt-test"
    assert isinstance(chat, ChatModelPort)
    assert completions.calls == [
        {"model": "gpt-test", "messages": messages, "temperature": 0.3}
    ]


async def test_generate_defaults_to_zero_temperature_and_empty_string_content():
    completions = _FakeCompletions(result=_response(None))
    chat = OpenAIChatModel(_FakeClient(completions), "gpt-test")
    assert await chat.generate([{"role": "user", "content": "q"}]) == ""
    assert completions.calls[0]["temperature"] == 0.0


async def test_timeout_maps_to_timeout_kind():
    err = await _expect_dependency_error(
        _chat_raising(openai.APITimeoutError(request=_request()))
    )
    assert err.kind == "timeout"
    assert err.component == "llm"
    assert err.target == "gpt-test"
    assert isinstance(err.__cause__, openai.APITimeoutError)


async def test_status_error_maps_to_unavailable():
    response = httpx.Response(500, request=_request())
    err = await _expect_dependency_error(
        _chat_raising(openai.APIStatusError("boom", response=response, body=None))
    )
    assert err.kind == "unavailable"
    assert err.component == "llm"
    assert err.target == "gpt-test"
    assert "boom" in err.detail
    assert "500" in err.detail


async def test_connection_error_maps_to_unavailable():
    err = await _expect_dependency_error(
        _chat_raising(openai.APIConnectionError(request=_request()))
    )
    assert err.kind == "unavailable"


async def test_unrelated_errors_propagate_untranslated():
    with pytest.raises(RuntimeError, match="bug"):
        await _chat_raising(RuntimeError("bug")).generate(
            [{"role": "user", "content": "q"}]
        )


def test_from_settings_builds_async_client_with_timeout_and_retries(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # SDK refuses to build without one
    settings = settings_factory(
        openai_model="gpt-x", openai_timeout_s=12.5, openai_max_retries=3
    )
    chat = OpenAIChatModel.from_settings(settings)
    assert chat.model_name == "gpt-x"
    assert isinstance(chat._client, openai.AsyncOpenAI)
    assert chat._client.timeout == 12.5
    assert chat._client.max_retries == 3


def test_importing_openai_chat_does_not_import_openai():
    code = (
        "import sys, rag_load_test.adapters.openai_chat; "
        "assert 'openai' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
