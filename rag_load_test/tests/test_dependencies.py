"""Settings -> adapter wiring. Construction only: no weights, no network."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from rag_load_test.adapters.chroma_store import ChromaVectorStore
from rag_load_test.adapters.local_embedder import SentenceTransformerEmbedder
from rag_load_test.adapters.local_reranker import CrossEncoderReranker
from rag_load_test.adapters.openai_chat import OpenAIChatModel
from rag_load_test.adapters.ovms_embedder import OvmsEmbedder
from rag_load_test.adapters.ovms_reranker import OvmsReranker
from rag_load_test.errors import RagConfigError
from rag_load_test.settings import RagSettings
from rag_load_test.testing import FakeChatModel, FakeEmbedder, FakeReranker
from rag_load_test.workflow.dependencies import (
    build_chat_model,
    build_dependencies,
    build_embedder,
    build_reranker,
    build_vector_store,
)
from rag_load_test.workflow.nodes import RagDependencies

SettingsFactory = Callable[..., RagSettings]


def _unique_collection() -> str:
    # EphemeralClient state is process-wide, so every test gets its own collection.
    return f"deps-{uuid.uuid4().hex[:8]}"


def _fake_settings(settings_factory: SettingsFactory, **overrides: Any) -> RagSettings:
    base: dict[str, Any] = {
        "embedder_backend": "fake",
        "reranker_backend": "fake",
        "llm_backend": "fake",
        "chroma_path": ":memory:",
        "chroma_collection": _unique_collection(),
    }
    return settings_factory(**{**base, **overrides})


def _no_network_client() -> httpx.AsyncClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call: {request.method} {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(refuse))


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)


async def test_fake_backends_build_fake_ports_and_chroma_store(
    settings_factory: SettingsFactory, executor: Executor
) -> None:
    settings = _fake_settings(settings_factory, top_k_retrieve=7, top_k_rerank=2)

    deps = build_dependencies(
        settings, http_client=_no_network_client(), executor=executor
    )

    assert isinstance(deps, RagDependencies)
    assert isinstance(deps.embedder, FakeEmbedder)
    assert isinstance(deps.reranker, FakeReranker)
    assert isinstance(deps.vector_store, ChromaVectorStore)
    assert isinstance(deps.chat_model, FakeChatModel)
    assert (deps.top_k_retrieve, deps.top_k_rerank) == (7, 2)
    assert deps.vector_store.embedder_model() == deps.embedder.model_name
    assert await deps.vector_store.count() == 0


def test_reranker_none_yields_no_reranker(
    settings_factory: SettingsFactory, executor: Executor
) -> None:
    settings = _fake_settings(settings_factory, reranker_backend="none")
    client = _no_network_client()

    assert (
        build_reranker(
            settings, http_client=client, executor=executor, token_provider=None
        )
        is None
    )
    assert (
        build_dependencies(settings, http_client=client, executor=executor).reranker
        is None
    )


def test_ovms_backends_construct_without_network(
    settings_factory: SettingsFactory, executor: Executor
) -> None:
    settings = _fake_settings(
        settings_factory,
        embedder_backend="ovms",
        reranker_backend="ovms",
        ovms_embeddings_url="http://emb:8002/",
        ovms_rerank_url="http://rr:8001",
    )

    deps = build_dependencies(
        settings, http_client=_no_network_client(), executor=executor
    )

    assert isinstance(deps.embedder, OvmsEmbedder)
    assert deps.embedder.model_name == "ovms:bge-small-en-v1.5"
    assert deps.embedder.embeddings_url == "http://emb:8002/v3/embeddings"
    assert isinstance(deps.reranker, OvmsReranker)
    assert deps.reranker.model_name == "ovms:bge-reranker-base"
    assert deps.reranker.rerank_url == "http://rr:8001/v3/rerank"
    # The store records the OVMS embedder so /readyz can flag an index mismatch.
    assert deps.vector_store.embedder_model() == "ovms:bge-small-en-v1.5"


async def test_ovms_embedder_sends_configured_bearer_token(
    settings_factory: SettingsFactory, executor: Executor
) -> None:
    seen_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("Authorization"))
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
        )

    settings = _fake_settings(
        settings_factory,
        embedder_backend="ovms",
        ovms_auth="static",
        ovms_bearer_token="tok",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        deps = build_dependencies(settings, http_client=client, executor=executor)
        assert await deps.embedder.embed_query("hi") == [0.1, 0.2]

    assert seen_headers == ["Bearer tok"]


def test_openai_backend_constructs_client_only(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    settings = _fake_settings(
        settings_factory, llm_backend="openai", openai_model="gpt-4o-mini"
    )

    chat = build_chat_model(settings)

    assert isinstance(chat, OpenAIChatModel)
    assert chat.model_name == "gpt-4o-mini"


def test_fake_chat_model_gets_configured_latency(
    settings_factory: SettingsFactory,
) -> None:
    chat = build_chat_model(_fake_settings(settings_factory, fake_llm_latency_ms=12.5))

    assert isinstance(chat, FakeChatModel)
    assert chat.latency_ms == 12.5


def test_local_backends_delegate_to_from_pretrained(
    settings_factory: SettingsFactory,
    executor: Executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, str, Executor]] = []

    def fake_embedder_load(
        cls: type, model_name: str, pool: Executor, **_: Any
    ) -> FakeEmbedder:
        loads.append(("embedder", model_name, pool))
        return FakeEmbedder()

    def fake_reranker_load(
        cls: type, model_name: str, pool: Executor, **_: Any
    ) -> FakeReranker:
        loads.append(("reranker", model_name, pool))
        return FakeReranker()

    monkeypatch.setattr(
        SentenceTransformerEmbedder, "from_pretrained", classmethod(fake_embedder_load)
    )
    monkeypatch.setattr(
        CrossEncoderReranker, "from_pretrained", classmethod(fake_reranker_load)
    )
    settings = _fake_settings(
        settings_factory,
        embedder_backend="local",
        reranker_backend="local",
        embedder_model="org/embedder",
        reranker_model="org/reranker",
    )

    deps = build_dependencies(
        settings, http_client=_no_network_client(), executor=executor
    )

    assert isinstance(deps.embedder, FakeEmbedder) and isinstance(
        deps.reranker, FakeReranker
    )
    assert loads == [
        ("embedder", "org/embedder", executor),
        ("reranker", "org/reranker", executor),
    ]


def test_vector_store_records_embedder_model(
    settings_factory: SettingsFactory, executor: Executor
) -> None:
    settings = _fake_settings(settings_factory)

    store = build_vector_store(settings, executor=executor, embedder_model="m-x")

    assert isinstance(store, ChromaVectorStore)
    assert store.embedder_model() == "m-x"


@pytest.mark.parametrize(
    ("field", "env_name", "build"),
    [
        ("embedder_backend", "RAG_EMBEDDER_BACKEND", "embedder"),
        ("reranker_backend", "RAG_RERANKER_BACKEND", "reranker"),
        ("llm_backend", "RAG_LLM_BACKEND", "llm"),
    ],
)
def test_unknown_backend_raises_config_error_with_value(
    field: str, env_name: str, build: str, executor: Executor
) -> None:
    # model_construct bypasses the Literal validation that normally rejects this,
    # which is exactly the path a direct caller could hit.
    settings = RagSettings.model_construct(**{field: "bogus"})
    kwargs: dict[str, Any] = {
        "http_client": _no_network_client(),
        "executor": executor,
        "token_provider": None,
    }
    builders: dict[str, Callable[[], Any]] = {
        "embedder": lambda: build_embedder(settings, **kwargs),
        "reranker": lambda: build_reranker(settings, **kwargs),
        "llm": lambda: build_chat_model(settings),
    }

    with pytest.raises(RagConfigError) as info:
        builders[build]()

    assert info.value.setting == env_name
    assert info.value.got == "bogus"
