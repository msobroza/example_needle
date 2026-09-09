"""FastAPI service driven through TestClient with fake ports (no weights, no network)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_load_test.adapters.chroma_store import ChromaVectorStore
from rag_load_test.api.__main__ import main
from rag_load_test.api.app import RagRuntime, create_app, readiness_checks
from rag_load_test.contracts.models import Passage, ScoredPassage
from rag_load_test.errors import RagDependencyError
from rag_load_test.settings import RagSettings
from rag_load_test.testing import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.workflow.nodes import RagDependencies

TOPOLOGY = "test-topo"
QUESTION = "how many vacation days do employees get"
STAGE_KEYS = {"embed", "retrieve", "rerank", "generate", "total"}
CORPUS: list[tuple[str, str]] = [
    ("p1", "employees get 25 vacation days per year"),
    ("p2", "vacation requests must be approved by a manager"),
    ("p3", "the cafeteria serves lunch from noon to two"),
    ("p4", "parking permits are issued by facilities"),
    ("p5", "remote work policy allows three days per week"),
    ("p6", "expense reports are due by the fifth of the month"),
]
SettingsFactory = Callable[..., RagSettings]


class RaisingReranker:
    """RerankerPort double that fails with the requested RagDependencyError kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        raise RagDependencyError(
            "reranker", "boom", target="http://ovms:8001/v3/rerank", kind=self.kind
        )

    async def ready(self) -> tuple[bool, str]:
        return True, "raising"


def _settings(settings_factory: SettingsFactory, **overrides: Any) -> RagSettings:
    base: dict[str, Any] = {
        "topology": TOPOLOGY,
        "reranker_backend": "fake",
        "embedder_backend": "fake",
        "llm_backend": "fake",
        "chroma_path": ":memory:",
    }
    return settings_factory(**{**base, **overrides})


def _corpus_rows(embedder: FakeEmbedder) -> tuple[list[Passage], list[list[float]]]:
    passages = [Passage(id=pid, text=text) for pid, text in CORPUS]
    return passages, [embedder.embed_text(p.text) for p in passages]


def _populated_store(embedder: FakeEmbedder) -> InMemoryVectorStore:
    # Sync tests only (TestClient drives its own loop); async tests await upsert.
    store = InMemoryVectorStore()
    asyncio.run(store.upsert(*_corpus_rows(embedder)))
    return store


def _deps(
    store: InMemoryVectorStore, embedder: FakeEmbedder, **overrides: Any
) -> RagDependencies:
    fields: dict[str, Any] = {
        "embedder": embedder,
        "vector_store": store,
        "reranker": FakeReranker(),
        "chat_model": FakeChatModel(),
        "top_k_retrieve": 4,
        "top_k_rerank": 3,
    }
    return RagDependencies(**{**fields, **overrides})


@pytest.fixture
def deps() -> RagDependencies:
    embedder = FakeEmbedder()
    return _deps(_populated_store(embedder), embedder)


@pytest.fixture
def client(
    settings_factory: SettingsFactory, deps: RagDependencies
) -> Iterator[TestClient]:
    app = create_app(_settings(settings_factory), dependencies=deps)
    with TestClient(app) as test_client:
        yield test_client


def _client_for(settings_factory: SettingsFactory, deps: RagDependencies) -> TestClient:
    return TestClient(create_app(_settings(settings_factory), dependencies=deps))


def test_query_returns_answer_passages_timings_and_headers(client: TestClient) -> None:
    response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].startswith("[fake-llm]")
    assert body["deployment"] == TOPOLOGY and body["mode"] == "query"
    assert 0 < len(body["passages"]) <= 5
    assert set(body["passages"][0]) == {
        "id",
        "text",
        "retrieval_score",
        "rerank_score",
        "metadata",
    }
    assert set(body["timings_ms"]) == STAGE_KEYS
    assert body["timings_ms"]["total"] > 0.0
    assert response.headers["X-RAG-Topology"] == TOPOLOGY
    assert response.headers["X-Request-ID"] == body["request_id"] != ""


def test_retrieve_has_no_answer(client: TestClient) -> None:
    response = client.post("/retrieve", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] is None and body["mode"] == "retrieve"
    assert body["timings_ms"]["generate"] == 0.0
    assert len(body["passages"]) == 3


def test_top_k_overrides_flow_to_the_graph(client: TestClient) -> None:
    body = client.post(
        "/query", json={"question": QUESTION, "top_k": 2, "rerank_top_k": 1}
    ).json()
    assert len(body["passages"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {},
        {"question": "q", "top_k": 0},
        {"question": "q", "rerank_top_k": 51},
    ],
)
def test_validation_422_on_bad_body(
    client: TestClient, payload: dict[str, Any]
) -> None:
    response = client.post("/query", json=payload)
    assert response.status_code == 422
    assert response.headers["X-RAG-Topology"] == TOPOLOGY  # middleware also wraps 422s


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_ready(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    checks = {c["name"]: c for c in body["checks"]}
    assert {"vector_store", "embedder", "reranker", "embedder_model_match"} <= set(
        checks
    )
    assert all(c["ok"] for c in checks.values())
    assert checks["vector_store"]["detail"] == "6 passages in rag_passages"


def test_readyz_not_ready_when_store_empty(settings_factory: SettingsFactory) -> None:
    embedder = FakeEmbedder()
    with _client_for(
        settings_factory, _deps(InMemoryVectorStore(), embedder)
    ) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    checks = {c["name"]: c for c in body["checks"]}
    assert checks["vector_store"]["ok"] is False
    assert checks["embedder"]["ok"] is True


async def test_readiness_checks_skip_reranker_and_warn_on_model_mismatch(
    settings_factory: SettingsFactory,
) -> None:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore(embedder_model="other-model")
    await store.upsert(*_corpus_rows(embedder))
    runtime = RagRuntime(
        settings=_settings(settings_factory),
        deps=_deps(store, embedder, reranker=None),
        graph=None,
    )

    checks = {c.name: c for c in await readiness_checks(runtime)}

    assert "reranker" not in checks
    assert all(c.ok for c in checks.values())
    assert "other-model" in checks["embedder_model_match"].detail
    assert embedder.model_name in checks["embedder_model_match"].detail


@pytest.mark.parametrize(("kind", "status"), [("unavailable", 503), ("timeout", 504)])
def test_dependency_error_maps_to_503_and_timeout_to_504(
    settings_factory: SettingsFactory, kind: str, status: int
) -> None:
    embedder = FakeEmbedder()
    deps = _deps(_populated_store(embedder), embedder, reranker=RaisingReranker(kind))
    with _client_for(settings_factory, deps) as client:
        response = client.post(
            "/query", json={"question": QUESTION}, headers={"X-Request-ID": "rid-1"}
        )

    assert response.status_code == status
    body = response.json()
    assert body["error"] == f"reranker_{kind}"
    assert "boom" in body["detail"]
    assert body["target"] == "http://ovms:8001/v3/rerank"
    assert response.headers["X-RAG-Topology"] == TOPOLOGY
    assert response.headers["X-Request-ID"] == "rid-1"


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": QUESTION}, headers={"X-Request-ID": "abc"}
    )

    assert response.headers["X-Request-ID"] == "abc"
    assert response.json()["request_id"] == "abc"


def test_each_request_logs_one_json_line(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="rag_load_test.api"):
        client.post(
            "/retrieve", json={"question": QUESTION}, headers={"X-Request-ID": "log-1"}
        )

    events = [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name.startswith("rag_load_test.api")
    ]
    requests = [e for e in events if e.get("event") == "rag_request"]
    assert len(requests) == 1
    logged = requests[0]
    assert (
        logged["request_id"],
        logged["topology"],
        logged["mode"],
        logged["status"],
    ) == ("log-1", TOPOLOGY, "retrieve", 200)
    assert set(logged["timings_ms"]) == STAGE_KEYS


def test_root_path_comes_from_domino_env(
    settings_factory: SettingsFactory,
    deps: RagDependencies,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/proxy/run-1")
    app = create_app(_settings(settings_factory), dependencies=deps)

    assert app.root_path == "/proxy/run-1"
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200  # routes stay at "/"


def test_lifespan_builds_dependencies_from_settings(
    settings_factory: SettingsFactory,
) -> None:
    settings = _settings(
        settings_factory, chroma_collection=f"api-{uuid.uuid4().hex[:8]}"
    )
    app = create_app(settings)

    with TestClient(app) as client:
        runtime = app.state.runtime
        assert isinstance(runtime, RagRuntime) and runtime.settings is settings
        assert isinstance(runtime.deps.vector_store, ChromaVectorStore)
        assert isinstance(runtime.deps.embedder, FakeEmbedder)
        assert client.get("/readyz").status_code == 503  # fresh collection is empty


def test_main_runs_uvicorn_with_cli_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: runs.append({"app": app, **kwargs})
    )

    assert main(["--host", "127.0.0.1", "--port", "9999"]) == 0

    assert len(runs) == 1
    assert isinstance(runs[0]["app"], FastAPI)
    assert (runs[0]["host"], runs[0]["port"], runs[0]["log_level"]) == (
        "127.0.0.1",
        9999,
        "info",
    )
