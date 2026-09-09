"""OVMS embedder/reranker adapters and the shared retrying POST over httpx.MockTransport.

Bearer-header behaviour (static and Domino tokens) is covered in ``test_auth.py``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from rag_load_test.adapters.auth import NoAuth, StaticBearerToken
from rag_load_test.adapters.http_retry import post_json_with_retry
from rag_load_test.adapters.ovms_embedder import OvmsEmbedder
from rag_load_test.adapters.ovms_reranker import OvmsReranker
from rag_load_test.contracts.models import ScoredPassage
from rag_load_test.contracts.ports import EmbedderPort, RerankerPort
from rag_load_test.errors import RagDependencyError

EMBED_BASE = "http://ovms:8002"
RERANK_BASE = "http://ovms:8001"
POST_URL = "http://ovms:9/v3/anything"
ADAPTER_KW: dict[str, Any] = {
    "token_provider": NoAuth(),
    "timeout_s": 5.0,
    "max_retries": 0,
}
Reply = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class RecordingServer:
    """MockTransport handler: records every request, replays scripted replies in order."""

    def __init__(self, *scripted: Reply, default: Reply | None = None) -> None:
        self._scripted = list(scripted)
        self._default = default
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._scripted.pop(0) if self._scripted else self._default
        if reply is None:
            raise AssertionError(
                f"unexpected request #{len(self.requests)}: {request.url}"
            )
        if isinstance(reply, Exception):
            raise reply
        return reply(request) if callable(reply) else reply

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def payloads(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


class CountingTokenProvider:
    """Hands out ``tok-<n>`` where n counts invalidations, like a refreshed Domino token."""

    def __init__(self) -> None:
        self.invalidations = 0

    async def token(self) -> str | None:
        return f"tok-{self.invalidations}"

    def invalidate(self) -> None:
        self.invalidations += 1


def embeddings_reply(request: httpx.Request) -> httpx.Response:
    """One 1-d vector per input, encoding the input's numeric suffix (``t3`` -> ``[3.0]``)."""
    inputs = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(t[1:])]} for i, t in enumerate(inputs)]
    return httpx.Response(200, json={"object": "list", "data": data})


def ok_reply() -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


def fake_sleep() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return sleeps, sleep


def make_embedder(
    server: RecordingServer, base_url: str = EMBED_BASE, **overrides: Any
) -> OvmsEmbedder:
    kwargs = {**ADAPTER_KW, **overrides}
    return OvmsEmbedder(server.client(), base_url, "bge-small-en-v1.5", **kwargs)


def make_reranker(server: RecordingServer, **overrides: Any) -> OvmsReranker:
    kwargs = {**ADAPTER_KW, **overrides}
    return OvmsReranker(server.client(), RERANK_BASE, "bge-reranker-base", **kwargs)


def passages(n: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text{i}", retrieval_score=0.9 - i / 10)
        for i in range(n)
    ]


async def retrying_post(server: RecordingServer, **overrides: Any) -> dict[str, Any]:
    kwargs = {**ADAPTER_KW, "max_retries": 2, **overrides}
    return await post_json_with_retry(
        server.client(), POST_URL, {"k": "v"}, component="embedder", **kwargs
    )


# --- embedder -----------------------------------------------------------------


async def test_embedder_posts_expected_payload_and_reorders_by_index() -> None:
    rows = [{"index": 1, "embedding": [1.0]}, {"index": 0, "embedding": [0.0]}]
    server = RecordingServer(httpx.Response(200, json={"data": rows}))
    embedder = make_embedder(server)

    assert await embedder.embed_documents(["a", "b"]) == [[0.0], [1.0]]
    assert server.requests[0].method == "POST"
    assert str(server.requests[0].url) == f"{EMBED_BASE}/v3/embeddings"
    assert server.payloads() == [{"model": "bge-small-en-v1.5", "input": ["a", "b"]}]
    assert embedder.model_name == "ovms:bge-small-en-v1.5"
    assert isinstance(embedder, EmbedderPort)


async def test_embedder_embed_query_returns_first_vector_and_passes_timeout() -> None:
    server = RecordingServer(default=embeddings_reply)
    assert await make_embedder(server, timeout_s=7.5).embed_query("t4") == [4.0]
    assert server.payloads() == [{"model": "bge-small-en-v1.5", "input": ["t4"]}]
    assert server.requests[0].extensions["timeout"]["read"] == 7.5


async def test_embedder_batches() -> None:
    server = RecordingServer(default=embeddings_reply)
    texts = [f"t{i}" for i in range(5)]
    vectors = await make_embedder(server, batch_size=2).embed_documents(texts)
    assert vectors == [[float(i)] for i in range(5)]
    assert [p["input"] for p in server.payloads()] == [texts[:2], texts[2:4], texts[4:]]


async def test_embedder_empty_input_makes_no_request() -> None:
    server = RecordingServer()
    assert await make_embedder(server).embed_documents([]) == []
    assert server.requests == []


async def test_embedder_tolerates_trailing_slash_in_base_url() -> None:
    server = RecordingServer(default=embeddings_reply)
    await make_embedder(server, base_url=EMBED_BASE + "/").embed_query("t1")
    assert str(server.requests[0].url) == f"{EMBED_BASE}/v3/embeddings"


async def test_embedder_malformed_body_is_dependency_error() -> None:
    server = RecordingServer(httpx.Response(200, json={"data": [{"index": 0}]}))
    with pytest.raises(RagDependencyError) as e:
        await make_embedder(server).embed_query("a")
    assert e.value.component == "embedder"
    assert e.value.target == f"{EMBED_BASE}/v3/embeddings"


async def test_embedder_retries_through_adapter() -> None:
    server = RecordingServer(httpx.Response(503, text="busy"), default=embeddings_reply)
    assert await make_embedder(server, max_retries=1).embed_query("t2") == [2.0]
    assert len(server.requests) == 2


# --- reranker -----------------------------------------------------------------


async def test_reranker_payload_and_ordering() -> None:
    scores = {2: 0.1, 0: 0.95, 1: 0.5}  # server order differs from passage order
    results = [{"index": i, "relevance_score": s} for i, s in scores.items()]
    server = RecordingServer(httpx.Response(200, json={"results": results}))
    reranker = make_reranker(server)
    candidates = passages(3)

    ranked = await reranker.rerank("q", candidates, top_n=2)

    assert [p.id for p in ranked] == ["p0", "p1"]
    assert [p.rerank_score for p in ranked] == [0.95, 0.5]
    assert [p.retrieval_score for p in ranked] == [0.9, 0.8]
    assert all(p.rerank_score is None for p in candidates)  # inputs are not mutated
    assert str(server.requests[0].url) == f"{RERANK_BASE}/v3/rerank"
    expected = {"model": "bge-reranker-base", "query": "q", "top_n": 2}
    assert server.payloads() == [{**expected, "documents": ["text0", "text1", "text2"]}]
    assert isinstance(reranker, RerankerPort)


async def test_reranker_empty_passages_makes_no_request() -> None:
    server = RecordingServer()
    assert await make_reranker(server).rerank("q", [], top_n=3) == []
    assert server.requests == []


async def test_reranker_malformed_result_is_dependency_error() -> None:
    bad = {"results": [{"index": 7, "relevance_score": 0.5}]}  # index past the end
    server = RecordingServer(httpx.Response(200, json=bad))
    with pytest.raises(RagDependencyError) as e:
        await make_reranker(server).rerank("q", passages(2), top_n=2)
    assert e.value.component == "reranker"
    assert e.value.target == f"{RERANK_BASE}/v3/rerank"


# --- retry policy (post_json_with_retry) --------------------------------------


async def test_retry_on_503_then_success() -> None:
    sleeps, sleep = fake_sleep()
    server = RecordingServer(httpx.Response(503, text="busy"), ok_reply())
    assert await retrying_post(server, max_retries=2, sleep=sleep) == {"ok": True}
    assert len(server.requests) == 2
    assert sleeps == [0.2]


async def test_401_invalidates_token_and_retries() -> None:
    sleeps, sleep = fake_sleep()
    provider = CountingTokenProvider()
    server = RecordingServer(httpx.Response(401), ok_reply())
    assert await retrying_post(server, token_provider=provider, sleep=sleep) == {
        "ok": True
    }
    assert provider.invalidations == 1
    tokens = [r.headers["authorization"] for r in server.requests]
    assert tokens == ["Bearer tok-0", "Bearer tok-1"]
    assert sleeps == [0.2]


async def test_4xx_fails_fast() -> None:
    sleeps, sleep = fake_sleep()
    server = RecordingServer(httpx.Response(400, text="bad request body"))
    with pytest.raises(RagDependencyError) as e:
        await retrying_post(server, sleep=sleep)
    assert e.value.kind == "unavailable"
    assert e.value.component == "embedder"
    assert e.value.target == POST_URL
    assert "HTTP 400" in e.value.detail and "bad request body" in e.value.detail
    assert len(server.requests) == 1
    assert sleeps == []


async def test_timeout_exhausts_to_timeout_kind() -> None:
    sleeps, sleep = fake_sleep()
    server = RecordingServer(default=httpx.ReadTimeout("read timed out"))
    with pytest.raises(RagDependencyError) as e:
        await retrying_post(server, max_retries=1, sleep=sleep)
    assert e.value.kind == "timeout"
    assert e.value.target == POST_URL
    assert "read timed out" in e.value.detail
    assert len(server.requests) == 2
    assert sleeps == [0.2]


async def test_5xx_exhausts_to_unavailable_kind_with_exponential_backoff() -> None:
    sleeps, sleep = fake_sleep()
    server = RecordingServer(default=httpx.Response(503, text="still busy"))
    with pytest.raises(RagDependencyError) as e:
        await retrying_post(server, max_retries=2, sleep=sleep)
    assert e.value.kind == "unavailable"
    assert "HTTP 503" in e.value.detail and "still busy" in e.value.detail
    assert len(server.requests) == 3
    assert sleeps == [0.2, 0.4]


async def test_connect_error_exhausts_to_unavailable_kind() -> None:
    sleeps, sleep = fake_sleep()
    server = RecordingServer(default=httpx.ConnectError("connection refused"))
    with pytest.raises(RagDependencyError) as e:
        await retrying_post(server, max_retries=0, sleep=sleep)
    assert e.value.kind == "unavailable"
    assert "connection refused" in e.value.detail
    assert len(server.requests) == 1
    assert sleeps == []


async def test_invalid_json_body_is_dependency_error() -> None:
    _, sleep = fake_sleep()
    server = RecordingServer(httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(RagDependencyError) as e:
        await retrying_post(server, sleep=sleep)
    assert "not json" in e.value.detail
    assert len(server.requests) == 1


# --- readiness ----------------------------------------------------------------


async def test_ready_true_and_false() -> None:
    up = RecordingServer(httpx.Response(200, json={"model_version_status": []}))
    ok, detail = await make_embedder(up, token_provider=StaticBearerToken("t")).ready()
    assert (ok, detail) == (True, f"{EMBED_BASE}/v3/models/bge-small-en-v1.5 -> 200")
    assert up.requests[0].method == "GET"
    assert up.requests[0].headers["authorization"] == "Bearer t"

    down = RecordingServer(default=httpx.ConnectError("connection refused"))
    ok, detail = await make_reranker(down).ready()
    assert ok is False and "connection refused" in detail

    missing = RecordingServer(httpx.Response(404, text="no such model"))
    ok, detail = await make_reranker(missing).ready()
    assert (ok, detail) == (False, f"{RERANK_BASE}/v3/models/bge-reranker-base -> 404")
