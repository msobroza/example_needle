"""Bearer-token providers for the OVMS adapters (token endpoint mocked, no network)."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from rag_load_test.adapters.auth import (
    BearerTokenProvider,
    DominoAccessToken,
    NoAuth,
    StaticBearerToken,
    build_token_provider,
)
from rag_load_test.adapters.ovms_embedder import OvmsEmbedder
from rag_load_test.errors import RagConfigError, RagDependencyError

TOKEN_URL = "http://localhost:8899/access-token"
EMBED_REPLY = {"data": [{"index": 0, "embedding": [1.0]}]}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _embedder(client: httpx.AsyncClient, provider: BearerTokenProvider) -> OvmsEmbedder:
    return OvmsEmbedder(
        client,
        "http://ovms:8002",
        "m",
        token_provider=provider,
        timeout_s=5,
        max_retries=0,
    )


async def test_static_and_none() -> None:
    assert await NoAuth().token() is None
    assert await StaticBearerToken("abc").token() == "abc"
    assert isinstance(NoAuth(), BearerTokenProvider)
    assert isinstance(StaticBearerToken("abc"), BearerTokenProvider)


async def test_invalidate_is_a_no_op_for_static_and_none() -> None:
    static = StaticBearerToken("abc")
    static.invalidate()
    NoAuth().invalidate()
    assert await static.token() == "abc"


async def test_domino_token_is_cached_until_ttl_and_after_invalidate() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=f"tok-{len(calls)}\n")

    now = [1000.0]
    provider = DominoAccessToken(
        _client(handler), TOKEN_URL, ttl_s=240, clock=lambda: now[0]
    )
    assert await provider.token() == "tok-1"
    assert await provider.token() == "tok-1"  # cached
    now[0] += 241
    assert await provider.token() == "tok-2"  # ttl expired
    provider.invalidate()
    assert await provider.token() == "tok-3"
    assert calls == ["/access-token"] * 3


async def test_domino_token_uses_get_with_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="tok")

    provider = DominoAccessToken(_client(handler), TOKEN_URL, timeout_s=2.5)
    assert await provider.token() == "tok"
    assert seen[0].method == "GET"
    assert seen[0].extensions["timeout"]["read"] == 2.5


async def test_domino_token_failure_is_dependency_error() -> None:
    provider = DominoAccessToken(
        _client(lambda r: httpx.Response(500)), "http://x/access-token"
    )
    with pytest.raises(RagDependencyError) as e:
        await provider.token()
    assert e.value.component == "domino_access_token"
    assert e.value.target == "http://x/access-token"
    assert e.value.kind == "unavailable"


async def test_domino_token_empty_body_is_dependency_error() -> None:
    provider = DominoAccessToken(
        _client(lambda r: httpx.Response(200, text="  \n")), TOKEN_URL
    )
    with pytest.raises(RagDependencyError) as e:
        await provider.token()
    assert "empty" in e.value.detail


async def test_domino_token_timeout_maps_to_timeout_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow token endpoint")

    provider = DominoAccessToken(_client(handler), TOKEN_URL)
    with pytest.raises(RagDependencyError) as e:
        await provider.token()
    assert e.value.kind == "timeout"
    assert "slow token endpoint" in e.value.detail


async def test_domino_token_is_refetched_after_failure() -> None:
    replies = [httpx.Response(503), httpx.Response(200, text="tok-ok")]
    provider = DominoAccessToken(_client(lambda r: replies.pop(0)), TOKEN_URL)
    with pytest.raises(RagDependencyError):
        await provider.token()
    assert await provider.token() == "tok-ok"


def test_build_token_provider_static_requires_token(settings_factory) -> None:
    with pytest.raises(RagConfigError) as e:
        build_token_provider(
            settings_factory(ovms_auth="static", ovms_bearer_token=""),
            httpx.AsyncClient(),
        )
    assert e.value.setting == "RAG_OVMS_BEARER_TOKEN"
    assert e.value.got == ""


async def test_build_token_provider_dispatches_on_auth_mode(settings_factory) -> None:
    client = _client(lambda r: httpx.Response(200, text="dom"))
    assert isinstance(
        build_token_provider(settings_factory(ovms_auth="none"), client), NoAuth
    )

    static = build_token_provider(
        settings_factory(ovms_auth="static", ovms_bearer_token="s3cret"), client
    )
    assert isinstance(static, StaticBearerToken)
    assert await static.token() == "s3cret"

    domino = build_token_provider(
        settings_factory(ovms_auth="domino", domino_access_token_url=TOKEN_URL), client
    )
    assert isinstance(domino, DominoAccessToken)
    assert await domino.token() == "dom"


async def test_static_token_becomes_bearer_header_on_ovms_calls() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=EMBED_REPLY)

    client = _client(handler)
    await _embedder(client, StaticBearerToken("t")).embed_query("a")
    await _embedder(client, NoAuth()).embed_query("a")
    assert seen[0].headers["authorization"] == "Bearer t"
    assert "authorization" not in seen[1].headers


async def test_domino_token_becomes_bearer_header_on_ovms_calls() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/access-token":
            return httpx.Response(200, text="dom-tok\n")
        return httpx.Response(200, json=EMBED_REPLY)

    client = _client(handler)
    provider = DominoAccessToken(client, TOKEN_URL)
    assert await _embedder(client, provider).embed_query("a") == [1.0]
    assert [r.url.path for r in seen] == ["/access-token", "/v3/embeddings"]
    assert seen[1].headers["authorization"] == "Bearer dom-tok"
