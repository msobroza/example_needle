"""Bearer-token providers for the OVMS HTTP adapters (``RAG_OVMS_AUTH``).

OVMS itself is unauthenticated, but on Domino each app sits behind a proxy that
expects the run's bearer token. That token is served by the run-local
``/access-token`` endpoint and expires, hence the TTL cache in
:class:`DominoAccessToken`; a 401 from OVMS calls ``invalidate()`` so the next
request fetches a fresh one.

Example::

    provider = build_token_provider(settings, client)
    token = await provider.token()          # None for RAG_OVMS_AUTH=none
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import httpx

from ..errors import RagConfigError, RagDependencyError
from ..settings import RagSettings
from .http_retry import dependency_error, describe_response

DOMINO_TOKEN_COMPONENT = "domino_access_token"


@runtime_checkable
class BearerTokenProvider(Protocol):
    """Source of the ``Authorization: Bearer`` value for OVMS calls."""

    async def token(self) -> str | None: ...

    def invalidate(self) -> None: ...


class NoAuth:
    """Provider for unauthenticated OVMS endpoints (``RAG_OVMS_AUTH=none``).

    Example::

        assert await NoAuth().token() is None
    """

    async def token(self) -> str | None:
        return None

    def invalidate(self) -> None:
        return None


class StaticBearerToken:
    """A fixed token from ``RAG_OVMS_BEARER_TOKEN`` (``RAG_OVMS_AUTH=static``).

    Example::

        assert await StaticBearerToken("abc").token() == "abc"
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def token(self) -> str | None:
        return self._token

    def invalidate(self) -> None:
        return None  # nothing to refresh: a static token cannot be renewed


class DominoAccessToken:
    """Fetch the Domino run token with ``GET url`` and cache it for ``ttl_s`` seconds.

    The body is the raw token (whitespace stripped). ``clock`` is injectable so
    tests can drive expiry without sleeping.

    Example::

        provider = DominoAccessToken(client, "http://localhost:8899/access-token")
        token = await provider.token()   # cached for 240 s, refetched on invalidate()
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        ttl_s: float = 240.0,
        timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._url = url
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s
        self._clock = clock
        self._cached: str | None = None
        self._fetched_at = 0.0
        self._refresh_lock = asyncio.Lock()

    async def token(self) -> str | None:
        if self._is_fresh():
            return self._cached
        async with (
            self._refresh_lock
        ):  # one refetch per expiry, not one per in-flight call
            if not self._is_fresh():
                self._cached = await self._fetch()
                self._fetched_at = self._clock()
        return self._cached

    def invalidate(self) -> None:
        self._cached = None

    def _is_fresh(self) -> bool:
        return (
            self._cached is not None and self._clock() - self._fetched_at < self._ttl_s
        )

    async def _fetch(self) -> str:
        try:
            response = await self._client.get(self._url, timeout=self._timeout_s)
        except httpx.TransportError as exc:
            raise dependency_error(
                DOMINO_TOKEN_COMPONENT, exc, target=self._url
            ) from exc
        if response.status_code != 200:
            detail = describe_response(response)
            raise RagDependencyError(DOMINO_TOKEN_COMPONENT, detail, target=self._url)
        token = response.text.strip()
        if not token:
            detail = "empty body, expected a bearer token"
            raise RagDependencyError(DOMINO_TOKEN_COMPONENT, detail, target=self._url)
        return token


def build_token_provider(
    settings: RagSettings, client: httpx.AsyncClient
) -> BearerTokenProvider:
    """Pick the provider for ``ovms_auth``: ``none`` | ``static`` | ``domino``.

    Example::

        provider = build_token_provider(RagSettings.from_env(), httpx.AsyncClient())
    """
    mode = settings.ovms_auth
    if mode == "none":
        return NoAuth()
    if mode == "static":
        return _static_from_settings(settings)
    if mode == "domino":
        return DominoAccessToken(client, settings.domino_access_token_url)
    raise RagConfigError(
        "RAG_OVMS_AUTH", got=mode, expected="one of none, static, domino"
    )


def _static_from_settings(settings: RagSettings) -> StaticBearerToken:
    token = settings.ovms_bearer_token.strip()
    if not token:
        raise RagConfigError(
            "RAG_OVMS_BEARER_TOKEN",
            got=settings.ovms_bearer_token,
            expected="non-empty token",
        )
    return StaticBearerToken(token)
