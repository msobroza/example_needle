"""Shared HTTP plumbing for the OVMS adapters: a bounded-retry JSON POST, the
readiness GET, and the error mapping both adapters (and the Domino token
provider) rely on.

Retry policy (spec section 13): attempts = ``max_retries + 1``. Transport errors
(timeouts included) and 5xx are retried after ``backoff_s * 2**attempt``; a 401
invalidates the bearer token first so the next attempt fetches a fresh one; any
other 4xx fails fast because retrying cannot fix a bad request. No circuit
breaker or rate limiting, per the project rules.

Example::

    body = await post_json_with_retry(
        client, "http://ovms:8002/v3/embeddings", {"model": "m", "input": ["a"]},
        component="embedder", token_provider=NoAuth(), timeout_s=30, max_retries=2,
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx

from ..errors import RagDependencyError

if TYPE_CHECKING:  # auth.py imports this module at runtime; keep the cycle type-only
    from .auth import BearerTokenProvider

SleepFn = Callable[[float], Awaitable[None]]

# Upstream body characters kept in error messages: enough to diagnose a failure
# without turning a multi-KB error page into a log flood.
DETAIL_LIMIT = 200


class RetryableStatus(Exception):
    """A 401 or 5xx reply: retried, and reported verbatim if retries run out."""


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    component: str,
    token_provider: BearerTokenProvider,
    timeout_s: float,
    max_retries: int,
    backoff_s: float = 0.2,
    sleep: SleepFn = asyncio.sleep,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON object.

    Raises ``RagDependencyError(component, ..., target=url)`` with ``kind="timeout"``
    when the last failure was an httpx timeout, ``"unavailable"`` otherwise.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        outcome = await _attempt(
            client,
            url,
            payload,
            component=component,
            token_provider=token_provider,
            timeout_s=timeout_s,
        )
        if isinstance(outcome, dict):
            return outcome
        last_error = outcome
        if attempt < max_retries:
            await sleep(backoff_s * (2**attempt))
    raise dependency_error(component, last_error, target=url)


async def _attempt(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    component: str,
    token_provider: BearerTokenProvider,
    timeout_s: float,
) -> dict[str, Any] | Exception:
    """One POST: the JSON body on success, the error if retryable, raise otherwise."""
    headers = await bearer_headers(token_provider)
    try:
        response = await client.post(
            url, json=payload, headers=headers, timeout=timeout_s
        )
    except httpx.TransportError as exc:
        return exc
    status = response.status_code
    if status == 401:
        token_provider.invalidate()  # stale Domino token: the next attempt refetches
        return RetryableStatus(describe_response(response))
    if status >= 500:
        return RetryableStatus(describe_response(response))
    if status >= 400:
        raise RagDependencyError(component, describe_response(response), target=url)
    return parse_json_object(response, component=component, target=url)


async def bearer_headers(token_provider: BearerTokenProvider) -> dict[str, str]:
    """``{"Authorization": "Bearer <token>"}``, or ``{}`` when the provider has none.

    Example::

        headers = await bearer_headers(StaticBearerToken("t"))
        assert headers == {"Authorization": "Bearer t"}
    """
    token = await token_provider.token()
    return {} if token is None else {"Authorization": f"Bearer {token}"}


async def probe_ready(
    client: httpx.AsyncClient,
    url: str,
    *,
    token_provider: BearerTokenProvider,
    timeout_s: float,
) -> tuple[bool, str]:
    """GET ``url`` once and return ``(ok, detail)``; no retries so /readyz stays fast.

    Example::

        ok, detail = await probe_ready(client, "http://ovms:8002/v3/models/m",
                                       token_provider=NoAuth(), timeout_s=5)
    """
    try:
        headers = await bearer_headers(token_provider)
        response = await client.get(url, headers=headers, timeout=timeout_s)
    except (httpx.TransportError, RagDependencyError) as exc:
        return False, describe_exception(exc)
    return response.status_code == 200, f"{url} -> {response.status_code}"


def parse_json_object(
    response: httpx.Response, *, component: str, target: str
) -> dict[str, Any]:
    """Decode a JSON object body or raise RagDependencyError carrying an excerpt."""
    try:
        body = response.json()
    except ValueError as exc:
        detail = f"invalid JSON body: {excerpt(response.text)}"
        raise RagDependencyError(component, detail, target=target) from exc
    if not isinstance(body, dict):
        detail = f"expected a JSON object, got {type(body).__name__}: {excerpt(body)}"
        raise RagDependencyError(component, detail, target=target)
    return body


def dependency_error(
    component: str, exc: BaseException | None, *, target: str
) -> RagDependencyError:
    """Map a transport/status failure to RagDependencyError.

    httpx timeouts become ``kind="timeout"`` (HTTP 504 at the API), everything
    else ``"unavailable"`` (HTTP 503).
    """
    kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "unavailable"
    return RagDependencyError(
        component, describe_exception(exc), target=target, kind=kind
    )


def describe_response(response: httpx.Response) -> str:
    """``"HTTP 503: <body excerpt>"``."""
    return f"HTTP {response.status_code}: {excerpt(response.text)}"


def describe_exception(exc: BaseException | None) -> str:
    """``"ReadTimeout: read timed out"`` (type kept: httpx messages can be empty)."""
    if exc is None:
        return "no attempt made"
    if isinstance(exc, RetryableStatus):
        return str(exc)
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def excerpt(value: Any) -> str:
    """The first ``DETAIL_LIMIT`` characters of ``str(value)`` for error messages."""
    return str(value)[:DETAIL_LIMIT]
