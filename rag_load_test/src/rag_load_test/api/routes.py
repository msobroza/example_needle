"""HTTP routes. Handlers see only ``RagRuntime`` and log one JSON line per request.

Example::

    app.include_router(router)
"""

from __future__ import annotations

import functools
import logging
import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..contracts.models import RagMode, RagResult
from ..workflow.graph import run_rag
from .app import RagRuntime, readiness_checks
from .schemas import QueryRequest, QueryResponse, ReadyResponse, to_query_response
from .tracing import log_json, traced

router = APIRouter()
logger = logging.getLogger("rag_load_test.api")
RunRag = Callable[..., Awaitable[RagResult]]


def get_runtime(request: Request) -> RagRuntime:
    """The ``RagRuntime`` the lifespan stored on ``app.state``.

    Example::

        runtime = get_runtime(request)
    """
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError(
            "app.state.runtime is unset (got None); the app must be started through "
            "its lifespan, e.g. `with TestClient(app) as client:`"
        )
    return runtime


def get_request_id(request: Request) -> str:
    """Request id assigned by ``RequestContextMiddleware`` (fresh uuid without it).

    Example::

        request_id = get_request_id(request)
    """
    return str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)


@functools.lru_cache(maxsize=2)
def traced_run_rag(enabled: bool) -> RunRag:
    """``run_rag`` wrapped by ``traced("rag_query")`` once per flag, not per request.

    Example::

        run = traced_run_rag(settings.domino_tracing)
    """
    return traced("rag_query", enabled=enabled)(run_rag)


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Full RAG: embed -> retrieve -> rerank -> generate."""
    runtime = get_runtime(request)
    run = traced_run_rag(runtime.settings.domino_tracing)
    return await _answer(request, runtime, body, mode="query", run=run)


@router.post("/retrieve", response_model=QueryResponse)
async def retrieve(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieval only (no LLM call): embed -> retrieve -> rerank."""
    runtime = get_runtime(request)
    return await _answer(request, runtime, body, mode="retrieve", run=run_rag)


async def _answer(
    request: Request,
    runtime: RagRuntime,
    body: QueryRequest,
    *,
    mode: RagMode,
    run: RunRag,
) -> QueryResponse:
    result = await run(
        runtime.graph,
        body.question,
        mode=mode,
        top_k=body.top_k,
        rerank_top_k=body.rerank_top_k,
    )
    response = to_query_response(
        result, request_id=get_request_id(request), deployment=runtime.settings.topology
    )
    log_json(
        logger,
        logging.INFO,
        event="rag_request",
        request_id=response.request_id,
        topology=response.deployment,
        mode=mode,
        timings_ms=response.timings_ms,
        status=200,
    )
    return response


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up (no dependency checks)."""
    return {"status": "ok"}


@router.get(
    "/readyz", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}}
)
async def readyz(request: Request) -> JSONResponse:
    """Readiness: 200 when every check passes, 503 with per-check detail otherwise."""
    checks = await readiness_checks(get_runtime(request))
    ready = all(check.ok for check in checks)
    payload = ReadyResponse(status="ready" if ready else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if ready else 503, content=payload.model_dump())
