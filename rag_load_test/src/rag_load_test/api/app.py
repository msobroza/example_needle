"""FastAPI application factory: settings -> adapters -> graph, wired in the lifespan.

Example::

    app = create_app(RagSettings.from_env())      # or build_app_from_env()
    uvicorn.run(app, host="0.0.0.0", port=8888)
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..adapters.executor import build_model_executor
from ..contracts.ports import EmbedderPort, RerankerPort, VectorStorePort
from ..errors import RagDependencyError
from ..settings import RagSettings
from ..workflow.dependencies import build_dependencies
from ..workflow.graph import build_rag_graph
from ..workflow.nodes import RagDependencies
from .schemas import ReadyCheck
from .tracing import log_json

APP_TITLE = "rag-load-test"
REQUEST_ID_HEADER = "X-Request-ID"
TOPOLOGY_HEADER = "X-RAG-Topology"
# Domino's reverse proxy prefixes every app URL with this path (spec section 8).
ROOT_PATH_ENV = "DOMINO_RUN_HOST_PATH"
logger = logging.getLogger("rag_load_test.api")

LifespanFn = Callable[[FastAPI], Any]


@dataclass
class RagRuntime:
    """Everything a request handler needs, built once by the lifespan.

    Example::

        runtime: RagRuntime = request.app.state.runtime
    """

    settings: RagSettings
    deps: RagDependencies
    graph: Any


async def readiness_checks(runtime: RagRuntime) -> list[ReadyCheck]:
    """Probe the store and model ports; feeds ``GET /readyz``.

    Example::

        checks = await readiness_checks(runtime)
        ready = all(check.ok for check in checks)
    """
    deps = runtime.deps
    checks = [
        await _vector_store_check(
            deps.vector_store, runtime.settings.chroma_collection
        ),
        await _port_check("embedder", deps.embedder),
    ]
    if deps.reranker is not None:
        checks.append(await _port_check("reranker", deps.reranker))
    checks.append(_embedder_model_match(deps))
    return checks


async def _vector_store_check(store: VectorStorePort, collection: str) -> ReadyCheck:
    try:
        count = await store.count()
    except Exception as exc:  # any failure means "not ready", never a 500 from /readyz
        return ReadyCheck(name="vector_store", ok=False, detail=_describe(exc))
    return ReadyCheck(
        name="vector_store", ok=count > 0, detail=f"{count} passages in {collection}"
    )


async def _port_check(name: str, port: EmbedderPort | RerankerPort) -> ReadyCheck:
    try:
        ok, detail = await port.ready()
    except Exception as exc:  # same rule as above: report, do not crash the probe
        return ReadyCheck(name=name, ok=False, detail=_describe(exc))
    return ReadyCheck(name=name, ok=ok, detail=detail)


def _embedder_model_match(deps: RagDependencies) -> ReadyCheck:
    recorded = deps.vector_store.embedder_model()
    serving = deps.embedder.model_name
    if recorded in (None, serving):
        detail = f"index embedder {recorded!r}, serving {serving!r}"
        return ReadyCheck(name="embedder_model_match", ok=True, detail=detail)
    # ok stays True: a mismatched index still serves, but its results are meaningless.
    detail = (
        f"WARNING: index built with {recorded!r} but serving embedder is {serving!r}"
    )
    return ReadyCheck(name="embedder_model_match", ok=True, detail=detail)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class RequestContextMiddleware:
    """Pure ASGI middleware: request id in ``request.state``, headers on every response.

    Being pure ASGI (not ``BaseHTTPMiddleware``) it also stamps the 4xx/5xx bodies
    produced by the exception handlers.

    Example::

        app.add_middleware(RequestContextMiddleware, topology="monolith")
    """

    def __init__(self, app: ASGIApp, *, topology: str) -> None:
        self.app = app
        self.topology = topology

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = Headers(scope=scope).get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                headers[TOPOLOGY_HEADER] = self.topology
            await send(message)

        await self.app(scope, receive, send_with_headers)


def dependency_error_handler(
    topology: str,
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    """Map ``RagDependencyError`` to 504 (``timeout``) or 503, logging one JSON line.

    Example::

        handler = dependency_error_handler("monolith")
        app.add_exception_handler(RagDependencyError, handler)
    """

    async def handle(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RagDependencyError), exc
        status = 504 if exc.kind == "timeout" else 503
        error = f"{exc.component}_{exc.kind}"
        log_json(
            logger,
            logging.WARNING,
            event="rag_request",
            request_id=getattr(request.state, "request_id", ""),
            topology=topology,
            status=status,
            error=error,
            detail=str(exc),
        )
        body = {"error": error, "detail": str(exc), "target": exc.target}
        return JSONResponse(status_code=status, content=body)

    return handle


def create_app(
    settings: RagSettings | None = None,
    *,
    dependencies: RagDependencies | None = None,
) -> FastAPI:
    """Build the FastAPI app; ``dependencies`` bypasses ``build_dependencies`` (tests).

    Example::

        app = create_app(settings, dependencies=fake_deps)
        with TestClient(app) as client: ...
    """
    # routes.py imports RagRuntime/readiness_checks from this module, so the
    # router is imported here rather than at module scope to avoid a cycle.
    from .routes import router

    resolved = settings if settings is not None else RagSettings.from_env()
    app = FastAPI(
        title=APP_TITLE,
        root_path=os.environ.get(ROOT_PATH_ENV, ""),
        lifespan=_lifespan_for(resolved, dependencies),
    )
    app.include_router(router)
    app.add_exception_handler(
        RagDependencyError, dependency_error_handler(resolved.topology)
    )
    app.add_middleware(RequestContextMiddleware, topology=resolved.topology)
    return app


def _lifespan_for(settings: RagSettings, dependencies: RagDependencies | None) -> Any:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        executor = build_model_executor(settings.model_threads)
        http_client = httpx.AsyncClient(timeout=settings.http_timeout_s)
        try:
            # Local weights load here so a broken model fails at startup, not on
            # the first request (spec section 13).
            deps = dependencies
            if deps is None:
                deps = build_dependencies(
                    settings, http_client=http_client, executor=executor
                )
            app.state.runtime = RagRuntime(
                settings=settings, deps=deps, graph=build_rag_graph(deps)
            )
            _log_startup(settings, deps)
            yield
        finally:
            await http_client.aclose()
            executor.shutdown(wait=False)

    return lifespan


def _log_startup(settings: RagSettings, deps: RagDependencies) -> None:
    log_json(
        logger,
        logging.INFO,
        event="rag_startup",
        topology=settings.topology,
        embedder=deps.embedder.model_name,
        reranker=settings.reranker_backend,
        llm=deps.chat_model.model_name,
        collection=settings.chroma_collection,
    )


def build_app_from_env() -> FastAPI:
    """``create_app(RagSettings.from_env())`` — the uvicorn factory entry point.

    Example::

        uvicorn rag_load_test.api.app:build_app_from_env --factory
    """
    return create_app(RagSettings.from_env())
