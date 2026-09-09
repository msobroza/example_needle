"""Optional Domino agent tracing for ``/query``, plus the JSON log-line helper.

``traced`` is identity unless ``RAG_DOMINO_TRACING`` is on *and* the Domino SDK
is importable, so the service runs unchanged outside Domino.

Example::

    run_query = traced("rag_query", enabled=settings.domino_tracing)(run_rag)
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)

# Domino 6.2 ships ``add_tracing`` under ``aisystems``; older SDKs under ``agents``.
TRACING_MODULES = ("domino.aisystems.tracing", "domino.agents.tracing")
AUTOLOG_FRAMEWORKS = ("langchain",)


def log_json(logger: logging.Logger, level: int, **fields: Any) -> None:
    """Emit ``fields`` (plus ``level``) as one JSON object on a single log line.

    Example::

        log_json(logger, logging.INFO, event="rag_request", status=200)
    """
    payload = {**fields, "level": logging.getLevelName(level)}
    logger.log(level, json.dumps(payload, default=str))


def resolve_add_tracing() -> Callable[..., Any] | None:
    """Return Domino's ``add_tracing`` factory, or ``None`` (with a JSON warning).

    Looked up by name from ``traced`` so tests can monkeypatch it.

    Example::

        add_tracing = resolve_add_tracing()
    """
    for dotted in TRACING_MODULES:
        add_tracing = _import_add_tracing(dotted)
        if add_tracing is not None:
            return add_tracing
    log_json(
        logger,
        logging.WARNING,
        event="domino_tracing_unavailable",
        tried=list(TRACING_MODULES),
        detail="RAG_DOMINO_TRACING is on but the Domino SDK is not importable; "
        "install rag-load-test[domino] or run without tracing",
    )
    return None


def _import_add_tracing(dotted: str) -> Callable[..., Any] | None:
    try:
        module = importlib.import_module(dotted)
    except ImportError:  # covers ModuleNotFoundError and ``None`` in sys.modules
        return None
    return getattr(module, "add_tracing", None)


def traced(name: str, *, enabled: bool) -> Callable[[F], F]:
    """Decorator factory: ``add_tracing(name=..., autolog_frameworks=["langchain"])``.

    With ``enabled=False`` (or no Domino SDK) the decorated function is returned
    untouched, so there is zero overhead on the hot path.

    Example::

        @traced("rag_query", enabled=True)
        async def answer(question: str) -> str: ...
    """

    def decorate(fn: F) -> F:
        if not enabled:
            return fn
        add_tracing = resolve_add_tracing()
        if add_tracing is None:
            return fn
        return add_tracing(name=name, autolog_frameworks=list(AUTOLOG_FRAMEWORKS))(fn)

    return decorate
