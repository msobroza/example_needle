"""``traced`` resolves Domino's ``add_tracing`` lazily and degrades to identity."""

from __future__ import annotations

import json
import logging
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from rag_load_test.api import tracing
from rag_load_test.api.tracing import log_json, resolve_add_tracing, traced

AISYSTEMS_MODULE = "domino.aisystems.tracing"
AGENTS_MODULE = "domino.agents.tracing"


async def sample(value: int) -> int:
    return value + 1


def _recording_add_tracing(calls: list[dict[str, Any]]) -> Callable[..., Any]:
    """A stand-in for ``add_tracing``: records kwargs, wraps with ``__wrapped__``."""

    def add_tracing(**kwargs: Any) -> Callable[[Any], Any]:
        calls.append(kwargs)

        def decorator(fn: Any) -> Any:
            async def wrapped(*args: Any, **kw: Any) -> Any:
                return await fn(*args, **kw)

            wrapped.__wrapped__ = fn  # type: ignore[attr-defined]
            return wrapped

        return decorator

    return add_tracing


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch, dotted: str, add_tracing: Any
) -> None:
    module = types.ModuleType(dotted)
    module.add_tracing = add_tracing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, dotted, module)


def test_disabled_returns_same_function_object() -> None:
    assert traced("x", enabled=False)(sample) is sample


def test_enabled_applies_add_tracing_with_name_and_langchain_autolog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tracing, "resolve_add_tracing", lambda: _recording_add_tracing(calls)
    )

    wrapped = traced("x", enabled=True)(sample)

    assert wrapped is not sample
    assert wrapped.__wrapped__ is sample  # type: ignore[attr-defined]
    assert calls == [{"name": "x", "autolog_frameworks": ["langchain"]}]


async def test_enabled_wrapper_still_calls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracing, "resolve_add_tracing", lambda: _recording_add_tracing([])
    )
    assert await traced("x", enabled=True)(sample)(1) == 2


def test_enabled_without_resolver_result_is_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "resolve_add_tracing", lambda: None)
    assert traced("x", enabled=True)(sample) is sample


def test_resolve_returns_none_and_warns_as_json_when_domino_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # ``None`` in sys.modules makes ``import`` raise ImportError deterministically,
    # whatever is installed in the environment.
    monkeypatch.setitem(sys.modules, AISYSTEMS_MODULE, None)
    monkeypatch.setitem(sys.modules, AGENTS_MODULE, None)

    with caplog.at_level(logging.WARNING, logger=tracing.__name__):
        assert resolve_add_tracing() is None

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    payload = json.loads(warnings[0].getMessage())
    assert payload["event"] == "domino_tracing_unavailable"
    assert AISYSTEMS_MODULE in payload["tried"] and AGENTS_MODULE in payload["tried"]


def test_resolve_prefers_aisystems_over_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    def from_aisystems(**kwargs: Any) -> None: ...

    def from_agents(**kwargs: Any) -> None: ...

    _install_fake_module(monkeypatch, AISYSTEMS_MODULE, from_aisystems)
    _install_fake_module(monkeypatch, AGENTS_MODULE, from_agents)

    assert resolve_add_tracing() is from_aisystems


def test_resolve_falls_back_to_agents_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def from_agents(**kwargs: Any) -> None: ...

    monkeypatch.setitem(sys.modules, AISYSTEMS_MODULE, None)
    _install_fake_module(monkeypatch, AGENTS_MODULE, from_agents)

    assert resolve_add_tracing() is from_agents


def test_log_json_emits_one_parseable_line(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("rag_load_test.tests.log_json")
    with caplog.at_level(logging.INFO, logger=logger.name):
        log_json(logger, logging.INFO, event="unit", status=200, nested={"a": 1.5})

    assert len(caplog.records) == 1
    assert json.loads(caplog.records[0].getMessage()) == {
        "event": "unit",
        "status": 200,
        "nested": {"a": 1.5},
        "level": "INFO",
    }
