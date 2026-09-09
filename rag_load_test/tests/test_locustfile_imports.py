"""Import ``loadtest/locustfile.py`` headlessly and inspect RagUser + StepLoadShape."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCUSTFILE = ROOT / "loadtest" / "locustfile.py"
_module_ids = itertools.count()


def _load_locustfile() -> ModuleType:
    # Fresh module name per load: the file reads task weights and the shape
    # switch from os.environ at import time, so each test needs a re-exec.
    name = f"locustfile_under_test_{next(_module_ids)}"
    spec = importlib.util.spec_from_file_location(name, LOCUSTFILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_loadtest_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCUST_SKIP_MONKEY_PATCH", "1")
    for key in (
        "RAG_LOADTEST_QUERY_WEIGHT",
        "RAG_LOADTEST_RETRIEVE_WEIGHT",
        "RAG_LOADTEST_SHAPE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_locustfile_defines_query_and_retrieve_tasks() -> None:
    module = _load_locustfile()
    names = [t.__name__ for t in module.RagUser.tasks]
    assert sorted(set(names)) == ["query", "retrieve"]
    # locust repeats each task ``weight`` times inside ``tasks``; defaults 1 and 3.
    assert names.count("query") == 1
    assert names.count("retrieve") == 3


def test_locustfile_tasks_has_two_entries_with_unit_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LOADTEST_QUERY_WEIGHT", "1")
    monkeypatch.setenv("RAG_LOADTEST_RETRIEVE_WEIGHT", "1")
    module = _load_locustfile()
    assert len(module.RagUser.tasks) == 2


def test_locustfile_registers_step_shape_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not hasattr(_load_locustfile(), "StepLoadShape")
    monkeypatch.setenv("RAG_LOADTEST_SHAPE", "step")
    assert hasattr(_load_locustfile(), "StepLoadShape")


def test_step_shape_ticks_through_steps_then_stops() -> None:
    from rag_load_test.loadtest.shapes import STEP_SECONDS, STEP_USERS, StepLoadShape

    assert STEP_USERS == (5, 10, 20, 40)
    assert STEP_SECONDS == 60
    shape = StepLoadShape()
    clock = {"t": 0.0}
    shape.get_run_time = lambda: clock["t"]  # type: ignore[method-assign]
    assert shape.tick() == (5, 5)
    clock["t"] = 3 * STEP_SECONDS + 1.0
    assert shape.tick() == (40, 40)
    clock["t"] = 240.0
    assert shape.tick() is None


class FakeLocustResponse:
    """Stand-in for locust's ResponseContextManager inside a catch_response block."""

    def __init__(
        self, status_code: int, text: str | None, error: Exception | None = None
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.error = error
        self.failures: list[str] = []

    def json(self) -> Any:
        return json.loads(self.text)  # type: ignore[arg-type]  # same as locust

    def failure(self, message: str) -> None:
        self.failures.append(message)


def test_json_body_or_fail_reports_transport_error_without_body() -> None:
    # Regression: connection refused gives status 0 and text None; the old
    # ``resp.text[:200]`` raised TypeError and the request was never reported.
    module = _load_locustfile()
    refused = ConnectionRefusedError("[Errno 61] Connection refused")
    resp = FakeLocustResponse(0, None, error=refused)
    assert module._json_body_or_fail(resp) is None
    assert resp.failures == ["HTTP 0: [Errno 61] Connection refused"]


def test_json_body_or_fail_marks_http_error_with_body_excerpt() -> None:
    module = _load_locustfile()
    resp = FakeLocustResponse(503, "x" * 300)
    assert module._json_body_or_fail(resp) is None
    assert resp.failures == ["HTTP 503: " + "x" * 200]


@pytest.mark.parametrize("text", ["not json", None])
def test_json_body_or_fail_marks_non_json_and_empty_bodies(text: str | None) -> None:
    module = _load_locustfile()
    resp = FakeLocustResponse(200, text)
    assert module._json_body_or_fail(resp) is None
    assert resp.failures == ["non-JSON body"]


def test_json_body_or_fail_rejects_non_object_json() -> None:
    module = _load_locustfile()
    resp = FakeLocustResponse(200, "[1, 2]")
    assert module._json_body_or_fail(resp) is None
    assert resp.failures == ["expected a JSON object, got list"]


def test_json_body_or_fail_returns_object() -> None:
    module = _load_locustfile()
    resp = FakeLocustResponse(200, '{"timings_ms": {"embed": 1}}')
    assert module._json_body_or_fail(resp) == {"timings_ms": {"embed": 1}}
    assert resp.failures == []


def test_fire_stage_events_emits_one_stage_request_per_nonzero_stage() -> None:
    from locust.env import Environment

    module = _load_locustfile()
    module.RagUser.host = "http://127.0.0.1:9"  # FastHttpUser refuses a None host
    env = Environment(user_classes=[module.RagUser])
    fired: list[dict[str, Any]] = []
    env.events.request.add_listener(lambda **kwargs: fired.append(kwargs))
    user = module.RagUser(env)
    user.on_start()
    body = {
        "timings_ms": {"embed": 4.0, "retrieve": 0.0, "rerank": 12.0, "total": 16.0}
    }
    user._fire_stage_events("/query", body)
    assert [(f["request_type"], f["name"], f["response_time"]) for f in fired] == [
        ("STAGE", "/query:embed", 4.0),
        ("STAGE", "/query:rerank", 12.0),
    ]
    assert all(f["exception"] is None for f in fired)
