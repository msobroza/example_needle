"""Locust users for the RAG service: ``POST /query`` and ``POST /retrieve``.

Run headless via ``scripts/run_loadtest.sh <name> <host>``. Each successful
response's ``timings_ms`` is re-emitted as ``STAGE`` request events so p50/p95
per workflow stage land in the same ``*_stats.csv`` as the endpoint totals.
Set ``RAG_LOADTEST_SHAPE=step`` to drive the run with ``StepLoadShape``.
"""

from __future__ import annotations

import os
import random
from typing import Any

from locust import FastHttpUser, between, task
from locust.contrib.fasthttp import ResponseContextManager

from rag_load_test.loadtest.helpers import (
    STAGE_REQUEST_TYPE,
    auth_headers,
    parse_timings,
    pick_question,
    stage_events,
    task_weights,
)
from rag_load_test.loadtest.questions import QUESTIONS

QUERY_WEIGHT, RETRIEVE_WEIGHT = task_weights(os.environ)


class RagUser(FastHttpUser):
    """One simulated client asking random bank questions with a short think time."""

    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.rng = random.Random()
        self.headers = auth_headers(os.environ)

    @task(QUERY_WEIGHT)
    def query(self) -> None:
        self._ask("/query")

    @task(RETRIEVE_WEIGHT)
    def retrieve(self) -> None:
        self._ask("/retrieve")

    def _ask(self, path: str) -> None:
        payload = {"question": pick_question(self.rng, QUESTIONS)}
        with self.client.post(
            path, json=payload, headers=self.headers, name=path, catch_response=True
        ) as resp:
            body = _json_body_or_fail(resp)
            if body is None:
                return
            self._fire_stage_events(path, body)

    def _fire_stage_events(self, path: str, body: dict[str, Any]) -> None:
        for name, ms in stage_events(path, parse_timings(body)):
            self.environment.events.request.fire(
                request_type=STAGE_REQUEST_TYPE,
                name=name,
                response_time=ms,
                response_length=0,
                exception=None,
                context={},
            )


def _json_body_or_fail(resp: ResponseContextManager) -> dict[str, Any] | None:
    """Return the JSON object body, or mark the response failed and return None."""
    if resp.status_code != 200:
        resp.failure(f"HTTP {resp.status_code}: {_failure_excerpt(resp)}")
        return None
    try:
        body = resp.json()
    except (ValueError, TypeError):
        # FastResponse.json() is json.loads(text): TypeError when the body is None.
        resp.failure("non-JSON body")
        return None
    if not isinstance(body, dict):
        resp.failure(f"expected a JSON object, got {type(body).__name__}")
        return None
    return body


def _failure_excerpt(resp: ResponseContextManager) -> str:
    # Transport errors (connection refused, timeout) yield status 0 and no body:
    # ``text`` is None and the exception lives in ``error`` (locust ErrorResponse).
    if resp.text:
        return resp.text[:200]
    return str(getattr(resp, "error", None) or "no response body")


if os.environ.get("RAG_LOADTEST_SHAPE") == "step":
    # locust auto-registers LoadTestShape subclasses found in the locustfile module.
    from rag_load_test.loadtest.shapes import StepLoadShape  # noqa: F401
