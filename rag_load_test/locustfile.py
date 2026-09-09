"""Locust users for the RAG service; ``make loadtest`` runs this file headless.

Each successful response's ``timings_ms`` is re-emitted as ``STAGE`` request
events so per-stage p50/p95 land in the same ``*_stats.csv`` as the endpoint
totals. ``RAG_LOADTEST_SHAPE=step`` enables the 5 -> 10 -> 20 -> 40 user ramp.
"""

from __future__ import annotations

import os
import random

from locust import FastHttpUser, LoadTestShape, between, task

from rag_load_test.loadtest import (
    QUESTIONS,
    auth_headers,
    parse_timings,
    stage_events,
    task_weights,
)

QUERY_WEIGHT, RETRIEVE_WEIGHT = task_weights(os.environ)


class RagUser(FastHttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.headers = auth_headers(os.environ)

    @task(QUERY_WEIGHT)
    def query(self) -> None:
        self._ask("/query")

    @task(RETRIEVE_WEIGHT)
    def retrieve(self) -> None:
        self._ask("/retrieve")

    def _ask(self, path: str) -> None:
        payload = {"question": random.choice(QUESTIONS)}
        with self.client.post(
            path, json=payload, headers=self.headers, name=path, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {(resp.text or '')[:200]}")
                return
            for name, ms in stage_events(path, parse_timings(resp.json())):
                self.environment.events.request.fire(
                    request_type="STAGE",
                    name=name,
                    response_time=ms,
                    response_length=0,
                    exception=None,
                    context={},
                )


if os.environ.get("RAG_LOADTEST_SHAPE") == "step":
    STEP_USERS, STEP_SECONDS = (5, 10, 20, 40), 60

    class StepLoadShape(LoadTestShape):  # locust picks up shape classes defined here
        def tick(self) -> tuple[int, float] | None:
            step = int(self.get_run_time() // STEP_SECONDS)
            if step >= len(STEP_USERS):
                return None
            return STEP_USERS[step], float(STEP_USERS[step])
