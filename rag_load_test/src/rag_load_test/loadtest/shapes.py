"""Opt-in step load shape (``RAG_LOADTEST_SHAPE=step``): 5 -> 10 -> 20 -> 40 users.

This module imports locust, so it must stay out of the service's import path;
``loadtest/locustfile.py`` imports it only when the shape is requested.

Example::

    shape = StepLoadShape()
    shape.tick()  # (5, 5.0) during the first 60 s, None after 240 s
"""

from __future__ import annotations

from locust import LoadTestShape

STEP_USERS: tuple[int, ...] = (5, 10, 20, 40)
STEP_SECONDS = 60


class StepLoadShape(LoadTestShape):
    """Hold each user count in ``STEP_USERS`` for ``STEP_SECONDS``, then stop."""

    def tick(self) -> tuple[int, float] | None:
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEP_USERS):
            return None
        users = STEP_USERS[step]
        # Spawn at ``users``/s so every step reaches its plateau within a second.
        return users, float(users)
