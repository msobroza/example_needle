"""Locust-free helpers for ``loadtest/locustfile.py``.

Everything here is plain Python so it can be unit-tested without importing
locust (which gevent-patches the interpreter at import time).

Example::

    headers = auth_headers(os.environ)
    for name, ms in stage_events("/query", parse_timings(body)):
        events.request.fire(request_type=STAGE_REQUEST_TYPE, name=name, ...)
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

STAGE_REQUEST_TYPE = "STAGE"
# Order matters: it is the order stages run in the workflow and appear in tables.
STAGES: tuple[str, ...] = ("embed", "retrieve", "rerank", "generate")

BEARER_TOKEN_ENV = "RAG_LOADTEST_BEARER_TOKEN"
DOMINO_API_KEY_ENV = "DOMINO_API_KEY"
QUERY_WEIGHT_ENV = "RAG_LOADTEST_QUERY_WEIGHT"
RETRIEVE_WEIGHT_ENV = "RAG_LOADTEST_RETRIEVE_WEIGHT"
DEFAULT_QUERY_WEIGHT = 1
DEFAULT_RETRIEVE_WEIGHT = 3


def auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    """Pick the auth header a Domino app expects from callers outside a run.

    ``RAG_LOADTEST_BEARER_TOKEN`` wins over ``DOMINO_API_KEY``; blank values
    count as unset so a templated ``.env`` never sends an empty header.

    Example::

        auth_headers({"DOMINO_API_KEY": "k"})  # {"X-Domino-Api-Key": "k"}
    """
    token = env.get(BEARER_TOKEN_ENV, "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    api_key = env.get(DOMINO_API_KEY_ENV, "")
    if api_key:
        return {"X-Domino-Api-Key": api_key}
    return {}


def stage_events(
    endpoint: str, timings_ms: Mapping[str, float]
) -> list[tuple[str, float]]:
    """Turn per-stage timings into ``(name, ms)`` pairs for extra locust events.

    ``total`` is excluded (the endpoint request itself already measures it) and
    absent or zero stages (``generate`` on ``/retrieve``) are skipped.

    Example::

        stage_events("/query", {"embed": 4.0, "total": 9.0})
        # [("/query:embed", 4.0)]
    """
    return [
        (f"{endpoint}:{stage}", timings_ms[stage])
        for stage in STAGES
        if stage in timings_ms and timings_ms[stage] > 0
    ]


def task_weights(env: Mapping[str, str]) -> tuple[int, int]:
    """Read the ``(query, retrieve)`` task weights; defaults are ``(1, 3)``.

    A weight of ``0`` disables that task; anything but a non-negative integer
    literal raises ``ValueError`` naming the variable and the offending value.

    Example::

        task_weights({"RAG_LOADTEST_RETRIEVE_WEIGHT": "0"})  # (1, 0)
    """
    return (
        _weight(env, QUERY_WEIGHT_ENV, DEFAULT_QUERY_WEIGHT),
        _weight(env, RETRIEVE_WEIGHT_ENV, DEFAULT_RETRIEVE_WEIGHT),
    )


def _weight(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    # isdecimal() rejects signs, blanks and floats, and int() accepts every
    # decimal digit it accepts, so this is the whole validation.
    if not raw.strip().isdecimal():
        raise ValueError(f"{key}: got {raw!r}, expected a non-negative integer")
    return int(raw)


def pick_question(rng: random.Random, questions: Sequence[str]) -> str:
    """Choose one question from the bank with the caller's RNG.

    Example::

        pick_question(random.Random(1), QUESTIONS)
    """
    if not questions:
        raise ValueError("question bank is empty, expected at least one question")
    return rng.choice(questions)


def parse_timings(body: Mapping[str, Any]) -> dict[str, float]:
    """Extract ``body["timings_ms"]`` as ``{stage: float}``; ``{}`` when malformed.

    Malformed means the key is missing, not a mapping, or holds a value that is
    not a plain number (bools and strings are rejected so ``"fast"`` never
    silently becomes a latency).

    Example::

        parse_timings({"timings_ms": {"embed": 3}})  # {"embed": 3.0}
    """
    raw = body.get("timings_ms")
    if not isinstance(raw, Mapping):
        return {}
    if not all(_is_number(value) for value in raw.values()):
        return {}
    return {str(stage): float(value) for stage, value in raw.items()}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
