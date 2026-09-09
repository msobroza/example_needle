"""Unit tests for the locust-free helpers behind ``loadtest/locustfile.py``."""

from __future__ import annotations

import random
from typing import Any

import pytest

from rag_load_test.loadtest.helpers import (
    STAGE_REQUEST_TYPE,
    auth_headers,
    parse_timings,
    pick_question,
    stage_events,
    task_weights,
)


def test_stage_request_type_constant() -> None:
    assert STAGE_REQUEST_TYPE == "STAGE"


# --- auth_headers -----------------------------------------------------------


def test_auth_headers_prefers_bearer_token_over_domino_key() -> None:
    env = {"RAG_LOADTEST_BEARER_TOKEN": "tok", "DOMINO_API_KEY": "key"}
    assert auth_headers(env) == {"Authorization": "Bearer tok"}


def test_auth_headers_falls_back_to_domino_api_key() -> None:
    assert auth_headers({"DOMINO_API_KEY": "key"}) == {"X-Domino-Api-Key": "key"}


def test_auth_headers_empty_without_credentials() -> None:
    assert auth_headers({}) == {}
    # A blank value (typical of a templated .env) counts as unset.
    assert auth_headers({"RAG_LOADTEST_BEARER_TOKEN": "", "DOMINO_API_KEY": ""}) == {}


# --- stage_events -----------------------------------------------------------


def test_stage_events_orders_stages_and_excludes_total() -> None:
    timings = {
        "total": 100.0,
        "generate": 40.0,
        "embed": 5.0,
        "retrieve": 10.0,
        "rerank": 20.0,
    }
    assert stage_events("/query", timings) == [
        ("/query:embed", 5.0),
        ("/query:retrieve", 10.0),
        ("/query:rerank", 20.0),
        ("/query:generate", 40.0),
    ]


def test_stage_events_skips_missing_and_zero_stages() -> None:
    timings = {"embed": 5.0, "retrieve": 0.0, "generate": 0.0, "total": 5.0}
    assert stage_events("/retrieve", timings) == [("/retrieve:embed", 5.0)]


def test_stage_events_empty_for_empty_timings() -> None:
    assert stage_events("/query", {}) == []


# --- task_weights -----------------------------------------------------------


def test_task_weights_defaults() -> None:
    assert task_weights({}) == (1, 3)


def test_task_weights_reads_env_and_allows_zero() -> None:
    env = {"RAG_LOADTEST_QUERY_WEIGHT": "2", "RAG_LOADTEST_RETRIEVE_WEIGHT": "0"}
    assert task_weights(env) == (2, 0)


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", ""])
def test_task_weights_rejects_bad_query_weight(value: str) -> None:
    with pytest.raises(ValueError, match="RAG_LOADTEST_QUERY_WEIGHT") as exc_info:
        task_weights({"RAG_LOADTEST_QUERY_WEIGHT": value})
    assert repr(value) in str(exc_info.value)


def test_task_weights_rejects_bad_retrieve_weight() -> None:
    with pytest.raises(ValueError, match="RAG_LOADTEST_RETRIEVE_WEIGHT"):
        task_weights({"RAG_LOADTEST_RETRIEVE_WEIGHT": "-3"})


# --- pick_question ----------------------------------------------------------


def test_pick_question_is_deterministic_for_seeded_rng() -> None:
    bank = ["a", "b", "c", "d", "e", "f"]
    first = pick_question(random.Random(7), bank)
    second = pick_question(random.Random(7), bank)
    assert first == second
    assert first in bank


def test_pick_question_rejects_empty_bank() -> None:
    with pytest.raises(ValueError, match="empty"):
        pick_question(random.Random(), [])


# --- parse_timings ----------------------------------------------------------


def test_parse_timings_coerces_numbers_to_float() -> None:
    body = {"timings_ms": {"embed": 5, "retrieve": 10.5, "total": 15.5}}
    parsed = parse_timings(body)
    assert parsed == {"embed": 5.0, "retrieve": 10.5, "total": 15.5}
    assert all(type(v) is float for v in parsed.values())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"timings_ms": None},
        {"timings_ms": "fast"},
        {"timings_ms": [1, 2]},
        {"timings_ms": {"embed": "slow"}},
        {"timings_ms": {"embed": None}},
        {"timings_ms": {"embed": True}},
    ],
)
def test_parse_timings_returns_empty_on_malformed(body: dict[str, Any]) -> None:
    assert parse_timings(body) == {}


# --- question bank ----------------------------------------------------------


def test_question_bank_is_fifty_nonempty_deterministic_questions() -> None:
    from rag_load_test.ingest.corpus import synthetic_questions
    from rag_load_test.loadtest.questions import QUESTIONS

    assert len(QUESTIONS) == 50
    assert all(isinstance(q, str) and q.strip() for q in QUESTIONS)
    assert QUESTIONS == synthetic_questions(50)
