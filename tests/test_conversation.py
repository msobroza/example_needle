"""Tests for conversation and response value objects."""

from __future__ import annotations

from conversational_core.domain.interaction.conversation import Conversation, Turn
from conversational_core.domain.interaction.query import Query
from conversational_core.domain.interaction.response import RetrievalResponse


def test_empty_conversation() -> None:
    conv = Conversation()
    assert len(conv) == 0
    assert conv.last is None
    assert conv.history_text() == ""
    assert list(conv) == []


def test_add_turn_flow() -> None:
    conv = Conversation()
    turn = conv.add_turn(Query("what is revenue?"))
    assert isinstance(turn, Turn)
    assert len(conv) == 1
    assert conv.last is turn
    assert conv.last.response is None


def test_turn_with_response() -> None:
    conv = Conversation()
    conv.add_turn(Query("first"))
    conv.add_turn(Query("second"), response="answer")
    assert len(conv) == 2
    assert conv.last.response == "answer"
    assert conv.last.query.query_text == "second"


def test_history_text_in_order() -> None:
    conv = Conversation()
    conv.add_turn(Query("alpha"))
    conv.add_turn(Query("beta"))
    conv.add_turn(Query("gamma"))
    assert conv.history_text() == "alpha\nbeta\ngamma"


def test_iteration_order() -> None:
    conv = Conversation()
    conv.add_turn(Query("one"))
    conv.add_turn(Query("two"))
    texts = [turn.query.query_text for turn in conv]
    assert texts == ["one", "two"]


def test_response_len_and_emptiness() -> None:
    resp = RetrievalResponse(query=Query("q"), results=[], took_ms=1.5)
    assert len(resp) == 0
    assert resp.is_empty is True
    assert resp.best is None


def test_response_best_and_len() -> None:
    resp = RetrievalResponse(query=Query("q"), results=["a", "b", "c"])
    assert len(resp) == 3
    assert resp.is_empty is False
    assert resp.best == "a"
