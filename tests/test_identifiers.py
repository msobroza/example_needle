"""Tests for typed identifier value objects."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from conversational_core.domain.identifiers import DocumentId, QueryId, VersionId


def test_str_returns_raw_value() -> None:
    assert str(DocumentId("doc_abc")) == "doc_abc"
    assert str(VersionId("ver_abc")) == "ver_abc"
    assert str(QueryId("q_abc")) == "q_abc"


def test_new_uses_expected_prefixes() -> None:
    assert DocumentId.new().value.startswith("doc_")
    assert VersionId.new().value.startswith("ver_")
    assert QueryId.new().value.startswith("q_")


def test_new_format_is_prefix_underscore_hex12() -> None:
    for cls, prefix in ((DocumentId, "doc"), (VersionId, "ver"), (QueryId, "q")):
        value = cls.new().value
        assert re.fullmatch(rf"{prefix}_[0-9a-f]{{12}}", value), value


def test_new_ids_are_unique() -> None:
    ids = {DocumentId.new().value for _ in range(100)}
    assert len(ids) == 100


def test_frozen_and_hashable() -> None:
    doc = DocumentId("doc_x")
    with pytest.raises(FrozenInstanceError):
        doc.value = "other"  # type: ignore[misc]
    # Hashable -> usable as dict keys / in sets.
    assert {DocumentId("doc_x"), DocumentId("doc_x")} == {DocumentId("doc_x")}


def test_equality_by_value() -> None:
    assert DocumentId("doc_x") == DocumentId("doc_x")
    assert DocumentId("doc_x") != DocumentId("doc_y")


@pytest.mark.parametrize("cls", [DocumentId, VersionId, QueryId])
def test_empty_value_rejected(cls: type) -> None:
    with pytest.raises(ValueError):
        cls("")
