"""Tests for declarative metadata schema validation."""

from __future__ import annotations

from conversational_core.domain.metadata.schema import MetadataField, MetadataSchema


def _schema() -> MetadataSchema:
    return MetadataSchema(
        fields=[
            MetadataField("title", type=str, required=True),
            MetadataField("year", type=int),
            MetadataField("lang", type=str, choices=("en", "fr")),
        ]
    )


def test_valid_metadata() -> None:
    schema = _schema()
    meta = {"title": "Report", "year": 2024, "lang": "en"}
    assert schema.validate(meta) == []
    assert schema.is_valid(meta) is True


def test_missing_required_field() -> None:
    schema = _schema()
    errors = schema.validate({"year": 2024})
    assert any("missing required field" in e and "title" in e for e in errors)
    assert schema.is_valid({"year": 2024}) is False


def test_optional_field_absent_is_ok() -> None:
    schema = _schema()
    assert schema.validate({"title": "x"}) == []


def test_wrong_type() -> None:
    schema = _schema()
    errors = schema.validate({"title": "x", "year": "not-a-number"})
    assert any("year" in e and "expected type int" in e for e in errors)


def test_bool_not_accepted_for_int() -> None:
    schema = _schema()
    errors = schema.validate({"title": "x", "year": True})
    assert any("year" in e and "bool" in e for e in errors)


def test_value_not_in_choices() -> None:
    schema = _schema()
    errors = schema.validate({"title": "x", "lang": "de"})
    assert any("lang" in e and "not in" in e for e in errors)


def test_value_in_choices_ok() -> None:
    schema = _schema()
    assert schema.validate({"title": "x", "lang": "fr"}) == []


def test_multiple_errors_collected() -> None:
    schema = _schema()
    errors = schema.validate({"year": "x", "lang": "de"})
    # missing title + wrong-type year + bad-choice lang
    assert len(errors) == 3


def test_bool_field_accepts_bool() -> None:
    schema = MetadataSchema(fields=[MetadataField("flag", type=bool)])
    assert schema.validate({"flag": True}) == []
