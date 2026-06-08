"""Tests for needle configuration objects."""

from __future__ import annotations

import pytest

from needle.config import RetrieverConfig
from needle.embedders.colpali_engine import BACKEND_SPECS, resolve_device
from needle.exceptions import ConfigurationError


def test_defaults_and_to_kwargs():
    config = RetrieverConfig()
    kwargs = config.to_kwargs()
    assert kwargs == {
        "index_path": config.index_path,
        "dpi": config.dpi,
        "batch_size": config.batch_size,
    }
    assert "model_name" not in kwargs  # omitted when None


def test_to_kwargs_includes_model_name_when_set():
    config = RetrieverConfig(model_name="some/model")
    assert config.to_kwargs()["model_name"] == "some/model"


def test_validation():
    with pytest.raises(ConfigurationError):
        RetrieverConfig(dpi=0)
    with pytest.raises(ConfigurationError):
        RetrieverConfig(batch_size=0)
    with pytest.raises(ConfigurationError):
        RetrieverConfig(name="")


def test_from_env(monkeypatch):
    monkeypatch.setenv("NEEDLE_RETRIEVER", "colpali")
    monkeypatch.setenv("NEEDLE_INDEX_PATH", "/tmp/idx.pkl")
    monkeypatch.setenv("NEEDLE_DPI", "200")
    monkeypatch.setenv("NEEDLE_BATCH_SIZE", "8")
    config = RetrieverConfig.from_env()
    assert config.name == "colpali"
    assert config.index_path == "/tmp/idx.pkl"
    assert config.dpi == 200
    assert config.batch_size == 8


def test_backend_specs_cover_known_backends():
    assert {"colpali", "colqwen2", "nomic", "modernvbert"} <= set(BACKEND_SPECS)
    assert BACKEND_SPECS["colpali"].multi_vector is True
    assert BACKEND_SPECS["nomic"].multi_vector is False


def test_resolve_device_returns_string():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device() in {"cpu", "cuda"}
