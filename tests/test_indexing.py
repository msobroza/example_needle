"""Tests for the :mod:`needle.indexing` index-store adapters.

These are deliberately torch-free: they exercise only NumPy arrays and plain
dict payloads so they run in the lightweight test environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from needle.exceptions import IndexStoreError
from needle.indexing import (
    BaseIndexStore,
    InMemoryIndexStore,
    NumpyIndexStore,
    PickleIndexStore,
)
from needle_core.domain.ports.index_store import IndexStorePort

ALL_STORES = [InMemoryIndexStore, PickleIndexStore, NumpyIndexStore]


def _sample_items():
    embeddings = [
        np.arange(4, dtype=np.float32),
        np.arange(8, dtype=np.float32).reshape(2, 4),
    ]
    payloads = [
        {"page": 1, "year": 2024},
        {"page": 2, "year": 2025},
    ]
    return embeddings, payloads


@pytest.mark.parametrize("store_cls", ALL_STORES)
def test_is_index_store_port(store_cls):
    assert isinstance(store_cls(), IndexStorePort)
    assert isinstance(store_cls(), BaseIndexStore)


@pytest.mark.parametrize("store_cls", ALL_STORES)
def test_add_len_and_access(store_cls):
    store = store_cls()
    assert len(store) == 0

    emb = np.array([1.0, 2.0, 3.0])
    payload = {"page": 1, "year": 2024}
    store.add(emb, payload)

    assert len(store) == 1
    assert store.payloads() == [payload]
    np.testing.assert_allclose(store.embeddings()[0], emb)


@pytest.mark.parametrize("store_cls", ALL_STORES)
def test_clear(store_cls):
    store = store_cls()
    store.add(np.zeros(3), {"page": 1, "year": 2024})
    store.clear()
    assert len(store) == 0
    assert store.embeddings() == []
    assert store.payloads() == []


@pytest.mark.parametrize("store_cls", ALL_STORES)
def test_extend(store_cls):
    store = store_cls()
    embeddings, payloads = _sample_items()
    store.extend(embeddings, payloads)

    assert len(store) == 2
    assert store.payloads() == payloads
    for got, want in zip(store.embeddings(), embeddings, strict=True):
        np.testing.assert_allclose(got, want)


@pytest.mark.parametrize("store_cls", ALL_STORES)
def test_repr(store_cls):
    store = store_cls()
    store.add(np.zeros(2), {"page": 1, "year": 2024})
    assert repr(store) == f"<{store_cls.__name__} items=1>"


def test_in_memory_save_raises():
    store = InMemoryIndexStore()
    with pytest.raises(IndexStoreError):
        store.save("anywhere.bin")


def test_in_memory_load_raises():
    store = InMemoryIndexStore()
    with pytest.raises(IndexStoreError):
        store.load("anywhere.bin")


def _assert_round_trip(saved, loaded, embeddings, payloads):
    assert len(loaded) == len(saved)
    assert loaded.payloads() == payloads
    for got, want in zip(loaded.embeddings(), embeddings, strict=True):
        np.testing.assert_allclose(got, want)


def test_pickle_round_trip(tmp_path):
    embeddings, payloads = _sample_items()
    store = PickleIndexStore()
    store.extend(embeddings, payloads)

    path = tmp_path / "index.pkl"
    store.save(path)
    assert path.exists()

    loaded = PickleIndexStore().load(path)
    _assert_round_trip(store, loaded, embeddings, payloads)


def test_pickle_format_matches_legacy_layout(tmp_path):
    """The pickle must be the exact ``{"embeddings", "payloads"}`` dict."""
    import pickle

    embeddings, payloads = _sample_items()
    store = PickleIndexStore()
    store.extend(embeddings, payloads)

    path = tmp_path / "index.pkl"
    store.save(path)

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    assert set(data) == {"embeddings", "payloads"}
    assert data["payloads"] == payloads


def test_numpy_round_trip(tmp_path):
    embeddings, payloads = _sample_items()
    store = NumpyIndexStore()
    store.extend(embeddings, payloads)

    path = tmp_path / "index.npz"
    store.save(path)
    assert path.exists()

    loaded = NumpyIndexStore().load(path)
    _assert_round_trip(store, loaded, embeddings, payloads)


def test_numpy_round_trip_uncompressed(tmp_path):
    embeddings, payloads = _sample_items()
    store = NumpyIndexStore(compressed=False)
    store.extend(embeddings, payloads)

    path = tmp_path / "index_raw.npz"
    store.save(path)

    loaded = NumpyIndexStore().load(path)
    _assert_round_trip(store, loaded, embeddings, payloads)
