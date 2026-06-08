"""Tests for the embedding backends (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from conversational_core.domain.ports.embedder import Embedder
from needle.embedders import BaseEmbedder, DeterministicEmbedder


@pytest.fixture
def image():
    from PIL import Image

    return Image.new("RGB", (24, 32), "white")


def test_base_embedder_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedder()  # type: ignore[abstract]


def test_deterministic_satisfies_embedder_port():
    emb = DeterministicEmbedder()
    assert isinstance(emb, Embedder)
    assert "multi-vector" in emb.regime
    assert "DeterministicEmbedder" in repr(emb)


def test_multi_vector_shapes_and_determinism(image):
    emb = DeterministicEmbedder(dim=16, num_tokens=8, multi_vector=True)
    a = emb.embed_images([image, image])
    assert len(a) == 2
    assert a[0].shape == (8, 16)
    assert np.allclose(a[0], a[1])  # same content -> same vector

    q1 = emb.embed_query("hello world")
    q2 = emb.embed_query("hello world")
    assert q1.ndim == 2 and q1.shape[1] == 16
    assert np.allclose(q1, q2)
    assert not np.allclose(q1, emb.embed_query("different"))


def test_dense_shapes(image):
    emb = DeterministicEmbedder(dim=16, multi_vector=False)
    assert emb.embed_images([image])[0].shape == (16,)
    assert emb.embed_query("x").shape == (16,)
    assert "dense" in emb.regime


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        DeterministicEmbedder(dim=0)
    with pytest.raises(ValueError):
        DeterministicEmbedder(num_tokens=0)
