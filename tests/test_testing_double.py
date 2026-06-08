"""Tests for the deterministic :class:`DummyEmbedderRetriever` test double."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from needle.testing import DummyEmbedderRetriever  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture
def image():
    from PIL import Image

    return Image.new("RGB", (32, 48), "white")


def test_query_embedding_is_deterministic(tmp_path):
    r = DummyEmbedderRetriever(index_path=str(tmp_path / "i.pkl"))
    a = r._embed_query("hello world")
    b = r._embed_query("hello world")
    assert np.allclose(a, b)
    assert not np.allclose(a, r._embed_query("different text"))


def test_multi_vector_shapes(tmp_path, image):
    r = DummyEmbedderRetriever(
        index_path=str(tmp_path / "i.pkl"), multi_vector=True, dim=16, num_tokens=8
    )
    embs = r._embed_images([image, image])
    assert len(embs) == 2
    assert embs[0].shape == (8, 16)
    q = r._embed_query("two words")
    assert q.ndim == 2 and q.shape[1] == 16


def test_dense_shapes(tmp_path, image):
    r = DummyEmbedderRetriever(
        index_path=str(tmp_path / "i.pkl"), multi_vector=False, dim=16
    )
    embs = r._embed_images([image])
    assert embs[0].shape == (16,)
    q = r._embed_query("anything")
    assert q.shape == (16,)


def test_same_image_same_embedding(tmp_path, image):
    r = DummyEmbedderRetriever(index_path=str(tmp_path / "i.pkl"))
    first = r._embed_images([image])[0]
    second = r._embed_images([image])[0]
    assert np.allclose(first, second)
