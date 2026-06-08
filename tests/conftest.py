"""Shared pytest fixtures.

Image fixtures use only Pillow (no torch), so the torch-free test modules can
run them anywhere. Retriever fixtures ``importorskip`` torch so the full
pipeline tests are skipped gracefully when PyTorch is not installed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def png_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes a small PNG/JPG and returns its path."""
    from PIL import Image, ImageDraw

    def _make(
        name: str = "page.png",
        color: str = "white",
        text: str = "hello",
        size: tuple[int, int] = (256, 320),
    ) -> Path:
        image = Image.new("RGB", size, color)
        ImageDraw.Draw(image).text((10, 10), text, fill="black")
        path = tmp_path / name
        image.save(path)
        return path

    return _make


@pytest.fixture
def sample_files(png_factory: Callable[..., Path]) -> dict[str, Path]:
    """Three small documents with distinct content."""
    return {
        "invoice": png_factory("invoice.png", "white", "INVOICE 2021"),
        "report": png_factory("report.png", "lightyellow", "REPORT 2024"),
        "memo": png_factory("memo.jpg", "lightblue", "MEMO 2024"),
    }


@pytest.fixture
def DummyRetriever():
    """The deterministic dummy retriever class (requires torch at import)."""
    pytest.importorskip("torch")
    from needle.testing import DummyEmbedderRetriever

    return DummyEmbedderRetriever


@pytest.fixture
def indexed_retriever(DummyRetriever, sample_files, tmp_path):
    """A DummyEmbedderRetriever with three documents already indexed."""
    from needle.retrieval.data import InputDocument

    retriever = DummyRetriever(
        index_path=str(tmp_path / "index.pkl"), batch_size=2, multi_vector=True
    )
    documents = [
        InputDocument.from_path(
            sample_files["invoice"], metadata={"year": 2021, "lang": "en"}
        ),
        InputDocument.from_path(
            sample_files["report"], metadata={"year": 2024, "lang": "en"}
        ),
        InputDocument.from_path(
            sample_files["memo"], metadata={"year": 2024, "lang": "fr"}
        ),
    ]
    retriever.index(documents)
    return retriever
