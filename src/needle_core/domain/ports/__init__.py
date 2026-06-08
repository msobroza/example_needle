"""Ports (interfaces) for the retrieval domain — the hexagonal boundary.

The domain defines *what* it needs (embed images, store vectors, render pages,
retrieve) as abstract ports; the :mod:`needle` package provides concrete
adapters. Depending on these ports rather than concrete classes keeps the
domain and application layers free of heavyweight infrastructure.
"""

from __future__ import annotations

from .embedder import Embedder
from .index_store import IndexStorePort
from .page_renderer import PageRenderer
from .retriever import RetrieverPort

__all__ = [
    "Embedder",
    "IndexStorePort",
    "PageRenderer",
    "RetrieverPort",
]
