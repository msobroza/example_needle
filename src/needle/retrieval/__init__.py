"""Retrieval subpackage.

The torch-free building blocks (data carriers, extractors, filter helpers) are
imported eagerly. The retriever classes — which depend on PyTorch — are
exposed lazily so that ``import needle.retrieval`` (and the data/extractor
modules) work without a heavy ML stack installed.
"""

from __future__ import annotations

from typing import Any

from .data import (
    InputDocument,
    PageAnnotation,
    PageAnnotationResult,
    PreannotationPageResult,
)
from .extractors import (
    IMAGE_EXTRACTORS,
    ImageFileExtractor,
    OfficeToImageExtractor,
    PageExtractor,
    PageToImageExtractor,
    PdfToImageExtractor,
    get_extractor,
)
from .page_retriever_utils import SimilarityMapVisualizer, matches_filter

_LAZY_RETRIEVERS = {
    "BasePageRetriever",
    "MultimodalEmbedderRetriever",
    "ColPaliRetriever",
    "ColQwen2Retriever",
    "NomicDenseRetriever",
    "TomoroColQwen3Retriever",
    "ColModernVBertRetriever",
    "ModernVBertRetriever",
}

_LAZY_REGISTRY = {"RETRIEVERS", "get_retriever", "available_retrievers"}

__all__ = [
    "InputDocument",
    "PageAnnotation",
    "PageAnnotationResult",
    "PreannotationPageResult",
    "PageExtractor",
    "PageToImageExtractor",
    "ImageFileExtractor",
    "PdfToImageExtractor",
    "OfficeToImageExtractor",
    "IMAGE_EXTRACTORS",
    "get_extractor",
    "matches_filter",
    "SimilarityMapVisualizer",
    *sorted(_LAZY_RETRIEVERS),
    *sorted(_LAZY_REGISTRY),
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy shim
    if name in _LAZY_RETRIEVERS:
        from . import page_retrievers

        return getattr(page_retrievers, name)
    if name in _LAZY_REGISTRY:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
