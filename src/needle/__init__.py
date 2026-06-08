"""needle — page-level multimodal document retrievers.

The heavy retriever classes (``ColPaliRetriever`` and friends) live in
:mod:`needle.retrieval.page_retrievers` and pull in PyTorch. To keep
``import needle`` cheap and dependency-light, those symbols are exposed
lazily via :pep:`562` module ``__getattr__`` — they are only imported the
first time you actually touch them.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

# Names that are resolved lazily from needle.retrieval.page_retrievers.
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

_LAZY_REGISTRY = {"get_retriever", "available_retrievers"}

__all__ = [
    "__version__",
    *sorted(_LAZY_REGISTRY),
    *sorted(_LAZY_RETRIEVERS),
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy shim
    if name in _LAZY_RETRIEVERS:
        from . import retrieval

        return getattr(retrieval, name)
    if name in _LAZY_REGISTRY:
        from .retrieval import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
