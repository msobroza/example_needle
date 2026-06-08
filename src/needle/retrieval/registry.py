"""A small registry/factory for the built-in retrievers.

Lets callers select a backend by name (e.g. from a CLI flag or config) without
importing concrete classes::

    from needle.retrieval.registry import get_retriever
    retriever = get_retriever("colqwen2", index_path="index.pkl")
"""

from __future__ import annotations

from typing import Any

from .page_retrievers import (
    BasePageRetriever,
    ColModernVBertRetriever,
    ColPaliRetriever,
    ColQwen2Retriever,
    ModernVBertRetriever,
    MultimodalEmbedderRetriever,
    NomicDenseRetriever,
    TomoroColQwen3Retriever,
)

#: Canonical name -> retriever class.
RETRIEVERS: dict[str, type[MultimodalEmbedderRetriever]] = {
    "colpali": ColPaliRetriever,
    "colqwen2": ColQwen2Retriever,
    "nomic": NomicDenseRetriever,
    "tomoro-colqwen3": TomoroColQwen3Retriever,
    "colmodernvbert": ColModernVBertRetriever,
    "modernvbert": ModernVBertRetriever,
}

#: Friendly aliases mapped onto canonical names.
_ALIASES = {
    "colqwen": "colqwen2",
    "nomic-dense": "nomic",
    "colqwen3": "tomoro-colqwen3",
    "tomoro": "tomoro-colqwen3",
    "modern-vbert": "modernvbert",
    "col-modernvbert": "colmodernvbert",
}


def available_retrievers() -> list[str]:
    """Return the sorted list of canonical retriever names."""
    return sorted(RETRIEVERS)


def get_retriever(name: str, **kwargs: Any) -> BasePageRetriever:
    """Instantiate a retriever by (case-insensitive) name or alias."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        cls = RETRIEVERS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown retriever {name!r}. Available: {available_retrievers()} "
            f"(aliases: {sorted(_ALIASES)})."
        ) from exc
    return cls(**kwargs)
