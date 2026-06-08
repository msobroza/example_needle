"""A composable, torch-free retrieval pipeline.

``RetrievalPipeline`` wires the pluggable building blocks of this package —
an :class:`~conversational_core.domain.ports.embedder.Embedder`, an index store,
a scoring strategy and the page extractors — into the same index/search flow
implemented by the concrete retrievers, but assembled from parts you choose.

Paired with :class:`needle.embedders.DeterministicEmbedder` it runs end to end
with no model weights, which is exactly how the examples and tests exercise it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np

from conversational_core.domain.exceptions import EmptyIndexError
from conversational_core.domain.interaction.query import Query
from conversational_core.domain.metadata.backend_filter_adapters import (
    MultiFieldFilterAdapter,
)
from conversational_core.domain.ports.embedder import Embedder
from conversational_core.domain.types import DocumentExtension

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_DPI, DEFAULT_TOP_K
from .indexing import InMemoryIndexStore
from .indexing.base import BaseIndexStore
from .preprocessing import batched
from .retrieval.data import InputDocument, PreannotationPageResult
from .retrieval.extractors import IMAGE_EXTRACTORS, PageToImageExtractor
from .retrieval.page_retriever_utils import matches_filter
from .scoring import ScoringStrategy, get_scorer, minmax_normalize

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Index and search page images using interchangeable components."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        store: Optional[BaseIndexStore] = None,
        scorer: Optional[ScoringStrategy] = None,
        extractors: Optional[dict[DocumentExtension, PageToImageExtractor]] = None,
        dpi: int = DEFAULT_DPI,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.embedder = embedder
        # Note: an empty store is falsy (it defines __len__), so compare to None.
        self.store: BaseIndexStore = (
            store if store is not None else InMemoryIndexStore()
        )
        self.scorer = scorer or get_scorer(embedder.multi_vector)
        self.extractors = extractors if extractors is not None else IMAGE_EXTRACTORS
        self.dpi = dpi
        self.batch_size = batch_size

    # -- indexing -------------------------------------------------------
    def index(
        self, documents: Sequence[InputDocument], reinit: bool = True
    ) -> RetrievalPipeline:
        if reinit:
            self.store.clear()
        for document in documents:
            self._index_one(document)
        logger.info("Indexed %d pages from %d documents", len(self), len(documents))
        return self

    def _index_one(self, document: InputDocument) -> None:
        doc = document.document
        version = document.document_version
        ext = doc.document_ext
        extractor = self.extractors[ext]
        path = Path(version.document_path)
        try:
            pages = extractor.extract(path, dpi=self.dpi)
        except Exception as exc:  # skip unreadable docs, like the retrievers do
            logger.warning("Skipping %s: %s", path.name, exc)
            return

        page_no = 0
        for chunk in batched(pages, self.batch_size):
            for embedding in self.embedder.embed_images(chunk):
                page_no += 1
                self.store.add(
                    np.asarray(embedding),
                    {
                        "file": path.name,
                        "format": ext,
                        "page": page_no,
                        "document": doc,
                        "document_version": version,
                        **version.document_metadata,
                    },
                )

    # -- search ---------------------------------------------------------
    def search(
        self, query: Query, top_k: int = DEFAULT_TOP_K
    ) -> list[PreannotationPageResult]:
        embeddings = self.store.embeddings()
        if not embeddings:
            raise EmptyIndexError("Index is empty. Call .index(...) first.")
        payloads = self.store.payloads()

        q_emb = self.embedder.embed_query(query.query_text)
        raw_scores = self.scorer.score_all(q_emb, embeddings)
        normalized = minmax_normalize(raw_scores)

        mask = np.ones(len(raw_scores), dtype=bool)
        filters = MultiFieldFilterAdapter().to_backend(
            spec=query.get_metadata_filter_spec()
        )
        for key, value in filters.items():
            mask &= np.array([matches_filter(p.get(key), value) for p in payloads])

        candidates = np.where(mask, raw_scores, -np.inf)
        normalized = np.where(mask, normalized, 0.0)
        top_idx = np.argsort(-candidates)[:top_k]
        return [
            PreannotationPageResult(
                query=query,
                document=payloads[i]["document"],
                document_version=payloads[i]["document_version"],
                page=payloads[i]["page"],
                score=float(candidates[i]),
                normalized_score=float(normalized[i]),
            )
            for i in top_idx
            if candidates[i] > -np.inf
        ]

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return (
            f"<RetrievalPipeline embedder={self.embedder!r} "
            f"store={self.store!r} pages={len(self)}>"
        )


__all__ = ["RetrievalPipeline"]
