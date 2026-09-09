"""VectorStorePort on chromadb (persistent on disk, or ephemeral for ``:memory:``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from typing import Any

from ..contracts.models import MetadataValue, Passage, ScoredPassage
from .executor import run_blocking

# Collection-level metadata key recording which embedder produced the vectors,
# so the service can refuse to serve an index built with a different model.
EMBEDDER_MODEL_KEY = "embedder_model"
# Chroma rejects empty per-row metadata dicts, so every row carries its id here.
PASSAGE_ID_KEY = "passage_id"
IN_MEMORY_PATH = ":memory:"
_QUERY_INCLUDE = ["documents", "metadatas", "distances"]


class ChromaVectorStore:
    """Chroma-backed vector store; all collection calls run on ``executor``.

    Example::

        store = ChromaVectorStore.open(
            "./data/chroma", "rag_passages", executor, embedder_model="bge-small"
        )
        await store.upsert(passages, vectors)
        hits = await store.query(query_vector, top_k=20)
    """

    def __init__(self, collection: Any, executor: Executor) -> None:
        self._collection = collection
        self._executor = executor

    @classmethod
    def open(
        cls,
        path: str,
        collection_name: str,
        executor: Executor,
        *,
        embedder_model: str | None = None,
    ) -> ChromaVectorStore:
        """Open (or create) ``collection_name``; ``path == ":memory:"`` is ephemeral."""
        collection = _open_client(path).get_or_create_collection(
            name=collection_name, metadata=_collection_metadata(embedder_model)
        )
        return cls(collection, executor)

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        rows = list(passages)
        vectors = [list(e) for e in embeddings]
        if len(rows) != len(vectors):
            raise ValueError(
                "passages/embeddings length mismatch: " f"{len(rows)} != {len(vectors)}"
            )
        if not rows:
            return
        await run_blocking(self._executor, self._upsert_rows, rows, vectors)

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        # Chroma warns (older versions raise) when n_results exceeds the row count.
        n_results = min(top_k, await self.count())
        if n_results <= 0:
            return []
        result = await run_blocking(
            self._executor, self._query_rows, list(embedding), n_results
        )
        return _to_scored_passages(result)

    async def count(self) -> int:
        return await run_blocking(self._executor, self._collection.count)

    def embedder_model(self) -> str | None:
        value = (self._collection.metadata or {}).get(EMBEDDER_MODEL_KEY)
        return None if value is None else str(value)

    def _upsert_rows(self, rows: list[Passage], vectors: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[p.id for p in rows],
            documents=[p.text for p in rows],
            embeddings=vectors,
            metadatas=[{PASSAGE_ID_KEY: p.id, **p.metadata} for p in rows],
        )

    def _query_rows(self, embedding: list[float], n_results: int) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=[embedding], n_results=n_results, include=_QUERY_INCLUDE
        )


def _open_client(path: str) -> Any:
    # Imported here so ``import rag_load_test`` stays cheap for the OVMS apps.
    import chromadb
    from chromadb.config import Settings

    settings = Settings(anonymized_telemetry=False)
    if path == IN_MEMORY_PATH:
        return chromadb.EphemeralClient(settings=settings)
    return chromadb.PersistentClient(path=path, settings=settings)


def _collection_metadata(embedder_model: str | None) -> dict[str, str]:
    metadata = {"hnsw:space": "cosine"}
    if embedder_model is not None:
        metadata[EMBEDDER_MODEL_KEY] = embedder_model
    return metadata


def _to_scored_passages(result: Mapping[str, Any]) -> list[ScoredPassage]:
    # Chroma nests one list per query embedding; we always send exactly one.
    columns = zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
        strict=True,
    )
    return [
        ScoredPassage(
            id=passage_id,
            text=document,
            metadata=_strip_passage_id(metadata),
            # cosine distance = 1 - cosine similarity
            retrieval_score=1.0 - float(distance),
        )
        for passage_id, document, metadata, distance in columns
    ]


def _strip_passage_id(
    metadata: Mapping[str, MetadataValue] | None,
) -> dict[str, MetadataValue]:
    return {k: v for k, v in (metadata or {}).items() if k != PASSAGE_ID_KEY}
