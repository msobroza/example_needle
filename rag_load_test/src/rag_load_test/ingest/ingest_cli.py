"""``rag-ingest``: load or generate a corpus, chunk it, embed it, upsert into the store.

Example::

    rag-ingest --synthetic 200            # embed with the RAG_* configured backend
    rag-ingest --corpus docs.jsonl --fake # FakeEmbedder + in-memory store (demo)

Adapters (sentence-transformers, Chroma, OVMS) are imported inside the helper
that needs them so ``--fake`` runs, and importing this module, stay cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from concurrent.futures import Executor
from contextlib import AsyncExitStack
from pathlib import Path

from ..contracts.models import Passage
from ..contracts.ports import EmbedderPort, VectorStorePort
from ..errors import RagConfigError
from ..settings import RagSettings
from ..testing.fakes import FakeEmbedder, InMemoryVectorStore
from .corpus import SourceDocument, chunk_document, load_jsonl_corpus, synthetic_corpus

FAKE_COLLECTION_LABEL = "in-memory"


async def ingest_passages(
    passages: Sequence[Passage],
    embedder: EmbedderPort,
    store: VectorStorePort,
    *,
    batch_size: int = 64,
) -> int:
    """Embed passages in batches and upsert them; returns the number upserted.

    Example::

        count = await ingest_passages(passages, FakeEmbedder(), InMemoryVectorStore())
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    upserted = 0
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        embeddings = await embedder.embed_documents([p.text for p in batch])
        await store.upsert(batch, embeddings)
        upserted += len(batch)
    return upserted


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``rag-ingest``.

    Example::

        args = build_parser().parse_args(["--synthetic", "50", "--fake"])
    """
    parser = argparse.ArgumentParser(
        prog="rag-ingest", description="Chunk, embed and upsert a corpus."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", type=int, default=200, help="number of synthetic documents"
    )
    source.add_argument("--corpus", type=Path, default=None, help="JSONL corpus file")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunk-words", type=int, default=120)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use FakeEmbedder and an in-memory store (tests/demos)",
    )
    return parser


def _load_documents(args: argparse.Namespace) -> list[SourceDocument]:
    if args.corpus is not None:
        return load_jsonl_corpus(args.corpus)
    return synthetic_corpus(args.synthetic, seed=args.seed)


def _chunk_all(
    docs: Sequence[SourceDocument], args: argparse.Namespace
) -> list[Passage]:
    return [
        passage
        for doc in docs
        for passage in chunk_document(
            doc, chunk_words=args.chunk_words, overlap_words=args.overlap_words
        )
    ]


def _open_executor(settings: RagSettings, stack: AsyncExitStack) -> Executor:
    from ..adapters.executor import build_model_executor

    executor = build_model_executor(settings.model_threads)
    stack.callback(executor.shutdown, True)
    return executor


def _local_embedder(settings: RagSettings, executor: Executor) -> EmbedderPort:
    from ..adapters.local_embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder.from_pretrained(
        settings.embedder_model, executor
    )


async def _ovms_embedder(settings: RagSettings, stack: AsyncExitStack) -> EmbedderPort:
    import httpx

    from ..adapters.auth import build_token_provider
    from ..adapters.ovms_embedder import OvmsEmbedder

    client = await stack.enter_async_context(httpx.AsyncClient())
    return OvmsEmbedder(
        client,
        settings.ovms_embeddings_url,
        settings.ovms_embeddings_model,
        token_provider=build_token_provider(settings, client),
        timeout_s=settings.http_timeout_s,
        max_retries=settings.http_max_retries,
    )


async def _build_embedder(
    settings: RagSettings, executor: Executor, stack: AsyncExitStack
) -> EmbedderPort:
    backend = settings.embedder_backend
    if backend == "fake":
        return FakeEmbedder()
    if backend == "local":
        return _local_embedder(settings, executor)
    if backend == "ovms":
        return await _ovms_embedder(settings, stack)
    raise RagConfigError(
        "RAG_EMBEDDER_BACKEND", got=backend, expected="one of local, ovms, fake"
    )


def _open_store(
    settings: RagSettings, executor: Executor, embedder_model: str
) -> VectorStorePort:
    from ..adapters.chroma_store import ChromaVectorStore

    return ChromaVectorStore.open(
        settings.chroma_path,
        settings.chroma_collection,
        executor,
        embedder_model=embedder_model,
    )


async def _ingest_fake(
    passages: Sequence[Passage], batch_size: int
) -> dict[str, object]:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    count = await ingest_passages(passages, embedder, store, batch_size=batch_size)
    return _summary(count, FAKE_COLLECTION_LABEL, embedder.model_name)


async def _ingest_configured(
    passages: Sequence[Passage], settings: RagSettings, batch_size: int
) -> dict[str, object]:
    async with AsyncExitStack() as stack:
        executor = _open_executor(settings, stack)
        embedder = await _build_embedder(settings, executor, stack)
        store = _open_store(settings, executor, embedder.model_name)
        count = await ingest_passages(passages, embedder, store, batch_size=batch_size)
    return _summary(count, settings.chroma_collection, embedder.model_name)


def _summary(count: int, collection: str, embedder: str) -> dict[str, object]:
    return {"passages": count, "collection": collection, "embedder": embedder}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``rag-ingest``; prints one JSON summary line and returns 0.

    Example::

        main(["--synthetic", "3", "--fake"])
        # {"event": "ingest_done", "documents": 3, "passages": 3, ...}
    """
    args = build_parser().parse_args(argv)
    settings = RagSettings.from_env()
    documents = _load_documents(args)
    passages = _chunk_all(documents, args)
    if args.fake:
        summary = asyncio.run(_ingest_fake(passages, args.batch_size))
    else:
        summary = asyncio.run(_ingest_configured(passages, settings, args.batch_size))
    print(json.dumps({"event": "ingest_done", "documents": len(documents), **summary}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
