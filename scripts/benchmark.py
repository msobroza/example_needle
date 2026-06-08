#!/usr/bin/env python3
"""Micro-benchmark of index + search throughput using the dummy embedder.

This measures the *pipeline* overhead (extraction, scoring, filtering,
normalisation), not any real model — handy for spotting regressions.

    python scripts/benchmark.py --pages 500 --queries 50
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path


def synth_pages(workdir: Path, n: int) -> list:
    from PIL import Image

    from needle.retrieval.data import InputDocument

    documents = []
    for i in range(n):
        path = workdir / f"page_{i:05d}.png"
        # Deterministic-ish colour so embeddings differ between pages.
        Image.new("RGB", (64, 80), (i * 7 % 256, i * 13 % 256, i * 29 % 256)).save(path)
        documents.append(InputDocument.from_path(path, metadata={"bucket": i % 10}))
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=200)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--multi-vector", action="store_true", default=True)
    args = parser.parse_args()

    from conversational_core.domain.interaction.query import Query
    from needle.testing import DummyEmbedderRetriever

    workdir = Path(tempfile.mkdtemp(prefix="needle-bench-"))
    documents = synth_pages(workdir, args.pages)

    retriever = DummyEmbedderRetriever(
        index_path=str(workdir / "index.pkl"),
        batch_size=16,
        multi_vector=args.multi_vector,
    )

    t0 = time.perf_counter()
    retriever.index(documents, save=False)
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(args.queries):
        retriever.search(
            Query.of(f"query number {i}", filters={"bucket": i % 10}), top_k=10
        )
    t_search = time.perf_counter() - t0

    print(f"pages          : {len(retriever)}")
    print(f"index time     : {t_index:.3f}s ({len(retriever)/t_index:.0f} pages/s)")
    print(f"search time    : {t_search:.3f}s ({args.queries/t_search:.0f} queries/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
