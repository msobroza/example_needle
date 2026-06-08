"""Compose a retrieval pipeline from interchangeable parts (no model weights).

python examples/pipeline_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from needle.factory import build_pipeline
from needle.indexing import PickleIndexStore
from needle.retrieval.data import InputDocument
from needle_core.domain.interaction.query import Query


def make_page(path: Path, text: str, color: str) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (300, 380), color)
    ImageDraw.Draw(image).text((16, 16), text, fill="black")
    image.save(path)
    return path


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="needle-pipeline-"))
    documents = [
        InputDocument.from_path(
            make_page(workdir / "a.png", "INVOICE 2021", "white"),
            metadata={"year": 2021, "lang": "en"},
        ),
        InputDocument.from_path(
            make_page(workdir / "b.png", "REPORT 2024", "lightyellow"),
            metadata={"year": 2024, "lang": "en"},
        ),
        InputDocument.from_path(
            make_page(workdir / "c.png", "MEMO 2024", "lightblue"),
            metadata={"year": 2024, "lang": "fr"},
        ),
    ]

    # A pipeline = embedder + store + scorer + extractors. Here we plug in a
    # persistent PickleIndexStore; the embedder defaults to the weight-free
    # DeterministicEmbedder.
    store = PickleIndexStore()
    pipeline = build_pipeline(store=store)
    pipeline.index(documents)
    print(f"Indexed {len(pipeline)} pages — {pipeline!r}\n")

    query = Query.of("annual report", filters={"lang": "en"}).with_filter(
        "year", 2024, "gte"
    )
    for rank, hit in enumerate(pipeline.search(query, top_k=5), 1):
        print(
            f"  {rank}. {hit.document_version.filename} "
            f"p{hit.page} score={hit.score:.3f}"
        )

    index_path = workdir / "index.pkl"
    pipeline.store.save(index_path)
    reloaded = PickleIndexStore().load(index_path)
    print(f"\nPersisted and reloaded {len(reloaded)} pages from {index_path}")


if __name__ == "__main__":
    main()
