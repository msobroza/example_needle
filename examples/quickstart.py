"""Quickstart: index a few generated pages and search them.

Uses the deterministic ``DummyEmbedderRetriever`` so it runs with no model
weights, no network and no GPU.

    python examples/quickstart.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from needle.retrieval.data import InputDocument
from needle.testing import DummyEmbedderRetriever
from needle_core.domain.interaction.query import Query


def make_page(path: Path, text: str, color: str) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 400), color)
    ImageDraw.Draw(image).text((16, 16), text, fill="black")
    image.save(path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    workdir = Path(tempfile.mkdtemp(prefix="needle-quickstart-"))

    documents = [
        InputDocument.from_path(
            make_page(workdir / "invoice.png", "INVOICE 2021", "white"),
            metadata={"year": 2021, "lang": "en", "kind": "invoice"},
        ),
        InputDocument.from_path(
            make_page(workdir / "report.png", "ANNUAL REPORT 2024", "lightyellow"),
            metadata={"year": 2024, "lang": "en", "kind": "report"},
        ),
        InputDocument.from_path(
            make_page(workdir / "memo.png", "MEMO 2024", "lightblue"),
            metadata={"year": 2024, "lang": "fr", "kind": "memo"},
        ),
    ]

    retriever = DummyEmbedderRetriever(
        index_path=str(workdir / "index.pkl"), batch_size=2
    )
    retriever.index(documents)
    print(f"\nIndexed {len(retriever)} pages into {workdir/'index.pkl'}\n")

    print("== Search: no filter ==")
    for rank, hit in enumerate(retriever.search(Query.of("annual report"), top_k=3), 1):
        print(
            f"  {rank}. {hit.document_version.filename} (page {hit.page}) "
            f"score={hit.score:.3f} norm={hit.normalized_score:.3f}"
        )

    print("\n== Search: only English documents from 2024 or later ==")
    query = Query.of("annual report", filters={"lang": "en"}).with_filter(
        "year", 2024, "gte"
    )
    for rank, hit in enumerate(retriever.search(query, top_k=5), 1):
        print(
            f"  {rank}. {hit.document_version.filename} (page {hit.page}) "
            f"score={hit.score:.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
