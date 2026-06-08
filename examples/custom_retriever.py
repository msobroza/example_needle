"""A tiny but *real* custom retriever.

``ColorPatchRetriever`` is a multi-vector retriever whose "tokens" are the mean
colours of a grid of image patches, and whose query embedding maps colour words
to RGB vectors. MaxSim late interaction then makes a search for ``"red"`` rank
the reddest page first — demonstrating exactly how to extend
``MultimodalEmbedderRetriever`` with your own ``_load_model`` / ``_embed_images``
/ ``_embed_query``.

    python examples/custom_retriever.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from conversational_core.domain.interaction.query import Query
from needle.retrieval.data import InputDocument
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever

_COLOR_WORDS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}


class ColorPatchRetriever(MultimodalEmbedderRetriever):
    """Multi-vector retriever over mean patch colours (a toy, but functional)."""

    multi_vector = True

    def __init__(self, *, grid: int = 4, **kwargs) -> None:
        self.grid = grid
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        # No model to load — the "embedding" is pure arithmetic on pixels.
        self.model = None
        self.processor = None

    def _embed_images(self, images: list) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for image in images:
            small = image.convert("RGB").resize((self.grid, self.grid))
            patches = np.asarray(small, dtype=np.float32).reshape(-1, 3) / 255.0
            out.append(patches.astype(np.float16))  # (grid*grid, 3)
        return out

    def _embed_query(self, query: str) -> np.ndarray:
        vectors = [
            _COLOR_WORDS[word] for word in query.lower().split() if word in _COLOR_WORDS
        ]
        if not vectors:
            vectors = [(0.5, 0.5, 0.5)]  # neutral grey if no colour word found
        return np.asarray(vectors, dtype=np.float32)  # (num_color_words, 3)


def _solid_page(path: Path, color: tuple[int, int, int]) -> Path:
    from PIL import Image

    Image.new("RGB", (128, 160), color).save(path)
    return path


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="needle-color-"))
    documents = [
        InputDocument.from_path(
            _solid_page(workdir / "red.png", (220, 20, 20)), metadata={"hue": "red"}
        ),
        InputDocument.from_path(
            _solid_page(workdir / "green.png", (20, 200, 20)), metadata={"hue": "green"}
        ),
        InputDocument.from_path(
            _solid_page(workdir / "blue.png", (20, 20, 220)), metadata={"hue": "blue"}
        ),
    ]

    retriever = ColorPatchRetriever(index_path=str(workdir / "index.pkl"), grid=4)
    retriever.index(documents)

    for query_text in ("red", "blue", "green"):
        hits = retriever.search(Query.of(query_text), top_k=3)
        ranking = ", ".join(
            f"{h.document_version.filename}={h.score:.2f}" for h in hits
        )
        winner = hits[0].document_version.filename
        print(f"query {query_text!r:8} -> winner: {winner:9}  ({ranking})")


if __name__ == "__main__":
    main()
