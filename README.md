# example_needle

> Find the needle in the haystack — **page-level multimodal document retrieval**.

`example_needle` indexes documents by **embedding each page as an image** with
modern late-interaction / dense visual embedders (ColPali, ColQwen2, Nomic,
ModernVBERT, …) and lets you search them with natural language plus structured
metadata filters. No OCR, no brittle text extraction — the model looks at the
page the way a human does.

```text
   PDF / DOCX / PNG ...            "what was Q3 revenue?"  +  {year >= 2024}
          │                                        │
   ┌──────▼───────┐   rasterise   ┌──────────────┐ │  embed   ┌───────────────┐
   │  extractors  ├──────────────▶│  page images  ├─┼─────────▶│  query vector │
   └──────────────┘               └──────┬────────┘ │         └───────┬───────┘
                                         │ embed     │                 │ MaxSim / cosine
                                  ┌──────▼────────┐  │   filter mask   ▼
                                  │  page vectors │──┴───────────▶  top-k pages
                                  └───────────────┘
```

## Why two packages?

| Package | Responsibility | Heavy deps? |
|---|---|---|
| **`conversational_core`** | Framework-agnostic *domain*: `Document`, `DocumentVersion`, `DocumentPage`, `Query`, and the metadata-filter model. | No — pure Python. |
| **`needle`** | The retrieval layer: page extractors, the `BasePageRetriever` hierarchy and concrete model backends. | Yes — PyTorch (loaded lazily). |

Importing `needle` is cheap; PyTorch is only imported the first time you touch
an actual retriever, so the domain model and helpers stay usable in lightweight
contexts (services, tests, notebooks).

## Install

```bash
pip install -e .                 # core: numpy, pillow, tqdm
pip install -e ".[retrieval]"    # + torch + colpali-engine + transformers
pip install -e ".[pdf]"          # + PyMuPDF, to rasterise real PDFs
pip install -e ".[dev]"          # + pytest, ruff, black, mypy
```

## Quickstart

The deterministic `DummyEmbedderRetriever` runs the **entire** index/search
pipeline with no model weights — perfect for trying the API or for CI:

```python
from conversational_core.domain.interaction.query import Query
from needle.retrieval.data import InputDocument
from needle.testing import DummyEmbedderRetriever

retriever = DummyEmbedderRetriever(index_path="index.pkl")

retriever.index([
    InputDocument.from_path("invoice.png", metadata={"year": 2021, "lang": "en"}),
    InputDocument.from_path("report.png",  metadata={"year": 2024, "lang": "en"}),
])

# Natural-language query, restricted to recent English documents.
query = Query.of("what was the total?", filters={"lang": "en"}).with_filter(
    "year", 2024, "gte"
)
for hit in retriever.search(query, top_k=5):
    print(hit.score, hit.document_version.filename, "page", hit.page)
```

Run it for real:

```bash
python examples/quickstart.py
```

## Using a real model

```python
from needle import ColQwen2Retriever          # lazy — imports torch here
from needle.retrieval.data import InputDocument

retriever = ColQwen2Retriever(index_path="index.pkl", dpi=150, batch_size=4)
retriever.index([InputDocument.from_path("report.pdf")])   # needs the [pdf] extra
results = retriever.search(Query.of("revenue by region"), top_k=5)
retriever.show([r.to_annotation_result() for r in results])
```

Or pick a backend by name:

```python
from needle.retrieval.registry import get_retriever, available_retrievers

print(available_retrievers())
# ['colmodernvbert', 'colpali', 'colqwen2', 'modernvbert', 'nomic', 'tomoro-colqwen3']

retriever = get_retriever("colpali", index_path="index.pkl")
```

## Built-in retrievers

| Name | Class | Scoring | colpali-engine model |
|---|---|---|---|
| `colpali` | `ColPaliRetriever` | multi-vector (MaxSim) | `ColPali` |
| `colqwen2` | `ColQwen2Retriever` | multi-vector (MaxSim) | `ColQwen2` |
| `tomoro-colqwen3` | `TomoroColQwen3Retriever` | multi-vector (MaxSim) | `ColQwen2` |
| `colmodernvbert` | `ColModernVBertRetriever` | multi-vector (MaxSim) | `ColModernVBert` |
| `nomic` | `NomicDenseRetriever` | dense (cosine) | `BiIdefics3` |
| `modernvbert` | `ModernVBertRetriever` | dense (cosine) | `BiModernVBert` |

## Add your own retriever

Subclass `MultimodalEmbedderRetriever` and implement three methods:

```python
import numpy as np
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever

class MyRetriever(MultimodalEmbedderRetriever):
    multi_vector = True   # MaxSim late interaction; set False for dense cosine

    def _load_model(self): ...
    def _embed_images(self, images) -> list[np.ndarray]: ...   # one (T, D) array per image
    def _embed_query(self, query: str) -> np.ndarray: ...      # one (Tq, D) array
```

See [`docs/contributing-a-retriever.md`](docs/contributing-a-retriever.md) and
the worked example in [`src/needle/testing.py`](src/needle/testing.py).

## Command line

```bash
needle list                                  # list backends
needle index report.pdf --retriever colqwen2 --index index.pkl
needle search "revenue by region" --index index.pkl --top-k 5
```

## Documentation

* [`docs/index.md`](docs/index.md) — start here
* [`docs/architecture.md`](docs/architecture.md) — how the pieces fit together
* [`docs/retrievers.md`](docs/retrievers.md) — the model backends
* [`docs/filters.md`](docs/filters.md) — metadata filtering
* [`docs/extractors.md`](docs/extractors.md) — turning files into page images
* [`docs/contributing-a-retriever.md`](docs/contributing-a-retriever.md)

## Development

```bash
make install-dev
make lint        # ruff + black --check
make test        # pytest (torch tests skip if torch is absent)
make cover       # pytest with coverage
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
