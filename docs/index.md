# example_needle

**Page-level multimodal document retrieval.**

`example_needle` indexes and searches documents by treating each **page as an
image**. Instead of extracting text and embedding word tokens, it rasterises
every page (PDF, DOCX, PPTX, PNG, …) into a picture and embeds that picture with
a *visual* document encoder such as
[ColPali](https://huggingface.co/vidore/colpali) or
[ColQwen2](https://huggingface.co/vidore/colqwen2-v1.0). Layout, tables, charts
and figures are preserved because nothing is ever flattened to plain text.

At query time your natural-language question is embedded by the same model, every
indexed page is scored, the scores are min–max normalised, an optional
**metadata filter** is applied as a boolean mask, and the top-*k* pages are
returned.

Two embedding regimes are supported:

* **Multi-vector / late-interaction** (ColPali, ColQwen2, …) — each page is a
  *set* of token vectors; scoring uses MaxSim.
* **Dense single-vector** (Nomic, ModernVBert) — each page is one vector; scoring
  uses cosine similarity.

## Package layout

The project ships two packages:

* **`conversational_core`** — a dependency-light domain layer (`Document`,
  `DocumentVersion`, `DocumentPage`, `Query`, metadata filters). No torch.
* **`needle`** — the retrieval layer (extractors, retrievers, registry). Pulls in
  torch, but only when a retriever is actually touched.

See [Architecture](architecture.md) for the full picture.

## Table of contents

| Page | What it covers |
| --- | --- |
| [Architecture](architecture.md) | The two-package design, the retriever class hierarchy, lazy torch imports, and the `index()` / `search()` pipelines. |
| [Retrievers](retrievers.md) | Every built-in retriever, multi-vector vs dense scoring, and how to add your own. |
| [Filters](filters.md) | Building metadata filters on a `Query` and how they become a Mongo-style backend dict. |
| [Extractors](extractors.md) | Turning files into page images, supported formats, DPI, optional deps, and custom extractors. |
| [Contributing a retriever](contributing-a-retriever.md) | A step-by-step tutorial on subclassing `MultimodalEmbedderRetriever`. |

## Quickstart

The snippet below uses `needle.testing.DummyEmbedderRetriever`, a fully-working
retriever whose "model" produces deterministic embeddings from the input bytes —
so it runs with **no model weights, no network and no GPU**.

```python
from needle.retrieval.data import InputDocument
from needle.testing import DummyEmbedderRetriever
from conversational_core.domain.interaction.query import Query

retriever = DummyEmbedderRetriever(index_path="demo.pkl")        # no weights needed
docs = [InputDocument.from_path("report.png", metadata={"year": 2023})]
retriever.index(docs)                                            # extract → embed → store

query = Query.of("quarterly revenue chart", filters={"year": 2023})
for hit in retriever.search(query, top_k=5):                     # embed → score → filter → top_k
    print(hit.score, hit.document_version.filename, "page", hit.page)
```

> **Tip:** Swap `DummyEmbedderRetriever` for `ColQwen2Retriever` (or any other
> built-in) to run against a real model — the `index()` / `search()` API is
> identical. See [Retrievers](retrievers.md).
