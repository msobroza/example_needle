# Retrievers

A retriever is responsible for turning page images into embeddings, indexing
them, and scoring a query against the index. All built-in retrievers subclass
`needle.retrieval.page_retrievers.MultimodalEmbedderRetriever`, which means they
share the same `index()` / `search()` / persistence machinery and differ only in:

1. which model + processor they load (`_load_model`),
2. how they embed images and queries (`_embed_images`, `_embed_query`), and
3. whether they are **multi-vector** or **dense** (the `multi_vector` flag).

## Built-in retrievers

| Retriever | `model_name` default | `multi_vector` | `colpali_engine` model / processor |
| --- | --- | --- | --- |
| `ColPaliRetriever` | `/domino/datasets/ModelHub-model-huggingface-vidore/colpali/main/` | `True` | `ColPali` / `ColPaliProcessor` |
| `ColQwen2Retriever` | `/domino/datasets/ModelHub-model-huggingface-vidore/colqwen2-v1.0-hf/main/` | `True` | `ColQwen2` / `ColQwen2Processor` |
| `TomoroColQwen3Retriever` | `TomoroAI/tomoro-colqwen3-embed-4b` | `True` | `ColQwen2` / `ColQwen2Processor` |
| `ColModernVBertRetriever` | `ModernVBERT/colmodernvbert` | `True` | `ColModernVBert` / `ColModernVBertProcessor` |
| `NomicDenseRetriever` | `nomic-ai/nomic-embed-multimodal-3b` | `False` | `BiIdefics3` / `BiIdefics3Processor` |
| `ModernVBertRetriever` | `ModernVBERT/modernvbert` | `False` | `BiModernVBert` / `BiModernVBertProcessor` |

> The two ColPali/ColQwen defaults point at a local model-hub mount. Override
> them with the `model_name=` argument (e.g. a Hugging Face repo id) when running
> elsewhere.

Every retriever accepts the same constructor keyword arguments via
`MultimodalEmbedderRetriever.__init__`:

```python
ColQwen2Retriever(
    model_name=...,          # model-specific default, override as needed
    index_path="index.pkl",  # where the pickled index lives
    batch_size=4,            # pages embedded per forward pass
    dpi=150,                 # rasterisation DPI (see Extractors)
    extractors=IMAGE_EXTRACTORS,  # extension → extractor map (see Extractors)
)
```

If `index_path` already exists, it is loaded automatically in `__init__`.

### Selecting a retriever by name

The registry lets you pick a backend by string without importing the class:

```python
from needle.retrieval.registry import (
    RETRIEVERS,            # {"colpali": ColPaliRetriever, ...}
    get_retriever,
    available_retrievers,  # sorted list of canonical names
)

print(available_retrievers())
# ['colmodernvbert', 'colpali', 'colqwen2', 'modernvbert', 'nomic', 'tomoro-colqwen3']

retriever = get_retriever("colqwen2", index_path="index.pkl")
```

`get_retriever` is case-insensitive and understands aliases: `colqwen` →
`colqwen2`, `nomic-dense` → `nomic`, `colqwen3`/`tomoro` → `tomoro-colqwen3`,
`modern-vbert` → `modernvbert`, `col-modernvbert` → `colmodernvbert`.

## Scoring: multi-vector vs dense

The base class picks the scoring function from the `multi_vector` flag:

```python
def _score(self, q_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    doc = doc_emb.astype(np.float32)
    if self.multi_vector:
        return (q_emb @ doc.T).max(axis=1).sum()
    return float(q_emb @ doc / (np.linalg.norm(q_emb) * np.linalg.norm(doc) + 1e-9))
```

### Multi-vector late-interaction (MaxSim)

Late-interaction models (ColPali, ColQwen2, ColModernVBert, TomoroColQwen3)
represent **each page as a matrix** of shape `(num_page_tokens, dim)` and **each
query as a matrix** of shape `(num_query_tokens, dim)`. The MaxSim score is:

```python
score = (q_emb @ doc.T).max(axis=1).sum()
```

For every query token, take its maximum similarity against any page token
(`.max(axis=1)`), then sum those per-token maxima (`.sum()`). This lets individual
query terms "find" the most relevant patch of the page, which is what makes
late-interaction strong on dense, visually-structured documents — at the cost of
storing one vector *per token per page*.

### Dense single-vector (cosine)

Dense models (`NomicDenseRetriever`, `ModernVBertRetriever`) represent **each page
as a single vector** of shape `(dim,)` and **each query as a single vector**
`(dim,)`. Scoring is plain cosine similarity. This is faster and far cheaper to
store (one vector per page), at some recall cost on layout-heavy pages.

### Comparison

| Property | Multi-vector (late-interaction) | Dense (single-vector) |
| --- | --- | --- |
| `multi_vector` | `True` | `False` |
| Per-page embedding shape | `(num_tokens, dim)` | `(dim,)` |
| Per-query embedding shape | `(num_query_tokens, dim)` | `(dim,)` |
| Scoring | MaxSim: `(q @ doc.T).max(axis=1).sum()` | Cosine: `q·doc / (‖q‖‖doc‖)` |
| Storage per page | Large (many vectors) | Small (one vector) |
| Quality on layout/figures | Higher | Lower |
| Examples | ColPali, ColQwen2, ColModernVBert, TomoroColQwen3 | Nomic, ModernVBert |

The `_minmax` normalisation that runs after scoring is identical for both regimes
— it only rescales raw scores to `[0, 1]` for display/ranking convenience.

## Adding your own retriever

Subclass `MultimodalEmbedderRetriever`, set `multi_vector`, and implement the
three hooks. You inherit `index()`, `search()`, persistence and filtering for
free.

```python
import numpy as np
import torch
from needle.retrieval.page_retrievers import MultimodalEmbedderRetriever


class MyVisualRetriever(MultimodalEmbedderRetriever):
    """A multi-vector late-interaction retriever wrapping my own model."""

    multi_vector = True  # use MaxSim scoring

    def __init__(self, model_name: str = "my-org/my-visual-encoder", **kwargs):
        self.model_name = model_name
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        # Imported lazily so torch/model deps are only needed when used.
        from my_library import MyModel, MyProcessor

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = MyModel.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = MyProcessor.from_pretrained(self.model_name)

    def _embed_images(self, images: list) -> np.ndarray:
        # Returns one (num_tokens, dim) array per image (multi_vector=True).
        with torch.no_grad():
            batch = self.processor.process_images(images).to(self.device)
            return self.model(**batch).cpu().float().numpy().astype(np.float16)

    def _embed_query(self, query: str) -> np.ndarray:
        # Returns a single (num_query_tokens, dim) array (multi_vector=True).
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.device)
            return self.model(**batch)[0].cpu().float().numpy()
```

For a **dense** variant, set `multi_vector = False` and return one `(dim,)` vector
per image and one `(dim,)` vector for the query.

To expose your retriever by name, register it in
`needle/retrieval/registry.py`. See
[Contributing a retriever](contributing-a-retriever.md) for the full contract,
input/output shapes, and a worked example built on
`needle.testing.DummyEmbedderRetriever`.
