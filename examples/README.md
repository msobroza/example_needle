# Examples

These scripts run with **no model weights** — they use deterministic toy
embedders so you can explore the API offline.

| Script | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Index a few generated pages with `DummyEmbedderRetriever`, then search with and without metadata filters. |
| [`custom_retriever.py`](custom_retriever.py) | Subclass `MultimodalEmbedderRetriever` with a toy *colour* embedder, so searching `"red"` actually finds the reddest page (a real, if tiny, multi-vector MaxSim retriever). |

Run them directly:

```bash
python examples/quickstart.py
python examples/custom_retriever.py
```

For the real model backends (ColPali, ColQwen2, …) install the extra and swap
in a concrete retriever:

```bash
pip install -e ".[retrieval]"
```

```python
from needle import ColQwen2Retriever
retriever = ColQwen2Retriever(index_path="index.pkl")
```
