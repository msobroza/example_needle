"""rag_load_test — one RAG workflow, three deployment topologies, one Locust suite.

The package is deliberately independent from ``needle``/``needle_core``: it is a
text RAG (sentence-transformers + Chroma + cross-encoder reranker + OpenAI)
orchestrated by LangGraph and exposed by FastAPI, built to be load tested on
Domino Data Lab as a monolith or split across OpenVINO Model Server apps.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
