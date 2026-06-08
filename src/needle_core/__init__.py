"""needle_core — domain model for conversational document retrieval.

This package holds the framework-agnostic, dependency-light *domain* layer:
documents, queries, metadata filters and the value types shared across the
retrieval pipeline. It deliberately avoids heavyweight ML dependencies so it
can be imported anywhere (services, tests, notebooks) without pulling in
PyTorch or model weights.
"""

from __future__ import annotations

from .domain.types import DocumentExtension

__all__ = ["DocumentExtension", "__version__"]

__version__ = "0.1.0"
