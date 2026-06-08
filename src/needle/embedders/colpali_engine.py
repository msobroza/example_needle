"""Loaders for real ``colpali-engine`` model backends.

These helpers centralise the (lazy) construction of the model + processor pairs
used by the concrete retrievers, mirroring their ``_load_model`` logic. Importing
this module is cheap: ``torch`` and ``colpali_engine`` are only imported inside
the loader functions, so nothing here requires a GPU or model weights at import
time.

Install the backends with::

    pip install -e ".[retrieval]"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class BackendSpec:
    """Names of the ``colpali_engine`` model + processor classes for a backend."""

    model_cls: str
    processor_cls: str
    multi_vector: bool


#: Logical backend name -> the colpali-engine classes that implement it.
BACKEND_SPECS: dict[str, BackendSpec] = {
    "colpali": BackendSpec("ColPali", "ColPaliProcessor", True),
    "colqwen2": BackendSpec("ColQwen2", "ColQwen2Processor", True),
    "colmodernvbert": BackendSpec("ColModernVBert", "ColModernVBertProcessor", True),
    "nomic": BackendSpec("BiIdefics3", "BiIdefics3Processor", False),
    "modernvbert": BackendSpec("BiModernVBert", "BiModernVBertProcessor", False),
}


def resolve_device(device: Optional[str] = None) -> str:
    """Return an explicit device, defaulting to CUDA when available."""
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch optional
        return "cpu"


def load_backend(
    backend: str,
    model_name: str,
    *,
    device: Optional[str] = None,
) -> tuple[Any, Any]:
    """Load ``(model, processor)`` for a named backend from ``model_name``.

    Requires the optional ``[retrieval]`` extra (``torch`` + ``colpali-engine``).
    """
    if backend not in BACKEND_SPECS:
        raise KeyError(f"Unknown backend {backend!r}. Known: {sorted(BACKEND_SPECS)}")
    spec = BACKEND_SPECS[backend]
    device = resolve_device(device)

    import torch  # lazy: only needed when actually loading a model
    from colpali_engine import models as cpm

    model_cls = getattr(cpm, spec.model_cls)
    processor_cls = getattr(cpm, spec.processor_cls)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = model_cls.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()
    processor = processor_cls.from_pretrained(model_name)
    return model, processor


__all__ = ["BackendSpec", "BACKEND_SPECS", "resolve_device", "load_backend"]
