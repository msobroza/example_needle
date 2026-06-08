"""Preprocessing subpackage — torch-free image and batching helpers.

Image operations import Pillow lazily, so importing this package stays cheap
and dependency-light.
"""

from __future__ import annotations

from .batching import batched
from .image_ops import ensure_max_side, mean_color, resize, to_rgb

__all__ = [
    "to_rgb",
    "resize",
    "ensure_max_side",
    "mean_color",
    "batched",
]
