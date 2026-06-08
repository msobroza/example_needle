"""Lightweight, torch-free image preprocessing helpers.

Pillow is imported lazily inside each function so that importing this module
(and the package) does not require Pillow until an operation is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


def to_rgb(image: Image.Image) -> Image.Image:
    """Return ``image`` converted to RGB mode."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize ``image`` to ``size`` (``(width, height)``)."""
    return image.resize(size)


def ensure_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale ``image`` so its longest side is ``<= max_side``.

    The aspect ratio is preserved. Images already within ``max_side`` are
    returned unchanged.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size)


def mean_color(image: Image.Image) -> np.ndarray:
    """Return the mean RGB colour of ``image`` as floats in ``[0, 1]``.

    The result is an array of shape ``(3,)``.
    """
    arr = np.asarray(to_rgb(image), dtype=np.float64) / 255.0
    return arr.reshape(-1, 3).mean(axis=0)
