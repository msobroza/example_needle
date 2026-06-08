"""Tests for :mod:`needle.preprocessing` (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from needle.preprocessing import (
    batched,
    ensure_max_side,
    mean_color,
    resize,
    to_rgb,
)


def test_to_rgb_converts_mode():
    grey = Image.new("L", (4, 4), color=128)
    out = to_rgb(grey)
    assert out.mode == "RGB"


def test_to_rgb_passthrough_for_rgb():
    rgb = Image.new("RGB", (4, 4), color=(10, 20, 30))
    out = to_rgb(rgb)
    assert out.mode == "RGB"


def test_resize_changes_size():
    image = Image.new("RGB", (8, 6))
    out = resize(image, (3, 2))
    assert out.size == (3, 2)


def test_ensure_max_side_caps_longest_and_preserves_aspect():
    image = Image.new("RGB", (200, 100))
    out = ensure_max_side(image, 50)
    assert max(out.size) <= 50
    # original aspect ratio is 2:1 -> width should stay double the height
    assert out.size == (50, 25)


def test_ensure_max_side_noop_when_within_limit():
    image = Image.new("RGB", (40, 30))
    out = ensure_max_side(image, 50)
    assert out.size == (40, 30)


def test_mean_color_solid_red():
    image = Image.new("RGB", (5, 5), color=(255, 0, 0))
    out = mean_color(image)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-9)


def test_batched_chunks():
    assert list(batched([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]


def test_batched_invalid_n_raises():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], 0))
