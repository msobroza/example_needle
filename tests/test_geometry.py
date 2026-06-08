"""Tests for the BoundingBox geometry value object."""

from __future__ import annotations

import pytest

from conversational_core.domain.geometry import BoundingBox


def test_dimensions_and_area() -> None:
    box = BoundingBox(0, 0, 4, 2)
    assert box.width == 4
    assert box.height == 2
    assert box.area == 8


def test_invalid_x_coords() -> None:
    with pytest.raises(ValueError):
        BoundingBox(5, 0, 1, 1)


def test_invalid_y_coords() -> None:
    with pytest.raises(ValueError):
        BoundingBox(0, 5, 1, 1)


def test_intersection_overlap() -> None:
    a = BoundingBox(0, 0, 2, 2)
    b = BoundingBox(1, 1, 3, 3)
    inter = a.intersection(b)
    assert inter == BoundingBox(1, 1, 2, 2)
    assert inter.area == 1


def test_intersection_disjoint_is_none() -> None:
    a = BoundingBox(0, 0, 1, 1)
    b = BoundingBox(2, 2, 3, 3)
    assert a.intersection(b) is None


def test_intersection_touching_edges_is_none() -> None:
    a = BoundingBox(0, 0, 1, 1)
    b = BoundingBox(1, 0, 2, 1)
    assert a.intersection(b) is None


def test_iou_identical_boxes() -> None:
    box = BoundingBox(0, 0, 2, 2)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_known_value() -> None:
    # Two unit-area boxes overlapping in a 1x1 region.
    a = BoundingBox(0, 0, 2, 2)
    b = BoundingBox(1, 1, 3, 3)
    # intersection area = 1, union = 4 + 4 - 1 = 7
    assert a.iou(b) == pytest.approx(1 / 7)


def test_iou_disjoint_is_zero() -> None:
    a = BoundingBox(0, 0, 1, 1)
    b = BoundingBox(5, 5, 6, 6)
    assert a.iou(b) == 0.0


def test_normalized() -> None:
    box = BoundingBox(10, 20, 50, 80)
    norm = box.normalized(100, 200)
    assert norm == BoundingBox(0.1, 0.1, 0.5, 0.4)


def test_normalized_invalid_dimensions() -> None:
    box = BoundingBox(0, 0, 1, 1)
    with pytest.raises(ValueError):
        box.normalized(0, 100)
    with pytest.raises(ValueError):
        box.normalized(100, -1)
