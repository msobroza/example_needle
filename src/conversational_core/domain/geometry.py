"""2D geometry value objects.

A :class:`BoundingBox` describes an axis-aligned rectangle in pixel (or any
consistent) coordinates with ``x0 <= x1`` and ``y0 <= y1``. It supports the
common operations needed when reasoning about regions on a page:
intersection, intersection-over-union and normalisation to ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle ``(x0, y0)`` (top-left) to ``(x1, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) must be <= x1 ({self.x1})")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) must be <= y1 ({self.y1})")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Return the overlapping box, or ``None`` when there is no overlap."""
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return BoundingBox(x0, y0, x1, y1)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union with ``other`` (``0.0`` when disjoint)."""
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union = self.area + other.area - inter.area
        if union <= 0:
            return 0.0
        return inter.area / union

    def normalized(self, width: float, height: float) -> BoundingBox:
        """Return a copy with coordinates divided by ``width``/``height``."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        return BoundingBox(
            self.x0 / width,
            self.y0 / height,
            self.x1 / width,
            self.y1 / height,
        )
