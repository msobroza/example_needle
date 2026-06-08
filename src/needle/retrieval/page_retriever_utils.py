"""Helpers for the page retrievers: metadata matching + similarity maps.

Two responsibilities live here:

* :func:`matches_filter` — evaluate a single Mongo-style criterion produced by
  :class:`conversational_core.domain.metadata.backend_filter_adapters.MultiFieldFilterAdapter`
  against an actual metadata value. This is the read side of the filter
  contract; the adapter is the write side.
* :class:`SimilarityMapVisualizer` — render a late-interaction similarity
  heat-map over a page image (notebook / debugging convenience). All of its
  heavy dependencies (matplotlib, colpali-engine) are imported lazily.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------
def matches_filter(actual: Any, criterion: Any) -> bool:
    """Return whether ``actual`` satisfies ``criterion``.

    ``criterion`` may be:

    * ``None`` — always matches (no constraint).
    * a Mongo-style dict, e.g. ``{"$gte": 2020, "$lte": 2024}`` — every
      operator must hold (logical AND).
    * a list / tuple / set — membership test (``actual in criterion``).
    * any scalar — equality test.
    """
    if criterion is None:
        return True
    if isinstance(criterion, dict):
        return all(
            _apply_operator(op, actual, expected) for op, expected in criterion.items()
        )
    if isinstance(criterion, (list, tuple, set)):
        return actual in criterion
    return actual == criterion


def _apply_operator(op: str, actual: Any, expected: Any) -> bool:
    key = op.lstrip("$").lower()
    if key == "eq":
        return actual == expected
    if key == "ne":
        return actual != expected
    if key == "in":
        return actual in (expected or [])
    if key == "nin":
        return actual not in (expected or [])
    if key in {"gt", "gte", "lt", "lte"}:
        return _compare(key, actual, expected)
    if key == "contains":
        try:
            return expected in actual  # type: ignore[operator]
        except TypeError:
            return False
    if key == "exists":
        return (actual is not None) == bool(expected)
    if key == "regex":
        return actual is not None and re.search(str(expected), str(actual)) is not None
    raise ValueError(f"Unsupported filter operator: {op!r}")


def _compare(key: str, actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return False
    try:
        if key == "gt":
            return actual > expected
        if key == "gte":
            return actual >= expected
        if key == "lt":
            return actual < expected
        return actual <= expected  # lte
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Similarity-map visualisation (notebook convenience)
# ---------------------------------------------------------------------------
class SimilarityMapVisualizer:
    """Render where a query "looks" on a page for late-interaction models."""

    @staticmethod
    def highlight_image(
        image: Any,
        query_text: str,
        model: Any,
        processor: Any,
        device: str = "cpu",
    ) -> Any:
        """Overlay a token-level similarity heat-map onto ``image``.

        Best-effort: if the optional interpretability / plotting stack is not
        installed, the raw image is displayed (or returned) instead of raising.
        """
        try:
            return SimilarityMapVisualizer._render_with_colpali(
                image, query_text, model, processor, device
            )
        except Exception as exc:  # pragma: no cover - visualisation is optional
            logger.warning("Falling back to plain image (no similarity map): %s", exc)
            try:
                from IPython.display import display

                display(image)
            except Exception:
                return image
            return image

    @staticmethod
    def _render_with_colpali(
        image: Any,
        query_text: str,
        model: Any,
        processor: Any,
        device: str,
    ) -> Any:  # pragma: no cover - requires heavy optional deps
        import matplotlib.pyplot as plt
        import torch
        from colpali_engine.interpretability import (
            get_similarity_maps_from_embeddings,
        )

        with torch.no_grad():
            img_batch = processor.process_images([image]).to(device)
            query_batch = processor.process_queries([query_text]).to(device)
            img_emb = model(**img_batch)
            query_emb = model(**query_batch)

        n_patches = processor.get_n_patches(image.size, patch_size=model.patch_size)
        image_mask = processor.get_image_mask(img_batch)
        maps = get_similarity_maps_from_embeddings(
            image_embeddings=img_emb,
            query_embeddings=query_emb,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        similarity_map = maps[0].mean(dim=0).float().cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(
            similarity_map,
            cmap="jet",
            alpha=0.5,
            extent=(0, image.size[0], image.size[1], 0),
        )
        ax.set_title(query_text)
        ax.axis("off")
        return fig
