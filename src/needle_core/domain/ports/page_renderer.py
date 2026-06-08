"""The :class:`PageRenderer` port — rasterise a document into page images."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PageRenderer(Protocol):
    """Structural interface implemented by the image extractors.

    :class:`needle.retrieval.extractors.PageToImageExtractor` satisfies this
    port: it renders a file into a list of one image per page.
    """

    def extract(self, path: Any, *, dpi: int = 150, **kwargs: Any) -> list:
        """Return one image per page of ``path``."""
        ...
