"""Core value types shared across the conversational retrieval domain."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Union


class DocumentExtension(str, Enum):
    """Supported document file extensions.

    The enum subclasses :class:`str` so members behave like their string
    value: they can be used directly as dictionary keys (see
    ``IMAGE_EXTRACTORS``), compared with raw strings, and serialised without
    any custom plumbing.
    """

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "md"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"DocumentExtension.{self.name}"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_string(cls, value: Union[str, DocumentExtension]) -> DocumentExtension:
        """Build an extension from a (possibly dotted/aliased) string."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().lstrip(".")
        if normalized in _EXTENSION_ALIASES:
            return _EXTENSION_ALIASES[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:  # pragma: no cover - error path
            raise ValueError(
                f"Unsupported document extension: {value!r}. "
                f"Known extensions: {sorted(e.value for e in cls)}"
            ) from exc

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> DocumentExtension:
        """Infer the extension from a filesystem path."""
        suffix = Path(path).suffix
        if not suffix:
            raise ValueError(f"Path has no file extension: {path!r}")
        return cls.from_string(suffix)

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------
    @property
    def is_image(self) -> bool:
        """Whether this extension is a raster image format."""
        return self in _IMAGE_EXTENSIONS

    @property
    def is_renderable(self) -> bool:
        """Whether pages of this format can be rasterised to images."""
        return self in _RENDERABLE_EXTENSIONS


_EXTENSION_ALIASES = {
    "tif": DocumentExtension.TIFF,
    "htm": DocumentExtension.HTML,
    "markdown": DocumentExtension.MARKDOWN,
    "mdown": DocumentExtension.MARKDOWN,
    "text": DocumentExtension.TXT,
}

_IMAGE_EXTENSIONS = frozenset(
    {
        DocumentExtension.PNG,
        DocumentExtension.JPG,
        DocumentExtension.JPEG,
        DocumentExtension.TIFF,
        DocumentExtension.WEBP,
        DocumentExtension.BMP,
        DocumentExtension.GIF,
    }
)

_RENDERABLE_EXTENSIONS = _IMAGE_EXTENSIONS | {
    DocumentExtension.PDF,
    DocumentExtension.DOCX,
    DocumentExtension.PPTX,
}
