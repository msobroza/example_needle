"""Page extractors — turn a file on disk into a list of per-page objects.

The retrievers in this package are *visual*: they embed page **images**. The
:class:`PageToImageExtractor` hierarchy rasterises each supported format into
a list of :class:`PIL.Image.Image` objects (one per page).

Pillow is imported lazily inside the methods so that simply importing this
module stays cheap and dependency-light; heavier backends (PyMuPDF for PDFs)
are optional and only required when their format is actually used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from conversational_core.domain.exceptions import UnsupportedExtensionError
from conversational_core.domain.types import DocumentExtension

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image


class PageExtractor(ABC):
    """Abstract base: extract a list of per-page objects from a file."""

    #: Extensions this extractor knows how to handle.
    supported_extensions: tuple[DocumentExtension, ...] = ()

    @abstractmethod
    def extract(self, path: Union[str, Path], **kwargs: Any) -> list:
        """Return a list of per-page objects for ``path``."""

    def supports(self, ext: DocumentExtension) -> bool:
        return ext in self.supported_extensions

    def _check_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such document: {path}")
        return path


class PageToImageExtractor(PageExtractor):
    """A :class:`PageExtractor` whose pages are :class:`PIL.Image.Image`."""

    @abstractmethod
    def extract(  # noqa: D401 - imperative is fine
        self, path: Union[str, Path], *, dpi: int = 150, **kwargs: Any
    ) -> list[Image]:
        """Rasterise ``path`` into one RGB image per page."""


class ImageFileExtractor(PageToImageExtractor):
    """Single-image formats (PNG/JPG/...): the file *is* the one page."""

    supported_extensions = (
        DocumentExtension.PNG,
        DocumentExtension.JPG,
        DocumentExtension.JPEG,
        DocumentExtension.TIFF,
        DocumentExtension.WEBP,
        DocumentExtension.BMP,
        DocumentExtension.GIF,
    )

    def extract(
        self, path: Union[str, Path], *, dpi: int = 150, **kwargs: Any
    ) -> list[Image]:
        from PIL import Image, ImageSequence

        path = self._check_path(path)
        with Image.open(path) as img:
            # Multi-frame images (animated GIF / multipage TIFF) → one page each.
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]
        return frames or []


class PdfToImageExtractor(PageToImageExtractor):
    """Rasterise PDF pages to images using PyMuPDF (``pip install pymupdf``)."""

    supported_extensions = (DocumentExtension.PDF,)

    def extract(
        self, path: Union[str, Path], *, dpi: int = 150, **kwargs: Any
    ) -> list[Image]:
        path = self._check_path(path)
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise UnsupportedExtensionError(
                "Rendering PDFs requires PyMuPDF. Install it with "
                "`pip install example-needle[pdf]` (or `pip install pymupdf`)."
            ) from exc
        from PIL import Image

        images: list[Image] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(
                    Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                )
        return images


class OfficeToImageExtractor(PageToImageExtractor):
    """Office formats (DOCX/PPTX): convert to PDF first, then rasterise.

    Conversion relies on an external LibreOffice/``soffice`` binary being
    available on ``PATH``. When it is not, a clear, actionable error is raised
    rather than failing deep inside a subprocess.
    """

    supported_extensions = (DocumentExtension.DOCX, DocumentExtension.PPTX)

    def __init__(self, pdf_extractor: PdfToImageExtractor | None = None) -> None:
        self.pdf_extractor = pdf_extractor or PdfToImageExtractor()

    def extract(
        self, path: Union[str, Path], *, dpi: int = 150, **kwargs: Any
    ) -> list[Image]:
        import shutil
        import subprocess
        import tempfile

        path = self._check_path(path)
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice is None:  # pragma: no cover - environment dependent
            raise UnsupportedExtensionError(
                "Rendering office documents requires LibreOffice (`soffice`) on "
                "PATH. Convert the file to PDF first, or install LibreOffice."
            )
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmp,
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
            pdf_path = Path(tmp) / (path.stem + ".pdf")
            return self.pdf_extractor.extract(pdf_path, dpi=dpi)


# A single shared instance is fine: extractors are stateless.
_IMAGE_FILE_EXTRACTOR = ImageFileExtractor()
_PDF_EXTRACTOR = PdfToImageExtractor()
_OFFICE_EXTRACTOR = OfficeToImageExtractor(pdf_extractor=_PDF_EXTRACTOR)

#: Default mapping from extension to the extractor that rasterises it.
IMAGE_EXTRACTORS: dict[DocumentExtension, PageToImageExtractor] = {
    DocumentExtension.PDF: _PDF_EXTRACTOR,
    DocumentExtension.DOCX: _OFFICE_EXTRACTOR,
    DocumentExtension.PPTX: _OFFICE_EXTRACTOR,
    DocumentExtension.PNG: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.JPG: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.JPEG: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.TIFF: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.WEBP: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.BMP: _IMAGE_FILE_EXTRACTOR,
    DocumentExtension.GIF: _IMAGE_FILE_EXTRACTOR,
}


def get_extractor(
    ext: Union[DocumentExtension, str],
    extractors: dict[DocumentExtension, PageToImageExtractor] | None = None,
) -> PageToImageExtractor:
    """Look up the extractor for an extension, raising a clear error if absent."""
    ext = DocumentExtension.from_string(ext)
    table = extractors if extractors is not None else IMAGE_EXTRACTORS
    try:
        return table[ext]
    except KeyError as exc:
        raise UnsupportedExtensionError(
            f"No extractor registered for {ext!r}. "
            f"Supported: {sorted(str(k) for k in table)}"
        ) from exc
