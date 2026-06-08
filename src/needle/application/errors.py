"""Application-layer errors."""

from __future__ import annotations

from ..exceptions import NeedleError


class ApplicationError(NeedleError):
    """Base class for application/use-case errors."""


class EmptyQueryError(ApplicationError):
    """Raised when a search is requested with empty query text."""


__all__ = ["ApplicationError", "EmptyQueryError"]
