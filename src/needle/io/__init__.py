"""IO subpackage — filesystem helpers and index manifests.

:mod:`paths` collects small path utilities (directory creation, recursive file
enumeration, byte formatting); :mod:`manifests` defines the JSON-serialisable
:class:`IndexManifest` / :class:`ManifestEntry` describing indexed documents.
"""

from __future__ import annotations

from .manifests import IndexManifest, ManifestEntry
from .paths import ensure_dir, human_size, iter_files

__all__ = [
    "ensure_dir",
    "iter_files",
    "human_size",
    "ManifestEntry",
    "IndexManifest",
]
