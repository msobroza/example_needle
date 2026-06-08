#!/usr/bin/env python3
"""Scan a directory of documents and export a JSON index manifest.

python scripts/export_manifest.py ./corpus --output manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from needle.io.manifests import IndexManifest, ManifestEntry
from needle_core.domain.document.discovery import (
    count_by_extension,
    discover_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory to scan")
    parser.add_argument("--output", type=Path, default=Path("manifest.json"))
    parser.add_argument(
        "--checksum", action="store_true", help="include file checksums"
    )
    args = parser.parse_args()

    pairs = discover_documents(args.root, with_checksum=args.checksum)
    manifest = IndexManifest()
    for document, version in pairs:
        manifest.add(
            ManifestEntry(
                file=version.filename,
                format=document.document_ext.value,
                pages=0,  # filled in at index time once pages are rendered
                metadata={
                    **version.document_metadata,
                    **({"checksum": version.checksum} if version.checksum else {}),
                },
            )
        )

    saved = manifest.save(args.output)
    print(f"Wrote {len(manifest)} entries to {saved}")
    print("By extension:")
    for ext, count in sorted(count_by_extension(args.root).items()):
        print(f"  {ext}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
