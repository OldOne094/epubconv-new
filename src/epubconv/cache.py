"""Per-page resumable cache.

Each page's pipeline result is written to disk as soon as it is produced, so an
interrupted run can resume without re-running OCR (the most expensive stage) on
pages already completed. The cache key is derived from the source file's content
hash, so editing the source invalidates stale results automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from .models import PageResult

_HASH_CHUNK = 1024 * 1024


def hash_file(path: Path) -> str:
    """Content hash of a source. Supports both a single PDF file and a folder of
    page images (epubconv accepts either as input), so the cache key must too.
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.iterdir() if p.is_file()):
            digest.update(child.name.encode("utf-8"))
            _hash_file_into(child, digest)
    else:
        _hash_file_into(path, digest)
    return digest.hexdigest()[:16]


def _hash_file_into(path: Path, digest) -> None:
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            digest.update(chunk)


class PageCache:
    """Filesystem-backed cache of :class:`PageResult` objects, one file per page."""

    def __init__(self, cache_root: Path, source_path: Path):
        self.source_path = source_path
        self.key = hash_file(source_path)
        self.dir = cache_root / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest_path = self.dir / "manifest.json"
        if manifest_path.exists():
            return
        manifest_path.write_text(
            json.dumps({"source": str(self.source_path), "key": self.key}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _page_path(self, index: int) -> Path:
        return self.dir / f"page_{index:05d}.json"

    def has(self, index: int) -> bool:
        return self._page_path(index).exists()

    def load(self, index: int) -> Optional[PageResult]:
        path = self._page_path(index)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return PageResult.from_dict(data)

    def save(self, result: PageResult) -> None:
        path = self._page_path(result.index)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=None), encoding="utf-8"
        )
        os.replace(tmp_path, path)

    def clear(self) -> None:
        for child in self.dir.glob("page_*.json"):
            child.unlink(missing_ok=True)
