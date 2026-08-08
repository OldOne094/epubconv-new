"""Streaming ingestion of source documents (PDF or a folder of images).

Pages are yielded one at a time as numpy arrays; the caller discards each page's
pixel data once consumed. This keeps peak memory bounded regardless of document
length — the prior version loaded every page into RAM before OCR began, which was
its primary performance bottleneck on long books.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pymupdf as fitz
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class SourcePage:
    index: int
    image: np.ndarray  # RGB, HxWx3, uint8


def list_images(dir_path: Path) -> list[Path]:
    return sorted(
        (p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def count_pages(source_path: Path) -> int:
    source_path = Path(source_path)
    if source_path.is_dir():
        return len(list_images(source_path))
    with fitz.open(source_path) as doc:
        return doc.page_count


def iter_pages(
    source_path: Path, dpi: int = 300, indices: Optional[Iterable[int]] = None
) -> Iterator[SourcePage]:
    """Yield one page image at a time without holding the whole document in memory.

    When ``indices`` is given, only those pages are decoded — used to resume a run
    without re-rendering pages whose results are already cached.
    """
    source_path = Path(source_path)
    wanted = set(indices) if indices is not None else None
    if source_path.is_dir():
        yield from _iter_image_dir(source_path, wanted)
    else:
        yield from _iter_pdf(source_path, dpi, wanted)


def render_page_png(source_path: Path, index: int, dpi: int = 300) -> bytes:
    """Render a single page as PNG bytes - shared by the review server's image
    preview and the EPUB builder's image-only pages, so both rasterize a page
    identically.
    """
    page = next(iter_pages(source_path, dpi=dpi, indices=[index]))
    buf = io.BytesIO()
    Image.fromarray(page.image).save(buf, format="PNG")
    return buf.getvalue()


def _iter_image_dir(dir_path: Path, wanted: Optional[set[int]]) -> Iterator[SourcePage]:
    for index, path in enumerate(list_images(dir_path)):
        if wanted is not None and index not in wanted:
            continue
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
        yield SourcePage(index=index, image=array)


def _pixmap_to_array(pixmap: "fitz.Pixmap") -> np.ndarray:
    if pixmap.n >= 4:
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
    arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _iter_pdf(pdf_path: Path, dpi: int, wanted: Optional[set[int]]) -> Iterator[SourcePage]:
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for index in range(doc.page_count):
            if wanted is not None and index not in wanted:
                continue
            page = doc.load_page(index)
            pixmap = page.get_pixmap(matrix=matrix)
            array = _pixmap_to_array(pixmap)
            yield SourcePage(index=index, image=array)
