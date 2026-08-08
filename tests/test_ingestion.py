from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from epubconv.ingestion import count_pages, iter_pages, list_images


def _make_pdf(path: Path, page_count: int) -> None:
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 20), f"page {i}")
    doc.save(path)
    doc.close()


def _make_image_dir(path: Path, count: int) -> None:
    path.mkdir()
    for i in range(count):
        Image.new("RGB", (30, 30), color=(i * 10 % 255, 0, 0)).save(path / f"{i:03d}.png")


def test_count_pages_pdf(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)
    assert count_pages(pdf_path) == 3


def test_iter_pages_pdf_yields_all_pages_in_order(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)

    pages = list(iter_pages(pdf_path, dpi=72))

    assert [p.index for p in pages] == [0, 1, 2]
    assert all(isinstance(p.image, np.ndarray) for p in pages)
    assert all(p.image.ndim == 3 for p in pages)


def test_iter_pages_pdf_with_indices_only_yields_requested_pages(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 5)

    pages = list(iter_pages(pdf_path, dpi=72, indices=[1, 3]))

    assert [p.index for p in pages] == [1, 3]


def test_count_pages_image_dir(tmp_path: Path):
    dir_path = tmp_path / "pages"
    _make_image_dir(dir_path, 4)
    assert count_pages(dir_path) == 4


def test_list_images_sorted_and_filtered(tmp_path: Path):
    dir_path = tmp_path / "pages"
    dir_path.mkdir()
    (dir_path / "b.png").write_bytes(b"")
    (dir_path / "a.png").write_bytes(b"")
    (dir_path / "notes.txt").write_bytes(b"")

    names = [p.name for p in list_images(dir_path)]

    assert names == ["a.png", "b.png"]


def test_iter_pages_image_dir_with_indices(tmp_path: Path):
    dir_path = tmp_path / "pages"
    _make_image_dir(dir_path, 4)

    pages = list(iter_pages(dir_path, indices=[0, 2]))

    assert [p.index for p in pages] == [0, 2]
