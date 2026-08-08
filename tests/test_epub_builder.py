import zipfile
from pathlib import Path

import fitz
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub

from epubconv.epub_builder import NO_SOURCE_FOR_IMAGE_MESSAGE, build_epub
from epubconv.models import (
    BBox,
    Block,
    BlockKind,
    ConversionResult,
    DocumentMeta,
    Line,
    PageResult,
    PageStatus,
    Word,
)


def _make_pdf(path: Path, page_count: int = 1) -> None:
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 20), f"page {i}")
    doc.save(path)
    doc.close()


def _ok_page(index: int, text: str, low_confidence: bool = False) -> PageResult:
    word = Word(text=text, bbox=BBox(0, 0, 10, 10), confidence=0.9, low_confidence=low_confidence)
    line = Line(words=(word,), bbox=BBox(0, 0, 10, 10))
    block = Block(kind=BlockKind.PARAGRAPH, lines=(line,))
    return PageResult(index=index, status=PageStatus.OK, blocks=(block,), total_words=1)


def test_build_epub_creates_file_with_all_pages_as_chapters(tmp_path: Path):
    pages = [
        _ok_page(0, "<script>alert(1)</script>"),
        PageResult(index=1, status=PageStatus.FAILED, error="OCR crashed"),
        _ok_page(2, "نص سليم", low_confidence=True),
    ]
    result = ConversionResult(meta=DocumentMeta(title="كتابي", author="مؤلف"), pages=pages)
    output_path = tmp_path / "out.epub"

    returned_path = build_epub(result, output_path)

    assert returned_path == output_path
    assert output_path.exists()

    book = epub.read_epub(str(output_path))
    html_items = [
        item
        for item in book.get_items()
        if item.get_type() == ITEM_DOCUMENT and item.file_name.startswith("page_")
    ]
    assert len(html_items) == 3

    combined = b"".join(item.get_content() for item in html_items).decode("utf-8")
    assert "<script>alert(1)</script>" not in combined
    assert "&lt;script&gt;" in combined
    assert "OCR crashed" in combined
    assert 'class="low-confidence"' in combined


def test_build_epub_sets_rtl_direction_in_the_actual_written_file(tmp_path: Path):
    # ebooklib's read_epub()/get_content() round-trip silently drops the `dir`
    # attribute, so the only reliable way to check it made it into the book is to
    # read the raw XHTML bytes straight out of the zip, as a real e-reader would.
    result = ConversionResult(meta=DocumentMeta(title="كتابي"), pages=[_ok_page(0, "نص")])
    output_path = tmp_path / "out.epub"

    build_epub(result, output_path)

    with zipfile.ZipFile(output_path) as archive:
        chapter_names = [n for n in archive.namelist() if n.endswith("page_00001.xhtml")]
        assert len(chapter_names) == 1
        chapter_bytes = archive.read(chapter_names[0]).decode("utf-8")
        opf_names = [n for n in archive.namelist() if n.endswith(".opf")]
        opf_bytes = archive.read(opf_names[0]).decode("utf-8")

    assert 'dir="rtl"' in chapter_bytes
    assert 'page-progression-direction="rtl"' in opf_bytes


def test_build_epub_embeds_the_actual_page_for_an_image_status_page(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    pages = [PageResult(index=0, status=PageStatus.IMAGE)]
    result = ConversionResult(meta=DocumentMeta(title="كتابي"), pages=pages)
    output_path = tmp_path / "out.epub"

    build_epub(result, output_path, source_path=pdf_path)

    book = epub.read_epub(str(output_path))
    image_items = [item for item in book.get_items() if item.get_type() == ITEM_IMAGE]
    assert len(image_items) == 1
    assert image_items[0].get_content().startswith(b"\x89PNG")

    html_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    chapter = next(item for item in html_items if item.file_name.startswith("page_"))
    assert "<img" in chapter.get_content().decode("utf-8")


def test_build_epub_falls_back_to_a_message_when_no_source_given_for_an_image_page(
    tmp_path: Path,
):
    pages = [PageResult(index=0, status=PageStatus.IMAGE)]
    result = ConversionResult(meta=DocumentMeta(title="كتابي"), pages=pages)
    output_path = tmp_path / "out.epub"

    build_epub(result, output_path)  # no source_path

    book = epub.read_epub(str(output_path))
    html_items = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    chapter = next(item for item in html_items if item.file_name.startswith("page_"))
    assert NO_SOURCE_FOR_IMAGE_MESSAGE in chapter.get_content().decode("utf-8")
