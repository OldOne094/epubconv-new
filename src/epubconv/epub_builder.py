"""Assemble a right-to-left Arabic EPUB3 book from converted pages.

Every page becomes its own chapter, in source order, so a failed page shows up as
a single flagged chapter instead of aborting the whole book — the prior version
stopped the entire conversion on a single page failure.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from ebooklib import epub

from . import ingestion
from .models import Block, BlockKind, ConversionResult, Line, PageResult, PageStatus

CSS = """
body { direction: rtl; text-align: right; font-family: serif; line-height: 1.8; }
h1, h2, h3, h4, h5, h6 { text-align: center; }
p { margin: 0 0 1em 0; text-indent: 1.5em; }
.low-confidence { background-color: #fff3b0; }
.page-error { color: #a00; font-style: italic; text-align: center; }
"""

NO_TEXT_MESSAGE = "لم يتم العثور على نص في هذه الصفحة."
FAILED_PAGE_MESSAGE = "تعذّر التعرف على هذه الصفحة"
NO_SOURCE_FOR_IMAGE_MESSAGE = "هذه الصفحة معلَّمة كصورة، لكن لا يوجد مصدر لعرضها."


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _render_line_html(line: Line) -> str:
    parts = []
    for word in line.words:
        escaped = _escape(word.text)
        if word.low_confidence:
            parts.append(f'<span class="low-confidence">{escaped}</span>')
        else:
            parts.append(escaped)
    return " ".join(parts)


def _render_block_html(block: Block) -> str:
    if block.kind == BlockKind.HEADING:
        level = min(max(block.level, 1), 6)
        inner = "<br/>".join(_render_line_html(line) for line in block.lines)
        return f"<h{level}>{inner}</h{level}>"
    inner = " ".join(_render_line_html(line) for line in block.lines)
    return f"<p>{inner}</p>"


def render_page_body(page: PageResult) -> str:
    if page.status == PageStatus.FAILED:
        detail = _escape(page.error) if page.error else ""
        suffix = f": {detail}" if detail else "."
        return f'<p class="page-error">{FAILED_PAGE_MESSAGE}{suffix}</p>'
    if not page.blocks:
        return f'<p class="page-error">{NO_TEXT_MESSAGE}</p>'
    return "\n".join(_render_block_html(block) for block in page.blocks)


def _add_image_page(book: epub.EpubBook, page: PageResult, source_path: Path, dpi: int) -> str:
    """Embed the page's actual raster as an EPUB image resource and return the
    chapter body referencing it - used for pages marked "keep as image" (covers,
    illustrations, ...) that were never sent through OCR at all.
    """
    image_bytes = ingestion.render_page_png(source_path, page.index, dpi=dpi)
    image_name = f"images/page_{page.index + 1:05d}.png"
    image_item = epub.EpubImage(
        uid=f"img_{page.index}", file_name=image_name, media_type="image/png", content=image_bytes
    )
    book.add_item(image_item)
    return f'<div style="text-align:center"><img src="{image_name}" alt=""/></div>'


def build_epub(
    result: ConversionResult,
    output_path: Path,
    source_path: Optional[Path] = None,
    dpi: int = 300,
) -> Path:
    book = epub.EpubBook()
    book.set_identifier(f"epubconv-{result.meta.title}-{len(result.pages)}")
    book.set_title(result.meta.title)
    book.set_language(result.meta.language)
    book.set_direction("rtl")
    if result.meta.author:
        book.add_author(result.meta.author)

    style = epub.EpubItem(
        uid="style", file_name="style/main.css", media_type="text/css", content=CSS
    )
    book.add_item(style)

    chapters = []
    for page in result.pages:
        chapter = epub.EpubHtml(
            title=f"صفحة {page.index + 1}",
            file_name=f"page_{page.index + 1:05d}.xhtml",
            lang=result.meta.language,
            direction="rtl",
        )
        if page.status == PageStatus.IMAGE:
            body = (
                _add_image_page(book, page, source_path, dpi)
                if source_path is not None
                else f'<p class="page-error">{NO_SOURCE_FOR_IMAGE_MESSAGE}</p>'
            )
        else:
            body = render_page_body(page)
        # ebooklib regenerates the XHTML from its own template on write, discarding
        # any `dir` attribute on this raw string — only `EpubHtml.direction` (set
        # above) actually survives into the written file.
        chapter.content = (
            f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{result.meta.language}">'
            f"<head><title>{_escape(chapter.title)}</title>"
            f'<link rel="stylesheet" href="style/main.css" type="text/css"/></head>'
            f"<body>{body}</body></html>"
        )
        chapter.add_item(style)
        book.add_item(chapter)
        chapters.append(chapter)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(output_path), book)
    return output_path
