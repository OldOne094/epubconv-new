"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from . import epub_builder, report
from .logging_setup import configure_logging
from .models import PageResult, PageStatus
from .pipeline import ConversionConfig, convert

logger = logging.getLogger("epubconv.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epubconv",
        description="Convert an Arabic PDF or a folder of page images into an EPUB3 book via OCR.",
    )
    parser.add_argument("source", type=Path, help="PDF file or folder of page images")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output .epub path (default: alongside source)"
    )
    parser.add_argument("--title", help="Book title (default: source file/folder name)")
    parser.add_argument("--author", help="Book author")
    parser.add_argument("--lang", default="ar", help="OCR/EPUB language code (default: ar)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default: 300)")
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel OCR worker processes (default: 1)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="OCR device: auto-detect, force CPU, or force GPU (default: auto)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per page before marking it failed (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence below which a word is flagged, not corrected (default: 0.70)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore any existing cache and reprocess every page"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".epubconv_cache"),
        help="Where per-page results are cached for resuming (default: .epubconv_cache)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Only convert the first N pages (for previewing before a full run)",
    )
    parser.add_argument("--report", type=Path, help="Where to write the HTML conversion report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def _default_output(source: Path) -> Path:
    stem = source.stem if source.is_file() else source.name
    return source.parent / f"{stem}.epub"


def _make_progress_logger(total_hint: int = 0):
    def on_page_done(result: PageResult, total: int) -> None:
        status = "OK" if result.status == PageStatus.OK else "FAILED"
        logger.info("[%d/%d] page %d: %s", result.index + 1, total, result.index + 1, status)

    return on_page_done


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if not args.source.exists():
        parser.error(f"Source not found: {args.source}")

    output_path = args.output or _default_output(args.source)
    title = args.title or (args.source.stem if args.source.is_file() else args.source.name)

    config = ConversionConfig(
        lang=args.lang,
        dpi=args.dpi,
        threshold=args.threshold,
        max_retries=args.max_retries,
        workers=max(1, args.workers),
        resume=not args.no_resume,
        cache_root=args.cache_dir,
        device=args.device,
    )

    result = convert(
        args.source, title, config, on_page_done=_make_progress_logger(), max_pages=args.max_pages
    )
    if args.author:
        result.meta.author = args.author

    epub_builder.build_epub(result, output_path, source_path=args.source, dpi=args.dpi)

    report_path = args.report or output_path.with_suffix(".report.html")
    report.write_report(result, report_path)

    failed = len(result.failed_pages)
    total = len(result.pages)
    logger.info("Done: %s (%d/%d pages succeeded)", output_path, total - failed, total)
    logger.info("Report: %s", report_path)

    return 1 if total > 0 and failed == total else 0


if __name__ == "__main__":
    sys.exit(main())
