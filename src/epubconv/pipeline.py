"""Pipeline orchestrator.

Streams pages one at a time (bounded to ``config.workers`` images in flight even
in parallel mode), isolates failures per page — a bad page is retried and then
flagged rather than aborting the whole run — and persists each page's result to
the cache immediately so an interrupted run can resume without redoing OCR.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import arabic, ingestion, ocr, preprocessing, structure
from .cache import PageCache
from .models import ConversionResult, DocumentMeta, PageResult, PageStatus

logger = logging.getLogger("epubconv.pipeline")

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

OnPageDone = Callable[[PageResult, int], None]


@dataclasses.dataclass
class ConversionConfig:
    lang: str = "ar"
    dpi: int = 300
    threshold: float = ocr.LOW_CONFIDENCE_THRESHOLD
    max_retries: int = DEFAULT_MAX_RETRIES
    workers: int = 1
    resume: bool = True
    cache_root: Path = dataclasses.field(default_factory=lambda: Path(".epubconv_cache"))
    # "auto" picks GPU if a CUDA-capable paddlepaddle build sees one, else CPU.
    device: str = "auto"


def process_page(
    image: np.ndarray,
    index: int,
    lang: str,
    threshold: float,
    device: str = "auto",
    manual_cuts: Optional[Sequence[float]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> PageResult:
    processed = preprocessing.preprocess(image)
    words = ocr.run_ocr_tiled(
        processed,
        lang=lang,
        threshold=threshold,
        device=device,
        manual_cuts=manual_cuts,
        on_progress=on_progress,
    )
    blocks = arabic.clean_blocks(structure.analyze(words))
    total_words = sum(len(line.words) for block in blocks for line in block.lines)
    low_confidence = sum(
        1 for block in blocks for line in block.lines for word in line.words if word.low_confidence
    )
    return PageResult(
        index=index,
        status=PageStatus.OK,
        blocks=tuple(blocks),
        total_words=total_words,
        low_confidence_words=low_confidence,
        attempts=1,
    )


def _process_with_retry(
    image: np.ndarray, index: int, lang: str, threshold: float, max_retries: int, device: str = "auto"
) -> PageResult:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 2):
        try:
            result = process_page(image, index, lang, threshold, device)
            result.attempts = attempt
            return result
        except Exception as exc:  # noqa: BLE001 - per-page isolation is the point
            last_error = exc
            logger.warning(
                "Page %d failed (attempt %d/%d): %s", index, attempt, max_retries + 1, exc
            )
            if attempt <= max_retries:
                time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * attempt)
    return PageResult(
        index=index, status=PageStatus.FAILED, error=str(last_error), attempts=max_retries + 1
    )


def _normalize_lang(lang: str) -> str:
    return "ar" if lang in ("ar", "arabic") else lang


def convert(
    source_path: Path,
    title: str,
    config: ConversionConfig,
    on_page_done: Optional[OnPageDone] = None,
    max_pages: Optional[int] = None,
) -> ConversionResult:
    source_path = Path(source_path)
    page_count = ingestion.count_pages(source_path)
    # Cache key is keyed off the source file itself (see PageCache), not the
    # requested subset, so a later full run reuses whatever a max_pages preview
    # already cached instead of redoing that OCR.
    indices_to_process = range(min(max_pages, page_count)) if max_pages is not None else range(page_count)
    cache = PageCache(config.cache_root, source_path) if config.resume else None

    pages: list[Optional[PageResult]] = [None] * page_count
    pending_indices: list[int] = []
    for index in indices_to_process:
        cached = cache.load(index) if cache is not None else None
        if cached is not None:
            pages[index] = cached
            if on_page_done is not None:
                on_page_done(cached, page_count)
        else:
            pending_indices.append(index)

    if pending_indices:
        runner = _run_parallel if config.workers > 1 else _run_sequential
        runner(source_path, pending_indices, config, cache, pages, on_page_done, page_count)

    meta = DocumentMeta(
        title=title,
        language=_normalize_lang(config.lang),
        source_path=source_path,
        page_count=page_count,
    )
    return ConversionResult(meta=meta, pages=[p for p in pages if p is not None])


def _run_sequential(
    source_path: Path,
    pending_indices: list[int],
    config: ConversionConfig,
    cache: Optional[PageCache],
    pages: list[Optional[PageResult]],
    on_page_done: Optional[OnPageDone],
    page_count: int,
) -> None:
    for source_page in ingestion.iter_pages(source_path, dpi=config.dpi, indices=pending_indices):
        result = _process_with_retry(
            source_page.image,
            source_page.index,
            config.lang,
            config.threshold,
            config.max_retries,
            config.device,
        )
        pages[source_page.index] = result
        if cache is not None:
            cache.save(result)
        if on_page_done is not None:
            on_page_done(result, page_count)


def _run_parallel(
    source_path: Path,
    pending_indices: list[int],
    config: ConversionConfig,
    cache: Optional[PageCache],
    pages: list[Optional[PageResult]],
    on_page_done: Optional[OnPageDone],
    page_count: int,
) -> None:
    page_iter = ingestion.iter_pages(source_path, dpi=config.dpi, indices=pending_indices)
    in_flight: dict[concurrent.futures.Future, int] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=config.workers) as executor:

        def submit_next() -> bool:
            source_page = next(page_iter, None)
            if source_page is None:
                return False
            future = executor.submit(
                _process_with_retry,
                source_page.image,
                source_page.index,
                config.lang,
                config.threshold,
                config.max_retries,
                config.device,
            )
            in_flight[future] = source_page.index
            return True

        for _ in range(config.workers):
            if not submit_next():
                break

        while in_flight:
            done, _pending = concurrent.futures.wait(
                list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                index = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - worker crash isolation
                    logger.warning("Page %d crashed in worker process: %s", index, exc)
                    result = PageResult(
                        index=index,
                        status=PageStatus.FAILED,
                        error=str(exc),
                        attempts=config.max_retries + 1,
                    )
                pages[index] = result
                if cache is not None:
                    cache.save(result)
                if on_page_done is not None:
                    on_page_done(result, page_count)
                submit_next()
