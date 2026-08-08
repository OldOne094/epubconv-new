"""PaddleOCR wrapper.

Model loading is expensive, so the engine is constructed once per process (lazily,
on first use) and reused across every page handled by that process, rather than
reloaded per page. ``paddleocr`` itself is only imported inside :func:`_build_engine`
so that importing this module — and running the unit tests — never requires the
heavy OCR stack to be installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .models import BBox, Word
from .structure import Y_OVERLAP_RATIO

logger = logging.getLogger("epubconv.ocr")

LOW_CONFIDENCE_THRESHOLD = 0.70

# Substring of the oneDNN crash message (see _build_engine's enable_mkldnn note).
# Sighted specifically when PaddleX silently falls back from GPU to CPU mid-
# session ("The specified device (GPU) is not available! Switching to CPU
# instead.") - enable_mkldnn=False protects an engine explicitly *built* for
# CPU from the start, but not this in-flight fallback path. Reproduced and
# confirmed the fix: rebuilding explicitly on CPU (not "auto") and retrying
# always succeeds where the silently-degraded engine crashes.
_ONEDNN_CRASH_MARKER = "ConvertPirAttribute2RuntimeAttribute"

# PaddleOCR's detector resizes the whole input to fit its own internal side-length
# limit before it ever looks for text. On a full book-page image (2000-2300px
# tall) that downscale is severe enough to destroy fine detail — verified on a
# real page where an ordinary sentence was recognized correctly when OCR ran on a
# ~400px-tall crop of it, but came out as unreadable character fragments when OCR
# ran on the full page. Splitting each page into shorter, overlapping strips
# keeps every strip well inside that limit. The overlap exists so a line that
# would otherwise sit exactly on a strip boundary (and get badly cut by both
# neighbors) is instead fully contained, undistorted, in at least one strip.
DEFAULT_STRIP_HEIGHT = 700
DEFAULT_STRIP_OVERLAP = 400

_engine: Optional[Any] = None
_engine_lock = threading.Lock()
_last_resolved_device: Optional[str] = None


def get_last_resolved_device() -> Optional[str]:
    """The device the current engine was actually built with ('gpu'/'cpu'), or
    None if no engine has been built yet in this process. Exposed so callers
    (e.g. the review server) can show the user what's really running — note this
    reflects engine *construction*, not necessarily every individual predict()
    call, since PaddleX can fall back to CPU per-call without telling us.
    """
    return _last_resolved_device


def _detect_device() -> str:
    """GPU if a CUDA-capable paddlepaddle build is installed and sees a device,
    otherwise CPU. Most users will only have the CPU build installed, so this
    must never raise for that (very common) case.
    """
    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:  # noqa: BLE001 - any failure here just means "no GPU"
        pass
    return "cpu"


def _build_engine(lang: str, device: str = "auto") -> Any:
    from paddleocr import PaddleOCR

    global _last_resolved_device
    resolved_device = _detect_device() if device == "auto" else device
    _last_resolved_device = resolved_device

    return PaddleOCR(
        lang=lang,
        use_textline_orientation=True,
        device=resolved_device,
        # These two default on and assume a photographed, potentially skewed/curved
        # page. Our pages come from rendering a PDF (or an already-flat scanned
        # image) — feeding that through document-unwarping actively distorts it.
        # Verified on a real page: with unwarping on, an entire diacritic-dense
        # sentence was destroyed into unrecognizable fragments; with it off, the
        # same sentence came back correct almost word-for-word. We already deskew
        # ourselves in preprocessing.py for genuinely tilted scans.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # oneDNN's CPU kernel for the detection model crashes on this PaddlePaddle
        # build (`ConvertPirAttribute2RuntimeAttribute not support ...DoubleAttribute`).
        # Always disabled, not just when we asked for CPU: PaddleX can silently fall
        # back to CPU per-call even when GPU was requested and initially available
        # (observed when a second process contended for the same GPU) - if that
        # happens with mkldnn still enabled, every OCR call on this engine crashes.
        # GPU inference doesn't go through oneDNN, so this costs nothing there.
        enable_mkldnn=False,
    )


def get_engine(lang: str = "ar", device: str = "auto") -> Any:
    """Reuses the cached engine unless the caller explicitly asks for a specific
    device that differs from the one already active — e.g. a user picking "CPU"
    in the review UI after an engine already came up on GPU. A plain "auto"
    request never forces a rebuild of an existing engine, since "auto" doesn't
    express a preference one way or the other.
    """
    global _engine
    with _engine_lock:
        needs_rebuild = _engine is None or (
            device != "auto" and _last_resolved_device is not None and device != _last_resolved_device
        )
        if needs_rebuild:
            _engine = _build_engine(lang, device)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine (mainly for tests and worker-process cleanup)."""
    global _engine, _last_resolved_device
    with _engine_lock:
        _engine = None
        _last_resolved_device = None


def _predict_recovering(engine: Any, image: np.ndarray, lang: str) -> tuple[Any, Any]:
    """engine.predict(image), with one recovery attempt for the silent-GPU-
    fallback oneDNN crash (see _ONEDNN_CRASH_MARKER). Returns (raw_results,
    engine_actually_used) so callers processing multiple strips can keep using
    the healthy engine for the rest of the page instead of hitting the same
    crash again on every remaining strip.
    """
    try:
        return engine.predict(image), engine
    except NotImplementedError as exc:
        if _ONEDNN_CRASH_MARKER not in str(exc):
            raise
        logger.warning(
            "OCR engine crashed (likely a silent GPU->CPU fallback hitting a "
            "oneDNN bug) - rebuilding explicitly on CPU and retrying"
        )
        reset_engine()
        cpu_engine = get_engine(lang, "cpu")
        return cpu_engine.predict(image), cpu_engine


def run_ocr(
    image: np.ndarray,
    lang: str = "ar",
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
    engine: Optional[Any] = None,
    device: str = "auto",
) -> list[Word]:
    engine = engine if engine is not None else get_engine(lang, device)
    raw_results, _engine_used = _predict_recovering(engine, image, lang)
    return parse_predict_results(raw_results, threshold=threshold)


def plan_tile_strips(
    height: int, strip_height: int = DEFAULT_STRIP_HEIGHT, overlap: int = DEFAULT_STRIP_OVERLAP
) -> list[tuple[int, int]]:
    """The (y_start, y_end) ranges run_ocr_tiled's automatic grid would use for an
    image this tall. Exposed so callers (e.g. the review server) can report a
    strip count for progress display before OCR actually starts.
    """
    if height <= strip_height:
        return [(0, height)]
    strips = []
    y = 0
    while True:
        y_end = min(y + strip_height, height)
        strips.append((y, y_end))
        if y_end >= height:
            return strips
        y += strip_height - overlap


def run_ocr_tiled(
    image: np.ndarray,
    lang: str = "ar",
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
    engine: Optional[Any] = None,
    device: str = "auto",
    strip_height: int = DEFAULT_STRIP_HEIGHT,
    overlap: int = DEFAULT_STRIP_OVERLAP,
    manual_cuts: Optional[Sequence[float]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[Word]:
    """Run OCR on horizontal strips of a page instead of the whole image at once.

    Splitting avoids the detector's internal downscale hurting recognition (see
    module docstring on ``DEFAULT_STRIP_HEIGHT``). By default strips overlap and
    most lines get detected redundantly by two neighboring strips;
    :func:`_merge_tile_words` resolves that into one set of words.

    ``manual_cuts``, if given, is a list of y-coordinates a person picked (in the
    review UI) as safe whitespace between lines. Strips are then cut exactly
    there with no overlap and no dedup needed — the person already guaranteed no
    line is split, which the automatic grid can't guarantee for every page.

    ``on_progress(strips_done, strips_total)`` is called after each strip so a
    caller can show progress during what can be a multi-second call.
    """
    engine = engine if engine is not None else get_engine(lang, device)
    height = image.shape[0]

    if manual_cuts:
        cuts = sorted(set(int(c) for c in manual_cuts if 0 < c < height))
        bounds = [0, *cuts, height]
        strips = list(zip(bounds[:-1], bounds[1:]))
        words: list[Word] = []
        for i, (y0, y1) in enumerate(strips):
            raw_results, engine = _predict_recovering(engine, image[y0:y1, :], lang)
            strip_words = parse_predict_results(raw_results, threshold=threshold)
            words.extend(
                Word(
                    text=w.text,
                    bbox=BBox(w.bbox.x0, w.bbox.y0 + y0, w.bbox.x1, w.bbox.y1 + y0),
                    confidence=w.confidence,
                    low_confidence=w.low_confidence,
                )
                for w in strip_words
            )
            if on_progress is not None:
                on_progress(i + 1, len(strips))
        return words

    strips = plan_tile_strips(height, strip_height, overlap)
    if len(strips) == 1:
        raw_results, _engine_used = _predict_recovering(engine, image, lang)
        if on_progress is not None:
            on_progress(1, 1)
        return parse_predict_results(raw_results, threshold=threshold)

    tagged: list[tuple[Word, int]] = []
    boundaries: list[tuple[float, float]] = []
    for strip_index, (y, y_end) in enumerate(strips):
        raw_results, engine = _predict_recovering(engine, image[y:y_end, :], lang)
        strip_words = parse_predict_results(raw_results, threshold=threshold)
        for word in strip_words:
            shifted = Word(
                text=word.text,
                bbox=BBox(word.bbox.x0, word.bbox.y0 + y, word.bbox.x1, word.bbox.y1 + y),
                confidence=word.confidence,
                low_confidence=word.low_confidence,
            )
            tagged.append((shifted, strip_index))
        if y_end < height:
            boundaries.append((y_end - overlap, y_end))
        if on_progress is not None:
            on_progress(strip_index + 1, len(strips))

    return _merge_tile_words(tagged, boundaries)


# A merge that would make the resulting line more than this many times taller
# than its shorter side is almost certainly two real lines, not one - reject it.
_MAX_LINE_MERGE_RATIO = 2.0


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _robust_line_span(words: list[Word]) -> tuple[float, float, float]:
    """Median-based (y0, y1, height) for a line of words.

    Deliberately NOT a min/max union: a single strip's own word-level boxes
    jitter a few px against each other (diacritics extend some boxes more
    than others), and that jitter compounds across a multi-word line into a
    union noticeably taller than any individual word - tall enough that a
    plain-union comparison can wrongly reject a real match against another
    strip's single, tighter whole-line box for the exact same printed line.
    The median is far less sensitive to that per-word jitter.
    """
    y0 = _median([w.bbox.y0 for w in words])
    y1 = _median([w.bbox.y1 for w in words])
    return y0, y1, y1 - y0


def _fits_as_one_line(words_a: list[Word], words_b: list[Word]) -> bool:
    a0, a1, ah = _robust_line_span(words_a)
    b0, b1, bh = _robust_line_span(words_b)
    top = max(a0, b0)
    bottom = min(a1, b1)
    overlap = max(0.0, bottom - top)
    reference = min(ah, bh)
    if reference <= 0 or overlap / reference < Y_OVERLAP_RATIO:
        return False
    merged_height = max(a1, b1) - min(a0, b0)
    return merged_height <= reference * _MAX_LINE_MERGE_RATIO


def _merge_tile_words(
    tagged: list[tuple[Word, int]], boundaries: list[tuple[float, float]]
) -> list[Word]:
    def in_overlap_zone(word: Word) -> bool:
        cy = (word.bbox.y0 + word.bbox.y1) / 2
        return any(lo <= cy <= hi for lo, hi in boundaries)

    confirmed = [word for word, _strip in tagged if not in_overlap_zone(word)]
    contested = [entry for entry in tagged if in_overlap_zone(entry[0])]

    # Group contested words into lines PER STRIP first, refusing any merge that
    # would blow past _MAX_LINE_MERGE_RATIO - this is what stops one bad
    # detection from fusing two of a strip's own real, separate lines.
    by_strip: dict[int, list[Word]] = {}
    for word, strip_index in contested:
        by_strip.setdefault(strip_index, []).append(word)

    strip_lines: list[tuple[int, list[Word]]] = []
    for strip_index, words in by_strip.items():
        words_sorted = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2)
        lines: list[list[Word]] = []
        for w in words_sorted:
            target = next((line for line in lines if _fits_as_one_line(line, [w])), None)
            if target is not None:
                target.append(w)
            else:
                lines.append([w])
        strip_lines.extend((strip_index, line) for line in lines)

    def line_confidence(words: list[Word]) -> float:
        return sum(w.confidence for w in words) / len(words)

    # Group strip-lines that represent the SAME real printed line across
    # strips using connected components, not just first-found pairs. The
    # tiling geometry can put THREE (or more) strips in contest for one line
    # at once (consecutive overlap zones overlap each other); pairing only
    # the first two matches found would leave any additional strip's
    # competing detection unmatched and therefore kept unconditionally,
    # producing a visible duplicate right next to the winning reading.
    n = len(strip_lines)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if strip_lines[i][0] == strip_lines[j][0]:
                continue
            if _fits_as_one_line(strip_lines[i][1], strip_lines[j][1]):
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        group = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            group.append(node)
            stack.extend(neighbor for neighbor in adjacency[node] if not visited[neighbor])
        best = max(group, key=lambda idx: line_confidence(strip_lines[idx][1]))
        confirmed.extend(strip_lines[best][1])

    return confirmed


def parse_predict_results(
    raw_results: Iterable[Any], threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> list[Word]:
    """Pure parsing of PaddleOCR's ``predict()`` output into :class:`Word` objects.

    Kept separate from :func:`run_ocr` so it can be unit-tested against a fabricated
    result structure without needing PaddleOCR installed.
    """
    words: list[Word] = []
    for page_result in raw_results:
        data = _as_dict(page_result)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        boxes = _extract_boxes(data, expected=len(texts))
        for text, score, box in zip(texts, scores, boxes):
            if not text or not text.strip():
                continue
            confidence = float(score)
            words.append(
                Word(
                    text=text,
                    bbox=box,
                    confidence=confidence,
                    low_confidence=confidence < threshold,
                )
            )
    return words


def _as_dict(page_result: Any) -> dict:
    if isinstance(page_result, dict):
        data = page_result
    elif hasattr(page_result, "json"):
        data = page_result.json
    else:
        raise TypeError(f"Unrecognized PaddleOCR result type: {type(page_result)!r}")
    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        return data["res"]
    return data


def _extract_boxes(data: dict, expected: int) -> list[BBox]:
    if data.get("rec_boxes") is not None:
        return [_box_from_rect(b) for b in data["rec_boxes"]]
    if data.get("rec_polys") is not None:
        return [_box_from_poly(p) for p in data["rec_polys"]]
    if data.get("dt_polys") is not None:
        return [_box_from_poly(p) for p in data["dt_polys"]]
    return [BBox(0.0, 0.0, 0.0, 0.0) for _ in range(expected)]


def _box_from_rect(rect: Any) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in rect)
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _box_from_poly(poly: Any) -> BBox:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
