"""Heuristic structural analysis.

Groups raw OCR word boxes into lines, then lines into paragraph/heading blocks,
following Arabic reading order (right-to-left within a line, top-to-bottom across
lines). Nothing here talks to the OCR engine, so it is fully unit-testable against
synthetic word boxes.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from .models import BBox, Block, BlockKind, Line, Word

Y_OVERLAP_RATIO = 0.5
# A gap larger than this fraction of the median line height starts a new block.
# Real page geometry has most within-paragraph line gaps at ~0 (adjacent line
# boxes touch or slightly overlap), so this only needs to clear noise, not split
# a stable "typical gap" — using the median gap itself as that reference is
# unstable: on real pages over half the gaps are exactly 0, which used to
# collapse every block on a page into one.
PARAGRAPH_GAP_HEIGHT_RATIO = 0.08
HEADING_HEIGHT_FACTOR = 1.3


def group_words_into_lines(words: Sequence[Word]) -> list[Line]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2)
    clusters: list[list[Word]] = []
    for word in ordered:
        target = next(
            (cluster for cluster in clusters if y_overlaps(cluster[-1].bbox, word.bbox)), None
        )
        if target is not None:
            target.append(word)
        else:
            clusters.append([word])

    lines: list[Line] = []
    for cluster in clusters:
        cluster.sort(key=lambda w: w.bbox.x0, reverse=True)  # RTL: rightmost word first
        lines.append(Line(words=tuple(cluster), bbox=_union_bbox([w.bbox for w in cluster])))
    lines.sort(key=lambda line: line.bbox.y0)
    return lines


def y_overlaps(a: BBox, b: BBox) -> bool:
    """Whether two boxes' vertical spans overlap enough to belong to the same line.

    Public: also used by ocr.py to cluster candidate duplicate detections along
    tile boundaries when merging tiled-OCR results back into one page.
    """
    top = max(a.y0, b.y0)
    bottom = min(a.y1, b.y1)
    overlap = max(0.0, bottom - top)
    shorter = min(a.height, b.height)
    if shorter <= 0:
        return False
    return (overlap / shorter) >= Y_OVERLAP_RATIO


def _union_bbox(boxes: Sequence[BBox]) -> BBox:
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def group_lines_into_blocks(lines: Sequence[Line]) -> list[Block]:
    if not lines:
        return []
    heights = [line.bbox.height for line in lines if line.bbox.height > 0]
    median_height = statistics.median(heights) if heights else 0.0
    gaps = _gaps_between(lines)
    gap_threshold = median_height * PARAGRAPH_GAP_HEIGHT_RATIO

    blocks: list[Block] = []
    current: list[Line] = [lines[0]]
    for i in range(1, len(lines)):
        gap = gaps[i - 1]
        same_block = gap <= gap_threshold
        if same_block:
            current.append(lines[i])
        else:
            blocks.append(_make_block(current, median_height))
            current = [lines[i]]
    blocks.append(_make_block(current, median_height))
    return blocks


def _gaps_between(lines: Sequence[Line]) -> list[float]:
    return [max(0.0, lines[i].bbox.y0 - lines[i - 1].bbox.y1) for i in range(1, len(lines))]


def _make_block(lines: Sequence[Line], median_height: float) -> Block:
    is_heading = (
        len(lines) == 1
        and median_height > 0
        and lines[0].bbox.height >= median_height * HEADING_HEIGHT_FACTOR
    )
    kind = BlockKind.HEADING if is_heading else BlockKind.PARAGRAPH
    return Block(kind=kind, lines=tuple(lines), level=1 if is_heading else 0)


def analyze(words: Sequence[Word]) -> list[Block]:
    return group_lines_into_blocks(group_words_into_lines(words))
