"""Arabic text reconstruction.

Only reversible, deterministic normalization happens here — no character
substitution or guessing. Low-confidence OCR output is left exactly as recognized
and flagged (``Word.low_confidence``) for downstream review, never silently
"corrected": doing so would risk inventing text that was never on the page.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .models import Block, Line, Word

TATWEEL = "ـ"
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WHITESPACE_RE = re.compile(r"[ \t]+")

# A parenthesized 1-2 digit number - e.g. "(1)" - regardless of surrounding
# whitespace, OR a bare digit with whitespace (or a string edge) on both
# sides - e.g. the "١" in "للمعجزة ١ كانت هنا". Both are the signature of a
# printed superscript footnote marker that the recognizer read correctly but
# returned merged into the body sentence, since it gives line-level text, not
# per-character boxes we could use to isolate it geometrically instead.
_FOOTNOTE_MARKER_RE = re.compile(
    r"(\(\s?[0-9٠-٩]{1,2}\s?\))" r"|(?<!\S)([0-9٠-٩])(?!\S)"
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(TATWEEL, "")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_word(word: Word) -> Word:
    cleaned = normalize_text(word.text)
    if cleaned == word.text:
        return word
    return Word(
        text=cleaned, bbox=word.bbox, confidence=word.confidence, low_confidence=word.low_confidence
    )


def flag_footnote_markers(word: Word) -> list[Word]:
    """Split off any embedded footnote-reference marker as its own flagged word.

    This is a text-pattern heuristic, not a correction: the marker's own text
    is left exactly as recognized and only ever flagged low_confidence, never
    removed or rewritten - a human decides in review whether it's really a
    footnote call-out (move it) or a genuine inline digit (leave it). It can
    occasionally flag a real inline number by mistake; it can never guess one
    away, which is the one thing this tool must not do.
    """
    matches = list(_FOOTNOTE_MARKER_RE.finditer(word.text))
    if not matches:
        return [word]

    pieces: list[Word] = []
    cursor = 0
    for m in matches:
        before = word.text[cursor : m.start()].strip()
        if before:
            pieces.append(Word(before, word.bbox, word.confidence, word.low_confidence))
        marker = m.group().strip()
        pieces.append(Word(marker, word.bbox, word.confidence, True))
        cursor = m.end()
    after = word.text[cursor:].strip()
    if after:
        pieces.append(Word(after, word.bbox, word.confidence, word.low_confidence))
    return pieces


def clean_line(line: Line) -> Optional[Line]:
    words: list[Word] = []
    for w in line.words:
        cleaned = clean_word(w)
        if not normalize_text(cleaned.text):
            continue
        words.extend(flag_footnote_markers(cleaned))
    if not words:
        return None
    return Line(words=tuple(words), bbox=line.bbox)


def clean_block(block: Block) -> Block:
    lines = tuple(cleaned for line in block.lines if (cleaned := clean_line(line)) is not None)
    return Block(kind=block.kind, lines=lines, level=block.level)


def clean_blocks(blocks: list[Block]) -> list[Block]:
    return [cleaned for block in blocks if (cleaned := clean_block(block)).lines]
