"""Shared data models passed between pipeline stages."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any, Optional


class PageStatus(enum.Enum):
    OK = "ok"
    FAILED = "failed"
    IMAGE = "image"


class BlockKind(enum.Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"


@dataclasses.dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BBox":
        return cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])


@dataclasses.dataclass(frozen=True)
class Word:
    text: str
    bbox: BBox
    confidence: float
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Word":
        return cls(
            text=data["text"],
            bbox=BBox.from_dict(data["bbox"]),
            confidence=data["confidence"],
            low_confidence=data.get("low_confidence", False),
        )


@dataclasses.dataclass(frozen=True)
class Line:
    words: tuple[Word, ...]
    bbox: BBox

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def to_dict(self) -> dict:
        return {"words": [w.to_dict() for w in self.words], "bbox": self.bbox.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "Line":
        return cls(
            words=tuple(Word.from_dict(w) for w in data["words"]),
            bbox=BBox.from_dict(data["bbox"]),
        )


@dataclasses.dataclass(frozen=True)
class Block:
    kind: BlockKind
    lines: tuple[Line, ...]
    level: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "level": self.level,
            "lines": [line.to_dict() for line in self.lines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        return cls(
            kind=BlockKind(data["kind"]),
            level=data.get("level", 0),
            lines=tuple(Line.from_dict(l) for l in data["lines"]),
        )


@dataclasses.dataclass
class PageResult:
    index: int
    status: PageStatus
    blocks: tuple[Block, ...] = ()
    low_confidence_words: int = 0
    total_words: int = 0
    error: Optional[str] = None
    attempts: int = 0

    @property
    def confidence_ratio(self) -> float:
        if self.total_words == 0:
            return 1.0
        return 1.0 - (self.low_confidence_words / self.total_words)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "status": self.status.value,
            "blocks": [b.to_dict() for b in self.blocks],
            "low_confidence_words": self.low_confidence_words,
            "total_words": self.total_words,
            "error": self.error,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PageResult":
        return cls(
            index=data["index"],
            status=PageStatus(data["status"]),
            blocks=tuple(Block.from_dict(b) for b in data.get("blocks", [])),
            low_confidence_words=data.get("low_confidence_words", 0),
            total_words=data.get("total_words", 0),
            error=data.get("error"),
            attempts=data.get("attempts", 0),
        )


@dataclasses.dataclass
class DocumentMeta:
    title: str
    language: str = "ar"
    author: Optional[str] = None
    source_path: Optional[Path] = None
    page_count: int = 0


@dataclasses.dataclass
class ConversionResult:
    meta: DocumentMeta
    pages: list[PageResult]
    output_path: Optional[Path] = None

    @property
    def failed_pages(self) -> list[PageResult]:
        return [p for p in self.pages if p.status == PageStatus.FAILED]

    @property
    def ok_pages(self) -> list[PageResult]:
        return [p for p in self.pages if p.status == PageStatus.OK]
