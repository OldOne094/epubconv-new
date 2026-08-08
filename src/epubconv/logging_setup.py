"""Logging configuration for the CLI."""

from __future__ import annotations

import logging
import sys


def _force_utf8_console() -> None:
    """Arabic titles/filenames otherwise crash `print`/logging on Windows, whose
    console defaults to a legacy codepage (cp1252/cp850) rather than UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_logging(verbose: bool = False) -> None:
    _force_utf8_console()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("epubconv").setLevel(level)
