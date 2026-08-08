"""Local review server.

Lets a person inspect flagged (low-confidence) pages next to the source image,
retry OCR on a page, or directly overwrite its text — all backed by the same
resumable page cache the CLI uses, so a subsequent ``epubconv`` run picks up
whatever was reviewed here without redoing OCR on pages left untouched.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

from .logging_setup import configure_logging

from . import epub_builder, ingestion, ocr, report
from .cache import PageCache
from .models import (
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
from .pipeline import ConversionConfig, process_page

logger = logging.getLogger("epubconv.review")


class ReviewState:
    """Owns the cache and source access; the HTTP layer just calls into this."""

    def __init__(self, source_path: Optional[Path], config: ConversionConfig):
        self.source_path = Path(source_path) if source_path is not None else None
        self.config = config
        self.cache = PageCache(config.cache_root, self.source_path) if self.source_path else None
        self.page_count = ingestion.count_pages(self.source_path) if self.source_path else 0
        self._lock = threading.Lock()
        # index -> {"current": int, "total": int}, present only while a retry for
        # that page is in flight. A plain dict is fine here: the review UI polls
        # this from a different thread than the one running retry(), and single
        # key read/writes are atomic enough under the GIL for a progress display
        # (not used for anything that needs stronger guarantees).
        self.progress: dict[int, dict] = {}
        self.bulk_state: dict = {"running": False, "done": 0, "total": 0, "current_index": None}
        self._bulk_stop_event = threading.Event()
        self._bulk_thread: Optional[threading.Thread] = None

    def _require_book(self) -> None:
        if self.source_path is None:
            raise ValueError("لا يوجد كتاب مفتوح حاليًا - افتح كتاب أولًا")

    def list_pages(self) -> list[dict]:
        pages = []
        for index in range(self.page_count):
            cached = self.cache.load(index)
            pages.append(_page_summary(cached, index))
        return pages

    def get_page(self, index: int) -> dict:
        if self.cache is None:
            return {"index": index, "status": "pending", "text": "", "error": None}
        return _page_view(self.cache.load(index), index)

    def get_progress(self, index: int) -> dict:
        return self.progress.get(index, {"current": 0, "total": 0})

    def get_device(self) -> dict:
        return {"device": ocr.get_last_resolved_device()}

    def get_tiling_defaults(self) -> dict:
        return {"strip_height": ocr.DEFAULT_STRIP_HEIGHT, "overlap": ocr.DEFAULT_STRIP_OVERLAP}

    def prepare_engine(self, device: Optional[str] = None) -> dict:
        """Build (or rebuild, if a different device is now requested) the OCR
        engine ahead of time, so the first real page retry isn't the one paying
        for model load time.
        """
        with self._lock:
            ocr.get_engine(self.config.lang, device or self.config.device)
            return {"device": ocr.get_last_resolved_device()}

    def render_image_png(self, index: int) -> bytes:
        self._require_book()
        return ingestion.render_page_png(self.source_path, index, dpi=self.config.dpi)

    def retry(
        self,
        index: int,
        manual_cuts: Optional[Sequence[float]] = None,
        device: Optional[str] = None,
    ) -> dict:
        self._require_book()
        with self._lock:
            self.progress[index] = {"current": 0, "total": 0}
            try:
                page = next(
                    ingestion.iter_pages(self.source_path, dpi=self.config.dpi, indices=[index])
                )

                def on_progress(current: int, total: int) -> None:
                    self.progress[index] = {"current": current, "total": total}

                result = process_page(
                    page.image,
                    index,
                    self.config.lang,
                    self.config.threshold,
                    device or self.config.device,
                    manual_cuts=manual_cuts,
                    on_progress=on_progress,
                )
                self.cache.save(result)
                return _page_view(result, index)
            finally:
                self.progress.pop(index, None)

    def save_manual_edit(self, index: int, text: str) -> dict:
        self._require_book()
        with self._lock:
            result = _build_manual_result(index, text)
            self.cache.save(result)
            return _page_view(result, index)

    def mark_as_image(self, index: int) -> dict:
        """Skip OCR for this page entirely - it'll be embedded as a plain image
        in the final EPUB instead of text (covers, illustrations, ...). A later
        "إعادة محاولة OCR" on the same page overwrites this with a real attempt,
        same as it would overwrite any other cached result.
        """
        self._require_book()
        with self._lock:
            result = PageResult(index=index, status=PageStatus.IMAGE)
            self.cache.save(result)
            return _page_view(result, index)

    def start_bulk_process(self, device: Optional[str] = None) -> dict:
        """Process every page that has no cached result yet, in page order, in a
        background thread - so this call returns immediately and the server
        keeps answering other requests (including a page's own progress poll)
        while it runs. Reuses retry(), so it shares the same lock as any
        single-page retry the person might also trigger, and the same recovery
        behaviour on a bad OCR attempt (isolate the page, move on).
        """
        self._require_book()
        if self.bulk_state["running"]:
            return dict(self.bulk_state)

        pending = [i for i in range(self.page_count) if self.cache.load(i) is None]
        self.bulk_state = {"running": True, "done": 0, "total": len(pending), "current_index": None}
        self._bulk_stop_event.clear()

        def worker() -> None:
            try:
                for index in pending:
                    if self._bulk_stop_event.is_set():
                        break
                    self.bulk_state["current_index"] = index
                    try:
                        self.retry(index, device=device)
                    except Exception:  # noqa: BLE001 - isolate one page, keep going
                        logger.exception("bulk processing failed on page %d", index)
                    self.bulk_state["done"] += 1
            finally:
                self.bulk_state["running"] = False
                self.bulk_state["current_index"] = None

        self._bulk_thread = threading.Thread(target=worker, daemon=True)
        self._bulk_thread.start()
        return dict(self.bulk_state)

    def get_bulk_progress(self) -> dict:
        view = dict(self.bulk_state)
        current = view.get("current_index")
        view["current_page_progress"] = self.get_progress(current) if current is not None else None
        return view

    def stop_bulk_process(self) -> dict:
        self._bulk_stop_event.set()
        return dict(self.bulk_state)

    def build_final_epub(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> dict:
        """Assemble the EPUB from whatever's in the cache right now - the same
        build epubconv itself would do, minus running OCR: a page with no
        cached result yet is written in as a failed placeholder (never
        silently dropped from the book) and counted so the person doing the
        save knows to run "معالجة كل الصفحات المتبقية" first if they want it
        complete.
        """
        self._require_book()
        pages: list[PageResult] = []
        missing = 0
        for index in range(self.page_count):
            cached = self.cache.load(index)
            if cached is None:
                missing += 1
                cached = PageResult(
                    index=index, status=PageStatus.FAILED, error="لم تتم معالجتها بعد"
                )
            pages.append(cached)

        meta = DocumentMeta(
            title=title or self.source_path.stem,
            language=self.config.lang,
            author=author or None,
            source_path=self.source_path,
            page_count=self.page_count,
        )
        result = ConversionResult(meta=meta, pages=pages)

        out_path = Path(output_path) if output_path else self.source_path.with_suffix(".epub")
        epub_builder.build_epub(result, out_path, source_path=self.source_path, dpi=self.config.dpi)
        report_path = out_path.with_suffix(".report.html")
        report.write_report(result, report_path)

        return {
            "output_path": str(out_path),
            "report_path": str(report_path),
            "page_count": self.page_count,
            "missing_pages": missing,
            "failed_pages": sum(1 for p in pages if p.status == PageStatus.FAILED),
        }


def list_books_in_dir(directory: Path) -> list[dict]:
    """PDFs found next to the currently open book - a convenience list, not the
    only way to open one (a person can also type any path directly).
    """
    if not directory.is_dir():
        return []
    return [
        {"path": str(p), "name": p.name}
        for p in sorted(directory.iterdir(), key=lambda p: p.name.lower())
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]


def books_dir() -> Path:
    """Where uploaded books are kept, so a person picking one via the browser's
    native file dialog doesn't need to know or type any path - just a fixed
    subfolder next to the project itself. Assumes an editable install (the
    install scripts always use ``pip install -e .``), which is what keeps
    __file__ pointing at the real source checkout instead of a site-packages
    copy.
    """
    root = Path(__file__).resolve().parent.parent.parent
    directory = root / "books"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


_SAFE_FILENAME_RE = re.compile(r"[^\w\-. ()؀-ۿ]+")


def save_uploaded_book(filename: str, content: bytes, directory: Optional[Path] = None) -> Path:
    """Save uploaded PDF bytes into books_dir(), sanitizing the name and
    avoiding collisions with an existing file instead of overwriting it.
    """
    stem = Path(filename).stem or "book"
    stem = _SAFE_FILENAME_RE.sub("_", stem).strip() or "book"
    if directory is None:
        directory = books_dir()
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}.pdf"
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem} ({n}).pdf"
        n += 1
    candidate.write_bytes(content)
    return candidate


def _build_manual_result(index: int, text: str) -> PageResult:
    zero_box = BBox(0.0, 0.0, 0.0, 0.0)
    blocks = []
    for raw_paragraph in text.split("\n\n"):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        lines = tuple(
            Line(
                words=(Word(text=line, bbox=zero_box, confidence=1.0, low_confidence=False),),
                bbox=zero_box,
            )
            for line in (segment.strip() for segment in paragraph.split("\n"))
            if line
        )
        if lines:
            blocks.append(Block(kind=BlockKind.PARAGRAPH, lines=lines))
    total_words = sum(1 for block in blocks for _ in block.lines)
    return PageResult(
        index=index,
        status=PageStatus.OK,
        blocks=tuple(blocks),
        total_words=total_words,
        low_confidence_words=0,
        attempts=0,
    )


def _page_summary(result: Optional[PageResult], index: int) -> dict:
    if result is None:
        return {"index": index, "status": "pending"}
    return {
        "index": index,
        "status": result.status.value,
        "total_words": result.total_words,
        "low_confidence_words": result.low_confidence_words,
        "confidence_ratio": result.confidence_ratio,
        "error": result.error,
    }


def _page_view(result: Optional[PageResult], index: int) -> dict:
    if result is None:
        return {"index": index, "status": "pending", "text": "", "error": None}
    paragraphs = []
    for block in result.blocks:
        paragraphs.append("\n".join(line.text for line in block.lines))
    words_view = [
        {"text": word.text, "confidence": word.confidence, "low_confidence": word.low_confidence}
        for block in result.blocks
        for line in block.lines
        for word in line.words
    ]
    return {
        "index": index,
        "status": result.status.value,
        "error": result.error,
        "total_words": result.total_words,
        "low_confidence_words": result.low_confidence_words,
        "text": "\n\n".join(paragraphs),
        "words": words_view,
    }


def _book_view(state: ReviewState) -> dict:
    scan_dir = state.source_path.parent if state.source_path is not None else books_dir()
    return {
        "current": str(state.source_path) if state.source_path is not None else None,
        "current_name": state.source_path.name if state.source_path is not None else None,
        "page_count": state.page_count,
        "books": list_books_in_dir(scan_dir),
    }


class ReviewHandler(BaseHTTPRequestHandler):
    state: ReviewState  # assigned on the class before serve_forever()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        logger.debug("%s - %s", self.address_string(), format % args)

    def _write_response(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # The browser navigated away or cancelled the request (e.g. it fires a
            # new page-image fetch before the previous one finished) mid-response.
            # Nothing to recover: the client is gone, so there's no one to report
            # an error to — log it and move on instead of letting it crash this
            # request's worker thread with a traceback.
            logger.debug("client disconnected mid-response for %s", self.path)

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._write_response(status, "application/json; charset=utf-8", body)

    def _send_html(self, body: str) -> None:
        self._write_response(200, "text/html; charset=utf-8", body.encode("utf-8"))

    def _send_png(self, data: bytes) -> None:
        self._write_response(200, "image/png", data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/pages":
            self._send_json(200, self.state.list_pages())
            return
        if path == "/api/device":
            self._send_json(200, self.state.get_device())
            return
        if path == "/api/tiling-defaults":
            self._send_json(200, self.state.get_tiling_defaults())
            return
        if path == "/api/books":
            self._send_json(200, _book_view(self.state))
            return
        if path == "/api/process-all/progress":
            self._send_json(200, self.state.get_bulk_progress())
            return
        match = re.match(r"^/api/pages/(\d+)$", path)
        if match:
            self._send_json(200, self.state.get_page(int(match.group(1))))
            return
        match = re.match(r"^/api/pages/(\d+)/progress$", path)
        if match:
            self._send_json(200, self.state.get_progress(int(match.group(1))))
            return
        match = re.match(r"^/api/pages/(\d+)/image$", path)
        if match:
            try:
                self._send_png(self.state.render_image_png(int(match.group(1))))
            except Exception as exc:  # noqa: BLE001 - report to the client, don't crash the server
                logger.exception("rendering page image failed")
                self._send_json(500, {"error": str(exc)})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        path = urlparse(self.path).path
        if path == "/api/prepare":
            try:
                payload = self._read_json_body()
                self._send_json(200, self.state.prepare_engine(payload.get("device") or None))
            except Exception as exc:  # noqa: BLE001
                logger.exception("engine prepare failed")
                self._send_json(500, {"error": str(exc)})
            return
        if path == "/api/process-all":
            try:
                payload = self._read_json_body()
                self._send_json(200, self.state.start_bulk_process(payload.get("device") or None))
            except Exception as exc:  # noqa: BLE001
                logger.exception("starting bulk processing failed")
                self._send_json(500, {"error": str(exc)})
            return
        if path == "/api/process-all/stop":
            self._send_json(200, self.state.stop_bulk_process())
            return
        if path == "/api/save":
            try:
                payload = self._read_json_body()
                self._send_json(
                    200,
                    self.state.build_final_epub(
                        title=(payload.get("title") or "").strip() or None,
                        author=(payload.get("author") or "").strip() or None,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("building final epub failed")
                self._send_json(500, {"error": str(exc)})
            return
        if path == "/api/books/open":
            try:
                payload = self._read_json_body()
                raw_path = (payload.get("path") or "").strip()
                if not raw_path:
                    self._send_json(400, {"error": "path is required"})
                    return
                new_path = Path(raw_path)
                if not new_path.exists():
                    self._send_json(404, {"error": f"not found: {raw_path}"})
                    return
                self.__class__.state = ReviewState(new_path, self.state.config)
                self._send_json(200, _book_view(self.state))
            except Exception as exc:  # noqa: BLE001
                logger.exception("opening book failed")
                self._send_json(500, {"error": str(exc)})
            return
        if path == "/api/books/upload":
            try:
                payload = self._read_json_body()
                filename = (payload.get("filename") or "").strip()
                content_b64 = payload.get("content") or ""
                if not filename or not content_b64:
                    self._send_json(400, {"error": "filename and content are required"})
                    return
                try:
                    content = base64.b64decode(content_b64, validate=True)
                except Exception:
                    self._send_json(400, {"error": "invalid file data"})
                    return
                if not content.startswith(b"%PDF"):
                    self._send_json(400, {"error": "الملف ده مش PDF صالح"})
                    return
                saved_path = save_uploaded_book(filename, content)
                self.__class__.state = ReviewState(saved_path, self.state.config)
                self._send_json(200, _book_view(self.state))
            except Exception as exc:  # noqa: BLE001
                logger.exception("uploading book failed")
                self._send_json(500, {"error": str(exc)})
            return
        match = re.match(r"^/api/pages/(\d+)/retry$", path)
        if match:
            index = int(match.group(1))
            try:
                payload = self._read_json_body()
                manual_cuts = payload.get("manual_cuts") or None
                device = payload.get("device") or None
                self._send_json(200, self.state.retry(index, manual_cuts=manual_cuts, device=device))
            except Exception as exc:  # noqa: BLE001
                logger.exception("retry failed for page %d", index)
                self._send_json(500, {"error": str(exc)})
            return
        match = re.match(r"^/api/pages/(\d+)/edit$", path)
        if match:
            index = int(match.group(1))
            try:
                payload = self._read_json_body()
                self._send_json(200, self.state.save_manual_edit(index, payload.get("text", "")))
            except Exception as exc:  # noqa: BLE001
                logger.exception("manual edit failed for page %d", index)
                self._send_json(500, {"error": str(exc)})
            return
        match = re.match(r"^/api/pages/(\d+)/mark-image$", path)
        if match:
            index = int(match.group(1))
            try:
                self._send_json(200, self.state.mark_as_image(index))
            except Exception as exc:  # noqa: BLE001
                logger.exception("mark-as-image failed for page %d", index)
                self._send_json(500, {"error": str(exc)})
            return
        self.send_response(404)
        self.end_headers()


def create_server(
    source_path: Optional[Path], config: ConversionConfig, port: int = 8765
) -> ThreadingHTTPServer:
    """Build the server without running it, so callers (including tests) control
    the run loop. Note: state lives on the ``ReviewHandler`` class itself, so
    only one review server should be active per process at a time.

    ``source_path`` may be ``None`` to start with no book loaded - the person
    picks one from the browser (upload, the books/ folder listing, or a typed
    path) instead of needing one up front on the command line.
    """
    ReviewHandler.state = ReviewState(source_path, config)
    return ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)


def serve(
    source_path: Optional[Path],
    config: ConversionConfig,
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    server = create_server(source_path, config, port)
    bound_port = server.server_address[1]
    url = f"http://127.0.0.1:{bound_port}"
    logger.info("Review server: %s (Ctrl+C to stop)", url)
    if open_browser:
        # The server is already listening at this point (the socket is bound
        # in ThreadingHTTPServer.__init__, before serve_forever) - safe to open
        # right away with no wait/race.
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a browser failing to launch must never crash the server
            logger.exception("could not open a browser automatically")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<title>مراجعة epubconv</title>
<style>
  :root {
    --accent: #1a73e8;
    --accent-dark: #1558b0;
    --bg: #f5f6f8;
    --panel: #ffffff;
    --border: #dde1e6;
    --text: #202124;
    --text-muted: #6b7280;
    --danger: #b3261e;
    --danger-bg: #fdecea;
    --warn-bg: #fff3b0;
    --image-bg: #eef2ff;
    --image-text: #3730a3;
    --radius: 8px;
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: var(--bg); color: var(--text);
  }
  button, select, input[type="text"] {
    font-family: inherit; font-size: 0.88em; padding: 0.4em 0.8em;
    border: 1px solid var(--border); border-radius: var(--radius); background: var(--panel); color: var(--text);
  }
  button { cursor: pointer; background: var(--accent); color: #fff; border-color: var(--accent); transition: background 0.15s; }
  button:hover { background: var(--accent-dark); }
  button:disabled { opacity: 0.55; cursor: default; }
  button.secondary { background: var(--panel); color: var(--text); border-color: var(--border); }
  button.secondary:hover { background: #eef1f4; }
  button.danger { background: var(--danger); border-color: var(--danger); }
  button.danger:hover { background: #8f1f19; }

  #topbar {
    display: flex; align-items: center; gap: 1.6em; flex-wrap: wrap;
    padding: 0.6em 1em; background: var(--panel); border-bottom: 1px solid var(--border);
  }
  .brand { font-weight: 700; color: var(--accent-dark); white-space: nowrap; }
  .topbar-group { display: flex; align-items: center; gap: 0.4em; flex-wrap: wrap; }
  .topbar-group label { color: var(--text-muted); font-size: 0.85em; }
  #bookPathInput { width: 14em; }
  #bookFileInput {
    position: absolute; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none;
  }
  .file-btn { display: inline-block; }
  #emptyState { padding: 1.2em; color: var(--text-muted); font-size: 0.9em; line-height: 1.8; }
  #bulkStatus, #saveStatus { color: var(--text-muted); font-size: 0.85em; }
  #bookTitleInput { width: 12em; }

  #bulkBarWrap { display: none; background: #e6e8eb; height: 6px; }
  #bulkBar { background: var(--accent); height: 100%; width: 0%; transition: width 0.2s; }

  #body { flex: 1; display: flex; overflow: hidden; }
  #sidebar { width: 270px; overflow-y: auto; border-left: 1px solid var(--border); flex-shrink: 0; background: var(--panel); }
  #sidebar div.page-row { padding: 0.55em 0.9em; cursor: pointer; border-bottom: 1px solid #f1f2f4; font-size: 0.88em; }
  #sidebar div.page-row:hover { background: #f0f4fb; }
  #sidebar div.page-row.selected { background: #dbe9ff; font-weight: 600; }
  #sidebar div.page-row.status-failed { color: var(--danger); }
  #sidebar div.page-row.status-pending { color: var(--text-muted); }
  #sidebar div.page-row.status-image { color: var(--image-text); }
  #sidebar div.page-row.low-confidence-page { background: var(--warn-bg); }

  #main { flex: 1; display: flex; overflow: hidden; }
  #imagePane, #textPane { flex: 1; overflow: auto; padding: 1em; }
  #imagePane { border-left: 1px solid var(--border); background: #fafbfc; }
  #imageWrap { position: relative; display: inline-block; }
  #imageWrap img { max-width: 100%; display: block; box-shadow: 0 1px 5px rgba(0,0,0,0.12); }
  #cutOverlay { position: absolute; inset: 0; cursor: crosshair; }
  .cut-line { position: absolute; left: 0; right: 0; height: 0; border-top: 2px dashed var(--accent); cursor: pointer; }
  .cut-hint { color: var(--text-muted); font-size: 0.82em; margin-top: 0.6em; line-height: 1.7; }

  .low-confidence { background-color: var(--warn-bg); border-radius: 3px; padding: 0 0.1em; }
  #preview {
    line-height: 2.1; margin-bottom: 0.8em; padding: 0.9em; background: var(--panel);
    border: 1px solid var(--border); border-radius: var(--radius); min-height: 2em;
  }
  #preview em { color: var(--text-muted); }
  textarea {
    width: 100%; height: 42%; font-size: 1em; direction: rtl; padding: 0.6em;
    border: 1px solid var(--border); border-radius: var(--radius); font-family: inherit;
  }

  .toolbar { margin-bottom: 0.6em; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5em; }
  #status { color: var(--text-muted); font-size: 0.85em; }
  #deviceIndicator {
    color: var(--text-muted); font-size: 0.8em; padding: 0.3em 0.8em;
    border: 1px solid var(--border); border-radius: 1em; background: var(--panel);
  }
  #progressBarWrap { display: none; background: #e6e8eb; border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 0.5em; }
  #progressBar { background: var(--accent); height: 100%; width: 0%; transition: width 0.2s; }
</style>
</head>
<body>
<div id="topbar">
  <div class="brand">epubconv</div>
  <div class="topbar-group">
    <label>الكتاب:</label>
    <select id="bookSelect" title="اختر من الكتب الموجودة"></select>
    <label for="bookFileInput" class="secondary file-btn">تصفّح واختر ملف PDF...</label>
    <input id="bookFileInput" type="file" accept="application/pdf,.pdf"/>
    <input id="bookPathInput" type="text" placeholder="أو الصق مسار ملف يدويًا"/>
    <button id="openBookBtn" class="secondary">فتح</button>
  </div>
  <div class="topbar-group">
    <button id="processAllBtn">معالجة كل الصفحات المتبقية</button>
    <button id="stopAllBtn" class="danger" style="display:none">إيقاف</button>
    <span id="bulkStatus"></span>
  </div>
  <div class="topbar-group">
    <input id="bookTitleInput" type="text" placeholder="عنوان الكتاب (اختياري)"/>
    <button id="saveEpubBtn">حفظ الكتاب النهائي (EPUB)</button>
    <span id="saveStatus"></span>
  </div>
</div>
<div id="bulkBarWrap"><div id="bulkBar"></div></div>
<div id="body">
  <div id="sidebar"></div>
  <div id="main">
    <div id="imagePane">
      <div id="imageWrap">
        <img id="pageImage" src="" alt=""/>
        <div id="cutOverlay"></div>
      </div>
      <div class="cut-hint">
        الخطوط الزرقاء المتقطّعة هي نقاط التقسيم (تلقائية بالبداية) — دوس على الصورة في الفراغ بين سطرين عشان تضيف نقطة، ودوس على خط موجود عشان تشيله.
        <button id="clearCutsBtn" class="secondary">امسح كل نقاط التقسيم</button>
      </div>
    </div>
    <div id="textPane">
      <div class="toolbar">
        <select id="deviceSelect" title="الجهاز المطلوب استخدامه">
          <option value="auto">تلقائي</option>
          <option value="cpu">المعالج (CPU)</option>
          <option value="gpu">كرت الشاشة (GPU)</option>
        </select>
        <button id="prepareBtn" class="secondary">تجهيز المحرك</button>
        <button id="retryBtn">إعادة محاولة OCR</button>
        <button id="markImageBtn" class="secondary">خلي الصفحة صورة</button>
        <button id="saveBtn">حفظ التعديل اليدوي</button>
        <span id="deviceIndicator">الجهاز: ...</span>
        <span id="status"></span>
      </div>
      <div id="progressBarWrap"><div id="progressBar"></div></div>
      <div id="preview"></div>
      <textarea id="editor" placeholder="اختر صفحة من القائمة"></textarea>
    </div>
  </div>
</div>
<script>
let currentIndex = null;
let manualCuts = [];
let cutsEdited = false;
let progressTimer = null;
let bulkTimer = null;
let tilingDefaults = { strip_height: 700, overlap: 400 };

const STATUS_LABELS = { ok: 'تم', failed: 'فشل', pending: 'لم تُعالج بعد', image: 'صورة' };
function statusLabel(status) { return STATUS_LABELS[status] || status; }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function loadDevice() {
  const res = await fetch('/api/device');
  const data = await res.json();
  document.getElementById('deviceIndicator').textContent =
    'الجهاز: ' + (data.device === 'gpu' ? 'كرت الشاشة (GPU)' : data.device === 'cpu' ? 'المعالج (CPU)' : 'غير معروف بعد');
}

async function loadTilingDefaults() {
  const res = await fetch('/api/tiling-defaults');
  tilingDefaults = await res.json();
}

async function loadBooks() {
  const res = await fetch('/api/books');
  const data = await res.json();
  const select = document.getElementById('bookSelect');
  select.innerHTML = '';
  if (!data.current) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'لا يوجد كتاب مفتوح';
    opt.selected = true;
    opt.disabled = true;
    select.appendChild(opt);
  }
  let sawCurrent = !data.current;
  for (const b of data.books) {
    const opt = document.createElement('option');
    opt.value = b.path;
    opt.textContent = b.name;
    if (b.path === data.current) { opt.selected = true; sawCurrent = true; }
    select.appendChild(opt);
  }
  if (!sawCurrent) {
    const opt = document.createElement('option');
    opt.value = data.current;
    opt.textContent = data.current_name;
    opt.selected = true;
    select.appendChild(opt);
  }
}

function resetUiForNewBook() {
  currentIndex = null;
  manualCuts = [];
  cutsEdited = false;
  document.getElementById('pageImage').src = '';
  document.getElementById('preview').innerHTML = '';
  document.getElementById('editor').value = '';
  document.getElementById('bookPathInput').value = '';
  document.getElementById('bookTitleInput').value = '';
  document.getElementById('saveStatus').textContent = '';
  document.getElementById('status').textContent = '';
}

async function openBook(path) {
  if (!path) return;
  document.getElementById('status').textContent = 'جاري فتح الكتاب...';
  try {
    const res = await fetch('/api/books/open', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل فتح الكتاب: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  }
  resetUiForNewBook();
  await Promise.all([loadPages(), loadDevice(), loadBooks()]);
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function uploadBook(file) {
  document.getElementById('status').textContent = 'جاري رفع الكتاب... (حسب حجم الملف)';
  try {
    const buffer = await file.arrayBuffer();
    const content = arrayBufferToBase64(buffer);
    const res = await fetch('/api/books/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, content: content }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل الرفع: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  }
  resetUiForNewBook();
  await Promise.all([loadPages(), loadDevice(), loadBooks()]);
}

document.getElementById('bookSelect').onchange = (e) => openBook(e.target.value);
document.getElementById('openBookBtn').onclick = () => {
  openBook(document.getElementById('bookPathInput').value.trim());
};
document.getElementById('bookFileInput').onchange = (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) uploadBook(file);
};

function computeDefaultCuts(naturalHeight) {
  // Mirrors ocr.plan_tile_strips()'s automatic grid, then takes the midpoint of
  // each strip-to-strip overlap zone as a single representative cut line, so the
  // person sees roughly where the automatic algorithm would split and can nudge
  // it from there instead of starting from a blank slate.
  const { strip_height, overlap } = tilingDefaults;
  if (!naturalHeight || naturalHeight <= strip_height) return [];
  const cuts = [];
  let y = 0;
  while (true) {
    const yEnd = Math.min(y + strip_height, naturalHeight);
    if (yEnd >= naturalHeight) break;
    cuts.push(Math.round(yEnd - overlap / 2));
    y += strip_height - overlap;
  }
  return cuts;
}

async function loadPages() {
  const res = await fetch('/api/pages');
  const pages = await res.json();
  const sidebar = document.getElementById('sidebar');
  sidebar.innerHTML = '';
  if (pages.length === 0) {
    const empty = document.createElement('div');
    empty.id = 'emptyState';
    empty.textContent = 'مفيش كتاب مفتوح لسه. اختر كتاب من القائمة فوق، دوس "تصفّح واختر ملف PDF..." عشان ترفع كتابك، أو الصق مسار ملف يدويًا.';
    sidebar.appendChild(empty);
    return;
  }
  for (const p of pages) {
    const row = document.createElement('div');
    row.className = 'page-row status-' + p.status;
    if (p.index === currentIndex) row.classList.add('selected');
    if (p.status === 'ok' && p.confidence_ratio !== undefined && p.confidence_ratio < 0.9) {
      row.className += ' low-confidence-page';
    }
    let label = 'صفحة ' + (p.index + 1) + ' — ' + statusLabel(p.status);
    if (p.low_confidence_words) label += ' (' + p.low_confidence_words + ' مشكوك فيها)';
    row.textContent = label;
    row.onclick = () => selectPage(p.index);
    sidebar.appendChild(row);
  }
}

function renderCutLines() {
  const overlay = document.getElementById('cutOverlay');
  overlay.innerHTML = '';
  const img = document.getElementById('pageImage');
  if (!img.naturalHeight) return;
  const scale = img.clientHeight / img.naturalHeight;
  manualCuts.forEach((y, i) => {
    const line = document.createElement('div');
    line.className = 'cut-line';
    line.style.top = Math.round(y * scale) + 'px';
    line.title = 'اضغط للحذف';
    line.onclick = (e) => {
      e.stopPropagation();
      manualCuts.splice(i, 1);
      cutsEdited = true;
      renderCutLines();
    };
    overlay.appendChild(line);
  });
}

document.getElementById('cutOverlay').onclick = (e) => {
  const img = document.getElementById('pageImage');
  if (!img.naturalHeight) return;
  const rect = img.getBoundingClientRect();
  const clickY = e.clientY - rect.top;
  const scale = img.naturalHeight / img.clientHeight;
  manualCuts.push(Math.round(clickY * scale));
  manualCuts.sort((a, b) => a - b);
  cutsEdited = true;
  renderCutLines();
};

document.getElementById('clearCutsBtn').onclick = () => {
  manualCuts = [];
  cutsEdited = false;
  renderCutLines();
};

async function selectPage(index) {
  currentIndex = index;
  manualCuts = [];
  cutsEdited = false;
  document.querySelectorAll('#sidebar .page-row').forEach((el, i) => {
    el.classList.toggle('selected', i === index);
  });
  const img = document.getElementById('pageImage');
  img.onload = () => {
    // These are only a visual starting suggestion for the person to nudge -
    // showing them here must NOT itself count as an edit (see cutsEdited).
    manualCuts = computeDefaultCuts(img.naturalHeight);
    renderCutLines();
  };
  img.src = '/api/pages/' + index + '/image';
  await refreshPage(index);
}

async function refreshPage(index) {
  const res = await fetch('/api/pages/' + index);
  const data = await res.json();
  const preview = document.getElementById('preview');
  if (data.status === 'image') {
    preview.innerHTML = '<em>هذه الصفحة معلَّمة كصورة — هتتضاف للكتاب النهائي كما هي، بدون نص وبدون OCR.</em>';
  } else if (data.words && data.words.length) {
    preview.innerHTML = data.words.map(w =>
      w.low_confidence
        ? '<span class="low-confidence">' + escapeHtml(w.text) + '</span>'
        : escapeHtml(w.text)
    ).join(' ');
  } else {
    preview.innerHTML = '';
  }
  document.getElementById('editor').value = data.text || '';
  let statusText = statusLabel(data.status) + (data.error ? (': ' + data.error) : '');
  if (data.status === 'ok' && (!data.words || data.words.length === 0)) {
    statusText += ' — نجحت العملية لكن لم يُعثر على أي نص؛ جرّب "إعادة محاولة OCR" مرة تانية.';
  }
  document.getElementById('status').textContent = statusText;
}

function startProgressPolling(index) {
  const wrap = document.getElementById('progressBarWrap');
  const bar = document.getElementById('progressBar');
  wrap.style.display = 'block';
  bar.style.width = '0%';
  progressTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/pages/' + index + '/progress');
      const data = await res.json();
      if (data.total > 0) {
        const pct = Math.round((data.current / data.total) * 100);
        bar.style.width = pct + '%';
        document.getElementById('status').textContent =
          'جاري المعالجة: شريحة ' + data.current + ' من ' + data.total + ' (' + pct + '%)';
      }
    } catch (e) { /* ignore transient poll errors */ }
  }, 400);
}

function stopProgressPolling() {
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
  document.getElementById('progressBarWrap').style.display = 'none';
  document.getElementById('progressBar').style.width = '0%';
}

function selectedDevice() {
  const value = document.getElementById('deviceSelect').value;
  return value === 'auto' ? null : value;
}

document.getElementById('retryBtn').onclick = async () => {
  if (currentIndex === null) return;
  document.getElementById('status').textContent = 'جاري إعادة المحاولة...';
  startProgressPolling(currentIndex);
  try {
    // Only send cuts the person actually placed by hand. The blue lines shown
    // by default are just a visual suggestion of where the automatic grid
    // would split, computed with no idea where the real text lines are - as
    // real cut points they can (and did) slice straight through a line.
    // Left untouched, retry must use the backend's own overlap+dedup tiling,
    // which is built to survive exactly that kind of bad split.
    const res = await fetch('/api/pages/' + currentIndex + '/retry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manual_cuts: cutsEdited ? manualCuts : null, device: selectedDevice() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشلت إعادة المحاولة: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  } finally {
    stopProgressPolling();
  }
  await refreshPage(currentIndex);
  await loadPages();
  await loadDevice();
};

document.getElementById('prepareBtn').onclick = async () => {
  const btn = document.getElementById('prepareBtn');
  btn.disabled = true;
  document.getElementById('status').textContent = 'جاري تجهيز المحرك (أول مرة بتاخد ثواني)...';
  try {
    const res = await fetch('/api/prepare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device: selectedDevice() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل التجهيز: ' + (err.error || res.status);
      return;
    }
    document.getElementById('status').textContent = 'المحرك جاهز.';
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  } finally {
    btn.disabled = false;
  }
  await loadDevice();
};

document.getElementById('markImageBtn').onclick = async () => {
  if (currentIndex === null) return;
  document.getElementById('status').textContent = 'جاري التحديد كصورة...';
  try {
    const res = await fetch('/api/pages/' + currentIndex + '/mark-image', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  }
  await refreshPage(currentIndex);
  await loadPages();
};

document.getElementById('saveBtn').onclick = async () => {
  if (currentIndex === null) return;
  const text = document.getElementById('editor').value;
  document.getElementById('status').textContent = 'جاري الحفظ...';
  try {
    const res = await fetch('/api/pages/' + currentIndex + '/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل الحفظ: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  }
  await refreshPage(currentIndex);
  await loadPages();
};

function startBulkPolling() {
  document.getElementById('bulkBarWrap').style.display = 'block';
  document.getElementById('processAllBtn').style.display = 'none';
  document.getElementById('stopAllBtn').style.display = 'inline-block';
  bulkTimer = setInterval(pollBulkProgress, 800);
}

function stopBulkPolling() {
  if (bulkTimer) { clearInterval(bulkTimer); bulkTimer = null; }
  document.getElementById('bulkBarWrap').style.display = 'none';
  document.getElementById('bulkBar').style.width = '0%';
  document.getElementById('processAllBtn').style.display = 'inline-block';
  document.getElementById('stopAllBtn').style.display = 'none';
  document.getElementById('bulkStatus').textContent = '';
}

async function pollBulkProgress() {
  try {
    const res = await fetch('/api/process-all/progress');
    const data = await res.json();
    if (data.total > 0) {
      const pct = Math.round((data.done / data.total) * 100);
      document.getElementById('bulkBar').style.width = pct + '%';
      let text = 'جاري معالجة الصفحات: ' + data.done + ' من ' + data.total;
      if (data.current_index !== null && data.current_index !== undefined) {
        text += ' (صفحة ' + (data.current_index + 1) + ')';
      }
      document.getElementById('bulkStatus').textContent = text;
    } else {
      document.getElementById('bulkStatus').textContent = 'كل الصفحات معالجة بالفعل.';
    }
    await loadPages();
    if (!data.running) {
      stopBulkPolling();
      if (currentIndex !== null) await refreshPage(currentIndex);
      await loadDevice();
    }
  } catch (e) { /* ignore transient poll errors */ }
}

document.getElementById('processAllBtn').onclick = async () => {
  try {
    const res = await fetch('/api/process-all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device: selectedDevice() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      document.getElementById('status').textContent = 'فشل بدء المعالجة: ' + (err.error || res.status);
      return;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'فشل الاتصال بالخادم: ' + e;
    return;
  }
  startBulkPolling();
};

document.getElementById('stopAllBtn').onclick = async () => {
  try {
    await fetch('/api/process-all/stop', { method: 'POST' });
  } catch (e) { /* server will stop on its own next poll either way */ }
};

document.getElementById('saveEpubBtn').onclick = async () => {
  const btn = document.getElementById('saveEpubBtn');
  const title = document.getElementById('bookTitleInput').value.trim();
  btn.disabled = true;
  document.getElementById('saveStatus').textContent = 'جاري بناء ملف EPUB...';
  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      document.getElementById('saveStatus').textContent = 'فشل الحفظ: ' + (data.error || res.status);
      return;
    }
    let text = 'تم الحفظ في: ' + data.output_path;
    if (data.missing_pages > 0) {
      text += ' (تنبيه: ' + data.missing_pages + ' صفحة لسه ماتعالجتش، اتحطّت كصفحات فاشلة في الكتاب)';
    }
    document.getElementById('saveStatus').textContent = text;
  } catch (e) {
    document.getElementById('saveStatus').textContent = 'فشل الاتصال بالخادم: ' + e;
  } finally {
    btn.disabled = false;
  }
};

loadPages();
loadDevice();
loadTilingDefaults();
loadBooks();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epubconv-review",
        description="Local web UI to inspect flagged OCR pages, retry them, or edit text by hand.",
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="PDF file or folder of page images (optional - omit to pick one from the browser)",
    )
    parser.add_argument("--lang", default="ar", help="OCR language code (default: ar)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default: 300)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Confidence below which a word is flagged (default: 0.70)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="OCR device for retries (default: auto)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".epubconv_cache"),
        help="Same cache directory used by the epubconv run being reviewed",
    )
    parser.add_argument("--port", type=int, default=8765, help="Local server port (default: 8765)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open a browser tab automatically"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.source is not None and not args.source.exists():
        parser.error(f"Source not found: {args.source}")

    config = ConversionConfig(
        lang=args.lang,
        dpi=args.dpi,
        threshold=args.threshold,
        device=args.device,
        cache_root=args.cache_dir,
    )
    serve(args.source, config, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
