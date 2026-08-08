import json
import threading
import urllib.request
from pathlib import Path

import fitz
import pytest

from epubconv import review_server
from epubconv.models import PageResult, PageStatus
from epubconv.pipeline import ConversionConfig
from epubconv.review_server import (
    ReviewState,
    _build_manual_result,
    _page_summary,
    _page_view,
    create_server,
)


def _make_pdf(path: Path, page_count: int) -> None:
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 20), f"page {i}")
    doc.save(path)
    doc.close()


def _ok_result(index: int, low_confidence_words: int = 0, total_words: int = 3) -> PageResult:
    return PageResult(
        index=index, status=PageStatus.OK, total_words=total_words,
        low_confidence_words=low_confidence_words,
    )


def test_page_summary_for_pending_page():
    assert _page_summary(None, 5) == {"index": 5, "status": "pending"}


def test_page_summary_for_ok_page():
    result = _ok_result(0, low_confidence_words=2, total_words=10)
    summary = _page_summary(result, 0)
    assert summary["status"] == "ok"
    assert summary["low_confidence_words"] == 2
    assert summary["total_words"] == 10


def test_page_view_for_pending_page():
    view = _page_view(None, 3)
    assert view == {"index": 3, "status": "pending", "text": "", "error": None}


def test_build_manual_result_splits_paragraphs_on_blank_lines():
    text = "أول فقرة\nسطر تاني\n\nفقرة تانية"
    result = _build_manual_result(0, text)
    assert result.status == PageStatus.OK
    assert len(result.blocks) == 2
    assert result.blocks[0].lines[0].text == "أول فقرة"
    assert result.blocks[0].lines[1].text == "سطر تاني"
    assert result.blocks[1].lines[0].text == "فقرة تانية"
    assert result.low_confidence_words == 0
    assert all(not w.low_confidence for b in result.blocks for l in b.lines for w in l.words)


def test_build_manual_result_ignores_empty_paragraphs():
    result = _build_manual_result(0, "\n\nنص\n\n\n")
    assert len(result.blocks) == 1


def test_page_view_round_trips_manual_result():
    result = _build_manual_result(0, "سطر واحد")
    view = _page_view(result, 0)
    assert view["text"] == "سطر واحد"
    assert view["words"][0]["low_confidence"] is False


# --- ReviewState ---


def test_review_state_list_pages_reports_pending_for_uncached(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    pages = state.list_pages()

    assert len(pages) == 3
    assert all(p["status"] == "pending" for p in pages)


def test_review_state_retry_saves_to_cache(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 2)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    monkeypatch.setattr(
        review_server, "process_page", lambda image, index, lang, threshold, device, manual_cuts=None, on_progress=None: _ok_result(index)
    )

    view = state.retry(1)

    assert view["status"] == "ok"
    assert state.cache.has(1)


def test_review_state_retry_forwards_manual_cuts_to_process_page(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    seen = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen["manual_cuts"] = manual_cuts
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.retry(0, manual_cuts=[100, 250])

    assert seen["manual_cuts"] == [100, 250]


def test_review_state_retry_reports_and_then_clears_progress(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    seen_mid_call = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        on_progress(1, 3)
        seen_mid_call["progress"] = dict(state.get_progress(index))
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    assert state.get_progress(0) == {"current": 0, "total": 0}
    state.retry(0)

    assert seen_mid_call["progress"] == {"current": 1, "total": 3}
    # cleared once the retry call has fully returned
    assert state.get_progress(0) == {"current": 0, "total": 0}


def test_review_state_get_device_reflects_ocr_module(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    monkeypatch.setattr(review_server.ocr, "get_last_resolved_device", lambda: "gpu")

    assert state.get_device() == {"device": "gpu"}


def test_review_state_get_tiling_defaults(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    assert state.get_tiling_defaults() == {
        "strip_height": review_server.ocr.DEFAULT_STRIP_HEIGHT,
        "overlap": review_server.ocr.DEFAULT_STRIP_OVERLAP,
    }


def test_review_state_prepare_engine_uses_requested_device(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache", device="auto")
    state = ReviewState(pdf_path, config)
    seen = {}

    def fake_get_engine(lang, device):
        seen["lang"] = lang
        seen["device"] = device
        return object()

    monkeypatch.setattr(review_server.ocr, "get_engine", fake_get_engine)
    monkeypatch.setattr(review_server.ocr, "get_last_resolved_device", lambda: "cpu")

    result = state.prepare_engine(device="cpu")

    assert seen == {"lang": config.lang, "device": "cpu"}
    assert result == {"device": "cpu"}


def test_review_state_prepare_engine_falls_back_to_config_device(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache", device="gpu")
    state = ReviewState(pdf_path, config)
    seen = {}

    monkeypatch.setattr(
        review_server.ocr, "get_engine", lambda lang, device: seen.setdefault("device", device)
    )
    monkeypatch.setattr(review_server.ocr, "get_last_resolved_device", lambda: "gpu")

    state.prepare_engine()

    assert seen["device"] == "gpu"


def test_review_state_retry_forwards_device_override(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache", device="auto")
    state = ReviewState(pdf_path, config)
    seen = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen["device"] = device
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.retry(0, device="cpu")

    assert seen["device"] == "cpu"


def test_review_state_retry_without_override_uses_config_device(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache", device="gpu")
    state = ReviewState(pdf_path, config)
    seen = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen["device"] = device
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.retry(0)

    assert seen["device"] == "gpu"


def test_review_state_save_manual_edit_persists(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    view = state.save_manual_edit(0, "نص معدَّل يدويًا")

    assert view["text"] == "نص معدَّل يدويًا"
    cached = state.cache.load(0)
    assert cached is not None
    assert cached.blocks[0].lines[0].text == "نص معدَّل يدويًا"


def test_review_state_render_image_png_returns_valid_png_bytes(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    png_bytes = state.render_image_png(0)

    assert png_bytes.startswith(b"\x89PNG")


def test_review_state_mark_as_image_persists_image_status(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)

    view = state.mark_as_image(0)

    assert view["status"] == "image"
    cached = state.cache.load(0)
    assert cached is not None
    assert cached.status == PageStatus.IMAGE
    assert cached.blocks == ()


def test_review_state_retry_after_mark_as_image_overwrites_it(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    monkeypatch.setattr(
        review_server, "process_page", lambda image, index, lang, threshold, device, manual_cuts=None, on_progress=None: _ok_result(index)
    )

    state.mark_as_image(0)
    view = state.retry(0)

    assert view["status"] == "ok"
    assert state.cache.load(0).status == PageStatus.OK


def test_list_books_in_dir_finds_pdfs_case_insensitively(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "B.PDF").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    books = review_server.list_books_in_dir(tmp_path)

    assert [b["name"] for b in books] == ["a.pdf", "B.PDF"]


def test_list_books_in_dir_on_missing_directory_returns_empty():
    assert review_server.list_books_in_dir(Path("does/not/exist")) == []


def test_books_dir_returns_a_books_folder_and_creates_it():
    directory = review_server.books_dir()
    assert directory.name == "books"
    assert directory.is_dir()


def test_save_uploaded_book_writes_into_the_given_directory(tmp_path: Path):
    saved = review_server.save_uploaded_book("my book.pdf", b"%PDF-1.4 fake", directory=tmp_path)

    assert saved.exists()
    assert saved.read_bytes() == b"%PDF-1.4 fake"
    assert saved.parent == tmp_path


def test_save_uploaded_book_sanitizes_unsafe_characters(tmp_path: Path):
    saved = review_server.save_uploaded_book("weird/name:*?.pdf", b"content", directory=tmp_path)

    assert saved.parent == tmp_path
    assert saved.suffix == ".pdf"
    assert "/" not in saved.name and ":" not in saved.name


def test_save_uploaded_book_avoids_overwriting_an_existing_file(tmp_path: Path):
    first = review_server.save_uploaded_book("book.pdf", b"first", directory=tmp_path)
    second = review_server.save_uploaded_book("book.pdf", b"second", directory=tmp_path)

    assert first != second
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_save_uploaded_book_preserves_arabic_filenames(tmp_path: Path):
    saved = review_server.save_uploaded_book("كتابي الجميل.pdf", b"content", directory=tmp_path)

    assert "كتابي" in saved.name


# --- ReviewState with no book loaded ---


def test_review_state_with_no_source_has_zero_pages(tmp_path: Path):
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(None, config)

    assert state.page_count == 0
    assert state.list_pages() == []
    assert state.source_path is None
    assert state.cache is None


def test_review_state_with_no_source_get_page_returns_pending(tmp_path: Path):
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(None, config)

    view = state.get_page(0)

    assert view["status"] == "pending"


@pytest.mark.parametrize(
    "method_call",
    [
        lambda state: state.render_image_png(0),
        lambda state: state.retry(0),
        lambda state: state.save_manual_edit(0, "نص"),
        lambda state: state.mark_as_image(0),
        lambda state: state.start_bulk_process(),
        lambda state: state.build_final_epub(),
    ],
)
def test_review_state_with_no_source_rejects_actions_needing_a_book(tmp_path: Path, method_call):
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(None, config)

    with pytest.raises(ValueError):
        method_call(state)


def test_review_state_start_bulk_process_runs_every_pending_page(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    seen_indices = []

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen_indices.append(index)
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.start_bulk_process()
    state._bulk_thread.join(timeout=5)

    assert sorted(seen_indices) == [0, 1, 2]
    assert state.bulk_state["running"] is False
    assert state.bulk_state["done"] == 3
    assert all(state.cache.has(i) for i in range(3))


def test_review_state_start_bulk_process_skips_already_cached_pages(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 2)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    state.cache.save(_ok_result(0))
    seen_indices = []

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen_indices.append(index)
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.start_bulk_process()
    state._bulk_thread.join(timeout=5)

    assert seen_indices == [1]


def test_review_state_start_bulk_process_ignores_a_second_call_while_running(
    tmp_path: Path, monkeypatch
):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 2)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    release = threading.Event()

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        release.wait(timeout=5)
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    first = state.start_bulk_process()
    second = state.start_bulk_process()
    release.set()
    state._bulk_thread.join(timeout=5)

    assert first["running"] is True
    assert second == first  # the second call was a no-op, not a fresh run


def test_review_state_stop_bulk_process_halts_before_remaining_pages(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    started = threading.Event()
    seen_indices = []

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen_indices.append(index)
        started.set()
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    state.start_bulk_process()
    started.wait(timeout=5)
    state.stop_bulk_process()
    state._bulk_thread.join(timeout=5)

    assert len(seen_indices) < 3


def test_review_state_build_final_epub_uses_source_stem_as_default_title(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    state.cache.save(_ok_result(0))

    summary = state.build_final_epub()

    output_path = Path(summary["output_path"])
    assert output_path.exists()
    assert output_path == pdf_path.with_suffix(".epub")
    assert Path(summary["report_path"]).exists()


def test_review_state_build_final_epub_uses_a_given_title(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    state.cache.save(_ok_result(0))

    state.build_final_epub(title="عنوان مخصّص")

    from ebooklib import epub

    book = epub.read_epub(str(pdf_path.with_suffix(".epub")))
    assert book.get_metadata("DC", "title")[0][0] == "عنوان مخصّص"


def test_review_state_build_final_epub_counts_pages_never_processed(tmp_path: Path):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 3)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    state.cache.save(_ok_result(0))
    # pages 1 and 2 are left uncached (never processed)

    summary = state.build_final_epub()

    assert summary["missing_pages"] == 2
    assert summary["failed_pages"] == 2
    assert summary["page_count"] == 3


def test_review_state_build_final_epub_never_drops_an_unprocessed_page(tmp_path: Path):
    # The exported EPUB must still have a chapter for every page, even one
    # never sent through OCR - a placeholder, never a silent gap in the book.
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 2)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    state = ReviewState(pdf_path, config)
    state.cache.save(_ok_result(0))

    state.build_final_epub()

    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(pdf_path.with_suffix(".epub")))
    chapters = [item for item in book.get_items() if item.get_type() == ITEM_DOCUMENT]
    page_chapters = [item for item in chapters if item.file_name.startswith("page_")]
    assert len(page_chapters) == 2


# --- HTTP layer ---


@pytest.fixture
def running_server(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 2)
    config = ConversionConfig(cache_root=tmp_path / "cache")
    monkeypatch.setattr(
        review_server, "process_page", lambda image, index, lang, threshold, device, manual_cuts=None, on_progress=None: _ok_result(index)
    )
    server = create_server(pdf_path, config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, payload: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_index_page_served(running_server):
    status, body = _get(running_server + "/")
    assert status == 200
    assert b"epubconv" in body or "مراجعة".encode("utf-8") in body


def test_list_pages_endpoint(running_server):
    status, body = _get(running_server + "/api/pages")
    assert status == 200
    pages = json.loads(body)
    assert len(pages) == 2
    assert all(p["status"] == "pending" for p in pages)


def test_page_image_endpoint_returns_png(running_server):
    status, body = _get(running_server + "/api/pages/0/image")
    assert status == 200
    assert body.startswith(b"\x89PNG")


def test_retry_endpoint_updates_cache_and_listing(running_server):
    status, body = _post(running_server + "/api/pages/0/retry")
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "ok"

    _, listing_body = _get(running_server + "/api/pages")
    pages = json.loads(listing_body)
    assert pages[0]["status"] == "ok"


def test_device_endpoint_reports_current_device(running_server, monkeypatch):
    monkeypatch.setattr(review_server.ocr, "get_last_resolved_device", lambda: "cpu")

    status, body = _get(running_server + "/api/device")

    assert status == 200
    assert json.loads(body) == {"device": "cpu"}


def test_progress_endpoint_returns_zero_when_nothing_in_flight(running_server):
    status, body = _get(running_server + "/api/pages/0/progress")
    assert status == 200
    assert json.loads(body) == {"current": 0, "total": 0}


def test_retry_endpoint_accepts_manual_cuts_in_body(running_server, monkeypatch):
    seen = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen["manual_cuts"] = manual_cuts
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    status, body = _post(running_server + "/api/pages/0/retry", {"manual_cuts": [50, 120]})

    assert status == 200
    assert seen["manual_cuts"] == [50, 120]


def test_retry_endpoint_accepts_device_override_in_body(running_server, monkeypatch):
    seen = {}

    def fake_process_page(image, index, lang, threshold, device, manual_cuts=None, on_progress=None):
        seen["device"] = device
        return _ok_result(index)

    monkeypatch.setattr(review_server, "process_page", fake_process_page)

    status, body = _post(running_server + "/api/pages/0/retry", {"device": "cpu"})

    assert status == 200
    assert seen["device"] == "cpu"


def test_tiling_defaults_endpoint(running_server):
    status, body = _get(running_server + "/api/tiling-defaults")
    assert status == 200
    data = json.loads(body)
    assert data == {
        "strip_height": review_server.ocr.DEFAULT_STRIP_HEIGHT,
        "overlap": review_server.ocr.DEFAULT_STRIP_OVERLAP,
    }


def test_prepare_endpoint_builds_engine_with_requested_device(running_server, monkeypatch):
    seen = {}

    def fake_get_engine(lang, device):
        seen["device"] = device
        return object()

    monkeypatch.setattr(review_server.ocr, "get_engine", fake_get_engine)
    monkeypatch.setattr(review_server.ocr, "get_last_resolved_device", lambda: "cpu")

    status, body = _post(running_server + "/api/prepare", {"device": "cpu"})

    assert status == 200
    assert json.loads(body) == {"device": "cpu"}
    assert seen["device"] == "cpu"


def test_edit_endpoint_saves_manual_text(running_server):
    status, body = _post(running_server + "/api/pages/1/edit", {"text": "نص يدوي"})
    assert status == 200
    data = json.loads(body)
    assert data["text"] == "نص يدوي"

    _, page_body = _get(running_server + "/api/pages/1")
    assert json.loads(page_body)["text"] == "نص يدوي"


def test_unknown_route_returns_404(running_server):
    status, _ = _get(running_server + "/api/nope")
    assert status == 404


def test_mark_image_endpoint_sets_status_and_reflects_in_listing(running_server):
    status, body = _post(running_server + "/api/pages/0/mark-image")
    assert status == 200
    assert json.loads(body)["status"] == "image"

    _, listing_body = _get(running_server + "/api/pages")
    listing = json.loads(listing_body)
    assert listing[0]["status"] == "image"


def test_books_endpoint_lists_pdfs_next_to_current_book(running_server, tmp_path: Path):
    _make_pdf(tmp_path / "other.pdf", 1)

    status, body = _get(running_server + "/api/books")

    assert status == 200
    data = json.loads(body)
    assert data["current_name"] == "book.pdf"
    names = {b["name"] for b in data["books"]}
    assert {"book.pdf", "other.pdf"} <= names


def test_books_open_endpoint_switches_the_active_book(running_server, tmp_path: Path):
    other_path = tmp_path / "other.pdf"
    _make_pdf(other_path, 5)

    status, body = _post(running_server + "/api/books/open", {"path": str(other_path)})

    assert status == 200
    data = json.loads(body)
    assert data["current_name"] == "other.pdf"
    assert data["page_count"] == 5

    _, pages_body = _get(running_server + "/api/pages")
    assert len(json.loads(pages_body)) == 5


def test_books_open_endpoint_rejects_a_missing_path(running_server, tmp_path: Path):
    status, body = _post(
        running_server + "/api/books/open", {"path": str(tmp_path / "nope.pdf")}
    )
    assert status == 404


def test_books_upload_endpoint_saves_and_switches_to_the_uploaded_book(
    running_server, tmp_path: Path, monkeypatch
):
    import base64

    upload_dir = tmp_path / "uploaded_books"
    monkeypatch.setattr(review_server, "books_dir", lambda: upload_dir)

    real_pdf_bytes = (tmp_path / "throwaway.pdf")
    _make_pdf(real_pdf_bytes, 4)
    content_b64 = base64.b64encode(real_pdf_bytes.read_bytes()).decode("ascii")

    status, body = _post(
        running_server + "/api/books/upload",
        {"filename": "كتاب صاحبي.pdf", "content": content_b64},
    )

    assert status == 200
    data = json.loads(body)
    assert data["page_count"] == 4
    assert upload_dir.is_dir()
    saved_files = list(upload_dir.glob("*.pdf"))
    assert len(saved_files) == 1
    assert "كتاب" in saved_files[0].name


def test_books_upload_endpoint_rejects_non_pdf_content(running_server, tmp_path: Path, monkeypatch):
    import base64

    monkeypatch.setattr(review_server, "books_dir", lambda: tmp_path / "uploaded_books")
    content_b64 = base64.b64encode(b"not actually a pdf").decode("ascii")

    status, body = _post(
        running_server + "/api/books/upload", {"filename": "fake.pdf", "content": content_b64}
    )

    assert status == 400


def test_books_upload_endpoint_requires_filename_and_content(running_server):
    status, _ = _post(running_server + "/api/books/upload", {"filename": "", "content": ""})
    assert status == 400


def test_process_all_endpoint_processes_every_pending_page(running_server):
    status, body = _post(running_server + "/api/process-all")
    assert status == 200
    assert json.loads(body)["running"] is True

    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        _, progress_body = _get(running_server + "/api/process-all/progress")
        if not json.loads(progress_body)["running"]:
            break
        time.sleep(0.05)

    _, pages_body = _get(running_server + "/api/pages")
    pages = json.loads(pages_body)
    assert all(p["status"] == "ok" for p in pages)


def test_process_all_stop_endpoint_is_reachable(running_server):
    _post(running_server + "/api/process-all")
    status, _ = _post(running_server + "/api/process-all/stop")
    assert status == 200


def test_save_endpoint_builds_epub_and_reports_missing_pages(running_server):
    _post(running_server + "/api/pages/0/retry")  # only page 0 processed, page 1 left pending

    status, body = _post(running_server + "/api/save")

    assert status == 200
    data = json.loads(body)
    assert Path(data["output_path"]).exists()
    assert data["missing_pages"] == 1


def test_save_endpoint_accepts_a_custom_title(running_server, tmp_path: Path):
    status, body = _post(running_server + "/api/save", {"title": "عنوان مخصّص"})
    assert status == 200

    from ebooklib import epub

    output_path = Path(json.loads(body)["output_path"])
    book = epub.read_epub(str(output_path))
    assert book.get_metadata("DC", "title")[0][0] == "عنوان مخصّص"


# --- CLI ---


def test_build_parser_defaults():
    parser = review_server.build_parser()
    args = parser.parse_args(["source.pdf"])
    assert args.lang == "ar"
    assert args.dpi == 300
    assert args.port == 8765
    assert args.device == "auto"


def test_main_errors_on_missing_source(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(SystemExit) as exc_info:
        review_server.main([str(missing)])
    assert exc_info.value.code != 0


def test_main_starts_server_and_serves_forever_until_stopped(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)

    created = {}

    class FakeServer:
        def __init__(self):
            self.server_address = ("127.0.0.1", 12345)

        def serve_forever(self):
            created["served"] = True

        def server_close(self):
            created["closed"] = True

    def fake_create_server(source_path, config, port):
        created["source_path"] = source_path
        created["port"] = port
        return FakeServer()

    monkeypatch.setattr(review_server, "create_server", fake_create_server)
    opened = []
    monkeypatch.setattr(review_server.webbrowser, "open", lambda url: opened.append(url))

    exit_code = review_server.main([str(pdf_path), "--port", "9999"])

    assert exit_code == 0
    assert created["port"] == 9999
    assert created["served"] is True
    assert created["closed"] is True
    assert opened == ["http://127.0.0.1:12345"]  # opens a browser by default


def test_main_no_browser_flag_skips_opening_a_browser(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    _make_pdf(pdf_path, 1)

    class FakeServer:
        server_address = ("127.0.0.1", 12345)

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(review_server, "create_server", lambda *a, **k: FakeServer())
    opened = []
    monkeypatch.setattr(review_server.webbrowser, "open", lambda url: opened.append(url))

    review_server.main([str(pdf_path), "--no-browser"])

    assert opened == []


def test_main_allows_no_source_argument(monkeypatch):
    class FakeServer:
        server_address = ("127.0.0.1", 12345)

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setattr(review_server, "create_server", lambda *a, **k: FakeServer())
    monkeypatch.setattr(review_server.webbrowser, "open", lambda url: None)

    exit_code = review_server.main(["--no-browser"])

    assert exit_code == 0
