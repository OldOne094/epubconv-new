import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image

from epubconv import pipeline
from epubconv.models import PageResult, PageStatus


def _make_image_dir(path: Path, count: int) -> None:
    path.mkdir()
    for i in range(count):
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(path / f"{i:03d}.png")


def _fake_ok_result(index: int) -> PageResult:
    return PageResult(index=index, status=PageStatus.OK, total_words=1)


def test_convert_processes_every_page_sequentially(tmp_path, monkeypatch):
    calls = []

    def fake_process_page(image, index, lang, threshold, device="auto"):
        calls.append(index)
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 4)
    config = pipeline.ConversionConfig(resume=False, cache_root=tmp_path / "cache")

    result = pipeline.convert(source_dir, "كتاب", config)

    assert sorted(calls) == [0, 1, 2, 3]
    assert len(result.pages) == 4
    assert all(p.status == PageStatus.OK for p in result.pages)


def test_convert_max_pages_only_processes_the_first_n_pages(tmp_path, monkeypatch):
    calls = []

    def fake_process_page(image, index, lang, threshold, device="auto"):
        calls.append(index)
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 5)
    config = pipeline.ConversionConfig(resume=False, cache_root=tmp_path / "cache")

    result = pipeline.convert(source_dir, "كتاب", config, max_pages=2)

    assert sorted(calls) == [0, 1]
    assert {p.index for p in result.pages} == {0, 1}
    assert result.meta.page_count == 5  # full book length is still recorded


def test_convert_max_pages_reuses_cache_on_a_later_full_run(tmp_path, monkeypatch):
    calls = []

    def fake_process_page(image, index, lang, threshold, device="auto"):
        calls.append(index)
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 4)
    cache_root = tmp_path / "cache"
    config = pipeline.ConversionConfig(resume=True, cache_root=cache_root)

    pipeline.convert(source_dir, "كتاب", config, max_pages=2)
    assert sorted(calls) == [0, 1]
    calls.clear()

    full_result = pipeline.convert(source_dir, "كتاب", config)

    assert sorted(calls) == [2, 3]  # pages 0,1 came from cache, not reprocessed
    assert {p.index for p in full_result.pages} == {0, 1, 2, 3}


def test_convert_isolates_a_failing_page_instead_of_aborting(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_: None)

    def fake_process_page(image, index, lang, threshold, device="auto"):
        if index == 2:
            raise ValueError("boom")
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 4)
    config = pipeline.ConversionConfig(resume=False, max_retries=1, cache_root=tmp_path / "cache")

    result = pipeline.convert(source_dir, "كتاب", config)

    assert len(result.pages) == 4
    failed = [p for p in result.pages if p.status == PageStatus.FAILED]
    assert [p.index for p in failed] == [2]
    assert failed[0].error == "boom"
    assert failed[0].attempts == 2  # 1 initial attempt + 1 retry
    assert all(p.status == PageStatus.OK for p in result.pages if p.index != 2)


def test_convert_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_: None)
    attempts_seen = {}

    def flaky_process_page(image, index, lang, threshold, device="auto"):
        attempts_seen[index] = attempts_seen.get(index, 0) + 1
        if index == 1 and attempts_seen[index] < 2:
            raise RuntimeError("transient")
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", flaky_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 2)
    config = pipeline.ConversionConfig(resume=False, max_retries=2, cache_root=tmp_path / "cache")

    result = pipeline.convert(source_dir, "كتاب", config)

    page1 = next(p for p in result.pages if p.index == 1)
    assert page1.status == PageStatus.OK
    assert page1.attempts == 2


def test_convert_resume_skips_already_cached_pages(tmp_path, monkeypatch):
    calls = []

    def fake_process_page(image, index, lang, threshold, device="auto"):
        calls.append(index)
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 3)
    cache_root = tmp_path / "cache"
    config = pipeline.ConversionConfig(resume=True, cache_root=cache_root)

    first = pipeline.convert(source_dir, "كتاب", config)
    assert sorted(calls) == [0, 1, 2]

    calls.clear()
    second = pipeline.convert(source_dir, "كتاب", config)

    assert calls == []  # nothing re-processed; everything came from cache
    assert len(second.pages) == 3
    assert {p.index for p in second.pages} == {0, 1, 2}


def test_convert_no_resume_reprocesses_even_with_existing_cache(tmp_path, monkeypatch):
    calls = []

    def fake_process_page(image, index, lang, threshold, device="auto"):
        calls.append(index)
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 2)
    cache_root = tmp_path / "cache"

    pipeline.convert(source_dir, "كتاب", pipeline.ConversionConfig(resume=True, cache_root=cache_root))
    calls.clear()

    pipeline.convert(
        source_dir, "كتاب", pipeline.ConversionConfig(resume=False, cache_root=cache_root)
    )

    assert sorted(calls) == [0, 1]


def test_convert_parallel_path_processes_all_pages(tmp_path, monkeypatch):
    # Use real threads instead of processes so the in-test monkeypatch of
    # process_page is visible to the workers (ProcessPoolExecutor would spawn
    # separate interpreters that never see this monkeypatch).
    monkeypatch.setattr(
        pipeline.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )

    def fake_process_page(image, index, lang, threshold, device="auto"):
        return _fake_ok_result(index)

    monkeypatch.setattr(pipeline, "process_page", fake_process_page)

    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 6)
    config = pipeline.ConversionConfig(resume=False, workers=3, cache_root=tmp_path / "cache")

    result = pipeline.convert(source_dir, "كتاب", config)

    assert {p.index for p in result.pages} == {0, 1, 2, 3, 4, 5}
    assert all(p.status == PageStatus.OK for p in result.pages)
