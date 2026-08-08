from pathlib import Path

from epubconv.cache import PageCache, hash_file
from epubconv.models import PageResult, PageStatus


def test_hash_file_is_stable_and_content_sensitive(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_a.write_bytes(b"hello world")
    file_b = tmp_path / "b.txt"
    file_b.write_bytes(b"hello world")
    file_c = tmp_path / "c.txt"
    file_c.write_bytes(b"different content")

    assert hash_file(file_a) == hash_file(file_b)
    assert hash_file(file_a) != hash_file(file_c)


def test_save_and_load_roundtrip(tmp_path: Path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-fake")
    cache = PageCache(tmp_path / "cache", source)

    assert cache.has(0) is False
    assert cache.load(0) is None

    result = PageResult(index=0, status=PageStatus.OK, total_words=5, low_confidence_words=1)
    cache.save(result)

    assert cache.has(0) is True
    loaded = cache.load(0)
    assert loaded.index == 0
    assert loaded.status == PageStatus.OK
    assert loaded.total_words == 5


def test_different_source_content_gets_different_cache_dir(tmp_path: Path):
    source1 = tmp_path / "book1.pdf"
    source1.write_bytes(b"content-one")
    source2 = tmp_path / "book2.pdf"
    source2.write_bytes(b"content-two")

    cache1 = PageCache(tmp_path / "cache", source1)
    cache2 = PageCache(tmp_path / "cache", source2)

    assert cache1.dir != cache2.dir


def test_clear_removes_page_files_but_keeps_manifest(tmp_path: Path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"content")
    cache = PageCache(tmp_path / "cache", source)
    cache.save(PageResult(index=0, status=PageStatus.OK))

    cache.clear()

    assert cache.has(0) is False
    assert (cache.dir / "manifest.json").exists()


def test_load_survives_corrupt_json(tmp_path: Path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"content")
    cache = PageCache(tmp_path / "cache", source)
    cache._page_path(0).write_text("{not valid json", encoding="utf-8")

    assert cache.load(0) is None
