from pathlib import Path

from PIL import Image

from epubconv import cli
from epubconv.models import ConversionResult, DocumentMeta, PageResult, PageStatus


def _make_image_dir(path: Path, count: int) -> None:
    path.mkdir()
    for i in range(count):
        Image.new("RGB", (16, 16), color=(0, 0, 0)).save(path / f"{i:03d}.png")


def test_build_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["source_dir"])
    assert args.lang == "ar"
    assert args.dpi == 300
    assert args.workers == 1
    assert args.max_retries == 2
    assert args.no_resume is False


def test_default_output_for_file(tmp_path: Path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-fake")
    assert cli._default_output(source) == tmp_path / "book.epub"


def test_default_output_for_directory(tmp_path: Path):
    source_dir = tmp_path / "pages"
    source_dir.mkdir()
    assert cli._default_output(source_dir) == tmp_path / "pages.epub"


def test_main_end_to_end_with_stubbed_conversion(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 2)

    def fake_convert(source_path, title, config, on_page_done=None, max_pages=None):
        pages = [PageResult(index=0, status=PageStatus.OK, total_words=3)]
        if on_page_done:
            on_page_done(pages[0], 1)
        return ConversionResult(meta=DocumentMeta(title=title), pages=pages)

    monkeypatch.setattr(cli, "convert", fake_convert)

    output_path = tmp_path / "out.epub"
    exit_code = cli.main([str(source_dir), "-o", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.with_suffix(".report.html").exists()


def test_main_returns_nonzero_when_every_page_fails(tmp_path, monkeypatch):
    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 1)

    def fake_convert(source_path, title, config, on_page_done=None, max_pages=None):
        pages = [PageResult(index=0, status=PageStatus.FAILED, error="boom")]
        return ConversionResult(meta=DocumentMeta(title=title), pages=pages)

    monkeypatch.setattr(cli, "convert", fake_convert)

    output_path = tmp_path / "out.epub"
    exit_code = cli.main([str(source_dir), "-o", str(output_path)])

    assert exit_code == 1


def test_main_forwards_max_pages_to_convert(tmp_path, monkeypatch):
    source_dir = tmp_path / "pages"
    _make_image_dir(source_dir, 5)
    seen = {}

    def fake_convert(source_path, title, config, on_page_done=None, max_pages=None):
        seen["max_pages"] = max_pages
        pages = [PageResult(index=0, status=PageStatus.OK, total_words=1)]
        return ConversionResult(meta=DocumentMeta(title=title), pages=pages)

    monkeypatch.setattr(cli, "convert", fake_convert)

    cli.main([str(source_dir), "-o", str(tmp_path / "out.epub"), "--max-pages", "10"])

    assert seen["max_pages"] == 10


def test_main_errors_on_missing_source(tmp_path, capsys):
    missing = tmp_path / "does_not_exist"
    try:
        cli.main([str(missing)])
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code != 0
