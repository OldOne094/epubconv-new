from epubconv.models import ConversionResult, DocumentMeta, PageResult, PageStatus
from epubconv.report import build_report


def test_report_escapes_html_in_title_and_errors():
    meta = DocumentMeta(title="<script>alert(1)</script>")
    page = PageResult(index=0, status=PageStatus.FAILED, error="<b>boom</b>")
    result = ConversionResult(meta=meta, pages=[page])

    html_out = build_report(result)

    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "&lt;b&gt;boom&lt;/b&gt;" in html_out


def test_report_summary_counts():
    meta = DocumentMeta(title="كتاب")
    pages = [
        PageResult(index=0, status=PageStatus.OK, total_words=10, low_confidence_words=0),
        PageResult(index=1, status=PageStatus.FAILED, error="oops"),
    ]
    result = ConversionResult(meta=meta, pages=pages)

    html_out = build_report(result)

    assert "صفحات ناجحة: 1" in html_out
    assert "صفحات فشلت: 1" in html_out
