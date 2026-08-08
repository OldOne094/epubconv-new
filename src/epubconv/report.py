"""Human-readable HTML conversion report.

All dynamic content is HTML-escaped before insertion, since page text and error
messages both flow from untrusted OCR/file input.
"""

from __future__ import annotations

import html
from pathlib import Path

from .models import ConversionResult, PageStatus

_TEMPLATE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<title>تقرير التحويل: {title}</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: right; }}
tr.failed {{ background-color: #fde2e2; }}
tr.low-confidence {{ background-color: #fff8dc; }}
.summary {{ margin-bottom: 1.5em; }}
</style>
</head>
<body>
<h1>تقرير تحويل: {title}</h1>
<div class="summary">
<p>عدد الصفحات: {page_count}</p>
<p>صفحات ناجحة: {ok_count}</p>
<p>صفحات فشلت: {failed_count}</p>
<p>متوسط نسبة الثقة: {avg_confidence:.1%}</p>
</div>
<table>
<tr><th>#</th><th>الحالة</th><th>عدد الكلمات</th><th>كلمات منخفضة الثقة</th><th>محاولات</th><th>ملاحظات</th></tr>
{rows}
</table>
</body>
</html>
"""

_ROW_TEMPLATE = (
    '<tr class="{css_class}"><td>{index}</td><td>{status}</td><td>{total_words}</td>'
    "<td>{low_confidence}</td><td>{attempts}</td><td>{note}</td></tr>"
)


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def build_report(result: ConversionResult) -> str:
    rows = []
    for page in result.pages:
        failed = page.status == PageStatus.FAILED
        is_image = page.status == PageStatus.IMAGE
        low_conf = not failed and not is_image and page.total_words > 0 and page.confidence_ratio < 0.9
        css_class = "failed" if failed else ("low-confidence" if low_conf else "")
        status = "فشل" if failed else ("صورة" if is_image else "تم")
        rows.append(
            _ROW_TEMPLATE.format(
                css_class=css_class,
                index=page.index + 1,
                status=status,
                total_words=page.total_words,
                low_confidence=page.low_confidence_words,
                attempts=page.attempts,
                note=_escape(page.error) if page.error else "",
            )
        )

    total = len(result.pages)
    ok = len(result.ok_pages)
    failed_count = len(result.failed_pages)
    avg_confidence = sum(p.confidence_ratio for p in result.ok_pages) / ok if ok else 0.0

    return _TEMPLATE.format(
        title=_escape(result.meta.title),
        page_count=total,
        ok_count=ok,
        failed_count=failed_count,
        avg_confidence=avg_confidence,
        rows="\n".join(rows),
    )


def write_report(result: ConversionResult, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(result), encoding="utf-8")
    return output_path
