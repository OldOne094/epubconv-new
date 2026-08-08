<div dir="auto" align="center">

# epubconv (new version)

**Arabic OCR → EPUB3 converter** · محوّل الكتب العربية (PDF أو صور) إلى EPUB3

Deterministic. No hallucination. Every low-confidence word is kept verbatim and left for human review.

</div>

---

## Table of Contents / فهرس المحتويات

- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Command-line usage](#command-line-usage)
- [Human review (web UI)](#human-review-web-ui)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [العربية](#العربية)

---

## About

`epubconv` converts scanned Arabic books — a **PDF file** or a **folder of page
images** — into structured, right-to-left **EPUB 3** files using PaddleOCR. It
follows a hard rule against hallucination:

> If the engine is not sure about a word, it is **never invented** — it is kept
> verbatim, flagged, and left for a person to review.

---

## Features

- **PDF / folder-of-images input** — PDFs are rasterized at configurable DPI
  (default 300); a folder of JPG/PNG/BMP/TIFF pages is read directly.
- **Arabic-aware OCR** via PaddleOCR (`lang="ar"`), CPU by default with optional
  NVIDIA GPU acceleration (verified at install time).
- **Deterministic output** — below the confidence threshold a word is *flagged*
  (`low-conf`), never guessed or silently "fixed".
- **RTL EPUB 3 output** — `dir="rtl" lang="ar"`, CSS, cover, TOC, and correct
  `title`/`author` metadata.
- **Resumable** — per-page results are cached (default `.epubconv_cache`), so an
  interrupted run picks up exactly where it stopped; `--no-resume` reprocesses.
- **Parallel OCR** — optional multi-worker processing.
- **Tall-page tiling** — very tall pages are split into overlapping strips for
  OCR, with automatic dedup of the overlap regions.
- **Local review web UI** — inspect flagged pages side-by-side with the source
  image, retry OCR (optionally with manual cut lines), edit text by hand, mark a
  page as an image, and rebuild the final EPUB from the cache.
- **Per-conversion HTML report** — pages, word counts, and low-confidence stats.
- **GPU auto-detection at install** — the installer probes for an NVIDIA GPU and
  installs a matching `paddlepaddle-gpu` build, verifying real computation
  before trusting it (falls back to CPU on any doubt).

---

## Requirements

- **Python 3.11 or 3.12** (the installer targets 3.12).
- Internet access is only needed **once** if the OCR models are not present
  locally (they are cached under `~/.paddlex/official_models`).

| Component | Notes |
|---|---|
| `paddleocr` 3.x | Arabic = `ar` |
| `paddlepaddle` / `paddlepaddle-gpu` | CPU by default; GPU via installer |
| `pymupdf` | PDF → rasters |
| `opencv-python`, `pillow` | image handling & preprocessing |
| `ebooklib` | EPUB 3 writer |
| `numpy` <2.5 | **required below 2.5** |

---

## Installation

On Windows, either double-click `install.cmd` or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer creates a `.venv` (Python 3.12), installs the package editable with
dev dependencies, detects an NVIDIA GPU, and creates a double-clickable launcher
(`تشغيل البرنامج.cmd`) that opens the review UI.

Two commands become available after activation (`.\.venv\Scripts\Activate.ps1`):

```
epubconv          # main converter (CLI)
epubconv-review   # local web UI for human review
```

---

## Command-line usage

```bash
epubconv <source> [-o out.epub] [options]
```

`<source>` is a PDF file or a folder of page images. The output defaults to a
`.epub` alongside the source.

| Option | Default | Description |
|---|---|---|
| `-o, --output` | alongside source | output `.epub` |
| `--title` | source name | EPUB title |
| `--author` | – | EPUB author |
| `--lang` | `ar` | OCR/EPUB language code |
| `--dpi` | `300` | rasterization DPI for PDFs |
| `--workers` | `1` | parallel OCR worker processes |
| `--device` | `auto` | OCR device: `auto`, `cpu`, `gpu` |
| `--max-retries` | `2` | retries per page before marking it failed |
| `--threshold` | `0.70` | below this confidence → flagged for review |
| `--no-resume` | off | ignore the existing cache and reprocess every page |
| `--cache-dir` | `.epubconv_cache` | where per-page results are cached |
| `--max-pages` | all | only convert the first N pages (preview) |
| `--report` | `<out>.report.html` | HTML conversion report path |
| `-v, --verbose` | off | verbose logging |

### Example

```bash
epubconv book.pdf -o out\book.epub --dpi 300 --workers 2 --title "رجال في الشمس" --author "غسان كنفاني"
epubconv pages\ --device gpu --max-pages 5
```

---

## Human review (web UI)

The pipeline never fabricates text: words below the threshold are kept verbatim
and flagged. To correct them, run the review server — it shares the same page
cache, so everything you fix here is picked up by the next `epubconv` run:

```bash
epubconv-review [source] [--port 8765]
```

It opens a browser at `http://127.0.0.1:8765` (Arabic UI). With no `source`, you
can pick one in the browser: upload a PDF, choose from the local `books/`
folder, or type a path.

In the UI you can:

- browse every page and see the source image next to the OCR text,
- **retry OCR** on a single page (CPU or GPU), optionally drawing manual cut
  lines where the text columns should be split,
- **edit text by hand** and save it back to the cache,
- **mark a page as an image** (covers, illustrations) so it is embedded as-is,
- **process all remaining pages** in the background with progress,
- **build the final EPUB** from the cache — pages with no result yet are written
  as failed placeholders (never silently dropped) and counted, so you know to
  process them before saving a complete book.

Options: `--lang`, `--dpi`, `--threshold`, `--device`, `--cache-dir` (must match
the `epubconv` run being reviewed), `--port` (default `8765`), `--no-browser`.

---

## How it works

```
CLI:  epubconv <source> -o <out.epub> [options]
UI:   epubconv-review [source]   ->  http://127.0.0.1:8765

1 ingest        PDF -> rasters (pymupdf @ dpi) | image folder -> pages (Pillow)
2 preprocess    grayscale -> CLAHE -> denoise -> deskew -> size capping
3 ocr           PaddleOCR (lang="ar") with tall-page tiling (overlapping strips)
4 gate          confidence < threshold -> low-conf flag (never invented, never fixed)
5 build         ebooklib EPUB3: xhtml dir=rtl lang=ar + CSS + cover + TOC
6 report        HTML report (pages, word counts, low-confidence stats)
7 resume        per-page results in cache-dir; --no-resume ignores it
```

---

## Limitations

- OCR quality depends on the source scan. Poor scans produce more flagged words —
  the no-hallucination policy bounds the damage, but real recovery needs real
  scans or photos.
- PDFs are always treated as images; there is no text-layer extraction.
- GPU is optional and detected only at install time on Windows; CPU works
  everywhere (slower).
- One page is processed per worker; very large pages are downscaled before OCR.

---

# العربية

---

## نبذة

**epubconv** أداة تحويل الكتب العربية الممسوحة — ملف **PDF** أو **مجلد صور صفحات**
— إلى **EPUB3** قابلة للإعادة التدفق بتخطيط **من اليمين إلى اليسار**، مع قاعدة
صارمة ضد **الهلوسة**:

> إن لم يكن المحرّك واثقاً من كلمة ما، **لا يختلقها أبداً** — بل يُبقيها كما هي،
> ويوسمها، ويتركها للمراجعة البشرية.

---

## الميزات

- **إدخال PDF / مجلد صور** — PDF يُنقَّط بدقة قابلة للضبط (افتراضي 300)؛ ومجلد
  صور (JPG/PNG/BMP/TIFF) يُقرأ مباشرة.
- **OCR عربي** عبر PaddleOCR (`lang="ar"`)، على المعالج افتراضياً مع خيار تسريع
  GPU لبطاقات NVIDIA (يُتحقَّق منه وقت التثبيت).
- **مخرَج حتمي** — الكلمة دون عتبة الثقة تُوسَم (`low-conf`)، لا تُخمَّن ولا
  تُصحَّح بصمت.
- **مخرَج EPUB3 باليمين-إلى-اليسار** — `dir="rtl" lang="ar"` + CSS + غلاف + فهرس
  وميتاداتا `title`/`author`.
- **استئناف** — نتائج كل صفحة تُخزَّن في كاش (افتراضي `.epubconv_cache`)،
  فيستأنف التشغيل من حيث توقف بالضبط؛ و`--no-resume` يعيد المعالجة.
- **توازي** — خيار معالجة متعدد العمليات.
- **تقطيع الصفحات الطويلة** — الصفحات الطويلة جداً تُقسَّم شرائح متداخلة
  تُدمج بذكاء بعد OCR.
- **واجهة مراجعة محلية** — افحص الصفحات المشكوك فيها بجانب الصورة الأصلية،
  وأعد محاولة OCR، وعدّل النص يدوياً، أو علّم صفحة كصورة، ثم ابنِ الـ EPUB
  النهائي من الكاش.
- **تقرير HTML لكل تحويل** — الصفحات، أعداد الكلمات، وإحصاءات منخفضة الثقة.
- **كشف GPU وقت التثبيت** — يفحص المثبّت وجود بطاقة NVIDIA ويُثبّت بناءً
  مطابقاً من `paddlepaddle-gpu` بعد التحقق من الحساب الفعلي (وإلا يعود للـ CPU).

---

## المتطلبات

- **Python 3.11 أو 3.12** (المثبّت يستهدف 3.12).
- الإنترنت مطلوب **مرة واحدة فقط** إن لم تكن نماذج OCR محليّة
  (تُخزَّن في `~/.paddlex/official_models`).

| المكوّن | ملاحظات |
|---|---|
| `paddleocr` 3.x | لغة عربي = `ar` |
| `paddlepaddle` / `paddlepaddle-gpu` | CPU افتراضياً؛ GPU عبر المثبّت |
| `pymupdf` | PDF → صور |
| `opencv-python`, `pillow` | فكّ/معالجة الصور |
| `ebooklib` | كاتب EPUB3 |
| `numpy` <2.5 | **إلزامي <2.5** |

---

## التثبيت

على ويندوز: انقر نقراً مزدوجاً على `install.cmd` أو شغّل:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

ينشئ المثبّت `.venv` (Python 3.12)، ويثبّت الحزمة مع تبعيات التطوير، ويكشف وجود
بطاقة NVIDIA، وينشئ مشغّلاً بالنقر المزدوج (`تشغيل البرنامج.cmd`) يفتح واجهة
المراجعة.

يتوفر أمران بعد التفعيل (`.\.venv\Scripts\Activate.ps1`):

| الأمر | الوظيفة |
|---|---|
| `epubconv` | المحوّل الرئيسي (سطر الأوامر) |
| `epubconv-review` | واجهة المراجعة البشرية عبر المتصفح |

---

## الاستخدام عبر سطر الأوامر

```bash
epubconv <source> [-o out.epub] [خيارات]
```

`<source>` هو ملف PDF أو مجلد صور صفحات. المخرَج الافتراضي `.epub` بجانب المصدر.

| الخيار | الافتراضي | الوصف |
|---|---|---|
| `-o, --output` | بجانب المصدر | ملف `.epub` المخرَج |
| `--title` | اسم المصدر | عنوان الـ EPUB |
| `--author` | – | مؤلف الـ EPUB |
| `--lang` | `ar` | كود اللغة |
| `--dpi` | `300` | DPI نقطية الـ PDF |
| `--workers` | `1` | عمليات OCR متوازية |
| `--device` | `auto` | الجهاز: `auto`, `cpu`, `gpu` |
| `--max-retries` | `2` | محاولات لكل صفحة قبل إعلان فشلها |
| `--threshold` | `0.70` | تحته → يُوسم للمراجعة |
| `--no-resume` | معطل | تجاهل الكاش وإعادة المعالجة الكاملة |
| `--cache-dir` | `.epubconv_cache` | مكان تخزين نتائج الصفحات |
| `--max-pages` | الكل | معالجة أول N صفحة فقط (معاينة) |
| `--report` | `<out>.report.html` | مسار تقرير HTML |
| `-v, --verbose` | معطل | تسجيل مفصّل |

### مثال

```bash
epubconv book.pdf -o out\book.epub --dpi 300 --workers 2 --title "رجال في الشمس" --author "غسان كنفاني"
epubconv pages\ --device gpu --max-pages 5
```

---

## المراجعة البشرية (واجهة الويب)

خط الأنبوب **لا يختلق النص** أبداً: الكلمات دون عتبة الثقة تبقى كما هي وتُوسَم.
لتصحيحها شغّل خادم المراجعة — يتشارك كاش الصفحات نفسه، فتُلتقط أي تعديلاتك في
تشغيل `epubconv` التالي:

```bash
epubconv-review [source] [--port 8765]
```

يفتح متصفحاً على `http://127.0.0.1:8765` (واجهة عربية). دون `source` يمكنك
اختيار الكتاب من المتصفح: رفع PDF، أو اختيار من مجلد `books/` المحلي، أو لصق مسار.

في الواجهة يمكنك:

- تصفّح كل صفحة ومشاهدة الصورة الأصلية بجانب نص OCR،
- **إعادة محاولة OCR** لصفحة واحدة (CPU أو GPU)، مع إمكانية رسم خطوط تقسيم
  يدوية لأعمدة النص،
- **تعديل النص يدوياً** وحفظه في الكاش،
- **تحديد صفحة كصورة** (أغلفة، رسوم) لتُضمَّن كما هي،
- **معالجة كل الصفحات المتبقية** في الخلفية مع مؤشر تقدم،
- **بناء الـ EPUB النهائي** من الكاش — الصفحات بلا نتائج تُكتب كصفحات فاشلة
  (لا تُحذف صامتاً) وتُحصى لتعرف بمعالجتها قبل حفظ كتاب كامل.

الخيارات: `--lang`, `--dpi`, `--threshold`, `--device`, `--cache-dir` (يجب أن
يطابق تشغيل `epubconv` قيد المراجعة), `--port` (افتراضي `8765`), `--no-browser`.

---

## كيف يعمل

```
CLI:  epubconv <source> -o <out.epub> [خيارات]
UI:   epubconv-review [source]   ->  http://127.0.0.1:8765

1 الإدخال     PDF -> نقطي (pymupdf @ dpi) | مجلد صور -> صفحات (Pillow)
2 المعالجة    رمادي -> CLAHE -> تقليل ضجيج -> تصحيح ميل -> سقف مقاسات
3 OCR         PaddleOCR (ar) مع تقطيع الصفحات الطويلة (شرائح متداخلة)
4 الحصص      conf < threshold -> وسم low-conf (لا اختراع ولا تصحيح صامت)
5 البناء      ebooklib EPUB3: xhtml dir=rtl lang=ar + CSS + غلاف + فهرس
6 التقرير     تقرير HTML (صفحات، أعداد كلمات، إحصاءات الثقة)
7 الاستئناف   نتائج الصفحات في cache-dir؛ --no-resume يتجاهلها
```

---

## قيود معروفة

- جودة OCR تعتمد على جودة المسح. المسوح الرديئة تُنتج كلمات موسومة أكثر —
  سياسة منع الهلوسة تحدد الضرر، لكن الاسترداد الحقيقي يحتاج مسوحاً/صوراً سليمة.
- يُعامَل الـ PDF دائماً كصور؛ لا استخراج طبقة نصية.
- GPU اختيارية تُكتشف عند التثبيت على ويندوز فقط؛ CPU يعمل في كل مكان (أبطأ).
- تُعالَج صفحة واحدة لكل عامل؛ الصفحات الكبيرة جداً تُقلَّص قبل OCR.
