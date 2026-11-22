"""Reading the text off the scanned pages, for books that have no digitised text.

Uses the reader built into Windows (no download, no account). The result is
rougher than a proofread text, which is fine here: it only has to agree with the
lesson's transcript often enough to tell which line is being read.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import fitz

OCR_DPI = 200


@dataclass
class OcrWord:
    text: str
    y: float      # middle of the word, fraction of the page height
    x: float      # middle of the word, fraction of the page width
    line: int = 0  # which line of the page it belongs to
    # The word's box, fractions of the page size (None in caches written before 0.5).
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None


def available() -> str | None:
    """None when Arabic can be read on this machine, else the reason it cannot."""
    try:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine
    except ImportError:
        return "مكتبة قراءة الصفحات غير مثبّتة (winrt)"
    if not OcrEngine.is_language_supported(Language("ar")):
        return "قراءة العربية غير مفعّلة في ويندوز: أضف اللغة العربية من إعدادات اللغة"
    return None


async def _read(engine, pix: fitz.Pixmap) -> list[OcrWord]:
    from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.storage.streams import DataWriter

    writer = DataWriter()
    writer.write_bytes(pix.samples)
    bitmap = SoftwareBitmap.create_copy_from_buffer(
        writer.detach_buffer(), BitmapPixelFormat.GRAY8, pix.width, pix.height)
    result = await engine.recognize_async(bitmap)
    words = []
    for n, line in enumerate(result.lines):
        row = []
        for w in line.words:
            r = w.bounding_rect
            row.append(OcrWord(w.text, (r.y + r.height / 2) / pix.height,
                               (r.x + r.width / 2) / pix.width, n,
                               r.x / pix.width, r.y / pix.height,
                               (r.x + r.width) / pix.width, (r.y + r.height) / pix.height))
        # The reader hands the words of a line left to right; Arabic runs the other way.
        words.extend(sorted(row, key=lambda w: -w.x))
    return words


def ocr_pages(pdf_path: Path, indexes: list[int], cache: Path | None = None,
              progress=None, boxes: bool = False) -> dict[int, list[OcrWord]]:
    """Words of the given PDF pages (0-based), in reading order.

    cache: a json file that keeps what was read, so a page is read only once.
    boxes: the words' boxes are needed; pages cached without them are read again."""
    from winrt.windows.globalization import Language
    from winrt.windows.media.ocr import OcrEngine

    done: dict[str, list] = {}
    if cache and cache.exists():
        done = json.loads(cache.read_text(encoding="utf-8"))
    def missing(i: int) -> bool:
        got = done.get(str(i))
        return got is None or (boxes and any(len(w) < 8 for w in got))

    todo = [i for i in indexes if missing(i)]
    if todo:
        engine = OcrEngine.try_create_from_language(Language("ar"))
        if engine is None:
            raise RuntimeError(available() or "تعذّر تشغيل قارئ الصفحات")
        doc = fitz.open(pdf_path)
        limit = OcrEngine.max_image_dimension

        async def run():
            for n, i in enumerate(todo):
                page = doc[i]
                dpi = min(OCR_DPI, int(72 * limit / max(page.rect.width, page.rect.height)))
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                done[str(i)] = [[w.text, round(w.y, 4), round(w.x, 4), w.line,
                                 round(w.x0, 4), round(w.y0, 4), round(w.x1, 4), round(w.y1, 4)]
                                for w in await _read(engine, pix)]
                if progress:
                    progress((n + 1) / len(todo))
                if cache and (n % 20 == 19 or n == len(todo) - 1):
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")

        asyncio.run(run())
    return {i: [OcrWord(*w) for w in done[str(i)]] for i in indexes}


def drop_running_heads(pages: dict[int, list[OcrWord]]) -> None:
    """Remove the line printed at the top of every page (book or chapter title).

    It is not part of the text being read, and it would sit between the last word
    of one page and the first word of the next."""
    from collections import Counter

    from .arabic import tokens

    def head(words: list[OcrWord]) -> tuple[str, ...]:
        if not words:
            return ()
        top = min(words, key=lambda w: w.y).line
        return tuple(tokens(" ".join(w.text for w in words if w.line == top)))

    heads = {i: head(ws) for i, ws in pages.items()}
    seen = Counter(h for h in heads.values() if h)
    for i, h in heads.items():
        if h and len(h) <= 8 and seen[h] >= max(3, 0.1 * len(pages)):
            top = min(pages[i], key=lambda w: w.y).line
            pages[i] = [w for w in pages[i] if w.line != top]
