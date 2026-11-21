"""The book: its text (word by word, with the page and height each word sits at)
and its pictures (cropped page views rendered from the PDF).

Text sources:
  * turath.io digitised text whose pagination matches the printed edition.
  * the PDF itself, read by OCR, for books that exist only as scans.
The PDF is always the source of the pictures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from .arabic import tokens

_FOOTNOTE_SPLIT = re.compile(r"\n_{5,}\s*\n?")
_FOOTNOTE_MARK = re.compile(r"\(\^[^)]*\)")
ANALYSIS_DPI = 100


# ----------------------------------------------------------------- text ----

def _cached_json(url: str, cache: Path) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        import requests
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return json.loads(cache.read_text(encoding="utf-8"))


def turath_book(book_id: int, cache_dir: Path) -> dict:
    return _cached_json(f"https://files.turath.io/books-v3/{book_id}.json",
                        cache_dir / f"turath_{book_id}.json")


def turath_volumes(book_id: int, cache_dir: Path) -> list[dict]:
    """Volumes of a book with their printed page range, in book order."""
    vols: dict[str, list[int]] = {}
    for rec in turath_book(book_id, cache_dir)["pages"]:
        if isinstance(rec.get("page"), int):
            vols.setdefault(str(rec.get("vol")), []).append(rec["page"])
    return [{"vol": v, "first": min(p), "last": max(p)} for v, p in vols.items()]


def turath_catalog(cache_dir: Path) -> list[dict]:
    """Every book of the library as {id, name, author}, for searching by name."""
    data = _cached_json("https://files.turath.io/data-v3.json", cache_dir / "turath_catalog.json")
    authors = data.get("authors", {})
    out = []
    for book in data.get("books", {}).values():
        author = authors.get(str(book.get("author_id")), {})
        out.append({"id": book["id"], "name": book.get("name", ""), "author": author.get("name", "")})
    return out


def load_turath_pages(book_id: int, vol: str, cache_dir: Path) -> dict[int, str]:
    """Return {printed page number: body text without footnotes} for one volume."""
    data = turath_book(book_id, cache_dir)
    pages: dict[int, str] = {}
    for rec in data["pages"]:
        if str(rec.get("vol")) != str(vol) or not isinstance(rec.get("page"), int):
            continue
        body = _FOOTNOTE_SPLIT.split(rec["text"], maxsplit=1)[0]
        body = _FOOTNOTE_MARK.sub(" ", body)
        # A printed page can be split over several records: append in order.
        pages[rec["page"]] = (pages.get(rec["page"], "") + "\n" + body).strip()
    return pages


# --------------------------------------------------------------- layout ----

@dataclass
class PageLayout:
    """Geometry of one PDF page. Heights and widths are fractions of the page size."""
    x0: float
    x1: float
    y0: float                 # top of the first printed line
    y1: float                 # bottom of the last printed line (footnotes included)
    body_top: float
    body_bottom: float        # where the body text ends (the footnote rule, if any)
    body_cum: np.ndarray      # cumulative share of body ink, one value per pixel row
    bands: list[tuple[float, float]]   # (top, bottom) of every printed line
    line_h: float
    body_ink: float = 0.0     # amount of ink in the body text

    def y_at(self, share: np.ndarray) -> np.ndarray:
        """Height at which the given share of the body text has been printed."""
        rows = np.searchsorted(self.body_cum, share)
        return self.body_top + rows / len(self.body_cum) * (self.body_bottom - self.body_top)


def _bands(row_ink: np.ndarray) -> list[tuple[int, int]]:
    on = row_ink > 0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.view(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _longest_run(row: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def analyze_page(page: fitz.Page) -> PageLayout | None:
    pix = page.get_pixmap(dpi=ANALYSIS_DPI, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = img < 128
    h, w = ink.shape
    # Ignore scanner specks at the very edge of the sheet.
    m = int(0.03 * w)
    ink[:, :m] = False
    ink[:, w - m:] = False
    ink[: int(0.02 * h)] = False
    ink[h - int(0.02 * h):] = False
    row_ink = ink.sum(axis=1)
    row_ink[row_ink < 2] = 0
    bands = _bands(row_ink)
    if not bands:
        return None
    heights = [b - a for a, b in bands]
    line_h = float(np.median([x for x in heights if x >= 6] or heights))

    # The footnote rule: a very thin band made of one long horizontal stroke.
    rule = None
    for n, (a, b) in enumerate(bands):
        if b - a <= max(3, 0.25 * line_h) and a > 0.15 * h:
            if max(_longest_run(ink[r]) for r in range(a, b)) > 0.12 * w:
                rule = n
                break
    if rule is not None:
        body = bands[:rule]
    else:
        body = list(bands)
        # Without footnotes the last band is the page number: narrow and alone.
        if len(body) > 1:
            a, b = body[-1]
            xs = np.flatnonzero(ink[a:b].any(axis=0))
            if len(xs) and (xs[-1] - xs[0]) < 0.12 * w:
                body.pop()
    if not body:
        return None
    top, bottom = body[0][0], body[-1][1]
    cum = np.cumsum(row_ink[top:bottom].astype(float))
    cols = np.flatnonzero(ink.any(axis=0))
    return PageLayout(
        x0=cols[0] / w, x1=(cols[-1] + 1) / w, y0=bands[0][0] / h, y1=bands[-1][1] / h,
        body_top=top / h, body_bottom=bottom / h, body_cum=cum / cum[-1],
        bands=[(a / h, b / h) for a, b in bands], line_h=line_h / h, body_ink=float(cum[-1]))


def guess_page_offset(pdf_path: Path, page_text: dict[int, str],
                      sample: int = 50) -> tuple[int, float]:
    """Find (PDF page number - printed page number) without reading the scan.

    Pages differ a lot in how much body text they carry (chapter ends, pages full of
    footnotes). The amount of text per printed page, from the digitised text, and the
    amount of ink per PDF page follow the same ups and downs when lined up correctly.
    Returns the best offset and how much better it correlates than the runner-up.
    """
    # The PDF may hold the whole volume or only the part being read, so every
    # possible position of a stretch of the PDF inside the volume is tried.
    doc = fitz.open(pdf_path)
    mid = doc.page_count // 2
    chosen = list(range(max(0, mid - sample // 2), min(doc.page_count, mid + sample // 2)))
    inks = []
    for idx in chosen:
        lay = analyze_page(doc[idx])
        inks.append(lay.body_ink if lay else 0.0)
    inks = np.array(inks)
    chars = {p: float(len(re.sub(r"\s", "", t))) for p, t in page_text.items()}
    scores = {}
    for k in range(1 - max(chars), doc.page_count):
        got = np.array([chars.get(idx + 1 - k, np.nan) for idx in chosen])
        ok = ~np.isnan(got)
        if ok.sum() >= max(8, 0.7 * len(chosen)) and got[ok].std() > 0 and inks[ok].std() > 0:
            scores[k] = float(np.corrcoef(got[ok], inks[ok])[0, 1])
    ranked = sorted(scores, key=scores.get, reverse=True)
    if not ranked:
        return 0, 0.0
    margin = scores[ranked[0]] - (scores[ranked[1]] if len(ranked) > 1 else 0.0)
    return ranked[0], margin


# ------------------------------------------------------------ the book ----

@dataclass
class Book:
    pdf_path: Path
    words: list[str]              # normalised words of the whole range, reading order
    word_page: np.ndarray         # PDF page index (0-based) of each word
    word_y: np.ndarray            # height on that page (fraction) of each word
    layouts: dict[int, PageLayout]
    printed: dict[int, int]       # PDF page index -> printed page number
    window_h: float               # height of a view, fraction of the page height
    x0: float                     # horizontal extent shared by all views
    x1: float

    def stops(self, page_idx: int) -> list[float]:
        """Tops of the windows this page is shown through; they travel over the body text."""
        lay = self.layouts[page_idx]
        top = max(0.0, lay.body_top - 0.6 * lay.line_h)
        travel = max(0.0, (lay.body_bottom + 0.6 * lay.line_h) - top - self.window_h)
        n = int(np.ceil(travel / (0.45 * self.window_h))) + 1
        return [top + f * travel for f in np.linspace(0, 1, n)]

    def pick_stop(self, page_idx: int, y: float, current: int | None) -> int:
        """Which window should be on screen while the line at height y is read."""
        stops, h = self.stops(page_idx), self.window_h
        if current is not None and current < len(stops) and \
                stops[current] + 0.04 * h <= y <= stops[current] + 0.85 * h:
            return current
        best = 0
        for i, top in enumerate(stops):
            if top <= y - 0.3 * h:
                best = i
        return best


def read_pdf_text(pdf_path: Path, cache_dir: Path, first: int | None = None,
                  last: int | None = None, progress=None) -> dict[int, list]:
    """Return {PDF page number: words read off the scan, each with its place on the page}.

    What was read is kept in cache_dir, so each page of a book is read once."""
    import hashlib

    from .ocr import drop_running_heads, ocr_pages

    info = pdf_path.stat()
    key = hashlib.sha1(f"{pdf_path.resolve()}|{info.st_size}|{int(info.st_mtime)}".encode()).hexdigest()[:10]
    count = fitz.open(pdf_path).page_count
    wanted = [i for i in range(count) if (not first or i + 1 >= first) and (not last or i + 1 <= last)]
    pages = ocr_pages(pdf_path, wanted, cache_dir / f"ocr_{key}.json", progress)
    drop_running_heads(pages)
    return {i + 1: ws for i, ws in pages.items()}


def build_book(pdf_path: Path, page_text: dict[int, str] | dict[int, list], page_offset: int = 0,
               first: int | None = None, last: int | None = None,
               window: float = 0.55) -> Book:
    """page_text: the text of each printed page, or the words read off each PDF page.
    page_offset: PDF page number (1-based) minus printed page number.
    window: share of the printed page height that is visible at once."""
    doc = fitz.open(pdf_path)
    words: list[str] = []
    wp: list[int] = []
    wy: list[float] = []
    layouts: dict[int, PageLayout] = {}
    printed: dict[int, int] = {}
    for pno in sorted(page_text):
        if (first and pno < first) or (last and pno > last):
            continue
        idx = pno + page_offset - 1
        if not 0 <= idx < doc.page_count:
            continue
        layout = analyze_page(doc[idx])
        if layout is None:
            continue
        if not isinstance(page_text[pno], str):
            # Words read off the scan know their own height; footnotes are left out.
            slack = 0.5 * layout.line_h
            got = [(t, w.y) for w in page_text[pno]
                   if layout.body_top - slack <= w.y <= layout.body_bottom + slack
                   for t in tokens(w.text)]
            if not got:
                continue
            layouts[idx] = layout
            printed[idx] = pno
            words.extend(t for t, _ in got)
            wp.extend([idx] * len(got))
            wy.extend(y for _, y in got)
            continue
        toks = tokens(page_text[pno])
        if not toks:
            continue
        layouts[idx] = layout
        printed[idx] = pno
        # A word sits where the same share of the page's ink has been printed.
        chars = np.array([len(t) for t in toks], dtype=float)
        share = (np.cumsum(chars) - chars / 2) / chars.sum()
        words.extend(toks)
        wp.extend([idx] * len(toks))
        wy.extend(layout.y_at(share).tolist())
    lays = list(layouts.values())
    span = float(np.median([l.y1 - l.y0 for l in lays]))
    return Book(pdf_path, words, np.array(wp), np.array(wy), layouts, printed,
                window_h=window * span,
                x0=float(np.percentile([l.x0 for l in lays], 5)),
                x1=float(np.percentile([l.x1 for l in lays], 95)))


# ---------------------------------------------------------------- views ----

def render_view(book: Book, page_idx: int, stop: int, out_dir: Path,
                size: tuple[int, int] = (1920, 1080), scale: float = 1.0,
                pos: str = "center", background: Path | None = None) -> Path:
    """Render one window of one page as a full frame and return its path.

    The page is a card laid on the background picture (the team's own design with
    the book title and logo), or on a plain colour when none is given.
    scale: card height as a share of the frame height. pos: left, center or right.
    """
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    look = f"{pos[0]}{round(scale * 100)}"
    if background:
        look += f"b{int(background.stat().st_mtime) % 100000}"
    # Ends with a letter, not a digit: Resolve must not read the stills as an image sequence.
    out = out_dir / f"{look}_page{book.printed[page_idx]:04d}_{'abcdefghij'[stop]}.png"
    if out.exists():
        return out
    lay = book.layouts[page_idx]
    page = fitz.open(book.pdf_path)[page_idx]
    pw, ph = page.rect.width, page.rect.height
    top = book.stops(page_idx)[stop]
    bottom = min(1.0, top + book.window_h)
    pad_x = 0.035
    clip = fitz.Rect(max(0, book.x0 - pad_x) * pw, top * ph,
                     min(1, book.x1 + pad_x) * pw, bottom * ph)
    fw, fh = size
    zoom = min(scale * fh / (book.window_h * ph), 0.96 * fw / clip.width)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csRGB)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # Blank out lines that the window edge would cut in half.
    draw = ImageDraw.Draw(img)
    for a, b in lay.bands:
        if b - a > 3 * lay.line_h:
            continue
        if a < top < b:
            draw.rectangle([0, 0, img.width, int((b - top) * ph * zoom) + 1], fill="white")
        if a < bottom < b:
            draw.rectangle([0, int((a - top) * ph * zoom) - 1, img.width, img.height], fill="white")
    if background:
        canvas = Image.open(background).convert("RGB").resize(size, Image.LANCZOS)
    else:
        canvas = Image.new("RGB", size, (96, 180, 164))
    margin = round(0.03 * fw)
    x = {"left": margin, "right": fw - img.width - margin}.get(pos, (fw - img.width) // 2)
    canvas.paste(img, (max(0, x), (fh - img.height) // 2))
    canvas.save(out, optimize=True)
    return out
