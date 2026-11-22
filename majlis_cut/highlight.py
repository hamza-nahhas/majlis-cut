"""Highlighting the word being read, and its line, on the page pictures.

Three steps:
  1. Where each book word sits on the scan (`page_boxes`): the words read off the
     page by OCR carry their boxes; the book's own words are lined up with them.
  2. When each word is read (`view_events`): the transcript's word times, through
     the alignment, give the book word being read at each moment.
  3. Drawing (`Highlight.frame`): a soft ribbon under the words being read and a
     faint band under their line. The ribbon flows from word to word without
     stopping: its front edge reaches the next word as it starts and its back edge
     follows a moment later. It rests, dimmed a little, only when the reader pauses.

The drawing is a pure function of time, so the Resolve clip, the review preview and
the interface's sample all show the same thing. The highlight sits under the ink:
the page track, then the highlight, then the ink alone on top, so letters stay sharp.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .arabic import tokens

HIGHLIGHT_VERSION = 2      # raise when the drawing or the clip format changes (2: the flowing ribbon)
DEFAULT_COLOR = "#4096ff"
CLIP_SCALE = 0.5           # the highlight is soft: half the frame size is enough, Resolve scales it up

# Look of the highlight (sizes in line heights, times in seconds).
BOX_ALPHA = 0.42
BAND_ALPHA = 0.10
BOX_PAD_X, BOX_PAD_Y = 0.16, 0.06
BAND_PAD_X = 0.2
BAND_RADIUS = 0.3
LEAD = 0.08                # show a word slightly before it is heard: a lag feels sluggish
HOLD = 0.7                 # after the last word, the ribbon stays this long before fading
REST_FADE = 0.3

MIN_RATIO = 0.35           # a page whose words match the scan less than this gets the line only


def parse_color(text: str | None) -> tuple[int, int, int]:
    s = (text or DEFAULT_COLOR).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        raise ValueError(f"لون غير صالح: {text}") from None


# ------------------------------------------------------------- where ----

@dataclass
class PageBoxes:
    """Where the book words of one page sit on the scan (fractions of the page)."""
    bands: list[tuple[float, float, float, float]]   # (top, bottom, left, right) of each printed line
    box: dict[int, tuple[float, float]] = field(default_factory=dict)   # book word -> (x0, x1)
    band: dict[int, int] = field(default_factory=dict)                  # book word -> line
    ratio: float = 0.0                                                  # share of words found on the scan


def _units(ocr_words, lay) -> list[tuple[str, float, float, int]]:
    """OCR words of the body text as (token, x0, x1, line), in reading order.

    A word the normaliser splits in two (a joined ﷺ, say) shares its box by length."""
    slack = 0.5 * lay.line_h
    out = []
    for w in ocr_words:
        if w.x0 is None or not (lay.body_top - slack <= w.y <= lay.body_bottom + slack):
            continue
        line = min(range(len(lay.bands)),
                   key=lambda n: abs((lay.bands[n][0] + lay.bands[n][1]) / 2 - w.y))
        toks = tokens(w.text)
        total = sum(len(t) for t in toks) or 1
        right = w.x1
        for t in toks:                      # right to left
            width = (w.x1 - w.x0) * len(t) / total
            out.append((t, right - width, right, line))
            right -= width
    # The reader hands lines in its own order: sort by line, then right to left.
    out.sort(key=lambda u: (lay.bands[u[3]][0], -u[2]))
    return out


def _match(book: list[str], scan: list[str]) -> list[tuple[list[int], list[int]]]:
    """Line up the book's words with the words read off the scan.

    Returns groups (book word indexes, scan word indexes) that belong together:
    one to one, several book words in one scan word (the reader runs Quranic words
    together), or one book word in two scan words. Unmatched words are left out."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    m, n = len(book), len(scan)
    if not m or not n:
        return []

    def sim(a: list[str], b: list[str]) -> np.ndarray:
        return cdist(a, b, scorer=Levenshtein.normalized_similarity, workers=1)

    s11 = sim(book, scan)
    joined2 = [book[i] + book[i + 1] for i in range(m - 1)]
    joined3 = [book[i] + book[i + 1] + book[i + 2] for i in range(m - 2)]
    s21 = sim(joined2, scan) if joined2 else np.zeros((0, n))
    s31 = sim(joined3, scan) if joined3 else np.zeros((0, n))
    scan2 = [scan[j] + scan[j + 1] for j in range(n - 1)]
    s12 = sim(book, scan2) if scan2 else np.zeros((m, 0))

    keep = 0.55               # below this a pairing costs more than leaving both out
    skip_book, skip_scan = -0.25, -0.15
    neg = -1e9
    score = np.full((m + 1, n + 1), neg)
    move = np.zeros((m + 1, n + 1), dtype=np.int8)
    score[0, :] = np.arange(n + 1) * skip_scan
    score[:, 0] = np.arange(m + 1) * skip_book
    move[0, 1:], move[1:, 0] = 2, 1
    width = max(25, int(0.2 * max(m, n)))
    for i in range(1, m + 1):
        mid = i * n / m
        for j in range(max(1, int(mid - width)), min(n, int(mid + width)) + 1):
            best, how = score[i - 1, j] + skip_book, 1
            v = score[i, j - 1] + skip_scan
            if v > best:
                best, how = v, 2
            v = score[i - 1, j - 1] + s11[i - 1, j - 1] - keep
            if v > best:
                best, how = v, 3
            if i >= 2:
                v = score[i - 2, j - 1] + s21[i - 2, j - 1] - keep - 0.1
                if v > best:
                    best, how = v, 4
            if i >= 3:
                v = score[i - 3, j - 1] + s31[i - 3, j - 1] - keep - 0.2
                if v > best:
                    best, how = v, 5
            if j >= 2:
                v = score[i - 1, j - 2] + s12[i - 1, j - 2] - keep - 0.1
                if v > best:
                    best, how = v, 6
            score[i, j], move[i, j] = best, how
    groups = []
    i, j = m, n
    while i > 0 and j > 0:
        how = move[i, j]
        if how == 0:                       # outside the band
            break
        if how == 1:
            i -= 1
        elif how == 2:
            j -= 1
        elif how == 3:
            groups.append(([i - 1], [j - 1])); i, j = i - 1, j - 1
        elif how == 4:
            groups.append(([i - 2, i - 1], [j - 1])); i, j = i - 2, j - 1
        elif how == 5:
            groups.append(([i - 3, i - 2, i - 1], [j - 1])); i, j = i - 3, j - 1
        else:
            groups.append(([i - 1], [j - 2, j - 1])); i, j = i - 1, j - 2
    return groups[::-1]


def page_boxes(book, page_idx: int, ocr_words) -> PageBoxes:
    """Boxes of the book words of one page, from the words read off its scan."""
    lay = book.layouts[page_idx]
    idx = np.flatnonzero(book.word_page == page_idx)
    units = _units(ocr_words, lay)
    bands = []
    for n, (a, b) in enumerate(lay.bands):
        on = [u for u in units if u[3] == n]
        left = min((u[1] for u in on), default=lay.x0)
        right = max((u[2] for u in on), default=lay.x1)
        bands.append((a, b, left, right))
    out = PageBoxes(bands)
    # Every word gets a line: from the scan when matched, else from its estimated height.
    centres = np.array([(a + b) / 2 for a, b in lay.bands])
    for j in idx:
        out.band[int(j)] = int(np.argmin(np.abs(centres - book.word_y[j])))
    if not len(idx) or not units:
        return out
    book_words = [book.words[j] for j in idx]
    groups = _match(book_words, [u[0] for u in units])
    found = 0
    for bs, ss in groups:
        x0 = min(units[s][1] for s in ss)
        x1 = max(units[s][2] for s in ss)
        line = units[ss[0]][3]
        total = sum(len(book_words[b]) for b in bs) or 1
        right = x1
        for b in bs:                        # several book words in one box: share it by length
            w = (x1 - x0) * len(book_words[b]) / total
            out.box[int(idx[b])] = (right - w, right)
            out.band[int(idx[b])] = line
            right -= w
        found += len(bs)
    out.ratio = found / len(idx)
    if out.ratio < MIN_RATIO:
        out.box.clear()
        return out
    # Short runs of unmatched words between two matched ones on the same line share the gap.
    have = sorted(out.box)
    for a, b in zip(have, have[1:]):
        gap = list(range(a + 1, b))
        if not gap or len(gap) > 4 or out.band[a] != out.band[b] or book.word_page[b] != page_idx:
            continue
        right, left = out.box[a][0], out.box[b][1]          # reading runs right to left
        if right <= left:
            continue
        chars = [len(book.words[g]) + 1.2 for g in gap]
        x = right
        for g, c in zip(gap, chars):
            w = (right - left) * c / sum(chars)
            out.box[g] = (x - w, x)
            out.band[g] = out.band[a]
            x -= w
    return out


# -------------------------------------------------------------- when ----

@dataclass
class Event:
    t: float                    # when the word starts being shown (seconds into the view)
    rest: float                 # when, without a next word, the box starts to fade (a pause)
    box: tuple[float, float, float, float] | None   # frame pixels at full size, None = line only
    band: tuple[float, float, float, float]          # the line's band, frame pixels at full size


def view_events(view: dict, words: list[dict], pos: np.ndarray, book, boxes: PageBoxes,
                frame) -> list[Event]:
    """The words read during one page view, in order, with where to draw them.

    The reader repeating a word or two does not send the box back; going back
    several words (re-reading a sentence) does. A few words the transcript skipped
    between two heard ones get a share of the time between them."""
    page = view["page"]
    heard = [(w["t0"], w["t1"], int(pos[n])) for n, w in enumerate(words)
             if view["t0"] - 0.5 <= w["t0"] < view["t1"] and pos[n] >= 0
             and book.word_page[pos[n]] == page]
    seq: list[tuple[float, float, int]] = []
    for t0, t1, j in heard:
        if seq:
            last = seq[-1][2]
            if last - 3 <= j <= last:      # a repeat: hold on
                seq[-1] = (seq[-1][0], max(seq[-1][1], t1), last)
                continue
            if last < j <= last + 5:       # fill in words the transcript ran over
                steps = j - last
                t_prev = seq[-1][0]
                for k in range(1, steps):
                    tk = t_prev + (t0 - t_prev) * k / steps
                    if tk - seq[-1][0] >= 0.12:
                        seq.append((tk, tk + 0.2, last + k))
        seq.append((t0, t1, j))
    events = []
    for t0, t1, j in seq:
        line = boxes.band.get(j)
        if line is None:
            continue
        top, bottom, left, right = boxes.bands[line]
        bx0, by0 = frame.to_frame(left, top)
        bx1, by1 = frame.to_frame(right, bottom)
        # Lines cut by the window edge are blanked on the picture: nothing to point at.
        if by0 < frame.y - 1 or by1 > frame.y + frame.h + 1:
            continue
        box = None
        if j in boxes.box:
            x0, x1 = boxes.box[j]
            wx0, _ = frame.to_frame(x0, top)
            wx1, _ = frame.to_frame(x1, top)
            box = (wx0, by0, wx1, by1)
        events.append(Event(t0 - view["t0"] - LEAD, t1 - view["t0"] + HOLD, box, (bx0, by0, bx1, by1)))
    return events


# ----------------------------------------------------------- drawing ----

def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _runs(events: list[Event]) -> list[list[int]]:
    """Stretches of words read along one line, right to left. A new stretch starts on
    a new line, after a word without a box, or when the reader goes back."""
    runs: list[list[int]] = []
    run: list[int] = []
    for n, e in enumerate(events):
        if e.box is None:
            if run:
                runs.append(run)
            run = []
            continue
        if run:
            prev = events[run[-1]]
            if prev.band != e.band or e.box[2] > prev.box[2] + 15.0:
                runs.append(run)
                run = []
        run.append(n)
    if run:
        runs.append(run)
    return runs


def _said_in(width: float) -> float:
    """About how long a word this wide (full-size pixels) takes to say."""
    return float(np.clip(0.22 + 0.0030 * width, 0.28, 0.68))


def _monotone(ts: list[float], xs: list[float], grid: np.ndarray) -> np.ndarray:
    """A smooth curve through the points that never overshoots them (PCHIP)."""
    from scipy.interpolate import PchipInterpolator

    if len(ts) < 2:
        return np.full_like(grid, xs[0])
    return PchipInterpolator(ts, xs, extrapolate=False)(np.clip(grid, ts[0], ts[-1]))


class Highlight:
    """The highlight of one page view, drawn at any moment.

    A soft ribbon flows along the line under the words being read. Its front edge
    (the left one: Arabic runs right to left) reaches the next word as it starts,
    and its back edge lets go of the word before a moment later, so it glides on
    without stopping at each word. In a real pause it comes to rest on the word and
    dims a little. The next line's band lights up just before the reading moves to it."""

    RIBBON_ALPHA = BOX_ALPHA
    FLOW = 0.115               # seconds over which the edges' motion is smoothed
    TRAIL = 0.12               # the back edge follows the front edge this late
    MIN_WIDTH = 1.05           # in line heights: short words still get a comfortable ribbon
    PAUSE = 0.50               # more time than a word needs, plus this, is a pause
    ENTER, EXIT = 0.18, 0.25
    LINE_AHEAD, LINE_CROSS = 0.24, 0.40
    DIM, DIM_AFTER, DIM_OVER = 0.68, 0.45, 0.60     # resting in a pause

    def __init__(self, events: list[Event], duration: float, color: tuple[int, int, int],
                 size: tuple[int, int] = (1920, 1080), scale: float = 1.0):
        from scipy.ndimage import gaussian_filter1d

        self.events = events
        self.duration = duration
        self.color = np.array(color, np.float32)
        self.scale = scale
        self.w, self.h = round(size[0] * scale), round(size[1] * scale)
        # each line's band: from its first word to the last word's rest
        self.lines: list[list] = []
        for e in events:
            if self.lines and self.lines[-1][0] == e.band:
                self.lines[-1][2] = max(self.lines[-1][2], e.rest)
            else:
                self.lines.append([e.band, e.t, e.rest])
        self.ribbons = []
        runs = _runs(events)
        step = 0.01
        for r, run in enumerate(runs):
            first, last = events[run[0]], events[run[-1]]
            lh = first.box[3] - first.box[1]
            pad = BOX_PAD_X * lh
            left_end = min(events[n].box[0] for n in run) - pad
            right_end = max(events[n].box[2] for n in run) + pad
            front: tuple[list, list] = ([], [])
            back: tuple[list, list] = ([], [])
            rests = []

            def add(edge, t, x):
                ts, xs = edge
                if ts and t <= ts[-1] + 0.015:
                    t = ts[-1] + 0.015
                if xs:
                    x = min(x, xs[-1])          # never back to the right
                ts.append(t)
                xs.append(x)

            for i, n in enumerate(run):
                e = events[n]
                t0 = e.t
                said = max(e.t + 0.08, e.rest - HOLD - LEAD)
                nxt = events[run[i + 1]].t if i + 1 < len(run) else said
                room = max(0.08, nxt - t0)
                need = _said_in(e.box[2] - e.box[0])
                wl, wr = e.box[0] - pad, e.box[2] + pad
                if wr - wl < self.MIN_WIDTH * lh:
                    mid = (wl + wr) / 2
                    wl = max(left_end, mid - self.MIN_WIDTH * lh / 2)
                    wr = min(right_end, mid + self.MIN_WIDTH * lh / 2)
                if i == 0:                       # comes in already moving
                    add(front, t0 - 0.06, min(right_end, wr - 0.45 * (wr - wl)))
                    add(back, t0 - 0.06, wr)
                if room > need + self.PAUSE:     # a pause: flow onto the word and rest there
                    hit = t0 + 0.62 * need
                    add(front, hit, wl)
                    add(back, hit + self.TRAIL, wr)
                    until = max(hit + self.TRAIL + 0.08, nxt - 0.14)
                    add(front, until, wl)
                    add(back, until, wr)
                    rests.append((hit + self.TRAIL, until))
                else:                            # speech goes on: one point per edge, no stop
                    hit = t0 + 0.30 * room
                    add(front, hit, wl)
                    add(back, hit + min(self.TRAIL, 0.45 * room), wr)
            grid = np.arange(min(front[0][0], back[0][0]) - 0.5, max(front[0][-1], back[0][-1]) + 0.5, step)
            edges = []
            for ts, xs in (front, back):
                x = np.clip(_monotone(ts, xs, grid), left_end, right_end)
                edges.append(np.minimum.accumulate(gaussian_filter1d(x, self.FLOW / step, mode="nearest")))
            gl, gr = edges
            thin = (gr - gl) < 0.85 * self.MIN_WIDTH * lh
            if thin.any():
                mid, half = (gl + gr) / 2, 0.85 * self.MIN_WIDTH * lh / 2
                gl = np.where(thin, np.maximum(left_end, mid - half), gl)
                gr = np.where(thin, np.minimum(right_end, mid + half), gr)
            self.ribbons.append({
                "grid": grid, "left": gl, "right": gr, "y0": first.box[1], "y1": first.box[3], "lh": lh,
                "on": first.t - 0.06, "rest": last.rest, "rests": rests,
                "next": events[runs[r + 1][0]].t if r + 1 < len(runs) else None})

    def _dim(self, t: float, rests: list[tuple[float, float]]) -> float:
        for a, b in rests:
            if a <= t <= b:
                return 1.0 - (1.0 - self.DIM) * float(_smooth((t - a - self.DIM_AFTER) / self.DIM_OVER)) \
                    * float(_smooth((b - t) / 0.22))
        return 1.0

    def prims(self, t: float) -> list[tuple]:
        """What is on screen at time t: (rect, radius, soft_y, soft_left, soft_right, opacity),
        rectangles in full-size frame pixels."""
        out = []
        for n, (band, t0, t1) in enumerate(self.lines):
            start = t0 - self.LINE_AHEAD
            if n + 1 < len(self.lines):
                end = self.lines[n + 1][1] - 0.35 * self.LINE_AHEAD
                gone = end + self.LINE_CROSS
            else:
                end, gone = t1, t1 + REST_FADE
            if not start <= t <= gone:
                continue
            op = BAND_ALPHA * float(_smooth((t - start) / self.LINE_CROSS)) \
                * (1 - float(_smooth((t - end) / max(0.01, gone - end))))
            if op > 0.002:
                x0, y0, x1, y1 = band
                lh = y1 - y0
                s = 0.12 * lh
                out.append(((x0 - BAND_PAD_X * lh, y0, x1 + BAND_PAD_X * lh, y1), BAND_RADIUS * lh, s, s, s, op))
        for rb in self.ribbons:
            off = rb["next"] + self.EXIT if rb["next"] is not None else rb["rest"] + REST_FADE
            if not rb["on"] <= t <= off:
                continue
            op = float(_smooth((t - rb["on"]) / self.ENTER))
            if rb["next"] is not None and t > rb["next"]:
                op *= 1 - float(_smooth((t - rb["next"]) / self.EXIT))
            op *= 1 - float(_smooth((t - rb["rest"]) / REST_FADE))
            op *= self._dim(t, rb["rests"])
            if op <= 0.002:
                continue
            g = rb["grid"]
            tc = min(max(t, g[0]), g[-1])
            xl, xr = float(np.interp(tc, g, rb["left"])), float(np.interp(tc, g, rb["right"]))
            lh = rb["lh"]
            out.append(((xl, rb["y0"] - BOX_PAD_Y * lh, xr, rb["y1"] + BOX_PAD_Y * lh),
                        0.28 * lh, 3.0, 0.22 * lh, 0.30 * lh, self.RIBBON_ALPHA * op))
        return out

    def _paint(self, a: np.ndarray, rect, radius: float, soft_y: float, soft_l: float, soft_r: float,
               op: float) -> tuple[int, int, int, int] | None:
        """Lay a rounded rectangle with its own softness on each side onto the opacity map
        (the brighter of the two wins). Returns the area touched, in map pixels."""
        s = self.scale
        x0, y0, x1, y1 = (v * s for v in rect)
        if x1 <= x0 or y1 <= y0 or op <= 0.002:
            return None
        sy = max(0.8, soft_y * s)
        sl, sr = max(sy, soft_l * s), max(sy, soft_r * s)
        ex_l, ex_r = (sl - sy) / 2, (sr - sy) / 2
        rx0, rx1 = x0 - ex_l, x1 + ex_r
        cx, cy = (rx0 + rx1) / 2, (y0 + y1) / 2
        hw, hh = (rx1 - rx0) / 2, (y1 - y0) / 2
        r = min(radius * s, hw, hh)
        m = sy + 2.0
        ax0, ax1 = max(0, int(rx0 - m)), min(self.w, int(rx1 + m) + 1)
        ay0, ay1 = max(0, int(y0 - m)), min(self.h, int(y1 + m) + 1)
        if ax0 >= ax1 or ay0 >= ay1:
            return None
        ys, xs = np.ogrid[ay0:ay1, ax0:ax1]
        xf, yf = xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5
        qx = np.abs(xf - cx) - (hw - r)
        qy = np.abs(yf - cy) - (hh - r)
        sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
        cover = _smooth(0.5 - sdf / sy)
        if ex_l > 0.01:                          # the sides fade out gently, so no letter is cut
            cover = cover * _smooth((xf - (x0 - sl / 2)) / sl)
        if ex_r > 0.01:
            cover = cover * _smooth(((x1 + sr / 2) - xf) / sr)
        patch = a[ay0:ay1, ax0:ax1]
        np.maximum(patch, (cover * np.float32(op)).astype(np.float32), out=patch)
        return ax0, ay0, ax1, ay1

    def alpha(self, t: float) -> np.ndarray:
        """Opacity of the highlight over the whole frame at time t (0..1)."""
        a = np.zeros((self.h, self.w), np.float32)
        for p in self.prims(t):
            self._paint(a, *p)
        return a

    def alpha_area(self, t: float) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """The opacity map at time t and the part of it that is not empty (None: nothing)."""
        a = np.zeros((self.h, self.w), np.float32)
        touched = [z for z in (self._paint(a, *p) for p in self.prims(t)) if z]
        if not touched:
            return a, None
        return a, (min(z[0] for z in touched), min(z[1] for z in touched),
                   max(z[2] for z in touched), max(z[3] for z in touched))

    def bind(self, page: np.ndarray | None = None, ink: np.ndarray | None = None) -> "Highlight":
        """What the highlight is drawn with in apply(): the ink alone (H x W x 4, 0..1),
        laid over the colour. (The page is not needed: the ribbon only adds colour.)"""
        self.ink = None if ink is None else np.asarray(ink, np.float32)
        return self

    def apply(self, picture: np.ndarray, t: float) -> np.ndarray | None:
        """The picture (H x W x 3 uint8, the page as shown) with the highlight at time t,
        the ink laid again on top; None when there is nothing to draw."""
        a, area = self.alpha_area(t)
        if area is None:
            return None
        x0, y0, x1, y1 = area
        out = picture.copy()
        region = out[y0:y1, x0:x1].astype(np.float32)
        cover = a[y0:y1, x0:x1, None]
        region = region * (1 - cover) + self.color * cover
        ink = getattr(self, "ink", None)
        if ink is not None:
            top = ink[y0:y1, x0:x1]
            region = region * (1 - top[..., 3:]) + top[..., :3] * 255 * top[..., 3:]
        out[y0:y1, x0:x1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
        return out

    def frame(self, t: float) -> np.ndarray:
        """The highlight at time t as RGBA (straight alpha)."""
        a = self.alpha(t)
        out = np.empty((self.h, self.w, 4), np.uint8)
        out[..., :3] = self.color.astype(np.uint8)
        out[..., 3] = (a * 255 + 0.5).astype(np.uint8)
        return out

    def frames(self, times):
        """RGBA frames for a run of times, as buffers to write out.

        One frame is kept and only the places that change are repainted: the
        highlight covers a small part of the picture."""
        buf = np.empty((self.h, self.w, 4), np.uint8)
        buf[..., :3] = self.color.astype(np.uint8)
        buf[..., 3] = 0
        a = np.zeros((self.h, self.w), np.float32)
        dirty: list[tuple[int, int, int, int]] = []
        last = None
        for t in times:
            prims = self.prims(t)
            key = tuple((tuple(round(v, 2) for v in p[0]), round(p[5], 4)) for p in prims)
            if key != last:
                for x0, y0, x1, y1 in dirty:
                    a[y0:y1, x0:x1] = 0
                touched = [z for z in (self._paint(a, *p) for p in prims) if z]
                for x0, y0, x1, y1 in dirty + touched:
                    buf[y0:y1, x0:x1, 3] = (a[y0:y1, x0:x1] * 255 + 0.5).astype(np.uint8)
                dirty, last = touched, key
            yield buf.data


def fps_text(fps: float) -> str:
    exact = Fraction(round(fps) * 1000, 1001) if abs(fps - round(fps)) > 0.005 else Fraction(round(fps))
    return f"{exact.numerator}/{exact.denominator}"


def frame_count(seconds: float, fps: float) -> int:
    exact = Fraction(round(fps) * 1000, 1001) if abs(fps - round(fps)) > 0.005 else Fraction(round(fps))
    return int(round(seconds * exact))


def clip_frames(t0: float, t1: float, fps: float) -> int:
    """Frames a clip placed from t0 to t1 must hold: counted the way the timeline counts
    them, plus one spare. Resolve calls a clip missing when it asks for one frame more
    than its file has."""
    return frame_count(t1, fps) - frame_count(t0, fps) + 1


def write_clip(hl: Highlight, fps: float, out: Path, n: int | None = None) -> Path:
    """The highlight of one view as a video with transparency (ProRes 4444), n frames long.

    Resolve reads ProRes 4444 with its alpha on Windows."""
    n = n or frame_count(hl.duration, fps)
    exact = Fraction(fps_text(fps))
    tmp = out.with_name(out.stem + ".part.mov")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{hl.w}x{hl.h}", "-r", fps_text(fps), "-i", "-",
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
           "-alpha_bits", "8", "-qscale:v", "24", "-vendor", "apl0", "-f", "mov", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for data in hl.frames(float(i / exact) for i in range(n)):
            proc.stdin.write(data)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    tmp.replace(out)
    return out


STYLES = ("ribbon", "glass", "glass-tail")


def highlight_for(style: str, events: list[Event], duration: float, color: tuple[int, int, int],
                  size: tuple[int, int] = (1920, 1080), scale: float = 1.0) -> Highlight:
    """The highlight of one view in the chosen style."""
    if style == "ribbon":
        return Highlight(events, duration, color, size, scale)
    from .glass import Glass
    return Glass(events, duration, color, style, size, scale)


def clip_scale(style: str) -> float:
    """The ribbon is soft and fits in half the frame; the glass bends letters, which need every pixel."""
    return CLIP_SCALE if style == "ribbon" else 1.0


def _picture(path: str, scale: float, mode: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert(mode)
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    return np.asarray(img, np.float32)


def _draw_clip(job: dict) -> str:
    """Draw one view's clip (run in a worker process)."""
    hl = highlight_for(job["style"], job["events"], job["duration"], job["color"], job["size"], job["scale"])
    if job["style"] != "ribbon":
        hl.bind(_picture(job["file"], job["scale"], "RGBA"), _picture(job["ink"], job["scale"], "RGBA") / 255)
    write_clip(hl, job["fps"], Path(job["out"]), job["frames"])
    return job["out"]


def workers() -> int:
    """Clips drawn at once: the machine's cores, leaving room for ffmpeg and the system."""
    import os
    return max(1, min(6, (os.cpu_count() or 2) // 3))


def make_clips(views: list[dict], words: list[dict], pos: np.ndarray, book, ocr: dict[int, list],
               color: tuple[int, int, int], fps: float, out_dir: Path,
               size: tuple[int, int] = (1920, 1080), scale: float = 1.0, place: str = "center",
               style: str = "ribbon", progress=None) -> dict:
    """Draw the highlight clip of every page view; set view["highlight"] to its file.

    ocr: words read off each page (with boxes); a page missing from it gets the line only.
    The glass styles need view["file"] (the page as shown) and view["ink"] (the ink alone).
    A clip is named after everything that shapes it, so an unchanged view reuses its
    clip from the last run, and changing only the colour redraws without reading again.
    Several views are drawn at once, in separate processes.
    Returns what a person should know: pages where only the line could be shown."""
    import hashlib
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .book import view_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = {p: page_boxes(book, p, ocr.get(p, [])) for p in sorted({v["page"] for v in views})}
    cs = clip_scale(style)
    used: set[Path] = set()
    jobs = []
    for v in views:
        frame = view_frame(book, v["page"], v["stop"], size, scale, place)
        events = view_events(v, words, pos, book, boxes[v["page"]], frame)
        v["highlight"] = None
        if not events:
            continue
        duration = v["t1"] - v["t0"]
        frames = clip_frames(v["t0"], v["t1"], fps)
        sig = json.dumps([HIGHLIGHT_VERSION, style, cs, color, fps_text(fps), frames,
                          v.get("file") if style != "ribbon" else None,
                          [[round(e.t, 3), round(e.rest, 3), e.box and [round(x, 1) for x in e.box],
                            [round(x, 1) for x in e.band]] for e in events]])
        out = out_dir / f"hl_{book.printed[v['page']]:04d}_{hashlib.sha1(sig.encode()).hexdigest()[:10]}.mov"
        v["highlight"] = str(out)
        v["highlight_s"] = float(frames / Fraction(fps_text(fps)))
        used.add(out)
        if not out.exists():
            jobs.append({"style": style, "events": events, "duration": duration, "color": color, "size": size,
                         "scale": cs, "file": v.get("file"), "ink": v.get("ink"), "fps": fps,
                         "out": str(out), "frames": frames})
    done = 0
    if len(jobs) > 1 and workers() > 1:
        with ProcessPoolExecutor(workers()) as pool:
            for f in as_completed([pool.submit(_draw_clip, j) for j in jobs]):
                f.result()
                done += 1
                if progress:
                    progress(done / len(jobs))
    else:
        for j in jobs:
            _draw_clip(j)
            done += 1
            if progress:
                progress(done / len(jobs))
    for old in out_dir.glob("hl_*"):
        if old not in used:
            old.unlink(missing_ok=True)
    return {"line_only": sorted(book.printed[p] for p, b in boxes.items() if not b.box),
            "bytes": sum(p.stat().st_size for p in used)}


def sample(book, page_idx: int, stop: int, frame, ocr_words, color: tuple[int, int, int],
           page: np.ndarray, ink: np.ndarray, style: str = "ribbon") -> np.ndarray | None:
    """The view with the highlight resting on one of its words, for judging the look.

    page: the view as shown (H x W x 3 or 4, 0..255); ink: the ink alone (H x W x 4, 0..1).
    Returns the picture (H x W x 3 uint8), or None when no word of the view is placed.
    Picks a word a little before the middle of what the view shows. Without the words'
    places (no reader) only a line is lit, as it would be in the result."""
    boxes = page_boxes(book, page_idx, ocr_words or [])
    view = {"page": page_idx, "t0": 0.0, "t1": 1.0}
    seen = []
    for j in np.flatnonzero(book.word_page == page_idx):
        j = int(j)
        if j not in boxes.band:
            continue
        fake = [{"t0": 0.5, "t1": 0.6}]
        ev = view_events(view, fake, np.array([j]), book, boxes, frame)
        if ev:
            seen.append(ev[0])
    if not seen:
        return None
    with_box = [e for e in seen if e.box is not None] or seen
    e = with_box[int(len(with_box) * 0.4)]
    e = Event(0.0, 99.0, e.box, e.band)
    # a moment when the ribbon has come to rest on the word, before it dims
    settled = 0.62 * _said_in(e.box[2] - e.box[0]) + Highlight.TRAIL + 0.3 if e.box else 1.0
    h, w = page.shape[:2]
    hl = highlight_for(style, [e], 3.0, color, (w, h)).bind(page, ink)
    shown = np.asarray(page, np.float32)
    if shown.shape[2] == 4:                          # see-through page: judge it on white
        a = shown[..., 3:] / 255
        shown = shown[..., :3] * a + 255 * (1 - a)
    shown = np.clip(shown + 0.5, 0, 255).astype(np.uint8)
    out = hl.apply(shown, settled)
    return shown if out is None else out
