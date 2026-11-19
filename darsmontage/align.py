"""Find where the transcript follows the book.

The reader reads the book word for word, the commentator speaks freely. So every
stretch of the transcript that matches the book, in order, is reading, and the
book position of the match tells which page is being read. Everything else is
commentary.

Method: exact word n-grams shared by transcript and book are anchors (i, j);
a chaining pass keeps the anchors that advance together through both texts,
allowing the transcript to wander off (commentary) and come back.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from .arabic import tokens
from .book import Book

K = 3              # words per anchor
MAX_OCC = 4        # an n-gram found more often than this in the book is too common to anchor on
LOOKBACK = 300
DIAG_SLACK = 8     # words the two texts may drift apart inside one reading run
DIAG_GAP = 30      # transcript words without any anchor that a reading run survives
JUMP = 3.0         # cost of leaving the book and coming back


@dataclass
class Run:
    i0: int        # transcript token range [i0, i1]
    i1: int
    j0: int        # book word range [j0, j1]
    j1: int
    anchors: list[tuple[int, int]]


def transcript_tokens(words: list[dict]) -> tuple[list[str], np.ndarray]:
    """Normalised tokens of the transcript and, for each, the index of its word."""
    toks: list[str] = []
    src: list[int] = []
    for n, w in enumerate(words):
        for t in tokens(w["w"]):
            toks.append(t)
            src.append(n)
    return toks, np.array(src, dtype=int)


def find_anchors(t: list[str], b: list[str]) -> np.ndarray:
    index: dict[tuple, list[int]] = defaultdict(list)
    for j in range(len(b) - K + 1):
        index[tuple(b[j:j + K])].append(j)
    out = []
    for i in range(len(t) - K + 1):
        js = index.get(tuple(t[i:i + K]))
        if js and len(js) <= MAX_OCC:
            out.extend((i, j) for j in js)
    return np.array(out, dtype=int).reshape(-1, 2)


def chain(anchors: np.ndarray) -> list[tuple[int, int]]:
    n = len(anchors)
    if n == 0:
        return []
    ti, bj = anchors[:, 0], anchors[:, 1]
    score = np.ones(n)
    prev = np.full(n, -1)
    far_best, far_idx = -np.inf, -1          # best chain end older than the lookback window
    for a in range(n):
        lo = max(0, a - LOOKBACK)
        if lo > 0 and score[lo - 1] > far_best:
            far_best, far_idx = score[lo - 1], lo - 1
        if a > lo:
            di = ti[a] - ti[lo:a]
            dj = bj[a] - bj[lo:a]
            off = np.abs(dj - di)
            diag = (dj > 0) & (off <= DIAG_SLACK) & (di <= DIAG_GAP)
            cost = np.where(diag, 0.3 * off,
                            JUMP + np.minimum(0.004 * np.abs(dj), 6.0) + np.where(dj <= 0, 2.0, 0.0))
            cand = score[lo:a] + 1.0 - cost
            cand[di <= 0] = -np.inf
            m = int(cand.argmax())
            if cand[m] > score[a]:
                score[a], prev[a] = cand[m], lo + m
        if far_idx >= 0 and far_best + 1.0 - (JUMP + 8.0) > score[a]:
            score[a], prev[a] = far_best + 1.0 - (JUMP + 8.0), far_idx
    a = int(score.argmax())
    path = []
    while a >= 0:
        path.append((int(ti[a]), int(bj[a])))
        a = int(prev[a])
    return path[::-1]


def split_runs(path: list[tuple[int, int]]) -> list[Run]:
    runs: list[Run] = []
    for i, j in path:
        if runs:
            r = runs[-1]
            pi, pj = r.anchors[-1]
            di, dj = i - pi, j - pj
            if dj > 0 and abs(dj - di) <= DIAG_SLACK and di <= DIAG_GAP:
                r.anchors.append((i, j))
                r.i1, r.j1 = i + K - 1, j + K - 1
                continue
        runs.append(Run(i, i + K - 1, j, j + K - 1, [(i, j)]))
    return runs


def _similar(a: str, b: str) -> bool:
    """A match strong enough to prove the reader is still on the book: short
    words such as «أن» or «في» turn up everywhere and prove nothing."""
    if len(a) < 3:
        return False
    return a == b or (len(a) >= 4 and fuzz.ratio(a, b) >= 75)


def extend_edges(runs: list[Run], t: list[str], b: list[str], reach: int = 20) -> None:
    """Grow each run over neighbouring words that still follow the book but had no exact anchor."""
    for n, r in enumerate(runs):
        floor = runs[n - 1].i1 + 1 if n else 0
        ceil = runs[n + 1].i0 - 1 if n + 1 < len(runs) else len(t) - 1
        for step in (-1, 1):
            i_edge, j_edge = (r.i0, r.j0) if step < 0 else (r.i1, r.j1)
            miss, best = 0, 0
            for m in range(1, reach + 1):
                i, j = i_edge + step * m, j_edge + step * m
                if not floor <= i <= ceil or not 1 <= j < len(b) - 1 or miss >= 2:
                    break
                if any(_similar(t[i], b[jj]) for jj in (j - 1, j, j + 1)):
                    best, miss = m, 0
                elif len(t[i]) >= 3:
                    miss += 1
            if step < 0:
                r.i0, r.j0 = r.i0 - best, r.j0 - best
            else:
                r.i1, r.j1 = r.i1 + best, r.j1 + best


# What the reader says before he starts a passage. It is his voice, so the page
# is put on screen from there.
LEAD_IN = set(tokens("أحسن الله إليكم إليك قال رحمه تعالى المصنف المؤلف بسم الرحمن الرحيم"))
LEAD_IN_KEY = set(tokens("أحسن قال بسم"))


def include_lead_in(runs: list[Run], t: list[str], reach: int = 9) -> None:
    for n, r in enumerate(runs):
        floor = runs[n - 1].i1 + 1 if n else 0
        i = r.i0
        while i - 1 >= floor and r.i0 - (i - 1) <= reach and t[i - 1] in LEAD_IN:
            i -= 1
        while i < r.i0 and t[i] not in LEAD_IN_KEY:     # «…بإذن الله» belongs to the commentator
            i += 1
        r.i0 = i


def align(words: list[dict], book: Book, min_read_s: float = 3.0,
          min_talk_s: float = 4.0, lead_in: bool = True, voice=None) -> dict:
    """Return the reading runs and, per transcript word, its book position (-1 = commentary)."""
    t, src = transcript_tokens(words)
    anchors = find_anchors(t, book.words)
    path = chain(anchors)
    runs = split_runs(path)
    t0 = np.array([w["t0"] for w in words])
    t1 = np.array([w["t1"] for w in words])

    def span(r: Run) -> float:
        return t1[src[r.i1]] - t0[src[r.i0]]

    def when(r: Run) -> tuple[float, float]:
        return float(t0[src[r.i0]]), float(t1[src[r.i1]])

    # A couple of matching words in the middle of free speech is a quotation, not reading.
    runs = [r for r in runs if len(r.anchors) >= 3]
    extend_edges(runs, t, book.words)

    # Learn the two voices from the parts the text match is sure about.
    heard = False
    if voice is not None and runs:
        sure_read = [(a + 1, b - 1) for a, b in map(when, runs) if b - a >= 8]
        edges = [0.0] + [x for r in runs for x in when(r)] + [float(t1[-1])]
        sure_talk = [(a + 3, b - 3) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 15]
        heard = voice.fit(sure_read, sure_talk)

    def reader_voice(a: float, b: float) -> float:
        score = voice.reader_score(a, b) if heard else None
        return 0.0 if score is None else score

    # The commentator repeating a sentence of the book is still commentary.
    runs = [r for r in runs if not (span(r) < 12 and reader_voice(*when(r)) < -0.4)]
    # Join runs split by a stumble or a short interjection while the book goes on.
    merged: list[Run] = []
    for r in runs:
        if merged:
            p = merged[-1]
            if when(r)[0] - when(p)[1] < min_talk_s and -5 <= r.j0 - p.j1 <= 80:
                p.i1, p.j1 = r.i1, max(p.j1, r.j1)
                p.anchors.extend(r.anchors)
                continue
        merged.append(r)
    runs = [r for r in merged if span(r) >= min_read_s]

    # Book position of every transcript word inside a run, interpolated between anchors.
    pos = np.full(len(words), -1, dtype=int)
    for r in runs:
        xs = [r.i0] + [a[0] for a in r.anchors] + [r.i1]
        ys = [r.j0] + [a[1] for a in r.anchors] + [r.j1]
        if xs[0] == xs[1]:
            xs, ys = xs[1:], ys[1:]
        ii = np.arange(r.i0, r.i1 + 1)
        jj = np.maximum.accumulate(np.interp(ii, xs, ys)).round().astype(int)
        pos[src[ii]] = np.clip(jj, 0, len(book.words) - 1)
    if lead_in:
        include_lead_in(runs, t)
    return {"runs": runs, "pos": pos, "tokens": t, "src": src,
            "n_anchors": len(anchors), "n_path": len(path),
            "voice": voice if heard else None,
            "voice_accuracy": voice.accuracy if heard else None}


def _boundary(end_prev: float, start_next: float) -> float:
    """Cut point in the pause between two words: close to the start of the next one."""
    gap = start_next - end_prev
    return start_next - 0.2 if gap > 0.5 else end_prev + gap / 2


def reading_spans(words: list[dict], result: dict, duration: float,
                  min_camera_s: float = 3.0) -> list[dict]:
    """Time ranges during which the book is on screen.

    They start as the matched runs. Then the reader's voice decides what the text
    cannot: reading that goes on past the matched words, and stretches the reader
    read that the transcript lost or the book text lacks (a verse cited in short).
    """
    src, pos, voice = result["src"], result["pos"], result["voice"]
    spans: list[dict] = []
    for r in result["runs"]:
        w0, w1 = int(src[r.i0]), int(src[r.i1])
        start = _boundary(words[w0 - 1]["t1"], words[w0]["t0"]) if w0 > 0 else 0.0
        end = _boundary(words[w1]["t1"], words[w1 + 1]["t0"]) if w1 + 1 < len(words) \
            else min(duration, words[w1]["t1"] + 0.3)
        if spans and start < spans[-1]["t1"]:
            start = spans[-1]["t1"]
        spans.append({"t0": start, "t1": end, "j0": r.j0, "j1": r.j1})
    if voice is None or not spans:
        return spans

    starts = np.array([w["t0"] for w in words])
    ends = np.array([w["t1"] for w in words])

    def snap(t: float) -> float:
        """Move a cut found by ear to the widest pause between words nearby."""
        n0, n1 = np.searchsorted(ends, t - 1.2), np.searchsorted(starts, t + 1.2)
        best, at = 0.25, t
        for n in range(max(1, n0 + 1), min(len(words), n1)):
            pause = starts[n] - ends[n - 1]
            if pause > best:
                best, at = pause, _boundary(ends[n - 1], starts[n])
        return float(at)

    def reader(a: float, b: float) -> float:
        score = voice.reader_score(a, b)
        return -9.0 if score is None else score

    out: list[dict] = []
    bounds = [None] + spans + [None]
    for prev, nxt in zip(bounds, bounds[1:]):
        a = prev["t1"] if prev else 0.0
        b = nxt["t0"] if nxt else duration
        if b - a >= 4:
            grid = np.arange(a, b - 3 + 1e-6, 1.0)             # 3 s windows, 1 s apart
            good = [reader(k, k + 3) > 0.45 for k in grid]
            lead = next((n for n, g in enumerate(good) if not g), len(good)) if prev else 0
            trail = next((n for n, g in enumerate(reversed(good)) if not g), len(good)) if nxt else 0
            whole = prev and nxt and b - a <= 30 and -5 <= nxt["j0"] - prev["j1"] <= 80 \
                and reader(a, b) > 0.2
            if (prev and nxt and lead == len(good)) or whole:    # the reader never stopped
                out[-1]["t1"], out[-1]["j1"] = nxt["t1"], nxt["j1"]
                continue
            if prev and lead >= 2:
                out[-1]["t1"] = min(b, snap(a + lead + 1))
            new_start = max(out[-1]["t1"] if out else 0.0, snap(b - trail - 1)) \
                if nxt and trail >= 2 else b
            # Islands: the reader alone for six seconds or more, away from any matched run.
            n = lead + 1 if prev else 0
            stop_at = len(good) - trail - (1 if nxt else 0)
            while n < stop_at:
                if good[n]:
                    m = n
                    while m + 1 < stop_at and good[m + 1]:
                        m += 1
                    if m - n + 1 >= 4:
                        out.append({"t0": snap(grid[n] + 0.5), "t1": snap(grid[m] + 2.5),
                                    "j0": None, "j1": None})
                    n = m + 1
                n += 1
            if nxt:
                nxt = dict(nxt, t0=new_start)
        if nxt:
            out.append(nxt)
    out = [s for s in out if s["t1"] - s["t0"] > 0.5]
    out.sort(key=lambda s: s["t0"])
    # A camera shot of a second or two between two readings is only a flicker: stay on the book.
    joined: list[dict] = []
    for s in out:
        if joined and s["t0"] - joined[-1]["t1"] < min_camera_s:
            joined[-1]["t1"] = max(joined[-1]["t1"], s["t1"])
            if joined[-1]["j0"] is None:
                joined[-1]["j0"] = s["j0"]
            joined[-1]["j1"] = s["j1"] if s["j1"] is not None else joined[-1]["j1"]
        else:
            joined.append(s)
    return joined


def build_plan(words: list[dict], book: Book, result: dict, duration: float,
               min_view_s: float = 2.0) -> dict:
    """Turn the alignment into what the editor needs: reading ranges and page views over time."""
    pos = result["pos"]
    starts = np.array([w["t0"] for w in words])
    segments: list[dict] = []
    views: list[dict] = []
    last_pos = None
    spans = reading_spans(words, result, duration)
    for n_span, span in enumerate(spans):
        start, end = span["t0"], span["t1"]
        w0, w1 = int(np.searchsorted(starts, start)), int(np.searchsorted(starts, end - 0.1)) - 1
        matched = [w for w in range(w0, w1 + 1) if pos[w] >= 0]
        if matched:
            track = [(words[w]["t0"], int(pos[w])) for w in matched]
        else:
            # Nothing matched here (heard, not read from the transcript): stay where the book is.
            later = next((int(pos[w]) for w in range(w1 + 1, len(words)) if pos[w] >= 0), None)
            where = last_pos if last_pos is not None else later
            if where is None:
                continue
            track = [(start, where)]
        last_pos = track[-1][1]
        seg_views: list[dict] = []
        page, stop = None, None
        for at_word, j in track:
            p, y = int(book.word_page[j]), float(book.word_y[j])
            s = book.pick_stop(p, y, stop if p == page else None)
            if p != page or s != stop:
                at = start if not seg_views else max(seg_views[-1]["t0"], at_word - 0.3)
                if seg_views and at - seg_views[-1]["t0"] < min_view_s:
                    seg_views[-1].update(page=p, stop=s)      # too brief to show: replace it
                else:
                    seg_views.append({"t0": at, "page": p, "stop": s})
                page, stop = p, s
        for n, v in enumerate(seg_views):
            v["t1"] = seg_views[n + 1]["t0"] if n + 1 < len(seg_views) else end
            v["printed"] = book.printed[v["page"]]
        views.extend(seg_views)
        segments.append({
            "t0": start, "t1": end, "kind": "read",
            "by": "text" if matched else "voice",
            "pages": sorted({v["printed"] for v in seg_views}),
            "first_words": " ".join(w["w"] for w in words[w0:w0 + 6]) if w1 >= w0 else "",
            "last_words": " ".join(w["w"] for w in words[max(w0, w1 - 5):w1 + 1]) if w1 >= w0 else "",
        })
    # Fill the rest with commentary.
    full: list[dict] = []
    cursor = 0.0
    def talk(a: float, b: float) -> dict:
        w0, w1 = int(np.searchsorted(starts, a)), int(np.searchsorted(starts, b - 0.1))
        said = [w["w"] for w in words[w0:w1]]
        return {"t0": a, "t1": b, "kind": "talk", "first_words": " ".join(said[:8]),
                "last_words": " ".join(said[-6:]) if len(said) > 8 else ""}

    for s in segments:
        if s["t0"] - cursor > 0.05:
            full.append(talk(cursor, s["t0"]))
        full.append(s)
        cursor = s["t1"]
    if duration - cursor > 0.05:
        full.append(talk(cursor, duration))
    return {"duration": duration, "segments": full, "views": views}
