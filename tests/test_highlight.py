"""Check of the reading highlight: matching words to the scan, the flowing ribbon,
and the clips' length on the timeline.

    python tests/test_highlight.py
"""
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.arabic import tokens  # noqa: E402
from majlis_cut.export import Timeline  # noqa: E402
from majlis_cut.highlight import (BOX_ALPHA, BOX_PAD_X, HOLD, Event, Highlight,  # noqa: E402
                                  _match, clip_frames, parse_color)


def toks(text):
    return tokens(text)


# --- matching the book's words to the words read off the scan ---------------

book = toks("وأما جهاد الشيطان فمرتبتان إحداهما جهاده على دفع ما يلقي إلى العبد")
# The same words, one misread, plus a footnote mark the reader picked up.
scan = toks("وأما جهاد الشيطان فمرتبتان ١ إحداهما جهاذه على دفع ما يلقي إلى العبد")
pairs = {b[0]: s[0] for b, s in _match(book, scan)}
assert len(pairs) == len(book), "every book word finds its scan word"
assert scan[pairs[book.index("جهاده")]] == toks("جهاذه")[0], "a misread word still matches"
assert all(scan[s] != "١" for s in pairs.values()), "the footnote mark is left out"

# The reader runs Quranic words together: one scan word holds two book words.
book = toks("فاصدع بما تؤمر واعرض عن المشركين")
scan = toks("فاصدع بماتؤمر واعرض عن المشركين")
groups = _match(book, scan)
assert ([1, 2], [1]) in groups, groups

# A scan with nothing in common: nothing is matched.
assert _match(toks("قال رسول الله"), toks("xx yy zz qq")) == []


# --- the flowing ribbon ------------------------------------------------------------

LH = 50.0
line1, line2, line3 = (100.0, 200.0, 1500.0, 250.0), (100.0, 300.0, 1500.0, 350.0), (100.0, 400.0, 1500.0, 450.0)


def word(t, right, line, width=100.0, said=0.35):
    return Event(t, t + said + HOLD, (right - width, line[1], right, line[3]), line)


# Line 1: four words at a steady pace, a 2 s pause, three more; line 2: two words;
# line 3: a word whose place on the scan is unknown.
events = [word(1.0 + 0.4 * k, 1450 - 130 * k, line1) for k in range(4)]
events += [word(4.0 + 0.4 * k, 1450 - 130 * (k + 4), line1) for k in range(3)]
events += [word(5.6 + 0.4 * k, 1450 - 130 * k, line2) for k in range(2)]
events += [Event(7.5, 7.5 + 0.35 + HOLD, None, line3)]
hl = Highlight(events, 10.0, (64, 150, 255), scale=0.5)


def ribbon(t):
    """The ribbon's rectangle and opacity at t (bands are drawn softer than 3 px)."""
    got = [(p[0], p[5]) for p in hl.prims(t) if p[2] == 3.0]
    return got[0] if got else (None, 0.0)


def bands(t):
    return [p[0][1] for p in hl.prims(t) if p[2] != 3.0]


assert hl.prims(0.3) == [], "nothing long before the first word"
assert bands(0.9) == [line1[1]] and ribbon(0.85)[0] is None, "the line lights up just before its first word"

# While the words follow one another the ribbon never stops and never goes back.
steady = [1.0 + i / 25 for i in range(int(1.55 * 25))]           # the four words up to the pause
edges = [ribbon(t)[0] for t in steady]
lefts, rights = [e[0] for e in edges], [e[2] for e in edges]
assert all(b <= a + 1e-6 for a, b in zip(lefts, lefts[1:])), "the front edge only moves on"
assert all(b <= a + 1e-6 for a, b in zip(rights, rights[1:])), "the back edge only moves on"
still = sum(abs(b - a) < 0.3 for a, b in zip(lefts[3:], lefts[4:]))
assert still <= 0.1 * len(lefts), f"it glides instead of stopping on words ({still} still frames)"
pad = BOX_PAD_X * LH
assert min(lefts) >= 1450 - 130 * 6 - 100 - pad - 1 and max(rights) <= 1450 + pad + 1, "it stays on the text"

# It is over each word while the word is being said.
for e in events[:3]:
    rect, _ = ribbon(e.t + 0.2)
    overlap = min(rect[2], e.box[2]) - max(rect[0], e.box[0])
    assert overlap >= 0.5 * (e.box[2] - e.box[0]), f"the ribbon covers the word said at {e.t}"

# In the pause it comes to rest on the last word and dims a little.
rest = [ribbon(t) for t in (3.0, 3.2, 3.4)]
assert all(abs(a - b) < 0.01 for r in rest[1:] for a, b in zip(r[0], rest[0][0])), "it rests in a pause"
assert Highlight.DIM * BOX_ALPHA - 1e-3 <= rest[2][1] < BOX_ALPHA, "and dims, but stays visible"
assert ribbon(1.5)[1] == BOX_ALPHA, "while reading it is fully shown"

# The next line lights up before the reading reaches it.
assert line2[1] in bands(5.6 - 0.1), "the next line's band comes before its first word"
assert ribbon(7.6)[0] is None and line3[1] in bands(7.6), "a word without a box: the line only"

# The frames written to the clip are the same as frames drawn from scratch.
times = [i / 25 for i in range(250)]
for i, data in enumerate(hl.frames(times)):
    if i % 5 == 0:
        got = np.frombuffer(data, np.uint8).reshape(hl.h, hl.w, 4)
        assert np.array_equal(got, hl.frame(times[i])), f"frame {i} differs"

assert parse_color("#4096FF") == (64, 150, 255) and parse_color("fa0") == (255, 170, 0)


# --- clip length on the timeline -----------------------------------------------

# Resolve calls a clip missing when it asks for a frame more than the file has.
random.seed(1)
for fps in (25.0, 29.97, 30.0, 23.976, 50.0):
    tl = Timeline("t", fps)
    for _ in range(200):
        t0 = random.uniform(0, 4000)
        t1 = t0 + random.uniform(0.3, 90)
        n = clip_frames(t0, t1, fps)
        assert n >= tl.frames(t1) - tl.frames(t0) + 1, (fps, t0, t1)

with tempfile.TemporaryDirectory() as tmp:
    tl = Timeline("t", 25.0)
    t0, t1 = 33.96, 44.82
    n = clip_frames(t0, t1, 25.0)
    tl.add_video(3, Path(tmp) / "hl.mov", t0, t1, src_duration=n / 25, size=(960, 540), alpha=True)
    out = tl.write_fcp7_xml(Path(tmp) / "t.xml")
    clip = ET.parse(out).getroot().find(".//clipitem")
    used = int(clip.find("out").text) - int(clip.find("in").text)
    assert used <= int(clip.find("file/duration").text), "the clip stays inside its file"
    assert clip.find("file/media/video/samplecharacteristics/width").text == "960"
    assert clip.find("alphatype").text == "straight"



# --- the glass looks ----------------------------------------------------------------

from majlis_cut.glass import Glass  # noqa: E402
from majlis_cut.highlight import STYLES, highlight_for  # noqa: E402

rng = np.random.default_rng(0)
W_, H_ = 640, 360
paper = np.full((H_, W_, 3), 245.0, np.float32)
paper[150:180:4, 60:600] = 20.0                           # lines of "ink"
ink = np.zeros((H_, W_, 4), np.float32)
ink[150:180:4, 60:600] = (20 / 255, 20 / 255, 20 / 255, 1.0)
gl_events = [word(0.5 + 0.4 * k, 1450 - 130 * k, line1) for k in range(5)]
for style in ("glass", "glass-tail"):
    g = highlight_for(style, gl_events, 4.0, (64, 150, 255), scale=W_ / 1920)
    assert isinstance(g, Glass)
    g.bind(paper, ink)
    shown = np.clip(paper + 0.5, 0, 255).astype(np.uint8)
    times = [i / 25 for i in range(100)]
    for i, data in enumerate(g.frames(times)):
        if i % 10:
            continue
        rgba = np.frombuffer(data, np.uint8).reshape(g.h, g.w, 4).astype(np.float32)
        over = shown * (1 - rgba[..., 3:] / 255) + rgba[..., :3] * rgba[..., 3:] / 255
        direct = g.apply(shown, times[i])
        if direct is not None:
            assert np.abs(over - direct).max() <= 2.0, f"{style}: the clip over the page is the glass ({i})"
    # on a see-through page nothing is bent: the patch is a pure see-through layer
    clear = np.dstack([paper, np.zeros((H_, W_), np.float32)])
    g2 = highlight_for(style, gl_events, 4.0, (64, 150, 255), scale=W_ / 1920).bind(clear, ink)
    patches = g2.patches(1.6)
    assert patches and all(np.all(a < 1.0) for _, _, a, _ in patches), "no solid rim pixels over a clear page"

# glass hanging over the frame's edge is cut there, not dropped
edge_events = [word(0.5, 1915, line1, width=200)]
g = highlight_for("glass", edge_events, 2.0, (64, 150, 255), scale=W_ / 1920).bind(paper, ink)
assert any(x0 + a.shape[1] == W_ for _, x0, a, _ in g.patches(1.0)), "the pane reaches the frame edge"
assert set(STYLES) == {"ribbon", "glass", "glass-tail"}

print("highlight: all checks passed")
