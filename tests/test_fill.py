"""Check of filling the empty places on screen with the book page.

    python tests/test_fill.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.align import fill_gaps  # noqa: E402


def view(t0, t1, page):
    return {"t0": t0, "t1": t1, "page": page, "stop": 0, "printed": page, "file": f"p{page}.png"}


def shown(views):
    return [(v["t0"], v["t1"], v["page"], bool(v.get("fill"))) for v in views]


# Sound only: commentary between readings is covered by the page read last,
# the start by the first page and the end by the last.
views = [view(10, 20, 1), view(20, 30, 2), view(50, 60, 3)]
assert shown(fill_gaps(views, [], 100)) == [(0, 20, 1, False), (20, 50, 2, False), (50, 100, 3, False)]
assert views[0]["t0"] == 10, "the input is left untouched"

# Cameras: only what no camera covers is filled.
cams = [{"t0": 30, "t1": 40}]
assert shown(fill_gaps(views, cams, 60)) == [(0, 20, 1, False), (20, 30, 2, False), (40, 50, 2, True), (50, 60, 3, False)]

# A short hole between the camera and the next reading: the next page comes in early.
cams = [{"t0": 30, "t1": 49}]
assert shown(fill_gaps(views, cams, 60)) == [(0, 20, 1, False), (20, 30, 2, False), (49, 60, 3, False)]

# A short hole between two camera clips stays empty, a long one gets the last page.
cams = [{"t0": 30, "t1": 35}, {"t0": 36, "t1": 40}, {"t0": 45, "t1": 50}]
assert shown(fill_gaps(views, cams, 60)) == [(0, 20, 1, False), (20, 30, 2, False), (40, 45, 2, True), (50, 60, 3, False)]

# A range the reviewer sent to the camera stays empty.
assert shown(fill_gaps(views, [], 60, keep_empty=[{"t0": 30, "t1": 50}])) == \
    [(0, 20, 1, False), (20, 30, 2, False), (50, 60, 3, False)]

assert fill_gaps([], [], 60) == []
print("fill: all checks passed")
