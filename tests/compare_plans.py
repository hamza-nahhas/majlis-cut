"""How far two results of the same lesson agree: python tests/compare_plans.py <out A> <out B>"""
import json
import sys
from pathlib import Path

a, b = (json.loads((Path(p) / "work" / "plan.json").read_text(encoding="utf-8")) for p in sys.argv[1:3])


def kind_at(plan, t):
    for s in plan["segments"]:
        if s["t0"] <= t < s["t1"]:
            return s["kind"]
    return "talk"


def view_at(plan, t):
    for v in plan["views"]:
        if v["t0"] <= t < v["t1"]:
            return v
    return None


same = differ = page_same = page_diff = stop_diff = 0
spots = []
t = 0.0
while t < a["duration"]:
    ka, kb = kind_at(a, t), kind_at(b, t)
    if ka == kb:
        same += 1
        if ka == "read":
            va, vb = view_at(a, t), view_at(b, t)
            if va and vb:
                if va["page"] != vb["page"]:
                    page_diff += 1
                elif va["stop"] != vb["stop"]:
                    stop_diff += 1
                else:
                    page_same += 1
    else:
        differ += 1
        if not spots or t - spots[-1][1] > 1.0:
            spots.append([t, t, ka, kb])
        spots[-1][1] = t
    t += 0.5
print(f"same kind {same / (same + differ):.1%}; differing {differ * 0.5:.0f} s")
print(f"while reading in both: same view {page_same * 0.5:.0f} s, other part of the page {stop_diff * 0.5:.0f} s, "
      f"other page {page_diff * 0.5:.0f} s")
for t0, t1, ka, kb in spots:
    if t1 - t0 >= 3:
        print(f"  {int(t0 // 60)}:{t0 % 60:04.1f} for {t1 - t0:.0f} s: A={ka} B={kb}")
