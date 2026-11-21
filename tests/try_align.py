"""Manual check of the alignment: python tests/try_align.py <pdf> <work_dir> <words.json|partial.jsonl>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.align import align, build_plan  # noqa: E402
from majlis_cut.book import build_book, load_turath_pages  # noqa: E402

pdf, work, src = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if src.suffix == ".jsonl":
    words = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
else:
    words = json.loads(src.read_text(encoding="utf-8"))["words"]
book = build_book(pdf, load_turath_pages(197, "3", work / "books"), first=1, last=80)
res = align(words, book)
plan = build_plan(words, book, res, words[-1]["t1"])
print(f"words {len(words)}  tokens {len(res['tokens'])}  anchors {res['n_anchors']}  on path {res['n_path']}  runs {len(res['runs'])}")


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


read = sum(s["t1"] - s["t0"] for s in plan["segments"] if s["kind"] == "read")
print(f"reading {read / 60:.1f} min of {plan['duration'] / 60:.1f} min")
for s in plan["segments"]:
    if s["kind"] == "read":
        print(f"{mmss(s['t0'])}-{mmss(s['t1'])} READ p{s['pages']}  «{s['first_words']}» … «{s['last_words']}»")
    else:
        ws = [w["w"] for w in words if s["t0"] <= w["t0"] < s["t1"]]
        print(f"{mmss(s['t0'])}-{mmss(s['t1'])} talk ({len(ws)} words)  «{' '.join(ws[:6])}» … «{' '.join(ws[-6:])}»")
print("views:", len(plan["views"]))
for v in plan["views"][:40]:
    print(f"   {mmss(v['t0'])}-{mmss(v['t1'])} page {v['printed']} view {'abcdefghij'[v['stop']]}")
