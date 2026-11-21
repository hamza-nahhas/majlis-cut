"""How well does OCR of the scan agree with the digitised text?

python tests/try_ocr.py <pdf> <turath id> <vol> <first page> <last page> [offset]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from majlis_cut.arabic import tokens
from majlis_cut.book import load_turath_pages
from majlis_cut.ocr import ocr_pages

pdf, book_id, vol, first, last = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
offset = int(sys.argv[6]) if len(sys.argv) > 6 else 0
truth = load_turath_pages(book_id, vol, Path(__file__).resolve().parent.parent / "cache")

t = time.time()
got = ocr_pages(pdf, [p + offset - 1 for p in range(first, last + 1)])
print(f"{(time.time() - t) / (last - first + 1):.2f} s per page")


def grams(ws, n=3):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


tot_w = tot_g = hit_w = hit_g = extra = 0
for p in range(first, last + 1):
    want = tokens(truth.get(p, ""))
    have = tokens(" ".join(w.text for w in got[p + offset - 1]))
    hw = sum(1 for w in want if w in set(have))
    hg = len(grams(want) & grams(have))
    tot_w += len(want); hit_w += hw
    tot_g += len(grams(want)); hit_g += hg
    extra += len(have) - len(want)
    print(f"page {p}: {len(want)} words in text, {len(have)} from OCR, "
          f"words found {hw / max(1, len(want)):.0%}, triples found {hg / max(1, len(grams(want))):.0%}")
print(f"all: words {hit_w / tot_w:.1%}, triples {hit_g / tot_g:.1%}, extra OCR words {extra}")
if "--show" in sys.argv:
    for w in got[first + offset - 1][:80]:
        print(f"{w.y:.3f} {w.text}")
