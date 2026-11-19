"""Show transcript words with their book match: python tests/dump_range.py <pdf> <work> <words> <from_s> <to_s>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.align import align  # noqa: E402
from darsmontage.book import build_book, load_turath_pages  # noqa: E402

pdf, work, src = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
a, b = float(sys.argv[4]), float(sys.argv[5])
if src.suffix == ".jsonl":
    words = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
else:
    words = json.loads(src.read_text(encoding="utf-8"))["words"]
book = build_book(pdf, load_turath_pages(197, "3", work / "books"), first=1, last=80)
res = align(words, book)
for n, w in enumerate(words):
    if a <= w["t0"] <= b:
        j = res["pos"][n]
        match = f"-> {book.words[j]} (p{book.printed[int(book.word_page[j])]} y{book.word_y[j]:.2f})" if j >= 0 else ""
        print(f"{w['t0']:7.2f}-{w['t1']:7.2f}  {w['w']:<14} {match}")
