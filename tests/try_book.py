"""Manual check of the book layer: python tests/try_book.py <pdf> <work_dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.book import build_book, load_turath_pages, render_view, view_stops  # noqa: E402

pdf, work = Path(sys.argv[1]), Path(sys.argv[2])
text = load_turath_pages(197, "3", work / "books")
book = build_book(pdf, text, first=5, last=50)
print("words", len(book.words), "pages", len(book.layouts))
for idx in sorted(book.layouts)[:48]:
    lay = book.layouts[idx]
    n = int((book.word_page == idx).sum())
    print(f"pdf#{idx + 1} printed {book.printed[idx]}: body lines {len(lay.body_lines):2d} "
          f"words {n:3d} body {lay.body_lines[0][0]:.2f}-{lay.body_lines[-1][1]:.2f} "
          f"content {lay.y0:.2f}-{lay.y1:.2f} stops {[round(s, 2) for s in view_stops(lay, 0.55)]}")
for pno, stop in [(13, 0), (13, 2), (31, 1)]:
    print(render_view(book, pno - 1, stop, 0.55, work / "views_test"))
