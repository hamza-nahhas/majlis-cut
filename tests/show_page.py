"""Print the turath text of printed pages: python tests/show_page.py <work_dir> <vol> <page> [page…]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.arabic import tokens  # noqa: E402
from darsmontage.book import load_turath_pages  # noqa: E402

pages = load_turath_pages(197, sys.argv[2], Path(sys.argv[1]) / "books")
for p in sys.argv[3:]:
    print(f"--- page {p} ---\n{pages[int(p)]}\n--- normalised ---\n{' '.join(tokens(pages[int(p)]))}\n")
