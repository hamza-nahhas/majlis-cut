"""Check the turath helpers: python tests/try_catalog.py <pdf> <work_dir>"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.book import guess_page_offset, load_turath_pages, turath_catalog, turath_volumes  # noqa: E402

pdf, cache = Path(sys.argv[1]), Path(sys.argv[2]) / "books"
cat = turath_catalog(cache)
print(len(cat), "books;", [b for b in cat if "زاد المعاد" in b["name"]])
print(turath_volumes(197, cache))
t = time.time()
print("offset, margin:", guess_page_offset(pdf, load_turath_pages(197, "3", cache)), f"{time.time() - t:.1f}s")
