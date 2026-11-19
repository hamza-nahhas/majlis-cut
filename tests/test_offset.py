"""Page-offset detection on the full volume and on excerpts cut from it.

    python tests/test_offset.py <pdf> <work_dir>
"""
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.book import guess_page_offset, load_turath_pages  # noqa: E402

pdf, work = Path(sys.argv[1]), Path(sys.argv[2])
text = load_turath_pages(197, "3", work.parent / "cache")
cases = [(None, 0)]
for first, last in [(11, 41), (300, 345), (600, 612)]:        # printed pages kept in the excerpt
    part = work / f"excerpt_{first}_{last}.pdf"
    doc = fitz.open(pdf)
    doc.select(list(range(first - 1, last)))
    doc.save(part)
    cases.append((part, 1 - first))
for path, want in cases:
    offset, margin = guess_page_offset(path or pdf, text)
    print(f"{(path or pdf).name[-28:]:30} offset {offset:5d} (true {want:5d})  margin {margin:.2f}")
    assert offset == want
print("OK")
