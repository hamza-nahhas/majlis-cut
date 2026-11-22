"""Check of removing the page's paper (--no-paper) on a yellowed, noisy scan.

    python tests/test_paper.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut.book import remove_paper  # noqa: E402

rng = np.random.default_rng(1)
page = Image.new("RGB", (600, 400), (236, 222, 188))           # yellowed paper
d = ImageDraw.Draw(page)
for y in range(40, 360, 40):
    d.rectangle([60, y, 540, y + 8], fill=(30, 26, 22))        # lines of black text
d.rectangle([60, 10, 300, 22], fill=(170, 30, 30))             # a red heading
noisy = np.clip(np.asarray(page, np.int16) + rng.integers(-10, 11, (400, 600, 1)), 0, 255).astype(np.uint8)

out = np.asarray(remove_paper(Image.fromarray(noisy)))
alpha = out[..., 3]
assert alpha[30:38, 60:540].max() == 0, "paper between lines is clear"
assert alpha[380:, :].mean() < 2, "paper grain is gone"
assert alpha[42:46, 100:500].min() > 240, "black text stays solid"
red = out[14:18, 80:280]
assert red[..., 3].min() > 100 and (red[..., 0].astype(int) - red[..., 1]).min() > 80, "red stays red"
print("paper: all checks passed")
