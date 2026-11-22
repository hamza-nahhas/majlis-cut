"""Draw the tool's icon (majlis_cut/icon.ico): an open book on the interface's teal.

    python tools/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=52, fill=(15, 122, 108))
# two pages of an open book
cx, top, bottom = S // 2, 70, 196
d.polygon([(cx, top + 14), (44, top), (44, bottom - 14), (cx, bottom)], fill=(251, 239, 216))
d.polygon([(cx, top + 14), (S - 44, top), (S - 44, bottom - 14), (cx, bottom)], fill=(255, 255, 255))
for k in range(4):                        # lines of text
    y = top + 34 + k * 24
    d.line([(64, y - 4), (cx - 16, y + 4)], fill=(154, 91, 6), width=7)
    d.line([(cx + 16, y + 4), (S - 64, y - 4)], fill=(27, 36, 34), width=7)
d.line([(cx, top + 14), (cx, bottom)], fill=(15, 122, 108), width=5)
out = Path(__file__).resolve().parents[1] / "majlis_cut" / "icon.ico"
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(out.with_suffix(".png"))
print(out)
