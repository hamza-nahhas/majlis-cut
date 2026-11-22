"""Liquid-glass looks for the reading highlight.

The glass follows the ribbon's motion (`highlight.Highlight`), and instead of a
colour under the ink it lays a pane of glass over the page: a faint tint, a lit rim,
a light band along the top, a glint and a shadow, and along the rim the letters
beneath are bent a little, as glass bends them. The pane stretches and leans a
little as it moves.

Because it bends what is under it, the glass clip carries the page's own pixels
along the rim and sits over the page (no separate ink track). Everything else on the
pane is a see-through layer, so the clip composited over the page gives the result.
On pages without paper (ink on a clear ground over the camera) nothing is bent: the
camera under the page is not in the picture.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .highlight import Highlight

# The looks offered in the interface. Sizes are shares of the pane's height,
# except pad and tail (shares of the line height).
LOOKS = {
    # closed glass: rounded, thick, the rim bending the letters a little
    "glass": dict(roundness=0.42, bend=0.12, edge=0.22, split=0.15, pad=0.10, open_len=0.0, tail=0.0),
    # open glass: the right side (where the reading came from) fades out along a tail
    "glass-tail": dict(roundness=0.42, bend=0.07, edge=0.16, split=0.06, pad=0.18, open_len=2.6, tail=1.8),
}
SHADOW = np.array([18, 40, 70], np.float32)
_PANES: dict = {}
WHITE = np.array([255, 255, 255], np.float32)


class GlassPane:
    """The pane's look for one (width, height, lean), worked out once and reused.

    All the layers on the pane, and its shadow, reduce to one blend per pixel,
    out = in * keep + add. The rim pixels also keep where each colour samples the
    page from, so a frame only has to bend the rim and apply the blend."""

    def __init__(self, color, look: dict, step: int = 4):
        self.col = np.asarray(color, np.float32)
        self.look = look
        self.step = step
        self.cache: dict = {}

    def template(self, w: int, h: int, lean: float) -> dict:
        key = (w, h, lean)
        if key in self.cache:
            return self.cache[key]
        lk, col = self.look, self.col
        radius = min(0.5, lk["roundness"]) * h
        dy, soft, sa = 0.22 * h, 0.30 * h, 0.26
        m = int(dy + soft) + 4
        YY, XX = np.mgrid[0:h + 2 * m, 0:w + 2 * m].astype(np.float32)
        cx, cy, hw = m + w / 2, m + h / 2, w / 2
        XXs = XX + (YY - cy) / (h / 2) * lean * h * np.clip((cx - XX) / max(1.0, hw), 0, 1)

        def box(xx, yy):
            dx, dyy = xx - cx, yy - cy
            qx, qy = np.abs(dx) - (hw - radius), np.abs(dyy) - (h / 2 - radius)
            ox, oy = np.maximum(qx, 0), np.maximum(qy, 0)
            out = np.hypot(ox, oy)
            sdf = out + np.minimum(np.maximum(qx, qy), 0) - radius
            corner = out > 0          # the direction of the nearest edge, straight from the shape
            nx = np.where(corner, ox / (out + 1e-6), (qx >= qy).astype(np.float32)) * np.sign(dx)
            ny = np.where(corner, oy / (out + 1e-6), (qy > qx).astype(np.float32)) * np.sign(dyy)
            return sdf, nx, ny

        sdf, nx, ny = box(XXs, YY)
        if lk["open_len"]:            # open on the right: everything fades towards the right end
            u = np.clip(((m + w) - XX) / max(1.0, lk["open_len"] * h), 0, 1)
            fade = (u * u * (3 - 2 * u)).astype(np.float32)
        else:
            fade = np.ones_like(sdf)
        inside = np.clip(0.5 - sdf / 0.8, 0, 1)
        sh_sdf, _, _ = box(XXs, YY - dy)
        sh = np.clip(0.5 - sh_sdf / soft, 0, 1) ** 2 * (1 - inside) * sa * fade
        t = np.clip(1 + sdf / (lk["edge"] * h), 0, 1) * (sdf < 0)
        ring = np.nonzero(t > 0)
        bend = lk["bend"] * h * t[ring] ** 2.4 * fade[ring]
        taps = [(ring[0] + ny[ring] * bend * f, ring[1] + nx[ring] * bend * f)
                for f in (1 + lk["split"], 1.0, 1 - lk["split"])]
        keep = np.ones(sdf.shape, np.float32)
        add = np.zeros(sdf.shape + (3,), np.float32)

        def layer(a, colour):
            nonlocal keep, add
            a = (a * fade).astype(np.float32)
            add = add * (1 - a)[..., None] + np.asarray(colour, np.float32) * a[..., None]
            keep = keep * (1 - a)

        up = np.clip(-ny, 0, 1) ** 1.3
        layer(inside * 0.08, col)                                                      # body tint
        layer(t ** 3 * inside * 0.45, col * 0.55)                                      # rim in the colour
        layer(np.exp(-((sdf + 1.6) / 1.1) ** 2) * inside * (0.30 + 0.70 * up), WHITE)  # lit rim
        layer(np.clip(1 + sdf / (0.16 * h), 0, 1) * inside * up * 0.45, WHITE)         # thickness
        layer(np.exp(-(sdf / 0.8) ** 2) * 0.45, col * 0.55)                            # outline
        gl = np.exp(-(((XX - (m + w - 0.30 * h)) / (0.32 * h)) ** 2 + ((YY - (m + 0.20 * h)) / (0.07 * h)) ** 2))
        layer(gl * inside * 0.8, WHITE)                                                # glint
        # the whole look as one see-through layer over the page: the shadow, then the pane
        alpha = 1 - (1 - sh) * keep
        colour = (SHADOW * sh[..., None] * keep[..., None] + add) / np.maximum(alpha, 1e-4)[..., None]
        tpl = {"m": m, "shadow_keep": (1 - sh).astype(np.float32), "shadow_add": SHADOW * sh[..., None],
               "ring": ring, "taps": taps, "keep": keep, "add": add,
               "alpha": alpha.astype(np.float32), "colour": np.clip(colour, 0, 255).astype(np.float32)}
        if len(self.cache) > 600:
            self.cache.clear()
        self.cache[key] = tpl
        return tpl

    def patch(self, page: np.ndarray, opaque: np.ndarray | None, rect, strength: float, speed: float, scale: float,
              tint=None):
        """The glass over `page` (H x W x 3, 0..255) as a see-through patch.

        tint(y0, y1, x0, x1, pixels) colours a part of the page as it is seen under the
        glass (the line band), so only the part the glass covers is worked out.

        rect: the ribbon's box in full-size frame pixels; the page is at `scale`.
        Returns (y0, x0, alpha, colour) with page * (1 - alpha) + colour * alpha the
        glass seen over the page, or None when it is outside the picture."""
        H, W = page.shape[:2]
        lk = self.look
        x0, y0, x1, y1 = (v * scale for v in rect)
        lh = y1 - y0
        x0, x1 = x0 - lk["pad"] * lh, x1 + lk["pad"] * lh + lk["tail"] * lh
        s = round(min(1.0, speed / 900.0) * 5) / 5        # thinner and leaning a little as it moves (5 steps)
        y0, y1 = y0 - 0.04 * lh + 0.03 * s * lh, y1 + 0.04 * lh - 0.03 * s * lh
        w = max(self.step, int(round((x1 - x0) / self.step)) * self.step)
        h = max(4, int(round(y1 - y0)))
        tpl = self.template(w, h, round(0.10 * s, 2))
        m = tpl["m"]
        ox, oy = int(round(x1 - w)) - m, int(round(y0)) - m
        th, tw = tpl["keep"].shape
        a0, b0, a1, b1 = max(0, oy), max(0, ox), min(H, oy + th), min(W, ox + tw)
        if a0 >= a1 or b0 >= b1:
            return None
        ta, tb = a0 - oy, b0 - ox
        sl = (slice(ta, ta + a1 - a0), slice(tb, tb + b1 - b0))
        alpha = tpl["alpha"][sl]
        colour = tpl["colour"][sl]
        # the rim: the page bent, where the page is solid (paper or a background)
        ry, rx = tpl["ring"]
        if len(ry) and lk["bend"] > 0:
            region = np.zeros((th, tw, 3), np.float32)
            region[sl] = page[a0:a1, b0:b1] if tint is None else tint(a0, a1, b0, b1, page[a0:a1, b0:b1])
            under = region * tpl["shadow_keep"][..., None] + tpl["shadow_add"]
            bent = np.empty((len(ry), 3), np.float32)
            for k, (sy, sx) in enumerate(tpl["taps"]):
                bent[:, k] = map_coordinates(under[..., k], [sy, sx], order=1, mode="nearest")
            bent = bent * tpl["keep"][ry, rx, None] + tpl["add"][ry, rx]
            inside = (ry >= sl[0].start) & (ry < sl[0].stop) & (rx >= sl[1].start) & (rx < sl[1].stop)
            ly, lx = ry[inside] - ta, rx[inside] - tb
            if opaque is not None:
                solid = opaque[a0:a1, b0:b1][ly, lx] > 0.99
                ly, lx, bent = ly[solid], lx[solid], bent[inside][solid]
            else:
                bent = bent[inside]
            alpha = alpha.copy()
            colour = colour.copy()
            alpha[ly, lx] = 1.0
            colour[ly, lx] = np.clip(bent, 0, 255)
        return a0, b0, alpha * float(strength), colour


class Glass(Highlight):
    """The reading highlight as liquid glass over the page.

    bind() gives it the picture it sits over (the page as shown, at this highlight's
    scale) and the ink alone, so the line band can stay off the letters."""

    def __init__(self, events, duration, color, look: str = "glass",
                 size: tuple[int, int] = (1920, 1080), scale: float = 1.0):
        super().__init__(events, duration, color, size, scale)
        # the panes' shapes are shared by every view of the lesson: the same widths come back
        key = (look, tuple(int(c) for c in color))
        if key not in _PANES:
            _PANES[key] = GlassPane(color, LOOKS[look])
        self.pane = _PANES[key]
        self.page = None
        self.opaque = None
        self.ink = None

    def bind(self, page: np.ndarray, ink: np.ndarray | None = None) -> "Glass":
        """page: H x W x 3 or x 4 (0..255) at this highlight's size; ink: H x W x 4 (0..1)."""
        page = np.asarray(page, np.float32)
        self.opaque = page[..., 3] / 255 if page.shape[2] == 4 else None
        self.page = page[..., :3]
        self.ink = None if ink is None else np.asarray(ink, np.float32)[..., 3]
        return self

    def _state(self, t: float):
        prims = self.prims(t)
        bands = [p for p in prims if p[2] != 3.0]
        ribbon = next((p for p in prims if p[2] == 3.0), None)
        if ribbon is None:
            return bands, None
        before = next((p for p in self.prims(t - 0.04) if p[2] == 3.0), None)
        speed = abs(ribbon[0][0] - before[0][0]) / 0.04 if before else 0.0
        return bands, (ribbon[0], ribbon[5] / self.RIBBON_ALPHA, speed)

    def patches(self, t: float) -> list[tuple]:
        """See-through patches for time t: (y0, x0, alpha, colour). Each one is laid over
        the bare page; where they overlap, the later one already includes the earlier."""
        bands, glass = self._state(t)
        if getattr(self, "_band", None) is None:
            self._band, self._band_key, self._painted = np.zeros((self.h, self.w), np.float32), None, []
        band = self._band
        key = tuple((tuple(round(v, 1) for v in p[0]), round(p[5], 4)) for p in bands)
        if key != self._band_key:                 # the band is drawn again only when it changes
            for x0, y0, x1, y1 in self._painted:
                band[y0:y1, x0:x1] = 0
            self._painted = [z for z in (self._paint(band, *p) for p in bands) if z]
            self._band_key = key
        painted = self._painted
        out = []

        def band_at(y0, y1, x0, x1):
            ab = band[y0:y1, x0:x1]
            return ab * (1 - self.ink[y0:y1, x0:x1]) if self.ink is not None else ab

        for x0, y0, x1, y1 in painted:
            a = band_at(y0, y1, x0, x1)
            out.append((y0, x0, a, np.broadcast_to(self.color, a.shape + (3,))))
        if glass is not None and self.page is not None:
            rect, strength, speed = glass

            def tint(y0, y1, x0, x1, pixels):
                ab = band_at(y0, y1, x0, x1)[..., None]
                return pixels * (1 - ab) + self.color * ab

            p = self.pane.patch(self.page, self.opaque, rect, strength, speed, self.scale, tint if painted else None)
            if p is not None:
                y0, x0, alpha, colour = p
                hh, ww = alpha.shape
                if painted:                       # one layer over the bare page: band, then glass
                    ab = band_at(y0, y0 + hh, x0, x0 + ww)
                    total = 1 - (1 - ab) * (1 - alpha)
                    colour = (self.color * (ab * (1 - alpha))[..., None] + colour * alpha[..., None])                         / np.maximum(total, 1e-4)[..., None]
                    alpha = total
                out.append((y0, x0, alpha, colour))
        return out

    def frames(self, times):
        buf = np.zeros((self.h, self.w, 4), np.uint8)
        dirty: list[tuple[int, int, int, int]] = []
        last = None
        for t in times:
            bands, glass = self._state(t)
            key = (tuple((tuple(round(v, 1) for v in p[0]), round(p[5], 3)) for p in bands),
                   None if glass is None else (tuple(round(v, 1) for v in glass[0]), round(glass[1], 3),
                                               round(min(1.0, glass[2] / 900.0) * 5)))
            if key != last:
                for y0, x0, y1, x1 in dirty:
                    buf[y0:y1, x0:x1] = 0
                dirty = []
                for y0, x0, alpha, colour in self.patches(t):
                    hh, ww = alpha.shape
                    buf[y0:y0 + hh, x0:x0 + ww, :3] = (colour + 0.5).astype(np.uint8)
                    buf[y0:y0 + hh, x0:x0 + ww, 3] = (alpha * 255 + 0.5).astype(np.uint8)
                    dirty.append((y0, x0, y0 + hh, x0 + ww))
                last = key
            yield buf.data

    def frame(self, t: float) -> np.ndarray:
        out = np.zeros((self.h, self.w, 4), np.uint8)
        for y0, x0, alpha, colour in self.patches(t):
            hh, ww = alpha.shape
            out[y0:y0 + hh, x0:x0 + ww, :3] = (colour + 0.5).astype(np.uint8)
            out[y0:y0 + hh, x0:x0 + ww, 3] = (alpha * 255 + 0.5).astype(np.uint8)
        return out

    def apply(self, picture: np.ndarray, t: float) -> np.ndarray | None:
        """The picture (H x W x 3 uint8, the page as shown) with the glass at time t."""
        got = self.patches(t)
        if not got:
            return None
        out = picture.copy()
        for y0, x0, alpha, colour in got:         # each over the bare picture, as in the clip
            hh, ww = alpha.shape
            reg = picture[y0:y0 + hh, x0:x0 + ww].astype(np.float32)
            mixed = reg * (1 - alpha[..., None]) + colour * alpha[..., None]
            out[y0:y0 + hh, x0:x0 + ww] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
        return out
