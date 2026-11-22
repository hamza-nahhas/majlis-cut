"""Outputs: a timeline DaVinci Resolve can import (FCP7 XML, works in the free
edition through File > Import > Timeline) and a quick low-resolution preview."""
from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path


def _rate(parent: ET.Element, fps: float) -> None:
    ntsc = abs(fps - round(fps)) > 0.005
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(round(fps))
    ET.SubElement(r, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _exact_fps(fps: float) -> Fraction:
    return Fraction(round(fps) * 1000, 1001) if abs(fps - round(fps)) > 0.005 else Fraction(round(fps))


class Timeline:
    """Collects clips as (track, source file, timeline start/end, source in) in seconds."""

    def __init__(self, name: str, fps: float, size: tuple[int, int] = (1920, 1080)):
        self.name, self.fps, self.size = name, fps, size
        self.video: dict[int, list[dict]] = {}
        self.audio: list[dict] = []

    def frames(self, seconds: float) -> int:
        return int(round(seconds * _exact_fps(self.fps)))

    def add_video(self, track: int, path: Path, t0: float, t1: float, src_in: float = 0.0,
                  still: bool = False, src_duration: float | None = None,
                  has_audio: bool = False, size: tuple[int, int] | None = None,
                  alpha: bool = False) -> None:
        """size: the file's own picture size when it differs from the timeline's.
        alpha: the file carries transparency (stills always may)."""
        self.video.setdefault(track, []).append(
            {"path": Path(path), "t0": t0, "t1": t1, "in": src_in, "still": still,
             "src_duration": src_duration, "has_audio": has_audio, "channels": 2,
             "size": size, "alpha": alpha})

    def add_audio(self, path: Path, t0: float, t1: float, src_in: float, src_duration: float,
                  channels: int = 2) -> None:
        self.audio.append({"path": Path(path), "t0": t0, "t1": t1, "in": src_in,
                           "src_duration": src_duration, "channels": channels})

    def write_fcp7_xml(self, out: Path) -> Path:
        root = ET.Element("xmeml", version="5")
        seq = ET.SubElement(root, "sequence")
        ET.SubElement(seq, "name").text = self.name
        end = max([c["t1"] for cs in self.video.values() for c in cs] + [c["t1"] for c in self.audio])
        ET.SubElement(seq, "duration").text = str(self.frames(end))
        _rate(seq, self.fps)
        tc = ET.SubElement(seq, "timecode")
        _rate(tc, self.fps)
        ET.SubElement(tc, "string").text = "00:00:00:00"
        ET.SubElement(tc, "frame").text = "0"
        ET.SubElement(tc, "displayformat").text = "NDF"
        media = ET.SubElement(seq, "media")
        files: dict[tuple[Path, str], str] = {}
        counter = [0]

        def file_el(parent: ET.Element, clip: dict, kind: str) -> None:
            path = clip["path"]
            # Picture and sound of one file get a definition each: Resolve did not find
            # the sound when its clips only pointed back at the picture's definition.
            key = (path, kind)
            if key in files:
                ET.SubElement(parent, "file", id=files[key])
                return
            files[key] = f"file-{len(files) + 1}"
            f = ET.SubElement(parent, "file", id=files[key])
            ET.SubElement(f, "name").text = path.name
            ET.SubElement(f, "pathurl").text = path.resolve().as_uri()
            _rate(f, self.fps)
            dur = clip["src_duration"] if clip.get("src_duration") else clip["t1"] - clip["t0"]
            ET.SubElement(f, "duration").text = str(self.frames(dur))
            m = ET.SubElement(f, "media")
            if kind == "video":
                v = ET.SubElement(m, "video")
                sc = ET.SubElement(v, "samplecharacteristics")
                w, h = clip.get("size") or self.size
                ET.SubElement(sc, "width").text = str(w)
                ET.SubElement(sc, "height").text = str(h)
                if clip.get("still"):
                    ET.SubElement(v, "stillframe").text = "TRUE"
            else:
                a = ET.SubElement(m, "audio")
                sc = ET.SubElement(a, "samplecharacteristics")
                ET.SubElement(sc, "depth").text = "16"
                ET.SubElement(sc, "samplerate").text = "48000"
                ET.SubElement(a, "channelcount").text = str(clip["channels"])

        def clip_el(track: ET.Element, clip: dict, kind: str, channel: int = 1) -> None:
            counter[0] += 1
            ci = ET.SubElement(track, "clipitem", id=f"clip-{counter[0]}")
            ET.SubElement(ci, "name").text = clip["path"].name
            ET.SubElement(ci, "enabled").text = "TRUE"
            start, stop = self.frames(clip["t0"]), self.frames(clip["t1"])
            src_in = self.frames(clip["in"])
            dur = clip["src_duration"] if clip.get("src_duration") else clip["t1"] - clip["t0"]
            ET.SubElement(ci, "duration").text = str(self.frames(dur))
            _rate(ci, self.fps)
            ET.SubElement(ci, "start").text = str(start)
            ET.SubElement(ci, "end").text = str(stop)
            ET.SubElement(ci, "in").text = str(src_in)
            ET.SubElement(ci, "out").text = str(src_in + stop - start)
            file_el(ci, clip, kind)
            if kind == "video" and clip.get("still"):
                ET.SubElement(ci, "stillframe").text = "TRUE"
            if kind == "video" and (clip.get("still") or clip.get("alpha")):
                ET.SubElement(ci, "alphatype").text = "straight"     # page pictures may be see-through
            if kind == "audio":
                st = ET.SubElement(ci, "sourcetrack")
                ET.SubElement(st, "mediatype").text = "audio"
                ET.SubElement(st, "trackindex").text = str(channel)

        video = ET.SubElement(media, "video")
        fmt = ET.SubElement(ET.SubElement(video, "format"), "samplecharacteristics")
        _rate(fmt, self.fps)
        ET.SubElement(fmt, "width").text = str(self.size[0])
        ET.SubElement(fmt, "height").text = str(self.size[1])
        ET.SubElement(fmt, "pixelaspectratio").text = "square"
        for n in range(1, max(self.video, default=0) + 1):
            track = ET.SubElement(video, "track")
            for clip in sorted(self.video.get(n, []), key=lambda c: c["t0"]):
                if self.frames(clip["t1"]) > self.frames(clip["t0"]):
                    clip_el(track, clip, "video")
        audio = ET.SubElement(media, "audio")
        ET.SubElement(audio, "numOutputChannels").text = "2"
        afmt = ET.SubElement(ET.SubElement(audio, "format"), "samplecharacteristics")
        ET.SubElement(afmt, "depth").text = "16"
        ET.SubElement(afmt, "samplerate").text = "48000"
        for channel in (1, 2):
            track = ET.SubElement(audio, "track")
            for clip in self.audio:
                if channel <= clip["channels"]:
                    clip_el(track, clip, "audio", channel)
        ET.indent(root)
        out.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
                       + ET.tostring(root, encoding="unicode"), encoding="utf-8")
        return out


def camera_placeholder(out: Path, size: tuple[int, int] = (1920, 1080)) -> Path:
    """Stand-in frame for the commentator's camera when no camera clips were given."""
    from PIL import Image, ImageDraw

    if out.exists():
        return out
    w, h = size
    img = Image.new("RGB", size, (28, 32, 38))
    d = ImageDraw.Draw(img)
    cx, cy, u = w // 2, h // 2, h // 12
    d.rounded_rectangle([cx - 2 * u, cy - u, cx + u, cy + u], radius=u // 4, fill=(90, 98, 110))
    d.polygon([(cx + u, cy - u // 3), (cx + 2 * u, cy - u), (cx + 2 * u, cy + u), (cx + u, cy + u // 3)],
              fill=(90, 98, 110))
    img.save(out)
    return out


def render_preview(views: list[dict], duration: float, audio: Path, placeholder: Path, out: Path,
                   fps: int = 5, height: int = 720, highlights: list[dict] | None = None) -> Path:
    """Low-resolution MP4 of the result: page views over reading, placeholder elsewhere.

    See-through page pictures are shown over the placeholder, as they would sit
    over the camera in Resolve.
    highlights: [{t0, t1, hl, ink}] the reading highlight of a view (a Highlight drawn
    at this preview's scale) and the ink picture laid over it.
    """
    from PIL import Image

    flat_dir = out.parent / "preview_frames"
    flat: dict[Path, Path] = {}

    def opaque(path: Path) -> Path:
        if path not in flat:
            img = Image.open(path)
            if img.mode != "RGBA":
                flat[path] = path
            else:
                flat_dir.mkdir(exist_ok=True)
                under = Image.open(placeholder).convert("RGBA").resize(img.size)
                flat[path] = flat_dir / f"{len(flat):05d}.png"
                Image.alpha_composite(under, img).convert("RGB").save(flat[path])
        return flat[path]

    shots: list[tuple[Path, float]] = []
    cursor = 0.0
    for v in views:
        if v["t0"] - cursor > 0.01:
            shots.append((placeholder, v["t0"] - cursor))
        shots.append((opaque(Path(v["file"])), v["t1"] - v["t0"]))
        cursor = v["t1"]
    if duration - cursor > 0.01:
        shots.append((placeholder, duration - cursor))
    if highlights:
        _preview_frames(shots, highlights, audio, out, fps, height)
        shutil.rmtree(flat_dir, ignore_errors=True)
        return out
    lines = []
    for path, dur in shots:
        lines.append(f"file '{path.resolve().as_posix()}'")
        lines.append(f"duration {dur:.3f}")
    lines.append(f"file '{shots[-1][0].resolve().as_posix()}'")
    listing = out.with_suffix(".txt")
    listing.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-i", str(audio), "-vf", f"fps={fps},scale=-2:{height}", "-c:v", "libx264",
         "-preset", "veryfast", "-tune", "stillimage", "-crf", "30", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "64k", "-shortest", str(out)], check=True)
    listing.unlink(missing_ok=True)
    shutil.rmtree(flat_dir, ignore_errors=True)
    return out


def _preview_frames(shots: list[tuple[Path, float]], highlights: list[dict], audio: Path, out: Path,
                    fps: int, height: int) -> None:
    """The preview drawn frame by frame, for the highlight to move in it."""
    import numpy as np
    from PIL import Image

    width = round(height * 16 / 9) // 2 * 2
    pictures: dict[Path, np.ndarray] = {}
    inks: dict[Path, np.ndarray] = {}

    def picture(path: Path) -> np.ndarray:
        if path not in pictures:
            img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
            pictures[path] = np.ascontiguousarray(np.asarray(img))
        return pictures[path]

    def ink(path: Path) -> np.ndarray:
        if path not in inks:
            img = Image.open(path).convert("RGBA").resize((width, height), Image.BILINEAR)
            inks[path] = np.asarray(img, np.float32) / 255
        return inks[path]

    starts = np.cumsum([0.0] + [d for _, d in shots])
    total = int(starts[-1] * fps)
    hls = sorted(highlights, key=lambda h: h["t0"])
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-i", str(audio), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    shot = hi = 0
    try:
        for i in range(total):
            t = i / fps
            while shot + 1 < len(shots) and starts[shot + 1] <= t:
                shot += 1
            while hi < len(hls) and hls[hi]["t1"] <= t:
                hi += 1
            base = picture(shots[shot][0])
            h = hls[hi] if hi < len(hls) and hls[hi]["t0"] <= t else None
            if h is None:
                proc.stdin.write(base.data)
                continue
            if not h.get("bound"):
                h["hl"].bind(base, ink(Path(h["ink"])) if h.get("ink") else None)
                h["bound"] = True
            frame = h["hl"].apply(base, t - h["t0"])
            proc.stdin.write((base if frame is None else frame).data)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
