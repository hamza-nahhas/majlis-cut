"""Place camera clips on the master audio by comparing their sound.

For each clip a few one-minute probes of its own audio are searched in the
master recording with normalised cross-correlation. The probes must agree, and
the difference between the first and the last one measures clock drift.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import fft

SR = 4000
PROBE_S = 60.0


def read_audio(path: Path, sr: int = SR) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, dtype=np.float32)
    return np.append(x[:1], x[1:] - 0.95 * x[:-1]).astype(np.float32)   # flatten the spectrum


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    num, den = info["streams"][0]["r_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "duration": float(info["format"]["duration"]),
            "width": info["streams"][0]["width"], "height": info["streams"][0]["height"]}


@dataclass
class Placement:
    path: str
    offset: float          # master time at which the clip's first frame falls
    duration: float
    fps: float
    confidence: float      # correlation of the best probe, 0..1
    drift_ms: float        # how far the clip's end slips against the master
    ok: bool


class Master:
    def __init__(self, path: Path):
        self.x = read_audio(path)
        self.n = fft.next_fast_len(len(self.x) + int(PROBE_S * SR), real=True)
        self.spectrum = fft.rfft(self.x, self.n)
        self.energy = np.concatenate(([0.0], np.cumsum(self.x.astype(np.float64) ** 2)))

    def locate(self, probe: np.ndarray) -> tuple[float, float]:
        """Best position (seconds) of the probe inside the master and its correlation."""
        m = len(probe)
        last = len(self.x) - m
        if last <= 0 or not probe.any():
            return 0.0, 0.0
        corr = fft.irfft(self.spectrum * np.conj(fft.rfft(probe, self.n)), self.n)[:last + 1]
        local = np.sqrt(np.maximum(self.energy[m:m + last + 1] - self.energy[:last + 1], 1e-9))
        ncc = corr / (local * float(np.linalg.norm(probe)) + 1e-9)
        # Ignore places where the master is almost silent: the ratio is meaningless there.
        ncc[local < 0.05 * np.median(local)] = 0.0
        k = int(np.argmax(ncc))
        return k / SR, float(ncc[k])


def place_clip(master: Master, path: Path, min_confidence: float = 0.12) -> Placement:
    info = probe_video(path)
    x = read_audio(path)
    m = int(min(PROBE_S, len(x) / SR * 0.8) * SR)
    starts = sorted({int(0.03 * len(x)), (len(x) - m) // 2, max(0, int(0.97 * len(x)) - m)})
    found = []
    for s in starts:
        at, score = master.locate(x[s:s + m])
        found.append((s / SR, at - s / SR, score))          # (position in clip, offset, score)
    best = max(found, key=lambda f: f[2])
    agree = [f for f in found if abs(f[1] - best[1]) < 0.25 and f[2] >= min_confidence]
    drift = (agree[-1][1] - agree[0][1]) * 1000 if len(agree) >= 2 else 0.0
    offset = float(np.median([f[1] for f in agree])) if agree else best[1]
    # One probe may fall outside the master (the clip starts before the recorder): two suffice.
    ok = best[2] >= min_confidence and (len(agree) >= 2 or len(found) == 1)
    return Placement(str(path), offset, info["duration"], info["fps"], best[2], drift, ok)


def place_all(master_audio: Path, clips: list[Path]) -> list[Placement]:
    master = Master(master_audio)
    return sorted((place_clip(master, c) for c in clips), key=lambda p: p.offset)


def camera_track(main: list[Placement], backup: list[Placement], duration: float,
                 trim: float = 0.5) -> list[dict]:
    """Clips for the camera track: the main camera, and the backup wherever it is missing."""
    track: list[dict] = []
    covered: list[tuple[float, float]] = []
    for p in (q for q in main if q.ok):
        t0, t1 = max(0.0, p.offset + trim), min(duration, p.offset + p.duration - trim)
        if covered and t0 < covered[-1][1]:
            t0 = covered[-1][1]
        if t1 - t0 > 0.2:
            track.append({"path": p.path, "t0": t0, "t1": t1, "in": t0 - p.offset,
                          "duration": p.duration, "camera": "main"})
            covered.append((t0, t1))
    edges = [0.0] + [x for c in covered for x in c] + [duration]
    for g0, g1 in zip(edges[::2], edges[1::2]):
        cursor = g0
        for p in (q for q in backup if q.ok):
            t0, t1 = max(cursor, p.offset), min(g1, p.offset + p.duration)
            if t1 - t0 > 0.2:
                track.append({"path": p.path, "t0": t0, "t1": t1, "in": t0 - p.offset,
                              "duration": p.duration, "camera": "backup"})
                cursor = t1
    return sorted(track, key=lambda c: c["t0"])


def save(placements: list[Placement], out: Path) -> None:
    out.write_text(json.dumps([asdict(p) for p in placements], ensure_ascii=False, indent=1),
                   encoding="utf-8")
