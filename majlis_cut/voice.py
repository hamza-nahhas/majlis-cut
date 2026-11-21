"""Tell the reader's voice from the commentator's, learned from the lesson itself.

The text match already labels most of the recording with certainty: long matched
passages are the reader, long unmatched stretches are the commentator. A small
classifier trained on those decides the doubtful spots (a verse the transcript
got differently, a sentence of the book quoted by the commentator).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.fft import dct

SR = 8000
N_FFT, HOP, N_MELS, N_CEPS = 256, 80, 24, 13
WIN, STEP = 150, 50          # frames per feature window (1.5 s) and step (0.5 s)


def load_audio(path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def _mel_bank() -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(mel(100), mel(SR / 2), N_MELS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(N_FFT, 1 / SR)
    bank = np.zeros((N_MELS, len(bins)))
    for m in range(N_MELS):
        lo, mid, hi = edges[m:m + 3]
        bank[m] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    return bank


_BANK = _mel_bank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def _frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cepstral coefficients (without the loudness term) and log energy per 10 ms frame."""
    if len(x) < N_FFT:
        return np.zeros((0, N_CEPS - 1)), np.zeros(0)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * _WINDOW
    power = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    logmel = np.log(power @ _BANK.T + 1e-10)
    return dct(logmel, type=2, axis=1, norm="ortho")[:, 1:N_CEPS], logmel.mean(axis=1)


class VoiceModel:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.w: np.ndarray | None = None
        self.accuracy = float("nan")

    def _windows(self, t0: float, t1: float) -> np.ndarray:
        ceps, energy = _frames(self.samples[int(t0 * SR):int(t1 * SR)])
        if len(ceps) < WIN // 2:
            return np.zeros((0, 2 * (N_CEPS - 1)))
        loud = energy > np.percentile(energy, 25)        # skip pauses
        out = []
        for a in range(0, max(1, len(ceps) - WIN + 1), STEP):
            sel = ceps[a:a + WIN][loud[a:a + WIN]]
            if len(sel) >= 40:
                out.append(np.concatenate([sel.mean(axis=0), sel.std(axis=0)]))
        return np.array(out).reshape(-1, 2 * (N_CEPS - 1))

    def fit(self, reader: list[tuple[float, float]], other: list[tuple[float, float]],
            budget_s: float = 900.0) -> bool:
        """reader / other: time ranges known to be each voice."""
        def collect(ranges):
            feats, used = [], 0.0
            for t0, t1 in sorted(ranges, key=lambda r: r[0] - r[1]):     # longest first
                if used >= budget_s:
                    break
                feats.append(self._windows(t0, t1))
                used += t1 - t0
            return [f for f in feats if len(f)]
        fr, fo = collect(reader), collect(other)
        if sum(map(len, fr)) < 60 or sum(map(len, fo)) < 60:
            return False

        def train(a, b):
            xa, xb = np.vstack(a), np.vstack(b)
            mu = np.vstack([xa, xb]).mean(axis=0)
            sd = np.vstack([xa, xb]).std(axis=0) + 1e-6
            za, zb = (xa - mu) / sd, (xb - mu) / sd
            sw = np.cov(za.T) + np.cov(zb.T) + 0.05 * np.eye(za.shape[1])
            w = np.linalg.solve(sw, za.mean(axis=0) - zb.mean(axis=0))
            pa, pb = za @ w, zb @ w
            mid, half = (pa.mean() + pb.mean()) / 2, (pa.mean() - pb.mean()) / 2
            return mu, sd, w / half, mid / half

        # Honest accuracy: train on every other range, test on the rest.
        if len(fr) >= 4 and len(fo) >= 4:
            mu, sd, w, mid = train(fr[::2], fo[::2])
            ta = ((np.vstack(fr[1::2]) - mu) / sd) @ w - mid
            tb = ((np.vstack(fo[1::2]) - mu) / sd) @ w - mid
            self.accuracy = float(((ta > 0).sum() + (tb < 0).sum()) / (len(ta) + len(tb)))
        self.mu, self.sd, self.w, self.mid = train(fr, fo)
        return True

    def reader_score(self, t0: float, t1: float) -> float | None:
        """> 0: sounds like the reader, < 0: like the commentator; about ±1 when typical."""
        if self.w is None:
            return None
        x = self._windows(t0, t1)
        if not len(x):
            return None
        return float(np.median(((x - self.mu) / self.sd) @ self.w - self.mid))
