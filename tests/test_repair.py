"""The skipped recitation at 4:12 of the sample lesson must come back after gap repair.

    python tests/test_repair.py <audio> <asr.json made before gap repair existed>
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.asr import repair_gaps  # noqa: E402

audio, cached = sys.argv[1], Path(sys.argv[2])
words = [w for w in json.loads(cached.read_text(encoding="utf-8"))["words"] if w["t1"] < 360]
pcm = subprocess.run(["ffmpeg", "-v", "error", "-t", "360", "-i", audio, "-ac", "1", "-ar", "16000",
                      "-f", "f32le", "-"], capture_output=True, check=True).stdout
samples = np.frombuffer(pcm, dtype=np.float32)
wm = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=8)
fixed, repaired = repair_gaps(wm, samples, words, beam=2)
added = [w for w in fixed if 252 < w["t0"] < 266]
print(f"gaps repaired: {repaired}; words now between 4:12 and 4:26: {len(added)}")
print(" ".join(w["w"] for w in added))
assert repaired >= 1 and len(added) >= 15
assert all(a["t0"] <= b["t0"] for a, b in zip(fixed, fixed[1:])), "words out of order"
print("OK")
