"""Does Whisper transcribe a stretch it skipped, when given that stretch alone?

    python tests/try_gap.py <audio> <from_s> <to_s>
"""
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

audio, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
pcm = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(a), "-t", str(b - a), "-i", audio, "-ac", "1",
                      "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
x = np.frombuffer(pcm, dtype=np.float32)
wm = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=8)
for label, kw in [("with VAD", {"vad_filter": True}), ("no VAD", {"vad_filter": False}),
                  ("no VAD, keep low-confidence", {"vad_filter": False, "no_speech_threshold": None,
                                                   "log_prob_threshold": None})]:
    segs, _ = wm.transcribe(x, language="ar", beam_size=2, temperature=0.0, word_timestamps=True,
                            condition_on_previous_text=False, **kw)
    print(f"--- {label}")
    for s in segs:
        print(f"  {a + s.start:7.1f}-{a + s.end:7.1f}  no_speech={s.no_speech_prob:.2f} logprob={s.avg_logprob:.2f}  {s.text}")
