"""Speech-to-text with word timestamps.

Every backend returns the same structure and it is cached as JSON in the lesson
folder, so the (slow or paid) transcription runs once per lesson:

    {"audio": path, "duration": seconds, "backend": name,
     "words": [{"t0": s, "t1": s, "w": "كلمة", "spk": "A" | None}, ...]}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

Progress = Callable[[float], None]
CHUNK_S = 300.0


def audio_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _quiet_point(samples, around: float, sr: int = 16000, reach: float = 15.0) -> float:
    """The quietest half second near `around`: a safe place to cut between two chunks."""
    import numpy as np

    a, b = int((around - reach) * sr), int((around + reach) * sr)
    step = sr // 10
    power = np.square(samples[a:b][: (b - a) // step * step]).reshape(-1, step).mean(axis=1)
    smooth = np.convolve(power, np.ones(5) / 5, mode="valid")
    return around - reach + (int(smooth.argmin()) + 2.5) * 0.1


def transcribe_whisper(audio: Path, out_json: Path, model: str = "large-v3-turbo",
                       threads: int = 8, beam: int = 2, progress: Progress | None = None) -> dict:
    """Local transcription with faster-whisper on CPU: free, takes about 40% of the lesson's length.

    Works in five-minute chunks and writes every word as soon as it has it, so a run
    that was stopped picks up where it left off.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    # Decode with ffmpeg ourselves: independent of the PyAV version installed.
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    samples = np.frombuffer(pcm, dtype=np.float32)
    duration = len(samples) / 16000
    partial = out_json.with_suffix(".partial.jsonl")
    words: list[dict] = []
    if partial.exists():
        words = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    pos = words[-1]["t1"] if words else 0.0
    wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
    with partial.open("a", encoding="utf-8") as fh:
        while pos < duration - 1.0:
            end = duration if duration - pos < CHUNK_S + 30 else _quiet_point(samples, pos + CHUNK_S)
            segments, _info = wm.transcribe(
                samples[int(pos * 16000):int(end * 16000)], language="ar", beam_size=beam,
                word_timestamps=True, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 700},
                condition_on_previous_text=False,
                # One retry only: on noise or music the default six retries per
                # half minute can stall a run for a very long time.
                temperature=(0.0, 0.4),
            )
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if not text:
                        continue
                    rec = {"t0": round(pos + w.start, 3), "t1": round(pos + w.end, 3),
                           "w": text, "spk": None}
                    words.append(rec)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if progress:
                    progress(min(1.0, (pos + seg.end) / duration))
            pos = end
            if progress:
                progress(min(1.0, pos / duration))
    result = {"audio": str(audio), "duration": duration, "backend": f"faster-whisper:{model}",
              "words": words}
    out_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return result


def transcribe_elevenlabs(audio: Path, out_json: Path, api_key: str | None = None,
                          progress: Progress | None = None) -> dict:
    """Cloud transcription with ElevenLabs Scribe: word timestamps and speaker labels."""
    import requests

    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    with audio.open("rb") as fh:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": api_key},
            data={"model_id": os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1"),
                  "language_code": "ara", "diarize": "true", "num_speakers": "2",
                  "timestamps_granularity": "word", "tag_audio_events": "false"},
            files={"file": (audio.name, fh)},
            timeout=3600,
        )
    resp.raise_for_status()
    data = resp.json()
    words = [
        {"t0": round(w["start"], 3), "t1": round(w["end"], 3), "w": w["text"].strip(),
         "spk": w.get("speaker_id")}
        for w in data.get("words", [])
        if w.get("type", "word") == "word" and w.get("text", "").strip()
    ]
    result = {"audio": str(audio), "duration": audio_duration(audio),
              "backend": "elevenlabs", "words": words}
    out_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def transcribe(audio: Path, out_json: Path, backend: str = "whisper",
               progress: Progress | None = None) -> dict:
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if backend == "whisper":
        return transcribe_whisper(audio, out_json, progress=progress)
    if backend == "elevenlabs":
        return transcribe_elevenlabs(audio, out_json, progress=progress)
    raise ValueError(f"unknown ASR backend: {backend}")


if __name__ == "__main__":
    transcribe(Path(sys.argv[1]), Path(sys.argv[2]),
               backend=sys.argv[3] if len(sys.argv) > 3 else "whisper",
               progress=lambda f: print(f"[asr] {f:.1%}", file=sys.stderr, flush=True))
