"""Build a short test case from a transcription still in progress.

    python tests/make_partial_case.py <audio> <partial.jsonl> <case_dir>
Writes <case_dir>/part.mp3 and <case_dir>/out/asr_whisper.json covering the words done so far.
"""
import json
import subprocess
import sys
from pathlib import Path

audio, partial, case = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
words = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines() if line.strip()]
end = words[-1]["t1"] + 0.5
(case / "out").mkdir(parents=True, exist_ok=True)
part = case / "part.mp3"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-t", f"{end:.2f}", "-c", "copy", str(part)],
               check=True)
(case / "out" / "asr_whisper.json").write_text(
    json.dumps({"audio": str(part), "duration": end, "backend": "partial", "words": words}, ensure_ascii=False),
    encoding="utf-8")
print(f"{len(words)} words, {end / 60:.1f} min")
