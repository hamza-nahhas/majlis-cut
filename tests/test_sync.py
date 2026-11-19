"""Sync check on fake camera clips cut from a real recording.

    python tests/test_sync.py <master_audio> <work_dir>

Each fake clip is a stretch of the master, band-limited, with echo, pink noise and a
30 ppm clock error, on a tiny black video: roughly what a camera microphone hears.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.sync import camera_track, place_all  # noqa: E402

master, work = Path(sys.argv[1]), Path(sys.argv[2]) / "fake_cams"
work.mkdir(parents=True, exist_ok=True)
CASES = {"main_1.mp4": (120.0, 600.0), "main_2.mp4": (735.0, 600.0), "backup_1.mp4": (700.0, 60.0)}
for name, (start, length) in CASES.items():
    out = work / name
    if out.exists():
        continue
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(master),
         "-f", "lavfi", "-i", f"anoisesrc=c=pink:a=0.03:d={length}",
         "-f", "lavfi", "-i", f"color=c=black:s=320x180:r=25:d={length}",
         "-filter_complex",
         "[0:a]aresample=48000,asetrate=48000*1.00003,aresample=48000,highpass=f=200,lowpass=f=3000,"
         "aecho=0.8:0.7:40:0.3,volume=0.6[a0];[a0][1:a]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "2:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(out)], check=True)

placed = place_all(master, [work / n for n in CASES])
worst = 0.0
for p in placed:
    want = CASES[Path(p.path).name][0]
    worst = max(worst, abs(p.offset - want))
    print(f"{Path(p.path).name:14} offset {p.offset:9.3f} s (true {want:7.1f})  error {1000 * (p.offset - want):+6.1f} ms  "
          f"confidence {p.confidence:.2f}  drift {p.drift_ms:+.1f} ms  ok={p.ok}")
main = [p for p in placed if "main" in p.path]
backup = [p for p in placed if "backup" in p.path]
for c in camera_track(main, backup, 1500.0):
    print(f"   {c['t0']:8.2f} - {c['t1']:8.2f}  {c['camera']:6}  {Path(c['path']).name}  in {c['in']:.2f}")
assert all(p.ok for p in placed) and worst < 0.04, f"sync error {worst * 1000:.0f} ms"
print("OK: all clips within one frame")
