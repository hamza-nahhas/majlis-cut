"""Voice score of every segment of a finished run: python tests/try_voice.py <audio> <out_dir>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darsmontage.voice import VoiceModel, load_audio  # noqa: E402

audio, out = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
vm = VoiceModel(load_audio(audio))
segs = plan["segments"]
ok = vm.fit([(s["t0"] + 1, s["t1"] - 1) for s in segs if s["kind"] == "read" and s["t1"] - s["t0"] >= 8],
            [(s["t0"] + 3, s["t1"] - 3) for s in segs if s["kind"] == "talk" and s["t1"] - s["t0"] >= 15])
print("fit", ok, "accuracy", vm.accuracy)
for s in segs:
    whole = vm.reader_score(s["t0"], s["t1"])
    parts = [vm.reader_score(a, min(a + 3, s["t1"])) for a in
             [s["t0"] + 3 * k for k in range(int((s["t1"] - s["t0"]) // 3) + 1)]][:14]
    print(f"{s['t0']:7.1f}-{s['t1']:7.1f} {s['kind']:4} score {whole if whole is None else round(whole, 2)}   "
          + " ".join("  . " if p is None else f"{p:+.1f}" for p in parts))
