"""Start a run through the interface's server and follow it: python tests/try_ui_run.py [port]"""
import json
import sys
import time
from pathlib import Path

import requests

base = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8765}"
form = json.loads((Path(__file__).resolve().parents[1] / "ui_state.json").read_text(encoding="utf-8"))
print("run:", requests.post(base + "/api/run", json=form).json())
seen = None
while True:
    job = requests.get(base + "/api/status").json()
    key = (job["state"], job["stage"], None if job["p"] is None else round(job["p"], 1))
    if key != seen:
        print(key)
        seen = key
    if job["state"] != "running":
        print("\n".join(job["log"][-4:]))
        break
    time.sleep(1)
