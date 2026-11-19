"""Correct a segment through the interface server and apply it: python tests/try_ui_edit.py <port>

The server must run with DARSMONTAGE_STATE pointing at a test state file.
"""
import sys
import time
from urllib.parse import quote

import requests

base = f"http://127.0.0.1:{sys.argv[1]}"
form = requests.get(base + "/api/state").json()["form"]
out = form["out"]


def run_and_wait():
    r = requests.post(base + "/api/run", json=form)
    assert r.ok, r.text
    while True:
        job = requests.get(base + "/api/status").json()
        if job["state"] != "running":
            assert job["state"] == "done", job
            return
        time.sleep(1)


def plan():
    return requests.get(base + "/api/result?out=" + quote(out)).json()


run_and_wait()
p = plan()
first_read = next(s for s in p["segments"] if s["kind"] == "read")
print("before:", len(p["segments"]), "segments; pending:", p["pending_edits"])
r = requests.post(base + "/api/edit", json={"out": out, "t0": first_read["t0"], "t1": first_read["t1"], "to": "talk"})
print("edit:", r.json(), "pending now:", plan()["pending_edits"])
run_and_wait()
p = plan()
row = next(s for s in p["segments"] if abs(s["t0"] - first_read["t0"]) < 0.1)
print("after apply:", row["kind"], "manual:", row["manual"], "pending:", p["pending_edits"])
assert row["kind"] == "talk" and row["manual"] and not p["pending_edits"]
requests.post(base + "/api/edit", json={"out": out, "t0": row["t0"], "t1": row["t1"], "undo": "one", "to": "read"})
run_and_wait()
p = plan()
row = next(s for s in p["segments"] if abs(s["t0"] - first_read["t0"]) < 0.1)
print("after undo:", row["kind"], "edits:", p["edits"])
assert row["kind"] == "read" and not p["edits"]
video = requests.get(base + "/api/file?path=" + quote(out + "\\review\\preview.mp4"), headers={"Range": "bytes=0-99"})
print("preview served:", video.status_code)
assert video.status_code == 206
print("OK")
