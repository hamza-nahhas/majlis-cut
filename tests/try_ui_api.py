"""Exercise the interface's server: python tests/try_ui_api.py <pdf> <out_dir> [port]"""
import sys
from urllib.parse import quote

import requests

pdf, out = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{sys.argv[3] if len(sys.argv) > 3 else 8765}"


def get(path):
    r = requests.get(base + path, timeout=120)
    return r


print("page:", get("/").status_code, len(get("/").content), "bytes")
print("state:", get("/api/state").json()["job"]["state"])
hits = get("/api/books?q=" + quote("زاد المعاد")).json()
print("search:", [(b["id"], b["name"]) for b in hits][:3])
print("vols:", get("/api/book?id=197").json()["vols"][:4])
print("offset:", get(f"/api/offset?pdf={quote(pdf)}&id=197&vol=3").json())
img = get(f"/api/pageimg?pdf={quote(pdf)}&page=31&offset=0")
print("pageimg:", img.status_code, img.headers["Content-Type"], len(img.content))
print("pagetext:", get("/api/pagetext?id=197&vol=3&page=31").json()["text"][:60])
print("save:", requests.post(base + "/api/state", json={"out": out, "pdf": pdf}).json())
plan = get("/api/result?out=" + quote(out)).json()
print("result:", {k: (len(v) if isinstance(v, list) else v) for k, v in plan.items()})
video = requests.get(base + "/api/file?path=" + quote(out + "\\preview.mp4"), headers={"Range": "bytes=100-299"})
print("video range:", video.status_code, video.headers.get("Content-Range"), len(video.content))
blocked = get("/api/file?path=" + quote("C:\\Windows\\win.ini"))
print("outside file blocked:", blocked.status_code)
bad = requests.post(base + "/api/run", json={"audio": "x"})
print("bad run:", bad.status_code, bad.json())
