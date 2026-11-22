"""Graphical interface: a local web page served from this machine only.

    python -m majlis_cut.ui        (or double-click start.bat)

The page collects the lesson's files, runs the same command-line tool in a
separate process and shows its progress and its result.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import fitz

from . import __version__, update
from .__main__ import STAGES
from .arabic import normalize
from .book import (build_book, guess_page_offset, load_turath_pages, ocr_cache, read_pdf_text,
                   render_view, turath_catalog, turath_volumes, view_frame)
from .highlight import DEFAULT_COLOR, parse_color

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
STATE_FILE = Path(os.environ.get("MAJLIS_CUT_STATE") or ROOT / "ui_state.json")   # tests use their own
PAGE = Path(__file__).with_name("ui.html")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PICKER = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
kind, start = sys.argv[1], sys.argv[2] or None
types = {"audio": [("Audio", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma *.mp4 *.mov"), ("All files", "*.*")],
         "pdf": [("PDF", "*.pdf")],
         "image": [("Image", "*.png *.jpg *.jpeg")],
         "videos": [("Video", "*.mp4 *.mov *.mts *.m2ts *.mxf *.avi *.mkv"), ("All files", "*.*")]}
if kind == "folder":
    picked = [filedialog.askdirectory(initialdir=start)]
elif kind == "videos":
    picked = list(filedialog.askopenfilenames(initialdir=start, filetypes=types[kind]))
else:
    picked = [filedialog.askopenfilename(initialdir=start, filetypes=types[kind])]
sys.stdout.buffer.write("\n".join(p for p in picked if p).encode("utf-8"))
"""


def page_as_read(pdf: Path, page: int) -> str:
    """The words read off one PDF page, line under line."""
    from .ocr import available
    if available():
        raise RuntimeError(available())
    lines: dict[int, list[str]] = {}
    for w in read_pdf_text(pdf, CACHE, page, page).get(page, []):
        lines.setdefault(w.line, []).append(w.text)
    return "\n".join(" ".join(ws) for ws in lines.values())


def sample_highlight(book, idx: int, shot: Path, tmp: Path, color, look: dict) -> Path:
    """The sample page with the reading highlight resting on one of its words."""
    import numpy as np
    from PIL import Image

    from .highlight import sample
    from .ocr import available, ocr_pages

    words = [] if available() else ocr_pages(book.pdf_path, [idx], ocr_cache(book.pdf_path, CACHE),
                                             boxes=True)[idx]
    frame = view_frame(book, idx, 0, scale=look["scale"], pos=look["pos"])
    hl = sample(book, idx, 0, frame, words, color)
    if hl is None:
        return shot
    page = Image.open(shot).convert("RGBA")
    ink = Image.open(render_view(book, idx, 0, tmp, ink_only=True, **look)).convert("RGBA")
    out = Image.alpha_composite(Image.alpha_composite(page, Image.fromarray(np.ascontiguousarray(hl))), ink)
    path = tmp / "sample_highlight.png"
    out.save(path)
    return path


class Job:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.state = "idle"          # idle | running | done | failed | cancelled
        self.stage = -1
        self.p: float | None = None
        self.out = ""
        self.has_cameras = False
        self.error = ""
        self.log: deque[str] = deque(maxlen=60)

    def status(self) -> dict:
        return {"state": self.state, "stage": self.stage, "p": self.p, "out": self.out,
                "error": self.error,
                "has_cameras": self.has_cameras, "stages": STAGES, "log": list(self.log)[-12:]}

    def start(self, cfg: dict) -> None:
        cmd = [sys.executable, "-m", "majlis_cut", "--json-progress",
               "--audio", cfg["audio"], "--pdf", cfg["pdf"], "--out", cfg["out"],
               "--window", str(float(cfg.get("window") or 0.55)),
               "--overlay-size", str(float(cfg.get("overlay_size") or 1.0)),
               "--overlay-pos", cfg.get("overlay_pos") or "center",
               "--asr", cfg.get("asr") or "whisper", "--cache", str(CACHE)]
        if cfg.get("text_source") != "pdf":       # otherwise the text is read off the PDF
            cmd += ["--turath", str(cfg["book_id"]), "--vol", str(cfg["vol"]),
                    "--page-offset", str(int(cfg.get("page_offset") or 0))]
        if cfg.get("pages"):
            cmd += ["--pages", cfg["pages"]]
        if cfg.get("background"):
            cmd += ["--background", cfg["background"]]
        if cfg.get("fill_with_book"):
            cmd += ["--fill-with-book"]
        if cfg.get("no_paper"):
            cmd += ["--no-paper"]
        if cfg.get("highlight"):
            cmd += ["--highlight", "--highlight-color", cfg.get("highlight_color") or DEFAULT_COLOR]
        for flag, key in (("--cam-main", "cam_main"), ("--cam-backup", "cam_backup")):
            paths = [p for p in cfg.get(key) or [] if p]
            if paths:
                cmd += [flag, *paths]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if cfg.get("elevenlabs_key"):
            env["ELEVENLABS_API_KEY"] = cfg["elevenlabs_key"]
        self.state, self.stage, self.p, self.out = "running", -1, None, cfg["out"]
        self.has_cameras = bool(cfg.get("cam_main") or cfg.get("cam_backup"))
        self.log.clear()
        self.error = ""
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", creationflags=NO_WINDOW)
        threading.Thread(target=self._read_progress, daemon=True).start()
        threading.Thread(target=self._read_log, daemon=True).start()

    def _read_progress(self) -> None:
        proc = self.proc
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "stage" in msg:
                self.stage, self.p = msg["stage"], msg.get("p")
            if "error" in msg:
                self.error = msg["error"]
        code = proc.wait()
        if self.proc is proc and self.state == "running":
            self.state = "done" if code == 0 else "failed"

    def _read_log(self) -> None:
        for line in self.proc.stderr:
            if line.strip():
                self.log.append(line.rstrip())

    def cancel(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.state = "cancelled"
            self.proc.terminate()


JOB = Job()
_catalog: list[dict] | None = None


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(cfg: dict) -> None:
    cfg = {k: v for k, v in cfg.items() if k != "elevenlabs_key"}      # never store the key
    STATE_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")


def search_books(q: str) -> list[dict]:
    global _catalog
    if _catalog is None:
        _catalog = [dict(b, key=" ".join(normalize(b["name"] + " " + b["author"]).split()))
                    for b in turath_catalog(CACHE)]
    terms = normalize(q).split()
    if not terms:
        return []
    hits = [b for b in _catalog if all(t in b["key"] for t in terms)]
    hits.sort(key=lambda b: len(b["name"]))
    return [{"id": b["id"], "name": b["name"], "author": b["author"]} for b in hits[:40]]


def pick(kind: str, start: str) -> list[str]:
    start_dir = start if start and Path(start).is_dir() else (str(Path(start).parent) if start else "")
    out = subprocess.run([sys.executable, "-c", PICKER, kind, start_dir], capture_output=True,
                         creationflags=NO_WINDOW).stdout.decode("utf-8")
    return [str(Path(p)) for p in out.splitlines() if p]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:      # keep the console quiet
        pass

    def _json(self, data, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        """Serve a file with byte ranges, so the browser can seek in the preview video."""
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(size - 1, int(b)) if a and b else size - 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionError, OSError):
                    return
                left -= len(chunk)

    def _inside_output(self, path: Path) -> bool:
        out = JOB.out or load_state().get("out") or ""
        return bool(out) and path.resolve().is_relative_to(Path(out).resolve())

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self._bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._json({"form": load_state(), "job": JOB.status()})
            elif url.path == "/favicon.png":
                self._bytes(PAGE.with_name("icon.png").read_bytes(), "image/png")
            elif url.path == "/api/version":
                self._json(update.check(force=q.get("force") == "1") if q.get("check") else
                           {"current": __version__})
            elif url.path == "/api/status":
                self._json(JOB.status())
            elif url.path == "/api/books":
                self._json(search_books(q.get("q", "")))
            elif url.path == "/api/book":
                self._json({"vols": turath_volumes(int(q["id"]), CACHE)})
            elif url.path == "/api/offset":
                text = load_turath_pages(int(q["id"]), q["vol"], CACHE)
                offset, margin = guess_page_offset(Path(q["pdf"]), text)
                middle = fitz.open(q["pdf"]).page_count // 2 + 1 - offset     # printed page shown
                self._json({"offset": offset, "sure": margin > 0.15, "sample": middle})
            elif url.path == "/api/pagetext":
                page = int(q["page"])
                if q.get("id"):
                    self._json({"text": load_turath_pages(int(q["id"]), q["vol"], CACHE).get(page, "")})
                else:
                    self._json({"text": page_as_read(Path(q["pdf"]), page)})
            elif url.path == "/api/pageimg":
                doc = fitz.open(q["pdf"])
                idx = int(q["page"]) + int(q.get("offset", 0)) - 1
                if not 0 <= idx < doc.page_count:
                    self._json({"error": "no such page"}, 404)
                else:
                    self._bytes(doc[idx].get_pixmap(dpi=80).tobytes("png"), "image/png")
            elif url.path == "/api/sample":
                # One page drawn with the chosen look, to judge it before running.
                page, offset = int(q["page"]), int(q.get("offset", 0))
                if q.get("id"):
                    text = load_turath_pages(int(q["id"]), q["vol"], CACHE)
                else:
                    text = read_pdf_text(Path(q["pdf"]), CACHE, page, page)
                book = build_book(Path(q["pdf"]), text, offset, page, page, float(q.get("window", 0.55)))
                idx = page + offset - 1
                if idx not in book.layouts:
                    self._json({"error": "no such page"}, 404)
                    return
                background = Path(q["background"]) if q.get("background") else None
                tmp = Path(tempfile.mkdtemp(prefix="majlis_cut_"))
                try:
                    look = {"scale": float(q.get("size", 1)), "pos": q.get("pos", "center")}
                    shot = render_view(book, idx, 0, tmp, background=background,
                                       paper=q.get("paper") != "0", **look)
                    if q.get("hl"):
                        shot = sample_highlight(book, idx, shot, tmp, parse_color(q["hl"]), look)
                    self._bytes(shot.read_bytes(), "image/png")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            elif url.path == "/api/result":
                out = Path(q.get("out") or JOB.out)
                plan = out / "work" / "plan.json"
                data = json.loads(plan.read_text(encoding="utf-8")) if plan.exists() else {}
                if data:
                    edits = out / "work" / "edits.json"
                    now = json.loads(edits.read_text(encoding="utf-8")) if edits.exists() else []
                    data["pending_edits"] = now != data.get("edits", [])
                self._json(data)
            elif url.path == "/api/file":
                path = Path(q["path"])
                if path.is_file() and self._inside_output(path):
                    self._file(path)
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "unknown"}, 404)
        except Exception as exc:  # noqa: BLE001 - report every failure to the page
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        try:
            if url.path == "/api/pick":
                picked = pick(body["kind"], body.get("start", ""))
                extra = {}
                if picked and body["kind"] == "audio":
                    audio = Path(picked[0])
                    extra["suggest_out"] = str(audio.parent / f"{audio.stem}_montage")
                self._json({"paths": picked, **extra})
            elif url.path == "/api/state":
                save_state(body)
                self._json({"ok": True})
            elif url.path == "/api/run":
                if JOB.state == "running":
                    self._json({"error": "يوجد تشغيل جارٍ"}, 409)
                    return
                need = ("audio", "pdf", "out") if body.get("text_source") == "pdf" \
                    else ("audio", "pdf", "out", "book_id", "vol")
                missing = [k for k in need if not body.get(k)]
                for key in ("audio", "pdf"):
                    if body.get(key) and not Path(body[key]).is_file():
                        self._json({"error": f"الملف غير موجود: {body[key]}"}, 400)
                        return
                if missing:
                    self._json({"error": "حقول ناقصة: " + "، ".join(missing)}, 400)
                    return
                save_state(body)
                JOB.start(body)
                self._json({"ok": True})
            elif url.path == "/api/edit":
                # Flip one segment between book and camera; flipping it back removes the edit.
                out = Path(body["out"])
                if not self._inside_output(out / "work"):
                    self._json({"error": "not allowed"}, 403)
                    return
                path = out / "work" / "edits.json"
                edits = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
                if body.get("undo") == "all":
                    edits = []
                elif body.get("undo"):
                    edits = [e for e in edits if not (e["t0"] < body["t1"] - 0.05 and e["t1"] > body["t0"] + 0.05)]
                else:
                    edits.append({"t0": body["t0"], "t1": body["t1"], "to": body["to"]})
                path.write_text(json.dumps(edits), encoding="utf-8")
                self._json({"count": len(edits)})
            elif url.path == "/api/update":
                if JOB.state == "running":
                    self._json({"error": "انتظر حتى ينتهي التشغيل الجاري ثم حدّث."}, 409)
                    return
                update.apply(self.server.server_address[1])
                self._json({"ok": True})
                restart_after_update(self.server)
            elif url.path == "/api/cancel":
                JOB.cancel()
                self._json({"ok": True})
            elif url.path == "/api/open":
                path = Path(body["path"])
                if path.exists() and self._inside_output(path):
                    os.startfile(path)                      # noqa: S606 - opens the output folder
                self._json({"ok": True})
            else:
                self._json({"error": "unknown"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)


class Server(ThreadingHTTPServer):
    # Windows would otherwise let two copies share one port, and requests meant
    # for one would land on the other.
    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


RESTARTING = threading.Event()


def restart_after_update(server: Server) -> None:
    """Close the interface so start.bat opens the new version (see update.py)."""
    RESTARTING.set()
    threading.Thread(target=server.shutdown, daemon=True).start()


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
    again = update.restart_port()            # reopening after an update: same page, same port
    if again:
        port = again
    while True:
        try:
            server = Server(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"majlis_cut {__version__}: {url}   (close this window to stop)")
    if "--no-browser" not in sys.argv and port != again:   # the old page reloads itself
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        JOB.cancel()
    server.server_close()
    if RESTARTING.is_set():
        print("Updated. Opening the new version...")
        sys.exit(update.RESTART_CODE)


if __name__ == "__main__":
    main()
