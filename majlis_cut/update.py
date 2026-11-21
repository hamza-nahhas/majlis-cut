"""Updating the tool from its GitHub releases.

The interface asks GitHub for the latest release when it opens. If it is newer
than this copy, the page offers it; one click downloads the release zip and puts
its files in place of the old ones. Nothing the user made is touched: the Python
environment (.venv), the cache, the saved form (ui_state.json) and every lesson
folder stay as they are. A copy made with `git clone` is updated with git instead.

After the files are replaced the interface closes with RESTART_CODE; start.bat
then refreshes the libraries if requirements.txt changed and opens it again.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import requests

from . import __version__

REPO = "hamza-nahhas/majlis-cut"
API = f"https://api.github.com/repos/{REPO}/releases/latest"
ROOT = Path(__file__).resolve().parents[1]
RESTART_CODE = 3
RESTART_FILE = ROOT / ".restart.json"       # tells the next start which port to reuse
KEEP = {".venv", "cache", "work", "ui_state.json", ".git", ".restart.json"}   # never replaced
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHECK_EVERY = 6 * 3600                      # seconds between asks to GitHub

_last: dict = {}
_checked_at = 0.0


def parse(version: str) -> tuple[int, ...]:
    """'v0.4.1' -> (0, 4, 1); anything after the numbers (like '-beta') is ignored."""
    m = re.match(r"v?(\d+(?:\.\d+)*)", version.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def newer(a: str, b: str) -> bool:
    """True when version a is later than version b."""
    pa, pb = parse(a), parse(b)
    width = max(len(pa), len(pb))
    return pa + (0,) * (width - len(pa)) > pb + (0,) * (width - len(pb))


def is_git() -> bool:
    return (ROOT / ".git").exists()


def check(force: bool = False) -> dict:
    """What the interface shows: this version, and the latest release if it is newer."""
    global _last, _checked_at
    if not force and _last and time.time() - _checked_at < CHECK_EVERY:
        return _last
    info = {"current": __version__, "available": False, "git": is_git()}
    try:
        r = requests.get(API, timeout=8, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        rel = r.json()
        tag = rel.get("tag_name") or ""
        asset = next((a for a in rel.get("assets", [])
                      if a["name"].startswith("majlis-cut-") and a["name"].endswith(".zip")), None)
        info.update(latest=tag.lstrip("v"), notes=rel.get("body") or "", page=rel.get("html_url") or "",
                    available=newer(tag, __version__) and (asset is not None or is_git()))
        if asset:
            info.update(zip_url=asset["browser_download_url"], digest=asset.get("digest") or "",
                        size=asset.get("size") or 0)
    except (requests.RequestException, ValueError, KeyError) as exc:
        info["error"] = str(exc)                 # offline: the page simply shows nothing
    _last, _checked_at = info, time.time()
    return info


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if out.returncode:
        raise RuntimeError((out.stderr or out.stdout).strip() or f"git {args[0]} failed")
    return out.stdout.strip()


def _update_git(tag: str) -> None:
    """Move a git checkout forward to the release tag, only when that is a clean fast-forward."""
    if shutil.which("git") is None:
        raise RuntimeError("هذه النسخة منسوخة بـgit لكن git غير موجود. حمّل النسخة الجديدة من صفحة الإصدار.")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("في هذه النسخة تعديلات محلية على الشيفرة؛ حدّثها يدويًّا بـgit pull.")
    _git("fetch", "--quiet", "origin", "tag", tag, "--no-tags")
    try:
        _git("merge", "--ff-only", "--quiet", tag)
    except RuntimeError as exc:
        raise RuntimeError("تعذّر التحديث تلقائيًّا لأن هذه النسخة تختلف عن الإصدار؛ حدّثها يدويًّا بـgit pull.") from exc


def _download(url: str, digest: str, dest: Path) -> None:
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
                sha.update(chunk)
    if digest.startswith("sha256:") and sha.hexdigest() != digest.split(":", 1)[1].lower():
        raise RuntimeError("الملف المحمَّل تالف (لا يطابق بصمته). أعد المحاولة.")


def _replace(src: Path, dst: Path, moved: list[tuple[Path, Path]]) -> None:
    """Put src at dst, keeping the old dst aside so everything can be put back on failure."""
    old = None
    if dst.exists():
        old = dst.with_name(dst.name + ".old-update")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink()
        dst.rename(old)
    moved.append((dst, old))
    shutil.move(str(src), str(dst))


def _update_zip(info: dict) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="majlis_cut_update_"))
    try:
        archive = tmp / "release.zip"
        _download(info["zip_url"], info.get("digest", ""), archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():            # refuse paths that would land outside tmp
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    raise RuntimeError("ملف الإصدار غير سليم.")
            z.extractall(tmp / "x")
        tops = [p for p in (tmp / "x").iterdir()]
        new = tops[0] if len(tops) == 1 and tops[0].is_dir() else tmp / "x"
        init = new / "majlis_cut" / "__init__.py"
        if not init.exists() or info["latest"] not in init.read_text(encoding="utf-8"):
            raise RuntimeError("ملف الإصدار لا يحتوي النسخة المنتظرة.")

        moved: list[tuple[Path, Path | None]] = []
        try:
            for item in sorted(new.iterdir()):
                if item.name not in KEEP:
                    _replace(item, ROOT / item.name, moved)
        except Exception:
            for dst, old in reversed(moved):     # put the old files back
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
                if old is not None:
                    old.rename(dst)
            raise
        for _, old in moved:
            if old is not None:
                shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def apply(port: int) -> None:
    """Install the latest release in place. The caller then exits with RESTART_CODE."""
    info = check(force=True)
    if not info.get("available"):
        raise RuntimeError(info.get("error") or "لا يوجد إصدار أحدث.")
    if info["git"]:
        _update_git("v" + info["latest"])
    else:
        _update_zip(info)
    RESTART_FILE.write_text(json.dumps({"port": port}), encoding="utf-8")


def restart_port() -> int | None:
    """The port of the interface that closed itself to update, read once."""
    try:
        port = int(json.loads(RESTART_FILE.read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    RESTART_FILE.unlink(missing_ok=True)
    return port
