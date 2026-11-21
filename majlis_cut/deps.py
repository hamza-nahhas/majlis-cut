"""Keep the libraries in .venv in step with requirements.txt.

    python -m majlis_cut.deps          install only if requirements.txt changed
    python -m majlis_cut.deps --force  install anyway

setup.bat and start.bat both run this, so after an update the libraries are
refreshed on the next start without anyone having to remember it. It uses only
the standard library: it must work even before the libraries are installed.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = Path(sys.prefix) / "majlis_cut_requirements.sha256"


def fingerprint() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current() -> bool:
    try:
        return STAMP.read_text(encoding="ascii").strip() == fingerprint()
    except OSError:
        return False


def install() -> int:
    print("Installing libraries (several minutes the first time)...", flush=True)
    pip = [sys.executable, "-m", "pip"]
    subprocess.run([*pip, "install", "--upgrade", "pip", "--quiet"], cwd=ROOT)
    code = subprocess.run([*pip, "install", "-r", str(REQUIREMENTS)], cwd=ROOT).returncode
    if code == 0:
        STAMP.write_text(fingerprint(), encoding="ascii")
    return code


def main() -> int:
    if "--force" not in sys.argv and current():
        return 0
    return install()


if __name__ == "__main__":
    sys.exit(main())
