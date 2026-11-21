"""Print the CHANGELOG.md section of one version (used by the release workflow).

    python tools/release_notes.py 0.4.0
"""
import re
import sys
from pathlib import Path

version = sys.argv[1].lstrip("v")
text = Path(__file__).resolve().parents[1].joinpath("CHANGELOG.md").read_text(encoding="utf-8")
m = re.search(rf"^## {re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
if not m:
    sys.exit(f"CHANGELOG.md has no section for {version}")
sys.stdout.buffer.write(m.group(1).strip().encode("utf-8") + b"\n")
