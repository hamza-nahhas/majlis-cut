"""Extract the page's script and syntax-check it with node: python tests/check_ui_js.py"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

html = (Path(__file__).resolve().parents[1] / "darsmontage" / "ui.html").read_text(encoding="utf-8")
script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
ids_used = set(re.findall(r'\$\("([\w]+)"\)', script))
ids_defined = set(re.findall(r'id="([\w]+)"', html))
print("ids used but not defined:", sorted(ids_used - ids_defined))
with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
    fh.write(script)
res = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True, shell=sys.platform == "win32")
print("node --check:", "OK" if res.returncode == 0 else res.stderr)
