"""Check of the updater: version comparison, and installing a release zip in place
without touching what the user made.

    python tests/test_update.py
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from majlis_cut import update  # noqa: E402

assert update.parse("v0.4.1") == (0, 4, 1)
assert update.newer("v0.10.0", "0.9.9") and update.newer("0.4.1", "0.4") and update.newer("1.0", "0.9.9")
assert not update.newer("0.4.0", "0.4") and not update.newer("v0.3.9", "0.4.0") and not update.newer("", "0.4.0")

tmp = Path(tempfile.mkdtemp(prefix="majlis_cut_test_"))
try:
    # An installed copy: old code, plus things the user made that must survive.
    root = tmp / "app"
    (root / "majlis_cut").mkdir(parents=True)
    (root / "majlis_cut" / "__init__.py").write_text('__version__ = "0.4.0"\n')
    (root / "majlis_cut" / "gone.py").write_text("old\n")
    (root / "start.bat").write_text("old\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "lib.txt").write_text("keep\n")
    (root / "cache").mkdir()
    (root / "cache" / "book.json").write_text("keep\n")
    (root / "ui_state.json").write_text("keep\n")

    # The release zip, as git archive makes it: everything under majlis-cut/.
    archive = tmp / "release.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("majlis-cut/majlis_cut/__init__.py", '__version__ = "0.5.0"\n')
        z.writestr("majlis-cut/majlis_cut/new.py", "new\n")
        z.writestr("majlis-cut/start.bat", "new\n")
        z.writestr("majlis-cut/requirements.txt", "numpy\n")

    update.ROOT = root
    update._download = lambda url, digest, dest: shutil.copy(archive, dest)
    update._update_zip({"zip_url": "x", "latest": "0.5.0"})

    assert "0.5.0" in (root / "majlis_cut" / "__init__.py").read_text()
    assert (root / "majlis_cut" / "new.py").exists()
    assert not (root / "majlis_cut" / "gone.py").exists(), "the code folder is replaced whole"
    assert (root / "start.bat").read_text() == "new\n" and (root / "requirements.txt").exists()
    for kept in (".venv/lib.txt", "cache/book.json", "ui_state.json"):
        assert (root / kept).read_text() == "keep\n", kept
    assert not list(root.glob("*.old-update")), "nothing left behind"

    # A zip with another version inside is refused and changes nothing.
    try:
        update._update_zip({"zip_url": "x", "latest": "0.6.0"})
        raise AssertionError("a wrong zip was installed")
    except RuntimeError:
        pass
    assert "0.5.0" in (root / "majlis_cut" / "__init__.py").read_text()

    # A zip that tries to write outside the folder is refused.
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("majlis-cut/../evil.txt", "x")
    try:
        update._update_zip({"zip_url": "x", "latest": "0.5.0"})
        raise AssertionError("an unsafe zip was accepted")
    except RuntimeError:
        pass
    assert not (tmp / "evil.txt").exists()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("update: all checks passed")
