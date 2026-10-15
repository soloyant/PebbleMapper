"""030 FR-025, FR-038, SC-004: an archive cannot write outside its folder."""
import os
import subprocess
import sys
import zipfile

import pytest

from installer import archive
from installer.errors import InstallError


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            if isinstance(data, zipfile.ZipInfo):
                zf.writestr(data, b"target")
            else:
                zf.writestr(name, data)
    return path


def _nothing_outside(tmp_path, dest):
    return sorted(p.name for p in tmp_path.iterdir()) == sorted(["a.zip", dest.name])


def test_a_clean_archive_unpacks(tmp_path):
    z = _zip(tmp_path / "a.zip", [("pkg/__init__.py", b""), ("pkg/run.py", b"print(1)")])
    dest = tmp_path / "code"
    assert sorted(archive.safe_unzip(z, dest)) == ["pkg/__init__.py", "pkg/run.py"]
    assert (dest / "pkg" / "run.py").read_bytes() == b"print(1)"


@pytest.mark.parametrize("bad", ["../escape.txt", "pkg/../../escape.txt",
                                 "/abs.txt", "C:/abs.txt", "C:\\abs.txt",
                                 "..\\escape.txt"])
def test_entries_leaving_the_folder_are_refused_before_anything_is_written(tmp_path, bad):
    z = _zip(tmp_path / "a.zip", [("ok.txt", b"1"), (bad, b"x")])
    dest = tmp_path / "code"
    with pytest.raises(InstallError) as ei:
        archive.safe_unzip(z, dest)
    assert ei.value.cls == "published_changed"
    assert not (dest / "ok.txt").exists(), "nothing applied"
    assert _nothing_outside(tmp_path, dest)


def test_a_symlink_entry_is_refused(tmp_path):
    info = zipfile.ZipInfo("pkg/link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    z = _zip(tmp_path / "a.zip", [("pkg/link", info)])
    with pytest.raises(InstallError):
        archive.safe_unzip(z, tmp_path / "code")


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_a_junction_inside_the_folder_cannot_carry_writes_out(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "code"
    dest.mkdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(dest / "pkg"), str(outside)],
                   check=True, capture_output=True)
    z = _zip(tmp_path / "a.zip", [("pkg/evil.txt", b"x")])
    try:
        with pytest.raises(InstallError):
            archive.safe_unzip(z, dest)
        assert list(outside.iterdir()) == []
    finally:
        os.rmdir(dest / "pkg")             # removes the junction, not its target


def test_transient_locks_are_retried_then_named(monkeypatch):
    monkeypatch.setattr(archive, "_RETRIES", (0, 0))
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise PermissionError(13, "locked", "x.dll")
        return "ok"
    assert archive.retry_io(flaky) == "ok"

    def locked():
        raise PermissionError(13, "locked", "y.dll")
    with pytest.raises(InstallError) as ei:
        archive.retry_io(locked)
    assert ei.value.cls == "file_locked" and "y.dll" in ei.value.sentence()
