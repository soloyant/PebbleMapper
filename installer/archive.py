"""Unpack a verified archive without letting it write outside its folder.

Every entry is checked before anything is written (030 FR-025): no absolute
path, no drive letter, no ``..``, no link entry; and every file must still
resolve inside the destination once written, so a junction planted in the
destination cannot redirect it. Any failure leaves nothing applied.
"""
from __future__ import annotations

import os
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from installer.errors import InstallError

_RETRIES = (0.5, 1.0, 2.0, 4.0)         # antivirus holds a fresh file briefly


def _refuse(archive: Path, name: str, why: str) -> InstallError:
    return InstallError("published_changed",
                        file=f"{archive.name} (entry {name!r}: {why})")


def _check_entry(archive: Path, info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    win = PureWindowsPath(name)
    posix = PurePosixPath(name.replace("\\", "/"))
    if win.drive or win.root or posix.is_absolute():
        raise _refuse(archive, name, "absolute path")
    if any(part == ".." for part in posix.parts):
        raise _refuse(archive, name, "leaves its folder")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise _refuse(archive, name, "link")
    return posix


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


def retry_io(fn, *args, **kwargs):
    """Run ``fn``, retrying transient access errors (FR-038); the last one
    becomes a ``file_locked`` :class:`InstallError` naming the file."""
    for i, delay in enumerate((0,) + _RETRIES):
        if delay:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except PermissionError as ex:
            last = ex
    raise InstallError("file_locked", path=getattr(last, "filename", "") or "?") from last


def safe_unzip(archive: Path, dest: Path) -> list:
    """Unpack ``archive`` into ``dest``; return the relative paths written."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        entries = [(info, _check_entry(archive, info)) for info in infos]
        targets = []
        for info, rel in entries:
            target = dest.joinpath(*rel.parts)
            # A junction or link already inside dest would carry the write out.
            probe = target
            while probe != dest and not probe.exists():
                probe = probe.parent
            if not _inside(probe, dest):
                raise _refuse(archive, info.filename, "would be written through a link")
            targets.append((info, rel, target))
        written = []
        for info, rel, target in targets:
            if info.is_dir():
                retry_io(target.mkdir, parents=True, exist_ok=True)
                continue
            retry_io(target.parent.mkdir, parents=True, exist_ok=True)
            if not _inside(target.parent, dest):
                raise _refuse(archive, info.filename, "would be written through a link")

            def _write(info=info, target=target):
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
            retry_io(_write)
            written.append(rel.as_posix())
    return written
