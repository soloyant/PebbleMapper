"""Long-path-aware reads and the unreadable-vs-empty distinction.

Composed output names in this project can exceed Windows' 260-char MAX_PATH.
``open_robust`` tries a normal open first, retries with the ``\\\\?\\``
extended-length prefix when the file exists there, and otherwise raises
:class:`UnreadableFileError` naming the path and the actual reason, so an
unreadable input is never reported as an empty one.
"""

from __future__ import annotations

import os as _os


class UnreadableFileError(OSError):
    """The file exists (or its status cannot be determined) but cannot be
    read. Distinct from FileNotFoundError so callers can tell 'missing',
    'unreadable' and 'empty' apart."""

    def __init__(self, path, reason: str):
        self.path = str(path)
        self.reason = reason
        super().__init__(f"could not read {self.path} ({reason})")


def extended_path(path) -> str:
    r"""The ``\\?\``-prefixed form of an absolute Windows path (UNC-aware).

    On non-Windows platforms, or for an already-prefixed path, returns the
    string unchanged.
    """
    s = str(path)
    if _os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    s = _os.path.abspath(s)
    if s.startswith("\\\\"):                 # UNC share
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def exists_robust(path) -> bool:
    """Existence check that survives >MAX_PATH paths."""
    return _os.path.exists(str(path)) or _os.path.exists(extended_path(path))


def open_robust(path, mode: str = "r", **kwargs):
    """``open()`` with a ``\\\\?\\`` retry; raises :class:`UnreadableFileError`
    with the real reason when the file exists but cannot be read, and plain
    :class:`FileNotFoundError` when it genuinely is not there."""
    try:
        return open(path, mode, **kwargs)
    except FileNotFoundError:
        ext = extended_path(path)
        if ext != str(path) and _os.path.exists(ext):
            try:
                return open(ext, mode, **kwargs)
            except OSError as exc2:
                raise UnreadableFileError(path, str(exc2)) from exc2
        if len(str(_os.path.abspath(str(path)))) >= 260 and _os.name == "nt":
            raise UnreadableFileError(
                path, "path is longer than this Windows configuration "
                      "allows (MAX_PATH); shorten the file name or move the "
                      "project to a shorter root") from None
        raise
    except OSError as exc:
        if exists_robust(path):
            raise UnreadableFileError(path, str(exc)) from exc
        raise


def read_text_robust(path, encoding: str = "utf-8", errors: str = "replace") -> str:
    with open_robust(path, "r", encoding=encoding, errors=errors) as fh:
        return fh.read()
