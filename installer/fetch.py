"""Download one listed file: resumable, size-capped, SHA-256 checked (030 R9).

Only addresses on the allow-list are fetched (FR-003), only over verified
HTTPS (FR-049; redirects too), and a download stops the moment it exceeds the
size the list gives (FR-026). A partial ``.part`` file is continued with an
HTTP ``Range`` request on the original address -- GitHub's signed redirect
expires after about an hour, so it is never reused. Requests carry nothing
but a product User-Agent (FR-050).
"""
from __future__ import annotations

import hashlib
import os
import socket
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from installer.errors import InstallError

CHUNK = 1 << 20

# Hosts a listed file may be fetched from (data-model §1).
ALLOWED_HOSTS = frozenset({
    "github.com", "raw.githubusercontent.com", "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "conda.anaconda.org",
    "download.pytorch.org", "download-r2.pytorch.org",
    "pypi.org", "files.pythonhosted.org",
    "dl.fbaipublicfiles.com",
    "huggingface.co",
    "zenodo.org",
})

# Tests, and the maintainer's rehearsals under the developer test-list
# setting (FR-052), serve files from 127.0.0.1 over plain HTTP.
_ALLOW_LOCAL_HTTP = False


class Cancelled(Exception):
    """The user cancelled; the ``.part`` file is kept for Resume."""


def _user_agent() -> str:
    try:
        from functions import __version__ as TOOL_VERSION
    except Exception:
        TOOL_VERSION = "dev"
    return f"PebbleMapper/{TOOL_VERSION}"


def _is_local(url: str) -> bool:
    p = urlparse(url)
    rehearsal = bool(os.environ.get("PEBBLEMAPPER_TEST_LIST", "").strip())   # FR-052
    return ((_ALLOW_LOCAL_HTTP or rehearsal) and p.scheme == "http"
            and p.hostname in ("127.0.0.1", "localhost"))


def check_url(url: str) -> str:
    """The URL's host when it may be fetched; :class:`InstallError` otherwise."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if _is_local(url):
        return host
    if p.scheme != "https" or host not in ALLOWED_HOSTS:
        raise InstallError("published_changed", file=url)
    return host


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not (newurl.startswith("https://") or _is_local(newurl)):
            raise InstallError("network", host=urlparse(newurl).hostname or newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    ctx = ssl.create_default_context()           # verification always on
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),
                                       _HttpsOnlyRedirect())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, size: int, sha256: str) -> bool:
    """True when ``path`` exists with exactly this size and SHA-256."""
    path = Path(path)
    return (path.is_file() and path.stat().st_size == size
            and sha256_of(path) == sha256.lower())


def _classify(ex: BaseException, host: str) -> InstallError:
    reason = getattr(ex, "reason", ex)
    if isinstance(reason, ssl.SSLCertVerificationError) or isinstance(ex, ssl.SSLCertVerificationError):
        return InstallError("clock", host=host)
    if isinstance(ex, urllib.error.HTTPError):
        if ex.code in (404, 410):
            return InstallError("published_changed", file=ex.url or host)
        return InstallError("network", host=host)
    return InstallError("network", host=host)


def download(url: str, dest: Path, size: int, sha256: str,
             on_progress: Optional[Callable[[int, int], None]] = None,
             cancel: Optional[Callable[[], bool]] = None,
             timeout: float = 30.0) -> Path:
    """Fetch ``url`` to ``dest`` and check it; return ``dest``.

    A ``dest`` that already matches is not fetched again (resume reuses
    verified files, FR-032). Raises :class:`InstallError` or
    :class:`Cancelled`.
    """
    dest = Path(dest)
    host = check_url(url)
    if verify_file(dest, size, sha256):
        if on_progress:
            on_progress(size, size)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in (1, 2):
        offset = part.stat().st_size if part.is_file() else 0
        if offset > size:
            part.unlink()
            offset = 0
        if offset < size:
            _fetch_into(url, host, part, offset, size, on_progress, cancel, timeout)
        if sha256_of(part) == sha256.lower():
            os.replace(part, dest)
            return dest
        part.unlink()
        if offset == 0:
            # A complete, fresh download over verified HTTPS that still does
            # not match: what is published is not what the list names.
            raise InstallError("published_changed", file=dest.name)
        # A resumed file can be damaged where the pieces join: once, afresh.
    raise InstallError("damaged_download", file=dest.name)


def _fetch_into(url, host, part, offset, size, on_progress, cancel, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    try:
        resp = _opener().open(req, timeout=timeout)
    except InstallError:
        raise
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, OSError) as ex:
        raise _classify(ex, host) from ex
    with resp:
        status = getattr(resp, "status", 200)
        if offset and not (status == 206 and str(resp.headers.get(
                "Content-Range", "")).startswith(f"bytes {offset}-")):
            offset = 0                            # the server sent it all again
        done = offset
        mode = "ab" if offset else "wb"
        try:
            with open(part, mode) as fh:
                while True:
                    if cancel and cancel():
                        raise Cancelled()
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    done += len(block)
                    if done > size:
                        raise InstallError("published_changed", file=part.name[:-5])
                    fh.write(block)
                    if on_progress:
                        on_progress(done, size)
        except (socket.timeout, ConnectionError, ssl.SSLError) as ex:
            raise InstallError("network", host=host) from ex
        except InstallError as ex:
            if ex.cls == "published_changed":
                part.unlink(missing_ok=True)      # too long: nothing to resume
            raise
    if done < size:
        raise InstallError("network", host=host)   # dropped; Resume continues
