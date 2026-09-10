"""Fetch the trained Mask R-CNN weights from Zenodo into ``model_weights/``.

    python -m detectors.download_weights            # fetch and verify
    python -m detectors.download_weights --list     # only show the record's files
    python -m detectors.download_weights --dest D   # somewhere else

The weights (``mask_rcnn_clasts.h5``, CC BY 4.0) are published on Zenodo under
the DOI in :data:`DOI`. The record is read through Zenodo's API at run time,
so the file's URL and MD5 come from the record itself: the download is checked
against that checksum, a file already in place is verified rather than fetched
again, and a mismatch is reported instead of silently overwritten. The plug-in
models each have a ``scripts/download_*.py`` doing the same for theirs, so
every model installs the same way: environment, weights script, done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DOI = "10.5281/zenodo.20779877"
FILE = "mask_rcnn_clasts.h5"
LICENCE = "CC BY 4.0"
CITATION = ("Soloy, A.; Turki, I.; Fournier, M.; Costa, S.; Peuziat, B.; Lecoq, N. (2026). "
            "PebbleMapper: trained Mask R-CNN weights for clast detection (mask_rcnn_clasts.h5). "
            f"Zenodo. https://doi.org/{DOI}")
ISSUES = "https://github.com/soloyant/pebblemapper/issues"
# False until the Zenodo record is published: every "weights missing"
# message then says how to obtain them in the meantime. Set it to True the
# day the record goes public.
WEIGHTS_PUBLIC = False
WEIGHTS_PENDING = (
    "The trained Mask R-CNN weights are not public yet: their licensing is being "
    "cleared, which should be settled in the coming weeks. In the meantime the "
    "author sends them privately on request: open an issue at " + ISSUES + ". "
    f"Place the file you receive at model_weights/{FILE}.")


def weights_missing_message() -> str:
    """What to tell a user whose Mask R-CNN weights are not in place."""
    if WEIGHTS_PUBLIC:
        return (f"The Mask R-CNN weights ({FILE}) are missing. Double-click "
                "'Install PebbleMapper' again, or run: python -m detectors.download_weights")
    return WEIGHTS_PENDING
API = "https://zenodo.org/api/records/"
_CHUNK = 1 << 20


EXIT_NOT_PUBLISHED = 4     # the record does not resolve yet: not an install failure


class RecordNotFound(RuntimeError):
    """The Zenodo record behind the DOI does not resolve (not published yet,
    or a different identifier)."""


def record_id(doi: str) -> str:
    """``10.5281/zenodo.20779877`` -> ``20779877``."""
    tail = doi.strip().rsplit("zenodo.", 1)
    if len(tail) != 2 or not tail[1].isdigit():
        raise ValueError(f"not a Zenodo DOI: {doi!r}")
    return tail[1]


def _open(url: str):
    """One place that touches the network, so tests can stand in for it."""
    return urlopen(Request(url, headers={"User-Agent": "PebbleMapper weights"}),
                   timeout=60)


def fetch_record(doi: str) -> dict:
    """The record's JSON; ``RecordNotFound`` when Zenodo has no such record."""
    url = API + record_id(doi)
    try:
        with _open(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as ex:
        if ex.code in (404, 410):
            raise RecordNotFound(
                f"Zenodo has no record for {doi} (HTTP {ex.code}). The record "
                "may not be published yet; download the file from the DOI page "
                f"by hand and place it at model_weights/{FILE}.") from ex
        raise RuntimeError(f"Zenodo answered HTTP {ex.code} for {url}") from ex
    except URLError as ex:
        raise RuntimeError(f"could not reach Zenodo ({ex.reason}); download "
                           f"{FILE} from https://doi.org/{doi} by hand") from ex


def pick_file(record: dict, name: str = FILE) -> dict:
    """``{"url", "size", "md5"}`` of ``name`` in the record's file list."""
    for entry in record.get("files") or []:
        if entry.get("key") == name:
            checksum = str(entry.get("checksum") or "")
            md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
            url = ((entry.get("links") or {}).get("self")
                   or f"{API}{record.get('id')}/files/{name}/content")
            return {"url": url, "size": int(entry.get("size") or 0), "md5": md5}
    names = [e.get("key") for e in record.get("files") or []]
    raise RuntimeError(f"the record holds no {name}; it holds {names}")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int = 0, md5: Optional[str] = None,
             log: Callable[[str], None] = print) -> Path:
    """Stream ``url`` to ``dest`` through a temporary file, hashing as it
    goes; the file only takes its final name once the checksum matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.md5()
    done, last, t0 = 0, 0.0, time.time()
    with _open(url) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if size and done - last >= size / 20:
                last = done
                log(f"  {done / 2**20:6.0f} / {size / 2**20:.0f} MB "
                    f"({100 * done / size:3.0f} %, {time.time() - t0:.0f} s)")
    got = h.hexdigest()
    if md5 and got != md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch: record says md5 {md5}, "
                           f"download is {got}; nothing was kept")
    os.replace(tmp, dest)
    return dest


def default_dest() -> Path:
    """Where the backend looks first (``model_weights/`` at the repo root)."""
    from detectors.maskrcnn import _weights_candidates
    return _weights_candidates()[0].parent


def main(argv=None, log: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doi", default=DOI, help=f"the Zenodo DOI (default {DOI})")
    ap.add_argument("--dest", default=None,
                    help="folder to put the file in (default: model_weights/)")
    ap.add_argument("--list", action="store_true",
                    help="show the record's files and stop")
    ap.add_argument("--force", action="store_true",
                    help="fetch again even when a file with another checksum is in place")
    args = ap.parse_args(argv)
    dest_dir = Path(args.dest) if args.dest else default_dest()
    target = dest_dir / FILE

    try:
        record = fetch_record(args.doi)
    except RecordNotFound as ex:
        if target.exists():
            log(f"{target} is in place ({target.stat().st_size / 2**20:.0f} MB); "
                "its checksum could not be checked against the record.")
            return 0
        log(str(ex) if WEIGHTS_PUBLIC else WEIGHTS_PENDING)
        return EXIT_NOT_PUBLISHED
    except RuntimeError as ex:
        log(str(ex))
        return 2

    meta = record.get("metadata") or {}
    licence = ((meta.get("license") or {}).get("id") or LICENCE)
    log(f"Zenodo record {record.get('id')}: {meta.get('title') or '?'} "
        f"(licence {licence}, version {meta.get('version') or '?'})")
    if args.list:
        for e in record.get("files") or []:
            log(f"  {e.get('key')}  {int(e.get('size') or 0) / 2**20:.1f} MB  "
                f"{e.get('checksum')}")
        return 0

    info = pick_file(record, FILE)
    if target.exists() and not args.force:
        have = md5_of(target)
        if info["md5"] is None or have == info["md5"]:
            log(f"have {target} (md5 {have}); nothing to do")
            return 0
        log(f"{target} is in place but its md5 {have} differs from the record's "
            f"{info['md5']}; run again with --force to replace it")
        return 3

    log(f"downloading {FILE} ({info['size'] / 2**20:.0f} MB) -> {target}")
    download(info["url"], target, info["size"], info["md5"], log)
    log(f"done: {target} (md5 verified)" if info["md5"] else f"done: {target}")
    log(f"The weights are {licence}; cite them as: {CITATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
