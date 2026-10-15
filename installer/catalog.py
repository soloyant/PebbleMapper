"""The maintainer's list of models: fetch, validate, keep (030 FR-001, FR-003–FR-006).

The list lives at one fixed address (FR-004). It is fetched over verified
HTTPS, checked against the format in ``contracts/list.schema.json`` by the
hand-written validator below (no new dependency), and refused as a whole on
any fault. A list whose ``list_version`` is older than the newest accepted
from the same source is refused too, so an old list cannot be replayed. The
last accepted list is kept, and shown whenever a fetch fails.

``PEBBLEMAPPER_TEST_LIST`` (a URL or a local file) is the maintainer's
developer setting for rehearsals (FR-052); it has its own accepted version.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from installer.paths import list_cache_dir, write_json_atomic

_log = logging.getLogger("csm.installer")

LIST_URL = "https://raw.githubusercontent.com/soloyant/pebblemapper-models/main/list.json"
FORMAT = 1
FETCH_TIMEOUT_S = 10
MAX_LIST_BYTES = 2_000_000
_CACHE_LOCK = threading.Lock()

_ID = re.compile(r"^[a-z0-9-]{1,32}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PY = re.compile(r"^[a-z_][a-z0-9_]*$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ROLES = {"archive", "upstream", "wheel", "check_image", "conda_lock",
          "pip_lock", "weights"}


class ListError(ValueError):
    """The list breaks the format; the message names where."""


@dataclass
class CatalogResult:
    data: Optional[dict]                  # the list in force, or None
    source: str
    fresh: bool                           # fetched now (True) or the kept copy
    message: str = ""                     # what the Models tab says, if anything
    too_old: bool = False
    problems: List[str] = field(default_factory=list)


# ------------------------------------------------------------- validation
def _req(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        raise ListError(f"{where}: missing {key!r}")
    return obj[key]


def _text(v, where, cap=300):
    if not isinstance(v, str) or len(v) > cap:
        raise ListError(f"{where}: must be text of at most {cap} characters")


def _allowed_url(url: str, where: str) -> None:
    from installer.fetch import ALLOWED_HOSTS, _is_local
    if not isinstance(url, str):
        raise ListError(f"{where}: not an address")
    if _is_local(url):
        return
    p = urlparse(url)
    if p.scheme != "https" or (p.hostname or "") not in ALLOWED_HOSTS:
        raise ListError(f"{where}: {url} is not on an allowed host over https")
    if p.hostname == "huggingface.co" and not re.search(r"/resolve/[0-9a-f]{40}/", p.path):
        raise ListError(f"{where}: a Hugging Face address must name a revision hash")


def _check_dest(dest: str, where: str) -> None:
    if (not isinstance(dest, str) or dest.startswith(("/", "\\")) or ":" in dest
            or any(part == ".." for part in re.split(r"[\\/]", dest))):
        raise ListError(f"{where}: dest {dest!r} must stay inside the model folder")


_APP_ROLES = {"archive", "file_list", "conda_lock", "pip_lock"}


def _validate_app_release(r) -> None:
    """One ``app_releases[]`` entry (031 data-model §1)."""
    rel = _req(r, "release", "app_releases")
    if not _SEMVER.match(str(rel)):
        raise ListError(f"app release: bad release {rel!r}")
    w = f"app release {rel}"
    if not _COMMIT.match(str(_req(r, "commit", w))):
        raise ListError(f"{w}: commit must be 40 hex characters")
    _text(_req(r, "summary", w), f"{w}.summary", 600)
    if not str(_req(r, "notes", w)).startswith("https://github.com/soloyant/PebbleMapper/"):
        raise ListError(f"{w}: notes must link to PebbleMapper's own pages")
    ms = _req(r, "min_starter", w)
    if not isinstance(ms, int) or isinstance(ms, bool) or ms < 1:
        raise ListError(f"{w}: min_starter must be a positive integer")
    if not isinstance(_req(r, "withdrawn", w), bool):
        raise ListError(f"{w}: withdrawn must be true or false")
    if "replacement" in r and not _SEMVER.match(str(r["replacement"])):
        raise ListError(f"{w}: bad replacement")
    variants = _req(r, "variants", w)
    if not isinstance(variants, dict) or not variants or set(variants) - {"gpu", "cpu"}:
        raise ListError(f"{w}: variants must be gpu and/or cpu")
    for vn, v in variants.items():
        wv = f"{w} {vn}"
        for k in ("download_bytes", "disk_bytes"):
            val = _req(v, k, wv)
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ListError(f"{wv}: {k} must be a whole number")
        files = _req(v, "files", wv)
        roles = set()
        for j, f in enumerate(files if isinstance(files, list) else []):
            wf = f"{wv} files[{j}]"
            role = _req(f, "role", wf)
            if role not in _APP_ROLES:
                raise ListError(f"{wf}: unknown role {role!r}")
            roles.add(role)
            _allowed_url(_req(f, "url", wf), wf)
            size = _req(f, "size", wf)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ListError(f"{wf}: size must be at least 1")
            if not _SHA.match(str(_req(f, "sha256", wf))):
                raise ListError(f"{wf}: sha256 must be 64 hex characters")
        if _APP_ROLES - roles:
            raise ListError(f"{wv}: missing {sorted(_APP_ROLES - roles)}")


def validate(data) -> None:
    """Raise :class:`ListError` unless ``data`` is a valid format-1 list.
    Mirrors ``contracts/list.schema.json``."""
    if not isinstance(data, dict):
        raise ListError("the list is not an object")
    if data.get("format") != FORMAT:
        raise ListError("format")
    lv = _req(data, "list_version", "list")
    if not isinstance(lv, int) or isinstance(lv, bool) or lv < 1:
        raise ListError("list_version must be a positive integer")
    if not _DATE.match(str(_req(data, "published", "list"))):
        raise ListError("published must be a date")
    app_rels = data.get("app_releases", [])
    if not isinstance(app_rels, list):
        raise ListError("app_releases must be a list")
    for r in app_rels:
        _validate_app_release(r)
    models = _req(data, "models", "list")
    if not isinstance(models, list):
        raise ListError("models must be a list")
    seen = set()
    for i, m in enumerate(models):
        w = f"models[{i}]"
        mid = _req(m, "id", w)
        if not isinstance(mid, str) or not _ID.match(mid) or mid in seen:
            raise ListError(f"{w}: bad or repeated id {mid!r}")
        seen.add(mid)
        w = f"model {mid}"
        for k in ("authors", "licence_code", "licence_weights", "description"):
            _text(_req(m, k, w), f"{w}.{k}")
        _text(_req(m, "citation", w), f"{w}.citation", 600)
        home = _req(m, "homepage", w)
        if not str(home).startswith("https://github.com/soloyant/pebblemapper-backend-"):
            raise ListError(f"{w}: homepage must be the plug-in's own repository")
        if _req(m, "runs_in", w) not in ("own", "app"):
            raise ListError(f"{w}: runs_in must be own or app")
        backends = _req(m, "backends", w)
        if not isinstance(backends, list) or not backends:
            raise ListError(f"{w}: no backends")
        for b in backends:
            if not _ID.match(str(_req(b, "name", w))):
                raise ListError(f"{w}: bad backend name")
            _text(_req(b, "display_name", w), f"{w}.display_name")
            for k in ("package", "module", "factory"):
                if not _PY.match(str(_req(b, k, w))):
                    raise ListError(f"{w}: backend {k} must be a Python name")
        releases = _req(m, "releases", w)
        if not isinstance(releases, list) or not releases:
            raise ListError(f"{w}: no releases")
        for r in releases:
            rel = _req(r, "release", w)
            if not _SEMVER.match(str(rel)):
                raise ListError(f"{w}: bad release {rel!r}")
            wr = f"{w} {rel}"
            if not _COMMIT.match(str(_req(r, "commit", wr))):
                raise ListError(f"{wr}: commit must be 40 hex characters")
            if not isinstance(_req(r, "withdrawn", wr), bool):
                raise ListError(f"{wr}: withdrawn must be true or false")
            if "summary" in r:
                _text(r["summary"], f"{wr}.summary")
            if "notes" in r and not str(r["notes"]).startswith("https://github.com/soloyant/"):
                raise ListError(f"{wr}: notes must link to the maintainer's pages")
            if "replacement" in r and not _SEMVER.match(str(r["replacement"])):
                raise ListError(f"{wr}: bad replacement")
            variants = _req(r, "variants", wr)
            if not isinstance(variants, dict) or not variants \
                    or set(variants) - {"gpu", "cpu"}:
                raise ListError(f"{wr}: variants must be gpu and/or cpu")
            for vn, v in variants.items():
                wv = f"{wr} {vn}"
                for k in ("download_bytes", "disk_bytes", "peak_bytes",
                          "longest_path", "check_timeout_s"):
                    val = _req(v, k, wv)
                    if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                        raise ListError(f"{wv}: {k} must be a whole number")
                mins = _req(v, "minutes", wv)
                if not (isinstance(mins, list) and len(mins) == 2):
                    raise ListError(f"{wv}: minutes must be [low, high]")
                files = _req(v, "files", wv)
                if not isinstance(files, list) or not files:
                    raise ListError(f"{wv}: no files")
                roles = set()
                for j, f in enumerate(files):
                    wf = f"{wv} files[{j}]"
                    role = _req(f, "role", wf)
                    if role not in _ROLES:
                        raise ListError(f"{wf}: unknown role {role!r}")
                    roles.add(role)
                    _allowed_url(_req(f, "url", wf), wf)
                    size = _req(f, "size", wf)
                    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                        raise ListError(f"{wf}: size must be at least 1")
                    if not _SHA.match(str(_req(f, "sha256", wf))):
                        raise ListError(f"{wf}: sha256 must be 64 hex characters")
                    if "dest" in f:
                        _check_dest(f["dest"], wf)
                need = {"archive", "check_image"} | (
                    {"conda_lock"} if m["runs_in"] == "own" else set())
                if need - roles:
                    raise ListError(f"{wv}: missing {sorted(need - roles)}")


# ------------------------------------------------------------ the source
def current_source() -> str:
    return os.environ.get("PEBBLEMAPPER_TEST_LIST", "").strip() or LIST_URL


def is_test_source(source: str) -> bool:
    return source != LIST_URL


def _key(source: str) -> str:
    return "main" if source == LIST_URL else \
        "test-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _cache_path(source: str) -> Path:
    return list_cache_dir() / f"{_key(source)}.json"


def _accepted_path(source: str) -> Path:
    return list_cache_dir() / f"{_key(source)}.accepted"


def accepted_version(source: str) -> int:
    try:
        return int(_accepted_path(source).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def kept_list(source: str) -> Optional[dict]:
    try:
        return json.loads(_cache_path(source).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _download(source: str) -> bytes:
    if not source.startswith(("http://", "https://")):
        return Path(source).read_bytes()        # a local test list
    from installer.fetch import _opener, _user_agent, check_url
    p = urlparse(source)
    if not (p.scheme == "https" and p.hostname in ("raw.githubusercontent.com", "github.com")):
        check_url(source)                       # test lists: local or allowed only
    req = urllib.request.Request(source, headers={"User-Agent": _user_agent()})
    with _opener().open(req, timeout=FETCH_TIMEOUT_S) as r:
        body = r.read(MAX_LIST_BYTES + 1)
    if len(body) > MAX_LIST_BYTES:
        raise ListError("the list is larger than any list should be")
    return body


def _offline_message(kept: Optional[dict]) -> str:
    if kept:
        return ("Could not reach the model list (no connection). Showing the "
                f"list from {kept.get('published', 'an earlier date')}.")
    return "Connect to the internet once to see the models you can add."


TOO_OLD = ("This version of PebbleMapper is too old to read the current model "
           "list; update PebbleMapper.")


def fetch_list(source: Optional[str] = None) -> CatalogResult:
    """Fetch, validate and keep the list; on any failure return the kept one
    with the sentence the Models tab shows. Never raises."""
    source = source or current_source()
    kept = kept_list(source)
    try:
        body = _download(source)
        data = json.loads(body.decode("utf-8"))
    except Exception as ex:                      # offline, proxy, DNS, TLS, 404
        _log.info("model list not fetched from %s: %s", source, ex)
        return CatalogResult(kept, source, False, _offline_message(kept))
    if isinstance(data, dict) and data.get("format") != FORMAT:
        _log.warning("model list format %r is not understood", data.get("format"))
        return CatalogResult(kept, source, False, TOO_OLD, too_old=True)
    try:
        validate(data)
    except ListError as ex:
        _log.warning("model list from %s refused: %s", source, ex)
        msg = "The model list could not be read; showing the list kept from before."
        return CatalogResult(kept, source, False, msg if kept else _offline_message(None),
                             problems=[str(ex)])
    if data["list_version"] < accepted_version(source):
        _log.warning("model list version %s is older than the accepted %s; refused",
                     data["list_version"], accepted_version(source))
        return CatalogResult(kept, source, False,
                             "The model list received is older than the one already "
                             "seen; showing the newer one kept from before.",
                             problems=["older list_version"])
    try:
        with _CACHE_LOCK:              # two pages refreshing at once write in turn
            if data["list_version"] >= accepted_version(source):
                write_json_atomic(_cache_path(source), data)
                write_json_atomic(_accepted_path(source), data["list_version"])
    except OSError as ex:
        # The list itself is good; only keeping a copy failed. Use it now.
        _log.warning("model list fetched but not kept (%s); it will be fetched "
                     "again next time", ex)
    return CatalogResult(data, source, True, "")


def latest_release(model: dict) -> Optional[dict]:
    rels = [r for r in model.get("releases", []) if not r.get("withdrawn")]
    if not rels:
        return None
    return max(rels, key=lambda r: tuple(int(x) for x in r["release"].split(".")))
