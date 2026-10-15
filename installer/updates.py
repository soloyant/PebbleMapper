"""Is there an update? (031 FR-001–FR-009)

At most once every 24 hours, a random 1–10 minutes after the first page, the
app fetches the model list (``installer.catalog``), which names PebbleMapper's
releases as well as the models'. Nothing else is sent (FR-002). Every failure
is silent and logged once at debug level (FR-003); the result is kept in
``~/.pebblemapper/updates.json`` for the header notice and the Models tab.

Automatic checks are on by default and switched off with ``config.json``
``update_checks`` (FR-004); **Check now** runs one whatever the setting.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from installer.paths import write_json_atomic

_log = logging.getLogger("csm.installer")

CHECK_EVERY_S = 24 * 3600
FIRST_DELAY_S = (60, 600)          # 1–10 min after the first page (FR-001)


# ------------------------------------------------------------------ state
def _state_file():
    from functions.layout import app_home
    return app_home() / "updates.json"


def read_state() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(state: dict) -> None:
    try:
        write_json_atomic(_state_file(), state)
    except OSError as ex:                      # best effort, never fatal
        _log.debug("update state not written: %s", ex)


def checks_enabled() -> bool:
    from functions.layout import _load_config
    return bool(_load_config().get("update_checks", True))


def set_checks_enabled(on: bool) -> None:
    from functions.layout import _load_config, _save_config
    cfg = _load_config()
    cfg["update_checks"] = bool(on)
    _save_config(cfg)


def version_tuple(v: str) -> tuple:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


# --------------------------------------------------------------- the check
def check_due(now: Optional[float] = None) -> bool:
    if not checks_enabled():
        return False
    now = time.time() if now is None else now
    last = read_state().get("last_check_ts", 0)
    return now - float(last or 0) >= CHECK_EVERY_S


def run_check(source: Optional[str] = None, now: Optional[float] = None) -> dict:
    """One check: fetch the list (silent by contract), record the outcome.
    Returns the state. Never raises."""
    from installer import catalog
    now = time.time() if now is None else now
    state = read_state()
    try:
        res = catalog.fetch_list(source)
        state["last_result"] = "ok" if res.fresh else ("too_old" if res.too_old else "offline")
        state["last_message"] = res.message
    except Exception as ex:                    # catalog never raises; belt and braces
        _log.debug("update check failed: %s", ex)
        state["last_result"], state["last_message"] = "offline", ""
    state["last_check_ts"] = now
    state["last_check"] = datetime.fromtimestamp(now).isoformat(timespec="minutes")
    _write_state(state)
    return state


def start_background_check(first_delay: Optional[float] = None) -> threading.Thread:
    """Once per app run, after the first page: wait 1–10 min, check if due,
    then look again every hour (a check runs only when 24 h have passed)."""
    delay = random.uniform(*FIRST_DELAY_S) if first_delay is None else first_delay

    def _loop():
        time.sleep(delay)
        while True:
            try:
                if check_due():
                    run_check()
            except Exception as ex:            # never disturb the app
                _log.debug("update check loop: %s", ex)
            time.sleep(3600)
    t = threading.Thread(target=_loop, name="pm-update-check", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------- what is new
def _installed_app_version() -> str:
    from functions import __version__
    return __version__


def _offered(releases: List[dict], current: str) -> Optional[dict]:
    """The release to offer above ``current``: the newest not withdrawn; if
    ``current`` itself was withdrawn, its named replacement even when older
    (FR-009)."""
    by_rel = {r["release"]: r for r in releases}
    mine = by_rel.get(current)
    if mine and mine.get("withdrawn") and mine.get("replacement") in by_rel:
        return by_rel[mine["replacement"]]
    good = [r for r in releases if not r.get("withdrawn")]
    newer = [r for r in good if version_tuple(r["release"]) > version_tuple(current)]
    return max(newer, key=lambda r: version_tuple(r["release"])) if newer else None


def available_updates(data: Optional[dict], installed: List[dict],
                      app_version: Optional[str] = None) -> List[dict]:
    """``[{"kind": "app"|"model", "key", "current", "release": {...}, ...}]``."""
    if not data:
        return []
    out = []
    app_version = app_version or _installed_app_version()
    rel = _offered(data.get("app_releases", []), app_version)
    if rel:
        out.append({"kind": "app", "key": f"app:{rel['release']}", "id": "app",
                    "name": "PebbleMapper", "current": app_version, "release": rel,
                    "withdrawn_current": _is_withdrawn(data.get("app_releases", []),
                                                       app_version)})
    models = {m["id"]: m for m in data.get("models", [])}
    newest_installed: Dict[str, dict] = {}
    for e in installed:
        cur = newest_installed.get(e["id"])
        if cur is None or version_tuple(e["release"]) > version_tuple(cur["release"]):
            newest_installed[e["id"]] = e
    for mid, e in newest_installed.items():
        m = models.get(mid)
        if not m:
            continue                           # removed from the list: keeps working
        rel = _offered(m["releases"], e["release"])
        if rel and rel["release"] != e["release"] and e.get("variant", "gpu") in rel["variants"]:
            out.append({"kind": "model", "key": f"model:{mid}:{rel['release']}", "id": mid,
                        "name": m["backends"][0]["display_name"], "current": e["release"],
                        "variant": e.get("variant", "gpu"), "release": rel, "model": m,
                        "withdrawn_current": _is_withdrawn(m["releases"], e["release"])})
    return out


def _is_withdrawn(releases: List[dict], release: str) -> bool:
    return any(r["release"] == release and r.get("withdrawn") for r in releases)


def dismiss(key: str) -> None:
    state = read_state()
    d = set(state.get("dismissed", []))
    d.add(key)
    state["dismissed"] = sorted(d)
    _write_state(state)


def to_announce(updates: List[dict]) -> List[dict]:
    """The updates the header line shows: not dismissed (FR-008). A withdrawn
    current release is always announced."""
    dismissed = set(read_state().get("dismissed", []))
    return [u for u in updates if u["key"] not in dismissed or u.get("withdrawn_current")]
