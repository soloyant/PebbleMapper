"""Per-tab queue persistence.

One JSON file per queue-bearing tab under ``app_home()/queues/``, written
atomically (temp + ``os.replace``) so a crash mid-write leaves the previous
version readable. A watcher thread snapshots the queue lists about once a
second and writes the ones whose content changed.

Restore semantics:

- a row persisted as ``running`` can only mean the process died mid-job;
  it comes back as ``interrupted``, never silently resumed;
- a ``done`` row whose recorded outputs are gone demotes to ``pending``;
- declining the restore offer archives the files under ``queues/declined/``
  (newest 10 kept) and they are not offered again;
- an unknown file version is refused by name, never guessed at.
"""

from __future__ import annotations

import copy as _copy
import datetime as _dt
import json as _json
import os as _os
import uuid as _uuid
from pathlib import Path
from typing import Optional

from functions.layout import app_home

FILE_VERSION = 1

#: tab name -> attribute on the GUI state object holding its job list.
TABS = {
    "detection": "det_jobs",
    "zonal": "zonal_jobs",
    "rasterize": "ras_jobs",
    "map": "map_jobs",
    "merge": "merge_jobs",
    "validate": "val_jobs",
}

#: A file whose rows are all terminal does not trigger a restore offer.
TERMINAL = {"done", "error"}

_DECLINED_KEEP = 10


def queues_dir() -> Path:
    return app_home() / "queues"


def _path(tab: str) -> Path:
    return queues_dir() / f"{tab}.json"


def sanitize_rows(rows) -> list:
    """A JSON-safe deep copy of the job dicts (non-serialisable values become strings)."""
    return _json.loads(_json.dumps(rows, default=str))


def save(tab: str, rows, *, project: str = "", date_layer: Optional[str] = None,
         app_version: str = "") -> bool:
    """Atomically write one tab's queue. An empty queue deletes the file:
    nothing to restore is represented by absence."""
    try:
        p = _path(tab)
        if not rows:
            if p.exists():
                p.unlink()
            return True
        payload = {
            "version": FILE_VERSION,
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "app_version": app_version,
            "project": project or None,
            "date_layer": date_layer or None,
            "rows": sanitize_rows(rows),
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp-{_uuid.uuid4().hex[:6]}")
        tmp.write_text(_json.dumps(payload, indent=1), encoding="utf-8")
        _os.replace(tmp, p)
        return True
    except OSError:
        return False


def load(tab: str) -> Optional[dict]:
    """The stored document, or None. An unknown version comes back as
    ``{"version_error": "..."}`` so the caller can refuse it by name."""
    p = _path(tab)
    try:
        doc = _json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return {"version_error": f"queue file {p.name} is unreadable/corrupt"}
    if doc.get("version") != FILE_VERSION:
        return {"version_error":
                f"queue file {p.name} is version {doc.get('version')!r}; "
                f"this build reads {FILE_VERSION}"}
    return doc


def pending_restore() -> dict:
    """{tab: {"rows": n_nonterminal, "total": n, "project": str|None,
    "saved_at": str, "version_error": str|None}} for every tab whose stored
    queue holds non-terminal rows (or cannot be read)."""
    out = {}
    for tab in TABS:
        doc = load(tab)
        if doc is None:
            continue
        if "version_error" in doc:
            out[tab] = {"rows": 0, "total": 0, "project": None,
                        "saved_at": None,
                        "version_error": doc["version_error"]}
            continue
        rows = doc.get("rows") or []
        nonterm = [r for r in rows
                   if str(r.get("status") or "pending") not in TERMINAL]
        if nonterm:
            out[tab] = {"rows": len(nonterm), "total": len(rows),
                        "project": doc.get("project"),
                        "saved_at": doc.get("saved_at"),
                        "version_error": None}
    return out


def _row_outputs(row: dict) -> list:
    """Every string in the row that looks like a recorded output path."""
    outs = []
    for key, val in row.items():
        if not isinstance(val, str) or not val:
            continue
        k = key.lower()
        if k in ("out_path", "out_csv", "out_png", "output", "output_path") \
                or k.startswith("out_"):
            outs.append(val)
    if isinstance(row.get("outputs"), list):
        outs.extend(str(o) for o in row["outputs"] if o)
    return outs


def revalidate_rows(rows: list) -> list:
    """Apply the restore rules to a loaded row list, in place: ``running`` ->
    ``interrupted``; ``done`` with missing outputs -> ``pending``."""
    for row in rows:
        status = str(row.get("status") or "pending")
        if status == "running":
            row["status"] = "interrupted"
        elif status == "done":
            outs = _row_outputs(row)
            if outs and not all(Path(o).exists() for o in outs):
                row["status"] = "pending"
                row["error"] = ("restored: recorded output no longer on "
                                "disk — demoted to pending")
    return rows


def restore_into(state_obj) -> dict:
    """Load every stored queue into ``state_obj``'s job lists (revalidated).
    Returns {tab: n_rows_restored}."""
    restored = {}
    for tab, attr in TABS.items():
        doc = load(tab)
        if not doc or "version_error" in doc:
            continue
        rows = revalidate_rows(doc.get("rows") or [])
        if not rows:
            continue
        getattr(state_obj, attr)[:] = rows
        restored[tab] = len(rows)
    return restored


def archive_declined() -> int:
    """Move every stored queue file to ``queues/declined/`` (newest
    ``_DECLINED_KEEP`` kept). Returns how many files were archived."""
    n = 0
    dec = queues_dir() / "declined"
    try:
        dec.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0
    for tab in TABS:
        p = _path(tab)
        if p.exists():
            stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            try:
                _os.replace(p, dec / f"{tab}-{stamp}.json")
                n += 1
            except OSError:
                pass
    try:
        old = sorted(dec.glob("*.json"), key=lambda q: q.name)
        for q in old[:-_DECLINED_KEEP]:
            q.unlink()
    except OSError:
        pass
    return n


class QueueWatcher:
    """Snapshots the queue lists ~once a second and writes changed ones.

    A plain daemon thread, not a ``ui.timer``: a timer created without a
    client context never fires. Reads are snapshot copies; a concurrent
    mutation at worst delays persistence one tick.
    """

    def __init__(self, state_obj, interval: float = 1.0, app_version: str = ""):
        self._state = state_obj
        self._interval = interval
        self._app_version = app_version
        self._last = {}
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def flush_once(self) -> None:
        for tab, attr in TABS.items():
            try:
                rows = _copy.deepcopy(list(getattr(self._state, attr)))
            except Exception:
                continue
            if not rows and tab not in self._last:
                # Never delete a stored queue this watcher has not itself written:
                # an unanswered restore offer's file must survive an empty start.
                continue
            try:
                blob = _json.dumps(rows, default=str, sort_keys=True)
            except Exception:
                continue
            if self._last.get(tab) == blob:
                continue
            proj = str(getattr(self._state, "current_project", "") or "")
            date = getattr(self._state, "current_date", None) or None
            if save(tab, rows, project=proj, date_layer=date,
                    app_version=self._app_version):
                self._last[tab] = blob

    def run_forever(self) -> None:  # pragma: no cover - thread loop
        import time
        while not self._stop:
            self.flush_once()
            time.sleep(self._interval)

    def start(self):
        import threading
        t = threading.Thread(target=self.run_forever, daemon=True,
                             name="pm-queue-watcher")
        t.start()
        return t
