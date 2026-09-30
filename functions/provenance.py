"""Which model, or which hand, produced each clast of a Digitize table.

The clast table keeps its 15 columns (tests/test_schema_contract.py); the
origin of each row lives in a sidecar beside the CSV,
``<csv stem>.provenance.json``::

    {"format": "pebblemapper-provenance", "version": 1,
     "csv": "IMG_4228_truth.csv", "image": "...", "written": "...",
     "models": [{"backend": "maskrcnn", "display_name": "Mask R-CNN ...",
                 "version": null, "tool_version": "...", "license": "...",
                 "weights": "...", "detect_scale": 0.5,
                 "min_confidence": 0.7, "date": "...", ...}],
     "counts": {"hand": 12, "Mask R-CNN ...": 312}, "edited": 4,
     "clasts": {"1": {"origin": "hand", "edited": false}, ...}}

``origin`` is :data:`HAND` for a clast drawn by the user, else the display
name of the model that proposed it; ``edited`` is true when a proposal's
geometry was changed by hand. A clast without an entry is a hand-drawn one.
Pure logic: no NiceGUI.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["HAND", "PROVENANCE_SUFFIX", "PROVENANCE_FORMAT",
           "provenance_path_for", "model_entry", "record_origin",
           "clast_origins", "origin_counts", "edited_counts", "model_names",
           "summary_line", "write_provenance", "read_provenance"]

HAND = "hand"
PROVENANCE_SUFFIX = ".provenance.json"
PROVENANCE_FORMAT = "pebblemapper-provenance"
PROVENANCE_VERSION = 1


def provenance_path_for(csv_path) -> Path:
    """``<csv stem>.provenance.json`` beside the CSV."""
    p = Path(str(csv_path))
    stem = p.name[:-4] if p.name.lower().endswith(".csv") else p.name
    return p.with_name(stem + PROVENANCE_SUFFIX)


def model_entry(backend: str, info=None, *, display_name: Optional[str] = None,
                detect_scale=None, min_confidence=None, date: Optional[str] = None,
                **extra) -> dict:
    """One model used on the photograph: the backend's name and, from its
    ``BackendInfo`` when given, display name, version, licence and weights,
    with the run's detection scale, minimum confidence and date."""
    try:
        from functions import __version__ as tool_version
    except Exception:  # pragma: no cover
        tool_version = "dev"
    entry = {
        "backend": str(backend),
        "display_name": str(display_name or getattr(info, "display_name", "")
                            or backend),
        "version": getattr(info, "version", None),
        "tool_version": tool_version,
        "license": getattr(info, "license", None),
        "weights": getattr(info, "weights", None),
        "detect_scale": (float(detect_scale) if detect_scale is not None
                         else None),
        "min_confidence": (float(min_confidence) if min_confidence is not None
                           else None),
        "date": date or datetime.now().isoformat(timespec="seconds"),
    }
    entry.update(extra)
    return entry


def record_origin(rec: Mapping) -> str:
    """A record's origin: its ``"origin"``, :data:`HAND` when it has none."""
    o = rec.get("origin") if isinstance(rec, Mapping) else None
    return str(o) if o else HAND


def clast_origins(records: Sequence[Mapping], positions: Iterable[int]) -> Dict[int, dict]:
    """``{clast_ID: {"origin", "edited"}}`` for a table whose row ``k``
    (``clast_ID`` k + 1) came from ``records[positions[k]]``."""
    out: Dict[int, dict] = {}
    for k, pos in enumerate(positions):
        rec = records[pos]
        origin = record_origin(rec)
        out[k + 1] = {"origin": origin,
                      "edited": bool(rec.get("edited")) and origin != HAND}
    return out


def origin_counts(clasts: Mapping) -> Dict[str, int]:
    """How many clasts per origin, hand-drawn first then models by count."""
    counts: Dict[str, int] = {}
    for v in clasts.values():
        o = str((v or {}).get("origin") or HAND)
        counts[o] = counts.get(o, 0) + 1
    ordered = {}
    if HAND in counts:
        ordered[HAND] = counts[HAND]
    for o, n in sorted(((o, n) for o, n in counts.items() if o != HAND),
                       key=lambda t: (-t[1], t[0])):
        ordered[o] = n
    return ordered


def edited_counts(clasts: Mapping) -> Dict[str, int]:
    """Edited proposals per model origin."""
    out: Dict[str, int] = {}
    for v in clasts.values():
        v = v or {}
        o = str(v.get("origin") or HAND)
        if o != HAND and v.get("edited"):
            out[o] = out.get(o, 0) + 1
    return out


def model_names(clasts: Mapping, models: Sequence[Mapping] = ()) -> List[str]:
    """The model display names behind the clasts, most clasts first; a model
    that ran but kept nothing is listed after them."""
    names = [o for o in origin_counts(clasts) if o != HAND]
    for m in models or []:
        n = str((m or {}).get("display_name") or "")
        if n and n not in names:
            names.append(n)
    return names


def summary_line(clasts: Mapping, models: Sequence[Mapping] = ()) -> str:
    """``"Detections: Segmenteverygrain (Sylvester), 312 kept, 4 edited;
    12 drawn by hand"``; ``"12 drawn by hand"`` with no model; ``"No
    clasts"`` for an empty table."""
    counts = origin_counts(clasts)
    edited = edited_counts(clasts)
    parts = []
    for name in model_names(clasts, models):
        n = counts.get(name, 0)
        txt = f"{name}, {n} kept"
        if edited.get(name):
            txt += f", {edited[name]} edited"
        parts.append(txt)
    line = ("Detections: " + "; ".join(parts)) if parts else ""
    if counts.get(HAND):
        hand = f"{counts[HAND]} drawn by hand"
        line = f"{line}; {hand}" if line else hand
    return line or "No clasts"


def write_provenance(csv_path, clasts: Mapping, models: Sequence[Mapping] = (),
                     *, image=None, extra: Optional[Mapping] = None) -> Optional[Path]:
    """Write the sidecar beside ``csv_path``; returns its path, or None when
    it could not be written (provenance never fails a save)."""
    try:
        body = {str(int(k)): {"origin": str((v or {}).get("origin") or HAND),
                              "edited": bool((v or {}).get("edited"))}
                for k, v in clasts.items()}
        doc = {"format": PROVENANCE_FORMAT, "version": PROVENANCE_VERSION,
               "csv": Path(str(csv_path)).name,
               "image": str(image) if image else None,
               "written": datetime.now().isoformat(timespec="seconds"),
               "models": [dict(m) for m in (models or [])],
               "counts": origin_counts(clasts),
               "edited": sum(1 for v in body.values() if v["edited"])}
        if extra:
            doc.update(dict(extra))
        doc["clasts"] = body
        out = provenance_path_for(csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, out)
        return out
    except Exception:
        return None


def read_provenance(csv_path) -> Optional[dict]:
    """The sidecar beside ``csv_path`` with ``clasts`` keyed by int, or None
    when there is none or it is unreadable."""
    try:
        p = provenance_path_for(csv_path)
        if not p.is_file():
            return None
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return None
        raw = doc.get("clasts") if isinstance(doc.get("clasts"), dict) else {}
        clasts = {}
        for k, v in raw.items():
            try:
                clasts[int(k)] = {"origin": str((v or {}).get("origin") or HAND),
                                  "edited": bool((v or {}).get("edited"))}
            except (TypeError, ValueError):
                continue
        doc["clasts"] = clasts
        doc["models"] = [m for m in (doc.get("models") or []) if isinstance(m, dict)]
        return doc
    except Exception:
        return None
