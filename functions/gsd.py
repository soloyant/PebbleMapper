"""Single reader for a quadrat photograph's ground-sample distance.

Orthorectification records the GSD in a JSON sidecar (``<image>.json`` with a
``PM_GSD`` key) and in the output filename (``..._GSD=0.003m.jpg``). A corrupt
or non-positive sidecar value yields a named warning plus an explicit fallback,
never a silent coercion.
"""

from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GsdInfo:
    """What we know about an image's own resolution.

    ``gsd`` in metres per pixel or None; ``source`` in {"sidecar", "filename",
    None}; ``warning`` a one-line human message when something looked like a
    GSD record but could not be trusted (the caller surfaces it once).
    """
    gsd: Optional[float]
    source: Optional[str]
    warning: Optional[str] = None


def parse_gsd_from_filename(filename) -> Optional[float]:
    """GSD from a filename, or None.

    Recognizes ``GSD=<value>m`` (e.g. GSD=0.005m) and the legacy
    ``GSD=<int>p<frac>mm`` (the 'p' replaces the decimal dot: GSD=2p100mm =
    2.100 mm = 0.0021 m).
    """
    name = Path(str(filename)).stem
    m = _re.search(r"GSD=(\d+(?:\.\d+)?)m(?:\b|_)", name)
    if m:
        try:
            v = float(m.group(1))
            return v if v > 0 else None
        except ValueError:
            pass
    m = _re.search(r"GSD=(\d+)p(\d+)mm(?:\b|_)", name)
    if m:
        try:
            mm = float(f"{m.group(1)}.{m.group(2)}")
            return mm / 1000.0 if mm > 0 else None
        except ValueError:
            pass
    return None


def sidecar_path(image_path) -> Path:
    return Path(str(image_path) + ".json")


def effective_gsd(image_path) -> GsdInfo:
    """The GSD this image carries about itself, with provenance.

    The sidecar's ``PM_GSD`` is authoritative when present and valid; the
    filename pattern is the fallback; otherwise (None, None) and the caller's
    global value applies.
    """
    sp = sidecar_path(image_path)
    if sp.exists():
        try:
            raw = _json.loads(sp.read_text(encoding="utf-8")).get("PM_GSD")
        except Exception as exc:
            return GsdInfo(
                parse_gsd_from_filename(image_path),
                "filename" if parse_gsd_from_filename(image_path) else None,
                f"{sp.name}: sidecar unreadable ({exc}) — "
                "falling back past it")
        if raw is not None:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = float("nan")
            if v > 0:
                return GsdInfo(v, "sidecar", None)
            return GsdInfo(
                None, None,
                f"{sp.name}: PM_GSD is {raw!r}, not a positive number — "
                "fix or re-rectify; using the global resolution for this "
                "file")
    v = parse_gsd_from_filename(image_path)
    if v:
        return GsdInfo(v, "filename", None)
    return GsdInfo(None, None, None)


def resolution_after_open(own_gsd, current, last_auto, typed):
    """What Digitize's Resolution field should hold once a photograph is
    open: ``(value, last_auto, typed)``.

    A photograph that names its GSD (sidecar or file name) fills the field
    and marks the value automatic. One that names none empties an automatic
    value — the boot default or another photograph's GSD, which a raw
    photograph would otherwise inherit in silence — and keeps
    a value the user typed, since a folder shot from one height is scaled
    once by hand.
    """
    if own_gsd:
        g = float(own_gsd)
        return g, g, False
    if typed:
        return current, last_auto, True
    return None, None, False
