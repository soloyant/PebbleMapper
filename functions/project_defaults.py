"""One resolver for "what should this field default to, given the active
project".

Answers, per field, with the newest sensible candidate (by mtime) and, where
unambiguous, a stem-matched partner (a raster's or CSV's source ortho). It
reads only and returns ``None``/empty when a project has nothing yet, so the
caller can say what the tab is waiting for. All paths resolve through
:func:`functions.layout.project_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from functions.layout import project_path
from functions import images as _images
from functions import naming

_IMAGE_EXTS = (".tif", ".tiff")
_PHOTO_EXTS = _images.CAMERA_EXTENSIONS       # .jpg .jpeg .png .heic .heif


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _dir(project: str, kind: str) -> Optional[Path]:
    if not project:
        return None
    try:
        d = project_path(project, kind)
        return d if d and Path(d).is_dir() else None
    except Exception:
        return None


def _list(project: str, kind: str, exts=None, predicate=None) -> list:
    return _list_path(_dir(project, kind), exts, predicate)


def _list_path(d, exts=None, predicate=None) -> list:
    if d is None or not Path(d).is_dir():
        return []
    d = Path(d)
    out = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        if predicate and not predicate(p):
            continue
        out.append(p)
    return sorted(out, key=_mtime, reverse=True)   # newest first


# --- Per-kind candidate lists (newest first) ---

def merged_csvs(project: str) -> list:
    """Merge-tool outputs whose name ends exactly in ``_merged.csv``, newest first.

    ``*_merged_individual_clast_values.csv`` and ``*_merged_zonal.csv`` also
    contain ``_merged`` but are not the per-clast merge a downstream run wants.
    """
    return _list(project, "vectors", (".csv",),
                 predicate=lambda p: p.name.endswith("_merged.csv"))


def detection_csvs(project: str) -> list:
    """Per-window and quadrat detection CSVs, newest first (the inputs a
    Zonal/Rasterize run consumes when no merge exists)."""
    return _list(project, "vectors", (".csv",),
                 predicate=lambda p: (naming.parse_window_size(p.name) is not None
                                      or p.name.endswith("_individual_clasts.csv"))
                 and not naming.is_merged(p.name))


def rasters(project: str) -> list:
    return _list(project, "rasters", (".tif", ".tiff"))


def orthos(project: str) -> list:
    return _list(project, "images", _IMAGE_EXTS)


def photos(project: str) -> list:
    return _list(project, "images", _PHOTO_EXTS)


def raw_photos(project: str) -> list:
    """Quadrat photographs as shot, newest first: ``validation/raw`` (the
    canonical place; Orthorectify writes beside it), else the photographs
    kept under ``input_data/images``."""
    return (_list(project, "validation_raw", _PHOTO_EXTS)
            or _list(project, "images", _PHOTO_EXTS))


def best_clast_csv(project: str) -> Optional[Path]:
    """The CSV a downstream tab should default to: newest merge if any, else
    newest detection CSV."""
    m = merged_csvs(project)
    if m:
        return m[0]
    d = detection_csvs(project)
    return d[0] if d else None


def source_ortho_for(name, project: str) -> Optional[Path]:
    """The ortho whose stem matches a CSV/raster name, else the newest ortho."""
    os_ = orthos(project)
    if not os_:
        return None
    stem = naming.image_stem(Path(str(name)).name)
    if stem:
        for o in os_:
            if naming.same_image(o.name, stem) or o.stem.startswith(stem) \
                    or stem.startswith(o.stem):
                return o
    return os_[0]


# --- Merge auto-fill ---

def window_csvs_by_stem(project: str) -> dict:
    """{image_stem: [(path, window_m), ...]} for every per-window detection CSV,
    sorted by window descending within a stem (the merge order)."""
    groups: dict = {}
    d = _dir(project, "vectors")
    if d is None:
        return groups
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if naming.is_merged(p.name) or "individual_clast_values" in p.name:
            continue
        w = naming.parse_window_size(p.name)
        if w is None:
            continue
        stem = naming.image_stem(p.name)
        groups.setdefault(stem, []).append((str(p), float(w)))
    for stem in groups:
        groups[stem].sort(key=lambda t: t[1], reverse=True)
    return groups


def mergeable_stems(project: str) -> list:
    """Stems that have ≥ 2 window CSVs (worth a merge), newest-activity first."""
    groups = window_csvs_by_stem(project)
    ready = [(s, rows) for s, rows in groups.items() if len(rows) >= 2]
    ready.sort(key=lambda sr: max(_mtime(Path(p)) for p, _ in sr[1]),
               reverse=True)
    return [s for s, _ in ready]


# --- Derivable inputs for Georeference, Validate and the Zonal profile ---
_ANY_IMAGE = _IMAGE_EXTS + _PHOTO_EXTS


def rectified_photos(project: str) -> list:
    """Rectified quadrat photographs, newest first: ``validation/orthorectified``
    (what Orthorectify writes beside ``validation/raw``), else
    ``input_data/images/orthorectified`` (what it writes when the raw
    photographs are kept under ``images/``). Empty when neither holds one."""
    out = _list(project, "validation_rectified", _ANY_IMAGE)
    if out:
        return out
    d = _dir(project, "images")
    return _list_path(d / "orthorectified", _ANY_IMAGE) if d is not None else []


def validation_images(project: str) -> list:
    """Quadrat photographs: the rectified ones first (what Detect and Digitize
    want), else the legacy validation/images bucket, else validation/."""
    return (rectified_photos(project)
            or _list(project, "validation_images", _ANY_IMAGE)
            or _list(project, "validation", _ANY_IMAGE))


def validation_csvs(project: str) -> list:
    """Hand-digitized truth CSVs (Digitize writes them under validation/)."""
    return (_list(project, "validation", (".csv",))
            + _list(project, "validation_images", (".csv",)))


def for_truth(truth_csv, candidates) -> Optional[Path]:
    """Among ``candidates`` (newest first), the one made on the photograph a
    truth CSV was digitised on. ``<stem>_truth.csv`` names ``<stem>``; the
    photograph is ``<stem>.<ext>`` and its detection CSV
    ``<project>__<stem>_individual_clasts.csv`` carries the stem in its name.
    None when there is no truth or nothing matches: the caller falls back to
    its usual newest-first choice."""
    if truth_csv is None or not candidates:
        return None
    stem = Path(truth_csv).stem
    if stem.endswith("_truth"):
        stem = stem[: -len("_truth")]
    if not stem:
        return None
    def _derived(p):
        n = Path(p).stem          # q1_georeferenced, <stem>_individual_clasts, P__<stem>_…
        return n.startswith(stem + "_") or ("__" + stem + "_") in n
    for p in candidates:
        if Path(p).stem == stem:
            return Path(p)
    for p in candidates:
        if _derived(p):
            return Path(p)
    return None


def georectified_images(project: str) -> list:
    """Placed quadrats (validation/georectified), newest first."""
    return _list(project, "validation_georectified", _ANY_IMAGE)


def transect_csvs(project: str) -> list:
    """Zonal transect samples (results/zonal/*.transects.csv), newest first."""
    return _list(project, "zonal", (".csv",),
                 predicate=lambda p: p.name.endswith(".transects.csv"))

