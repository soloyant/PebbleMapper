"""Canonical per-project directory layout.

The canonical tree for a project named ``<proj>`` rooted at ``DATASETS_ROOT``
is::

    datasets/<proj>/
        input_data/                   ortho-images / quadrat photographs
        output_results/
            vectors/                  *.run.csv, paired .csv, run sidecars
                checkpoints/          partial-run snapshots
            rasters/                  rasterized statistics + sidecar JSON
            figures/                  diagnostic / publication figures
            maps/                     Map-tab PNG / PDF deliverables
            zonal/                    per-polygon / transect CSV + plots
            gauge/                    Digitize object-scale CSV, sidecar, figures
            reports/                  PDF reports + build manifests
            logs/                     build / run logs
        validation/
            images/                   ground-truth ortho / quadrat images
            results/                  paired validation CSVs, plots, JSON

``PATH_KINDS`` lists the legal ``kind`` values for ``project_path`` and
``default_starting_dir``. A legacy layout (``images/`` and ``results/`` in
place of ``input_data/`` and ``output_results/``) is resolved transparently
when it exists on disk; new projects get the canonical names.
"""
from __future__ import annotations

import json as _json
import os as _os
import re as _re
from pathlib import Path
from typing import Iterable

_SAFE_PROJECT_NAME_RE = _re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-. ]*$')
# Optional date level under a project: YYYY, YYYY-MM, or YYYY-MM-DD.
_SAFE_DATE_RE = _re.compile(r'^\d{4}(-\d{2}(-\d{2})?)?$')


def _validate_date(date: str) -> None:
    """Raise ValueError for a date-folder name that isn't a safe date token."""
    if not date:
        raise ValueError("Date must not be empty.")
    if not _SAFE_DATE_RE.match(str(date)):
        raise ValueError(
            f"Date {date!r} must be YYYY, YYYY-MM, or YYYY-MM-DD.")


def _validate_project_name(project: str) -> None:
    """Raise ValueError for project names that could escape DATASETS_ROOT."""
    if not project:
        raise ValueError("Project name must not be empty.")
    if ".." in project:
        raise ValueError(f"Project name {project!r} contains '..' (path traversal).")
    if any(c in project for c in ("/", "\\", "\x00")):
        raise ValueError(f"Project name {project!r} contains a path separator.")
    if not _SAFE_PROJECT_NAME_RE.match(project):
        raise ValueError(
            f"Project name {project!r} contains characters not allowed in a "
            "directory name. Use letters, digits, hyphens, underscores, dots, "
            "and spaces only."
        )

REPO_ROOT = Path(__file__).resolve().parent.parent

# Datasets root precedence: PEBBLEMAPPER_DATASETS_ROOT env var (CSM_DATASETS_ROOT
# as a legacy alias), then the root persisted in config.json, then
# <repo>/datasets. Runtime-mutable via set_datasets_root; readers consult the
# module global at call time.
_CONFIG_DIR = Path.home() / ".pebblemapper"


def app_home() -> Path:
    """The per-user application-state directory (config.json, persisted
    queues, crash breadcrumb). ``PEBBLEMAPPER_HOME`` overrides the default
    ``~/.pebblemapper``; resolved at call time so the override wins even
    when set after import."""
    env = _os.environ.get("PEBBLEMAPPER_HOME")
    if env:
        return Path(env)
    return _CONFIG_DIR


def _load_config() -> dict:
    try:
        return _json.loads((app_home() / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        home = app_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.json").write_text(_json.dumps(cfg, indent=2),
                                          encoding="utf-8")
    except Exception:
        pass  # config persistence is best-effort, never fatal


def _initial_datasets_root() -> Path:
    env_override = (_os.environ.get("PEBBLEMAPPER_DATASETS_ROOT")
                    or _os.environ.get("CSM_DATASETS_ROOT"))
    if env_override:
        return Path(env_override).resolve()
    saved = _load_config().get("datasets_root")
    if saved:
        try:
            return Path(saved).resolve()
        except Exception:
            pass
    return REPO_ROOT / "datasets"


DATASETS_ROOT = _initial_datasets_root()


def get_datasets_root() -> Path:
    """The current datasets root (runtime-mutable via ``set_datasets_root``)."""
    return DATASETS_ROOT


def set_datasets_root(path, persist: bool = True) -> Path:
    """Point the app at a new datasets root, immediately and (by default)
    persistently. An explicit ``PEBBLEMAPPER_DATASETS_ROOT`` env var still
    wins at the next startup."""
    global DATASETS_ROOT
    DATASETS_ROOT = Path(path).resolve()
    if persist:
        cfg = _load_config()
        cfg["datasets_root"] = str(DATASETS_ROOT)
        _save_config(cfg)
    return DATASETS_ROOT


# --- Optional per-project date level (multi-temporal analysis) ---
# A project is "flat" (subfolders under <project>/) or "dated" (under
# <project>/<date>/). project_path inserts the active date when the requested
# project matches the one it was set for; pairing date with project keeps a
# stale date from leaking onto a different project after a switch.
_ACTIVE_DATE = None
_ACTIVE_DATE_PROJECT = None


def set_active_date(date, project=None) -> None:
    """Set (or clear, with an empty ``date``) the active date for ``project``."""
    global _ACTIVE_DATE, _ACTIVE_DATE_PROJECT
    _ACTIVE_DATE = date or None
    _ACTIVE_DATE_PROJECT = project


def get_active_date():
    """The active date string, or None when the active project is flat."""
    return _ACTIVE_DATE

# Canonical (kind, subpath) registry: the single source of truth for kinds.
_KIND_SUBPATHS: dict[str, str] = {
    # inputs
    "images":               "input_data/images",
    "dem":                  "input_data/dem",
    "geometries":           "input_data/geometries",  # ROIs + zones/transects
    # outputs
    "vectors":              "output_results/vectors",
    "checkpoints":          "output_results/vectors/checkpoints",
    "rasters":              "output_results/rasters",
    "figures":              "output_results/figures",
    "maps":                 "output_results/maps",

    "zonal":                "output_results/zonal",
    "gauge":                "output_results/gauge",    # Digitize object-scale outputs
    "reports":              "output_results/reports",
    "logs":                 "output_results/logs",
    "detections":           "output_results/vectors",  # alias
    # validation tree
    "validation":           "validation",
    "validation_images":    "validation/images",   # legacy flat bucket
    "validation_results":   "validation/results",
    "validation_raw":          "validation/raw",           # unrectified photos
    "validation_rectified":    "validation/orthorectified",  # what Orthorectify writes
    "validation_georectified": "validation/georectified",  # georeferenced GeoTIFFs
    "validation_gps":          "validation/gps",           # GPS points / grid-photo geometry
    "validation_reports":   "validation/results",  # legacy alias
}

# Legacy subpath registry, used when the canonical path is absent on disk but
# the legacy one exists. Unchanged kinds (validation/*) mirror _KIND_SUBPATHS.
_LEGACY_KIND_SUBPATHS: dict[str, str] = {
    "images":               "images",
    "vectors":              "results/vectors",
    "checkpoints":          "results/vectors/checkpoints",
    "rasters":              "results/rasters",
    "figures":              "results/figures",
    "maps":                 "results/maps",
    "zonal":                "results/zonal",
    "gauge":                "results/gauge",
    "reports":              "results/reports",
    "logs":                 "results/logs",
    "detections":           "results/vectors",
    "dem":                  "input_data/dem",
    "geometries":           "input_data/geometries",
    "validation":           "validation",
    "validation_images":    "validation/images",
    "validation_results":   "validation/results",
    "validation_raw":          "validation/raw",
    "validation_rectified":    "validation/orthorectified",
    "validation_georectified": "validation/georectified",
    "validation_gps":          "validation/gps",
    "validation_reports":   "validation/results",
}

# Locations for the input kinds, canonical first; project_path returns the
# first that exists on disk, else the canonical entry.
_INPUT_KIND_CANDIDATES: dict[str, tuple[str, ...]] = {
    "images":     ("input_data/images", "input_data", "images"),
    "dem":        ("input_data/dem", "dem"),
    "geometries": ("input_data/geometries",),
}

PATH_KINDS: tuple[str, ...] = tuple(_KIND_SUBPATHS.keys())


def _base_uses_legacy_layout(base: Path) -> bool:
    """True when the tree at ``base`` (project or ``<project>/<date>`` root)
    uses the legacy folder names (``images/``, ``results/``)."""
    has_new = (base / "input_data").is_dir() or (base / "output_results").is_dir()
    if has_new:
        return False
    has_old = (base / "images").is_dir() or (base / "results").is_dir()
    return has_old




def project_path(project: str, kind: str = "", date=None) -> Path:
    """Absolute path of a per-project subfolder.

    ``project`` empty returns ``DATASETS_ROOT``; ``kind`` empty returns the
    project root, and an unknown kind falls back to the project root rather
    than raising. A legacy on-disk layout resolves to its legacy path.
    """
    if not project:
        return DATASETS_ROOT
    # Explicit date, else the active date if it was set for this project.
    eff_date = date if date is not None else (
        _ACTIVE_DATE if project == _ACTIVE_DATE_PROJECT else None)
    base = DATASETS_ROOT / project
    if eff_date:
        base = base / eff_date
    if not kind:
        return base
    # Input kinds: first existing location, else the canonical one.
    cands = _INPUT_KIND_CANDIDATES.get(kind)
    if cands is not None:
        for _c in cands:
            p = base / _c
            if p.is_dir():
                return p
        return base / cands[0]
    if _base_uses_legacy_layout(base):
        sub = _LEGACY_KIND_SUBPATHS.get(kind)
    else:
        sub = _KIND_SUBPATHS.get(kind)
    if sub is None:
        return base
    return base / sub


def list_projects() -> list[str]:
    """Sorted list of existing project directory names under DATASETS_ROOT."""
    if not DATASETS_ROOT.exists():
        return []
    return sorted([p.name for p in DATASETS_ROOT.iterdir()
                   if p.is_dir() and not p.name.startswith(".")])


def list_dates(project: str) -> list[str]:
    """Sorted date subfolders of ``project``; empty for a flat project."""
    if not project:
        return []
    base = DATASETS_ROOT / project
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir()
                   if p.is_dir() and _SAFE_DATE_RE.match(p.name)])


def is_dated_project(project: str) -> bool:
    """True when ``project`` has at least one date subfolder."""
    return bool(list_dates(project))


def ensure_project_layout(project: str,
                          extra_kinds: Iterable[str] = (),
                          date=None) -> None:
    """Create the canonical subfolders for a project. Idempotent.
    ``extra_kinds`` forces creation of optional folders (e.g. ``("zonal",)``);
    ``date`` materialises the tree under ``<project>/<date>/``."""
    _validate_project_name(project)
    base = DATASETS_ROOT / project
    if date:
        _validate_date(date)
        base = base / date
    core = (
        "images",
        "dem",
        "geometries",
        "vectors",
        "checkpoints",
        "rasters",
        "figures",
        "maps",
        "validation_raw",
        "validation_rectified",
        "validation_georectified",
        "validation_gps",
        "validation_results",
    )
    for kind in (*core, *extra_kinds):
        sub = _KIND_SUBPATHS.get(kind)
        if sub:
            (base / sub).mkdir(parents=True, exist_ok=True)
    _write_project_readme(base)


# Per-project guide written into each project root.
_PROJECT_README = """\
PebbleMapper project — where to put your files
==============================================

input_data/          everything YOU provide
  images/            imagery to analyse — quadrat photographs OR ortho GeoTIFFs
  dem/               optional elevation rasters (DEMs), GeoTIFF
  geometries/        optional regions of interest (Detection) and zones /
                     transects (Zonal), as GeoJSON or shapefile

output_results/      everything the tool GENERATES (you don't edit these)
  vectors/  rasters/  figures/  maps/  zonal/  gauge/  reports/  logs/

scaling_objects.json the scaling-object library of Digitize's object-scale
                     option (name and, when known, length); editable

validation/          ground-truth workflow (optional, separate from production)
  raw/               raw quadrat photographs (uncropped, unrectified)
  orthorectified/    rectified photographs (Orthorectify writes them here)
  georectified/      georeferenced GeoTIFFs
  gps/               GPS points / grid-photo geometry (GeoJSON or shapefile)
  results/           validation analysis output

Tip: every Browse… dialog in the app already defaults to the right folder here.
"""


def _write_project_readme(base: Path) -> None:
    """Write README.txt at the project (or date) root. Best-effort; never
    overwrites an existing file and never raises."""
    try:
        readme = base / "README.txt"
        if not readme.exists():
            readme.write_text(_PROJECT_README, encoding="utf-8")
    except Exception:
        pass


class Resolution:
    """The outcome of resolving a kind against a project root: unlike a bare
    ``Path`` it says whether the folder was found or guessed, so a reader
    cannot silently read nothing from a guessed path. ``path`` is always
    present for writers."""

    __slots__ = ("path", "how", "problem")

    def __init__(self, path: Path, how: str, problem: str = ""):
        self.path = path
        self.how = how          # canonical | legacy | input-candidate | fallback
        self.problem = problem  # "" | unknown-kind | unrecognised-layout

    @property
    def found(self) -> bool:
        """True when a folder was actually located on disk."""
        return self.how != "fallback"

    @property
    def unrecognised(self) -> bool:
        """True when the directory is a project-like tree in no known layout."""
        return self.problem == "unrecognised-layout"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Resolution(path={self.path!r}, how={self.how!r}, problem={self.problem!r})"


def _layout_is_recognised(project_root: Path) -> bool:
    """True when ``project_root`` is empty, absent, or in a known layout.
    Only a populated tree matching neither layout is unrecognised; an empty
    directory is a new project, and a writer needs the canonical target."""
    if not project_root.is_dir():
        return True
    if _base_uses_legacy_layout(project_root):
        return True
    if (project_root / "input_data").is_dir() or (project_root / "output_results").is_dir():
        return True
    if (project_root / "validation").is_dir():
        return True
    return not any(p.is_dir() for p in project_root.iterdir())


def resolve_project_subfolder_checked(project_root: Path, kind: str) -> Resolution:
    """Resolve a kind against a project root, saying how the answer was
    reached. Prefer this over :func:`resolve_project_subfolder` when reading:
    it distinguishes found from guessed and flags an unrecognised layout."""
    project_root = Path(project_root)
    canonical_sub = _KIND_SUBPATHS.get(kind)
    legacy_sub = _LEGACY_KIND_SUBPATHS.get(kind)

    if not canonical_sub and not legacy_sub:
        # An unknown kind is a caller bug, not a broken project on disk.
        return Resolution(project_root, "fallback", "unknown-kind")

    if canonical_sub:
        cand = project_root / canonical_sub
        if cand.exists() or canonical_sub == legacy_sub:
            # canonical_sub == legacy_sub: kinds with no fallback by design.
            how = "canonical" if cand.exists() else "fallback"
            problem = "" if cand.exists() else _unrecognised_or_blank(project_root)
            return Resolution(cand, how, problem)
        if legacy_sub:
            legacy_cand = project_root / legacy_sub
            if legacy_cand.exists():
                return Resolution(legacy_cand, "legacy")
        return Resolution(cand, "fallback", _unrecognised_or_blank(project_root))

    legacy_cand = project_root / legacy_sub
    if legacy_cand.exists():
        return Resolution(legacy_cand, "legacy")
    return Resolution(legacy_cand, "fallback", _unrecognised_or_blank(project_root))


def _unrecognised_or_blank(project_root: Path) -> str:
    return "" if _layout_is_recognised(project_root) else "unrecognised-layout"


def describe_unrecognised_layout(project_root: Path) -> str:
    """A refusal the user can act on: names the directory and what to do next."""
    return (
        f"{project_root} is not a project layout PebbleMapper recognises. "
        "Its folders match neither the current layout (input_data/, "
        "output_results/) nor the legacy one (images/, results/), so there is "
        "nothing to read. Migrate the project to the canonical layout and try "
        "again."
    )


def resolve_project_subfolder(project_root: Path, kind: str) -> Path:
    """Resolve a subfolder of an arbitrary project root (a Path, not a project
    name), canonical name first, legacy name second. Returns a bare ``Path``
    that falls back to the canonical target (which may not exist), so it
    cannot say whether the folder was found; for reads prefer
    :func:`resolve_project_subfolder_checked`."""
    return resolve_project_subfolder_checked(project_root, kind).path


def default_starting_dir(kind: str, project: str) -> str:
    """Seed directory for Browse... dialogs. Unlike ``project_path`` this
    creates the target for an existing project, since a file dialog silently
    ignores a nonexistent path; it never creates a project that does not
    exist."""
    if project:
        if project_path(project).is_dir():
            target = project_path(project, kind)
            target.mkdir(parents=True, exist_ok=True)
            return str(target)
        if DATASETS_ROOT.exists():
            return str(DATASETS_ROOT)
        return str(REPO_ROOT)
    if DATASETS_ROOT.exists():
        return str(DATASETS_ROOT)
    return str(REPO_ROOT)


# --- Legacy project migration ---

# Filename suffix -> kind; the first matching pattern wins.
_LEGACY_FILE_HINTS: tuple[tuple[str, str], ...] = (
    (".run.csv",        "vectors"),
    (".validation.csv", "validation_results"),
    (".validation.json","validation_results"),
    ("_paired.csv",     "validation_results"),
    (".tif",            "rasters"),
    (".tiff",           "rasters"),
    (".pdf",            "reports"),
    ("_map.png",        "maps"),
    ("_map.pdf",        "maps"),
    (".png",            "figures"),
    (".jpg",            "images"),
    (".jpeg",           "images"),
    (".heic",           "images"),
    (".heif",           "images"),
)


def migrate_legacy_project(project: str, dry_run: bool = True) -> list[str]:
    """Move loose files in a legacy project (root and ``results/``) into the
    canonical tree by the hint patterns above. Returns printable actions;
    with ``dry_run=True`` (default) nothing is moved. Files whose target
    already exists are skipped and reported with a "skip" prefix."""
    if not project:
        return []
    base = DATASETS_ROOT / project
    if not base.exists():
        return []

    known_subs = {
        (base / sub).resolve()
        for sub in _KIND_SUBPATHS.values()
    }

    actions: list[str] = []
    # Only the project root and results/, never the kind folders themselves.
    candidates = []
    for entry in base.iterdir():
        if entry.is_file():
            candidates.append(entry)
    results_dir = base / "results"
    if results_dir.is_dir():
        for entry in results_dir.iterdir():
            if entry.is_file():
                candidates.append(entry)

    for src in candidates:
        if any(p in known_subs for p in src.parents):
            continue
        name = src.name.lower()
        target_kind = None
        for suffix, kind in _LEGACY_FILE_HINTS:
            if name.endswith(suffix):
                target_kind = kind
                break
        if target_kind is None:
            continue
        dst_dir = project_path(project, target_kind)
        dst = dst_dir / src.name
        if dst.exists():
            actions.append(f"skip (target exists): {src} -> {dst}")
            continue
        if dry_run:
            actions.append(f"would move: {src} -> {dst}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            actions.append(f"moved: {src} -> {dst}")
    return actions


def _is_georectified_tif(path) -> bool:
    """True if a .tif/.tiff carries georeferencing (a projection or a
    non-identity geotransform). Best-effort; False if GDAL can't read it."""
    if Path(path).suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            return False
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        ds = None
        identity = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return bool(proj) or (gt is not None and tuple(gt) != identity)
    except Exception:
        return False


def migrate_validation_structure(project: str, dry_run: bool = True,
                                 date=None) -> list[str]:
    """Sort existing validation inputs into the structured subfolders. Only
    georeferenced GeoTIFFs are moved (to ``validation/georectified``); plain
    photos could be raw or rectified and are left for manual sorting. Dry-run
    by default; returns human-readable action strings."""
    if not project:
        return []
    val_root = project_path(project, "validation", date=date)
    if not val_root.exists():
        return []
    georect_dir = project_path(project, "validation_georectified", date=date)
    targets = {project_path(project, k, date=date).resolve()
               for k in ("validation_raw", "validation_rectified",
                         "validation_georectified", "validation_results")}

    actions: list[str] = []
    for d in (val_root, val_root / "images"):
        if not d.is_dir() or d.resolve() in targets:
            continue
        for src in sorted(d.iterdir()):
            if not src.is_file() or not _is_georectified_tif(src):
                continue
            dst = georect_dir / src.name
            if dst.exists():
                actions.append(f"skip (target exists): {src} -> {dst}")
                continue
            if dry_run:
                actions.append(f"would move: {src} -> {dst}")
            else:
                georect_dir.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                actions.append(f"moved: {src} -> {dst}")
    return actions


def migrate_input_structure(project: str, dry_run: bool = True,
                            date=None) -> list[str]:
    """Relocate existing inputs into the ``input_data/`` tree: loose imagery
    and ROI sidecars in ``input_data/``, a top-level ``dem/`` and a legacy
    ``images/`` folder. Dry-run by default; returns human-readable actions."""
    if not project:
        return []
    base = project_path(project, date=date)
    if not base.exists():
        return []
    images_dir = base / "input_data" / "images"
    dem_dir = base / "input_data" / "dem"
    geom_dir = base / "input_data" / "geometries"
    _img_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".heic", ".heif")
    actions: list[str] = []

    def _move(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        if dst.exists():
            actions.append(f"skip (target exists): {src} -> {dst}")
            return
        if dry_run:
            actions.append(f"would move: {src} -> {dst}")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            actions.append(f"moved: {src} -> {dst}")

    in_dir = base / "input_data"
    if in_dir.is_dir():
        for src in sorted(in_dir.iterdir()):
            if not src.is_file():
                continue
            if src.name.lower().endswith("_roi.geojson"):
                _move(src, geom_dir)
            elif src.suffix.lower() in _img_exts:
                _move(src, images_dir)
    legacy_images = base / "images"
    if (legacy_images.is_dir()
            and legacy_images.resolve() != images_dir.resolve()):
        for src in sorted(legacy_images.iterdir()):
            if src.is_file() and src.suffix.lower() in _img_exts:
                _move(src, images_dir)
    top_dem = base / "dem"
    if top_dem.is_dir() and top_dem.resolve() != dem_dir.resolve():
        for src in sorted(top_dem.iterdir()):
            if src.is_file():
                _move(src, dem_dir)
    return actions


__all__ = [
    "REPO_ROOT",
    "DATASETS_ROOT",
    "get_datasets_root",
    "set_datasets_root",
    "set_active_date",
    "get_active_date",
    "PATH_KINDS",
    "project_path",
    "list_projects",
    "list_dates",
    "is_dated_project",
    "ensure_project_layout",
    "default_starting_dir",
    "migrate_legacy_project",
    "migrate_validation_structure",
    "migrate_input_structure",
    "resolve_project_subfolder",
    "resolve_project_subfolder_checked",
    "Resolution",
    "describe_unrecognised_layout",
]
