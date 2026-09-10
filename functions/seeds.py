"""Where each quadrat was, so the matcher knows where to look.

Reads the file a field GPS or a notebook already produces (a CSV pairing a
photo with a coordinate), forgiving about how it is spelled. It does not
guess: a row it cannot parse is reported with its line number and skipped,
and a coordinate that lands nowhere near the ortho is called out.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["Seed", "SeedTable", "load_seed_table", "seed_for_photo",
           "ResolvedSeed", "resolve_seed"]

# Header spellings seen in the wild. Order matters only for reporting.
# Columns that name the photograph, most specific first: a survey export
# carries the quadrat's label in 'name' (Q25) AND the file in 'photograph'
# (IMG_0955.JPG); keyed by the label, no photo ever matched.
_PHOTO_KEYS = ("photo", "photograph", "picture", "image", "img", "file",
               "filename", "name", "quadrat")
_X_KEYS = ("x", "lon", "long", "longitude", "easting", "east", "e")
_Y_KEYS = ("y", "lat", "latitude", "northing", "north", "n")
_CRS_KEYS = ("crs", "epsg", "srs")


@dataclass(frozen=True)
class Seed:
    """One quadrat's approximate position, as supplied."""
    photo: str
    x: float
    y: float
    crs: str = ""
    line: int = 0

    @property
    def key(self) -> str:
        return normalise_photo(self.photo)


@dataclass
class SeedTable:
    """Everything a seed file yielded, including what it could not."""
    seeds: Dict[str, Seed]
    problems: List[str]
    columns: Tuple[str, str, str]        # the photo/x/y headers actually used
    default_crs: str = ""

    def __len__(self) -> int:
        return len(self.seeds)

    def get(self, photo) -> Optional[Seed]:
        """The seed for a photo, exact name first, then its base name.

        Rectified outputs carry the GSD in their name, so re-rectifying
        renames every file; the base name (``DJI_0904``) is the stable
        identity of a quadrat across rectifications.
        """
        exact = self.seeds.get(normalise_photo(photo))
        if exact is not None:
            return exact
        want = _base_photo(photo)
        if not want:
            return None
        for key, seed in self.seeds.items():
            if _base_photo(key) == want:
                return seed
        return None


def normalise_photo(name) -> str:
    """Compare photos by name, ignoring folder, extension and case."""
    stem = Path(str(name or "").strip()).name
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem.strip().lower()


from functions.naming import TOOL_SUFFIXES as _TOOL_SUFFIXES


def _base_photo(name) -> str:
    """``dji_0904`` from any rectification of it."""
    stem = normalise_photo(name)
    for suf in _TOOL_SUFFIXES:
        i = stem.find(suf)
        if i > 0:
            return stem[:i]
    return stem


def _pick(header: List[str], candidates) -> Optional[str]:
    lowered = {h.strip().lower(): h for h in header}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _to_float(text, decimal: str = "") -> Optional[float]:
    """Parse a coordinate, tolerating a comma decimal separator.

    With ``decimal`` unset a comma is a decimal point only when there is no
    dot to contradict it. Stated explicitly, the given symbol wins and the
    other is a thousands separator, the only way to read "1.234,56" correctly.
    """
    if text is None:
        return None
    s = str(text).strip().replace(" ", "")
    if not s:
        return None
    if decimal == ",":
        s = s.replace(".", "").replace(",", ".")
    elif decimal == ".":
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# Separators offered to the user; "auto" sniffs.
SEPARATORS = {"comma": ",", "semicolon": ";", "tab": "\t", "pipe": "|"}


def _read_rows(path, separator: str = "") -> Tuple[List[List[str]], str]:
    """Rows of the file, plus a note about how it was split.

    ``separator`` is a literal character, one of the :data:`SEPARATORS` names,
    or empty/``auto`` to sniff.
    """
    name = str(separator or "").strip().lower()
    sep = "" if name in ("", "auto") else SEPARATORS.get(name, str(separator))
    if len(sep) > 1:                     # a name nobody knows; sniff instead
        sep = ""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    note = ""
    if not sep:
        head = text.split("\n", 1)[0]
        counts = {c: head.count(c) for c in (",", ";", "\t", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) else ","
        name = {v: k for k, v in SEPARATORS.items()}.get(sep, sep)
        note = f"Columns split on the detected separator ({name})."
    return list(csv.reader(io.StringIO(text), delimiter=sep)), note


def load_seed_table(path, default_crs: str = "", *, separator: str = "comma",
                    has_header: bool = True, decimal: str = "") -> SeedTable:
    """Read a photo-to-coordinate CSV.

    Never raises on content: an unreadable file, a missing header or a
    malformed row all come back as problems to report.

    ``separator`` takes a :data:`SEPARATORS` name, a literal character, or
    ``"auto"`` to sniff. ``decimal`` empty guesses; ``"."`` or ``","`` states
    it, which is the only way to read ``1.234,56`` correctly. Without a header
    the first three columns are taken as photo, x, y (x first, the GIS order)
    and a note says so, because a transposed coordinate searches hundreds of
    kilometres away while looking reasonable.
    """
    problems: List[str] = []
    seeds: Dict[str, Seed] = {}
    p = Path(path)
    try:
        rows, sep_note = _read_rows(p, separator)
    except (OSError, UnicodeDecodeError) as ex:
        return SeedTable({}, [f"Could not read {p.name}: {ex}"], ("", "", ""),
                         default_crs)
    if sep_note:
        problems.append(sep_note)
    if not rows:
        return SeedTable({}, [f"{p.name} is empty."], ("", "", ""), default_crs)

    if has_header:
        header = [h.strip() for h in rows[0]]
        body = rows[1:]
        first_line = 2
    else:
        # No names to match on, so position decides; said explicitly, since
        # x and y the wrong way round is the failure this module exists to prevent.
        header = (["photo", "x", "y"]
                  + [f"col{i}" for i in range(3, len(rows[0]))])
        body = rows
        first_line = 1
        problems.append(
            "No header row, so the first three columns were read as photo, x "
            "(longitude/easting), y (latitude/northing) in that order.")
    ph_col = _pick(header, _PHOTO_KEYS)
    x_col = _pick(header, _X_KEYS)
    y_col = _pick(header, _Y_KEYS)
    crs_col = _pick(header, _CRS_KEYS)
    missing = [n for n, c in (("photo", ph_col), ("x/longitude", x_col),
                              ("y/latitude", y_col)) if c is None]
    if missing:
        return SeedTable(
            {}, problems
            + [f"{p.name} has no {' and no '.join(missing)} column. "
               f"Found: {', '.join(header) or '(no header)'}."
               + (" If the columns look merged, the separator is wrong."
                  if len(header) <= 1 else "")],
            ("", "", ""), default_crs)

    idx = {h: i for i, h in enumerate(header)}
    for n, row in enumerate(body, start=first_line):
        if not any(str(c).strip() for c in row):
            continue
        def cell(col):
            i = idx.get(col, -1)
            return row[i] if 0 <= i < len(row) else ""
        photo = str(cell(ph_col)).strip()
        if not photo:
            problems.append(f"line {n}: no photo name; skipped.")
            continue
        x = _to_float(cell(x_col), decimal)
        y = _to_float(cell(y_col), decimal)
        if x is None or y is None:
            problems.append(
                f"line {n} ({photo}): could not read a coordinate from "
                f"{cell(x_col)!r}, {cell(y_col)!r}; skipped.")
            continue
        crs = str(cell(crs_col)).strip() if crs_col else ""
        if crs and crs.isdigit():
            crs = f"EPSG:{crs}"
        key = normalise_photo(photo)
        if key in seeds:
            problems.append(
                f"line {n}: {photo} appears more than once; the last wins.")
        seeds[key] = Seed(photo=photo, x=x, y=y, crs=crs or default_crs,
                          line=n)
    return SeedTable(seeds, problems, (ph_col, x_col, y_col), default_crs)


def seed_for_photo(table: Optional[SeedTable], photo,
                   pins: Optional[dict] = None) -> Optional[Seed]:
    """The seed to use for one photo: a pin (just placed on the ortho) wins
    over the file."""
    key = normalise_photo(photo)
    if pins:
        pinned = pins.get(key)
        if pinned is not None:
            x, y, crs = (list(pinned) + [""])[:3]
            return Seed(photo=str(photo), x=float(x), y=float(y),
                        crs=str(crs or ""), line=0)
    if table is None:
        return None
    return table.get(photo)


@dataclass(frozen=True)
class ResolvedSeed:
    """Where to search, and what said so."""
    x: Optional[float]
    y: Optional[float]
    crs: str
    provenance: str            # "a pin", "the seed list", ... or a refusal
    ok: bool = True


def resolve_seed(photo, table=None, pins=None, typed=(None, None),
                 typed_crs: str = "", search_dirs=(), bounds=None,
                 crs_transform=None) -> ResolvedSeed:
    """Pin, then seed list, then typed coordinate, then the photograph's own fix.

    Resolved when a search runs, not when a button is pressed: "Use the seed
    list / pin" copies its coordinate into the typed boxes, so provenance has
    to be re-derived from the sources each time.

    A typed coordinate counts as supplied only when it is neither ``(0, 0)``
    nor empty: the boxes default to ``0.0``, which is the sentinel for "pin it
    instead", and a cleared NiceGUI number field is ``None``.
    """
    name = Path(str(photo or "")).name
    seed = seed_for_photo(table, name, pins or None)
    if seed is not None:
        src = ("a pin" if pins and normalise_photo(name) in pins
               else "the seed list")
        return ResolvedSeed(float(seed.x), float(seed.y),
                            str(seed.crs or typed_crs), src)

    # `list(None)` raises; a resolver must refuse, not traceback, on a missing coordinate.
    try:
        tx, ty = (list(() if typed is None else typed) + [None, None])[:2]
    except TypeError:
        tx, ty = None, None
    try:
        supplied = (tx is not None and ty is not None
                    and not (abs(float(tx)) < 1e-12 and abs(float(ty)) < 1e-12))
    except (TypeError, ValueError):
        supplied = False
    if supplied:
        return ResolvedSeed(float(tx), float(ty), typed_crs,
                            "the coordinates you typed")

    from functions import exif_seed as _xs
    fix, why = _xs.seed_from_photo(photo, search_dirs=search_dirs,
                                   bounds=bounds, crs_transform=crs_transform)
    if fix is not None:
        return ResolvedSeed(
            float(fix["lon"]), float(fix["lat"]), "EPSG:4326",
            f"the photograph's own GPS fix ({Path(fix['source']).name})")
    return ResolvedSeed(None, None, typed_crs, why or "nothing", ok=False)


def outside_footprint(x, y, bounds) -> bool:
    """True when a seed lands outside the ortho.

    Latitude and longitude the wrong way round put the seed somewhere
    impossible, and saying so beats "no match found".
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    except (TypeError, ValueError):
        return False
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    return not (lo_x <= float(x) <= hi_x and lo_y <= float(y) <= hi_y)


# --- Checking a whole survey at once ---
@dataclass
class QuadratCheck:
    """What happened when one quadrat was looked for."""
    photo: str
    status: str            # located | not_located | no_seed | already_placed
                           # | unreadable | stopped
    detail: str = ""
    n_inliers: int = 0
    residual_m: float = float("nan")
    seed_source: str = ""  # pin | list | (none)
    # The placement itself, kept so a user who then opens one quadrat does not
    # pay the match cost again. None for a refusal, as match_quadrat returns it.
    matrix: Optional[list] = None
    seed_world: Optional[Tuple[float, float]] = None
    # Placements the matcher found and could not choose between, best first:
    # the one failure a person can usually settle by looking at the ortho.
    candidates: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("located", "already_placed")


def check_quadrats(photo_paths, ortho_path, table=None, pins=None,
                   *, quadrat_gsd_m=None, search_radius_m=None,
                   frame_inset_m=None, settings=None, log_fn=None,
                   progress_fn=None) -> List[QuadratCheck]:
    """Ask, for each photo, whether it can be placed in the ortho.

    Reports every photo and never stops at a failure. Writes nothing.
    """
    from functions import georef as _gr
    import numpy as _np

    out: List[QuadratCheck] = []
    ortho_wkt = _gr.ortho_crs(ortho_path)
    bounds = _ortho_bounds(ortho_path)

    for path in list(photo_paths):
        p = Path(path)
        if progress_fn:
            try:
                progress_fn(p.name)
            except Exception:
                pass
        placed = _gr.describe_georeferencing(p)
        if placed.get("georeferenced"):
            out.append(QuadratCheck(p.name, "already_placed",
                                    placed.get("reason", "")))
            continue
        # The same order the single-quadrat path uses: pin, seed list, then
        # the photograph's own GPS fix.
        r = resolve_seed(p, table=table, pins=pins, typed=(None, None),
                         search_dirs=[p.parent])
        if not r.ok:
            out.append(QuadratCheck(
                p.name, "no_seed",
                "No position for this photo, so there is nowhere to start "
                "looking: " + (r.provenance or "none found") +
                " Add it to the seed list, pin it on the ortho, or keep the "
                "original photograph in a raw/ folder beside this one."))
            continue
        source = ("pin" if r.provenance == "a pin" else
                  "list" if r.provenance == "the seed list" else
                  "photo" if r.provenance.startswith("the photograph")
                  else "typed")
        sx, sy, note = _gr.transform_seed(
            r.x, r.y, r.crs or _gr.DEFAULT_SEED_CRS, ortho_wkt)
        if bounds and outside_footprint(sx, sy, bounds):
            out.append(QuadratCheck(
                p.name, "not_located",
                "The given position is outside this ortho-image. Check the "
                "coordinate and its CRS — latitude and longitude the wrong "
                "way round land exactly like this."
                + (f" ({note})" if note else ""),
                seed_source=source))
            continue
        try:
            from PIL import Image as _PI
            with _PI.open(str(p)) as im:
                quad = _np.asarray(im.convert("L"), dtype=float)
        except Exception as ex:
            out.append(QuadratCheck(p.name, "unreadable",
                                    f"Could not read the image: {ex}",
                                    seed_source=source))
            continue
        try:
            m = _gr.locate_quadrat(
                quad, *_window_reader(ortho_path),
                quadrat_gsd_m=(quadrat_gsd_m
                               or _gr.describe_georeferencing(p).get("gsd_m")
                               or 0.001),
                ortho_gsd_m=_ortho_gsd(ortho_path),
                seed_xy=(sx, sy),
                ortho_origin_xy=_ortho_origin(ortho_path),
                search_radius_m=search_radius_m,
                # Cap the widening: without it plan_search doubles the radius on
                # failure until it covers the entire ortho.
                max_radius_m=search_radius_m,
                frame_inset_m=frame_inset_m, settings=settings,
                log_fn=log_fn)
        except Exception as ex:
            out.append(QuadratCheck(p.name, "unreadable",
                                    f"The search failed: {ex}",
                                    seed_source=source))
            continue
        q = m.quality
        mat = (None if m.matrix is None
               else [list(map(float, row)) for row in m.matrix])
        if m.accepted:
            out.append(QuadratCheck(
                p.name, "located",
                f"{q.n_inliers} of {q.n_correspondences} correspondences "
                f"agreed; {q.residual_m * 100:.1f} cm mean error.",
                q.n_inliers, q.residual_m, source, mat, (sx, sy)))
        else:
            out.append(QuadratCheck(
                p.name, "not_located", " ".join(q.reasons),
                q.n_inliers, q.residual_m, source, mat, (sx, sy)))
    return out


def _ortho_ds(ortho_path):
    from osgeo import gdal
    return gdal.Open(str(ortho_path))


def _ortho_gsd(ortho_path) -> float:
    ds = _ortho_ds(ortho_path)
    if ds is None:
        return 0.0
    gt = ds.GetGeoTransform()
    ds = None
    return abs(float(gt[1])) if gt else 0.0


def _ortho_origin(ortho_path):
    ds = _ortho_ds(ortho_path)
    if ds is None:
        return (0.0, 0.0)
    gt = ds.GetGeoTransform()
    ds = None
    return (float(gt[0]), float(gt[3])) if gt else (0.0, 0.0)


def _ortho_bounds(ortho_path):
    ds = _ortho_ds(ortho_path)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    ds = None
    if not gt:
        return None
    return (gt[0], gt[3] + h * gt[5], gt[0] + w * gt[1], gt[3])


def _window_reader(ortho_path):
    """(read_window, shape) for one ortho, reading windows on demand."""
    ds = _ortho_ds(ortho_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open the ortho: {ortho_path}")
    shape = (ds.RasterYSize, ds.RasterXSize)
    ds = None

    def read(r0, c0, r1, c1):
        import numpy as _np
        d = _ortho_ds(ortho_path)
        if d is None:
            return None
        arr = d.ReadAsArray(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
        d = None
        if arr is not None and getattr(arr, "ndim", 2) == 3:
            arr = _np.moveaxis(arr, 0, -1)
        return arr
    return read, shape


def _quadrat_gsd(path, fallback=None) -> float:
    """The scale of ONE quadrat, preferring what the file itself says.

    A rectified image's GSD is in its geotransform or in the name the tool
    wrote (``<stem>_rectified_GSD=0.000641m.png``); a survey rectified with
    automatic GSD has a different value per photograph, so one typed figure
    cannot fit them all within `scale_tolerance`.
    """
    try:
        from functions.quadrat_validation import detect_gsd_from_path
        got = detect_gsd_from_path(str(path)).get("gsd")
        if got and float(got) > 0:
            return float(got)
    except Exception:
        pass
    try:
        if fallback and float(fallback) > 0:
            return float(fallback)
    except (TypeError, ValueError):
        pass
    return 0.001


# --- A survey, tile by tile rather than quadrat by quadrat ---
def check_survey(photo_paths, ortho_path, table=None, pins=None, *,
                 quadrat_gsd_m=None, search_radius_m=None, frame_inset_m=None,
                 settings=None, log_fn=None, progress_fn=None,
                 should_stop=None, full_grid=False) -> List[QuadratCheck]:
    """Check a whole survey, reading each ortho tile once.

    Walks TILE-major over `georef.tile_grid`, one grid fixed to the ortho, so
    quadrats near one another ask for the same rectangle and it is read once
    (quadrat-major reading pulls the same ground off disk repeatedly).
    Preserved from the per-quadrat path: nearest-first per quadrat, early
    exit once a quadrat places, and every gate of `match_quadrat`.

    ``progress_fn(done, total, message)`` is called as quadrats resolve and
    ``should_stop()`` is polled between rounds.

    ``full_grid`` sweeps the whole ortho for quadrats that have no seed at
    all. It is off unless asked for, being the most expensive thing this tool
    can do; without it an unseeded quadrat is reported as ``no_seed``. When
    on, every unseeded quadrat walks the same grid in the same order, so each
    tile is read once and matched against all of them.
    """
    from functions import georef as _gr
    import numpy as _np
    from PIL import Image as _PI

    cfg = dict(_gr.DEFAULTS)
    if settings:
        cfg.update(settings)
    radius = float(search_radius_m if search_radius_m is not None
                   else cfg["search_radius_m"])

    read_window, ortho_shape = _window_reader(ortho_path)
    gsd = _ortho_gsd(ortho_path)
    ox, oy = _ortho_origin(ortho_path)
    ortho_wkt = _gr.ortho_crs(ortho_path)
    bounds = _ortho_bounds(ortho_path)

    paths = [Path(p) for p in photo_paths]
    total = len(paths)
    out: Dict[str, QuadratCheck] = {}
    pending: List[dict] = []

    def tell(msg):
        if progress_fn:
            try:
                progress_fn(len(out), total, msg)
            except Exception:
                pass

    # --- pass 1: resolve seeds and plan tiles. No pixels read yet. ---
    for p in paths:
        tell(f"reading {p.name}")
        placed = _gr.describe_georeferencing(p)
        if placed.get("georeferenced"):
            out[p.name] = QuadratCheck(p.name, "already_placed",
                                       placed.get("reason", ""))
            continue
        r = resolve_seed(p, table=table, pins=pins, typed=(None, None),
                         search_dirs=[p.parent])
        if not r.ok and not full_grid:
            # A whole-ortho sweep is never started without being asked for.
            out[p.name] = QuadratCheck(
                p.name, "no_seed",
                "No position for this photo, so there is nowhere to start "
                "looking: " + (r.provenance or "none found") +
                " Add it to the seed list, pin it on the ortho, keep the "
                "original photograph in a raw/ folder beside this one, or ask "
                "for a whole-ortho search.")
            continue
        if not r.ok:
            # Asked for: the quadrat joins the sweep with no seed of its own.
            try:
                with _PI.open(str(p)) as im:
                    quad = _np.asarray(im.convert("L"), dtype=float)
            except Exception as ex:
                out[p.name] = QuadratCheck(p.name, "unreadable",
                                           f"Could not read the image: {ex}")
                continue
            qgsd = _quadrat_gsd(p, quadrat_gsd_m)
            pending.append(dict(
                path=p, quad=quad, qgsd=qgsd, seed=None, source="grid",
                tiles=list(_gr.tile_grid(ortho_shape, gsd,
                                         max(quad.shape) * qgsd)),
                at=0, last=None))
            continue
        source = ("pin" if r.provenance == "a pin" else
                  "list" if r.provenance == "the seed list" else
                  "photo" if r.provenance.startswith("the photograph")
                  else "typed")
        sx, sy, note = _gr.transform_seed(
            r.x, r.y, r.crs or _gr.DEFAULT_SEED_CRS, ortho_wkt)
        if bounds and outside_footprint(sx, sy, bounds):
            out[p.name] = QuadratCheck(
                p.name, "not_located",
                "The given position is outside this ortho-image. Check the "
                "coordinate and its CRS - latitude and longitude the wrong "
                "way round land exactly like this."
                + (f" ({note})" if note else ""), seed_source=source)
            continue
        try:
            with _PI.open(str(p)) as im:
                quad = _np.asarray(im.convert("L"), dtype=float)
        except Exception as ex:
            out[p.name] = QuadratCheck(p.name, "unreadable",
                                       f"Could not read the image: {ex}",
                                       seed_source=source)
            continue
        qgsd = _quadrat_gsd(p, quadrat_gsd_m)
        extent = max(quad.shape) * qgsd
        seed_px = (int(round((oy - sy) / gsd)), int(round((sx - ox) / gsd)))
        pending.append(dict(
            path=p, quad=quad, qgsd=qgsd, seed=(sx, sy), source=source,
            tiles=_gr.tiles_for_seed(ortho_shape, gsd, extent, seed_px, radius),
            at=0, last=None))

    # --- pass 2: rounds. Each round groups every quadrat's NEXT tile, so a
    # tile several quadrats want is read once and matched several times. ---
    stopped = False
    while pending and not stopped:
        if should_stop is not None:
            try:
                stopped = bool(should_stop())
            except Exception:
                stopped = False
        if stopped:
            for job in pending:
                # Its own status: reported as "not located" it was
                # indistinguishable from a quadrat the matcher rejected.
                out.setdefault(job["path"].name, QuadratCheck(
                    job["path"].name, "stopped",
                    "Stopped before this quadrat was checked.",
                    seed_source=job["source"]))
            break

        by_tile: Dict[tuple, List[dict]] = {}
        for job in pending:
            if job["at"] < len(job["tiles"]):
                by_tile.setdefault(job["tiles"][job["at"]], []).append(job)
        if not by_tile:
            break

        for tile, jobs in by_tile.items():
            r0, c0, r1, c1 = tile
            try:
                win = read_window(r0, c0, r1, c1)
            except Exception as ex:
                if log_fn:
                    log_fn(f"could not read tile {tile}: {ex}")
                win = None
            usable = win is not None and _np.asarray(win).size > 0
            for job in jobs:
                job["at"] += 1
                if not usable or job["path"].name in out:
                    continue
                # A quadrat swept without a seed has no centre to judge against,
                # so the radius gate must not be applied to it.
                m = _gr.match_quadrat(
                    job["quad"], win, quadrat_gsd_m=job["qgsd"],
                    ortho_gsd_m=gsd,
                    window_origin_xy=(ox + c0 * gsd, oy - r0 * gsd),
                    seed_xy=job["seed"],
                    search_radius_m=(radius if job["seed"] is not None
                                     else None),
                    frame_inset_m=frame_inset_m, settings=cfg)
                job["last"] = m
                if m.accepted:
                    q = m.quality
                    mat = (None if m.matrix is None
                           else [list(map(float, row)) for row in m.matrix])
                    out[job["path"].name] = QuadratCheck(
                        job["path"].name, "located",
                        f"{q.n_inliers} of {q.n_correspondences} "
                        f"correspondences agreed; "
                        f"{q.residual_m * 100:.1f} cm mean error.",
                        q.n_inliers, q.residual_m, job["source"], mat,
                        job["seed"])
                    # "placed" / "no match": "located" / "not located" differ by
                    # one leading word and misread at a glance.
                    tell(f"✓ placed {job['path'].name}")

        still = []
        for job in pending:
            if job["path"].name in out:
                continue
            if job["at"] >= len(job["tiles"]):
                m = job["last"]
                q = m.quality if m is not None else None
                out[job["path"].name] = QuadratCheck(
                    job["path"].name, "not_located",
                    " ".join(q.reasons) if q is not None and q.reasons
                    else "No tile in the search area produced a usable match.",
                    q.n_inliers if q is not None else 0,
                    q.residual_m if q is not None else float("nan"),
                    job["source"], None, job["seed"],
                    # Carry the rival placements out with the refusal.
                    list(getattr(q, "candidates", []) or []) if q else [])
                tell(f"✗ no match for {job['path'].name}")
                continue
            still.append(job)
        pending = still

    return [out[p.name] for p in paths if p.name in out]


def save_located(results, photo_dir, ortho_path, out_dir, *,
                 quadrat_gsd_m=None, overwrite=False, progress_fn=None):
    """Write a GeoTIFF for every quadrat the survey placed.

    Returns ``{"written": [...], "skipped": [(photo, why), ...]}``.

    Refusals are never written: a quadrat the matcher declined is an absence
    of evidence, and a GeoTIFF of it could not be told from a measured one.
    """
    from functions import georef as _gr
    import numpy as _np
    from PIL import Image as _PI

    photo_dir = Path(photo_dir)
    out_dir = Path(out_dir)
    ogsd = _ortho_gsd(ortho_path)
    wkt = _gr.ortho_crs(ortho_path)
    written, skipped = [], []
    todo = [r for r in results if r.status == "located" and r.matrix]
    for i, r in enumerate(todo):
        if progress_fn:
            try:
                progress_fn(i, len(todo), f"saving {r.photo}")
            except Exception:
                pass
        p = photo_dir / r.photo
        if not p.is_file():
            skipped.append((r.photo, "the photograph is no longer there"))
            continue
        try:
            with _PI.open(str(p)) as im:
                arr = _np.asarray(im.convert("RGB"))
        except Exception as ex:
            skipped.append((r.photo, f"could not read it: {ex}"))
            continue
        q = _gr.MatchQuality(
            n_inliers=int(r.n_inliers or 0),
            residual_m=float(r.residual_m) if r.residual_m == r.residual_m
            else float("nan"),
            accepted=True)
        m = _gr.QuadratMatch(_np.asarray(r.matrix, dtype=float), q,
                             r.seed_world, _quadrat_gsd(p, quadrat_gsd_m),
                             ogsd)
        try:
            got = _gr.write_georeferenced(
                m, arr, out_dir, Path(r.photo).stem, crs_wkt=wkt,
                source_files={"quadrat": str(p), "ortho": str(ortho_path)},
                overwrite=overwrite)
            written.append(str(got.get("geotiff", "")))
        except FileExistsError:
            skipped.append((r.photo, "an output already exists — "
                                     "tick overwrite to replace it"))
        except Exception as ex:
            skipped.append((r.photo, f"{type(ex).__name__}: {ex}"))
    if progress_fn:
        try:
            progress_fn(len(todo), len(todo), "done")
        except Exception:
            pass
    return {"written": written, "skipped": skipped}
