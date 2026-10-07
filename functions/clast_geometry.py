"""Clast geometry shared by every drawer: the orientation convention, the
axis chords, the ellipse outline, and the contour sidecar next to a CSV.

The orientation convention (pinned by tests/test_clast_geometry.py)
--------------------------------------------------------------------
``Orientation`` is the **bearing of the clast's long axis: degrees clockwise
from image-up, axial on [0, 180)**. Image-up is grid north for an ortho
(world coordinates, y up) and the top of the photograph for a quadrat image
(the CSV stores ``y = image height - row``, so y is up there too). 0° is a
long axis running up-down (north-south), 90° left-right (east-west), 45°
from lower-left to upper-right on screen, 135° from upper-left to
lower-right. The value is the same number whether the clast is described in
world coordinates or in image pixels, because both frames are seen the
same way on screen. What does depend on the frame is the *vector* of the
axis in data coordinates, which is why every helper here takes ``y_down``.

``_measure_clast`` computes it as ``phi + 90`` where ``phi`` is the long
axis's angle in the raw (column, row) algebra: with rows counting down that
angle runs clockwise on screen, so ``phi = -t`` for a long axis seen ``t``
degrees anticlockwise from +x, and ``Orientation = 90 - t``. A consumer
that feeds ``Orientation`` to ``matplotlib.patches.Ellipse(angle=...)`` or
``cv2.ellipse`` unchanged mirrors every clast across the diagonal: use
:func:`axis_angle_deg` instead.

Axes helpers
------------
``axis_direction(orientation, y_down)`` is the unit vector of the long axis
in a data frame whose y counts down (image rows) or up (world, quadrat
CSV). ``axis_chords`` gives the major and minor chords through the centroid
with the clast's ``Clast_length`` and ``Clast_width`` (not the ellipse
axes); ``ellipse_outline`` the ellipse with given full axes (the IoU
footprint of merge deduplication).

Contour sidecar
---------------
``<csv stem>.contours.json`` next to a clast CSV maps ``clast_ID`` to the
mask outline, a list of ``[x, y]`` in the CSV's own frame (image pixels
with y up for a quadrat photograph or a Digitize run, world coordinates for
an ortho), simplified to half a pixel. ``write_contours`` / ``read_contours``
are the only readers and writers; ``draw_clasts`` draws outlines and
chords the same way on every figure. A DataFrame returned by a detector
carries its outlines in ``df.attrs["contours"]`` (a :class:`ContourSet`,
see :func:`attach_contours`), so callers that never wrote the CSV
themselves can still persist or draw them.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "CONTOURS_SUFFIX", "CONTOURS_FORMAT", "CONTOUR_TOLERANCE_PX",
    "MAJOR_COLOR", "MINOR_COLOR", "NO_CONTOURS_NOTE",
    "normalise_orientation", "axis_direction", "axis_angle_deg",
    "axis_chords", "ellipse_outline",
    "simplify_contour", "contour_from_measurement", "world_decimals",
    "quadrat_frame", "world_frame",
    "ContourSet", "attach_contours", "contours_of",
    "contours_path_for", "write_contours", "read_contours",
    "discard_stale_contours", "transform_contours", "contours_for_frame",
    "length_colours", "draw_clasts",
    "SIZE_CLASS_COLORS", "size_classes", "add_size_legend", "muted_background",
]

CONTOURS_SUFFIX = ".contours.json"
CONTOURS_FORMAT = "pebblemapper-contours"
CONTOURS_VERSION = 1
# Half a pixel: the outline stays within the mask's own rasterisation error.
CONTOUR_TOLERANCE_PX = 0.5

# The clast figure style, the same in the app, the report and the README:
# the photograph in darkened greyscale (muted_background), each clast filled
# by its size class, a thin black outline, the length chord solid and the width
# chord dotted in the same black, and a small black dot at the centroid.
# Fills show the half-phi size class (size_classes, with add_size_legend);
# CLAST_PALETTE is for figures where colour carries no quantity.
MAJOR_COLOR = "#000000"
MINOR_COLOR = "#000000"
OUTLINE_COLOR = "#000000"
HALO_COLOR = "#ffffff"
MINOR_DASH = (0, (1.0, 1.2))
# Bright hues ordered small to large (HSV hue 0.60, 0.40, 0.16, 0.08, 0.0 at
# saturation 0.75, value 0.95): blue, green, yellow, orange, red.
SIZE_CLASS_COLORS = ("#3d85f2", "#3df285", "#f2eb3d", "#f2943d", "#f23d3d")
CLAST_PALETTE = ("#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6",
                 "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2")
BACKGROUND_BRIGHTNESS = 0.72
NO_CONTOURS_NOTE = ("no contour file for this CSV: axes only "
                    "(re-run the detection to get outlines)")

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


# --------------------------------------------------------------------------- #
#  Orientation and axes                                                        #
# --------------------------------------------------------------------------- #
def normalise_orientation(orientation) -> float:
    """The bearing folded onto [0, 180); NaN stays NaN."""
    o = float(orientation)
    if not math.isfinite(o):
        return o
    return o % 180.0


def axis_direction(orientation, *, y_down: bool) -> Point:
    """Unit vector of the long axis in a data frame whose y counts up
    (``y_down=False``: world coordinates, the quadrat CSV frame) or down
    (``y_down=True``: image columns and rows)."""
    o = math.radians(normalise_orientation(orientation))
    dx, dy = math.sin(o), math.cos(o)
    return (dx, -dy) if y_down else (dx, dy)


def axis_angle_deg(orientation, *, y_down: bool) -> float:
    """The long axis as an angle from +x in the data frame's own algebra
    (anticlockwise when y is up, clockwise on screen when y is down), in
    degrees: what ``matplotlib.patches.Ellipse(angle=...)`` and
    ``cv2.ellipse`` take when drawing in that frame."""
    dx, dy = axis_direction(orientation, y_down=y_down)
    return math.degrees(math.atan2(dy, dx))


def axis_chords(x, y, length, width, orientation, *,
                y_down: bool) -> Tuple[Segment, Segment]:
    """The major and minor chords through the centroid ``(x, y)``: the
    ``Clast_length`` chord along the long axis and the ``Clast_width`` chord
    across it, in the same units as ``x``/``y``. Returns
    ``((p0, p1), (q0, q1))``."""
    x, y = float(x), float(y)
    dx, dy = axis_direction(orientation, y_down=y_down)
    hl, hw = float(length) / 2.0, float(width) / 2.0
    major = ((x - dx * hl, y - dy * hl), (x + dx * hl, y + dy * hl))
    nx, ny = -dy, dx          # a quarter turn: perpendicular in either frame
    minor = ((x - nx * hw, y - ny * hw), (x + nx * hw, y + ny * hw))
    return major, minor


def ellipse_outline(x, y, major, minor, orientation, *, y_down: bool,
                    n_points: int = 64) -> np.ndarray:
    """``(n_points, 2)`` vertices of the ellipse with full axes ``major``
    (along the long axis) and ``minor``, centred on ``(x, y)``."""
    dx, dy = axis_direction(orientation, y_down=y_down)
    nx, ny = -dy, dx
    a, b = float(major) / 2.0, float(minor) / 2.0
    t = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    u, v = a * np.cos(t), b * np.sin(t)
    return np.column_stack([float(x) + u * dx + v * nx,
                            float(y) + u * dy + v * ny])


# --------------------------------------------------------------------------- #
#  Contours                                                                    #
# --------------------------------------------------------------------------- #
def simplify_contour(cols, rows, tolerance: float = CONTOUR_TOLERANCE_PX
                     ) -> np.ndarray:
    """Douglas–Peucker on a closed outline in pixel space; returns
    ``(n, 2)`` ``(col, row)`` vertices without the closing duplicate.
    Falls back to the raw ring when shapely cannot build a polygon."""
    cols = np.asarray(cols, dtype=float).ravel()
    rows = np.asarray(rows, dtype=float).ravel()
    pts = np.column_stack([cols, rows])
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 4 or not tolerance or tolerance <= 0:
        return pts
    try:
        from shapely.geometry import Polygon
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty or poly.geom_type != "Polygon":
            return pts
        simple = poly.simplify(float(tolerance), preserve_topology=True)
        ring = np.asarray(simple.exterior.coords, dtype=float)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        return ring if len(ring) >= 3 else pts
    except Exception:
        return pts


def quadrat_frame(height) -> Callable:
    """``to_frame`` for a photograph: ``(col, row) -> (col, height - row)``,
    the frame Quadrat mode writes ``x``/``y`` in."""
    h = float(height)

    def to_frame(cols, rows):
        return np.asarray(cols, dtype=float), h - np.asarray(rows, dtype=float)
    return to_frame


def world_frame(geotransform) -> Callable:
    """``to_frame`` for a georeferenced image: GDAL's affine geotransform
    applied to ``(col, row)``."""
    gt = [float(v) for v in geotransform]

    def to_frame(cols, rows):
        c = np.asarray(cols, dtype=float)
        r = np.asarray(rows, dtype=float)
        return gt[0] + c * gt[1] + r * gt[2], gt[3] + c * gt[4] + r * gt[5]
    return to_frame


def world_decimals(metres_per_px) -> int:
    """Decimals that keep a tenth of a pixel in world units (at least 2)."""
    try:
        g = abs(float(metres_per_px))
        if not (g > 0 and math.isfinite(g)):
            return 4
        return max(2, int(math.ceil(-math.log10(g))) + 1)
    except (TypeError, ValueError):
        return 4


def contour_from_measurement(meas: Mapping, *, to_frame: Callable,
                             offset: Point = (0.0, 0.0),
                             tolerance: float = CONTOUR_TOLERANCE_PX,
                             decimals: int = 2) -> list:
    """The outline ``_measure_clast`` found, as ``[[x, y], ...]`` in the
    CSV's frame. ``to_frame(cols, rows)`` maps arrays of image pixels to
    that frame (:func:`quadrat_frame`, :func:`world_frame`); ``offset``
    ``(col, row)`` is added to the pixels first (the crop origin of a tile
    or a bounding-box mask). Simplified in pixel space to ``tolerance`` and
    rounded to ``decimals``. Returns ``[]`` when the measurement carries no
    outline."""
    try:
        pts = simplify_contour(meas["_contour_x"], meas["_contour_y"],
                               tolerance)
    except (KeyError, TypeError, ValueError):
        return []
    if len(pts) < 3:
        return []
    cols = pts[:, 0] + float(offset[0])
    rows = pts[:, 1] + float(offset[1])
    xs, ys = to_frame(cols, rows)
    xs = np.round(np.asarray(xs, dtype=float), int(decimals))
    ys = np.round(np.asarray(ys, dtype=float), int(decimals))
    return [[float(a), float(b)] for a, b in zip(xs, ys)]


class ContourSet(dict):
    """``{clast_ID: [[x, y], ...]}`` with its ``frame`` (``"pixels"`` or
    ``"world"``), safe to keep in ``DataFrame.attrs``: pandas deep-copies
    attrs on most operations, and a set of thousands of outlines is shared
    instead of copied."""

    def __init__(self, *args, frame: str = "pixels", **kw):
        super().__init__(*args, **kw)
        self.frame = str(frame)

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def __eq__(self, other):          # attrs are compared by pandas.concat
        return self is other

    def __ne__(self, other):
        return self is not other

    __hash__ = object.__hash__


def attach_contours(df, contours: Mapping, *, frame: str):
    """Keep ``contours`` on ``df`` (``df.attrs["contours"]``); returns df."""
    try:
        cs = contours if isinstance(contours, ContourSet) else \
            ContourSet({int(k): v for k, v in dict(contours).items()},
                       frame=frame)
        cs.frame = str(frame)
        df.attrs["contours"] = cs
    except Exception:
        pass
    return df


def contours_of(df) -> Optional["ContourSet"]:
    """The outlines a detector attached to ``df``, or None."""
    try:
        cs = df.attrs.get("contours")
    except Exception:
        return None
    return cs if isinstance(cs, dict) else None


def contours_path_for(csv_path) -> Path:
    """``<csv stem>.contours.json`` beside the CSV (``foo.csv`` ->
    ``foo.contours.json``)."""
    p = Path(str(csv_path))
    stem = p.name[:-4] if p.name.lower().endswith(".csv") else p.name
    return p.with_name(stem + CONTOURS_SUFFIX)


def write_contours(csv_path, contours: Mapping, *, frame: Optional[str] = None,
                   extra: Optional[Mapping] = None) -> Optional[Path]:
    """Write the sidecar for ``csv_path``. ``contours`` maps ``clast_ID`` to
    ``[[x, y], ...]``; ``frame`` is ``"pixels"`` (image pixels, y up, as
    the quadrat CSV) or ``"world"`` (the ortho's CRS), taken from a
    :class:`ContourSet` when omitted. Returns the path, or None when nothing
    could be written: a missing outline never fails a detection run."""
    try:
        if frame is None:
            frame = getattr(contours, "frame", "pixels")
        out = contours_path_for(csv_path)
        body = {}
        for k, v in (contours or {}).items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
                continue
            body[str(int(k))] = [[float(a), float(b)] for a, b in arr]
        doc = {"format": CONTOURS_FORMAT, "version": CONTOURS_VERSION,
               "csv": Path(str(csv_path)).name, "frame": str(frame),
               "n": len(body)}
        if extra:
            doc.update(dict(extra))
        doc["contours"] = body
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(doc, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, out)
        return out
    except Exception:
        return None


def read_contours(csv_path) -> Optional[Dict[int, np.ndarray]]:
    """``{clast_ID: (n, 2) array}`` from the sidecar beside ``csv_path``,
    or None when there is none (an older CSV) or it is unreadable."""
    try:
        if csv_path is None:
            return None
        p = contours_path_for(csv_path)
        if not p.is_file():
            return None
        doc = json.loads(p.read_text(encoding="utf-8"))
        raw = doc.get("contours") if isinstance(doc, dict) else None
        if not isinstance(raw, dict):
            return None
        out = ContourSet(frame=str(doc.get("frame", "pixels")))
        for k, v in raw.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3:
                out[int(k)] = arr
        return out
    except Exception:
        return None


def discard_stale_contours(csv_path, slack_s: float = 2.0) -> bool:
    """Remove a sidecar older than the CSV beside it (a rewrite by a writer
    that kept no outlines would otherwise draw another run's contours).
    Returns True when one was removed. Never raises."""
    try:
        csv = Path(str(csv_path))
        side = contours_path_for(csv)
        if csv.exists() and side.exists() and \
                side.stat().st_mtime < csv.stat().st_mtime - slack_s:
            side.unlink()
            return True
    except OSError:
        pass
    return False


def transform_contours(contours: Optional[Mapping], fn: Callable
                       ) -> Optional[Dict[int, np.ndarray]]:
    """Apply ``fn(xs, ys) -> (xs, ys)`` to every outline (a GSD scaling, a
    geotransform, a flip); None stays None."""
    if contours is None:
        return None
    out = ContourSet(frame=getattr(contours, "frame", "pixels"))
    for k, arr in contours.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or len(a) == 0:
            continue
        xs, ys = fn(a[:, 0], a[:, 1])
        out[int(k)] = np.column_stack([np.asarray(xs, dtype=float),
                                       np.asarray(ys, dtype=float)])
    return out


def contours_for_frame(df, contours: Optional[Mapping]) -> Dict[int, np.ndarray]:
    """The outlines of the rows of ``df`` keyed by row position: a DataFrame
    that was filtered keeps its ``clast_ID`` column, which is the key."""
    if contours is None or df is None or "clast_ID" not in df.columns:
        return {}
    out = {}
    ids = df["clast_ID"].to_numpy()
    for pos, cid in enumerate(ids):
        try:
            key = int(cid)
        except (TypeError, ValueError):
            continue
        if key in contours:
            out[pos] = np.asarray(contours[key], dtype=float)
    return out


# --------------------------------------------------------------------------- #
#  Drawing                                                                     #
# --------------------------------------------------------------------------- #
def length_colours(lengths, cmap: str = "viridis"):
    """``(colours, norm, cmap_obj)`` for colouring outlines by clast length:
    the 2–98 % quantile range of ``lengths`` mapped onto ``cmap``."""
    import matplotlib
    from matplotlib import colors as _mcolors
    v = np.asarray(lengths, dtype=float).ravel()
    ok = v[np.isfinite(v) & (v > 0)]
    if len(ok):
        vmin, vmax = float(np.quantile(ok, 0.02)), float(np.quantile(ok, 0.98))
        if not vmax > vmin:
            vmin, vmax = float(ok.min()), float(ok.max())
        if not vmax > vmin:
            vmin, vmax = (vmin * 0.9 if vmin > 0 else 0.0,
                          vmax * 1.1 if vmax > 0 else 1.0)
    else:
        vmin, vmax = 0.0, 1.0
    norm = _mcolors.Normalize(vmin=vmin, vmax=vmax)
    try:
        cmap_obj = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        from matplotlib import cm
        cmap_obj = cm.get_cmap(cmap)
    colours = [cmap_obj(norm(float(l))) if np.isfinite(l) else cmap_obj(0.5)
               for l in v]
    return colours, norm, cmap_obj


def muted_background(image, brightness: float = BACKGROUND_BRIGHTNESS):
    """The photograph as darkened greyscale RGB (uint8), the ground the
    pastel clast fills are drawn on."""
    a = np.asarray(image)
    if a.ndim == 2:
        g = a.astype(float)
    else:
        rgb = a[..., :3].astype(float)
        if rgb.max() <= 1.0 and a.dtype.kind == "f":
            rgb = rgb * 255.0
        g = rgb @ np.array([0.299, 0.587, 0.114])
    g = np.clip(g * float(brightness), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def size_classes(lengths, *, unit: str = "mm") -> dict:
    """Five half-phi size classes of ``lengths`` (in the display unit):
    four consecutive edges at 2^(k/2) units, placed where they spread the
    clasts most evenly over the five classes (the fullest smallest class).
    Returns ``{"colours": one per clast, "edges": the four inner edges,
    "labels": five legend labels, "counts": clasts per class}``."""
    v = np.asarray(lengths, dtype=float).ravel()
    ok = v[np.isfinite(v) & (v > 0)]
    med = float(np.median(ok)) if len(ok) else 1.0
    m = int(np.floor(2.0 * np.log2(med)))
    best = None
    for start in range(m - 4, m + 1):
        e = [2.0 ** (k / 2.0) for k in range(start, start + 4)]
        c = np.bincount(np.digitize(ok, e), minlength=5) if len(ok) else np.zeros(5)
        score = (int(c.min()), -abs(start - (m - 1)))
        if best is None or score > best[0]:
            best = (score, e)
    edges = best[1]
    cls = np.digitize(np.where(np.isfinite(v), v, med), edges)
    fmt = lambda e: f"{e:.3g}"
    u = f" {unit}" if unit else ""
    labels = ([f"< {fmt(edges[0])}{u}"]
              + [f"{fmt(edges[i])}–{fmt(edges[i + 1])}{u}" for i in range(3)]
              + [f"≥ {fmt(edges[3])}{u}"])
    counts = [int((cls[np.isfinite(v)] == i).sum()) for i in range(5)]
    return {"colours": [SIZE_CLASS_COLORS[int(c)] for c in cls],
            "edges": edges, "labels": labels, "counts": counts}


def add_size_legend(ax, classes: dict, *, fontsize: float = 8.0, ncol: int = 5,
                    outside_figure: bool = False):
    """The size-class legend under ``ax``: one swatch per class with its
    range and count. ``outside_figure`` puts it under everything else in a
    constrained-layout figure (below an axis label or footer)."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=SIZE_CLASS_COLORS[i], edgecolor=OUTLINE_COLOR,
                     linewidth=0.6,
                     label=f"{classes['labels'][i]}  ({classes['counts'][i]})")
               for i in range(5)]
    kw = dict(handles=handles, ncol=ncol, frameon=False, fontsize=fontsize,
              handlelength=1.2, columnspacing=1.0)
    if outside_figure:
        return ax.figure.legend(loc="outside lower center", **kw)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), **kw)


def _is_single_colour(c) -> bool:
    if isinstance(c, str):
        return True
    if isinstance(c, (tuple, list, np.ndarray)) and len(c) in (3, 4):
        return all(isinstance(v, (int, float, np.floating, np.integer))
                   for v in c)
    return False


def draw_clasts(ax, clasts, *, contours: Optional[Mapping] = None,
                y_down: bool, to_data: Optional[Callable] = None,
                units_per_m: float = 1.0, edge_colors=None,
                face_alpha: float = 0.75, outline_width: float = 0.6,
                chord_width: float = 0.45, halo: bool = False,
                note: bool = True, zorder: int = 5,
                chords: bool = True, outlines: bool = True,
                centres: bool = True) -> dict:
    """Draw every clast of ``clasts`` (rows with ``x, y, Clast_length,
    Clast_width, Orientation``; ``clast_ID`` when ``contours`` is given) on
    ``ax`` as its contour polygon, when ``contours`` has it, filled with a
    translucent colour under a thin black outline, its length chord solid
    and its width chord dotted in the same black, and a small black dot at
    its centroid (drawn in data units, so it sits exactly where the chords
    cross). No ellipse is drawn.

    ``to_data(xs, ys)`` maps the CSV frame to the axes' data frame (for a
    photograph drawn as rows, ``(x, H - y)``); ``y_down`` says whether that
    data frame counts y downward. Lengths are multiplied by ``units_per_m``
    (1 for metres on a georeferenced axis, ``1 / gsd`` or ``px_per_unit``
    for pixels). ``edge_colors`` is one colour or one per row (a length
    classes, say: ``size_classes``) and colours the fill; left None, clasts
    take turns through ``CLAST_PALETTE`` so neighbours differ. ``face_alpha`` is the
    fill opacity; the outline is always ``OUTLINE_COLOR``.
    When ``note`` is set and no contours are drawn, a one-line note says
    the figure shows axes only. Returns ``{"n_outlines", "n_chords",
    "n_rows"}``."""
    import pandas as pd
    from matplotlib import colors as _mcolors
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patheffects import withStroke

    df = clasts if isinstance(clasts, pd.DataFrame) else pd.DataFrame(clasts)
    n = int(len(df))
    if n == 0:
        return {"n_outlines": 0, "n_chords": 0, "n_rows": 0}
    if to_data is None:
        def to_data(xs, ys):
            return xs, ys
    xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    dx, dy = to_data(xs, ys)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    def _col(name):
        if name not in df.columns:
            return np.full(n, np.nan)
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    L = _col("Clast_length") * float(units_per_m)
    Wd = _col("Clast_width") * float(units_per_m)
    O = _col("Orientation")

    if edge_colors is None:
        edge_colors = [CLAST_PALETTE[(i * 5) % len(CLAST_PALETTE)]
                       for i in range(n)]
    if _is_single_colour(edge_colors):
        edge_list = [edge_colors] * n
    else:
        edge_list = list(edge_colors)
        if len(edge_list) != n:
            edge_list = [edge_list[0] if edge_list else "#d97706"] * n

    outline_rgba = _mcolors.to_rgba(OUTLINE_COLOR, alpha=1.0)
    n_outlines = 0
    if outlines and contours:
        by_pos = contours_for_frame(df, contours)
        verts, ec, fc = [], [], []
        for pos, arr in by_pos.items():
            if arr.ndim != 2 or len(arr) < 3:
                continue
            px, py = to_data(arr[:, 0], arr[:, 1])
            verts.append(np.column_stack([np.asarray(px, dtype=float),
                                          np.asarray(py, dtype=float)]))
            col = edge_list[pos]
            ec.append(outline_rgba)
            try:
                r, g, b, _a = _mcolors.to_rgba(col)
                fc.append((r, g, b, float(face_alpha)))
            except (ValueError, TypeError):
                fc.append((0, 0, 0, 0))
        if verts:
            ax.add_collection(PolyCollection(
                verts, closed=True, facecolors=fc, edgecolors=ec,
                linewidths=outline_width, zorder=zorder))
            n_outlines = len(verts)

    n_chords = 0
    if chords:
        maj, mnr = [], []
        for i in range(n):
            if not (np.isfinite(L[i]) and np.isfinite(Wd[i]) and np.isfinite(O[i])
                    and np.isfinite(dx[i]) and np.isfinite(dy[i])
                    and L[i] > 0 and Wd[i] > 0):
                continue
            (p0, p1), (q0, q1) = axis_chords(dx[i], dy[i], L[i], Wd[i], O[i],
                                             y_down=y_down)
            maj.append([p0, p1])
            mnr.append([q0, q1])
        if maj:
            halo_fx = [withStroke(linewidth=chord_width + 1.2,
                                  foreground=HALO_COLOR, alpha=0.55)] if halo else None
            ax.add_collection(LineCollection(
                mnr, colors=_mcolors.to_rgba(MINOR_COLOR, alpha=1.0),
                linewidths=chord_width, zorder=zorder + 1,
                linestyles=MINOR_DASH, path_effects=halo_fx))
            ax.add_collection(LineCollection(
                maj, colors=_mcolors.to_rgba(MAJOR_COLOR, alpha=1.0),
                linewidths=chord_width, zorder=zorder + 2,
                capstyle="round", path_effects=halo_fx))
            n_chords = len(maj)

    if centres:
        # Small discs in data units: a marker in points is snapped to whole
        # pixels and lands beside the chords' crossing. The radius follows
        # the drawn extent and never exceeds a tenth of a clast's width.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span = max(abs(x1 - x0), abs(y1 - y0))
        if not (np.isfinite(span) and span > 0 and (x0, x1) != (0.0, 1.0)):
            ok = np.isfinite(dx) & np.isfinite(dy)
            span = float(max(np.ptp(dx[ok]), np.ptp(dy[ok]))) if ok.any() else 0.0
        base = 0.0015 * span if span > 0 else 0.0
        t = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        discs = []
        for i in range(n):
            if not (np.isfinite(dx[i]) and np.isfinite(dy[i])):
                continue
            r = base
            if np.isfinite(Wd[i]) and Wd[i] > 0:
                r = min(r, 0.1 * Wd[i]) if r > 0 else 0.1 * Wd[i]
            if r > 0:
                discs.append(np.column_stack([dx[i] + r * np.cos(t), dy[i] + r * np.sin(t)]))
        if discs:
            ax.add_collection(PolyCollection(discs, closed=True, facecolors=OUTLINE_COLOR,
                                             edgecolors="none", zorder=zorder + 3))

    if note and outlines and n_outlines == 0:
        ax.text(0.99, 0.01, NO_CONTOURS_NOTE, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#333",
                zorder=zorder + 3,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.8))
    return {"n_outlines": n_outlines, "n_chords": n_chords, "n_rows": n}
