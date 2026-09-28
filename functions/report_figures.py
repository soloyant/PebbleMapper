"""Matplotlib (Agg) figure builders the PDF report embeds.

Every public builder returns a :class:`matplotlib.figure.Figure` built from
plain data — numpy arrays, pandas DataFrames, file paths — and never touches
the GUI. The report converts the figure itself (``report._figure_image``).
When the data cannot support a figure the builder raises ``ValueError`` with a
one-line reason instead of drawing an empty frame, so the report can print
the reason where the figure would have been.

Each figure carries ``fig.pm_meta``: a plain dict with the numbers a caption
needs (counts, percentiles, fit statistics …), computed from the same arrays
the figure was drawn from.

Conventions shared by every builder
-----------------------------------
* Lengths arrive in **metres** (the stored unit of the detection CSVs) and
  are shown in the display unit through ``scale`` (metres → display; 1000 for
  millimetres — see :mod:`functions.units`).
* ``Orientation`` is in degrees, axial [0, 180): the bearing of the long
  axis clockwise from image-up (grid north for UAV world coordinates, the top
  of the photograph for quadrat pixel coordinates). Clasts are drawn as their
  mask outlines with their major and minor axes through
  :func:`functions.clast_geometry.draw_clasts`, the one implementation of
  that convention; no builder draws ellipses.
* Figures are 6.5 in wide (single) or 7.2 in (multi-panel), constrained
  layout, 9–10 pt labels, units in parentheses in axis labels, thousands
  separators, U+2212 for minus signs, viridis / cividis / Greys plus one
  accent colour. No figure title unless the caller passes ``title``.
* Georeferencing: ``transform`` may be a rasterio / ``affine.Affine`` (or any
  object with ``a … f`` attributes), a 6-tuple **GDAL geotransform** as
  returned by ``ds.GetGeoTransform()``, or a 9-tuple in Affine order.
  When it is omitted the raster's own georeferencing is read from the file;
  when neither exists the image is treated as pixel space.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from functions import images as _images

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm as _cm  # noqa: E402
from matplotlib import colors as _mcolors  # noqa: E402
from matplotlib import ticker as _mtick  # noqa: E402
from matplotlib import transforms as _mtrans  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

__all__ = [
    "detection_overlay",
    "cdf_overlay",
    "histogram_adaptive",
    "validation_diagnostics",
    "cell_count_map",
    "orientation_rose",
    "zone_location_map",
    "axial_circular_mean",
]

# --- House style ---
SINGLE_WIDTH_IN = 6.5
PANEL_WIDTH_IN = 7.2
FONT_PT = 9
ACCENT = "#d97706"          # the amber the report already uses for fits
MINUS = "−"

_RC = {
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT + 1,
    "axes.titlesize": FONT_PT + 1,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT - 1,
    "axes.unicode_minus": True,
    "hatch.linewidth": 0.6,
    "figure.dpi": 100,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}

_LINESTYLES = ["-", "--", "-.", ":"]


def _um(s: str) -> str:
    """Typographic minus in a string this module formatted itself."""
    return str(s).replace("-", MINUS)


def _fmt(v: float, decimals: Optional[int] = None) -> str:
    """Thousands separators, U+2212, sensible decimals for a display value."""
    if v is None or not np.isfinite(v):
        return "—"
    if decimals is None:
        a = abs(v)
        decimals = 0 if a >= 100 else (1 if a >= 10 else 2)
    return _um(f"{v:,.{decimals}f}")


def _plain_tick(v, _pos=None):
    """'20', '50', '1,000' on a log axis — never '2e+01'."""
    return _um(f"{v:,g}")


def _series_colors(k: int) -> List[tuple]:
    """``k`` viridis colours, stopping short of the yellow end so every line
    survives on white paper and in greyscale."""
    if k <= 1:
        return [_cm.viridis(0.25)]
    return [_cm.viridis(t) for t in np.linspace(0.0, 0.62, k)]


def _new_figure(width: float, height: float) -> Figure:
    """A Figure that is not registered with pyplot (no global state)."""
    return Figure(figsize=(width, height), constrained_layout=True)


def _apply_log_axis(ax):
    """1-2-5 major ticks with plain labels, finer minor ticks, light grid."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(
        _mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.xaxis.set_minor_locator(
        _mtick.LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
                          numticks=24))
    ax.xaxis.set_major_formatter(_mtick.FuncFormatter(_plain_tick))
    ax.xaxis.set_minor_formatter(_mtick.NullFormatter())
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)


def _mm_per_display(unit: str, scale: float) -> Optional[float]:
    """Millimetres per display unit, or None when the axis is not a length."""
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return None
    if unit not in ("mm", "cm", "m", "µm", "um"):
        return None
    return 1000.0 / float(scale)


def _add_phi_axis(ax, lo_display: float, hi_display: float,
                  mm_per_unit: Optional[float], outward_pt: float = 0.0):
    """Krumbein φ (= −log₂ D_mm) along the top of a log-x axis.

    Implemented as a twin axis with ticks at the display-unit positions of
    whole φ values — a ``secondary_xaxis`` inherits the log scale and cannot
    represent negative φ. Skipped when fewer than two whole φ fall inside
    the data range.
    """
    if mm_per_unit is None or lo_display <= 0 or hi_display <= lo_display:
        return None
    lo_mm, hi_mm = lo_display * mm_per_unit, hi_display * mm_per_unit
    phi_hi = int(math.floor(-math.log2(hi_mm)))   # most negative
    phi_lo = int(math.ceil(-math.log2(lo_mm)))    # least negative
    ticks = list(range(phi_hi, phi_lo + 1))
    if len(ticks) < 2:
        return None
    # Thin out very long φ ranges so labels never collide.
    while len(ticks) > 9:
        ticks = ticks[::2]
    sec = ax.twiny()
    sec.set_xscale("log")
    sec.set_xlim(ax.get_xlim())
    sec.set_xticks([2.0 ** (-t) / mm_per_unit for t in ticks])
    sec.set_xticklabels([_um(f"{t:d}") for t in ticks])
    sec.xaxis.set_minor_locator(_mtick.NullLocator())
    sec.set_xlabel(r"$\phi = -\log_{2}\,D_{\mathrm{mm}}$", fontsize=FONT_PT)
    sec.tick_params(labelsize=FONT_PT)
    if outward_pt:
        sec.spines["top"].set_position(("outward", outward_pt))
    return sec


def _clean_positive(values, what: str = "values") -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    v = v[v > 0]
    return v


def _nice_length(span: float) -> float:
    """A 1-2-5 length about a fifth of ``span`` (same rule as map_export)."""
    target = span * 0.20
    if not np.isfinite(target) or target <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(target))
    for nice in (1, 2, 5, 10):
        if nice * mag >= target:
            return nice * mag
    return 10 * mag


def _length_label(length_m: float) -> str:
    if length_m < 1.0:
        return f"{length_m * 100:g} cm"
    if length_m >= 1000.0:
        return f"{length_m / 1000:g} km"
    return f"{length_m:g} m"


def _add_scale_bar(ax, length_data: float, label: str, *, loc="lower left"):
    """A single black bar of ``length_data`` data units, labelled, boxed.

    x in data coordinates (so it scales with the map), y in axes fraction
    (so it sits at the same height whatever the extent).
    """
    tr = _mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    if loc == "lower left":
        x0 = xmin + 0.04 * abs(span) * np.sign(span)
    else:
        x0 = xmax - 0.04 * abs(span) * np.sign(span) - length_data * np.sign(span)
    y0 = 0.045
    h = 0.014
    ax.add_patch(Rectangle((x0, y0), length_data * np.sign(span), h,
                           transform=tr, facecolor="black",
                           edgecolor="black", lw=0.5, zorder=30))
    ax.text(x0 + length_data * np.sign(span) / 2.0, y0 + h + 0.008, label,
            transform=tr, ha="center", va="bottom", fontsize=FONT_PT,
            zorder=31,
            path_effects=[withStroke(linewidth=2.5, foreground="white")])
    # A translucent white pad so the bar reads on any background.
    pad_x = 0.012 * abs(span)
    ax.add_patch(Rectangle((min(x0, x0 + length_data * np.sign(span)) - pad_x,
                            y0 - 0.012),
                           length_data + 2 * pad_x, h + 0.05,
                           transform=tr, facecolor="white", edgecolor="none",
                           alpha=0.6, zorder=29))


def _add_north_arrow(ax):
    """A small arrow + 'N' in the upper-right corner (axes fraction)."""
    ax.annotate("N", xy=(0.95, 0.96), xytext=(0.95, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="top", fontsize=FONT_PT + 1,
                fontweight="bold", zorder=31,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                shrinkA=0, shrinkB=0),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="none", alpha=0.7))


def _use_full_coordinates(ax):
    """Full projected coordinates on both axes: no offset, no exponent,
    thousands separators."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(_mtick.MaxNLocator(nbins=5, steps=[1, 2, 5]))
        axis.set_major_formatter(_mtick.FuncFormatter(
            lambda v, _p: _um(f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}")))
    ax.tick_params(axis="x", labelrotation=0)


def _map_height(width_in: float, x_span: float, y_span: float,
                extra_in: float = 0.9, lo: float = 3.2, hi: float = 7.5
                ) -> float:
    """Figure height that follows the map's aspect (equal-aspect axes leave
    blank bands otherwise), with room for labels and a colorbar."""
    if not (x_span > 0 and y_span > 0):
        return 5.4
    h = width_in * 0.8 * (y_span / x_span) + extra_in
    return float(min(hi, max(lo, h)))


def _crs_box(ax, text: str):
    """The CRS in the lower-right corner (the scale bar owns the lower left)."""
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FONT_PT - 1, zorder=31,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", lw=0.5, alpha=0.75))


# --- Affine helpers ---
def _coerce_transform(transform) -> Optional[Tuple[float, ...]]:
    """→ ``(a, b, c, d, e, f)`` with x = a·col + b·row + c, y = d·col + e·row + f.

    Accepts an Affine-like object, a 6-tuple GDAL geotransform, or a 6/9-tuple
    in Affine order (9 = the tuple form of ``affine.Affine``). ``None`` and the
    identity both mean "not georeferenced".
    """
    if transform is None:
        return None
    if all(hasattr(transform, k) for k in "abcdef"):
        t = tuple(float(getattr(transform, k)) for k in "abcdef")
    else:
        seq = [float(v) for v in transform]
        if len(seq) == 9:
            t = tuple(seq[:6])
        elif len(seq) == 6:
            # GDAL order: (x0, dx, rx, y0, ry, dy)
            x0, dx, rx, y0, ry, dy = seq
            t = (dx, rx, x0, ry, dy, y0)
        else:
            raise ValueError("transform must be an Affine, a 6-tuple GDAL "
                             "geotransform or a 9-tuple Affine")
    if t == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0):
        return None
    if not all(np.isfinite(t)):
        raise ValueError("transform contains non-finite values")
    return t


def _pix_to_world(t, col, row):
    a, b, c, d, e, f = t
    return a * col + b * row + c, d * col + e * row + f


def _world_to_pix(t, x, y):
    a, b, c, d, e, f = t
    det = a * e - b * d
    if det == 0:
        raise ValueError("transform is singular")
    col = (e * (x - c) - b * (y - f)) / det
    row = (-d * (x - c) + a * (y - f)) / det
    return col, row


# --- Raster access (GDAL -> rasterio -> PIL) ---
class _Raster:
    """Read windows of an image with whichever backend the env provides.

    GDAL is tried first because the rest of PebbleMapper reads orthos with
    it; rasterio next; PIL last (plain photographs). ``transform`` is the
    file's georeferencing as ``(a…f)`` or None; ``crs_label`` is "EPSG:nnnn"
    or the CRS name when known.
    """

    def __init__(self, path):
        self.path = str(path)
        if not Path(self.path).is_file():
            raise ValueError(f"image not found: {self.path}")
        self.backend = None
        self.width = self.height = 0
        self.count = 0
        self.transform: Optional[Tuple[float, ...]] = None
        self.crs_label: Optional[str] = None
        self._ds = None
        self._open()

    # -- opening ------------------------------------------------------------
    def _open(self):
        try:
            from osgeo import gdal, osr
            if _images.is_heif(self.path):     # no GDAL HEIF driver: skip
                raise ValueError("HEIF: not a GDAL raster")
            ds = gdal.Open(self.path)
            if ds is not None:
                self.backend = "gdal"
                self._ds = ds
                self.width, self.height = ds.RasterXSize, ds.RasterYSize
                self.count = ds.RasterCount
                gt = ds.GetGeoTransform(can_return_null=True) \
                    if hasattr(ds, "GetGeoTransform") else None
                self.transform = _coerce_transform(gt) if gt else None
                wkt = ds.GetProjection()
                if wkt:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(wkt)
                    self.crs_label = _srs_label(srs)
                return
        except Exception:
            pass
        try:
            import rasterio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = rasterio.open(self.path)
            self.backend = "rasterio"
            self._ds = ds
            self.width, self.height = ds.width, ds.height
            self.count = ds.count
            t = ds.transform
            self.transform = None if t.is_identity else _coerce_transform(t)
            if ds.crs:
                epsg = ds.crs.to_epsg()
                self.crs_label = (f"EPSG:{epsg}" if epsg
                                  else ds.crs.to_string())
            return
        except Exception:
            pass
        try:
            from PIL import Image
            im = Image.open(self.path)
            im.load()
            self.backend = "pil"
            self._ds = im.convert("RGB")
            self.width, self.height = im.size
            self.count = 3
            return
        except Exception as ex:
            raise ValueError(f"cannot open image {self.path}: "
                             f"{type(ex).__name__}: {ex}")

    def close(self):
        if self.backend == "pil" and self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
        elif self.backend == "rasterio" and self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
        self._ds = None

    # -- reading ------------------------------------------------------------
    def read(self, col0: int, row0: int, w: int, h: int,
             max_px: int = 1500) -> np.ndarray:
        """``(h', w', 3)`` float32 in 0–1 for the pixel window, resampled so
        the longest side is at most ``max_px``."""
        if w <= 0 or h <= 0:
            raise ValueError("the window falls outside the image")
        scale = min(1.0, float(max_px) / float(max(w, h)))
        ow = max(1, int(round(w * scale)))
        oh = max(1, int(round(h * scale)))
        if self.backend == "gdal":
            arr = self._ds.ReadAsArray(int(col0), int(row0), int(w), int(h),
                                       buf_xsize=ow, buf_ysize=oh)
            if arr is None:
                raise ValueError("GDAL could not read the window")
            if arr.ndim == 2:
                arr = arr[None, ...]
            arr = arr[:3]
        elif self.backend == "rasterio":
            from rasterio.windows import Window
            idx = list(range(1, min(3, self.count) + 1))
            arr = self._ds.read(idx, window=Window(col0, row0, w, h),
                                out_shape=(len(idx), oh, ow))
        else:
            im = self._ds.crop((col0, row0, col0 + w, row0 + h))
            if (ow, oh) != (w, h):
                im = im.resize((ow, oh))
            arr = np.asarray(im).transpose(2, 0, 1)
        arr = np.asarray(arr)
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        rgb = arr.transpose(1, 2, 0)
        return _normalise_rgb(rgb)


def _srs_label(srs) -> Optional[str]:
    try:
        auth = srs.GetAuthorityName(None)
        code = srs.GetAuthorityCode(None)
        if auth and code:
            return f"{auth}:{code}"
        name = srs.GetName()
        return name or None
    except Exception:
        return None


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0
    out = rgb.astype(np.float32)
    finite = out[np.isfinite(out)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            out = (out - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out), 0.0, 1.0)


def _pixel_size_guard(t):
    """Refuse a geographic (degree) transform: lengths would be nonsense."""
    a, _b, _c, _d, e, _f = t
    if max(abs(a), abs(e)) < 1e-5:
        raise ValueError("raster pixel size is < 1e-5 units — the CRS looks "
                         "geographic (degrees); a projected CRS in metres is "
                         "required")


# --- 1. detection_overlay ---
def _densest_window_center(x: np.ndarray, y: np.ndarray,
                           window: float) -> Tuple[float, float]:
    """Centre of the ``window``×``window`` box holding the most points,
    searched on a half-window lattice."""
    if len(x) == 1:
        return float(x[0]), float(y[0])
    step = window / 2.0
    x0, y0 = float(x.min()), float(y.min())
    nx = int(math.ceil((float(x.max()) - x0) / step)) + 1
    ny = int(math.ceil((float(y.max()) - y0) / step)) + 1
    if nx * ny > 4_000_000:
        return float(np.median(x)), float(np.median(y))
    H, _xe, _ye = np.histogram2d(
        x, y, bins=[nx, ny],
        range=[[x0, x0 + nx * step], [y0, y0 + ny * step]])
    Hp = np.zeros((nx + 1, ny + 1))
    Hp[:nx, :ny] = H
    S = Hp[:-1, :-1] + Hp[1:, :-1] + Hp[:-1, 1:] + Hp[1:, 1:]
    i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    return x0 + (i + 1) * step, y0 + (j + 1) * step


def detection_overlay(image_path, clasts: pd.DataFrame, *, center_xy=None,
                      window_m: float = 2.0, gsd_m: Optional[float] = None,
                      transform=None, title: Optional[str] = None,
                      contours=None, csv_path=None) -> Figure:
    """A crop of the source image with the detected clasts as their mask
    outlines and their major and minor axes.

    ``clasts`` needs ``x, y, Clast_length, Clast_width, Orientation``
    (lengths in metres) and ``clast_ID`` for the outlines. For a GeoTIFF the
    world coordinates are located through ``transform`` (or the file's own
    georeferencing); for a plain photograph ``x, y`` are pixels with ``y``
    measured upward from the bottom edge (as Quadrat mode writes them) and
    ``gsd_m`` (metres per pixel) sizes the axes and the scale bar. The crop
    is ``window_m`` wide, centred on ``center_xy`` (in the CSV's frame) or,
    by default, on the densest window.

    Outlines come from ``contours`` (``{clast_ID: [[x, y], ...]}`` in the
    CSV's frame), else from the ``.contours.json`` beside ``csv_path``, else
    from ``clasts.attrs``; without any, the axes are drawn alone and the
    figure says so. The image is shown in darkened greyscale; each clast is
    filled by its half-phi size class (legend under the frame) under a thin
    black outline, with its length chord solid, its width chord dotted and
    its centroid as a dot. The scale bar is 10 cm for windows up to 3 m and
    1 m beyond.

    ``fig.pm_meta``: ``n_shown, n_total, window_m, center_xy, mode,
    n_outlines``.
    """
    from functions import clast_geometry as CG
    need = ["x", "y", "Clast_length", "Clast_width", "Orientation"]
    missing = [c for c in need if c not in clasts.columns]
    if missing:
        raise ValueError(f"clasts is missing column(s) {missing}")
    if contours is None and csv_path is not None:
        contours = CG.read_contours(csv_path)
    if contours is None:
        contours = CG.contours_of(clasts)
    cols = need + (["clast_ID"] if "clast_ID" in clasts.columns else [])
    df = clasts[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=need)
    df = df[(df["Clast_length"] > 0) & (df["Clast_width"] > 0)]
    if len(df) == 0:
        raise ValueError("no clasts with finite x, y, length and width")
    whole_image = window_m is None
    if not whole_image and not (np.isfinite(window_m) and window_m > 0):
        raise ValueError("window_m must be a positive length in metres, or "
                         "None for the whole image")

    ras = _Raster(image_path)
    try:
        t = _coerce_transform(transform) if transform is not None \
            else ras.transform
        if t is not None:
            _pixel_size_guard(t)
            mode = "georeferenced"
            units_per_m = 1.0          # data coordinates are metres
            px_m = math.hypot(t[0], t[3])   # metres per pixel column step
        else:
            if gsd_m is None or not (np.isfinite(gsd_m) and gsd_m > 0):
                raise ValueError("gsd_m (metres per pixel) is required when "
                                 "the image is not georeferenced")
            mode = "pixels"
            units_per_m = 1.0 / float(gsd_m)
            px_m = float(gsd_m)

        if mode == "pixels":
            # The CSV counts y up from the bottom edge; the axes count rows.
            H_px = float(ras.height)

            def to_data(xs, ys):
                return (np.asarray(xs, dtype=float),
                        H_px - np.asarray(ys, dtype=float))
            if center_xy is not None:
                center_xy = (float(center_xy[0]), H_px - float(center_xy[1]))
        else:
            def to_data(xs, ys):
                return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        x, y = to_data(df["x"].to_numpy(float), df["y"].to_numpy(float))
        if whole_image:
            # The whole photograph (quadrat photographs are one frame).
            c0, c1, r0, r1 = 0, ras.width, 0, ras.height
            span_m = max(ras.width, ras.height) * px_m
            if mode == "georeferenced":
                cxy = _pix_to_world(t, ras.width / 2.0, ras.height / 2.0)
            else:
                cxy = (ras.width / 2.0, ras.height / 2.0)
            cx, cy = cxy
        else:
            window = window_m * units_per_m
            span_m = float(window_m)
            if center_xy is None:
                cx, cy = _densest_window_center(x, y, window)
            else:
                cx, cy = float(center_xy[0]), float(center_xy[1])
            half = window / 2.0
            xlo, xhi, ylo, yhi = cx - half, cx + half, cy - half, cy + half

            # Pixel window, clipped to the raster.
            if mode == "georeferenced":
                corners = [_world_to_pix(t, xx, yy)
                           for xx in (xlo, xhi) for yy in (ylo, yhi)]
                cols = [c for c, _r in corners]
                rows = [r for _c, r in corners]
            else:
                cols, rows = [xlo, xhi], [ylo, yhi]
            c0 = int(max(0, math.floor(min(cols))))
            c1 = int(min(ras.width, math.ceil(max(cols))))
            r0 = int(max(0, math.floor(min(rows))))
            r1 = int(min(ras.height, math.ceil(max(rows))))
        if c1 <= c0 or r1 <= r0:
            raise ValueError("the window falls outside the image")
        img = ras.read(c0, r0, c1 - c0, r1 - r0, max_px=1500)

        if mode == "georeferenced":
            x_left, y_top = _pix_to_world(t, c0, r0)
            x_right, y_bot = _pix_to_world(t, c1, r1)
            extent = (x_left, x_right, y_bot, y_top)
        else:
            extent = (c0, c1, r1, r0)

        inside = ((x >= min(extent[0], extent[1]))
                  & (x <= max(extent[0], extent[1]))
                  & (y >= min(extent[2], extent[3]))
                  & (y <= max(extent[2], extent[3])))
        sub = df[inside]

        with matplotlib.rc_context(_RC):
            # Height follows the crop's aspect (a clipped window or a whole
            # 4:3 photograph is not square).
            aspect = (r1 - r0) / float(c1 - c0)
            fig = _new_figure(SINGLE_WIDTH_IN,
                              float(min(8.0, max(3.0, SINGLE_WIDTH_IN * aspect))))
            ax = fig.add_subplot(111)
            ax.imshow(CG.muted_background(img), extent=extent, origin="upper",
                      interpolation="bilinear", zorder=1)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            classes = CG.size_classes(
                pd.to_numeric(sub["Clast_length"], errors="coerce") * 1000.0, unit="mm")
            drawn = CG.draw_clasts(
                ax, sub, contours=contours, y_down=(mode == "pixels"),
                to_data=to_data, units_per_m=units_per_m,
                edge_colors=classes["colours"])
            if len(sub):
                CG.add_size_legend(ax, classes, fontsize=7)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            bar_m = 0.1 if span_m <= 3.0 else 1.0
            _add_scale_bar(ax, bar_m * units_per_m, _length_label(bar_m))
            if title:
                ax.set_title(title)
        fig.pm_meta = {
            "n_shown": int(len(sub)),
            "n_total": int(len(df)),
            "window_m": None if whole_image else float(window_m),
            "span_m": float(span_m),
            "center_xy": (float(cx), float(cy)),
            "mode": mode,
            "scale_bar_m": bar_m,
            "n_outlines": int(drawn["n_outlines"]),
        }
        return fig
    finally:
        ras.close()


# --- 2. cdf_overlay ---
_MARK_STYLES = {50: "o", 84: "s", 95: "^", 16: "v", 10: "D", 90: "P"}


def _draw_cdfs(ax, series: Mapping[str, np.ndarray], *, scale: float,
               marks: Sequence[int], dmin_display: Dict[str, float],
               legend_loc="lower right", show_marks_legend=True,
               legend_kwargs: Optional[dict] = None, unit: str = "mm"):
    """Shared by :func:`cdf_overlay` and the validation panel.

    ``series`` values are already cleaned (finite, > 0, n ≥ 2, metres).
    Returns ``(quantiles, xlim)`` where quantiles is ``{name: {q: value}}``
    in display units.
    """
    names = list(series)
    colors = _series_colors(len(names))
    all_disp = np.concatenate([np.asarray(series[n]) * scale for n in names])
    xlo, xhi = float(all_disp.min()) * 0.8, float(all_disp.max()) * 1.25
    for v in dmin_display.values():
        if v is not None and np.isfinite(v) and v > 0:
            xlo = min(xlo, v * 0.8)
    quantiles: Dict[str, Dict[int, float]] = {}
    handles = []
    for k, name in enumerate(names):
        v = np.sort(np.asarray(series[name], dtype=float)) * scale
        n = len(v)
        yy = np.arange(1, n + 1) / n
        col = colors[k]
        ls = _LINESTYLES[k % len(_LINESTYLES)]
        (ln,) = ax.plot(v, yy, color=col, ls=ls, lw=1.6,
                        drawstyle="steps-post", zorder=4 + k,
                        label=f"{name} (n = {n:,})")
        handles.append(ln)
        qd = {}
        for q in marks:
            qv = float(np.quantile(v, q / 100.0))
            qd[int(q)] = qv
            ax.plot([qv], [q / 100.0], marker=_MARK_STYLES.get(int(q), "o"),
                    ms=5, color=col, mec="white", mew=0.7, ls="none",
                    zorder=10 + k)
        quantiles[name] = qd
    # Detection limits: a band, a dashed line and a label per distinct
    # value, in the colour of the population it belongs to; the populations
    # are named only when their limits differ.
    limits: Dict[float, List[str]] = {}
    for name in names:
        dm = dmin_display.get(name)
        if dm is not None and np.isfinite(dm) and dm > xlo:
            limits.setdefault(round(float(dm), 6), []).append(name)
    for dm, owners in limits.items():
        col = colors[names.index(owners[0])] if len(limits) > 1 else "#555"
        ax.axvspan(xlo, dm, color=col, alpha=0.07, lw=0, zorder=1)
        ax.axvline(dm, color=col, lw=1.0, ls=(0, (4, 3)), zorder=3)
        # Two series of the same instrument share a limit and a short name:
        # "UAV ortho, UAV ortho" says nothing twice.
        short = list(dict.fromkeys(o.split(",")[0].strip() for o in owners))
        who = "" if len(limits) == 1 else ", ".join(short) + ": "
        ax.text(dm, 0.985, f" {who}detection limit {dm:.3g} {unit} ",
                rotation=90, ha="right", va="top", fontsize=FONT_PT - 2,
                color=col, zorder=12)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(_mtick.FuncFormatter(
        lambda v, _p: f"{v:g}"))
    for q in marks:
        ax.axhline(q / 100.0, color="grey", lw=0.5, ls=":", alpha=0.6,
                   zorder=2)
    if show_marks_legend:
        for q in marks:
            handles.append(Line2D([], [], marker=_MARK_STYLES.get(int(q), "o"),
                                  color="#444", mec="white", ls="none", ms=5,
                                  label=f"D{int(q)}"))
    kw = dict(loc=legend_loc, frameon=True, framealpha=0.85,
              edgecolor="none")
    kw.update(legend_kwargs or {})
    ax.legend(handles=handles, **kw)
    return quantiles, (xlo, xhi)


def _clean_series(series: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not series:
        raise ValueError("no series to draw")
    out: Dict[str, np.ndarray] = {}
    for name, vals in series.items():
        v = _clean_positive(vals)
        if len(v) < 2:
            raise ValueError(f"series '{name}' has n = {len(v)} (< 2)")
        out[str(name)] = v
    return out


def cdf_overlay(series: Mapping[str, np.ndarray], *,
                field_label: str = "Clast length", unit: str = "mm",
                scale: float = 1000.0, marks: Sequence[int] = (50, 84),
                dmin_mm: Union[Mapping[str, float], float, None] = None
                ) -> Figure:
    """Empirical CDFs of several populations on one log-x axis.

    ``series`` maps a legend name to values in metres. Each curve gets its
    own viridis hue **and** line style, so the overlay reads in greyscale
    and under deuteranopia. ``marks`` percentiles are drawn as small
    markers on each curve (circle D50, square D84). ``dmin_mm`` — one value
    or one per series, in millimetres — shades a "below detection limit"
    band. A φ axis runs along the top when the field is a length.

    ``fig.pm_meta``: ``n`` and the marked percentiles per series, in display
    units.
    """
    clean = _clean_series(series)
    mm_per_unit = _mm_per_display(unit, scale)
    dmin_display: Dict[str, float] = {}
    if dmin_mm is not None:
        if isinstance(dmin_mm, Mapping):
            src = {str(k): v for k, v in dmin_mm.items()}
        else:
            src = {name: float(dmin_mm) for name in clean}
        for name, v in src.items():
            if v is None or not np.isfinite(v):
                continue
            dmin_display[name] = (float(v) / mm_per_unit if mm_per_unit
                                  else float(v))
    with matplotlib.rc_context(_RC):
        fig = _new_figure(SINGLE_WIDTH_IN, 4.0)
        ax = fig.add_subplot(111)
        _apply_log_axis(ax)
        quant, (xlo, xhi) = _draw_cdfs(ax, clean, scale=scale, marks=marks,
                                       dmin_display=dmin_display, unit=unit)
        ax.set_xlabel(f"{field_label} ({unit})" if unit else field_label)
        ax.set_ylabel("Cumulative fraction")
        _add_phi_axis(ax, xlo, xhi, mm_per_unit)
    fig.pm_meta = {
        "n": {k: int(len(v)) for k, v in clean.items()},
        "marks": [int(q) for q in marks],
        "quantiles": quant,
        "unit": unit,
        "dmin": dmin_display or None,
    }
    for q in marks:
        fig.pm_meta[f"D{int(q)}"] = {k: quant[k][int(q)] for k in clean}
    return fig


# --- 3. histogram_adaptive ---
def _fd_bins_log(v: np.ndarray, lo: int = 8, hi: int = 60) -> int:
    """Freedman–Diaconis bin count on log10 values, clamped to [lo, hi]."""
    lv = np.log10(v)
    n = len(lv)
    q75, q25 = np.percentile(lv, [75, 25])
    iqr = q75 - q25
    rng = float(lv.max() - lv.min())
    if rng <= 0:
        return lo
    if iqr <= 0:
        return lo
    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    k = int(math.ceil(rng / h)) if h > 0 else hi
    return int(min(hi, max(lo, k)))


def histogram_adaptive(values_m: np.ndarray, *, unit: str = "mm",
                       scale: float = 1000.0,
                       dmin_mm: Optional[float] = None,
                       fit: bool = True, field_label: str = "Clast length"
                       ) -> Figure:
    """Log-x histogram with an adaptive bin count, KDE and log-normal fit.

    Bins: Freedman–Diaconis on log10 values, clamped to 8…60, geometric
    edges. D50 / D84 / D95 are marked with dotted lines and labelled in a
    strip **above** the axes (the φ axis is pushed outward to make room), so
    the labels never sit on the curves. Values below ``dmin_mm`` are shaded
    grey and labelled "below detection limit".

    ``fig.pm_meta``: ``n, bins, D50, D84, D95`` (display units) and
    ``fit = {mu_ln, sigma_ln, median, ks_stat, ks_p}`` (``None`` when the
    fit is off or SciPy is unavailable).
    """
    v = _clean_positive(values_m)
    if len(v) < 2:
        raise ValueError(f"n = {len(v)} finite positive values (< 2)")
    vd = v * scale
    if float(vd.max()) <= float(vd.min()):
        raise ValueError("all values are identical — no distribution to draw")
    mm_per_unit = _mm_per_display(unit, scale)
    nb = _fd_bins_log(vd)
    lo, hi = float(vd.min()), float(vd.max())
    edges = np.geomspace(lo, hi, nb + 1)
    d50, d84, d95 = (float(np.quantile(vd, q)) for q in (0.5, 0.84, 0.95))

    fit_meta = None
    with matplotlib.rc_context(_RC):
        fig = _new_figure(SINGLE_WIDTH_IN, 4.2)
        ax = fig.add_subplot(111)
        ax.hist(vd, bins=edges, density=True, color=_cm.viridis(0.45),
                edgecolor="white", linewidth=0.3, zorder=3,
                label=f"Histogram ({nb} bins)")
        if fit:
            try:
                from scipy.stats import gaussian_kde, lognorm, kstest
                xs = np.geomspace(lo, hi, 400)
                kde = gaussian_kde(vd)
                ax.plot(xs, kde(xs), color=_cm.viridis(0.0), lw=1.6,
                        zorder=5, label="KDE")
                shape, _loc, sc = lognorm.fit(vd, floc=0)
                ax.plot(xs, lognorm.pdf(xs, shape, loc=0, scale=sc),
                        color=ACCENT, lw=1.6, ls="--", zorder=6,
                        label="Log-normal fit")
                ks = kstest(vd, "lognorm", args=(shape, 0, sc))
                fit_meta = {
                    "mu_ln": float(np.log(sc)),
                    "sigma_ln": float(shape),
                    "median": float(sc),
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                }
            except Exception:
                fit_meta = None
        _apply_log_axis(ax)
        xlo, xhi = lo * 0.8, hi * 1.25
        dmin_display = None
        if dmin_mm is not None and np.isfinite(dmin_mm) and dmin_mm > 0:
            dmin_display = (float(dmin_mm) / mm_per_unit if mm_per_unit
                            else float(dmin_mm))
            xlo = min(xlo, dmin_display * 0.8)
            ax.axvspan(xlo, dmin_display, color=_cm.Greys(0.25), lw=0,
                       zorder=1)
            ax.text(dmin_display, 0.5, "below detection limit ",
                    transform=_mtrans.blended_transform_factory(
                        ax.transData, ax.transAxes),
                    rotation=90, ha="right", va="center",
                    fontsize=FONT_PT - 1, color="#444", zorder=12)
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel(f"{field_label} ({unit})" if unit else field_label)
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper right", framealpha=0.85, edgecolor="none")
        # Percentile lines inside; labels in a strip above the axes.
        # D84 sits one row higher than D50/D95 so neighbours never overlap.
        for name, val, row in (("D50", d50, 0), ("D84", d84, 1),
                               ("D95", d95, 0)):
            ax.axvline(val, color="#333", lw=0.8, ls=":", zorder=7)
            ax.annotate(f"{name} = {_fmt(val)}", xy=(val, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 3 + 12 * row), textcoords="offset points",
                        ha="center", va="bottom", fontsize=FONT_PT - 1,
                        color="#222", annotation_clip=False)
        _add_phi_axis(ax, xlo, xhi, mm_per_unit, outward_pt=30)
    fig.pm_meta = {
        "n": int(len(v)),
        "bins": int(nb),
        "D50": d50, "D84": d84, "D95": d95,
        "unit": unit,
        "fit": fit_meta,
        "dmin": dmin_display,
    }
    # The fit statistics are also exposed flat, which is how captions read
    # them (``meta.get("mu_ln")``); ``None`` when there is no fit.
    for key in ("mu_ln", "sigma_ln", "ks_stat", "ks_p"):
        fig.pm_meta[key] = fit_meta.get(key) if fit_meta else None
    return fig


# --- 4. validation_diagnostics ---
def validation_diagnostics(paired: pd.DataFrame, *, field: str = "Clast_length",
                           unit: str = "mm", scale: float = 1000.0,
                           truth_col: str = "truth", detect_col: str = "detect"
                           ) -> Figure:
    """1×3 panel: truth-vs-detection scatter, Bland–Altman, CDF overlay.

    ``paired`` holds one row per matched clast with the truth and detected
    value of ``field`` in metres. (a) scatter with the 1:1 line, R² (of the
    least-squares line) and n; (b) difference (detection − truth) against
    the pair mean with the bias and ±1.96 σ limits of agreement labelled;
    (c) the two empirical CDFs with D50 / D84 marks.

    Raises ``ValueError`` when fewer than 5 pairs remain or when the two
    columns are identical.

    ``fig.pm_meta``: ``n, r2, rmse, bias, sd_diff, loa_low, loa_high`` and
    ``D50/D84`` for each population, all in display units.
    """
    for c in (truth_col, detect_col):
        if c not in paired.columns:
            raise ValueError(f"paired is missing column '{c}'")
    t = pd.to_numeric(paired[truth_col], errors="coerce").to_numpy(float)
    d = pd.to_numeric(paired[detect_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(d) & (t > 0) & (d > 0)
    t, d = t[ok], d[ok]
    n = len(t)
    if n < 5:
        raise ValueError(f"n = {n} paired clasts (< 5)")
    if np.array_equal(t, d):
        raise ValueError("truth and detection are identical — no diagnostic")
    td, dd = t * scale, d * scale
    label = field.replace("_", " ").capitalize()
    ul = f" ({unit})" if unit else ""

    # Statistics
    diff = dd - td
    mean = (dd + td) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    if td.std() > 0 and dd.std() > 0:
        r2 = float(np.corrcoef(td, dd)[0, 1] ** 2)
    else:
        r2 = float("nan")

    with matplotlib.rc_context(_RC):
        fig = _new_figure(PANEL_WIDTH_IN, 3.1)
        axs = fig.subplots(1, 3)
        col = _cm.viridis(0.35)

        # (a) scatter
        ax = axs[0]
        lim_lo = 0.0
        lim_hi = float(max(td.max(), dd.max())) * 1.08
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#333", lw=0.8,
                ls="--", zorder=2, label="1:1")
        ax.scatter(td, dd, s=10, color=col, alpha=0.7, edgecolor="none",
                   zorder=3)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Truth {label.lower()}{ul}")
        ax.set_ylabel(f"Detected {label.lower()}{ul}")
        ax.text(0.04, 0.96, f"R² = {_fmt(r2, 2)}\nn = {n:,}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT_PT - 1)
        ax.set_title("(a) Truth vs detection", loc="left")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(_mtick.FuncFormatter(_plain_tick))

        # (b) Bland–Altman
        ax = axs[1]
        ax.scatter(mean, diff, s=10, color=col, alpha=0.7, edgecolor="none",
                   zorder=3)
        ax.axhline(0, color="#999", lw=0.6, zorder=1)
        ax.axhline(bias, color=ACCENT, lw=1.2, zorder=4)
        ax.axhline(loa_lo, color="#333", lw=0.8, ls="--", zorder=4)
        ax.axhline(loa_hi, color="#333", lw=0.8, ls="--", zorder=4)
        xr = ax.get_xlim()
        for yv, txt in ((bias, f"bias {_fmt(bias)}"),
                        (loa_hi, f"+1.96 σ {_fmt(loa_hi)}"),
                        (loa_lo, f"{MINUS}1.96 σ {_fmt(loa_lo)}")):
            ax.annotate(txt, xy=(xr[1], yv), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=FONT_PT - 2, color="#222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
        ax.set_xlabel(f"Mean of pair{ul}")
        ax.set_ylabel(f"Detection {MINUS} truth{ul}")
        ax.set_title("(b) Bland–Altman", loc="left")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(_mtick.FuncFormatter(_plain_tick))

        # (c) CDFs
        ax = axs[2]
        _apply_log_axis(ax)
        # The panel is too small for an inset legend: it goes below the axes.
        quant, _ = _draw_cdfs(ax, {"Truth": t, "Detection": d}, scale=scale,
                              marks=(50, 84), dmin_display={},
                              legend_loc="upper center",
                              legend_kwargs=dict(bbox_to_anchor=(0.5, -0.32),
                                                 ncol=2, columnspacing=1.0,
                                                 handlelength=1.6))
        ax.set_xlabel(f"{label}{ul}")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("(c) Size distributions", loc="left")

    fig.pm_meta = {
        "n": int(n),
        "r2": r2,
        "rmse": rmse,
        "bias": bias,
        "sd_diff": sd,
        "loa_low": float(loa_lo),
        "loa_high": float(loa_hi),
        "D50_truth": quant["Truth"][50],
        "D84_truth": quant["Truth"][84],
        "D50_detect": quant["Detection"][50],
        "D84_detect": quant["Detection"][84],
        "unit": unit,
    }
    return fig


# --- 5. cell_count_map ---
def cell_count_map(clasts: pd.DataFrame, *, cellsize_m: float = 1.0,
                   extent=None, crs_label: Optional[str] = None,
                   title: Optional[str] = None, min_n: int = 30) -> Figure:
    """Clasts per grid cell as a discrete viridis raster.

    ``clasts.x, clasts.y`` are projected coordinates in metres. ``extent``
    is ``(xmin, xmax, ymin, ymax)``; by default the data extent snapped
    outward to whole cells. Empty cells are left blank; occupied cells with
    fewer than ``min_n`` clasts are hatched (too few for a stable D50).
    Scale bar, north arrow, full coordinates and the ``crs_label`` box are
    drawn.

    ``fig.pm_meta``: ``n_min, n_median, n_max, frac_cells_lt30, n_cells``
    over occupied cells, plus ``n_cells_total, n_empty, cellsize_m``.
    """
    for c in ("x", "y"):
        if c not in clasts.columns:
            raise ValueError(f"clasts is missing column '{c}'")
    x = pd.to_numeric(clasts["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(clasts["y"], errors="coerce").to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        raise ValueError("no clasts with finite x, y")
    if not (np.isfinite(cellsize_m) and cellsize_m > 0):
        raise ValueError("cellsize_m must be a positive length in metres")
    cs = float(cellsize_m)
    if extent is None:
        xmin = math.floor(float(x.min()) / cs) * cs
        ymin = math.floor(float(y.min()) / cs) * cs
        xmax = max(math.ceil(float(x.max()) / cs) * cs, xmin + cs)
        ymax = max(math.ceil(float(y.max()) / cs) * cs, ymin + cs)
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in extent)
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("extent must be (xmin, xmax, ymin, ymax) with "
                             "xmax > xmin and ymax > ymin")
    nx = int(math.ceil((xmax - xmin) / cs - 1e-9))
    ny = int(math.ceil((ymax - ymin) / cs - 1e-9))
    if nx * ny > 4_000_000:
        raise ValueError(f"grid would have {nx * ny:,} cells; increase "
                         f"cellsize_m")
    xe = xmin + cs * np.arange(nx + 1)
    ye = ymin + cs * np.arange(ny + 1)
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    counts = H.T.astype(int)                    # rows = y, cols = x
    occupied = counts > 0
    if not occupied.any():
        raise ValueError("no clasts fall inside the extent")
    occ = counts[occupied]
    n_lt = int((occ < min_n).sum())

    # Discrete integer colour scale. Up to 12 clasts: one colour per count.
    # Beyond: equal integer ranges "1–8", "9–16", … so every occupied cell,
    # the sparsest included, sits inside a labelled class.
    vmax = int(occ.max())
    if vmax <= 12:
        bounds = np.arange(0.5, vmax + 1.5, 1.0)
        tick_pos = bounds[:-1] + 0.5
        tick_lab = [f"{int(v)}" for v in tick_pos]
    else:
        tv = _mtick.MaxNLocator(nbins=8, integer=True).tick_values(0, vmax)
        step = max(1, int(round(tv[1] - tv[0])))
        edges = list(range(0, vmax + step, step))
        if edges[-1] < vmax:
            edges.append(edges[-1] + step)
        bounds = np.array([e + 0.5 for e in edges], dtype=float)
        tick_pos = (bounds[:-1] + bounds[1:]) / 2.0
        tick_lab = [f"{e0 + 1:,}–{e1:,}" for e0, e1 in zip(edges[:-1],
                                                            edges[1:])]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(_cm.Greys(0.12))
    norm = _mcolors.BoundaryNorm(bounds, ncolors=cmap.N)
    masked = np.ma.masked_where(~occupied, counts)

    with matplotlib.rc_context(_RC):
        fig = _new_figure(SINGLE_WIDTH_IN,
                          _map_height(SINGLE_WIDTH_IN * 0.85,
                                      xe[-1] - xe[0], ye[-1] - ye[0]))
        ax = fig.add_subplot(111)
        im = ax.imshow(masked, extent=(xe[0], xe[-1], ye[0], ye[-1]),
                       origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest", zorder=2)
        # Hatch the under-populated cells.
        rects = []
        for r, c in zip(*np.where(occupied & (counts < min_n))):
            rects.append(Rectangle((xe[c], ye[r]), cs, cs))
        if rects:
            ax.add_collection(PatchCollection(
                rects, facecolor="none", edgecolor="white", hatch="////",
                linewidth=0.0, zorder=3))
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        # 'datalim' keeps the axes box where constrained layout put it and
        # pads the view instead — a fixed box under constrained layout can
        # shrink after the margins were computed and clip the y label.
        ax.set_aspect("equal", adjustable="datalim")
        _use_full_coordinates(ax)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02,
                            ticks=list(tick_pos))
        cbar.set_label("Clasts per cell (n)")
        cbar.ax.set_yticklabels(tick_lab)
        bar_m = _nice_length(xe[-1] - xe[0])
        _add_scale_bar(ax, bar_m, _length_label(bar_m))
        _add_north_arrow(ax)
        if crs_label:
            _crs_box(ax, str(crs_label))
        ax.legend(handles=[Patch(facecolor=_cm.viridis(0.5), edgecolor="white",
                                 hatch="////", label=f"n < {min_n}")],
                  loc="upper left", framealpha=0.85, edgecolor="none")
        if title:
            ax.set_title(title)
    fig.pm_meta = {
        "n_min": int(occ.min()),
        "n_median": float(np.median(occ)),
        "n_max": vmax,
        "frac_cells_lt30": float(n_lt / len(occ)),
        "n_cells": int(len(occ)),
        "n_cells_total": int(nx * ny),
        "n_empty": int(nx * ny - len(occ)),
        "cellsize_m": cs,
        "extent": (float(xe[0]), float(xe[-1]), float(ye[0]), float(ye[-1])),
        "min_n": int(min_n),
    }
    return fig


# --- 6. orientation_rose + 8. axial_circular_mean ---
def axial_circular_mean(deg) -> Tuple[float, float]:
    """``(mean_deg in [0, 180), R̄)`` of axial data by the doubled-angle method.

    Each direction θ is doubled (so 5° and 175° become 10° and 350°, which
    are neighbours), the unit vectors averaged, and the mean halved back.
    R̄ is the mean resultant length in the doubled space: 1 = perfectly
    aligned, 0 = no preferred direction.
    """
    a = np.asarray(deg, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if len(a) == 0:
        raise ValueError("no finite orientations")
    th = np.deg2rad(np.mod(a, 180.0)) * 2.0
    C, S = float(np.cos(th).mean()), float(np.sin(th).mean())
    R = math.hypot(C, S)
    mean = math.degrees(math.atan2(S, C)) / 2.0
    mean = mean % 180.0
    if mean >= 180.0 - 1e-9:
        mean = 0.0
    return float(mean), float(R)


def orientation_rose(orientation_deg, *, title: Optional[str] = None,
                     bin_deg: float = 10.0) -> Figure:
    """Axial rose of clast orientations (0–180° mirrored onto the circle).

    Angles are bearings as stored: clockwise from north (image-up), so the
    rose reads like a compass (0° up, 90° right). The axial mean direction
    is drawn as a diameter whose length is R̄ of the radial range, and the
    statistics are printed under the plot.

    ``fig.pm_meta``: ``mean_deg, R, n, bin_deg``.
    """
    a = np.asarray(orientation_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if len(a) < 2:
        raise ValueError(f"n = {len(a)} finite orientations (< 2)")
    if not (0 < bin_deg <= 90) or (180.0 / bin_deg) % 1 != 0:
        raise ValueError("bin_deg must divide 180 evenly")
    a = np.mod(a, 180.0)
    mean_deg, R = axial_circular_mean(a)
    nb = int(round(180.0 / bin_deg))
    counts, _edges = np.histogram(a, bins=nb, range=(0.0, 180.0))
    counts_full = np.concatenate([counts, counts])        # mirrored
    centres = np.deg2rad(np.arange(nb * 2) * bin_deg + bin_deg / 2.0)
    width = np.deg2rad(bin_deg)

    with matplotlib.rc_context(_RC):
        fig = _new_figure(SINGLE_WIDTH_IN, 6.0)
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(centres, counts_full, width=width, bottom=0.0,
               color=_cm.viridis(0.45), edgecolor="white", linewidth=0.5,
               alpha=0.95, zorder=3)
        rmax = float(counts_full.max()) * 1.05 if counts_full.max() > 0 else 1.0
        ax.set_ylim(0, rmax)
        th = np.deg2rad(mean_deg)
        ax.plot([th, th + np.pi], [R * rmax, R * rmax], color=ACCENT, lw=2.0,
                solid_capstyle="round", zorder=5,
                path_effects=[withStroke(linewidth=3.5, foreground="white")])
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
        ax.set_xticklabels([f"{d}°" for d in range(0, 360, 30)])
        ax.yaxis.set_major_formatter(_mtick.FuncFormatter(
            lambda v, _p: "" if v == 0 else f"{v:,.0f}"))
        ax.set_rlabel_position(255)
        ax.tick_params(axis="y", labelsize=FONT_PT - 1, colors="#444")
        ax.grid(True, alpha=0.35)
        ax.text(0.5, -0.06,
                f"axial mean {_fmt(mean_deg, 1)}°, "
                f"R̄ = {_fmt(R, 2)}, n = {len(a):,}, "
                f"{bin_deg:g}° bins",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=FONT_PT)
        if title:
            ax.set_title(title, pad=14)
    fig.pm_meta = {"mean_deg": mean_deg, "R": R, "n": int(len(a)),
                   "bin_deg": float(bin_deg)}
    return fig


# --- 7. zone_location_map ---
def _iter_geojson_features(path) -> List[dict]:
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        raise ValueError(f"cannot read GeoJSON {p.name}: "
                         f"{type(ex).__name__}: {ex}")
    if not isinstance(doc, dict):
        raise ValueError(f"{p.name} is not a GeoJSON object")
    gtype = doc.get("type")
    if gtype == "FeatureCollection":
        feats = doc.get("features") or []
    elif gtype == "Feature":
        feats = [doc]
    elif gtype in ("Polygon", "MultiPolygon", "LineString",
                   "MultiLineString", "Point", "MultiPoint"):
        feats = [{"type": "Feature", "properties": {}, "geometry": doc}]
    else:
        feats = []
    return [f for f in feats if isinstance(f, dict) and f.get("geometry")]


def _feature_label(feat: dict, fid: int) -> str:
    props = feat.get("properties") or {}
    for key in ("name", "Name", "NAME", "id", "ID", "label"):
        v = props.get(key)
        if v not in (None, ""):
            return str(v)
    if feat.get("id") not in (None, ""):
        return str(feat["id"])
    return str(fid)


def _geometry_parts(geom: dict) -> List[Tuple[str, np.ndarray]]:
    """``[(kind, (n, 2) array)]`` — kind in {'poly', 'line', 'point'}."""
    gt = geom.get("type")
    coords = geom.get("coordinates")
    parts: List[Tuple[str, np.ndarray]] = []
    try:
        if gt == "Polygon":
            parts.append(("poly", np.asarray(coords[0], float)[:, :2]))
        elif gt == "MultiPolygon":
            for poly in coords:
                parts.append(("poly", np.asarray(poly[0], float)[:, :2]))
        elif gt == "LineString":
            parts.append(("line", np.asarray(coords, float)[:, :2]))
        elif gt == "MultiLineString":
            for ln in coords:
                parts.append(("line", np.asarray(ln, float)[:, :2]))
        elif gt == "Point":
            parts.append(("point", np.asarray([coords[:2]], float)))
        elif gt == "MultiPoint":
            parts.append(("point", np.asarray(coords, float)[:, :2]))
    except (TypeError, IndexError, ValueError):
        return []
    return [(k, a) for k, a in parts if a.ndim == 2 and len(a) > 0]


def zone_location_map(image_path, geojson_paths: Sequence, *, transform=None,
                      labels: bool = True, title: Optional[str] = None,
                      max_px: int = 1500) -> Figure:
    """The ortho (downsampled to ≤ ``max_px``) with zone polygons / transects.

    Every feature of every GeoJSON file is drawn — polygons as outlines,
    LineStrings as lines, Points as markers — and labelled by its ``name``
    property (else ``id``, else its position in the file). With several
    files each file gets its own hue and a legend entry (file stem). The
    GeoJSON coordinates must be in the raster's frame: world coordinates
    for a georeferenced ortho, pixels for a plain photograph. Scale bar,
    north arrow and the CRS box are drawn only when the raster is
    georeferenced.

    ``fig.pm_meta``: ``n_features, labels, files, georeferenced, crs``.
    """
    paths = [Path(p) for p in (geojson_paths or [])]
    if not paths:
        raise ValueError("no GeoJSON files given")
    per_file: List[Tuple[Path, List[dict]]] = []
    for p in paths:
        feats = _iter_geojson_features(p)
        per_file.append((p, feats))
    n_feat = sum(len(f) for _p, f in per_file)
    if n_feat == 0:
        raise ValueError("no features in the GeoJSON files")

    ras = _Raster(image_path)
    try:
        t = _coerce_transform(transform) if transform is not None \
            else ras.transform
        img = ras.read(0, 0, ras.width, ras.height, max_px=max_px)
        if t is not None:
            x_left, y_top = _pix_to_world(t, 0, 0)
            x_right, y_bot = _pix_to_world(t, ras.width, ras.height)
            extent = (x_left, x_right, y_bot, y_top)
        else:
            extent = (0, ras.width, ras.height, 0)
        crs = ras.crs_label
    finally:
        ras.close()

    file_colors = ([ACCENT] if len(per_file) == 1
                   else _series_colors(len(per_file)))
    drawn_labels: List[str] = []
    with matplotlib.rc_context(_RC):
        fig = _new_figure(SINGLE_WIDTH_IN,
                          _map_height(SINGLE_WIDTH_IN,
                                      abs(extent[1] - extent[0]),
                                      abs(extent[3] - extent[2]),
                                      extra_in=0.7))
        ax = fig.add_subplot(111)
        ax.imshow(img, extent=extent, origin="upper",
                  interpolation="bilinear", zorder=1)
        halo = [withStroke(linewidth=3.0, foreground="white")]
        handles = []
        for k, (p, feats) in enumerate(per_file):
            col = file_colors[k]
            for fid, feat in enumerate(feats, start=1):
                parts = _geometry_parts(feat["geometry"])
                if not parts:
                    continue
                for kind, arr in parts:
                    if kind == "poly":
                        ax.plot(arr[:, 0], arr[:, 1], color=col, lw=1.4,
                                zorder=5, path_effects=halo)
                    elif kind == "line":
                        ax.plot(arr[:, 0], arr[:, 1], color=col, lw=1.6,
                                zorder=5, path_effects=halo,
                                marker="|", ms=6, mew=1.4)
                    else:
                        ax.plot(arr[:, 0], arr[:, 1], ls="none", marker="o",
                                ms=5, color=col, mec="white", zorder=6)
                if labels:
                    allpts = np.vstack([a for _k, a in parts])
                    cx, cy = allpts.mean(axis=0)
                    lab = _feature_label(feat, fid)
                    drawn_labels.append(lab)
                    box = dict(boxstyle="round,pad=0.2", fc="white",
                               ec=col, lw=0.8, alpha=0.85)
                    if any(kind == "poly" for kind, _a in parts):
                        # Above the polygon, so a small zone (a quadrat)
                        # stays visible under its label.
                        ax.annotate(lab, (cx, allpts[:, 1].max()),
                                    xytext=(0, 4), textcoords="offset points",
                                    ha="center", va="bottom", fontsize=FONT_PT,
                                    color="#111", zorder=7, bbox=box)
                    else:
                        ax.text(cx, cy, lab, ha="center", va="center",
                                fontsize=FONT_PT, color="#111", zorder=7,
                                bbox=box)
            if len(per_file) > 1:
                handles.append(Line2D([], [], color=col, lw=1.6,
                                      label=p.stem))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="datalim")   # see cell_count_map
        if t is not None:
            _use_full_coordinates(ax)
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            bar_m = _nice_length(abs(extent[1] - extent[0]))
            _add_scale_bar(ax, bar_m, _length_label(bar_m))
            _add_north_arrow(ax)
            if crs:
                _crs_box(ax, crs)
        else:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        if handles:
            ax.legend(handles=handles, loc="best", framealpha=0.85,
                      edgecolor="none")
        if title:
            ax.set_title(title)
    fig.pm_meta = {
        "n_features": int(n_feat),
        "labels": drawn_labels,
        "files": [str(p) for p in paths],
        "georeferenced": t is not None,
        "crs": crs,
    }
    return fig
