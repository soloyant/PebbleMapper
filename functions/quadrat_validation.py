"""Quadrat-aware validation engine.

Three quadrat-specification modes, each producing a polygon (or buffered
point) that both the truth and the detection CSVs are clipped against
before any comparison:

* ``geotiff_aligned``: the truth image is itself a georeferenced raster whose
  bounds define the quadrat.
* ``centroid_dims``: an axis-aligned rectangle of the given width x height
  centred on a known centroid.
* ``point_corner``: one recorded point with its compass identity (N, NE, E,
  SE, S, SW, W, NW) plus the nominal dimensions. The point is translated by
  half the diagonal toward the centroid, and the footprint is a disc of that
  radius, generous enough to contain the true quadrat whatever the field
  team's corner-pointing convention.

GSD handling
------------
``detect_gsd_from_path()`` returns the ground sampling distance (m/pixel) of
an image from, in order of preference, a non-identity GDAL GeoTransform, a
filename token such as ``GSD=0.0023m`` or the legacy ``GSD=2p100mm``, else
``None`` (the caller must supply a manual GSD). The returned dict flags
``source`` so the UI can show where the value came from.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from osgeo import gdal

from functions.gsd import parse_gsd_from_filename


# 8-point compass -> unit vector pointing OUTWARD from the centroid.
_CORNER_UNIT = {
    "N":  ( 0.0,  1.0),
    "NE": ( 1.0,  1.0),
    "E":  ( 1.0,  0.0),
    "SE": ( 1.0, -1.0),
    "S":  ( 0.0, -1.0),
    "SW": (-1.0, -1.0),
    "W":  (-1.0,  0.0),
    "NW": (-1.0,  1.0),
}


# --- GSD auto-detection ---

# gdal.Open() on a CSV prints a noisy "ERROR 4: not recognized" line on
# stderr; only open extensions GDAL could plausibly read. HEIC/HEIF stay
# out on purpose: GDAL has no HEIF driver in this environment (checked
# 2026-09-11, GDAL 3.9.3), and a photograph carries no geotransform anyway.
_GDAL_READABLE_EXTS = {
    ".tif", ".tiff", ".vrt", ".img", ".jp2", ".png", ".jpg", ".jpeg",
    ".bmp", ".gif", ".hdf", ".nc", ".dem", ".asc", ".bil", ".dt2",
}


def _gsd_from_geotransform(path: str) -> Optional[float]:
    """The column step's length (hypot of the x-scale and rotation terms,
    so a rotated georectified quadrat reads its true pixel size) when the
    dataset has a non-identity affine, else None."""
    if not path:
        return None
    if Path(path).suffix.lower() not in _GDAL_READABLE_EXTS:
        return None
    # Keep a soft failure on a recognised extension out of the log.
    try:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
    except Exception:
        pass
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
    except Exception:
        ds = None
    finally:
        try:
            gdal.PopErrorHandler()
        except Exception:
            pass
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    ds = None
    if gt is None:
        return None
    if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        return None
    import math
    return float(math.hypot(float(gt[1]), float(gt[2])))


def detect_gsd_from_path(path: str) -> dict:
    """``{"gsd": float|None, "source": "geotransform"|"filename"|None}``."""
    if not path or not Path(path).exists():
        return {"gsd": None, "source": None}
    gt = _gsd_from_geotransform(path)
    if gt is not None:
        return {"gsd": gt, "source": "geotransform"}
    fn = parse_gsd_from_filename(path)
    if fn is not None:
        return {"gsd": fn, "source": "filename"}
    return {"gsd": None, "source": None}


# --- Quadrat footprint construction ---

@dataclass
class QuadratFootprint:
    """A quadrat spec resolved into a clipping geometry.

    ``kind`` is ``"box"`` (axis-aligned rectangle) or ``"buffer"`` (disc).
    ``bounds`` is always (xmin, ymin, xmax, ymax) of the bounding box.
    ``polygon_xy`` is the closed boundary for the point-in-footprint test
    (four corners + closing vertex, or an N-segment circle).
    """
    kind: str
    bounds: tuple[float, float, float, float]
    polygon_xy: list[tuple[float, float]]
    centroid: tuple[float, float]
    radius: float = 0.0      # only meaningful when kind == "buffer"
    width: float = 0.0       # only meaningful when kind == "box"
    height: float = 0.0


def placement_provenance(path: str) -> dict:
    """How the quadrat in this raster came to be where it is.

    ``{"placement": "fitted"|"hand-edited"|"manual"|"", "agreement_score":
    float|None}``, read from the GDAL metadata ``georef.write_georeferenced``
    stores on the raster itself, so a metric over a mixed set can be
    recomputed over fitted placements only.
    """
    out = {"placement": "", "agreement_score": None}
    try:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    except Exception:
        ds = None
    if ds is None:
        return out
    meta = ds.GetMetadata() or {}
    ds = None
    out["placement"] = str(meta.get("PM_PLACEMENT", "") or "")
    raw = meta.get("PM_AGREEMENT_SCORE")
    if raw is not None:
        try:
            out["agreement_score"] = float(raw)
        except (TypeError, ValueError):
            pass
    return out


def raster_world_bounds(gt, cols: int, rows: int):
    """``(xmin, ymin, xmax, ymax)`` of a raster in world units, from all
    four corners of its affine geotransform. A georectified quadrat is a
    rotated raster (its rotation terms gt[2] and gt[4] are not zero): taken
    from the origin and the far corner alone, the box of a square rotated
    by 40° was a strip a tenth as high as the quadrat."""
    xs, ys = [], []
    for c, r in ((0, 0), (cols, 0), (0, rows), (cols, rows)):
        xs.append(gt[0] + c * gt[1] + r * gt[2])
        ys.append(gt[3] + c * gt[4] + r * gt[5])
    return min(xs), min(ys), max(xs), max(ys)


def quadrat_from_geotiff(truth_path: str) -> QuadratFootprint:
    """Build a footprint from a georeferenced truth GeoTIFF's bounds."""
    ds = gdal.Open(truth_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open truth raster: {truth_path}")
    gt = ds.GetGeoTransform()
    if gt is None or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        ds = None
        raise ValueError(
            f"{truth_path}: not georeferenced (identity GeoTransform). "
            "Use centroid_dims or point_corner mode instead.")
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    # All four corners: north-up rasters have gt[5] < 0, rotated ones
    # (every georectified quadrat) have gt[2] and gt[4] as well.
    xmin, ymin, xmax, ymax = raster_world_bounds(gt, cols, rows)
    ds = None
    poly = [
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax),
        (xmin, ymin),
    ]
    return QuadratFootprint(
        kind="box",
        bounds=(xmin, ymin, xmax, ymax),
        polygon_xy=poly,
        centroid=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0),
        width=xmax - xmin,
        height=ymax - ymin,
    )


def quadrat_from_centroid(cx: float, cy: float,
                            width: float,
                            height: Optional[float] = None) -> QuadratFootprint:
    """Axis-aligned rectangle of width × height centred on (cx, cy)."""
    if height is None or height <= 0:
        height = width
    hw = width / 2.0
    hh = height / 2.0
    xmin, xmax = cx - hw, cx + hw
    ymin, ymax = cy - hh, cy + hh
    poly = [
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax),
        (xmin, ymin),
    ]
    return QuadratFootprint(
        kind="box",
        bounds=(xmin, ymin, xmax, ymax),
        polygon_xy=poly,
        centroid=(cx, cy),
        width=width,
        height=height,
    )


def quadrat_from_point_and_corner(
    px: float, py: float,
    corner: str,
    width: float,
    height: Optional[float] = None,
    n_segments: int = 64,
) -> QuadratFootprint:
    """Buffered-disc footprint from a single recorded point + its corner identity.

    The point is translated inward, opposite to its compass identity, by
    half the quadrat diagonal to the inferred centroid; the footprint is a
    disc of that radius, generous enough to absorb pointing error and any
    rotation of the quadrat in the field.
    """
    if height is None or height <= 0:
        height = width
    diag = math.hypot(width, height)
    half_diag = diag / 2.0

    corner_key = (corner or "").upper().strip()
    if corner_key not in _CORNER_UNIT:
        raise ValueError(
            f"Unknown corner identity {corner!r}. Expected one of "
            f"{sorted(_CORNER_UNIT)}.")
    ux, uy = _CORNER_UNIT[corner_key]
    # Normalise so diagonal corners don't get sqrt(2) x half-diag.
    mag = math.hypot(ux, uy) or 1.0
    ux, uy = ux / mag, uy / mag

    cx = px - ux * half_diag
    cy = py - uy * half_diag

    poly = []
    for i in range(n_segments + 1):
        a = 2.0 * math.pi * i / n_segments
        poly.append((cx + half_diag * math.cos(a),
                      cy + half_diag * math.sin(a)))
    return QuadratFootprint(
        kind="buffer",
        bounds=(cx - half_diag, cy - half_diag,
                cx + half_diag, cy + half_diag),
        polygon_xy=poly,
        centroid=(cx, cy),
        radius=half_diag,
        width=width,
        height=height,
    )


# --- Clipping ---

def clip_to_footprint(df: pd.DataFrame,
                       footprint: QuadratFootprint,
                       x_col: str = "x",
                       y_col: str = "y") -> pd.DataFrame:
    """Rows of df whose (x, y) lies inside the quadrat footprint.

    A bounding-box pre-filter precedes the point-in-polygon test.
    """
    from matplotlib.path import Path as _MplPath
    if df.empty:
        return df.copy()
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"DataFrame missing {x_col!r}/{y_col!r}. "
            f"Available: {list(df.columns)}")
    xmin, ymin, xmax, ymax = footprint.bounds
    inside_box = (
        (df[x_col] >= xmin) & (df[x_col] <= xmax) &
        (df[y_col] >= ymin) & (df[y_col] <= ymax)
    )
    sub = df.loc[inside_box].copy()
    if sub.empty or footprint.kind == "box":
        return sub
    path = _MplPath(np.array(footprint.polygon_xy))
    pts = sub[[x_col, y_col]].to_numpy()
    return sub.loc[path.contains_points(pts)].copy()


# --- Nodata-aware clipping ---
# A quadrat truth GeoTIFF's bounding box includes nodata margins
# (fully-dark or fully-bright pixels left by orthorectification), so a UAV
# detection landing on one is inside the extent but outside the real
# photographic coverage, and counting it as a false positive would be
# mis-attribution. A valid-pixel mask built from the truth image (the same
# nodata convention as ``map_export._read_ortho_image``) drops those.


def valid_pixel_mask_from_image(image_path: str) -> Optional[dict]:
    """Build a (mask, gt) bundle describing valid photographic pixels.

    Returns ``None`` when the image isn't georeferenced. Otherwise::

        {"mask": np.ndarray[bool, (H, W)],
         "gt":   tuple(GeoTransform),
         "W":    int, "H": int}

    Band 4 is treated as alpha when present; otherwise pixels at the dtype
    extremes across all RGB bands count as nodata.
    """
    ds = gdal.Open(str(image_path), gdal.GA_ReadOnly)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    if gt is None or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        ds = None
        return None
    n_bands = ds.RasterCount
    W = ds.RasterXSize
    H = ds.RasterYSize
    bands = []
    for i in range(1, min(5, n_bands + 1)):
        bands.append(ds.GetRasterBand(i).ReadAsArray())
    ds = None
    if not bands:
        return None

    if n_bands >= 4:
        alpha = bands[3]
        mask = alpha > 0
        return {"mask": mask, "gt": gt, "W": W, "H": H}

    arr0 = bands[0]
    dtype = arr0.dtype
    if dtype == np.uint8:
        dark_v, bright_v = 0, 255
    elif dtype == np.uint16:
        dark_v, bright_v = 0, 65535
    else:
        dark_v = float(arr0.min())
        bright_v = float(arr0.max())
    if len(bands) >= 3:
        full_dark = np.logical_and(
            np.logical_and(bands[0] <= dark_v, bands[1] <= dark_v),
            bands[2] <= dark_v)
        full_bright = np.logical_and(
            np.logical_and(bands[0] >= bright_v, bands[1] >= bright_v),
            bands[2] >= bright_v)
    else:
        full_dark = arr0 <= dark_v
        full_bright = arr0 >= bright_v
    mask = ~(full_dark | full_bright)
    return {"mask": mask, "gt": gt, "W": W, "H": H}


def _world_to_pixel(gt, x: float, y: float) -> tuple[int, int]:
    """Invert the GeoTransform: world (x, y) -> integer pixel (col, row).

    GDAL convention::

        x_world = gt[0] + col * gt[1] + row * gt[2]
        y_world = gt[3] + col * gt[4] + row * gt[5]

    so the forward matrix is ``[[gt[1], gt[2]], [gt[4], gt[5]]]`` with
    determinant ``gt[1]*gt[5] - gt[2]*gt[4]``.
    """
    g0, g1, g2, g3, g4, g5 = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
    det = g1 * g5 - g2 * g4
    if det == 0:
        return -1, -1
    dx = x - g0
    dy = y - g3
    col = (g5 * dx - g2 * dy) / det
    row = (-g4 * dx + g1 * dy) / det
    return int(round(col)), int(round(row))


def reproject_pixels_to_world(
    image_path: str,
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    y_up: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Forward-transform PIXEL coordinates to world coordinates.

    A clast CSV measured on a photograph (Quadrat-mode detection, Digitize)
    stores each centroid in pixels with ``x = col`` and ``y = height - row``
    (y counted up from the bottom edge, see the clast table in docs/user-manual.md), whereas
    an ortho detection CSV is already in the world CRS. ``y_up=False``
    reads ``y`` as a row index instead, for CSVs made by other tools. The
    full affine GeoTransform is applied (a scalar GSD multiply would ignore
    the origin and the north-up row flip)::

        row     = height - y        (y_up, the PebbleMapper convention)
        x_world = gt[0] + col * gt[1] + row * gt[2]
        y_world = gt[3] + col * gt[4] + row * gt[5]

    Returns ``(df, reprojected)``. ``df`` is returned unchanged (flag
    ``False``) when the frame is empty or lacks the columns, the image is not
    georeferenced, or the coordinates already look like world coordinates
    (the majority of finite points fall inside the raster's world bounds),
    so the function is safe to call on either frame.
    """
    if df is None or df.empty:
        return df, False
    if x_col not in df.columns or y_col not in df.columns:
        return df, False
    ds = gdal.Open(str(image_path), gdal.GA_ReadOnly)
    if ds is None:
        return df, False
    gt = ds.GetGeoTransform()
    W = ds.RasterXSize
    H = ds.RasterYSize
    ds = None
    if gt is None or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        return df, False

    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        return df, False

    # Pixel coordinates of this raster all lie on its grid; anything
    # beyond it is a world frame already. The earlier test — "most points
    # inside the raster's world bounds" — took a whole-ortho CSV in world
    # coordinates for quadrat pixels, because most of the ortho lies
    # outside one quadrat.
    margin_x, margin_y = 0.01 * W + 1.0, 0.01 * H + 1.0
    on_grid = (
        (xs[finite] >= -margin_x) & (xs[finite] <= W + margin_x) &
        (ys[finite] >= -margin_y) & (ys[finite] <= H + margin_y)
    )
    if not on_grid.all():
        return df, False

    rows = (float(H) - ys) if y_up else ys
    out = df.copy()
    out[x_col] = gt[0] + xs * gt[1] + rows * gt[2]
    out[y_col] = gt[3] + xs * gt[4] + rows * gt[5]
    return out, True


def clip_to_valid_pixels(
    df: pd.DataFrame,
    mask_bundle: dict,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Drop rows whose (x, y) world coord lands on a False pixel of
    ``mask_bundle`` (from ``valid_pixel_mask_from_image``) or outside it."""
    if df.empty or mask_bundle is None:
        return df.copy()
    mask = mask_bundle["mask"]
    gt = mask_bundle["gt"]
    W = mask_bundle["W"]
    H = mask_bundle["H"]
    if x_col not in df.columns or y_col not in df.columns:
        return df.copy()
    keep = np.zeros(len(df), dtype=bool)
    xs = df[x_col].to_numpy(dtype=float)
    ys = df[y_col].to_numpy(dtype=float)
    for i in range(len(df)):
        col, row = _world_to_pixel(gt, xs[i], ys[i])
        if 0 <= col < W and 0 <= row < H:
            keep[i] = bool(mask[row, col])
    return df.loc[keep].copy()


__all__ = [
    "QuadratFootprint",
    "detect_gsd_from_path",
    "placement_provenance",
    "quadrat_from_geotiff",
    "quadrat_from_centroid",
    "quadrat_from_point_and_corner",
    "clip_to_footprint",
    "valid_pixel_mask_from_image",
    "reproject_pixels_to_world",
    "clip_to_valid_pixels",
]
