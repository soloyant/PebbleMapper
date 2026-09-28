"""Zonal statistics over rasterized grain-size maps.

For each polygon in a vector layer, summarise the underlying data within the
polygon footprint (count, mean, median, percentiles, std, Folk-Ward moments).
For each transect (LineString), sample a raster along the line at a user-set
step and emit a longitudinal profile, optionally co-sampling a DEM.

No GUI dependency: reads vector / raster paths and writes CSV / PNG.

Design notes
------------
* No `rasterstats` dependency: a pure-numpy mask-and-reduce loop over GDAL
  is fast enough for the polygon counts expected and is deterministic.
* Polygon -> mask is done by rasterizing the polygon onto the source raster's
  grid via gdal.RasterizeLayer.
* Transect -> samples are done analytically: walk each polyline in
  equal-distance steps and bilinear-interpolate (nearest is offered as a flag).
* Percentile labels follow the rest of the package: `D` letters are reserved
  for grain-size fields, everything else uses `P<q>` or `median`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from osgeo import gdal, ogr, osr

from functions.units import (
    is_size_field_for_phi,
    folk_ward_sorting_phi,
    folk_ward_skewness_phi,
    folk_ward_kurtosis_phi,
    field_display,
    field_unit_and_factor,
)


# --- Axis-label helper ---

def _humanize_axis_label(field_name: str, *, unit_override: Optional[str] = None
                         ) -> str:
    """Axis label from a raw column name: ``Clast_length`` -> ``"Clast length [mm]"``.

    Display name and unit come from the shared ``units`` registry;
    ``unit_override`` forces a unit string (the transect plotters display
    millimetres regardless of the field's stored metre unit).
    """
    display, unit, _factor = field_display(field_name)
    if not display:
        display = str(field_name).replace("_", " ").strip() or str(field_name)
    use_unit = unit_override if unit_override is not None else unit
    return f"{display} [{use_unit}]" if use_unit else display


# --- Geometry helpers ---

def _world_to_pixel(gt, x: float, y: float) -> tuple[float, float]:
    """Invert the GeoTransform: world (x,y) -> fractional pixel (col,row).

    GDAL convention::

        x_world = gt[0] + col * gt[1] + row * gt[2]
        y_world = gt[3] + col * gt[4] + row * gt[5]

    Works for north-up and rotated transforms.
    """
    g0, g1, g2, g3, g4, g5 = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
    det = g1 * g5 - g2 * g4
    if det == 0:
        raise ValueError("Degenerate GeoTransform (zero determinant)")
    dx = x - g0
    dy = y - g3
    col = (g5 * dx - g2 * dy) / det
    row = (-g4 * dx + g1 * dy) / det
    return col, row


# --- Polygon -> masked samples ---

def _polygon_mask(raster_ds, polygon_geom) -> np.ndarray:
    """Rasterize a single OGR polygon onto the source raster's grid; True inside.

    Uses an in-memory layer + RasterizeLayer because osgeo's Python wrapper
    has no pure-numpy mask helper.
    """
    cols = raster_ds.RasterXSize
    rows = raster_ds.RasterYSize
    gt = raster_ds.GetGeoTransform()
    proj = raster_ds.GetProjection()

    mem_drv = gdal.GetDriverByName("MEM")
    mask_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(gt)
    if proj:
        mask_ds.SetProjection(proj)

    vec_drv = ogr.GetDriverByName("Memory")
    vec_ds = vec_drv.CreateDataSource("mask_poly")
    srs = osr.SpatialReference()
    if proj:
        srs.ImportFromWkt(proj)
    layer = vec_ds.CreateLayer("poly", srs=srs, geom_type=ogr.wkbPolygon)
    feat_def = layer.GetLayerDefn()
    feat = ogr.Feature(feat_def)
    feat.SetGeometry(polygon_geom.Clone())
    layer.CreateFeature(feat)
    feat = None  # release

    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
    mask = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
    return mask


@dataclass
class PolygonStats:
    polygon_id: str
    band: int
    count: int
    mean: float
    std: float
    percentiles: dict[int, float]  # e.g. {16: ..., 50: ..., 84: ...}


@dataclass
class DistributionStats:
    """Per-polygon distribution summary computed from the vector CSV.

    Count + density, range, mean/median, spread (std, IQR, CV), shape
    (skewness, kurtosis), Folk-Ward moments in phi (NaN for non-size fields)
    and optional DEM summaries. Values are in the field's native unit.
    """
    polygon_id: str
    count: int
    area_m2: float       # geometric area of the polygon in CRS² units
    density: float       # count / area_m2 (clasts per m² for metric CRS)
    minimum: float
    maximum: float
    range: float
    mean: float
    median: float
    std: float
    cv: float            # coefficient of variation = std / mean
    iqr: float           # P75 − P25
    skewness: float      # statistical (Fisher-Pearson)
    kurtosis: float      # statistical excess kurtosis
    # Folk-Ward moments: NaN for non-size fields so the CSV schema stays constant.
    mean_phi: float
    sigma_phi: float     # Folk-Ward sorting σφ
    sk_phi: float        # Folk-Ward graphic skewness Sk_φ
    kg_phi: float        # Folk-Ward graphic kurtosis K_G
    percentiles: dict[int, float]
    # NaN when no DEM was supplied.
    dem_mean: float = float("nan")
    dem_std: float = float("nan")
    dem_min: float = float("nan")
    dem_max: float = float("nan")
    dem_range: float = float("nan")
    # "inside": count is a measured value (0 means no clasts here).
    # "outside": the polygon does not intersect the data, so count=0 is
    # missing data, not an observation.
    coverage: str = "inside"


def profile_out_path(explicit, base_dir, *, when=None):
    """Where a profile figure goes: the path the user typed, else a fresh
    timestamped one under ``base_dir``.

    ``build_profile_outputs`` appends ``__profile.png`` (and the three table
    names) to the stem, so a stem already used is a file already written: the
    auto name is never remembered and never reused. Keeping the first stamp
    in the field made every later render overwrite the figure and its three
    tables; two renders in the same second collide the
    same way, so an unused suffix is found here.
    """
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    if explicit:
        return _Path(str(explicit))
    base = _Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = (when or _dt.now()).strftime("%Y%m%d_%H%M%S")
    stem, n = f"zonal_{stamp}", 1
    while (base / f"{stem}__profile.png").exists():
        n += 1
        stem = f"zonal_{stamp}-{n}"
    return base / f"{stem}.png"


def _percentile_label(field_name: str, q: int) -> str:
    """Match the report-side label policy: D<q> for grain-size, P<q> else."""
    if is_size_field_for_phi(field_name):
        return f"D{q}"
    # P50, not "median": the moment block already has a column of that name,
    # and a CSV with two identical headers is read back wrong.
    return f"P{q}"


def _phi(D_meters_array: np.ndarray) -> np.ndarray:
    """Krumbein φ = −log₂(D / 1 mm), array form, positives only."""
    D_mm = np.asarray(D_meters_array, dtype=float) * 1000.0
    D_mm = D_mm[D_mm > 0]
    if D_mm.size == 0:
        return np.array([])
    return -np.log2(D_mm)


def _folk_skewness_from_phi(phi_vals: np.ndarray) -> float:
    """Folk-Ward graphic skewness Sk_φ from a φ-array. NaN-safe."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p16, p50, p84, p95 = np.percentile(phi_vals, [5, 16, 50, 84, 95])
    return folk_ward_skewness_phi(p5, p16, p50, p84, p95)


def _folk_kurtosis_from_phi(phi_vals: np.ndarray) -> float:
    """Folk-Ward graphic kurtosis K_G from a φ-array. NaN-safe."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p25, p75, p95 = np.percentile(phi_vals, [5, 25, 75, 95])
    return folk_ward_kurtosis_phi(p5, p25, p75, p95)


def _folk_sorting_phi(phi_vals: np.ndarray) -> float:
    """Folk-Ward (1957) graphic sorting sigma_phi; NaN below 5 values.
    Mirrors ``functions.validation._folk_sorting_phi``."""
    phi_vals = np.asarray(phi_vals, dtype=float)
    phi_vals = phi_vals[np.isfinite(phi_vals)]
    if phi_vals.size < 5:
        return float("nan")
    p5, p16, p84, p95 = np.percentile(phi_vals, [5, 16, 84, 95])
    return folk_ward_sorting_phi(p5, p16, p84, p95)


def _statistical_skewness(values: np.ndarray) -> float:
    """Fisher-Pearson moment skewness. NaN on too-small samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return float("nan")
    m = float(values.mean())
    s = float(values.std(ddof=0))
    if s == 0.0:
        return float("nan")
    return float(np.mean((values - m) ** 3) / (s ** 3))


def _statistical_kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis (kurtosis − 3). NaN on too-small samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    m = float(values.mean())
    s = float(values.std(ddof=0))
    if s == 0.0:
        return float("nan")
    return float(np.mean((values - m) ** 4) / (s ** 4) - 3.0)


def _polygon_area_from_geom(geom) -> float:
    """Geometric area of an OGR polygon in CRS² units. Robust to multi-polys."""
    try:
        return float(geom.GetArea())
    except Exception:
        return float("nan")


def _ring_to_path_coords(geom):
    """``[(x_array, y_array)]``, one per exterior ring of a (Multi)Polygon.
    Holes are ignored: clasts inside a hole count as inside."""
    rings: list[tuple[np.ndarray, np.ndarray]] = []
    gtype = geom.GetGeometryType()
    if gtype in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        ring = geom.GetGeometryRef(0)
        if ring is None:
            return rings
        n = ring.GetPointCount()
        if n < 3:
            return rings
        xs = np.empty(n, dtype=float)
        ys = np.empty(n, dtype=float)
        for i in range(n):
            x, y, *_ = ring.GetPoint(i)
            xs[i] = x
            ys[i] = y
        rings.append((xs, ys))
    elif gtype in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
        for k in range(geom.GetGeometryCount()):
            sub = geom.GetGeometryRef(k)
            rings.extend(_ring_to_path_coords(sub))
    return rings


def _points_in_rings(points_xy: np.ndarray,
                      rings: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Boolean mask: True where the point is inside any ring."""
    from matplotlib.path import Path as _MplPath
    if points_xy.size == 0 or not rings:
        return np.zeros(points_xy.shape[0], dtype=bool)
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    for xs, ys in rings:
        path = _MplPath(np.column_stack([xs, ys]))
        inside |= path.contains_points(points_xy)
    return inside


def _dem_stats_inside_polygon(
    dem_path: str | Path,
    geom,
) -> tuple[float, float, float, float, float]:
    """(mean, std, min, max, range) of DEM pixels inside the polygon; all NaN
    when the footprint is empty."""
    dds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
    if dds is None:
        return (float("nan"),) * 5
    band = dds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    mask = _polygon_mask(dds, geom)
    if not mask.any():
        dds = None
        return (float("nan"),) * 5
    arr = band.ReadAsArray().astype(np.float64)
    vals = arr[mask]
    if nodata is not None:
        vals = vals[(vals != nodata) & np.isfinite(vals)]
    else:
        vals = vals[np.isfinite(vals)]
    dds = None
    if vals.size == 0:
        return (float("nan"),) * 5
    return (
        float(vals.mean()),
        float(vals.std(ddof=0)),
        float(vals.min()),
        float(vals.max()),
        float(vals.max() - vals.min()),
    )


def describe_crs_mismatch(vector_path, raster_path) -> Optional[str]:
    """Human-readable message when a vector layer and a raster declare
    different coordinate reference systems, else ``None``.

    A CRS mismatch does not raise anywhere in the pipeline: zones silently
    land in the wrong place and the run reports success over an empty result.
    Returns ``None`` whenever the question cannot be answered (either side
    unreadable or carrying no CRS).
    """
    try:
        from osgeo import gdal, ogr, osr
    except Exception:
        return None
    # A GeoJSON with no "crs" member is WGS 84 by definition (RFC 7946), but
    # that is exactly what PebbleMapper writes for its own projected zone
    # sets, so an absent declaration means "unknown", not "WGS 84".
    if str(vector_path).lower().endswith((".geojson", ".json")):
        try:
            import json as _json
            doc = _json.loads(Path(vector_path).read_text(encoding="utf-8"))
            if not doc.get("crs"):
                return None
        except Exception:
            return None
    vsrs = rsrs = None
    try:
        vds = ogr.Open(str(vector_path), 0)
        if vds is not None:
            lyr = vds.GetLayer(0)
            if lyr is not None:
                vsrs = lyr.GetSpatialRef()
            vds = None
        ds = gdal.Open(str(raster_path))
        if ds is not None:
            wkt = ds.GetProjection()
            ds = None
            if wkt:
                rsrs = osr.SpatialReference()
                rsrs.ImportFromWkt(wkt)
    except Exception:
        return None
    if vsrs is None or rsrs is None:
        return None
    try:
        if vsrs.IsSame(rsrs):
            return None
        vname = vsrs.GetName() or "unknown"
        rname = rsrs.GetName() or "unknown"
    except Exception:
        return None
    return (f"The vector layer is in {vname} but the data is in {rname}. "
            f"Coordinates are compared as plain numbers, so the zones will "
            f"not land where you expect. Reproject one of them to match "
            f"(QGIS: Vector ▸ Data Management Tools ▸ Reproject Layer).")


def zonal_polygon_stats_from_csv(
    detection_csv: str | Path,
    vector_path: str | Path,
    out_csv: str | Path,
    *,
    field_name: str = "Clast_length",
    id_field: str = "",
    percentiles: Iterable[int] = (5, 16, 25, 50, 75, 84, 95),
    x_col: str = "x",
    y_col: str = "y",
    dem_path: Optional[str | Path] = None,
) -> list[DistributionStats]:
    """Per-polygon distribution stats computed from the vector clast CSV.

    The preferred polygon-mode entry point: each clast's centroid is tested
    for inclusion in each polygon, so percentiles come from the actual clast
    values rather than rasterised cell aggregates.

    Parameters
    ----------
    detection_csv
        Path to the per-clast CSV (``*_individual_clasts.csv`` or merged).
    vector_path
        OGR-readable polygon layer (shapefile, GeoJSON, GeoPackage...).
    out_csv
        Output CSV. Parent dir is auto-created.
    field_name
        Column to summarise (e.g. ``"Clast_length"``).
    id_field
        Attribute on the vector layer to use as the polygon ID column;
        falls back to the OGR FID when empty.
    percentiles
        Integer percentiles to compute.
    x_col, y_col
        Coordinate columns in the detection CSV, in the polygon layer's CRS.
    dem_path
        Optional DEM raster path; per-polygon elevation summaries are
        written when supplied.
    """
    detection_csv = str(detection_csv)
    vector_path = str(vector_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(detection_csv)
    for required in (x_col, y_col, field_name):
        if required not in df.columns:
            raise KeyError(
                f"Detection CSV missing column {required!r}. "
                f"Available: {list(df.columns)}")
    pts = df[[x_col, y_col]].to_numpy(dtype=float)
    vals_all = df[field_name].to_numpy(dtype=float)

    vds = ogr.Open(vector_path, 0)
    if vds is None:
        raise FileNotFoundError(f"Cannot open vector: {vector_path}")
    vlayer = vds.GetLayer(0)

    pcts = sorted({int(q) for q in percentiles if 0 <= int(q) <= 100})
    size_field = is_size_field_for_phi(field_name)
    rows_out: list[DistributionStats] = []

    # Bounding box of the clast centroids, to tell a measured zero apart from
    # a polygon that does not intersect the survey.
    if pts.size:
        _data_x0, _data_y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        _data_x1, _data_y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    else:
        _data_x0 = _data_y0 = _data_x1 = _data_y1 = float("nan")

    for feat in vlayer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        gtype = geom.GetGeometryType()
        if gtype not in (ogr.wkbPolygon, ogr.wkbMultiPolygon,
                         ogr.wkbPolygon25D, ogr.wkbMultiPolygon25D):
            continue
        if id_field:
            try:
                pid = str(feat.GetField(id_field))
            except Exception:
                pid = str(feat.GetFID())
        else:
            pid = str(feat.GetFID())

        area = _polygon_area_from_geom(geom)
        rings = _ring_to_path_coords(geom)
        mask_pts = _points_in_rings(pts, rings)
        vals = vals_all[mask_pts]
        # phi is undefined on D <= 0, so size fields drop non-positives.
        if size_field:
            vals = vals[np.isfinite(vals) & (vals > 0)]
        else:
            vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            stats = DistributionStats(
                polygon_id=pid, count=0, area_m2=area, density=0.0,
                minimum=float("nan"), maximum=float("nan"),
                range=float("nan"), mean=float("nan"),
                median=float("nan"), std=float("nan"),
                cv=float("nan"), iqr=float("nan"),
                skewness=float("nan"), kurtosis=float("nan"),
                mean_phi=float("nan"), sigma_phi=float("nan"),
                sk_phi=float("nan"), kg_phi=float("nan"),
                percentiles={q: float("nan") for q in pcts},
            )
        else:
            p25, p75 = np.percentile(vals, [25, 75])
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            stats = DistributionStats(
                polygon_id=pid,
                count=int(vals.size),
                area_m2=float(area),
                density=(float(vals.size) / float(area))
                          if area and np.isfinite(area) and area > 0
                          else float("nan"),
                minimum=float(vals.min()),
                maximum=float(vals.max()),
                range=float(vals.max() - vals.min()),
                mean=mean,
                median=float(np.percentile(vals, 50)),
                std=std,
                cv=(std / mean) if mean != 0 else float("nan"),
                iqr=float(p75 - p25),
                skewness=_statistical_skewness(vals),
                kurtosis=_statistical_kurtosis(vals),
                mean_phi=(float(_phi(vals).mean())
                          if size_field and _phi(vals).size else float("nan")),
                sigma_phi=(float(_folk_sorting_phi(_phi(vals)))
                           if size_field and _phi(vals).size >= 5
                           else float("nan")),
                sk_phi=(_folk_skewness_from_phi(_phi(vals))
                        if size_field else float("nan")),
                kg_phi=(_folk_kurtosis_from_phi(_phi(vals))
                        if size_field else float("nan")),
                percentiles={q: float(np.percentile(vals, q)) for q in pcts},
            )

        # OGR envelope is (minX, maxX, minY, maxY).
        try:
            ex0, ex1, ey0, ey1 = geom.GetEnvelope()
            if _data_x0 != _data_x0:  # NaN: no data points to compare with
                stats.coverage = "inside"
            elif (ex0 <= _data_x1 and ex1 >= _data_x0
                    and ey0 <= _data_y1 and ey1 >= _data_y0):
                stats.coverage = "inside"
            else:
                stats.coverage = "outside"
        except Exception:
            stats.coverage = "inside"

        if dem_path:
            dm, ds_, dmn, dmx, dr = _dem_stats_inside_polygon(dem_path, geom)
            stats.dem_mean = dm
            stats.dem_std = ds_
            stats.dem_min = dmn
            stats.dem_max = dmx
            stats.dem_range = dr

        rows_out.append(stats)

    vds = None

    # Column order groups the moments (mean, std, skewness, kurtosis), then
    # robust spread, then bounds, as pandas/scipy `describe()` readers expect.
    import csv
    pct_cols = [_percentile_label(field_name, q) for q in pcts]
    header = [
        "polygon_id", "count", "area_m2", "density",
        "mean", "std", "skewness", "kurtosis",
        "median", "iqr", "cv",
        "min", "max", "range",
        "mean_phi", "sigma_phi", "sk_phi", "kg_phi",
        *pct_cols,
    ]
    has_dem = dem_path is not None
    if has_dem:
        header += ["dem_mean", "dem_std", "dem_min", "dem_max", "dem_range"]
    # Field-derived statistics are written in DISPLAY units (mm for lengths)
    # so the CSV agrees with the report and the maps; the returned
    # DistributionStats stay in stored units. The trailing field/unit columns
    # record what was summarised. Columns not covered by `unit`: area_m2 (m2),
    # density (clasts per m2), cv / skewness / kurtosis / kg_phi
    # (dimensionless), mean_phi / sigma_phi / sk_phi (phi), dem_* (metres).
    from functions.units import field_unit_and_factor
    field_unit_str, unit_factor = field_unit_and_factor(field_name)

    def _conv(v):
        """Scale a field-derived statistic into display units."""
        return v * unit_factor

    header += ["coverage", "field", "unit"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows_out:
            row = [
                r.polygon_id, r.count,
                f"{r.area_m2:.6g}", f"{r.density:.6g}",
                f"{_conv(r.mean):.6g}", f"{_conv(r.std):.6g}",
                f"{r.skewness:.6g}", f"{r.kurtosis:.6g}",
                f"{_conv(r.median):.6g}", f"{_conv(r.iqr):.6g}", f"{r.cv:.6g}",
                f"{_conv(r.minimum):.6g}", f"{_conv(r.maximum):.6g}",
                f"{_conv(r.range):.6g}",
                f"{r.mean_phi:.6g}", f"{r.sigma_phi:.6g}",
                f"{r.sk_phi:.6g}", f"{r.kg_phi:.6g}",
                *[f"{_conv(r.percentiles[q]):.6g}" for q in pcts],
            ]
            if has_dem:
                row += [
                    f"{r.dem_mean:.6g}", f"{r.dem_std:.6g}",
                    f"{r.dem_min:.6g}", f"{r.dem_max:.6g}",
                    f"{r.dem_range:.6g}",
                ]
            row += [r.coverage, field_name, field_unit_str]
            w.writerow(row)

    return rows_out


def zonal_polygon_stats(
    raster_path: str | Path,
    vector_path: str | Path,
    out_csv: str | Path,
    *,
    field_name: str = "value",
    band: int = 1,
    id_field: str = "",
    percentiles: Iterable[int] = (16, 50, 84),
    nodata: Optional[float] = None,
) -> list[PolygonStats]:
    """Compute per-polygon zonal statistics and write a CSV.

    Parameters
    ----------
    raster_path
        Path to a GeoTIFF (typically the rasterize-tab output).
    vector_path
        Path to a polygon vector layer readable by OGR (shapefile, GeoJSON).
    out_csv
        Output CSV path. Existing files are overwritten.
    field_name
        Logical name of what the raster represents, e.g. ``"Clast_length"``
        or ``"Clast_circularity"``. Drives column labels (D50 vs P50 etc.).
    band
        Raster band index (1-based, GDAL convention). Default 1.
    id_field
        Attribute field on the vector layer to use as the polygon's ID
        column. Empty string falls back to the OGR FID.
    percentiles
        Integer percentiles in [0, 100] to compute per polygon. Default
        D16/D50/D84 (matches grain-size convention).
    nodata
        Optional nodata sentinel to exclude. If None, the raster's declared
        NoData value is used; if that's missing, NaN-handling is the only
        exclusion.

    Returns
    -------
    list[PolygonStats]
        One entry per polygon, in input order.
    """
    raster_path = str(raster_path)
    vector_path = str(vector_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if rds is None:
        raise FileNotFoundError(f"Cannot open raster: {raster_path}")
    rband = rds.GetRasterBand(int(band))
    if rband is None:
        raise ValueError(f"Band {band} not found in {raster_path}")
    arr = rband.ReadAsArray().astype(np.float64)
    if nodata is None:
        nodata = rband.GetNoDataValue()

    vds = ogr.Open(vector_path, 0)
    if vds is None:
        raise FileNotFoundError(f"Cannot open vector: {vector_path}")
    vlayer = vds.GetLayer(0)
    if vlayer is None:
        raise ValueError(f"No layer 0 in {vector_path}")

    rows: list[PolygonStats] = []
    pcts = sorted({int(q) for q in percentiles if 0 <= int(q) <= 100})

    for feat in vlayer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        gtype = geom.GetGeometryType()
        if gtype not in (ogr.wkbPolygon, ogr.wkbMultiPolygon,
                         ogr.wkbPolygon25D, ogr.wkbMultiPolygon25D):
            continue

        if id_field:
            try:
                poly_id = str(feat.GetField(id_field))
            except Exception:
                poly_id = str(feat.GetFID())
        else:
            poly_id = str(feat.GetFID())

        mask = _polygon_mask(rds, geom)
        if not mask.any():
            rows.append(PolygonStats(
                polygon_id=poly_id, band=band, count=0,
                mean=float("nan"), std=float("nan"),
                percentiles={q: float("nan") for q in pcts},
            ))
            continue

        values = arr[mask]
        finite = np.isfinite(values)
        if nodata is not None:
            finite &= (values != nodata)
        values = values[finite]

        if values.size == 0:
            rows.append(PolygonStats(
                polygon_id=poly_id, band=band, count=0,
                mean=float("nan"), std=float("nan"),
                percentiles={q: float("nan") for q in pcts},
            ))
            continue

        rows.append(PolygonStats(
            polygon_id=poly_id,
            band=band,
            count=int(values.size),
            mean=float(values.mean()),
            std=float(values.std(ddof=0)),
            percentiles={q: float(np.percentile(values, q)) for q in pcts},
        ))

    rds = None
    vds = None

    import csv
    header = ["polygon_id", "band", "count", "mean", "std"]
    for q in pcts:
        header.append(_percentile_label(field_name, q))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.polygon_id, r.band, r.count,
                f"{r.mean:.6g}", f"{r.std:.6g}",
                *[f"{r.percentiles[q]:.6g}" for q in pcts],
            ])
    return rows


# --- Transect sampling ---

@dataclass
class TransectSample:
    transect_id: str
    distance_m: np.ndarray   # 1D, monotonically increasing
    raster_value: np.ndarray # 1D, same shape
    elevation_m: Optional[np.ndarray] = None  # 1D or None


def _sample_raster_along(
    raster_arr: np.ndarray,
    gt,
    coords_world: np.ndarray,  # (N, 2) world (x, y)
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Sample a 2D raster along a sequence of world coordinates."""
    pix = np.empty_like(coords_world)
    for i in range(coords_world.shape[0]):
        col, row = _world_to_pixel(gt, coords_world[i, 0], coords_world[i, 1])
        pix[i, 0] = col
        pix[i, 1] = row

    rows = raster_arr.shape[0]
    cols = raster_arr.shape[1]
    if interpolation == "nearest":
        cc = np.clip(np.round(pix[:, 0]).astype(int), 0, cols - 1)
        rr = np.clip(np.round(pix[:, 1]).astype(int), 0, rows - 1)
        return raster_arr[rr, cc]

    # scipy is imported lazily so the engine stays importable without it.
    from scipy.ndimage import map_coordinates
    # map_coordinates expects (row, col) ordering.
    rc = np.stack([pix[:, 1], pix[:, 0]], axis=0)
    return map_coordinates(raster_arr, rc, order=1, mode="constant",
                            cval=np.nan)


def zonal_transect_profile(
    raster_path: str | Path,
    vector_path: str | Path,
    out_csv: str | Path,
    *,
    step_m: float = 0.5,
    band: int = 1,
    id_field: str = "",
    dem_path: Optional[str | Path] = None,
    interpolation: str = "bilinear",
    field_name: str = "",
) -> list[TransectSample]:
    """Sample a raster (and optionally a DEM) along each LineString.

    Parameters
    ----------
    raster_path
        Statistic raster to sample (e.g. D50 GeoTIFF from the Rasterize tab).
    vector_path
        Polyline vector layer (LineString or MultiLineString features).
    out_csv
        Long-format CSV with one row per sample point::

            transect_id,distance_m,raster_value[,elevation_m]
    step_m
        Sampling step along the line, in raster CRS units (typically metres).
    band
        Raster band index (1-based).
    id_field
        Attribute to use as the transect's ID column; falls back to FID.
    dem_path
        Optional DEM raster. Must share the same CRS / overlap the
        transects. Sampled with the same interpolation rule.
    interpolation
        ``"bilinear"`` (default) or ``"nearest"``.

    Returns
    -------
    list[TransectSample]
        One entry per LineString. MultiLineStrings are merged into a single
        TransectSample (segments concatenated with their cumulative distance).
    """
    raster_path = str(raster_path)
    vector_path = str(vector_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if rds is None:
        raise FileNotFoundError(f"Cannot open raster: {raster_path}")
    rband = rds.GetRasterBand(int(band))
    arr = rband.ReadAsArray().astype(np.float64)
    gt = rds.GetGeoTransform()

    # Mask no-data cells to NaN before sampling so the profile breaks over
    # empty ground instead of dropping to a spurious 0 mm. Two sentinels, as
    # in map_export._read_raster_layer: the declared NoDataValue and 0, which
    # clasts_rasterize writes for "no clasts in this cell".
    _nodata = rband.GetNoDataValue()
    if _nodata is not None:
        arr = np.where(arr == _nodata, np.nan, arr)
    arr = np.where(arr == 0, np.nan, arr)

    dem_arr = None
    dem_gt = None
    if dem_path:
        dds = gdal.Open(str(dem_path), gdal.GA_ReadOnly)
        if dds is None:
            raise FileNotFoundError(f"Cannot open DEM: {dem_path}")
        dem_arr = dds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        # 0 is a valid elevation (sea level): only the declared NoDataValue is masked.
        _dem_nodata = dds.GetRasterBand(1).GetNoDataValue()
        if _dem_nodata is not None:
            dem_arr = np.where(dem_arr == _dem_nodata, np.nan, dem_arr)
        dem_gt = dds.GetGeoTransform()
        dds = None

    vds = ogr.Open(vector_path, 0)
    if vds is None:
        raise FileNotFoundError(f"Cannot open vector: {vector_path}")
    vlayer = vds.GetLayer(0)

    results: list[TransectSample] = []

    def _walk_linestring(geom_ls) -> tuple[np.ndarray, np.ndarray]:
        """Return (cumulative distance, world-coord samples) for one LineString."""
        n = geom_ls.GetPointCount()
        if n < 2:
            return np.array([]), np.empty((0, 2))
        seg_pts: list[np.ndarray] = []
        seg_dist: list[np.ndarray] = []
        acc = 0.0
        for i in range(n - 1):
            x0, y0, *_ = geom_ls.GetPoint(i)
            x1, y1, *_ = geom_ls.GetPoint(i + 1)
            seg_len = float(np.hypot(x1 - x0, y1 - y0))
            if seg_len <= 0:
                continue
            n_samples = max(2, int(np.floor(seg_len / max(step_m, 1e-9))) + 1)
            t = np.linspace(0.0, 1.0, n_samples)
            xs = x0 + t * (x1 - x0)
            ys = y0 + t * (y1 - y0)
            ds = acc + t * seg_len
            seg_pts.append(np.stack([xs, ys], axis=1))
            seg_dist.append(ds)
            acc += seg_len
        if not seg_pts:
            return np.array([]), np.empty((0, 2))
        all_pts = np.concatenate(seg_pts, axis=0)
        all_d = np.concatenate(seg_dist, axis=0)
        return all_d, all_pts

    for feat in vlayer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        gtype = geom.GetGeometryType()
        if id_field:
            try:
                tid = str(feat.GetField(id_field))
            except Exception:
                tid = str(feat.GetFID())
        else:
            tid = str(feat.GetFID())

        if gtype in (ogr.wkbLineString, ogr.wkbLineString25D):
            d, pts = _walk_linestring(geom)
        elif gtype in (ogr.wkbMultiLineString, ogr.wkbMultiLineString25D):
            d_acc: list[np.ndarray] = []
            p_acc: list[np.ndarray] = []
            running = 0.0
            for k in range(geom.GetGeometryCount()):
                sub = geom.GetGeometryRef(k)
                d_k, p_k = _walk_linestring(sub)
                if d_k.size == 0:
                    continue
                d_acc.append(d_k + running)
                p_acc.append(p_k)
                running += d_k[-1]
            if not d_acc:
                continue
            d = np.concatenate(d_acc, axis=0)
            pts = np.concatenate(p_acc, axis=0)
        else:
            continue

        if d.size == 0:
            continue
        vals = _sample_raster_along(arr, gt, pts, interpolation=interpolation)
        elev = None
        if dem_arr is not None:
            elev = _sample_raster_along(dem_arr, dem_gt, pts,
                                          interpolation=interpolation)
        results.append(TransectSample(
            transect_id=tid,
            distance_m=d,
            raster_value=vals,
            elevation_m=elev,
        ))

    rds = None
    vds = None

    import csv
    header = ["transect_id", "distance_m", "raster_value"]
    has_dem = any(t.elevation_m is not None for t in results)
    if has_dem:
        header.append("elevation_m")
    # raster_value is written in the raster's STORED unit (metres for lengths):
    # the profile composer converts on render, so scaling here would
    # double-convert, and no `unit` column is emitted for the same reason.
    has_field = bool(field_name)
    if has_field:
        header.append("field")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for t in results:
            n = t.distance_m.size
            for i in range(n):
                row = [t.transect_id, f"{t.distance_m[i]:.6g}",
                       f"{t.raster_value[i]:.6g}"]
                if has_dem:
                    e = (t.elevation_m[i] if t.elevation_m is not None
                         else float("nan"))
                    row.append(f"{e:.6g}")
                if has_field:
                    row.append(field_name)
                w.writerow(row)
    return results


# --- Quick-look plotting (publication figures are produced by report.py) ---

def plot_transect_profile(
    sample: TransectSample,
    out_png: str | Path,
    *,
    field_name: str = "value",
    parameter: Optional[str] = None,
) -> Path:
    """Write a two-panel plot: raster value vs distance, DEM vs distance.

    ``parameter`` is the raster's per-cell statistic (``D50``, ``sorting``…,
    read from its file name by the caller): with it the values are converted
    to display units and labelled as that statistic (``Clast length — median
    [mm]``). The DEM panel is omitted when no elevation samples are present.
    """
    from functions.units import resolve_display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    has_dem = sample.elevation_m is not None
    fig, axes = plt.subplots(
        2 if has_dem else 1, 1,
        figsize=(8, 6 if has_dem else 3.5),
        sharex=True,
    )
    if not has_dem:
        axes = [axes]

    ax0 = axes[0]
    # Display units and label from the field and the statistic together, as
    # the Map tab labels the same raster.
    _disp, _unit, _factor, _cmap, _div = resolve_display(field_name, parameter)
    if not _disp:
        _disp = str(field_name).replace("_", " ") or "value"
    values = np.asarray(sample.raster_value, dtype=float) * float(_factor or 1.0)
    ax0.plot(sample.distance_m, values, "-", color="#225599", linewidth=1.4)
    ax0.set_ylabel(f"{_disp} [{_unit}]" if _unit else _disp)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(f"Transect {sample.transect_id}: {_disp}")

    if has_dem:
        ax1 = axes[1]
        ax1.plot(sample.distance_m, sample.elevation_m, "-",
                 color="#8b4513", linewidth=1.2)
        ax1.set_ylabel("Elevation (m)")
        ax1.set_xlabel("Distance along transect (m)")
        ax1.grid(True, alpha=0.3)
    else:
        ax0.set_xlabel("Distance along transect (m)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


# --- Combined plot composer ---

def plot_combined(
    layers: list[dict],
    out_png: str | Path,
    *,
    title: str = "",
    primary_label: Optional[str] = None,
    secondary_label: Optional[str] = None,
    figsize: tuple[float, float] = (9.0, 5.5),
    dpi: int = 160,
) -> Path:
    """Render a single multi-layer plot with optional secondary y-axis.

    Each ``layer`` is a dict with the following keys:

    * ``csv`` — path to a CSV emitted by the zonal engine. Typically a
      transect profile (long-format) or polygon summary (one row per
      polygon).
    * ``x_field`` — column to plot along the x-axis (e.g. ``distance_m``
      for transects, ``polygon_id`` for polygons).
    * ``y_field`` — column to plot. For envelope layers this is the
      *lower* bound; the upper bound is ``y2_field``.
    * ``y2_field`` (optional, envelope layers only) — upper bound column.
    * ``axis`` — ``"primary"`` or ``"secondary"``. Layers on the
      secondary axis are drawn on a twin y-axis aligned to the same x.
    * ``label`` — legend label.
    * ``color`` — any matplotlib colour spec.
    * ``style`` — ``"line"`` (default) or ``"envelope"``.

    The function returns the resolved output PNG path. Caller can stage
    1–6 layers; more are accepted but the legend gets crowded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = None  # created on the first secondary layer
    handles, labels = [], []

    for layer in layers:
        csv_path = layer.get("csv")
        if not csv_path or not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        x_field = layer.get("x_field") or ""
        y_field = layer.get("y_field") or ""
        if x_field not in df.columns or y_field not in df.columns:
            # One broken layer must not kill the whole render.
            continue

        axis_target = ax1
        if layer.get("axis") == "secondary":
            if ax2 is None:
                ax2 = ax1.twinx()
            axis_target = ax2

        color = layer.get("color") or None
        label = layer.get("label") or Path(csv_path).stem
        style = layer.get("style", "line")
        x = df[x_field].to_numpy()
        y = df[y_field].to_numpy()

        if style == "envelope":
            y2_field = layer.get("y2_field") or ""
            if y2_field not in df.columns:
                continue
            y2 = df[y2_field].to_numpy()
            handle = axis_target.fill_between(
                x, y, y2,
                color=color, alpha=0.25, linewidth=0,
                label=label,
            )
            handles.append(handle)
            labels.append(label)
        else:
            line, = axis_target.plot(
                x, y, "-", color=color, linewidth=1.5, label=label,
            )
            handles.append(line)
            labels.append(label)

    if primary_label:
        ax1.set_ylabel(primary_label)
    if ax2 is not None and secondary_label:
        ax2.set_ylabel(secondary_label)
    if title:
        ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    if handles:
        ax1.legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


# --- Publication-quality zonal map (product raster + zones + transects) ---
def _flatten_type(gtype):
    try:
        return ogr.GT_Flatten(gtype)
    except Exception:
        return gtype


def _exterior_rings(geom):
    """Exterior-ring coordinate lists for a (Multi)Polygon."""
    gt = _flatten_type(geom.GetGeometryType())
    rings = []
    if gt == ogr.wkbPolygon:
        if geom.GetGeometryCount() > 0:
            ring = geom.GetGeometryRef(0)  # exterior ring
            rings.append([(ring.GetX(i), ring.GetY(i))
                          for i in range(ring.GetPointCount())])
    elif gt == ogr.wkbMultiPolygon:
        for k in range(geom.GetGeometryCount()):
            rings.extend(_exterior_rings(geom.GetGeometryRef(k)))
    return rings


def _linestrings(geom):
    """Coordinate lists for a (Multi)LineString."""
    gt = _flatten_type(geom.GetGeometryType())
    lines = []
    if gt == ogr.wkbLineString:
        lines.append([(geom.GetX(i), geom.GetY(i))
                      for i in range(geom.GetPointCount())])
    elif gt == ogr.wkbMultiLineString:
        for k in range(geom.GetGeometryCount()):
            lines.extend(_linestrings(geom.GetGeometryRef(k)))
    return lines


def _feature_label(feat, gtype, polygon_id_field, transect_id_field):
    """Best-effort human label: the requested id field, else a common name
    property, else the OGR FID."""
    is_poly = gtype in (ogr.wkbPolygon, ogr.wkbMultiPolygon)
    field = polygon_id_field if is_poly else transect_id_field
    if field:
        try:
            v = feat.GetField(field)
            if v not in (None, ""):
                return str(v)
        except Exception:
            pass
    for cand in ("name", "Name", "NAME", "label", "Label", "id", "ID"):
        try:
            if feat.GetFieldIndex(cand) >= 0:
                v = feat.GetField(cand)
                if v not in (None, ""):
                    return str(v)
        except Exception:
            pass
    try:
        return str(feat.GetFID())
    except Exception:
        return ""


def plot_zonal_map(
    raster_path: str | Path,
    vector_path: str | Path,
    out_png: str | Path,
    *,
    field_name: str = "Clast_length",
    polygon_id_field: str = "",
    transect_id_field: str = "",
    cmap: str = "viridis",
    title: str = "",
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 200,
    show_north_arrow: bool = True,
    show_scale_bar: bool = True,
    log_fn: Optional[callable] = None,
) -> Path:
    """Publication-quality zonal map.

    Renders the *product raster* used for the zonal calculation as a coloured
    basemap (with a colourbar), then overlays the zone polygons and the
    transect lines, each labelled with its name/ID, plus a north arrow and a
    scale bar (the same decorations as the Map tab). Returns the PNG path.

    The vector layer may hold polygons (drawn as zones) and/or LineStrings
    (drawn as named transects); both are rendered when present, matching the
    Zonal tab's combined feature set.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from functions import map_export as _mx

    raster_path = str(raster_path)
    vector_path = str(vector_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    emit = log_fn or (lambda _m: None)

    arr, (xmin, xmax, ymin, ymax) = _mx._read_raster_layer(raster_path)
    # Display units, as the Map tab resolves them: a size raster stored in
    # metres is drawn in millimetres on both tabs, not metres on one of them.
    stem = Path(raster_path).stem
    parameter = _mx._parameter_from_stem(stem)
    unit_str, factor = _mx._resolve_unit_for_field(
        stem, arr[np.isfinite(arr)], "metric", "auto", parameter=parameter)
    arr = arr * factor
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)  # transparent no-data / empty cells
    im = ax.imshow(np.ma.masked_invalid(arr),
                   extent=(xmin, xmax, ymin, ymax), origin="upper",
                   cmap=cmap_obj, interpolation="nearest")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.ticklabel_format(style="plain", useOffset=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
    try:
        cbar.set_label(_mx._raster_label(stem, parameter, unit_str))
    except Exception:
        cbar.set_label(str(field_name))

    n_poly = n_line = 0
    vds = ogr.Open(vector_path, 0)
    if vds is None:
        emit(f"[zonal-map] could not open vector {vector_path}; raster only.")
    else:
        vlayer = vds.GetLayer(0)
        for feat in vlayer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            gt = _flatten_type(geom.GetGeometryType())
            label = _feature_label(feat, gt, polygon_id_field, transect_id_field)
            if gt in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
                for ring in _exterior_rings(geom):
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    # White halo under a thin black outline reads on any cmap.
                    ax.plot(xs, ys, color="white", lw=2.4, solid_capstyle="round")
                    ax.plot(xs, ys, color="black", lw=1.0, solid_capstyle="round")
                n_poly += 1
                try:
                    c = geom.Centroid()
                    if c is not None and label:
                        ax.annotate(label, (c.GetX(), c.GetY()), color="black",
                                    fontsize=9, fontweight="bold",
                                    ha="center", va="center",
                                    bbox=dict(boxstyle="round,pad=0.2",
                                              fc="white", ec="none", alpha=0.75))
                except Exception:
                    pass
            elif gt in (ogr.wkbLineString, ogr.wkbMultiLineString):
                for line in _linestrings(geom):
                    xs = [p[0] for p in line]
                    ys = [p[1] for p in line]
                    ax.plot(xs, ys, color="white", lw=4.0, solid_capstyle="round")
                    ax.plot(xs, ys, color="#d62728", lw=2.2,
                            solid_capstyle="round")
                    if label and xs:
                        mid = len(xs) // 2
                        ax.annotate(label, (xs[mid], ys[mid]), color="#d62728",
                                    fontsize=9, fontweight="bold",
                                    ha="left", va="bottom",
                                    bbox=dict(boxstyle="round,pad=0.2",
                                              fc="white", ec="none", alpha=0.8))
                n_line += 1
        vds = None

    ax.set_title(title or f"Zonal map — {Path(raster_path).stem}")
    if show_north_arrow:
        try:
            _mx._add_north_arrow(ax)
        except Exception:
            pass
    if show_scale_bar:
        try:
            _mx._add_scale_bar(ax, units="m")
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    emit(f"[zonal-map] wrote {out_png.name} "
         f"({n_poly} zone(s), {n_line} transect(s)).")
    return out_png


# --- Profile figures and tables: one transect, several dates, with topography ---

_PROFILE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ProfileSeries:
    """One transect, at one date, ready to plot.

    ``value`` stays in the CSV's stored unit; conversion happens once at
    render time so the file and the figure can never drift apart.
    """
    label: str
    date: str
    transect_id: str
    field: str
    distance_m: np.ndarray
    value: np.ndarray
    elevation_m: Optional[np.ndarray] = None


def infer_profile_date(csv_path) -> str:
    """Survey date for a transect CSV, taken from the project's date folder.

    Returns '' when no path component is a YYYY-MM-DD date, which is the
    normal case for a flat (undated) project.
    """
    try:
        parts = Path(csv_path).resolve().parts
    except Exception:
        parts = Path(csv_path).parts
    for part in reversed(parts):
        if _PROFILE_DATE_RE.match(part):
            return part
    return ""


def profile_label(date: str, transect_id: str) -> str:
    """Legend text for a series: "<date> - <transect>", or the transect alone
    when the project carries no date. A filename is never used."""
    date = (date or "").strip()
    tid = (transect_id or "").strip()
    if date and tid:
        return f"{date} — {tid}"
    return tid or date or "profile"


def load_profile_series(
    csv_paths,
    *,
    labels=None,
    distance_field: str = "distance_m",
    value_field: str = "raster_value",
    transect_id_field: str = "transect_id",
    elevation_field: str = "elevation_m",
) -> tuple[list[ProfileSeries], list[str]]:
    """Read transect CSVs into plottable series.

    Returns ``(series, skipped)``. ``skipped`` names files that carried no
    samples, so the caller can say which input was ignored instead of
    rendering an empty line for it.

    ``labels`` optionally overrides the auto-generated label per input path,
    keyed by the path as given — for an undated project, or for labels like
    "pre-storm".

    Raises ValueError when the inputs describe more than one field: plotting
    Clast_length and Equivalent_diameter on one axis states something untrue.
    """
    import pandas as pd

    labels = labels or {}
    series: list[ProfileSeries] = []
    skipped: list[str] = []
    fields_seen: dict[str, str] = {}
    seen_paths: set[str] = set()

    for raw in csv_paths:
        p = Path(raw)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen_paths:
            # One line, not two — but the caller has to know, or a second row
            # with its own label looks plotted while nothing changed and the
            # change table comes out empty.
            skipped.append(f"{p.name} (added twice; the second row was ignored)")
            continue
        seen_paths.add(key)
        # An unreadable file must not be reported like an empty one:
        # open_robust distinguishes missing, blocked and readable.
        from functions.io_robust import UnreadableFileError, open_robust
        try:
            with open_robust(p, "r", encoding="utf-8",
                             errors="replace") as _fh:
                df = pd.read_csv(_fh)
        except FileNotFoundError:
            skipped.append(f"{p.name} (not found)")
            continue
        except UnreadableFileError as _ex:
            skipped.append(f"{p.name} (could not be read: {_ex.reason})")
            continue
        except Exception as _ex:
            skipped.append(f"{p.name} (unparseable: {_ex})")
            continue
        if df.empty:
            skipped.append(f"{p.name} (carried no samples)")
            continue
        if distance_field not in df.columns:
            skipped.append(f"{p.name} (no {distance_field!r} column — not a "
                           "transect CSV?)")
            continue

        field = ""
        if "field" in df.columns and len(df):
            val = df["field"].iloc[0]
            if pd.notna(val):
                field = str(val).strip()
        if field:
            fields_seen.setdefault(field, p.name)

        override = labels.get(str(raw), labels.get(key))
        date = infer_profile_date(p)
        for tid, grp in df.groupby(transect_id_field, sort=True):
            grp = grp.sort_values(distance_field)
            dist = pd.to_numeric(grp[distance_field], errors="coerce").to_numpy(float)
            vals = pd.to_numeric(grp[value_field], errors="coerce").to_numpy(float)
            elev = None
            if elevation_field in grp.columns:
                e = pd.to_numeric(grp[elevation_field],
                                  errors="coerce").to_numpy(float)
                if np.isfinite(e).any():
                    elev = e
            if dist.size == 0:
                continue
            series.append(ProfileSeries(
                label=override or profile_label(date, str(tid)),
                date=date, transect_id=str(tid), field=field,
                distance_m=dist, value=vals, elevation_m=elev))

    if len(fields_seen) > 1:
        names = ", ".join(f"{f} (in {src})" for f, src in sorted(fields_seen.items()))
        raise ValueError(
            f"These inputs hold different measurements: {names}. Plotting them "
            f"on one axis would label them as the same quantity. Render one "
            f"field per figure.")
    return series, skipped


def _profile_display(series: list[ProfileSeries]) -> tuple[str, float, str]:
    """``(y_label, factor, unit)`` for a set of series."""
    field = next((s.field for s in series if s.field), "")
    if not field:
        # Older CSVs carry no field column.
        return "Value", 1.0, ""
    unit, factor = field_unit_and_factor(field)
    return _humanize_axis_label(field, unit_override=unit), factor, unit


def profile_summary_rows(series: list[ProfileSeries]) -> list[dict]:
    """One row per transect per date, in display units."""
    _lbl, factor, unit = _profile_display(series)
    rows = []
    for s in series:
        v = s.value[np.isfinite(s.value)] * factor
        d = s.distance_m[np.isfinite(s.distance_m)]
        length = float(d.max() - d.min()) if d.size else 0.0
        grad = float("nan")
        if v.size >= 2 and d.size == s.value.size and length > 0:
            ok = np.isfinite(s.value) & np.isfinite(s.distance_m)
            if ok.sum() >= 2:
                grad = float(np.polyfit(s.distance_m[ok],
                                        s.value[ok] * factor, 1)[0])
        rows.append({
            "date": s.date, "transect_id": s.transect_id,
            "field": s.field, "unit": unit,
            "n_samples": int(v.size),
            "length_m": f"{length:.6g}",
            "mean": f"{float(v.mean()):.6g}" if v.size else "nan",
            "median": f"{float(np.median(v)):.6g}" if v.size else "nan",
            "min": f"{float(v.min()):.6g}" if v.size else "nan",
            "max": f"{float(v.max()):.6g}" if v.size else "nan",
            "gradient_per_m": f"{grad:.6g}",
        })
    return rows


def profile_change_rows(series: list[ProfileSeries]) -> list[dict]:
    """Consecutive date pairs per transect, with the difference stated."""
    _lbl, factor, unit = _profile_display(series)
    by_transect: dict[str, list[ProfileSeries]] = {}
    for s in series:
        by_transect.setdefault(s.transect_id, []).append(s)
    rows = []
    for tid, group in sorted(by_transect.items()):
        group = sorted(group, key=lambda s: s.date)
        for a, b in zip(group, group[1:]):
            va = a.value[np.isfinite(a.value)] * factor
            vb = b.value[np.isfinite(b.value)] * factor
            if not va.size or not vb.size:
                continue
            mean_a, mean_b = float(va.mean()), float(vb.mean())
            med_a, med_b = float(np.median(va)), float(np.median(vb))
            rows.append({
                "transect_id": tid, "field": a.field, "unit": unit,
                "date_from": a.date, "date_to": b.date,
                "mean_from": f"{mean_a:.6g}", "mean_to": f"{mean_b:.6g}",
                "mean_difference": f"{mean_b - mean_a:.6g}",
                "median_from": f"{med_a:.6g}", "median_to": f"{med_b:.6g}",
                "median_difference": f"{med_b - med_a:.6g}",
            })
    return rows


def _validate_bin_width(bin_width_m) -> float:
    """Bin width as a positive float, or raise. Shared so the orchestrator
    can reject a bad width before it writes anything."""
    if not bin_width_m or float(bin_width_m) <= 0:
        raise ValueError(
            f"Bin width must be greater than zero (got {bin_width_m!r}).")
    return float(bin_width_m)


def profile_binned_rows(series: list[ProfileSeries],
                        bin_width_m: float = 1.0) -> list[dict]:
    """Mean value per fixed distance bin, short enough to print."""
    bin_width_m = _validate_bin_width(bin_width_m)
    _lbl, factor, unit = _profile_display(series)
    rows = []
    for s in series:
        ok = np.isfinite(s.distance_m) & np.isfinite(s.value)
        if not ok.any():
            continue
        d = s.distance_m[ok]
        v = s.value[ok] * factor
        d0 = float(d.min())
        length = float(d.max()) - d0
        n_bins = max(1, int(math.ceil(length / bin_width_m)))
        for i in range(n_bins):
            lo = d0 + i * bin_width_m
            hi = lo + bin_width_m
            sel = (d >= lo) & (d < hi) if i < n_bins - 1 else (d >= lo)
            vals = v[sel]
            rows.append({
                "date": s.date, "transect_id": s.transect_id,
                "field": s.field, "unit": unit,
                "bin_start_m": f"{lo:.6g}", "bin_end_m": f"{hi:.6g}",
                "n_samples": int(vals.size),
                "mean": f"{float(vals.mean()):.6g}" if vals.size else "nan",
            })
    return rows


def _write_rows_csv(rows: list[dict], path: Path) -> Path:
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if not rows:
            fh.write("")
            return path
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def plot_profile_figure(
    series: list[ProfileSeries],
    out_png,
    *,
    title: str = "",
    caption: str = "",
    figsize: tuple[float, float] = (9.0, 6.0),
    dpi: int = 160,
) -> Path:
    """Stacked profile figure: grain size above, elevation below.

    Two panels sharing one distance axis rather than a dual y-axis, which
    invites misattributing a curve to the wrong scale. The elevation panel is
    omitted when no series carries elevation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    y_label, factor, _unit = _profile_display(series)
    has_elev = any(s.elevation_m is not None for s in series)

    if has_elev:
        fig, (ax, ax_e) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
        ax_e = None

    for s in series:
        ax.plot(s.distance_m, s.value * factor, lw=1.8, label=s.label)
        if ax_e is not None and s.elevation_m is not None:
            ax_e.plot(s.distance_m, s.elevation_m, lw=1.4, label=s.label)

    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    # Legend outside the axes so it never covers the data.
    if series:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, frameon=False)
    if ax_e is not None:
        ax_e.set_ylabel("Elevation [m]")
        ax_e.grid(True, alpha=0.3)
        ax_e.set_xlabel("Distance along transect [m]")
    else:
        ax.set_xlabel("Distance along transect [m]")

    if caption:
        fig.text(0.01, 0.005, caption, fontsize=8, va="bottom", ha="left",
                 wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_profile_outputs(
    csv_paths,
    out_dir,
    stem: str,
    *,
    labels=None,
    bin_width_m: float = 1.0,
    title: str = "",
    caption: str = "",
) -> dict:
    """Render the profile figure and write its three tables.

    Returns ``{"png", "summary", "change", "binned", "skipped", "series"}``.
    """
    out_dir = Path(out_dir)
    # Validate before writing anything: a rejected run must leave nothing behind.
    bin_width_m = _validate_bin_width(bin_width_m)
    series, skipped = load_profile_series(csv_paths, labels=labels)
    if not series:
        raise ValueError(
            "No transect samples to plot"
            + (f" — every input was skipped: {'; '.join(skipped)}."
               if skipped else "."))
    png = plot_profile_figure(series, out_dir / f"{stem}__profile.png",
                              title=title, caption=caption)
    summary = _write_rows_csv(profile_summary_rows(series),
                              out_dir / f"{stem}__summary.csv")
    change = _write_rows_csv(profile_change_rows(series),
                             out_dir / f"{stem}__change.csv")
    binned = _write_rows_csv(profile_binned_rows(series, bin_width_m),
                             out_dir / f"{stem}__binned.csv")
    return {"png": png, "summary": summary, "change": change,
            "binned": binned, "skipped": skipped, "series": series}


__all__ = [
    "PolygonStats",
    "DistributionStats",
    "TransectSample",
    "ProfileSeries",
    "zonal_polygon_stats",
    "zonal_polygon_stats_from_csv",
    "zonal_transect_profile",
    "plot_transect_profile",
    "plot_combined",
    "plot_zonal_map",
    "infer_profile_date",
    "profile_label",
    "load_profile_series",
    "profile_summary_rows",
    "profile_change_rows",
    "profile_binned_rows",
    "plot_profile_figure",
    "build_profile_outputs",
]
