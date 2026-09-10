import pandas as pd
import scipy
import math
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import operations as op
from functions._logging import get_logger
from functions.units import (
    folk_ward_sorting_phi,
    folk_ward_skewness_phi,
    folk_ward_kurtosis_phi,
)

_log = get_logger(__name__)

# Krumbein φ reference grain size: φ = -log2(D/D0), D0 = 1 mm (Krumbein 1934).
_PHI_REF_M = 1e-3

# Percentile ranks for d_percentiles and folk_ward_* outputs.
D_PERCENTILES = [0.05, 0.16, 0.50, 0.84, 0.95]


# --- Internal helpers ---


def mode_value(values) -> float:
    """The most frequent value of ``values`` as a float, NaN when there is
    none. scipy's mode() returns a ModeResult (an array in older releases,
    a scalar since 1.11); storing it in a cell raised 'float() argument
    must be... not ModeResult'."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    res = scipy.stats.mode(arr, nan_policy='omit')
    m = np.asarray(getattr(res, "mode", res)).ravel()
    return float(m[0]) if m.size else np.nan

def _compute_raster_grid(local_clasts, field, parameter, cellsize, percentile,
                          bin_edges, bin_mode, bin_field, min_cell_density):
    """Per-cell statistics grid from a clast DataFrame whose per-field
    transforms are already applied; x/y in metric CRS units, cellsize in the
    same units. Cells with fewer than ``min_cell_density`` clasts become NaN.

    Returns (p, density_grid, bin_labels, bins_m, n_rows, n_cols): ``p`` is
    ``(n_rows, n_cols)`` for scalar parameters or ``(n_rows, n_cols, n_bands)``
    for multi-band ones; ``bin_labels``/``bins_m`` (metres) are None for
    single-band output.
    """
    local_clasts = local_clasts.copy()
    local_clasts['y'] = local_clasts['y'] - np.min(local_clasts['y'])
    local_clasts['x'] = local_clasts['x'] - np.min(local_clasts['x'])
    n_rows = math.ceil(np.max(local_clasts['y']) / cellsize)
    n_cols = math.ceil(np.max(local_clasts['x']) / cellsize)

    p = np.zeros((n_rows, n_cols))
    p1 = np.zeros((n_rows, n_cols))
    p2 = np.zeros((n_rows, n_cols))
    if str.lower(parameter) == "distribution":
        p  = np.zeros((n_rows, n_cols, 100))
        p1 = np.zeros((n_rows, n_cols, 100))
        p2 = np.zeros((n_rows, n_cols, 100))
    if str.lower(parameter) == "d_percentiles":
        p = np.zeros((n_rows, n_cols, len(D_PERCENTILES)))

    PHI_REF_M = _PHI_REF_M

    _bins_m = None
    _bin_labels = None
    if (bin_edges is not None and len(bin_edges) > 0 and
            str.lower(parameter) in ("packing_index", "packing_clustering")):
        edges = np.asarray(bin_edges, dtype=float)
        if str.lower(bin_mode) == "percentile":
            src = (local_clasts[bin_field].dropna().values
                   if bin_field in local_clasts.columns else [])
            if len(src) == 0:
                raise ValueError(
                    f"bin_mode='percentile' but no usable values in column "
                    f"'{bin_field}'.")
            edges = np.quantile(src, np.clip(edges, 0.0, 1.0))
        edges = np.sort(np.unique(edges))
        _bins_m = np.concatenate([[0.0], edges, [np.inf]])

        def _fmt(v):
            mm = v * 1000.0
            return f"{int(round(mm))}" if mm >= 1 else f"{mm:.1f}"

        labels = []
        for i in range(len(_bins_m) - 1):
            lo, hi = _bins_m[i], _bins_m[i + 1]
            labels.append(f"{_fmt(lo)}+mm" if hi == np.inf
                          else f"{_fmt(lo)}-{_fmt(hi)}mm")
        _bin_labels = labels

    if str.lower(parameter) == "packing_index" and _bins_m is not None:
        p = np.zeros((n_rows, n_cols, len(_bin_labels)))
    if str.lower(parameter) == "packing_clustering" and _bins_m is None:
        raise ValueError(
            "parameter='packing_clustering' requires bin_edges. "
            "Pass at least one edge (in metres) so the per-cell size "
            "distribution can be evaluated across bins.")

    density_grid = np.zeros((n_rows, n_cols), dtype=int)

    # Kurtosis/skewness/std/mode are undefined for axial circular data
    # (Mardia & Jupp §13.1): emit NaN and a warning, not a meaningless raster.
    _ORIENTATION_INVALID_PARAMS = {
        "kurtosis", "skewness", "std", "mode", "cv",
        "folk_ward_kurtosis", "folk_ward_skewness",
    }
    _orientation_with_invalid_param = (
        str.lower(field) == "orientation"
        and str.lower(parameter) in _ORIENTATION_INVALID_PARAMS
    )
    if _orientation_with_invalid_param:
        import warnings
        warnings.warn(
            f"parameter={parameter!r} is not defined for axial circular data "
            f"(field='Orientation'). The output raster will be all-NaN. "
            "Use parameter='average', 'quantile', or 'density' for orientation.",
            UserWarning, stacklevel=3,
        )

    for m in tqdm(range(0, n_rows)):
        for n in range(0, n_cols):
            crop = local_clasts[
                (local_clasts['y'] >= m * cellsize) &
                (local_clasts['y'] <  (m + 1) * cellsize) &
                (local_clasts['x'] >= n * cellsize) &
                (local_clasts['x'] <  (n + 1) * cellsize)
            ]
            density_grid[m, n] = len(crop)
            if str.lower(field) == "orientation":
                if _orientation_with_invalid_param:
                    p[m, n] = np.nan
                    continue
                if np.shape(crop)[0] > 0:
                    if str.lower(parameter) == "quantile":
                        p1[m, n] = np.nanquantile(crop['u'], percentile)
                        p2[m, n] = np.nanquantile(crop['v'], percentile)
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "density":
                        p[m, n] = (np.shape(crop[field])[0] + 1) / (cellsize ** 2)
                    if str.lower(parameter) == "average":
                        p1[m, n] = np.nanmean(crop['u'])
                        p2[m, n] = np.nanmean(crop['v'])
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "mode":
                        p1[m, n] = mode_value(crop['u'])
                        p2[m, n] = mode_value(crop['v'])
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "kurtosis":
                        p1[m, n] = scipy.stats.kurtosis(crop['u'], nan_policy='omit')
                        p2[m, n] = scipy.stats.kurtosis(crop['v'], nan_policy='omit')
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "skewness":
                        p1[m, n] = scipy.stats.skew(crop['u'], nan_policy='omit')
                        p2[m, n] = scipy.stats.skew(crop['v'], nan_policy='omit')
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "std":
                        p1[m, n] = np.nanstd(crop['u'])
                        p2[m, n] = np.nanstd(crop['v'])
                        p[m, n] = (np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
                                   / 2.0)
                    if str.lower(parameter) == "distribution":
                        for l in range(1, 100):
                            p1[m, n, l - 1] = np.nanquantile(crop['u'], l / 100)
                            p2[m, n, l - 1] = np.nanquantile(crop['v'], l / 100)
                            p[m, n, l - 1] = np.rad2deg(
                                op.cart2pol(p1[m, n, l - 1], p2[m, n, l - 1])[1])
                    if str.lower(parameter) == "sorting":
                        # Graphic-spread arithmetic on the (u, v) components,
                        # not the Folk-Ward grain-size σφ; deliberately not
                        # routed through units.folk_ward_sorting_phi.
                        p1[m, n] = (
                            (np.nanquantile(crop['u'], 0.84) -
                             np.nanquantile(crop['u'], 0.16)) / 4
                            + (np.nanquantile(crop['u'], 0.95) -
                               np.nanquantile(crop['u'], 0.05)) / 6.6
                        )
                        p2[m, n] = (
                            (np.nanquantile(crop['v'], 0.84) -
                             np.nanquantile(crop['v'], 0.16)) / 4
                            + (np.nanquantile(crop['v'], 0.95) -
                               np.nanquantile(crop['v'], 0.05)) / 6.6
                        )
                        p[m, n] = np.rad2deg(op.cart2pol(p1[m, n], p2[m, n])[1])
            else:
                if np.shape(crop)[0] > 0:
                    if str.lower(parameter) == "quantile":
                        p[m, n] = np.nanquantile(crop[field], percentile)
                    if str.lower(parameter) == "density":
                        p[m, n] = (np.shape(crop[field])[0] + 1) / (cellsize ** 2)
                    if str.lower(parameter) == "average":
                        p[m, n] = np.nanmean(crop[field])
                    if str.lower(parameter) == "mode":
                        p[m, n] = mode_value(crop[field])
                    if str.lower(parameter) == "kurtosis":
                        p[m, n] = scipy.stats.kurtosis(crop[field], nan_policy='omit')
                    if str.lower(parameter) == "skewness":
                        p[m, n] = scipy.stats.skew(crop[field], nan_policy='omit')
                    if str.lower(parameter) == "std":
                        p[m, n] = np.nanstd(crop[field])
                    if str.lower(parameter) == "cv":
                        # std / mean; NaN where the mean is zero or undefined.
                        _mu = np.nanmean(crop[field])
                        p[m, n] = (np.nanstd(crop[field]) / _mu
                                   if (np.isfinite(_mu) and _mu != 0)
                                   else np.nan)
                    if str.lower(parameter) == "distribution":
                        for l in range(1, 100):
                            p[m, n, l - 1] = np.nanquantile(crop[field], l / 100)
                    if str.lower(parameter) == "sorting":
                        # Graphic spread on the raw field values (no φ
                        # transform), distinct from "folk_ward_sorting" below;
                        # deliberately not routed through
                        # units.folk_ward_sorting_phi.
                        p[m, n] = (
                            (np.nanquantile(crop[field], 0.84) -
                             np.nanquantile(crop[field], 0.16)) / 4
                            + (np.nanquantile(crop[field], 0.95) -
                               np.nanquantile(crop[field], 0.05)) / 6.6
                        )
                    if str.lower(parameter) == "d_percentiles":
                        for li, q in enumerate(D_PERCENTILES):
                            p[m, n, li] = np.nanquantile(crop[field], q)
                    if str.lower(parameter) in ("folk_ward_sorting",
                                                  "folk_ward_skewness",
                                                  "folk_ward_kurtosis"):
                        D_values = crop[field].values
                        D_values = D_values[D_values > 0]
                        if len(D_values) >= 5:
                            phi_values = -np.log2(D_values / PHI_REF_M)
                            if str.lower(parameter) == "folk_ward_sorting":
                                phi_05 = np.nanquantile(phi_values, 0.05)
                                phi_16 = np.nanquantile(phi_values, 0.16)
                                phi_84 = np.nanquantile(phi_values, 0.84)
                                phi_95 = np.nanquantile(phi_values, 0.95)
                                p[m, n] = folk_ward_sorting_phi(
                                    phi_05, phi_16, phi_84, phi_95)
                            elif str.lower(parameter) == "folk_ward_skewness":
                                phi_05 = np.nanquantile(phi_values, 0.05)
                                phi_16 = np.nanquantile(phi_values, 0.16)
                                phi_50 = np.nanquantile(phi_values, 0.50)
                                phi_84 = np.nanquantile(phi_values, 0.84)
                                phi_95 = np.nanquantile(phi_values, 0.95)
                                p[m, n] = folk_ward_skewness_phi(
                                    phi_05, phi_16, phi_50, phi_84, phi_95)
                            elif str.lower(parameter) == "folk_ward_kurtosis":
                                phi_05 = np.nanquantile(phi_values, 0.05)
                                phi_25 = np.nanquantile(phi_values, 0.25)
                                phi_75 = np.nanquantile(phi_values, 0.75)
                                phi_95 = np.nanquantile(phi_values, 0.95)
                                p[m, n] = folk_ward_kurtosis_phi(
                                    phi_05, phi_25, phi_75, phi_95)
                        else:
                            p[m, n] = np.nan
                    if str.lower(parameter) == "packing_index":
                        cell_area = float(cellsize) ** 2
                        if 'Surface_area' not in crop.columns:
                            if _bins_m is None:
                                p[m, n] = np.nan
                            else:
                                p[m, n, :] = np.nan
                        elif _bins_m is None:
                            covered = float(crop['Surface_area'].sum())
                            p[m, n] = max(0.0, min(1.0, covered / cell_area))
                        else:
                            sizes = crop.get(bin_field)
                            areas = crop['Surface_area'].values
                            if sizes is None:
                                p[m, n, :] = np.nan
                            else:
                                idx = (np.digitize(sizes.values, _bins_m,
                                                   right=False) - 1)
                                idx = np.clip(idx, 0, len(_bin_labels) - 1)
                                for b in range(len(_bin_labels)):
                                    s = (float(areas[idx == b].sum())
                                         if (idx == b).any() else 0.0)
                                    p[m, n, b] = max(0.0, min(1.0, s / cell_area))
                    if str.lower(parameter) == "packing_clustering":
                        if ('Surface_area' not in crop.columns
                                or bin_field not in crop.columns):
                            p[m, n] = np.nan
                        else:
                            sizes = crop[bin_field].values
                            areas = crop['Surface_area'].values
                            idx = (np.digitize(sizes, _bins_m, right=False) - 1)
                            idx = np.clip(idx, 0, len(_bin_labels) - 1)
                            per_bin = np.zeros(len(_bin_labels))
                            for b in range(len(_bin_labels)):
                                if (idx == b).any():
                                    per_bin[b] = float(areas[idx == b].sum())
                            tot = per_bin.sum()
                            if tot <= 0 or np.count_nonzero(per_bin) < 2:
                                p[m, n] = np.nan
                            else:
                                pi = per_bin / tot
                                nz = pi[pi > 0]
                                H = -np.sum(nz * np.log(nz))
                                p[m, n] = float(H / np.log(len(_bin_labels)))

    if min_cell_density and min_cell_density > 0 and \
            str.lower(parameter) != "density":
        p = p.astype(np.float64)
        low = density_grid < int(min_cell_density)
        if p.ndim == 2:
            p[low] = np.nan
        else:
            p[low, ...] = np.nan
        _log.info(
            "Applied min_cell_density=%d: %d of %d cells set to NaN.",
            min_cell_density, int(low.sum()), p.shape[0] * p.shape[1])

    return p, density_grid, _bin_labels, _bins_m, n_rows, n_cols


def _write_raster(p, north_up_gt, crs_wkt, RasterFileWritingPath,
                   parameter, _bin_labels, _bins_m, sidecar_meta):
    """Write the result array (2-D, or 3-D for multi-band) to a GeoTIFF with
    a north-up geotransform ``[origin_x, cellsize, 0, origin_y_top, 0,
    -cellsize]`` and the source CRS, plus a sidecar JSON carrying
    ``sidecar_meta`` verbatim."""
    n_rows, n_cols = p.shape[0], p.shape[1]
    arr_out = np.flip(p, axis=0).copy()
    driver = gdal.GetDriverByName("GTiff")
    _log.info("Saving raster to %s …", RasterFileWritingPath)

    if str.lower(parameter) == "distribution":
        n_band = arr_out.shape[2] if arr_out.ndim == 3 else 1
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows,
                                n_band - 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(north_up_gt)
        outdata.SetProjection(crs_wkt)
        for i in range(1, n_band):
            outdata.GetRasterBand(i).WriteArray(arr_out[:, :, i - 1])
            outdata.GetRasterBand(i).SetNoDataValue(0)
            outdata.GetRasterBand(i).SetDescription('D' + str(i))

    elif str.lower(parameter) == "d_percentiles":
        n_band = len(D_PERCENTILES)
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows,
                                n_band, gdal.GDT_Float64)
        outdata.SetGeoTransform(north_up_gt)
        outdata.SetProjection(crs_wkt)
        band_names = [f'D{int(round(q * 100))}' for q in D_PERCENTILES]
        for i, name in enumerate(band_names):
            outdata.GetRasterBand(i + 1).WriteArray(arr_out[:, :, i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(0)
            outdata.GetRasterBand(i + 1).SetDescription(name)

    elif str.lower(parameter) == "packing_index" and _bins_m is not None:
        n_band = len(_bin_labels)
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows,
                                n_band, gdal.GDT_Float64)
        outdata.SetGeoTransform(north_up_gt)
        outdata.SetProjection(crs_wkt)
        for i, name in enumerate(_bin_labels):
            outdata.GetRasterBand(i + 1).WriteArray(arr_out[:, :, i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(0)
            outdata.GetRasterBand(i + 1).SetDescription(name)

    else:
        outdata = driver.Create(RasterFileWritingPath, n_cols, n_rows,
                                1, gdal.GDT_Float64)
        outdata.SetGeoTransform(north_up_gt)
        outdata.SetProjection(crs_wkt)
        outdata.GetRasterBand(1).WriteArray(arr_out)
        outdata.GetRasterBand(1).SetNoDataValue(0)

    outdata.FlushCache()
    outdata = None
    del outdata

    # Sidecar JSON, best-effort.
    try:
        import json as _json
        from datetime import datetime as _dt
        try:
            from functions import __version__ as _ver
        except Exception:
            _ver = "dev"
        _n_bands = 1
        if str.lower(parameter) == "distribution":
            _n_bands = max(1, p.shape[2] - 1) if p.ndim == 3 else 1
        elif str.lower(parameter) == "d_percentiles":
            _n_bands = len(D_PERCENTILES)
        elif str.lower(parameter) == "packing_index" and _bins_m is not None:
            _n_bands = len(_bin_labels)
        meta = {
            "schema_version":    1,
            "rasterize_version": str(_ver),
            "generated_at":      _dt.now().isoformat(timespec="seconds"),
            "raster_path":       os.path.basename(RasterFileWritingPath),
            "parameter":         parameter,
            "n_bands":           int(_n_bands),
            **sidecar_meta,
        }
        sidecar = RasterFileWritingPath + ".json"
        with open(sidecar, "w", encoding="utf-8") as fh:
            _json.dump(meta, fh, indent=2, ensure_ascii=False)
        _log.info("Sidecar written: %s", sidecar)
    except Exception as _meta_ex:
        _log.warning("Sidecar JSON write failed: %s", _meta_ex)


def _render_raster_figure(p, raster, field, parameter, percentile, figuresize):
    """Render a stacked ortho-image / raster diagnostic figure."""
    image = np.dstack([
        raster.GetRasterBand(1).ReadAsArray(),
        raster.GetRasterBand(2).ReadAsArray(),
        raster.GetRasterBand(3).ReadAsArray(),
    ])
    fig = plt.figure(figsize=figuresize)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(image, interpolation='none')
    ax1.set_title("Ortho-image")
    ax2 = fig.add_subplot(2, 1, 2)
    pos = ax2.imshow(np.flipud(p), interpolation='none')
    if parameter == "quantile":
        ax2.set_title("D" + str(int(percentile * 100)) + " map")
    else:
        ax2.set_title(parameter + " map")
    fig.colorbar(pos, ax=ax2)


# --- Public API ---

def clasts_rasterize(ClastImageFilePath, ClastSizeListCSVFilePath,
                     RasterFileWritingPath, field="Clast_length",
                     parameter="quantile", cellsize=1, percentile=0.5,
                     plot=True, figuresize=(15, 20), T=10,
                     bin_edges=None, bin_mode="fixed",
                     bin_field="Clast_length",
                     min_cell_density=0,
                     rho_water: float = 1025.0):

#   Rasterise a per-clast CSV over the geotiff it was detected on.
#   field: CSV column or derived quantity to aggregate: Clast_length, Clast_width,
#       Ellipse_major/minor_axis, Equivalent_diameter, Score, Orientation,
#       Surface_area, Clast/Ellipse_elongation, Clast/Ellipse_circularity,
#       Van_Rijn_dimensionless_diameter (Van Rijn 1993), Soulsby_critical_shields
#       (Soulsby 1997), Shields_critical_shear_stress / _shear_velocity /
#       _grain_reynolds_number (Shields 1936), Hjulstrom_deposition_velocity
#       (Sundborg 1956 fit: U = 77·D/(1+24·D), D in m), Hjulstrom_erosion_velocity
#       (Soulsby 1997 Shields parameterisation; diverges from Hjulström for
#       D > 64 mm), Leroux_wave_orbital_velocity (Le Roux, DOI
#       10.1016/S0037-0738(01)00105-1).
#   parameter: per-cell statistic: quantile (at ``percentile``), density, average,
#       std, kurtosis, skewness, sorting, cv, mode, distribution, d_percentiles,
#       folk_ward_sorting/skewness/kurtosis, packing_index, packing_clustering.
#   cellsize: output cell size in the geotiff's units.
#   T: wave period (s) for Le Roux.
#   rho_water: kg/m³ for the transport-threshold fields; 1025 seawater, 1000 fresh.
#   bin_edges: size-bin edges in METRES; makes packing_index multi-band (one band
#       per bin, Σ Surface_area / cell area) and is required by packing_clustering
#       (Shannon evenness across bins).
#   bin_mode: "fixed" takes bin_edges as metres; "percentile" takes them as
#       quantiles of ``bin_field`` over the whole CSV.
#   bin_field: column to bin on; Surface_area is always what is summed.

    # --- 1. Load + filter ---
    clasts = pd.read_csv(ClastSizeListCSVFilePath)
    local_clasts = clasts.copy()

    local_clasts = local_clasts[local_clasts['Clast_length'].isnull() == False]
    local_clasts = local_clasts[local_clasts['Clast_length'] > 0]
    local_clasts = local_clasts[local_clasts['Clast_width'] > 0]
    local_clasts = local_clasts.reset_index(drop=True)

    # --- 2. User-defined addon columns ---
    try:
        from pathlib import Path as _P
        from functions.addons import (
            load_addons as _la,
            apply_addons_to_dataframe as _aatd,
        )
        _csv_p = _P(ClastSizeListCSVFilePath)
        _repo_root = _P(__file__).resolve().parent.parent
        # Shipped hydraulic indices (lowest precedence), then project addons,
        # then user addons; must match the Rasterize tab's loader.
        _candidates = [_repo_root / "hydraulic_addons.json"]
        try:
            _candidates.append(_csv_p.parent.parent.parent / "addons.json")
        except Exception:
            pass
        _candidates.append(_repo_root / "user_addons.json")
        _addons = _la(*_candidates)
        if _addons:
            _aatd(local_clasts, _addons)
            _log.info("Applied %d user addon(s): %s",
                      len(_addons), [a.name for a in _addons])
    except Exception as _ex:
        _log.warning("Addon evaluation skipped: %s", _ex)

    # --- 3. Per-field transforms ---
    # Orientation is axial circular: double the angle, work in (u, v) space.
    if str.lower(field) == "orientation":
        local_clasts['u'], local_clasts['v'] = op.pol2cart(
            np.ones(np.shape(local_clasts)[0]),
            np.deg2rad(local_clasts['Orientation']) * 2.0
        )

    if str.lower(field) == "clast_elongation":
        local_clasts['Clast_elongation'] = (local_clasts['Clast_width']
                                             / local_clasts['Clast_length'])
    if str.lower(field) == "ellipse_elongation":
        local_clasts['Ellipse_elongation'] = (local_clasts['Ellipse_minor_axis']
                                               / local_clasts['Ellipse_major_axis'])

    # Equivalent diameter feeds the transport-threshold and circularity fields.
    if str.lower(field) in (
        "equivalent_diameter", "hjulstrom_deposition_velocity",
        "hjulstrom_erosion_velocity", "leroux_wave_orbital_velocity",
        "van_rijn_dimensionless_diameter", "soulsby_critical_shields",
        "shields_critical_shear_stress", "shields_critical_shear_velocity",
        "shields_critical_grain_reynolds_number", "clast_circularity",
        "ellipse_circularity",
    ):
        local_clasts['Equivalent_diameter'] = (
            2 * np.sqrt(local_clasts['Surface_area'] / np.pi))

    if str.lower(field) == "clast_circularity":
        local_clasts['Clast_circularity'] = (local_clasts['Equivalent_diameter']
                                              / local_clasts['Clast_length'])
    if str.lower(field) == "ellipse_circularity":
        local_clasts['Ellipse_circularity'] = (local_clasts['Equivalent_diameter']
                                                / local_clasts['Ellipse_major_axis'])

    # "sorting": convert size fields to phi in place so the per-cell loop
    # operates on phi values.
    if str.lower(parameter) == "sorting":
        local_clasts['Clast_length'] = (
            -np.log2(local_clasts['Clast_length'] / _PHI_REF_M))
        local_clasts['Clast_width'] = (
            -np.log2(local_clasts['Clast_width'] / _PHI_REF_M))
        for _ell in ('Ellipse_major_axis', 'Ellipse_minor_axis'):
            if _ell in local_clasts.columns:
                local_clasts[_ell] = -np.log2(local_clasts[_ell] / _PHI_REF_M)
        if 'Perimeter' in local_clasts.columns:
            local_clasts['Perimeter'] = (
                -np.log2(local_clasts['Perimeter'] / _PHI_REF_M))
        if str.lower(field) == "equivalent_diameter":
            local_clasts['Equivalent_diameter'] = (
                -np.log2(local_clasts['Equivalent_diameter'] / _PHI_REF_M))

    # Sediment-motion thresholds (SI throughout: density kg/m³, mu Pa·s, D m).
    if str.lower(field) in (
        "van_rijn_dimensionless_diameter", "soulsby_critical_shields",
        "shields_critical_shear_stress", "shields_critical_shear_velocity",
        "shields_critical_grain_reynolds_number",
    ):
        rho_sed = 2650.0
        Rd  = rho_sed / rho_water
        g   = 9.81
        mu  = 1.07e-3
        nu  = mu / rho_water

        local_clasts['Van_Rijn_dimensionless_diameter'] = (
            local_clasts['Equivalent_diameter']
            * ((g * (Rd - 1)) / (nu ** 2)) ** (1 / 3)
        )
        Dstar = local_clasts['Van_Rijn_dimensionless_diameter']
        local_clasts['Soulsby_critical_shields'] = (
            0.30 / (1.0 + 1.2 * Dstar)
            + 0.055 * (1.0 - np.exp(-0.020 * Dstar))
        )
        local_clasts['Shields_critical_shear_stress'] = (
            local_clasts['Soulsby_critical_shields']
            * (rho_sed - rho_water) * g * local_clasts['Equivalent_diameter']
        )
        local_clasts['Shields_critical_shear_velocity'] = (
            local_clasts['Shields_critical_shear_stress'] / rho_water
        ) ** 0.5
        local_clasts['Shields_critical_grain_reynolds_number'] = (
            local_clasts['Shields_critical_shear_velocity']
            * local_clasts['Equivalent_diameter'] / nu
        )

    if str.lower(field) == "hjulstrom_deposition_velocity":
        # Sundborg (1956) fit to the Hjulström deposition envelope:
        # U_dep = 77·D / (1 + 24·D), D in m, U in m/s; valid D ≈ 0.01 mm – 1 m.
        local_clasts['Hjulstrom_deposition_velocity'] = (
            77 * (local_clasts['Equivalent_diameter']
                  / (1 + 24 * local_clasts['Equivalent_diameter']))
        )

    if str.lower(field) == "hjulstrom_erosion_velocity":
        # Shields-based critical erosion velocity, Soulsby (1997) parameterisation.
        rho_sed = 2650.0
        Rd  = rho_sed / rho_water
        g   = 9.81
        mu  = 1.07e-3
        nu  = mu / rho_water
        D   = local_clasts['Equivalent_diameter']
        local_clasts['Hjulstrom_erosion_velocity'] = (
            1.5 * ((nu / D) ** 0.8)
            + 0.85 * ((nu / D) ** 0.35)
            + 9.5 * ((Rd * g * D) / (1 + (2.25 * Rd * g * D)))
        )

    if str.lower(field) == "leroux_wave_orbital_velocity":
        # Le Roux (2007), kept in cgs to match the source paper.
        rho_water_cgs = rho_water / 1000.0
        rho_sed = 2.600
        g       = 981
        D       = local_clasts['Equivalent_diameter'] * 100  # m → cm
        rho_gam = rho_sed - rho_water_cgs
        mu      = 1.07e-2
        Dd = D * (((rho_water_cgs * g * rho_gam) / (mu ** 2))) ** (1 / 3)

        Dd_arr = Dd.values if hasattr(Dd, 'values') else np.asarray(Dd)
        Wd = np.where(
            Dd_arr < 1.2538,
            (0.2354 * Dd_arr) ** 2,
            np.where(
                Dd_arr < 2.9074,
                (0.208 * Dd_arr - 0.0652) ** (3 / 2),
                np.where(
                    Dd_arr < 22.9866,
                    (0.2636 * Dd_arr - 0.37),
                    np.where(
                        Dd_arr < 134.92150,
                        (0.8255 * Dd_arr - 5.4) ** (2 / 3),
                        (2.531 * Dd_arr + 160) ** (1 / 2),
                    ),
                ),
            ),
        )
        Wd[~np.isfinite(Wd)] = np.nan
        Wd[Wd == 0] = np.nan

        theta_wl = (0.0246 * Wd) ** (-0.55)
        local_clasts['Leroux_wave_orbital_velocity'] = (
            (theta_wl * g * D * rho_gam)
            / (((np.pi * T) / (rho_water_cgs * mu)) ** 0.5)
        ) / 100  # cm/s → m/s

    # --- 4. Compute grid ---
    p, density_grid, _bin_labels, _bins_m, n_rows, n_cols = _compute_raster_grid(
        local_clasts, field, parameter, cellsize, percentile,
        bin_edges, bin_mode, bin_field, min_cell_density,
    )

    # --- 5. Source CRS; north-up geotransform from the clast coordinates ---
    raster = gdal.Open(ClastImageFilePath, gdal.GA_ReadOnly)
    origin_x     = float(np.min(clasts['x']))
    origin_y_top = float(np.min(clasts['y'])) + n_rows * cellsize
    north_up_gt  = [origin_x, cellsize, 0, origin_y_top, 0, -cellsize]

    # --- 6. Write raster + sidecar JSON ---
    sidecar_meta = {
        "source_image":     os.path.basename(ClastImageFilePath),
        "source_csv":       os.path.basename(ClastSizeListCSVFilePath),
        "field":            field,
        "cellsize_m":       float(cellsize),
        "percentile":       (float(percentile)
                             if str.lower(parameter) == "quantile" else None),
        "bin_mode":         bin_mode,
        "bin_field":        bin_field,
        "bin_edges_m":      (list(map(float, bin_edges))
                             if bin_edges is not None else None),
        "wave_period_s":    float(T),
        "min_cell_density": int(min_cell_density or 0),
    }
    _write_raster(p, north_up_gt, raster.GetProjection(),
                  RasterFileWritingPath, parameter, _bin_labels, _bins_m,
                  sidecar_meta)

    if plot:
        _render_raster_figure(p, raster, field, parameter, percentile, figuresize)

    _log.info("Rasterize complete: %s", RasterFileWritingPath)
    return p
