"""Publication-quality map export: a QGIS-style map of a per-clast CSV
(vector) or a rasterized grain-size map (raster) over the source ortho and a
tile basemap, with grid, colorbar, CRS box, north arrow and scale bar.
``make_publication_map`` returns a matplotlib Figure; every element except
the data is optional."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from functions._logging import get_logger

_log = get_logger(__name__)

import numpy as np
import pandas as pd
# Non-interactive backend before pyplot: renders run from worker threads.
import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Basemap providers exposed to the GUI. Google tiles are excluded: their
# terms require an API key.
DEFAULT_BASEMAPS = {
    "ESRI World Imagery (satellite)": "Esri.WorldImagery",
    "ESRI World Topographic":         "Esri.WorldTopoMap",
    "ESRI World Street Map":          "Esri.WorldStreetMap",
    "ESRI World Terrain":             "Esri.WorldTerrain",
    "ESRI World Physical":            "Esri.WorldPhysical",
    "Stadia Satellite":               "Stadia.AlidadeSatellite",
    "Stadia Outdoors":                "Stadia.Outdoors",
    "USGS US Imagery (US only)":      "USGS.USImagery",
    "USGS US Imagery + Topo":         "USGS.USImageryTopo",
    "OpenStreetMap":                  "OpenStreetMap.Mapnik",
    "OpenTopoMap":                    "OpenTopoMap",
    "CartoDB Voyager":                "CartoDB.Voyager",
    "CartoDB Positron (light)":       "CartoDB.Positron",
    "CartoDB Dark Matter":            "CartoDB.DarkMatter",
    "(none)":                         None,
}

# Colormaps offered in the GUI (order matters); cyclic ones for angular fields last.
DEFAULT_COLORMAPS = [
    "viridis", "magma", "plasma", "inferno", "cividis",
    "terrain", "RdYlBu_r", "Spectral_r", "turbo", "gray",
    "twilight", "twilight_shifted", "hsv",
]

# Substituted for axial fields: perceptually uniform and wraps, so 175° and
# 5° render near-identically.
DEFAULT_CYCLIC_COLORMAP = "twilight"

# Axial data span half a turn; pinned so every orientation map reads on the same scale.
CYCLIC_RANGE_DEG = (0.0, 180.0)
CYCLIC_TICKS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0)

# Not perceptually uniform, not colourblind-safe; refused for quantitative
# maps unless ``allow_rainbow=True``.
RAINBOW_COLORMAPS = frozenset({
    "jet", "rainbow", "turbo", "gist_rainbow", "nipy_spectral", "hsv",
    "jet_r", "rainbow_r", "turbo_r", "gist_rainbow_r", "nipy_spectral_r",
    "hsv_r",
})
DEFAULT_QUANTITATIVE_COLORMAP = "viridis"

# Bumped whenever the rendering changes in a way a cached PNG could be stale
# against; written into every metadata sidecar.
EXPORTER_VERSION = "2.0"

# Appended to the FULL output filename: ``map.png.json``.
METADATA_SIDECAR_SUFFIX = ".json"


def _is_cyclic_field(field: str) -> bool:
    """True iff ``field`` is an axial quantity (Orientation, 0° ≡ 180°).
    Substring match so compound filename stems are recognised."""
    if not field:
        return False
    return "orientation" in str(field).lower()

# Physical units for output fields, keyed by exact column name.
_FIELD_UNITS = {
    "Clast_length":              "m",
    "Clast_width":               "m",
    "Ellipse_major_axis":        "m",
    "Ellipse_minor_axis":        "m",
    "Equivalent_diameter":       "m",
    "Surface_area":              "m²",
    "Score":                     "",            # confidence score, dimensionless
    "Orientation":               "°",
    "Clast_elongation":          "",
    "Ellipse_elongation":        "",
    "Clast_circularity":         "",
    "Ellipse_circularity":       "",
    "Van_Rijn_dimensionless_diameter":              "",
    "Soulsby_critical_shields":                     "",
    "Shields_critical_shear_stress":                "Pa",
    "Shields_critical_shear_velocity":              "m/s",
    "Shields_critical_grain_reynolds_number":       "",
    "Hjulstrom_deposition_velocity":                "m/s",
    "Hjulstrom_erosion_velocity":                   "m/s",
    "Leroux_wave_orbital_velocity":                 "m/s",
}


def _format_colorbar_label(field_or_filename: str, parameter: str = None) -> str:
    """Publication-ready colorbar label from a snake_case field name or a
    compound filename stem: ``Clast_length_D50`` -> ``'Clast length [m]'``.
    A ``parameter`` other than quantile/average is appended."""
    name = field_or_filename
    # Canonical field from a compound stem: exact, then prefix, then longest
    # substring match (longest avoids false positives on shared substrings).
    raw = name
    if name in _FIELD_UNITS:
        raw = name
    else:
        prefix_match = max(
            (k for k in _FIELD_UNITS if name.startswith(k)),
            key=len,
            default=None,
        )
        if prefix_match is not None:
            raw = prefix_match
        else:
            substr_match = max(
                (k for k in _FIELD_UNITS if k in name),
                key=len,
                default=None,
            )
            if substr_match is not None:
                raw = substr_match
    unit = _FIELD_UNITS.get(raw, None)
    pretty = raw.replace("_", " ")
    pretty = pretty[0].upper() + pretty[1:] if pretty else pretty
    if unit:
        pretty = f"{pretty} [{unit}]"
    if parameter and parameter not in ("quantile", "average"):
        pretty = f"{pretty} – {parameter}"
    return pretty


# --- Unit conversion ---
# Data is stored in SI (m, m², m/s, Pa) and never modified; only the display
# is relabelled and rescaled.

_SIZE_UNIT_TABLE = {
    "m":  ("m",  1.0),
    "cm": ("cm", 100.0),
    "mm": ("mm", 1000.0),
    "ft": ("ft", 3.28084),
    "in": ("in", 39.3701),
}


def _pick_size_unit(values, unit_system: str = "metric", override: str = "auto"):
    """(unit_str, factor) for size-like data; the factor multiplies metres."""
    if override and override != "auto" and override in _SIZE_UNIT_TABLE:
        return _SIZE_UNIT_TABLE[override]

    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        magnitude = 1.0
    else:
        magnitude = float(np.median(np.abs(finite)))

    if unit_system == "imperial":
        magnitude_ft = magnitude * 3.28084
        if magnitude_ft >= 1.0:
            return _SIZE_UNIT_TABLE["ft"]
        return _SIZE_UNIT_TABLE["in"]
    if magnitude >= 1.0:
        return _SIZE_UNIT_TABLE["m"]
    if magnitude >= 0.01:
        return _SIZE_UNIT_TABLE["cm"]
    return _SIZE_UNIT_TABLE["mm"]


# Per-field semantic kind, looked up by longest substring match so a compound
# name like "Clast_width_packing_index_…" lands on the parameter, not the size.
_FIELD_KIND = {
    "Clast_length": "size",       "Clast_width": "size",
    "Ellipse_major_axis": "size", "Ellipse_minor_axis": "size",
    "Equivalent_diameter": "size",
    "Surface_area": "area",
    "Score": "dimensionless",
    "Orientation": "angle",
    "Clast_elongation": "dimensionless", "Ellipse_elongation": "dimensionless",
    "Clast_circularity": "dimensionless", "Ellipse_circularity": "dimensionless",
    "Van_Rijn_dimensionless_diameter": "dimensionless",
    "Soulsby_critical_shields": "dimensionless",
    "Shields_critical_shear_stress": "pressure",
    "Shields_critical_shear_velocity": "velocity",
    "Shields_critical_grain_reynolds_number": "dimensionless",
    "Hjulstrom_deposition_velocity": "velocity",
    "Hjulstrom_erosion_velocity":   "velocity",
    "Leroux_wave_orbital_velocity": "velocity",
    # Per-cell parameters with dimensionless values (φ renders as dimensionless).
    "packing_index":         "dimensionless",
    "packing_clustering":    "dimensionless",
    "folk_ward_sorting":     "dimensionless",
    "folk_ward_skewness":    "dimensionless",
    "folk_ward_kurtosis":    "dimensionless",
    "sorting":               "dimensionless",
    "skewness":              "dimensionless",
    "kurtosis":              "dimensionless",
}




def _registry_known_unit(field: str, parameter: Optional[str] = None):
    """``(unit, factor)`` when ``functions.units`` knows the quantity, else
    None. Looked up against the table itself because ``field_unit_and_factor``
    answers ``("m", 1.0)`` for unknown fields too. A per-cell ``parameter``
    that replaces the field's unit (sorting → φ, density → clasts/m²) wins."""
    from functions.units import FIELD_UNIT_TABLE, PARAMETER_DISPLAY
    if parameter:
        row = PARAMETER_DISPLAY.get(str(parameter))
        if row is not None and row[1] is not None:
            return (row[1], float(row[2] if row[2] is not None else 1.0))
    if not field:
        return None
    needle = str(field).lower()
    best, best_len = None, -1
    for key, val in FIELD_UNIT_TABLE:
        if key in needle and len(key) > best_len:
            best, best_len = val, len(key)
    return best


def _parameter_from_stem(stem: Optional[str]) -> Optional[str]:
    """Per-cell parameter baked into a rasterize filename (``..._D50_...`` →
    ``'D50'``), from ``functions.units.PARAMETER_DISPLAY`` keys bounded by
    non-alphanumerics; longest key wins. None when nothing matches."""
    if not stem:
        return None
    import re as _re
    from functions.units import PARAMETER_DISPLAY
    s = str(stem)
    best, best_len = None, -1
    for key in PARAMETER_DISPLAY:
        if _re.search(r"(?<![A-Za-z0-9])" + _re.escape(key) + r"(?![A-Za-z0-9])",
                      s) and len(key) > best_len:
            best, best_len = key, len(key)
    return best


def _field_kind(field: str):
    """Semantic kind of ``field`` from ``_FIELD_KIND`` (longest match), or None."""
    kind = _FIELD_KIND.get(field, None)
    if kind is None:
        best_len = -1
        for known, k in _FIELD_KIND.items():
            if known in field and len(known) > best_len:
                best_len = len(known)
                kind = k
    return kind


def _resolve_unit_for_field(field: str, values, unit_system: str, size_unit: str,
                            parameter: Optional[str] = None):
    """``(display_unit_label, factor)`` for a field; the factor multiplies
    stored SI values. ``('', 1.0)`` for dimensionless or unknown fields.

    Precedence: an explicit ``size_unit`` wins for size/area fields; imperial
    output uses the magnitude rule; otherwise ``functions.units`` is the
    single authority (so the same field never renders in cm on one map and
    mm on the next); only a field the registry does not know falls back to
    ``_FIELD_KIND`` and the magnitude rule.
    """
    kind = _field_kind(field)
    explicit = bool(size_unit) and size_unit != "auto"
    imperial = unit_system == "imperial"

    if not explicit and not imperial:
        known = _registry_known_unit(field, parameter)
        if known is not None:
            unit, factor = known
            if unit in _SIZE_UNIT_TABLE:
                return _SIZE_UNIT_TABLE[unit]
            return (unit, float(factor))

    if kind is None or kind == "dimensionless":
        return ("", 1.0)
    if kind == "angle":
        return ("°", 1.0)
    if kind == "size":
        return _pick_size_unit(values, unit_system, size_unit)
    if kind == "area":
        suffix, factor = _pick_size_unit(
            np.sqrt(np.abs(np.asarray(values, dtype=float))),
            unit_system, size_unit)
        return (f"{suffix}²", factor ** 2)
    if kind == "velocity":
        if unit_system == "imperial":
            return ("ft/s", 3.28084)
        return ("m/s", 1.0)
    if kind == "pressure":
        if unit_system == "imperial":
            return ("psi", 1.0 / 6894.76)
        return ("Pa", 1.0)
    return ("", 1.0)


def _raster_label(stem: str, parameter, unit_str: str) -> str:
    """Colorbar label for a raster map, from its file name.

    The field is the longest known quantity inside the stem; when the
    per-cell parameter replaces the field's unit (density in clasts/m²,
    sorting in φ) the parameter names the quantity, since the field's name
    with the parameter's unit is a contradiction. Before this the whole stem
    was prettified, and a map read "Example 03 Etretat etretat 20200610 ortho
    crop merged density cellsize=1.0m [clasts/m²]".
    """
    from functions.units import PARAMETER_DISPLAY
    row = PARAMETER_DISPLAY.get(str(parameter)) if parameter else None
    if row is not None and row[1] is not None:
        pretty = str(row[0])
        return f"{pretty[:1].upper()}{pretty[1:]} [{unit_str}]" if unit_str \
            else f"{pretty[:1].upper()}{pretty[1:]}"
    field = max((k for k in _FIELD_KIND if k in str(stem)), key=len, default=None)
    if field is None:
        field = str(parameter or stem)
    return _format_label_with_unit(field, unit_str)


def _format_label_with_unit(field_or_filename: str, unit_str: str) -> str:
    """'Pretty name [unit]' label with an externally resolved unit string
    (from _resolve_unit_for_field), unlike _format_colorbar_label."""
    raw = field_or_filename
    for known in _FIELD_KIND:
        if known in field_or_filename:
            raw = known
            break
    pretty = raw.replace("_", " ")
    if pretty:
        pretty = pretty[0].upper() + pretty[1:]
    if unit_str:
        pretty = f"{pretty} [{unit_str}]"
    return pretty


# --- Coordinate system helpers ---
def _read_ortho_crs(tif_path: str) -> Tuple[str, str, int]:
    """(epsg_code_str, friendly_name, epsg_int) for a GeoTIFF;
    ('?', 'Unknown CRS', 0) on any error."""
    try:
        from osgeo import gdal, osr
        ds = gdal.Open(tif_path)
        if ds is None:
            return "?", "Unknown CRS", 0
        wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        epsg = srs.GetAttrValue("AUTHORITY", 1)
        name = srs.GetAttrValue("PROJCS") or srs.GetAttrValue("GEOGCS") or "Unknown"
        return str(epsg), name, int(epsg) if epsg else 0
    except Exception:
        return "?", "Unknown CRS", 0


def _read_ortho_image(tif_path: str, transparent_edges: bool = True):
    """Read a 3-band ortho-image; returns ``(rgba_array, (xmin, xmax, ymin,
    ymax))`` in the ortho's CRS for ``imshow(extent=...)``. With
    ``transparent_edges`` the alpha is 0 wherever every band is fully dark
    or fully bright, which is almost always nodata margin."""
    from osgeo import gdal
    ds = gdal.Open(tif_path)
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open {tif_path}")
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    xmin = gt[0]
    xmax = gt[0] + nx * gt[1]
    ymax = gt[3]
    ymin = gt[3] + ny * gt[5]
    # A full-resolution float32 promotion of a 20k+ px ortho needs several GB;
    # let GDAL resample on read (nearest neighbour), capped on the longest
    # side. The extent stays in world coordinates.
    MAX_READ_DIM = 8000
    biggest = max(nx, ny)
    if biggest > MAX_READ_DIM:
        scale = float(MAX_READ_DIM) / float(biggest)
        read_nx = max(1, int(round(nx * scale)))
        read_ny = max(1, int(round(ny * scale)))
    else:
        read_nx, read_ny = nx, ny
    bands = []
    for i in range(1, min(4, ds.RasterCount + 1)):
        bands.append(
            ds.GetRasterBand(i).ReadAsArray(
                buf_xsize=read_nx, buf_ysize=read_ny))
    if len(bands) >= 3:
        rgb = np.dstack(bands[:3])
    else:
        rgb = bands[0]

    if rgb.dtype == np.uint8:
        full_dark_val, full_bright_val = 0, 255
    elif rgb.dtype == np.uint16:
        full_dark_val, full_bright_val = 0, 65535
    else:
        full_dark_val = float(rgb.min())
        full_bright_val = float(rgb.max())

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        rgb = rgb.astype(np.float32) / 65535.0
    else:
        rgb_min, rgb_max = float(rgb.min()), float(rgb.max())
        if rgb_max > rgb_min:
            rgb = (rgb.astype(np.float32) - rgb_min) / (rgb_max - rgb_min)

    if not transparent_edges or rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb, (xmin, xmax, ymin, ymax)

    # Alpha 0 where all three bands sit at the dtype's extreme value.
    if full_bright_val > 0:
        bright_threshold = 1.0
        dark_threshold = 0.0
    else:
        bright_threshold = 1.0
        dark_threshold = 0.0
    full_dark = np.all(rgb <= dark_threshold + 1e-6, axis=2)
    full_bright = np.all(rgb >= bright_threshold - 1e-6, axis=2)
    nodata_mask = full_dark | full_bright
    alpha = np.where(nodata_mask, 0.0, 1.0).astype(np.float32)
    rgba = np.dstack([rgb, alpha])
    return rgba, (xmin, xmax, ymin, ymax)


def _raster_band_count(tif_path) -> int:
    """How many bands the file holds. A multi-band raster (one band per size
    bin, or the 99 quantiles of a distribution) is drawn from its first band,
    and the map has to say so instead of passing for the whole file."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(tif_path))
        return int(ds.RasterCount) if ds is not None else 1
    except Exception:
        return 1


def _read_raster_layer(tif_path: str):
    """Read a single-band rasterized map as (array, extent); nodata and
    zero (no clasts in the cell) become NaN."""
    from osgeo import gdal
    ds = gdal.Open(tif_path)
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open {tif_path}")
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    xmin = gt[0]
    xmax = gt[0] + nx * gt[1]
    ymax = gt[3]
    ymin = gt[3] + ny * gt[5]
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(arr == 0, np.nan, arr)
    return arr, (xmin, xmax, ymin, ymax)


# --- Decorations: north arrow, CRS box, scale bar ---
def _add_north_arrow(ax, x_frac=0.93, y_frac=0.90, length_frac=0.06):
    """Spearhead-style N arrow at (x_frac, y_frac) in axes coordinates: a
    hollow left half and a solid right half with a notch at the base, an 'N'
    label above, on a translucent box."""
    from matplotlib.patches import Polygon
    # Layout constants in axes coordinates.
    box_half_width = 0.025
    pad_top = 0.04
    pad_bottom = 0.025
    label_gap = 0.012
    label_height = 0.025

    y_tip = y_frac + length_frac / 2
    y_base = y_frac - length_frac / 2
    y_notch_top = y_base + length_frac * 0.30
    spear_half_w = length_frac * 0.30

    label_y = y_tip + label_gap
    box_y_lo = y_base - pad_bottom
    box_y_hi = label_y + pad_top
    box_height = box_y_hi - box_y_lo

    bg = Rectangle(
        (x_frac - box_half_width, box_y_lo),
        2 * box_half_width, box_height,
        transform=ax.transAxes,
        facecolor="white", edgecolor="black", lw=0.8,
        alpha=0.7, zorder=22,
    )
    ax.add_patch(bg)

    left_tri = Polygon(
        [
            (x_frac,                y_tip),
            (x_frac - spear_half_w, y_base),
            (x_frac,                y_notch_top),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="white", edgecolor="black", lw=1.5,
        zorder=23,
    )
    ax.add_patch(left_tri)

    right_tri = Polygon(
        [
            (x_frac,                y_tip),
            (x_frac,                y_notch_top),
            (x_frac + spear_half_w, y_base),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="#222222", edgecolor="black", lw=1.5,
        zorder=23,
    )
    ax.add_patch(right_tri)

    ax.text(
        x_frac, label_y, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        zorder=24,
    )


def _add_crs_box(ax, crs_text, x_frac=0.02, y_frac=0.02):
    """Add a small CRS-info box in the bottom-left corner."""
    ax.text(
        x_frac, y_frac, crs_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        zorder=24,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black",
                  alpha=0.7),
    )


_FEET_PER_M = 3.28084
_FEET_PER_MILE = 5280.0


def _scale_bar_length(span_x: float, unit_system: str = "metric"):
    """``(bar length in data units, label of that length, unit label)``.

    The axes are always in metres (a projected CRS); in imperial mode the bar
    is a round number of feet, or of miles once it passes half a mile, so the
    bar matches the units the rest of the map is labelled in.
    """
    target = span_x * 0.20

    def _nice(value):
        magnitude = 10 ** int(np.floor(np.log10(value)))
        for nice in (1, 2, 5, 10):
            if nice * magnitude >= value:
                return nice * magnitude
        return 10 * magnitude

    if str(unit_system) == "imperial":
        feet = _nice(target * _FEET_PER_M)
        if feet >= _FEET_PER_MILE / 2:
            miles = _nice(feet / _FEET_PER_MILE)
            return miles * _FEET_PER_MILE / _FEET_PER_M, miles, "mi"
        return feet / _FEET_PER_M, feet, "ft"
    metres = _nice(target)
    return metres, metres, "m"


def _add_scale_bar(ax, units="m", unit_system="metric"):
    """QGIS-style two-segment scale bar (black / white) with 0, midpoint and
    full-length labels, inside a translucent box in the lower-right corner."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin
    span_y = ymax - ymin

    bar_total, bar_label_value, units = _scale_bar_length(span_x, unit_system)
    bar_half = bar_total / 2.0

    # Margin large enough to clear the zebra border plus the box padding.
    bar_height = span_y * 0.012
    pad_x = span_x * 0.06
    pad_y = span_y * 0.06
    x_end = xmax - pad_x
    x_mid = x_end - bar_half
    x_start = x_end - bar_total
    y_bar = ymin + pad_y

    # Outer box first (lower zorder) so the bars and labels paint on top.
    box_inner_pad_x = span_x * 0.012
    box_inner_pad_y = span_y * 0.012
    label_y_offset = bar_height * 1.8
    box_y_lo = y_bar - label_y_offset - bar_height * 1.5   # room for descenders
    box_y_hi = y_bar + bar_height + box_inner_pad_y
    box_x_lo = x_start - box_inner_pad_x
    units_padding = span_x * 0.012 * (1 + len(units))
    box_x_hi = x_end + box_inner_pad_x + units_padding
    ax.add_patch(Rectangle(
        (box_x_lo, box_y_lo),
        box_x_hi - box_x_lo,
        box_y_hi - box_y_lo,
        facecolor="white", edgecolor="black", lw=0.8,
        alpha=0.7, zorder=22,
    ))

    ax.add_patch(Rectangle(
        (x_start, y_bar), bar_half, bar_height,
        facecolor="black", edgecolor="black", lw=0.8, zorder=23,
    ))
    ax.add_patch(Rectangle(
        (x_mid, y_bar), bar_half, bar_height,
        facecolor="white", edgecolor="black", lw=0.8, zorder=23,
    ))

    label_y = y_bar - label_y_offset * 0.55
    label_kwargs = dict(ha="center", va="top", fontsize=8, zorder=24)
    def _fmt(v):
        return f"{int(v)}" if v == int(v) else f"{v:.1f}"
    ax.text(x_start, label_y, "0", **label_kwargs)
    # The bar is drawn in data units (metres); the numbers are in the unit
    # the map is labelled in, so an imperial map reads feet, not metres.
    ax.text(x_mid, label_y, _fmt(bar_label_value / 2.0), **label_kwargs)
    ax.text(x_end, label_y, f"{_fmt(bar_label_value)} {units}", **label_kwargs)


def _add_zebra_border(ax, n_segments=10):
    """Cartographic zebra border: n_segments alternating black/white
    rectangles along each edge, in axes coordinates."""
    border_thickness = 0.012  # fraction of axes height/width

    def _rect(x, y, w, h, color):
        ax.add_patch(Rectangle(
            (x, y), w, h,
            transform=ax.transAxes,
            facecolor=color, edgecolor="black", lw=0.5,
            zorder=18,
        ))

    # Drawn just inside the axes box so the tick labels outside stay clear.
    for i in range(n_segments):
        x = i / n_segments
        w = 1.0 / n_segments
        color = "black" if i % 2 == 0 else "white"
        _rect(x, 0.0, w, border_thickness, color)
        _rect(x, 1.0 - border_thickness, w, border_thickness, color)
    for i in range(n_segments):
        y = i / n_segments
        h = 1.0 / n_segments
        color = "black" if i % 2 == 0 else "white"
        _rect(0.0, y, border_thickness, h, color)
        _rect(1.0 - border_thickness, y, border_thickness, h, color)


def _compute_zoom_from_extent(ax, crs_int):
    """Tile zoom level from the axes extent, clamped to [10, 19]. Contextily's
    auto-zoom can pick z=21+ for a tiny UAV extent (404s) or fall back to
    z=0 (whole-globe tiles painted over the ortho)."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    # Approximate metres; 1° ≈ 111 km is close enough for zoom picking.
    if crs_int and crs_int == 4326:
        avg_y = (ymin + ymax) / 2.0
        width_m = width * 111000 * max(0.1, np.cos(np.radians(avg_y)))
        height_m = height * 111000
    else:
        width_m = float(width)
        height_m = float(height)
    span_m = max(width_m, height_m)
    if span_m <= 0:
        return 18
    # Tile width at zoom z is ~ EARTH_CIRC / 2^z; pick z so ~2 tiles span the extent.
    EARTH_CIRC = 40075016.686
    z = int(np.log2(EARTH_CIRC / max(span_m, 1.0)))
    return max(10, min(19, z))


def _is_network_available() -> bool:
    """Quick TCP probe (8.8.8.8:53, 2 s) so offline machines skip the tile
    fetch instead of waiting on HTTP timeouts."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("8.8.8.8", 53))
        sock.close()
        return True
    except (socket.error, OSError):
        return False


def _use_full_coordinates(ax) -> None:
    """Print full projected coordinates on both axes, no offset notation."""
    import matplotlib.ticker as _mtick
    for axis in (ax.xaxis, ax.yaxis):
        fmt = _mtick.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def _add_attribution_below(ax, provider_name) -> None:
    """Draw the tile attribution under the axes rather than inside them."""
    label = {
        "Esri.WorldImagery": "Tiles © Esri — Source: Esri, Maxar, Earthstar "
                             "Geographics and the GIS User Community",
        "OpenStreetMap.Mapnik": "© OpenStreetMap contributors",
    }.get(str(provider_name), f"Basemap: {provider_name}")
    try:
        ax.annotate(label, xy=(0.0, -0.085), xycoords="axes fraction",
                    ha="left", va="top", fontsize=5.5, color="#555555",
                    annotation_clip=False)
    except Exception:
        pass


def _basemap_is_placeholder(ax) -> bool:
    """True when the last image added to ``ax`` is a near-uniform mosaic,
    the signature of a provider's "no imagery here" placeholder tiles. The
    threshold is low so genuine imagery of water or sand is never discarded."""
    try:
        import numpy as _np
        if not ax.images:
            return False
        arr = _np.asarray(ax.images[-1].get_array(), dtype=float)
        if arr.size == 0:
            return False
        if arr.ndim == 3:
            arr = arr[..., :3]
        return float(_np.nanstd(arr)) < 6.0
    except Exception:
        return False


def _drop_last_basemap(ax) -> None:
    """Remove the most recently added image from ``ax``."""
    try:
        if ax.images:
            ax.images[-1].remove()
    except Exception:
        pass


def _add_basemap_safe(ax, crs_int, provider_name, log_fn=None):
    """Add a contextily basemap at an extent-derived zoom, falling back to
    lower zooms; returns whether one was added. Failures are reported through
    the logger and ``log_fn`` when given."""
    if provider_name is None:
        return False

    import re as _re
    _PROVIDER_NAME_RE = _re.compile(
        r'^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*$')
    if not _PROVIDER_NAME_RE.match(provider_name):
        _msg = (f"Basemap provider name {provider_name!r} is invalid "
                "(expected 'Provider.Style', e.g. 'Esri.WorldImagery'). "
                "Skipping basemap.")
        _log.warning(_msg)
        if log_fn is not None:
            log_fn(f"[map] {_msg}")
        return False

    def _emit(msg: str, level: str = "info") -> None:
        getattr(_log, level)(msg)
        if log_fn is not None:
            log_fn(f"[map] {msg}")

    if not _is_network_available():
        _emit(
            "Network not available — skipping basemap. "
            "Connect to the internet and re-run the Map tab to add a basemap.",
            level="warning",
        )
        return False

    try:
        import contextily as cx
        provider = cx.providers
        for part in provider_name.split("."):
            provider = getattr(provider, part)

        crs_arg = f"EPSG:{crs_int}" if crs_int else None

        # Never zoom='auto': it can 404 at z=21+ or paint whole-globe tiles.
        primary_zoom = _compute_zoom_from_extent(ax, crs_int)
        attempts = [
            {"zoom": primary_zoom,                "label": f"z={primary_zoom}"},
            {"zoom": max(10, primary_zoom - 2),   "label": f"z={max(10, primary_zoom - 2)}"},
            {"zoom": max(10, primary_zoom - 4),   "label": f"z={max(10, primary_zoom - 4)}"},
            {"zoom": 12,                          "label": "z=12"},
        ]
        seen_zooms = set()
        unique_attempts = []
        for a in attempts:
            if a["zoom"] not in seen_zooms:
                seen_zooms.add(a["zoom"])
                unique_attempts.append(a)
        attempts = unique_attempts

        last_err = None
        for attempt in attempts:
            try:
                # Attribution is drawn below the axes; contextily's would land
                # on the scale bar.
                cx.add_basemap(ax, crs=crs_arg, source=provider,
                               attribution="",
                               zoom=attempt["zoom"])
                if _basemap_is_placeholder(ax):
                    _drop_last_basemap(ax)
                    _emit(f"Provider returned no imagery at "
                          f"{attempt['label']}; trying a wider zoom.")
                    continue
                _emit(f"Basemap loaded at {attempt['label']} (extent-derived zoom).")
                return True
            except Exception as e:
                last_err = e
                continue

        _emit(
            f"Basemap fetch failed for {provider_name!r}: "
            f"{type(last_err).__name__}: {last_err}. "
            "Map will render without basemap. "
            "Try a different provider (e.g. 'OpenStreetMap' or 'Stadia Satellite') "
            "or check your internet connection.",
            level="warning",
        )
        return False
    except Exception as e:
        _emit(f"Basemap unavailable: {type(e).__name__}: {e}", level="warning")
        return False


# --- Frame, class-label and metadata helpers ---
def _valid_data_bbox(arr, extent):
    """World-coordinate ``(xmin, xmax, ymin, ymax)`` of the finite cells of
    ``arr``, or the full ``extent`` when nothing is finite."""
    xmin, xmax, ymin, ymax = extent
    a = np.asarray(arr)
    if a.ndim != 2 or a.size == 0:
        return extent
    finite = np.isfinite(a)
    rows = np.flatnonzero(finite.any(axis=1))
    cols = np.flatnonzero(finite.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return extent
    ny, nx = a.shape
    px = (xmax - xmin) / float(nx)
    py = (ymax - ymin) / float(ny)
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    return (xmin + c0 * px, xmin + (c1 + 1) * px,
            ymax - (r1 + 1) * py, ymax - r0 * py)


def _expand_bbox(bbox, margin: float):
    """Grow ``(xmin, xmax, ymin, ymax)`` by ``margin`` × its span on each
    side; a zero-span axis borrows the other's span (or 1 m) so the frame
    never has zero width."""
    xmin, xmax, ymin, ymax = (float(v) for v in bbox)
    sx, sy = xmax - xmin, ymax - ymin
    if sx <= 0 and sy <= 0:
        sx = sy = 1.0
    elif sx <= 0:
        sx = sy
    elif sy <= 0:
        sy = sx
    return (xmin - sx * margin, xmax + sx * margin,
            ymin - sy * margin, ymax + sy * margin)


def _default_decimals(unit: str, v_hi: float) -> Optional[int]:
    """Decimal places for a colorbar label when no uncertainty is known;
    ``None`` means two significant figures (velocities, stresses)."""
    top = abs(float(v_hi)) if np.isfinite(v_hi) else 0.0
    if unit == "mm":
        return 0 if top >= 10.0 else 1
    if unit == "cm":
        return 1 if top >= 1.0 else 2
    if unit == "m":
        return 3
    if unit in ("in", "ft"):
        return 2
    if unit == "°":
        return 0
    if unit in ("m/s", "ft/s", "Pa", "psi"):
        return None
    if unit in ("mm²",):
        return 0
    if unit in ("cm²", "m²", "in²", "ft²", "clasts/m²"):
        return 1
    return 2


def _tick_formatter(unit: str, u_display: Optional[float], v_hi: float):
    """Callable ``value → label`` for colorbar ticks: with an uncertainty (in
    display units) the label stops at the digit it supports, otherwise
    ``_default_decimals`` applies."""
    decimals: Optional[int]
    if u_display is not None and np.isfinite(u_display) and u_display > 0:
        from functions.precision import decimals_for
        decimals = int(decimals_for(float(u_display), unit))
    else:
        decimals = _default_decimals(unit, v_hi)

    def _fmt(v, _pos=None):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        if not np.isfinite(v):
            return ""
        if decimals is None:
            if v == 0:
                return "0"
            return f"{v:.2g}"
        s = f"{v:.{decimals}f}"
        # "-0" reads as a sign error on a scale that starts at zero.
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
        return s
    return _fmt


def _json_safe(value):
    """Recursively coerce numpy scalars/arrays and Paths into JSON types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def metadata_sidecar_path(out_path) -> Path:
    """``<png>.json`` beside an exported map: ``map.png`` → ``map.png.json``."""
    p = Path(out_path)
    return p.with_name(p.name + METADATA_SIDECAR_SUFFIX)


def write_map_metadata(out_path, metadata: dict) -> Path:
    """Write ``metadata`` as the sidecar JSON for ``out_path``; return its path."""
    import json
    side = metadata_sidecar_path(out_path)
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(_json_safe(metadata), indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return side


def read_map_metadata(out_path) -> Optional[dict]:
    """Metadata dict from the sidecar beside ``out_path``, or None if absent
    or unreadable."""
    import json
    side = metadata_sidecar_path(out_path)
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_publication_map(fig, out_path, *, dpi: int = 300, sidecar: bool = True,
                         **savefig_kwargs) -> dict:
    """Save ``fig`` at ``dpi`` and, when ``sidecar`` is true, the
    ``<out_path>.json`` sidecar carrying ``fig.map_metadata`` plus output,
    dpi and format. Returns the metadata written; the figure is not closed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(str(out_path), dpi=dpi, **savefig_kwargs)
    meta = dict(getattr(fig, "map_metadata", None) or {})
    meta["output"] = str(out_path)
    meta["dpi"] = int(dpi)
    meta["format"] = (savefig_kwargs.get("format")
                      or out_path.suffix.lstrip(".").lower() or None)
    if sidecar:
        meta["sidecar"] = str(write_map_metadata(out_path, meta))
    return meta


# --- Main entry point ---
def make_publication_map(
    *,
    ortho_tif: str,
    clast_csv: Optional[str] = None,
    raster_tif: Optional[str] = None,
    layer: str = "vector",                  # 'vector' or 'raster'
    field: str = "Clast_length",            # CSV column to color by (vector mode)
    cmap: str = "viridis",
    basemap: Optional[str] = "Esri.WorldImagery",
    show_ortho: bool = True,
    ortho_alpha: float = 0.7,
    raster_alpha: float = 0.85,
    show_grid: bool = True,
    show_legend: bool = True,
    show_crs_info: bool = True,
    show_north_arrow: bool = True,
    show_scale_bar: bool = True,
    show_zebra_border: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    point_size: float = 8.0,
    vmin: Optional[float] = None,           # None = autoscale per color_scale
    vmax: Optional[float] = None,
    color_scale: str = "quantile",          # 'linear' or 'quantile'
    quantile_low: float = 0.02,             # clip percentiles for 'quantile'
    quantile_high: float = 0.98,
    # Clip percentiles for 'linear' when vmin/vmax are unset: a single huge
    # clast would otherwise crush the colour range. 0/1 = true min/max.
    linear_low_pct: float = 0.05,
    linear_high_pct: float = 0.95,
    colorbar_label: Optional[str] = None,   # None = auto-format from field/raster name
    # Triangular extension caps on the colorbar; off for cyclic colormaps.
    show_colorbar_extends: bool = True,
    # Padding around the data extent as a fraction of it, each side.
    extent_padding: float = 0.05,
    # 'metric' or 'imperial'; size_unit 'auto' picks m/cm/mm (or ft/in) from
    # the data magnitude. Display only; the data is never modified.
    unit_system: str = "metric",
    size_unit: str = "auto",
    log_fn: Optional[callable] = None,
    # Frame: 'project' = union of padded data extent and ortho; 'raster' (alias
    # 'data') = the valid-data bounding box plus extent_margin each side, so a
    # raster covering a corner of the ortho fills the figure.
    extent: str = "project",
    extent_margin: float = 0.10,
    # None = continuous colorbar; an int = that many discrete classes
    # (quantile-spaced or equal-interval per color_scale). Ignored for cyclic fields.
    classes: Optional[int] = None,
    # Uncertainty of the plotted quantity in DISPLAY units; colorbar labels
    # stop at the digit it supports (functions.precision.decimals_for).
    uncertainty: Optional[float] = None,
    # Rainbow colormaps are replaced by viridis unless allowed.
    allow_rainbow: bool = False,
    # Per-cell parameter of a rasterized product ('D50', 'sorting', ...);
    # None = recovered from the raster filename. Drives the unit (a
    # 'sorting' raster stores φ) and the metadata.
    parameter: Optional[str] = None,
    # True returns (fig, metadata); the dict is always on fig.map_metadata.
    return_metadata: bool = False,
):
    """Render a publication-quality map and return the matplotlib Figure.

    ``clast_csv`` is required for layer='vector', ``raster_tif`` (a
    clasts_rasterize output) for layer='raster'; ``ortho_tif`` supplies the
    georeferencing. Cyclic fields (Orientation) always render with
    ``twilight`` on a fixed 0–180° scale; rainbow colormaps are replaced by
    viridis unless ``allow_rainbow``. The keyword comments above describe
    the remaining options.

    The caption metadata is attached as ``fig.map_metadata`` (field,
    parameter, layer, cell_size, unit, unit_factor, cmap, classification,
    class_edges, value_min/max, data_min/max, n_valid, n_total, crs,
    crs_name, extent_mode, extent, data_extent, colorbar_label, source,
    ortho, basemap, basemap_added, exporter_version, exported_at);
    ``save_publication_map`` writes it as a ``<png>.json`` sidecar.
    """
    import matplotlib.colors as mcolors
    from datetime import datetime, timezone

    if layer == "vector" and not clast_csv:
        raise ValueError("layer='vector' requires clast_csv")
    if layer == "raster" and not raster_tif:
        raise ValueError("layer='raster' requires raster_tif")
    if layer not in ("vector", "raster"):
        raise ValueError(f"layer must be 'vector' or 'raster', got {layer!r}")
    if extent not in ("project", "raster", "data"):
        raise ValueError(
            f"extent must be 'project' or 'raster', got {extent!r}")
    if color_scale not in ("linear", "quantile"):
        raise ValueError(
            f"color_scale must be 'linear' or 'quantile', got {color_scale!r}")
    if classes is not None:
        classes = int(classes)
        if classes < 2:
            raise ValueError(f"classes must be >= 2, got {classes}")

    def _say(msg: str) -> None:
        if log_fn is not None:
            log_fn(f"[map] {msg}")

    raster_stem = Path(raster_tif).stem if raster_tif else None
    if parameter is None and layer == "raster":
        parameter = _parameter_from_stem(raster_stem)

    # Axial fields: colormap, range and ticks are fixed whatever the caller
    # passed (twilight, 0–180°, every 45°), and values are folded into
    # [0, 180) since axial data is sometimes stored in −90..90.
    _cyclic_field_name = field if layer == "vector" else raster_stem
    is_cyclic = _is_cyclic_field(_cyclic_field_name)
    if is_cyclic:
        if cmap != DEFAULT_CYCLIC_COLORMAP:
            _say(f"Orientation is an axial/cyclic field; using cyclic "
                 f"colormap {DEFAULT_CYCLIC_COLORMAP!r} instead of {cmap!r} "
                 f"so 0° and 180° read the same.")
            _log.info(
                "Cyclic field %r: overriding colormap %r -> %r.",
                _cyclic_field_name, cmap, DEFAULT_CYCLIC_COLORMAP)
        cmap = DEFAULT_CYCLIC_COLORMAP
        show_colorbar_extends = False
        if classes is not None:
            _say("classes is ignored for a cyclic field: the scale is "
                 "fixed at 0–180°.")
            classes = None
    elif cmap in RAINBOW_COLORMAPS and not allow_rainbow:
        # Substituted, not raised, so the job still produces a figure.
        _say(f"Colormap {cmap!r} is a rainbow ramp and is not used for "
             f"quantitative maps; using {DEFAULT_QUANTITATIVE_COLORMAP!r}. "
             f"Pass allow_rainbow=True to keep it.")
        _log.info("Rainbow colormap %r refused -> %r.", cmap,
                  DEFAULT_QUANTITATIVE_COLORMAP)
        cmap = DEFAULT_QUANTITATIVE_COLORMAP

    # --- Georeferencing: from the ortho, else from the raster ---
    if ortho_tif:
        epsg_str, crs_name, crs_int = _read_ortho_crs(ortho_tif)
        rgb, ortho_extent = _read_ortho_image(ortho_tif)
    else:
        epsg_str, crs_name, crs_int = "—", "(no ortho)", None
        rgb, ortho_extent = None, None
        if layer == "raster" and raster_tif:
            try:
                epsg_str, crs_name, crs_int = _read_ortho_crs(raster_tif)
            except Exception:
                pass

    # --- Data layer first: the frame is set from its extent before the basemap ---
    df = None
    raster_arr = None
    raster_extent = None
    cell_size = None
    if layer == "vector":
        df = pd.read_csv(clast_csv)
        if field not in df.columns:
            raise ValueError(f"Column {field!r} not found in {clast_csv}. "
                             f"Available: {list(df.columns)}")
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError(f"CSV must contain x and y columns, got {list(df.columns)}")
        xs = df["x"].values
        ys = df["y"].values
        data_extent = (float(np.min(xs)), float(np.max(xs)),
                       float(np.min(ys)), float(np.max(ys)))
        valid_bbox = data_extent
    else:
        raster_arr, raster_extent = _read_raster_layer(raster_tif)
        data_extent = raster_extent
        valid_bbox = _valid_data_bbox(raster_arr, raster_extent)
        ny_r, nx_r = raster_arr.shape[:2]
        if nx_r > 0 and ny_r > 0:
            cell_size = float(abs(raster_extent[1] - raster_extent[0]) / nx_r)

    dxmin, dxmax, dymin, dymax = data_extent

    fig, ax = plt.subplots(figsize=figsize)
    extent_mode = "raster" if extent in ("raster", "data") else "project"
    if extent_mode == "raster":
        ax_xmin, ax_xmax, ax_ymin, ax_ymax = _expand_bbox(
            valid_bbox, float(extent_margin))
    else:
        # Union of the padded data extent and the ortho, so the ortho is
        # fully visible even when the data covers a small patch.
        span_x = dxmax - dxmin
        span_y = dymax - dymin
        pad_x = span_x * extent_padding
        pad_y = span_y * extent_padding
        if ortho_extent is not None:
            oxmin, oxmax, oymin, oymax = ortho_extent
            ax_xmin = min(dxmin - pad_x, oxmin)
            ax_xmax = max(dxmax + pad_x, oxmax)
            ax_ymin = min(dymin - pad_y, oymin)
            ax_ymax = max(dymax + pad_y, oymax)
        else:
            ax_xmin = dxmin - pad_x
            ax_xmax = dxmax + pad_x
            ax_ymin = dymin - pad_y
            ax_ymax = dymax + pad_y
    ax.set_xlim(ax_xmin, ax_xmax)
    ax.set_ylim(ax_ymin, ax_ymax)
    ax.set_aspect("equal")

    # --- Basemap (bottom of the z-order) ---
    basemap_added = False
    if basemap is not None:
        basemap_added = _add_basemap_safe(ax, crs_int, basemap, log_fn=log_fn)

    # --- Source ortho overlay ---
    if show_ortho and rgb is not None and ortho_extent is not None:
        ax.imshow(rgb, extent=ortho_extent, origin="upper",
                  alpha=ortho_alpha, zorder=2)

    # --- Data layer ---
    cbar = None
    # 'linear' = min-max mapping; 'quantile' = BoundaryNorm over 256
    # quantile-spaced edges (GIS "quantile classification": equal area per
    # colour, which percentile-clipping vmin/vmax does NOT give);
    # classes=N = N discrete classes with the edges as colorbar ticks.
    fixed_range = vmin is not None and vmax is not None
    if color_scale == "quantile":
        clip_lo, clip_hi = quantile_low * 100.0, quantile_high * 100.0
    else:
        clip_lo, clip_hi = linear_low_pct * 100.0, linear_high_pct * 100.0

    def _clip_text() -> str:
        if fixed_range:
            return "fixed range"
        return f"{clip_lo:g}–{clip_hi:g} % clip"

    def _autoscale(values):
        """``(vmin, vmax, norm, cmap_obj, class_edges)`` honouring explicit
        user limits. ``norm`` is None (linear) or a Normalize to pass to
        imshow/scatter; ``cmap_obj`` is a ListedColormap for a classed map;
        ``class_edges`` is the boundary list or None. Empty or all-equal
        input falls back to a tiny non-zero window so a zero-width
        normalisation never reaches matplotlib."""
        base_cmap = plt.get_cmap(cmap)
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]

        if is_cyclic:
            # Fixed axial scale; the caller's vmin/vmax do not apply.
            lo, hi = CYCLIC_RANGE_DEG
            return lo, hi, None, base_cmap, None

        v_lo = vmin
        v_hi = vmax
        norm = None
        if v_lo is None or v_hi is None:
            if len(finite) == 0:
                if v_lo is None: v_lo = 0.0
                if v_hi is None: v_hi = 1.0
                return v_lo, v_hi, None, base_cmap, None
            computed_lo = float(np.nanpercentile(finite, clip_lo))
            computed_hi = float(np.nanpercentile(finite, clip_hi))
            if v_lo is None:
                v_lo = computed_lo
            if v_hi is None:
                v_hi = computed_hi
            # A zero-width range breaks BoundaryNorm's edge slicing.
            if v_lo == v_hi:
                pad = max(abs(v_lo) * 1e-6, 1e-12)
                v_lo -= pad
                v_hi += pad

        if classes is not None:
            if color_scale == "quantile" and len(finite) > 0:
                inside = finite[(finite >= v_lo) & (finite <= v_hi)]
                if len(inside) == 0:
                    inside = finite
                edges = np.nanpercentile(
                    inside, np.linspace(0.0, 100.0, classes + 1))
                edges[0], edges[-1] = v_lo, v_hi
            else:
                edges = np.linspace(v_lo, v_hi, classes + 1)
            edges = np.unique(np.asarray(edges, dtype=float))
            if len(edges) < 2 or edges[-1] - edges[0] < 1e-12:
                edges = np.linspace(v_lo, v_hi if v_hi > v_lo else v_lo + 1.0,
                                    classes + 1)
            n_cls = len(edges) - 1
            listed = mcolors.ListedColormap(
                base_cmap(np.linspace(0.0, 1.0, n_cls)),
                name=f"{cmap}_{n_cls}classes")
            # Out-of-range values take the end colours (extend caps).
            listed.set_under(base_cmap(0.0))
            listed.set_over(base_cmap(1.0))
            norm = mcolors.BoundaryNorm(edges, ncolors=n_cls, clip=False)
            return v_lo, v_hi, norm, listed, [float(e) for e in edges]

        if color_scale == "quantile":
            # BoundaryNorm wants N+1 edges for N bins.
            n_bins = 256
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.nanpercentile(finite, qs * 100)
            edges[0] = min(edges[0], v_lo)
            edges[-1] = max(edges[-1], v_hi)
            # BoundaryNorm needs monotonic edges; repeated values break that.
            edges = np.maximum.accumulate(edges)
            if edges[-1] - edges[0] < 1e-12:
                edges = np.linspace(v_lo, v_hi if v_hi > v_lo else v_lo + 1.0,
                                    n_bins + 1)
            norm = mcolors.BoundaryNorm(edges, ncolors=256, clip=True)
        return v_lo, v_hi, norm, base_cmap, None

    extend_kw = "both" if show_colorbar_extends else "neither"

    def _classification_text(class_edges) -> str:
        if is_cyclic:
            return "cyclic, fixed 0–180° scale"
        if class_edges is not None:
            n = len(class_edges) - 1
            kind = "quantile" if color_scale == "quantile" else "equal interval"
            return f"{kind}, {n} classes, {_clip_text()}"
        if color_scale == "quantile":
            return f"quantile, continuous (256 quantile bins), {_clip_text()}"
        return f"linear, {_clip_text()}"

    def _format_cbar(cbar, is_quantile: bool, unit_str: str, v_hi: float,
                     class_edges):
        from matplotlib.ticker import FormatStrFormatter, FuncFormatter
        if is_quantile or class_edges is not None or is_cyclic:
            # BoundaryNorm yields a busy minor-tick set.
            cbar.ax.minorticks_off()
        if is_cyclic:
            fmt = _tick_formatter("°", None, CYCLIC_RANGE_DEG[1])
            cbar.set_ticks(list(CYCLIC_TICKS_DEG))
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        elif class_edges is not None:
            fmt = _tick_formatter(unit_str, uncertainty, v_hi)
            cbar.set_ticks(list(class_edges))
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        elif uncertainty is not None:
            fmt = _tick_formatter(unit_str, uncertainty, v_hi)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        else:
            # Three decimals captures mm resolution in any of m/cm/mm.
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        cbar.ax.tick_params(labelsize=8)

    def _colorbar(mappable, label: str, class_edges):
        kw = dict(shrink=0.5, pad=0.02, extend=extend_kw, fraction=0.04)
        if class_edges is not None:
            # Equal-height segments; the labels carry the unequal intervals.
            kw["spacing"] = "uniform"
        cb = fig.colorbar(mappable, ax=ax, **kw)
        cb.set_label(label, fontsize=10)
        return cb

    if layer == "vector":
        c_raw = df[field].values.astype(float)
        n_total = int(len(c_raw))
        unit_str, factor = _resolve_unit_for_field(field, c_raw, unit_system,
                                                   size_unit)
        c = c_raw * factor
        if is_cyclic:
            c = np.where(np.isfinite(c), np.mod(c, CYCLIC_RANGE_DEG[1]), c)
        plotted = c
        v_lo, v_hi, norm, cmap_obj, class_edges = _autoscale(c)
        # Dense point clouds hide the backdrop: shrink markers and add
        # transparency as N grows (clamped, monotone; ≤ 2000 points untouched).
        n_pts = int(np.count_nonzero(np.isfinite(c)))
        n_valid = n_pts
        eff_size = point_size
        pt_alpha = 1.0
        if n_pts > 2000:
            # 2k pts → ~1.0×, 20k → ~0.55×, 200k → ~0.30×.
            shrink = float(np.clip(1.0 - 0.22 * np.log10(n_pts / 2000.0),
                                   0.30, 1.0))
            eff_size = max(point_size * shrink, 2.0)
            pt_alpha = float(np.clip(1.0 - 0.18 * np.log10(n_pts / 2000.0),
                                     0.35, 1.0))
        if norm is not None:
            sc = ax.scatter(xs, ys, c=c, cmap=cmap_obj, s=eff_size,
                            norm=norm, alpha=pt_alpha,
                            edgecolors="none", zorder=4)
        else:
            sc = ax.scatter(xs, ys, c=c, cmap=cmap_obj, s=eff_size,
                            vmin=v_lo, vmax=v_hi, alpha=pt_alpha,
                            edgecolors="none", zorder=4)
        mappable = sc
        label_text = colorbar_label or _format_label_with_unit(field, unit_str)
        if show_legend:
            cbar = _colorbar(sc, label_text, class_edges)
            _format_cbar(cbar, is_quantile=(norm is not None and class_edges is None),
                         unit_str=unit_str, v_hi=v_hi, class_edges=class_edges)
        source_path = clast_csv
    else:  # raster
        raster_field_for_unit = raster_stem
        n_total = int(raster_arr.size)
        unit_str, factor = _resolve_unit_for_field(
            raster_field_for_unit, raster_arr[~np.isnan(raster_arr)],
            unit_system, size_unit, parameter=parameter)
        raster_disp = raster_arr * factor
        if is_cyclic:
            raster_disp = np.where(np.isfinite(raster_disp),
                                   np.mod(raster_disp, CYCLIC_RANGE_DEG[1]),
                                   raster_disp)
        plotted = raster_disp[~np.isnan(raster_disp)]
        n_valid = int(plotted.size)

        v_lo, v_hi, norm, cmap_obj, class_edges = _autoscale(plotted)
        if norm is not None:
            im = ax.imshow(raster_disp, extent=raster_extent, origin="upper",
                           cmap=cmap_obj, norm=norm,
                           alpha=raster_alpha, zorder=4)
        else:
            im = ax.imshow(raster_disp, extent=raster_extent, origin="upper",
                           cmap=cmap_obj, vmin=v_lo, vmax=v_hi,
                           alpha=raster_alpha, zorder=4)
        mappable = im
        label_text = (colorbar_label
                      or _raster_label(raster_stem, parameter, unit_str))
        _n_bands = _raster_band_count(raster_tif)
        if _n_bands > 1:
            label_text = f"{label_text} — band 1 of {_n_bands}"
        if show_legend:
            cbar = _colorbar(im, label_text, class_edges)
            _format_cbar(cbar, is_quantile=(norm is not None and class_edges is None),
                         unit_str=unit_str, v_hi=v_hi, class_edges=class_edges)
        source_path = raster_tif

    # --- Grid: above the data layer (z=4), below the decorations (z=22+) ---
    if show_grid:
        ax.grid(True, color="white", linewidth=0.6, alpha=0.5, linestyle="--",
                zorder=10)
        ax.set_axisbelow(False)
        ax.tick_params(direction="in", labelsize=8)
        plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_zebra_border:
        _add_zebra_border(ax)

    if show_crs_info:
        crs_text = f"EPSG:{epsg_str}\n{crs_name}"
        _add_crs_box(ax, crs_text)

    if show_north_arrow:
        _add_north_arrow(ax)

    if show_scale_bar:
        # The axes are metres (projected CRS); the labels follow the unit
        # system, as the colorbar's do.
        _add_scale_bar(ax, unit_system=unit_system)

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    # No offset notation ("+6.9815e6" in the corner) on a publication map.
    _use_full_coordinates(ax)
    if basemap_added:
        _add_attribution_below(ax, basemap)

    fig.tight_layout()

    # --- Caption metadata ---
    finite_plotted = np.asarray(plotted, dtype=float)
    finite_plotted = finite_plotted[np.isfinite(finite_plotted)]
    crs_label = (f"EPSG:{crs_int}" if crs_int
                 else (f"EPSG:{epsg_str}" if epsg_str not in ("?", "—") else None))
    metadata = {
        "field": field if layer == "vector" else raster_stem,
        "parameter": parameter,
        "layer": layer,
        "cell_size": cell_size,
        "unit": unit_str,
        "unit_factor": float(factor),
        "cmap": getattr(cmap_obj, "name", str(cmap)) if class_edges is None else cmap,
        "cyclic": bool(is_cyclic),
        "color_scale": "cyclic" if is_cyclic else color_scale,
        "classification": _classification_text(class_edges),
        "classes": (len(class_edges) - 1) if class_edges is not None else None,
        "class_edges": class_edges,
        "value_min": float(v_lo),
        "value_max": float(v_hi),
        "data_min": float(finite_plotted.min()) if finite_plotted.size else None,
        "data_max": float(finite_plotted.max()) if finite_plotted.size else None,
        "uncertainty": (float(uncertainty) if uncertainty is not None else None),
        "n_valid": int(n_valid),
        "n_total": int(n_total),
        "crs": crs_label,
        "crs_name": crs_name,
        "extent_mode": extent_mode,
        "extent": [float(ax_xmin), float(ax_xmax), float(ax_ymin), float(ax_ymax)],
        "data_extent": [float(v) for v in data_extent],
        "colorbar_label": label_text,
        "source": str(source_path) if source_path else None,
        "ortho": str(ortho_tif) if ortho_tif else None,
        "basemap": basemap,
        "basemap_added": bool(basemap_added),
        "exporter_version": EXPORTER_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fig.map_metadata = metadata
    fig._pm_colorbar = cbar      # test hook: the colorbar this map drew
    fig._pm_mappable = mappable  # test hook: the scatter / image artist
    if return_metadata:
        return fig, metadata
    return fig
