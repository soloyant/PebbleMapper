"""Shared per-clast field / per-cell parameter registry.

Single source of truth for how a per-clast field is displayed (unit and
multiplier from stored SI to display value), which colormap a per-cell map
uses, and which per-cell statistic (the rasterize "parameter") overrides the
field's unit/colormap.

No heavy imports (no matplotlib, GDAL or pandas) so any tab can import it at
module load; stdlib + numpy only.

Lookup convention: keys are matched case-insensitively as substrings of the
queried field/parameter name and the longest matching key wins, so
"folk_ward_sorting" beats "sorting" beats "Clast_length" for the string
`Clast_length_folk_ward_sorting`. The rasterize tab bakes both the source
field and the per-cell parameter into the filename, and the report must
recover both.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np


# --- Field unit table: per-clast quantities ---
# Every column the pipeline writes to a per-clast CSV needs a row, or the
# report falls back to a bare number with no unit.
# Row: (key_substring, (display_unit, multiplier_from_stored_to_display)).
# Longest substring match wins.
FIELD_UNIT_TABLE: list[tuple[str, tuple[str, float]]] = [
    # --- Linear sizes — stored in metres, displayed in millimetres ---
    ("clast_length",          ("mm",  1000.0)),
    ("clast_width",           ("mm",  1000.0)),
    ("ellipse_major_axis",    ("mm",  1000.0)),
    ("ellipse_minor_axis",    ("mm",  1000.0)),
    ("major_axis",            ("mm",  1000.0)),
    ("minor_axis",            ("mm",  1000.0)),
    ("equivalent_diameter",   ("mm",  1000.0)),
    ("perimeter",             ("mm",  1000.0)),
    # --- Areas — stored in m², displayed in mm² ---
    ("surface_area",          ("mm²", 1_000_000.0)),
    # --- Angles — already in degrees, range [0, 180) for axis directions ---
    ("orientation",           ("°",   1.0)),
    # --- Velocities (Hjulström curve, Le Roux wave-orbital) — m/s ---
    ("hjulstrom_deposition_velocity", ("m/s", 1.0)),
    ("hjulstrom_erosion_velocity",    ("m/s", 1.0)),
    ("leroux_wave_orbital_velocity",  ("m/s", 1.0)),
    # --- Dimensionless transport parameters ---
    ("van_rijn_dimensionless_diameter",        ("", 1.0)),
    ("shields_critical_grain_reynolds_number", ("", 1.0)),
    ("soulsby_critical_shields",               ("", 1.0)),
    # --- Stresses / velocities derived from Shields ---
    ("shields_critical_shear_stress",   ("Pa",  1.0)),
    ("shields_critical_shear_velocity", ("m/s", 1.0)),
    # --- Dimensionless shape / packing metrics (0–1 bounded) ---
    ("clast_circularity",     ("",    1.0)),
    ("ellipse_circularity",   ("",    1.0)),
    ("circularity",           ("",    1.0)),
    ("clast_elongation",      ("",    1.0)),
    ("ellipse_elongation",    ("",    1.0)),
    ("elongation",            ("",    1.0)),
    ("eccentricity",          ("",    1.0)),
    ("solidity",              ("",    1.0)),
    ("mean_intensity",        ("",    1.0)),
    ("packing_index",         ("",    1.0)),
    ("packing_clustering",    ("",    1.0)),
    # --- Folk-Ward statistics on the phi scale (rare as per-clast columns) ---
    ("folk_ward_sorting",     ("φ",   1.0)),
    ("folk_ward_skewness",    ("φ",   1.0)),
    ("folk_ward_kurtosis",    ("",    1.0)),  # K_G is dimensionless
    # --- Score from the detection model (0–1) ---
    ("score",                 ("",    1.0)),
    # --- Density / count ---
    ("density",               ("clasts/m²", 1.0)),
    # --- Fallback ---
    ("clast_id",              ("",    1.0)),
]


# Fields declared by user add-ons (functions.addons), keyed by lower-case
# name: exact-name matches take precedence over the substring table.
_REGISTERED_FIELD_UNITS: dict = {}

_DIMENSIONLESS_WORDS = {"", "dimensionless", "none", "-", "1", "unitless", "ratio"}


def register_field_unit(field: str, unit: Optional[str], factor: float = 1.0) -> None:
    """Declare the display unit and multiplier of a field that is not in the
    built-in table, typically a user add-on. ``unit`` words meaning "no unit"
    become the empty string; ``factor`` multiplies the stored value for display."""
    if not field:
        return
    u = (unit or "").strip()
    if u.lower() in _DIMENSIONLESS_WORDS:
        u = ""
    try:
        f = float(factor) if factor else 1.0
    except (TypeError, ValueError):
        f = 1.0
    _REGISTERED_FIELD_UNITS[str(field).strip().lower()] = (u, f)


def registered_field_units() -> dict:
    """A copy of the add-on field registry."""
    return dict(_REGISTERED_FIELD_UNITS)


def clear_registered_field_units() -> None:
    """Forget every add-on unit registered so far."""
    _REGISTERED_FIELD_UNITS.clear()


def field_unit_and_factor(field: Optional[str]) -> Tuple[str, float]:
    """``(display_unit, multiplier)`` for a per-clast field: an add-on
    registered under exactly this name first, else the longest
    case-insensitive substring match in the table. Unknown fields fall back
    to ``("m", 1.0)``."""
    if not field:
        return ("m", 1.0)
    needle = field.lower()
    hit = _REGISTERED_FIELD_UNITS.get(needle.strip())
    if hit is not None:
        return hit
    best: Optional[Tuple[str, float]] = None
    best_len = -1
    for key, val in FIELD_UNIT_TABLE:
        if key in needle and len(key) > best_len:
            best = val
            best_len = len(key)
    return best if best is not None else ("m", 1.0)


def field_unit(field: Optional[str]) -> str:
    """The display-unit string for ``field``."""
    return field_unit_and_factor(field)[0]


def format_value_with_unit(value: Optional[float], field: str,
                           u_display: Optional[float] = None) -> str:
    """Format ``value`` (stored unit) as a string in the field's display unit.

    ``u_display`` is the measurement's uncertainty in the DISPLAY unit. When
    given, the value is rounded to the precision that uncertainty supports
    instead of to a fixed per-unit convention.

    Examples (field -> output)::

        Clast_length:     0.0034 m  ->  '3.40 mm'
        Surface_area:    1.2e-05 m² ->  '12.0 mm²'
        Orientation:     1.5°       ->  '1.50 °'
        Clast_circularity: 0.45     ->  '0.450'
        Hjulstrom_*:      0.123 m/s ->  '0.123 m/s'
    """
    if value is None or not np.isfinite(value):
        return "—"
    unit, factor = field_unit_and_factor(field)
    disp = value * factor
    if u_display is not None:
        from functions.precision import format_value as _fmt
        return f"{_fmt(disp, u_display, unit)} {unit}".strip()
    if unit == "mm":
        return f"{disp:.2f} mm"
    if unit == "mm²":
        return f"{disp:.1f} mm²"
    if unit == "°":
        return f"{disp:.2f} °"
    if unit == "m/s":
        return f"{disp:.3f} m/s"
    if unit == "Pa":
        return f"{disp:.3g} Pa"
    if unit == "φ":
        return f"{disp:.2f} φ"
    if unit == "clasts/m²":
        return f"{disp:.2f} clasts/m²"
    if unit == "":
        return f"{disp:.3f}"
    return f"{disp:.3f} {unit}"


# --- Field registry: display name + colormap (per-cell raster panels) ---
# Row: (key, display, unit, factor, cmap, diverging). Keys match as
# case-insensitive substrings, longest wins. ``diverging`` means the values
# are signed and the colorbar centres on 0.
# Colormap conventions: sizes / velocities viridis; signed deviations RdBu_r;
# cyclic (orientation) twilight; bounded shape / packing single-hue.
FIELD_REGISTRY: list[tuple[str, str, str, float, str, bool]] = [
    # Sizes: rasters store metres, display millimetres.
    ("Clast_length",        "Clast length",        "mm",         1000.0, "viridis", False),
    ("Clast_width",         "Clast width",         "mm",         1000.0, "viridis", False),
    ("Equivalent_diameter", "Equivalent diameter", "mm",         1000.0, "viridis", False),
    ("Perimeter",           "Perimeter",           "mm",         1000.0, "viridis", False),
    ("Surface_area",        "Surface area",        "mm²",     1_000_000.0, "viridis", False),
    # Ellipse-axis columns of older CSVs.
    ("Ellipse_major_axis",  "Ellipse major axis",  "mm",         1000.0, "viridis", False),
    ("Ellipse_minor_axis",  "Ellipse minor axis",  "mm",         1000.0, "viridis", False),

    # Axis direction, range [0, 180): a cyclic palette so 175 and 5 degrees look alike.
    ("Orientation",         "Orientation",         "°",            1.0, "twilight", False),

    # Bounded shape (0-1).
    ("Clast_circularity",   "Circularity",         "",             1.0, "YlGnBu", False),
    ("Clast_elongation",    "Clast elongation",    "",             1.0, "YlGnBu", False),
    ("Eccentricity",        "Eccentricity",        "",             1.0, "YlGnBu", False),
    ("Solidity",            "Solidity",            "",             1.0, "YlGnBu", False),
    # Per-grain mean brightness (0–255) under the detection mask.
    ("Mean_intensity",      "Mean intensity",      "",             1.0, "gray",   False),

    # Sediment-transport thresholds, base units.
    ("Hjulstrom_deposition_velocity",  "Hjulström deposition velocity",  "m/s", 1.0, "viridis", False),
    ("Hjulstrom_erosion_velocity",     "Hjulström erosion velocity (Shields-based)", "m/s", 1.0, "viridis", False),
    ("Leroux_wave_orbital_velocity",   "Le Roux wave orbital velocity",  "m/s", 1.0, "viridis", False),
    ("Van_Rijn_dimensionless_diameter","Van Rijn D*",                    "",    1.0, "viridis", False),
    ("Shields_critical_grain_reynolds_number",
                                       "Shields critical grain Re_p",    "",    1.0, "viridis", False),
    ("Shields_critical_shear_stress",  "Shields critical shear stress",  "Pa",  1.0, "viridis", False),
    ("Shields_critical_shear_velocity","Shields critical shear velocity","m/s", 1.0, "viridis", False),
    ("Soulsby_critical_shields",       "Soulsby critical Shields",       "",    1.0, "viridis", False),

    # Packing metrics: bounded 0–1, single-hue.
    ("packing_index",       "Packing index",       "",             1.0, "BuPu",    False),
    ("packing_clustering",  "Packing clustering",  "",             1.0, "BuPu",    False),

    # Folk-Ward as per-clast columns (rare).
    ("folk_ward_sorting",   "Folk-Ward sorting σφ", "φ",           1.0, "cividis", False),
    ("folk_ward_skewness",  "Folk-Ward Sk φ",       "φ",           1.0, "RdBu_r",  True),
    ("folk_ward_kurtosis",  "Folk-Ward K_G",        "",            1.0, "cividis", False),

    # Density / abundance.
    ("density",             "Clast density",        "clasts/m²",   1.0, "inferno", False),

    # Detection model score (0–1).
    ("score",               "Detection score",      "",             1.0, "cividis", False),
]


def field_registry_lookup(field: Optional[str]):
    """The registry row whose key is the longest substring of ``field``, or None."""
    if not field:
        return None
    needle = field.lower()
    best = None
    best_len = -1
    for row in FIELD_REGISTRY:
        key = row[0].lower()
        if key in needle and len(key) > best_len:
            best = row
            best_len = len(key)
    return best


def field_display(field: Optional[str]) -> Tuple[str, str, float]:
    """``(display_name, unit, factor)`` for a cleaned field string.

    Unregistered fields get a tidied raw name and no unit. Density rasters
    (no real per-clast field) get an empty display string so the caller
    renders the parameter alone.
    """
    row = field_registry_lookup(field)
    if row:
        _, display, unit, factor, _cmap, _div = row
        return display, unit, factor
    if not field:
        return "", "", 1.0
    cleaned = clean_field_name(field)
    if not cleaned:
        return "", "", 1.0
    return cleaned.replace("_", " "), "", 1.0


# --- Parameter registry: per-cell statistic ---
# The "parameter" half of a rasterize filename. Some parameters change the
# physical unit of the raster's stored values: sorting / skewness / kurtosis
# are phi-derived (rasterize converts the size field to phi first), density
# is clasts per m2. When they apply, the parameter's unit / factor / cmap
# must win over the field's.
# Row: key -> (display_name, unit, factor, cmap_or_None, diverging).
# ``unit`` None means the field's unit/factor apply; ``cmap_or_None`` None
# means the field's cmap applies.
PARAMETER_DISPLAY: dict[str, tuple[str, Optional[str], Optional[float], Optional[str], bool]] = {
    # Parameters that preserve the field's unit. Percentile names stay neutral
    # (no "D50" suffix): the D-notation only makes sense for linear sizes and
    # the heading already names the field.
    "average":  ("average",          None, None, None,      False),
    "std":     ("standard deviation", None, None, "cividis", False),
    "D5":       ("5th percentile",    None, None, None,      False),
    "D16":      ("16th percentile",   None, None, None,      False),
    "D50":      ("median",            None, None, None,      False),
    "D84":      ("84th percentile",   None, None, None,      False),
    "D95":      ("95th percentile",   None, None, None,      False),
    "d_percentiles": ("D-percentile bands (D5, D16, D50, D84, D95)",
                      None, None, None, False),
    "distribution":  ("size-bin distribution",
                      None, None, None, False),

    # Parameters that replace the unit (phi-derived per-cell statistics)
    "sorting":            ("sorting σφ",            "φ", 1.0, "cividis", False),
    "skewness":           ("skewness Sk φ",         "φ", 1.0, "RdBu_r",  True),
    "kurtosis":           ("kurtosis K_G",          "",  1.0, "cividis", False),
    "folk_ward_sorting":  ("Folk-Ward sorting σφ",  "φ", 1.0, "cividis", False),
    "folk_ward_skewness": ("Folk-Ward Sk φ",        "φ", 1.0, "RdBu_r",  True),
    "folk_ward_kurtosis": ("Folk-Ward K_G",         "",  1.0, "cividis", False),

    # Parameters whose unit is independent of the field
    "density":            ("count per cell",        "clasts/m²", 1.0, "inferno", False),
    "packing_index":      ("packing index",         "",  1.0, "BuPu",    False),
    "packing_clustering": ("packing clustering",    "",  1.0, "BuPu",    False),
}


# Percentile + average panels of one field share a colorbar so the spatial
# gradient is visible as colour shift, not just colorbar rescaling.
PERCENTILE_FAMILY: set[str] = {"D5", "D16", "D50", "D84", "D95", "average"}


def parameter_display(parameter: Optional[str]):
    """The PARAMETER_DISPLAY row for ``parameter``, or None."""
    if not parameter:
        return None
    return PARAMETER_DISPLAY.get(parameter)


def resolve_display(field: Optional[str], parameter: Optional[str]
                    ) -> Tuple[str, str, float, str, bool]:
    """Combine field + parameter into ``(display, unit, factor, cmap, diverging)``.

    The parameter wins for unit/factor when it sets a unit, and for cmap when
    it pins one; otherwise the field's values apply. The display string is
    ``"<field> - <parameter>"``.
    """
    f_display, f_unit, f_factor = field_display(field)
    p_row = parameter_display(parameter)

    if p_row is None:
        cmap, diverging = "viridis", False
        row = field_registry_lookup(field)
        if row:
            cmap, diverging = row[4], row[5]
        if parameter:
            display = f"{f_display} — {parameter.replace('_', ' ')}" if f_display else parameter
        else:
            display = f_display
        return display, f_unit, f_factor, cmap, diverging

    p_display, p_unit, p_factor, p_cmap, p_diverging = p_row
    unit = p_unit if p_unit is not None else f_unit
    factor = p_factor if p_factor is not None else f_factor

    if p_cmap is not None:
        cmap = p_cmap
        diverging = p_diverging
    else:
        row = field_registry_lookup(field)
        if row:
            cmap, diverging = row[4], row[5]
        else:
            cmap, diverging = "viridis", False

    if f_display:
        display = f"{f_display} — {p_display}"
    else:
        display = p_display
    return display, unit, factor, cmap, diverging


# --- Helpers used by both the report and the validate tab ---
# Fields where phi (Krumbein) and Folk-Ward statistics make physical sense;
# anything else skips the phi block (phi of an orientation in degrees is
# defined but meaningless). Case-insensitive substring matching.
SIZE_FIELDS_FOR_PHI: tuple[str, ...] = (
    "clast_length", "clast_width",
    "ellipse_major_axis", "ellipse_minor_axis",
    "major_axis", "minor_axis",
    "equivalent_diameter",
    # Surface_area is excluded: phi is defined on linear sizes, not areas.
)


def is_size_field_for_phi(field: Optional[str]) -> bool:
    """True iff ``field`` is a linear-size column where φ-statistics make sense."""
    if not field:
        return False
    needle = field.lower()
    return any(k in needle for k in SIZE_FIELDS_FOR_PHI)


# --- Folk-Ward graphic statistics: the one shared arithmetic ---
# Folk, R.L. & Ward, W.C. (1957) "Brazos River bar: a study in the
# significance of grain size parameters", J. Sediment. Petrol. 27(1):3.
# These take PRE-COMPUTED phi-quantiles, not raw samples; each caller
# extracts its quantiles its own way and keeps its own sample-size gates.
# Not for the rasterize velocity-field "sorting" branch, which applies the
# same arithmetic to wave-orbital velocity components, not to phi sizes.
# Krumbein phi: phi = -log2(D / D0), D0 = 1 mm (Krumbein 1934).
PHI_REF_M: float = 1e-3


def folk_ward_sorting_phi(p5: float, p16: float, p84: float, p95: float) -> float:
    """Folk-Ward graphic sorting: (phi84 - phi16) / 4 + (phi95 - phi5) / 6.6.
    NaN in -> NaN out; sample-size gates stay at the call site."""
    if not (np.isfinite(p5) and np.isfinite(p16)
            and np.isfinite(p84) and np.isfinite(p95)):
        return float("nan")
    return float((p84 - p16) / 4.0 + (p95 - p5) / 6.6)


def folk_ward_skewness_phi(p5: float, p16: float, p50: float,
                           p84: float, p95: float) -> float:
    """Folk-Ward graphic skewness:
    (phi16 + phi84 - 2 phi50) / (2 (phi84 - phi16)) + (phi5 + phi95 - 2 phi50) / (2 (phi95 - phi5)).
    NaN in -> NaN out, NaN when a denominator is zero."""
    if not (np.isfinite(p5) and np.isfinite(p16) and np.isfinite(p50)
            and np.isfinite(p84) and np.isfinite(p95)):
        return float("nan")
    denom_a = 2.0 * (p84 - p16)
    denom_b = 2.0 * (p95 - p5)
    if denom_a == 0.0 or denom_b == 0.0:
        return float("nan")
    return float((p16 + p84 - 2.0 * p50) / denom_a
                 + (p5 + p95 - 2.0 * p50) / denom_b)


def folk_ward_kurtosis_phi(p5: float, p25: float,
                           p75: float, p95: float) -> float:
    """Folk-Ward graphic kurtosis: (phi95 - phi5) / (2.44 (phi75 - phi25)).
    NaN in -> NaN out, NaN when the denominator is zero."""
    if not (np.isfinite(p5) and np.isfinite(p25)
            and np.isfinite(p75) and np.isfinite(p95)):
        return float("nan")
    denom = 2.44 * (p75 - p25)
    if denom == 0.0:
        return float("nan")
    return float((p95 - p5) / denom)


def clean_field_name(s: Optional[str]) -> str:
    """Strip the source-CSV stem that Rasterize bakes ahead of the
    ``<Field>_<parameter>_cellsize=...`` suffix, in short and legacy forms.

    Density rasters have no per-clast field, so the result is an empty
    string and callers render the heading from the parameter alone.
    """
    if not s:
        return s or ""
    # Keep whatever follows the LAST naming token.
    m = None
    for _m in re.finditer(r"_(?:window_size=|ws)[0-9.]+m_", s):
        m = _m
    if m is not None:
        s = s[m.end():]
    else:
        for marker in ("individual_clast_values_", "_merged_", "_xprs_"):
            i = s.rfind(marker)
            if i >= 0:
                s = s[i + len(marker):]
                break
    # Density rasters end at "..._individual_clast_values" with no trailing field.
    if s.endswith("individual_clast_values"):
        s = s[: -len("individual_clast_values")].rstrip("_")
    return s
