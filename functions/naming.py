"""Output-filename conventions: build and parse, in one place.

Every analysis output starts with an ORIGIN that says where the data came
from, so a file stays self-describing outside its folder::

    <origin> := [<project>[_<YYYYMMDD>]__]<image_stem>

The site (project name) and the date (dated projects only) sit before a
double underscore; either is omitted when the image stem already carries it.
Processed images (rectified, georeferenced) keep their photograph's stem so
they still pair with the raw file; only analysis outputs carry the origin.

Written forms::

    <origin>_ws<size>m.csv                 detection at one window size
    <origin>_ws<size>m.run.csv             partial-run checkpoint
    <origin>_individual_clasts.csv         quadrat (full-image) detection
    <origin>_merged.csv                    merge-tool output
    <origin>_xprs_ws<size>m.csv            Express-tab detection
    <origin>_xprs_merged.csv               Express-tab merge
    <csv stem>_<field>_<param>_cellsize=<c>m.tif      raster
    <raster stem>_map.png                             map
    <csv stem>_zones=<set>_field=<f>.polygons.csv     zonal, polygon mode
    <raster stem>_zones=<set>.transects.csv           zonal, transect mode

Legacy forms (still parsed)::

    <stem>_window_size=<size>m_individual_clast_values.csv
    <stem>_merged_individual_clast_values.csv
    <base>__<zones stem>[__<field tag>].<mode>.csv
    <stem>_xprs_<field>_<param>.tif / <stem>_xprs_<layer>_<field>_<i>.png

The ``_xprs`` tag marks any artefact produced by the one-click Express tab.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

ORIGIN_SEP = "__"

# Suffixes the Orthorectify and Georeference tools append to a photograph's
# stem. A rectified name also carries the quadrat size and the GSD, so it is
# not a stable identity for the photograph; strip these to get back to it.
TOOL_SUFFIXES = ("_rectified", "_corrected", "_georeferenced")
_LEGACY_SUFFIX = "_individual_clast_values"
_QUADRAT_SUFFIX = "_individual_clasts"
# Matches either the short ``_ws2.5m`` or the legacy ``_window_size=2.5m`` token.
_WINDOW_RE = re.compile(r"_(?:window_size=|ws)([0-9]+(?:\.[0-9]+)?)m")
_ZONAL_RE = re.compile(r"^(?P<base>.+?)_zones=(?P<zone>[^=]*?)(?:_field=(?P<field>[^=]*?))?(?:_id=(?P<id>[^=]*))?$")


def _fmt(window) -> str:
    return f"{float(window):g}"


def _norm(s) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s or "").lower())


# --- Origin: site, date, image ---
def origin_stem(image_stem: str, project: Optional[str] = None,
                date: Optional[str] = None) -> str:
    """``[<project>[_<YYYYMMDD>]__]<image_stem>``.

    A part is left out when the image stem already contains it (compared
    without case or separators), so ``02_etretat_20200610_alpha`` under
    ``Normandy_Etretat / 2020-06-10`` becomes
    ``Normandy_Etretat__02_etretat_20200610_alpha``.
    """
    image_stem = str(image_stem or "")
    if ORIGIN_SEP in image_stem:          # already carries an origin
        return image_stem
    img_n = _norm(image_stem)
    parts = []
    project = str(project or "").strip()
    if project and _norm(project) not in img_n:
        parts.append(project)
    digits = re.sub(r"[^0-9]", "", str(date or ""))
    if digits and digits not in img_n:
        parts.append(digits)
    if not parts:
        return image_stem
    return "_".join(parts) + ORIGIN_SEP + image_stem


def split_origin(stem: str) -> tuple:
    """``(prefix, image_part)``; the prefix is ``""`` for a legacy name."""
    s = str(stem or "")
    if ORIGIN_SEP in s:
        pre, rest = s.split(ORIGIN_SEP, 1)
        return pre, rest
    return "", s


def strip_origin(stem: str) -> str:
    return split_origin(stem)[1]


# --- Build ---
def detection_csv_name(stem: str, window, *, express: bool = False,
                       run: bool = False) -> str:
    """Filename for a single-window detection CSV (or its .run.csv checkpoint)."""
    tag = "_xprs" if express else ""
    ext = ".run.csv" if run else ".csv"
    return f"{stem}{tag}_ws{_fmt(window)}m{ext}"


_LABEL_SET_RE = re.compile(r"^(?P<stem>.+)_labels=(?P<model>[A-Za-z0-9_-]+)\.csv$")


def label_set_id(model: str) -> str:
    """The token a model's label set is known by, in its file name too."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(model or "").strip()) or "model"


DEFAULT_MODEL = "maskrcnn"
_MODEL_TOKEN_RE = re.compile(r"_model=([A-Za-z0-9_-]+)$")


def with_model(stem: str, model: Optional[str]) -> str:
    """The stem one model's outputs of a photograph are written under.

    Mask R-CNN keeps the plain stem, so every existing output keeps its
    name; any other model adds ``_model=<id>``, so a second model's run of
    the same photograph writes a second file instead of replacing the
    first. A token already on ``stem`` is replaced."""
    stem = _MODEL_TOKEN_RE.sub("", str(stem or ""))
    if not model or label_set_id(model) == DEFAULT_MODEL:
        return stem
    return f"{stem}_model={label_set_id(model)}"


def model_of(name: str) -> str:
    """The model a detection output's name says made it (Mask R-CNN when
    the name carries no token)."""
    m = _MODEL_TOKEN_RE.search(run_stem(name))
    return m.group(1) if m else DEFAULT_MODEL


def label_set_csv_name(image_stem: str, model: str) -> str:
    """``<image stem>_labels=<model>.csv``: a photograph's clasts as one model
    proposed them (and as they were then edited), kept beside its truth."""
    return f"{image_stem}_labels={label_set_id(model)}.csv"


def parse_label_set_name(name: str) -> Optional[tuple]:
    """``(image_stem, model)`` of a label-set CSV name, else None."""
    m = _LABEL_SET_RE.match(Path(str(name)).name)
    return (m.group("stem"), m.group("model")) if m else None


def quadrat_csv_name(stem: str) -> str:
    """Filename for a full-image (quadrat photograph) detection CSV."""
    return f"{stem}{_QUADRAT_SUFFIX}.csv"


# The pre-rename spelling, kept for one release; new code calls ``quadrat_csv_name``.
terrestrial_csv_name = quadrat_csv_name


def merged_csv_name(stem: str, *, express: bool = False) -> str:
    """Filename for a merge-tool output CSV."""
    tag = "_xprs" if express else ""
    return f"{stem}{tag}_merged.csv"


def bins_token(bin_edges, bin_mode: str = "fixed") -> str:
    """The size bins as one filename-safe token: ``16-64-256``, or
    ``phi-1--2--3`` in phi (where the minus signs of phi values survive)."""
    edges = [f"{float(e):g}" for e in (bin_edges or [])]
    if not edges:
        return ""
    head = "phi" if str(bin_mode).lower().startswith("phi") else ""
    return head + "-".join(edges)


def raster_name(csv_stem: str, field: str, parameter: str, cellsize, *,
                bin_edges=None, bin_mode: str = "fixed", min_density=None) -> str:
    """``<csv stem>_<field>_<parameter>_cellsize=<c>m[_bins=…][_mind=…].tif``.

    The bins and the minimum cell density change what is in the raster, so
    they belong in its name: without them a binned run and an unbinned one,
    or two runs at different densities, wrote the same file and the second
    silently replaced the first. They come
    after ``cellsize=`` so that everything reading that token still reads it.
    """
    parts = [str(csv_stem), str(field or ""), str(parameter or "")]
    name = "_".join(p for p in parts if p) + f"_cellsize={cellsize}m"
    tok = bins_token(bin_edges, bin_mode)
    if tok:
        name += f"_bins={tok}"
    if min_density:
        name += f"_mind={int(min_density)}"
    return name + ".tif"


def map_name(raster_or_csv_stem: str) -> str:
    return f"{raster_or_csv_stem}_map.png"


def zone_token(zone_set: str) -> str:
    """The zone set as it appears after ``zones=``: no leading ``zones_``,
    spaces as hyphens, only filename-safe characters."""
    z = str(zone_set or "").strip()
    z = re.sub(r"^zones[_-]", "", z, flags=re.IGNORECASE)
    z = re.sub(r"\s+", "-", z)
    z = re.sub(r"[^0-9A-Za-z._-]+", "_", z).strip("_")
    return z or "all"


def field_token(field: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(field or "")).strip("_")


def zonal_out_name(base_stem: str, zone_set: str, field: str, mode: str,
                   id_field=None) -> str:
    """Zonal result CSV.

    In polygon mode the base is the detection CSV, the same file for every
    field, so the field must appear in the name. In transect mode the base is
    the raster stem, which already names the field. ``id_field`` labels the
    zones (``None`` or empty: the feature index); it is in the name so that
    two runs of the same field, labelled differently, are two files.
    """
    tag = field_token(field)
    name = f"{base_stem}_zones={zone_token(zone_set)}"
    if tag and tag.lower() not in str(base_stem).lower():
        name += f"_field={tag}"
    if mode == "polygons":
        name += f"_id={field_token(id_field) or 'index'}"
    return f"{name}.{mode}.csv"


def zonal_map_name(raster_stem: str, zone_set: str) -> str:
    return f"{raster_stem}_zones={zone_token(zone_set)}.zonal_map.png"


# --- Parse (accepts both short and legacy forms) ---
def parse_window_size(name: str) -> Optional[float]:
    """Window size in metres parsed from a filename, or None if absent."""
    m = _WINDOW_RE.search(str(name))
    return float(m.group(1)) if m else None


def is_merged(name: str) -> bool:
    return "_merged" in str(name)


def is_express(name: str) -> bool:
    return "_xprs" in str(name)


def run_stem(name: str) -> str:
    """The stem with the package's tokens removed and the origin kept: what a
    downstream output derives its own name from."""
    s = re.sub(r"\.(run\.csv|csv|tiff?|png|jpe?g|json|txt)$", "", str(name),
               flags=re.IGNORECASE)                     # drop extension
    s = s.replace(_LEGACY_SUFFIX, "")
    s = re.sub(r"_(?:window_size=|ws)[0-9.]+m.*$", "", s)
    s = re.sub(r"_merged.*$", "", s)
    s = re.sub(r"_xprs.*$", "", s)
    s = re.sub(rf"{_QUADRAT_SUFFIX}$", "", s)
    return s


def image_stem(name: str) -> str:
    """The source-image stem: the package's naming tokens (short or legacy
    form), the model token and the origin prefix stripped."""
    return strip_origin(_MODEL_TOKEN_RE.sub("", run_stem(name)))


def same_image(name_a: str, name_b: str) -> bool:
    """Do two names (outputs or images) come from the same source image?"""
    return image_stem(name_a) == image_stem(name_b)


def parse_zonal_name(name: str) -> Optional[dict]:
    """``{"mode", "base_stem", "zone_set", "field_tag"}`` for a zonal CSV
    name, in either grammar; None when the name is not a zonal output."""
    s = str(name)
    s = re.sub(r"\.csv$", "", s, flags=re.IGNORECASE)
    mode = None
    for suf in (".polygons", ".transects"):
        if s.lower().endswith(suf):
            mode = suf[1:]
            s = s[: -len(suf)]
            break
    if mode is None:
        return None
    m = _ZONAL_RE.match(s)
    if m:
        return {"mode": mode, "base_stem": m.group("base"),
                "zone_set": m.group("zone") or None,
                "field_tag": m.group("field") or None,
                "id_tag": m.group("id") or None}
    # Legacy: <base>__<zones stem>[__<field tag>]; the base never carried "__".
    base, vec, field = s, None, None
    if ORIGIN_SEP in s:
        base, rest = s.split(ORIGIN_SEP, 1)
        if ORIGIN_SEP in rest:
            vec, field = rest.split(ORIGIN_SEP, 1)
        else:
            vec = rest
    return {"mode": mode, "base_stem": base, "zone_set": vec or None,
            "field_tag": field or None, "id_tag": None}


__all__ = [
    "TOOL_SUFFIXES", "bins_token",
    "ORIGIN_SEP", "origin_stem", "split_origin", "strip_origin",
    "detection_csv_name", "quadrat_csv_name", "terrestrial_csv_name",
    "merged_csv_name",
    "raster_name", "map_name", "zone_token", "field_token",
    "zonal_out_name", "zonal_map_name",
    "parse_window_size", "is_merged", "is_express", "run_stem",
    "image_stem", "same_image", "parse_zonal_name",
]
