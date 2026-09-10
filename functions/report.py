"""Project-level structured PDF report.

Walks a project directory and assembles a single PDF: cover, summary,
data and processing, detection results, spatial statistics, zonal
statistics, validation, interpretation, appendices. Driven by the Report tab
but importable for scripted use.

Entry points:

    collect_project_inventory(project_root) -> dict
        Walk the project and return a structured snapshot of what's there.

    compute_aggregate_stats(inventory) -> dict
        Concatenate the canonical detection CSVs and compute the headline
        statistics (D-percentiles, Folk-Ward moments, runtime, validation).

    build_pdf(project_root, out_path, options, log_fn=None) -> Path
        Orchestrate everything and write the PDF; sections are gated by the
        `options` dict.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import json as _json

import numpy as np
import pandas as pd

from functions._logging import get_logger
from functions import units as _u
from functions import branding as _brand

import contextlib as _contextlib
import logging as _logging
import warnings as _warnings


@_contextlib.contextmanager
def _quiet_degenerate_mpl():
    """Silence the benign numpy/matplotlib warnings a degenerate figure extent
    produces (``divide by zero`` in BboxTransformFrom, ``posx and posy should
    be finite values``); the rendered output is unaffected."""
    _txt_log = _logging.getLogger("matplotlib.text")
    _prev_level = _txt_log.level
    _txt_log.setLevel(_logging.ERROR)
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore", message=".*divide by zero.*")
                _warnings.filterwarnings(
                    "ignore", message=".*invalid value encountered.*")
                yield
    finally:
        _txt_log.setLevel(_prev_level)
from functions import modes, naming

_log = get_logger(__name__)

# Bump when the schema written by _persist_validation_result changes
# incompatibly; readers warn on a lower schema_version.
_VALIDATION_SCHEMA_VERSION = 2


def _recompute_v1_sorting_phi(payload: dict, json_path) -> None:
    """Standardise a schema_version<2 payload's sorting to the Folk-Ward value.

    Schema-1 validation JSONs store ``truth_sorting_phi`` / ``detect_sorting_phi``
    as std(phi), not the Folk-Ward graphic formula, and the phi samples were
    not persisted, so the source CSVs named in the payload are re-read and
    sigma_phi recomputed with the shared helper.

    Mutates ``payload['distribution']`` in place. On success sets
    ``payload['_v1_sorting_recomputed'] = True``; on any fallback (CSV missing
    / unreadable / non-size field / too few samples) leaves the v1 value and
    sets ``payload['_v1_sorting_unrecomputable'] = True`` so the report can
    footnote it. Never raises.
    """
    from functions import validation as _v

    name = Path(json_path).name
    field = payload.get("field") or "Clast_length"

    # Only meaningful for linear-size fields (mirrors validation's phi_meaningful gate).
    if not _u.is_size_field_for_phi(field):
        _log.info(
            "%s: schema_version<2 but field %r is not a size field; "
            "leaving σφ as-is (no Folk-Ward standardisation needed).",
            name, field)
        return

    dist = payload.get("distribution")
    if not isinstance(dist, dict):
        # Nothing to replace — flag so the caption can note the limitation.
        payload["_v1_sorting_unrecomputable"] = True
        _log.warning(
            "%s: schema_version<2 size field but no 'distribution' block; "
            "cannot standardise σφ — flagging caption fallback.", name)
        return

    def _sorting_from_csv(csv_path, side):
        """Read ``field`` from one CSV (metres) → Folk-Ward σφ, or None."""
        if not csv_path:
            _log.warning("%s: schema_version<2 %s_csv path missing; "
                         "cannot recompute σφ.", name, side)
            return None
        p = Path(csv_path)
        if not p.is_file():
            _log.warning("%s: schema_version<2 %s_csv %s not found on disk; "
                         "cannot recompute σφ.", name, side, p)
            return None
        try:
            df = pd.read_csv(p)
        except Exception as exc:  # unreadable / malformed CSV
            _log.warning("%s: schema_version<2 %s_csv %s unreadable (%s); "
                         "cannot recompute σφ.", name, side, p, exc)
            return None
        if field not in df.columns:
            _log.warning("%s: schema_version<2 %s_csv %s has no %r column; "
                         "cannot recompute σφ.", name, side, p, field)
            return None
        vals = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 5:
            _log.warning("%s: schema_version<2 %s_csv %s has only %d positive "
                         "%r samples (<5); cannot recompute σφ.",
                         name, side, p, vals.size, field)
            return None
        phi = _v._phi(vals)
        sigma = _v._folk_sorting_phi(phi)
        if not np.isfinite(sigma):
            _log.warning("%s: schema_version<2 %s σφ recompute produced "
                         "non-finite value; falling back.", name, side)
            return None
        return float(sigma)

    truth_sigma = _sorting_from_csv(payload.get("truth_csv"), "truth")
    detect_sigma = _sorting_from_csv(payload.get("detect_csv"), "detect")

    if truth_sigma is None or detect_sigma is None:
        payload["_v1_sorting_unrecomputable"] = True
        _log.warning(
            "%s: schema_version<2 σφ could NOT be standardised from source "
            "CSVs; keeping the pre-r17 std(φ) value and flagging caption.",
            name)
        return

    old_t = dist.get("truth_sorting_phi")
    old_d = dist.get("detect_sorting_phi")
    dist["truth_sorting_phi"] = truth_sigma
    dist["detect_sorting_phi"] = detect_sigma
    payload["_v1_sorting_recomputed"] = True
    _log.info(
        "%s: schema_version<2 σφ standardised to Folk-Ward — truth %r→%r, "
        "detect %r→%r (recomputed from source CSVs).",
        name, old_t, truth_sigma, old_d, detect_sigma)


def _load_validation_json(path) -> "dict | None":
    """Read a .validation.json file and warn if its schema is outdated.

    Returns the parsed dict, or *None* if the file cannot be read/parsed.
    A ``schema_version`` of 1 stores the raw std of the phi distribution as
    ``sorting_phi``; it is recomputed from the source CSVs when they are
    available, else kept and flagged for a caption note.
    """
    try:
        payload = _json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("Could not read validation JSON %s: %s", path, exc)
        return None
    version = payload.get("schema_version", 1)
    if version < _VALIDATION_SCHEMA_VERSION:
        _log.warning(
            "%s uses schema_version=%d (current=%d). "
            "The 'sorting_phi' field may use the old std(φ) estimator; "
            "attempting to standardise it to the Folk-Ward graphic formula "
            "from the source CSVs.",
            Path(path).name, version, _VALIDATION_SCHEMA_VERSION,
        )
        try:
            _recompute_v1_sorting_phi(payload, path)
        except Exception as exc:  # defensive: never break report assembly
            payload["_v1_sorting_unrecomputable"] = True
            _log.warning("%s: σφ standardisation raised %s; keeping v1 value.",
                         Path(path).name, exc)
    return payload


# --- Inventory ---
# Detection logs are append-only: each run starts with a "[Run <ISO ts>]"
# block followed by [Inputs] / [Parameters] / [Outputs] sections, then a
# trailing "Cumulative elapsed (s): <float>" line that accumulates across resumes.
_RUN_HEAD_RE = re.compile(r"^\[Run\s+([^\]]+)\]\s*$", re.MULTILINE)
_CUMULATIVE_RE = re.compile(
    r"^Cumulative elapsed \(s\):\s*([0-9.]+)\s*$", re.MULTILINE)
# Generic "key: value" line. Keys may contain spaces (e.g. "tile grid",
# "tiles to process", "this run elapsed (s)") so we allow any non-colon run.
_PARAM_RE = re.compile(r"^\s*([^:\[\]]+?)\s*:\s*(.+?)\s*$")


def _kv_from_chunk(chunk: str) -> dict:
    """Pull key:value pairs from a section body (skip [...] headers)."""
    kvs = {}
    for line in chunk.splitlines():
        s = line.strip()
        if not s or s.startswith("[") or s.startswith("="):
            continue
        mm = _PARAM_RE.match(line)
        if mm:
            kvs[mm.group(1).strip()] = mm.group(2).strip()
    return kvs


def _parse_detection_log(path: Path) -> dict:
    """Parse a *_detection_log.txt file into a structured dict.

    The runners emit an append-only, multi-run log:

        [Run <ISO timestamp>]
          outcome             : complete
          this run start      : <ISO timestamp>
          this run elapsed (s): 2980.54
          kstart this run     : 0

        [Inputs]
          image path          : ...
        [Parameters]
          metric_cropsize     : 1.0 m
...
        [Outputs]
          output CSV          : ...
          records in CSV      : 11808

        Cumulative elapsed (s): 2980.54

    Every run's outcome / elapsed / kstart goes into `runs`, the most recent
    run's [Parameters] / [Inputs] / [Outputs] into the top-level keys, and
    the bottom "Cumulative elapsed" into `cumulative_s`.
    """
    out = {
        "path": path,
        "elapsed_s": 0.0,        # most recent run
        "cumulative_s": 0.0,     # total across resumes
        "parameters": {},
        "outputs": {},
        "inputs": {},
        "runs": [],
    }
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    # Cumulative elapsed: last occurrence wins.
    matches = list(_CUMULATIVE_RE.finditer(text))
    if matches:
        try:
            out["cumulative_s"] = float(matches[-1].group(1))
        except ValueError:
            pass

    run_starts = [(m.start(), m.group(1)) for m in _RUN_HEAD_RE.finditer(text)]
    section_names = ("Inputs", "Parameters", "Outputs")
    section_map = {"Inputs": "inputs", "Parameters": "parameters",
                   "Outputs": "outputs"}
    for i, (idx, ts) in enumerate(run_starts):
        end = run_starts[i + 1][0] if i + 1 < len(run_starts) else len(text)
        block = text[idx:end]
        head_end = min((block.find(f"\n[{n}]") for n in section_names
                        if block.find(f"\n[{n}]") >= 0), default=len(block))
        head = _kv_from_chunk(block[:head_end])
        try:
            elapsed_s = float(head.get("this run elapsed (s)", 0))
        except ValueError:
            elapsed_s = 0.0
        try:
            kstart = int(head.get("kstart this run", 0))
        except ValueError:
            kstart = 0
        out["runs"].append({
            "ts": ts,
            "outcome": head.get("outcome", "—"),
            "start_ts": head.get("this run start", ""),
            "elapsed_s": elapsed_s,
            "kstart": kstart,
        })
        # Sections overwrite, so the final iteration leaves the most recent run's.
        for name in section_names:
            sidx = block.find(f"[{name}]")
            if sidx < 0:
                continue
            tail = block[sidx + len(name) + 2:]
            next_hdr = re.search(r"\n\[", tail)
            chunk = tail[:next_hdr.start()] if next_hdr else tail
            out[section_map[name]] = _kv_from_chunk(chunk)

    if out["runs"]:
        out["elapsed_s"] = out["runs"][-1]["elapsed_s"]
        if not out["cumulative_s"]:
            out["cumulative_s"] = out["elapsed_s"]
    return out


def _file_hash_short(path: Path, n: int = 12) -> str:
    """SHA-256 short prefix for reproducibility traceability."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except (OSError, ValueError):
        return "—"


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _identify_image_kind(path: Path) -> str:
    """Heuristic: ortho (.tif/.tiff georeferenced) vs quadrat photograph
    (.jpg, .png, .heic ...). Returns ``modes.ORTHO``, ``modes.QUADRAT`` or
    ``"other"``."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return modes.ORTHO
    if ext in (".jpg", ".jpeg", ".png", ".heic", ".heif"):
        return modes.QUADRAT
    return "other"


# Parameter names the filename parser recognises, unioned at runtime with
# functions.units.PARAMETER_DISPLAY (the authoritative registry).
_EXTRA_PARAM_NAMES = (
    "quantile", "mode",  # legacy names not in PARAMETER_DISPLAY
)


def _known_parameter_names() -> tuple[str, ...]:
    """Union of PARAMETER_DISPLAY keys + legacy extras, longest-first so
    "folk_ward_sorting" wins over the bare "sorting" suffix."""
    from functions.units import PARAMETER_DISPLAY
    names = set(PARAMETER_DISPLAY.keys()) | set(_EXTRA_PARAM_NAMES)
    return tuple(sorted(names, key=len, reverse=True))


def _read_raster_sidecar(raster_path: Path) -> Optional[dict]:
    """The sidecar JSON metadata next to ``raster_path`` (``<raster>.tif.json``),
    or ``None``.

    The sidecar is the authoritative source of ``field`` / ``parameter`` /
    ``cellsize_m`` / ``n_bands``; rasters without one fall back to
    :func:`_parse_raster_meta_from_name`. Returns the same shape as that
    parser, enriched with the extra sidecar fields.
    """
    try:
        sidecar = raster_path.with_suffix(raster_path.suffix + ".json")
        if not sidecar.is_file():
            return None
        with open(sidecar, "r", encoding="utf-8") as fh:
            payload = _json.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    out = {
        "field":      payload.get("field"),
        "parameter":  payload.get("parameter"),
        "cellsize":   (float(payload["cellsize_m"])
                        if isinstance(payload.get("cellsize_m"),
                                       (int, float)) else None),
        # Pass-through fields the report does not consume yet.
        "source_image":      payload.get("source_image"),
        "source_csv":        payload.get("source_csv"),
        "rasterize_version": payload.get("rasterize_version"),
        "generated_at":      payload.get("generated_at"),
        "_from_sidecar":     True,
    }
    return out


def _parse_raster_meta_from_name(name: str) -> dict:
    """Extract ``(field, parameter, cellsize)`` from a rasterize filename::

        <project_stem>_<field>_<parameter>_cellsize=<n>m.tif

    The trailing ``_cellsize=<n>m.tif`` is stripped first, then the
    registered parameter names are matched by suffix, longest first; the
    remainder is the field. Digits are allowed throughout so ``D50`` parses.
    """
    out = {"field": None, "parameter": None, "cellsize": None}

    m = re.search(r"_cellsize=([0-9.]+)m", name)
    if m:
        try:
            out["cellsize"] = float(m.group(1).rstrip("."))
        except ValueError:
            pass

    stem = re.sub(r"_cellsize=[0-9.]+m.*$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"\.(tif|tiff)$", "", stem, flags=re.IGNORECASE)
    if not stem:
        return out

    def _field_only(prefix: str):
        """The field name at the end of ``<csv stem>_<field>``: a registered
        field when one ends the prefix, else what follows the last naming
        token (``_merged_``, ``_xprs_``, ``_ws<W>m_``, the quadrat and
        legacy suffixes). The CSV stem carries the origin, so the prefix
        cannot be taken as the field whole."""
        if not prefix:
            return None
        try:
            known = sorted((k for k, _v in _u.FIELD_UNIT_TABLE),
                           key=len, reverse=True)
        except Exception:
            known = []
        for k in known:
            if prefix == k or prefix.endswith("_" + k):
                return k
        m2 = re.search(r"(?:_merged|_xprs|_ws[0-9.]+m|_individual_clasts|"
                       r"_individual_clast_values)_(?P<f>[^/]+)$", prefix)
        if m2:
            return m2.group("f") or None
        return prefix

    # The parameter may be the entire stem (e.g. density, where the field is implied).
    for p in _known_parameter_names():
        if stem.endswith("_" + p):
            out["parameter"] = p
            out["field"] = _field_only(stem[: -(len(p) + 1)])
            break
        if stem == p:
            out["parameter"] = p
            out["field"] = None
            break
    else:
        # Unregistered parameter: take the trailing token so it stays visible.
        i = stem.rfind("_")
        if i >= 0:
            out["parameter"] = stem[i + 1:]
            out["field"] = _field_only(stem[:i])
        else:
            out["parameter"] = stem

    return out


def collect_project_inventory(project_root: Path) -> dict:
    """Walk the project and return a structured inventory: the single source
    the PDF builder reads from. Deterministic for a given disk state."""
    project_root = Path(project_root)
    inv = {
        "project_root": project_root,
        "project_name": project_root.name,
        "images": [],
        "vectors": [],
        "rasters": [],
        "figures": [],
        "maps": [],
        "logs": [],
        # Zonal CSVs under results/zonal/, with the mode (polygons / transects)
        # parsed from the filename.
        "zonal": [],
        "validation": {"truth_csvs": [], "images": [], "results": []},
        "weights": {"path": None, "hash": None, "size_bytes": 0},
        # Set when the project's folders match neither the canonical layout nor
        # the legacy one.
        "unrecognised_layout": "",
    }
    # The checked layout-aware resolver says whether a folder was found or
    # merely guessed, so an unrecognised tree is detected on the first read.
    from functions.layout import (
        resolve_project_subfolder as _rsf,
        resolve_project_subfolder_checked as _rsf_checked,
        describe_unrecognised_layout as _describe_unrecognised,
    )
    _images_res = _rsf_checked(project_root, "images")
    if _images_res.unrecognised:
        inv["unrecognised_layout"] = _describe_unrecognised(project_root)
    images_dir = _images_res.path
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if p.is_file():
                inv["images"].append({
                    "path": p,
                    "size_bytes": _safe_size(p),
                    "kind": _identify_image_kind(p),
                })

    # Detection vectors
    vectors_dir = _rsf(project_root, "vectors")
    if vectors_dir.is_dir():
        for p in sorted(vectors_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".csv":
                # Detection logs picked up here too
                if p.name.endswith("_detection_log.txt"):
                    inv["logs"].append({
                        "path": p,
                        **_parse_detection_log(p),
                    })
                continue
            if p.name.endswith(".run.csv"):
                continue  # partial-state, skipped
            if not _is_detection_csv(p):
                # Derived tables live here too; a polygon summary is not a pile of clasts.
                continue
            n_rows = 0
            try:
                with open(p) as fh:
                    n_rows = max(0, sum(1 for _ in fh) - 1)
            except OSError:
                pass
            # Merge-tool output is always ortho (the Merge tab is ortho-only) even
            # though it carries no window-size token.
            window_size = naming.parse_window_size(p.name)
            is_merged = naming.is_merged(p.name)
            # Recover the source-image stem so all variants of one image group.
            image_stem = naming.image_stem(p.name)
            run_stem = naming.run_stem(p.name)
            inv["vectors"].append({
                "path": p,
                "image_stem": image_stem,
                "run_stem": run_stem,
                "window_size": window_size,
                "is_merged": is_merged,
                "n_rows": n_rows,
            })

    # Rasters
    rasters_dir = _rsf(project_root, "rasters")
    if rasters_dir.is_dir():
        for p in sorted(rasters_dir.iterdir()):
            if p.suffix.lower() != ".tif":
                continue
            # Skip smoke-test artifacts from the Rasterize tab's verification utilities.
            if p.stem.startswith(("_smoke_", "_verify_")):
                continue
            # The sidecar JSON is authoritative; the filename parser is best-effort.
            sidecar_meta = _read_raster_sidecar(p)
            meta = sidecar_meta or _parse_raster_meta_from_name(p.name)
            n_bands = 1
            try:
                from osgeo import gdal
                ds = gdal.Open(str(p))
                if ds is not None:
                    n_bands = ds.RasterCount
                    ds = None
            except Exception:
                pass
            inv["rasters"].append({"path": p, "n_bands": n_bands, **meta})

    # Figures and publication maps
    for kind, key in (("figures", "figures"),
                       ("maps", "maps")):
        d = _rsf(project_root, kind)
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg",
                                                     ".pdf", ".svg"):
                inv[key].append({"path": p})

    # Zonal-stats artifacts: ``<base>__<vec>.polygons.csv`` (per-polygon
    # summary), ``<base>__<vec>.transects.csv`` (profile samples),
    # ``<stem>__<id>.png`` (quick-look profile), ``combined_*.png``. Each CSV
    # entry carries the parsed mode plus its associated artifacts.
    zonal_dir = _rsf(project_root, "zonal")
    combined_plots: list[Path] = []
    # Profile figures and their summary tables, paired by stem:
    # "<stem>__profile.png" alongside "<stem>__summary.csv".
    profile_figures: list[dict] = []
    if zonal_dir.is_dir():
        for p in sorted(zonal_dir.iterdir()):
            if p.is_file() and p.name.lower().endswith("__profile.png"):
                stem = p.name[: -len("__profile.png")]
                summary = p.parent / f"{stem}__summary.csv"
                profile_figures.append({
                    "path": p,
                    "summary": summary if summary.exists() else None,
                })
        # Combined-plot PNGs become a gallery at the end of the zonal section.
        for p in sorted(zonal_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".png":
                if p.name.lower().startswith("combined_"):
                    combined_plots.append(p)
        for p in sorted(zonal_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() != ".csv":
                continue
            # Profile tables are rendered beside their figure, not as sections.
            if p.name.lower().endswith(("__summary.csv", "__change.csv",
                                        "__binned.csv")):
                continue
            mode = "unknown"
            stem_lc = p.stem.lower()
            if stem_lc.endswith(".polygons"):
                mode = "polygons"
            elif stem_lc.endswith(".transects"):
                mode = "transects"
            # The vector layer is the token after the "__" separator, before the ".<mode>" suffix.
            vector = None
            # The summarised field and its unit travel inside the CSV as
            # provenance columns; older files yield None here.
            field = None
            unit = None
            try:
                with open(p, encoding="utf-8") as fh:
                    hdr = fh.readline().rstrip("\r\n").split(",")
                    if "field" in hdr:
                        vals = fh.readline().rstrip("\r\n").split(",")
                        if len(vals) == len(hdr):
                            field = vals[hdr.index("field")].strip() or None
                            if "unit" in hdr:
                                unit = vals[hdr.index("unit")].strip() or None
            except (OSError, UnicodeDecodeError):
                pass

            # Both grammars: "<base>_zones=<set>[_field=<f>]" and the legacy
            # "<base>__<zones stem>[__<field tag>]".
            parsed = naming.parse_zonal_name(p.name) or {}
            raster_stem = parsed.get("base_stem") or p.stem
            vector = parsed.get("zone_set")
            if not field and parsed.get("field_tag"):
                field = parsed["field_tag"]
            # CSVs without provenance columns: the raster stem still names the
            # field, so recover it once here.
            if not field:
                m = re.match(r"(.*_cellsize=[\d.]+m)", raster_stem)
                if m:
                    field = _parse_raster_meta_from_name(
                        m.group(1) + ".tif").get("field") or None
            # Which source image this came from. Two runs over one zone set
            # differ only here, so a heading needs it to stay distinct.
            source_image = naming.run_stem(
                raster_stem.split("_merged_", 1)[0]) or None
            n_rows = 0
            try:
                with open(p) as fh:
                    n_rows = max(0, sum(1 for _ in fh) - 1)
            except OSError:
                pass
            inv["zonal"].append({
                "path": p,
                "mode": mode,
                "vector": vector,
                "field": field,
                "unit": unit,
                "source_image": source_image,
                "n_rows": n_rows,
            })
    # Stash combined-plot list on the inventory under a sentinel key so
    # the renderer can find it without walking the dir again.
    inv["zonal_combined_plots"] = combined_plots
    inv["zonal_profile_figures"] = profile_figures

    # Validation
    val_dir = project_root / "validation"
    if val_dir.is_dir():
        for p in sorted(val_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() == ".csv":
                inv["validation"]["truth_csvs"].append({"path": p})
        vimg = val_dir / "images"
        if vimg.is_dir():
            for p in sorted(vimg.iterdir()):
                if p.is_file() and _identify_image_kind(p) != "other":
                    inv["validation"]["images"].append({"path": p})
        # Validation results live under "results/"; older projects call it
        # "reports/". Both are walked, de-duplicated by filename with the
        # canonical folder winning, and the mixed situation is noted in the log.
        seen_names = set()
        for sub_name in ("results", "reports"):
            vrep = val_dir / sub_name
            if not vrep.is_dir():
                continue
            # rglob("*") picks up both the flat layout and the per-detection-CSV
            # subfolder layout (validation/results/<detect_stem>/).
            for p in sorted(vrep.rglob("*")):
                if not p.is_file():
                    continue
                if p.name in seen_names:
                    continue
                seen_names.add(p.name)
                inv["validation"]["results"].append({"path": p})
        # Surface the mixed-folder situation in the inventory so
        # downstream consumers can warn the user.
        if (val_dir / "results").is_dir() and (val_dir / "reports").is_dir():
            inv["validation"]["_legacy_reports_present"] = True

    # Mask R-CNN weights
    weights = (project_root.parent.parent / "model_weights"
               / "mask_rcnn_clasts.h5")
    if weights.is_file():
        inv["weights"] = {
            "path": weights,
            "hash": _file_hash_short(weights),
            "size_bytes": _safe_size(weights),
        }

    # Precision of this project's numbers, resolved once per build and consumed
    # by every table that prints a measured magnitude.
    try:
        inv["uncertainty"] = _resolve_uncertainties(inv)
    except Exception:
        inv["uncertainty"] = {}
    return inv


# --- Aggregate stats ---
_PHI_REF_M = 0.001  # phi = -log2(D / 1 mm)


def _phi(d_m):
    """Convert metres to the phi (φ) scale used by Folk-Ward statistics."""
    arr = np.asarray(d_m, dtype=float)
    arr = arr[arr > 0]
    return -np.log2(arr / _PHI_REF_M)


def _quantiles(arr, qs):
    if len(arr) == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.nanquantile(arr, q)) for q in qs}


def _is_detection_csv(path) -> bool:
    """True when this CSV is a per-clast detection table, decided on the
    header: ``output_results/vectors/`` also collects derived tables such as
    zonal polygon summaries, whose rows are not clasts."""
    try:
        import csv as _csv
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(_csv.reader(fh))
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    cols = {c.strip() for c in header}
    if "polygon_id" in cols or "transect_id" in cols:
        return False
    # A detection row identifies one clast at a position.
    return bool({"x", "y"} <= cols and (cols & {
        "Clast_length", "Ellipse_major_axis", "Equivalent_diameter"}))


def _canonical_vectors(inventory: dict) -> list[dict]:
    """The deduplicated detection-CSV list: one canonical CSV per image stem.

    When a stem has a merged CSV, that is the canonical population;
    otherwise every per-window CSV of the stem is kept. Every
    population-level summary must go through this so nothing double-counts.
    """
    by_stem = {}
    for entry in inventory["vectors"]:
        by_stem.setdefault(entry["image_stem"], []).append(entry)
    canonical = []
    for entries in by_stem.values():
        merged = [e for e in entries if e.get("is_merged")]
        if merged:
            # One merge, one population: "<stem>_merged.csv" and
            # "<stem>_merged_individual_clast_values.csv" hold the same clasts.
            # Prefer the plainest name; ties break on path for reproducibility.
            # `path` may arrive as a str from hand-built inventories.
            canonical.append(min(
                merged,
                key=lambda e: (len(Path(e["path"]).name), str(e["path"]))))
        else:
            canonical.extend(entries)
    return canonical


def _csv_row_count(path) -> int:
    """Count rows in ``path`` (excluding header). Returns 0 on any I/O error."""
    try:
        with open(path) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def compute_aggregate_stats(inventory: dict) -> dict:
    """Compute headline project statistics: total clast count, D-percentiles,
    Folk-Ward sorting/skewness/kurtosis, total cumulative runtime, UAV
    survey extent, and aggregated validation metrics (if any)."""
    out = {
        "n_images": len(inventory["images"]),
        "n_vectors": len(inventory["vectors"]),
        "n_clasts_total": 0,
        "per_image_counts": {},
        # Counts by full CSV stem, unfiltered: one bar per file in the summary chart.
        "per_csv_counts": {},
        "clast_length_m": {},
        "phi": {},
        "folk_ward": {},
        "runtime_total_s": 0.0,
        "extent": None,
        "validation_aggregate": None,
    }
    # Totals must not double-count: the merged file is the deduplicated union
    # of the per-window runs, so only canonical CSVs feed them.
    canonical_vectors = _canonical_vectors(inventory)

    # First pass: one bar per file, even when merged + per-window CSVs coexist.
    for entry in inventory["vectors"]:
        n = _csv_row_count(entry["path"])
        if entry.get("is_merged"):
            key = f"{entry['image_stem']} (merged)"
        elif entry.get("window_size") is not None:
            key = f"{entry['image_stem']} @ {entry['window_size']:g} m"
        else:
            key = entry["image_stem"]
        if key in out["per_csv_counts"]:
            i = 2
            while f"{key} (#{i})" in out["per_csv_counts"]:
                i += 1
            key = f"{key} (#{i})"
        out["per_csv_counts"][key] = n

    # Second pass: canonical CSVs only. The spatial extent uses only ortho
    # vectors (quadrat CSVs carry pixel coordinates), and lengths stay
    # partitioned by detection kind so a mixed project reports each
    # population separately rather than a chimeric distribution.
    all_lengths_by_kind = {modes.ORTHO: [], modes.QUADRAT: []}
    all_x = []
    all_y = []
    for entry in canonical_vectors:
        try:
            df = pd.read_csv(entry["path"], usecols=["x", "y", "Clast_length"])
        except (OSError, ValueError):
            continue
        n = len(df)
        out["n_clasts_total"] += n
        if entry.get("is_merged"):
            key = f"{entry['image_stem']} (merged)"
        elif entry.get("window_size") is not None:
            key = f"{entry['image_stem']} @ {entry['window_size']:g} m"
        else:
            key = entry["image_stem"]
        out["per_image_counts"][key] = n
        out["clast_length_m"]["count"] = out["clast_length_m"].get("count", 0) + n
        is_uav = bool(entry.get("is_merged")
                      or entry.get("window_size") is not None)
        kind = modes.ORTHO if is_uav else modes.QUADRAT
        all_lengths_by_kind[kind].append(df["Clast_length"].dropna().values)
        if is_uav:
            all_x.append(df["x"].dropna().values)
            all_y.append(df["y"].dropna().values)

    def _length_stats(lengths_list):
        """Compute D-percentiles, Folk-Ward σ_φ / Sk_φ / K_G for one
        kind. Returns ``(length_summary, folk_ward, phi_q)`` or
        ``(None, None, None)`` if no positive lengths.
        """
        if not lengths_list:
            return None, None, None
        L = np.concatenate(lengths_list)
        L = L[L > 0]
        if not len(L):
            return None, None, None
        qs = (0.05, 0.16, 0.50, 0.84, 0.95)
        d_q = _quantiles(L, qs)
        length_summary = {
            **{f"D{int(q*100)}": d_q[q] for q in qs},
            "mean":   float(np.mean(L)),
            "median": float(np.median(L)),
            "std":    float(np.std(L)),
            "count":  int(len(L)),
        }
        phi_vals = _phi(L)
        folk_ward = None
        phi_q_out = {}
        if len(phi_vals):
            phi_q = _quantiles(phi_vals,
                               (0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95))
            phi_q_out = {f"phi_{int(q*100):02d}": phi_q[q]
                         for q in (0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95)}
            p5, p16, p25, p50, p75, p84, p95 = (
                phi_q[0.05], phi_q[0.16], phi_q[0.25], phi_q[0.50],
                phi_q[0.75], phi_q[0.84], phi_q[0.95])
            sigma = _u.folk_ward_sorting_phi(p5, p16, p84, p95)
            skew = _u.folk_ward_skewness_phi(p5, p16, p50, p84, p95)
            kurt = _u.folk_ward_kurtosis_phi(p5, p25, p75, p95)
            folk_ward = {
                "sorting_phi":     float(sigma),
                "skewness_phi":    float(skew),
                "kurtosis_phi":    float(kurt),
                "sorting_verbal":  _classify_sorting(sigma),
                "skewness_verbal": _classify_skewness(skew),
                "kurtosis_verbal": _classify_kurtosis(kurt),
            }
        return length_summary, folk_ward, phi_q_out

    # Per-kind blocks, always populated (empty when the kind has no data).
    by_kind = {}
    for kind in modes.MODES:
        summary, fw, phi_q = _length_stats(all_lengths_by_kind[kind])
        by_kind[kind] = {
            "clast_length_m": summary or {},
            "folk_ward":      fw or {},
            "phi":            phi_q or {},
            "n_clasts":       int(summary["count"]) if summary else 0,
        }
    out["clast_length_m_by_kind"] = by_kind

    # Pooled blocks keep the output shape; a mixed project hides them on the
    # cover in favour of the per-kind breakdown.
    pooled_lengths = (all_lengths_by_kind[modes.ORTHO]
                       + all_lengths_by_kind[modes.QUADRAT])
    pooled_summary, pooled_fw, pooled_phi = _length_stats(pooled_lengths)
    if pooled_summary:
        out["clast_length_m"] = pooled_summary
        if pooled_fw:
            out["folk_ward"] = pooled_fw
        out["phi"] = pooled_phi
    if all_x and all_y:
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)
        if len(X) and len(Y):
            out["extent"] = {
                "xmin": float(np.min(X)), "xmax": float(np.max(X)),
                "ymin": float(np.min(Y)), "ymax": float(np.max(Y)),
            }

    out["runtime_total_s"] = sum(log["cumulative_s"]
                                  for log in inventory["logs"])

    # Validation: the persisted JSONs written by the Validate tab, else re-pair
    # the truth CSVs.
    out["validation_aggregate"] = _aggregate_validation(inventory)
    return out


def _aggregate_validation(inventory: dict):
    """Compute per-project validation metrics, grouped by field.

    Returns a dict shaped like::

        {
          "n_comparisons":   3,                 # UNIQUE truth/detect pairings
          "n_field_runs":    12,                # total JSONs (= comparisons x fields)
          "by_field": {
              "Clast_length": { "n_comparisons": 3, "recall": ..., "rmse": ..., ... },
              ...
          },
          "overall": { "recall": ..., "precision": ..., "f1": ..., "r2": ...,
                       "ks_p_value": ... }
        }

    Metrics whose units depend on the field (RMSE, bias) stay per-field.
    Recall / precision / F1 are field-independent (the pairing is spatial),
    so the overall block averages them across unique pairings, not JSONs.

    Reading priority: ``validation/results/*.validation.json``, else
    re-pairing the ``validation/*.csv`` truth files.
    """
    reports = inventory.get("validation", {}).get("results", [])
    json_reports = [r for r in reports
                    if Path(r["path"]).name.endswith(".validation.json")]

    # Per-field accumulator: field -> {metric: [values across comparisons]}.
    by_field = {}
    def _bucket(field):
        return by_field.setdefault(field, {
            "n_comparisons": 0, "recall": [], "precision": [], "f1": [],
            "r2": [], "rmse": [], "bias": [],
            "D50_relerr": [], "D84_relerr": [], "ks_p_value": [],
        })

    # A unique pairing is one (truth_csv, detect_csv, tolerance, truth_gsd,
    # detect_gsd) tuple; recall / precision / F1 are collected once per pairing.
    def _pairing_key(payload: dict) -> tuple:
        return (
            payload.get("truth_csv"),
            payload.get("detect_csv"),
            payload.get("tolerance_m"),
            payload.get("truth_gsd_m_per_px"),
            payload.get("detect_gsd_m_per_px"),
        )
    overall_recall = {}    # pairing_key -> recall
    overall_prec   = {}
    overall_f1     = {}

    # Detection kinds the saved pairings covered, so the summary can say
    # "validated against the quadrat photographs only".
    detect_kinds = set()
    def _detect_kind(detect_path: str) -> str:
        if not detect_path:
            return "unknown"
        name = Path(detect_path).name
        if naming.is_merged(name) or naming.parse_window_size(name) is not None:
            return modes.ORTHO
        return modes.QUADRAT

    n_field_runs = 0       # total JSONs read (= field-runs)
    if json_reports:
        for r in json_reports:
            payload = _load_validation_json(r["path"])
            if payload is None:
                continue
            field = payload.get("field") or "Clast_length"
            bucket = _bucket(field)
            metrics = payload.get("metrics") or {}
            bucket["n_comparisons"] += 1
            n_field_runs += 1
            for k in ("recall", "precision", "f1", "r2", "rmse", "bias",
                      "D50_relerr", "D84_relerr"):
                v = metrics.get(k)
                if v is not None:
                    bucket[k].append(v)
            dist = payload.get("distribution") or {}
            if dist.get("ks_p_value") is not None:
                bucket["ks_p_value"].append(dist["ks_p_value"])
            # Pairing-scoped metrics are identical across fields of one pairing.
            pk = _pairing_key(payload)
            if metrics.get("recall") is not None:
                overall_recall[pk] = metrics["recall"]
            if metrics.get("precision") is not None:
                overall_prec[pk] = metrics["precision"]
            if metrics.get("f1") is not None:
                overall_f1[pk] = metrics["f1"]
            detect_kinds.add(_detect_kind(payload.get("detect_csv") or ""))

    # Fallback: re-pair truth/detection CSVs when no JSONs exist.
    if n_field_runs == 0:
        val_csvs = inventory["validation"]["truth_csvs"]
        if val_csvs:
            from functions import validation as _v
            for entry in val_csvs:
                truth_path = entry["path"]
                stem = truth_path.stem.replace("_truth", "")
                detect = next((v for v in inventory["vectors"]
                               if v["image_stem"].startswith(stem)
                               or stem.startswith(v["image_stem"])), None)
                if detect is None:
                    continue
                try:
                    tdf = pd.read_csv(truth_path)
                    ddf = pd.read_csv(detect["path"])
                    pair = _v.pair_csvs(tdf, ddf, x_col="x", y_col="y",
                                         tolerance=None, size_col=None)
                    m = _v.detection_metrics(pair)
                    bucket = _bucket("Clast_length")
                    bucket["n_comparisons"] += 1
                    n_field_runs += 1
                    bucket["recall"].append(m["recall"])
                    bucket["precision"].append(m["precision"])
                    bucket["f1"].append(m["f1"])
                    pk = (str(truth_path), str(detect["path"]), None, None, None)
                    if m["recall"] is not None: overall_recall[pk] = m["recall"]
                    if m["precision"] is not None: overall_prec[pk] = m["precision"]
                    if m["f1"] is not None: overall_f1[pk] = m["f1"]
                    detect_kinds.add(_detect_kind(str(detect["path"])))
                    if pair["matched_pairs"]:
                        field = "Clast_length"
                        if field in tdf.columns and field in ddf.columns:
                            ti = [p[0] for p in pair["matched_pairs"]]
                            di = [p[1] for p in pair["matched_pairs"]]
                            paired = _v.compute_paired_stats(
                                tdf[field].iloc[ti].to_numpy(),
                                ddf[field].iloc[di].to_numpy())
                            if paired:
                                bucket["r2"].append(paired.get("r_squared", float("nan")))
                                bucket["rmse"].append(paired.get("rmse", float("nan")))
                except Exception:
                    continue

    if not n_field_runs:
        return None

    def _mean(xs):
        xs = [v for v in xs if v is not None and np.isfinite(v)]
        return float(np.mean(xs)) if xs else None

    # Per-field means
    per_field_out = {}
    for field, bucket in by_field.items():
        per_field_out[field] = {
            "n_comparisons": bucket["n_comparisons"],
            "recall":     _mean(bucket["recall"]),
            "precision":  _mean(bucket["precision"]),
            "f1":         _mean(bucket["f1"]),
            "r2":         _mean(bucket["r2"]),
            "rmse":       _mean(bucket["rmse"]),
            "bias":       _mean(bucket["bias"]),
            "D50_relerr": _mean(bucket["D50_relerr"]),
            "D84_relerr": _mean(bucket["D84_relerr"]),
            "ks_p_value": _mean(bucket["ks_p_value"]),
        }

    # The overall block is averaged across unique pairings; r2 and ks_p_value
    # are field-dependent and stay per-field.
    overall = {
        "recall":     _mean(list(overall_recall.values())),
        "precision":  _mean(list(overall_prec.values())),
        "f1":         _mean(list(overall_f1.values())),
        "r2":         None,
        "ks_p_value": None,
    }

    return {
        # Headline count: distinct pairings, whatever the number of fields each
        # was validated against.
        "n_comparisons":  len(overall_recall) or len(overall_prec)
                           or len(overall_f1) or 0,
        # Secondary count: total JSON files (field-runs).
        "n_field_runs":   n_field_runs,
        "by_field":       per_field_out,
        "overall":        overall,
        "detect_kinds":   sorted(detect_kinds),
    }


# Field units and parameter display resolution live in functions/units.py,
# shared with the GUI.
from functions.units import field_unit_and_factor as _field_unit_and_factor




def _classify_sorting(sigma):
    if not np.isfinite(sigma):
        return "—"
    if sigma < 0.35: return "very well sorted"
    if sigma < 0.50: return "well sorted"
    if sigma < 0.70: return "moderately well sorted"
    if sigma < 1.00: return "moderately sorted"
    if sigma < 2.00: return "poorly sorted"
    if sigma < 4.00: return "very poorly sorted"
    return "extremely poorly sorted"


def _classify_skewness(sk):
    if not np.isfinite(sk):
        return "—"
    if sk < -0.30: return "very coarse skewed"
    if sk < -0.10: return "coarse skewed"
    if sk < +0.10: return "near-symmetrical"
    if sk < +0.30: return "fine skewed"
    return "very fine skewed"


def _classify_kurtosis(k):
    if not np.isfinite(k):
        return "—"
    if k < 0.67: return "very platykurtic"
    if k < 0.90: return "platykurtic"
    if k < 1.11: return "mesokurtic"
    if k < 1.50: return "leptokurtic"
    if k < 3.00: return "very leptokurtic"
    return "extremely leptokurtic"


# --- PDF helpers ---
def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _format_elapsed(seconds):
    if not seconds or seconds <= 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _figure_image(fig, dpi=100, max_width_in=6.5, max_height_in=8.0,
                  format="jpg", jpeg_quality=80):
    """Render a matplotlib Figure to a ReportLab Image flowable.

    JPEG at modest dpi keeps a many-raster report to a few megabytes and
    speeds the second multiBuild pass; callers wanting higher fidelity pass
    ``format="png"`` or a higher ``dpi``.
    """
    from reportlab.platypus import Image as RImage
    from reportlab.lib.utils import ImageReader
    import matplotlib
    matplotlib.use("Agg")
    buf = io.BytesIO()
    fmt = format.lower()
    savefig_kwargs = {"format": fmt, "dpi": dpi, "bbox_inches": "tight"}
    if fmt in ("jpg", "jpeg"):
        # matplotlib axes are RGBA by default, which flattens to black in JPEG.
        savefig_kwargs["facecolor"] = "white"
        savefig_kwargs["pil_kwargs"] = {"quality": int(jpeg_quality),
                                        "optimize": True,
                                        "progressive": True}
    fig.savefig(buf, **savefig_kwargs)
    buf.seek(0)
    # bbox_inches="tight" trims, so read the pixel size back instead of fig.get_size_inches().
    try:
        ir = ImageReader(buf)
        px_w, px_h = ir.getSize()
        buf.seek(0)
        aspect = px_h / px_w if px_w else 0.75
    except Exception:
        aspect = 0.75  # safe fallback
    # Fit to max_width first, then clamp height if it would overflow the page.
    w_pt = max_width_in * 72.0
    h_pt = w_pt * aspect
    max_h_pt = max_height_in * 72.0
    if h_pt > max_h_pt:
        h_pt = max_h_pt
        w_pt = h_pt / aspect if aspect else w_pt
    return RImage(buf, width=w_pt, height=h_pt)


# --- PDF builder: section helpers ---
def _styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.colors import HexColor
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="CoverTitle", parent=ss["Title"], fontSize=28, leading=34,
        alignment=TA_CENTER, textColor=HexColor("#1f3556"),
        spaceAfter=20))
    ss.add(ParagraphStyle(
        name="CoverMeta", parent=ss["BodyText"], fontSize=12, leading=16,
        alignment=TA_CENTER, textColor=HexColor("#5a6878"),
        spaceAfter=6))
    ss.add(ParagraphStyle(
        name="H1Custom", parent=ss["Heading1"], fontSize=18, leading=22,
        textColor=HexColor("#1f3556"), spaceBefore=14, spaceAfter=8))
    ss.add(ParagraphStyle(
        name="H2Custom", parent=ss["Heading2"], fontSize=14, leading=18,
        textColor=HexColor("#1f3556"), spaceBefore=10, spaceAfter=6))
    ss.add(ParagraphStyle(
        name="Body", parent=ss["BodyText"], fontSize=10, leading=13,
        alignment=TA_LEFT, spaceAfter=6))
    ss.add(ParagraphStyle(
        name="Mono", parent=ss["Code"], fontSize=8, leading=10,
        textColor=HexColor("#222222")))
    ss.add(ParagraphStyle(
        name="Caption", parent=ss["BodyText"], fontSize=9, leading=11,
        textColor=HexColor("#6b7280"), spaceBefore=2, spaceAfter=10))
    # ToC level styles, picked up by reportlab's TableOfContents flow.
    ss.add(ParagraphStyle(
        name="TOCH1", parent=ss["BodyText"], fontSize=11, leading=15,
        textColor=HexColor("#1f3556"), leftIndent=4, spaceBefore=4,
        spaceAfter=2))
    ss.add(ParagraphStyle(
        name="TOCH2", parent=ss["BodyText"], fontSize=9, leading=12,
        textColor=HexColor("#5a6878"), leftIndent=20, spaceBefore=1,
        spaceAfter=1))
    ss.add(ParagraphStyle(
        name="TOCFigure", parent=ss["BodyText"], fontSize=9, leading=12,
        leftIndent=4, spaceBefore=1, spaceAfter=1,
        textColor=HexColor("#5a6878")))
    ss.add(ParagraphStyle(
        name="TOCTable", parent=ss["BodyText"], fontSize=9, leading=12,
        leftIndent=4, spaceBefore=1, spaceAfter=1,
        textColor=HexColor("#5a6878")))
    ss.add(ParagraphStyle(
        name="TOCTitle", parent=ss["Heading1"], fontSize=16, leading=20,
        textColor=HexColor("#1f3556"), spaceAfter=10))
    # FigureCaption sits below its figure, TableCaption above its table; the
    # style name lets the doc template's afterFlowable route them to the
    # List of Figures / List of Tables.
    ss.add(ParagraphStyle(
        name="FigureCaption", parent=ss["BodyText"], fontSize=9,
        leading=11, textColor=HexColor("#374151"),
        spaceBefore=4, spaceAfter=10, leftIndent=4, rightIndent=4))
    ss.add(ParagraphStyle(
        name="TableCaption", parent=ss["BodyText"], fontSize=9,
        leading=11, textColor=HexColor("#374151"),
        spaceBefore=8, spaceAfter=4, leftIndent=4, rightIndent=4))
    return ss


# --- Numbered figure / table captions ---
# Counters live on the stylesheet for one build. Numbering is decided at
# story-build time, not in afterFlowable, so it is stable across multiBuild's
# two passes.

def _ensure_counters(ss):
    """Initialise per-build figure / table counters on ``ss`` if needed."""
    if not hasattr(ss, "_counters"):
        # A class so AttributeError is loud for a counter that was not set up.
        ss._counters = type("_Counters", (), {"fig": 0, "tbl": 0})()
    return ss._counters


# Embedding fidelity per preset. A preset never adds or drops content (the
# include_* flags do that), so two builds differ only in resolution. Figures
# are re-rendered rather than copied from the Map tab's exports, whose
# colorbar unit and attribution are baked in.
REPORT_PRESETS = {
    "full": {
        "label": "Full archive",
        "dpi": 300,
        "quality": 95,
    },
    "shareable": {
        "label": "Shareable",
        "dpi": 90,
        "quality": 70,
    },
}


def _provenance(source: str) -> str:
    """Trailing provenance fragment for a caption: the long source filename,
    demoted to the end in a smaller, greyer font."""
    if not source:
        return ""
    return (f" <font size=6 color='#777'>Source: "
            f"<code>{source}</code></font>")


def _fig_caption(text: str, ss, source: str = ""):
    """Numbered "Figure N. ..." caption Paragraph. Place it directly after the
    figure flowable so the caption sits beneath the image and the ToC entry
    lands on the same page. ``source`` is appended as a provenance line."""
    from reportlab.platypus import Paragraph
    c = _ensure_counters(ss)
    c.fig += 1
    return Paragraph(f"<b>Figure {c.fig}.</b> {text}{_provenance(source)}",
                     ss["FigureCaption"])


def _tbl_caption(text: str, ss, source: str = ""):
    """Numbered "Table N. ..." caption Paragraph. Place it directly before the
    table flowable. ``source`` is appended as a provenance line."""
    from reportlab.platypus import Paragraph
    c = _ensure_counters(ss)
    c.tbl += 1
    return Paragraph(f"<b>Table {c.tbl}.</b> {text}{_provenance(source)}",
                     ss["TableCaption"])


def _table(data, col_widths=None, header_bg="#e8eef7", zebra=True,
           font_size=8, wrap_threshold=25, ss=None,
           total_width=7.0 * 72):
    """Build a styled ReportLab Table from a 2D list (first row = header).

    ReportLab's ``Table`` does not wrap bare-string cells, so any string
    longer than ``wrap_threshold`` (or carrying inline markup) is wrapped in
    a ``Paragraph`` styled to match the table font; flowables pass through.

    When ``col_widths`` is None, columns are equal-width summing to
    ``total_width`` rather than ReportLab's auto-layout, which lets wide
    CSV-derived tables overflow the printable area. The default 504 pt is
    the printable width of A4 with the template's 36 pt margins.
    """
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle

    if ss is not None and "Body" in ss:
        body_style = ss["Body"]
        cell_style = ParagraphStyle(
            "_TableCell", parent=body_style,
            fontSize=font_size, leading=font_size + 2,
            spaceBefore=0, spaceAfter=0)
    else:
        cell_style = ParagraphStyle(
            "_TableCell", fontName="Helvetica",
            fontSize=font_size, leading=font_size + 2,
            textColor=HexColor("#222222"),
            spaceBefore=0, spaceAfter=0)

    def _maybe_wrap(cell):
        if isinstance(cell, str):
            if len(cell) > wrap_threshold or "<" in cell:
                return Paragraph(cell, cell_style)
            return cell
        return cell  # Already a flowable (Paragraph, Table, Image, …)

    wrapped = [[_maybe_wrap(c) for c in row] for row in data]
    if col_widths is None and wrapped and len(wrapped[0]) > 0:
        n_cols = len(wrapped[0])
        col_widths = [total_width / n_cols] * n_cols
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#1f3556")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r),
                              HexColor("#f5f7fb")))
    t.setStyle(TableStyle(style))
    return t


# Bump when the drawing changes: the cached map is keyed on it.
_OVERVIEW_MAP_VERSION = "overview-map-v2"


def _build_project_overview_map(inv, out_path):
    """Draw every UAV ortho-image in the project on a basemap, save it to
    ``out_path``, and return ``out_path`` on success or ``None`` on failure.

    Best-effort: any failure (missing GDAL, no projected CRS, no UAV images,
    contextily import error) returns ``None`` and the cover falls back.

    Cached: regenerated only when a source ortho is newer than the PNG, the
    set of orthos changes, or the renderer version changes, so contextily
    tiles are not re-fetched on every build.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    try:
        from osgeo import gdal, osr
    except Exception:
        return None

    uav_images = [i for i in inv.get("images", [])
                  if modes.is_ortho(i.get("kind"))]
    if len(uav_images) < 1:
        return None

    out_path = Path(out_path)
    try:
        if out_path.is_file():
            cached_mtime = out_path.stat().st_mtime
            stale = False
            for img in uav_images:
                try:
                    if img["path"].stat().st_mtime > cached_mtime:
                        stale = True
                        break
                except OSError:
                    stale = True
                    break
            # The manifest of source images catches "a new ortho was added".
            manifest_path = out_path.with_suffix(".manifest.txt")
            # The renderer's version is part of the key, so a change to the
            # drawing code regenerates a cached map.
            current_manifest = _OVERVIEW_MAP_VERSION + "\n" + "\n".join(
                sorted(str(i["path"]) for i in uav_images))
            if manifest_path.is_file():
                try:
                    if manifest_path.read_text(
                            encoding="utf-8") != current_manifest:
                        stale = True
                except OSError:
                    stale = True
            else:
                # No manifest yet means we should refresh, then write one.
                stale = True
            if not stale:
                return out_path
    except Exception:
        pass

    # Collect each ortho's extent + projection.
    extents = []          # list of (xmin, xmax, ymin, ymax, stem)
    crs_wkt = None
    epsg = None
    for img in uav_images:
        try:
            ds = gdal.Open(str(img["path"]))
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            nx, ny = ds.RasterXSize, ds.RasterYSize
            xmin = gt[0]
            xmax = gt[0] + nx * gt[1]
            ymax = gt[3]
            ymin = gt[3] + ny * gt[5]
            if crs_wkt is None:
                crs_wkt = ds.GetProjection()
                if crs_wkt:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(crs_wkt)
                    try:
                        epsg = int(srs.GetAttrValue("AUTHORITY", 1))
                    except (TypeError, ValueError):
                        epsg = None
            extents.append((xmin, xmax, ymin, ymax, img["path"].stem))
            ds = None
        except Exception:
            continue

    if not extents:
        return None

    # Union extent, plus a 5 % margin on each side so the boxes don't
    # touch the figure edges.
    xmins = [e[0] for e in extents]
    xmaxs = [e[1] for e in extents]
    ymins = [e[2] for e in extents]
    ymaxs = [e[3] for e in extents]
    x_lo, x_hi = min(xmins), max(xmaxs)
    y_lo, y_hi = min(ymins), max(ymaxs)
    pad_x = (x_hi - x_lo) * 0.10 or 1.0
    pad_y = (y_hi - y_lo) * 0.10 or 1.0
    x_lo -= pad_x; x_hi += pad_x
    y_lo -= pad_y; y_hi += pad_y

    # For a single small quadrat ortho the auto-zoom and tile warp collapse to
    # a ~1-px image and a blank basemap; enforce a minimum view span.
    _MIN_SPAN_M = 50.0
    cx_mid = (x_lo + x_hi) / 2.0
    cy_mid = (y_lo + y_hi) / 2.0
    if (x_hi - x_lo) < _MIN_SPAN_M:
        x_lo, x_hi = cx_mid - _MIN_SPAN_M / 2.0, cx_mid + _MIN_SPAN_M / 2.0
    if (y_hi - y_lo) < _MIN_SPAN_M:
        y_lo, y_hi = cy_mid - _MIN_SPAN_M / 2.0, cy_mid + _MIN_SPAN_M / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")

    # Contextily basemap, skipped when it is not installed or the CRS is not
    # reachable. The global socket timeout is capped for the call so a stalled
    # tile fetch falls through to a basemap-less map rather than freezing the build.
    basemap_added = False
    if epsg:
        import socket as _socket
        _orig_timeout = _socket.getdefaulttimeout()
        try:
            _socket.setdefaulttimeout(8.0)
            import contextily as cx
            # Esri serves "Map data not yet available" tiles beyond zoom
            # 18 over most of the coast; a footprint a few tens of metres
            # wide would otherwise be drawn on that placeholder.
            _span = max(x_hi - x_lo, y_hi - y_lo)
            cx.add_basemap(ax, crs=f"EPSG:{epsg}",
                            source=cx.providers.Esri.WorldImagery,
                            zoom=18 if _span < 800 else "auto",
                            attribution="")
            basemap_added = True
        except Exception as ex:
            # Log to stdout so the worker's capture_stdout_to_log
            # surfaces the skip reason in the report tab's log.
            print(f"[overview-map] basemap skipped: "
                  f"{type(ex).__name__}: {ex}")
        finally:
            try:
                _socket.setdefaulttimeout(_orig_timeout)
            except Exception:
                pass

    # Each ortho's actual pixels, with fully-dark / fully-bright margins
    # transparent so only the real footprint shows; a coloured edge and a
    # label keep several orthos distinguishable.
    from functions.map_export import _read_ortho_image
    cmap = plt.get_cmap("tab10")
    for i, (xmin, xmax, ymin, ymax, stem) in enumerate(extents):
        color = cmap(i % 10)
        try:
            rgba, ext = _read_ortho_image(
                str([img["path"] for img in uav_images
                     if img["path"].stem == stem][0]),
                transparent_edges=True)
            ax.imshow(rgba, extent=ext, origin="upper", zorder=3,
                       interpolation="nearest")
        except Exception as ex:
            print(f"[overview-map] ortho '{stem}' draw failed, "
                  f"using extent rectangle: "
                  f"{type(ex).__name__}: {ex}")
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              linewidth=2.0, edgecolor=color,
                              facecolor=color, alpha=0.15, zorder=3)
            ax.add_patch(rect)
        edge = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          linewidth=1.5, edgecolor=color,
                          facecolor="none", alpha=0.8, zorder=5)
        ax.add_patch(edge)
        label = stem
        for marker in ("_of_UAV_ortho_image", "_ortho", "_uav"):
            label = label.replace(marker, "")
        label = label.strip("_") or stem
        ax.text(xmin, ymax, " " + label, color="black", fontsize=8,
                weight="bold",
                ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor=color,
                           alpha=0.85, pad=2.0), zorder=20)

    ax.set_xlabel("Easting (m)" if basemap_added else "X")
    ax.set_ylabel("Northing (m)" if basemap_added else "Y")
    try:
        from functions.map_export import (_use_full_coordinates as _ufc,
                                          _add_attribution_below as _aab)
        _ufc(ax)
        if basemap_added:
            _aab(ax, "Esri World Imagery")
    except Exception:
        pass
    title = (f"Project footprint — {len(extents)} ortho-image"
             f"{'s' if len(extents) > 1 else ''}"
             + (f"  ·  EPSG:{epsg}" if epsg else ""))
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=10)

    # The same decorations map_export draws on its publication maps.
    try:
        from functions.map_export import (
            _add_zebra_border, _add_scale_bar, _add_north_arrow,
        )
        _add_zebra_border(ax)
        if basemap_added or epsg:
            _add_scale_bar(ax, units="m")
        _add_north_arrow(ax)
    except Exception as ex:
        print(f"[overview-map] decoration skipped: "
              f"{type(ex).__name__}: {ex}")

    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        # Sidecar manifest of the source orthos, for the freshness check.
        try:
            manifest_path = out_path.with_suffix(".manifest.txt")
            manifest_path.write_text(
                _OVERVIEW_MAP_VERSION + "\n"
                + "\n".join(sorted(str(i["path"]) for i in uav_images)),
                encoding="utf-8")
        except Exception:
            pass
        return out_path
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None




from functions.units import resolve_display as _resolve_display


# Polygon-table columns that do NOT carry the summarised field's magnitude:
# identifiers, counts, ratios, and the phi-space moments (already unitless by
# construction). Everything else in a zonal polygon CSV is in the field's unit.
_POLY_UNITLESS = {
    "polygon_id", "zone_id", "fid", "id", "name", "label",
    "count", "n", "density", "coverage", "field", "unit",
    "sigma_phi", "sk_phi", "kg_phi", "mean_phi", "skewness", "kurtosis",
    "area", "area_m2", "perimeter",
}


def _poly_value_cols(header) -> list:
    """Indices of the columns carrying the summarised field's own magnitude.

    These are the ones a display-unit conversion applies to, and the only
    ones whose header should gain a unit bracket. Percentile columns arrive
    in either D- or P-notation depending on whether the field is a grain
    size, so match both.
    """
    out = []
    for i, h in enumerate(header):
        low = re.sub(r"\s*\(.*\)\s*$", "", str(h)).strip().lower()
        if low in _POLY_UNITLESS:
            continue
        if low in ("mean", "std", "median", "iqr", "min", "max", "range"):
            out.append(i)
        elif re.fullmatch(r"[dp]\d+", low):
            out.append(i)
    return out




def _project_gsd_m(inv) -> Optional[float]:
    """Ground sample distance of the project's imagery, metres per pixel.

    Read from the first georeferenced image's GeoTransform. Used as the
    precision floor when no validation result exists: nothing measured from an
    image is known to better than one pixel.
    """
    for entry in (inv.get("images") or []):
        try:
            from osgeo import gdal
            ds = gdal.Open(str(entry["path"]))
            if ds is None:
                continue
            gt = ds.GetGeoTransform()
            ds = None
            if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                px = abs(float(gt[1]))
                if px > 0 and math.isfinite(px):
                    return px
        except Exception:
            continue
    return None


def _resolve_uncertainties(inv) -> dict:
    """Field → Uncertainty for this project, read ONCE per build.

    Validation results live one JSON per quadrat; pooling them gives the
    mission's error. Where a field has none, one pixel of the imagery stands in
    — uncertainty scales with resolution, so a fixed constant would be wrong at
    every GSD but one.
    """
    from functions import precision as _prec

    out: dict = {}
    dirs = {Path(e["path"]).parent
            for e in (inv.get("validation", {}).get("results") or [])
            if str(e.get("path", "")).endswith(".validation.json")}
    for d in sorted(dirs):
        for field, records in _prec.load_validation_uncertainties(d).items():
            u = _prec.pooled(records, field)
            if u is None:
                continue
            # Several missions in one report: keep the larger error rather than
            # pooling across dates, which would describe no mission at all.
            prev = out.get(field)
            if prev is None or u.value > prev.value:
                out[field] = u
    gsd = _project_gsd_m(inv)
    out["__gsd_m__"] = gsd
    return out


# Units the ground sample distance can bound. The GSD is a length in metres
# per pixel, so it says nothing about a velocity, a shear stress or a
# dimensionless index — using it there would be dimensionally meaningless.
_GSD_BOUNDABLE_UNITS = {"mm", "cm", "m"}


def _uncertainty_display(uncerts: dict, field: str, factor: float):
    """(uncertainty in display units, Uncertainty or None) for one field.

    A measured validation RMSE is used wherever one exists. The one-pixel
    fallback applies only to lengths, because that is all a ground sample
    distance can bound.
    """
    from functions import precision as _prec

    gsd = (uncerts or {}).get("__gsd_m__")
    u = (uncerts or {}).get(field)
    if u is None:
        unit = _field_unit_and_factor(field)[0] if field else ""
        if gsd and unit in _GSD_BOUNDABLE_UNITS:
            u = _prec.from_gsd(gsd, field)
    if u is None:
        return None, None
    # Floor at a pixel only for lengths, for the same reason.
    unit = _field_unit_and_factor(field)[0] if field else ""
    floor = gsd if unit in _GSD_BOUNDABLE_UNITS else None
    return u.effective(floor) * float(factor), u


def _fmt_measured(value: float, unit: str, u_display=None) -> str:
    """Format a measured quantity without inventing precision.

    ``u_display`` is the field's uncertainty in the SAME unit as ``value`` —
    the mission's validation RMSE, floored at one pixel of the imagery. The
    value is rendered so its last digit sits at that magnitude, so 74.0348 mm
    against ± 10 mm prints as "74". Without an uncertainty this keeps the
    per-unit convention.
    """
    from functions import precision as _prec

    if value != value:                     # NaN
        return "—"
    if u_display is not None:
        return _prec.format_value(value, u_display, unit)
    if unit == "mm":
        return f"{value:.1f}"
    return f"{value:.4g}"


def _pretty_set_name(name: str) -> str:
    """Readable zone-set name for a heading.

    The identifier comes from a filename stem, so it arrives as
    ``zones_example_n1_of_UAV_ortho_image``. A heading should name the set,
    not print a path fragment; the exact filename stays in the caption's
    provenance line.
    """
    s = re.sub(r"^zones[_-]", "", str(name or "")).replace("_", " ").strip()
    if len(s) > 48:
        s = s[:47].rstrip() + "…"
    return s or str(name)


def _illustrations(story, ss, inv, options, section_num=4):
    """User-supplied illustration images, each with a custom title + caption.

    ``options["illustrations"]`` is a list of dicts:
    ``{"path": str, "title": str, "description": str}``. Each image is embedded
    at page width with the title as a sub-heading and the description as the
    figure caption, so it registers in the List of Figures like any other
    figure.
    """
    from reportlab.platypus import Paragraph, PageBreak, Image as RImage
    items = list(options.get("illustrations") or [])
    story.append(Paragraph(f"{section_num}. Illustrations", ss["H1Custom"]))
    if not items:
        story.append(Paragraph("No illustration images were added.", ss["Body"]))
        story.append(PageBreak())
        return
    _embeddable = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
    for idx, item in enumerate(items, start=1):
        path = Path(str(item.get("path", "")))
        title = (str(item.get("title") or "").strip()
                 or path.stem or f"Illustration {idx}")
        desc = str(item.get("description") or "").strip()
        heading = f"{section_num}.{idx} {title}" if len(items) > 1 else title
        story.append(Paragraph(heading, ss["H2Custom"]))
        if path.suffix.lower() in _embeddable and path.exists():
            try:
                story.append(RImage(str(path), width=6.5 * 72, height=5.5 * 72,
                                    kind="proportional"))
                cap = f"<i>{title}</i>." + (f" {desc}" if desc else "")
                story.append(_fig_caption(cap, ss))
            except Exception:
                story.append(Paragraph(
                    f"<i>Could not embed {path.name}.</i>", ss["Body"]))
                if desc:
                    story.append(Paragraph(desc, ss["Body"]))
        else:
            story.append(Paragraph(
                f"<i>Image not found or unsupported format: "
                f"<code>{path}</code>.</i>", ss["Body"]))
            if desc:
                story.append(Paragraph(desc, ss["Body"]))
    story.append(PageBreak())


# --- PDF document template + numbered canvas (ToC, bookmarks, Page N of M) ---
def _bookmark(canv, key):
    """Make ``key`` a destination on the canvas's current page. The numbered
    canvas below replays its pages in save(), so a destination made while
    the story is laid out would refer to the first page — the only page the
    document holds until then — and every bookmark of the outline opened
    the cover. Such a canvas notes the key
    with its page and places it during the replay."""
    later = getattr(canv, "bookmark_later", None)
    if later is not None:
        later(key)
    else:
        canv.bookmarkPage(key)


class _PDFDocTemplate:
    """Lazy factory for a ``BaseDocTemplate`` subclass whose ``afterFlowable``
    populates the :class:`TableOfContents` and emits PDF outline bookmarks
    for H1 / H2 paragraphs. Built lazily so importing this module does not
    pull ReportLab."""
    @staticmethod
    def build(out_path, page_size, project_name):
        from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame

        class _Doc(BaseDocTemplate):
            def afterFlowable(self, flowable):
                # H1 / H2 go to the ToC and the outline; captions go to their
                # own List of Figures / List of Tables channels only.
                style_name = getattr(getattr(flowable, "style", None),
                                      "name", "")
                if style_name == "H1Custom":
                    text = flowable.getPlainText()
                    self.notify("TOCEntry", (0, text, self.page))
                    key = f"h1_p{self.page}_{text[:40]}"
                    _bookmark(self.canv, key)
                    self.canv.addOutlineEntry(text, key, level=0,
                                               closed=False)
                elif style_name == "H2Custom":
                    text = flowable.getPlainText()
                    self.notify("TOCEntry", (1, text, self.page))
                    key = f"h2_p{self.page}_{text[:40]}"
                    _bookmark(self.canv, key)
                    self.canv.addOutlineEntry(text, key, level=1)
                elif style_name == "FigureCaption":
                    text = flowable.getPlainText()
                    self.notify("FigEntry", (0, text, self.page))
                elif style_name == "TableCaption":
                    text = flowable.getPlainText()
                    self.notify("TableEntry", (0, text, self.page))

        frame = Frame(0.5 * 72, 0.5 * 72,
                       page_size[0] - 1.0 * 72,
                       page_size[1] - 1.1 * 72,
                       id="normal")
        page_tpl = PageTemplate(id="default", frames=[frame],
                                 pagesize=page_size)
        # Landscape template for image overlays: a much bigger drawing area
        # for a multi-megapixel quadrat photo.
        ls_size = (page_size[1], page_size[0])
        ls_frame = Frame(0.5 * 72, 0.5 * 72,
                          ls_size[0] - 1.0 * 72,
                          ls_size[1] - 1.1 * 72,
                          id="landscape_frame")
        ls_tpl = PageTemplate(id="landscape", frames=[ls_frame],
                               pagesize=ls_size)
        doc = _Doc(
            str(out_path),
            pagesize=page_size,
            leftMargin=0.5 * 72, rightMargin=0.5 * 72,
            topMargin=0.6 * 72, bottomMargin=0.5 * 72,
            title=f"{_brand.APP_NAME} — {project_name} project report",
        )
        doc.addPageTemplates([page_tpl, ls_tpl])
        return doc


def _make_numbered_canvas(project_name):
    """Build a ``canvas.Canvas`` subclass that draws "Page N of M" and the
    project-name footer. M is unknown while pages stream, so ``showPage``
    buffers state per page and ``save`` finalises the text."""
    from reportlab.pdfgen import canvas as _rl_canvas

    class _NumberedCanvas(_rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            _rl_canvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_states = []
            # page number -> destination keys to place when that page is
            # replayed in save() (see _bookmark).
            self._pending_bookmarks = {}

        def bookmark_later(self, key):
            self._pending_bookmarks.setdefault(self._pageNumber, []).append(key)

        def showPage(self):
            self._saved_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_states)
            pending = self._pending_bookmarks
            for state in self._saved_states:
                self.__dict__.update(state)
                for key in pending.get(self._pageNumber, ()):
                    self.bookmarkPage(key)
                self._draw_footer(total)
                _rl_canvas.Canvas.showPage(self)
            _rl_canvas.Canvas.save(self)

        def _draw_footer(self, total):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColorRGB(0.4, 0.45, 0.55)
            self.drawString(0.5 * 72, 0.3 * 72,
                             f"{_brand.APP_NAME} — {project_name}")
            # Package version in the footer; "dev" when not installed with one.
            try:
                from functions import __version__ as _ver
            except Exception:
                _ver = "dev"
            self.drawCentredString(
                self._pagesize[0] / 2, 0.3 * 72,
                f"v{_ver}")
            self.drawRightString(self._pagesize[0] - 0.5 * 72, 0.3 * 72,
                                  f"Page {self._pageNumber} of {total}")
            self.restoreState()

    return _NumberedCanvas


def build_pdf(project_root, out_path, options=None, log_fn=None):
    """Build the project PDF report.

    options dict (all optional):
        author, affiliation, description, cover_image
        illustrations: list of {"path", "title", "description"} — user-supplied
            images rendered in a dedicated "Illustrations" section (auto-included
            when non-empty).
        include_cover, include_overview, include_methodology,
        include_detection, include_spatial_maps, include_validation,
        include_publication_figures, include_appendix_logs,
        include_appendix_samples, include_appendix_field_reference,
        include_toc (default True),
        include_list_of_figures (default True),
        include_list_of_tables (default True)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, PageBreak
    project_root = Path(project_root)
    out_path = Path(out_path)
    options = dict(options or {})
    # The Illustrations section exists only when images were supplied.
    options.setdefault("include_illustrations",
                       bool(options.get("illustrations")))
    # The preset controls embedding fidelity only, never which sections appear.
    _preset_name = str(options.get("preset", "full")).lower()
    if _preset_name not in REPORT_PRESETS:
        _preset_name = "full"
    _preset = REPORT_PRESETS[_preset_name]
    log = log_fn or (lambda _msg: None)
    log(f"[report] preset: {_preset_name} ({_preset['label']})")

    # Log this module's on-disk mtime so a long-running GUI that has not
    # picked up an edit can be recognised from the log.
    try:
        _this = Path(__file__).resolve()
        _mtime = datetime.fromtimestamp(_this.stat().st_mtime)
        log(f"[report] functions/report.py — mtime {_mtime:%Y-%m-%d %H:%M:%S}"
            f" — {_this}")
    except Exception:
        pass

    log(f"Walking project at {project_root}…")
    inv = collect_project_inventory(project_root)
    log(f"  {len(inv['images'])} image(s), {len(inv['vectors'])} CSV(s), "
        f"{len(inv['rasters'])} raster(s), {len(inv['maps'])} map(s), "
        f"{len(inv['logs'])} log(s).")
    log("Computing aggregate statistics…")
    stats = compute_aggregate_stats(inv)
    log(f"  total clasts: {stats['n_clasts_total']:,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = _PDFDocTemplate.build(out_path, A4, inv["project_name"])
    ss = _styles()
    story = []

    # Each section is guarded in report_sections, so one malformed artefact
    # costs its section, not the report.
    from functions import report_facts as _facts_mod
    from functions import report_sections as _sections
    log("Deriving the facts the report states…")
    facts = _facts_mod.gather_facts(inv, stats)
    _ensure_counters(ss)
    with _quiet_degenerate_mpl():
        _sections.build_story(story, ss, inv, stats, facts, options, log,
                              dpi=_preset["dpi"], quality=_preset["quality"])

    # Two passes: the ToC picks up page numbers from pass 1; the numbered
    # canvas finalises "Page N of M" in save().
    doc.multiBuild(
        story,
        canvasmaker=_make_numbered_canvas(inv["project_name"]),
    )
    log(f"[done] {out_path}")
    return out_path
