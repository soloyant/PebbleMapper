"""Derived facts the PDF report states: one computation per build.

``gather_facts(inventory, stats)`` turns the project inventory into what the
document says — per image: pixel size, ground sample distance, detection
limit, imaged area, the canonical population and its percentiles, clast
density, detected areal cover, orientation fabric; the detection parameters
read from the logs; which validation comparisons can be trusted; the rasters
grouped by source image and category.

Everything here is plain data (dicts, floats, numpy arrays) so the section
builders only format, and the numbers can be tested without ReportLab.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from functions import images as _images
from functions import modes, naming
from functions import units as _u

# Soloy et al. (2020): the smallest reliably segmented clast is about eight
# pixels along its long axis (4 cm at 5 mm/px).
DETECTION_LIMIT_PX = 8
# Percentiles need this many clasts before a per-cell value is quoted.
MIN_CELL_N_PERCENTILE = 30
MIN_CELL_N_MOMENTS = 50
# A validation comparison below this many paired clasts says nothing about a
# distribution.
MIN_VALIDATION_PAIRS = 5

_RASTER_CATEGORY = (
    ("density", "density"),
    ("packing", "cover"),
    ("orientation", "orientation"),
    ("hjulstrom", "transport"), ("shields", "transport"),
    ("soulsby", "transport"), ("van_rijn", "transport"), ("leroux", "transport"),
    ("circularity", "shape"), ("elongation", "shape"),
    ("eccentricity", "shape"), ("solidity", "shape"),
    ("surface_area", "size"), ("perimeter", "size"),
    ("length", "size"), ("width", "size"), ("axis", "size"),
    ("diameter", "size"),
)


def _site_and_date(project_root: Path) -> tuple:
    name = project_root.name
    if re.match(r"^\d{4}-\d{2}-\d{2}$", name):
        return project_root.parent.name, name
    return name, None


def _gdal_image_info(path: Path) -> dict:
    """Width, height, GSD, CRS of a georeferenced image; empty on failure."""
    out: dict = {}
    try:
        from osgeo import gdal, osr
        ds = gdal.Open(str(path))
        if ds is None:
            return out
        out["width_px"], out["height_px"] = ds.RasterXSize, ds.RasterYSize
        gt = ds.GetGeoTransform()
        if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            out["gsd_m"] = abs(float(gt[1]))
            out["transform"] = tuple(float(g) for g in gt)
        wkt = ds.GetProjection()
        if wkt:
            srs = osr.SpatialReference(wkt=wkt)
            code = srs.GetAuthorityCode(None)
            name = srs.GetName() if hasattr(srs, "GetName") else None
            out["crs"] = (f"EPSG:{code}" if code else "") + (
                f" ({name})" if name else "")
            out["crs"] = out["crs"].strip() or None
        ds = None
    except Exception:
        pass
    return out


def _pil_image_size(path: Path) -> dict:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return {"width_px": im.size[0], "height_px": im.size[1]}
    except Exception:
        return {}


def _quantiles_mm(values_m: np.ndarray) -> dict:
    v = np.asarray(values_m, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return {}
    qs = (5, 16, 25, 50, 75, 84, 95)
    pct = np.percentile(v, qs)
    out = {f"D{q}": float(p) * 1000.0 for q, p in zip(qs, pct)}
    out["n"] = int(v.size)
    out["mean"] = float(v.mean()) * 1000.0
    out["std"] = float(v.std()) * 1000.0
    return out


def _folk_ward(values_m: np.ndarray) -> dict:
    v = np.asarray(values_m, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 5:
        return {}
    phi = -np.log2(v * 1000.0)
    p5, p16, p25, p50, p75, p84, p95 = np.percentile(phi, (5, 16, 25, 50, 75, 84, 95))
    from functions import report as R
    sigma = _u.folk_ward_sorting_phi(p5, p16, p84, p95)
    skew = _u.folk_ward_skewness_phi(p5, p16, p50, p84, p95)
    kurt = _u.folk_ward_kurtosis_phi(p5, p25, p75, p95)
    return {
        "sorting_phi": float(sigma), "skewness_phi": float(skew),
        "kurtosis_phi": float(kurt),
        "sorting_verbal": R._classify_sorting(sigma),
        "skewness_verbal": R._classify_skewness(skew),
        "kurtosis_verbal": R._classify_kurtosis(kurt),
        "mean_phi": float(np.mean(phi)),
    }


def bootstrap_ci_mm(values_m: np.ndarray, q: float = 50, n_boot: int = 300,
                    seed: int = 7) -> Optional[tuple]:
    """95 % bootstrap interval (mm) of one percentile; None when n < 20."""
    v = np.asarray(values_m, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 20:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.percentile(v[idx], q, axis=1) * 1000.0
    lo, hi = np.percentile(boots, (2.5, 97.5))
    return float(lo), float(hi)


def axial_circular_mean(deg) -> tuple:
    """(mean direction in [0, 180), mean resultant length) of axial data."""
    a = np.asarray(deg, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    t = np.deg2rad(a) * 2.0
    c, s = np.mean(np.cos(t)), np.mean(np.sin(t))
    r = float(math.hypot(c, s))
    mean = math.degrees(math.atan2(s, c)) / 2.0
    return float(mean % 180.0), r


def _read_clasts(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _log_key_for(vector: dict) -> Optional[str]:
    """The detection-log stem a CSV owns (its own naming generation)."""
    w = vector.get("window_size")
    if w is None:
        return vector.get("run_stem") or vector.get("image_stem")
    stem = vector.get("run_stem") or vector["image_stem"]
    if "_window_size=" in Path(vector["path"]).name:
        return f"{stem}_window_size={w}m"
    return naming.detection_csv_name(stem, w)[:-4]


def _run_rows(inv: dict, image_stem: str, canonical_path: Optional[Path]) -> list:
    log_index = {Path(l["path"]).stem.replace("_detection_log", ""): l
                 for l in inv.get("logs", [])}
    rows = []
    for v in inv.get("vectors", []):
        if v["image_stem"] != image_stem:
            continue
        log = log_index.get(_log_key_for(v) or "", {})
        runs = log.get("runs") or []
        complete = [r for r in runs if str(r.get("outcome", "")).startswith("complete")]
        last = runs[-1] if runs else {}
        params = log.get("parameters", {})
        rows.append({
            "path": Path(v["path"]),
            "name": Path(v["path"]).name,
            "window_m": v.get("window_size"),
            "merged": bool(v.get("is_merged")),
            "n_clasts": int(v.get("n_rows") or 0),
            "counted": (canonical_path is not None
                        and Path(v["path"]) == canonical_path),
            "overlap": params.get("overlap"),
            "tiles": _first_int(params.get("tiles to process")
                                or params.get("tiles processed")
                                or params.get("tile grid")),
            "date": (runs[0].get("start_ts") or runs[0].get("ts")) if runs else None,
            "elapsed_s": (complete[-1]["elapsed_s"] if complete
                          else (last.get("elapsed_s") or 0.0)),
            "n_runs": len(runs),
            "n_aborted": sum(1 for r in runs
                             if not str(r.get("outcome", "")).startswith("complete")),
            "outcome": last.get("outcome", "—") if runs else "—",
            "parameters": params,
        })
    rows.sort(key=lambda r: (0 if r["merged"] else 1,
                             r["window_m"] if r["window_m"] is not None else 0,
                             r["name"]))
    return rows


def _first_int(s) -> Optional[int]:
    if not s:
        return None
    m = re.match(r"\s*(\d+)", str(s))
    return int(m.group(1)) if m else None


def _raster_category(field: Optional[str], parameter: Optional[str]) -> str:
    key = f"{field or ''} {parameter or ''}".lower()
    if "packing" in key:
        return "cover"
    if parameter and parameter.lower() == "density":
        return "density"
    for needle, cat in _RASTER_CATEGORY:
        if needle in key:
            return cat
    return "other"


def _all_bands_empty(path: Path) -> bool:
    """True when no band of the raster holds a finite value."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            return True
        for b in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(b)
            arr = np.asarray(band.ReadAsArray(), dtype=float)
            nd = band.GetNoDataValue()
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            if np.isfinite(arr).any():
                return False
        return True
    except Exception:
        return False


def _param_key(r: dict) -> str:
    """The statistic as a display key: a ``quantile`` raster becomes its
    percentile (``D50``) when the sidecar records it."""
    param = r.get("parameter") or ""
    if param.lower() == "quantile":
        try:
            import json
            side = json.loads(Path(str(r["path"]) + ".json").read_text(encoding="utf-8"))
            pct = side.get("percentile")
            if pct is not None:
                return f"D{int(round(float(pct) * 100))}"
        except Exception:
            pass
        return "quantile"
    return param


_TOOL_SUFFIXES = naming.TOOL_SUFFIXES


def photo_key(stem: str) -> str:
    """One key for a photograph and every file derived from it: the origin
    stripped, GSD tags and tool suffixes removed, case folded."""
    s = naming.image_stem(str(stem))
    s = re.sub(r"_GSD=[0-9.]+m(?:_per_px)?", "", s)
    s = re.sub(r"_GSD=[0-9]+p[0-9]+mm", "", s)
    for suf in _TOOL_SUFFIXES:
        s = s.replace(suf, "")
    return s.lower()


def _raster_source_stem(r: dict, image_stems: list) -> Optional[str]:
    src = r.get("source_image")
    if src:
        return naming.image_stem(Path(src).name)
    name = Path(r["path"]).stem
    bare = naming.strip_origin(name)
    best = None
    for s in image_stems:
        if (name.startswith(s) or bare.startswith(s)) and (best is None or len(s) > len(best)):
            best = s
    if best:
        return best
    rec = naming.image_stem(name)
    return rec if rec != name else None


def gather_facts(inv: dict, stats: dict) -> dict:
    """Everything the report states, derived once from the inventory."""
    from functions import report as R
    root = Path(inv["project_root"])
    site, date = _site_and_date(root)
    canonical = R._canonical_vectors(inv)
    canon_by_stem = {}
    for e in canonical:
        canon_by_stem.setdefault(e["image_stem"], []).append(e)

    images = []
    for entry in inv.get("images", []):
        # The inventory may carry an older spelling; compare canonical values only.
        kind = modes.normalise_mode(entry.get("kind"), default="")
        if kind not in modes.MODES:
            continue
        path = Path(entry["path"])
        info = _gdal_image_info(path) if kind == modes.ORTHO else _pil_image_size(path)
        if kind == modes.QUADRAT or not info.get("gsd_m"):
            try:
                from functions.gsd import effective_gsd
                gi = effective_gsd(path)
                if gi.gsd:
                    info["gsd_m"] = float(gi.gsd)
                    info["gsd_source"] = gi.source
            except Exception:
                pass
        gsd = info.get("gsd_m")
        w, h = info.get("width_px"), info.get("height_px")
        footprint = (w * h * gsd * gsd) if (w and h and gsd) else None
        rec = {
            "name": path.name, "stem": path.stem, "kind": kind, "path": path,
            "width_px": w, "height_px": h, "gsd_m": gsd,
            "gsd_source": info.get("gsd_source", "georeference" if kind == modes.ORTHO else None),
            "crs": info.get("crs"), "transform": info.get("transform"),
            "footprint_m2": footprint,
            "width_m": (w * gsd) if (w and gsd) else None,
            "height_m": (h * gsd) if (h and gsd) else None,
            "dmin_m": (DETECTION_LIMIT_PX * gsd) if gsd else None,
            "canonical": None, "detect_image": None,
            "n_clasts": 0, "length_mm": {}, "width_mm": {},
            "folk_ward": {}, "density_per_m2": None, "areal_cover": None,
            "orientation": {}, "runs": [], "repeatability": [], "merged_variants": [],
            "d50_ci_mm": None, "d84_ci_mm": None, "clasts": None,
        }
        cands = [e for e in canon_by_stem.get(path.stem, [])] or [
            e for stem, es in canon_by_stem.items() for e in es
            if naming.same_image(stem, path.stem)] or [
            e for stem, es in canon_by_stem.items() for e in es
            if photo_key(stem) == photo_key(path.stem)]
        if cands:
            # One canonical CSV per image: the merged one when present.
            cands.sort(key=lambda e: (0 if e.get("is_merged") else 1,
                                      len(Path(e["path"]).name)))
            canon = cands[0]
            rec["canonical"] = Path(canon["path"])
            # The file the detector read: the photograph itself, or the
            # rectified copy named in the CSV when one exists.
            want = naming.image_stem(rec["canonical"].name)
            if want != path.stem:
                for cand in path.parent.rglob(want + ".*"):
                    if cand.suffix.lower() in _images.PHOTO_EXTENSIONS:
                        rec["detect_image"] = cand
                        try:
                            from functions.gsd import effective_gsd
                            gi = effective_gsd(cand)
                            if gi.gsd and not rec["gsd_m"]:
                                rec["gsd_m"] = float(gi.gsd)
                                rec["dmin_m"] = DETECTION_LIMIT_PX * rec["gsd_m"]
                        except Exception:
                            pass
                        break
            df = _read_clasts(rec["canonical"])
            if df is not None and len(df):
                rec["clasts"] = df
                rec["n_clasts"] = int(len(df))
                if "Clast_length" in df:
                    L = df["Clast_length"].to_numpy(dtype=float)
                    rec["length_mm"] = _quantiles_mm(L)
                    rec["folk_ward"] = _folk_ward(L)
                    rec["d50_ci_mm"] = bootstrap_ci_mm(L, 50)
                    rec["d84_ci_mm"] = bootstrap_ci_mm(L, 84)
                    if rec["dmin_m"]:
                        rec["n_below_dmin"] = int(np.sum(L < rec["dmin_m"]))
                if "Clast_width" in df:
                    rec["width_mm"] = _quantiles_mm(df["Clast_width"].to_numpy(dtype=float))
                if footprint:
                    rec["density_per_m2"] = rec["n_clasts"] / footprint
                    if "Surface_area" in df:
                        area = float(np.nansum(df["Surface_area"].to_numpy(dtype=float)))
                        if area > 0:
                            rec["areal_cover"] = min(1.0, area / footprint)
                if "Orientation" in df:
                    mean_deg, rbar = axial_circular_mean(df["Orientation"].to_numpy(dtype=float))
                    rec["orientation"] = {"mean_deg": mean_deg, "R": rbar}
        run_key = path.stem
        if path.stem not in canon_by_stem:
            run_key = next((s for s in canon_by_stem if naming.same_image(s, path.stem)
                            or photo_key(s) == photo_key(path.stem)), path.stem)
        rec["runs"] = _run_rows(inv, run_key, rec["canonical"])
        # Two complete runs at the same window: the detector's repeatability.
        by_w: dict = {}
        for r in rec["runs"]:
            if r["window_m"] is not None and not r["merged"] and r["n_clasts"]:
                by_w.setdefault((r["window_m"], str(r["overlap"])), []).append(r["n_clasts"])
        rec["repeatability"] = [(w, ns) for (w, _o), ns in sorted(by_w.items()) if len(ns) > 1]
        rec["merged_variants"] = [r for r in rec["runs"] if r["merged"]]
        images.append(rec)

    # A canonical CSV whose image is not in the project (orthos are often
    # archived elsewhere) is still a population: report it from the CSV
    # alone, with the footprint as the bounding box of the clasts.
    matched = {str(i["canonical"]) for i in images if i["canonical"]}
    for stem, es in canon_by_stem.items():
        es = sorted(es, key=lambda e: (0 if e.get("is_merged") else 1, len(Path(e["path"]).name)))
        canon = es[0]
        if str(canon["path"]) in matched:
            continue
        df = _read_clasts(Path(canon["path"]))
        if df is None or not len(df) or "Clast_length" not in df:
            continue
        is_uav = bool(canon.get("is_merged") or canon.get("window_size") is not None)
        L = df["Clast_length"].to_numpy(dtype=float)
        rec = {
            "name": stem, "stem": stem, "kind": modes.ORTHO if is_uav else modes.QUADRAT,
            "path": None, "width_px": None, "height_px": None, "gsd_m": None,
            "gsd_source": None, "crs": None, "transform": None,
            "footprint_m2": None, "footprint_kind": "bounding box",
            "width_m": None, "height_m": None, "dmin_m": None,
            "canonical": Path(canon["path"]), "detect_image": None,
            "n_clasts": int(len(df)), "length_mm": _quantiles_mm(L),
            "width_mm": (_quantiles_mm(df["Clast_width"].to_numpy(dtype=float))
                         if "Clast_width" in df else {}),
            "folk_ward": _folk_ward(L), "density_per_m2": None, "areal_cover": None,
            "orientation": {}, "runs": _run_rows(inv, stem, Path(canon["path"])),
            "repeatability": [], "merged_variants": [],
            "d50_ci_mm": bootstrap_ci_mm(L, 50), "d84_ci_mm": bootstrap_ci_mm(L, 84),
            "clasts": df, "image_missing": True,
        }
        if is_uav and {"x", "y"} <= set(df.columns):
            x = df["x"].to_numpy(dtype=float)
            y = df["y"].to_numpy(dtype=float)
            if x.size and np.isfinite(x).any():
                w = float(np.nanmax(x) - np.nanmin(x))
                h = float(np.nanmax(y) - np.nanmin(y))
                if w > 0 and h > 0:
                    rec["footprint_m2"] = w * h
                    rec["width_m"], rec["height_m"] = w, h
                    rec["density_per_m2"] = rec["n_clasts"] / (w * h)
        if "Orientation" in df:
            mean_deg, rbar = axial_circular_mean(df["Orientation"].to_numpy(dtype=float))
            rec["orientation"] = {"mean_deg": mean_deg, "R": rbar}
        rec["merged_variants"] = [r for r in rec["runs"] if r["merged"]]
        images.append(rec)

    # Detection parameters, from the logs.
    params: dict = {"windows_m": set(), "overlap": set(), "dedup": set(),
                    "min_confidence": set(), "model": set()}
    for log in inv.get("logs", []):
        p = log.get("parameters", {}) or {}
        w = p.get("metric_cropsize")
        if w:
            try:
                params["windows_m"].add(float(str(w).split()[0]))
            except ValueError:
                pass
        for key, dst in (("overlap", "overlap"), ("min_confidence", "min_confidence"),
                         ("dedup_overlap", "dedup"), ("model", "model"),
                         ("backend", "model")):
            if p.get(key):
                params[dst].add(str(p[key]).strip())
    params = {k: sorted(v) for k, v in params.items()}

    # Validation comparisons, with the flags that decide whether they count.
    validations = []
    for r in inv.get("validation", {}).get("results", []):
        p = Path(r["path"])
        if not p.name.endswith(".validation.json"):
            continue
        payload = R._load_validation_json(p)
        if not payload:
            continue
        truth = Path(payload.get("truth_csv") or "")
        detect = Path(payload.get("detect_csv") or "")
        m = payload.get("metrics") or {}
        d = payload.get("distribution") or {}
        n_pairs = int(m.get("n_matched") or 0)
        truth_rows = int(d.get("truth_n") or m.get("n_truth") or 0)
        detect_rows = int(d.get("detect_n") or m.get("n_detect") or 0)
        self_cmp = (truth.name == detect.name) or (
            truth.exists() and detect.exists()
            and truth.resolve() == detect.resolve())
        flags = []
        if self_cmp:
            flags.append("truth and detection are the same file")
        if truth_rows == 0:
            flags.append("truth file holds no clasts")
        elif n_pairs < MIN_VALIDATION_PAIRS:
            flags.append(f"only {n_pairs} paired clasts")
        detect_gsd = payload.get("detect_gsd_m_per_px")
        validations.append({
            "path": p, "payload": payload, "field": payload.get("field") or "Clast_length",
            "truth": truth, "detect": detect, "n_pairs": n_pairs,
            "truth_rows": truth_rows, "detect_rows": detect_rows,
            "detect_gsd_m": float(detect_gsd) if detect_gsd else None,
            "dmin_m": (DETECTION_LIMIT_PX * float(detect_gsd)) if detect_gsd else None,
            "usable": not flags, "flags": flags,
            "detect_kind": (modes.ORTHO if (naming.is_merged(detect.name)
                                            or naming.parse_window_size(detect.name) is not None)
                            else modes.QUADRAT),
        })
    truth_files = []
    for t in inv.get("validation", {}).get("truth_csvs", []):
        tp = Path(t["path"])
        n = 0
        try:
            with open(tp, encoding="utf-8") as fh:
                n = max(0, sum(1 for _ in fh) - 1)
        except OSError:
            pass
        truth_files.append({"path": tp, "rows": n})

    # Rasters by source image and category.
    image_stems = [i["stem"] for i in images]
    rasters_by_image: dict = {}
    for r in inv.get("rasters", []):
        src = _raster_source_stem(r, image_stems) or "(unknown source)"
        field, param = r.get("field"), r.get("parameter")
        key = _param_key(r)
        disp, unit, factor, cmap, diverging = _u.resolve_display(field, key)
        rasters_by_image.setdefault(src, []).append({
            **r, "path": Path(r["path"]), "param_key": key, "display": disp,
            "unit": unit, "factor": factor, "cmap": cmap, "diverging": diverging,
            "category": _raster_category(field, param),
            "empty": _all_bands_empty(Path(r["path"])),
        })

    for rec in images:
        rec.setdefault("footprint_kind", "image")
        rec.setdefault("image_missing", False)
    n_ortho = sum(1 for i in images if i["kind"] == modes.ORTHO)
    n_quadrat = sum(1 for i in images if i["kind"] == modes.QUADRAT)
    return {
        "site": site, "date": date, "project_root": root,
        "images": images, "n_ortho": n_ortho, "n_quadrat": n_quadrat,
        "params": params, "validations": validations, "truth_files": truth_files,
        "rasters_by_image": rasters_by_image,
        "zonal": inv.get("zonal") or [],
        "profile_figures": inv.get("zonal_profile_figures") or [],
        "weights": inv.get("weights") or {},
        "uncertainty": inv.get("uncertainty") or {},
    }


def pooled_lengths(facts: dict, kind: Optional[str] = None) -> np.ndarray:
    """Clast lengths (m) of every canonical population of one kind."""
    parts = []
    for img in facts["images"]:
        if kind and img["kind"] != kind:
            continue
        df = img.get("clasts")
        if df is not None and "Clast_length" in df:
            parts.append(df["Clast_length"].to_numpy(dtype=float))
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def headline_kind(facts: dict) -> Optional[str]:
    """The population the summary leads with: ortho when present."""
    if facts["n_ortho"]:
        return modes.ORTHO
    if facts["n_quadrat"]:
        return modes.QUADRAT
    return None
