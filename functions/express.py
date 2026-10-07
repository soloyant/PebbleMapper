"""Express pipeline: one-click detect→merge→rasterise→map→report with presets.
GUI-free; the Express tab only drives these functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from functions import modes, naming

PRESETS: dict[str, dict] = {
    "fast":   {"overlap": 0.0,  "min_confidence": 0.80, "dedup_overlap": 0.30,
               "merge": False, "min_cell_density": 0},
    "medium": {"overlap": 0.20, "min_confidence": 0.70, "dedup_overlap": 0.30,
               "merge": True,  "min_cell_density": 3},
    "slow":   {"overlap": 0.25, "min_confidence": 0.60, "dedup_overlap": 0.30,
               "merge": True,  "min_cell_density": 5},
}


# Detection window size(s) in metres per preset. Larger windows mean fewer
# tiles but coarser detection; multi-window presets merge their results.
_WINDOWS = {
    "fast":   [5.0],
    "medium": [2.5, 5.0],
    "slow":   [1.0, 2.5, 5.0],
}


def resolve_window_sizes(extent: dict, preset_name: str) -> list[float]:
    """Detection window size(s) in metres for a preset. ``extent`` is accepted
    for API symmetry but does not influence the result."""
    return list(_WINDOWS.get(preset_name, [2.5]))


def read_ortho_extent(ortho_path) -> dict:
    """Real-world extent + GSD of an ortho via GDAL. is_georeferenced is False
    for an identity GeoTransform; Express is ortho-only and blocks those."""
    from osgeo import gdal
    ds = gdal.Open(str(ortho_path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open ortho: {ortho_path}")
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    ds = None
    px, py = abs(gt[1]), abs(gt[5])
    identity = gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    is_geo = (not identity) and px > 0
    return {
        "width_m": nx * px,
        "height_m": ny * py,
        "gsd_m": px if is_geo else None,
        "is_georeferenced": bool(is_geo),
    }


# Report sections the quick pipeline never produces inputs for.
_ALWAYS_OFF = {"include_zonal_statistics", "include_validation"}
_ALL_REPORT_KEYS = (
    "include_cover", "include_overview", "include_methodology", "include_detection",
    "include_data_quality", "include_spatial_maps", "include_zonal_statistics",
    "include_publication_figures", "include_validation", "include_appendix_logs",
    "include_appendix_samples", "include_appendix_field_reference",
)
_REPORT_ON = {
    "fast":   {"include_cover", "include_overview", "include_detection",
               "include_spatial_maps", "include_publication_figures"},
    "medium": {"include_cover", "include_overview", "include_methodology",
               "include_detection", "include_data_quality", "include_spatial_maps",
               "include_publication_figures", "include_appendix_logs",
               "include_appendix_field_reference"},
    "slow":   set(_ALL_REPORT_KEYS) - _ALWAYS_OFF,
}
# Rasters per preset: (field, parameter, percentile). Maps: (layer, field, basemap).
_RASTERS = {
    "fast":   [("Clast_length", "quantile", 0.5)],
    # "density" is a per-cell statistic of a field, not a field: the
    # rasterizer counts the clasts of the field's column per cell.
    "medium": [("Clast_length", "quantile", 0.5), ("Clast_length", "density", None),
               ("Clast_length", "folk_ward_sorting", None)],
    "slow":   [("Clast_length", "quantile", 0.5), ("Clast_length", "quantile", 0.84),
               ("Clast_length", "density", None), ("Clast_length", "folk_ward_sorting", None),
               ("Equivalent_diameter", "quantile", 0.5)],
}
_MAPS = {
    "fast":   [("vector", "Clast_length", None)],
    "medium": [("vector", "Clast_length", None), ("raster", "Clast_length", None)],
    "slow":   [("vector", "Clast_length", "Esri.WorldImagery"),
               ("raster", "Clast_length", "Esri.WorldImagery"),
               ("raster", "density", "Esri.WorldImagery")],
}


@dataclass
class ResolvedPlan:
    preset: str
    windows: list[float]
    overlap: float
    min_confidence: float
    dedup_overlap: float
    merge: bool
    cellsize: float
    min_cell_density: int
    rasters: list[dict]
    maps: list[dict]
    report_options: dict
    devicemode: str = "gpu"
    devicenumber: int = 0
    model: str = "maskrcnn"   # detection backend name (detectors.registry)


def resolve_plan(ortho_path, preset_name: str, windows=None,
                 model: str = "maskrcnn") -> ResolvedPlan:
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}")
    ext = read_ortho_extent(ortho_path)
    if not ext["is_georeferenced"]:
        raise ValueError("Ortho is not georeferenced (identity GeoTransform); "
                         "Express is ortho-only. Use the Detect tab.")
    spec = PRESETS[preset_name]
    # User-supplied windows are used as-is (deduped, ascending); merge is
    # driven purely by how many windows there are.
    if windows:
        windows = sorted({round(float(w), 4) for w in windows
                          if float(w) > 0})
        if not windows:
            raise ValueError("No valid (>0) window size provided.")
    else:
        windows = resolve_window_sizes(ext, preset_name)
    merge = len(windows) >= 2
    cellsize = max(1.0, windows[0]) if preset_name == "fast" else 1.0
    rasters = [{"field": f, "parameter": p, "percentile": q,
                "cellsize": cellsize, "min_cell_density": spec["min_cell_density"]}
               for (f, p, q) in _RASTERS[preset_name]]
    maps = [{"layer": l, "field": f, "basemap": b} for (l, f, b) in _MAPS[preset_name]]
    report_options = {k: (k in _REPORT_ON[preset_name] and k not in _ALWAYS_OFF)
                      for k in _ALL_REPORT_KEYS}
    return ResolvedPlan(
        preset=preset_name, windows=windows, overlap=spec["overlap"],
        min_confidence=spec["min_confidence"], dedup_overlap=spec["dedup_overlap"],
        merge=merge, cellsize=cellsize,
        min_cell_density=spec["min_cell_density"], rasters=rasters, maps=maps,
        report_options=report_options, model=model,
    )


@dataclass
class StageStatus:
    name: str
    status: str = "pending"   # pending|running|done|error|skipped|stopped
    detail: str = ""

@dataclass
class ExpressResult:
    stages: list
    report_pdf: Optional[Path] = None
    rasters: list = field(default_factory=list)
    maps: list = field(default_factory=list)
    merged_csv: Optional[Path] = None
    window_csvs: list = field(default_factory=list)


# --- thin stage wrappers (each calls one backend fn; monkeypatched in tests) ---
def _stage_detect(project, ortho, window, plan, *, log_fn, progress_cb, stop_check):
    """Run ortho detection for one window size; return the written CSV path.
    The detector writes into ``output_dir`` and returns DataFrames, so the
    CSV is located by name afterwards."""
    from functions import clasts_detection, layout
    out_dir = layout.project_path(project, "vectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Route through the selected detection backend; fall back to the direct
    # call if the registry is unavailable.
    try:
        from detectors import get_backend
        _backend = get_backend(getattr(plan, "model", "maskrcnn") or "maskrcnn")
    except Exception:
        _backend = None
    _detect = (_backend.detect_jobs if _backend is not None
               else clasts_detection.clasts_detect_jobs)
    log_fn(f"[express]   loading the detection model and tiling the ortho at "
           f"{window:g} m — the first run can take up to a minute to initialise "
           f"(no output appears until the model is ready)…")
    # Only the backend path accepts log_fn.
    _extra = {"log_fn": log_fn} if _backend is not None else {}
    origin = naming.origin_stem(Path(ortho).stem, project,
                                layout.get_active_date())
    _detect(
        modes.ORTHO, [{"path": str(ortho), "kstart": 0, "out_stem": origin}],
        metric_cropsize=window, overlap=plan.overlap,
        min_confidence=plan.min_confidence, dedup_method="iou",
        dedup_overlap=plan.dedup_overlap, plot=False, saveresults=True,
        devicemode=plan.devicemode, devicenumber=plan.devicenumber,
        output_dir=str(out_dir), progress_callback=progress_cb,
        stop_check=stop_check, **_extra)
    log_fn(f"[express]   detection at {window:g} m complete.")
    stem = origin
    # The detector writes the standard detection name; the _xprs tag applies
    # only to express-specific artefacts.
    cand = out_dir / naming.detection_csv_name(stem, window)
    if cand.exists():
        return cand
    # Fallback glob over both the short and the legacy long form, newest wins.
    matches = sorted(
        [p for p in out_dir.glob(f"{stem}*.csv")
         if not p.name.endswith(".run.csv")
         and naming.parse_window_size(p.name) is not None
         and not naming.is_merged(p.name)],
        key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else cand

def _stage_merge(csvs, out):
    """Fold N window CSVs (smallest window first) into one by merging pairs
    iteratively; the larger window wins conflicts."""
    import shutil
    import tempfile
    from functions import clasts_merge
    csvs = [str(c) for c in csvs]
    if len(csvs) == 1:
        shutil.copyfile(csvs[0], str(out))
        return Path(out)
    acc = csvs[0]
    tmps: list[str] = []
    last = len(csvs) - 1
    for i in range(1, len(csvs)):
        dest = str(out) if i == last else tempfile.mkstemp(suffix=".csv")[1]
        if i != last:
            tmps.append(dest)
        clasts_merge.merge_csvs(acc, csvs[i], dest, "iou", 0.30, 32)
        acc = dest
    for t in tmps:
        try:
            Path(t).unlink()
        except OSError:
            pass
    return Path(out)

def _stage_rasterize(ortho, csv, out, spec):
    # Keyword arguments deliberately: a positional call would break silently
    # if clasts_rasterize's long parameter list ever changed order.
    from functions import clasts_rasterize
    clasts_rasterize.clasts_rasterize(str(ortho), str(csv), str(out),
                                      field=spec["field"],
                                      parameter=spec["parameter"],
                                      cellsize=spec["cellsize"],
                                      percentile=spec.get("percentile") or 0.5,
                                      min_cell_density=spec["min_cell_density"],
                                      plot=False)
    return Path(out)

def _stage_map(ortho, src, out, spec):
    from functions import map_export
    # One ortho pixel is the finest length the map can mean: the colourbar
    # labels stop there instead of printing 80.016 mm on a 3.4 mm/px image.
    try:
        gsd = read_ortho_extent(ortho).get("gsd_m")
    except Exception:
        gsd = None
    fig = map_export.make_publication_map(
        ortho_tif=str(ortho),
        clast_csv=str(src) if spec["layer"] == "vector" else None,
        raster_tif=str(src) if spec["layer"] == "raster" else None,
        layer=spec["layer"], field=spec["field"], basemap=spec["basemap"],
        uncertainty=(round(float(gsd) * 1000.0, 2)
                     if gsd and spec["field"] != "density" else None))
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close(fig)
    return Path(out)

def _stage_report(project_root, out, opts, log_fn):
    # ``project_root`` is the absolute project directory, not a project name.
    from functions import report
    return report.build_pdf(project_root, out, options=opts, log_fn=log_fn)


def run_express_pipeline(project, ortho_path, preset_name, *,
                         windows=None, model="maskrcnn", log_fn=None,
                         progress_cb=None, on_stage=None, stop_check=None,
                         devicemode="gpu", devicenumber=0) -> ExpressResult:
    """Run each stage best-effort: on failure mark it 'error' and continue to
    stages whose inputs still exist. ``windows`` overrides the preset's window
    sizes; ``on_stage(name, status)`` is called on every status change."""
    from functions import layout
    log = log_fn or (lambda _m: None)
    stop = stop_check or (lambda: False)
    plan = resolve_plan(ortho_path, preset_name, windows=windows, model=model)
    plan.devicemode = devicemode
    plan.devicenumber = devicenumber
    stages = {n: StageStatus(n) for n in
              ("Detection", "Merge", "Rasterize", "Map", "Report")}
    res = ExpressResult(stages=list(stages.values()))

    def _set(name, status, detail=""):
        stages[name].status = status
        if detail:
            stages[name].detail = detail
        if on_stage:
            try:
                on_stage(name, status)
            except Exception:
                pass
    log(f"[express] starting '{preset_name}' run — detection window(s): "
        f"{', '.join(f'{w:g} m' for w in plan.windows)}; "
        f"merge {'on' if plan.merge else 'off'}.")

    # Detection (one run per window)
    _set("Detection", "running")
    try:
        for i, w in enumerate(plan.windows, 1):
            if stop():
                break
            log(f"[express] Detection {i}/{len(plan.windows)} — {w:g} m window…")
            csv = _stage_detect(project, ortho_path, w, plan,
                                log_fn=log, progress_cb=progress_cb, stop_check=stop)
            if csv and Path(csv).exists():
                res.window_csvs.append(Path(csv))
        if stop():
            # Stop halts the whole run: the window that was running is
            # partial (its CSV and log say "partial (stopped)"; the .run.csv
            # checkpoint stays), and no later stage is built on it.
            _set("Detection", "stopped")
            for name in ("Merge", "Rasterize", "Map", "Report"):
                _set(name, "skipped")
            log("[express] stopped — the current window's detection is partial "
                "(its log says 'partial (stopped)'; the checkpoint stays in "
                "output_results/vectors); merge, rasters, maps and report "
                "were not run.")
            return res
        _set("Detection", "done" if res.window_csvs else "error")
        if res.window_csvs:
            log(f"[express] Detection done — {len(res.window_csvs)} window CSV(s).")
    except Exception as ex:
        _set("Detection", "error", str(ex))
        log(f"[express] detection failed: {ex}")

    # Choose the canonical CSV: merged (if merge & ≥2 windows) else the largest window
    canonical = None
    if plan.merge and len(res.window_csvs) >= 2:
        _set("Merge", "running")
        try:
            vdir = layout.project_path(project, "vectors")
            vdir.mkdir(parents=True, exist_ok=True)
            merged = vdir / naming.merged_csv_name(
                naming.run_stem(res.window_csvs[-1].name), express=True)
            log("[express] Merging detection windows…")
            res.merged_csv = _stage_merge(res.window_csvs, merged)
            canonical = res.merged_csv; _set("Merge", "done")
        except Exception as ex:
            _set("Merge", "error", str(ex))
            log(f"[express] merge failed: {ex}")
    else:
        _set("Merge", "skipped")
    if canonical is None and res.window_csvs:
        canonical = res.window_csvs[-1]  # fall back to last/largest window

    # Rasterize
    if canonical:
        _set("Rasterize", "running")
        rdir = layout.project_path(project, "rasters")
        rdir.mkdir(parents=True, exist_ok=True)
        log(f"[express] Rasterising {len(plan.rasters)} surface(s)…")
        for spec in plan.rasters:
            if stop(): break
            tag = f"{spec['field']}_{spec['parameter']}"
            # Standard raster name: the report and the Map tab parse field,
            # parameter and cell size from it.
            csv_stem = naming.run_stem(canonical.name) + "_xprs"
            # A quantile raster is named by its percentile (D50), as the
            # Rasterize tab names it.
            token = spec["parameter"]
            if token == "quantile":
                token = f"D{int(round(float(spec.get('percentile') or 0.5) * 100))}"
            out = rdir / naming.raster_name(csv_stem, spec["field"], token,
                                            spec["cellsize"],
                                            min_density=spec.get("min_cell_density"))
            try:
                res.rasters.append(_stage_rasterize(ortho_path, canonical, out, spec))
            except Exception as ex:
                log(f"[express] rasterize {tag} failed: {ex}")
        _set("Rasterize", "done" if res.rasters else "error")
    else:
        _set("Rasterize", "skipped")

    # Map
    _set("Map", "running")
    mdir = layout.project_path(project, "maps")
    mdir.mkdir(parents=True, exist_ok=True)
    log(f"[express] Rendering {len(plan.maps)} map(s)…")
    for i, spec in enumerate(plan.maps):
        if stop(): break
        if spec["layer"] == "vector":
            src = canonical
        else:
            # The raster this map is about: the density surface for a
            # density map, else the first (D50) surface.
            want = "_density_" if spec["field"] == "density" else None
            src = next((r for r in res.rasters if want and want in Path(r).name),
                       res.rasters[0] if res.rasters else None)
        if src is None:
            continue
        base = (Path(src).stem if spec["layer"] == "raster"
                else f"{naming.run_stem(Path(src).name)}_xprs_{spec['field']}")
        out = mdir / naming.map_name(base)
        # A re-run overwrites its map like every other Express output; only
        # two maps of the same run may not share a name.
        if any(m == out for m in res.maps):
            out = mdir / naming.map_name(f"{base}_{i}")
        try:
            res.maps.append(_stage_map(ortho_path, src, out, spec))
        except Exception as ex:
            log(f"[express] map {i} failed: {ex}")
    _set("Map", "done" if res.maps else ("skipped" if not canonical else "error"))

    # Report (always attempt — build_pdf no-ops sections without inputs)
    _set("Report", "running")
    try:
        repdir = layout.project_path(project, "reports")
        repdir.mkdir(parents=True, exist_ok=True)
        log("[express] Building the PDF report…")
        # build_pdf walks the project root for its inventory, so it needs the
        # absolute directory, not the bare project name.
        project_root = layout.project_path(project)
        out_pdf = repdir / f"{project_root.name}_xprs_report.pdf"
        res.report_pdf = _stage_report(project_root, out_pdf, plan.report_options, log)
        _set("Report", "done")
    except Exception as ex:
        _set("Report", "error", str(ex))
        log(f"[express] report failed: {ex}")
    return res
