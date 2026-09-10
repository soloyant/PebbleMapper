# PebbleMapper API reference

The GUI is a thin layer over the `functions` and `detectors` packages; everything a tab does can be called from a script or a notebook. This page lists the public entry points in pipeline order. Each signature is copied from the `def` line of the module named in the heading. Paths are accepted as `str` or `pathlib.Path`. Lengths are in metres, areas in m² and angles in degrees unless stated otherwise.

Related pages: [the clast table](../user-manual.md#the-clast-table) (the per-clast table every stage exchanges), [custom equations](../user-manual.md#custom-equations), [project folders](../user-manual.md#project-folders), [adding a detection model](adding-a-detection-model.md).

---

## 1. Environment and imports

Scripts run from the repository root (the directory holding `functions/`, `detectors/`, `mrcnn/`) in the `maskrcnn` conda environment:

```bash
conda activate maskrcnn
cd /path/to/pebblemapper
python my_script.py
```

From another directory, put the repository root on `sys.path` first (`sys.path.insert(0, r"C:\path\to\pebblemapper")`).

- Importing `functions` sets GDAL, OGR and OSR to the return-None error model: `gdal.Open(bad_path)` returns `None` and does not raise.
- `functions.clasts_detection` and `detectors.base` import TensorFlow and the vendored `mrcnn` package. Every other module is TensorFlow-free.
- `functions.layout` resolves the **datasets root** once at import: the `PEBBLEMAPPER_DATASETS_ROOT` environment variable (legacy alias `CSM_DATASETS_ROOT`), then `datasets_root` in `<app home>/config.json`, then `<repo>/datasets`. The app home is `~/.pebblemapper`, or `PEBBLEMAPPER_HOME` when set. Set the variable before importing `functions.layout`.
- Mask R-CNN weights are looked up at `<repo>/models/maskrcnn/mask_rcnn_clasts.h5`, then `<repo>/model_weights/mask_rcnn_clasts.h5`.
- Photographs (JPEG, PNG, TIFF, HEIC/HEIF) are read through Pillow. Importing `functions.images`, or any module that reads photographs (`gauge`, `orthorectify`, `clasts_detection`, `exif_seed`, `zonal_canvas`), registers the HEIC/HEIF opener. See section 8.
- Figure functions use `matplotlib.pyplot`. In a headless script call `matplotlib.use("Agg")` first and pass `plot=False` where the parameter exists.

---

## 2. Project layout and output names

### `functions.layout`

The registry of where a project's files live. Scripts that resolve paths through it write files the GUI and the report find. The tree under `<DATASETS_ROOT>/<project>/` (optionally `/<YYYY-MM-DD>/` for dated projects) is described in [project folders](../user-manual.md#project-folders); a legacy tree (`images/`, `results/`) is resolved transparently.

| Function | Signature | Returns |
|---|---|---|
| `get_datasets_root` | `get_datasets_root() -> Path` | the current datasets root |
| `set_datasets_root` | `set_datasets_root(path, persist: bool = True) -> Path` | the new root; `persist` saves it to `config.json` (an environment variable still wins at the next start) |
| `project_path` | `project_path(project: str, kind: str = "", date=None) -> Path` | a project folder: empty `project` gives the datasets root; empty or unknown `kind` gives the project root; `date=None` uses the active date if one is set for this project |
| `list_projects` | `list_projects() -> list[str]` | sorted directory names under the root |
| `list_dates` | `list_dates(project: str) -> list[str]` | date subfolders (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`); empty for a flat project |
| `is_dated_project` | `is_dated_project(project: str) -> bool` | |
| `set_active_date` / `get_active_date` | `set_active_date(date, project=None) -> None`, `get_active_date()` | the date `project_path` inserts when `date` is not given |
| `ensure_project_layout` | `ensure_project_layout(project: str, extra_kinds: Iterable[str] = (), date=None) -> None` | creates the canonical folders and a `README.txt`; idempotent; `ValueError` on an unsafe name |
| `resolve_project_subfolder_checked` | `resolve_project_subfolder_checked(project_root: Path, kind: str) -> Resolution` | `Resolution(path, how, problem)`; `.found` is `False` when the folder was guessed, `.unrecognised` when the tree matches no known layout |
| `resolve_project_subfolder` | `resolve_project_subfolder(project_root: Path, kind: str) -> Path` | the path only |
| `default_starting_dir` | `default_starting_dir(kind: str, project: str) -> str` | creates the folder for an existing project |
| `migrate_legacy_project` | `migrate_legacy_project(project: str, dry_run: bool = True) -> list[str]` | printable actions; nothing moves unless `dry_run=False` |
| `migrate_validation_structure` | `migrate_validation_structure(project: str, dry_run: bool = True, date=None) -> list[str]` | |
| `migrate_input_structure` | `migrate_input_structure(project: str, dry_run: bool = True, date=None) -> list[str]` | |

`PATH_KINDS`: `images`, `dem`, `geometries`, `vectors`, `checkpoints`, `rasters`, `figures`, `maps`, `zonal`, `gauge`, `reports`, `logs`, `detections` (alias of `vectors`), `validation`, `validation_images`, `validation_results`, `validation_raw`, `validation_rectified`, `validation_georectified`, `validation_gps`, `validation_reports` (alias of `validation_results`). For input kinds the first existing candidate is returned (`input_data/images`, then `input_data`, then `images`).

### `functions.naming`

Builds and parses output filenames. Every analysis output begins with an **origin** naming its source:

```
<origin> := [<project>[_<YYYYMMDD>]__]<image_stem>
```

The project (and, for dated projects, the date digits) precede a double underscore; each part is omitted when the image stem already contains it. A detection made by any model other than Mask R-CNN adds `_model=<id>` to the origin (`with_model`), so two models' runs on one photograph write two files. Later outputs inherit the token through `run_stem`; `image_stem` strips it. Processed images (rectified, georeferenced) keep the photograph's own stem.

| Function | Signature | Produces |
|---|---|---|
| `origin_stem` | `origin_stem(image_stem: str, project: Optional[str] = None, date: Optional[str] = None) -> str` | `Normandy_Etretat_20200610__DJI_0904` |
| `with_model` | `with_model(stem: str, model: Optional[str]) -> str` | `<stem>` for Mask R-CNN (or no model), else `<stem>_model=<id>`; an existing token is replaced |
| `model_of` | `model_of(name: str) -> str` | the model id in an output's name; `"maskrcnn"` when there is none |
| `detection_csv_name` | `detection_csv_name(stem: str, window, *, express: bool = False, run: bool = False) -> str` | `<stem>_ws<size>m.csv`; `run=True` gives the `.run.csv` checkpoint; `express=True` inserts `_xprs` |
| `quadrat_csv_name` | `quadrat_csv_name(stem: str) -> str` | `<stem>_individual_clasts.csv`; alias `terrestrial_csv_name` |
| `merged_csv_name` | `merged_csv_name(stem: str, *, express: bool = False) -> str` | `<stem>_merged.csv` / `<stem>_xprs_merged.csv` |
| `raster_name` | `raster_name(csv_stem: str, field: str, parameter: str, cellsize) -> str` | `<csv stem>_<field>_<parameter>_cellsize=<c>m.tif`; `cellsize` is formatted with `str()` (`1` → `cellsize=1m`) |
| `map_name` | `map_name(raster_or_csv_stem: str) -> str` | `<stem>_map.png` |
| `zonal_out_name` | `zonal_out_name(base_stem: str, zone_set: str, field: str, mode: str) -> str` | `<base>_zones=<set>[_field=<f>].<mode>.csv`; the field tag is added only when the base stem does not already contain the field |
| `zonal_map_name` | `zonal_map_name(raster_stem: str, zone_set: str) -> str` | `<raster stem>_zones=<set>.zonal_map.png` |
| `zone_token` / `field_token` | `zone_token(zone_set: str) -> str`, `field_token(field: str) -> str` | filename-safe tokens (`zone_token` strips a leading `zones_`) |
| `split_origin` / `strip_origin` | `split_origin(stem: str) -> tuple`, `strip_origin(stem: str) -> str` | `(prefix, image_part)`; prefix `""` for a legacy name |
| `parse_window_size` | `parse_window_size(name: str) -> Optional[float]` | metres, or `None` |
| `is_merged` / `is_express` | `is_merged(name: str) -> bool`, `is_express(name: str) -> bool` | |
| `run_stem` | `run_stem(name: str) -> str` | extension and package tokens removed, origin kept: the base a downstream output is named from |
| `image_stem` | `image_stem(name: str) -> str` | `run_stem` with the origin stripped |
| `same_image` | `same_image(name_a: str, name_b: str) -> bool` | whether two names come from the same source image |
| `parse_zonal_name` | `parse_zonal_name(name: str) -> Optional[dict]` | `{"mode", "base_stem", "zone_set", "field_tag"}` or `None` |

`window` is in metres, formatted with `:g` (`2.5` → `ws2.5m`, `5.0` → `ws5m`). The parsers also accept the legacy `_window_size=<s>m_individual_clast_values` and `_merged_individual_clast_values` forms. `_xprs` marks Express outputs. `TOOL_SUFFIXES = ("_rectified", "_corrected", "_georeferenced")` are the suffixes the rectify and georeference tools append.

```python
from functions import naming

origin = naming.origin_stem("DJI_0904", "Normandy_Etretat", "2020-06-10")
csv    = naming.detection_csv_name(origin, 2.5)   # Normandy_Etretat_20200610__DJI_0904_ws2.5m.csv
tif    = naming.raster_name(naming.run_stem(csv), "Clast_length", "D50", 1)
print(naming.parse_window_size(csv), naming.image_stem(csv))   # 2.5 DJI_0904
```

### `functions.project_defaults`

The per-project file finders every tab pre-fills from. Each takes the project name and returns `Path` objects, newest first.

| Function | Returns |
|---|---|
| `merged_csvs(project)`, `detection_csvs(project)`, `rasters(project)`, `orthos(project)`, `photos(project)`, `raw_photos(project)`, `rectified_photos(project)`, `validation_images(project)`, `validation_csvs(project)`, `georectified_images(project)`, `transect_csvs(project)` | files of that kind (`raw_photos`: `validation/raw`, else `input_data/images`; `rectified_photos`: `validation/orthorectified`, else `input_data/images/orthorectified`) |
| `best_clast_csv(project)` | the newest merge, else the newest detection CSV, else `None` |
| `source_ortho_for(name, project)` | the ortho a CSV or raster name was made from |
| `window_csvs_by_stem(project)`, `mergeable_stems(project)` | per-window CSVs grouped by image, and the images with more than one |
| `for_truth(truth_csv, candidates)` | among `candidates`, the file made from the photograph a `<stem>_truth.csv` was digitised on (`<stem>.<ext>`, `<stem>_georeferenced.tif`, `<project>__<stem>_individual_clasts.csv`); `None` when nothing matches |

---

## 3. Detection

### `functions.clasts_detection.clasts_detect_jobs`

Runs Mask R-CNN and writes the canonical per-clast table (the 15 columns of `_CLAST_COLUMNS`; see [the clast table](../user-manual.md#the-clast-table)). **Ortho** mode processes a georeferenced ortho-image in tiles; **Quadrat** mode processes any photograph of known scale whole, including a quadrat photographed from a drone.

```python
clasts_detect_jobs(mode, jobs, resolution=0.001, metric_cropsize=1,
                   plot=True, saveplot=False, saveresults=False,
                   devicemode="gpu", devicenumber=0, ksaveint=1,
                   min_confidence=None, overlap=0.0, dedup_method='iou',
                   dedup_overlap=None, stop_check=None,
                   output_dir=None, figures_dir=None,
                   progress_callback=None,
                   dark_threshold=None, bright_threshold=None,
                   nodata_max_frac=0.95, roi_path=None,
                   save_checkpoints=None, checkpoint_dir=None,
                   clean_checkpoints_on_completion=None)
```

| Parameter | Meaning |
|---|---|
| `mode` | `'ortho'` or `'quadrat'`; `'uav'` and `'terrestrial'` are aliases |
| `jobs` | one dict per image: `path` (required), `kstart` (Ortho tile to start at, default 0), `out_stem` (output stem, default the image stem; pass `naming.origin_stem(...)` for origin-prefixed names), `roi_path` (overrides the global ROI) |
| `resolution` | m/px of a quadrat photograph; in Ortho mode the GeoTransform supplies it |
| `metric_cropsize` | Ortho window size in m; tile side in pixels is `metric_cropsize / resolution` |
| `plot` | calls `plt.show()` per image or tile; set `False` in scripts |
| `saveplot` | writes the diagnostic PNGs below |
| `saveresults` | **must be `True` for the final CSV, log and figures to be written**; the `.run.csv` checkpoint and `.run.grid.json` are written, and the DataFrames returned, either way |
| `devicemode`, `devicenumber` | `"gpu"` or `"cpu"`, and the device index |
| `ksaveint` | ignored |
| `min_confidence` | confidence floor (0–1) built into the graph; `None` keeps the config value `0` (every detection kept). GUI presets use 0.60–0.80 |
| `overlap` | Ortho tile overlap in `[0, 0.95)`; stride is `cropsize * (1 - overlap)` |
| `dedup_method` | `'iou'` (ellipse intersection-over-union) or `'centroid'` (distance as a fraction of mean length); see `dedup_clasts` |
| `dedup_overlap` | threshold for removing tile-overlap duplicates after a clean Ortho completion; `None` disables it. The GUI passes 0.30 |
| `stop_check` | polled between jobs and tiles; returning `True` stops cleanly and keeps the checkpoint |
| `output_dir`, `figures_dir` | CSV and PNG directories; default the image's own directory |
| `progress_callback` | `progress_callback(job_index, event, payload)` with `event` in `"start"`, `"done"` (`payload["n_clasts"]`), `"stopped"`, `"error"` (`payload["message"]`, `payload["traceback"]`) |
| `dark_threshold`, `bright_threshold` | Ortho tile filter (0–255): pixels whose mean RGB is below / above are nodata; `None` (or 0 / 255) disables |
| `nodata_max_frac` | a tile whose nodata fraction exceeds this is skipped; uniform tiles (fewer than 3 distinct values) are always skipped |
| `roi_path` | GeoJSON of up to 1000 `Polygon`/`MultiPolygon` features. Ortho: world CRS; a tile runs when its centre is inside. Quadrat: image pixels `(col, row)`; a clast is kept when its centroid is inside. An unreadable ROI logs a warning and filters nothing |
| `save_checkpoints`, `checkpoint_dir`, `clean_checkpoints_on_completion` | deprecated, ignored |

Returns `list[pandas.DataFrame]`, one per job in input order; a failed job, or one stopped before it started, gives an empty frame. `x`, `y` are in the ortho's CRS (Ortho) or image pixels with `y` upward from the bottom edge (Quadrat).

Files written with `saveresults=True`:

| Mode | File | Content |
|---|---|---|
| quadrat | `<stem>_individual_clasts.csv` | per-clast table |
| quadrat, `saveplot` | `<figures_dir>/<stem>_overlay.png`, `<stem>_histogram.png` | mask outlines filled with the length colour, major axis white, minor pink; size histogram |
| ortho | `<stem>_ws<size>m.csv` | per-clast table, written on completion, stop and error |
| ortho | `<stem>_ws<size>m_detection_log.txt` | append-mode log: outcome, parameters, tile grid, cumulative elapsed time |
| ortho | `<stem>_ws<size>m.run.csv`, `.run.grid.json`, `.run.contours.jsonl` | checkpoint, its grid (overlap, tile and stride in px, tile counts), its outlines |
| ortho, on exception | `<stem>_ws<size>m.crash.json` | tool version, last tile, error, whether out-of-memory, clasts recovered |
| ortho, `saveplot` | `<figures_dir>/tiles/<stem>_tile_k=<k>_n=<n>_m=<m>.png` | one PNG per tile |

**Checkpoint and resume.** Each tile's clasts are appended to `<stem>_ws<size>m.run.csv` (canonical columns plus `_tile_idx`; an empty tile writes a marker row with a blank `clast_ID`). If the file exists when a run starts, the run resumes from `max(kstart, last_tile + 1)` with the saved clasts; a truncated file is salvaged line by line, and a checkpoint made with another overlap is discarded. A clean completion deletes it; delete it yourself to force a fresh run. `inspect_run_state(output_dir, image_path, metric_cropsize, out_stem=None, overlap=None) -> dict` reports `exists`, `path`, `last_tile`, `n_clasts`, `resume_k`, `grid_mismatch` (`None` when the checkpoint fits) without running anything.

**GPU out-of-memory.** The per-tile detection caps are halved and the model rebuilt, down to the stock floor; then a `RuntimeError` suggests CPU mode or a smaller window.

**Model cache.** The model is built once per `(devicemode, devicenumber, min_confidence)`. `get_model_status() -> dict` (`loaded`, `key`, `loaded_at`, `age_s`) and `clear_model_cache() -> None` (frees the GPU) are public.

`clasts_detect_jobs` writes no manifest. Call detection through the registry to get `<csv>.manifest.json`.

### `detectors` (registry and backends)

A backend implements `detectors.base.DetectorBackend` and produces the canonical table; downstream stages do not depend on which model ran. See [adding a detection model](adding-a-detection-model.md).

| Name | Signature | Meaning |
|---|---|---|
| `detectors.get_backend` | `get_backend(name: str) -> Optional[DetectorBackend]` | `None` for an unknown name |
| `detectors.available_backends` | `available_backends() -> List[DetectorBackend]` | backends whose `is_available()` is true |
| `detectors.all_backends` | `all_backends() -> List[DetectorBackend]` | every registered backend |
| `detectors.default_backend_name` | `default_backend_name() -> str` | `"maskrcnn"` |
| `detectors.register` | `register(backend: DetectorBackend) -> None` | adds a backend built in your own code |
| `detectors.registry.load_user_backends` | `load_user_backends(path=None) -> List[str]` | imports the backends declared in `PEBBLEMAPPER_DETECTORS` or `<repo>/user_detectors.json` (`{"module", "factory", "path"}` entries); runs at import; an unavailable backend logs its `install_hint` |
| `detectors.run_detect_jobs` | `run_detect_jobs(backend, mode, jobs, **kwargs) -> List` | calls `backend.detect_jobs`, guarantees one terminal `progress_callback` event per job, and discards a stale manifest beside a rewritten CSV. The GUI calls backends through this |
| `detectors.output_csv_name` | `output_csv_name(mode, stem, window=None) -> str` | `<stem>_ws<window>m.csv` (ortho) or `<stem>_individual_clasts.csv` (quadrat) |
| `detectors.measure.measure_mask` | `measure_mask(mask, score, resolution) -> dict or None` | the shared per-instance measurement |
| `detectors.measure.measure_instances` | `measure_instances(instances, resolution, *, height=None, geotransform=None, image=None, roi_paths=None, log_fn=None, csv_path=None) -> DataFrame` | measures instances into the canonical table in world coordinates; outlines go to `df.attrs["contours"]` and, with `csv_path`, to `<csv stem>.contours.json` |
| `detectors.subprocess_runner.run_backend` | `run_backend(env, script, spec, *, conda_exe=None, timeout=None, log_fn=None, stream=False, stop_check=None) -> CompletedProcess` | `conda run` a subprocess backend; `stream=True` forwards output to `log_fn` and kills the process tree (raising `BackendStopped`) when `stop_check()` turns true |
| `detectors.subprocess_runner.read_instances_npz` | `read_instances_npz(path, pad=2) -> List[Instance]` | reads a `<csv>.instances.npz` (`labels` int32 H×W, `scores` float32 N) into bounding-box masks |
| `detectors.CANONICAL_COLUMNS` | `list[str]` | the 15 column names, in order |
| `DetectorBackend.info` | `BackendInfo` | `name`, `display_name`, `framework`, `license`, `output_type`, `env`, `in_process`, `weights`, `install_hint`, `description`, `dev_only` (hidden from the model list unless `PEBBLEMAPPER_SHOW_DEV_BACKENDS=1`), `version` (recorded in Digitize's provenance sidecar) |
| `DetectorBackend.is_available` | `is_available(self) -> bool` | |
| `DetectorBackend.detect_jobs` | `detect_jobs(self, mode: str, jobs: list, **kwargs) -> List` | the arguments of `clasts_detect_jobs`, plus `log_fn` and `timeout` (read by subprocess backends) |

Shipped backends: `"maskrcnn"` (`detectors.maskrcnn.MaskRCNNBackend`, in-process; calls `clasts_detect_jobs` and writes `<csv>.manifest.json` beside every CSV the run produced) and `"stub"` (`detectors.stub_backend.StubBackend`, a cross-environment proof, available when the `pebble-stub` conda environment exists).

The manifest (`detectors.base.DetectionManifest`) holds `model`, `model_version`, `weights`, `params` (the detection keyword arguments), `crs`, `gsd_m`, `n_tiles`, `tool_version`, `license`, `timestamp`. It is written only beside a CSV newer than the run start, so a failed re-run does not refresh an older result's manifest.

```python
import matplotlib; matplotlib.use("Agg")
from functions import layout, naming
from detectors import get_backend

backend = get_backend("maskrcnn")
project = "Normandy_Etretat"
ortho = layout.project_path(project, "images") / "DJI_0904.tif"
origin = naming.origin_stem(ortho.stem, project, layout.get_active_date())

frames = backend.detect_jobs(
    "ortho", [{"path": str(ortho), "kstart": 0, "out_stem": origin}],
    metric_cropsize=2.5, overlap=0.20, min_confidence=0.70, dedup_overlap=0.30,
    plot=False, saveresults=True,
    output_dir=str(layout.project_path(project, "vectors")))
```

### `functions.gauge`

The engine behind the Digitize tab's Detect button and its *No GSD? Scale from an object* option: clast sizes from one photograph, scaled by its GSD (`GaugeScale.from_gsd`) or by scaling objects drawn on it. The detector runs on the whole frame in pixel units (`clasts_detect_jobs(mode="quadrat", resolution=1.0)`) and every length is divided by a pixels-per-unit scale. An object scale applies no perspective or lens correction; `DISCLAIMER` states the expected error and is printed on the figures and stored in the sidecar.

The scaling-object library is `<project root>/scaling_objects.json`, a list of `{"name", "length", "unit"}` where `length` and `unit` may be `null` (size unknown).

| Name | Signature | Meaning |
|---|---|---|
| `ScalingObject` | dataclass `(name, length=None, unit=None)` | `.metres` (`None` when unknown), `.is_metric`, `.label()`, `.to_dict()`, `ScalingObject.from_dict(d)` (`None` without a name) |
| `load_library` | `load_library(project_root) -> list[ScalingObject]` | the file's objects, malformed entries skipped, duplicates dropped; `default_library()` (one *scale bar* of unknown size) when the file is absent, unreadable or empty |
| `save_library` | `save_library(project_root, objects) -> Path` | writes `ScalingObject`s or dicts; an empty library is written as the default |
| `library_path`, `find_object`, `default_library` | | helpers |
| `GaugeSegment` | dataclass `(p0, p1, object_name="")` | one drawn segment in image pixels; `.px_length` |
| `result_unit_options` | `result_unit_options(segments, library) -> dict` | `{"mode", "options", "default", "unknown", "missing"}`: `"metric"` when every object used has a known length (options `mm cm m in ft`, default mm), else `"custom"` (options: the unknown objects used). With no segments, metric when any library object has a known length |
| `resolve_scale` | `resolve_scale(segments, library, result_unit=None) -> GaugeScale` | metric: the mean pixels per metre over segments, `spread = (max − min) / mean`; custom: the unit is one unknown object (`result_unit`, else the first drawn), other objects go to `.ignored` with a `.note`. `ValueError` for no segment, a sub-pixel segment, an unknown object or a bad `result_unit` |
| `scale_from_segments` | `scale_from_segments(segments_px, object_length=1.0) -> dict` | single-object form: `{"px_per_unit", "per_segment", "spread", "n"}` |
| `GaugeScale` | dataclass | `object_name`, `object_length`, `unit` (a `UNITS` key or `"custom"`), `px_per_unit`, `spread`, `segments_px`, `mode`, `segments`, `ignored`, `note`; properties `unit_label`, `is_metric`, `metres_per_unit`, `metres_per_px` (`None` for custom), `valid`; `GaugeScale.from_segments(...)`; `GaugeScale.from_gsd(metres_per_px, unit="mm")` (`ValueError` for a non-positive GSD or unknown unit); `.to_dict()` |
| `run_gauge` | `run_gauge(image_path, scale, out_dir, *, min_confidence=0.7, devicemode="gpu", devicenumber=0, log_fn=None, stop_check=None, detect_fn=None, out_stem=None, model="maskrcnn", source_image=None, detect_scale=1.0, scale_source=None, disclaimer=DISCLAIMER, exclude_segments=True, segments=None, exclude_frame=True, write=True) -> dict` | see below |
| `backend_detect_fn` | `backend_detect_fn(backend, work_dir, *, log_fn=None, keep=None, timeout=None)` | a `detect_fn` running any registered backend in Quadrat mode through `run_detect_jobs`, keeping its masks (or outlines) in `keep` for Digitize's proposals |
| `resolve_detect_scale` | `resolve_detect_scale(label, model_name: str = "maskrcnn") -> float` | the factor a detection-scale label stands for; `"Auto"` is 0.5 for Mask R-CNN and 1 for other models |
| `DETECT_SCALES`, `DETECT_SCALE_OPTIONS`, `DETECT_SCALE_DEFAULT`, `DETECT_SCALE_HINT` | | `{"1": 1.0, "1/2": 0.5, "1/3": 1/3, "1/4": 0.25}`; `"Auto"` plus those labels; `"Auto"`; the GUI's one-line explanation |
| `is_detector_summary_line` | `is_detector_summary_line(line) -> bool` | the detector's centimetre summary line, which Digitize's console drops |
| `gauge_stats` | `gauge_stats(df, scale, column="Clast_length") -> dict` | `n, D16, D50, D84, mean` in the unit, `unit`, `column`; `sorting_phi` (Folk–Ward) in metric mode with n ≥ 2 |
| `gauge_overlay_figure` | `gauge_overlay_figure(image_path, df, scale, *, n_labels=8, cmap="viridis", stats=None, library=None, disclaimer=DISCLAIMER, contours=None, note=None) -> Figure` | the photograph with clast outlines coloured by `Clast_length`, major (white) and minor (pink) chords, labelled segments, `n_labels` annotated clasts, an n / D50 / D84 box, a scale bar in metric mode, and `disclaimer` below |
| `gauge_distribution_figure` | `gauge_distribution_figure(df, scale, *, column="Clast_length", disclaimer=DISCLAIMER) -> Figure` | log-x histogram with KDE and D16 / D50 / D84, and the empirical CDF; `ValueError` below two values |
| `write_gauge_figures` | `write_gauge_figures(image_path, df, scale, out_dir, stem, *, dpi=200, n_labels=8, stats=None, library=None, log_fn=None, disclaimer=DISCLAIMER, contours=None, note=None) -> dict` | `{"overlay", "distribution"}`: `<stem>_gauge_overlay.png`, `<stem>_gauge_distribution.png` (`None` below two clasts) |
| `format_length` | `format_length(value, unit_label="", *, short=False) -> str` | `"63.2 mm"`, `"0.71 boot width"` |
| `open_photo` | `open_photo(path) -> PIL.Image` | `functions.images.open_photo(path, upright=True)` |
| `pool_sample_set` | `pool_sample_set(entries) -> (pooled, per_photo, notes)` | pools several gauge CSVs (`entries`: dicts with `csv`, `scale`, …) into one sample set |
| `sample_set_figure` | `sample_set_figure(pooled, *, unit=None, …) -> Figure` | the pooled distribution |
| `write_sample_set` | `write_sample_set(entries, out_dir, stem, *, …) -> dict` | writes the pooled CSV, per-photograph table and figure, named by `sample_set_names(stem) -> dict` |
| `needs_working_copy`, `working_copy`, `prepare_photo` | `prepare_photo(path, out_dir) -> str` | from `functions.images`: the photograph itself, or, when its EXIF orientation is not upright, `<out_dir>/<stem>.jpg`, an upright RGB JPEG (quality 95). A HEIC never needs the copy |

`run_gauge` detects, drops detections crossing a scale segment (`exclude_segments`) or lying on the quadrat frame band (`exclude_frame`), and divides `LENGTH_COLUMNS` by px/unit and `AREA_COLUMNS` by (px/unit)², leaving `x`/`y` in pixels (`y` upward) and adding a `unit` column. It writes `<stem>_gauge.csv`, `<stem>_gauge.json` and `<stem>_gauge.contours.json`, and returns `{"csv", "json", "df", "stats", "stem", "out_dir", "contours", "model", "excluded_by_segments", "removed_ids", "excluded_by_frame", "frame_removed_ids", "frame", "sidecar"}`. `model` is recorded under `model.backend` in the sidecar; `detect_fn` replaces `clasts_detect_jobs`; `source_image` is the user's photograph when `image_path` is its working copy. `detect_scale` in (0, 1] resamples the photograph (LANCZOS, temporary `<stem>_gauge_detect.jpg`) and detects at `resolution = 1 / detect_scale`, so results return in original pixels. `scale_source` (`"object"`, `"gsd:filename"`, `"gsd:sidecar"`) and `disclaimer` (`None` for a GSD-scaled run) go into the sidecar. Raises `RuntimeError` on a detector error or a stop. `UNITS` maps `mm cm m in ft` to metres.

---

## 4. Merging window sizes

### `functions.clasts_merge`

Combines detections of the same ortho at different window sizes into one table and removes near-duplicates. No TensorFlow import.

```python
merge_csvs(input_filepath_small, input_filepath_large, output_filepath,
           method='iou', overlap=0.30, n_points=32)
```

| Parameter | Meaning |
|---|---|
| `input_filepath_small`, `input_filepath_large` | detection CSVs from the smaller and larger window; the larger wins conflicts |
| `output_filepath` | merged CSV to write |
| `method` | `'iou'`: conflict when ellipse IoU ≥ `overlap`; `'centroid'`: conflict when centroid distance < `overlap` × mean length |
| `overlap` | IoU threshold or length fraction |
| `n_points` | vertices per ellipse polygon (`'iou'` only) |

Returns the merged `DataFrame`, also written (`float_format='%.5f'`). Rows with a non-positive `Clast_length`, `Clast_width`, `Ellipse_major_axis` or `Ellipse_minor_axis` are dropped first; `clast_ID` is renumbered from 1; canonical columns come first. For more than two windows, fold pairwise: the Merge tab starts from the largest window and merges each smaller CSV into the accumulator (`n_points=24`); Express starts from the smallest and merges upward with `("iou", 0.30, 32)`. Name the result `naming.merged_csv_name(naming.run_stem(<largest csv>))`.

```python
dedup_clasts(df, method='iou', overlap=0.30, priority_col=None,
             n_points=32, search_radius_factor=1.0, return_kept_mask=False)
```

Greedy keep-the-best scan over a KD-tree. `df` needs `x`, `y`, `Clast_length`, `Clast_width`, `Orientation` (`Score` breaks ties). `priority_col` names a column whose larger values win; `search_radius_factor` widens the neighbour query for elongated clasts. Returns the deduplicated frame (and the kept mask on request).

---

## 5. Rasterizing and custom fields

### `functions.clasts_rasterize.clasts_rasterize`

Aggregates a per-clast CSV into a per-cell statistic GeoTIFF over the ortho it was detected on.

```python
clasts_rasterize(ClastImageFilePath, ClastSizeListCSVFilePath,
                 RasterFileWritingPath, field="Clast_length",
                 parameter="quantile", cellsize=1, percentile=0.5,
                 plot=True, figuresize=(15, 20), T=10,
                 bin_edges=None, bin_mode="fixed",
                 bin_field="Clast_length",
                 min_cell_density=0,
                 rho_water: float = 1025.0)
```

| Parameter | Meaning |
|---|---|
| `ClastImageFilePath` | source GeoTIFF; supplies the CRS and the diagnostic figure's RGB |
| `ClastSizeListCSVFilePath` | detection or merged CSV; rows with null or non-positive `Clast_length`/`Clast_width` are dropped |
| `RasterFileWritingPath` | output `.tif`; use `naming.raster_name` |
| `field`, `parameter` | what to aggregate and how (lists below) |
| `cellsize` | output cell size in CRS units (m) |
| `percentile` | quantile (0–1) for `parameter="quantile"`; 0.5 gives D50 |
| `plot`, `figuresize` | build the ortho/raster diagnostic figure (not shown or saved); size in inches |
| `T` | wave period in s for `Leroux_wave_orbital_velocity` |
| `bin_edges` | size-bin edges (m, or quantiles); makes `packing_index` one band per bin; required by `packing_clustering` |
| `bin_mode` | `"fixed"` (edges in metres) or `"percentile"` (edges are quantiles of `bin_field` over the CSV) |
| `bin_field` | column to bin on; `Surface_area` is always what is summed |
| `min_cell_density` | cells with fewer clasts are left empty |
| `rho_water` | water density in kg/m³ for transport thresholds (1025 sea, 1000 fresh) |

Returns the grid as a NumPy array: `(n_rows, n_cols)`, or `(n_rows, n_cols, n_bands)` for `distribution`, `d_percentiles` and binned `packing_index`.

`field`: any CSV column (including addon columns), or a derived quantity: `Clast_elongation` (width/length), `Ellipse_elongation`, `Clast_circularity` (equivalent diameter/length), `Ellipse_circularity`, `Van_Rijn_dimensionless_diameter`, `Soulsby_critical_shields`, `Shields_critical_shear_stress` (Pa), `Shields_critical_shear_velocity` (m/s), `Shields_critical_grain_reynolds_number`, `Hjulstrom_deposition_velocity` (m/s), `Hjulstrom_erosion_velocity` (m/s), `Leroux_wave_orbital_velocity` (m/s). `Orientation` is treated as axial (doubled angle, vector mean).

`parameter`: `quantile`, `density` ((clast count + 1) per cell area; `field` must name a real column), `average`, `std`, `cv`, `mode`, `skewness`, `kurtosis`, `sorting` (Folk-Ward sorting in φ), `folk_ward_sorting`, `folk_ward_skewness`, `folk_ward_kurtosis`, `distribution` (99 bands, D1…D99), `d_percentiles` (D5, D16, D50, D84, D95), `packing_index`, `packing_clustering` (Shannon evenness across bins).

Files written: the GeoTIFF (Float64, north-up geotransform `[x_min, cellsize, 0, y_top, 0, -cellsize]` from the clast coordinates, source CRS, NoData 0, band descriptions on multi-band outputs) and `<tif>.json` with `schema_version, rasterize_version, generated_at, raster_path, parameter, n_bands, source_image, source_csv, field, cellsize_m, percentile` (quantile only)`, bin_mode, bin_field, bin_edges_m, wave_period_s, min_cell_density`.

The Map tab and the report parse `field`, `parameter` and `cellsize` from the filename. The Rasterize tab writes `<csv stem>_<field>_D<percentile×100>_cellsize=<c>m.tif` for quantiles, `<csv stem>_<field>_<parameter>_cellsize=<c>m.tif` otherwise, and `<csv stem>_density_cellsize=<c>m.tif` for density. Express uses the same names on the `_xprs` merged stem, except that its density raster carries the field token. Use the `D50` form for a raster the report should label as a median.

Addons are applied before aggregation, loaded in this order (later wins on a name clash): `<repo>/hydraulic_addons.json`, `<csv dir>/../../addons.json` (the project root for a CSV in `output_results/vectors/`), `<repo>/user_addons.json`.

### `functions.addons`

User-defined per-clast equations from JSON, evaluated by a whitelist AST walker (no `eval`). The JSON format is described in [custom equations](../user-manual.md#custom-equations).

| Name | Signature | Meaning |
|---|---|---|
| `load_addons` | `load_addons(*paths: str | Path) -> list[Addon]` | reads each existing file and skips invalid entries with a warning; de-duplicated by `name`, later paths override. Registers each addon's `units` and `factor` with `functions.units.register_field_unit` |
| `apply_addons_to_dataframe` | `apply_addons_to_dataframe(df, addons: list[Addon])` | adds one column per addon; mutates and returns `df`; a row that raises gets NaN; an addon with a missing input column is skipped |
| `Addon` | dataclass `name, inputs, expression, units, description, factor` | `validate() -> bool`, `evaluate(row: dict) -> float` |

Keys: `name` (required), `inputs`, `expression`, `units`, `description`, `factor` (default 1.0, stored-to-display multiplier). Expressions allow arithmetic, comparisons, conditional expressions and calls to `sqrt, log, log2, log10, exp, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, floor, ceil, fabs, abs, min, max, round, pow` (`pow` exponent ≤ 64).

---

## 6. Publication maps

### `functions.map_export`

Renders a detection CSV (points) or a raster product (cells) over the ortho, with an optional web basemap, north arrow, scale bar and colorbar, and records what was plotted.

```python
make_publication_map(
    *, ortho_tif: str, clast_csv: Optional[str] = None, raster_tif: Optional[str] = None,
    layer: str = "vector", field: str = "Clast_length", cmap: str = "viridis",
    basemap: Optional[str] = "Esri.WorldImagery", show_ortho: bool = True,
    ortho_alpha: float = 0.7, raster_alpha: float = 0.85, show_grid: bool = True,
    show_legend: bool = True, show_crs_info: bool = True, show_north_arrow: bool = True,
    show_scale_bar: bool = True, show_zebra_border: bool = True, title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10), point_size: float = 8.0,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    color_scale: str = "quantile", quantile_low: float = 0.02, quantile_high: float = 0.98,
    linear_low_pct: float = 0.05, linear_high_pct: float = 0.95,
    colorbar_label: Optional[str] = None, show_colorbar_extends: bool = True,
    extent_padding: float = 0.05, unit_system: str = "metric", size_unit: str = "auto",
    log_fn: Optional[callable] = None, extent: str = "project", extent_margin: float = 0.10,
    classes: Optional[int] = None, uncertainty: Optional[float] = None,
    allow_rainbow: bool = False, parameter: Optional[str] = None,
    return_metadata: bool = False)
```

All arguments are keyword-only. The `show_*` flags toggle decorations; `ortho_alpha`, `raster_alpha`, `figsize`, `point_size` and `title` (`None` derives one) set appearance.

| Parameter | Meaning |
|---|---|
| `ortho_tif` | GeoTIFF supplying the georeferencing and, with `show_ortho`, the background |
| `layer`, `clast_csv`, `raster_tif`, `field` | `"vector"` colours the points of `clast_csv` by `field`; `"raster"` draws `raster_tif`, a `clasts_rasterize` output |
| `cmap` | matplotlib colormap (`DEFAULT_COLORMAPS` lists the GUI's). Rainbow maps (`RAINBOW_COLORMAPS`) are replaced by viridis unless `allow_rainbow`; `Orientation` always uses `twilight` on a fixed 0–180° scale |
| `basemap` | xyzservices provider (values of `DEFAULT_BASEMAPS`) or `None`; needs network access; a failed fetch is logged and skipped |
| `vmin`, `vmax`, `color_scale` | colour limits; when unset, `"quantile"` clips at `quantile_low`/`quantile_high` and `"linear"` at `linear_low_pct`/`linear_high_pct` (0 and 1 give the true extremes) |
| `colorbar_label` | `None` builds one from the field or raster name with its unit |
| `unit_system`, `size_unit` | `"metric"`/`"imperial"`; `"auto"` picks m/cm/mm (ft/in) from the data magnitude; display only |
| `extent`, `extent_padding`, `extent_margin` | `"project"`: union of the ortho and the data extent padded by `extent_padding` per side; `"raster"` (alias `"data"`): the valid-data bounding box plus `extent_margin` per side |
| `classes` | `None` continuous; an integer ≥ 2 gives discrete classes (quantile-spaced or equal-interval per `color_scale`); ignored for axial fields |
| `uncertainty` | uncertainty in display units; colorbar labels are rounded to the digit it supports |
| `parameter` | a raster's per-cell parameter (`D50`, `sorting`, ...); `None` reads it from the filename. Sets the unit (a `sorting` raster stores φ) |
| `return_metadata` | `True` returns `(fig, metadata)`; the dict is always attached as `fig.map_metadata` |
| `log_fn` | receives `"[map] ..."` strings |

Returns the `Figure` (not saved, not closed). `fig.map_metadata` holds `field, parameter, layer, cell_size, unit, unit_factor, cmap, classification, class_edges, classes, color_scale, cyclic, uncertainty, value_min/max, data_min/max, n_valid, n_total, crs, crs_name, extent_mode, extent, data_extent, colorbar_label, source, ortho, basemap, basemap_added, exporter_version, exported_at`.

```python
save_publication_map(fig, out_path, *, dpi: int = 300, sidecar: bool = True, **savefig_kwargs) -> dict
```

Saves the figure (`bbox_inches="tight"` unless overridden; `.png`, `.pdf` or `.svg` by extension) and, with `sidecar`, writes `<out_path>.json` with `fig.map_metadata` plus `output`, `dpi`, `format`, `sidecar`. Returns the metadata written. Helpers: `metadata_sidecar_path(out_path) -> Path`, `write_map_metadata(out_path, metadata: dict) -> Path`, `read_map_metadata(out_path) -> Optional[dict]`. The GUI and Express save with `fig.savefig` (no sidecar) to `output_results/maps/<stem>_map.png`.

---

## 7. Zonal statistics

### `functions.zonal_stats`

Summarises clasts or raster cells inside polygons, samples rasters along transects, and plots the results. Reads OGR vector layers (GeoJSON, shapefile, GeoPackage) and GDAL rasters. The vector layer must share the CRS of the raster or the clast coordinates: `describe_crs_mismatch(vector_path, raster_path) -> Optional[str]` returns a message when they differ. A mismatch raises nothing and yields an empty result.

**Polygons, from the clast CSV** (the Zonal tab's polygon mode):

```python
zonal_polygon_stats_from_csv(detection_csv, vector_path, out_csv, *,
    field_name: str = "Clast_length", id_field: str = "",
    percentiles: Iterable[int] = (5, 16, 25, 50, 75, 84, 95),
    x_col: str = "x", y_col: str = "y",
    dem_path: Optional[str | Path] = None) -> list[DistributionStats]
```

Non-polygon features are ignored. `id_field` names the attribute used as `polygon_id` (empty: the OGR FID). Percentiles are labelled `D<q>` for size fields, `median` and `P<q>` otherwise. `x_col`, `y_col` must be in the polygon layer's CRS. `dem_path` adds `dem_mean, dem_std, dem_min, dem_max, dem_range`.

Returns one `DistributionStats` per polygon in **stored** units (`count, area_m2, density, minimum, maximum, range, mean, median, std, cv, iqr, skewness, kurtosis, mean_phi, sigma_phi, sk_phi, kg_phi, percentiles, dem_*, coverage`). The CSV (`polygon_id, count, area_m2, density, mean, std, skewness, kurtosis, median, iqr, cv, min, max, range, mean_phi, sigma_phi, sk_phi, kg_phi, <percentiles>, [dem_*], coverage, field, unit`) is in **display** units (mm for lengths), matching the report. `coverage` is `"inside"` or `"outside"` (the polygon does not touch the data). Folk-Ward φ moments are NaN for non-size fields.

**Polygons, from a raster:**

```python
zonal_polygon_stats(raster_path, vector_path, out_csv, *,
    field_name: str = "value", band: int = 1, id_field: str = "",
    percentiles: Iterable[int] = (16, 50, 84),
    nodata: Optional[float] = None) -> list[PolygonStats]
```

Cells whose centre falls in each polygon are reduced to `count, mean, std` and the percentiles; `nodata=None` uses the raster's declared NoData. CSV columns: `polygon_id, band, count, mean, std, <percentiles>`.

**Transects:**

```python
zonal_transect_profile(raster_path, vector_path, out_csv, *,
    step_m: float = 0.5, band: int = 1, id_field: str = "",
    dem_path: Optional[str | Path] = None, interpolation: str = "bilinear",
    field_name: str = "") -> list[TransectSample]
```

Samples `raster_path` every `step_m` (CRS units) along each LineString (a MultiLineString is concatenated into one transect with cumulative distance). Declared NoData and 0 (the rasterizer's empty cell) become NaN; only the DEM's declared NoData is masked. `interpolation` is `"bilinear"` or `"nearest"`. The long-format CSV has `transect_id, distance_m, raster_value[, elevation_m][, field]` in the raster's **stored** unit; `field_name` fills the trailing `field` column, which the profile figure uses for its axis label. Returns `TransectSample(transect_id, distance_m, raster_value, elevation_m)` per feature.

**Plots:**

| Function | Signature | Writes |
|---|---|---|
| `plot_transect_profile` | `plot_transect_profile(sample: TransectSample, out_png, *, field_name: str = "value") -> Path` | value against distance, plus a DEM panel when present |
| `plot_zonal_map` | `plot_zonal_map(raster_path, vector_path, out_png, *, field_name: str = "Clast_length", polygon_id_field: str = "", transect_id_field: str = "", cmap: str = "viridis", title: str = "", figsize=(10.0, 10.0), dpi: int = 200, show_north_arrow: bool = True, show_scale_bar: bool = True, log_fn=None) -> Path` | the raster with labelled zones and transects |
| `plot_combined` | `plot_combined(layers: list[dict], out_png, *, title: str = "", primary_label=None, secondary_label=None, figsize=(9.0, 5.5), dpi: int = 160) -> Path` | several zonal CSVs on one axis (optional twin axis); layer dict: `csv, x_field, y_field, [y2_field], axis ("primary"/"secondary"), label, color, style ("line"/"envelope")` |

**Multi-date transect profiles:**

```python
load_profile_series(csv_paths, *, labels=None, distance_field="distance_m",
    value_field="raster_value", transect_id_field="transect_id",
    elevation_field="elevation_m") -> tuple[list[ProfileSeries], list[str]]
build_profile_outputs(csv_paths, out_dir, stem: str, *, labels=None,
    bin_width_m: float = 1.0, title: str = "", caption: str = "") -> dict
```

`load_profile_series` reads transect CSVs (the survey date comes from a `YYYY-MM-DD` path component; `labels` maps path to legend text) and raises `ValueError` when the inputs describe more than one field. `build_profile_outputs` writes `<stem>__profile.png` (grain size above, elevation below), `<stem>__summary.csv` (per transect and date, display units), `<stem>__change.csv` (consecutive-date differences) and `<stem>__binned.csv` (mean per `bin_width_m` bin), and returns `{"png", "summary", "change", "binned", "skipped", "series"}`. The report includes any `*__profile.png` in the zonal folder. Also public: `profile_summary_rows`, `profile_change_rows`, `profile_binned_rows(series, bin_width_m=1.0)`, `plot_profile_figure(series, out_png, *, title="", caption="", figsize=(9.0, 6.0), dpi=160)`.

GUI output names (under `output_results/zonal/`): polygon CSV `naming.zonal_out_name(<csv stem>, <vector stem>, field, "polygons")`; transect CSV `naming.zonal_out_name(<raster stem>, <vector stem>, field, "transects")`; per-transect PNG `<transect csv stem>__<transect_id>.png`; zonal map `naming.zonal_map_name(<raster stem>, <vector stem>)`; combined profile stem `transects_overlay`.

### `functions.zonal_canvas`

Serves the browser canvases. `prepare_browser_image(src_path: str, *, current_project: str = "", max_dim: int = 4000, cache_subdir: str = "canvas_cache") -> (png_path, disp_scale)` transcodes a GeoTIFF, a HEIC or an EXIF-rotated photograph to a cached, downsampled PNG in the stored pixel frame; `render_canvas_image(...)` (same arguments) returns a dict with the geotransform and display sizes.

---

## 8. Quadrat photographs: rectify and georeference

### `functions.images`

Photograph reading: which files are photographs, how HEIC/HEIF opens, and the pixel frame every measurement uses.

| Name | Signature | Meaning |
|---|---|---|
| `PHOTO_EXTENSIONS` | tuple | `.jpg .jpeg .png .tif .tiff .heic .heif`: what a folder of photographs may hold (Orthorectify, Digitize, Georeference, Validate) |
| `CAMERA_EXTENSIONS` | tuple | `.jpg .jpeg .png .heic .heif`: no TIFF (Detect's Quadrat listing and `project_defaults.photos`, where a TIFF means an ortho) |
| `HEIF_EXTENSIONS`, `is_heif`, `is_photo` | `is_heif(path) -> bool` | by file name |
| `register_heif_opener` | `register_heif_opener() -> bool` | registers pillow-heif with Pillow; `False` when the package is absent. Runs at import; `HEIF_AVAILABLE` holds the result |
| `heif_import_error`, `heif_failure_hint` | `heif_failure_hint(path) -> str` | `heif_import_error()` is why HEIC support is unavailable (`""` when available); `heif_failure_hint` is the text appended to a *could not read image* message for a HEIC/HEIF file (`""` for other files). The diagnostic bundle's `versions.json` records `pillow_heif`, `heif_available`, `heif_error` |
| `open_photo` | `open_photo(path, *, upright=False) -> PIL.Image` | a loaded image in the stored pixel frame; `upright=True` applies the EXIF orientation. `RuntimeError` naming pillow-heif for a HEIC without the package |
| `read_rgb` | `read_rgb(path) -> np.ndarray` | `H × W × 3` uint8 in the stored pixel frame; alpha dropped; 16-bit and float scaled to 0..255 |
| `exif_orientation` | `exif_orientation(path) -> int` | the tag; 1 when absent, 0 when unreadable; 1 for every HEIC |
| `needs_working_copy`, `working_copy`, `prepare_photo` | `prepare_photo(path, out_dir) -> str` | the upright JPEG copy of an EXIF-rotated photograph; see `functions.gauge` |

**The stored pixel frame** is what Pillow returns, without an EXIF-orientation transpose. The canvases and every measurement use it, because corner picks and digitised polygons are saved in it. Browsers apply the EXIF tag, so `zonal_canvas.prepare_browser_image` serves a photograph unchanged only when the tag is 1 and otherwise transcodes it, unrotated, to a tagless PNG. For HEIC, libheif applies the container's rotation and pillow-heif resets the tag to 1, so the stored frame is the upright picture. OpenCV's `imread`/`imdecode` apply the tag, so no photograph is read through OpenCV.

### `functions.orthorectify`

Rectifies a photograph containing a quadrat of known size from its four corner pixels, producing an image of known GSD.

```python
orthorectify(image, corners_px, seg_lengths_m, gsd_m=None,
             camera_matrix=None, dist_coeffs=None) -> (rectified, gsd_actual)
```

| Parameter | Meaning |
|---|---|
| `image` | uint8 RGB array (H, W, 3) |
| `corners_px` | 4 × (x, y); corners 0, 1, 2, 3 map to the rectified top-left, top-right, bottom-right, bottom-left |
| `seg_lengths_m` | metres along 0→1, 1→2, 2→3, 3→0; opposite sides are averaged |
| `gsd_m` | output m/px; `None`/`'auto'` keeps the finest source axis' pixel count (`auto_gsd`) |
| `camera_matrix`, `dist_coeffs` | 3×3 and (k1, k2, p1, p2[, k3…]); when both are given, image and corners are undistorted first |

Imports OpenCV lazily. Helpers: `compute_homography_target(corners_px, seg_lengths_m) -> (src_pts, dst_pts, target_w_m, target_h_m)`, `auto_gsd(corners_px, target_w_m, target_h_m) -> float`, `parse_calibration_file(path) -> (fx, fy, cx, cy, [dist]) | None` (OpenCV JSON, YAML or `.npz`; `camera_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`).

```python
rectify_one(src_path, corners, seg_lengths_m, out_path, *,
            gsd_m=None, camera_matrix=None, dist_coeffs=None,
            frame_thickness_m=None, seg_confirmed=False,
            overwrite=True, tool_name="PebbleMapper") -> RectifyOutcome
```

The file-writing entry point. `src_path` is read with `functions.images.read_rgb` in the stored pixel frame, where the corners were picked. `gsd_m=0` means auto. `frame_thickness_m` is recorded as `PM_FRAME_THICKNESS_M`, the default frame inset for georeferencing and the frame band for `functions.quadrat_frame`. The default segment length `DEFAULT_SEG_M = 1.0` gives a warning unless `seg_confirmed=True`. With `overwrite=False` an existing target is a refusal. `tool_name` is written into the record.

Returns `RectifyOutcome(ok, out_path, gsd_m, shape, refusal, notes, warnings, record)`; a refusal (`ok=False`) writes nothing. Refusals: a missing source, fewer than four corners, non-positive lengths, an output equal to the source or resembling a raw camera file, an existing target without `overwrite`. A rectified size disagreeing with the segment lengths by more than 2 % is a warning.

Files written: `<out stem>_GSD=<g>m<ext>` (`gsd_tagged_name(out_path, gsd_m) -> Path`; an existing `GSD=` in the stem is kept); the sidecar `<that file>.json` with the rectification record (`PM_RECTIFIED`, `PM_RECTIFIED_ON`, `PM_SOURCE`, `PM_GSD`, `PM_SIZE`, `PM_CORNERS`, `PM_SEGMENTS_M`, `PM_FRAME_THICKNESS_M`, `PM_TOOL`, and `lat`/`lon` when the source has a GPS fix); and the same record embedded as image metadata (`exif_seed.write_placement_metadata`).

Batch helpers: `is_ready_to_rectify(corners, seg_lengths_m, seg_confirmed=False) -> (ready, reason)`; `plan_rectify_folder(rows) -> (ready, skipped)` with `rows = [(name, corners, seg_lengths_m, seg_confirmed, already_done)]`; `rectify_folder(jobs, *, overwrite=False, progress_fn=None, should_stop=None, tool_name="PebbleMapper") -> list[BatchItem]` with jobs `dict(photo=, src=, corners=, segs=, out=, gsd=, thickness=, confirmed=, camera_matrix=, dist_coeffs=)` (two jobs writing the same file are both skipped); `batch_manifest(items, when="") -> str`; `rectified_output_for(out_path, gsd_m=None, listing=None)`.

### `functions.quadrat_frame`

The frame band of a rectified quadrat photograph, read from its sidecar (`PM_FRAME_THICKNESS_M`), so that no detector measures the frame as clasts.

| Function | Signature | Returns |
|---|---|---|
| `frame_thickness_m` | `frame_thickness_m(image_path) -> Optional[float]` | the recorded thickness, or `None` |
| `frame_inset` | `frame_inset(image_path) -> Optional[FrameInset]` | the inset in pixels (thickness plus a 10 % margin, at least 2 px) and the inner size in metres; `None` without a thickness or when it is 0 |
| `inner_rectangle_geojson` | `inner_rectangle_geojson(fi, source_name="") -> dict` | the inner rectangle as a GeoJSON polygon in image pixels |
| `write_inner_roi` | `write_inner_roi(image_path, out_dir) -> Optional[Path]` | writes that polygon as the job's ROI file |
| `describe` | `describe(fi) -> str` | the one-line note the file list shows |

### `functions.quadrat_detect`

Orthorectify's **Suggest a position**: an exemplar of the frame from a photograph already rectified in the folder, matched on the next one.

| Function | Signature | Returns |
|---|---|---|
| `build_exemplar` | `build_exemplar(image, corners, frame_thickness_m, quadrat_side_m, …) -> Exemplar` | the frame's appearance around the four picked corners |
| `propose` | `propose(image, exemplar, …)` | four corner proposals, or `None` when the two independent checks disagree |
| `refine_corners` | `refine_corners(image, exemplar, seed, …)` | snaps rough corners onto the frame |

### `functions.georef`

Places a rectified quadrat photograph in a UAV ortho's coordinate frame by feature matching, with acceptance gates (inlier count, RMS residual, fitted scale against the GSD ratio, ambiguity). Rejections write nothing.

`DEFAULTS` (override through `settings`): `search_radius_m 10.0`, `frame_inset_m 0.0`, `min_inliers 12`, `min_inlier_fraction 0.03`, `max_residual_m 0.05`, `scale_tolerance 0.15`, `ambiguity_ratio 0.75`, `ransac_seed 20260805`, `descriptor "sift"`, `match_ratio 0.75`, `orb_keypoints 800`, `orb_keypoints_per_megapixel 50000`, `orb_keypoints_max 20000`.

```python
locate_quadrat(quadrat_gray, read_window, ortho_shape, *,
    quadrat_gsd_m: float, ortho_gsd_m: float,
    seed_xy: Optional[Tuple[float, float]] = None,
    ortho_origin_xy: Tuple[float, float] = (0.0, 0.0),
    search_radius_m: Optional[float] = None, max_radius_m: Optional[float] = None,
    frame_inset_m: Optional[float] = None, settings: Optional[dict] = None,
    log_fn=None) -> QuadratMatch
```

| Parameter | Meaning |
|---|---|
| `quadrat_gray` | the rectified quadrat as a 2-D array (RGB is converted) |
| `read_window` | `read_window(r0, c0, r1, c1)` returning that ortho window's pixels; the ortho is never loaded whole |
| `ortho_shape` | `(rows, cols)` of the ortho |
| `quadrat_gsd_m`, `ortho_gsd_m` | m/px on each side; they fix the expected scale |
| `seed_xy` | approximate world position; the search expands outward from it and the first accepted ring wins; without a seed the whole ortho is tiled |
| `ortho_origin_xy` | world coordinate of the ortho's top-left pixel |
| `search_radius_m`, `max_radius_m` | radius around the seed; cap for the outward search |
| `frame_inset_m` | metres of frame excluded from matching |

Returns `QuadratMatch(matrix, quality, seed_xy, quadrat_gsd_m, ortho_gsd_m)`. `matrix` is the 3×3 similarity from quadrat pixel `(col, row)` to ortho world coordinates. `quality` is a `MatchQuality` with `n_correspondences, n_inliers, inlier_fraction, residual_m, scale, scale_expected, rotation_deg, accepted, rival_inliers, rival_checked` (`False` when the ambiguity search could not run, which is not a pass)`, rival_error, frame_inset_px, candidates, reasons`. Two distant placements that both pass are reported as ambiguous (`accepted=False`, `candidates` filled). `match_quadrat(quadrat_gray, ortho_window_gray, *, quadrat_gsd_m, ortho_gsd_m, window_origin_xy, seed_xy=None, search_radius_m=None, frame_inset_m=None, settings=None) -> QuadratMatch` is the single-window primitive.

```python
write_georeferenced(match, quadrat_image, out_dir, stem: str, *,
    crs_wkt: str = "", clasts=None, x_col: str = "x", y_col: str = "y",
    source_files: Optional[dict] = None, hand_placed: Optional[dict] = None,
    overwrite: bool = False) -> dict
```

Raises `ValueError` unless `match.accepted` or `hand_placed={"matrix": 3x3, "provenance": ..., "agreement_score": ...}` is given, and `FileExistsError` when an output exists without `overwrite`. Writes `<stem>.tif` (the affine as GeoTransform, `crs_wkt` as projection, `PM_PLACEMENT` metadata); `<stem>_individual_clasts.csv` when `clasts` (a per-clast table in quadrat pixels) is given, with `x`/`y` in world coordinates and a `placement` column; and `<stem>.georef.json` (transform, provenance, quality, seed, GSDs, CRS, sources). Returns `{"geotiff", "csv", "sidecar"}` paths (or `*_error` strings).

Also public: `describe_georeferencing(image_path) -> {georeferenced, crs, gsd_m, reason}`; `ortho_crs(ortho_path) -> str` (WKT); `transform_seed(x, y, source_crs="EPSG:4326", target_crs="") -> (x, y, note)` (lon/lat order, forced); `world_from_pixels(matrix, cols, rows) -> (xs, ys)`; `survey_overlay_figure(placements, ortho_path, *, max_px=1600, quadrat_shapes=None) -> Figure` with `placements = {photo_name: matrix}`.

### `functions.placement`

The hand-placement editor behind `write_georeferenced(hand_placed=…)`: a `Placement` (centre, rotation and scale of the quadrat in the ortho) and the moves the Georeference canvas applies.

| Function | Signature | Returns |
|---|---|---|
| `matrix_from_placement`, `placement_from_matrix` | `matrix_from_placement(p) -> ndarray`, `placement_from_matrix(matrix, quad_shape) -> Placement` | the 3×3 similarity and back |
| `placement_from_seed` | `placement_from_seed(seed_xy, quad_shape, quadrat_gsd_m, …) -> Placement` | an unrotated placement centred on a seed |
| `corners_world` | `corners_world(p) -> ndarray` | the four corners in world coordinates |
| `translate`, `nudge`, `set_rotation`, `drag_corner` | `translate(p, dx_m, dy_m)`, `nudge(p, dcol, drow, ortho_gsd_m)`, `set_rotation(p, rotation_deg)`, `drag_corner(p, corner_index, target_xy)` | a new `Placement` |
| `placement_delta` | `placement_delta(fitted, saved) -> dict` | shift, rotation and scale between a hand placement and the fitted one |
| `blend_layer`, `agreement_score` | | the canvas blend of quadrat and ortho window, and the edge-agreement score |

### `functions.seeds`

Approximate quadrat positions for the matcher, and the survey-level driver most scripts call instead of `locate_quadrat`.

| Function | Signature | Meaning |
|---|---|---|
| `load_seed_table` | `load_seed_table(path, default_crs: str = "", *, separator: str = "comma", has_header: bool = True, decimal: str = "") -> SeedTable` | reads a photo-to-coordinate CSV; recognises headers such as `photo/image/file`, `x/lon/easting`, `y/lat/northing`, `crs/epsg`; `separator` is `comma`, `semicolon`, `tab`, `pipe`, a literal character or `"auto"`; never raises on content; `table.problems` lists what was skipped |
| `SeedTable.get` | `get(self, photo) -> Optional[Seed]` | exact name, then the base photo name (rectification renames files) |
| `seed_for_photo` | `seed_for_photo(table, photo, pins=None) -> Optional[Seed]` | a pin (`{normalised name: (x, y[, crs])}`) wins over the table |
| `resolve_seed` | `resolve_seed(photo, table=None, pins=None, typed=(None, None), typed_crs: str = "", search_dirs=(), bounds=None, crs_transform=None) -> ResolvedSeed` | pin, table, typed coordinate, then the photograph's EXIF fix; `ResolvedSeed(x, y, crs, provenance, ok)` |
| `check_survey` | `check_survey(photo_paths, ortho_path, table=None, pins=None, *, quadrat_gsd_m=None, search_radius_m=None, frame_inset_m=None, settings=None, log_fn=None, progress_fn=None, should_stop=None, full_grid=False) -> List[QuadratCheck]` | places every photograph, reading each ortho tile once; seeds are converted to the ortho CRS; `quadrat_gsd_m=None` reads each photograph's GSD; `full_grid=True` sweeps the whole ortho for unseeded photographs (otherwise `no_seed`). Writes nothing |
| `check_quadrats` | `check_quadrats(photo_paths, ortho_path, table=None, pins=None, *, quadrat_gsd_m=None, search_radius_m=None, frame_inset_m=None, settings=None, log_fn=None, progress_fn=None) -> List[QuadratCheck]` | per-quadrat variant (each reads its own windows) |
| `save_located` | `save_located(results, photo_dir, ortho_path, out_dir, *, quadrat_gsd_m=None, overwrite=False, progress_fn=None) -> dict` | writes a GeoTIFF and `.georef.json` (through `write_georeferenced`) for every `located` result; returns `{"written": [...], "skipped": [(photo, why), ...]}` |

`QuadratCheck(photo, status, detail, n_inliers, residual_m, seed_source, matrix, seed_world, candidates)`; `status` is `located`, `not_located`, `no_seed`, `already_placed` or `unreadable`.

### `functions.exif_seed`

| Function | Signature | Meaning |
|---|---|---|
| `gps_from_image` | `gps_from_image(path) -> Optional[dict]` | `{"lon", "lat", "alt", "hpe_m", "source"}` in EPSG:4326, or `None`; never raises |
| `seed_from_photo` | `seed_from_photo(path, search_dirs=(), bounds=None, crs_transform=None) -> Tuple[Optional[dict], str]` | a usable fix (from the photograph, else its raw sibling in `search_dirs` or a sibling `raw/` folder), or the reason there is none; a fix more than `EXIF_SEED_RADIUS_M` (10 m) outside `bounds`, or with a reported error above `HPE_LIMIT_M` (40 m), is refused |
| `raw_sibling` | `raw_sibling(path, search_dirs=()) -> Optional[Path]` | the raw photograph a rectified file came from |
| `read_rectification_record` | `read_rectification_record(path) -> dict` | the `PM_*` record embedded by `rectify_one` |
| `write_placement_metadata` | `write_placement_metadata(path, fields: dict) -> bool` | attaches metadata to an image the tool produced, without re-encoding pixels |

---

## 9. Validation against ground truth

### `functions.digitize`

Turns hand-drawn outlines into the canonical 15 columns (the Digitize tab's records).

| Function | Signature | Returns |
|---|---|---|
| `polygon_to_cropped_mask`, `circle_to_cropped_mask`, `ellipse_to_cropped_mask` | `(vertices, image_shape)`, `(center, radius, image_shape)`, `(center, axes, angle_deg, image_shape)` | a boolean mask cropped to the shape's bounding box, with its offset |
| `mask_centroid` | `mask_centroid(mask)` | the centroid of a mask |
| `measure_records` | `measure_records(records, image_height, resolution, *, with_contours=False) -> DataFrame` | one canonical row per record (`resolution` in m/px; `y` from the bottom), outlines attached on request |
| `digitize_records_to_dataframe` | `digitize_records_to_dataframe(records, image_height, resolution) -> DataFrame` | the truth CSV the tab exports |

### `functions.provenance`

The `<csv stem>.provenance.json` sidecar of a truth CSV: which clasts were drawn by hand, which came from which model, and which were edited.

| Function | Signature | Returns |
|---|---|---|
| `write_provenance`, `read_provenance` | `write_provenance(csv_path, clasts, models=(), *, image=None, extra=None) -> Optional[Path]`, `read_provenance(csv_path) -> Optional[dict]` | the sidecar |
| `model_entry` | `model_entry(backend, info=None, *, display_name=None, …) -> dict` | one model's entry (name, version, parameters) |
| `summary_line` | `summary_line(clasts, models=()) -> str` | the one-line origin summary shown in the tab and the report |

### `functions.validation`

Compares a detection CSV with a truth CSV. Pure functions on DataFrames and arrays; no I/O.

```python
pair_csvs(truth_df, detect_df, x_col="x", y_col="y", tolerance=None, size_col=None) -> dict
```

Mutual nearest-neighbour matching; each detection is used at most once. Both frames must share CRS and units. `tolerance` is the maximum pairing distance; `None` uses half the median nearest-neighbour distance of the truth points (0.05 m with a single truth point). With `size_col`, a pair whose sizes differ by more than 50 % is rejected. Returns `{"matched_pairs": [(truth_idx, detect_idx)], "unmatched_truth": [...], "unmatched_detect": [...], "used_tolerance": float}`.

```python
detection_metrics(pair_result) -> dict
```

`n_matched, n_truth, n_detect, false_negatives, false_positives, recall, precision, f1` (NaN when undefined).

```python
compute_distribution_stats(truth_vals, detect_vals, field=None) -> dict
```

Two independent samples. With `field` given, the φ block runs only for linear-size fields (`units.is_size_field_for_phi`); with `field=None` it always runs. Keys: `truth_n, detect_n, phi_meaningful`; `truth_d16/d50/d84, detect_d16/d50/d84, delta_d16/d50/d84` (input units, detect minus truth); `truth_/detect_` `mean_phi, sorting_phi, skewness_phi, kurtosis_phi` (Folk-Ward; NaN for non-size fields); `ks_statistic, ks_p_value` (two-sample K-S, on φ for size fields); `qq_truth_quantiles, qq_detect_quantiles` (101 matched quantiles).

```python
compute_paired_stats(truth_paired, detect_paired) -> dict
```

Equal-length arrays paired by index (non-finite and non-positive pairs dropped). Keys: `n, slope, intercept` (detect = slope × truth + intercept), `r_squared, rmse, mae, bias` (mean of detect − truth), `bland_altman_mean, bland_altman_diff` (per pair), `bland_altman_loa_lo, bland_altman_loa_hi` (bias ± 1.96 σ).

```python
compute_batch_summary(jobs) -> dict
```

Aggregates the GUI's completed validation job dicts into per-quadrat points, violin arrays and a QC text.

**Figures and the result file are written by the GUI.** In `gui/app.py`, `_build_validation_figures` and `_save_validation_figures` write `validation/results/figures/<hash>/<hash>_<name>.png` (`name` in `hist, cdf, qq, paired_scatter, bland_altman, ccdf, detection_function, spatial, overlay`), and `_persist_validation_result` writes `validation/results/<truth stem>__vs__<detect stem>__<hash>.validation.json`, which the report reads:

```
schema_version (2), generated_at, truth_csv, detect_csv, field,
truth_gsd_m_per_px, detect_gsd_m_per_px, tolerance_m,
metrics {n_truth, n_detect, n_matched, recall, precision, f1, r2, rmse, bias,
         truth_D50, detect_D50, truth_D84, detect_D84, D50_relerr, D84_relerr,
         ks_statistic, ks_p_value},
distribution (compute_distribution_stats output, arrays as lists),
paired (compute_paired_stats output), figures {name: relative path}
```

To include a scripted comparison in the PDF, write a file of that shape with the `.validation.json` suffix under `validation/results/`.

### `functions.quadrat_validation`

Clips both frames to the quadrat before comparison, and converts a truth CSV digitised in pixels to world coordinates.

| Function | Signature | Meaning |
|---|---|---|
| `detect_gsd_from_path` | `detect_gsd_from_path(path: str) -> dict` | `{"gsd": m/px or None, "source": "geotransform" / "filename" / None}` |
| `quadrat_from_geotiff` | `quadrat_from_geotiff(truth_path: str) -> QuadratFootprint` | the truth raster's bounds; `ValueError` for an identity GeoTransform |
| `quadrat_from_centroid` | `quadrat_from_centroid(cx: float, cy: float, width: float, height: Optional[float] = None) -> QuadratFootprint` | axis-aligned rectangle; `height=None` means square |
| `quadrat_from_point_and_corner` | `quadrat_from_point_and_corner(px: float, py: float, corner: str, width: float, height: Optional[float] = None, n_segments: int = 64) -> QuadratFootprint` | one recorded corner (`N, NE, E, SE, S, SW, W, NW`) moved half the diagonal to the centroid; the footprint is a disc of that radius |
| `clip_to_footprint` | `clip_to_footprint(df, footprint, x_col: str = "x", y_col: str = "y") -> pd.DataFrame` | rows inside the footprint |
| `valid_pixel_mask_from_image` | `valid_pixel_mask_from_image(image_path: str) -> Optional[dict]` | `{"mask", "gt", "W", "H"}` of non-nodata pixels; `None` for an ungeoreferenced image |
| `clip_to_valid_pixels` | `clip_to_valid_pixels(df, mask_bundle: dict, x_col: str = "x", y_col: str = "y") -> pd.DataFrame` | drops rows on nodata margins |
| `reproject_pixels_to_world` | `reproject_pixels_to_world(image_path: str, df, x_col: str = "x", y_col: str = "y") -> tuple[pd.DataFrame, bool]` | applies the truth raster's affine to `(col, row)` centroids; the flag says whether it was applied |
| `placement_provenance` | `placement_provenance(path: str) -> dict` | `{"placement": "fitted" / "hand-edited" / "manual" / "", "agreement_score"}` from the metadata `write_georeferenced` stores |

`QuadratFootprint(kind, bounds, polygon_xy, centroid, radius, width, height)`; `kind` is `"box"` or `"buffer"`.

A typical comparison clips both frames with `clip_to_footprint(df, quadrat_from_geotiff(truth_tif))` and `clip_to_valid_pixels`, then calls `pair_csvs`, `detection_metrics`, `compute_distribution_stats` and, on the matched pairs, `compute_paired_stats`.

### `functions.truncation`

The detection-limit analysis shared by the Validate tab and the report: sizes below `k` pixels are not resolved by the imagery.

| Function | Signature | Returns |
|---|---|---|
| `compute_d_min` | `compute_d_min(gsd_m, k_pixels=DEFAULT_K_PIXELS) -> float` | `k × GSD` in metres (`DEFAULT_K_PIXELS = 8`) |
| `truncate_above_dmin` | `truncate_above_dmin(values, d_min) -> ndarray` | values at or above the limit |
| `detection_function_from_pair` | `detection_function_from_pair(truth_sizes, matched_truth_idx, *, n_bins=10, …)` | fraction of truth clasts found per size bin |
| `plot_ccdf_log_log`, `plot_detection_function` | | the two figures the Validate tab and the report draw |

---

## 10. Report and Express

### `functions.report.build_pdf`

Inventories a project folder (images, CSVs, rasters and sidecars, maps, zonal outputs, `.validation.json` files, detection logs) and builds the PDF report.

```python
build_pdf(project_root, out_path, options=None, log_fn=None) -> Path
```

`project_root` is the **absolute project directory** (`layout.project_path(name)`, the date folder for a dated project), not the project name. `out_path`'s parent is created. `log_fn` receives progress strings.

| `options` key | Default | Meaning |
|---|---|---|
| `author`, `affiliation`, `description` | `""` | cover text |
| `cover_image` | none | cover picture |
| `illustrations` | none | `{"path", "title", "description"}` entries for an Illustrations section, present only when non-empty |
| `preset` | `"full"` | embedding fidelity: `"full"` (300 dpi, JPEG quality 95) or `"shareable"` (90 dpi, quality 70); see `REPORT_PRESETS`. Does not change the sections |
| `include_cover`, `include_toc`, `include_spatial_maps`, `include_zonal_statistics`, `include_validation` | `True` | cover and summary, table of contents, spatial statistics, zonal, validation sections |
| `include_appendix_logs`, `include_appendix_samples`, `include_appendix_field_reference` | `True` | run register, data dictionary and glossary appendices |
| `include_overview`, `include_methodology`, `include_detection`, `include_data_quality`, `include_publication_figures`, `include_list_of_figures`, `include_list_of_tables` | | accepted but not consulted; the data-and-processing, detection-results and interpretation sections are always built |

A section whose artefact is malformed is replaced by a one-line notice. Sections with no inputs on disk are skipped.

### `functions.express`

The Express pipeline in one call: detect (one run per window), merge, rasterize, map and report. UAV orthos only; an identity GeoTransform is refused.

`PRESETS`:

| Preset | windows (m) | overlap | min_confidence | dedup_overlap | merge | min_cell_density | cellsize (m) |
|---|---|---|---|---|---|---|---|
| `fast` | 5 | 0.0 | 0.80 | 0.30 | no | 0 | max(1, first window) |
| `medium` | 2.5, 5 | 0.20 | 0.70 | 0.30 | yes | 3 | 1 |
| `slow` | 1, 2.5, 5 | 0.25 | 0.60 | 0.30 | yes | 5 | 1 |

Rasters (`field`, `parameter`, `percentile`): fast `(Clast_length, quantile, 0.5)`; medium adds `(Clast_length, density)` and `(Clast_length, folk_ward_sorting)`; slow adds `(Clast_length, quantile, 0.84)` and `(Equivalent_diameter, quantile, 0.5)`. Maps: fast, one vector map; medium, vector and raster; slow, vector and two raster maps on `Esri.WorldImagery`.

```python
run_express_pipeline(project, ortho_path, preset_name, *,
                     windows=None, model="maskrcnn", log_fn=None,
                     progress_cb=None, on_stage=None, stop_check=None,
                     devicemode="gpu", devicenumber=0) -> ExpressResult
```

| Parameter | Meaning |
|---|---|
| `project` | project **name** under the datasets root |
| `ortho_path` | georeferenced GeoTIFF |
| `preset_name` | `"fast"`, `"medium"` or `"slow"` |
| `windows` | overrides the preset's window sizes (m, deduplicated, ascending); merging runs whenever two or more windows run |
| `model` | detection backend name |
| `log_fn` | receives `"[express] ..."` strings |
| `progress_cb` | forwarded to detection as `progress_callback` |
| `on_stage` | `on_stage(name, status)`; stages `Detection, Merge, Rasterize, Map, Report`; statuses `pending, running, done, stopped, error, skipped` |
| `stop_check` | polled between windows, rasters and maps |

Each stage is best-effort: a failure marks it `error` and later stages run on the inputs that exist. Returns `ExpressResult(stages, report_pdf, rasters, maps, merged_csv, window_csvs)`.

Files written (origin = `naming.origin_stem(<ortho stem>, project, active date)`):

| Stage | File |
|---|---|
| Detection | `output_results/vectors/<origin>_ws<w>m.csv` (+ log, manifest) per window |
| Merge | `output_results/vectors/<origin>_xprs_merged.csv` |
| Rasterize | `output_results/rasters/<origin>_xprs_<field>_<parameter>_cellsize=<c>m.tif` (+ `.json`) |
| Map | `output_results/maps/<origin>_xprs_<field>_map.png` (vector), `<raster stem>_map.png` (raster); `_<i>` appended on a clash |
| Report | `output_results/reports/<project dir name>_xprs_report.pdf` |

Also public: `resolve_plan(ortho_path, preset_name: str, windows=None, model: str = "maskrcnn") -> ResolvedPlan` (everything the run will do, without running it) and `read_ortho_extent(ortho_path) -> dict` (`width_m, height_m, gsd_m, is_georeferenced`).

---

## 11. Units and ground sample distance

### `functions.units`

How a per-clast field or per-cell parameter is displayed. Stored units are SI; display units are what maps, tables and the report show. Lookups are case-insensitive longest-substring matches, so `Clast_length_folk_ward_sorting` resolves to the `folk_ward_sorting` row.

| Function | Signature | Returns |
|---|---|---|
| `field_unit_and_factor` | `field_unit_and_factor(field: Optional[str]) -> Tuple[str, float]` | `(display_unit, stored→display multiplier)`: an exactly registered field first, else the longest substring match; unknown fields give `("m", 1.0)` |
| `register_field_unit` | `register_field_unit(field: str, unit: Optional[str], factor: float = 1.0) -> None` | declares a field's display unit and multiplier; `"dimensionless"`, `"none"`, `"ratio"`, `"-"`, `"1"` and `""` mean no unit |
| `registered_field_units` | `registered_field_units() -> dict` | a copy of the registry, `{lower-case name: (unit, factor)}` |
| `field_unit` | `field_unit(field: Optional[str]) -> str` | the unit string |
| `format_value_with_unit` | `format_value_with_unit(value: Optional[float], field: str, u_display: Optional[float] = None) -> str` | `0.0034` for `Clast_length` → `'3.40 mm'`; with `u_display` the value is rounded to the digit the uncertainty supports |
| `field_display` | `field_display(field: Optional[str]) -> Tuple[str, str, float]` | `(display_name, unit, factor)`; unregistered fields get a tidied name and no unit |
| `field_registry_lookup` | `field_registry_lookup(field: Optional[str])` | the `FIELD_REGISTRY` row `(key, display, unit, factor, cmap, diverging)` or `None` |
| `parameter_display` | `parameter_display(parameter: Optional[str])` | the `PARAMETER_DISPLAY` row `(display_name, unit, factor, cmap, diverging)` or `None` |
| `resolve_display` | `resolve_display(field: Optional[str], parameter: Optional[str]) -> Tuple[str, str, float, str, bool]` | `(display, unit, factor, cmap, diverging)`; a parameter that sets a unit (`sorting`, `skewness`, `density`, …) takes precedence; display reads `"<field> — <parameter>"` |
| `clean_field_name` | `clean_field_name(s: Optional[str]) -> str` | strips the CSV stem before `<Field>_<parameter>_cellsize=...`; empty for density rasters |
| `is_size_field_for_phi` | `is_size_field_for_phi(field: Optional[str]) -> bool` | linear-size fields where φ statistics apply (`SIZE_FIELDS_FOR_PHI`; not `Surface_area`) |
| `folk_ward_sorting_phi` | `folk_ward_sorting_phi(p5, p16, p84, p95) -> float` | `(φ84 − φ16)/4 + (φ95 − φ5)/6.6` |
| `folk_ward_skewness_phi` | `folk_ward_skewness_phi(p5, p16, p50, p84, p95) -> float` | Folk-Ward graphic skewness |
| `folk_ward_kurtosis_phi` | `folk_ward_kurtosis_phi(p5, p25, p75, p95) -> float` | `(φ95 − φ5) / (2.44 (φ75 − φ25))` |

The Folk-Ward helpers take φ quantiles (`φ = −log2(D / PHI_REF_M)`, `PHI_REF_M = 1e-3` m) and return NaN for NaN input or a zero denominator; sample-size checks are the caller's. Tables: `FIELD_UNIT_TABLE`, `FIELD_REGISTRY`, `PARAMETER_DISPLAY`, `PERCENTILE_FAMILY` (`D5, D16, D50, D84, D95, average` share a colorbar).

### `functions.gsd`

Reads a quadrat photograph's GSD, which rectification records in a sidecar (`<image>.json`, key `PM_GSD`) and in the filename (`..._GSD=0.0021m.jpg`).

| Function | Signature | Returns |
|---|---|---|
| `parse_gsd_from_filename` | `parse_gsd_from_filename(filename) -> Optional[float]` | m/px from `GSD=<value>m` or the legacy `GSD=<int>p<frac>mm` (`GSD=2p100mm` = 0.0021 m); `None` when absent or non-positive |
| `sidecar_path` | `sidecar_path(image_path) -> Path` | `<image_path>.json` |
| `effective_gsd` | `effective_gsd(image_path) -> GsdInfo` | `GsdInfo(gsd, source, warning)`: the sidecar's valid `PM_GSD`, then the filename, else `(None, None)`; `source` is `"sidecar"`, `"filename"` or `None`; `warning` when a sidecar existed but could not be trusted |
| `resolution_after_open` | `resolution_after_open(own_gsd, current, last_auto, typed) -> Optional[float]` | the resolution a Digitize photograph opens with: its own GSD, else the typed value; never another photograph's automatic value |

---

## 12. Which function writes which file

`<origin>` is `naming.origin_stem(image_stem, project, date)`, plus `_model=<id>` for a model other than Mask R-CNN; `<w>` the window size in metres (`:g`); `<c>` the cell size; `<set>` the zone-set token. Folders are `layout` kinds.

| Function | Folder (kind) | File |
|---|---|---|
| `clasts_detect_jobs` / `DetectorBackend.detect_jobs` (ortho) | `vectors` | `<origin>_ws<w>m.csv`, `_detection_log.txt`, `.run.csv` (while incomplete), `.crash.json` (on error) |
| `MaskRCNNBackend.detect_jobs`, `run_detect_jobs` | `vectors` | `<csv>.manifest.json` |
| `clasts_detect_jobs` (quadrat) | `vectors` | `<stem>_individual_clasts.csv` |
| `clasts_detect_jobs` (`saveplot`) | `figures` | `<stem>_overlay.png`, `<stem>_histogram.png` (quadrat); `tiles/<origin>_tile_k=<k>_n=<n>_m=<m>.png` (ortho) |
| detection, `merge_csvs`, `run_gauge` | beside the CSV | `<csv stem>.contours.json` |
| `merge_csvs` | `vectors` | `<origin>_merged.csv`; Express: `<origin>_xprs_merged.csv` |
| `clasts_rasterize` | `rasters` | `<csv stem>_<field>_<parameter>_cellsize=<c>m.tif` (GUI: `D<pct>` for quantiles, `<csv stem>_density_cellsize=<c>m.tif` for density), plus `<tif>.json` |
| `save_publication_map` | `maps` | `<raster or csv stem>_map.png`, plus `<png>.json` |
| `zonal_polygon_stats_from_csv` / `zonal_polygon_stats` | `zonal` | `<csv stem>_zones=<set>_field=<field>.polygons.csv` |
| `zonal_transect_profile` | `zonal` | `<raster stem>_zones=<set>.transects.csv` |
| `plot_transect_profile` | `zonal` | `<transect csv stem>__<transect_id>.png` |
| `plot_zonal_map` | `zonal` | `<raster stem>_zones=<set>.zonal_map.png` |
| `build_profile_outputs` | `zonal` | `<stem>__profile.png`, `__summary.csv`, `__change.csv`, `__binned.csv` |
| `run_gauge` | `gauge` | `<origin>_gauge.csv`, `<origin>_gauge.json`, `<origin>_gauge.contours.json` |
| `write_gauge_figures` | `gauge` | `<origin>_gauge_overlay.png`, `<origin>_gauge_distribution.png` |
| `prepare_photo` (EXIF-rotated photographs) | `gauge` | `<photo stem>.jpg` |
| `save_library` | project root | `scaling_objects.json` |
| `rectify_one` / `rectify_folder` | `validation_rectified` | `<photo stem>_rectified_GSD=<g>m.<ext>`, `<that>.json` |
| `write_georeferenced` / `save_located` | `validation_georectified` | `<stem>.tif`, `<stem>_individual_clasts.csv`, `<stem>.georef.json` |
| Validate tab (GUI) | `validation_results` | `<truth stem>__vs__<detect stem>__<hash>.validation.json`, `figures/<hash>/<hash>_<name>.png` |
| `build_pdf` | `reports` | the `out_path` passed; Express: `<project dir>_xprs_report.pdf` |
| `run_express_pipeline` | as above | per-window CSVs keep standard names; merge, rasters, maps and report carry `_xprs` after the origin |
| `ensure_project_layout` | project root | the folder tree and `README.txt` |

---

## 13. Clast geometry: orientation, axes and contours

### `functions.clast_geometry`

**Orientation convention.** `Orientation` in every clast CSV is the bearing of the long axis in degrees, **clockwise from image-up** (grid north for an ortho, the top of the photograph for a quadrat image), axial on [0, 180): 0° up-down, 90° left-right, 45° lower-left to upper-right. A long axis at screen angle `t` (anticlockwise from +x) has `Orientation = 90 − t`. It is not an angle for `matplotlib.patches.Ellipse` or `cv2.ellipse`; convert with `axis_angle_deg`.

| Function | Signature | Purpose |
|---|---|---|
| `axis_direction` | `axis_direction(orientation, *, y_down) -> (dx, dy)` | unit vector of the long axis in a frame whose y counts down (rows) or up (world, quadrat CSV) |
| `axis_angle_deg` | `axis_angle_deg(orientation, *, y_down) -> float` | the angle from +x in that frame, for `Ellipse(angle=)` / `cv2.ellipse` |
| `axis_chords` | `axis_chords(x, y, length, width, orientation, *, y_down) -> (major, minor)` | the `Clast_length` and `Clast_width` chords through the centroid |
| `ellipse_outline` | `ellipse_outline(x, y, major, minor, orientation, *, y_down, n_points=64) -> ndarray` | ellipse vertices (merge's IoU footprint) |
| `contour_from_measurement` | `contour_from_measurement(meas, *, to_frame, offset=(0, 0), tolerance=0.5, decimals=2) -> list` | a `_measure_clast` outline in the CSV frame (`quadrat_frame(height)`, `world_frame(geotransform)`), Douglas–Peucker at half a pixel |
| `write_contours` / `read_contours` | `write_contours(csv_path, contours, *, frame=None, extra=None)`, `read_contours(csv_path) -> {clast_ID: ndarray} or None` | the `<csv stem>.contours.json` sidecar (`frame` `"pixels"` or `"world"`); never raises |
| `attach_contours` / `contours_of` | `attach_contours(df, contours, *, frame)`, `contours_of(df)` | outlines carried in `df.attrs["contours"]` |
| `contours_for_frame` | `contours_for_frame(df, contours) -> {row position: ndarray}` | the outlines of a (filtered) frame, matched by `clast_ID` |
| `draw_clasts` | `draw_clasts(ax, clasts, *, contours=None, y_down, to_data=None, units_per_m=1.0, edge_colors=None, face_alpha=0.45, ..., chords=True, outlines=True, note=True) -> dict` | each clast as a translucent fill (its `edge_colors` entry, else a soft palette) under a thin dark outline, with its major and minor chords; without outlines, the axes and a note |

The contours sidecar is written beside the CSV by Quadrat and Ortho detection, by `measure_instances`/`run_detect_jobs` for external backends, by `run_gauge`, and by `merge_csvs` (kept rows, renumbered IDs). An Ortho run keeps per-tile outlines in `<stem>.run.contours.jsonl`, so a resumed run recovers them; clean completion removes that file and maps the outlines through tile deduplication to the final IDs. `report_figures.detection_overlay` takes `contours=` or `csv_path=` to find the sidecar.
