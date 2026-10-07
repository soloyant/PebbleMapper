# Adding a detection model

A detection model is added to PebbleMapper as a backend: a Python class that implements
`detectors.base.DetectorBackend`, declared in a JSON file. PebbleMapper's source is not
edited. Mask R-CNN (`detectors/maskrcnn.py`) is the built-in backend; the `stub` backend
(`detectors/stub_backend.py`) is the template for a backend that runs in its own conda
environment.

1. [Output table](#1-output-table)
2. [`detect_jobs`](#2-detect_jobs)
3. [Backend types](#3-backend-types)
4. [Registration](#4-registration)
5. [In-process backend](#5-in-process-backend)
6. [Subprocess backend](#6-subprocess-backend)
7. [Checklist](#7-checklist)
8. [Existing plug-ins](#8-existing-plug-ins)

## 1. Output table

A backend writes the per-clast table defined in the
[clast table section of the user manual](../user-manual.md#the-clast-table): same columns,
same order, same units. In code the column list is
`functions.clasts_detection._CLAST_COLUMNS`, re-exported as `detectors.CANONICAL_COLUMNS`.

| Quantity | Frame and unit |
|---|---|
| `x`, `y` (clast centre) | Ortho: the ortho's CRS. Quadrat: image pixels, `y = image height − row` |
| Linear sizes | metres |
| `Surface_area` | m² |

## 2. `detect_jobs`

```python
backend.detect_jobs(mode, jobs, **kwargs) -> list[pandas.DataFrame]
```

### 2.1 Arguments

| Argument | Content |
|---|---|
| `mode` | `functions.modes.ORTHO` or `functions.modes.QUADRAT`. Normalise with `functions.modes.normalise_mode`, which also accepts older spellings. |
| `jobs` | list of `{"path", "kstart", "out_stem", "roi_path"}`. `out_stem`, when present, is the stem of the output file: it already carries the origin prefix and, for any model other than Mask R-CNN, the `_model=<id>` token (`functions.naming.with_model`). Use it as given. `roi_path` is an optional GeoJSON of regions of interest. |
| `kwargs` | see [2.5](#25-keyword-arguments). |

### 2.2 Outputs

For each job, in input order:

1. **CSV**: the output table, written to `kwargs["output_dir"]` as
   `detectors.output_csv_name(mode, stem, window)`, with
   `stem = job.get("out_stem") or Path(job["path"]).stem` and
   `window = kwargs["metric_cropsize"]`. This gives `<stem>_ws<window>m.csv` in Ortho mode
   and `<stem>_individual_clasts.csv` in Quadrat mode, the names the Detect tab, Validate
   and the report look for.
2. **Manifest**: `<csv>.manifest.json`, written with
   `detectors.base.DetectionManifest(...).write_for_completed_csv(csv_path, run_started)`.
   `model_version` defaults to PebbleMapper's version: set it to the model's. `weights` is a
   free string. The wrapper adds `params["frame_excluded"]` when it excluded a quadrat frame.
3. **Return value**: one `pandas.DataFrame` per job; an empty frame for a job that failed or
   was stopped.

A model that returns instance geometry (masks or polygons) measures each instance with
`detectors.measure`, so that sizes are defined as for Mask R-CNN
([6.3](#63-instances-file)).

### 2.3 Progress events

`kwargs["progress_callback"](job_index, event, payload)` updates the Detect tab's queue rows.

| `event` | `payload` |
|---|---|
| `"start"` | `{"path"}` |
| `"done"` | `{"n_clasts": n}` |
| `"stopped"` | `{}` |
| `"error"` | `{"message", "traceback"}` |

Emit events at job boundaries, as `functions.clasts_detection.clasts_detect_jobs` does, and
wrap each call in `try/except`: the callback is GUI code.

The Detect tab and Digitize call backends through
`detectors.run_detect_jobs(backend, mode, jobs, **kwargs)`, which settles every job:

| Situation when `detect_jobs` ends | Status given |
|---|---|
| returns, job without a terminal event | `"done"` with the returned row count, or `"stopped"` if `stop_check()` is true |
| raises | `"error"` for every unsettled job |

These statuses appear only when the call returns; emit events to show a live count.
Express calls `backend.detect_jobs()` directly and relies on the backend's own events.

### 2.4 Stop

`kwargs["stop_check"]` is a callable returning `True` once the user presses **Stop**. Poll
it between tiles (in-process backend) or while the subprocess runs
(`run_backend(stream=True, stop_check=...)` does this), and report `"stopped"` for the
interrupted job.

### 2.5 Keyword arguments

`kwargs` holds the Detect tab's parameters. A backend uses those it supports and ignores the
others.

| Key | Meaning | Required |
|---|---|---|
| `resolution` | metres per pixel of a quadrat photograph; in Ortho mode the GeoTIFF's geotransform applies | Quadrat |
| `metric_cropsize` | ortho tile size in metres; also the `window` of the CSV name | Ortho |
| `output_dir`, `figures_dir` | output folders for the CSV and any PNG | yes |
| `saveresults`, `saveplot`, `plot` | write the CSV / write `<stem>_overlay.png` / show figures | write the CSV unless `saveresults` is false |
| `min_confidence` | Mask R-CNN's detection threshold | no; record it in the manifest and log when not applied |
| `overlap`, `dedup_method`, `dedup_overlap` | tile overlap fraction and removal of duplicates across tiles | no, when the backend handles tile seams itself |
| `dark_threshold`, `bright_threshold`, `nodata_max_frac` | ortho tile filter (uniform or nodata tiles skipped) | no |
| `devicemode` (`"gpu"`/`"cpu"`), `devicenumber` | device | when possible |
| `roi_path` | GeoJSON polygons, also per job in `jobs[i]["roi_path"]`. For a rectified quadrat photograph whose sidecar records the frame thickness, the wrapper supplies the rectangle inside the frame (`functions.quadrat_frame`), unless the job or the call has an ROI or the caller passes `exclude_frame=False` (Digitize does). | yes; filter centroids with `detectors.measure.load_roi_paths` |
| `log_fn`, `timeout` | log-line sink; subprocess time limit in seconds | subprocess backends |
| `stop_check`, `progress_callback` | [2.4](#24-stop), [2.3](#23-progress-events) | yes |

Settings specific to a backend are read from a file or an environment variable and recorded
in the manifest; the app has no per-backend parameter panel.

### 2.6 Ortho mode

In Ortho mode the backend tiles the ortho, maps pixels to world coordinates, resumes from
`kstart`, skips nodata tiles and removes duplicates across tile seams; there is no shared
tiling module. The reference implementation is `functions.clasts_detection._detect_ortho`,
which `detectors/maskrcnn.py` calls. Minimum steps: read the geotransform (`osgeo.gdal`),
run the model on windows of `metric_cropsize / gsd` pixels with overlap, and map each
centroid with `detectors.measure.world_xy(cx, cy, geotransform=gt)`.

In Quadrat mode the photograph is processed whole, in pixels, and centroids are mapped with
`world_xy(cx, cy, height=H)`.

## 3. Backend types

| | In-process | Subprocess |
|---|---|---|
| Runs in | the `maskrcnn` conda environment | its own conda environment |
| Suited to | models whose dependencies are compatible with TensorFlow 2.10 | models with conflicting dependencies (e.g. PyTorch) or a GPL/AGPL licence |
| Example | `detectors/maskrcnn.py` | `detectors/stub_backend.py` + `detectors/stub/run.py` |
| `BackendInfo` | `in_process=True`, `env=None` | `in_process=False`, `env="<env>"` |

`detectors/maskrcnn.py` implements the same interface over `functions.clasts_detection`. It
is the default on first launch and the fallback when the selected backend is unavailable,
and Digitize takes its masks in memory.

Licences: permissive models (Apache, BSD, MIT) may run in-process and be bundled. GPL and
AGPL models run as a subprocess in an environment the user installs, and are not bundled
with or imported by PebbleMapper, which is MIT-licensed.

## 4. Registration

A backend is a Python module with a factory function, taking no arguments, that returns a
`DetectorBackend`. The module is listed in a JSON file: the file named by the
`PEBBLEMAPPER_DETECTORS` environment variable when it is set (then the only file read; a
missing file disables discovery), otherwise `user_detectors.json` at the repository root
(git-ignored).

```json
[
  {"module": "my_package.my_backend", "factory": "make_backend",
   "path": "../my_backends"}
]
```

| Field | Meaning |
|---|---|
| `module` | module to import |
| `factory` | factory function (default `make_backend`) |
| `path` | optional folder added to `sys.path` before the import, relative to the JSON file. Without it the module must be importable (installed in the `maskrcnn` environment or on `PYTHONPATH`). |

At start-up each entry is imported. An entry that fails to import, build or validate is
logged and skipped. A backend whose `is_available()` is false is left out of the
**Detection model** list, and its `install_hint` is printed once on the console.

Keep the backend's folder short (for example `C:\pm_backends\<name>\`): output names reach
about 130 characters, and Windows limits paths to 260.

## 5. In-process backend

```python
# my_backend.py
from pathlib import Path
from detectors.base import (BackendInfo, DetectionManifest, DetectorBackend,
                            output_csv_name)
from functions import modes

INFO = BackendInfo(
    name="mymodel", display_name="My Model (Author, 2026)",
    framework="tensorflow", license="MIT", output_type="instance",
    version="1.0", in_process=True, weights="C:/pm_backends/mymodel/weights.h5",
    install_hint="Download weights.h5 into C:/pm_backends/mymodel/.")

class MyModelBackend(DetectorBackend):
    info = INFO

    def is_available(self):
        return Path(INFO.weights).exists()

    def detect_jobs(self, mode, jobs, **kwargs):
        mode = modes.normalise_mode(mode)
        out = []
        for ji, job in enumerate(jobs):
            stem = job.get("out_stem") or Path(job["path"]).stem
            csv_path = Path(kwargs["output_dir"]) / output_csv_name(
                mode, stem, kwargs.get("metric_cropsize"))
            ...  # run the model, measure with detectors.measure, build df
            df.to_csv(csv_path, index=False)
            DetectionManifest(model=INFO.name, model_version="1.0",
                              weights=INFO.weights, license=INFO.license,
                              params={...}).write(csv_path)
            out.append(df)
        return out

def make_backend():
    return MyModelBackend()
```

Once declared, the backend appears in the **Detection model** list (left panel, Settings)
whenever `is_available()` is true; Detect, Express and Digitize use the selected model. A
backend distributed with PebbleMapper is added to `_BACKENDS` in `detectors/registry.py`
instead.

## 6. Subprocess backend

### 6.1 `InstanceSubprocessBackend`

For a model that produces a label image, subclass
`detectors.instance_backend.InstanceSubprocessBackend`. The base class:

- runs the script in the model's conda environment and streams its log;
- stops the process tree on **Stop**;
- reads the instances file ([6.3](#63-instances-file)) and measures every instance;
- applies the ROI, writes the CSV and the manifest;
- in Quadrat mode with `saveplot`, draws `<stem>_overlay.png`.

`run_detect_jobs` writes the `.contours.json` outline file.

The subclass describes the model and the parameters its script needs:

```python
from pathlib import Path
from detectors.base import BackendInfo
from detectors.instance_backend import InstanceSubprocessBackend

INFO = BackendInfo(name="mymodel", display_name="My model (Author)",
                   framework="pytorch", license="BSD-3-Clause", output_type="instance",
                   env="pm-mymodel", in_process=False, weights="models/mymodel.pt",
                   install_hint="conda env create -f environment.yml; python scripts/download.py",
                   description="What it is and how to cite it.")

def make_backend():
    return MyBackend()

class MyBackend(InstanceSubprocessBackend):
    info = INFO
    env_name = "pm-mymodel"
    required_modules = ("torch",)
    script = Path(__file__).with_name("run.py")
    model_version = "mymodel 1.2"
    log_prefix = "mymodel"

    def is_available(self):
        return super().is_available() and Path("models/mymodel.pt").exists()

    def spec_params(self, mode, kwargs, resolution):
        return {"weights_dir": "models", "opts": {}}   # read by run.py from the spec
```

The base `is_available()` requires the script, the conda environment `env_name`, and every
module of `required_modules` in that environment (`missing_modules()` lists absent ones).
`run.py` runs in `pm-mymodel`, reads the spec and writes each job's `instances_path`.
`tests/test_instance_backend.py` tests such a backend without conda by replacing
`_run_streaming`.

### 6.2 Custom driver

For a model that does not produce a label image, use `detectors/stub_backend.py` (driver, in
the core environment) and `detectors/stub/run.py` (subprocess) as the template.

| Part | Role |
|---|---|
| Environment | the model's dependencies; `detectors/stub/environment.yml` is the smallest example. |
| `run.py` | run as `conda run -n <env> --no-capture-output python run.py --spec <json>`. Reads the spec (`{mode, params, canonical_columns, jobs: [{path, kstart, out_csv, instances_path}]}`), runs the model on each image and writes each job's instances file. Imports only its own environment, never `functions`, `detectors` or `mrcnn`. Prints progress with `flush=True`; the driver forwards it to the app's log. |
| Driver | a `DetectorBackend` in the core environment. Computes each CSV path with `output_csv_name`, builds the spec, runs the subprocess with `subprocess_runner.run_backend(env, script, spec, log_fn=..., stream=True, stop_check=...)`, reads each instances file with `subprocess_runner.read_instances_npz`, measures with `detectors.measure.measure_instances`, writes the CSV and the manifest, and emits the progress events. `is_available()` checks `subprocess_runner.conda_env_exists(<env>)` and the weights. |

The driver computes the file names and passes absolute paths in the spec, so the subprocess
does not need `functions.naming`. `run_backend` locates conda; `PEBBLE_CONDA_EXE` overrides
it.

### 6.3 Instances file

The measurement function (`detectors.measure.measure_mask`) runs in the core environment, so
the subprocess writes geometry and the driver measures it. Per job the subprocess writes
`instances_path` = `<out_csv>.instances.npz`:

| Array | dtype / shape | Content |
|---|---|---|
| `labels` | `int32`, `H×W` | 0 = background, `k` = pixels of instance `k` (1…N) |
| `scores` | `float32`, `N` | one confidence per instance (classifier score, mean mask probability or another monotone value); becomes the `Score` column |
| `shape` | `int64`, `(H, W)` | shape of the label image |

`numpy.savez_compressed(path, labels=..., scores=..., shape=...)` writes it; the stub writes
the same file with the standard library only. In the driver:

```python
instances = subprocess_runner.read_instances_npz(npz_path)      # [Instance(mask, score, y0, x0)]
df = measure_instances(instances, resolution, height=H)          # quadrat: y = H - row
df = measure_instances(instances, gsd, geotransform=gt)          # ortho: CRS coordinates
```

| Function | Behaviour |
|---|---|
| `read_instances_npz` | crops every instance to its bounding box, holes filled |
| `measure_instances` | runs `measure_mask` on each instance, offsets the centroid, maps it to world coordinates, applies an optional ROI (`roi_paths=load_roi_paths(job["roi_path"])`), fills `Mean_intensity` when given the image, and returns the output table. An instance `measure_mask` rejects (disconnected mask) is skipped and counted in the log. Keeps every outline in `df.attrs["contours"]`, in the frame of `x`/`y`; with `csv_path=` it writes `<csv stem>.contours.json`, otherwise `run_detect_jobs` writes it after `detect_jobs` returns and removes an outline file older than the CSV. |

Overlays and the report draw clasts from the outline file; without it they show the axes
only. Digitize runs the selected backend in Quadrat mode on the photograph and converts its
instances or outlines into editable proposals.

When a model cannot provide geometry, the subprocess may write `out_csv` itself (header equal
to `canonical_columns`, sizes in metres). The driver reads it with
`subprocess_runner.read_canonical_csv(csv, CANONICAL_COLUMNS)`, which rejects any other
header. Its sizes are then those of the model's own measurement, not PebbleMapper's.

### 6.4 `subprocess_runner.run_backend`

```python
subprocess_runner.run_backend(ENV, SCRIPT, spec, log_fn=log_fn,
                              stream=True, stop_check=stop_check,
                              timeout=kwargs.get("timeout"))
```

| `stream` | Behaviour |
|---|---|
| `True` | stdout and stderr forwarded line by line to `log_fn` (`TQDM_DISABLE=1` set for the child); `stop_check()` polled every 0.5 s. On stop the process tree is killed (`taskkill /T /F` on Windows, since `conda run` starts the interpreter as a grandchild) and `subprocess_runner.BackendStopped` is raised: catch it, emit `"stopped"` for the jobs and return empty frames. |
| `False` (default) | `subprocess.run` with captured output. |

A non-zero exit raises `RuntimeError` with the last lines of output.

### 6.5 Stub backend

```bash
conda env create -f detectors/stub/environment.yml   # creates 'pebble-stub'
```

The stub (`BackendInfo(dev_only=True)`) writes three synthetic ellipses per image, sized by
`metric_cropsize`, to test the round trip. `available_backends()` returns it, but the
**Detection model** list shows it only when the app starts with
`PEBBLEMAPPER_SHOW_DEV_BACKENDS=1`. Set `dev_only=True` on any test backend. Remove it with
`conda env remove -n pebble-stub`.

## 7. Checklist

- [ ] `info.name` is unique; `display_name` is shown to users; `install_hint` says what to install.
- [ ] `is_available()` is fast and true only when the model can run.
- [ ] An in-process backend that keeps its model loaded overrides `clear_cache()` (drops the model, returns `True`) so **Reload model** works; a subprocess backend keeps the default (returns `False`).
- [ ] `detect_jobs()` normalises `mode`, writes the CSVs to `output_dir` as `detectors.output_csv_name(mode, stem, metric_cropsize)`, and returns one DataFrame per job.
- [ ] Instance models measure through `detectors.measure`; a subprocess writes `.instances.npz` and the driver measures.
- [ ] `progress_callback` receives `"start"` / `"done"` / `"stopped"` / `"error"` per job; `stop_check` is polled; a subprocess runs with `stream=True`.
- [ ] Ortho mode reads the geotransform and handles tiling and world coordinates.
- [ ] GPL/AGPL models run as a subprocess in a user-installed environment and are not bundled.
- [ ] A manifest is written beside every CSV.
- [ ] The backend is declared in `user_detectors.json` (with `path` if the package is not installed) and sits in a short folder.
- [ ] Tests replace the subprocess as `tests/test_subprocess_backend.py` does, and skip real round trips when the environment is absent.

## 8. Existing plug-ins

Four plug-ins provide five models, each plug-in in its own repository with its licence,
environment file, download scripts and README. All four repositories are public.

| Repository | Model | Method | Licence | Runs on |
|---|---|---|---|---|
| [`soloyant/pebblemapper-backend-segmenteverygrain`](https://github.com/soloyant/pebblemapper-backend-segmenteverygrain) | [Segmenteverygrain](https://github.com/zsylvester/segmenteverygrain) (Sylvester et al., 2025) | U-Net prompts + SAM 2.1 masks | adapter MIT; code and U-Net weights Apache-2.0 (© Zoltán Sylvester); SAM 2.1 code and weights Apache-2.0 (© Meta) | GPU (about 1.5 GB) |
| [`soloyant/pebblemapper-backend-imagegrains`](https://github.com/soloyant/pebblemapper-backend-imagegrains) | [ImageGrains](https://github.com/dmair1989/imagegrains) 2.0 (Mair et al., 2026) and 1.2 (Mair et al., 2023), two backends (`make_backend`, `make_backend_v1`) | Cellpose-SAM transformer (2.0); Cellpose-2 CNN (1.2) | adapter MIT; code BSD-3 (© David Mair; Cellpose © HHMI); models CC BY 4.0 on Zenodo | GPU, 3 GB or more (2.0); CPU (1.2) |
| [`soloyant/pebblemapper-backend-pebblecounts`](https://github.com/soloyant/pebblemapper-backend-pebblecounts) | [PebbleCountsAuto](https://github.com/UP-RS-ESP/PebbleCounts) (Purinton & Bookhagen, 2019) | edge detection + ellipse filtering, no neural network | GPL-3.0-or-later (adapter too); downloaded by script, not bundled | CPU |
| [`soloyant/pebblemapper-backend-orthosam`](https://github.com/soloyant/pebblemapper-backend-orthosam) | [OrthoSAM](https://github.com/UP-RS-ESP/OrthoSAM) (Chan, Rheinwalt & Bookhagen, 2026) | Segment Anything (SAM v1) prompted on a point grid over tiles, with coarser passes | adapter MIT; code Apache-2.0 (© the OrthoSAM authors); SAM code and weights Apache-2.0 (© Meta) | GPU (4 GB with ViT-B) |

Each adapter calls its model's published API; its install script downloads the code from PyPI
or GitHub and the weights from where the authors publish them. The [README](../../README.md)
compares the six models on one quadrat photograph of `example_03_Etretat`.
