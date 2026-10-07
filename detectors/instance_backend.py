"""A ready-made driver for a backend that segments in its own environment.

The plug-in contract asks a backend for the canonical per-clast CSV. A model
that produces *instance geometry* (a label image) should not measure it
itself: :func:`detectors.measure.measure_mask` defines the numbers, so every
backend's table means the same thing as Mask R-CNN's.

:class:`InstanceSubprocessBackend` is the part every such backend would
otherwise repeat -- launch a script in the model's conda environment, stream
its log, honour Stop, read back the ``<csv>.instances.npz`` the script wrote
(``labels`` int32 HxW, 0 = background, k = instance k; ``scores`` float32 N),
measure each instance, write the CSV and the manifest, and draw the Detect
tab's overlay. A concrete backend supplies its :class:`BackendInfo`, the
environment name, the script, and the parameters that script needs; see
``docs/developer/adding-a-detection-model.md`` for a worked example of about
seventy lines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from detectors.base import (CANONICAL_COLUMNS, DetectionManifest,
                            DetectorBackend)
from detectors import subprocess_runner


class InstanceSubprocessBackend(DetectorBackend):
    """Base class: a model in its own environment that writes a label image.

    Subclasses set ``info``, ``env_name``, ``script`` and ``model_version``,
    and may override :meth:`spec_params`, :meth:`is_available` and
    ``log_prefix``.
    """

    info = None                      # BackendInfo
    env_name = ""                    # conda environment the script runs in
    required_modules: tuple = ()     # what that env must hold to be complete
    script: Optional[Path] = None    # the subprocess entry point
    model_version = ""               # what the manifest records
    log_prefix = "backend"

    # ------------------------------------------------------------------ gate
    def is_available(self) -> bool:
        if not (self.script and Path(self.script).exists()
                and subprocess_runner.conda_env_exists(self.env_name)):
            return False
        return not self.missing_modules()

    def missing_modules(self) -> List[str]:
        """The ``required_modules`` its env lacks: an env can exist by name
        and still be half built."""
        if not self.required_modules:
            return []
        prefix = subprocess_runner.conda_env_prefix(self.env_name)
        if prefix is None:
            return list(self.required_modules)
        return subprocess_runner.missing_modules(prefix, self.required_modules)

    # ------------------------------------------------------------ parameters
    def spec_params(self, mode: str, kwargs: dict, resolution: float) -> dict:
        """What the subprocess needs, on top of the common keys."""
        return {}

    # ------------------------------------------------------------------- run
    def detect_jobs(self, mode: str, jobs: list, **kwargs) -> List:
        import pandas as pd
        from functions import modes, naming

        mode = modes.normalise_mode(mode)
        out_dir = kwargs.get("output_dir")
        log_fn = kwargs.get("log_fn") or (lambda s: None)
        stop_check = kwargs.get("stop_check")
        progress = kwargs.get("progress_callback")
        crop = float(kwargs.get("metric_cropsize", 1.0) or 1.0)
        resolution = float(kwargs.get("resolution", 0.001) or 0.001)

        def _emit(ji, event, payload):
            if progress:
                try:
                    progress(ji, event, payload)
                except Exception:
                    pass

        if stop_check is not None and stop_check():
            for ji in range(len(jobs)):
                _emit(ji, "stopped", {})
            return [pd.DataFrame() for _ in jobs]

        geo = {}
        if mode == modes.ORTHO:
            for job in jobs:
                geo[job["path"]] = _geotransform(job["path"])
            first = geo[jobs[0]["path"]] if jobs else None
            if first is not None:
                resolution = abs(first[1])

        spec_jobs, out_csvs, npz_paths = [], [], []
        for job in jobs:
            stem = job.get("out_stem") or Path(job["path"]).stem
            csv_dir = Path(out_dir) if out_dir else Path(job["path"]).parent
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = (csv_dir / naming.detection_csv_name(stem, crop)
                        if mode == modes.ORTHO
                        else csv_dir / naming.quadrat_csv_name(stem))
            npz = Path(str(csv_path) + ".instances.npz")
            out_csvs.append(csv_path)
            npz_paths.append(npz)
            spec_jobs.append({"path": str(job["path"]),
                              "kstart": int(job.get("kstart", 0)),
                              "out_csv": str(csv_path),
                              "instances_path": str(npz)})

        params = {k: kwargs[k] for k in (
            "resolution", "metric_cropsize", "overlap", "min_confidence",
            "devicemode") if k in kwargs}
        params["resolution"] = resolution
        params.update(self.spec_params(mode, kwargs, resolution))
        spec = {"mode": mode, "params": params,
                "canonical_columns": list(CANONICAL_COLUMNS),
                "jobs": spec_jobs}

        run_started = time.time()
        for ji in range(len(jobs)):
            _emit(ji, "start", {"path": jobs[ji]["path"]})
        try:
            self._run_streaming(spec, log_fn, stop_check, kwargs.get("timeout"))
        except Exception as ex:
            import traceback
            for ji in range(len(jobs)):
                _emit(ji, "error", {"message": str(ex),
                                    "traceback": traceback.format_exc()})
            raise
        if stop_check is not None and stop_check():
            for ji in range(len(jobs)):
                _emit(ji, "stopped", {})
            return [pd.DataFrame() for _ in jobs]

        results = []
        for ji, (job, csv_path, npz) in enumerate(zip(jobs, out_csvs, npz_paths)):
            try:
                df, n_raw, n_skipped = self._measure_job(
                    mode, job, npz, resolution, geo.get(job["path"]), kwargs)
                if kwargs.get("saveresults", True):
                    df.to_csv(csv_path, index=False, float_format="%.5f")
                log_fn(f"[{self.log_prefix}] {Path(job['path']).name}: {n_raw} "
                       f"instances, {len(df)} measured ({n_skipped} unmeasurable "
                       f"skipped) -> {csv_path.name}")
                DetectionManifest(
                    model=self.info.name, model_version=self.model_version,
                    weights=self.info.weights, license=self.info.license,
                    params={k: v for k, v in params.items()
                            if k not in ("weights_dir",)},
                    gsd_m=resolution,
                ).write_for_completed_csv(csv_path, run_started)
                if kwargs.get("saveplot") and mode == modes.QUADRAT:
                    self._overlay(job, df, npz, kwargs.get("figures_dir"),
                                  log_fn, resolution)
                results.append(df)
                _emit(ji, "done", {"n_clasts": len(df)})
            except Exception as ex:
                import traceback
                log_fn(f"[{self.log_prefix}] measurement failed for "
                       f"{job['path']}: {ex}")
                _emit(ji, "error", {"message": str(ex),
                                    "traceback": traceback.format_exc()})
                results.append(pd.DataFrame())
        return results

    # ---------------------------------------------------------------- pieces
    def _run_streaming(self, spec, log_fn, stop_check, timeout):
        """Popen + stdout streaming + stop polling: what a long-running model
        needs and what ``subprocess_runner.run_backend`` does not offer."""
        conda_exe = subprocess_runner.find_conda_exe()
        if not conda_exe:
            raise RuntimeError("conda not found; set PEBBLE_CONDA_EXE.")
        fd, spec_path = tempfile.mkstemp(prefix=f"pm_{self.info.name}_spec_",
                                         suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(spec, fh)
        cmd = subprocess_runner.build_command(self.env_name, self.script,
                                              spec_path, conda_exe)
        log_fn(f"[subprocess] {self.env_name}: {Path(self.script).name} "
               f"({len(spec['jobs'])} job(s))")
        tail: List[str] = []
        try:
            env = dict(os.environ, TQDM_DISABLE="1", PYTHONIOENCODING="utf-8")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    bufsize=1, env=env)

            def _pump():
                for line in proc.stdout:
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    tail.append(line)
                    del tail[:-40]
                    log_fn(line)

            t = threading.Thread(target=_pump, daemon=True)
            t.start()
            t0 = time.time()
            while proc.poll() is None:
                if stop_check is not None and stop_check():
                    log_fn(f"[{self.log_prefix}] stop requested; terminating "
                           "the subprocess tree")
                    _kill_tree(proc)
                    t.join(5)
                    return
                if timeout and (time.time() - t0) > float(timeout):
                    _kill_tree(proc)
                    raise RuntimeError(f"Subprocess backend '{self.env_name}' "
                                       f"timed out after {timeout}s.")
                time.sleep(0.5)
            t.join(5)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Subprocess backend '{self.env_name}' failed "
                    f"(exit {proc.returncode}).\n" + "\n".join(tail[-20:]))
        finally:
            try:
                os.remove(spec_path)
            except OSError:
                pass

    @staticmethod
    def _measure_job(mode, job, npz_path, resolution, geotransform, kwargs):
        """Every instance through the core's shared measurement step."""
        import numpy as np
        import pandas as pd
        from scipy import ndimage as ndi
        from detectors.measure import measure_mask
        from functions import images as _images
        from functions import modes
        from functions.clasts_detection import (_load_roi_paths,
                                                _mask_mean_intensity)

        if not Path(npz_path).exists():
            raise RuntimeError(f"subprocess did not write {npz_path}")
        data = np.load(npz_path)
        labels = data["labels"]
        scores = data["scores"]
        n = int(scores.shape[0])
        image = (_read_ortho_rgb(job["path"]) if mode == modes.ORTHO
                 else _images.read_rgb(job["path"]))
        height = labels.shape[0]
        roi_paths = (_load_roi_paths(job.get("roi_path") or kwargs.get("roi_path"))
                     if mode == modes.QUADRAT else [])
        # Outlines in the CSV frame, as detectors.measure attaches them: the
        # wrapper writes the .contours.json and every figure draws them.
        from functions import clast_geometry as CG
        if mode == modes.ORTHO and geotransform is not None:
            to_frame = CG.world_frame(geotransform)
            decimals = CG.world_decimals(geotransform[1])
        else:
            to_frame, decimals = CG.quadrat_frame(height), 2
        contours = {}

        records, skipped = [], 0
        slices = ndi.find_objects(labels, max_label=n)
        pad = 2
        for k in range(1, n + 1):
            sl = slices[k - 1]
            if sl is None:
                skipped += 1
                continue
            y0 = max(0, sl[0].start - pad)
            y1 = min(labels.shape[0], sl[0].stop + pad)
            x0 = max(0, sl[1].start - pad)
            x1 = min(labels.shape[1], sl[1].stop + pad)
            mask = ndi.binary_fill_holes(labels[y0:y1, x0:x1] == k)
            meas = measure_mask(mask, float(scores[k - 1]), resolution)
            if meas is None:
                skipped += 1
                continue
            cx = meas["center_x"] + x0
            cy = meas["center_y"] + y0
            if roi_paths and not any(rp.contains_point((cx, cy)) for rp in roi_paths):
                continue
            if mode == modes.ORTHO and geotransform is not None:
                gt = geotransform
                wx = gt[0] + cx * gt[1] + cy * gt[2]
                wy = gt[3] + cx * gt[4] + cy * gt[5]
            else:
                wx, wy = cx, height - cy
            try:
                outline = CG.contour_from_measurement(
                    meas, to_frame=to_frame, offset=(x0, y0), decimals=decimals)
                if outline:
                    contours[len(records) + 1] = outline
            except Exception:
                pass    # an outline never costs a measurement
            records.append({
                "clast_ID": len(records) + 1, "x": wx, "y": wy,
                "Clast_length": meas["Clast_length"],
                "Clast_width": meas["Clast_width"],
                "Ellipse_major_axis": meas["Ellipse_major_axis"],
                "Ellipse_minor_axis": meas["Ellipse_minor_axis"],
                "Surface_area": meas["Surface_area"],
                "Perimeter": meas["Perimeter"],
                "Equivalent_diameter": meas["Equivalent_diameter"],
                "Eccentricity": meas["Eccentricity"],
                "Solidity": meas["Solidity"],
                "Mean_intensity": _mask_mean_intensity(image[y0:y1, x0:x1], mask),
                "Score": meas["Score"], "Orientation": meas["Orientation"],
            })
        df = pd.DataFrame(records, columns=list(CANONICAL_COLUMNS))
        df.attrs["contours"] = contours
        return df, n, skipped

    def _overlay(self, job, df, npz_path, figures_dir, log_fn, resolution):
        """``<stem>_overlay.png``: instance outlines and fitted ellipses, so
        the Detect tab's preview has something to show."""
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            from skimage.segmentation import find_boundaries
            from functions import images as _images
            image = _images.read_rgb(job["path"])
            labels = np.load(npz_path)["labels"]
            stem = job.get("out_stem") or Path(job["path"]).stem
            fig_dir = Path(figures_dir) if figures_dir else Path(job["path"]).parent
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image)
            ov = np.zeros(image.shape[:2] + (4,), dtype=float)
            ov[find_boundaries(labels, mode="outer")] = (1.0, 1.0, 0.0, 0.9)
            ax.imshow(ov)
            h = image.shape[0]
            for _, r in df.iterrows():
                ax.add_patch(Ellipse((r["x"], h - r["y"]),
                                     r["Ellipse_major_axis"] / resolution,
                                     r["Ellipse_minor_axis"] / resolution,
                                     angle=-(r["Orientation"] - 90),
                                     fill=False, ec="red", lw=0.8))
            ax.set_axis_off()
            ax.set_title(f"{self.info.display_name}: {len(df)} clasts")
            out = fig_dir / f"{stem}_overlay.png"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            log_fn(f"[{self.log_prefix}] overlay -> {out}")
        except Exception as ex:
            log_fn(f"[{self.log_prefix}] overlay skipped: {ex}")


def _geotransform(path):
    try:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            return None
        gt = ds.GetGeoTransform()
        ds = None
        return tuple(gt)
    except Exception:
        return None


def _read_ortho_rgb(path):
    import numpy as np
    from osgeo import gdal
    ds = gdal.Open(str(path))
    n = min(3, ds.RasterCount)
    arr = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(n)],
                   axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
        arr = (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    return arr


def _kill_tree(proc):
    """conda run spawns the real python as a grandchild; kill the whole tree."""
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass
