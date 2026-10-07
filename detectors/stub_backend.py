"""Stub detection backend: the in-core driver of the cross-environment proof.

Does no real detection. It runs ``detectors/stub/run.py`` in a separate conda
environment (``pebble-stub``), reads back the instances file it writes,
measures every instance with the core's shared measurement step and writes
the canonical CSV + manifest — the mechanism a backend sharing neither
dependencies nor licence with the core env uses. ``is_available()`` is False
unless that environment exists (``conda env create -f
detectors/stub/environment.yml``), so it never appears in the GUI by
accident. It is the reference template for a real subprocess backend:
streamed launch, Stop support, progress events, canonical naming.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

from detectors.base import (BackendInfo, CANONICAL_COLUMNS, DetectionManifest,
                            DetectorBackend, output_csv_name)
from detectors import subprocess_runner

ENV_NAME = "pebble-stub"
_SCRIPT = Path(__file__).parent / "stub" / "run.py"

INFO = BackendInfo(
    name="stub",
    display_name="Stub (cross-env proof)",
    framework="classical",
    license="MIT",
    output_type="instance",
    env=ENV_NAME,
    in_process=False,
    weights=None,
    install_hint=("Developer/proof backend. Create its environment with "
                  "`conda env create -f detectors/stub/environment.yml`."),
    description=("Throwaway backend that fabricates synthetic clasts in a "
                 "separate conda env — proves the subprocess mechanism only."),
    dev_only=True,
)


class StubBackend(DetectorBackend):
    info = INFO

    def is_available(self) -> bool:
        """Only available once its dedicated conda env has been created."""
        return (self._SCRIPT_EXISTS()
                and subprocess_runner.conda_env_exists(ENV_NAME))

    @staticmethod
    def _SCRIPT_EXISTS() -> bool:
        return _SCRIPT.exists()

    def detect_jobs(self, mode: str, jobs: list, **kwargs) -> List:
        """Drive the stub in its own env; return one DataFrame per job and
        write each job's canonical CSV (+ manifest) into ``output_dir``."""
        import pandas as pd
        from functions import modes
        from detectors.measure import measure_instances

        mode = modes.normalise_mode(mode)
        out_dir = kwargs.get("output_dir")
        if not out_dir:
            raise ValueError("Stub backend requires output_dir.")
        crop = kwargs.get("metric_cropsize", 1.0)
        resolution = float(kwargs.get("resolution", 0.001) or 0.001)
        log_fn = kwargs.get("log_fn")
        stop_check = kwargs.get("stop_check")
        progress = kwargs.get("progress_callback")

        def _emit(ji, event, payload):
            if progress is None:
                return
            try:
                progress(ji, event, payload)
            except Exception:
                pass

        if stop_check is not None and stop_check():
            for ji in range(len(jobs)):
                _emit(ji, "stopped", {})
            return [pd.DataFrame() for _ in jobs]

        # The core app owns the naming convention; the subprocess receives
        # the destination paths and never needs the naming module.
        spec_jobs, out_csvs = [], []
        for job in jobs:
            stem = job.get("out_stem") or Path(job["path"]).stem
            csv_path = Path(out_dir) / output_csv_name(mode, stem, crop)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            out_csvs.append(csv_path)
            spec_jobs.append({
                "path": str(job["path"]),
                "kstart": int(job.get("kstart", 0)),
                "out_csv": str(csv_path),
                "instances_path": str(
                    subprocess_runner.instances_path_for(csv_path))})

        # Only the parameters the stub reads. min_confidence, the dedup and
        # tile-filter kwargs are Mask R-CNN's and are ignored here.
        params = {k: kwargs[k] for k in (
            "resolution", "metric_cropsize", "overlap", "min_confidence")
            if k in kwargs}
        spec = {"mode": mode, "params": params,
                "canonical_columns": list(CANONICAL_COLUMNS),
                "jobs": spec_jobs}

        run_started = time.time()
        for ji in range(len(jobs)):
            _emit(ji, "start", {"path": jobs[ji]["path"]})
        try:
            subprocess_runner.run_backend(
                ENV_NAME, _SCRIPT, spec, log_fn=log_fn,
                timeout=kwargs.get("timeout"),
                stream=True, stop_check=stop_check)
        except subprocess_runner.BackendStopped:
            for ji in range(len(jobs)):
                _emit(ji, "stopped", {})
            return [pd.DataFrame() for _ in jobs]

        results = []
        for ji, (job, csv_path) in enumerate(zip(jobs, out_csvs)):
            try:
                npz = subprocess_runner.instances_path_for(csv_path)
                instances = subprocess_runner.read_instances_npz(npz)
                shape = subprocess_runner.read_instances_shape(npz)
                height = (shape[0] if shape else
                          max((i.y0 + i.mask.shape[0] for i in instances),
                              default=0))
                df = measure_instances(instances, resolution, height=height,
                                       log_fn=log_fn)
                if kwargs.get("saveresults", True):
                    df.to_csv(csv_path, index=False, float_format="%.5f")
                    from functions.clast_geometry import (contours_of,
                                                          write_contours)
                    write_contours(csv_path, contours_of(df) or {})
                    DetectionManifest(
                        model=self.info.name, weights=self.info.weights,
                        license=self.info.license, params=params,
                        gsd_m=resolution).write_for_completed_csv(
                            csv_path, run_started)
                results.append(df)
                _emit(ji, "done", {"n_clasts": len(df)})
            except Exception as ex:
                import traceback
                if log_fn:
                    log_fn(f"[stub] {Path(job['path']).name}: {ex}")
                _emit(ji, "error", {"message": str(ex),
                                    "traceback": traceback.format_exc()})
                results.append(pd.DataFrame())
        return results
