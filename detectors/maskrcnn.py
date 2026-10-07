"""Mask R-CNN detection backend.

A thin wrapper over ``functions.clasts_detection.clasts_detect_jobs`` that
adds the backend interface and a best-effort provenance manifest; detection
itself runs unchanged in the core ``maskrcnn`` env.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from detectors.base import (BackendInfo, DetectionManifest, DetectorBackend,
                            output_csv_name)
from functions import modes

INFO = BackendInfo(
    name="maskrcnn",
    display_name="Mask R-CNN (Soloy et al., 2020)",
    framework="tensorflow",
    license="MIT (code) / CC-BY-4.0 weights (doi:10.5281/zenodo.20779877)",
    output_type="instance",
    env=None,
    in_process=True,
    weights="model_weights/mask_rcnn_clasts.h5 (or models/maskrcnn/)",
    install_hint="Included in the default 'maskrcnn' conda environment.",
    description=("Instance segmentation of individual clasts; the project's "
                 "trained Mask R-CNN. The original PebbleMapper detector."),
)


def _weights_candidates():
    """Locations the trained weights may live in."""
    from functions.clasts_detection import ROOT_DIR
    root = Path(ROOT_DIR)
    return [root / "model_weights" / "mask_rcnn_clasts.h5",
            root / "models" / "maskrcnn" / "mask_rcnn_clasts.h5"]


def _read_georef(image_path):
    """(crs, gsd_m) best-effort, for the manifest. Never raises.

    A plain photograph gets GDAL's default geotransform (0, 1, 0, 0, 0, 1),
    whose pixel size of 1.0 is a sentinel, not a measurement; only a real
    geotransform yields a GSD here. For rectified images the caller falls back
    to ``functions.gsd.effective_gsd``.
    """
    try:
        from osgeo import gdal, osr
        ds = gdal.Open(str(image_path))
        if ds is None:
            return None, None
        gt = ds.GetGeoTransform()
        _DEFAULT_GT = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        gsd = abs(gt[1]) if gt and tuple(gt) != _DEFAULT_GT else None
        crs = None
        wkt = ds.GetProjection()
        if wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(wkt)
            srs.AutoIdentifyEPSG()
            auth, code = srs.GetAuthorityName(None), srs.GetAuthorityCode(None)
            crs = f"{auth}:{code}" if auth and code else None
        ds = None
        return crs, gsd
    except Exception:
        return None, None


class MaskRCNNBackend(DetectorBackend):
    info = INFO

    def is_available(self) -> bool:
        return any(p.exists() for p in _weights_candidates())

    def clear_cache(self) -> bool:
        """Drop the cached model and TF session (and the cached ortho read):
        the next run rebuilds the model from the weights on disk. Always
        True: Mask R-CNN keeps its model between runs."""
        from functions.clasts_detection import clear_model_cache
        clear_model_cache()
        return True

    def detect_jobs(self, mode: str, jobs: list, **kwargs) -> List:
        """Forward to the in-process detector, then write a provenance
        manifest next to each written CSV (best-effort)."""
        from functions import clasts_detection
        mode = modes.normalise_mode(mode)
        # log_fn and timeout are meaningful only to subprocess backends.
        fwd = {k: v for k, v in kwargs.items()
               if k not in ("log_fn", "timeout")}
        import time as _time
        run_started = _time.time()
        results = clasts_detection.clasts_detect_jobs(mode, jobs, **fwd)
        self._write_manifests(mode, jobs, fwd, run_started)
        return results

    def _write_manifests(self, mode, jobs, kwargs, run_started):
        """One ``<csv>.manifest.json`` beside each CSV this run wrote.

        Everything this needs is imported at module level: a NameError here
        would be swallowed by the per-job guard and leave the previous
        model's manifest beside a fresh CSV.
        """
        out_dir = kwargs.get("output_dir")
        if not out_dir:
            return
        crop = kwargs.get("metric_cropsize")
        params = {k: kwargs[k] for k in (
            "resolution", "metric_cropsize", "overlap", "min_confidence",
            "devicemode", "devicenumber", "dedup_method", "dedup_overlap",
            "dark_threshold", "bright_threshold", "nodata_max_frac")
            if k in kwargs}
        for job in jobs:
            try:
                stem = job.get("out_stem") or Path(job["path"]).stem
                # The manifest must sit beside the CSV this mode writes.
                csv_path = Path(out_dir) / output_csv_name(mode, stem, crop)
                crs, gsd = _read_georef(job["path"])
                if gsd is None:
                    # A rectified quadrat photograph knows its GSD via sidecar/filename.
                    try:
                        from functions.gsd import effective_gsd
                        gsd = effective_gsd(job["path"]).gsd
                    except Exception:
                        pass
                DetectionManifest(
                    model=self.info.name, weights=self.info.weights,
                    license=self.info.license, params=params,
                    crs=crs, gsd_m=gsd).write_for_completed_csv(
                        csv_path, run_started)
            except Exception:
                pass  # provenance is best-effort
