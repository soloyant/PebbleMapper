"""Every detection path leaves ``<csv stem>.contours.json`` beside its CSV.

Mask R-CNN in Quadrat mode (a mocked model returning synthetic masks) and in
Ortho mode through a stop and a resume, the shared ``measure_instances`` and
the ``run_detect_jobs`` wrapper for external backends, Digitize's
``run_gauge`` (injected detector, resampled detection) and a backend driven
through ``gauge.backend_detect_fn``, and merge renumbering clast_IDs. Each
outline must lie in the CSV's own frame: the row's centroid falls inside
its polygon. The drawer's model options hide developer backends.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from functions import clast_geometry as CG


def _ellipse(shape, cx, cy, a, b, t_deg):
    t = math.radians(t_deg)
    dx, dy = math.cos(t), -math.sin(t)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    u = (cc - cx) * dx + (rr - cy) * dy
    v = -(cc - cx) * dy + (rr - cy) * dx
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def _assert_outlines_hold_centroids(df, contours, *, min_n=None):
    assert contours is not None
    ids = [int(i) for i in df["clast_ID"]]
    assert set(contours) == set(ids)
    if min_n is not None:
        assert len(ids) >= min_n
    for _, r in df.iterrows():
        poly = Polygon(np.asarray(contours[int(r["clast_ID"])], dtype=float))
        assert poly.is_valid and poly.contains(Point(r["x"], r["y"])), r["clast_ID"]


class _MaskModel:
    """``model.detect`` returning fixed rotated ellipses on any image."""

    def __init__(self, specs):
        self.specs = specs            # [(fx, fy, a, b, t)] in image fractions
        self.calls = 0

    def detect(self, images, verbose=0):
        self.calls += 1
        h, w = images[0].shape[:2]
        masks = [_ellipse((h, w), fx * w, fy * h, a, b, t)
                 for fx, fy, a, b, t in self.specs]
        n = len(masks)
        return [{"masks": np.dstack(masks) if n else np.zeros((h, w, 0), bool),
                 "scores": np.linspace(0.9, 0.8, n),
                 "rois": np.zeros((n, 4)), "class_ids": np.ones(n, int)}]


def _photo(path, w=240, h=180, seed=3):
    from PIL import Image
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(40, 220, (h, w, 3), dtype=np.uint8)).save(path)
    return path


# --------------------------------------------------------------------------- #
#  Mask R-CNN, Quadrat                                                         #
# --------------------------------------------------------------------------- #
def test_quadrat_detection_writes_contours_and_the_overlay(tmp_path):
    pytest.importorskip("tensorflow")
    from functions.clasts_detection import _detect_quadrat
    img = _photo(tmp_path / "q.jpg")
    model = _MaskModel([(0.25, 0.3, 30, 12, 20), (0.6, 0.6, 25, 10, 120),
                        (0.8, 0.25, 18, 14, 80)])
    df = _detect_quadrat(model, str(img), 0.001, False, True, True,
                         output_dir=str(tmp_path / "v"),
                         figures_dir=str(tmp_path / "f"))
    assert len(df) == 3
    csv = tmp_path / "v" / "q_individual_clasts.csv"
    side = CG.contours_path_for(csv)
    assert csv.exists() and side.exists()
    doc = json.loads(side.read_text(encoding="utf-8"))
    assert doc["frame"] == "pixels" and doc["n"] == 3
    back = pd.read_csv(csv)
    _assert_outlines_hold_centroids(back, CG.read_contours(csv))
    assert CG.contours_of(df) is not None
    assert (tmp_path / "f" / "q_overlay.png").exists()
    assert not (tmp_path / "f" / "q_ellipses.png").exists()


# --------------------------------------------------------------------------- #
#  Mask R-CNN, Ortho, stopped and resumed                                      #
# --------------------------------------------------------------------------- #
def _geotiff(path, w=200, h=200, gt=(500000.0, 0.01, 0.0, 6000000.0, 0.0, -0.01)):
    from osgeo import gdal
    rng = np.random.default_rng(7)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 3, gdal.GDT_Byte)
    ds.SetGeoTransform(gt)
    for b in range(3):
        ds.GetRasterBand(b + 1).WriteArray(
            rng.integers(30, 220, (h, w), dtype=np.uint8))
    ds.FlushCache()
    ds = None
    return path


def test_ortho_contours_survive_a_stop_and_a_resume(tmp_path):
    pytest.importorskip("tensorflow")
    from functions import clasts_detection as D
    tif = _geotiff(tmp_path / "o.tif")
    out = tmp_path / "vec"
    model = _MaskModel([(0.4, 0.5, 20, 8, 30), (0.75, 0.3, 12, 9, 100)])
    polls = {"n": 0}

    def stop_after_two_tiles():
        polls["n"] += 1
        return polls["n"] > 2

    kw = dict(plot=False, saveplot=False, saveresults=True, ksaveint=1,
              overlap=0.0, dedup_method="iou", output_dir=str(out))
    D._ORTHO_READ_CACHE["key"] = None
    part = D._detect_ortho(model, str(tif), 1.0, kstart=0, dedup_overlap=None,
                           stop_check=stop_after_two_tiles, **kw)
    assert len(part) == 4                                  # two tiles of 2x2
    run_csv = out / "o_ws1m.run.csv"
    run_side = Path(D._run_contours_path_for(str(run_csv)))
    assert run_csv.exists() and run_side.exists()
    # A partial run leaves outlines beside its partial CSV too.
    _assert_outlines_hold_centroids(pd.read_csv(out / "o_ws1m.csv"),
                                    CG.read_contours(out / "o_ws1m.csv"))

    full = D._detect_ortho(model, str(tif), 1.0, kstart=0, dedup_overlap=0.3,
                           stop_check=None, **kw)
    assert len(full) == 8 and model.calls == 4             # resumed, not redone
    csv = out / "o_ws1m.csv"
    contours = CG.read_contours(csv)
    assert contours.frame == "world"
    _assert_outlines_hold_centroids(pd.read_csv(csv), contours, min_n=8)
    assert not run_csv.exists() and not run_side.exists()
    # World coordinates carry a tenth of a pixel (0.01 m px -> 3 decimals).
    sample = next(iter(contours.values()))
    assert all(round(v, 3) == pytest.approx(v) for v in sample.ravel())


# --------------------------------------------------------------------------- #
#  External backends: measure_instances and the run_detect_jobs wrapper        #
# --------------------------------------------------------------------------- #
def _labels(h=90, w=120):
    lab = np.zeros((h, w), dtype=np.int32)
    lab[_ellipse((h, w), 35, 40, 22, 9, 35)] = 1
    lab[_ellipse((h, w), 85, 55, 15, 11, 150)] = 2
    return lab


def test_measure_instances_attaches_and_writes_outlines_in_both_frames(tmp_path):
    from detectors import measure, subprocess_runner as SR
    npz = tmp_path / "i.npz"
    np.savez(npz, labels=_labels(), scores=np.array([0.9, 0.8], np.float32),
             shape=np.array([90, 120]))
    inst = SR.read_instances_npz(npz)
    assert SR.read_instances_shape(npz) == (90, 120)
    csv = tmp_path / "img_individual_clasts.csv"
    df = measure.measure_instances(inst, 0.001, height=90, csv_path=csv)
    _assert_outlines_hold_centroids(df, CG.contours_of(df))
    _assert_outlines_hold_centroids(df, CG.read_contours(csv))
    gt = (400000.0, 0.005, 0.0, 5000000.0, 0.0, -0.005)
    dfw = measure.measure_instances(inst, 0.005, geotransform=gt)
    assert CG.contours_of(dfw).frame == "world"
    _assert_outlines_hold_centroids(dfw, CG.contours_of(dfw))


class _FakeSeg:
    """An in-process backend that segments discs, measures them with the
    shared step and writes only the CSV (no sidecar of its own)."""

    def __new__(cls, name="fakeseg", with_npz=True):
        from detectors.base import BackendInfo, DetectorBackend

        class Impl(DetectorBackend):
            info = BackendInfo(name=name, display_name="Fake segmenter",
                               framework="classical", license="MIT")

            def is_available(self):
                return True

            def detect_jobs(self, mode, jobs, **kw):
                from PIL import Image
                from detectors import measure, subprocess_runner as SR
                from detectors.base import output_csv_path
                self.seen = kw
                out = []
                for job in jobs:
                    csv = output_csv_path(mode, job, kw)
                    csv.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(job["path"]) as im:
                        w, h = im.size
                    r = max(4, int(round(0.06 * min(h, w))))
                    lab = np.zeros((h, w), np.int32)
                    for k, fx in enumerate((0.25, 0.5, 0.75), start=1):
                        lab[_ellipse((h, w), fx * w, 0.5 * h, r, r, 0)] = k
                    npz = SR.instances_path_for(csv)
                    np.savez(npz, labels=lab,
                             scores=np.array([0.9, 0.8, 0.95], np.float32),
                             shape=np.array([h, w]))
                    df = measure.measure_instances(
                        SR.read_instances_npz(npz),
                        float(kw.get("resolution", 1.0)), height=h)
                    if not with_npz:
                        npz.unlink()
                    df.to_csv(csv, index=False)
                    out.append(df)
                return out
        return Impl()


def test_run_detect_jobs_persists_attached_outlines_and_drops_stale_ones(tmp_path):
    from detectors.base import run_detect_jobs, output_csv_path
    from functions import modes
    img = _photo(tmp_path / "p.jpg", w=160, h=120)
    b = _FakeSeg()
    job = {"path": str(img)}
    kw = dict(output_dir=str(tmp_path / "o"), resolution=0.001)
    frames = run_detect_jobs(b, modes.QUADRAT, [job], **kw)
    csv = output_csv_path(modes.QUADRAT, job, kw)
    _assert_outlines_hold_centroids(pd.read_csv(csv), CG.read_contours(csv))
    assert len(frames[0]) == 3

    # A backend that rewrites the CSV with no outlines: the old sidecar goes.
    from detectors.base import BackendInfo, DetectorBackend

    class Bare(DetectorBackend):
        info = BackendInfo(name="bare", display_name="Bare", framework="x",
                           license="MIT")

        def is_available(self):
            return True

        def detect_jobs(self, mode, jobs, **k):
            df = pd.read_csv(csv)
            df.to_csv(csv, index=False)
            return [df]
    import os
    side = CG.contours_path_for(csv)
    os.utime(side, (side.stat().st_mtime - 60, side.stat().st_mtime - 60))
    run_detect_jobs(Bare(), modes.QUADRAT, [job], **kw)
    assert not side.exists()


# --------------------------------------------------------------------------- #
#  Digitize: run_gauge                                                         #
# --------------------------------------------------------------------------- #
def test_run_gauge_writes_outlines_in_original_pixels(tmp_path):
    from functions import gauge as Q
    img = _photo(tmp_path / "g.jpg", w=400, h=300)
    seen = {}

    def detect_fn(**kw):
        from functions.clasts_detection import _measure_clast
        path = kw["jobs"][0]["path"]
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        seen["size"] = (w, h)
        rows, outlines = [], {}
        for k, (fx, t) in enumerate(((0.3, 30), (0.7, 120)), start=1):
            m = _ellipse((h, w), fx * w, 0.5 * h, 30, 12, t)
            meas = _measure_clast(m, 0.9, kw["resolution"])
            outlines[k] = CG.contour_from_measurement(
                meas, to_frame=CG.quadrat_frame(h))
            rows.append({"clast_ID": k, "x": meas["center_x"],
                         "y": h - meas["center_y"],
                         "Clast_length": meas["Clast_length"],
                         "Clast_width": meas["Clast_width"],
                         "Orientation": meas["Orientation"], "Score": 0.9})
        return [CG.attach_contours(pd.DataFrame(rows), outlines, frame="pixels")]

    scale = Q.GaugeScale.from_gsd(0.001, "mm")
    res = Q.run_gauge(img, scale, tmp_path / "out", detect_fn=detect_fn,
                      detect_scale=0.5, model="maskrcnn")
    assert seen["size"] == (200, 150)
    csv = Path(res["csv"])
    side = CG.contours_path_for(csv)
    assert Path(res["contours"]) == side and side.name == "g_gauge.contours.json"
    assert json.loads(Path(res["json"]).read_text())["contours"] == side.name
    back = pd.read_csv(csv)
    contours = CG.read_contours(csv)
    _assert_outlines_hold_centroids(back, contours)
    # Original pixels: the ellipse spans 2 * 26.7 px across on the half-size
    # copy, so about 107 px in the photograph.
    span = np.ptp(np.asarray(contours[1])[:, 0])
    assert 95 < span < 115
    figs = Q.write_gauge_figures(img, res["df"], scale, tmp_path / "out",
                                 res["stem"], disclaimer=None)
    assert Path(figs["overlay"]).exists()
    fig = Q.gauge_overlay_figure(img, res["df"], scale, disclaimer=None)
    assert fig.pm_meta["n_outlines"] == 2 and fig.pm_meta["n_chords"] == 2


def test_backend_detect_fn_runs_a_backend_and_keeps_its_instances(tmp_path, monkeypatch):
    from detectors import registry
    from functions import gauge as Q
    fake = _FakeSeg()
    monkeypatch.setitem(registry._BACKENDS, "fakeseg", fake)
    img = _photo(tmp_path / "b.jpg", w=400, h=300)
    kept = {}
    fn = Q.backend_detect_fn(fake, tmp_path / "work", keep=kept)
    scale = Q.GaugeScale.from_gsd(0.002, "mm")
    res = Q.run_gauge(img, scale, tmp_path / "out", detect_fn=fn,
                      detect_scale=0.5, model="fakeseg")
    assert fake.seen["resolution"] == 2.0 and fake.seen["saveresults"] is True
    assert len(kept["instances"]) == 3 and kept["shape"] == (150, 200)
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["model"]["backend"] == "fakeseg"
    back = pd.read_csv(res["csv"])
    assert len(back) == 3
    _assert_outlines_hold_centroids(back, CG.read_contours(res["csv"]))
    np.testing.assert_allclose(back["Clast_length"], 72.0, rtol=0.1)


def test_backend_detect_fn_falls_back_to_the_outlines_without_instances(tmp_path):
    from functions import gauge as Q
    fake = _FakeSeg(with_npz=False)
    img = _photo(tmp_path / "c.jpg", w=200, h=150)
    kept = {}
    frames = Q.backend_detect_fn(fake, tmp_path / "w", keep=kept)(
        mode="quadrat", jobs=[{"path": str(img)}], resolution=1.0)
    assert kept["instances"] is None and len(kept["contours"]) == 3
    assert CG.contours_of(frames[0]) is not None


# --------------------------------------------------------------------------- #
#  Merge renumbers and keeps the outlines of the kept rows                     #
# --------------------------------------------------------------------------- #
def _frame(rows):
    cols = ["clast_ID", "x", "y", "Clast_length", "Clast_width",
            "Ellipse_major_axis", "Ellipse_minor_axis", "Score", "Orientation"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)


def _square(x, y, h):
    return [[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]]


def test_merge_carries_outlines_under_renumbered_ids(tmp_path):
    from functions.clasts_merge import merge_csvs
    small = _frame([(4, 1.0, 1.0, 0.2, 0.1, 0.2, 0.1, 0.9, 90.0),     # dup of large 7
                    (9, 5.0, 5.0, 0.2, 0.1, 0.2, 0.1, 0.9, 0.0)])
    large = _frame([(7, 1.01, 1.0, 0.2, 0.1, 0.2, 0.1, 0.8, 90.0),
                    (2, 9.0, 1.0, 0.3, 0.1, 0.3, 0.1, 0.8, 45.0)])
    ps, pl = tmp_path / "s_ws1m.csv", tmp_path / "l_ws2m.csv"
    small.to_csv(ps, index=False)
    large.to_csv(pl, index=False)
    CG.write_contours(ps, {4: _square(1.0, 1.0, 0.05), 9: _square(5.0, 5.0, 0.05)},
                      frame="world")
    CG.write_contours(pl, {7: _square(1.01, 1.0, 0.06), 2: _square(9.0, 1.0, 0.07)},
                      frame="world")
    out = tmp_path / "merged.csv"
    merged = merge_csvs(str(ps), str(pl), str(out))
    assert len(merged) == 3 and list(merged["clast_ID"]) == [1, 2, 3]
    contours = CG.read_contours(out)
    assert contours.frame == "world"
    _assert_outlines_hold_centroids(merged, contours)
    # The duplicate kept the large window's row and its (wider) outline.
    row = merged[np.isclose(merged["x"], 1.01)].iloc[0]
    assert np.ptp(contours[int(row["clast_ID"])][:, 0]) == pytest.approx(0.12)

    # Inputs without sidecars: no merged sidecar, and a stale one is removed.
    CG.contours_path_for(ps).unlink()
    CG.contours_path_for(pl).unlink()
    merge_csvs(str(ps), str(pl), str(out))
    assert CG.read_contours(out) is None


# --------------------------------------------------------------------------- #
#  The drawer's options hide developer backends                                #
# --------------------------------------------------------------------------- #
def test_selector_options_hide_dev_only_backends_unless_asked(monkeypatch):
    from detectors import registry
    from detectors.base import BackendInfo, DetectorBackend

    class Dev(DetectorBackend):
        info = BackendInfo(name="devproof", display_name="Dev proof",
                           framework="x", license="MIT", dev_only=True)

        def is_available(self):
            return True

        def detect_jobs(self, mode, jobs, **kw):
            return []
    monkeypatch.setitem(registry._BACKENDS, "devproof", Dev())
    monkeypatch.delenv(registry.DEV_BACKENDS_ENV, raising=False)
    assert registry.get_backend("stub").info.dev_only is True
    assert "devproof" in [b.info.name for b in registry.available_backends()]
    opts = registry.selector_options()
    assert "devproof" not in opts and "stub" not in opts
    assert next(iter(opts)) == "maskrcnn"
    assert registry.normalise_model_name("devproof") == "maskrcnn"
    monkeypatch.setenv(registry.DEV_BACKENDS_ENV, "1")
    assert "devproof" in registry.selector_options()
    assert registry.normalise_model_name("devproof") == "devproof"
