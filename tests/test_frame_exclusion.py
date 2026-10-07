"""The quadrat frame is left out of detection, for every backend.

A rectified photograph carries its frame's thickness in its sidecar. The
detection wrapper turns that into an inner-rectangle ROI before any backend
runs, so a clast on the frame band is not counted, the manifest says what was
excluded, and the Detect tab shows the inset in its file list and carries the
choice in each job.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from detectors import InstanceSubprocessBackend, run_detect_jobs  # noqa: E402
from detectors.base import BackendInfo  # noqa: E402
from functions import quadrat_frame as QF  # noqa: E402

W, H, GSD, THICK = 400, 300, 0.001, 0.02      # 20 px of frame + 2 px margin on each edge


def _rectified(folder, name="q_rectified_GSD=0.001m.jpg", thickness=THICK):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    img = rng.integers(60, 200, (H, W, 3), dtype=np.uint8)
    p = folder / name
    cv2.imencode(".jpg", img)[1].tofile(str(p))
    side = {"PM_RECTIFIED": "yes", "PM_GSD": f"{GSD:.6f}", "PM_SIZE": f"{W}x{H}"}
    if thickness:
        side[QF.SIDECAR_KEY] = f"{thickness:.4f}"
    (folder / (name + ".json")).write_text(json.dumps(side), encoding="utf-8")
    return p


def _labels():
    """Two discs: one well inside, one sitting on the frame band."""
    lab = np.zeros((H, W), dtype=np.int32)
    yy, xx = np.mgrid[:H, :W]
    lab[(xx - 200) ** 2 + (yy - 150) ** 2 <= 20 ** 2] = 1
    lab[(xx - 10) ** 2 + (yy - 150) ** 2 <= 8 ** 2] = 2
    return lab


class _Fake(InstanceSubprocessBackend):
    info = BackendInfo(name="fake", display_name="Fake", framework="none",
                       license="MIT", output_type="instance", env="nowhere",
                       in_process=False, weights="w")
    env_name = "nowhere"
    script = Path("run.py")
    model_version = "fake"
    log_prefix = "fake"

    def __init__(self):
        self.jobs_seen = []

    def _run_streaming(self, spec, log_fn, stop_check, timeout):
        self.jobs_seen = list(spec["jobs"])
        for job in spec["jobs"]:
            np.savez_compressed(job["instances_path"], labels=_labels(),
                                scores=np.ones(2, dtype=np.float32),
                                shape=np.array([H, W]))


def test_the_wrapper_excludes_the_frame_band_and_says_so(tmp_path):
    p = _rectified(tmp_path / "in")
    logs = []
    be = _Fake()
    frames = run_detect_jobs(be, "quadrat", [{"path": str(p)}], resolution=GSD,
                             output_dir=str(tmp_path / "out"), log_fn=logs.append)
    df = frames[0]
    assert len(df) == 1, "the disc on the frame band is not a clast"
    assert df.iloc[0]["x"] == pytest.approx(200, abs=1)
    assert any("[frame]" in l and "20 px + 2 px margin" in l for l in logs), logs
    csv = tmp_path / "out" / "q_rectified_GSD=0.001m_individual_clasts.csv"
    man = json.loads(Path(str(csv) + ".manifest.json").read_text(encoding="utf-8"))
    assert man["params"]["frame_excluded"] == {
        "thickness_m": 0.02, "inset_px": 22, "measured_size_m": [0.356, 0.256]}


def test_the_switch_and_a_user_roi_are_respected(tmp_path):
    p = _rectified(tmp_path / "in")
    # Switched off: both discs count.
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(p)}], resolution=GSD,
                         output_dir=str(tmp_path / "off"), exclude_frame=False)[0]
    assert len(df) == 2
    # The user's own ROI wins over the automatic one.
    roi = tmp_path / "roi.geojson"
    roi.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon",
         "coordinates": [[[0, 0], [30, 0], [30, 300], [0, 300], [0, 0]]]}}]}),
        encoding="utf-8")
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(p), "roi_path": str(roi)}],
                         resolution=GSD, output_dir=str(tmp_path / "user"))[0]
    assert len(df) == 1 and df.iloc[0]["x"] == pytest.approx(10, abs=1)
    # The Detect tab passes the job's ROI as the call-wide kwarg: that ROI
    # wins too.
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(p)}], resolution=GSD,
                         output_dir=str(tmp_path / "kw"), roi_path=str(roi))[0]
    assert len(df) == 1 and df.iloc[0]["x"] == pytest.approx(10, abs=1)
    # A photograph without a frame record is measured whole.
    q = _rectified(tmp_path / "plain", name="p_rectified_GSD=0.001m.jpg", thickness=None)
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(q)}], resolution=GSD,
                         output_dir=str(tmp_path / "plain_out"))[0]
    assert len(df) == 2


def test_an_ortho_job_is_never_inset(tmp_path):
    p = _rectified(tmp_path / "in")
    from detectors.base import frame_rois_for
    jobs = [{"path": str(p)}]
    assert frame_rois_for("ortho", jobs, tmp_path) == {}
    assert "roi_path" not in jobs[0]
    assert list(frame_rois_for("quadrat", jobs, tmp_path)) == [0]
    assert Path(jobs[0]["roi_path"]).exists()


def test_a_thickness_given_by_hand_is_kept_beside_the_photograph(tmp_path):
    p = _rectified(tmp_path / "in", thickness=None)
    side = Path(str(p) + ".json")
    assert QF.frame_inset(p) is None
    QF.set_frame_thickness(p, 0.02)
    rec = json.loads(side.read_text(encoding="utf-8"))
    assert rec[QF.SIDECAR_KEY] == "0.0200"
    assert rec["PM_GSD"] == f"{GSD:.6f}", "the rest of the record is kept"
    assert QF.frame_inset(p).frame_px == 20
    QF.set_frame_thickness(p, None)
    rec = json.loads(side.read_text(encoding="utf-8"))
    assert QF.SIDECAR_KEY not in rec and rec["PM_GSD"] == f"{GSD:.6f}"
    # A record it cannot read is left as it is, not overwritten.
    side.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        QF.set_frame_thickness(p, 0.02)
    assert side.read_text(encoding="utf-8") == "{not json"


def test_detect_can_assume_a_frame_for_a_photograph_without_a_record(tmp_path):
    q = _rectified(tmp_path / "plain", name="p_rectified_GSD=0.001m.jpg",
                   thickness=None)
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(q)}], resolution=GSD,
                         output_dir=str(tmp_path / "whole"))[0]
    assert len(df) == 2
    df = run_detect_jobs(_Fake(), "quadrat", [{"path": str(q)}], resolution=GSD,
                         output_dir=str(tmp_path / "assumed"),
                         frame_fallback_m=THICK)[0]
    assert len(df) == 1, "the disc on the assumed band is left out"
    # A recorded thickness wins over the assumed one.
    p = _rectified(tmp_path / "rec", thickness=THICK)
    assert QF.frame_inset(p, 0.1).thickness_m == THICK
