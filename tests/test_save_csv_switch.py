"""Save CSV (Advanced) off: the run reports its count and writes nothing,
and an older CSV's manifest and outlines beside it are left alone
."""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from detectors import base


class _Quiet(base.DetectorBackend):
    info = base.BackendInfo(name="_quiet", display_name="q",
                            framework="classical", license="MIT")

    def __init__(self, frame):
        self._frame = frame

    def is_available(self):
        return True

    def detect_jobs(self, mode, jobs, **kwargs):
        assert kwargs.get("saveresults") is False
        return [self._frame]


def test_save_csv_off_leaves_an_older_csv_and_its_manifest_alone(tmp_path):
    job = {"path": str(tmp_path / "IMG_1.jpg")}
    csv = base.output_csv_path("quadrat", job, {"output_dir": str(tmp_path)})
    csv.write_text("clast_ID,x,y\n1,0,0\n", encoding="utf-8")
    manifest = csv.with_name(csv.name + ".manifest.json")
    manifest.write_text(json.dumps({"params": {"older": True}}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(csv, (old, old))
    os.utime(manifest, (old, old))

    events = []
    frame = pd.DataFrame({"clast_ID": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    base.run_detect_jobs(_Quiet(frame), "quadrat", [job],
                         output_dir=str(tmp_path), saveresults=False,
                         progress_callback=lambda ji, ev, p: events.append((ev, p)))
    assert events[-1][0] == "done" and events[-1][1]["n_clasts"] == 2
    assert csv.read_text(encoding="utf-8").count("\n") == 2, "the older CSV is untouched"
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"params": {"older": True}}
    assert not list(tmp_path.glob("*.contours.json"))
