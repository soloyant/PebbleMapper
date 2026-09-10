"""What a stop means for the job that was running.

A quadrat photograph is one detection: a stop asked during it takes effect
after it, and its complete result is "done". A tiled ortho stopped mid-run
is partial, and stays "stopped".
"""
from __future__ import annotations

import pandas as pd
import pytest

from functions import clasts_detection as CD
from functions import modes


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(CD, "_build_model", lambda *a, **k: object())
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    monkeypatch.setattr(CD, "_detect_quadrat", lambda *a, **k: frame.copy())
    monkeypatch.setattr(CD, "_detect_ortho", lambda *a, **k: frame.copy())
    return frame


def _run(mode, tmp_path):
    events = []
    asked = {"stop": False}

    def stop_check():
        return asked["stop"]

    def cb(ji, event, payload):
        events.append(event)
        if event == "start":
            asked["stop"] = True      # the user presses Stop during the job
    CD.clasts_detect_jobs(mode, [{"path": str(tmp_path / "a.jpg")}],
                          resolution=0.001, output_dir=str(tmp_path),
                          stop_check=stop_check, progress_callback=cb,
                          saveresults=False, saveplot=False, plot=False)
    return events


def test_a_quadrat_photograph_stopped_during_its_detection_is_done(stub_model, tmp_path):
    assert _run(modes.QUADRAT, tmp_path) == ["start", "done"]


def test_a_tiled_ortho_stopped_mid_run_stays_stopped(stub_model, tmp_path):
    assert _run(modes.ORTHO, tmp_path) == ["start", "stopped"]


def test_a_stop_before_a_job_leaves_it_stopped_and_unstarted(stub_model, tmp_path):
    events = []
    CD.clasts_detect_jobs(modes.QUADRAT, [{"path": str(tmp_path / "a.jpg")}],
                          resolution=0.001, output_dir=str(tmp_path),
                          stop_check=lambda: True,
                          progress_callback=lambda ji, e, p: events.append(e),
                          saveresults=False, saveplot=False, plot=False)
    assert events == ["stopped"]
