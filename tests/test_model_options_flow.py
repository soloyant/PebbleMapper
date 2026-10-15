"""032 T015 (US3): only the values the user changed reach a model, never one
its release does not describe; what the model reports it used is recorded in
the result's record, and a release that reports nothing says so."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from detectors import InstanceSubprocessBackend  # noqa: E402
from detectors.base import BackendInfo  # noqa: E402

from tests.test_instance_backend import _labels, _photo  # noqa: E402
from tests.test_model_settings import DESC  # noqa: E402


def _desc_file(tmp_path, model="fakeplug"):
    d = copy.deepcopy(DESC)
    d["model"] = model
    f = tmp_path / "pm_settings.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    return f


class _Plug(InstanceSubprocessBackend):
    info = BackendInfo(name="fakeplug", display_name="Fake plug-in", framework="none",
                       license="MIT", output_type="instance", env="nowhere",
                       in_process=False, weights="w.bin")
    env_name = "nowhere"
    script = Path("run.py")
    model_version = "fake 1.1"
    log_prefix = "fakeplug"

    def __init__(self, settings_file=None, report=True):
        self.settings_file = settings_file
        self.report = report
        self.specs = []
        self.kwargs_seen = []

    def spec_params(self, mode, kwargs, resolution):
        self.kwargs_seen.append(dict(kwargs))
        return {"fake": dict(kwargs.get("model_options") or {})}

    def _run_streaming(self, spec, log_fn, stop_check, timeout):
        self.specs.append(spec)
        lab = _labels()
        used = {"threshold": 50, "min_size": 10, "use_sam": True, "dilation": 3}
        used.update(spec["params"]["fake"])
        resolved = {"format": 1, "settings": used,
                    "versions": {"fakepkg": "2.0"},
                    "device": {"kind": "cuda", "index": 0, "name": "Fake GPU"},
                    "reduction": {"factor": 0.5, "reason": "largest side 100 px"},
                    "tiles": {"size_px": 2000, "overlap_px": 600}}
        for job in spec["jobs"]:
            extra = {"resolved": np.array(json.dumps(resolved))} if self.report else {}
            np.savez_compressed(job["instances_path"], labels=lab,
                                scores=np.full(int(lab.max()), 0.5, dtype=np.float32),
                                shape=np.array(lab.shape), **extra)


def _run(tmp_path, backend, **kw):
    img = _photo(tmp_path / "IMG_1.jpg")
    out = tmp_path / "out"
    backend.detect_jobs("quadrat", [{"path": str(img)}], output_dir=str(out),
                        resolution=0.001, min_confidence=0.7, devicemode="GPU",
                        devicenumber=0, **kw)
    man = next(out.glob("*.manifest.json"))
    return json.loads(man.read_text(encoding="utf-8"))["params"]


def test_only_changed_and_described_values_are_sent(tmp_path):
    b = _Plug(_desc_file(tmp_path))
    _run(tmp_path, b, model_options={"threshold": 70, "nonsense": 1, "method": "b"})
    # 'nonsense' is not described; 'method' is ortho only: neither is sent.
    assert b.specs[0]["params"]["fake"] == {"threshold": 70}


def test_a_release_without_a_description_is_sent_nothing(tmp_path):
    b = _Plug(settings_file=tmp_path / "absent.json")
    _run(tmp_path, b, model_options={"threshold": 70})
    assert b.specs[0]["params"]["fake"] == {}


def test_an_invalid_value_is_never_sent(tmp_path):
    b = _Plug(_desc_file(tmp_path))
    _run(tmp_path, b, model_options={"threshold": 999})
    assert b.specs[0]["params"]["fake"] == {}


def test_the_record_holds_what_the_model_used(tmp_path):
    b = _Plug(_desc_file(tmp_path))
    p = _run(tmp_path, b, model_options={"threshold": 70})
    assert p["settings_used"] == {"threshold": 70, "min_size": 10, "use_sam": True,
                                  "dilation": 3}
    assert p["settings_changed"] == {"threshold": 70}
    assert p["model_versions"] == {"fakepkg": "2.0"}
    assert p["image_reduction"] == {"factor": 0.5, "reason": "largest side 100 px"}
    assert p["effective_tiles_px"] == {"size_px": 2000, "overlap_px": 600}
    assert p["device"]["mode"] == "GPU" and p["device"]["number"] == 0
    # Shared controls the model does not use are recorded as not applied.
    assert p["not_applied"] == {"confidence": 0.7}
    assert "min_confidence" not in p


def test_a_release_that_reports_nothing_says_so(tmp_path):
    b = _Plug(settings_file=tmp_path / "absent.json", report=False)
    p = _run(tmp_path, b)
    assert p["settings_used"] == "built-in, not reported"
    assert p["settings_changed"] == {}


def test_the_record_names_the_weights_the_install_checked(tmp_path):
    b = _Plug(_desc_file(tmp_path))
    b.installed_entry = {"files": [{"path": "weights/w.pth", "sha256": "ab" * 32},
                                   {"path": "check/x.jpg", "sha256": "cd" * 32}]}
    p = _run(tmp_path, b)
    assert p["weights_sha256"] == {"weights/w.pth": "ab" * 32}
    b2 = _Plug(_desc_file(tmp_path))
    assert _run(tmp_path, b2)["weights_sha256"].startswith("not recorded")


def test_the_chosen_card_is_the_only_one_the_run_sees(monkeypatch):
    from detectors import settings as S
    monkeypatch.setattr(S, "_GPU_NAMES", ["Card A", "Card B"])
    b = _Plug()
    assert b._visible_device({"devicemode": "GPU", "devicenumber": 1}) == "1"
    assert b._visible_device({"devicemode": "CPU", "devicenumber": 1}) is None
    with pytest.raises(RuntimeError, match="Graphics card 2 does not exist"):
        b._visible_device({"devicemode": "GPU", "devicenumber": 2})
    b.variant = "cpu"
    assert b._visible_device({"devicemode": "GPU", "devicenumber": 1}) is None


def test_setting_variables_are_removed_and_said(monkeypatch, capsys):
    from detectors import settings as S
    monkeypatch.setenv("PM_SEG_OPTIONS", '{"min_area": 5}')
    monkeypatch.setenv("PM_ORTHOSAM_MODEL", "vit_h")
    monkeypatch.setenv("PM_SEG_WEIGHTS", r"C:\w")           # a location: kept
    removed = S.strip_setting_env()
    import os
    assert set(removed) == {"PM_SEG_OPTIONS", "PM_ORTHOSAM_MODEL"}
    assert "PM_SEG_OPTIONS" not in os.environ and os.environ["PM_SEG_WEIGHTS"] == r"C:\w"
    assert "ignored" in capsys.readouterr().out
