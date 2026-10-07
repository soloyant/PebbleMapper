# tests/test_detectors.py — pluggable-detection scaffolding.
import json
import detectors
from detectors import base, measure, registry
from functions import clasts_detection, modes


def test_canonical_columns_are_the_detector_schema():
    # Single source of truth — the contract equals the detector's columns.
    assert detectors.CANONICAL_COLUMNS == clasts_detection._CLAST_COLUMNS
    assert detectors.CANONICAL_COLUMNS[:3] == ["clast_ID", "x", "y"]


def test_shared_measurement_reexports_the_detector_measure():
    assert measure.measure_mask is clasts_detection._measure_clast


def test_registry_has_maskrcnn_as_default():
    assert registry.default_backend_name() == "maskrcnn"
    names = [b.info.name for b in registry.all_backends()]
    assert "maskrcnn" in names
    assert registry.get_backend("maskrcnn") is not None
    assert registry.get_backend("does-not-exist") is None


def test_maskrcnn_backend_info_and_availability():
    b = registry.get_backend("maskrcnn")
    assert b.info.framework == "tensorflow"
    assert b.info.in_process is True
    assert b.info.output_type == "instance"
    assert isinstance(b.is_available(), bool)  # depends on weights presence
    # it must expose the legacy entry point, not reimplement detection
    assert hasattr(b, "detect_jobs")


def test_manifest_writes_sidecar(tmp_path):
    csv = tmp_path / "x_ws2.5m.csv"
    csv.write_text("clast_ID,x,y\n")
    m = base.DetectionManifest(model="maskrcnn", weights="zenodo",
                               license="MIT", params={"metric_cropsize": 2.5},
                               crs="EPSG:32630", gsd_m=0.004)
    out = m.write(csv)
    assert out is not None and out.exists()
    d = json.loads(out.read_text())
    assert d["model"] == "maskrcnn"
    assert d["params"]["metric_cropsize"] == 2.5
    assert d["crs"] == "EPSG:32630" and d["tool_version"]


def test_maskrcnn_backend_forwards_verbatim(monkeypatch, tmp_path):
    # Golden equivalence: the Mask R-CNN backend must pass every argument
    # straight through to clasts_detect_jobs and return its result unchanged,
    # so routing detection through the backend cannot alter the output.
    import pandas as pd
    from functions import clasts_detection
    captured = {}
    sentinel = [pd.DataFrame({"clast_ID": [1], "x": [0.0], "y": [0.0]})]

    def spy(mode, jobs, **kwargs):
        captured["mode"] = mode
        captured["jobs"] = jobs
        captured["kwargs"] = dict(kwargs)
        return sentinel
    monkeypatch.setattr(clasts_detection, "clasts_detect_jobs", spy)

    b = registry.get_backend("maskrcnn")
    out = b.detect_jobs(
        modes.ORTHO, [{"path": str(tmp_path / "o.tif"), "kstart": 0}],
        resolution=0.004, metric_cropsize=2.5, overlap=0.0,
        min_confidence=0.8, dedup_method="iou", dedup_overlap=0.3,
        devicemode="gpu", devicenumber=0, output_dir=str(tmp_path))

    assert out is sentinel                       # result returned unchanged
    assert captured["mode"] == modes.ORTHO
    assert captured["jobs"][0]["kstart"] == 0
    assert captured["kwargs"]["metric_cropsize"] == 2.5
    assert captured["kwargs"]["min_confidence"] == 0.8
    assert captured["kwargs"]["overlap"] == 0.0
    assert captured["kwargs"]["output_dir"] == str(tmp_path)


def test_register_adds_backend():
    class _Dummy(base.DetectorBackend):
        info = base.BackendInfo(name="_dummy_test", display_name="d",
                                framework="classical", license="MIT")
        def is_available(self):
            return False
        def detect_jobs(self, mode, jobs, **kwargs):
            return []
    registry.register(_Dummy())
    assert registry.get_backend("_dummy_test") is not None
    # unavailable backends are excluded from the available list
    assert "_dummy_test" not in [b.info.name for b in registry.available_backends()]


# --------------------------------------------------------------------------- #
#  Third-party backends: registerable without modifying the source            #
# --------------------------------------------------------------------------- #
def _write_third_party_backend(tmp_path):
    """A backend written the way an outside user would write one."""
    pkg = tmp_path / "outside_model"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "backend.py").write_text(
        "from detectors.base import BackendInfo, DetectorBackend\n"
        "\n"
        "class OutsideBackend(DetectorBackend):\n"
        "    info = BackendInfo(name='outside',\n"
        "                       display_name='Outside model',\n"
        "                       framework='pytorch',\n"
        "                       license='MIT',\n"
        "                       description='third-party model')\n"
        "    def is_available(self):\n"
        "        return True\n"
        "    def detect_jobs(self, mode, jobs, **kwargs):\n"
        "        return []\n"
        "\n"
        "def make_backend():\n"
        "    return OutsideBackend()\n",
        encoding="utf-8")
    return pkg


def test_third_party_backend_registers_without_touching_the_source(
        tmp_path, monkeypatch):
    """The whole point of the pluggable interface: someone else's model must
    become selectable without editing detectors/registry.py."""
    import sys
    from detectors import registry

    _write_third_party_backend(tmp_path)
    decl = tmp_path / "user_detectors.json"
    decl.write_text(
        '[{"module": "outside_model.backend", "factory": "make_backend"}]',
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PEBBLEMAPPER_DETECTORS", str(decl))
    sys.modules.pop("outside_model.backend", None)

    added = registry.load_user_backends()
    try:
        assert "outside" in added
        assert registry.get_backend("outside") is not None
        assert "outside" in [b.info.name for b in registry.available_backends()]
    finally:
        registry._BACKENDS.pop("outside", None)


def test_a_broken_third_party_backend_cannot_stop_startup(tmp_path, monkeypatch):
    """A model that explodes on import must be skipped, not fatal."""
    from detectors import registry

    decl = tmp_path / "user_detectors.json"
    decl.write_text(
        '[{"module": "no_such_module_anywhere", "factory": "make_backend"},'
        ' {"module": "", "factory": "x"}]',
        encoding="utf-8")
    monkeypatch.setenv("PEBBLEMAPPER_DETECTORS", str(decl))

    assert registry.load_user_backends() == []
    # The shipped backend is still there.
    assert registry.get_backend(registry.default_backend_name()) is not None


def test_register_rejects_something_that_is_not_a_backend():
    """A half-registered backend is worse than an absent one."""
    import pytest as _pytest
    from detectors import registry

    with _pytest.raises(TypeError):
        registry.register(object())


def test_missing_declaration_file_is_not_an_error(tmp_path, monkeypatch):
    from detectors import registry
    monkeypatch.setenv("PEBBLEMAPPER_DETECTORS", str(tmp_path / "absent.json"))
    assert registry.user_detectors_file() is None
    assert registry.load_user_backends() == []


def test_maskrcnn_backend_normalises_the_mode_before_forwarding(monkeypatch, tmp_path):
    # Scripts and persisted queues still pass the pre-rename spellings; the
    # backend hands the detector the canonical value whatever it was given.
    import pandas as pd
    captured = []
    monkeypatch.setattr(clasts_detection, "clasts_detect_jobs",
                        lambda mode, jobs, **kw: captured.append(mode) or [pd.DataFrame()])
    b = registry.get_backend("maskrcnn")
    for spelling in ("uav", "UAV", "Ortho", "terrestrial", "Quadrat", "QUADRAT"):
        b.detect_jobs(spelling, [{"path": str(tmp_path / "o.tif"), "kstart": 0}],
                      output_dir=str(tmp_path))
    assert captured == [modes.ORTHO, modes.ORTHO, modes.ORTHO,
                        modes.QUADRAT, modes.QUADRAT, modes.QUADRAT]
