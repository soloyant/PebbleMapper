"""Detect takes a photograph's object scale, drawn in Digitize.

*No GSD? Scale from an object* stays in Digitize; it is another way of
measuring the GSD, so Detect reads the segments saved beside a photograph's
Digitize CSV and runs that file with them. An object of known length gives
metres per pixel like any GSD; one of unknown length gives its own unit, and
the run then says so (a unit column, no metre-labelled plots).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from functions import gauge as Q  # noqa: E402

BOOT = {"name": "Boot", "length": 300.0, "unit": "mm"}
UNKNOWN = {"name": "scale bar", "length": None, "unit": None}
SEGS = [{"p0": [100.0, 40.0], "p1": [200.0, 40.0], "object": "Boot"}]


def _photo(path, w=60, h=40, seed=4):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))
    return path


# --------------------------------------------------------------------------- #
#  The engine: the sidecar alone carries the scale                             #
# --------------------------------------------------------------------------- #
def test_the_sidecar_gives_the_scale_without_the_project_library(tmp_path):
    csv = tmp_path / "IMG_1_truth.csv"
    Q.write_scale_segments(csv, "IMG_1.jpg", SEGS, [BOOT])
    scale, path = Q.scale_from_sidecar([csv])
    assert path == Q.scale_segments_path_for(csv)
    assert scale.is_metric and scale.unit_label == "mm"
    # 100 px = 300 mm.
    assert scale.metres_per_px == pytest.approx(0.003)
    assert scale.px_per_unit == pytest.approx(1 / 3, rel=1e-6)


def test_an_object_of_unknown_length_is_its_own_unit(tmp_path):
    csv = tmp_path / "IMG_1_truth.csv"
    Q.write_scale_segments(csv, "IMG_1.jpg",
                           [dict(SEGS[0], object="scale bar")], [UNKNOWN])
    scale, _ = Q.scale_from_sidecar([csv])
    assert not scale.is_metric
    assert scale.unit_label == "scale bar" and scale.px_per_unit == pytest.approx(100.0)


def test_the_first_candidate_with_a_sidecar_wins_and_none_means_none(tmp_path):
    beside = tmp_path / "beside" / "IMG_1_truth.csv"
    validation = tmp_path / "validation" / "IMG_1_truth.csv"
    Q.write_scale_segments(beside, "IMG_1.jpg", SEGS, [BOOT])
    assert Q.scale_from_sidecar([validation, beside])[1] == \
        Q.scale_segments_path_for(beside)
    assert Q.scale_from_sidecar([validation]) == (None, None)
    assert Q.scale_from_sidecar([]) == (None, None)


# --------------------------------------------------------------------------- #
#  The tab: a queued job runs in that scale                                    #
# --------------------------------------------------------------------------- #
_FIELDS = ("current_project", "det_dir", "det_mode", "det_files",
           "det_files_checked", "det_jobs", "det_resolution",
           "det_exclude_frame")


@pytest.fixture
def clean():
    from gui.app import state
    saved = {k: getattr(state, k) for k in _FIELDS}
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    state.det_jobs = []
    state.det_files = []
    state.det_files_checked = {}
    state.det_resolution = 0.001
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


@pytest.fixture
def project(tmp_path):
    import functions.layout as layout
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    (tmp_path / "P").mkdir()
    images = layout.project_path("P", "images")
    _photo(images / "IMG_1.jpg")
    yield {"root": tmp_path / "P", "images": images,
           "validation": layout.project_path("P", "validation")}
    layout.set_datasets_root(original, persist=False)


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _build(clean):
    from nicegui import ui
    from gui.app import build_detection_tab
    with ui.column() as col:
        build_detection_tab()
    clean[1].append(col)
    return col


def _press(col, label):
    from nicegui.events import ClickEventArguments, handle_event
    btn = next(e for e in _walk(col) if e.__class__.__name__ == "Button"
               and str(getattr(e, "text", "")) == label)
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))


def _queue_one(clean, project, objects, segments=SEGS):
    from gui.app import MODE_LABELS  # noqa: F401  (import guards the module)
    state, _ = clean
    Q.write_scale_segments(project["validation"] / "IMG_1_truth.csv",
                           "IMG_1.jpg", segments, objects)
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    col = _build(clean)
    # Typing the folder is what lists it; a pre-set value would not fire.
    inp = next(e for e in _walk(col)
               if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    assert state.det_files == ["IMG_1.jpg"]
    _press(col, "Add to queue")
    assert len(state.det_jobs) == 1
    return col, state.det_jobs[0]


def test_a_queued_job_takes_the_photographs_object_scale(clean, project):
    col, job = _queue_one(clean, project, [BOOT])
    assert job["resolution"] == pytest.approx(0.003), "300 mm over 100 px"
    assert job["gsd_source"] == "object: mm"
    assert job["unit_label"] is None, "a known length is metres, like a GSD"
    # The file list and the queue row say where the scale came from.
    texts = [str(getattr(e, "text", "")) for e in _walk(col)]
    assert any(t.startswith("Object scale 3.000 mm/px") for t in texts), texts
    assert any("object: mm" in t for t in texts)


def test_an_unknown_length_runs_in_its_own_unit(clean, project):
    col, job = _queue_one(clean, project,
                          [UNKNOWN], [dict(SEGS[0], object="scale bar")])
    assert job["unit_label"] == "scale bar"
    assert job["resolution"] == pytest.approx(0.01), "1 px = 1/100 scale bar"
    assert job["gsd_source"] == "object: scale bar"
    texts = [str(getattr(e, "text", "")) for e in _walk(col)]
    assert any("not metres" in t for t in texts), texts


def test_without_a_sidecar_the_gsd_rules(clean, project):
    state, _ = clean
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    state.det_resolution = 0.002
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    _press(col, "Add to queue")
    job = state.det_jobs[0]
    assert job["resolution"] == pytest.approx(0.002)
    assert job["gsd_source"] == "global" and job["unit_label"] is None


# --------------------------------------------------------------------------- #
#  The batch pass: draw a segment on each photograph without leaving Detect    #
# --------------------------------------------------------------------------- #
def _canvas_click(col, x, y):
    from nicegui.events import GenericEventArguments, handle_event
    img = [e for e in _walk(col)
           if e.__class__.__name__ == "InteractiveImage"][-1]
    mouse = [l for l in img._event_listeners.values() if l.type == "mouse"][0]
    handle_event(mouse.handler, GenericEventArguments(
        sender=img, client=img.client,
        args={"mouse_event_type": "click", "image_x": x, "image_y": y}))


def _scale_status(col):
    return next(e for e in _walk(col)
                if "pm-det-scale-status" in (e._classes or []))


def test_the_batch_pass_scales_each_photograph_from_the_detect_tab(
        clean, project, monkeypatch):
    import nicegui
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    state, _ = clean
    _photo(project["images"] / "IMG_2.jpg", seed=5)
    Q.save_library(project["root"], [BOOT])
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    state.det_scale_canvas.image = ""
    state.det_scale_canvas.features = []
    state.det_scale_object = "Boot"
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    assert state.det_files == ["IMG_1.jpg", "IMG_2.jpg"]

    # Next without a scale opens the first photograph; two clicks commit a
    # segment, which is saved beside its Digitize CSV at once.
    _press(col, "Next without a scale")
    assert Path(state.det_scale_canvas.image).name == "IMG_1.jpg"
    _canvas_click(col, 10.0, 20.0)
    _canvas_click(col, 30.0, 20.0)          # 20 px = one Boot = 300 mm
    side = Q.scale_segments_path_for(
        project["validation"] / "IMG_1_truth.csv")
    doc = json.loads(side.read_text(encoding="utf-8"))
    assert doc["segments"] == [{"p0": [10.0, 20.0], "p1": [30.0, 20.0],
                                "object": "Boot"}]
    assert doc["objects"] == [{"name": "Boot", "length": 300.0, "unit": "mm"}]
    assert "15.000 mm/px" in _scale_status(col).text

    # The next photograph without one is the second; the first is done.
    _press(col, "Next without a scale")
    assert Path(state.det_scale_canvas.image).name == "IMG_2.jpg"
    assert state.det_scale_canvas.features == [], "its own canvas, still empty"
    _canvas_click(col, 0.0, 0.0)
    _canvas_click(col, 40.0, 0.0)
    assert Q.scale_segments_path_for(
        project["validation"] / "IMG_2_truth.csv").exists()

    # Both are queued with the scale each was given.
    _press(col, "Add to queue")
    res = {j["image_name"]: j["resolution"] for j in state.det_jobs}
    assert res["IMG_1.jpg"] == pytest.approx(0.015)
    assert res["IMG_2.jpg"] == pytest.approx(0.0075)

    # Nothing left to scale, and the file list offers the ruler button.
    _press(col, "Next without a scale")
    assert sum(1 for e in _walk(col) if e.__class__.__name__ == "Button"
               and e._props.get("icon") == "straighten") == 2,         "one ruler button per listed photograph"

    # Clear takes the segment off the canvas and out of the saved file.
    _press(col, "Clear segments")
    assert state.det_scale_canvas.features == []
    assert not Q.scale_segments_path_for(
        project["validation"] / "IMG_2_truth.csv").exists()
    assert len(json.loads(side.read_text(encoding="utf-8"))["segments"]) == 1,         "the other photograph keeps its own"


# --------------------------------------------------------------------------- #
#  The third mode: Ortho | Quadrat | Object scale                              #
# --------------------------------------------------------------------------- #
def test_object_scale_is_the_third_mode_and_runs_as_a_quadrat(clean, project):
    from gui.app import OBJECT_SCALE, det_run_mode, det_photo_mode
    state, _ = clean
    _photo(project["images"] / "IMG_2.jpg", seed=6)
    Q.write_scale_segments(project["validation"] / "IMG_1_truth.csv",
                           "IMG_1.jpg", SEGS, [BOOT])
    state.current_project = "P"
    state.det_mode = OBJECT_SCALE
    state.det_dir = ""
    col = _build(clean)
    # The toggle offers it, and the mode survived the build (an unknown
    # value would have been normalised away).
    toggle = next(e for e in _walk(col) if e.__class__.__name__ == "Toggle")
    assert [o["label"] for o in toggle._props["options"]] == \
        ["Ortho", "Quadrat", "Object scale"]
    assert state.det_mode == OBJECT_SCALE
    # It lists photographs and runs as a Quadrat everywhere downstream.
    assert det_photo_mode(OBJECT_SCALE) and det_run_mode(OBJECT_SCALE) == "quadrat"
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    assert state.det_files == ["IMG_1.jpg", "IMG_2.jpg"]
    texts = [str(getattr(e, "text", "")) for e in _walk(col)]
    assert any(t.startswith("Object scale 3.000 mm/px") for t in texts)
    assert any(t.startswith("No scale yet") for t in texts), \
        "the unscaled photograph says so"

    # Only the scaled photograph is queued; the other is left out by name.
    _press(col, "Add to queue")
    assert [j["image_name"] for j in state.det_jobs] == ["IMG_1.jpg"]
    job = state.det_jobs[0]
    assert job["mode"] == "quadrat", "the detector and the outputs see Quadrat"
    assert job["resolution"] == pytest.approx(0.003)
    assert job["gsd_source"] == "object: mm"


# --------------------------------------------------------------------------- #
#  The quadrat frame is left out, and the tab says so                          #
# --------------------------------------------------------------------------- #
def test_the_frame_switch_is_on_by_default_and_travels_with_the_job(clean, project):
    state, _ = clean
    # IMG_1 is a plain photograph; IMG_2 is rectified, with a frame record:
    # 1 cm of frame at 1 mm/px on a 100 x 80 px photograph.
    _photo(project["images"] / "IMG_2.jpg", w=100, h=80, seed=7)
    (project["images"] / "IMG_2.jpg.json").write_text(json.dumps(
        {"PM_GSD": "0.001000", "PM_FRAME_THICKNESS_M": "0.0100"}),
        encoding="utf-8")
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    state.det_exclude_frame = True
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    assert state.det_files == ["IMG_1.jpg", "IMG_2.jpg"]

    def _notes():
        return [e.text for e in _walk(col) if "pm-det-frame" in (e._classes or [])]
    box = next(e for e in _walk(col) if "pm-det-exclude-frame" in (e._classes or []))
    assert box.value is True and box.visible
    assert _notes() == ["frame 1.0 cm = 10 px + 2 px margin left out on each "
                        "edge; 0.08 x 0.06 m measured"], "only the framed photograph"
    _press(col, "Add to queue")
    assert [j["exclude_frame"] for j in state.det_jobs] == [True, True]

    # Off: the note goes with it, and the jobs carry the choice.
    state.det_jobs = []
    box.value = False
    assert state.det_exclude_frame is False
    assert _notes() == []
    _press(col, "Add to queue")
    assert [j["exclude_frame"] for j in state.det_jobs] == [False, False]

    # The switch belongs to the photograph modes, not to Ortho.
    from nicegui import binding
    state.det_mode = "ortho"
    binding._refresh_step()
    assert not box.visible
    state.det_mode = "quadrat"
    binding._refresh_step()
    assert box.visible


def test_run_all_queued_does_not_die_on_the_drawer_button(clean, project, monkeypatch):
    """The drawer's Reload model button is registered per page and may be
    absent (a test, a page without the drawer): Run all queued must still
    start, not raise NameError before the first job."""
    from gui import app as A
    import nicegui
    from detectors import maskrcnn as _mr
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    # The weights are not part of the repository (a CI runner has none).
    monkeypatch.setattr(_mr.MaskRCNNBackend, "is_available", lambda self: True)
    state, _ = clean
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    A._RELOAD_MODEL_BTN["btn"] = None
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    _press(col, "Add to queue")
    assert len(state.det_jobs) == 1
    started = {}
    import threading

    class _NoThread:
        def __init__(self, target=None, daemon=None, **kw):
            started["target"] = target
        def start(self):
            started["started"] = True
    monkeypatch.setattr(threading, "Thread", _NoThread)
    _press(col, "Run all queued")
    assert started.get("started"), "the run never reached its worker thread"


def test_a_length_given_later_in_the_library_reaches_the_sidecars_scale(tmp_path):
    """The segment was drawn while the object had no length; the length
    given afterwards in Digitize (the project library) is the current word
."""
    csv = tmp_path / "IMG_1_truth.csv"
    Q.write_scale_segments(csv, "IMG_1.jpg", [dict(SEGS[0], object="boot width")],
                           [Q.ScalingObject("boot width", None, None)])
    unknown, _ = Q.scale_from_sidecar([csv])
    assert not unknown.is_metric
    known, _ = Q.scale_from_sidecar([csv], [Q.ScalingObject("boot width", 110.0, "mm")])
    assert known.is_metric and abs(known.metres_per_px * 1000 - 110.0 / unknown.px_per_unit) < 1e-9
    # An object the library no longer has still comes from the file.
    other, _ = Q.scale_from_sidecar([csv], [Q.ScalingObject("card", 85.6, "mm")])
    assert not other.is_metric


def _named_fake(name):
    """A plug-in stand-in: two discs, the CSV where the wrapper expects it."""
    import numpy as np
    from detectors.base import BackendInfo, DetectorBackend, output_csv_path

    class Fake(DetectorBackend):
        info = BackendInfo(name=name, display_name=f"Fake {name}",
                           framework="classical", license="MIT")

        def is_available(self):
            return True

        def detect_jobs(self, mode, jobs, **kw):
            from PIL import Image
            from detectors import measure, subprocess_runner as SR
            out = []
            for job in jobs:
                csv = output_csv_path(mode, job, kw)
                csv.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(job["path"]) as im:
                    w, h = im.size
                yy, xx = np.mgrid[0:h, 0:w]
                lab = np.zeros((h, w), np.int32)
                lab[(xx - w // 3) ** 2 + (yy - h // 2) ** 2 <= 100] = 1
                lab[(xx - 2 * w // 3) ** 2 + (yy - h // 2) ** 2 <= 100] = 2
                npz = SR.instances_path_for(csv)
                np.savez(npz, labels=lab, scores=np.array([0.9, 0.9], np.float32),
                         shape=np.array([h, w]))
                df = measure.measure_instances(SR.read_instances_npz(npz),
                                               float(kw["resolution"]), height=h)
                df.to_csv(csv, index=False)
                out.append(df)
            return out
    return Fake()


def test_two_models_on_one_photograph_write_two_files(clean, project, monkeypatch):
    """A second model's run of a photograph used to write over the first
    model's CSV: both carried the photograph's name only."""
    import threading
    import nicegui
    from gui import app as A
    from detectors import registry
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    monkeypatch.setitem(registry._BACKENDS, "fakea", _named_fake("fakea"))
    monkeypatch.setitem(registry._BACKENDS, "fakeb", _named_fake("fakeb"))
    state, _ = clean
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    A._RELOAD_MODEL_BTN["btn"] = None
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    started = {}

    class _Inline:
        def __init__(self, target=None, daemon=None, **kw):
            started["target"] = target

        def start(self):
            started["target"]()
    monkeypatch.setattr(threading, "Thread", _Inline)
    written = []
    for model in ("fakea", "fakeb"):
        state.det_model = model
        state.det_jobs = []
        _press(col, "Add to queue")
        assert len(state.det_jobs) == 1
        _press(col, "Run all queued")
        written = sorted(p.name for p in project["root"].rglob("*_individual_clasts.csv"))
    assert len(written) == 2, written
    assert any("_model=fakea_" in n for n in written)
    assert any("_model=fakeb_" in n for n in written)


def test_run_all_queued_refuses_without_the_weights_and_says_how_to_get_them(
        clean, project, monkeypatch):
    """Without Mask R-CNN's weights (not public yet), Run all queued stops
    before the worker and shows the notice of detectors.download_weights."""
    from gui import app as A
    import nicegui
    import threading
    from detectors import maskrcnn as _mr
    from detectors.download_weights import weights_missing_message
    monkeypatch.setattr(nicegui.ui, "run_javascript", lambda *a, **k: None)
    monkeypatch.setattr(_mr.MaskRCNNBackend, "is_available", lambda self: False)
    notes = []
    monkeypatch.setattr(nicegui.ui, "notify", lambda msg, **kw: notes.append(msg))
    state, _ = clean
    state.current_project = "P"
    state.det_mode = "quadrat"
    state.det_dir = ""
    A._RELOAD_MODEL_BTN["btn"] = None
    col = _build(clean)
    inp = next(e for e in _walk(col) if e.__class__.__name__ == "Input"
               and e._props.get("label") == "Image directory")
    inp.value = str(project["images"])
    _press(col, "Add to queue")
    started = {}

    class _NoThread:
        def __init__(self, target=None, daemon=None, **kw):
            pass
        def start(self):
            started["started"] = True
    monkeypatch.setattr(threading, "Thread", _NoThread)
    _press(col, "Run all queued")
    assert not started, "the run started without the weights"
    assert weights_missing_message() in notes
