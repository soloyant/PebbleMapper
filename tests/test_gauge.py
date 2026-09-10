"""``functions.gauge``, the engine behind Digitize's Detect and its *No GSD?
Scale from an object* option.

Pure logic: the scale from segments or from a GSD, the scaling-object
library and the metric / custom rule, unit conversion, the run with an
injected detector (sidecar: scale source, disclaimer or none), both figures
rendered to PNG with or without the disclaimer footer, the HEIC helpers,
then the layout kind in both project layouts. The Digitize section itself
is tested in tests/test_digitize_scale.py.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from functions import gauge as Q

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def _body(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)]


# --------------------------------------------------------------------------- #
#  The scale from segments                                                     #
# --------------------------------------------------------------------------- #
def test_scale_from_one_segment_is_its_pixel_length_over_the_object_length():
    r = Q.scale_from_segments([((0, 0), (300, 0))], object_length=1.0)
    assert r["px_per_unit"] == pytest.approx(300.0)
    assert r["spread"] == 0.0 and r["n"] == 1
    r = Q.scale_from_segments([((10, 10), (10, 310))], object_length=0.3)
    assert r["px_per_unit"] == pytest.approx(1000.0)


def test_scale_from_several_segments_is_the_mean_and_the_spread_is_relative():
    r = Q.scale_from_segments([((0, 0), (300, 0)), ((0, 0), (0, 330))], 1.0)
    assert r["px_per_unit"] == pytest.approx(315.0)
    assert r["per_segment"] == pytest.approx([300.0, 330.0])
    assert r["spread"] == pytest.approx(30.0 / 315.0)
    assert r["n"] == 2


@pytest.mark.parametrize("segs, length, msg", [
    ([], 1.0, "no scaling segment"),
    ([((5, 5), (5, 5))], 1.0, "degenerate"),
    ([((0, 0), (0, 0.4))], 1.0, "degenerate"),
    ([((0, 0), (100, 0))], 0.0, "must be > 0"),
    ([((0, 0), (100, 0))], "x", "must be a number"),
])
def test_scale_rejects_nothing_degenerate_and_bad_lengths(segs, length, msg):
    with pytest.raises(ValueError, match=msg):
        Q.scale_from_segments(segs, length)


def test_gaugescale_from_segments_labels_custom_and_metric_units():
    custom = Q.GaugeScale.from_segments([((0, 0), (200, 0))],
                                        object_name="boot width")
    assert custom.unit == Q.CUSTOM and custom.unit_label == "boot width"
    assert not custom.is_metric and custom.metres_per_px is None
    assert custom.mode == Q.CUSTOM_MODE and custom.valid
    metric = Q.GaugeScale.from_segments([((0, 0), (200, 0))],
                                        object_name="ruler",
                                        object_length=100, unit="mm")
    assert metric.unit_label == "mm" and metric.is_metric
    assert metric.px_per_unit == pytest.approx(2.0)
    assert metric.metres_per_px == pytest.approx(0.0005)
    assert metric.mode == Q.METRIC_MODE
    blank = Q.GaugeScale(object_name="  ")
    assert blank.unit_label == "unit" and not blank.valid
    with pytest.raises(ValueError, match="unit must be"):
        Q.GaugeScale.from_segments([((0, 0), (10, 0))], unit="furlong")


def test_format_length_rounds_sensibly():
    assert Q.format_length(63.2, "mm") == "63.2 mm"
    assert Q.format_length(63.2, "mm", short=True) == "63 mm"
    assert Q.format_length(0.71, "boot width") == "0.71 boot width"
    assert Q.format_length(0.71, "boot width", short=True) == "0.7 boot width"
    assert Q.format_length(1.0, "boot width", short=True) == "1 boot width"
    assert Q.format_length(30.0, "cm") == "30 cm"
    assert Q.format_length(1234.5, "mm") == "1,234 mm"
    assert Q.format_length(float("nan"), "mm") == "—"


# --------------------------------------------------------------------------- #
#  The library                                                                 #
# --------------------------------------------------------------------------- #
def test_scaling_object_knows_its_metres():
    assert Q.ScalingObject("ruler", 30, "cm").metres == pytest.approx(0.30)
    assert Q.ScalingObject("foot", 1, "ft").metres == pytest.approx(0.3048)
    assert Q.ScalingObject("foot", None, None).metres is None
    assert Q.ScalingObject("foot", 0.27, None).metres is None
    assert Q.ScalingObject("foot", None, "m").metres is None
    assert Q.ScalingObject("foot", -1, "m").metres is None
    assert Q.ScalingObject("ruler", 30, "cm").is_metric
    assert Q.ScalingObject("ruler", 30, "cm").label() == "ruler (30 cm)"
    assert Q.ScalingObject("foot").label() == "foot (size unknown)"


def test_scaling_object_from_dict_is_forgiving():
    assert Q.ScalingObject.from_dict({"name": " foot ", "length": "0.27", "unit": "m"}) \
        == Q.ScalingObject("foot", 0.27, "m")
    assert Q.ScalingObject.from_dict({"name": "x", "length": "abc", "unit": "yard"}) \
        == Q.ScalingObject("x", None, None)
    assert Q.ScalingObject.from_dict({"length": 1}) is None
    assert Q.ScalingObject.from_dict("nope") is None


def test_library_absent_is_the_default_one_unknown_object(tmp_path):
    lib = Q.load_library(tmp_path)
    assert [o.to_dict() for o in lib] == [
        {"name": "scale bar", "length": None, "unit": None}]
    assert not Q.library_path(tmp_path).exists(), "loading must not write"


def test_library_round_trips_through_the_project_root(tmp_path):
    objs = [Q.ScalingObject("boot width", 0.27, "m"),
            Q.ScalingObject("hammer length", None, None),
            {"name": "ruler", "length": 30, "unit": "cm"}]
    p = Q.save_library(tmp_path, objs)
    assert p == tmp_path / "scaling_objects.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw == [
        {"name": "boot width", "length": 0.27, "unit": "m"},
        {"name": "hammer length", "length": None, "unit": None},
        {"name": "ruler", "length": 30.0, "unit": "cm"},
    ]
    back = Q.load_library(tmp_path)
    assert [o.name for o in back] == ["boot width", "hammer length", "ruler"]
    assert back[0].metres == pytest.approx(0.27)
    assert back[1].metres is None


def test_library_skips_malformed_entries_and_duplicate_names(tmp_path):
    Q.library_path(tmp_path).write_text(json.dumps([
        {"name": "a", "length": 1, "unit": "m"}, {"length": 2},
        {"name": "a", "length": 5, "unit": "cm"}, "junk"]), encoding="utf-8")
    lib = Q.load_library(tmp_path)
    assert [(o.name, o.length) for o in lib] == [("a", 1.0)]
    Q.library_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert [o.name for o in Q.load_library(tmp_path)] == ["scale bar"]
    Q.save_library(tmp_path, [])
    assert [o.name for o in Q.load_library(tmp_path)] == ["scale bar"]


# --------------------------------------------------------------------------- #
#  The scale rule                                                              #
# --------------------------------------------------------------------------- #
LIB = [Q.ScalingObject("boot width", None, None),
       Q.ScalingObject("hammer length", None, None),
       Q.ScalingObject("ruler", 30, "cm"),
       Q.ScalingObject("card", 85.6, "mm")]


def _seg(px, name, y=0.0):
    return Q.GaugeSegment((0.0, y), (float(px), y), name)


def test_two_known_objects_give_metric_mode_in_the_chosen_unit():
    segs = [_seg(300, "ruler"), _seg(85.6 * 1.1, "card", y=50)]
    opts = Q.result_unit_options(segs, LIB)
    assert opts["mode"] == "metric" and opts["default"] == "mm"
    assert opts["options"] == ["mm", "cm", "m", "in", "ft"]
    sc = Q.resolve_scale(segs, LIB, "mm")
    assert sc.mode == Q.METRIC_MODE and sc.unit == "mm" and sc.is_metric
    # ruler: 300 px / 0.30 m = 1000 px/m; card: 94.16 px / 0.0856 m = 1100 px/m
    assert sc.px_per_unit == pytest.approx(1050.0 * 0.001)
    assert sc.spread == pytest.approx(100.0 / 1050.0)
    assert sc.ignored == [] and sc.note == ""
    assert [s.object_name for s in sc.segments] == ["ruler", "card"]
    in_cm = Q.resolve_scale(segs, LIB, "cm")
    assert in_cm.px_per_unit == pytest.approx(10.5)
    default = Q.resolve_scale(segs, LIB)
    assert default.unit == "mm"


def test_one_unknown_object_gives_custom_mode_named_after_it():
    segs = [_seg(300, "boot width"), _seg(320, "boot width", y=10)]
    opts = Q.result_unit_options(segs, LIB)
    assert opts == {"mode": "custom", "options": ["boot width"],
                    "default": "boot width", "unknown": ["boot width"],
                    "missing": []}
    sc = Q.resolve_scale(segs, LIB)
    assert sc.mode == Q.CUSTOM_MODE and sc.unit == Q.CUSTOM
    assert sc.unit_label == "boot width" and not sc.is_metric
    assert sc.px_per_unit == pytest.approx(310.0)
    assert sc.spread == pytest.approx(20.0 / 310.0)
    assert sc.metres_per_px is None and sc.ignored == []


def test_two_unknown_objects_the_chosen_one_defines_the_unit_the_other_is_ignored():
    segs = [_seg(300, "boot width"), _seg(100, "hammer length", y=20),
            _seg(310, "boot width", y=40), _seg(200, "ruler", y=60)]
    opts = Q.result_unit_options(segs, LIB)
    assert opts["mode"] == "custom"
    assert opts["options"] == ["boot width", "hammer length"]
    sc = Q.resolve_scale(segs, LIB)                       # default: first drawn
    assert sc.unit_label == "boot width"
    assert sc.px_per_unit == pytest.approx(305.0)
    assert sc.ignored == ["hammer length", "ruler"]
    assert "hammer length" in sc.note and "ratio" in sc.note
    other = Q.resolve_scale(segs, LIB, "hammer length")
    assert other.unit_label == "hammer length"
    assert other.px_per_unit == pytest.approx(100.0)
    assert other.ignored == ["boot width", "ruler"]
    with pytest.raises(ValueError, match="not one of the unknown objects"):
        Q.resolve_scale(segs, LIB, "mm")


def test_the_scale_rule_names_what_is_missing():
    with pytest.raises(ValueError, match="no scaling segment"):
        Q.resolve_scale([], LIB)
    with pytest.raises(ValueError, match="degenerate"):
        Q.resolve_scale([_seg(0.2, "ruler")], LIB)
    with pytest.raises(ValueError, match="no scaling object"):
        Q.resolve_scale([_seg(100, "")], LIB)
    with pytest.raises(ValueError, match="not in the scaling-object library"):
        Q.resolve_scale([_seg(100, "banana")], LIB)
    assert Q.result_unit_options([_seg(100, "banana")], LIB)["missing"] == ["banana"]


def test_before_any_segment_the_default_unit_follows_the_library():
    # A library with a known length somewhere: metric, mm.
    opts = Q.result_unit_options([], LIB)
    assert opts["mode"] == "metric" and opts["default"] == "mm"
    # Every object unknown: the first object's name.
    unknown_only = [Q.ScalingObject("scale bar"), Q.ScalingObject("boot width")]
    opts = Q.result_unit_options([], unknown_only)
    assert opts["mode"] == "custom"
    assert opts["options"] == ["scale bar", "boot width"]
    assert opts["default"] == "scale bar"
    assert Q.result_unit_options([], [])["default"] == ""


def test_the_detectors_centimetre_summary_is_recognised():
    assert Q.is_detector_summary_line("  Clast Width D90 = 7786.25 cm")
    assert Q.is_detector_summary_line("INFO   Clast Length D10 = 12.00 cm")
    assert Q.is_detector_summary_line("  Equivalent Diameter D50 = 1,234.50 cm")
    assert not Q.is_detector_summary_line("Detected clasts: 153")
    assert not Q.is_detector_summary_line("[gauge] 19 clast(s); D50 = 48.2 mm, D84 = 72.6 mm")


def test_detect_scale_resamples_the_photograph_and_reports_original_pixels(tmp_path, photo):
    from PIL import Image
    seen = {}

    def fake_detect(**kw):
        p = kw["jobs"][0]["path"]
        with Image.open(p) as im:
            seen["size"] = im.size
        seen["path"] = p
        seen["resolution"] = kw["resolution"]
        # What the real detector returns at resolution r: lengths in
        # detection pixels × r (= original pixels), centroids in
        # detection pixels.
        df = _pixel_frame(n=4)
        df["x"] = [100.0, 200.0, 300.0, 400.0]
        df["y"] = [50.0, 60.0, 70.0, 80.0]
        return [df]

    scale = Q.resolve_scale([_seg(300, "ruler")], LIB, "mm")      # 1 px = 1 mm
    out = tmp_path / "o"
    res = Q.run_gauge(photo, scale, out, detect_fn=fake_detect, detect_scale=0.5)
    assert seen["size"] == (450, 300), "the 900×600 photo is resampled by 1/2"
    assert seen["resolution"] == pytest.approx(2.0)
    assert seen["path"] == str(out / "IMG_0001_gauge_detect.jpg")
    assert not Path(seen["path"]).exists(), "the resampled file is removed afterwards"
    df = res["df"]
    np.testing.assert_allclose(df["x"], [200.0, 400.0, 600.0, 800.0])
    np.testing.assert_allclose(df["y"], [100.0, 120.0, 140.0, 160.0])
    np.testing.assert_allclose(df["Clast_length"], _pixel_frame(n=4)["Clast_length"])
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["model"]["detect_scale"] == 0.5
    assert side["model"]["detection_size_px"] == [450, 300]
    # Factor 1: nothing resampled, centroids untouched.
    res1 = Q.run_gauge(photo, scale, out, detect_fn=fake_detect, detect_scale=1.0)
    assert seen["size"] == (900, 600) and seen["path"] == str(photo)
    np.testing.assert_allclose(res1["df"]["x"], [100.0, 200.0, 300.0, 400.0])
    with pytest.raises(ValueError, match="detect_scale"):
        Q.run_gauge(photo, scale, out, detect_fn=fake_detect, detect_scale=0)
    assert Q.DETECT_SCALES == {"1": 1.0, "1/2": 0.5, "1/3": pytest.approx(1 / 3), "1/4": 0.25}


# --------------------------------------------------------------------------- #
#  The run with an injected detector                                           #
# --------------------------------------------------------------------------- #
def _pixel_frame(n=12, seed=1):
    rng = np.random.default_rng(seed)
    L = rng.uniform(40, 160, n)
    Wd = L * rng.uniform(0.5, 0.9, n)
    return pd.DataFrame({
        "clast_ID": np.arange(1, n + 1),
        "x": rng.uniform(50, 850, n), "y": rng.uniform(50, 550, n),
        "Clast_length": L, "Clast_width": Wd,
        "Ellipse_major_axis": L * 1.02, "Ellipse_minor_axis": Wd * 0.98,
        "Surface_area": math.pi * L * Wd / 4, "Perimeter": 3.0 * L,
        "Equivalent_diameter": np.sqrt(L * Wd),
        "Eccentricity": 0.6, "Solidity": 0.97, "Mean_intensity": 120.0,
        "Score": 0.9, "Orientation": rng.uniform(0, 180, n),
    })


def _photo(path, w=900, h=600, seed=2):
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def photo(tmp_path):
    return _photo(tmp_path / "IMG_0001.jpg")


def test_run_gauge_converts_lengths_areas_keeps_pixels_and_writes_csv_and_sidecar(
        tmp_path, photo):
    df_px = _pixel_frame()
    calls = []

    def fake_detect(**kw):
        calls.append(kw)
        return [df_px.copy()]

    segs = [_seg(300, "boot width", y=500)]
    scale = Q.resolve_scale(segs, LIB)
    out = tmp_path / "out"
    lines = []
    # The conversion itself: the segment-crossing rule (tested below) is
    # off, or the random clasts near the segment would be dropped.
    res = Q.run_gauge(photo, scale, out, detect_fn=fake_detect,
                      log_fn=lines.append, min_confidence=0.6,
                      devicemode="cpu", devicenumber=0,
                      exclude_segments=False)
    kw = calls[0]
    assert kw["mode"] == "quadrat" and kw["resolution"] == 1.0
    assert kw["saveresults"] is False and kw["plot"] is False
    assert kw["jobs"] == [{"path": str(photo)}]
    assert kw["min_confidence"] == 0.6 and kw["devicemode"] == "cpu"

    df = res["df"]
    ppu = 300.0
    for c in Q.LENGTH_COLUMNS:
        np.testing.assert_allclose(df[c], df_px[c] / ppu)
    np.testing.assert_allclose(df["Surface_area"], df_px["Surface_area"] / ppu ** 2)
    np.testing.assert_allclose(df["x"], df_px["x"])
    np.testing.assert_allclose(df["y"], df_px["y"])
    assert (df["unit"] == "boot width").all()
    assert set(df.columns) == set(df_px.columns) | {"unit"}

    csv = Path(res["csv"])
    assert csv == out / "IMG_0001_gauge.csv" and csv.exists()
    back = pd.read_csv(csv)
    assert len(back) == len(df_px)
    np.testing.assert_allclose(back["Clast_length"], df["Clast_length"], rtol=1e-4)
    assert back["unit"].iloc[0] == "boot width"

    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert Path(res["json"]) == out / "IMG_0001_gauge.json"
    assert side["scale"]["mode"] == "custom"
    assert side["scale"]["unit_label"] == "boot width"
    assert side["scale"]["px_per_unit"] == pytest.approx(300.0)
    assert side["scale"]["segments"] == [
        {"p0": [0.0, 500.0], "p1": [300.0, 500.0], "object": "boot width",
         "px_length": 300.0}]
    assert side["image"] == str(photo.resolve()) and side["detection_image"] == str(photo.resolve())
    assert side["model"] == {"backend": "maskrcnn", "min_confidence": 0.6,
                             "devicemode": "cpu", "devicenumber": 0,
                             "detect_scale": 1.0, "detection_size_px": None}
    assert side["disclaimer"] == Q.DISCLAIMER
    assert side["n"] == 12 and side["stats"]["sorting_phi"] is None
    assert "timestamp" in side and side["columns"]["pixels"] == ["x", "y"]

    st = res["stats"]
    assert st["n"] == 12
    assert st["D50"] == pytest.approx(float(np.quantile(df_px["Clast_length"] / ppu, 0.5)))
    assert math.isnan(st["sorting_phi"])
    assert any("300.00 px per boot width" in l for l in lines)


def test_run_gauge_in_metric_mode_records_source_and_folk_ward_sorting(tmp_path, photo):
    scale = Q.resolve_scale([_seg(300, "ruler")], LIB, "mm")      # 1 px = 1 mm
    res = Q.run_gauge(photo, scale, tmp_path / "o",
                      detect_fn=lambda **kw: [_pixel_frame()],
                      out_stem="Site_20260911__IMG_0001",
                      source_image=tmp_path / "IMG_0001.heic")
    assert Path(res["csv"]).name == "Site_20260911__IMG_0001_gauge.csv"
    df = res["df"]
    np.testing.assert_allclose(df["Clast_length"], _pixel_frame()["Clast_length"])
    assert (df["unit"] == "mm").all()
    assert math.isfinite(res["stats"]["sorting_phi"])
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["image"].endswith("IMG_0001.heic")
    assert side["detection_image"] == str(photo.resolve())
    assert side["scale"]["metres_per_px"] == pytest.approx(0.001)


def test_run_gauge_surfaces_the_detectors_error_and_a_stop(tmp_path, photo):
    scale = Q.resolve_scale([_seg(300, "ruler")], LIB)

    def failing(**kw):
        kw["progress_callback"](0, "error", {"message": "GPU ran out of memory"})
        return [pd.DataFrame()]
    with pytest.raises(RuntimeError, match="GPU ran out of memory"):
        Q.run_gauge(photo, scale, tmp_path / "o", detect_fn=failing)

    def stopped(**kw):
        kw["progress_callback"](0, "stopped", {})
        return [pd.DataFrame()]
    with pytest.raises(RuntimeError, match="stopped"):
        Q.run_gauge(photo, scale, tmp_path / "o", detect_fn=stopped)
    with pytest.raises(FileNotFoundError):
        Q.run_gauge(tmp_path / "nope.jpg", scale, tmp_path, detect_fn=failing)
    with pytest.raises(ValueError, match="no scale"):
        Q.run_gauge(photo, Q.GaugeScale(), tmp_path, detect_fn=failing)


def test_run_gauge_with_no_detection_writes_an_empty_table(tmp_path, photo):
    scale = Q.resolve_scale([_seg(300, "ruler")], LIB)
    res = Q.run_gauge(photo, scale, tmp_path / "o",
                      detect_fn=lambda **kw: [pd.DataFrame(columns=list(_pixel_frame().columns))])
    assert res["stats"]["n"] == 0 and Path(res["csv"]).exists()
    assert math.isnan(res["stats"]["D50"])


# --------------------------------------------------------------------------- #
#  The figures                                                                 #
# --------------------------------------------------------------------------- #
def test_overlay_figure_builds_with_axes_segments_labels_and_scale_bar(photo):
    from matplotlib.figure import Figure
    from functions.clast_geometry import NO_CONTOURS_NOTE
    scale = Q.resolve_scale([_seg(300, "ruler", y=550), _seg(94, "card", y=30)],
                            LIB, "mm")
    df = Q.convert_pixels_to_units(_pixel_frame(), scale)
    fig = Q.gauge_overlay_figure(photo, df, scale, n_labels=5, library=LIB)
    assert isinstance(fig, Figure)
    m = fig.pm_meta
    assert m["n"] == 12 and m["unit"] == "mm" and m["n_segments"] == 2
    # No ellipses: every clast's major and minor axes, and (without a
    # contour file) the note saying the outlines are missing.
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Ellipse
    assert not any(isinstance(p, Ellipse) for p in fig.axes[0].patches)
    assert not any(isinstance(c, PatchCollection) for c in fig.axes[0].collections)
    assert m["n_chords"] == 12 and m["n_outlines"] == 0
    assert NO_CONTOURS_NOTE in [t.get_text() for t in fig.axes[0].texts]
    assert len(m["labelled"]) == 5
    assert m["scale_bar_m"] is not None and m["scale_bar_m"] > 0
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("n = 12") and "D50 = " in t and "mm" in t for t in texts)
    assert "ruler: 30 cm" in texts and "card: 85.6 mm" in texts
    assert sum(1 for t in texts if re.fullmatch(r"[\d,.]+ mm", t)) == 5
    assert not ax.get_xticks().size and not ax.get_yticks().size
    leg = fig.legends[0] if fig.legends else None
    assert leg is not None, "the size-class legend is missing"
    labels = [t.get_text() for t in leg.get_texts()]
    assert len(labels) == 5 and all("mm" in t for t in labels)


def test_overlay_figure_in_custom_mode_has_no_scale_bar_and_names_the_unit(photo):
    segs = [_seg(300, "boot width", y=550), _seg(100, "hammer length", y=30)]
    scale = Q.resolve_scale(segs, LIB)
    df = Q.convert_pixels_to_units(_pixel_frame(), scale)
    fig = Q.gauge_overlay_figure(photo, df, scale, n_labels=3, library=LIB)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "1 boot width" in texts
    assert "hammer length (not used)" in texts
    assert fig.pm_meta["scale_bar_m"] is None
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert len(labels) == 5 and all("boot width" in t for t in labels)
    # Three annotated clasts ("0.3 boot width"), the box excluded.
    assert sum(1 for t in texts
               if re.fullmatch(r"[\d.]+ boot width", t) and t != "1 boot width") == 3


def test_overlay_figure_scale_label_is_rotated_along_its_segment(photo):
    # A segment going down-right in image pixels (y down) reads at -45 on
    # screen; drawn right to left it is flipped to the same -45, never
    # upside down. The horizontal one stays at 0.
    segs = [Q.GaugeSegment((100.0, 100.0), (400.0, 400.0), "boot width"),
            Q.GaugeSegment((700.0, 350.0), (500.0, 150.0), "hammer length"),
            Q.GaugeSegment((100.0, 550.0), (400.0, 550.0), "boot width")]
    scale = Q.resolve_scale(segs, LIB)
    df = Q.convert_pixels_to_units(_pixel_frame(), scale)
    fig = Q.gauge_overlay_figure(photo, df, scale, n_labels=3, library=LIB)
    by_text = {}
    for t in fig.axes[0].texts:
        by_text.setdefault(t.get_text(), []).append(t)
    foot = by_text["1 boot width"]
    # matplotlib reports rotations within [0, 360): -45 comes back as 315.
    assert [round(t.get_rotation(), 6) for t in foot] == [315.0, 0.0]
    assert foot[0].get_rotation_mode() == "anchor"
    assert foot[0].get_ha() == "center" and foot[0].get_va() == "bottom"
    # Anchored a little off the midpoint on the text's "up" side (towards
    # smaller image y), never on the line.
    x, y = foot[0].get_position()
    assert x > 250.0 and y < 250.0 and math.isclose(x - 250.0, -(y - 250.0), abs_tol=1e-6)
    x, y = foot[1].get_position()
    assert x == pytest.approx(250.0) and 540.0 < y < 550.0
    (rev,) = by_text["hammer length (not used)"]
    assert rev.get_rotation() == pytest.approx(315.0)
    # The per-clast labels are left unrotated.
    assert all(t.get_rotation() == 0.0 for t in fig.axes[0].texts
               if re.fullmatch(r"[\d.]+ boot width", t.get_text())
               and t.get_text() != "1 boot width")


def test_distribution_figure_has_two_panels_in_the_unit():
    scale = Q.resolve_scale([_seg(300, "boot width")], LIB)
    df = Q.convert_pixels_to_units(_pixel_frame(n=40), scale)
    fig = Q.gauge_distribution_figure(df, scale)
    assert len(fig.axes) == 2, "no φ twin axis for a custom unit"
    for ax in fig.axes:
        assert ax.get_xlabel() == "Clast length (boot width)"
        assert ax.get_xscale() == "log"
    m = fig.pm_meta
    assert m["n"] == 40 and m["unit"] == "boot width"
    assert m["D16"] < m["D50"] < m["D84"]
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert any(l.startswith("D16 = ") for l in labels)
    assert any(l.startswith("D50 = ") for l in labels)
    assert any(l.startswith("D84 = ") for l in labels)
    with pytest.raises(ValueError, match="< 2"):
        Q.gauge_distribution_figure(df.iloc[:1], scale)


def test_write_gauge_figures_saves_both_pngs_at_200_dpi(tmp_path, photo):
    from PIL import Image
    scale = Q.resolve_scale([_seg(300, "ruler")], LIB, "cm")
    df = Q.convert_pixels_to_units(_pixel_frame(), scale)
    out = tmp_path / "gauge"
    paths = Q.write_gauge_figures(photo, df, scale, out, "IMG_0001", library=LIB)
    ov, di = Path(paths["overlay"]), Path(paths["distribution"])
    assert ov == out / "IMG_0001_gauge_overlay.png" and ov.exists()
    assert di == out / "IMG_0001_gauge_distribution.png" and di.exists()
    with Image.open(ov) as im:
        assert im.info.get("dpi", (200,))[0] == pytest.approx(200, abs=1)
        assert im.size[0] > 1500
    # One clast: the overlay still draws, the distribution is skipped.
    paths1 = Q.write_gauge_figures(photo, df.iloc[:1], scale, out, "one", library=LIB)
    assert Path(paths1["overlay"]).exists() and paths1["distribution"] is None


# --------------------------------------------------------------------------- #
#  Photographs: HEIC and orientation                                           #
# --------------------------------------------------------------------------- #
def test_open_photo_names_pillow_heif_when_it_is_missing(tmp_path, monkeypatch):
    fake = tmp_path / "IMG_9.heic"
    fake.write_bytes(b"\x00" * 16)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)   # import -> ImportError
    with pytest.raises(RuntimeError, match=r"pillow-heif.*IMG_9\.heic|IMG_9\.heic.*pillow-heif"):
        Q.open_photo(fake)
    assert Q.register_heif_opener() is False


def test_open_photo_honours_the_exif_orientation_and_the_working_copy_is_upright(tmp_path):
    from PIL import Image
    p = tmp_path / "IMG_6.jpg"
    im = Image.fromarray(np.zeros((40, 100, 3), dtype=np.uint8))
    exif = im.getexif()
    exif[0x0112] = 6                    # rotate 90° clockwise to display
    im.save(p, exif=exif.tobytes())
    with Q.open_photo(p) as up:
        assert up.size == (40, 100)
    assert Q.needs_working_copy(p)
    copy = Q.working_copy(p, tmp_path / "gauge")
    assert Path(copy) == tmp_path / "gauge" / "IMG_6.jpg"
    with Image.open(copy) as c:
        assert c.size == (40, 100) and c.mode == "RGB"
        assert c.getexif().get(0x0112, 1) == 1
    assert Q.prepare_photo(p, tmp_path / "gauge") == copy
    # An upright JPEG needs no copy and is used as it is.
    upright = _photo(tmp_path / "IMG_1.jpg", w=50, h=30)
    assert not Q.needs_working_copy(upright)
    assert Q.prepare_photo(upright, tmp_path / "gauge") == str(upright)
    assert not (tmp_path / "gauge" / "IMG_1.jpg").exists()


def test_a_heic_opens_and_needs_no_jpeg_copy_when_pillow_heif_is_installed(tmp_path):
    """pillow-heif hands Pillow the upright pixels (tag 1), the canvas
    transcodes a HEIC like a TIFF and the detector reads it through
    functions.images.read_rgb: the tab draws on and detects on the HEIC
    itself. Only an EXIF-rotated JPEG still gets the upright copy."""
    pytest.importorskip("pillow_heif")
    from PIL import Image
    from functions import images as I
    assert Q.register_heif_opener()
    p = tmp_path / "IMG_4228.heic"
    Image.fromarray(np.full((30, 50, 3), 90, dtype=np.uint8)).save(p, format="HEIF")
    with Q.open_photo(p) as im:
        assert im.size == (50, 30)
    assert not Q.needs_working_copy(p)
    assert Q.prepare_photo(p, tmp_path / "gauge") == str(p)
    assert not (tmp_path / "gauge").exists()
    assert I.read_rgb(p).shape == (30, 50, 3)


def test_the_folder_listing_includes_heic(tmp_path):
    from gui.app import list_images
    for n in ("b.HEIC", "a.jpg", "c.heif", "d.txt", "e.png"):
        (tmp_path / n).write_bytes(b"")
    assert list_images(str(tmp_path), list(Q.PHOTO_EXTENSIONS)) == \
        ["a.jpg", "b.HEIC", "c.heif", "e.png"]
    assert ".heic" in Q.PHOTO_EXTENSIONS and ".heif" in Q.PHOTO_EXTENSIONS


# --------------------------------------------------------------------------- #
#  The layout kind                                                             #
# --------------------------------------------------------------------------- #
def test_gauge_is_a_layout_kind_in_both_layouts(tmp_path):
    from functions import layout
    assert "gauge" in layout.PATH_KINDS
    assert layout._KIND_SUBPATHS["gauge"] == "output_results/gauge"
    assert layout._LEGACY_KIND_SUBPATHS["gauge"] == "results/gauge"
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "New" / "output_results").mkdir(parents=True)
        assert layout.project_path("New", "gauge") == \
            tmp_path / "New" / "output_results" / "gauge"
        (tmp_path / "Old" / "results").mkdir(parents=True)
        assert layout.project_path("Old", "gauge") == \
            tmp_path / "Old" / "results" / "gauge"
        assert layout.project_path("Fresh", "gauge") == \
            tmp_path / "Fresh" / "output_results" / "gauge"
    finally:
        layout.set_datasets_root(original, persist=False)


def test_the_layout_page_lists_the_kind():
    manual = (Path(__file__).resolve().parents[1] / "docs" / "user-manual.md") \
        .read_text(encoding="utf-8")
    page = manual.split("## Project folders", 1)[1].split("\n## ", 1)[0]
    assert "| `gauge` | `output_results/gauge` | `results/gauge` |" in page
    assert "21 kinds" in page and "20 kinds" not in page


# --------------------------------------------------------------------------- #
#  The GSD scale and the disclaimer                                            #
# --------------------------------------------------------------------------- #
def test_from_gsd_is_metric_with_no_segments():
    s = Q.GaugeScale.from_gsd(0.002)                  # 2 mm per pixel
    assert s.is_metric and s.unit == "mm" and s.mode == "metric"
    assert s.px_per_unit == pytest.approx(0.5)
    assert s.metres_per_px == pytest.approx(0.002)
    assert s.segments == [] and s.segments_px == [] and s.spread == 0.0
    assert s.valid and s.unit_label == "mm"
    assert Q.GaugeScale.from_gsd(0.002, "cm").px_per_unit == pytest.approx(5.0)
    for bad in (0, -1, "x", float("nan")):
        with pytest.raises(ValueError):
            Q.GaugeScale.from_gsd(bad)
    with pytest.raises(ValueError, match="unit"):
        Q.GaugeScale.from_gsd(0.002, "furlong")


def test_run_gauge_records_the_scale_source_and_the_disclaimer_or_none(tmp_path, photo):
    res = Q.run_gauge(photo, Q.GaugeScale.from_gsd(0.001), tmp_path / "g",
                      detect_fn=lambda **kw: [_pixel_frame()],
                      scale_source="gsd:filename", disclaimer=None)
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["scale_source"] == "gsd:filename"
    assert side["disclaimer"] is None
    assert side["scale"]["segments"] == [] and side["scale"]["unit"] == "mm"
    assert side["tool"] == "PebbleMapper Digitize"
    assert (res["df"]["unit"] == "mm").all()
    np.testing.assert_allclose(res["df"]["Clast_length"],
                               _pixel_frame()["Clast_length"])
    # The defaults follow the scale: an object scale keeps the disclaimer.
    res2 = Q.run_gauge(photo, Q.resolve_scale([_seg(300, "ruler")], LIB),
                       tmp_path / "o", detect_fn=lambda **kw: [_pixel_frame()])
    side2 = json.loads(Path(res2["json"]).read_text(encoding="utf-8"))
    assert side2["scale_source"] == "object"
    assert side2["disclaimer"] == Q.DISCLAIMER
    res3 = Q.run_gauge(photo, Q.GaugeScale.from_gsd(0.001), tmp_path / "h",
                       detect_fn=lambda **kw: [_pixel_frame()])
    side3 = json.loads(Path(res3["json"]).read_text(encoding="utf-8"))
    assert side3["scale_source"] == "gsd"


def test_the_figures_carry_the_disclaimer_only_when_asked(photo):
    scale = Q.resolve_scale([_seg(300, "ruler")], LIB, "mm")
    df = Q.convert_pixels_to_units(_pixel_frame(n=40), scale)
    fig = Q.gauge_overlay_figure(photo, df, scale, library=LIB)
    assert fig.axes[0].get_xlabel().replace("\n", " ") == Q.DISCLAIMER
    fig = Q.gauge_overlay_figure(photo, df, scale, library=LIB, disclaimer=None)
    assert fig.axes[0].get_xlabel() == ""
    fig2 = Q.gauge_distribution_figure(df, scale)
    assert fig2.get_supxlabel().replace("\n", " ") == Q.DISCLAIMER
    fig2 = Q.gauge_distribution_figure(df, scale, disclaimer=None)
    assert fig2.get_supxlabel() == ""


def test_write_gauge_figures_passes_the_disclaimer_through(tmp_path, photo, monkeypatch):
    seen = {}
    real_overlay, real_dist = Q.gauge_overlay_figure, Q.gauge_distribution_figure

    def overlay(*a, **kw):
        seen["overlay"] = kw.get("disclaimer", "unset")
        return real_overlay(*a, **kw)

    def dist(*a, **kw):
        seen["distribution"] = kw.get("disclaimer", "unset")
        return real_dist(*a, **kw)
    monkeypatch.setattr(Q, "gauge_overlay_figure", overlay)
    monkeypatch.setattr(Q, "gauge_distribution_figure", dist)
    scale = Q.GaugeScale.from_gsd(0.001)
    df = Q.convert_pixels_to_units(_pixel_frame(), scale)
    Q.write_gauge_figures(photo, df, scale, tmp_path / "f", "s", disclaimer=None)
    assert seen == {"overlay": None, "distribution": None}
    Q.write_gauge_figures(photo, df, scale, tmp_path / "f", "s")
    assert seen == {"overlay": Q.DISCLAIMER, "distribution": Q.DISCLAIMER}


# --------------------------------------------------------------------------- #
#  Detections crossing a scale segment are removed                             #
# --------------------------------------------------------------------------- #
PHOTO_H = 600          # the ``photo`` fixture is 900 x 600


def _disc_frame(discs, height=PHOTO_H, with_contours=True):
    """Disc clasts ``[(clast_ID, col, row, radius), ...]`` as a pixel-unit
    table in the gauge CSV frame (y = height - row) with their outlines."""
    rows, outlines = [], {}
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for cid, col, row, r in discs:
        d = 2.0 * r
        rows.append({"clast_ID": cid, "x": float(col), "y": float(height - row),
                     "Clast_length": d, "Clast_width": d,
                     "Ellipse_major_axis": d, "Ellipse_minor_axis": d,
                     "Surface_area": math.pi * r * r, "Perimeter": 2 * math.pi * r,
                     "Equivalent_diameter": d, "Eccentricity": 0.0,
                     "Solidity": 1.0, "Mean_intensity": 120.0, "Score": 0.9,
                     "Orientation": 0.0})
        outlines[cid] = [[float(col + r * math.cos(a)),
                          float(height - (row + r * math.sin(a)))] for a in t]
    df = pd.DataFrame(rows)
    if with_contours:
        from functions import clast_geometry as CG
        CG.attach_contours(df, outlines, frame="pixels")
    return df


# A horizontal segment along row 100 from column 100 to 420: disc 1 sits on
# it, disc 2's left edge (column 418) covers its end, disc 3 is far away.
_THREE = [(1, 250, 100, 30), (2, 450, 100, 32), (3, 700, 450, 30)]
_SEGMENT = Q.GaugeSegment((100.0, 100.0), (420.0, 100.0), "boot width")


def test_the_crossed_and_the_end_touched_clasts_go_the_far_one_stays():
    from functions import clast_geometry as CG
    df = _disc_frame(_THREE)
    kept, removed = Q.drop_crossing_segments(
        df, CG.contours_of(df), [_SEGMENT], image_height=PHOTO_H)
    assert removed == [1, 2]
    assert list(kept["clast_ID"]) == [3] and list(kept.index) == [0]
    assert sorted(CG.contours_of(kept)) == [3]
    # The same rule without outlines, on the fitted ellipses.
    bare = _disc_frame(_THREE, with_contours=False)
    kept2, removed2 = Q.drop_crossing_segments(bare, None, [_SEGMENT],
                                               image_height=PHOTO_H)
    assert removed2 == [1, 2] and list(kept2["clast_ID"]) == [3]
    # Segments as dicts or point pairs work the same; a proposal ring in
    # image rows is tested with the same rule.
    _, r3 = Q.drop_crossing_segments(df, CG.contours_of(df),
                                     [{"p0": [100, 100], "p1": [420, 100]}],
                                     image_height=PHOTO_H)
    assert r3 == [1, 2]
    ring_rows = [[250 + 30 * math.cos(a), 100 + 30 * math.sin(a)]
                 for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    assert Q.ring_crosses_segments(ring_rows, [((100, 100), (420, 100))])
    assert not Q.ring_crosses_segments(ring_rows, [((100, 500), (420, 500))])


def test_the_segment_is_read_in_image_rows_not_in_the_csv_frame():
    """A segment near the TOP of the photograph removes the clast near the
    top, not the one at its vertical mirror (CSV y = height - row)."""
    from functions import clast_geometry as CG
    df = _disc_frame([(1, 300, 60, 25), (2, 300, 540, 25)])
    top = Q.GaugeSegment((250.0, 60.0), (350.0, 60.0), "ruler")
    kept, removed = Q.drop_crossing_segments(df, CG.contours_of(df), [top],
                                             image_height=PHOTO_H)
    assert removed == [1] and list(kept["clast_ID"]) == [2]
    bare = _disc_frame([(1, 300, 60, 25), (2, 300, 540, 25)], with_contours=False)
    _, removed2 = Q.drop_crossing_segments(bare, None, [top], image_height=PHOTO_H)
    assert removed2 == [1]


def test_no_segment_removes_nothing(tmp_path, photo):
    from functions import clast_geometry as CG
    df = _disc_frame(_THREE)
    kept, removed = Q.drop_crossing_segments(df, CG.contours_of(df), [],
                                             image_height=PHOTO_H)
    assert removed == [] and len(kept) == 3
    res = Q.run_gauge(photo, Q.GaugeScale.from_gsd(0.001), tmp_path / "g",
                      detect_fn=lambda **kw: [_disc_frame(_THREE)])
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert res["excluded_by_segments"] == 0 and side["excluded_by_segments"] == 0
    assert len(pd.read_csv(res["csv"])) == 3


def test_run_gauge_removes_them_from_every_output_and_records_the_count(tmp_path, photo):
    from functions import clast_geometry as CG
    scale = Q.resolve_scale([_SEGMENT], [Q.ScalingObject("boot width")])
    lines = []
    res = Q.run_gauge(photo, scale, tmp_path / "o", log_fn=lines.append,
                      detect_fn=lambda **kw: [_disc_frame(_THREE)])
    assert res["excluded_by_segments"] == 2 and res["removed_ids"] == [1, 2]
    assert list(res["df"]["clast_ID"]) == [3] and res["stats"]["n"] == 1
    back = pd.read_csv(res["csv"])
    assert list(back["clast_ID"]) == [3]
    assert sorted(CG.read_contours(res["csv"])) == [3]
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["excluded_by_segments"] == 2 and side["n"] == 1
    assert side["excluded_clast_IDs"] == [1, 2]
    assert side["exclusion_rule"] == Q.SEGMENT_EXCLUSION_RULE
    assert side["exclusion_segments"] == [[[100.0, 100.0], [420.0, 100.0]]]
    assert any("2 detection(s) crossing a scale segment removed" in l for l in lines)
    # A GSD-scaled run on a photograph that still has segments: explicit.
    res2 = Q.run_gauge(photo, Q.GaugeScale.from_gsd(0.001), tmp_path / "p",
                       detect_fn=lambda **kw: [_disc_frame(_THREE)],
                       segments=[_SEGMENT])
    assert res2["removed_ids"] == [1, 2]
    # The rule can be turned off.
    res3 = Q.run_gauge(photo, scale, tmp_path / "q", exclude_segments=False,
                       detect_fn=lambda **kw: [_disc_frame(_THREE)])
    assert res3["excluded_by_segments"] == 0 and len(res3["df"]) == 3
    # Detection on a half-size copy: the frame comes back in original
    # pixels before the rule is applied.
    def half(**kw):
        df = _disc_frame([(c, x / 2, r_ / 2, rad / 2) for c, x, r_, rad in _THREE],
                         height=PHOTO_H // 2)
        return [df]
    res4 = Q.run_gauge(photo, scale, tmp_path / "h", detect_fn=half,
                       detect_scale=0.5)
    assert res4["removed_ids"] == [1, 2]


# --------------------------------------------------------------------------- #
#  The quadrat frame: what sits on the band is not a clast of the bed         #
# --------------------------------------------------------------------------- #
def _framed_photo(tmp_path, thickness=0.02):
    """A 900 x 600 rectified photograph at 1 mm/px whose record says the
    frame is ``thickness`` m thick: 20 px along each edge by default."""
    p = _photo(tmp_path / "IMG_0955_rectified_GSD=0.001m.jpg")
    (tmp_path / (p.name + ".json")).write_text(json.dumps(
        {"PM_RECTIFIED": "yes", "PM_GSD": "0.001000", "PM_SIZE": "900x600",
         "PM_FRAME_THICKNESS_M": f"{thickness:.4f}"}), encoding="utf-8")
    return p


# Disc 1 sits on the left band (column 10), disc 2 on the bottom band
# (row 590), disc 3 well inside.
_FRAMED = [(1, 10, 300, 8), (2, 450, 590, 8), (3, 450, 300, 30)]


def test_the_frame_band_is_left_out_and_the_rule_can_be_turned_off(tmp_path):
    from functions import clast_geometry as CG
    from functions import quadrat_frame as QF
    p = _framed_photo(tmp_path)
    fi = QF.frame_inset(p)
    assert fi is not None and fi.inset_px == 22 and fi.height == 600
    df = _disc_frame(_FRAMED)
    kept, removed = Q.drop_outside_frame(df, CG.contours_of(df), fi)
    assert removed == [1, 2] and list(kept["clast_ID"]) == [3]
    assert sorted(CG.contours_of(kept)) == [3] and list(kept.index) == [0]
    # Without outlines the centroid rule is the same.
    _, removed2 = Q.drop_outside_frame(
        _disc_frame(_FRAMED, with_contours=False), None, fi)
    assert removed2 == [1, 2]
    assert Q.drop_outside_frame(df, CG.contours_of(df), None) == (df, [])

    lines = []
    res = Q.run_gauge(p, Q.GaugeScale.from_gsd(0.001), tmp_path / "g",
                      log_fn=lines.append,
                      detect_fn=lambda **kw: [_disc_frame(_FRAMED)])
    assert res["excluded_by_frame"] == 2 and res["frame_removed_ids"] == [1, 2]
    assert res["excluded_by_segments"] == 0 and res["frame"] == fi
    assert list(res["df"]["clast_ID"]) == [3] and res["stats"]["n"] == 1
    assert list(pd.read_csv(res["csv"])["clast_ID"]) == [3]
    assert sorted(CG.read_contours(res["csv"])) == [3]
    side = json.loads(Path(res["json"]).read_text(encoding="utf-8"))
    assert side["excluded_by_frame"] == 2
    assert side["excluded_by_frame_clast_IDs"] == [1, 2] and side["n"] == 1
    assert side["frame"] == {"thickness_m": 0.02, "inset_px": 22,
                             "measured_size_m": [0.856, 0.556]}
    assert side["frame_rule"] == Q.FRAME_EXCLUSION_RULE
    assert any("frame 2.0 cm = 20 px + 2 px margin left out on each edge" in l
               and "2 detection(s) on the frame removed (clast_ID 1, 2)" in l
               for l in lines), lines
    # The rule can be turned off; the sidecar then says there was none.
    res2 = Q.run_gauge(p, Q.GaugeScale.from_gsd(0.001), tmp_path / "h",
                       exclude_frame=False,
                       detect_fn=lambda **kw: [_disc_frame(_FRAMED)])
    assert res2["excluded_by_frame"] == 0 and len(res2["df"]) == 3
    side2 = json.loads(Path(res2["json"]).read_text(encoding="utf-8"))
    assert side2["frame"] is None and side2["frame_rule"] is None
    # A photograph without the record has no frame to leave out.
    plain = _photo(tmp_path / "plain" / "IMG_0002.jpg")
    res3 = Q.run_gauge(plain, Q.GaugeScale.from_gsd(0.001), tmp_path / "i",
                       detect_fn=lambda **kw: [_disc_frame(_FRAMED)])
    assert res3["excluded_by_frame"] == 0 and len(res3["df"]) == 3
    assert res3["frame"] is None


def test_the_frame_is_applied_in_original_pixels_after_a_resampled_detection(tmp_path):
    """Detection on a half-size copy: the centroids come back in original
    pixels before the frame rule (22 px at full size) is applied."""
    p = _framed_photo(tmp_path)

    def half(**kw):
        return [_disc_frame([(c, x / 2, r_ / 2, rad / 2) for c, x, r_, rad in _FRAMED],
                            height=PHOTO_H // 2)]
    res = Q.run_gauge(p, Q.GaugeScale.from_gsd(0.001), tmp_path / "j",
                      detect_scale=0.5, detect_fn=half)
    assert res["frame_removed_ids"] == [1, 2] and list(res["df"]["clast_ID"]) == [3]


def test_the_frame_is_read_beside_the_photograph_the_user_picked(tmp_path):
    """A working copy (a converted HEIC) has no sidecar of its own: the
    record is read beside the source photograph."""
    src = _framed_photo(tmp_path)
    copy = _photo(tmp_path / "work" / "copy.jpg")
    assert Q.frame_inset_for(copy, src) is not None
    assert Q.frame_inset_for(copy, None) is None
    res = Q.run_gauge(copy, Q.GaugeScale.from_gsd(0.001), tmp_path / "k",
                      source_image=src,
                      detect_fn=lambda **kw: [_disc_frame(_FRAMED)])
    assert res["frame_removed_ids"] == [1, 2]


def test_auto_detection_scale_is_half_for_maskrcnn_and_full_for_a_plugin():
    """A plug-in sets its smallest grain in pixels, so a half-size copy loses
    the small clasts; Mask R-CNN keeps almost all of them at half size."""
    assert Q.DETECT_SCALE_DEFAULT == Q.DETECT_SCALE_AUTO
    assert Q.resolve_detect_scale("Auto", "maskrcnn") == 0.5
    assert Q.resolve_detect_scale("Auto", "seg") == 1.0
    assert Q.resolve_detect_scale("Auto", "pebblecounts") == 1.0
    assert Q.resolve_detect_scale(None, "imagegrains") == 1.0
    assert Q.resolve_detect_scale("1/4", "seg") == 0.25
    assert Q.resolve_detect_scale("1", "maskrcnn") == 1.0
    assert Q.DETECT_SCALE_OPTIONS[0] == "Auto"
    assert set(Q.DETECT_SCALES) <= set(Q.DETECT_SCALE_OPTIONS)
