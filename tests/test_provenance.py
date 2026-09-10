"""Digitize's records as a table, their provenance sidecar, and the sample set.

Pure logic, no NiceGUI: ``functions.digitize.measure_records`` (a record is
measured on a window of its mask, the same numbers as on the full mask, and
says which record each row is), ``functions.provenance`` (the
``<csv stem>.provenance.json`` round trip, the counts and the model line)
and ``functions.gauge.pool_sample_set`` / ``sample_set_figure`` /
``write_sample_set``. The Digitize controls that use them are tested in
tests/test_digitize_scale.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from functions import gauge as Q
from functions import provenance as PV
from functions.digitize import (circle_to_mask, measure_record_px,
                                measure_records, polygon_to_mask)


def _disc(cx, cy, r, shape=(300, 400), **kw):
    rec = {"shape": "circle", "params": {"center": [cx, cy], "radius": r},
           "mask": circle_to_mask((cx, cy), r, shape).astype(bool),
           "centroid_x": float(cx), "centroid_y": float(cy), "label": ""}
    rec.update(kw)
    return rec


def test_a_record_is_measured_on_a_window_with_the_full_mask_numbers():
    from functions.clasts_detection import _measure_clast
    verts = [[120, 80], [220, 95], [240, 170], [150, 210], [100, 150]]
    mask = polygon_to_mask(verts, (300, 400)).astype(bool)
    rec = {"shape": "polygon", "params": {"vertices": verts}, "mask": mask,
           "score": 0.8}
    full = _measure_clast(mask, 0.8, 1.0)
    win = measure_record_px(rec)
    # What the mask itself gives is identical ...
    for k in ("Ellipse_major_axis", "Ellipse_minor_axis", "Surface_area",
              "Perimeter", "Equivalent_diameter", "Solidity"):
        assert win[k] == pytest.approx(full[k], rel=1e-9), k
    # ... the algebraic ellipse fit behind the chords and the orientation
    # is conditioned by the coordinates' magnitude: within 0.5 % / 0.5 deg.
    for k in ("Clast_length", "Clast_width"):
        assert win[k] == pytest.approx(full[k], rel=5e-3), k
    assert abs(win["Orientation"] - full["Orientation"]) < 0.5
    assert win["center_x"] == pytest.approx(full["center_x"], abs=0.05)
    assert win["center_y"] == pytest.approx(full["center_y"], abs=0.05)
    np.testing.assert_allclose(np.sort(win["_contour_x"]), np.sort(full["_contour_x"]))
    # Cached for as long as the mask is the same array; an edit replaces it.
    assert measure_record_px(rec) is win
    rec["mask"] = polygon_to_mask(verts[:4], (300, 400)).astype(bool)
    assert measure_record_px(rec) is not win


def test_measure_records_says_which_record_each_row_is_and_scales_once():
    empty = {"shape": "circle", "params": {"center": [5, 5], "radius": 0},
             "mask": np.zeros((300, 400), bool)}
    recs = [_disc(100, 100, 20), empty, _disc(300, 200, 12, label="big one")]
    df, pos, contours = measure_records(recs, 300, 0.002, with_contours=True)
    assert pos == [0, 2] and list(df["clast_ID"]) == [1, 2]
    assert list(df["Label"]) == ["", "big one"]
    df1, _ = measure_records(recs, 300, 1.0)
    np.testing.assert_allclose(df["Clast_length"], df1["Clast_length"] * 0.002)
    np.testing.assert_allclose(df["Surface_area"], df1["Surface_area"] * 0.002 ** 2)
    np.testing.assert_allclose(df["y"], 300 - np.array([100, 200]), atol=0.5)
    # Outlines in the CSV frame (y up), unscaled.
    ring = np.asarray(contours[1])
    assert abs(ring[:, 1].mean() - 200) < 1.5 and abs(ring[:, 0].mean() - 100) < 1.5


def test_the_provenance_sidecar_round_trips_with_edits(tmp_path):
    recs = [_disc(100, 100, 20), _disc(200, 100, 20, origin="Mask R-CNN (Soloy et al., 2020)"),
            _disc(300, 100, 20, origin="Mask R-CNN (Soloy et al., 2020)", edited=True),
            _disc(300, 220, 20, origin="Segment Every Grain (Sylvester)")]
    clasts = PV.clast_origins(recs, [0, 1, 2, 3])
    assert clasts[1] == {"origin": "hand", "edited": False}
    assert clasts[3] == {"origin": "Mask R-CNN (Soloy et al., 2020)", "edited": True}
    models = [PV.model_entry("maskrcnn", None, display_name="Mask R-CNN (Soloy et al., 2020)",
                             detect_scale=0.5, min_confidence=0.7, license="MIT")]
    csv = tmp_path / "IMG_truth.csv"
    out = PV.write_provenance(csv, clasts, models, image="IMG.heic")
    assert out == tmp_path / "IMG_truth.provenance.json"
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format"] == "pebblemapper-provenance" and doc["edited"] == 1
    assert doc["counts"] == {"hand": 1, "Mask R-CNN (Soloy et al., 2020)": 2,
                             "Segment Every Grain (Sylvester)": 1}
    assert doc["models"][0]["detect_scale"] == 0.5
    assert doc["models"][0]["min_confidence"] == 0.7
    back = PV.read_provenance(csv)
    assert back["clasts"] == clasts
    assert PV.summary_line(back["clasts"], back["models"]) == (
        "Detections: Mask R-CNN (Soloy et al., 2020), 2 kept, 1 edited; "
        "Segment Every Grain (Sylvester), 1 kept; 1 drawn by hand")
    assert PV.summary_line({1: {"origin": "hand"}}) == "1 drawn by hand"
    assert PV.read_provenance(tmp_path / "nothing.csv") is None
    # A hand-drawn clast is never "edited".
    assert PV.clast_origins([_disc(1, 1, 1, edited=True)], [0])[1]["edited"] is False


def _truth(lengths_m, unit=None):
    L = np.asarray(lengths_m, float)
    df = pd.DataFrame({"clast_ID": np.arange(1, len(L) + 1), "x": 1.0, "y": 1.0,
                       "Clast_length": L, "Clast_width": L * 0.7,
                       "Ellipse_major_axis": L, "Ellipse_minor_axis": L * 0.7,
                       "Surface_area": L * L * 0.55, "Perimeter": 3 * L,
                       "Equivalent_diameter": L * 0.8, "Eccentricity": 0.5,
                       "Solidity": 0.95, "Mean_intensity": np.nan, "Score": 1.0,
                       "Orientation": 45.0})
    if unit:
        df["unit"] = unit
    return df


def test_pooling_keeps_metric_tables_together_in_mm_and_skips_a_custom_unit():
    rng = np.random.default_rng(3)
    a = rng.uniform(0.02, 0.09, 40)           # metres, photograph at 1 mm/px
    b = rng.uniform(0.01, 0.05, 25)           # metres, photograph at 2 mm/px
    c = rng.uniform(0.1, 0.5, 30)             # in boot widths
    prov = {"clasts": {1: {"origin": "hand"}, 2: {"origin": "Mask R-CNN"}},
            "models": [{"display_name": "Mask R-CNN"}]}
    pooled, summary, skipped = Q.pool_sample_set([
        {"photo": "a.jpg", "df": _truth(a), "provenance": prov},
        {"photo": "b.jpg", "df": _truth(b)},
        {"photo": "c.jpg", "df": _truth(c, unit="boot width")}])
    assert pooled.attrs["unit"] == "mm" and pooled.attrs["metric"] is True
    assert list(pooled.columns[:2]) == ["photo", "clast_ID"]
    assert set(pooled["photo"]) == {"a.jpg", "b.jpg"} and len(pooled) == 65
    assert (pooled["unit"] == "mm").all()
    np.testing.assert_allclose(pooled["Clast_length"], np.r_[a, b] * 1000)
    np.testing.assert_allclose(pooled["Surface_area"], np.r_[a * a, b * b] * 0.55 * 1e6)
    assert skipped == [{"photo": "c.jpg",
                        "reason": "in 'boot width' units, not pooled with the millimetre set"}]
    assert list(summary["photo"]) == ["a.jpg", "b.jpg", "(pooled)"]
    row = summary.set_index("photo").loc["(pooled)"]
    assert row["n"] == 65
    assert row["D50"] == pytest.approx(float(np.median(np.r_[a, b] * 1000)))
    assert math.isfinite(row["sorting_phi"])
    assert summary.set_index("photo").loc["a.jpg", "D50"] == pytest.approx(np.median(a) * 1000)
    assert summary.set_index("photo").loc["a.jpg", "models"] == "Mask R-CNN"
    assert summary.set_index("photo").loc["(pooled)", "origins"] == "hand: 1; Mask R-CNN: 1"


def test_tables_in_the_same_custom_unit_pool_and_the_largest_group_wins():
    c1, c2 = [0.2, 0.3, 0.4], [0.25, 0.35]
    pooled, summary, skipped = Q.pool_sample_set([
        {"photo": "c1.jpg", "df": _truth(c1, unit="boot width")},
        {"photo": "c2.jpg", "df": _truth(c2, unit="boot width")},
        {"photo": "m.jpg", "df": _truth([0.03, 0.04])},
        {"photo": "e.jpg", "df": _truth([])}])
    assert pooled.attrs["unit"] == "boot width" and pooled.attrs["metric"] is False
    np.testing.assert_allclose(pooled["Clast_length"], c1 + c2)
    row = summary.set_index("photo").loc["(pooled)"]
    assert row["D50"] == pytest.approx(0.3) and math.isnan(row["sorting_phi"])
    reasons = {s["photo"]: s["reason"] for s in skipped}
    assert reasons["m.jpg"].startswith("in metres") and reasons["e.jpg"] == "no saved clasts"
    # A tie goes to the metric group.
    pooled2, _s, sk2 = Q.pool_sample_set([
        {"photo": "c1.jpg", "df": _truth(c1, unit="boot width")},
        {"photo": "m.jpg", "df": _truth([0.03, 0.04])}])
    assert pooled2.attrs["metric"] is True and sk2[0]["photo"] == "c1.jpg"


def test_the_sample_set_figure_and_files(tmp_path):
    rng = np.random.default_rng(4)
    entries = [{"photo": f"p{k:02d}.jpg", "df": _truth(rng.uniform(0.01, 0.08, 30))}
               for k in range(3)]
    res = Q.write_sample_set(entries, tmp_path / "gauge", "Site__images",
                             disclaimer=Q.DISCLAIMER)
    for key, name in (("csv", "Site__images_sampleset.csv"),
                      ("summary", "Site__images_sampleset_summary.csv"),
                      ("distribution", "Site__images_sampleset_distribution.png")):
        assert Path(res[key]) == tmp_path / "gauge" / name and Path(res[key]).exists()
    back = pd.read_csv(res["summary"])
    assert list(back["photo"]) == ["p00.jpg", "p01.jpg", "p02.jpg", "(pooled)"]
    fig = Q.sample_set_figure(res["pooled"], disclaimer=Q.DISCLAIMER)
    assert fig.pm_meta["photos"] == 3 and fig.pm_meta["per_photo_legend"] is True
    assert fig.get_supxlabel().replace("\n", " ") == Q.DISCLAIMER
    ax2 = fig.axes[1]
    assert any(l.get_linewidth() > 2 for l in ax2.get_lines())
    # Thirty photographs: grey lines, no per-photograph legend.
    many = [{"photo": f"q{k:02d}.jpg", "df": _truth(rng.uniform(0.01, 0.08, 12))}
            for k in range(30)]
    pooled, _s, _k = Q.pool_sample_set(many)
    fig = Q.sample_set_figure(pooled)
    assert fig.pm_meta["photos"] == 30 and fig.pm_meta["per_photo_legend"] is False
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert labels == ["Pooled (n = 360)"]
    assert fig.get_supxlabel() == ""
