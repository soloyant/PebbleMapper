"""One clast, counted once.

The report's headline population summed every CSV variant it found. A stem
routinely carries "<stem>_merged.csv" AND
"<stem>_merged_individual_clast_values.csv" — the same clasts, same columns,
different order — and `output_results/vectors/` also collects derived tables
such as a zonal polygon summary. On the example project that inflated the
cover's "Total clasts" from 15,673 to 31,241, roughly double.
"""
from __future__ import annotations

import pytest

from functions import report as rpt


def _write(p, header, rows=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines += [",".join(str(i) for _ in header) for i in range(rows)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


DETECT_COLS = ["clast_ID", "x", "y", "Ellipse_major_axis", "Clast_length",
               "Surface_area", "Score", "Orientation"]
ZONAL_COLS = ["polygon_id", "count", "area_m2", "density", "mean", "std"]
TRANSECT_COLS = ["transect_id", "distance_m", "raster_value"]


def test_a_zonal_summary_is_not_a_pile_of_clasts(tmp_path):
    zonal = _write(tmp_path / "img_merged_zonal.csv", ZONAL_COLS, rows=2)
    assert rpt._is_detection_csv(zonal) is False


def test_a_transect_sample_table_is_not_detections(tmp_path):
    t = _write(tmp_path / "img.transects.csv", TRANSECT_COLS, rows=40)
    assert rpt._is_detection_csv(t) is False


def test_a_detection_table_is_recognised(tmp_path):
    d = _write(tmp_path / "img_merged.csv", DETECT_COLS, rows=10)
    assert rpt._is_detection_csv(d) is True


def test_an_unreadable_or_empty_file_is_not_counted(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert rpt._is_detection_csv(empty) is False
    assert rpt._is_detection_csv(tmp_path / "absent.csv") is False


def test_one_merged_population_per_stem(tmp_path):
    """The bug: several merged variants of one image were all counted."""
    inv = {"vectors": [
        {"path": tmp_path / "img_merged.csv", "image_stem": "img",
         "is_merged": True, "window_size": None, "n_rows": 8300},
        {"path": tmp_path / "img_merged_individual_clast_values.csv",
         "image_stem": "img", "is_merged": True, "window_size": None,
         "n_rows": 8201},
    ]}
    can = rpt._canonical_vectors(inv)
    assert len(can) == 1
    assert can[0]["path"].name == "img_merged.csv", (
        "the merge step's own output is the population; the rest are "
        "re-exports of it")


def test_per_window_runs_are_kept_when_nothing_merged_them(tmp_path):
    inv = {"vectors": [
        {"path": tmp_path / "img_ws1m.csv", "image_stem": "img",
         "is_merged": False, "window_size": 1.0, "n_rows": 100},
        {"path": tmp_path / "img_ws2.5m.csv", "image_stem": "img",
         "is_merged": False, "window_size": 2.5, "n_rows": 40},
    ]}
    assert len(rpt._canonical_vectors(inv)) == 2


def test_a_merge_supersedes_the_windows_that_fed_it(tmp_path):
    inv = {"vectors": [
        {"path": tmp_path / "img_ws1m.csv", "image_stem": "img",
         "is_merged": False, "window_size": 1.0, "n_rows": 100},
        {"path": tmp_path / "img_merged.csv", "image_stem": "img",
         "is_merged": True, "window_size": None, "n_rows": 120},
    ]}
    can = rpt._canonical_vectors(inv)
    assert [e["path"].name for e in can] == ["img_merged.csv"]


def test_separate_images_are_both_counted(tmp_path):
    inv = {"vectors": [
        {"path": tmp_path / "a_merged.csv", "image_stem": "a",
         "is_merged": True, "window_size": None, "n_rows": 10},
        {"path": tmp_path / "b_merged.csv", "image_stem": "b",
         "is_merged": True, "window_size": None, "n_rows": 20},
    ]}
    assert len(rpt._canonical_vectors(inv)) == 2


def test_the_choice_is_deterministic(tmp_path):
    """Two builds of one project must report the same population."""
    entries = [
        {"path": tmp_path / "img_merged_b.csv", "image_stem": "img",
         "is_merged": True, "window_size": None, "n_rows": 1},
        {"path": tmp_path / "img_merged_a.csv", "image_stem": "img",
         "is_merged": True, "window_size": None, "n_rows": 2},
    ]
    first = rpt._canonical_vectors({"vectors": list(entries)})
    second = rpt._canonical_vectors({"vectors": list(reversed(entries))})
    assert first[0]["path"] == second[0]["path"]
