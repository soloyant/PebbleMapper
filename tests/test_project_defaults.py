"""functions/project_defaults.py — the shared per-field resolver."""
import time
from pathlib import Path

import pytest

import functions.layout as layout
import functions.project_defaults as pd


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "DATASETS_ROOT", tmp_path)
    layout.ensure_project_layout("P")
    return "P"


def _touch(project, kind, name, *, age=0.0, body="clast_ID,x,y\n1,0,0\n"):
    d = layout.project_path(project, kind)
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body, encoding="utf-8")
    if age:
        t = time.time() - age
        import os
        os.utime(p, (t, t))
    return p


def test_empty_project_returns_nothing(proj):
    assert pd.merged_csvs(proj) == []
    assert pd.best_clast_csv(proj) is None
    assert pd.orthos(proj) is None or pd.orthos(proj) == []
    assert pd.mergeable_stems(proj) == []


def test_newest_merged_csv_wins(proj):
    _touch(proj, "vectors", "a_merged.csv", age=100)
    newest = _touch(proj, "vectors", "b_merged.csv", age=1)
    got = pd.merged_csvs(proj)
    assert got[0] == newest                       # newest by mtime, clarified
    # the derived companion and the zonal output are excluded — only the
    # bare <stem>_merged.csv qualifies
    _touch(proj, "vectors", "b_merged_individual_clast_values.csv", age=0.1)
    _touch(proj, "vectors", "b_merged_zonal.csv", age=0.05)
    names = [p.name for p in pd.merged_csvs(proj)]
    assert names == ["b_merged.csv", "a_merged.csv"]


def test_best_clast_csv_prefers_merge_then_detection(proj):
    det = _touch(proj, "vectors", "img_ws5m.csv")
    assert pd.best_clast_csv(proj) == det         # only a detection CSV so far
    mg = _touch(proj, "vectors", "img_merged.csv", age=0.01)
    assert pd.best_clast_csv(proj) == mg          # merge now wins


def test_source_ortho_stem_match(proj):
    _touch(proj, "images", "example_n1_of_UAV_ortho_image.tif", body="x")
    n2 = _touch(proj, "images", "example_n2_of_UAV_ortho_image.tif", body="x",
                age=500)  # older, but stem-matched should still win
    got = pd.source_ortho_for(
        "example_n2_of_UAV_ortho_image_merged.csv", proj)
    assert got == n2


def test_window_csvs_by_stem_and_mergeable(proj):
    _touch(proj, "vectors", "n2_ws5m.csv")
    _touch(proj, "vectors", "n2_ws2.5m.csv")
    _touch(proj, "vectors", "n2_ws1m.csv")
    _touch(proj, "vectors", "solo_ws1m.csv")       # only one window
    groups = pd.window_csvs_by_stem(proj)
    assert set(w for _, w in groups["n2"]) == {5.0, 2.5, 1.0}
    # sorted descending by window (merge order)
    assert [w for _, w in groups["n2"]] == [5.0, 2.5, 1.0]
    assert pd.mergeable_stems(proj) == ["n2"]      # solo has <2 windows

def test_source_ortho_pairs_a_prefixed_csv_with_its_image(proj):
    """A CSV named with the site/date origin prefix still finds the image it
    came from; falling through to the newest ortho would pair it wrongly."""
    n1 = _touch(proj, "images", "example_n1_of_UAV_ortho_image.tif", body="x")
    n2 = _touch(proj, "images", "example_n2_of_UAV_ortho_image.tif", body="x",
                age=500)
    got = pd.source_ortho_for(
        "Site_20200610__example_n2_of_UAV_ortho_image_merged.csv", proj)
    assert got == n2
    assert pd.source_ortho_for(
        "Site__example_n1_of_UAV_ortho_image_ws2.5m.csv", proj) == n1


def test_window_csvs_group_prefixed_names_by_image(proj):
    _touch(proj, "vectors", "Site__n2_ws5m.csv")
    _touch(proj, "vectors", "Site__n2_ws1m.csv")
    groups = pd.window_csvs_by_stem(proj)
    assert len(groups) == 1
    (stem, rows), = groups.items()
    assert [w for _, w in rows] == [5.0, 1.0]


def test_rectified_photos_come_from_either_orthorectified_folder(proj):
    """Detect and Digitize open the rectified quadrat photographs first:
    validation/orthorectified (Orthorectify's output beside validation/raw),
    else input_data/images/orthorectified (beside raw photographs kept under
    images/); the legacy validation/images bucket only after those."""
    assert pd.rectified_photos(proj) == []
    legacy = _touch(proj, "validation_images", "old.jpg", body="x")
    assert pd.validation_images(proj) == [legacy]
    under_images = layout.project_path(proj, "images") / "orthorectified"
    under_images.mkdir(parents=True)
    b = under_images / "b_rectified_GSD=0.001m.jpg"
    b.write_text("x", encoding="utf-8")
    assert pd.rectified_photos(proj) == [b]
    assert pd.validation_images(proj) == [b]
    a = _touch(proj, "validation_rectified", "a_rectified_GSD=0.001m.jpg", body="x")
    assert layout.project_path(proj, "validation_rectified").name == "orthorectified"
    assert pd.rectified_photos(proj) == [a], "the validation folder wins"
    assert pd.validation_images(proj) == [a]


def test_validate_seeds_the_truth_s_own_photograph_and_detection(proj):
    """Validate pre-fills the source image and the detection CSV of the
    photograph the truth CSV was digitised on, not the newest file in each
    folder (on the caliper example the newest photograph was another one,
    and the overlay was drawn on it in silence)."""
    a = _touch(proj, "validation_rectified", "DSC_0854_rectified_GSD=0.000243m.jpg", body="x")
    b = _touch(proj, "validation_rectified", "DSC_0855_rectified_GSD=0.000229m.jpg", body="x", age=60)
    truth = _touch(proj, "validation", "DSC_0855_rectified_GSD=0.000229m_truth.csv")
    da = _touch(proj, "vectors", "P__DSC_0854_rectified_GSD=0.000243m_individual_clasts.csv")
    db = _touch(proj, "vectors", "P__DSC_0855_rectified_GSD=0.000229m_individual_clasts.csv", age=60)
    assert pd.validation_images(proj)[0] == a and pd.best_clast_csv(proj) == da, "newest first"
    assert pd.for_truth(truth, pd.validation_images(proj)) == b
    assert pd.for_truth(truth, pd.detection_csvs(proj)) == db
    # nothing to match on -> None, and the caller keeps its newest-first choice
    assert pd.for_truth(None, pd.validation_images(proj)) is None
    other = _touch(proj, "validation", "IMG_0001_truth.csv")
    assert pd.for_truth(other, pd.validation_images(proj)) is None
    assert pd.for_truth(truth, []) is None
    # a placed copy of the photograph counts (Georeference's *_georeferenced.tif);
    # a longer name that merely begins with the same characters does not
    g = _touch(proj, "validation_georectified", "DSC_0855_rectified_GSD=0.000229m_georeferenced.tif", body="x")
    assert pd.for_truth(truth, pd.georectified_images(proj)) == g
    assert pd.for_truth(Path("q1_truth.csv"), [Path("q10.jpg"), Path("q1.jpg")]) == Path("q1.jpg")
    assert pd.for_truth(Path("q1_truth.csv"), [Path("q10.jpg")]) is None


def test_raw_photos_prefer_validation_raw(proj):
    """Orthorectify opens on the photographs as shot: validation/raw first,
    else the photographs kept under input_data/images."""
    assert pd.raw_photos(proj) == []
    under_images = _touch(proj, "images", "b.jpg", body="x")
    assert pd.raw_photos(proj) == [under_images]
    raw = _touch(proj, "validation_raw", "a.JPG", body="x")
    assert pd.raw_photos(proj) == [raw]
