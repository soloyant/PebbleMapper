"""ROI loading + point-in-polygon — the core of ortho tile-skip and quadrat
clast-drop ROI filtering (functions.clasts_detection._load_roi_paths)."""
import json

import pytest

pytest.importorskip("matplotlib")


def _write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection",
                                "features": features}), encoding="utf-8")


def test_load_roi_paths_polygon_contains(tmp_path):
    from functions.clasts_detection import _load_roi_paths
    gj = tmp_path / "roi.geojson"
    # A 10x10 square (pixel/world coords are taken as-is).
    _write_geojson(gj, [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[
            [0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}])
    paths = _load_roi_paths(str(gj))
    assert len(paths) == 1
    assert paths[0].contains_point((5, 5)) is True
    assert paths[0].contains_point((20, 20)) is False


def test_load_roi_paths_multipolygon(tmp_path):
    from functions.clasts_detection import _load_roi_paths
    gj = tmp_path / "roi.geojson"
    _write_geojson(gj, [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "MultiPolygon", "coordinates": [
            [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
            [[[8, 8], [10, 8], [10, 10], [8, 10], [8, 8]]],
        ]}}])
    paths = _load_roi_paths(str(gj))
    assert len(paths) == 2
    # A point inside either sub-polygon is "inside the ROI".
    assert any(p.contains_point((1, 1)) for p in paths)
    assert any(p.contains_point((9, 9)) for p in paths)
    assert not any(p.contains_point((5, 5)) for p in paths)


def test_load_roi_paths_robust_to_missing_and_bad(tmp_path):
    from functions.clasts_detection import _load_roi_paths
    assert _load_roi_paths(None) == []
    assert _load_roi_paths("") == []
    assert _load_roi_paths(str(tmp_path / "nope.geojson")) == []
    bad = tmp_path / "bad.geojson"
    bad.write_text("{not valid json", encoding="utf-8")
    assert _load_roi_paths(str(bad)) == []
    # Degenerate polygon (<3 points) is skipped, not crashed on.
    degen = tmp_path / "degen.geojson"
    _write_geojson(degen, [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]}}])
    assert _load_roi_paths(str(degen)) == []
