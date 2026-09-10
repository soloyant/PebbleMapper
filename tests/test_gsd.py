"""functions/gsd.py — the one GSD reader, plus the default-geotransform sentinel."""
import json
from pathlib import Path

import pytest

from functions.gsd import GsdInfo, effective_gsd, parse_gsd_from_filename


def _jpg(tmp_path, name="photo_GSD=0.005m.jpg"):
    """A real (tiny) JPG so PIL/GDAL-facing tests exercise real readers."""
    from PIL import Image
    p = tmp_path / name
    Image.new("RGB", (8, 8), (120, 120, 120)).save(p, format="JPEG")
    return p


def test_filename_new_format():
    assert parse_gsd_from_filename("x_rectified_GSD=0.003m.jpg") == 0.003
    assert parse_gsd_from_filename("a_GSD=0.0021m_b.png") == 0.0021


def test_filename_legacy_format():
    assert parse_gsd_from_filename("x_GSD=2p100mm.jpg") == pytest.approx(0.0021)


def test_filename_none_and_nonpositive():
    assert parse_gsd_from_filename("plain_photo.jpg") is None
    assert parse_gsd_from_filename("x_GSD=0m.jpg") is None


def test_sidecar_wins_over_filename(tmp_path):
    img = _jpg(tmp_path)                      # filename says 0.005
    Path(str(img) + ".json").write_text(
        json.dumps({"PM_GSD": "0.003000"}), encoding="utf-8")
    info = effective_gsd(img)
    assert info == GsdInfo(0.003, "sidecar", None)


def test_filename_fallback_without_sidecar(tmp_path):
    img = _jpg(tmp_path)
    info = effective_gsd(img)
    assert info.gsd == 0.005 and info.source == "filename"
    assert info.warning is None


def test_corrupt_sidecar_warns_and_falls_back(tmp_path):
    img = _jpg(tmp_path)
    Path(str(img) + ".json").write_text("{ not json", encoding="utf-8")
    info = effective_gsd(img)
    assert info.warning and "unreadable" in info.warning
    assert info.gsd == 0.005 and info.source == "filename"  # past the sidecar


def test_nonpositive_pm_gsd_warns_and_uses_global(tmp_path):
    img = _jpg(tmp_path, "plain.jpg")
    Path(str(img) + ".json").write_text(
        json.dumps({"PM_GSD": "zero"}), encoding="utf-8")
    info = effective_gsd(img)
    assert info.gsd is None and info.source is None
    assert "not a positive number" in info.warning


def test_no_signals_is_clean_none(tmp_path):
    info = effective_gsd(_jpg(tmp_path, "plain.jpg"))
    assert info == GsdInfo(None, None, None)


def test_read_georef_reports_no_gsd_for_plain_jpg(tmp_path):
    """GDAL's default geotransform (px size 1.0) must never be recorded
    as a measured GSD for a non-georeferenced image."""
    from detectors.maskrcnn import _read_georef
    crs, gsd = _read_georef(str(_jpg(tmp_path, "plain.jpg")))
    assert gsd is None
