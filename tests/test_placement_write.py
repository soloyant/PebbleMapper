"""Writing a placement a person decided on, and saying so everywhere it matters.

The theme is that a hand placement must be *distinguishable* downstream: the
sidecar has exactly one reader in this repository, a test, while the Validate
tab reads the GeoTIFF and the CSV, so a flag that lives only in the sidecar
marks nothing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from functions import georef as G
from functions import placement as P


ORTHO_GSD = 0.005
QUAD_GSD = 0.001


def _accepted_match(scale=QUAD_GSD, rot=12.0):
    """A QuadratMatch that passed, built directly rather than by matching."""
    p = P.Placement(558260.0, 6981590.0, rot, scale, (300, 300))
    q = G.MatchQuality(n_correspondences=200, n_inliers=60,
                       inlier_fraction=0.3, residual_m=0.01,
                       scale=scale / ORTHO_GSD, scale_expected=scale / ORTHO_GSD,
                       rotation_deg=rot, accepted=True)
    return G.QuadratMatch(P.matrix_from_placement(p), q, (558260.0, 6981590.0),
                          QUAD_GSD, ORTHO_GSD), p


def _img():
    rng = np.random.default_rng(0)
    return (rng.random((300, 300, 3)) * 255).astype(np.uint8)


def _clasts():
    return pd.DataFrame({"clast_ID": [1, 2], "x": [10.0, 200.0],
                         "y": [20.0, 250.0], "Clast_length": [0.03, 0.05]})


# --------------------------------------------------------------------------- #
#  Requirement 28 — the old door stays shut                                    #
# --------------------------------------------------------------------------- #
def test_a_rejected_match_is_still_refused_the_old_way(tmp_path):
    q = G.MatchQuality(accepted=False, reasons=["not enough correspondences"])
    m = G.QuadratMatch(None, q)
    with pytest.raises(ValueError, match="not accepted"):
        G.write_georeferenced(m, _img(), tmp_path, "q")
    assert list(tmp_path.iterdir()) == []


def test_a_hand_placement_needs_its_own_named_argument(tmp_path):
    """So no existing caller can write an unaccepted match by accident."""
    q = G.MatchQuality(accepted=False, reasons=["ambiguous"])
    rejected = G.QuadratMatch(None, q)
    p = P.Placement(558260.0, 6981590.0, 0.0, QUAD_GSD, (300, 300))
    out = G.write_georeferenced(
        rejected, _img(), tmp_path, "q",
        hand_placed={"matrix": P.matrix_from_placement(p),
                     "provenance": "manual", "agreement_score": 0.41})
    assert out["sidecar"].exists()


# --------------------------------------------------------------------------- #
#  Requirement 12 — the score floor                                            #
# --------------------------------------------------------------------------- #
def test_a_placement_below_the_score_floor_is_refused(tmp_path):
    m, p = _accepted_match()
    with pytest.raises(ValueError, match="floor"):
        G.write_georeferenced(
            m, _img(), tmp_path, "q",
            hand_placed={"matrix": P.matrix_from_placement(p),
                         "agreement_score": 0.05})
    assert not (tmp_path / "q.tif").exists()


def test_a_placement_above_the_floor_is_written(tmp_path):
    m, p = _accepted_match()
    G.write_georeferenced(m, _img(), tmp_path, "q",
                          hand_placed={"matrix": P.matrix_from_placement(p),
                                       "agreement_score": 0.42})
    assert (tmp_path / "q.georef.json").exists()


# --------------------------------------------------------------------------- #
#  Requirement 26 — not over the reference placements                          #
# --------------------------------------------------------------------------- #
def test_writing_over_an_existing_stem_is_refused(tmp_path):
    """`validation/georectified/` holds the hand-made placements this method is
    validated against; clobbering one with an estimate is unrecoverable."""
    m, _ = _accepted_match()
    G.write_georeferenced(m, _img(), tmp_path, "q")
    with pytest.raises(FileExistsError) as ex:
        G.write_georeferenced(m, _img(), tmp_path, "q")
    msg = str(ex.value)
    assert "already exists" in msg and "fitted" in msg and "overwrite=True" in msg


def test_overwrite_is_possible_when_asked_for(tmp_path):
    m, _ = _accepted_match()
    G.write_georeferenced(m, _img(), tmp_path, "q")
    G.write_georeferenced(m, _img(), tmp_path, "q", overwrite=True)


# --------------------------------------------------------------------------- #
#  Requirements 22-24, 27 — what gets recorded, and where                      #
# --------------------------------------------------------------------------- #
def test_the_geotiff_and_the_csv_both_carry_the_saved_transform(tmp_path):
    """Not the fitted one. Leaving the CSV on the fit while the raster moved
    would put the two outputs in different frames, and the CSV is what the
    validation statistics are computed from."""
    from osgeo import gdal
    m, fitted = _accepted_match()
    moved = P.translate(P.set_rotation(fitted, fitted.rotation_deg + 3.0),
                        0.40, -0.25)
    Msaved = P.matrix_from_placement(moved)

    out = G.write_georeferenced(
        m, _img(), tmp_path, "q", clasts=_clasts(),
        hand_placed={"matrix": Msaved, "provenance": "hand-edited",
                     "agreement_score": 0.39,
                     "delta": P.placement_delta(fitted, moved)})

    ds = gdal.Open(str(out["geotiff"]))
    gt = ds.GetGeoTransform()
    meta = ds.GetMetadata()
    ds = None
    assert gt[0] == pytest.approx(Msaved[0, 2], abs=1e-6)
    assert gt[1] == pytest.approx(Msaved[0, 0], abs=1e-9)

    df = pd.read_csv(out["csv"])
    wx, wy = G.world_from_pixels(Msaved, [10.0], [20.0])
    assert df["x"].iloc[0] == pytest.approx(wx[0], abs=1e-6)
    assert df["y"].iloc[0] == pytest.approx(wy[0], abs=1e-6)

    # and provenance rides on BOTH, not only in the sidecar
    assert meta.get("PM_PLACEMENT") == "hand-edited"
    assert set(df["placement"]) == {"hand-edited"}


def test_the_sidecar_records_the_edit_rather_than_implying_the_fit(tmp_path):
    m, fitted = _accepted_match()
    moved = P.translate(fitted, 0.10, 0.0)
    out = G.write_georeferenced(
        m, _img(), tmp_path, "q",
        hand_placed={"matrix": P.matrix_from_placement(moved),
                     "provenance": "hand-edited", "agreement_score": 0.38,
                     "delta": P.placement_delta(fitted, moved),
                     "operator": "ac", "timestamp": "2026-08-06T10:00:00",
                     "view": {"mode": "high-pass", "opacity": 0.55}})
    side = json.loads(out["sidecar"].read_text(encoding="utf-8"))

    assert side["provenance"] == "hand-edited"
    assert side["fitted_transform"] is not None
    assert side["delta"]["translation_m"] == pytest.approx(0.10, abs=1e-9)
    assert side["delta"]["rotation_deg"] == pytest.approx(0.0, abs=1e-9)
    assert side["agreement_score"] == pytest.approx(0.38)
    assert side["operator"] == "ac"
    assert side["view"]["mode"] == "high-pass"
    assert side["ransac_seed"] == G.DEFAULTS["ransac_seed"]
    # the inlier count belongs to the FIT and must say so
    assert "fitted" in side["quality_describes"]
    assert side["quality"]["n_inliers"] == 60


def test_a_placement_with_no_fit_records_no_inlier_count(tmp_path):
    """Requirement 23. Not a zero -- a zero reads as 'measured, and it was
    none', which is a different and misleading claim."""
    q = G.MatchQuality(accepted=False, reasons=["not in this ortho"])
    rejected = G.QuadratMatch(None, q)
    p = P.Placement(558260.0, 6981590.0, 0.0, QUAD_GSD, (300, 300))
    out = G.write_georeferenced(
        rejected, _img(), tmp_path, "q",
        hand_placed={"matrix": P.matrix_from_placement(p),
                     "provenance": "manual", "agreement_score": 0.36})
    side = json.loads(out["sidecar"].read_text(encoding="utf-8"))
    assert side["quality"] is None
    assert side["provenance"] == "manual"
    assert "no fit" in side["quality_describes"]


def test_an_ordinary_fitted_save_is_unchanged_apart_from_added_fields(tmp_path):
    """Requirement 27's 'additively'. Anything reading today's sidecar keeps
    working; the transform itself is untouched."""
    m, p = _accepted_match()
    out = G.write_georeferenced(m, _img(), tmp_path, "q", clasts=_clasts())
    side = json.loads(out["sidecar"].read_text(encoding="utf-8"))
    for key in ("transform", "quality", "seed_xy", "quadrat_gsd_m",
                "ortho_gsd_m", "crs", "sources"):
        assert key in side
    np.testing.assert_allclose(np.array(side["transform"]),
                               P.matrix_from_placement(p), atol=1e-9)
    assert side["provenance"] == "fitted"
    assert "fitted_transform" not in side          # nothing was edited
    assert set(pd.read_csv(out["csv"])["placement"]) == {"fitted"}


# --------------------------------------------------------------------------- #
#  Requirement 25 — provenance readable downstream                             #
# --------------------------------------------------------------------------- #
def test_the_validate_engine_can_read_the_placement_back(tmp_path):
    """The sidecar has one reader in this repository, a test. The Validate tab
    reads the raster, so provenance has to be ON the raster to mark anything."""
    from functions import quadrat_validation as qv
    m, p = _accepted_match()
    out = G.write_georeferenced(
        m, _img(), tmp_path, "hand", clasts=_clasts(),
        hand_placed={"matrix": P.matrix_from_placement(p),
                     "provenance": "hand-edited", "agreement_score": 0.41})
    got = qv.placement_provenance(str(out["geotiff"]))
    assert got["placement"] == "hand-edited"
    assert got["agreement_score"] == pytest.approx(0.41, abs=1e-4)


def test_a_fitted_placement_reads_back_as_fitted(tmp_path):
    from functions import quadrat_validation as qv
    m, _ = _accepted_match()
    out = G.write_georeferenced(m, _img(), tmp_path, "auto")
    assert qv.placement_provenance(str(out["geotiff"]))["placement"] == "fitted"


def test_provenance_of_an_unmarked_raster_is_empty_not_wrong(tmp_path):
    """An ordinary GeoTIFF from anywhere else must not be reported as fitted."""
    from functions import quadrat_validation as qv
    from osgeo import gdal
    path = tmp_path / "plain.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 1)
    ds.SetGeoTransform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    ds = None
    got = qv.placement_provenance(str(path))
    assert got["placement"] == ""
    assert got["agreement_score"] is None


# --------------------------------------------------------------------------- #
#  Requirement 29 — the batch keeps what it found                              #
# --------------------------------------------------------------------------- #
def test_a_quadrat_check_carries_its_placement():
    """The batch computes a placement for every photo and used to discard it,
    so opening one afterwards paid the match cost again -- 7.9 s to 82 s a
    quadrat, twice over a season."""
    from functions.seeds import QuadratCheck
    c = QuadratCheck("a.jpg", "located", "", 40, 0.01, "list",
                     matrix=[[1.0, 0, 5.0], [0, -1.0, 6.0], [0, 0, 1.0]],
                     seed_world=(5.0, 6.0))
    assert c.matrix is not None and c.seed_world == (5.0, 6.0)
    p = P.placement_from_matrix(np.asarray(c.matrix, float), (10, 10))
    assert p.easting == pytest.approx(10.0)

    refused = QuadratCheck("b.jpg", "not_located", "ambiguous", 3, float("nan"),
                           "pin", matrix=None, seed_world=(1.0, 2.0))
    assert refused.matrix is None          # a refusal has no transform...
    assert refused.seed_world == (1.0, 2.0)   # ...but the editor can start here
