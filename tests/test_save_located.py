"""Keeping what a survey found.

The survey check writes nothing on purpose -- it exists to say where digitising
time is worth spending. But once it HAS placed five quadrats, the only way to
keep them was to open each row in the editor and press Save, five times, with
the ortho re-read each time. The placements are already computed and stored.

The load-bearing test here is the one asserting a refusal is never written. A
quadrat the matcher declined is not a low-confidence placement, it is an
absence of evidence, and a GeoTIFF on disk is indistinguishable from a measured
one.
"""
from __future__ import annotations

import numpy as np
import pytest

from functions import seeds as S

pytest.importorskip("osgeo")

from test_seeds import survey, ORIGIN, ORTHO_GSD      # noqa: F401,E402


def _run(survey):
    csvp = survey["dir"] / "seeds.csv"
    sx, sy = survey["seed"]
    csvp.write_text(f"photo,x,y,crs\nq_good,{sx},{sy},EPSG:2154\n",
                    encoding="utf-8")
    return S.check_survey(sorted(survey["photos"].glob("*.png")),
                          survey["ortho"], S.load_seed_table(csvp), None,
                          quadrat_gsd_m=0.001, search_radius_m=1.0)


def test_a_located_quadrat_is_written(survey, tmp_path):
    res = _run(survey)
    assert any(r.status == "located" for r in res), "nothing to save"
    got = S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    assert got["written"], got["skipped"]
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(got["written"][0])
    assert ds is not None
    assert ds.GetGeoTransform()[0] != 0.0
    assert ds.GetProjection(), "written without a CRS"
    ds = None


def test_a_refusal_is_never_written(survey, tmp_path):
    """The one that matters. q_alien is refused; nothing about it may reach
    disk, because a GeoTIFF cannot be told apart from a measured placement."""
    res = _run(survey)
    refused = [r for r in res if r.status != "located"]
    assert refused, "the fixture no longer produces a refusal"
    S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    names = {p.stem for p in (tmp_path / "out").rglob("*.tif")}
    for r in refused:
        assert not any(r.photo.split(".")[0] in n for n in names), \
            f"{r.photo} was refused and written anyway"


def test_the_placement_written_is_the_one_that_was_found(survey, tmp_path):
    res = _run(survey)
    ok = [r for r in res if r.status == "located"][0]
    S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    from osgeo import gdal
    gdal.UseExceptions()
    tif = next((tmp_path / "out").rglob("*.tif"))
    gt = gdal.Open(str(tif)).GetGeoTransform()
    M = np.asarray(ok.matrix, float)
    assert gt[0] == pytest.approx(M[0, 2], abs=1e-6)


def test_writing_twice_refuses_rather_than_clobbers(survey, tmp_path):
    """`validation/georectified/` holds hand-made reference placements; a
    batch that silently overwrote one would destroy the only ground truth."""
    res = _run(survey)
    first = S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    assert first["written"]
    second = S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    assert not second["written"]
    assert second["skipped"]
    assert "already exists" in second["skipped"][0][1]


def test_overwrite_is_possible_when_asked(survey, tmp_path):
    res = _run(survey)
    S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    again = S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out",
                           overwrite=True)
    assert again["written"]


def test_a_missing_photograph_is_reported_not_crashed(survey, tmp_path):
    res = _run(survey)
    got = S.save_located(res, tmp_path / "nowhere", survey["ortho"], tmp_path / "out")
    assert not got["written"]
    assert got["skipped"]
    assert "no longer there" in got["skipped"][0][1]


def test_nothing_located_writes_nothing_and_says_so(survey, tmp_path):
    res = S.check_survey(sorted(survey["photos"].glob("*.png")),
                         survey["ortho"], None, None, quadrat_gsd_m=0.001)
    got = S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out")
    assert got == {"written": [], "skipped": []}
    assert not list((tmp_path / "out").rglob("*.tif")) if (tmp_path / "out").exists() else True


def test_progress_is_reported_while_saving(survey, tmp_path):
    seen = []
    res = _run(survey)
    S.save_located(res, survey["photos"], survey["ortho"], tmp_path / "out",
                   progress_fn=lambda d, t, m: seen.append((d, t, m)))
    assert seen and seen[-1][0] == seen[-1][1]
