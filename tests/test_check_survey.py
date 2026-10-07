"""A survey checked tile by tile instead of quadrat by quadrat.

`check_quadrats` plans and reads windows per photo, so two quadrats a metre
apart pull the same ground off disk twice. Measured on a real
20-quadrat survey: 1620 window reads, 1620 distinct, 1307 megapixels
from a 588-megapixel ortho. `check_survey` walks one grid fixed to the ortho,
so the same search wants 256 distinct tiles and 207 megapixels.

The saving is only worth having if it is free. These assert that the tile-major
path finds the SAME quadrats, to the same accuracy, with the same seed
provenance -- and that it can report progress and be stopped, which the old one
could not.
"""
from __future__ import annotations

import numpy as np
import pytest

from functions import seeds as S

pytest.importorskip("osgeo")


# The `survey` fixture (a real GeoTIFF ortho plus quadrats cut from it) lives
# in test_seeds.py; reuse it rather than build a second one that could drift.
from test_seeds import survey, ORIGIN, ORTHO_GSD      # noqa: F401,E402


def _seed_csv(survey, name="seeds.csv"):
    p = survey["dir"] / name
    sx, sy = survey["seed"]
    p.write_text(f"photo,x,y,crs\nq_good,{sx},{sy},EPSG:2154\n",
                 encoding="utf-8")
    return S.load_seed_table(p)


def _by_photo(results):
    return {r.photo: r for r in results}


# --------------------------------------------------------------------------- #
#  Same answers as the path it replaces                                        #
# --------------------------------------------------------------------------- #
def test_it_finds_what_the_quadrat_major_path_finds(survey):
    photos = sorted(survey["photos"].glob("*.png"))
    table = _seed_csv(survey)
    kw = dict(quadrat_gsd_m=0.001, search_radius_m=1.0)

    old = _by_photo(S.check_quadrats(photos, survey["ortho"], table, None, **kw))
    new = _by_photo(S.check_survey(photos, survey["ortho"], table, None, **kw))

    assert set(old) == set(new), "a photo went missing"
    assert old["q_good.png"].status == "located", old["q_good.png"].detail
    assert new["q_good.png"].status == "located", new["q_good.png"].detail


def test_the_placement_agrees_to_a_few_millimetres(survey):
    """The grid moved; the answer must not. Anything worse than this and the
    84% saving is being paid for in accuracy."""
    photos = [survey["photos"] / "q_good.png"]
    table = _seed_csv(survey)
    kw = dict(quadrat_gsd_m=0.001, search_radius_m=1.0)

    old = S.check_quadrats(photos, survey["ortho"], table, None, **kw)[0]
    new = S.check_survey(photos, survey["ortho"], table, None, **kw)[0]
    assert old.matrix is not None and new.matrix is not None

    from PIL import Image
    with Image.open(str(photos[0])) as im:
        w, h = im.size

    def centre(c):
        M = np.asarray(c.matrix, float)
        p = M @ np.array([w / 2.0, h / 2.0, 1.0])
        return float(p[0]), float(p[1])

    ax, ay = centre(old)
    bx, by = centre(new)
    assert np.hypot(ax - bx, ay - by) < 0.005, "placements differ by over 5 mm"


def test_a_photo_with_no_seed_is_still_reported_not_dropped(survey):
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_survey(photos, survey["ortho"], None, None,
                         quadrat_gsd_m=0.001)
    assert len(res) == len(photos), "every photo must be reported"
    assert {r.status for r in res} == {"no_seed"}


def test_the_seed_source_survives(survey):
    """The batch's own table shows where each seed came from; losing it in the
    rewrite would make the results unreadable."""
    photos = [survey["photos"] / "q_good.png"]
    res = S.check_survey(photos, survey["ortho"], _seed_csv(survey), None,
                         quadrat_gsd_m=0.001, search_radius_m=1.0)
    assert res[0].seed_source == "list"


# --------------------------------------------------------------------------- #
#  Things the old path could not do                                            #
# --------------------------------------------------------------------------- #
def test_progress_is_reported_as_it_goes(survey):
    """The shipped batch awaits one call over the whole list and renders once
    at the end, so a 90-minute run showed a spinner and nothing else."""
    seen = []
    photos = sorted(survey["photos"].glob("*.png"))
    S.check_survey(photos, survey["ortho"], _seed_csv(survey), None,
                   quadrat_gsd_m=0.001, search_radius_m=1.0,
                   progress_fn=lambda d, t, m: seen.append((d, t, m)))
    assert seen, "no progress was reported at all"
    assert all(t == len(photos) for _, t, _ in seen)
    assert [d for d, _, _ in seen] == sorted(d for d, _, _ in seen)
    assert seen[-1][0] == len(photos), "the last report is not complete"
    assert any("q_good" in m for _, _, m in seen)
    # the two outcomes must not share a substring: "located" / "not located"
    # read the same at a glance in a one-line progress label
    assert not any(m.startswith("located") and "no match" in m for _, _, m in seen)


def test_a_run_can_be_stopped_and_keeps_what_it_found(survey):
    """Three hours of work must not be lost to a change of mind."""
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_survey(photos, survey["ortho"], _seed_csv(survey), None,
                         quadrat_gsd_m=0.001, search_radius_m=1.0,
                         should_stop=lambda: True)
    assert len(res) == len(photos), "stopping must still report every photo"
    assert all(r.status in ("located", "not_located", "no_seed", "stopped",
                            "already_placed", "unreadable") for r in res)


def test_a_stop_that_never_fires_changes_nothing(survey):
    photos = [survey["photos"] / "q_good.png"]
    table = _seed_csv(survey)
    kw = dict(quadrat_gsd_m=0.001, search_radius_m=1.0)
    a = S.check_survey(photos, survey["ortho"], table, None, **kw)[0]
    b = S.check_survey(photos, survey["ortho"], table, None,
                       should_stop=lambda: False, **kw)[0]
    assert a.status == b.status


def test_each_tile_is_read_once_however_many_quadrats_want_it(survey):
    """The point of the rewrite. Counts the reads rather than trusting that
    grouping happened."""
    photos = sorted(survey["photos"].glob("*.png"))
    table = _seed_csv(survey)

    real = S._window_reader
    calls = {"n": 0, "rects": set()}

    def counting(path):
        read, shape = real(path)

        def wrapped(r0, c0, r1, c1):
            calls["n"] += 1
            calls["rects"].add((r0, c0, r1, c1))
            return read(r0, c0, r1, c1)
        return wrapped, shape

    S._window_reader = counting
    try:
        S.check_survey(photos, survey["ortho"], table, None,
                       quadrat_gsd_m=0.001, search_radius_m=1.0)
    finally:
        S._window_reader = real
    assert calls["n"] == len(calls["rects"]), (
        f"{calls['n']} reads for {len(calls['rects'])} distinct tiles — "
        "a tile was read more than once")


# --------------------------------------------------------------------------- #
#  The search stays inside the radius the user asked for                       #
# --------------------------------------------------------------------------- #
def test_the_batch_does_not_widen_past_the_radius(survey, monkeypatch):
    """plan_search DOUBLES the radius when it finds nothing, until it covers
    the whole ortho. The single-quadrat path caps that (gui passes
    max_radius_m); the batch did not, so on the Swimbeach survey one unfindable
    quadrat cost 3268 tiles instead of 81 -- about 76 minutes instead of two,
    and most of that survey is unfindable."""
    from functions import georef as G

    seen = []
    real = G.plan_search

    def spy(*a, **kw):
        seen.append(kw.get("max_radius_m"))
        return real(*a, **kw)

    monkeypatch.setattr(G, "plan_search", spy)
    photos = [survey["photos"] / "q_alien.png"]      # deliberately unfindable
    csvp = survey["dir"] / "seeds.csv"
    sx, sy = survey["seed"]
    csvp.write_text(f"photo,x,y,crs\nq_alien,{sx},{sy},EPSG:2154\n",
                    encoding="utf-8")
    S.check_quadrats(photos, survey["ortho"], S.load_seed_table(csvp), None,
                     quadrat_gsd_m=0.001, search_radius_m=1.0)
    assert seen, "plan_search was never called"
    assert all(c is not None for c in seen), \
        "the batch still lets the search widen without limit"
    assert all(c == 1.0 for c in seen)


def test_check_survey_never_widens_either(survey):
    """The tile-major path bounds itself by construction -- tiles_for_seed
    takes a radius and returns the tiles reaching it, with no doubling."""
    from functions import georef as G
    tight = G.tiles_for_seed((4000, 4000), 0.004, 1.2, (2000, 2000), 2.0)
    wide = G.tiles_for_seed((4000, 4000), 0.004, 1.2, (2000, 2000), 20.0)
    assert len(tight) < len(wide), "radius has no effect on the tile set"
    # and a small radius must not quietly reach the far corner
    assert not any(t[0] == 0 and t[1] == 0 for t in tight)


# --------------------------------------------------------------------------- #
#  The whole-ortho sweep: only when asked, and shared when it runs             #
# --------------------------------------------------------------------------- #
def test_an_unseeded_quadrat_is_not_swept_unless_asked(survey):
    """Sweeping the whole ortho for one quadrat is the most expensive thing
    this tool can do -- 3268 tiles on the Swimbeach ortho, about 76 minutes.
    It must never start on its own."""
    photos = sorted(survey["photos"].glob("*.png"))
    reads = {"n": 0}
    real = S._window_reader

    def counting(path):
        read, shape = real(path)

        def wrapped(*a):
            reads["n"] += 1
            return read(*a)
        return wrapped, shape

    S._window_reader = counting
    try:
        res = S.check_survey(photos, survey["ortho"], None, None,
                             quadrat_gsd_m=0.001)
    finally:
        S._window_reader = real
    assert {r.status for r in res} == {"no_seed"}
    assert reads["n"] == 0, "the ortho was read for a quadrat with no seed"
    assert any("whole-ortho search" in r.detail for r in res), \
        "the refusal does not tell the user the sweep exists"


def test_asking_for_it_actually_sweeps(survey):
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_survey(photos, survey["ortho"], None, None,
                         quadrat_gsd_m=0.001, full_grid=True)
    assert len(res) == len(photos)
    assert "no_seed" not in {r.status for r in res}, \
        "a photo was still refused although a sweep was asked for"
    assert any(r.seed_source == "grid" for r in res)


def test_the_sweep_finds_the_quadrat_without_any_seed(survey):
    """The point of paying for it."""
    res = S.check_survey([survey["photos"] / "q_good.png"], survey["ortho"],
                         None, None, quadrat_gsd_m=0.001, full_grid=True)
    assert res[0].status == "located", res[0].detail
    assert res[0].seed_source == "grid"


def test_every_unseeded_quadrat_shares_each_tile_read(survey):
    """For a grid search over several unseeded quadrats, they
    are all run against each tile. If they were swept one after another the
    ortho would be read once PER QUADRAT."""
    photos = sorted(survey["photos"].glob("*.png"))
    assert len(photos) >= 2, "need at least two quadrats to share anything"

    real = S._window_reader
    calls = {"n": 0, "rects": []}

    def counting(path):
        read, shape = real(path)

        def wrapped(r0, c0, r1, c1):
            calls["n"] += 1
            calls["rects"].append((r0, c0, r1, c1))
            return read(r0, c0, r1, c1)
        return wrapped, shape

    S._window_reader = counting
    try:
        S.check_survey(photos, survey["ortho"], None, None,
                       quadrat_gsd_m=0.001, full_grid=True)
    finally:
        S._window_reader = real

    assert calls["n"] == len(set(calls["rects"])), (
        f"{calls['n']} reads for {len(set(calls['rects']))} distinct tiles — "
        "the unseeded quadrats were swept separately instead of together")


def test_a_seeded_and_an_unseeded_quadrat_can_run_in_one_pass(survey):
    """A real survey is usually mixed. The seeded ones must still stop early
    at their own tile while the unseeded ones walk the grid."""
    photos = sorted(survey["photos"].glob("*.png"))
    csvp = survey["dir"] / "mixed.csv"
    sx, sy = survey["seed"]
    csvp.write_text(f"photo,x,y,crs\nq_good,{sx},{sy},EPSG:2154\n",
                    encoding="utf-8")
    res = _by_photo(S.check_survey(photos, survey["ortho"],
                                   S.load_seed_table(csvp), None,
                                   quadrat_gsd_m=0.001, search_radius_m=1.0,
                                   full_grid=True))
    assert res["q_good.png"].seed_source == "list"
    assert res["q_alien.png"].seed_source == "grid"


def test_a_sweep_can_be_stopped_too(survey):
    photos = sorted(survey["photos"].glob("*.png"))
    res = S.check_survey(photos, survey["ortho"], None, None,
                         quadrat_gsd_m=0.001, full_grid=True,
                         should_stop=lambda: True)
    assert len(res) == len(photos)


# --------------------------------------------------------------------------- #
#  Each quadrat is measured at its OWN scale                                   #
# --------------------------------------------------------------------------- #
def test_a_quadrats_own_gsd_beats_the_one_typed_in_the_tab(tmp_path):
    """A survey rectified with automatic GSD has a different scale per photo.
    One real survey's files span 0.000527 to 0.000694 m -- 1.32x --
    while scale_tolerance is 0.15, so no single typed value can fit them all
    and the extremes are refused for arithmetic rather than for evidence."""
    from PIL import Image
    import numpy as np
    p = tmp_path / "S8_IMG_6762_rectified_GSD=0.000527m.png"
    Image.fromarray(np.zeros((16, 16, 3), "uint8")).save(str(p))
    assert S._quadrat_gsd(p, 0.0005) == pytest.approx(0.000527)


def test_the_typed_value_is_used_when_the_file_says_nothing(tmp_path):
    from PIL import Image
    import numpy as np
    p = tmp_path / "plain.png"
    Image.fromarray(np.zeros((16, 16, 3), "uint8")).save(str(p))
    assert S._quadrat_gsd(p, 0.0007) == pytest.approx(0.0007)


def test_there_is_always_a_scale_to_fall_back_on(tmp_path):
    from PIL import Image
    import numpy as np
    p = tmp_path / "plain2.png"
    Image.fromarray(np.zeros((16, 16, 3), "uint8")).save(str(p))
    assert S._quadrat_gsd(p, None) > 0
    assert S._quadrat_gsd(p, 0) > 0


def test_a_quadrat_the_run_never_reached_says_so(survey):
    """Stop marked every unchecked quadrat "not_located", which is what a
    quadrat the matcher rejected is called: the summary then read "7 of 7
    checked" and blamed the matcher for one it never looked at."""
    photos = sorted(survey["photos"].glob("*.png"))
    table = _seed_csv(survey)
    res = S.check_survey(photos, survey["ortho"], table, None,
                         quadrat_gsd_m=0.001, search_radius_m=1.0,
                         should_stop=lambda: True)
    assert len(res) == len(photos), "a stopped survey still reports every photograph"
    stopped = [r for r in res if r.status == "stopped"]
    assert stopped, "the quadrats the rounds never reached say so"
    assert all("Stopped before" in r.detail for r in stopped)
    # A photograph with no seed is refused before the rounds start, and keeps
    # its own reason; nothing the run skipped is called "not located".
    assert "not_located" not in {r.status for r in res}
    # ... and the same survey without the stop does not use that status.
    ok = S.check_survey(photos, survey["ortho"], table, None,
                        quadrat_gsd_m=0.001, search_radius_m=1.0)
    assert "stopped" not in {r.status for r in ok}

