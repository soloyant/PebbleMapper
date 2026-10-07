"""Rectifying a folder, and the guards that make that safe.

Rectification used to live in a 273-line GUI closure that read twenty `state`
fields and pushed strings into a widget, so nothing about it could be tested
and a batch could not exist. It is a function now.

Most of this file is about the refusals rather than the happy path. This write
path has already truncated a real `.tif` to a 229-byte husk while 494 tests
were green, and a batch is that failure times forty -- so the guards are
asserted by pointing the output at things it must never touch and checking the
bytes afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from functions import orthorectify as O          # noqa: E402


SQUARE = [[10.0, 10.0], [210.0, 10.0], [210.0, 210.0], [10.0, 210.0]]


def _photo(path, w=260, h=260, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    assert ok
    buf.tofile(str(path))
    return path


# --------------------------------------------------------------------------- #
#  One rectification, without a browser                                        #
# --------------------------------------------------------------------------- #
def test_it_rectifies_without_nicegui(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    out = tmp_path / "orthorectified" / "q_rectified.jpg"
    r = O.rectify_one(src, SQUARE, [1.185] * 4, out)
    assert r.ok, r.refusal
    assert Path(r.out_path).is_file()
    assert r.gsd_m > 0 and r.shape is not None
    assert "GSD=" in Path(r.out_path).name


def test_the_output_carries_the_record(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    r = O.rectify_one(src, SQUARE, [1.185] * 4,
                      tmp_path / "orthorectified" / "q_rectified.jpg",
                      frame_thickness_m=0.06)
    assert r.record["PM_SOURCE"] == "q.jpg"
    assert r.record["PM_SEGMENTS_M"].startswith("1.1850")
    assert r.record["PM_FRAME_THICKNESS_M"] == "0.0600"
    side = Path(r.out_path + ".json")
    assert side.is_file(), "no sidecar beside the output"


def test_what_the_tab_used_to_log_comes_back_as_data(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    r = O.rectify_one(src, SQUARE, [1.185] * 4,
                      tmp_path / "orthorectified" / "q_rectified.jpg")
    assert any("says it is" in n for n in r.notes)
    assert any("Wrote" in n for n in r.notes)


# --------------------------------------------------------------------------- #
#  The guards travel with the write                                            #
# --------------------------------------------------------------------------- #
def test_it_refuses_to_write_over_its_own_source(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    before = src.read_bytes()
    r = O.rectify_one(src, SQUARE, [1.185] * 4, src)
    assert not r.ok and "source photograph itself" in r.refusal
    assert src.read_bytes() == before, "the source photograph was written over"


def test_it_refuses_to_write_over_a_raw_photograph(tmp_path, monkeypatch):
    """A neighbour in the same folder, named by mistake. The photographs are
    field data and cannot be taken again."""
    src = _photo(tmp_path / "raw" / "q.jpg")
    victim = _photo(tmp_path / "raw" / "neighbour.jpg", seed=9)
    before = victim.read_bytes()
    monkeypatch.setattr("functions.exif_seed.looks_like_raw_photograph",
                        lambda p: Path(str(p)).name == "neighbour.jpg")
    r = O.rectify_one(src, SQUARE, [1.185] * 4, victim)
    assert not r.ok and "straight from a camera" in r.refusal
    assert victim.read_bytes() == before, "a raw photograph was written over"


def test_an_existing_output_is_kept_unless_overwrite_is_asked_for(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    out = tmp_path / "orthorectified" / "q_rectified.jpg"
    first = O.rectify_one(src, SQUARE, [1.185] * 4, out)
    assert first.ok
    stamp = Path(first.out_path).read_bytes()

    again = O.rectify_one(src, SQUARE, [1.185] * 4, out, overwrite=False)
    assert not again.ok and "already exists" in again.refusal
    assert Path(first.out_path).read_bytes() == stamp

    third = O.rectify_one(src, SQUARE, [1.185] * 4, out, overwrite=True)
    assert third.ok


def test_a_missing_source_is_a_refusal_not_a_crash(tmp_path):
    r = O.rectify_one(tmp_path / "gone.jpg", SQUARE, [1.185] * 4,
                      tmp_path / "out.jpg")
    assert not r.ok and "not there" in r.refusal


def test_three_corners_is_a_refusal(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    r = O.rectify_one(src, SQUARE[:3], [1.185] * 4, tmp_path / "out.jpg")
    assert not r.ok and "3/4" in r.refusal


# --------------------------------------------------------------------------- #
#  A wrong size warns; it does not refuse                                      #
# --------------------------------------------------------------------------- #
def test_the_untouched_default_is_a_warning_carried_with_the_result(tmp_path):
    """The number came from the user, and only they can say whether it is
    wrong -- so it is not a refusal. But it is returned rather than logged, so
    a batch can show it against the file it belongs to."""
    src = _photo(tmp_path / "raw" / "q.jpg")
    r = O.rectify_one(src, SQUARE, [1.0] * 4,
                      tmp_path / "orthorectified" / "q_rectified.jpg")
    assert r.ok
    assert any("default" in w for w in r.warnings)


def test_a_confirmed_default_does_not_warn(tmp_path):
    src = _photo(tmp_path / "raw" / "q.jpg")
    r = O.rectify_one(src, SQUARE, [1.0] * 4,
                      tmp_path / "orthorectified" / "q_rectified.jpg",
                      seg_confirmed=True)
    assert r.ok and not any("default" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
#  Readiness: what a batch is allowed to touch                                 #
# --------------------------------------------------------------------------- #
def test_a_file_still_at_the_default_is_not_ready():
    """The whole argument for a batch is that it does only work the user has
    already specified. Rectifying a folder at a size nobody chose is the lost
    survey, forty times."""
    ok, why = O.is_ready_to_rectify([[0, 0]] * 4, [1.0] * 4)
    assert not ok and "default" in why


def test_readiness_covers_the_other_ways_of_not_being_ready():
    assert O.is_ready_to_rectify([[0, 0]] * 4, [1.185] * 4)[0]
    assert O.is_ready_to_rectify([[0, 0]] * 4, [1.0] * 4, True)[0]
    assert not O.is_ready_to_rectify([[0, 0]] * 2, [1.185] * 4)[0]
    assert not O.is_ready_to_rectify([[0, 0]] * 4, None)[0]
    assert not O.is_ready_to_rectify([[0, 0]] * 4, [0.0] * 4)[0]


def test_the_plan_names_every_skip_and_its_reason():
    ready, skipped = O.plan_rectify_folder([
        ("a.jpg", [[0, 0]] * 4, [1.185] * 4, False, False),
        ("b.jpg", [[0, 0]] * 4, [1.0] * 4, False, False),
        ("c.jpg", [[0, 0]] * 2, [1.185] * 4, False, False),
        ("d.jpg", [[0, 0]] * 4, [1.185] * 4, False, True),
    ])
    assert ready == ["a.jpg"]
    reasons = dict(skipped)
    assert "default" in reasons["b.jpg"]
    assert "2/4" in reasons["c.jpg"]
    assert reasons["d.jpg"] == "already rectified"


# --------------------------------------------------------------------------- #
#  The folder run                                                              #
# --------------------------------------------------------------------------- #
def _job(tmp_path, name, segs=(1.185,) * 4):
    src = _photo(tmp_path / "raw" / name)
    return dict(photo=name, src=str(src), corners=SQUARE, segs=list(segs),
                out=str(tmp_path / "orthorectified"
                        / (Path(name).stem + "_rectified.jpg")))


def test_a_run_writes_one_output_per_ready_file(tmp_path):
    jobs = [_job(tmp_path, f"q{i}.jpg") for i in range(3)]
    items = O.rectify_folder(jobs, overwrite=True)
    assert [i.status for i in items] == ["rectified"] * 3
    written = list((tmp_path / "orthorectified").glob("*.jpg"))
    assert len(written) == 3


def test_progress_is_reported_per_file(tmp_path):
    """A ninety-minute run that showed only a spinner is on the record."""
    jobs = [_job(tmp_path, f"q{i}.jpg") for i in range(3)]
    seen = []
    O.rectify_folder(jobs, overwrite=True,
                     progress_fn=lambda d, t, m: seen.append((d, t, m)))
    assert [d for d, _, _ in seen] == [1, 2, 3]
    assert all(t == 3 for _, t, _ in seen)
    assert seen[-1][0] == 3


def test_a_stop_keeps_what_was_already_written(tmp_path):
    """Three hours of work must not be lost to a change of mind."""
    jobs = [_job(tmp_path, f"q{i}.jpg") for i in range(4)]
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2          # let the first two through
    items = O.rectify_folder(jobs, overwrite=True, should_stop=stop)
    assert len(items) == 4, "stopping must still report every file"
    assert [i.status for i in items[:2]] == ["rectified"] * 2
    assert all(i.status == "skipped" for i in items[2:])
    assert len(list((tmp_path / "orthorectified").glob("*.jpg"))) == 2


def test_one_bad_file_does_not_cost_the_others(tmp_path):
    jobs = [_job(tmp_path, "good1.jpg"),
            dict(photo="bad.jpg", src=str(tmp_path / "raw" / "missing.jpg"),
                 corners=SQUARE, segs=[1.185] * 4,
                 out=str(tmp_path / "orthorectified" / "bad_rectified.jpg")),
            _job(tmp_path, "good2.jpg")]
    items = O.rectify_folder(jobs, overwrite=True)
    assert [i.status for i in items] == ["rectified", "skipped", "rectified"]
    assert "not there" in items[1].reason


def test_the_run_is_written_down(tmp_path):
    """`state` does not survive a reconnect, and six months later "which of
    these forty did it write, and why is one missing" has to be answerable
    without rerunning anything."""
    jobs = [_job(tmp_path, "q0.jpg"), _job(tmp_path, "q1.jpg", segs=(1.0,) * 4)]
    items = O.rectify_folder(jobs, overwrite=True)
    text = O.batch_manifest(items, when="2026-08-21")
    assert "2 rectified" in text
    assert "q0.jpg" in text and "q1.jpg" in text
    # The warning travels with the file it belongs to.
    assert "default" in text


# --------------------------------------------------------------------------- #
#  Two photographs must never land on one file                                 #
# --------------------------------------------------------------------------- #
def test_two_sources_that_would_share_an_output_are_both_refused(tmp_path):
    """`01.jpg` and `01.tif` side by side both resolve to
    `orthorectified/01_rectified.jpg`, and the GSD tag does not separate them
    because it is the same quadrat at the same resolution. Writing both leaves
    one file, two "rectified" rows, and a manifest that says so confidently --
    the manifest whose whole purpose is to outlive the session."""
    for ext in (".jpg", ".tif"):
        _photo(tmp_path / "raw" / f"01{ext}")
    out = str(tmp_path / "orthorectified" / "01_rectified.jpg")
    jobs = [dict(photo=f"01{ext}", src=str(tmp_path / "raw" / f"01{ext}"),
                 corners=SQUARE, segs=[1.185] * 4, out=out)
            for ext in (".jpg", ".tif")]

    assert O.colliding_outputs(jobs), "the collision was not detected"
    items = O.rectify_folder(jobs, overwrite=True)
    assert [i.status for i in items] == ["skipped", "skipped"]
    assert all("same file as" in i.reason for i in items)
    assert not list((tmp_path / "orthorectified").glob("*.jpg")), (
        "a rectification was written despite the clash")


def test_distinct_outputs_are_not_mistaken_for_a_clash(tmp_path):
    jobs = [_job(tmp_path, "a.jpg"), _job(tmp_path, "b.jpg")]
    assert O.colliding_outputs(jobs) == {}
    assert [i.status for i in O.rectify_folder(jobs, overwrite=True)] == \
        ["rectified", "rectified"]


def test_a_write_that_produced_nothing_is_not_reported_as_success(tmp_path,
                                                                  monkeypatch):
    """The tab discarded cv2.imwrite's return and logged a tick on the next
    line, then told the user "the image is written and unchanged". Two
    untruths, no exception, and nothing for a folder run to record."""
    src = _photo(tmp_path / "raw" / "q.jpg")
    real = cv2.imencode

    def empty(ext, img, *a, **k):
        # Encoder says yes and hands back nothing -- a zero-byte file on disk,
        # which is the shape of the .tif truncation this project has already
        # suffered: the write "succeeded" and the file was a husk.
        return True, np.empty((0,), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imencode", empty)
    r = O.rectify_one(src, SQUARE, [1.185] * 4,
                      tmp_path / "orthorectified" / "q_rectified.jpg")
    monkeypatch.setattr(cv2, "imencode", real)
    assert not r.ok and "was not written" in r.refusal
    assert not any(
        p.stat().st_size for p in (tmp_path / "orthorectified").glob("*")
    ), "a husk was left behind and reported"
