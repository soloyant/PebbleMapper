"""Orthorectify corner save/load — retro-compatible <stem>_corners.txt."""
import pytest

pytest.importorskip("nicegui")


def test_ortho_corner_roundtrip(tmp_path):
    from gui.app import (_load_ortho_corners, _save_ortho_corners,
                         _ortho_corners_path)
    src = tmp_path / "01.png"
    src.write_bytes(b"x")  # helpers only use the path stem
    corners = [[601.9, 1181.7], [2266.8, 1008.9],
               [2405.0, 2737.1], [704.4, 2845.4]]
    p = _save_ortho_corners(str(src), corners)
    assert p and _ortho_corners_path(str(src)).name == "01_corners.txt"
    loaded = _load_ortho_corners(str(src))
    assert len(loaded) == 4
    for (a, b), (c, d) in zip(corners, loaded):
        assert abs(a - c) < 1e-6 and abs(b - d) < 1e-6


def test_ortho_corner_reads_existing_format(tmp_path):
    """Parse the existing 'x,y' per-line corner files verbatim."""
    from gui.app import _load_ortho_corners
    (tmp_path / "02_corners.txt").write_text(
        "601.936027136567,1181.7555999209\n"
        "2266.82817397125,1008.92944041781\n"
        "2405.08916671209,2737.19104754766\n"
        "704.479707554039,2845.4954510138\n", encoding="utf-8")
    pts = _load_ortho_corners(str(tmp_path / "02.png"))
    assert len(pts) == 4
    assert abs(pts[0][0] - 601.936027136567) < 1e-6
    assert abs(pts[2][1] - 2737.19104754766) < 1e-6


def test_ortho_corner_missing_file(tmp_path):
    from gui.app import _load_ortho_corners
    assert _load_ortho_corners(str(tmp_path / "nope.png")) == []


def test_corners_are_written_from_clicks_or_marked_as_suggested():
    """The sidecar means 'a person put these here'.

    An earlier version of this idea failed on exactly this point
    -- *"one drag would have laundered three guesses into evidence"*. That
    remains forbidden. What is now allowed is the one deliberate path where a
    person looks at a detected frame and presses Accept: a human decision, not
    an accident, but still not a hand-placed corner. So the rule is not "only
    clicks" any more, it is **either clicks, or provenance recorded**.

    Asserted on the source because the property is about which call sites
    exist, and a behavioural test can only sample the paths it thinks of.
    """
    import inspect
    import re
    from gui import app as A

    src = inspect.getsource(A)

    def _args_at(start):
        """The argument text of the call whose '(' is at `start`.

        A regex cannot do this: `[^)]*` stops inside `_effective_segments()`
        and silently reports a shorter argument list than was written, which
        is how this guard first read a marked call site as unmarked.
        """
        depth, i = 0, start
        while i < len(src):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    return src[start + 1:i]
            i += 1
        return src[start + 1:]

    writers = [_args_at(m.end() - 1) for m in
               re.finditer(r"(?<!def )_save_ortho_corners\(", src)]
    assert writers, "no call sites found -- the guard has gone stale"
    for c in writers:
        one = " ".join(c.split())
        assert "ortho_guide" not in c, \
            "a suggestion reaches a corner file directly, unmarked: %s" % one
        assert "state.ortho_corners" in c or "source=" in c, \
            ("a corner file is written from neither the user's clicks nor a "
             "marked source: %s" % one)


def test_an_accepted_frame_is_recorded_as_suggested(tmp_path):
    """The marker has to survive the round trip, or it is decoration.

    A quadrat measured to a fraction of a per cent and one accepted at a few
    per cent must not read identically a year later.
    """
    from gui.app import (_save_ortho_corners, _load_ortho_corners,
                         _load_ortho_segments, _load_ortho_corner_source)
    src = tmp_path / "07.png"
    src.write_bytes(b"x")
    corners = [[10.0, 10.0], [90.0, 12.0], [88.0, 91.0], [11.0, 89.0]]

    _save_ortho_corners(str(src), corners, [1.0, 1.0, 1.0, 1.0], 0.054,
                        source="suggested")
    assert _load_ortho_corner_source(str(src)) == "suggested"
    # The annotation must not damage what shares the file with it.
    assert _load_ortho_segments(str(src))[0] == pytest.approx(1.0)
    assert len(_load_ortho_corners(str(src))) == 4

    # And a hand-placed set stays unmarked -- absence is the ordinary case.
    plain = tmp_path / "08.png"
    plain.write_bytes(b"x")
    _save_ortho_corners(str(plain), corners, [1.0, 1.0, 1.0, 1.0], 0.054)
    assert _load_ortho_corner_source(str(plain)) is None
    assert _load_ortho_segments(str(plain))[0] == pytest.approx(1.0)
