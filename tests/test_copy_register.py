"""The copy register.

No control group carries more than one line of on-page prose; the Overview
describes the twelve tabs once and stays under 120 words. Gate messages
name the field and its band. The former paragraphs live in the tab sections
of docs/user-manual.md and README section 6.1, so nothing was lost -- it was moved.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"
TABS = ["build_overview_tab", "build_express_tab", "build_orthorectify_tab",
        "build_detection_tab", "build_merge_tab", "build_rasterize_tab",
        "build_map_tab", "build_zonal_tab", "build_georeference_tab",
        "build_digitize_tab", "build_validate_tab", "build_report_tab"]


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def _texts(els):
    out = []
    for e in els:
        k = e.__class__.__name__
        t = getattr(e, "content", "") if k == "Markdown" else (
            getattr(e, "text", "") if k == "Label" else "")
        if isinstance(t, str) and t.strip():
            out.append(t)
    return out


@pytest.fixture
def clean():
    from gui.app import state
    saved = state.current_project
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    state.current_project = ""
    built = []
    yield built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    state.current_project = saved


@pytest.mark.parametrize("name", TABS)
def test_no_control_group_carries_a_paragraph(clean, name):
    """Baselines before the rewrite: Overview 929 words, Georeference 287,
    Zonal 198, Validate 185, Orthorectify 178, Rasterize 131, Map 107,
    Digitize 79."""
    from nicegui import ui
    from gui import app as A
    with ui.column() as col:
        getattr(A, name)()
    clean.append(col)
    texts = _texts(_walk(col))
    long = [t for t in texts if len(t.split()) > 30]
    assert not long, f"{name}: {[t[:70] for t in long]}"
    assert not any(re.search(r"\bStep \d", t) for t in texts), name
    if name == "build_overview_tab":
        total = sum(len(t.split()) for t in texts)
        # 11 tab lines + root + project + two captions (<= 120).
        assert total <= 120, f"Overview carries {total} words (baseline 929)"
        joined = " ".join(texts)
        assert joined.count("Express") <= 3, "the tabs are described more than once"


def test_the_displaced_prose_was_moved_not_lost():
    notes = (APP.parents[1] / "docs" / "user-manual.md").read_text(encoding="utf-8")
    for anchor in ("Tile overlap", "Dedup IoU", "What to expect",
                   "Coordinate frame", "Rationale"):
        assert anchor in notes, f"{anchor!r} missing from docs/user-manual.md"
    readme = (APP.parents[1] / "README.md").read_text(encoding="utf-8")
    for tab in ("Overview", "Express", "Orthorectify", "Detect", "Merge",
                "Rasterize", "Map", "Zonal", "Digitize", "Validate", "Report"):
        assert tab in readme, f"README no longer describes the {tab} tab"


def test_gate_messages_name_field_and_band(src):
    """Before the rewrite, 23 of 163 gate messages named field + location. The
    rewritten gates say where to go; the plain ones left are data facts
    ('No clasts detected'), never 'do X first' without a place."""
    gone = ('"Select an image directory first."', '"Pick a valid Truth CSV first."',
            '"Pick a valid Detection CSV first."', '"Set a Detection CSV first."',
            '"Pick an output CSV path first."', '"Load the source image first."',
            '"Pick a source image first."', '"Select a project first."',
            '"Pick a project first."', '"Run the survey check first."',
            '"Check for a match first"', '"Choose a quadrat photograph first"',
            '"Add at least one transect CSV first."',
            '"Select a project and a UAV ortho first."',
            '"The list above says why."', "Set 'Frame thickness (m)' first")
    for g in gone:
        assert g not in src, f"unplaced gate still present: {g}"
    # Any remaining "... first." gate must say where.
    bad = []
    for m in re.finditer(r'ui\.notify\(\s*(f?"[^"]*first[^"]*")', src):
        frag = m.group(1)
        if not re.search(r"\((Inputs|Work surface|Results|Selection|strip|Project|left panel|above|below|this panel|Active project|toolbar|canvas|Map content)", frag) \
                and "above" not in frag and "below" not in frag \
                and not re.search(r"\b[A-Z][a-z]+ tab\b", frag):   # a named tab is a place
            bad.append(frag[:80])
    assert not bad, bad
