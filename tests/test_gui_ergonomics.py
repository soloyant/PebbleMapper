"""Layout guarantees for the Zonal, Validate and Detection tabs.

These assert structural decisions against `gui/app.py`'s source. They are
deliberately source-level: the module pulls in TensorFlow and GDAL and serves
a 16,000-line single-page app, so standing the whole UI up per assertion
costs more than it proves. What they guard is exactly what regressed before
— a second mode chooser creeping back
in, a section drifting below the run queue, or a collapsible being nested
inside another.

The rendered result was checked by driving the running app.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture(scope="module")
def src() -> str:
    return APP.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Zonal — one mode chooser, three sub-tabs                                    #
# --------------------------------------------------------------------------- #
def test_zonal_has_polygon_transect_and_profile_subtabs(src):
    """Polygons and transects share no processing and produce incompatible
    outputs, so they are separate sub-tabs rather than one long page."""
    for name in ("polygons", "transects", "profile"):
        assert re.search(rf'ui\.tab\(\s*"{name}"', src), f"missing sub-tab {name}"


def test_zonal_asks_the_mode_question_exactly_once(src):
    """The same polygons/transects choice used to be asked by two toggles
    bound to the same state, 27% of the tab apart."""
    toggles = re.findall(r'ui\.toggle\(\s*(?:\[|\{)\s*["\']polygons["\']', src)
    assert toggles == [], (
        f"{len(toggles)} polygons/transects toggle(s) remain; the sub-tab bar "
        "is the only control that may set this")


def test_zonal_view_state_exists_and_defaults_to_polygons(src):
    assert re.search(r'zonal_view:\s*str\s*=\s*"polygons"', src)


def test_zonal_cards_are_bound_to_the_subtab(src):
    """Canvas, analysis, run row and results follow the sub-tab; the profile
    composer is its own view so nothing trails the run queue."""
    for handle in ("_canvas_card", "_analysis_card", "_run_row",
                   "_results_holder"):
        block = src.split(f"{handle}.bind_visibility_from(", 1)
        assert len(block) == 2, f"{handle} is not visibility-bound"
        assert '"zonal_view"' in block[1][:120], f"{handle} not bound to zonal_view"
    assert re.search(
        r'_profile_card\.bind_visibility_from\(\s*state,\s*"zonal_view",\s*value="profile"',
        src), "profile card must be its own sub-tab"


def test_zonal_intro_does_not_name_a_removed_feature(src):
    """The tab opened by describing a "Combined plot composer" that no longer
    exists, and claiming both modes fed it. The bottom section takes transect
    CSVs only."""
    assert "Combined plot" not in src


# --------------------------------------------------------------------------- #
#  Validate — working a list of quadrats                                       #
# --------------------------------------------------------------------------- #
def test_validate_has_a_quadrat_navigator(src):
    """Digitize already solved list-work; Validate had no equivalent."""
    for attr in ("val_truth_dir", "val_truth_files", "val_truth_index"):
        assert re.search(rf"\b{attr}\b", src), f"missing state field {attr}"
    assert 'label="Quadrat"' in src
    assert '"Previous", icon="chevron_left"' in src
    assert '"Next", icon="chevron_right"' in src


def test_validate_navigator_disables_rather_than_hides(src):
    """A control that vanishes on a single-quadrat project reads as a bug."""
    assert "prev_btn.set_enabled(" in src
    assert "next_btn.set_enabled(" in src


def test_validate_names_the_batch_route_before_the_configuration(src):
    """The batch controls existed but sat at the foot of the tab, so nothing
    suggested a folder could be worked through at once."""
    # The label is wrapped across source lines, so match the tail of it.
    hint = src.find("from folder…* at the foot of this tab")
    truth_input = src.find('"Truth CSV (manual measurements)"')
    assert hint != -1, "the navigator must name the batch route"
    assert hint < truth_input, (
        "the batch route must be mentioned before the per-quadrat inputs")


# --------------------------------------------------------------------------- #
#  No collapsible inside a collapsible, anywhere                               #
# --------------------------------------------------------------------------- #
def test_no_expansion_is_nested_inside_another_expansion(src):
    """Detection's ROI definition sat inside Advanced parameters — the only
    such case in the application, and it cost two clicks to reach."""
    lines = src.splitlines()
    open_expansions: list[int] = []   # indentation columns
    offenders: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # A `with` block at or left of a recorded expansion closes it.
        while open_expansions and indent <= open_expansions[-1]:
            open_expansions.pop()
        if re.search(r"\bwith ui\.expansion\(", line):
            if open_expansions:
                offenders.append(f"line {i + 1}: {line.strip()[:70]}")
            open_expansions.append(indent)
    assert not offenders, "nested collapsible(s):\n" + "\n".join(offenders)
