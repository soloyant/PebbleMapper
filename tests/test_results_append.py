"""Map, Rasterize, Express and the Zonal profile append their results below
the action that produced them, and the Map's styling sits next to its
preview.

Rendered-tree assertions in DOM order.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


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


def _label(e):
    lab = str((getattr(e, "_props", {}) or {}).get("label", "") or "")
    if not lab and e.__class__.__name__ in ("Checkbox", "Button", "Toggle"):
        lab = str(getattr(e, "text", "") or "")
    return lab


def _classes(e):
    return list(getattr(e, "_classes", []) or [])


def _kind(e):
    return e.__class__.__name__


def _first(els, pred):
    return next(i for i, e in enumerate(els) if pred(e))


def _texts(els):
    return [str(getattr(e, "content", "")) if _kind(e) == "Markdown"
            else str(getattr(e, "text", "")) for e in els
            if _kind(e) in ("Markdown", "Label")]


@pytest.fixture
def clean():
    from gui.app import state
    saved = {k: getattr(state, k) for k in ("current_project", "zonal_view")}
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
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def _build(clean, builder):
    from nicegui import ui
    with ui.column() as col:
        builder()
    clean[1].append(col)
    return _walk(col)


def test_express_log_and_stages_follow_run(clean):
    from gui.app import build_express_tab
    els = _build(clean, build_express_tab)
    i_run = _first(els, lambda e: _label(e) == "Run")
    i_bar = _first(els, lambda e: _kind(e) == "LinearProgress")
    i_log = _first(els, lambda e: _kind(e) == "Log")
    assert i_run < i_bar < i_log, f"run {i_run} bar {i_bar} log {i_log}"
    assert not any(len(t.split()) > 30 for t in _texts(els))


def test_rasterize_queue_follows_add_and_summary_follows_the_cards(clean):
    from gui.app import build_rasterize_tab
    els = _build(clean, build_rasterize_tab)
    i_add = _first(els, lambda e: _label(e) == "Add to queue")
    i_queue = _first(els, lambda e: "pm-ras-queue" in _classes(e))
    i_cell = _first(els, lambda e: _label(e).startswith("Cell size"))
    assert i_cell < i_add < i_queue, f"cell {i_cell} add {i_add} queue {i_queue}"
    i_cards = _first(els, lambda e: "pm-ras-cards" in _classes(e))
    i_sum = _first(els, lambda e: "pm-ras-summary" in _classes(e))
    assert i_cards < i_sum
    assert not any(len(t.split()) > 30 for t in _texts(els))


def test_map_preview_follows_the_action_and_styling_precedes_it(clean):
    from gui.app import build_map_tab
    els = _build(clean, build_map_tab)
    i_dpi = _first(els, lambda e: _label(e) == "DPI")
    i_basemap = _first(els, lambda e: _label(e) == "Basemap")
    i_vmin = _first(els, lambda e: _label(e) == "vmin")
    i_unit = _first(els, lambda e: _label(e).startswith("Size unit"))
    i_preview_btn = _first(els, lambda e: _label(e) == "Preview")
    i_plot = _first(els, lambda e: "pm-map-preview" in _classes(e))
    i_log = _first(els, lambda e: _kind(e) == "Log")
    assert i_dpi < i_basemap < i_vmin < i_unit < i_preview_btn < i_plot < i_log, (
        f"dpi {i_dpi} basemap {i_basemap} vmin {i_vmin} unit {i_unit} "
        f"preview {i_preview_btn} plot {i_plot} log {i_log}")
    # Nothing but the action bar sits between the last styling input and the preview.
    between = [e for e in els[i_unit + 1:i_plot]
               if _kind(e) in ("Number", "Input", "Select", "Checkbox", "Toggle")]
    assert not between, [(_kind(e), _label(e)) for e in between]
    assert not any(len(t.split()) > 30 for t in _texts(els))


def test_profile_figure_and_tables_follow_render(clean):
    from gui.app import build_zonal_tab
    state, _ = clean
    state.zonal_view = "profile"
    els = _build(clean, build_zonal_tab)
    i_btn = _first(els, lambda e: _label(e).startswith("Render profile"))
    i_prev = _first(els, lambda e: "pm-profile-preview" in _classes(e))
    assert i_btn < i_prev


def test_profile_tables_reach_the_page(src):
    """'Profile figure and 3 tables saved' -- the tables are now shown."""
    body = src[src.index("def build_zonal_tab("):]
    body = body[:body.index("\ndef ")]
    assert "_display_profile_tables(res)" in body
    fn = body[body.index("def _display_profile_tables"):][:1800]
    assert "ui.table(" in fn and '"summary", "change", "binned"' in fn
