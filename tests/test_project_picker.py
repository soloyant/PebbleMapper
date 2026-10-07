"""The project picker lives in the left drawer, once per page. Tabs no longer
draw a project strip; they register a hook that the drawer's picker fires
when the project or the date changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


@pytest.fixture
def clean():
    from gui.app import state, _PROJECT_HOOKS
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
    _PROJECT_HOOKS.clear()
    state.current_project = ""
    built = []
    yield built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    _PROJECT_HOOKS.clear()
    state.current_project = saved


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


def test_the_drawer_holds_the_one_picker_and_no_tab_draws_a_strip():
    src = APP.read_text(encoding="utf-8")
    drawer = src[src.index("with ui.left_drawer()"):][:400]
    assert "render_project_picker()" in drawer
    strip = src[src.index("def render_project_strip"):src.index("def _fire_project_hooks")]
    assert "ui.row" not in strip and "ui.select" not in strip, (
        "the per-tab strip is drawn again")
    # Gates that used to point at the strip point at the drawer.
    assert "(strip, above)" not in src
    assert src.count("(Project, left panel)") >= 5


def test_a_tab_registers_a_hook_and_the_picker_fires_it(clean):
    from nicegui import ui
    from gui import app as A
    fired = []
    with ui.column() as col:
        A.render_project_picker()
        A.render_project_strip(on_change=lambda: fired.append("tab"))
    clean.append(col)
    els = _walk(col)
    assert not any(getattr(e, "text", "") == "Project:" for e in els), (
        "the strip row is drawn inside the tab")
    sel = next(e for e in els if e.__class__.__name__ == "Select")
    assert "" in sel.options and sel.options[""] == "— no project —"
    # The select is bound to the state: a project set anywhere (Overview,
    # the drawer) reaches it and fires the registered hook.
    A.state.current_project = ""
    assert fired == []
    # A value the select does not offer would be reset (a second change);
    # offer it first, as a real project would be.
    sel.options["Some_project"] = "Some_project"
    sel.update()
    sel.value = "Some_project"
    assert fired == ["tab"], fired
    assert A.state.current_project == "Some_project"


def test_a_failing_hook_does_not_stop_the_others(clean):
    from nicegui import ui
    from gui import app as A
    fired = []
    with ui.column() as col:
        A.render_project_strip(on_change=lambda: 1 / 0)
        A.render_project_strip(on_change=lambda: fired.append("second"))
    clean.append(col)
    A._fire_project_hooks()
    assert fired == ["second"]


def _press(btn):
    from nicegui.events import ClickEventArguments, handle_event
    for l in btn._event_listeners.values():
        if l.type == "click":
            handle_event(l.handler, ClickEventArguments(sender=btn, client=btn.client))


def _drawer_select(col, label=None):
    return next(e for e in _walk(col) if e.__class__.__name__ == "Select"
                and (e._props.get("label") == label) and "" in (e.options or {}))


def test_a_created_project_becomes_active_everywhere(clean, tmp_path):
    """Manual 2.3 step 3: Create makes the folders and the project becomes
    active. The drawer's picker used to be a select two-way bound to the
    state whose options lacked the new name, so it unset the project."""
    import functions.layout as layout
    from nicegui import ui
    from gui import app as A
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        A._PROJECT_PICKERS.clear()
        A.state.current_project = ""
        A.state.current_date = ""
        with ui.column() as col:
            A.render_project_picker()
            A.build_overview_tab()
        clean.append(col)
        els = _walk(col)
        name = next(e for e in els if e.__class__.__name__ == "Input"
                    and e._props.get("label") == "New project")
        name.value = "fresh"
        _press(next(e for e in els if e.__class__.__name__ == "Button"
                    and str(getattr(e, "text", "")) == "Create"))
        assert (tmp_path / "fresh" / "validation" / "orthorectified").is_dir()
        assert A.state.current_project == "fresh", "the new project is active"
        # The drawer's picker was rebuilt with the new name and shows it.
        sel = _drawer_select(col)
        assert "fresh" in sel.options and sel.value == "fresh"
        assert any(str(getattr(e, "text", "")).endswith("fresh")
                   for e in _walk(col) if e.__class__.__name__ == "Label")
        # A date added to it becomes the active date, and the drawer offers it.
        els = _walk(col)
        date = next(e for e in els if e.__class__.__name__ == "Input"
                    and e._props.get("label") == "Add date")
        date.value = "2026-09-21"
        _press(next(e for e in els if e.__class__.__name__ == "Button"
                    and str(getattr(e, "text", "")) == "Add date"))
        assert (tmp_path / "fresh" / "2026-09-21" / "input_data").is_dir()
        assert A.state.current_project == "fresh"
        assert A.state.current_date == "2026-09-21"
        dsel = next(e for e in _walk(col) if e.__class__.__name__ == "Select"
                    and e._props.get("label") == "Date")
        assert dsel.value == "2026-09-21" and "2026-09-21" in list(dsel.options)
        # Picking a project from the drawer sets the state (no binding needed).
        sel = _drawer_select(col)
        sel.value = ""
        assert A.state.current_project == ""
    finally:
        A.state.current_project = ""
        A.state.current_date = ""
        layout.set_datasets_root(original, persist=False)
