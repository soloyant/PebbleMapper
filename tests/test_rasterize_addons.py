"""A project's add-on fields appear in Rasterize when the project is picked
."""
from __future__ import annotations

import json

import pytest


def _walk(root):
    out = []

    def rec(e):
        out.append(e)
        for c in e.default_slot.children:
            rec(c)
    rec(root)
    return out


@pytest.fixture
def clean():
    from gui.app import state
    saved = {k: getattr(state, k) for k in ("current_project", "ras_csv", "ras_tif")}
    try:
        from nicegui import Client
        for e in list(Client.auto_index_client.content.default_slot.children):
            try:
                e.delete()
            except Exception:
                pass
    except Exception:
        pass
    built = []
    yield state, built
    for col in built:
        try:
            col.delete()
        except Exception:
            pass
    for k, v in saved.items():
        setattr(state, k, v)


def test_a_projects_addon_field_appears_when_the_project_is_picked(clean, tmp_path):
    import functions.layout as layout
    from nicegui import ui
    from gui.app import build_rasterize_tab, _fire_project_hooks
    state, built = clean
    original = layout.get_datasets_root()
    layout.set_datasets_root(tmp_path, persist=False)
    try:
        (tmp_path / "P").mkdir()
        (tmp_path / "P" / "addons.json").write_text(json.dumps([{
            "name": "Flatness", "inputs": ["Clast_width", "Clast_length"],
            "expression": "Clast_width / Clast_length", "units": "-"}]), encoding="utf-8")
        state.current_project = ""
        with ui.column() as col:
            build_rasterize_tab()
        built.append(col)
        names = lambda: [str(getattr(e, "text", "")) for e in _walk(col)
                         if e.__class__.__name__ == "Checkbox"]
        assert "Flatness" not in names(), "no project: the shipped fields only"
        state.current_project = "P"
        _fire_project_hooks()
        assert "Flatness" in names(), "the project's add-on field is listed"
        assert "Clast_length" in names()
    finally:
        layout.set_datasets_root(original, persist=False)
