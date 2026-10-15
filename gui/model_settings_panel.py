"""The model settings panel and the "N setting(s) changed" label (spec 032).

Built from the model's description alone (``detectors.settings``); values
are read from and written to the project's store, never bound to shared
page state (a select bound to shared state nulls it from other pages, 027).
The contract is ``specs/032-model-options/contracts/panel.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from nicegui import Client, context, ui

from detectors import settings as S

# (client, refresh) of every page's label, told when values change, so every
# page follows (two tabs on one project, 032 edge case); pages that went
# away are dropped.
_LISTENERS: List[tuple] = []


@dataclass
class Context:
    """What the panel acts on: the selected model, its backend, the project
    folder (None: no project) and the mode."""
    model_key: str
    display_name: str
    backend: object
    project_dir: Optional[Path]
    mode: str


def store_for(ctx: Context) -> S.ProjectSettings:
    return S.ProjectSettings(S.store_path(ctx.project_dir))


def description_for(ctx: Context) -> Optional[dict]:
    return S.load_description(ctx.backend) if ctx.backend is not None else None


def changed_values(ctx: Context) -> Dict[str, object]:
    """The values that will be sent for this model and mode."""
    desc = description_for(ctx)
    if desc is None:
        return {}
    store = store_for(ctx)
    store.check_against(ctx.model_key, desc)
    return store.changed(ctx.model_key, ctx.mode, desc)


def changed_label_text(ctx: Context) -> str:
    n = len(changed_values(ctx))
    return "" if not n else f"{n} setting{'s' if n > 1 else ''} changed"


def notify_changed() -> None:
    alive = []
    for client, fn in list(_LISTENERS):
        if getattr(client, "id", None) not in Client.instances:      # page gone
            continue
        try:
            fn()
            alive.append((client, fn))
        except Exception as ex:
            print(f"[models] settings label refresh failed: {ex}", flush=True)
    _LISTENERS[:] = alive


def _fmt(s: dict, v, detail: bool = False) -> str:
    if isinstance(v, dict) and "derived" in v:
        return v["derived"] if detail else "worked out by the model"
    if v is None:
        return "none given"
    if s["kind"] == "bool":
        return "on" if v else "off"
    if s["kind"] == "choice":
        return next((str(c["label"]) for c in s["choices"] if c["value"] == v), str(v))
    unit = S.unit_text(s)
    return S.number_text(v) + (("" if unit == "%" else " ") + unit if unit else "")


def _range(s: dict) -> str:
    """The allowed range of a number setting, as the refusal states it."""
    if s["kind"] != "number" or "min" not in s or "max" not in s:
        return ""
    return S.range_text(s)


def build_entry(get_ctx: Callable[[], Optional[Context]],
                dialog_host=None, on_change: Optional[Callable[[], None]] = None):
    """The "Settings" button and the "N setting(s) changed" label, for the
    drawer and the run areas. Returns a refresh function."""
    with ui.row().classes("items-center gap-2 pm-model-settings-entry"):
        btn = ui.button("Settings", icon="tune",
                        on_click=lambda: open_panel(get_ctx(), dialog_host, on_change)) \
            .props("flat dense no-caps").classes("pm-model-settings") \
            .tooltip("The settings of the selected detection model: what each one "
                     "changes, PebbleMapper's value and the authors'.")
        label = ui.label("").classes("text-caption text-warning cursor-pointer "
                                     "pm-settings-changed")
        label.on("click", lambda: open_panel(get_ctx(), dialog_host, on_change))

    def refresh():
        ctx = get_ctx()
        btn.set_enabled(ctx is not None)
        text = changed_label_text(ctx) if ctx is not None else ""
        label.set_text(text)
        label.set_visibility(bool(text))
    _LISTENERS.append((context.client, refresh))
    refresh()
    return refresh


def open_panel(ctx: Optional[Context], host=None,
               on_change: Optional[Callable[[], None]] = None) -> None:
    if ctx is None:
        ui.notify("Choose a detection model first.", type="warning")
        return
    desc = description_for(ctx)
    store = store_for(ctx)
    notices = store.check_against(ctx.model_key, desc)
    container = host if host is not None else ui.element("div")
    with container, ui.dialog() as dlg, \
            ui.card().classes("w-[44rem] max-w-full pm-settings-panel"):
        # A closed panel is removed, so panels do not pile up in the page.
        dlg.on("hide", lambda _e: dlg.delete())
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label(f"Settings — {ctx.display_name}").classes("text-h6 col")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round")
        ui.label(f"For {ctx.mode} images. "
                 + ("Saved with this project." if ctx.project_dir else
                    "No project is open: saved for PebbleMapper as a whole.")) \
            .classes("text-caption text-grey-7")
        for n in notices:
            ui.label(n).classes("text-sm text-warning")
        if store.problem:
            ui.label(store.problem).classes("text-sm text-negative")
        if desc is None:
            _no_description(ctx)
            dlg.open()
            return
        # Results of this model already in the project: a change makes later
        # results not comparable with them, said at the first change (FR-013).
        try:
            has_results = bool(ctx.project_dir) and bool(
                S.recorded_value_sets(ctx.project_dir, ctx.model_key))
        except Exception:
            has_results = False
        warn = ui.label("This project already holds results of this model: results made "
                        "from now on will not be comparable with the earlier ones made "
                        "with other values.") \
            .classes("text-sm text-warning pm-not-comparable")
        warn.set_visibility(False)
        body = ui.column().classes("w-full gap-1")
        errs: Dict[str, object] = {}        # per-setting error labels (never on desc)
        open_details: set = set()           # rows whose details are shown, kept across renders

        def changed_any():
            if has_results:
                warn.set_visibility(True)
            notify_changed()
            if on_change:
                on_change()

        def render():
            body.clear()
            values = store.values(ctx.model_key, ctx.mode)
            with body:
                main = [s for s in desc["settings"] if s["main"]]
                more = [s for s in desc["settings"] if not s["main"]]
                for s in main:
                    _row(s, values)
                if more:
                    with ui.expansion(f"More settings ({len(more)})").classes("w-full"):
                        for s in more:
                            _row(s, values)
                if desc.get("fixed"):
                    with ui.expansion("Fixed by PebbleMapper").classes("w-full"):
                        for f in desc["fixed"]:
                            ui.label(f"{f['label']}: {f.get('value', '')} — {f['reason']}") \
                                .classes("text-sm")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Reset all to PebbleMapper's values", icon="restart_alt",
                              on_click=lambda: (store.reset(ctx.model_key, ctx.mode),
                                                render(), changed_any())) \
                        .props("flat no-caps").classes("pm-settings-reset-all")

        def _row(s, values):
            ok, why = S.applies(s, ctx.mode, values, desc)
            default = S.default_for(s, ctx.mode)
            current = values.get(s["key"], default)
            is_changed = s["key"] in values and values[s["key"]] != default
            authors = s.get("authors")
            differs = authors is not None and authors.get("value") is not None \
                and not isinstance(default, dict) \
                and default is not None and authors["value"] != default
            # Unused in this mode and without a value here: nothing to compare.
            no_value_here = not ok and default is None
            with ui.column().classes("w-full gap-0 q-py-xs pm-setting-row") \
                    .style("" if ok else "opacity: 0.55"):
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    with ui.row().classes("col items-center gap-x-2 gap-y-0"):
                        ui.label(s["label"]).classes("text-subtitle2")
                        if is_changed:
                            ui.badge("changed", color="warning").classes("pm-badge-changed")
                        if differs:
                            ui.badge("differs from the authors", color="grey-6") \
                                .classes("pm-badge-differs")
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        _control(s, current, default, ok)
                        info = ui.button(icon="info_outline").props("flat dense round size=sm") \
                            .classes("pm-setting-details-toggle") \
                            .tooltip("Details: values, source, allowed range")
                        reset = ui.button(icon="restart_alt", on_click=lambda k=s["key"]: (
                            store.reset(ctx.model_key, ctx.mode, k), render(),
                            changed_any())) \
                            .props("flat dense round size=sm").classes("pm-setting-reset") \
                            .tooltip("Reset to PebbleMapper's value")
                        # Kept in place while hidden, so the controls stay aligned.
                        if not is_changed:
                            reset.style("visibility: hidden")
                ui.label(s["help"] if ok else why[0].upper() + why[1:] + ".") \
                    .classes("text-sm text-grey-8")
                short_authors = None
                if differs:
                    short_authors = ui.label(f"Authors: {_fmt(s, authors['value'])}") \
                        .classes("text-caption text-grey-7")
                    short_authors.set_visibility(s["key"] not in open_details)
                with ui.column().classes("gap-0 q-pl-sm pm-setting-details") as details:
                    if not ok:
                        ui.label(s["help"]).classes("text-caption text-grey-7")
                    if not no_value_here:
                        ui.label(f"PebbleMapper: {_fmt(s, default, detail=True)}") \
                            .classes("text-caption text-grey-7")
                    # No authors' value: their source says why (asked on screen,
                    # not part of the tool, another unit…).
                    ui.label("Authors: " + (
                        "none given" if not authors else
                        authors["source"] if authors.get("value") is None else
                        f"{_fmt(s, authors['value'])} ({authors['source']})")) \
                        .classes("text-caption text-grey-7")
                    rng = _range(s)
                    if rng:
                        ui.label(f"Allowed: {rng}").classes("text-caption text-grey-7")
                    if s.get("differs_because") and (differs or not authors
                                                     or authors.get("value") is None):
                        ui.label(("Why PebbleMapper differs: " if differs else
                                  "Why PebbleMapper chose this value: ")
                                 + s["differs_because"]) \
                            .classes("text-caption text-grey-7")
                    if s.get("cost"):
                        ui.label(s["cost"]).classes("text-caption text-grey-7")
                details.set_visibility(s["key"] in open_details)

                def toggle(k=s["key"], d=details, short=short_authors):
                    open_details.symmetric_difference_update({k})
                    d.set_visibility(k in open_details)
                    if short is not None:       # the details repeat it, with its source
                        short.set_visibility(k not in open_details)
                info.on_click(toggle)
                err = ui.label("").classes("text-caption text-negative pm-setting-error")
                err.set_visibility(False)
                errs[s["key"]] = err
            ui.separator().classes("pm-setting-sep")

        def _control(s, current, default, enabled):
            shown = None if isinstance(current, dict) else current

            def commit(value, el=None):
                try:
                    store.set(ctx.model_key, ctx.mode, desc, s["key"], value)
                except ValueError as ex:
                    err = errs.get(s["key"])
                    if err is not None:
                        err.set_text(str(ex))
                        err.set_visibility(True)
                    if el is not None:              # refused: the previous value stays
                        el.value = shown
                    return
                render()
                changed_any()

            if s["kind"] == "bool":
                el = ui.switch(value=bool(shown))
                el.on_value_change(lambda e: commit(bool(e.value)))
            elif s["kind"] == "choice":
                el = ui.select({c["value"]: c["label"] for c in s["choices"]}, value=shown) \
                    .props("dense")
                el.on_value_change(lambda e: commit(e.value))
            else:
                # No min/max here: the field would clamp silently; the check
                # below refuses and says the range (FR-009).
                el = ui.number(value=shown,
                               step=s.get("step", 1),
                               placeholder="worked out" if shown is None and enabled
                               else None,
                               suffix=S.unit_text(s) or None) \
                    .props("dense").classes("w-32")

                def on_blur(_e=None, el=el):
                    v = el.value
                    if v is None or v == shown:
                        return
                    commit(int(v) if s.get("integer") and float(v) == int(v) else v, el)
                el.on("blur", on_blur)
                # Enter leaves the field, so its value is synced first (blur commits).
                el.on("keydown.enter", js_handler="(e) => e.target.blur()")
            el.set_enabled(enabled)
            el.classes(f"pm-setting-{s['key']}")
        render()
    dlg.open()


def _no_description(ctx: Context) -> None:
    problem = S.description_problem(ctx.backend) if ctx.backend is not None else ""
    if problem:
        ui.label("This release's settings description could not be read, so its settings "
                 f"are not offered in PebbleMapper ({problem}).").classes("text-sm")
    else:
        ui.label("This release's settings are not offered in PebbleMapper.") \
            .classes("text-sm")
    newer = _newer_release_with_settings(ctx)
    if newer:
        ui.label(f"Release {newer} of this model offers them: update it on the Models tab.") \
            .classes("text-sm pm-settings-newer")


def _newer_release_with_settings(ctx: Context) -> Optional[str]:
    """The newest release in the kept model list that describes its settings,
    when it is newer than the installed one (spec US6)."""
    try:
        from installer import catalog
        from installer.updates import version_tuple
        data = catalog.kept_list(catalog.current_source()) or {}
    except Exception:
        return None
    name = S.model_name(ctx.model_key)
    current = ctx.model_key.split("@", 1)[1].replace("-cpu", "") if "@" in ctx.model_key else None
    if current is None:
        return None                       # hand-installed: no update to offer
    best = None
    for m in data.get("models", []):
        if not any(b.get("name") == name for b in m.get("backends", [])):
            continue
        for r in m.get("releases", []):
            if r.get("describes_settings") and not r.get("withdrawn") \
                    and version_tuple(r["release"]) > version_tuple(current):
                if best is None or version_tuple(r["release"]) > version_tuple(best):
                    best = r["release"]
    return best
