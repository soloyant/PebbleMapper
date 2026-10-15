"""The Models tab: install, follow and recover detection models (spec 030).

Built by ``gui.app`` (``build_models_tab``), which also places the header
indicator and the notice line built here. The work itself runs in the install
worker (``python -m installer.worker``); this page writes requests and reads
``job.json`` on a timer built at build time (a ``ui.timer`` created inside a
handler never fires). Wording follows ``specs/030-.../contracts/ui.md``: on
screen it is the "graphics-card version" and the "processor version", never
a variant, lock, prefix or environment (FR-028).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from nicegui import background_tasks, run, ui

from installer import catalog, jobs, registration, sysinfo
from installer.errors import human_size
from installer.updates import version_tuple
from installer.paths import models_state_dir, write_json_atomic

_STEP_TYPICAL = {"env-conda": (3, 15), "env-pip": (5, 20), "check-run": (1, 10)}
_START_CHECK = {"done": False}          # the start check runs once per app run


# ------------------------------------------------------------- settings
def _settings_file() -> Path:
    return models_state_dir() / "settings.json"


def models_dir() -> Path:
    try:
        d = json.loads(_settings_file().read_text(encoding="utf-8")).get("models_dir")
        if d:
            return Path(d)
    except (OSError, ValueError):
        pass
    return sysinfo.default_models_dir()


def set_models_dir(path: Path) -> None:
    write_json_atomic(_settings_file(), {"models_dir": str(path)})


def restart_app() -> None:
    """Stop the server cleanly and leave with the starter's restart code
    (031 FR-011); gui.app does the exit once the server is down."""
    from nicegui import app as _ng_app
    from functions import crashsafe
    from installer import appupdate
    appupdate.RESTART["requested"] = True
    crashsafe.mark_clean_exit()
    _ng_app.shutdown()


# --------------------------------------------------------------- worker
def start_worker() -> None:
    """Start the worker unless one runs (FR-034). It inherits the launcher's
    console, so its title can say an install is running (US2)."""
    if jobs.worker_alive():
        return
    subprocess.Popen([sys.executable, "-m", "installer.worker"],
                     cwd=str(Path(__file__).resolve().parents[1]),
                     stdin=subprocess.DEVNULL)


# ------------------------------------------------------------- helpers
def _installed_ids() -> dict:
    return {e["id"]: e for e in registration.read_installed()}


def _active_jobs() -> list:
    return [j for j in jobs.all_jobs()
            if j["state"] in jobs.ACTIVE + jobs.RESUMABLE]


def _step_line(job: dict) -> str:
    steps = [s for s in job.get("steps", []) if s.get("status") != "skipped"]
    cur = next((i for i, s in enumerate(steps) if s.get("status") == "running"), None)
    if job["state"] == "paused":
        return "Paused: battery low. Plug in to continue."
    if job["state"] == "queued":
        return "Waiting for the install in progress to finish"
    if cur is None:
        return {"paused": "Paused: battery low. Plug in to continue.",
                "cancelled": "Cancelled", "interrupted": "Interrupted",
                "failed": "Stopped on an error"}.get(job["state"], job["state"])
    s = steps[cur]
    line = f"{s['label']} — step {cur + 1} of {len(steps)}"
    if s.get("bytes_total"):
        line += (f" — {s.get('bytes_done', 0) / 1e6:,.0f} of "
                 f"{s['bytes_total'] / 1e6:,.0f} MB")
    elif s["name"] in _STEP_TYPICAL:
        lo, hi = _STEP_TYPICAL[s["name"]]
        line += f" — this step usually takes {lo}–{hi} min"
    return line


def _elapsed(job: dict) -> str:
    t0 = job.get("started")
    if not t0 or job["state"] not in ("running", "paused"):
        return ""
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m} min {s:02d} s"


# ------------------------------------------------------- header pieces
def build_header_indicator(open_tab: Callable[[], None]):
    """A compact line in the header while a job runs; click → Models tab."""
    box = ui.row().classes("items-center gap-1 cursor-pointer pm-install-indicator")
    box.on("click", lambda: open_tab())
    with box:
        spin = ui.spinner(size="sm", color="white")
        lbl = ui.label("").classes("text-caption text-white")
    box.set_visibility(False)

    def tick():
        running = [j for j in jobs.all_jobs() if j["state"] in ("running", "paused")]
        if not running:
            box.set_visibility(False)
            return
        j = running[0]
        verb = {"repair": "Repairing", "uninstall": "Removing",
                "app-update": "Updating"}.get(j.get("kind"), "Installing")
        lbl.set_text(f"{verb} {j.get('display_name') or j.get('model')}: {_step_line(j)}")
        spin.set_visibility(j["state"] == "running")
        box.set_visibility(True)
    ui.timer(1.0, tick)
    return box


def build_notice_line(open_tab: Callable[[], None]):
    """One dismissable line under the header (US2): interrupted installs,
    and the permanent test-list banner (FR-052)."""
    if catalog.is_test_source(catalog.current_source()):
        with ui.row().classes("w-full items-center bg-orange-2 q-px-md q-py-xs "
                              "pm-test-list-banner"):
            ui.icon("science")
            ui.label("Test model list in use: "
                     f"{catalog.current_source()}").classes("text-sm")
    jobs.detect_interrupted()             # a running job with no worker left
    interrupted = [j for j in jobs.all_jobs() if j["state"] == "interrupted"]
    if not interrupted:
        return
    with ui.row().classes("w-full items-center bg-amber-1 q-px-md q-py-xs "
                          "pm-notice-line") as bar:
        ui.icon("info", color="amber-9")
        noun = {"repair": "The repair", "uninstall": "The removal",
                "app-update": "The update"}
        text = "; ".join(f"{noun.get(j.get('kind'), 'The install')} of "
                         f"{j.get('display_name') or j.get('model')}" for j in interrupted)
        ui.label(f"{text} was interrupted — Resume or Remove on the Models tab") \
            .classes("text-sm")
        ui.button("Open", on_click=lambda: open_tab()).props("flat dense no-caps")
        ui.space()
        ui.button(icon="close", on_click=lambda: bar.set_visibility(False)) \
            .props("flat dense round").tooltip("Dismiss")


_BACKGROUND_CHECK = {"started": False}


def ensure_background_check() -> None:
    """Start the update check once per run of the app, after the first page
    (031 FR-001). It never touches the interface."""
    if _BACKGROUND_CHECK["started"]:
        return
    _BACKGROUND_CHECK["started"] = True
    from installer import updates
    updates.start_background_check()


def current_updates() -> list:
    """Updates against the list kept from the last fetch (no network)."""
    from installer import updates
    data = catalog.kept_list(catalog.current_source())
    return updates.available_updates(data, registration.read_installed())


def build_update_notice(open_tab: Callable[[], None]):
    """The header's update line (031 FR-006): one line per update not
    dismissed, never a dialog; refreshed from the kept state every 30 s."""
    from installer import updates
    box = ui.column().classes("w-full gap-0 pm-update-notice")

    @ui.refreshable
    def render():
        for u in updates.to_announce(current_updates()):
            rel = u["release"]
            with ui.row().classes("w-full items-center bg-blue-1 q-px-md q-py-xs"):
                ui.icon("system_update", color="primary")
                if u.get("withdrawn_current"):
                    text = (f"{u['name']} {u['current']} was withdrawn by the maintainer; "
                            f"use {rel['release']} instead.")
                else:
                    text = f"Update available: {u['name']} {rel['release']}."
                summary = rel.get("summary", "")
                ui.label(f"{text} {summary}".strip()).classes("text-sm")
                if rel.get("notes"):
                    ui.link("Notes", rel["notes"], new_tab=True).classes("text-sm")
                ui.button("Open", on_click=lambda: open_tab()).props("flat dense no-caps")
                ui.space()

                def _dismiss(k=u["key"]):
                    updates.dismiss(k)
                    render.refresh()
                ui.button(icon="close", on_click=_dismiss).props("flat dense round") \
                    .tooltip("Dismiss (it stays on the Models tab)")

    with box:
        render()
    ui.timer(30.0, render.refresh)
    return box


_SWITCH_TEXT = {
    # The starter records a rollback as a switch from the failed version to
    # the one put back.
    "rolled_back": ("PebbleMapper {from} did not start, so {to} was put back as it was. "
                    "Nothing of your projects changed. Please build a diagnostic bundle "
                    "and report it."),
    "damaged": ("The prepared update to {to} was damaged on disk and was discarded; "
                "{from} started as before. Check for updates again on the Models tab."),
    "failed": ("The update to {to} could not be applied ({detail}); the version you had "
               "started as before. Please build a diagnostic bundle and report it."),
    "returned": "PebbleMapper is back to {to}, as you asked.",
    "ok": "PebbleMapper {to} is running.",
}


def build_switch_banner():
    """After the starter switched versions (031 FR-014, FR-015): what
    happened, once, and every shipped file the user had changed and that was
    kept beside the new one as ``.mine``."""
    from installer import appupdate
    ls = appupdate.unseen_switch()
    if not ls:
        return None
    result = ls.get("result", "")
    bad = result in ("rolled_back", "damaged", "failed")
    facts = {"to": ls.get("to", "?"), "from": ls.get("from", "the version you had"),
             "detail": ls.get("detail", "")}
    with ui.column().classes("w-full gap-1 q-px-md q-py-sm pm-switch-banner "
                             + ("bg-orange-1" if bad else "bg-green-1")) as box:
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("warning" if bad else "check_circle",
                    color="warning" if bad else "positive")
            ui.label(_SWITCH_TEXT.get(result, "{to}").format(**facts)).classes("text-sm")
            ui.space()
            if bad:
                def _bundle():
                    from functions import feedback as _fb
                    try:
                        out = _fb.build_bundle(None, note=f"App update {result}")
                    except Exception as ex:
                        ui.notify(f"Could not build the bundle: {ex}", type="negative")
                        return
                    ui.notify(f"Bundle written: {out}. Attach it to an issue at "
                              f"{_fb.ISSUES_URL}.", type="positive", timeout=15000,
                              multi_line=True, close_button=True)
                ui.button("Build diagnostic bundle", icon="medical_services",
                          on_click=_bundle).props("flat dense no-caps")

            def _close():
                appupdate.mark_switch_seen()
                box.delete()
            ui.button(icon="close", on_click=_close).props("flat dense round") \
                .tooltip("Dismiss")
        mine = ls.get("kept_mine") or []
        if mine:
            ui.label(f"You had changed {len(mine)} file(s) that came with PebbleMapper. "
                     "The new version's file is in place, and yours is kept beside it "
                     "with .mine added to its name:").classes("text-sm")
            for p in mine[:20]:
                ui.label(p).classes("text-caption font-mono q-pl-md")
            if len(mine) > 20:
                ui.label(f"… and {len(mine) - 20} more").classes("text-caption q-pl-md")
    return box


# ----------------------------------------------------------- the tab
def build_models_tab(state, on_models_changed: Callable[[], None],
                     register_tab_refresh: Optional[Callable] = None):
    """The tab. ``on_models_changed`` refreshes every page's model select
    after an install or a removal."""
    page = {"catalog": None, "fetching": False, "last_jobs": None,
            "known_done": {j["id"] for j in jobs.all_jobs() if j["state"] == "done"}}

    ui.label("Detection models").classes("text-h5")
    ui.label("Install another detection model in one click. PebbleMapper "
             "downloads the release the maintainer tested, checks every file, "
             "tries the model once on a sample photograph, and adds it to the "
             "Detection model list. Nothing to type, nothing to restart.") \
        .classes("text-sm text-grey-8 pm-models-intro")
    list_status = ui.label("").classes("text-caption text-grey-7 pm-list-status")

    updates_box = ui.column().classes("w-full gap-2")
    # Dialogs opened from the Updates cards live here: built inside a card,
    # the card's refresh deleted them before they showed (rehearsal, 031).
    dialog_host = ui.element("div")
    jobs_box = ui.column().classes("w-full gap-2")
    installed_box = ui.column().classes("w-full gap-2")
    available_box = ui.column().classes("w-full gap-2")
    folder_box = ui.column().classes("w-full gap-1")

    # ---------------------------------------------------------- jobs
    @ui.refreshable
    def render_jobs():
        active = _active_jobs()
        if not active:
            return
        ui.label("Installing").classes("text-h6")
        for j in active:
            with ui.card().classes("w-full pm-job-card"):
                with ui.row().classes("w-full items-center gap-2"):
                    if j["state"] == "running":
                        ui.spinner(size="sm")
                    ui.label(j.get("display_name") or j.get("model")) \
                        .classes("text-subtitle1")
                    ui.label(f"release {j.get('release')}").classes("text-caption text-grey-7")
                    ui.space()
                    el = _elapsed(j)
                    if el:
                        ui.label(el).classes("text-caption text-grey-7")
                ui.label(_step_line(j)).classes("text-sm pm-job-step")
                if j["state"] == "running":
                    ui.label("Keeping the computer awake while this runs.") \
                        .classes("text-caption text-grey-7")
                err = j.get("error")
                if j["state"] == "failed" and err:
                    ui.label(err.get("message", "")).classes("text-sm text-negative")
                    ui.label(err.get("action", "")).classes("text-sm")
                with ui.row().classes("gap-2"):
                    if j["state"] in jobs.ACTIVE:
                        ui.button("Cancel", icon="stop",
                                  on_click=lambda jid=j["id"]: _cancel(jid)) \
                            .props("outline dense no-caps")
                    if j["state"] in jobs.RESUMABLE:
                        ui.button("Resume", icon="play_arrow",
                                  on_click=lambda jid=j["id"]: _resume(jid)) \
                            .props("dense no-caps")
                        ui.button("Remove", icon="delete",
                                  on_click=lambda jid=j["id"]: _remove_job(jid)) \
                            .props("outline dense no-caps")
                    if j["state"] == "failed":
                        ui.button("Build diagnostic bundle", icon="bug_report",
                                  on_click=_bundle).props("flat dense no-caps")
                with ui.expansion("Show details").classes("w-full"):
                    for line in jobs.JobLog(Path(j["log"])).tail(20):
                        ui.label(line).classes("text-caption font-mono") \
                            .style("white-space: pre-wrap; word-break: break-all;")

    def _cancel(jid):
        jobs.request_cancel(jid)
        ui.notify("Stopping…")
        render_jobs.refresh()

    def _resume(jid):
        jobs.resume(jid)
        start_worker()
        render_jobs.refresh()

    def _remove_job(jid):
        from installer import removal
        job = jobs.read_job(jid)
        try:
            freed = removal.remove_job(job)
        except Exception as ex:
            ui.notify(f"Could not remove it: {ex}", type="negative")
            return
        ui.notify(f"Removed; {human_size(freed)} freed.", type="positive")
        render_jobs.refresh()
        render_available.refresh()

    def _bundle():
        from functions import feedback as _fb
        try:
            out = _fb.build_bundle(None, note="Model install failed")
        except Exception as ex:
            ui.notify(f"Could not build the bundle: {ex}", type="negative")
            return
        ui.notify(f"Bundle written: {out}. Attach it to an issue at {_fb.ISSUES_URL}.",
                  type="positive", timeout=15000, multi_line=True, close_button=True)

    # ------------------------------------------------------ installed
    @ui.refreshable
    def render_installed():
        entries = registration.read_installed()
        listed = {m["id"]: m for m in ((page["catalog"] or {}).get("models", []))}
        if not entries:
            return
        ui.label("Installed").classes("text-h6")
        for e in entries:
            names = ", ".join(b.get("display_name", b["name"]) for b in e["backends"])
            with ui.card().classes("w-full pm-installed-card"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(names).classes("text-subtitle1")
                    ui.label(f"release {e['release']}" +
                             (" — processor version" if e.get("variant") == "cpu" else "")) \
                        .classes("text-caption text-grey-7")
                    if e.get("state") == "needs_repair":
                        ui.badge("Needs repair", color="negative")
                ui.label(f"{e['root']} — {human_size(e.get('size_bytes', 0))}") \
                    .classes("text-caption text-grey-7").style("word-break: break-all;")
                if e.get("state") == "needs_repair" and e.get("reason"):
                    ui.label(e["reason"]).classes("text-sm text-negative")
                m = listed.get(e["id"])
                newer = catalog.latest_release(m) if m else None
                if newer and tuple(map(int, newer["release"].split("."))) > \
                        tuple(map(int, e["release"].split("."))):
                    ui.label(f"A newer release ({newer['release']}) exists.") \
                        .classes("text-sm")
                cit = e.get("citation") or (m or {}).get("citation", "")
                if cit:
                    with ui.row().classes("items-center gap-1"):
                        ui.label(cit).classes("text-caption")
                        ui.button(icon="content_copy",
                                  on_click=lambda c=cit: ui.clipboard.write(c)) \
                            .props("flat dense round").tooltip("Copy citation")
                busy = any(j.get("model") == e["id"] for j in _active_jobs())
                with ui.row().classes("gap-2"):
                    if e.get("state") == "needs_repair":
                        ui.button("Repair", icon="build",
                                  on_click=lambda e=e: _repair(e)) \
                            .props("dense no-caps").classes("pm-repair-button") \
                            .set_enabled(not busy)
                    older = any(o["id"] == e["id"] and o is not e
                                and version_tuple(o["release"]) > version_tuple(e["release"])
                                for o in entries)
                    label = (f"Remove older release (frees {human_size(e.get('size_bytes', 0))})"
                             if older else "Uninstall")
                    ui.button(label, icon="delete",
                              on_click=lambda e=e: _confirm_uninstall(e)) \
                        .props("outline dense no-caps").classes("pm-uninstall-button") \
                        .set_enabled(not busy)
        for b, m in _manual_models():
            with ui.card().classes("w-full pm-manual-card"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(b["display_name"]).classes("text-subtitle1")
                    ui.badge("installed manually", color="grey-7")
                ui.label("Declared by hand in user_detectors.json. PebbleMapper does "
                         "not change or remove it; to remove it, delete its entry "
                         "from user_detectors.json and its conda environment, then "
                         "restart. Remove it first to have PebbleMapper install and "
                         "manage this model instead.").classes("text-caption text-grey-7")

    def _manual_models():
        """Listed backends registered from user_detectors.json (FR-044)."""
        try:
            from detectors import registry
            names = {k for k in registry._BACKENDS if "@" not in k}
        except Exception:
            names = set()
        out = []
        for m in (page["catalog"] or {}).get("models", []):
            for b in m["backends"]:
                if b["name"] in names:
                    out.append((b, m))
        return out

    def _confirm_uninstall(e):
        from detectors.instance_backend import environments_in_use
        name = ", ".join(b.get("display_name", b["name"]) for b in e["backends"])
        if str(e["env"]) in environments_in_use():
            ui.notify(f"{name} is running a detection; remove it when the run has "
                      "finished.", type="warning", multi_line=True)
            return
        with ui.dialog() as dlg, ui.card().classes("max-w-md"):
            ui.label(f"Remove {name}?").classes("text-h6")
            ui.label(f"This frees {human_size(e.get('size_bytes', 0))}. Results made "
                     "with it are kept.").classes("text-sm")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps")

                def _go():
                    dlg.close()
                    jobs.new_job("uninstall", {
                        "model": e["id"], "release": e["release"],
                        "variant": e["variant"], "display_name": name,
                        "models_dir": str(Path(e["root"]).parent)})
                    start_worker()
                    render_jobs.refresh()
                    render_installed.refresh()
                ui.button("Remove", icon="delete", on_click=_go) \
                    .props("color=negative no-caps").classes("pm-uninstall-confirm")
        dlg.open()

    def _repair(e):
        data = page["catalog"]
        if not data or not any(m["id"] == e["id"] for m in data.get("models", [])):
            ui.notify("Repair needs the model list; connect to the internet and open "
                      "this tab again.", type="warning")
            return
        from installer.worker import request_for
        try:
            req = request_for(data, e["id"], e["variant"], str(Path(e["root"]).parent),
                              e["release"])
        except SystemExit as ex:
            ui.notify(f"This release can no longer be repaired: {ex}. Uninstall it "
                      "and install the current one.", type="warning", multi_line=True)
            return
        jobs.new_job("repair", req)
        start_worker()
        ui.notify(f"Repairing {req['display_name']}.", type="info")
        render_jobs.refresh()
        render_installed.refresh()

    # ------------------------------------------------------ available
    @ui.refreshable
    def render_available():
        data = page["catalog"]
        if not data:
            return
        installed = _installed_ids()
        busy = {j.get("model") for j in _active_jobs()}
        cards = page.get("gpus")
        if cards is None:
            cards = page["gpus"] = sysinfo.gpus()
        mdir = models_dir()
        manual = {m["id"] for _, m in _manual_models()}
        offered = [m for m in data.get("models", [])
                   if m["id"] not in installed and m["id"] not in manual
                   and catalog.latest_release(m)]
        if not offered:
            ui.label("Every listed model is installed.").classes("text-sm text-grey-7")
            return
        ui.label("Available").classes("text-h6")
        for m in offered:
            rel = catalog.latest_release(m)
            with ui.card().classes("w-full pm-available-card"):
                names = ", ".join(b["display_name"] for b in m["backends"])
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label(names).classes("text-subtitle1")
                    ui.label(f"release {rel['release']}").classes("text-caption text-grey-7")
                ui.label(m["description"]).classes("text-sm")
                ui.label(f"{m['authors']} · Code: {m['licence_code']} · "
                         f"Weights: {m['licence_weights']}").classes("text-caption text-grey-7")
                force = {"cpu": False}
                both = "gpu" in rel["variants"] and "cpu" in rel["variants"]
                info = ui.column().classes("gap-0")

                def _render_info(m=m, rel=rel, force=force, info=info):
                    info.clear()
                    v = sysinfo.verdict(rel, cards, force_cpu=force["cpu"])
                    with info:
                        ui.label(v.text).classes(
                            "text-sm " + ("text-positive" if v.variant == "gpu" else
                                          "text-warning" if v.variant == "cpu" else
                                          "text-negative"))
                        if v.variant:
                            vi = rel["variants"][v.variant]
                            lo, hi = vi["minutes"]
                            ui.label(f"Download {human_size(vi['download_bytes'])} · "
                                     f"needs {sysinfo.describe_need(mdir, vi)} · "
                                     f"about {lo}–{hi} min").classes("text-caption")
                            short = sysinfo.disk_shortfall(mdir, vi)
                            if short:
                                ui.label("Not enough space: " + ", ".join(
                                    f"free {human_size(n)} more on {d}" for d, n in short.items())) \
                                    .classes("text-sm text-negative")
                            btn = ui.button(f"Install ({human_size(vi['download_bytes'])} download)",
                                            icon="download",
                                            on_click=lambda m=m, rel=rel, var=v.variant:
                                            _install(m, rel, var)) \
                                .props("no-caps").classes("pm-install-button")
                            if short or m["id"] in busy:
                                btn.disable()
                _render_info()
                if both:
                    ui.checkbox("Use the processor version",
                                on_change=lambda e, f=force, r=_render_info:
                                (f.__setitem__("cpu", bool(e.value)), r())) \
                        .props("dense")

    async def _install(m, rel, variant):
        mdir = models_dir()
        vi = rel["variants"][variant]
        chk = sysinfo.check_models_folder(mdir, vi.get("longest_path", 0))
        if not chk.ok:
            ui.notify(f"The models folder {mdir} cannot be used: {chk.reason}. "
                      "Choose another folder below.", type="negative", multi_line=True)
            return
        if sysinfo.disk_shortfall(mdir, vi):
            ui.notify("Not enough free space; see the line above the button.",
                      type="negative")
            return
        warn = []
        is_metered = await run.io_bound(sysinfo.metered)
        p = sysinfo.power()
        if p.on_battery:
            lo, hi = vi["minutes"]
            warn.append(f"The computer is on battery; the install takes about "
                        f"{lo}–{hi} min.")
            if sysinfo.modern_standby():
                warn.append("This laptop may sleep during the install on battery; "
                            "plug it in.")
        if is_metered:
            warn.append(f"You are on a metered connection. Download "
                        f"{human_size(vi['download_bytes'])}?")
        if warn:
            with ui.dialog() as dlg, ui.card().classes("max-w-md"):
                for w in warn:
                    ui.label(w).classes("text-sm")
                with ui.row().classes("justify-end w-full"):
                    ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat no-caps")
                    ui.button("Download" if is_metered else "Install",
                              on_click=lambda: dlg.submit(True)).props("no-caps")
            if not await dlg:
                return
        from installer.worker import request_for
        req = request_for(page["catalog"], m["id"], variant, str(mdir), rel["release"])
        jobs.new_job("install", req)
        start_worker()
        ui.notify(f"Installing {req['display_name']}. You can keep working; "
                  "progress is shown here and in the header.", type="info")
        render_jobs.refresh()
        render_available.refresh()

    # --------------------------------------------------------- folder
    @ui.refreshable
    def render_folder():
        mdir = models_dir()
        ui.label("Models folder").classes("text-h6")
        with ui.row().classes("items-center gap-2"):
            ui.label(str(mdir)).classes("text-sm font-mono pm-models-folder")
            try:
                ui.label(f"{human_size(sysinfo.free_bytes(mdir))} free on "
                         f"{sysinfo.drive_of(mdir)}").classes("text-caption text-grey-7")
            except OSError:
                pass
            ui.button("Change…", on_click=_change_folder).props("flat dense no-caps")
        ui.label("New installs go here; models already installed stay where they are.") \
            .classes("text-caption text-grey-7")

    def _change_folder():
        with ui.dialog() as dlg, ui.card().classes("min-w-[28rem]"):
            ui.label("Models folder").classes("text-h6")
            inp = ui.input("Folder", value=str(models_dir())).classes("w-full")
            msg = ui.label("").classes("text-sm text-negative")

            def _ok():
                p = Path(inp.value.strip())
                if not p.is_absolute():
                    msg.set_text("Give a full path, such as D:\\PebbleMapper-models.")
                    return
                chk = sysinfo.check_models_folder(p)
                if not chk.ok:
                    msg.set_text(f"This folder cannot be used: {chk.reason}.")
                    return
                set_models_dir(p)
                dlg.close()
                render_folder.refresh()
                render_available.refresh()
                if chk.warning:
                    ui.notify(chk.warning, type="warning")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                ui.button("Use this folder", on_click=_ok).props("no-caps")
        dlg.open()

    # ------------------------------------------------------- refreshes
    # ---------------------------------------------------------- updates
    @ui.refreshable
    def render_updates():
        from installer import updates as _up
        data = page["catalog"]
        ups = _up.available_updates(data, registration.read_installed()) if data else []
        st = _up.read_state()
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Updates").classes("text-h6")
            last = st.get("last_check")
            ui.label(f"Last check: {last}, {st.get('last_result', '')}" if last
                     else "Not checked yet").classes("text-caption text-grey-7")
            ui.space()
            ui.button("Check now", icon="refresh", on_click=_check_now) \
                .props("flat dense no-caps").classes("pm-check-now")
        _render_previous_version()
        if not ups:
            ui.label("Everything is up to date.").classes("text-sm text-grey-7")
            return
        busy = {j.get("model") for j in _active_jobs()}
        for u in ups:
            rel = u["release"]
            with ui.card().classes("w-full pm-update-card"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"{u['name']}: {u['current']} → {rel['release']}") \
                        .classes("text-subtitle1")
                    if u.get("withdrawn_current"):
                        ui.badge("current release withdrawn", color="negative")
                if rel.get("summary"):
                    ui.label(rel["summary"]).classes("text-sm")
                if rel.get("results_may_change"):
                    ui.label("Detections may differ from those of the release you have; "
                             "results record which release made them.") \
                        .classes("text-sm text-warning")
                if rel.get("notes"):
                    ui.link("Release notes", rel["notes"], new_tab=True).classes("text-caption")
                if u["kind"] == "model":
                    vi = rel["variants"][u["variant"]]
                    ui.label(f"Download {human_size(vi.get('download_bytes', 0))} · needs "
                             f"{human_size(vi.get('peak_bytes', 0))} · the current release "
                             "stays until you remove it").classes("text-caption")
                    ui.button(f"Update ({human_size(vi.get('download_bytes', 0))} download)",
                              icon="system_update", on_click=lambda u=u: _update_model(u)) \
                        .props("no-caps").classes("pm-update-model") \
                        .set_enabled(u["id"] not in busy)
                else:
                    _render_app_update(u, busy)

    def _render_previous_version():
        """The version before this one, kept after an update until removed
        (031 FR-015, US2 scenario 7)."""
        from installer import appupdate
        prev = appupdate.previous_version()
        if not prev:
            return
        with ui.card().classes("w-full pm-previous-version"):
            if prev["return_requested"]:
                ui.label(f"PebbleMapper returns to {prev['release']} at the next start.") \
                    .classes("text-sm")
                with ui.row().classes("gap-2"):
                    ui.button("Restart now", icon="restart_alt", on_click=_confirm_restart) \
                        .props("no-caps").classes("pm-restart-now")
                    ui.button("Keep this version",
                              on_click=lambda: (appupdate.cancel_return(),
                                                render_updates.refresh())) \
                        .props("flat no-caps")
                return
            if page.get("removing_previous"):
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label(f"Removing {prev['release']}: its software is several GB, "
                             "this takes a few minutes.").classes("text-sm")
                return
            can_return = appupdate.can_return()
            if can_return:
                text = (f"The previous version, {prev['release']}, is kept so you can go back "
                        "to it.")
            elif prev.get("last_result") == "rolled_back":
                text = (f"{prev['release']} did not start and is no longer used. Remove it to "
                        "free its space.")
            else:
                text = f"{prev['release']} is no longer used. Remove it to free its space."
            ui.label(text).classes("text-sm")
            with ui.row().classes("gap-2"):
                if can_return:
                    ui.button(f"Return to {prev['release']}", icon="undo",
                              on_click=_return_to_previous) \
                        .props("outline no-caps").classes("pm-return-previous")
                ui.button(f"Remove {prev['release']}", icon="delete",
                          on_click=_remove_previous) \
                    .props("flat no-caps").classes("pm-remove-previous")

    def _return_to_previous():
        from installer import appupdate
        try:
            appupdate.request_restart_to_previous()
        except Exception as ex:
            ui.notify(f"Cannot return: {ex}", type="negative")
            return
        render_updates.refresh()
        _confirm_restart()

    async def _remove_previous():
        from installer import appupdate
        prev = appupdate.previous_version() or {}
        with dialog_host, ui.dialog() as dlg, ui.card().classes("max-w-md"):
            ui.label(f"Remove PebbleMapper {prev.get('release', '')}?").classes("text-h6")
            ui.label("Its files and its environment are deleted, and you can no longer "
                     "return to it from here. Your projects and results are not touched.") \
                .classes("text-sm")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat no-caps")
                ui.button("Remove", on_click=lambda: dlg.submit(True)) \
                    .props("no-caps color=negative").classes("pm-remove-previous-confirm")
        if not await dlg:
            return
        page["removing_previous"] = True
        render_updates.refresh()
        try:
            freed = await run.io_bound(appupdate.remove_previous)
        except Exception as ex:
            ui.notify(f"Could not remove it: {ex}", type="negative")
            return
        finally:
            page["removing_previous"] = False
            render_updates.refresh()
        ui.notify(f"PebbleMapper {prev.get('release', '')} removed; {human_size(freed)} freed.",
                  type="positive")

    def _render_app_update(u, busy):
        """PebbleMapper itself: prepared in the background, applied by the
        starter at the next start (031 US2, FR-010–FR-011)."""
        from installer import appupdate
        rel = u["release"]
        refusal = appupdate.in_place_refusal()
        if refusal:
            ui.label(f"It cannot be updated from here: {refusal}.") \
                .classes("text-caption text-grey-7")
            return
        if appupdate.staged_release() == rel["release"]:
            ui.label(f"{rel['release']} is ready. It replaces {u['current']} at the next "
                     "start; your projects and results are not touched.") \
                .classes("text-sm text-positive")
            ui.button("Restart now", icon="restart_alt", on_click=_confirm_restart) \
                .props("no-caps").classes("pm-restart-now")
            return
        vi = rel["variants"].get("gpu") or next(iter(rel["variants"].values()))
        # The release's own files are small; its software is most of the download.
        ui.label(f"Download {human_size(vi.get('download_bytes', 0))} · needs "
                 f"{human_size(vi.get('disk_bytes', 0))} · prepared in the background, "
                 "applied at the next start").classes("text-caption")
        ui.button("Update", icon="system_update", on_click=lambda: _update_app(u)) \
            .props("no-caps").classes("pm-update-app").set_enabled("pebblemapper" not in busy)

    def _update_app(u):
        from installer import appupdate
        rel = u["release"]
        variant = "gpu" if "gpu" in rel["variants"] else next(iter(rel["variants"]))
        jobs.new_job("app-update", appupdate.request_for_app(page["catalog"],
                                                             rel["release"], variant))
        start_worker()
        ui.notify(f"Preparing PebbleMapper {rel['release']}; you can keep working.",
                  type="info")
        render_jobs.refresh()
        render_updates.refresh()

    def _confirm_restart():
        with dialog_host, ui.dialog() as dlg, ui.card().classes("max-w-md"):
            ui.label("Restart PebbleMapper now?").classes("text-h6")
            ui.label("The server stops, switches version and starts again; this page "
                     "reconnects by itself. A detection that is running is interrupted; "
                     "queued jobs are saved and offered back.").classes("text-sm text-grey-7")
            if any(j.get("kind") != "app-update" for j in _active_jobs()):
                ui.label("A model install is running: it pauses and can be resumed "
                         "from the Models tab.").classes("text-sm text-warning")
            with ui.row().classes("justify-end gap-2 w-full"):
                ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                ui.button("Restart now", icon="restart_alt",
                          on_click=lambda: (dlg.close(), restart_app())) \
                    .props("no-caps").classes("pm-restart-confirm")
        dlg.open()

    async def _check_now():
        from installer import updates as _up
        st = await run.io_bound(_up.run_check)
        page["catalog"] = catalog.kept_list(catalog.current_source())
        if st.get("last_result") != "ok":
            list_status.set_text(st.get("last_message") or
                                 "Could not reach the model list (no connection).")
        render_updates.refresh()
        render_installed.refresh()
        render_available.refresh()

    async def _update_model(u):
        rel = u["release"]
        if rel.get("results_may_change"):
            with ui.dialog() as dlg, ui.card().classes("max-w-md"):
                ui.label(f"Update {u['name']} to {rel['release']}?").classes("text-h6")
                ui.label("Detections may differ from those of release "
                         f"{u['current']}. Both releases stay installed until you "
                         "remove the old one, so an analysis can be finished with "
                         "the release it started with.").classes("text-sm")
                with ui.row().classes("justify-end w-full"):
                    ui.button("Cancel", on_click=lambda: dlg.submit(False)).props("flat no-caps")
                    ui.button("Update", on_click=lambda: dlg.submit(True)).props("no-caps")
            if not await dlg:
                return
        cur = next(e for e in registration.read_installed()
                   if e["id"] == u["id"] and e["release"] == u["current"])
        from installer.worker import request_for
        req = request_for(page["catalog"], u["id"], u["variant"],
                          str(Path(cur["root"]).parent), rel["release"])
        req["replaces"] = u["current"]
        jobs.new_job("install", req)
        start_worker()
        ui.notify(f"Updating {u['name']} to {rel['release']}; {u['current']} stays "
                  "installed until you remove it.", type="info", multi_line=True)
        render_jobs.refresh()
        render_updates.refresh()

    def _reselect_after_update(job):
        """A model update finished: if the replaced release was the selected
        model, select the new one (031 FR-020). Every page's select then
        refreshes through on_models_changed."""
        try:
            req = json.loads((jobs.job_dir(job["id"]) / "request.json")
                             .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        old = req.get("replaces")
        if not old or job.get("kind") != "install":
            return
        suffix = "-cpu" if req.get("variant") == "cpu" else ""
        for b in req.get("entry", {}).get("backends", []):
            if state.det_model == f"{b['name']}@{old}{suffix}":
                state.det_model = f"{b['name']}@{req['release']}{suffix}"
        render_updates.refresh()

    with updates_box:
        render_updates()
    with jobs_box:
        render_jobs()
    with installed_box:
        render_installed()
    with available_box:
        render_available()
    with folder_box:
        render_folder()

    async def refresh_list():
        """Opening the tab is a user action: fetch the list (FR-006)."""
        if page["fetching"]:
            return
        page["fetching"] = True
        try:
            if not _START_CHECK["done"]:
                # Once per run of the app, after the first page (FR-043).
                _START_CHECK["done"] = True
                from installer import health
                broken = await run.io_bound(health.check_all)
                for b in broken:
                    print(f"[models] {b['id']} {b['release']} needs repair: {b['reason']}")
            res = await run.io_bound(catalog.fetch_list)
        finally:
            page["fetching"] = False
        page["catalog"] = res.data
        if res.message:
            list_status.set_text(res.message)
        elif res.data:
            list_status.set_text(f"Model list from {res.data.get('published', '?')}.")
        render_updates.refresh()
        render_installed.refresh()
        render_available.refresh()

    def tick():
        """Re-draw the job cards when a job file changed; on a finished job,
        reload the registry and every page's model select (FR-035)."""
        active = _active_jobs()
        # A worker that died (killed, crashed) leaves its job "running": look
        # every 10 s, so the card offers Resume instead of a clock that runs on.
        if any(j["state"] in ("running", "paused") for j in active) \
                and time.time() - page.get("alive_checked", 0) > 10:
            page["alive_checked"] = time.time()
            if jobs.detect_interrupted():
                active = _active_jobs()
        # A long conda step writes nothing for minutes; redraw running jobs
        # every 5 s anyway so the elapsed time moves (FR-027).
        tick5 = int(time.time() // 5) if any(j["state"] == "running" for j in active) else 0
        snap = (tick5,) + tuple((j["id"], j["state"], j.get("heartbeat")) for j in active)
        if snap != page["last_jobs"]:
            page["last_jobs"] = snap
            render_jobs.refresh()
        done = [j for j in jobs.all_jobs() if j["state"] == "done"
                and j["id"] not in page["known_done"]]
        for j in done:
            page["known_done"].add(j["id"])
            try:
                from detectors.registry import reload_installed_backends
                reload_installed_backends()
            except Exception as ex:
                print(f"[models] registry reload failed: {ex}")
            _reselect_after_update(j)
            on_models_changed()
            name = j.get("display_name") or j.get("model")
            if j.get("kind") == "uninstall":
                msg = (f"{name} removed; {human_size(j.get('freed_bytes', 0))} freed. "
                       "Results made with it are kept.")
            elif j.get("kind") == "repair":
                msg = f"{name} is repaired."
            elif j.get("kind") == "app-update":
                msg = (f"PebbleMapper {j.get('release')} is ready: it starts at the next "
                       "start, or now with Restart now on the Models tab.")
                render_updates.refresh()
            else:
                msg = f"{name} is ready — choose it under Detection model."
            ui.notify(msg, type="positive", timeout=10000)
            render_installed.refresh()
            render_available.refresh()

    ui.timer(1.0, tick)
    if register_tab_refresh:
        register_tab_refresh("Models", lambda: background_tasks.create(refresh_list()))
    ui.timer(0.1, refresh_list, once=True)
