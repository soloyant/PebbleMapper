"""Homogenization static assertions over gui/app.py source text.

These prove that the job-queue and log-console homogenization *actually
took* — shared components are used and the old ad-hoc patterns are gone.
"Looks the same" is not evidence; these file-text assertions ARE. (The one
console-less tab is Digitize, which is not a job-queue tab.)

No running GUI is needed: every assertion reads gui/app.py / job_queue.py as
text and applies a regex.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PY = REPO_ROOT / "gui" / "app.py"
JOB_QUEUE_PY = REPO_ROOT / "gui" / "components" / "job_queue.py"


def _app_src() -> str:
    return APP_PY.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop whole-line ``#`` comments so a regex doesn't match a literal that
    only survives inside an explanatory comment. Cheap and good enough for the
    string-literal checks below (does not handle inline ``#`` after code, which
    is acceptable for these assertions)."""
    return "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))


# --------------------------------------------------------------------------- #
#  A — Job queues                                                              #
# --------------------------------------------------------------------------- #
def test_status_color_map_lives_only_in_job_queue_component():
    """Exactly ONE status->color mapping, and it lives in job_queue.py."""
    assert JOB_QUEUE_PY.exists(), (
        "gui/components/job_queue.py does not exist — not landed")
    # The inline status-color dict literal (e.g. "running": "blue") must be
    # gone from app.py; it should live only in the shared component.
    app_hits = re.findall(r'["\']running["\']\s*:\s*["\'][a-zA-Z#]', _app_src())
    assert not app_hits, (
        f"gui/app.py still contains inline status-color dict literal(s): "
        f"{app_hits} — A homogenization not complete")
    component_src = JOB_QUEUE_PY.read_text(encoding="utf-8")
    assert re.search(r'["\']running["\']\s*:\s*["\']', component_src), (
        "the single status->color mapping was not found in job_queue.py")


def test_no_inline_status_color_dict_in_app():
    """app.py contains no inline status-color dict."""
    src = _strip_comments(_app_src())
    assert not re.search(r'"running"\s*:\s*"blue"', src), (
        'gui/app.py still declares an inline "running": "blue" status map')


def test_all_six_tab_builders_use_shared_render_queue():
    """All six queue tabs reference the shared render_queue component.

    The six queue-bearing tabs are Detection, Rasterize, Merge, Map, Zonal,
    Validate. They must all route through the shared
    renderer rather than their six private ``_render_queue`` / ``_render_map_
    queue`` / ``_rebuild_job_list`` functions.
    """
    src = _app_src()
    n = len(re.findall(r"render_queue\s*\(", src))
    assert n >= 6, (
        f"expected >= 6 calls to the shared render_queue (one per queue tab); "
        f"found {n} — A homogenization not complete")
    # The old per-tab private renderers should be gone.
    for stale in ("_render_map_queue", "_rebuild_job_list"):
        assert stale not in src, (
            f"stale private queue renderer '{stale}' still present in app.py")


# --------------------------------------------------------------------------- #
#  B — Log console                                                            #
# --------------------------------------------------------------------------- #
def test_every_console_goes_through_build_log_console():
    """No inline ``ui.log(...)`` re-declares the #1e2530 background.

    After B, every console is created via ``build_log_console``; no tab inlines
    a ``ui.log(...)`` whose ``.style(...)`` re-declares the dark background
    literal ``#1e2530``.
    """
    src = _app_src()
    assert "build_log_console" in src, (
        "build_log_console is not referenced in app.py — B not landed")
    # Find each ui.log( ... ) call region and assert it does not carry an
    # inline #1e2530 background style.
    offenders = []
    for m in re.finditer(r"ui\.log\(", src):
        window = src[m.start():m.start() + 400]
        if "#1e2530" in window:
            offenders.append(src[:m.start()].count("\n") + 1)
    assert not offenders, (
        f"inline ui.log(...) with hardcoded #1e2530 background at line(s) "
        f"{offenders} — should route through build_log_console")


def test_legacy_text_color_literal_gone():
    """The legacy Zonal text colour ``#d0d8e8`` appears nowhere."""
    assert "#d0d8e8" not in _app_src(), (
        "legacy console text colour #d0d8e8 still present in app.py — "
        "B homogenization not complete")


def test_validate_tab_builds_a_console():
    """The Validate tab builds a console via the shared builder.

    The Validate tab — which runs jobs — should build its log console through
    ``build_log_console`` like the other queue tabs. We assert a console builder
    is invoked within the Validate tab builder's own source span (bounded to the
    next top-level ``def build_*_tab`` so we don't bleed into neighbouring tabs).
    """
    src = _app_src()
    m = re.search(r"^def build_validate_tab\b", src, re.MULTILINE)
    assert m is not None, "could not locate build_validate_tab in app.py"
    start = m.start()
    nxt = re.search(r"^def build_\w+_tab\b", src[start + 1:], re.MULTILINE)
    end = (start + 1 + nxt.start()) if nxt else len(src)
    span = src[start:end]
    assert "build_log_console" in span, (
        "build_validate_tab does not build a log console via build_log_console — "
        "B homogenization incomplete")


# --------------------------------------------------------------------------- #
#  Dark-panel palette centralisation (backlog dedup)                          #
# --------------------------------------------------------------------------- #
STYLES_PY = REPO_ROOT / "gui" / "components" / "styles.py"


def test_dark_panel_palette_centralized_in_styles():
    """The dark UI palette lives only in gui/components/styles.py.

    Dark-mode colours were once hand-copied 10+ times across app.py.
    They are now named constants (PANEL_BG/BORDER/TEXT/MUTED/MUTED_2) in
    styles.py; no raw dark-palette hex literal should remain in app.py.
    """
    assert STYLES_PY.exists(), "gui/components/styles.py missing"
    styles_src = STYLES_PY.read_text(encoding="utf-8")
    app_src = _app_src()
    for hexlit in ("#1e2530", "#3a4555", "#e6e6e6", "#6b7a8f", "#8fa0b8"):
        assert hexlit in styles_src, f"{hexlit} not defined in styles.py"
        assert hexlit not in app_src, (
            f"{hexlit} still hardcoded in app.py — should use the styles.* constant")
