"""Shared builder for the per-tab log / prompt-console windows.

Defines the single style vocabulary (``LOG_STYLE``) and the factory
(``build_log_console``) so every console shares one look. Must not import
``gui.app`` (circular import).
"""
from nicegui import ui

from gui.components.styles import PANEL_BG, PANEL_TEXT

# Colours come from the shared dark-panel palette so consoles and inspector panels match.
LOG_STYLE = (
    f"background:{PANEL_BG}; color:{PANEL_TEXT}; "
    "font-family:monospace; font-size:0.85em;"
)

# Default scrollback; tabs with heavy tile-by-tile streaming override it.
DEFAULT_MAX_LINES = 1000


def build_log_console(*, max_lines=DEFAULT_MAX_LINES, height="h-40",
                      extra_classes=""):
    """Create a styled ``ui.log`` console and return the element.

    Parameters
    ----------
    max_lines:
        Scrollback line cap.
    height:
        A Tailwind/Quasar height utility class (e.g. ``"h-32"``, ``"h-64"``).
    extra_classes:
        Extra utility classes appended after the standard ones; use this
        rather than re-inlining ``.style(...)`` for a per-site tweak.

    The returned element is a plain ``ui.log`` (``.push()`` / ``.clear()``
    work as usual).
    """
    return ui.log(max_lines=max_lines).classes(
        f"w-full {height} mt-2 {extra_classes}".strip()
    ).style(LOG_STYLE)
