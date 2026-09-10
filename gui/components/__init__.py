"""Reusable NiceGUI building blocks for the PebbleMapper GUI.

Components here must not import ``gui.app`` (circular import); they depend
only on NiceGUI and the standard library.
"""
from .job_queue import STATUS_COLORS, status_color, render_queue
from .log_console import LOG_STYLE, DEFAULT_MAX_LINES, build_log_console

__all__ = [
    "STATUS_COLORS", "status_color", "render_queue",
    "LOG_STYLE", "DEFAULT_MAX_LINES", "build_log_console",
]
