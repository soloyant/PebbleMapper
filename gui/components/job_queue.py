"""Shared job-queue renderer for the PebbleMapper GUI.

One builder, :func:`render_queue`, and one status->colour vocabulary for every
tab that runs batched jobs, with extension slots for the tab-specific per-row
controls. Must not import ``gui.app`` (circular import).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from nicegui import ui

# Single status vocabulary; unknown statuses fall back to grey via status_color().
STATUS_COLORS = {
    "queued": "grey",
    "pending": "grey",
    "running": "blue",
    "done": "green",
    "stopped": "orange",
    "error": "red",
}

#: Card style shared by every queue.
CARD_STYLE = "background:#fafafa; border:1px solid #ddd;"

Job = Any
TextFn = Callable[[Job, int], str]
RowFn = Callable[[Job, int], None]


def status_color(status: Optional[str]) -> str:
    """Return the badge color for ``status``; grey for unknown/missing."""
    return STATUS_COLORS.get(status or "", "grey")


def render_queue(
    container,
    jobs: Sequence[Job],
    *,
    title: str = "Queued jobs",
    count_label=None,
    empty_text: Optional[str] = None,
    primary_text: TextFn,
    params_text: Optional[TextFn] = None,
    render_extra: Optional[RowFn] = None,
    render_header_extra: Optional[Callable[[], None]] = None,
    render_row_footer: Optional[RowFn] = None,
    on_delete: Optional[Callable[[int], None]] = None,
    deletable_when: Optional[Callable[[Job], bool]] = None,
    default_status: str = "pending",
    render_delete: Optional[RowFn] = None,
) -> None:
    """Clear ``container`` and rebuild the job queue inside it.

    The builder owns the container: it calls ``container.clear()`` and creates
    the card + rows within ``with container:`` itself, so callers must not
    wrap a second ``@ui.refreshable`` around it.

    Parameters
    ----------
    container:
        A NiceGUI element (typically ``ui.column``) to clear and fill.
    jobs:
        The list of job dicts to render (one row each).
    title:
        Bold header label for the card.
    count_label:
        Optional ``ui.label`` whose text is set to ``"{n} queued"`` when
        non-empty and ``""`` when empty. ``None`` to skip.
    empty_text:
        Optional italic, muted placeholder rendered when ``jobs`` is empty.
    primary_text:
        ``primary_text(job, idx) -> str`` supplies the main row label.
    params_text:
        Optional ``params_text(job, idx) -> str`` for the muted params label.
    render_extra:
        Optional ``render_extra(job, idx)`` invoked inside the row after the
        params label, before the error text / delete button.
    render_header_extra:
        Optional ``render_header_extra()`` invoked inside the header row after
        the title.
    render_row_footer:
        Optional ``render_row_footer(job, idx)`` invoked after the row, at card
        level, for a full-width footer line.
    on_delete:
        Optional ``on_delete(idx)`` wired to a trailing delete icon button,
        shown only when ``deletable_when(job)`` is True. Ignored if
        ``render_delete`` is supplied.
    deletable_when:
        Optional predicate ``deletable_when(job) -> bool``; defaults to
        "status != 'running'".
    default_status:
        Status assumed when a job dict has no ``"status"`` key.
    render_delete:
        Optional ``render_delete(job, idx)`` that fully owns the trailing
        controls for a row; ``on_delete`` / ``deletable_when`` are then unused.
    """
    container.clear()
    n = len(jobs)

    if count_label is not None:
        count_label.set_text(f"{n} queued" if n else "")

    if n == 0:
        if empty_text:
            with container:
                ui.label(empty_text).classes("text-grey-6 italic m-3")
        return

    if deletable_when is None:
        def deletable_when(job):  # noqa: E306 - small inline default
            return job.get("status", default_status) != "running"

    with container:
        with ui.card().classes("w-full").style(CARD_STYLE):
            with ui.row().classes("items-center gap-3 w-full"):
                ui.label(title).classes("text-sm font-bold")
                if render_header_extra is not None:
                    render_header_extra()

            for idx, job in enumerate(jobs):
                status = job.get("status", default_status)
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.badge(status, color=status_color(status))
                    ui.label(f"#{idx + 1}").classes("font-mono w-8")
                    ui.label(primary_text(job, idx)).classes("text-xs")
                    if params_text is not None:
                        ui.label(params_text(job, idx)) \
                            .classes("text-xs text-grey-7")
                    if render_extra is not None:
                        render_extra(job, idx)
                    if status == "error":
                        ui.label(job.get("error", "?")) \
                            .classes("text-xs text-red-7 truncate")
                    if render_delete is not None:
                        render_delete(job, idx)
                    elif on_delete is not None and deletable_when(job):
                        # Default arg binds idx per iteration.
                        def _del(_e=None, _idx=idx):
                            on_delete(_idx)
                        ui.button(icon="delete", on_click=_del) \
                            .props("flat dense")

                if render_row_footer is not None:
                    render_row_footer(job, idx)
