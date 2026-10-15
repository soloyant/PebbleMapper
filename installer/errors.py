"""What went wrong, in plain words, and what to do next (030 FR-037).

Every failure the installer can report is one of the classes below. Each
renders as one sentence saying what failed and one saying what to do, neither
needing a terminal. On screen the app never says "variant", "lock", "prefix"
or "environment" (FR-028); the tests check every class for those words.
"""
from __future__ import annotations

from typing import Dict, Tuple

ISSUES_URL = "https://github.com/soloyant/PebbleMapper/issues"

# class -> (what failed, what to do). ``{name}`` fields come from the facts.
_MESSAGES: Dict[str, Tuple[str, str]] = {
    "disk_full": (
        "There is not enough free space on drive {drive}: {missing} more is needed.",
        "Free {missing} on drive {drive}, or choose a models folder on another drive, then try again."),
    "network": (
        "PebbleMapper could not reach {host}.",
        "Check the internet connection, then click Resume. If you are behind a proxy, ask your IT service to allow {host}."),
    "clock": (
        "The secure connection to {host} was refused, which usually means this computer's date or time is wrong.",
        "Check the computer's date and time, then click Resume."),
    "damaged_download": (
        "The download of {file} was damaged.",
        "Click Resume to download it again."),
    "published_changed": (
        "The published file {file} is no longer the one the model list names.",
        "Nothing can be done on this computer: the maintainer must update the model list. Please report it."),
    "cannot_run": (
        "This model cannot run on this computer: {reason}.",
        "Choose another model, or install it on a computer that meets its needs."),
    "conda_too_old": (
        "The conda that came with this PebbleMapper installation is too old to install models.",
        "Run the PebbleMapper installer again, then try again."),
    "starter_too_old": (
        "This update needs a newer PebbleMapper starter than this installation has.",
        "Run the PebbleMapper installer again; it keeps your data and installs the update."),
    "folder_refused": (
        "The models folder {folder} cannot be used: {reason}.",
        "Choose another folder on the Models tab, then try again."),
    "timeout": (
        "The step '{step}' took far longer than it should and was stopped.",
        "Click Resume to try again. If it stops again, build a diagnostic bundle and report it."),
    "check_run_failed": (
        "The model was installed but its test run on a sample photograph did not succeed.",
        "Click Remove, then install again. If it fails again, build a diagnostic bundle and report it."),
    "file_locked": (
        "The file {path} was kept locked by another program.",
        "Close programs that may use it, then click Resume. If an antivirus keeps blocking it, exclude the models folder from scanning."),
    "unknown": (
        "The install stopped on an unexpected error: {detail}.",
        "Build a diagnostic bundle and report it."),
}

CLASSES = tuple(_MESSAGES)


class InstallError(Exception):
    """A failure the user is told about. ``cls`` is one of :data:`CLASSES`;
    ``facts`` fill the sentence. Never a value the happy path could return
    (Principle I: sentinels must not be attainable)."""

    def __init__(self, cls: str, **facts):
        if cls not in _MESSAGES:
            raise ValueError(f"unknown install error class {cls!r}")
        self.cls = cls
        self.facts = facts
        super().__init__(self.sentence())

    def sentence(self) -> str:
        return _render(_MESSAGES[self.cls][0], self.facts)

    def action(self) -> str:
        return _render(_MESSAGES[self.cls][1], self.facts)

    def as_dict(self) -> dict:
        return {"class": self.cls, "message": self.sentence(),
                "action": self.action(),
                "facts": {k: str(v) for k, v in self.facts.items()}}


class _Blank(dict):
    def __missing__(self, key):
        return "?"


def _render(template: str, facts: dict) -> str:
    return template.format_map(_Blank({k: str(v) for k, v in facts.items()}))


def human_size(n_bytes: float) -> str:
    """``1.8 GB``, ``340 MB``: sizes as the Models tab writes them."""
    n = float(n_bytes)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("bytes", "KB", "MB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"
