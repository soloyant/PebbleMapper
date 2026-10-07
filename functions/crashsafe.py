"""Crash observability for the GUI process.

- Trace arming: ``faulthandler`` writes per-thread stacks into a timestamped
  ``crash_*.log`` in the active project's ``logs/`` (app-level ``logs/`` when
  no project is active). On Windows a stack overflow can defeat any in-process
  handler, so ``functions/worker.py`` appends its own post-mortem here when a
  job subprocess dies traceless.
- Session breadcrumb: ``app_home()/last-session.json`` records where the
  current trace lives and whether shutdown was clean, so the next boot can
  announce a crash.
- Test fault injection: env-gated, inert without
  ``PEBBLEMAPPER_ENABLE_TEST_FAULTS=1``.

Nothing here may block the tool: every recorder failure degrades to a
returned ``None``/message for the caller to surface once.
"""

from __future__ import annotations

import datetime as _dt
import faulthandler
import json as _json
import os as _os
import sys as _sys
from pathlib import Path
from typing import Optional

from functions.layout import app_home

TRACE_PREFIX = "crash_"
TRACE_SUFFIX = ".log"
TRACE_KEEP = 10                      # per location
BREADCRUMB_NAME = "last-session.json"
BREADCRUMB_VERSION = 1

# faulthandler writes through the trace file's fd, so the file object must
# stay referenced here.
_state = {
    "file": None,
    "path": None,
    "prev_crash": None,     # payload from previous_session_crashed(), set by boot()
    "announced": False,
    "degraded_reason": None,
}


# --- Locations ---

def app_log_dir() -> Path:
    """The no-project fallback location for crash traces."""
    return app_home() / "logs"


def current_trace_path() -> Optional[Path]:
    return _state["path"]


def degraded_reason() -> Optional[str]:
    """One-line reason when the recorder could not do its job, else None."""
    return _state["degraded_reason"]


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


# --- Trace arming / re-pointing / pruning ---

def is_empty_trace(path) -> bool:
    """True when a trace file holds only the header this module writes: the
    run it belonged to ended without a crash, so the file says nothing."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all((not line.strip()) or line.lstrip().startswith("#")
               for line in text.splitlines())


def prune_traces(trace_dir: Path, keep: int = TRACE_KEEP) -> None:
    """Keep the newest ``keep`` crash_*.log files in ``trace_dir``, and drop
    the header-only ones of earlier runs, so a clean exit leaves nothing
    behind. A trace another process still holds open cannot be removed on
    Windows, and is left alone."""
    try:
        for p in Path(trace_dir).glob(TRACE_PREFIX + "*" + TRACE_SUFFIX):
            if _state["path"] is not None and Path(p) == Path(_state["path"]):
                continue
            if is_empty_trace(p):
                try:
                    p.unlink()
                except OSError:
                    pass
    except OSError:
        pass
    try:
        traces = sorted(
            (p for p in Path(trace_dir).glob(TRACE_PREFIX + "*" + TRACE_SUFFIX)),
            key=lambda p: p.name,
        )
        for p in traces[:-keep] if keep > 0 else traces:
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def arm(trace_dir) -> Optional[Path]:
    """Point faulthandler at a fresh timestamped file under ``trace_dir``.

    Returns the trace path, or None when the location is unwritable (the
    previous target stays armed and :func:`degraded_reason` says why).
    """
    try:
        d = Path(trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{TRACE_PREFIX}{_timestamp()}{TRACE_SUFFIX}"
        f = open(path, "w", encoding="utf-8", errors="replace")
    except OSError as exc:
        _state["degraded_reason"] = (
            f"Crash traces cannot be written to {trace_dir} ({exc}). "
            "The previous trace location stays in effect."
        )
        return None
    f.write(
        f"# PebbleMapper crash trace — armed {_dt.datetime.now().isoformat()} "
        f"pid={_os.getpid()}\n"
        "# If this file ends after this header, the process died without an\n"
        "# in-process trace (on Windows a stack overflow can do that); check\n"
        "# below for a worker post-mortem block, and the console for the exit "
        "code.\n"
    )
    f.flush()
    old = _state["file"]
    faulthandler.enable(file=f, all_threads=True)
    _state["file"], _state["path"] = f, path
    _state["degraded_reason"] = None
    if old is not None:
        try:
            old.close()
        except OSError:
            pass
    prune_traces(d)
    _update_breadcrumb_trace(path)
    return path


def repoint(project_logs_dir=None) -> Optional[Path]:
    """Re-arm into the active project's logs dir (or the app fallback).
    No-op when already pointing there; safe to call on every render."""
    target = Path(project_logs_dir) if project_logs_dir else app_log_dir()
    cur = _state["path"]
    if cur is not None and Path(cur).parent == target:
        return cur
    return arm(target)


def disarm() -> None:
    """Restore faulthandler to stderr and close the trace handle (tests)."""
    try:
        faulthandler.enable(file=_sys.stderr, all_threads=True)
    except Exception:
        faulthandler.disable()
    f = _state["file"]
    _state["file"], _state["path"] = None, None
    if f is not None:
        try:
            f.close()
        except OSError:
            pass


# --- Session breadcrumb ---

def _breadcrumb_path() -> Path:
    return app_home() / BREADCRUMB_NAME


def read_breadcrumb() -> Optional[dict]:
    try:
        return _json.loads(_breadcrumb_path().read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_breadcrumb(payload: dict) -> bool:
    try:
        home = app_home()
        home.mkdir(parents=True, exist_ok=True)
        tmp = _breadcrumb_path().with_suffix(".tmp")
        tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        _os.replace(tmp, _breadcrumb_path())
        return True
    except OSError as exc:
        _state["degraded_reason"] = (
            f"The session breadcrumb cannot be written under {app_home()} "
            f"({exc}); crash announcements are unavailable this session."
        )
        return False


def write_breadcrumb(trace_path) -> bool:
    """Record this session as running-uncleanly-until-proven-otherwise."""
    return _write_breadcrumb({
        "version": BREADCRUMB_VERSION,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "pid": _os.getpid(),
        "trace_path": str(trace_path) if trace_path else "",
        "clean_exit": False,
    })


def _update_breadcrumb_trace(trace_path) -> None:
    bc = read_breadcrumb()
    if bc and bc.get("pid") == _os.getpid() and not bc.get("clean_exit"):
        bc["trace_path"] = str(trace_path)
        _write_breadcrumb(bc)


def mark_clean_exit() -> None:
    bc = read_breadcrumb()
    if bc and bc.get("pid") == _os.getpid():
        bc["clean_exit"] = True
        _write_breadcrumb(bc)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(h)
    try:
        _os.kill(pid, 0)
        return True
    except OSError:
        return False


def trace_has_evidence(trace_path) -> bool:
    """True when a session's trace holds more than the armed header -- a
    faulthandler dump or a worker post-mortem -- which is what a real crash
    leaves behind. A window closed on a healthy server leaves the header
    alone, and that is a stop, not a crash."""
    try:
        if not trace_path:
            return False
        p = Path(trace_path)
        if not p.is_file():
            return False
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip() and not line.startswith("#"):
                    return True
        return False
    except OSError:
        return False


def previous_session_crashed() -> Optional[dict]:
    """The announcement payload if the last session did not exit cleanly,
    else None.

    Requires a breadcrumb that says the exit was not clean and whose pid is no
    longer alive (a live pid is another running instance, not a crash). The
    payload's ``evidence`` says what the unclean exit was: ``"trace"`` when
    the session's trace file holds a crash dump, ``"none"`` when it holds
    only its header -- the console window was closed, the process was
    killed, the machine went down. Only the first deserves a warning.
    """
    bc = read_breadcrumb()
    if not bc or bc.get("clean_exit"):
        return None
    pid = int(bc.get("pid") or 0)
    if pid == _os.getpid() or _pid_alive(pid):
        return None
    trace = bc.get("trace_path") or ""
    return {
        "trace_path": trace,
        "started_at": bc.get("started_at") or "unknown time",
        "evidence": "trace" if trace_has_evidence(trace) else "none",
    }


# --- Closing the console window is a stop, not a crash ---

_console_handler_ref = None      # keeps the ctypes callback alive


def install_console_close_handler() -> bool:
    """On Windows, mark this session's exit clean when its console window is
    closed, the user logs off or the machine shuts down. The launcher tells
    users to close the window to stop the server, so the breadcrumb must not
    call that a crash. Ctrl+C is left to the server's own graceful shutdown,
    which reaches :func:`mark_clean_exit` through ``on_shutdown``."""
    global _console_handler_ref
    if _os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _on_console_event(event):
            console_close_marks_clean_exit(int(event))
            return False               # the default handling still ends the process

        _console_handler_ref = HANDLER(_on_console_event)
        return bool(ctypes.windll.kernel32.SetConsoleCtrlHandler(
            _console_handler_ref, True))
    except Exception:
        return False


def console_close_marks_clean_exit(event: int) -> bool:
    """What the installed handler does for ``event`` (tests call this; the
    console calls the ctypes callback)."""
    if int(event) in (2, 5, 6):
        mark_clean_exit()
        return True
    return False


# --- Boot / announcement plumbing for the GUI ---

def boot(project_logs_dir=None) -> Optional[dict]:
    """Arm everything for a fresh app session; returns the crash announcement
    payload for the previous session (or None). The previous breadcrumb must
    be read before this session overwrites it."""
    prev = previous_session_crashed()
    _state["prev_crash"] = prev
    _state["announced"] = False
    path = arm(project_logs_dir or app_log_dir())
    write_breadcrumb(path)
    install_console_close_handler()
    if prev and prev.get("evidence") == "trace":
        print(f"[crashsafe] The previous session ({prev['started_at']}) crashed "
              f"— trace: {prev['trace_path']}", flush=True)
    elif prev:
        print(f"[crashsafe] The previous session ({prev['started_at']}) stopped "
              "without a clean shutdown and left no crash trace (window closed, "
              "process ended); nothing to report.", flush=True)
    return prev


def pending_announcement() -> Optional[dict]:
    """The boot-time crash payload, until a client has shown it once."""
    if _state["announced"]:
        return None
    return _state["prev_crash"]


def mark_announced() -> None:
    _state["announced"] = True


# --- Worker post-mortem ---

def write_worker_postmortem(exit_code, op: str, envelope_path,
                            started_at: str, ended_at: str) -> Optional[Path]:
    """Append a worker-death record to the current trace location when a job
    subprocess dies without writing its result file. Returns the trace path
    written to, or None."""
    code = int(exit_code) if exit_code is not None else -1
    block = (
        "\n---- worker post-mortem ----\n"
        f"recorded_at : {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"op          : {op}\n"
        f"exit_code   : {code} (0x{code & 0xFFFFFFFF:08X})\n"
        f"envelope    : {envelope_path}\n"
        f"started_at  : {started_at}\n"
        f"ended_at    : {ended_at}\n"
        "----------------------------\n"
    )
    f = _state["file"]
    if f is not None:
        try:
            f.write(block)
            f.flush()
            return _state["path"]
        except OSError:
            pass
    # No armed trace: best-effort file in the app log dir.
    try:
        d = app_log_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{TRACE_PREFIX}{_timestamp()}{TRACE_SUFFIX}"
        path.write_text(block, encoding="utf-8")
        return path
    except OSError:
        return None


# --- NiceGUI binding-propagation hardening ---

#: (type_name, attr) pairs already warned about — warn once each.
_cycle_warned: set = set()


def harden_binding_propagation() -> bool:
    """Close a reentrancy hole in NiceGUI 2.x binding propagation.

    ``binding._propagate`` resets the ``propagation_visited`` cycle guard even
    when a cascade is already in flight, so a two-way-bound value with
    ``x != x`` (a float NaN) recurses until the native stack overflows. The
    patch reuses the active visited set on reentry and logs the first dampened
    cycle per (owner type, attribute) so the offending value can be found.

    Returns True when the patch is installed (or already was), False when
    NiceGUI is absent or its internals no longer match.
    """
    try:
        from nicegui import binding as _b
    except Exception:
        return False
    if getattr(_b, "_pm_hardened", False):
        return True
    if not all(hasattr(_b, n) for n in
               ("_propagate", "_propagate_recursively", "propagation_visited")):
        print("[crashsafe] NiceGUI binding internals changed — propagation "
              "hardening NOT installed; re-verify against this version.",
              flush=True)
        return False

    def _warn_not_self_equal(source_obj, source_name):
        key = (type(source_obj).__name__, str(source_name))
        if key in _cycle_warned:
            return
        try:
            if _b._has_attribute(source_obj, source_name):
                v = _b._get_attribute(source_obj, source_name)
                if v != v:  # NaN and friends — the recursion trigger
                    _cycle_warned.add(key)
                    print(f"[crashsafe] binding cycle dampened at "
                          f"{key[0]}.{key[1]} — the bound value compares "
                          "unequal to itself (NaN?). Harmless now, but worth "
                          "fixing at the source.", flush=True)
        except Exception:
            pass

    def _safe_propagate(source_obj, source_name):
        visited = _b.propagation_visited.get()
        if visited is not None:
            # Reentrant call from inside an active cascade: keep the guard.
            _warn_not_self_equal(source_obj, source_name)
            _b._propagate_recursively(source_obj, source_name)
            return
        token = _b.propagation_visited.set(set())
        try:
            _b._propagate_recursively(source_obj, source_name)
        finally:
            _b.propagation_visited.reset(token)

    _b._propagate = _safe_propagate
    _b._pm_hardened = True
    return True


# --- Test fault injection (ships inert) ---

def induce_test_fault(kind: str = "segv") -> str:
    """Deliberately kill this process, only under
    ``PEBBLEMAPPER_ENABLE_TEST_FAULTS=1``; otherwise refuses and says so.

    kinds: ``segv`` (native access violation), ``stack`` (native stack
    overflow, which faulthandler may miss), ``hard-exit`` (immediate
    ``os._exit``, a worker vanishing without a result file).
    """
    if _os.environ.get("PEBBLEMAPPER_ENABLE_TEST_FAULTS") != "1":
        return ("Refused: test faults are disabled. Set "
                "PEBBLEMAPPER_ENABLE_TEST_FAULTS=1 to enable (tests only).")
    if kind == "hard-exit":
        _os._exit(0xAD)
    if kind == "stack":
        try:
            faulthandler._stack_overflow()  # noqa: SLF001 — CPython test hook
        except AttributeError:
            pass  # fall through to segv on builds without the helper
    faulthandler._sigsegv()  # noqa: SLF001 — CPython test hook
    return "unreachable"


def quiet_windows_connection_resets(loop=None) -> None:
    """Keep asyncio's Windows proactor from printing a traceback every time
    a client drops a connection.

    On Windows the proactor transport calls ``socket.shutdown`` while
    closing a connection the peer has already reset, and asyncio reports
    the ConnectionResetError through the loop's exception handler as
    "Exception in callback _ProactorBasePipeTransport._call_connection_lost"
    with a full traceback: once per probe of the address, per closed tab,
    per reload. Nothing is wrong; the handler below drops
    that one report and leaves every other to the default handler."""
    import asyncio
    try:
        loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        return
    default = loop.get_exception_handler()

    def handler(lp, context):
        if is_windows_connection_reset(context):
            return
        if default is not None:
            default(lp, context)
        else:
            lp.default_exception_handler(context)
    loop.set_exception_handler(handler)


def is_windows_connection_reset(context) -> bool:
    """True for the proactor's report of a peer that reset the connection."""
    exc = context.get("exception")
    handle = str(context.get("handle", "")) + str(context.get("message", ""))
    return (isinstance(exc, ConnectionResetError)
            and "_call_connection_lost" in handle)
