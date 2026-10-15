"""Keep the computer awake while a job runs (030 FR-029, research R4).

``PowerSetRequest`` with *system required* and *execution required*, falling
back to ``SetThreadExecutionState``. Neither needs administrator rights. On a
Modern Standby laptop on battery, Windows ends both after the sleep timeout;
the Models tab warns before such an install starts.
"""
from __future__ import annotations

import ctypes
import os

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
PowerRequestSystemRequired = 1
PowerRequestExecutionRequired = 3


class _ReasonContext(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("Flags", ctypes.c_ulong),
                ("SimpleReasonString", ctypes.c_wchar_p)]


class KeepAwake:
    """``with KeepAwake("Installing OrthoSAM"):`` -- released on exit, even
    on an exception."""

    def __init__(self, reason: str = "PebbleMapper is installing a model"):
        self.reason = reason
        self._handle = None
        self._fallback = False
        self.active = False

    def __enter__(self):
        if os.name != "nt":
            return self
        k32 = ctypes.windll.kernel32
        try:
            ctx = _ReasonContext(POWER_REQUEST_CONTEXT_VERSION,
                                 POWER_REQUEST_CONTEXT_SIMPLE_STRING, self.reason)
            k32.PowerCreateRequest.restype = ctypes.c_void_p
            h = k32.PowerCreateRequest(ctypes.byref(ctx))
            if h and h != ctypes.c_void_p(-1).value:
                ok1 = k32.PowerSetRequest(ctypes.c_void_p(h), PowerRequestSystemRequired)
                ok2 = k32.PowerSetRequest(ctypes.c_void_p(h), PowerRequestExecutionRequired)
                self._handle = h
                self.active = bool(ok1 or ok2)
        except Exception:
            self._handle = None
        if not self.active:
            try:
                self._fallback = bool(k32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
                self.active = self._fallback
            except Exception:
                pass
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            return False
        k32 = ctypes.windll.kernel32
        if self._handle:
            for kind in (PowerRequestSystemRequired, PowerRequestExecutionRequired):
                try:
                    k32.PowerClearRequest(ctypes.c_void_p(self._handle), kind)
                except Exception:
                    pass
            try:
                k32.CloseHandle(ctypes.c_void_p(self._handle))
            except Exception:
                pass
            self._handle = None
        if self._fallback:
            try:
                k32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass
        self.active = False
        return False
