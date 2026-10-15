"""What this computer can do: graphics card, disk, battery, connection, folders.

Read once when the Models tab needs it (030 FR-017–FR-022, research R3–R7).
Every probe that fails says so in its result rather than guessing: a missing
``nvidia-smi`` means "no NVIDIA card", an unreadable battery means "unknown".
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from installer.errors import human_size

MAX_PATH_CHARS = 259
LOW_BATTERY = 15

# Compute capability by GPU name, for drivers older than R510 whose
# nvidia-smi does not report it (research R3). Only what the reference
# hardware and common field laptops carry; an unknown card is "unknown".
_CC_BY_NAME = {
    "quadro p600": 6.1, "quadro p1000": 6.1, "quadro p2000": 6.1,
    "geforce gtx 1050": 6.1, "geforce gtx 1060": 6.1, "geforce gtx 1070": 6.1,
    "geforce gtx 1080": 6.1, "geforce mx150": 6.1, "geforce mx250": 6.1,
    "geforce gtx 1650": 7.5, "geforce gtx 1660": 7.5, "quadro t1000": 7.5,
    "quadro t2000": 7.5, "geforce rtx 2060": 7.5, "geforce rtx 2070": 7.5,
    "geforce rtx 3050": 8.6, "geforce rtx 3060": 8.6, "geforce rtx 3070": 8.6,
    "geforce rtx 4050": 8.9, "geforce rtx 4060": 8.9, "geforce rtx 4070": 8.9,
}


# --------------------------------------------------------------- graphics
@dataclass
class Gpu:
    name: str
    driver: str
    compute_cap: Optional[float]
    vram_mb: int


def _nvidia_smi() -> Optional[str]:
    for p in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
              Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
        if p.is_file():
            return str(p)
    return shutil.which("nvidia-smi")


def parse_smi(text: str) -> List[Gpu]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, driver, cc, mem = parts[:4]
        try:
            ccf = float(cc)
        except ValueError:
            ccf = _CC_BY_NAME.get(name.lower().replace("nvidia ", "").strip())
        try:
            vram = int(float(mem))
        except ValueError:
            vram = 0
        gpus.append(Gpu(name, driver, ccf, vram))
    return gpus


def gpus() -> List[Gpu]:
    exe = _nvidia_smi()
    if not exe:
        return []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    q = ["--query-gpu=name,driver_version,compute_cap,memory.total",
         "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run([exe, *q], capture_output=True, text=True, timeout=20,
                             creationflags=flags)
        if out.returncode != 0 and "compute_cap" in (out.stdout + out.stderr):
            q[0] = "--query-gpu=name,driver_version,name,memory.total"
            out = subprocess.run([exe, *q], capture_output=True, text=True,
                                 timeout=20, creationflags=flags)
        return parse_smi(out.stdout) if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def _ver(v: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (0,)


@dataclass
class Verdict:
    variant: Optional[str]      # "gpu", "cpu" or None (cannot run)
    text: str                   # the sentence the Models tab shows
    reason: str = ""


def verdict(release: dict, cards: List[Gpu], force_cpu: bool = False) -> Verdict:
    """Graphics card, processor, or cannot run (FR-019)."""
    variants = release.get("variants", {})
    gpu_v, cpu_v = variants.get("gpu"), variants.get("cpu")
    why = ""
    if gpu_v and not force_cpu:
        if not cards:
            why = "it needs an NVIDIA graphics card"
        else:
            g = max(cards, key=lambda c: c.vram_mb)
            lo, hi = (gpu_v.get("compute_cap") or [0, 99])
            if gpu_v.get("min_driver") and _ver(g.driver) < _ver(gpu_v["min_driver"]):
                why = (f"the graphics driver ({g.driver}) is older than "
                       f"{gpu_v['min_driver']}; update it from NVIDIA's site")
            elif g.compute_cap is None:
                why = f"the graphics card ({g.name}) could not be identified"
            elif not (lo <= g.compute_cap <= hi):
                why = f"the graphics card ({g.name}) is not supported by this model"
            elif gpu_v.get("min_vram_mb") and g.vram_mb < gpu_v["min_vram_mb"]:
                why = (f"the graphics card has {g.vram_mb / 1024:.0f} GB of memory; "
                       f"this model needs {gpu_v['min_vram_mb'] / 1024:.0f} GB")
            else:
                return Verdict("gpu", "Will run on your graphics card")
    if cpu_v:
        n = cpu_v.get("speed_vs_gpu")
        txt = ("Will run on the processor, about "
               f"{int(n)} times slower" if n else "Will run on the processor")
        return Verdict("cpu", txt, why)
    return Verdict(None, f"Cannot run on this computer: {why}" if why
                   else "Cannot run on this computer", why)


# ------------------------------------------------------------------- disk
def drive_of(path: Path) -> str:
    return Path(path).resolve().drive or str(Path(path).resolve().anchor)


def disk_needs(models_dir: Path, variant: dict) -> Dict[str, int]:
    """Peak bytes per drive (FR-020). Downloads, caches, temporary files and
    the result all go under the models folder, so its drive carries the peak."""
    return {drive_of(models_dir): int(variant.get("peak_bytes", 0))}


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def disk_shortfall(models_dir: Path, variant: dict) -> Dict[str, int]:
    """``{drive: missing bytes}``; empty when the install fits."""
    short = {}
    for drive, need in disk_needs(models_dir, variant).items():
        free = free_bytes(Path(models_dir))
        if free < need:
            short[drive] = need - free
    return short


_USB_CACHE: Optional[set] = None


def usb_drives() -> set:
    """Drive letters on USB disks. Windows reports a USB hard drive as a
    fixed disk, so the bus type is asked of the Storage module (~6 s once,
    cached). Unknown → empty set."""
    global _USB_CACHE
    if _USB_CACHE is not None:
        return _USB_CACHE
    found = set()
    if os.name == "nt":
        ps = ("Get-Partition | Where-Object DriveLetter | ForEach-Object { "
              "'{0} {1}' -f $_.DriveLetter, (Get-Disk -Number $_.DiskNumber).BusType }")
        try:
            out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                                  "-Command", ps], capture_output=True, text=True,
                                 timeout=60,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in out.stdout.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].upper() == "USB":
                    found.add(parts[0].upper())
        except (OSError, subprocess.SubprocessError):
            pass
    _USB_CACHE = found
    return found


def default_models_dir() -> Path:
    """``PebbleMapper-models`` at the root of the internal fixed drive with
    the most free space (FR-017). USB disks are skipped: models there stop
    working whenever the drive is unplugged."""
    best, best_free = Path.home().anchor, -1
    usb = usb_drives()
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if letter in usb:
            continue
        if os.name == "nt":
            try:
                if ctypes.windll.kernel32.GetDriveTypeW(root) != 3:   # DRIVE_FIXED
                    continue
            except Exception:
                continue
        if not os.path.exists(root):
            continue
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            continue
        if free > best_free:
            best, best_free = root, free
    return Path(best) / "PebbleMapper-models"


# ---------------------------------------------------------------- battery
@dataclass
class Power:
    on_battery: Optional[bool]          # None = unknown
    percent: Optional[int]


def power() -> Power:
    if os.name != "nt":
        return Power(None, None)

    class _SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = _SPS()
    try:
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return Power(None, None)
    except Exception:
        return Power(None, None)
    if s.BatteryFlag == 128:                       # no battery
        return Power(False, None)
    on_batt = {0: True, 1: False}.get(s.ACLineStatus)
    pct = None if s.BatteryLifePercent == 255 else int(s.BatteryLifePercent)
    return Power(on_batt, pct)


def battery_low(p: Power) -> bool:
    return bool(p.on_battery) and p.percent is not None and p.percent < LOW_BATTERY


def modern_standby() -> bool:
    """True on laptops using S0 low-power idle, where Windows ends a
    keep-awake request on battery after the sleep timeout (R4)."""
    if os.name != "nt":
        return False

    class _SPC(ctypes.Structure):          # SYSTEM_POWER_CAPABILITIES, first 19 BOOLEANs
        _fields_ = [(n, ctypes.c_ubyte) for n in (
            "PowerButtonPresent", "SleepButtonPresent", "LidPresent", "SystemS1",
            "SystemS2", "SystemS3", "SystemS4", "SystemS5", "HiberFilePresent",
            "FullWake", "VideoDimPresent", "ApmPresent", "UpsPresent",
            "ThermalControl", "ProcessorThrottle", "ProcessorMinThrottle",
            "ProcessorMaxThrottle", "FastSystemS4", "Hiberboot")] + [
            ("WakeAlarmPresent", ctypes.c_ubyte), ("AoAc", ctypes.c_ubyte),
            ("_rest", ctypes.c_ubyte * 64)]
    caps = _SPC()
    try:
        if not ctypes.windll.powrprof.GetPwrCapabilities(ctypes.byref(caps)):
            return False
    except Exception:
        return False
    return bool(caps.AoAc)


# ------------------------------------------------------------- connection
def parse_cost(text: str) -> Optional[bool]:
    """``True`` metered, ``False`` not, ``None`` offline/unknown."""
    t = (text or "").strip()
    if not t:
        return None
    return t.lower() in ("fixed", "variable")


def metered() -> Optional[bool]:
    """One Windows PowerShell 5.1 call to WinRT (R6); ~0.7 s, so call it once
    when Install is pressed, off the interface thread."""
    if os.name != "nt":
        return None
    ps = ("$p=[Windows.Networking.Connectivity.NetworkInformation,"
          "Windows.Networking.Connectivity,ContentType=WindowsRuntime]"
          "::GetInternetConnectionProfile(); if ($p) { $p.GetConnectionCost().NetworkCostType }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                              "-Command", ps], capture_output=True, text=True,
                             timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return parse_cost(out.stdout)
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------- folders
_CLOUD_ATTRS = 0x40000 | 0x80000 | 0x100000 | 0x400000 | 0x1000


def _onedrive_roots() -> List[Path]:
    roots = [os.environ.get(k) for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\OneDrive\Accounts") as k:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        with winreg.OpenKey(k, sub) as s:
                            roots.append(winreg.QueryValueEx(s, "UserFolder")[0])
                    except OSError:
                        pass
        except OSError:
            pass
    return [Path(r) for r in roots if r]


def _dropbox_roots() -> List[Path]:
    out = []
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA")):
        if not base:
            continue
        f = Path(base) / "Dropbox" / "info.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out += [Path(v["path"]) for v in data.values() if isinstance(v, dict) and "path" in v]
        except (OSError, ValueError):
            pass
    return out


def _gdrive_roots() -> List[Path]:
    out = [Path.home() / "My Drive"]
    if os.name == "nt":
        buf = ctypes.create_unicode_buffer(261)
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            try:
                if ctypes.windll.kernel32.GetVolumeInformationW(
                        root, buf, 261, None, None, None, None, 0) and buf.value == "Google Drive":
                    out.append(Path(root))
            except Exception:
                pass
    return out


def _under(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


def _long_paths_enabled() -> bool:
    if os.name != "nt":
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return bool(winreg.QueryValueEx(k, "LongPathsEnabled")[0])
    except OSError:
        return False


@dataclass
class FolderCheck:
    ok: bool
    reason: str = ""            # a refusal, in plain words
    warning: str = ""           # a folder the app could not classify


def check_models_folder(folder: Path, longest_relative: int = 0) -> FolderCheck:
    """FR-018. Long paths are judged at 259 characters even when Windows long
    paths are on: conda's native tools ignore that setting (R7)."""
    folder = Path(folder)
    for name, roots in (("OneDrive", _onedrive_roots()), ("Dropbox", _dropbox_roots()),
                        ("Google Drive", _gdrive_roots())):
        if any(_under(folder, r) for r in roots):
            return FolderCheck(False, f"it is synced by {name}")
    p = folder
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        if getattr(os.stat(p), "st_file_attributes", 0) & _CLOUD_ATTRS:
            return FolderCheck(False, "it is synced by a cloud storage program")
    except OSError:
        pass
    budget = MAX_PATH_CHARS - len(str(folder)) - 1 - 20     # "\<id>-<rel>-g\" margin
    if longest_relative and budget < longest_relative:
        return FolderCheck(False, f"its path is too long for this model by "
                                  f"{longest_relative - budget} characters; choose a shorter one")
    letter = (folder.drive or "").rstrip(":").upper()
    if letter and letter in usb_drives():
        return FolderCheck(True, warning=(
            f"Drive {letter}: is a USB drive: models installed there stop working "
            "while it is unplugged, and are lost if it fails."))
    return FolderCheck(True)


def describe_need(models_dir: Path, variant: dict) -> str:
    return ", ".join(f"{human_size(n)} on {d}" for d, n in disk_needs(models_dir, variant).items())
