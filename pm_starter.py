"""PebbleMapper's starter: applies a prepared update, then starts the app.

Run by ``PebbleMapper.cmd`` with conda's base Python, so it needs no app
environment; standard library only. Written by the installer and never moved
or replaced by an update (031 FR-012), which is why it must not import any
file of the app.

On every start (contracts/starter.md):

1. a requested return to the previous version is applied;
2. a staged update is verified file by file, then switched in: every path the
   old or the new release owns is moved by a rename on the app's volume — an
   unchanged old file to ``.pm-update/previous/``, a file the user changed is
   left beside the new one as ``<name>.mine`` — and nothing else in the
   folder is touched (projects, outputs, weights, personal files);
3. the app is started in the environment the pointer names; after a switch
   the start is provisional: if ``/pm-health`` does not answer with the new
   version, the switch is reversed and the previous version started;
4. an app that exits with :data:`RESTART_CODE` (**Restart now**) is started
   again from step 1, so a prepared update is applied without the user
   looking for the shortcut.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STARTER_VERSION = 1
POINTER = ".pm-version.json"
FILES = ".pm-files.json"
UPDATE_DIR = ".pm-update"
PARTIAL = "journal.partial"     # the moves of a switch in progress
# Never in a release's file list, so never switched by an update; only the
# installer writes them (tools/make_app_release.py STARTER_FILES).
STARTER_FILES = ("pm_starter.py", "PebbleMapper.cmd")
HEALTH_WAIT_S = 120
HEALTH_STAY_S = 30
RESTART_CODE = 75        # the app's exit code for "start me again"


# ------------------------------------------------------------------ files
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path, entry: dict) -> bool:
    try:
        return path.is_file() and path.stat().st_size == entry["size"] \
            and _sha256(path) == entry["sha256"]
    except OSError:
        return False


def _rename(src: Path, dst: Path, tries: int = 6) -> None:
    """A rename on one volume, retried while antivirus or an editor holds
    the file for a moment."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.1
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def installed_version(app: Path) -> str:
    """The version of an installation without a pointer (1.0.x)."""
    import re
    try:
        text = (app / "functions" / "__init__.py").read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def load_pointer(app: Path) -> dict:
    p = _read_json(app / POINTER, None)
    if p is None:                                # a 1.0.x installation
        p = {"starter": STARTER_VERSION, "current": installed_version(app),
             "env": "maskrcnn", "previous": None, "previous_env": None,
             "staged": None, "provisional": False, "return_to_previous": False}
    return p


def _safe(rel: str) -> str:
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ":" in rel or ".." in parts:
        raise ValueError(f"path {rel!r} leaves the app folder")
    return rel


# ----------------------------------------------------------------- switch
def verify_stage(stage: Path, new_list: dict) -> list:
    """Staged files that do not match the new file list (031 FR-013)."""
    return [f["path"] for f in new_list["files"] if not _matches(stage / _safe(f["path"]), f)]


def switch(app: Path, stage: Path, old_list: dict, new_list: dict) -> dict:
    """Move the new release in. Returns ``{"journal", "kept_mine"}``; on any
    failure every move already made is reversed and the error re-raised.

    Every move is written to ``journal.partial`` (and flushed to disk) before
    it is made, so a switch cut short by a crash or a power loss is undone at
    the next start by :func:`recover_interrupted` (031 SC-006)."""
    upd = app / UPDATE_DIR
    upd.mkdir(parents=True, exist_ok=True)
    prev = upd / "previous"
    # Only one previous version is kept; its journal goes with it, so that a
    # stale journal can never be reversed over another version's files.
    for stale in ("journal.json", "previous_files.json"):
        (upd / stale).unlink(missing_ok=True)
    if prev.exists():
        shutil.rmtree(prev)
    old = {f["path"]: f for f in old_list.get("files", [])}
    new = {f["path"]: f for f in new_list["files"]}
    journal, kept_mine = [], []
    with open(upd / PARTIAL, "w", encoding="utf-8") as fh:
        def move(kind, src, dst):
            fh.write(json.dumps([kind, str(src), str(dst)]) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            _rename(src, dst)
            journal.append([kind, str(src), str(dst)])

        def keep_mine(cur):
            mine = cur.with_name(cur.name + ".mine")
            n = 1
            while mine.exists():
                n += 1
                mine = cur.with_name(f"{cur.name}.mine{n}")
            move("mine", cur, mine)
            kept_mine.append(str(mine.relative_to(app)))

        try:
            for rel in sorted(set(old) | set(new)):
                _safe(rel)
                if rel in STARTER_FILES:
                    continue                  # whatever a list says, never the starter
                cur = app / rel
                if rel in old and cur.exists():
                    if _matches(cur, old[rel]):
                        move("moved", cur, prev / rel)
                    elif rel in new:
                        keep_mine(cur)
                    # changed by the user and gone from the new release: left alone
                elif rel not in old and rel in new and cur.exists():
                    if _matches(cur, new[rel]):
                        # already the new release's bytes (an installation
                        # with no file list yet): an ordinary replacement
                        move("moved", cur, prev / rel)
                    else:
                        keep_mine(cur)       # a user file where the release puts one
                if rel in new:
                    move("placed", stage / rel, cur)
        except Exception:
            reverse(journal)
            fh.close()
            (upd / PARTIAL).unlink(missing_ok=True)
            raise
    _write_json(upd / "journal.json", {"journal": journal, "kept_mine": kept_mine})
    (upd / PARTIAL).unlink(missing_ok=True)
    return {"journal": journal, "kept_mine": kept_mine}


def recover_interrupted(app: Path) -> bool:
    """Undo a switch that a crash or a power loss cut short: every move its
    partial journal records, last first (a torn last line is skipped). The
    staged update is whole again afterwards and is applied anew."""
    p = app / UPDATE_DIR / PARTIAL
    if not p.exists():
        return False
    entries = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    reverse(entries)
    p.unlink()
    return True


def reverse(journal: list) -> None:
    """Undo a switch, last move first."""
    for kind, src, dst in reversed(journal):
        s, d = Path(src), Path(dst)
        if d.exists() or d.is_symlink():
            _rename(d, s)


# -------------------------------------------------------------- the app
def conda_exe(app: Path) -> str:
    text = (app / ".conda-location").read_text(encoding="utf-8").strip().strip('"')
    root = Path(text).parent.parent
    for exe in (root / "Scripts" / "conda.exe", root / "bin" / "conda"):
        if exe.is_file():
            return str(exe)
    raise FileNotFoundError("conda named in .conda-location is missing; re-run the installer")


def app_command(app: Path, env: str) -> list:
    flag = "-p" if ("/" in env or "\\" in env) else "-n"
    return [conda_exe(app), "run", flag, env, "--no-capture-output",
            "python", "-u", "-m", "gui.app"]


def port() -> int:
    for name in ("PEBBLEMAPPER_PORT", "CSM_PORT"):
        try:
            v = int(os.environ.get(name, ""))
            if 1 <= v <= 65535:
                return v
        except ValueError:
            pass
    return 8081


def healthy(version: str, proc, wait_s=HEALTH_WAIT_S, stay_s=HEALTH_STAY_S) -> bool:
    """The new version answers ``/pm-health`` with its version, and is still
    running ``stay_s`` later (031 FR-014). A browser is not needed."""
    url = f"http://127.0.0.1:{port()}/pm-health"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if json.loads(r.read().decode("utf-8")).get("version") == version:
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        return False
    time.sleep(stay_s)
    return proc.poll() is None


def _kill(proc) -> None:
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)


def say(msg: str) -> None:
    print(f"[PebbleMapper] {msg}", flush=True)


def apply_staged(app: Path, ptr: dict) -> dict:
    """Verify and switch the staged release; returns the updated pointer."""
    st = ptr["staged"]
    stage = Path(st["stage_dir"])
    new_list = _read_json(Path(st["file_list"]), None)
    old_list = _read_json(app / FILES, {"files": []})
    bad = verify_stage(stage, new_list) if new_list else ["(file list missing)"]
    if bad:
        say(f"The prepared update to {st['release']} is damaged ({len(bad)} file(s)); "
            "it is discarded and the current version starts.")
        shutil.rmtree(stage, ignore_errors=True)
        ptr.update(staged=None, last_switch={"from": ptr["current"], "to": st["release"],
                                             "result": "damaged"})
        return ptr
    say(f"Applying the update to {st['release']}…")
    try:
        res = switch(app, stage, old_list, new_list)
    except Exception as ex:
        say(f"The update could not be applied ({ex}); the current version starts.")
        # The switch reversed itself, but the previous version it replaced
        # is gone: nothing to return to any more.
        ptr.update(staged=None, previous=None, previous_env=None,
                   last_switch={"from": ptr["current"], "to": st["release"],
                                "result": "failed", "detail": str(ex)})
        return ptr
    _write_json(app / UPDATE_DIR / "previous_files.json", old_list)
    _write_json(app / FILES, new_list)
    ptr.update(previous=ptr["current"], previous_env=ptr["env"], current=st["release"],
               env=st["env"], staged=None, provisional=True,
               last_switch={"from": ptr["current"], "to": st["release"], "result": "applied",
                            "kept_mine": res["kept_mine"]})
    shutil.rmtree(stage, ignore_errors=True)
    return ptr


def roll_back(app: Path, ptr: dict, result: str) -> dict:
    j = _read_json(app / UPDATE_DIR / "journal.json", {"journal": []})
    reverse(j["journal"])
    prev_files = _read_json(app / UPDATE_DIR / "previous_files.json", None)
    if prev_files is not None:
        _write_json(app / FILES, prev_files)
    ptr.update(current=ptr["previous"], env=ptr["previous_env"], previous=ptr["current"],
               previous_env=ptr["env"], provisional=False, return_to_previous=False,
               last_switch={"from": ptr["current"], "to": ptr["previous"], "result": result})
    (app / UPDATE_DIR / "journal.json").unlink(missing_ok=True)
    return ptr


# --------------------------------------------------------- the installer
def fresh_pointer(version: str) -> dict:
    return {"starter": STARTER_VERSION, "current": version, "env": "maskrcnn",
            "previous": None, "previous_env": None, "staged": None,
            "provisional": False, "return_to_previous": False}


def record_install(app: Path) -> bool:
    """Run by install.ps1 once the ``maskrcnn`` environment is built and
    checked: record what is installed. A pointer for the same version keeps
    its environment when that environment still exists (a repair of a copy
    updated in the app); any other case starts a fresh pointer. Returns
    whether the folder knows its release's files (unpacked from a release
    download), without which the app cannot update itself."""
    version = installed_version(app)
    old = _read_json(app / POINTER, None)
    if old and old.get("current") == version:
        ptr = old
        env = ptr.get("env") or "maskrcnn"
        if env != "maskrcnn" and not Path(env).is_dir():
            ptr["env"] = "maskrcnn"
        ptr.update(starter=STARTER_VERSION, provisional=False)
    else:
        ptr = fresh_pointer(version)
    _write_json(app / POINTER, ptr)
    return (app / FILES).is_file()


def _old_list_for(dest: Path, known: dict) -> dict:
    """The files a folder without a file list was installed with: for each
    path a published version shipped, the entry its bytes match, else the
    first version's entry (so a changed file is seen as the user's)."""
    by_path: dict = {}
    for v in known.get("versions", []):
        for f in v["files"]:
            by_path.setdefault(f["path"], []).append(f)
    files = []
    for rel, entries in sorted(by_path.items()):
        cur = dest / _safe(rel)
        files.append(next((e for e in entries if _matches(cur, e)), entries[0]))
    return {"files": files}


def migrate_into(src: Path, dest: Path) -> dict:
    """Update the installation in ``dest`` to the release unpacked in ``src``
    (031 R6): the releases' own files are switched exactly as an in-app
    update would, so projects, weights and personal files stay as they are
    and a shipped file the user changed is kept as ``.mine``. ``src`` is
    only read. Returns ``{"from", "to", "kept_mine"}``."""
    src, dest = Path(src).resolve(), Path(dest).resolve()
    if src == dest:
        raise ValueError("the release and the installation are the same folder")
    if (dest / ".git").exists():
        raise ValueError(f"{dest} is a development copy; update it with git")
    if not (dest / "functions" / "__init__.py").is_file():
        raise ValueError(f"{dest} does not hold a PebbleMapper installation")
    new_list = _read_json(src / FILES, None)
    if not new_list:
        raise ValueError(f"{src} was not unpacked from a release download")
    old_list = _read_json(dest / FILES, None) or _old_list_for(
        dest, _read_json(src / "installer" / "known_files.json", {}))
    before = installed_version(dest)
    stage = dest / UPDATE_DIR / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    for f in new_list["files"]:
        rel = _safe(f["path"])
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, stage / rel)
    bad = verify_stage(stage, new_list)
    if bad:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError(f"{len(bad)} file(s) of the release did not copy correctly")
    res = switch(dest, stage, old_list, new_list)
    shutil.rmtree(stage, ignore_errors=True)
    for name in STARTER_FILES:                    # the installer's job, not a switch's
        if (src / name).is_file():
            shutil.copyfile(src / name, dest / name)
    _write_json(dest / UPDATE_DIR / "previous_files.json", old_list)
    _write_json(dest / FILES, new_list)
    old_ptr = _read_json(dest / POINTER, None) or fresh_pointer(before)
    ptr = fresh_pointer(installed_version(dest))
    ptr.update(previous=before, previous_env=old_ptr.get("env") or "maskrcnn",
               last_switch={"from": before, "to": ptr["current"], "result": "ok",
                            "kept_mine": res["kept_mine"]})
    _write_json(dest / POINTER, ptr)
    return {"from": before, "to": ptr["current"], "kept_mine": res["kept_mine"]}


def installer_command(argv: list) -> int:
    """``--record-install`` and ``--migrate-into <folder>``, run by install.ps1."""
    app = Path(__file__).resolve().parent
    try:
        if argv[0] == "--record-install":
            if not record_install(app):
                say("This folder was not unpacked from a release download, so PebbleMapper "
                    "cannot update itself from the app. Download the release ZIP from "
                    "https://github.com/soloyant/PebbleMapper/releases to get in-app updates.")
            return 0
        if argv[0] == "--migrate-into" and len(argv) == 2:
            r = migrate_into(app, Path(argv[1]))
            say(f"Updated {argv[1]} from {r['from']} to {r['to']}.")
            for p in r["kept_mine"]:
                say(f"  you had changed {p.rsplit('.mine', 1)[0]}; your version is kept as {p}")
            return 0
    except (OSError, ValueError) as ex:
        say(f"ERROR: {ex}")
        return 1
    say(f"unknown installer command: {' '.join(argv)}")
    return 2


def main(argv=None) -> int:
    """Start the app; start it again, applying what was prepared, each time
    it exits asking to be restarted (**Restart now**, 031 FR-011)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0].startswith("--"):
        return installer_command(argv)
    app = Path(__file__).resolve().parent
    restarted = False
    while True:
        code = start_once(app, restarted)
        if code != RESTART_CODE:
            return code
        say("Restarting…")
        restarted = True


def child_env(quiet: bool) -> dict:
    """The app knows it was started here, so it can offer Restart now; a
    restart reuses the open browser tab, which reconnects by itself."""
    env = dict(os.environ, PEBBLEMAPPER_STARTER=str(STARTER_VERSION))
    if quiet:
        env["PEBBLEMAPPER_NO_BROWSER"] = "1"
    return env


def start_once(app: Path, restarted: bool = False) -> int:
    if recover_interrupted(app):
        say("An update was interrupted while it was being applied (the computer "
            "stopped); it was undone.")
    ptr = load_pointer(app)
    if ptr.get("return_to_previous") and ptr.get("previous"):
        say(f"Returning to {ptr['previous']}…")
        ptr = roll_back(app, ptr, "returned")
        _write_json(app / POINTER, ptr)
    if ptr.get("staged"):
        ptr = apply_staged(app, ptr)
        _write_json(app / POINTER, ptr)
    while True:
        proc = subprocess.Popen(app_command(app, ptr["env"]), cwd=str(app),
                                env=child_env(restarted))
        if ptr.get("provisional"):
            if healthy(ptr["current"], proc):
                ptr["provisional"] = False
                ptr["last_switch"]["result"] = "ok"
                _write_json(app / POINTER, ptr)
            else:
                say(f"The update to {ptr['current']} did not start; returning to "
                    f"{ptr['previous']}.")
                _kill(proc)
                ptr = roll_back(app, ptr, "rolled_back")
                _write_json(app / POINTER, ptr)
                restarted = True                  # the tab is open already
                continue                          # start the previous version
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
