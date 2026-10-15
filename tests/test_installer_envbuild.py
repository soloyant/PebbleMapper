"""030 R1, R2, FR-010, FR-012, FR-032: environments are built from the locks
alone, ignore the user's own settings, and a failed build leaves nothing."""
import os
from pathlib import Path

import pytest

from installer import envbuild
from installer.errors import InstallError


def test_conda_argv_reads_only_the_explicit_lock(tmp_path):
    argv = envbuild.conda_create_argv("conda.exe", tmp_path / "env", tmp_path / "c.lock.txt")
    assert argv[:4] == ["conda.exe", "create", "-p", str(tmp_path / "env")]
    for flag in ("--file", "--no-default-packages", "--override-channels", "--yes"):
        assert flag in argv
    assert argv[argv.index("-c") + 1] == "conda-forge"


def test_conda_variables_keep_caches_in_the_job_and_plugins_out(tmp_path):
    v = envbuild.conda_env_vars(tmp_path)
    assert v["CONDA_PKGS_DIRS"] == str(tmp_path / "pkgs")
    assert v["CONDA_NO_PLUGINS"] == "true"
    assert v["CONDA_CHANNEL_PRIORITY"] == "strict"


def test_pip_argv_requires_hashes_and_no_resolution(tmp_path):
    argv = envbuild.pip_install_argv(tmp_path / "env", tmp_path / "p.lock.txt")
    assert argv[1:4] == ["-m", "pip", "install"]
    for flag in ("--isolated", "--require-hashes", "--no-deps"):
        assert flag in argv
    assert argv[argv.index("--only-binary") + 1] == ":all:"
    v = envbuild.pip_env_vars(tmp_path)
    assert v["PIP_CONFIG_FILE"] == os.devnull
    assert v["PIP_CACHE_DIR"] == str(tmp_path / "pip")


def test_a_users_pip_variables_never_reach_pip(tmp_path, monkeypatch):
    seen = {}

    class _P:
        stdout = iter(())
        returncode = 0
        pid = 1

        def poll(self):
            return 0

    def fake_popen(argv, **kw):
        seen.update(kw["env"])
        return _P()
    monkeypatch.setenv("PIP_INDEX_URL", "https://evil.example.org/simple")
    monkeypatch.setattr(envbuild.subprocess, "Popen", fake_popen)
    envbuild._run(["x"], envbuild.pip_env_vars(tmp_path), "step", None,
                  lambda s: None, 10)
    assert "PIP_INDEX_URL" not in seen
    assert seen["PIP_CONFIG_FILE"] == os.devnull


def test_index_lines_alone_are_not_packages(tmp_path):
    p = tmp_path / "pip.lock.txt"
    p.write_text("--index-url https://download.pytorch.org/whl/cu121\n# none\n")
    assert envbuild.lock_has_packages(p) is False
    p.write_text("--index-url x\ntorch==2.5.1+cu121 --hash=sha256:" + "a" * 64 + "\n")
    assert envbuild.lock_has_packages(p) is True
    assert envbuild.lock_has_packages(None) is False


@pytest.mark.parametrize("tail,cls", [
    (["OSError: [Errno 28] No space left on device"], "disk_full"),
    (["SSL: CERTIFICATE_VERIFY_FAILED certificate verify failed https://conda.anaconda.org/x"], "clock"),
    (["CondaHTTPError: connection timed out https://conda.anaconda.org/x"], "network"),
    (["ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE."], "published_changed"),
    (["something else entirely"], "unknown"),
])
def test_failures_are_classified(tail, cls):
    err = envbuild._classify_failure(tail, "Installing", "E:")
    assert err.cls == cls
    if cls == "disk_full":
        assert "E:" in err.sentence()


def _locks(tmp_path):
    c = tmp_path / "conda.lock.txt"
    c.write_text("@EXPLICIT\n")
    p = tmp_path / "pip.lock.txt"
    p.write_text("six==1.16.0 --hash=sha256:" + "b" * 64 + "\n")
    return c, p


def test_a_failed_build_leaves_no_environment(tmp_path, monkeypatch):
    c, p = _locks(tmp_path)
    prefix = tmp_path / "m" / "env"
    monkeypatch.setattr(envbuild, "conda_version", lambda exe: (26, 1))
    calls = []

    def fake_run(argv, env, step, cancel, log, timeout, drive):
        calls.append(argv[1])
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / "conda-meta").mkdir(exist_ok=True)
        if argv[1] == "-m":                       # the pip step fails
            raise InstallError("network", host="files.pythonhosted.org")
    monkeypatch.setattr(envbuild, "_run", fake_run)
    with pytest.raises(InstallError):
        envbuild.create_env(prefix, c, p, tmp_path / "work", log=lambda s: None,
                            conda_exe="conda.exe")
    assert calls == ["create", "-m"]
    assert not prefix.exists()


def test_a_partial_environment_is_rebuilt_not_repaired(tmp_path, monkeypatch):
    c, p = _locks(tmp_path)
    prefix = tmp_path / "m" / "env"
    (prefix / "Lib").mkdir(parents=True)
    (prefix / "Lib" / "stale.py").write_text("")
    monkeypatch.setattr(envbuild, "conda_version", lambda exe: (26, 1))
    seen_before = []

    def fake_run(argv, *a, **k):
        seen_before.append((prefix / "Lib" / "stale.py").exists())
    monkeypatch.setattr(envbuild, "_run", fake_run)
    envbuild.create_env(prefix, c, p, tmp_path / "work", log=lambda s: None,
                        conda_exe="conda.exe")
    assert seen_before[0] is False


def test_an_old_conda_is_refused(tmp_path, monkeypatch):
    c, p = _locks(tmp_path)
    monkeypatch.setattr(envbuild, "conda_version", lambda exe: (24, 7))
    with pytest.raises(InstallError) as ei:
        envbuild.create_env(tmp_path / "env", c, p, tmp_path / "w",
                            log=lambda s: None, conda_exe="conda.exe")
    assert ei.value.cls == "conda_too_old"


def test_the_conda_on_this_machine_is_new_enough():
    from functions.conda_locate import find_conda_exe
    exe = find_conda_exe()
    if not exe:
        pytest.skip("no conda here")
    assert envbuild.conda_version(exe) >= envbuild.MIN_CONDA


def test_shipped_wheels_are_found_in_the_job_folder(tmp_path):
    argv = envbuild.pip_install_argv(tmp_path / "env", tmp_path / "p.txt", tmp_path / "files")
    assert argv[argv.index("--find-links") + 1] == str(tmp_path / "files")
    assert "--require-hashes" in argv
    assert "--find-links" not in envbuild.pip_install_argv(tmp_path / "env", tmp_path / "p.txt")


def test_user_site_packages_never_count(tmp_path):
    assert envbuild.pip_env_vars(tmp_path)["PYTHONNOUSERSITE"] == "1"
    assert envbuild.conda_env_vars(tmp_path)["PYTHONNOUSERSITE"] == "1"


def test_pip_cache_goes_to_the_job_folder_on_the_command_line(tmp_path):
    # --isolated makes pip ignore PIP_CACHE_DIR, so the folder must be an option.
    argv = envbuild.pip_install_argv(tmp_path / "env", tmp_path / "p.txt",
                                     cache_dir=tmp_path / "work" / "pip")
    assert argv[argv.index("--cache-dir") + 1] == str(tmp_path / "work" / "pip")
    assert "--no-cache-dir" in envbuild.pip_install_argv(tmp_path / "env", tmp_path / "p.txt")
