"""030 FR-001, FR-003–FR-006, FR-052, SC-004: the list is validated as a
whole, kept, and never replaced by an older one."""
import copy
import json
from pathlib import Path

import pytest

from installer import catalog

H = "a" * 64
GOOD = {
    "format": 1, "list_version": 3, "published": "2026-10-13", "models": [{
        "id": "orthosam", "authors": "A", "citation": "C", "licence_code": "MIT",
        "licence_weights": "Apache-2.0", "description": "d", "runs_in": "own",
        "homepage": "https://github.com/soloyant/pebblemapper-backend-orthosam",
        "backends": [{"name": "orthosam", "display_name": "OrthoSAM",
                      "package": "pm_orthosam_backend", "module": "backend",
                      "factory": "make_backend"}],
        "releases": [{"release": "1.0.0", "commit": "b" * 40, "withdrawn": False,
                      "notes": "https://github.com/soloyant/x/releases/tag/v1.0.0",
                      "variants": {"gpu": {
                          "files": [
                              {"role": "archive", "size": 10, "sha256": H,
                               "url": "https://github.com/soloyant/x/releases/download/v1.0.0/a.zip"},
                              {"role": "conda_lock", "size": 10, "sha256": H,
                               "url": "https://github.com/soloyant/x/releases/download/v1.0.0/c.txt"},
                              {"role": "check_image", "size": 10, "sha256": H,
                               "url": "https://github.com/soloyant/x/releases/download/v1.0.0/check.jpg"},
                              {"role": "weights", "size": 10, "sha256": H, "dest": "weights/w.pth",
                               "url": "https://dl.fbaipublicfiles.com/segment_anything/w.pth"}],
                          "download_bytes": 1, "disk_bytes": 1, "peak_bytes": 1,
                          "longest_path": 100, "minutes": [10, 30],
                          "check_timeout_s": 900}}}]}]}


def _bad(path, value):
    d = copy.deepcopy(GOOD)
    obj = d
    for k in path[:-1]:
        obj = obj[k]
    if value is KeyError:
        del obj[path[-1]]
    else:
        obj[path[-1]] = value
    return d


V = ("models", 0, "releases", 0, "variants", "gpu")
F = V + ("files", 3)


def test_the_good_list_validates():
    catalog.validate(GOOD)


def test_the_schema_file_and_the_validator_agree_on_required_keys():
    path = (Path(__file__).resolve().parents[1] / "specs" / "030-plugin-installer-updates" /
            "contracts" / "list.schema.json")
    if not path.is_file():
        pytest.skip("the specs live in the development repository only")
    schema = json.loads(path.read_text(encoding="utf-8"))
    for name in ("model", "release", "variant", "file", "backend"):
        for key in schema["$defs"][name]["required"]:
            path = {"model": ("models", 0), "release": ("models", 0, "releases", 0),
                    "variant": V, "file": F, "backend": ("models", 0, "backends", 0)}[name]
            with pytest.raises(catalog.ListError):
                catalog.validate(_bad(path + (key,), KeyError))


@pytest.mark.parametrize("path,value", [
    (("format",), 2),
    (("list_version",), 0),
    (("models", 0, "id"), "Bad_Id"),
    (("models", 0, "description"), "x" * 301),
    (("models", 0, "homepage"), "https://evil.example.org/"),
    (("models", 0, "runs_in"), "somewhere"),
    (("models", 0, "backends", 0, "package"), "pm-orthosam"),
    (("models", 0, "releases", 0, "release"), "1.0"),
    (("models", 0, "releases", 0, "commit"), "main"),
    (("models", 0, "releases", 0, "notes"), "https://evil.example.org/notes"),
    (("models", 0, "releases", 0, "variants"), {"tpu": {}}),
    (F + ("url",), "http://dl.fbaipublicfiles.com/w.pth"),
    (F + ("url",), "https://evil.example.org/w.pth"),
    (F + ("url",), "https://huggingface.co/x/y/resolve/main/w.pth"),
    (F + ("sha256",), "abc"),
    (F + ("size",), 0),
    (F + ("dest",), "../outside/w.pth"),
    (F + ("dest",), "C:/w.pth"),
    (F + ("role",), "script"),
])
def test_faults_refuse_the_whole_list(path, value):
    with pytest.raises(catalog.ListError):
        catalog.validate(_bad(path, value))


def test_a_huggingface_revision_hash_is_accepted():
    catalog.validate(_bad(F + ("url",), "https://huggingface.co/x/y/resolve/" + "c" * 40 + "/w.pth"))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    return tmp_path


def _serve_file(tmp_path, data, name="list.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_a_good_list_is_kept_and_an_older_one_refused(home):
    src = _serve_file(home, GOOD)
    r = catalog.fetch_list(src)
    assert r.fresh and r.data["list_version"] == 3 and r.message == ""
    older = dict(copy.deepcopy(GOOD), list_version=2)
    _serve_file(home, older)
    r = catalog.fetch_list(src)
    assert not r.fresh and r.data["list_version"] == 3
    assert "older" in r.message


def test_a_failed_fetch_shows_the_kept_list_with_its_date(home):
    src = _serve_file(home, GOOD)
    catalog.fetch_list(src)
    Path(src).unlink()
    r = catalog.fetch_list(src)
    assert r.data["list_version"] == 3
    assert r.message == ("Could not reach the model list (no connection). "
                         "Showing the list from 2026-10-13.")


def test_no_list_ever(home):
    r = catalog.fetch_list(str(home / "absent.json"))
    assert r.data is None
    assert r.message == "Connect to the internet once to see the models you can add."


def test_a_newer_format_says_update_pebblemapper(home):
    src = _serve_file(home, dict(copy.deepcopy(GOOD), format=2))
    r = catalog.fetch_list(src)
    assert r.too_old and "too old" in r.message


def test_each_source_has_its_own_accepted_version(home):
    a = _serve_file(home, dict(copy.deepcopy(GOOD), list_version=50), "a.json")
    b = _serve_file(home, GOOD, "b.json")
    catalog.fetch_list(a)
    assert catalog.fetch_list(b).fresh, "a test list must not lock out another source"


def test_the_main_address_is_fixed(monkeypatch):
    monkeypatch.delenv("PEBBLEMAPPER_TEST_LIST", raising=False)
    assert catalog.current_source() == catalog.LIST_URL
    assert catalog.LIST_URL.startswith("https://raw.githubusercontent.com/soloyant/")
    monkeypatch.setenv("PEBBLEMAPPER_TEST_LIST", "x.json")
    assert catalog.is_test_source(catalog.current_source())


def test_latest_release_skips_withdrawn():
    m = {"releases": [{"release": "1.2.0", "withdrawn": True},
                      {"release": "1.10.0", "withdrawn": False},
                      {"release": "1.9.0", "withdrawn": False}]}
    assert catalog.latest_release(m)["release"] == "1.10.0"


def test_simultaneous_refreshes_never_raise(home):
    """Found in the GUI boot check: two pages refreshing the list at once
    collided on one temporary file and fetch_list raised."""
    import threading
    src = _serve_file(home, GOOD)
    errors, results = [], []

    def go():
        try:
            results.append(catalog.fetch_list(src))
        except Exception as ex:          # the contract: never raises
            errors.append(ex)
    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(r.data["list_version"] == 3 for r in results)
    assert catalog.accepted_version(src) == 3
    assert not list((home / "home" / "list").glob("*.tmp"))


APP = {"release": "1.1.0", "commit": "c" * 40, "summary": "Models can be updated.",
       "notes": "https://github.com/soloyant/PebbleMapper/releases/tag/v1.1.0",
       "min_starter": 1, "withdrawn": False,
       "variants": {"gpu": {"download_bytes": 1, "disk_bytes": 1, "files": [
           {"role": r, "size": 10, "sha256": H,
            "url": f"https://github.com/soloyant/PebbleMapper/releases/download/v1.1.0/{r}"}
           for r in ("archive", "file_list", "conda_lock", "pip_lock")]}}}


def test_an_app_release_validates():
    catalog.validate(dict(copy.deepcopy(GOOD), app_releases=[APP]))


@pytest.mark.parametrize("key,value", [
    ("release", "1.1"), ("commit", "main"), ("summary", "x" * 601),
    ("notes", "https://evil.example.org/"), ("min_starter", 0), ("withdrawn", "no"),
    ("variants", {"arm": {}}), ("replacement", "one")])
def test_app_release_faults_refuse_the_list(key, value):
    bad = copy.deepcopy(APP)
    bad[key] = value
    with pytest.raises(catalog.ListError):
        catalog.validate(dict(copy.deepcopy(GOOD), app_releases=[bad]))


def test_an_app_release_without_its_file_list_is_refused():
    bad = copy.deepcopy(APP)
    bad["variants"]["gpu"]["files"] = [f for f in bad["variants"]["gpu"]["files"]
                                       if f["role"] != "file_list"]
    with pytest.raises(catalog.ListError, match="file_list"):
        catalog.validate(dict(copy.deepcopy(GOOD), app_releases=[bad]))
