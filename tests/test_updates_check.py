"""031 FR-001–FR-009: when to check, what to offer, what to announce."""
import json
import time

import pytest

from installer import catalog, updates


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PEBBLEMAPPER_HOME", str(tmp_path / "home"))
    return tmp_path


def _rel(v, withdrawn=False, replacement=None, variants=("gpu",)):
    r = {"release": v, "withdrawn": withdrawn, "variants": {k: {} for k in variants}}
    if replacement:
        r["replacement"] = replacement
    return r


def _list(app=(), models=None):
    return {"app_releases": list(app), "models": models or []}


def test_due_only_when_on_and_a_day_has_passed():
    now = time.time()
    assert updates.check_due(now)                      # never checked
    updates.run_check(source="Z:/no/such/list.json", now=now)
    assert not updates.check_due(now + 3600)
    assert updates.check_due(now + 24 * 3600 + 1)
    updates.set_checks_enabled(False)
    assert not updates.check_due(now + 48 * 3600)


def test_a_failed_check_is_silent_and_recorded():
    st = updates.run_check(source="Z:/no/such/list.json")
    assert st["last_result"] == "offline"
    assert updates.read_state()["last_check"]


def test_the_first_delay_is_between_one_and_ten_minutes():
    lo, hi = updates.FIRST_DELAY_S
    assert (lo, hi) == (60, 600)


def test_app_and_model_updates_are_found():
    data = _list(app=[_rel("1.0.0"), _rel("1.1.0")],
                 models=[{"id": "o", "backends": [{"display_name": "O"}],
                          "releases": [_rel("1.0.0"), _rel("1.0.1")]}])
    ups = updates.available_updates(data, [{"id": "o", "release": "1.0.0", "variant": "gpu"}],
                                    app_version="1.0.0")
    assert sorted(u["key"] for u in ups) == ["app:1.1.0", "model:o:1.0.1"]


def test_nothing_newer_means_nothing_offered():
    data = _list(app=[_rel("1.0.0")])
    assert updates.available_updates(data, [], app_version="1.0.0") == []
    assert updates.available_updates(None, [], app_version="1.0.0") == []


def test_withdrawn_releases_are_skipped_and_replacements_offered_even_when_older():
    data = _list(app=[_rel("1.0.0"), _rel("1.1.0", withdrawn=True, replacement="1.0.0")])
    assert updates.available_updates(data, [], app_version="1.0.0") == []
    [u] = updates.available_updates(data, [], app_version="1.1.0")
    assert u["release"]["release"] == "1.0.0" and u["withdrawn_current"]


def test_a_model_removed_from_the_list_is_left_alone():
    assert updates.available_updates(_list(), [{"id": "gone", "release": "1.0.0"}],
                                     app_version="9.9.9") == []


def test_dismissed_updates_are_not_announced_but_withdrawn_ones_are():
    ups = [{"key": "app:1.1.0"}, {"key": "model:o:1.0.1"},
           {"key": "app:1.0.0", "withdrawn_current": True}]
    updates.dismiss("app:1.1.0")
    updates.dismiss("app:1.0.0")
    assert [u["key"] for u in updates.to_announce(ups)] == ["model:o:1.0.1", "app:1.0.0"]


def test_versions_compare_as_numbers():
    assert updates.version_tuple("1.10.0") > updates.version_tuple("1.9.3")
