"""032 T003-T007: settings descriptions are checked in full, and a project
keeps only the values the user changed — never sending, never losing one
the model does not accept."""
import copy
import json

import pytest

from detectors import settings as S

DESC = {
    "format": 1,
    "model": "fake",
    "shared_controls": ["tile_size", "overlap", "device", "device_number"],
    "not_used": {"confidence": "it gives no detector confidence.",
                 "dedup": "it merges its own grains."},
    "settings": [
        {"key": "threshold", "label": "Threshold", "help": "Higher: fewer, cleaner grains.",
         "unit": "", "kind": "number", "min": 0, "max": 255, "step": 1, "integer": True,
         "default": 50, "authors": {"value": None, "source": "asked at each run"},
         "differs_because": "the program asks for it interactively.", "main": True},
        {"key": "min_size", "label": "Minimum size", "help": "Higher: small grains are dropped.",
         "unit": "px", "kind": "number", "min": 0, "max": 500, "step": 1, "integer": True,
         "default_by_mode": {"quadrat": 10, "ortho": {"derived": "from the pixel size"}},
         "authors": {"value": 20, "source": "manual section 4"},
         "differs_because": "kept from the first release.", "main": True},
        {"key": "use_sam", "label": "Use SAM", "help": "On: outlines refined.", "kind": "bool",
         "default": True},
        {"key": "dilation", "label": "Dilation", "help": "Higher: outlines grow.", "unit": "px",
         "kind": "number", "min": 0, "max": 10, "integer": True, "default": 3,
         "requires": {"key": "use_sam", "value": False}},
        {"key": "method", "label": "Method", "help": "Which method.", "kind": "choice",
         "choices": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}],
         "default": "a", "modes": ["ortho"]},
    ],
    "fixed": [{"label": "Checkpoint", "value": "small", "reason": "the one pinned."}],
}


def _bad(mutate):
    d = copy.deepcopy(DESC)
    mutate(d)
    with pytest.raises(S.DescriptionError):
        S.validate_description(d)


def test_a_good_description_is_accepted_and_filled():
    d = S.validate_description(copy.deepcopy(DESC))
    by = {s["key"]: s for s in d["settings"]}
    assert S.default_for(by["min_size"], "quadrat") == 10
    assert S.default_for(by["min_size"], "ortho") == {"derived": "from the pixel size"}
    assert S.default_for(by["method"], "quadrat") is None
    assert by["use_sam"]["unit"] == "" and by["threshold"]["main"] is True


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(format=2),
    lambda d: d.pop("model"),
    lambda d: d["shared_controls"].append("weather"),
    lambda d: d["not_used"].pop("dedup"),                       # unused control, no reason
    lambda d: d["settings"][0].update(key="Threshold"),
    lambda d: d["settings"].append(dict(d["settings"][0])),     # duplicate key
    lambda d: d["settings"][0].pop("help"),
    lambda d: d["settings"][0].update(unit="furlong"),
    lambda d: d["settings"][0].update(kind="slider"),
    lambda d: d["settings"][0].update(min=300),                 # min above max
    lambda d: d["settings"][0].update(default=999),             # default out of range
    lambda d: d["settings"][0].update(default=5.5),             # not an integer
    lambda d: d["settings"][1]["default_by_mode"].pop("ortho"),
    lambda d: d["settings"][1]["default_by_mode"].update(ortho={"derived": ""}),
    lambda d: d["settings"][1].pop("differs_because"),          # differs from the authors
    lambda d: d["settings"][1]["authors"].update(source=""),
    lambda d: d["settings"][3].update(requires={"key": "nothing", "value": 1}),
    lambda d: d["settings"][4].update(choices=[{"value": "a", "label": "A"}, {"value": "a", "label": "B"}]),
    lambda d: d["settings"][4].update(default="z"),
    lambda d: d["settings"][4].update(modes=["photo"]),
    lambda d: d["fixed"].append({"label": "x"}),
])
def test_one_fault_refuses_the_whole_description(mutate):
    _bad(mutate)


def test_unknown_fields_are_ignored():
    d = copy.deepcopy(DESC)
    d["future"] = 1
    d["settings"][0]["colour"] = "red"
    S.validate_description(d)


def test_values_are_checked_and_the_range_said():
    s = next(x for x in S.validate_description(copy.deepcopy(DESC))["settings"]
             if x["key"] == "min_size")
    assert S.validate_value(s, 30) == 30
    assert S.validate_value(s, 30.0) == 30
    for bad in (-1, 501, 2.5, "10", True, float("nan")):
        with pytest.raises(ValueError, match="Allowed: 0 to 500 px"):
            S.validate_value(s, bad)


def test_applies_follows_modes_and_dependencies():
    d = S.validate_description(copy.deepcopy(DESC))
    by = {s["key"]: s for s in d["settings"]}
    assert S.applies(by["method"], "quadrat", {}, d) == (False, "used for ortho only")
    ok, why = S.applies(by["dilation"], "quadrat", {}, d)
    assert not ok and why == "used only when 'Use SAM' is off"
    assert S.applies(by["dilation"], "quadrat", {"use_sam": False}, d) == (True, "")


# ------------------------------------------------------- finding the file
class _Backend:
    class info:
        name = "fake"


def test_a_description_is_found_beside_the_backend_module_or_named(tmp_path):
    f = tmp_path / "pm_settings.json"
    f.write_text(json.dumps(DESC), encoding="utf-8")
    b = _Backend()
    b.settings_file = f
    assert S.load_description(b)["model"] == "fake"
    f.write_text("{not json", encoding="utf-8")
    import os
    os.utime(f, (1, 1))                                  # a new mtime: re-read
    assert S.load_description(b) is None and S.description_problem(b)
    b2 = _Backend()
    b2.settings_file = tmp_path / "missing.json"
    assert S.load_description(b2) is None and S.description_problem(b2) == ""


def test_a_description_of_another_model_is_refused(tmp_path):
    f = tmp_path / "pm_settings.json"
    f.write_text(json.dumps(dict(DESC, model="other")), encoding="utf-8")
    b = _Backend()
    b.settings_file = f
    assert S.load_description(b) is None
    assert "describes 'other'" in S.description_problem(b)


def test_the_mask_rcnn_description_is_valid():
    from detectors.maskrcnn import MaskRCNNBackend
    d = S.load_description(MaskRCNNBackend())
    assert d is not None, S.description_problem(MaskRCNNBackend())
    assert {s["key"] for s in d["settings"]} == {"min_confidence", "dedup_overlap"}


# ---------------------------------------------------------- project store
def test_model_names_drop_release_and_variant():
    assert S.model_name("seg@1.1.0") == "seg"
    assert S.model_name("seg@1.1.0-cpu") == "seg"
    assert S.model_name("maskrcnn") == "maskrcnn"


def test_only_changes_are_stored_and_sent(tmp_path):
    d = S.validate_description(copy.deepcopy(DESC))
    ps = S.ProjectSettings(tmp_path / S.STORE_FILE)
    assert ps.changed("fake@1.1.0", "quadrat", d) == {}
    ps.set("fake@1.1.0", "quadrat", d, "threshold", 70)
    ps.set("fake@1.1.0", "quadrat", d, "min_size", 10)        # PebbleMapper's own value
    assert ps.changed("fake@1.1.0", "quadrat", d) == {"threshold": 70}
    assert ps.changed("fake@1.1.0", "ortho", d) == {}
    with pytest.raises(ValueError):
        ps.set("fake", "quadrat", d, "threshold", 300)
    with pytest.raises(ValueError):
        ps.set("fake", "quadrat", d, "method", "b")           # ortho only
    again = S.ProjectSettings(tmp_path / S.STORE_FILE)
    assert again.values("fake@2.0.0-cpu", "quadrat") == {"threshold": 70}, "any release"
    again.reset("fake", "quadrat", "threshold")
    assert S.ProjectSettings(tmp_path / S.STORE_FILE).values("fake", "quadrat") == {}


def test_values_a_release_does_not_accept_are_kept_not_sent(tmp_path):
    d = S.validate_description(copy.deepcopy(DESC))
    ps = S.ProjectSettings(tmp_path / S.STORE_FILE)
    ps.set("fake", "quadrat", d, "threshold", 200)
    ps.set("fake", "quadrat", d, "dilation", 5)
    narrower = copy.deepcopy(DESC)
    narrower["settings"][0]["max"] = 100
    narrower["settings"] = [s for s in narrower["settings"] if s["key"] != "dilation"]
    n = S.validate_description(narrower)
    notices = ps.check_against("fake", n)
    assert len(notices) == 2 and all("kept in the project" in x for x in notices)
    assert ps.changed("fake", "quadrat", n) == {}
    assert ps.kept("fake", "quadrat") == {"threshold": 200, "dilation": 5}
    # Back on the wider release, the kept values return.
    assert ps.check_against("fake", d) == []
    assert ps.changed("fake", "quadrat", d) == {"threshold": 200, "dilation": 5}


def test_a_hand_edited_file_with_a_bad_value_is_not_sent(tmp_path):
    d = S.validate_description(copy.deepcopy(DESC))
    p = tmp_path / S.STORE_FILE
    p.write_text(json.dumps({"format": 1, "models": {"fake": {"values": {
        "quadrat": {"threshold": "seventy"}}}}}), encoding="utf-8")
    ps = S.ProjectSettings(p)
    assert ps.check_against("fake", d) and ps.changed("fake", "quadrat", d) == {}


def test_an_unreadable_file_is_kept_aside_on_the_next_save(tmp_path):
    d = S.validate_description(copy.deepcopy(DESC))
    p = tmp_path / S.STORE_FILE
    p.write_text("{broken", encoding="utf-8")
    ps = S.ProjectSettings(p)
    assert ps.problem and ps.values("fake", "quadrat") == {}
    ps.set("fake", "quadrat", d, "threshold", 60)
    assert (tmp_path / (S.STORE_FILE + ".unreadable")).read_text() == "{broken"
    assert json.loads(p.read_text())["models"]["fake"]["values"]["quadrat"] == {"threshold": 60}


def test_the_run_header_names_model_release_device_and_changes(monkeypatch):
    monkeypatch.setattr(S, "_GPU_NAMES", ["Card A", "Card B"])
    monkeypatch.setattr(S, "load_description",
                        lambda b: S.validate_description(copy.deepcopy(DESC)))

    class _B:
        class info:
            display_name = "Fake"
            name = "fake"
        registry_key = "fake@1.1.0"
        variant = "gpu"
    lines = S.run_header(_B(), "quadrat", {"threshold": 70}, "GPU", 1)
    assert lines[0] == "[settings] model: Fake, release 1.1.0; device: graphics card 1 (Card B)"
    assert lines[1] == "[settings] settings changed: threshold=70"
    assert S.run_header(_B(), "quadrat", {}, "CPU", 0)[1] == \
        "[settings] settings: PebbleMapper's values"
    assert "device: processor" in S.run_header(_B(), "quadrat", {}, "CPU", 0)[0]


def test_recorded_value_sets_reads_the_projects_records(tmp_path):
    v = tmp_path / "output_results" / "vectors"
    v.mkdir(parents=True)
    for i, (model, params) in enumerate([
            ("seg", {"settings_changed": {"min_area": 150}}),
            ("seg", {"settings_changed": {"min_area": 150}}),     # same set
            ("seg", {}),                                          # before 032
            ("maskrcnn", {"settings_changed": {"min_confidence": 0.5}})]):
        (v / f"r{i}.csv.manifest.json").write_text(
            json.dumps({"model": model, "params": params}), encoding="utf-8")
    (v / "broken.csv.manifest.json").write_text("{", encoding="utf-8")
    assert S.recorded_value_sets(tmp_path, "seg@1.1.0") == [{"min_area": 150}, {}] or \
        S.recorded_value_sets(tmp_path, "seg@1.1.0") == [{}, {"min_area": 150}]
    assert S.recorded_value_sets(tmp_path, "maskrcnn") == [{"min_confidence": 0.5}]
    assert S.recorded_value_sets(tmp_path / "nothing", "seg") == []
    assert S.describe_values({}) == "PebbleMapper's values"


def test_an_older_release_is_pointed_to_a_newer_one_that_offers_settings(monkeypatch):
    """032 US6: the kept model list names the newer release with settings;
    a hand-installed model (no release in its key) is offered nothing."""
    from gui import model_settings_panel as P
    from installer import catalog
    data = {"models": [{"backends": [{"name": "seg"}], "releases": [
        {"release": "1.0.0", "withdrawn": False},
        {"release": "1.1.0", "withdrawn": False, "describes_settings": True},
        {"release": "1.2.0", "withdrawn": True, "describes_settings": True}]}]}
    monkeypatch.setattr(catalog, "kept_list", lambda src: data)
    ctx = P.Context("seg@1.0.0", "Seg", None, None, "quadrat")
    assert P._newer_release_with_settings(ctx) == "1.1.0"
    assert P._newer_release_with_settings(P.Context("seg@1.1.0", "Seg", None, None,
                                                    "quadrat")) is None
    assert P._newer_release_with_settings(P.Context("seg", "Seg", None, None,
                                                    "quadrat")) is None


def test_the_report_states_each_model_release_and_set_of_values(tmp_path):
    from functions.report_facts import _detection_models
    from functions.report_sections import _model_settings_texts
    v = tmp_path / "output_results" / "vectors"
    v.mkdir(parents=True)
    for img, model, params in [
            ("A", "seg", {"release": "1.1.0", "settings_changed": {"min_area": 150}}),
            ("B", "seg", {"release": "1.1.0", "settings_changed": {"min_area": 150}}),
            ("C", "seg", {"release": "1.1.0", "settings_changed": {}}),
            ("D", "maskrcnn", {}),
            ("E", "pc", {"release": "1.0.0", "settings_used": "built-in, not reported",
                         "settings_changed": {}})]:
        (v / f"{img}.csv.manifest.json").write_text(
            json.dumps({"model": model, "params": params}), encoding="utf-8")
    groups = _detection_models(tmp_path)
    assert [(g["model"], g["release"]) for g in groups] == [
        ("maskrcnn", ""), ("pc", "1.0.0"), ("seg", "1.1.0")]
    texts = _model_settings_texts(groups)
    assert texts[0] == "<b>Mask R-CNN (Soloy et al. 2020)</b>: every setting was PebbleMapper's value."
    assert "does not report the values it used (images: E)" in texts[1]
    assert "min_area = 150 (images: A, B)" in texts[2]
    assert "PebbleMapper's values for the other images (C)" in texts[2]


def test_the_run_header_lists_only_what_is_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_GPU_NAMES", ["Card A"])
    f = tmp_path / "pm_settings.json"
    f.write_text(json.dumps(DESC), encoding="utf-8")

    class _B:
        class info:
            display_name = "Fake"
            name = "fake"
        settings_file = f
        variant = "gpu"
    lines = S.run_header(_B(), "quadrat", {"threshold": 70, "nonsense": 1, "method": "b",
                                           "min_size": -5}, "GPU", 0)
    assert lines[1] == "[settings] settings changed: threshold=70"

    class _Old(_B):
        settings_file = tmp_path / "absent.json"
    assert "built-in values" in S.run_header(_Old(), "quadrat", {"threshold": 70}, "GPU", 0)[1]
