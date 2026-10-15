"""031 US3 scenario 4: results from another release of the selected model
are found in the project from their output names."""
import pytest

from functions import naming


@pytest.fixture(scope="module")
def other_release_in_folder():
    from gui.app import other_release_in_folder as f
    return f


def _touch(folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.csv").write_text("x\n")


def test_another_release_is_found(tmp_path, other_release_in_folder):
    vec = tmp_path / "output_results" / "vectors"
    _touch(vec, naming.with_model("Site__IMG_1", "orthosam@1.0.0") + "_individual_clasts")
    assert other_release_in_folder(tmp_path, "orthosam@1.0.1") == "1.0.0"


def test_the_same_release_or_another_model_is_not(tmp_path, other_release_in_folder):
    vec = tmp_path / "v"
    _touch(vec, naming.with_model("S__I", "orthosam@1.0.1"))
    _touch(vec, naming.with_model("S__I", "seg@1.0.0"))
    _touch(vec, naming.with_model("S__I", "orthosam"))          # a hand install
    assert other_release_in_folder(tmp_path, "orthosam@1.0.1") is None


def test_the_newest_other_release_is_named(tmp_path, other_release_in_folder):
    for r in ("1.0.0", "1.2.0", "1.10.0"):
        _touch(tmp_path, naming.with_model(f"S__I{r}", f"orthosam@{r}"))
    assert other_release_in_folder(tmp_path, "orthosam@2.0.0") == "1.10.0"


def test_not_an_installed_model_means_no_question(tmp_path, other_release_in_folder):
    assert other_release_in_folder(tmp_path, "maskrcnn") is None
