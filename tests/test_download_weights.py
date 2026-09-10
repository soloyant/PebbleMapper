"""detectors.download_weights: the weights fetch, without the network.

Zenodo is stood in for at the one function that opens a URL, so what is
checked is the script's own behaviour: the record is read through the API,
the right file is picked, the download is verified against the record's
checksum and kept only when it matches, a file already in place is verified
rather than fetched again, and an unpublished record is reported clearly.
"""
from __future__ import annotations

import hashlib
import io
import json
from urllib.error import HTTPError

import pytest

from detectors import download_weights as dw

PAYLOAD = b"not really 367 MB of weights, but enough to hash\n" * 64
MD5 = hashlib.md5(PAYLOAD).hexdigest()
RECORD = {
    "id": 20779877,
    "metadata": {"title": "PebbleMapper Mask R-CNN weights", "version": "1.0",
                 "license": {"id": "cc-by-4.0"}},
    "files": [
        {"key": "README.txt", "size": 12, "checksum": "md5:0" * 1,
         "links": {"self": "https://zenodo.org/api/records/20779877/files/README.txt/content"}},
        {"key": dw.FILE, "size": len(PAYLOAD), "checksum": f"md5:{MD5}",
         "links": {"self": "https://zenodo.org/api/records/20779877/files/mask_rcnn_clasts.h5/content"}},
    ],
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _serve(monkeypatch, record=RECORD, payload=PAYLOAD, missing=False):
    calls = []

    def fake_open(url):
        calls.append(url)
        if missing:
            raise HTTPError(url, 404, "NOT FOUND", {}, None)
        if url.endswith("/records/20779877"):
            return _Resp(json.dumps(record).encode("utf-8"))
        if url.endswith("/content"):
            return _Resp(payload)
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(dw, "_open", fake_open)
    return calls


def test_the_record_id_comes_from_the_doi():
    assert dw.record_id("10.5281/zenodo.20779877") == "20779877"
    with pytest.raises(ValueError):
        dw.record_id("10.1002/esp.5755")


def test_fetches_verifies_and_places_the_file(monkeypatch, tmp_path):
    calls = _serve(monkeypatch)
    lines = []
    rc = dw.main(["--dest", str(tmp_path)], log=lines.append)
    assert rc == 0
    out = tmp_path / dw.FILE
    assert out.read_bytes() == PAYLOAD
    assert not out.with_name(out.name + ".part").exists()
    assert calls[0].endswith("/api/records/20779877")
    assert calls[1].endswith("/files/mask_rcnn_clasts.h5/content")
    assert any("md5 verified" in l for l in lines)
    assert any("CC BY" in l.upper().replace("-", " ") or "cc-by" in l for l in lines)
    # A second run verifies the file in place and fetches nothing.
    calls.clear()
    lines.clear()
    assert dw.main(["--dest", str(tmp_path)], log=lines.append) == 0
    assert calls == [calls[0]] if calls else True
    assert not any(u.endswith("/content") for u in calls)
    assert any("nothing to do" in l for l in lines)


def test_a_corrupt_download_is_not_kept(monkeypatch, tmp_path):
    _serve(monkeypatch, payload=PAYLOAD[:-1])
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dw.main(["--dest", str(tmp_path)])
    assert not (tmp_path / dw.FILE).exists()
    assert not (tmp_path / (dw.FILE + ".part")).exists()


def test_a_different_file_in_place_is_reported_not_replaced(monkeypatch, tmp_path):
    _serve(monkeypatch)
    (tmp_path / dw.FILE).write_bytes(b"older weights")
    lines = []
    assert dw.main(["--dest", str(tmp_path)], log=lines.append) == 3
    assert (tmp_path / dw.FILE).read_bytes() == b"older weights"
    assert any("--force" in l for l in lines)
    assert dw.main(["--dest", str(tmp_path), "--force"], log=lines.append) == 0
    assert (tmp_path / dw.FILE).read_bytes() == PAYLOAD


def test_list_shows_the_record_and_fetches_nothing(monkeypatch, tmp_path):
    calls = _serve(monkeypatch)
    lines = []
    assert dw.main(["--dest", str(tmp_path), "--list"], log=lines.append) == 0
    assert not (tmp_path / dw.FILE).exists()
    assert len(calls) == 1
    assert any(dw.FILE in l and "MB" in l for l in lines)


def test_an_unpublished_record_is_said_plainly(monkeypatch, tmp_path):
    _serve(monkeypatch, missing=True)
    lines = []
    assert dw.main(["--dest", str(tmp_path)], log=lines.append) == dw.EXIT_NOT_PUBLISHED
    # While the weights are not public, the message says how to get them.
    assert any("not public yet" in l and dw.ISSUES in l for l in lines)
    # Once they are, the record's own message is shown.
    monkeypatch.setattr(dw, "WEIGHTS_PUBLIC", True)
    lines.clear()
    assert dw.main(["--dest", str(tmp_path)], log=lines.append) == dw.EXIT_NOT_PUBLISHED
    assert any("no record" in l for l in lines)
    assert "Install PebbleMapper" in dw.weights_missing_message()
    # With the file already there, that is only a note, not a failure.
    (tmp_path / dw.FILE).write_bytes(PAYLOAD)
    assert dw.main(["--dest", str(tmp_path)], log=lines.append) == 0
    assert any("could not be checked" in l for l in lines)


def test_the_default_destination_is_where_the_backend_looks():
    from detectors.maskrcnn import _weights_candidates
    assert dw.default_dest() == _weights_candidates()[0].parent
    assert dw.default_dest().name == "model_weights"
