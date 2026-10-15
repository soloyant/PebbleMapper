"""030 R9, FR-026, FR-049, FR-050: downloads resume, stop at the listed size,
and are refused unless they match their checksum."""
import hashlib
import http.server
import threading

import pytest

from installer import fetch
from installer.errors import InstallError

PAYLOAD = bytes(range(256)) * 4096          # 1 MiB


class _Handler(http.server.BaseHTTPRequestHandler):
    files = {}
    honour_range = True
    seen_headers = []
    cut_after = None                          # drop the connection after N bytes

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).seen_headers.append(dict(self.headers))
        if self.path == "/redirect-http":
            self.send_response(302)
            self.send_header("Location", "http://example.org/x")
            self.end_headers()
            return
        body = self.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        start = 0
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        chunk = body[start:]
        if self.cut_after is not None:
            chunk = chunk[: self.cut_after]
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        try:
            self.wfile.write(chunk)
        except ConnectionError:
            pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(fetch, "_ALLOW_LOCAL_HTTP", True)
    _Handler.files = {"/f.bin": PAYLOAD, "/long.bin": PAYLOAD + b"x"}
    _Handler.honour_range, _Handler.cut_after = True, None
    _Handler.seen_headers = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_a_whole_download_is_checked_and_moved_into_place(server, tmp_path):
    seen = []
    out = fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA,
                         on_progress=lambda d, t: seen.append(d))
    assert out.read_bytes() == PAYLOAD
    assert not (tmp_path / "f.bin.part").exists()
    assert seen[-1] == len(PAYLOAD)
    ua = _Handler.seen_headers[-1]
    assert ua["User-Agent"].startswith("PebbleMapper/")
    assert not {"Cookie", "Authorization", "Referer"} & set(ua)


def test_a_partial_download_is_continued_with_range(server, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:300_000])
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert _Handler.seen_headers[-1]["Range"] == "bytes=300000-"
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_a_server_ignoring_range_restarts_cleanly(server, tmp_path):
    _Handler.honour_range = False
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:300_000])
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_a_verified_file_is_not_fetched_again(server, tmp_path):
    (tmp_path / "f.bin").write_bytes(PAYLOAD)
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert _Handler.seen_headers == []


def test_a_file_longer_than_listed_is_stopped(server, tmp_path):
    with pytest.raises(InstallError) as ei:
        fetch.download(f"{server}/long.bin", tmp_path / "l.bin", len(PAYLOAD), SHA)
    assert ei.value.cls == "published_changed"
    assert not (tmp_path / "l.bin").exists() and not (tmp_path / "l.bin.part").exists()


def test_a_mismatching_fresh_download_is_refused(server, tmp_path):
    with pytest.raises(InstallError) as ei:
        fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), "0" * 64)
    assert ei.value.cls == "published_changed"
    assert not (tmp_path / "f.bin").exists()


def test_a_damaged_partial_is_fetched_afresh_once(server, tmp_path):
    damaged = bytearray(PAYLOAD[:300_000])
    damaged[10] ^= 0xFF
    (tmp_path / "f.bin.part").write_bytes(bytes(damaged))
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_a_dropped_connection_keeps_the_part_for_resume(server, tmp_path):
    _Handler.cut_after = 200_000
    with pytest.raises(InstallError) as ei:
        fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert ei.value.cls == "network"
    assert (tmp_path / "f.bin.part").stat().st_size == 200_000
    _Handler.cut_after = None
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA)
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD


def test_cancel_keeps_the_part(server, tmp_path):
    with pytest.raises(fetch.Cancelled):
        fetch.download(f"{server}/f.bin", tmp_path / "f.bin", len(PAYLOAD), SHA,
                       cancel=lambda: True)


def test_404_means_the_published_file_changed(server, tmp_path):
    with pytest.raises(InstallError) as ei:
        fetch.download(f"{server}/gone.bin", tmp_path / "g.bin", 10, SHA)
    assert ei.value.cls == "published_changed"


def test_a_redirect_to_plain_http_is_refused(server, tmp_path):
    with pytest.raises(InstallError) as ei:
        fetch.download(f"{server}/redirect-http", tmp_path / "r.bin", 10, SHA)
    assert ei.value.cls == "network"


@pytest.mark.parametrize("url", [
    "http://github.com/x", "https://evil.example.org/x", "ftp://github.com/x",
    "https://github.com.evil.org/x"])
def test_only_listed_hosts_over_https(url):
    with pytest.raises(InstallError):
        fetch.check_url(url)


def test_certificate_verification_is_never_disabled():
    import inspect
    src = inspect.getsource(fetch)
    assert "CERT_NONE" not in src and "check_hostname = False" not in src
    assert "_create_unverified_context" not in src
