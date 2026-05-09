"""Tests for the hivemind transcript-sink integration.

Covers:

  * mDNS discovery (mocked zeroconf ServiceInfo)
  * Transcript batch POST shape (in-process HTTP server)
  * Flush cadence: 60s, 30 segments, EOF/close()
  * Persistent device_id across runs
  * origin_device wiring through to the wire payload
  * No-sink path: log + drop, no POST attempt
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from network import hivemind
from network.hivemind import (
    HIVEMIND_SERVICE_TYPE,
    HivemindBrowser,
    HivemindClient,
    HivemindMode,
    Sink,
    configure,
    get_or_create_device_id,
)


# ── helpers ─────────────────────────────────────────────────────────────


class _CapturingPoster:
    """Stand-in for ``post_batch`` that records the (sink, body) it saw."""

    def __init__(self, response: dict | None = None, exc: Exception | None = None):
        self.calls: list[tuple[Sink, dict]] = []
        self.response = response or {"cid": "abc123"}
        self.exc = exc

    def __call__(self, sink: Sink, batch: dict) -> dict:
        self.calls.append((sink, batch))
        if self.exc is not None:
            raise self.exc
        return self.response


class _FakeClock:
    """A deterministic monotonic-style clock for cadence tests."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def _make_client(
    *,
    poster: _CapturingPoster | None = None,
    sink: Sink | None = None,
    clock: _FakeClock | None = None,
    flush_seconds: float = 60.0,
    flush_segments: int = 30,
    device_id: str = "11111111-1111-4111-8111-111111111111",
    location: str = "",
    record_id: str | None = "transcript-test",
    mode: HivemindMode = HivemindMode.AUTO,
) -> tuple[HivemindClient, _CapturingPoster, _FakeClock]:
    poster = poster or _CapturingPoster()
    clock = clock or _FakeClock()
    sink = sink or Sink(host="127.0.0.1", port=7777, node="convent")
    client = HivemindClient(
        device_id=device_id,
        record_id=record_id,
        location=location,
        sink=sink,
        browser=None,
        mode=mode,
        flush_seconds=flush_seconds,
        flush_segments=flush_segments,
        clock=clock,
        poster=poster,
    )
    return client, poster, clock


# ── 1. mDNS discovery ───────────────────────────────────────────────────


class _FakeServiceInfo:
    """Just enough of zeroconf.ServiceInfo for HivemindBrowser._parse."""

    def __init__(
        self,
        host_bytes: bytes,
        port: int,
        properties: dict[bytes, bytes],
    ):
        import socket as _s
        self.addresses = [host_bytes]
        self.port = port
        self.properties = properties
        # Sanity check — _parse should still treat these as bytes.
        assert isinstance(host_bytes, (bytes, bytearray))


def test_mdns_discovery_finds_sink():
    import socket
    from zeroconf import ServiceStateChange

    info = _FakeServiceInfo(
        host_bytes=socket.inet_aton("192.168.1.42"),
        port=7777,
        properties={
            b"version": b"swf-bundle-v1",
            b"proto": b"shape-rotator-hivemind/v1",
            b"node": b"convent-box",
            b"pubkey": b"ed25519-deadbeef",
            b"port": b"7777",
        },
    )

    class _FakeZeroconf:
        def get_service_info(self, service_type, name):
            return info

    browser = HivemindBrowser()
    # Bypass start() so we don't open a real Zeroconf — just exercise
    # the state-change handler directly with our fake info.
    browser._on_state_change(
        zeroconf=_FakeZeroconf(),
        service_type=HIVEMIND_SERVICE_TYPE,
        name=f"convent-box.{HIVEMIND_SERVICE_TYPE}",
        state_change=ServiceStateChange.Added,
    )

    assert HIVEMIND_SERVICE_TYPE == "_sr-hivemind._tcp.local."
    sinks = browser.sinks()
    assert len(sinks) == 1
    s = sinks[0]
    assert s.host == "192.168.1.42"
    assert s.port == 7777
    assert s.pubkey == "ed25519-deadbeef"
    assert s.node == "convent-box"
    assert s.transcripts_url == "http://192.168.1.42:7777/hivemind/transcripts"
    assert browser.active_sink() == s


def test_mdns_browser_tracks_most_recent_advertisement():
    import socket
    from zeroconf import ServiceStateChange

    a = _FakeServiceInfo(
        socket.inet_aton("10.0.0.1"), 7777,
        {b"node": b"a", b"port": b"7777"},
    )
    b = _FakeServiceInfo(
        socket.inet_aton("10.0.0.2"), 7778,
        {b"node": b"b", b"port": b"7778"},
    )

    class _Zeroconf:
        def __init__(self):
            self._map = {}

        def add(self, name, info):
            self._map[name] = info

        def get_service_info(self, service_type, name):
            return self._map[name]

    zc = _Zeroconf()
    zc.add("a." + HIVEMIND_SERVICE_TYPE, a)
    zc.add("b." + HIVEMIND_SERVICE_TYPE, b)

    browser = HivemindBrowser()
    browser._on_state_change(zc, HIVEMIND_SERVICE_TYPE,
                             "a." + HIVEMIND_SERVICE_TYPE,
                             ServiceStateChange.Added)
    browser._on_state_change(zc, HIVEMIND_SERVICE_TYPE,
                             "b." + HIVEMIND_SERVICE_TYPE,
                             ServiceStateChange.Added)
    assert browser.active_sink().host == "10.0.0.2"

    # Removing the active one promotes the other one back to active.
    browser._on_state_change(zc, HIVEMIND_SERVICE_TYPE,
                             "b." + HIVEMIND_SERVICE_TYPE,
                             ServiceStateChange.Removed)
    assert browser.active_sink().host == "10.0.0.1"


# ── 2. Transcript batch POST ────────────────────────────────────────────


class _CapturingHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self):  # noqa: N802 (stdlib name)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = {"raw": body}
        self.__class__.received.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": parsed,
        })
        resp = json.dumps({"cid": "test-cid"}).encode("utf-8")
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *args, **kwargs):  # silence the test output
        pass


@pytest.fixture
def http_capture():
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_post_batch_to_sink(http_capture):
    """End-to-end: a real HTTP server captures the POST shape."""
    host, port = http_capture.server_address
    sink = Sink(host=host, port=port, node="test-sink")

    client = HivemindClient(
        device_id="22222222-2222-4222-8222-222222222222",
        record_id="transcript-test",
        location="test-room",
        sink=sink,
        mode=HivemindMode.AUTO,
    )

    client.add_segment(0.0, "alice", "hello")
    client.add_segment(1.5, "bob",   "world")
    assert client.flush_now() is True

    assert len(_CapturingHandler.received) == 1
    captured = _CapturingHandler.received[0]
    assert captured["path"] == "/hivemind/transcripts"
    assert captured["headers"]["Content-Type"] == "application/json"

    body = captured["body"]
    assert body["record_id"] == "transcript-test"
    assert body["batch_index"] == 0
    assert body["origin_device"] == "22222222-2222-4222-8222-222222222222"
    assert body["location"] == "test-room"
    assert "started_at" in body and body["started_at"].endswith("Z")
    assert "ended_at" in body and body["ended_at"].endswith("Z")
    assert body["segments"] == [
        {"t": 0.0, "speaker": "alice", "text": "hello"},
        {"t": 1.5, "speaker": "bob",   "text": "world"},
    ]
    assert client.batches_sent == 1


# ── 3. Flush cadence: 60s ──────────────────────────────────────────────


def test_flush_cadence_60s():
    client, poster, clock = _make_client(flush_seconds=60.0, flush_segments=1000)

    client.add_segment(0.0, "alice", "tick")
    assert poster.calls == [], "should not flush on first segment"

    # Walk forward 59 seconds — still under the threshold.
    clock.advance(59.0)
    assert client.maybe_flush() is False
    assert poster.calls == []

    # Cross 60s: nudge with a maybe_flush() (the cadence is checked
    # there as well as in add_segment).
    clock.advance(2.0)
    assert client.maybe_flush() is True
    assert len(poster.calls) == 1

    # Adding more segments after the flush starts a fresh window.
    client.add_segment(60.0, "bob", "next")
    assert len(poster.calls) == 1, "second batch shouldn't fire yet"
    clock.advance(61.0)
    assert client.maybe_flush() is True
    assert len(poster.calls) == 2
    # Sequential batch_indexes:
    assert poster.calls[0][1]["batch_index"] == 0
    assert poster.calls[1][1]["batch_index"] == 1


# ── 4. Flush cadence: 30 segments ──────────────────────────────────────


def test_flush_cadence_30_segments():
    client, poster, clock = _make_client(flush_seconds=10_000.0, flush_segments=30)

    for i in range(29):
        client.add_segment(float(i), "alice", f"seg-{i}")
    # Below the threshold — no flush.
    assert poster.calls == []

    # 30th segment trips the flush automatically.
    client.add_segment(29.0, "alice", "seg-29")
    assert len(poster.calls) == 1
    body = poster.calls[0][1]
    assert len(body["segments"]) == 30
    assert body["segments"][0]["text"] == "seg-0"
    assert body["segments"][-1]["text"] == "seg-29"


# ── 5. Flush on EOF / close() ──────────────────────────────────────────


def test_flush_on_eof():
    client, poster, _ = _make_client()

    client.add_segment(0.0, "alice", "leftover")
    # Cadence not tripped; nothing posted yet.
    assert poster.calls == []

    flushed = client.close()
    assert flushed is True
    assert len(poster.calls) == 1
    assert poster.calls[0][1]["segments"] == [
        {"t": 0.0, "speaker": "alice", "text": "leftover"},
    ]

    # After close, further segments are dropped silently.
    client.add_segment(1.0, "alice", "after-close")
    assert client.flush_now() is False
    assert len(poster.calls) == 1


def test_close_with_no_pending_segments_is_noop():
    client, poster, _ = _make_client()
    assert client.close() is False
    assert poster.calls == []


# ── 6. Persistent device_id ────────────────────────────────────────────


def test_device_id_persists_across_runs(tmp_path):
    p = tmp_path / "device_id"

    a = get_or_create_device_id(p)
    b = get_or_create_device_id(p)
    assert a == b

    # Validate it really is a v4 UUID, not just any string.
    parsed = uuid.UUID(a)
    assert parsed.version == 4

    # Persisted to disk in plain text.
    assert p.read_text(encoding="utf-8").strip() == a


def test_device_id_regenerates_on_corrupt_file(tmp_path):
    p = tmp_path / "device_id"
    p.write_text("not-a-uuid\n", encoding="utf-8")

    new_id = get_or_create_device_id(p)
    parsed = uuid.UUID(new_id)
    assert parsed.version == 4
    assert p.read_text(encoding="utf-8").strip() == new_id


# ── 7. origin_device threads through to the wire payload ───────────────


def test_origin_device_in_payload(tmp_path):
    p = tmp_path / "device_id"
    device_id = get_or_create_device_id(p)

    poster = _CapturingPoster()
    sink = Sink(host="127.0.0.1", port=7777)
    client = HivemindClient(
        device_id=device_id,
        record_id="transcript-x",
        sink=sink,
        poster=poster,
    )
    client.add_segment(0.0, "alice", "hi")
    client.flush_now()

    body = poster.calls[0][1]
    assert body["origin_device"] == device_id


# ── 8. No-sink path: log + drop, no POST attempt ───────────────────────


def test_no_sink_logs_and_drops(caplog):
    poster = _CapturingPoster()
    client = HivemindClient(
        device_id="33333333-3333-4333-8333-333333333333",
        sink=None,
        browser=None,
        mode=HivemindMode.AUTO,
        poster=poster,
    )

    with caplog.at_level("INFO", logger="voxterm.hivemind"):
        client.add_segment(0.0, "alice", "ghost batch")
        flushed = client.flush_now()

    assert flushed is False
    assert poster.calls == []
    assert client.batches_sent == 0
    assert client.batches_dropped == 1
    # Log mentions that we dropped — for human triage.
    assert any("no sink" in rec.message.lower() for rec in caplog.records)


# ── extras: hardening that earned its keep during development ──────────


def test_hivemind_off_is_total_noop():
    poster = _CapturingPoster()
    client = HivemindClient(
        device_id="44444444-4444-4444-8444-444444444444",
        sink=Sink(host="127.0.0.1", port=7777),
        mode=HivemindMode.OFF,
        poster=poster,
    )
    client.add_segment(0.0, "alice", "ignored")
    assert client.flush_now() is False
    assert poster.calls == []
    assert client.pending_segments == 0


def test_post_failure_drops_batch_does_not_raise():
    import urllib.error

    poster = _CapturingPoster(exc=urllib.error.URLError("connection refused"))
    client, _, _ = _make_client(poster=poster)
    client.add_segment(0.0, "alice", "doomed")
    flushed = client.flush_now()
    assert flushed is False
    assert client.batches_dropped == 1
    assert client.batches_sent == 0


def test_sink_from_url_accepts_host_port_only():
    s = Sink.from_url("http://convent.local:7777")
    assert s.host == "convent.local"
    assert s.port == 7777
    assert s.transcripts_url == "http://convent.local:7777/hivemind/transcripts"


def test_sink_from_url_rejects_garbage():
    with pytest.raises(ValueError):
        Sink.from_url("not a url")
    with pytest.raises(ValueError):
        Sink.from_url("ftp://convent.local:7777")


def test_hivemind_mode_parse():
    assert HivemindMode.parse(None) == HivemindMode.AUTO
    assert HivemindMode.parse("auto") == HivemindMode.AUTO
    assert HivemindMode.parse("ON") == HivemindMode.ON
    assert HivemindMode.parse("Off") == HivemindMode.OFF
    with pytest.raises(ValueError):
        HivemindMode.parse("noisy")


def test_configure_off_returns_none(tmp_path):
    client = configure(
        mode=HivemindMode.OFF,
        device_id_path=tmp_path / "device_id",
    )
    assert client is None


def test_configure_with_explicit_url_skips_mdns(tmp_path):
    client = configure(
        mode=HivemindMode.AUTO,
        sink_url="http://127.0.0.1:7777",
        device_id_path=tmp_path / "device_id",
    )
    assert client is not None
    sink = client.active_sink()
    assert sink is not None
    assert sink.host == "127.0.0.1"
    assert sink.port == 7777


def test_configure_on_mode_raises_when_no_sink(tmp_path, monkeypatch):
    """ON mode with no explicit URL and no mDNS hits should raise fast."""
    # Patch HivemindBrowser to a no-op so wait_for_sink returns None.
    class _NoOpBrowser:
        def start(self): pass
        def stop(self): pass
        def active_sink(self): return None
        def wait_for_sink(self, timeout): return None
        def sinks(self): return []

    monkeypatch.setattr(hivemind, "HivemindBrowser", _NoOpBrowser)

    with pytest.raises(RuntimeError, match="no sink discovered"):
        configure(
            mode=HivemindMode.ON,
            sink_url=None,
            device_id_path=tmp_path / "device_id",
            discovery_timeout=0.05,
        )


def test_empty_segments_are_filtered():
    client, poster, _ = _make_client()
    client.add_segment(0.0, "alice", "")
    client.add_segment(0.5, "alice", "   ")
    assert client.pending_segments == 0
    assert client.flush_now() is False
    assert poster.calls == []


def test_batch_index_monotonic_across_flushes():
    client, poster, clock = _make_client(flush_segments=2)
    client.add_segment(0.0, "alice", "a")
    client.add_segment(0.1, "alice", "b")  # triggers
    client.add_segment(0.2, "alice", "c")
    client.add_segment(0.3, "alice", "d")  # triggers
    client.add_segment(0.4, "alice", "e")
    client.close()
    indexes = [call[1]["batch_index"] for call in poster.calls]
    assert indexes == [0, 1, 2]
