"""Tests for the out-of-process event stream (tui/events.py)."""
import json
import threading

import pytest

from tui.events import EventLogger, NullEventLogger


def test_disabled_logger_is_noop(tmp_path):
    """NullEventLogger and an EventLogger constructed with path=None never write."""
    null = NullEventLogger()
    null.open()
    null.emit("text", text="hello")
    null.close()
    assert not list(tmp_path.iterdir())  # nothing on disk

    disabled = EventLogger(path=None)
    disabled.open()
    disabled.emit("text", text="hello")
    disabled.close()
    assert disabled.enabled is False


def test_emit_writes_one_jsonl_record_per_call(tmp_path):
    path = tmp_path / "out.jsonl"
    log = EventLogger(path)
    log.open()
    log.emit("text", speaker="Speaker 1", text="hello")
    log.emit("vad", on=True)
    log.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["kind"] == "text" and rec0["text"] == "hello"
    assert rec1["kind"] == "vad" and rec1["on"] is True
    # Every record carries a timestamp
    assert isinstance(rec0["t"], float)


def test_emit_before_open_is_silently_dropped(tmp_path):
    """Calling emit() before open() must not crash and must not create a file."""
    log = EventLogger(tmp_path / "out.jsonl")
    log.emit("text", text="early")  # no open() yet
    log.close()
    assert not (tmp_path / "out.jsonl").exists()


def test_emit_creates_parent_directory(tmp_path):
    """The log file lives under LIVE_DIR which may not exist yet."""
    path = tmp_path / "nested" / "deep" / "out.jsonl"
    log = EventLogger(path)
    log.open()
    log.emit("session", phase="start")
    log.close()
    assert path.exists()


def test_unserialisable_payload_drops_event_does_not_crash(tmp_path):
    path = tmp_path / "out.jsonl"
    log = EventLogger(path)
    log.open()
    # An object that isn't JSON-serialisable
    class Junk:
        pass
    log.emit("text", obj=Junk())
    # A valid event after a bad one still lands
    log.emit("text", text="ok")
    log.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "ok"


def test_concurrent_emit_is_thread_safe(tmp_path):
    """Many threads emitting simultaneously must produce intact JSON per line."""
    path = tmp_path / "out.jsonl"
    log = EventLogger(path)
    log.open()

    def producer():
        for i in range(100):
            log.emit("amplitude", rms=0.1)

    threads = [threading.Thread(target=producer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 800
    # Every line must parse — no torn writes.
    for ln in lines:
        rec = json.loads(ln)
        assert rec["kind"] == "amplitude"


def test_close_is_idempotent(tmp_path):
    log = EventLogger(tmp_path / "out.jsonl")
    log.open()
    log.emit("session", phase="start")
    log.close()
    log.close()  # second close must not raise


def test_open_is_idempotent(tmp_path):
    log = EventLogger(tmp_path / "out.jsonl")
    log.open()
    log.open()  # no-op
    log.emit("session", phase="start")
    log.close()
    assert len(((tmp_path / "out.jsonl").read_text()).splitlines()) == 1


def test_emit_after_close_is_silently_dropped(tmp_path):
    path = tmp_path / "out.jsonl"
    log = EventLogger(path)
    log.open()
    log.emit("text", text="first")
    log.close()
    log.emit("text", text="after-close")  # must not raise, must not write

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "first"
