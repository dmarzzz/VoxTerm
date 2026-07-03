import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from network.upload import (
    TranscriptUploadClient,
    UploadMode,
    UploadQueue,
    build_upload_record,
)
import tui.app as tui_app
from tui.app import UploadSettingsScreen, VoxTerm


def test_upload_mode_parse_and_flags():
    assert UploadMode.parse("local").saves_local is True
    assert UploadMode.parse("local").uploads_remote is False
    assert UploadMode.parse("local+remote") is UploadMode.LOCAL_AND_REMOTE
    assert UploadMode.parse("local+remote").saves_local is True
    assert UploadMode.parse("local+remote").uploads_remote is True
    assert UploadMode.parse("remote-only") is UploadMode.REMOTE_ONLY
    assert UploadMode.parse("remote-only").saves_local is False
    assert UploadMode.parse("bad-value") is UploadMode.LOCAL_ONLY


class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.captured.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
            }
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        return


def _server():
    httpd = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    httpd.captured = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}/upload"


def test_enqueue_and_flush_posts_json_with_auth(tmp_path):
    httpd, endpoint = _server()
    try:
        queue = UploadQueue(tmp_path / "queue.jsonl")
        client = TranscriptUploadClient(
            endpoint=endpoint,
            auth_token="secret-token",
            queue=queue,
        )

        assert client.enqueue("hello transcript", {"session_id": "s1"}) == 1
        result = client.flush()

        assert result.sent == 1
        assert result.pending == 0
        assert queue.load() == []
        captured = httpd.captured[0]  # type: ignore[attr-defined]
        assert captured["path"] == "/upload"
        assert captured["headers"]["Authorization"] == "Bearer secret-token"
        payload = json.loads(captured["body"].decode("utf-8"))
        assert payload["kind"] == "voxterm.finalized_transcript"
        assert payload["metadata"]["session_id"] == "s1"
        assert payload["transcript_markdown"] == "hello transcript"
        assert "attempts" not in payload
        assert "last_error" not in payload
    finally:
        httpd.shutdown()


def test_failed_flush_keeps_record_in_queue(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    client = TranscriptUploadClient(endpoint="http://example.invalid/upload", queue=queue)
    client.enqueue("hello", {})

    def fail(record):
        raise RuntimeError("boom")

    client._post = fail  # type: ignore[method-assign]
    result = client.flush()

    assert result.sent == 0
    assert result.pending == 1
    assert result.last_error == "boom"
    [record] = queue.load()
    assert record["attempts"] == 1
    assert record["last_error"] == "boom"


def test_missing_endpoint_leaves_queue_unattempted(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    client = TranscriptUploadClient(endpoint="", queue=queue)
    client.enqueue("hello", {})

    result = client.flush()

    assert result.attempted == 0
    assert result.pending == 1
    assert result.last_error == "missing upload endpoint"
    [record] = queue.load()
    assert record["attempts"] == 0


def test_queue_status_reports_pending_attempted_and_last_error(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    queue.replace(
        [
            {"id": "a", "attempts": 0, "last_error": ""},
            {"id": "b", "attempts": 2, "last_error": "timeout"},
        ]
    )

    status = queue.status()

    assert status.pending == 2
    assert status.attempted == 1
    assert status.last_error == "timeout"


def test_include_audio_embeds_retained_audio(tmp_path):
    wav = tmp_path / "take.wav"
    wav.write_bytes(b"RIFFfake")

    record = build_upload_record(
        "hello",
        {"session_id": "s1"},
        include_audio=True,
        audio_path=wav,
    )

    assert record["metadata"]["audio_included"] is True
    assert record["audio"]["filename"] == "take.wav"
    assert record["audio"]["content_type"] == "audio/wav"
    assert base64.b64decode(record["audio"]["base64"]) == b"RIFFfake"


def test_include_audio_without_file_marks_unavailable():
    record = build_upload_record("hello", {}, include_audio=True)

    assert record["audio"] is None
    assert record["metadata"]["audio_included"] is False
    assert "no retained audio" in record["metadata"]["audio_note"]


def test_upload_settings_status_includes_queue_state():
    screen = UploadSettingsScreen(
        mode="local_remote",
        endpoint="https://example.test/upload",
        include_audio=False,
        has_token=True,
        queue_pending=3,
        queue_last_error="temporary upstream outage with a verbose message",
    )

    status = screen._status_text()

    assert "Local + Remote" in status
    assert "token saved" in status
    assert "queue 3 pending" in status
    assert "last error" in status


def test_manual_upload_flush_uses_current_config(monkeypatch):
    app = object.__new__(VoxTerm)
    messages = []
    flushes = []

    class _Config:
        def get(self, key):
            return {"upload_endpoint": "https://upload.test"}[key]

    class _Secrets:
        def get(self, key):
            assert key == "upload_auth_token"
            return "token"

    class _Status:
        pending = 2
        last_error = ""

    class _Queue:
        def status(self):
            return _Status()

    class _Transcript:
        def system_message(self, message, *args, **kwargs):
            messages.append(message)

    monkeypatch.setattr(tui_app, "_get_config", lambda: _Config())
    monkeypatch.setattr(tui_app, "SecretStore", _Secrets)
    monkeypatch.setattr(tui_app, "UploadQueue", _Queue)
    monkeypatch.setattr(app, "query_one", lambda _cls: _Transcript())
    monkeypatch.setattr(
        app,
        "_flush_upload_queue",
        lambda endpoint, token: flushes.append((endpoint, token)),
    )

    app._flush_upload_queue_from_config()

    assert messages == ["remote upload flush requested (2 pending)"]
    assert flushes == [("https://upload.test", "token")]
