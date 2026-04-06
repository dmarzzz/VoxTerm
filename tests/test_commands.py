"""Tests for the voice command interpreter and executor."""

import json
import http.server
import threading

import pytest

from commands.interpreter import interpret, Command
from commands.executor import execute


class _FakeCommandHandler(http.server.BaseHTTPRequestHandler):
    """Mimics llama-swap returning structured command JSON."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            utterance = body["messages"][1]["content"]

            # Simple intent mapping for tests
            if "play" in utterance.lower() and "music" in utterance.lower():
                reply = json.dumps({
                    "action": "play_music",
                    "params": {"query": ""},
                    "speech": "Playing music.",
                })
            elif "volume up" in utterance.lower():
                reply = json.dumps({
                    "action": "system",
                    "params": {"command": "volume_up"},
                    "speech": "Volume up.",
                })
            elif "save" in utterance.lower():
                reply = json.dumps({
                    "action": "save_transcript",
                    "params": {"filename": "meeting-notes"},
                    "speech": "Saved.",
                })
            elif "summarize" in utterance.lower():
                reply = json.dumps({
                    "action": "summarize",
                    "params": {"strength": "medium"},
                    "speech": "Summarizing.",
                })
            else:
                reply = json.dumps({
                    "action": "unknown",
                    "params": {"original": utterance},
                    "speech": "I didn't understand that.",
                })

            response = {
                "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


@pytest.fixture()
def fake_llm():
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeCommandHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_interpret_play_music(fake_llm):
    cmd = interpret("play my music files", base_url=fake_llm)
    assert cmd.action == "play_music"
    assert cmd.speech == "Playing music."


def test_interpret_system_command(fake_llm):
    cmd = interpret("turn the volume up", base_url=fake_llm)
    assert cmd.action == "system"
    assert cmd.params["command"] == "volume_up"


def test_interpret_save(fake_llm):
    cmd = interpret("save this as meeting notes", base_url=fake_llm)
    assert cmd.action == "save_transcript"
    assert cmd.params["filename"] == "meeting-notes"


def test_interpret_summarize(fake_llm):
    cmd = interpret("summarize what we discussed", base_url=fake_llm)
    assert cmd.action == "summarize"


def test_interpret_unknown(fake_llm):
    cmd = interpret("flurble garble nonsense", base_url=fake_llm)
    assert cmd.action == "unknown"


def test_interpret_empty():
    cmd = interpret("")
    assert cmd.action == "unknown"


def test_interpret_server_down():
    from summarizer.engine import SummarizerError
    with pytest.raises(SummarizerError):
        interpret("play music", base_url="http://127.0.0.1:1")


def test_execute_unknown():
    result = execute("unknown", {"original": "blah"})
    assert "Didn't understand" in result


def test_execute_save_transcript(tmp_path):
    import paths
    old_dir = paths.SESSIONS_DIR
    paths.SESSIONS_DIR = tmp_path
    try:
        result = execute(
            "save_transcript",
            {"filename": "test-meeting"},
            {"transcript_markdown": "# Test\n\nHello world"},
        )
        assert "Saved" in result
        assert (tmp_path / "test-meeting.md").exists()
        assert "Hello world" in (tmp_path / "test-meeting.md").read_text()
    finally:
        paths.SESSIONS_DIR = old_dir


def test_execute_summarize_delegates():
    result = execute("summarize", {"strength": "medium"})
    assert result == "__delegate_summarize__"
