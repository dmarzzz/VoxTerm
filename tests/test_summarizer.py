"""Tests for the summarizer engine."""

import json
import http.server
import threading

import pytest

from summarizer.engine import (
    summarize,
    is_server_available,
    SummarizerError,
    _SYSTEM_PROMPTS,
    DEFAULT_BASE_URL,
)


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler that mimics llama-server's OpenAI-compatible API."""

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
            # Echo back the system prompt strength hint so tests can verify
            system_msg = body["messages"][0]["content"]
            reply = f"Summary of transcript. System was: {system_msg[:40]}"
            response = {
                "choices": [
                    {"message": {"content": reply}, "finish_reason": "stop"}
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # silence request logs during tests


@pytest.fixture()
def fake_llm_server():
    """Start a fake llama-server on a random port, return its base URL."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_summarize_returns_text(fake_llm_server):
    result = summarize("Hello world transcript", base_url=fake_llm_server)
    assert "Summary of transcript" in result


def test_summarize_empty_transcript():
    result = summarize("   ")
    assert result == ""


def test_summarize_low_strength(fake_llm_server):
    result = summarize("Some transcript", strength="low", base_url=fake_llm_server)
    assert "concise meeting summarizer" in result


def test_summarize_high_strength(fake_llm_server):
    result = summarize("Some transcript", strength="high", base_url=fake_llm_server)
    assert "detailed meeting minutes" in result


def test_summarize_server_unreachable():
    with pytest.raises(SummarizerError, match="Cannot reach LLM server"):
        summarize("transcript", base_url="http://127.0.0.1:1")


def test_is_server_available_true(fake_llm_server):
    assert is_server_available(base_url=fake_llm_server) is True


def test_is_server_available_false():
    assert is_server_available(base_url="http://127.0.0.1:1") is False


def test_all_strengths_have_prompts():
    for strength in ("low", "medium", "high"):
        assert strength in _SYSTEM_PROMPTS
        assert len(_SYSTEM_PROMPTS[strength]) > 20
