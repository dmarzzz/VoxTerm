"""Tests for P2P wire protocol message definitions and validation."""

import pytest

from network.protocol import (
    MSG_HELLO,
    MSG_HEARTBEAT,
    MSG_HEARTBEAT_ACK,
    MSG_PARTIAL,
    MSG_FINAL,
    MSG_BYE,
    build_hello,
    build_heartbeat,
    build_heartbeat_ack,
    build_partial,
    build_final,
    build_bye,
    validate_message,
)


class TestBuilders:
    def test_build_hello(self):
        msg = build_hello("node-1", "halcyon")
        assert msg["type"] == MSG_HELLO
        assert msg["node_id"] == "node-1"
        assert msg["display_name"] == "halcyon"
        assert msg["proto_v"] == 1
        assert msg["sample_rate"] == 16000
        assert msg["channels"] == 1
        assert msg["encoding"] == "pcm_s16le"

    def test_build_hello_custom(self):
        msg = build_hello("n", "bob", proto_v=2, sample_rate=48000, channels=2, encoding="opus")
        assert msg["proto_v"] == 2
        assert msg["sample_rate"] == 48000

    def test_build_heartbeat(self):
        msg = build_heartbeat("node-1", seq=42, local_ts=100.5)
        assert msg["type"] == MSG_HEARTBEAT
        assert msg["node_id"] == "node-1"
        assert msg["seq"] == 42
        assert msg["local_ts"] == 100.5

    def test_build_heartbeat_auto_ts(self):
        msg = build_heartbeat("node-1", seq=0)
        assert msg["local_ts"] > 0  # auto-filled with time.monotonic()

    def test_build_heartbeat_ack(self):
        msg = build_heartbeat_ack("node-2", echo_ts=100.5, echo_node_id="node-1", local_ts=100.6)
        assert msg["type"] == MSG_HEARTBEAT_ACK
        assert msg["echo_ts"] == 100.5
        assert msg["echo_node_id"] == "node-1"

    def test_build_partial(self):
        msg = build_partial("node-1", "halcyon", seq=7, text="hello world", start_ts=50.0)
        assert msg["type"] == MSG_PARTIAL
        assert msg["text"] == "hello world"
        assert msg["seq"] == 7

    def test_build_final(self):
        msg = build_final("node-1", "halcyon", seq=7, text="hello world", start_ts=50.0, end_ts=52.5, confidence=0.95)
        assert msg["type"] == MSG_FINAL
        assert msg["end_ts"] == 52.5
        assert msg["confidence"] == 0.95

    def test_build_bye(self):
        msg = build_bye("node-1")
        assert msg["type"] == MSG_BYE
        assert msg["reason"] == "user_quit"

    def test_build_bye_custom_reason(self):
        msg = build_bye("node-1", reason="crash")
        assert msg["reason"] == "crash"


class TestValidation:
    def test_all_builders_produce_valid_messages(self):
        messages = [
            build_hello("n", "alice"),
            build_heartbeat("n", 0, 1.0),
            build_heartbeat_ack("n", 1.0, "m", 1.1),
            build_partial("n", "alice", 1, "hi", 1.0),
            build_final("n", "alice", 1, "hi there", 1.0, 2.0, 0.9),
            build_bye("n"),
        ]
        for msg in messages:
            assert validate_message(msg), f"Failed for {msg['type']}"

    def test_missing_field_is_invalid(self):
        msg = build_hello("n", "alice")
        del msg["display_name"]
        assert not validate_message(msg)

    def test_wrong_type_is_invalid(self):
        assert not validate_message({"type": "unknown_msg"})

    def test_not_a_dict_is_invalid(self):
        assert not validate_message("hello")
        assert not validate_message(None)
        assert not validate_message(42)

    def test_missing_type_is_invalid(self):
        assert not validate_message({"node_id": "n"})

    def test_extra_fields_are_ok(self):
        msg = build_hello("n", "alice")
        msg["extra"] = "data"
        assert validate_message(msg)


class TestJsonRoundTrip:
    """Verify messages survive JSON serialization."""

    def test_all_types_json_round_trip(self):
        import json

        messages = [
            build_hello("n", "alice"),
            build_heartbeat("n", 0, 1.0),
            build_heartbeat_ack("n", 1.0, "m", 1.1),
            build_partial("n", "alice", 1, "hi", 1.0),
            build_final("n", "alice", 1, "hi there", 1.0, 2.0, 0.9),
            build_bye("n"),
        ]
        for msg in messages:
            serialized = json.dumps(msg, separators=(",", ":"))
            deserialized = json.loads(serialized)
            assert deserialized == msg
            assert validate_message(deserialized)
