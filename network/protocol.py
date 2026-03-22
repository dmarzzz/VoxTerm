"""P2P wire protocol — message definitions and builders.

Follows the same pattern as diarization/ipc.py: string constants for
message types, plain dicts for messages, JSON serialization.
"""

from __future__ import annotations

import time

# ── message types ─────────────────────────────────────────────

MSG_HELLO = "hello"
MSG_HEARTBEAT = "heartbeat"
MSG_HEARTBEAT_ACK = "heartbeat_ack"
MSG_PARTIAL = "partial"
MSG_FINAL = "final"
MSG_BYE = "bye"
MSG_AUDIO_FRAME = "audio_frame"  # UDP only — binary, not JSON

# Fields required per message type (for validation)
_REQUIRED_FIELDS: dict[str, set[str]] = {
    MSG_HELLO: {"type", "node_id", "display_name", "proto_v", "sample_rate", "channels", "encoding"},
    MSG_HEARTBEAT: {"type", "node_id", "local_ts", "seq"},
    MSG_HEARTBEAT_ACK: {"type", "node_id", "local_ts", "echo_ts", "echo_node_id"},
    MSG_PARTIAL: {"type", "node_id", "speaker_name", "seq", "text", "start_ts"},
    MSG_FINAL: {"type", "node_id", "speaker_name", "seq", "text", "start_ts", "end_ts", "confidence"},
    MSG_BYE: {"type", "node_id", "reason"},
}


# ── builders ──────────────────────────────────────────────────

def build_hello(
    node_id: str,
    display_name: str,
    proto_v: int = 1,
    sample_rate: int = 16000,
    channels: int = 1,
    encoding: str = "pcm_s16le",
) -> dict:
    return {
        "type": MSG_HELLO,
        "node_id": node_id,
        "display_name": display_name,
        "proto_v": proto_v,
        "sample_rate": sample_rate,
        "channels": channels,
        "encoding": encoding,
    }


def build_heartbeat(node_id: str, seq: int, local_ts: float | None = None) -> dict:
    return {
        "type": MSG_HEARTBEAT,
        "node_id": node_id,
        "local_ts": local_ts if local_ts is not None else time.monotonic(),
        "seq": seq,
    }


def build_heartbeat_ack(
    node_id: str,
    echo_ts: float,
    echo_node_id: str,
    local_ts: float | None = None,
) -> dict:
    return {
        "type": MSG_HEARTBEAT_ACK,
        "node_id": node_id,
        "local_ts": local_ts if local_ts is not None else time.monotonic(),
        "echo_ts": echo_ts,
        "echo_node_id": echo_node_id,
    }


def build_partial(
    node_id: str,
    speaker_name: str,
    seq: int,
    text: str,
    start_ts: float,
) -> dict:
    return {
        "type": MSG_PARTIAL,
        "node_id": node_id,
        "speaker_name": speaker_name,
        "seq": seq,
        "text": text,
        "start_ts": start_ts,
    }


def build_final(
    node_id: str,
    speaker_name: str,
    seq: int,
    text: str,
    start_ts: float,
    end_ts: float,
    confidence: float,
) -> dict:
    return {
        "type": MSG_FINAL,
        "node_id": node_id,
        "speaker_name": speaker_name,
        "seq": seq,
        "text": text,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "confidence": confidence,
    }


def build_bye(node_id: str, reason: str = "user_quit") -> dict:
    return {
        "type": MSG_BYE,
        "node_id": node_id,
        "reason": reason,
    }


# ── validation ────────────────────────────────────────────────

def validate_message(msg: dict) -> bool:
    """Check that a message dict has all required fields for its type."""
    if not isinstance(msg, dict):
        return False
    msg_type = msg.get("type")
    if msg_type not in _REQUIRED_FIELDS:
        return False
    return _REQUIRED_FIELDS[msg_type].issubset(msg.keys())
