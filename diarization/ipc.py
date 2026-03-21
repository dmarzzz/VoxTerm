"""Binary IPC protocol for main process ↔ diarizer subprocess communication.

Wire format per message:
    [4 bytes: uint32 LE payload length] [N bytes: JSON payload]

Binary blobs (audio, embeddings) are base64-encoded inside the JSON
to keep the protocol simple and debuggable while avoiding a dependency.
Typical overhead is negligible vs the ~100ms diarization latency.
"""

from __future__ import annotations

import json
import struct
import sys
from typing import IO

import numpy as np

# ── message types ─────────────────────────────────────────────

# Subprocess → main
MSG_READY = "ready"
MSG_RESULT = "result"
MSG_PONG = "pong"
MSG_ACK = "ack"
MSG_ERROR = "error"

# Main → subprocess
MSG_IDENTIFY = "identify"
MSG_IDENTIFY_MULTI = "identify_multi"
MSG_SET_NAME = "set_name"
MSG_GET_STATE = "get_state"
MSG_GET_EMBEDDINGS = "get_embeddings"
MSG_GET_CENTROID = "get_centroid"
MSG_IS_STABLE = "is_stable"
MSG_MARK_MATCHED = "mark_matched"
MSG_IS_MATCHED = "is_matched"
MSG_MERGE = "merge"
MSG_RESET = "reset"
MSG_PING = "ping"
MSG_SHUTDOWN = "shutdown"
MSG_GET_COLOR = "get_color"
MSG_GET_NAME = "get_name"
MSG_GET_NAMES = "get_names"
MSG_NUM_SPEAKERS = "num_speakers"

_HEADER = struct.Struct("<I")  # uint32 little-endian


# ── encoding helpers ──────────────────────────────────────────

def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as a hex string for JSON transport."""
    return arr.astype(np.float32).tobytes().hex()


def decode_array(hex_str: str) -> np.ndarray:
    """Decode a hex string back to a float32 numpy array."""
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.float32).copy()


# ── send / receive ────────────────────────────────────────────

def send_msg(pipe: IO[bytes], msg: dict) -> None:
    """Send a length-prefixed JSON message to a pipe."""
    payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    pipe.write(_HEADER.pack(len(payload)))
    pipe.write(payload)
    pipe.flush()


def recv_msg(pipe: IO[bytes]) -> dict | None:
    """Read a length-prefixed JSON message from a pipe.

    Returns None on EOF (subprocess exited).
    """
    header = _read_exact(pipe, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > 50_000_000:  # sanity: 50MB max
        return None
    payload = _read_exact(pipe, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _read_exact(pipe: IO[bytes], n: int) -> bytes | None:
    """Read exactly n bytes from a pipe. Returns None on EOF."""
    data = b""
    while len(data) < n:
        chunk = pipe.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data
