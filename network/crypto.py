"""Session code generation, key derivation, and AES-256-GCM encryption.

The session code serves dual purpose: it is how peers join a session AND
how the encryption key is derived.  No separate key exchange step.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import socket
import string
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

# ── constants ─────────────────────────────────────────────────

_CODE_ALPHABET = string.ascii_uppercase + string.digits  # A-Z 0-9
_CODE_LENGTH = 8  # XXXX-XXXX (displayed with hyphen, stored without)
_SALT = hashlib.sha256(b"voxterm-p2p-v1").digest()
_INFO = b"voxterm-session-key"
_KEY_LENGTH = 32  # AES-256
_NONCE_LENGTH = 12  # GCM standard
_TAG_LENGTH = 16  # GCM tag is appended by AESGCM

_TCP_HEADER = struct.Struct("<I")  # uint32 LE frame length
_MAX_MSG_SIZE = 10_000_000  # 10MB sanity limit

log = logging.getLogger("p2p.crypto")


class DecryptionError(Exception):
    """Raised when decryption fails (wrong key, tampered data, bad nonce)."""


# ── session codes ─────────────────────────────────────────────

def generate_session_code() -> str:
    """Generate a random session code in XXXX-XXXX format."""
    chars = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
    return f"{chars[:4]}-{chars[4:]}"


def normalize_session_code(code: str) -> str:
    """Strip hyphens/spaces and uppercase for key derivation."""
    return code.replace("-", "").replace(" ", "").upper()


# ── key derivation ────────────────────────────────────────────

def derive_session_key(session_code: str) -> bytes:
    """Derive a 256-bit symmetric key from a session code via HKDF-SHA256."""
    normalized = normalize_session_code(session_code)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=_SALT,
        info=_INFO,
    )
    key = hkdf.derive(normalized.encode("utf-8"))
    log.debug("Key derived for session ...%s", normalized[-4:])
    return key


# ── AES-256-GCM primitives ───────────────────────────────────

def encrypt(key: bytes, plaintext: bytes, nonce: bytes | None = None) -> tuple[bytes, bytes]:
    """Encrypt with AES-256-GCM.

    Returns (nonce, ciphertext_with_tag).
    The AESGCM class appends the 16-byte tag to the ciphertext.
    """
    if nonce is None:
        nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ct


def decrypt(key: bytes, nonce: bytes, ciphertext_with_tag: bytes) -> bytes:
    """Decrypt AES-256-GCM.  Raises DecryptionError on failure."""
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ciphertext_with_tag, None)
    except Exception as exc:
        raise DecryptionError(str(exc)) from exc


# ── TCP encrypted framing ────────────────────────────────────
#
# Wire format:
#   [4 bytes: uint32 LE total frame length (covers nonce + ct)]
#   [12 bytes: GCM nonce]
#   [N bytes: AES-256-GCM ciphertext + 16-byte tag]

def send_encrypted_msg(sock: socket.socket, key: bytes, msg: dict) -> None:
    """Send a length-prefixed, AES-256-GCM encrypted JSON message over TCP."""
    plaintext = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    nonce, ct = encrypt(key, plaintext)
    frame = nonce + ct
    log.debug("TX encrypted msg: %d bytes plaintext, type=%s", len(plaintext), msg.get("type"))
    sock.sendall(_TCP_HEADER.pack(len(frame)) + frame)


def recv_encrypted_msg(sock: socket.socket, key: bytes) -> dict | None:
    """Read a length-prefixed, encrypted JSON message from TCP.

    Returns None on EOF or decryption failure.
    """
    header = _recv_exact(sock, _TCP_HEADER.size)
    if header is None:
        log.debug("RX: EOF reading header")
        return None
    (length,) = _TCP_HEADER.unpack(header)
    if length > _MAX_MSG_SIZE or length < _NONCE_LENGTH + _TAG_LENGTH:
        log.warning("RX: invalid frame length %d (min=%d, max=%d)",
                     length, _NONCE_LENGTH + _TAG_LENGTH, _MAX_MSG_SIZE)
        return None
    frame = _recv_exact(sock, length)
    if frame is None:
        log.debug("RX: EOF reading frame body (%d bytes expected)", length)
        return None
    nonce = frame[:_NONCE_LENGTH]
    ct = frame[_NONCE_LENGTH:]
    try:
        plaintext = decrypt(key, nonce, ct)
    except DecryptionError:
        log.warning("RX: decryption failed (frame=%d bytes)", length)
        return None
    msg = json.loads(plaintext.decode("utf-8"))
    log.debug("RX decrypted msg: %d bytes, type=%s", len(plaintext), msg.get("type") if isinstance(msg, dict) else "?")
    return msg


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from a socket.  Returns None on EOF."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


# ── UDP encrypted audio frames ────────────────────────────────
#
# Datagram format:
#   [4 bytes] magic: 0x564F5854 ("VOXT")
#   [16 bytes] node_id (UUID bytes)
#   [4 bytes] sequence number (uint32 LE, plaintext — needed for nonce)
#   [12 bytes] nonce
#   [N bytes] AES-256-GCM encrypted payload (PCM + 16-byte tag)

_UDP_MAGIC = b"VOXT"
_UDP_HEADER = struct.Struct("<4s16sI")  # magic + node_id + seq


def encrypt_audio_frame(
    key: bytes,
    node_id: bytes,
    seq: int,
    timestamp: float,
    pcm_bytes: bytes,
) -> bytes:
    """Build an encrypted UDP audio datagram."""
    # Embed timestamp in the plaintext payload (8 bytes float64 LE + PCM)
    payload = struct.pack("<d", timestamp) + pcm_bytes
    nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, payload, None)
    header = _UDP_HEADER.pack(_UDP_MAGIC, node_id[:16].ljust(16, b"\x00"), seq)
    datagram = header + nonce + ct
    log.debug("UDP TX: encrypted frame seq=%d, %d bytes PCM, %d bytes datagram", seq, len(pcm_bytes), len(datagram))
    return datagram


def decrypt_audio_frame(
    key: bytes,
    datagram: bytes,
) -> tuple[bytes, bytes, int, float] | None:
    """Decrypt a UDP audio datagram.

    Returns (node_id, pcm_bytes, seq, timestamp) or None on failure.
    """
    min_size = _UDP_HEADER.size + _NONCE_LENGTH + _TAG_LENGTH + 8  # 8 for timestamp
    if len(datagram) < min_size:
        log.debug("UDP RX: datagram too small (%d < %d bytes)", len(datagram), min_size)
        return None
    magic, node_id, seq = _UDP_HEADER.unpack_from(datagram)
    if magic != _UDP_MAGIC:
        log.debug("UDP RX: bad magic %r", magic)
        return None
    offset = _UDP_HEADER.size
    nonce = datagram[offset : offset + _NONCE_LENGTH]
    ct = datagram[offset + _NONCE_LENGTH :]
    aesgcm = AESGCM(key)
    try:
        payload = aesgcm.decrypt(nonce, ct, None)
    except Exception:
        log.warning("UDP RX: decrypt failed for seq=%d", seq)
        return None
    timestamp = struct.unpack("<d", payload[:8])[0]
    pcm_bytes = payload[8:]
    log.debug("UDP RX: decrypted frame seq=%d, %d bytes PCM", seq, len(pcm_bytes))
    return node_id.rstrip(b"\x00"), pcm_bytes, seq, timestamp
