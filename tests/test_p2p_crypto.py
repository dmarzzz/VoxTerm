"""Tests for P2P encryption: session codes, key derivation, AES-256-GCM."""

import json
import socket
import threading

import pytest

from network.crypto import (
    DecryptionError,
    generate_session_code,
    normalize_session_code,
    derive_session_key,
    encrypt,
    decrypt,
    send_encrypted_msg,
    recv_encrypted_msg,
    encrypt_audio_frame,
    decrypt_audio_frame,
)


class TestSessionCodes:
    def test_format(self):
        code = generate_session_code()
        assert len(code) == 9  # XXXX-XXXX
        assert code[4] == "-"
        parts = code.replace("-", "")
        assert len(parts) == 8
        assert parts.isalnum()
        assert parts == parts.upper()

    def test_uniqueness(self):
        codes = {generate_session_code() for _ in range(100)}
        assert len(codes) == 100  # all unique (astronomically unlikely to collide)

    def test_normalize(self):
        assert normalize_session_code("ABCD-1234") == "ABCD1234"
        assert normalize_session_code("abcd-1234") == "ABCD1234"
        assert normalize_session_code("AB CD 12 34") == "ABCD1234"


class TestKeyDerivation:
    def test_deterministic(self):
        k1 = derive_session_key("ABCD-1234")
        k2 = derive_session_key("ABCD-1234")
        assert k1 == k2

    def test_key_length(self):
        key = derive_session_key("ABCD-1234")
        assert len(key) == 32  # 256 bits

    def test_different_codes_different_keys(self):
        k1 = derive_session_key("ABCD-1234")
        k2 = derive_session_key("WXYZ-5678")
        assert k1 != k2

    def test_normalization_applied(self):
        k1 = derive_session_key("ABCD-1234")
        k2 = derive_session_key("abcd1234")
        assert k1 == k2


class TestAESGCM:
    def test_encrypt_decrypt_round_trip(self):
        key = derive_session_key("TEST-CODE")
        plaintext = b"hello, world!"
        nonce, ct = encrypt(key, plaintext)
        result = decrypt(key, nonce, ct)
        assert result == plaintext

    def test_wrong_key_fails(self):
        key1 = derive_session_key("CODE-AAAA")
        key2 = derive_session_key("CODE-BBBB")
        nonce, ct = encrypt(key1, b"secret")
        with pytest.raises(DecryptionError):
            decrypt(key2, nonce, ct)

    def test_tampered_ciphertext_fails(self):
        key = derive_session_key("TEST-CODE")
        nonce, ct = encrypt(key, b"secret")
        tampered = bytearray(ct)
        tampered[0] ^= 0xFF
        with pytest.raises(DecryptionError):
            decrypt(key, nonce, bytes(tampered))

    def test_nonce_uniqueness(self):
        key = derive_session_key("TEST-CODE")
        nonces = set()
        for _ in range(100):
            nonce, _ = encrypt(key, b"data")
            nonces.add(nonce)
        assert len(nonces) == 100

    def test_explicit_nonce(self):
        key = derive_session_key("TEST-CODE")
        nonce = b"\x01" * 12
        n, ct = encrypt(key, b"data", nonce=nonce)
        assert n == nonce
        assert decrypt(key, nonce, ct) == b"data"

    def test_empty_plaintext(self):
        key = derive_session_key("TEST-CODE")
        nonce, ct = encrypt(key, b"")
        assert decrypt(key, nonce, ct) == b""

    def test_large_plaintext(self):
        key = derive_session_key("TEST-CODE")
        data = b"x" * 100_000
        nonce, ct = encrypt(key, data)
        assert decrypt(key, nonce, ct) == data


class TestTCPEncryptedFraming:
    """Test send/recv encrypted messages over real TCP sockets."""

    def _socket_pair(self):
        """Create a connected TCP socket pair on loopback."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        conn, _ = server.accept()
        server.close()
        return client, conn

    def test_send_recv_round_trip(self):
        key = derive_session_key("TEST-CODE")
        client, conn = self._socket_pair()
        try:
            msg = {"type": "hello", "data": "test123"}
            send_encrypted_msg(client, key, msg)
            result = recv_encrypted_msg(conn, key)
            assert result == msg
        finally:
            client.close()
            conn.close()

    def test_multiple_messages(self):
        key = derive_session_key("TEST-CODE")
        client, conn = self._socket_pair()
        try:
            for i in range(10):
                send_encrypted_msg(client, key, {"seq": i})
            for i in range(10):
                result = recv_encrypted_msg(conn, key)
                assert result == {"seq": i}
        finally:
            client.close()
            conn.close()

    def test_wrong_key_returns_none(self):
        key1 = derive_session_key("CODE-AAAA")
        key2 = derive_session_key("CODE-BBBB")
        client, conn = self._socket_pair()
        try:
            send_encrypted_msg(client, key1, {"secret": True})
            result = recv_encrypted_msg(conn, key2)
            assert result is None
        finally:
            client.close()
            conn.close()

    def test_eof_returns_none(self):
        key = derive_session_key("TEST-CODE")
        client, conn = self._socket_pair()
        client.close()
        result = recv_encrypted_msg(conn, key)
        assert result is None
        conn.close()


class TestUDPAudioFrames:
    def test_round_trip(self):
        key = derive_session_key("TEST-CODE")
        node_id = b"0123456789abcdef"
        pcm = b"\x00\x01" * 320  # 20ms of 16-bit mono at 16kHz
        datagram = encrypt_audio_frame(key, node_id, seq=42, timestamp=100.5, pcm_bytes=pcm)
        result = decrypt_audio_frame(key, datagram)
        assert result is not None
        r_node_id, r_pcm, r_seq, r_ts = result
        assert r_node_id == node_id
        assert r_pcm == pcm
        assert r_seq == 42
        assert abs(r_ts - 100.5) < 1e-9

    def test_wrong_key_returns_none(self):
        key1 = derive_session_key("CODE-AAAA")
        key2 = derive_session_key("CODE-BBBB")
        datagram = encrypt_audio_frame(key1, b"node" * 4, 0, 0.0, b"\x00" * 640)
        assert decrypt_audio_frame(key2, datagram) is None

    def test_truncated_datagram_returns_none(self):
        key = derive_session_key("TEST-CODE")
        assert decrypt_audio_frame(key, b"short") is None

    def test_bad_magic_returns_none(self):
        key = derive_session_key("TEST-CODE")
        datagram = encrypt_audio_frame(key, b"node" * 4, 0, 0.0, b"\x00" * 640)
        bad = b"NOPE" + datagram[4:]
        assert decrypt_audio_frame(key, bad) is None

    def test_short_node_id_padded(self):
        key = derive_session_key("TEST-CODE")
        node_id = b"short"
        datagram = encrypt_audio_frame(key, node_id, seq=1, timestamp=1.0, pcm_bytes=b"\x00" * 10)
        result = decrypt_audio_frame(key, datagram)
        assert result is not None
        assert result[0] == node_id  # trailing zeros stripped
