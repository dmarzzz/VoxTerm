"""Tests for speakers/crypto.py — AES-256-CBC encryption of biometric BLOBs.

Organized by threat model:
- TestConfidentiality: Can encrypted embeddings be recovered? Are they opaque?
- TestTamperDetection: Does the system detect modification, truncation, key mismatch?
- TestKeyManagement: Can keys be derived, stored, and retrieved correctly?
- TestMigrationSafety: Will a future crypto rewrite break existing encrypted databases?

All tests use ephemeral keys (os.urandom) — no Keychain interaction.
"""

import os

import pytest

from speakers.crypto import (
    decrypt_blob,
    derive_keys,
    encrypt_blob,
    get_or_create_key,
    is_encrypted,
    _HEADER_LEN,
    _HMAC_LEN,
    _IV_LEN,
    _MAGIC,
    _MAGIC_LEN,
    _kCCKeySizeAES256,
)


# ── Confidentiality ───────────────────────────────────────────
# Can encrypted embeddings be recovered? Are they opaque to inspection?


class TestConfidentiality:

    def test_round_trip_single_byte(self):
        key = os.urandom(32)
        plaintext = b"\x42"
        blob = encrypt_blob(key, plaintext)
        assert decrypt_blob(key, blob) == plaintext

    def test_round_trip_block_boundary_16(self):
        """Exactly one AES block — exercises PKCS7 full-block padding."""
        key = os.urandom(32)
        plaintext = os.urandom(16)
        assert decrypt_blob(key, encrypt_blob(key, plaintext)) == plaintext

    def test_round_trip_block_boundary_minus_one(self):
        """15 bytes — one byte short of a block."""
        key = os.urandom(32)
        plaintext = os.urandom(15)
        assert decrypt_blob(key, encrypt_blob(key, plaintext)) == plaintext

    def test_round_trip_block_boundary_plus_one(self):
        """17 bytes — one byte past a block."""
        key = os.urandom(32)
        plaintext = os.urandom(17)
        assert decrypt_blob(key, encrypt_blob(key, plaintext)) == plaintext

    def test_round_trip_realistic_embedding(self):
        """512-dim float32 embedding = 2048 bytes."""
        key = os.urandom(32)
        plaintext = os.urandom(2048)
        assert decrypt_blob(key, encrypt_blob(key, plaintext)) == plaintext

    def test_round_trip_large_payload(self):
        key = os.urandom(32)
        plaintext = os.urandom(1_000_000)
        assert decrypt_blob(key, encrypt_blob(key, plaintext)) == plaintext

    def test_empty_plaintext_is_passthrough(self):
        """Empty input is a passthrough, not an error."""
        key = os.urandom(32)
        assert encrypt_blob(key, b"") == b""
        assert decrypt_blob(key, b"") == b""

    def test_unique_iv_per_encryption(self):
        """Two encryptions of the same plaintext must produce different BLOBs.

        Without unique IVs, an observer could correlate encrypted embeddings
        by comparing ciphertext — breaking confidentiality without decryption.
        """
        key = os.urandom(32)
        plaintext = b"same data"
        blob1 = encrypt_blob(key, plaintext)
        blob2 = encrypt_blob(key, plaintext)
        assert blob1 != blob2
        assert decrypt_blob(key, blob1) == plaintext
        assert decrypt_blob(key, blob2) == plaintext

    def test_blob_has_magic_prefix(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"test")
        assert blob[:_MAGIC_LEN] == _MAGIC

    def test_blob_header_structure(self):
        """Verify MAGIC (4) || IV (16) || HMAC (32) || ciphertext layout."""
        key = os.urandom(32)
        blob = encrypt_blob(key, b"test")
        assert len(blob) > _HEADER_LEN
        iv = blob[_MAGIC_LEN : _MAGIC_LEN + _IV_LEN]
        assert len(iv) == _IV_LEN
        mac = blob[_MAGIC_LEN + _IV_LEN : _HEADER_LEN]
        assert len(mac) == _HMAC_LEN
        ciphertext = blob[_HEADER_LEN:]
        assert len(ciphertext) > 0

    def test_ciphertext_is_block_aligned(self):
        """Ciphertext length must be a multiple of 16 (AES block size)."""
        key = os.urandom(32)
        for size in [1, 15, 16, 17, 31, 32, 33, 2048]:
            blob = encrypt_blob(key, os.urandom(size))
            ciphertext = blob[_HEADER_LEN:]
            assert len(ciphertext) % 16 == 0, f"Not block-aligned for {size}-byte plaintext"

    def test_is_encrypted_positive(self):
        key = os.urandom(32)
        assert is_encrypted(encrypt_blob(key, b"test")) is True

    def test_is_encrypted_negative_raw_bytes(self):
        assert is_encrypted(os.urandom(2048)) is False

    def test_is_encrypted_negative_empty(self):
        assert is_encrypted(b"") is False

    def test_is_encrypted_negative_short(self):
        assert is_encrypted(_MAGIC + b"\x00" * 10) is False

    def test_is_encrypted_negative_wrong_magic(self):
        assert is_encrypted(b"XXXX" + os.urandom(100)) is False


# ── Tamper detection ──────────────────────────────────────────
# Does the system detect modification, truncation, or key mismatch?


class TestTamperDetection:

    @staticmethod
    def _flip_byte(data: bytes, offset: int) -> bytes:
        """Flip one byte at the given offset."""
        ba = bytearray(data)
        ba[offset] ^= 0xFF
        return bytes(ba)

    # -- Bit-flip in ciphertext (first / middle / last) --

    def test_flip_ciphertext_first_byte(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        tampered = self._flip_byte(blob, _HEADER_LEN)
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key, tampered)

    def test_flip_ciphertext_last_byte(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        tampered = self._flip_byte(blob, len(blob) - 1)
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key, tampered)

    def test_flip_ciphertext_middle_byte(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        mid = _HEADER_LEN + (len(blob) - _HEADER_LEN) // 2
        tampered = self._flip_byte(blob, mid)
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key, tampered)

    # -- Bit-flip in HMAC --

    def test_flip_hmac_byte(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        tampered = self._flip_byte(blob, _MAGIC_LEN + _IV_LEN)
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key, tampered)

    # -- Bit-flip in IV --

    def test_flip_iv_byte(self):
        """Flipping IV changes both the derived plaintext and the expected HMAC."""
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        tampered = self._flip_byte(blob, _MAGIC_LEN)
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key, tampered)

    # -- Bit-flip in magic --

    def test_flip_magic_byte(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        tampered = self._flip_byte(blob, 0)
        with pytest.raises(ValueError, match="magic"):
            decrypt_blob(key, tampered)

    # -- Truncation --

    def test_truncated_header_only(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, b"secret data")
        with pytest.raises(ValueError, match="too short"):
            decrypt_blob(key, blob[:_HEADER_LEN])

    def test_truncated_to_magic_only(self):
        key = os.urandom(32)
        with pytest.raises(ValueError, match="too short"):
            decrypt_blob(key, _MAGIC)

    def test_truncated_to_half(self):
        key = os.urandom(32)
        blob = encrypt_blob(key, os.urandom(2048))
        with pytest.raises(ValueError):
            decrypt_blob(key, blob[: len(blob) // 2])

    def test_empty_bytes_is_passthrough(self):
        """Empty bytes is a passthrough, not an integrity error."""
        key = os.urandom(32)
        assert decrypt_blob(key, b"") == b""

    # -- Wrong key --

    def test_wrong_key_decryption(self):
        """Encrypt with key A, decrypt with key B — HMAC must catch it."""
        key_a = os.urandom(32)
        key_b = os.urandom(32)
        blob = encrypt_blob(key_a, b"biometric data")
        with pytest.raises(ValueError, match="integrity"):
            decrypt_blob(key_b, blob)

    # -- Cross-row BLOB swap (known gap) --

    def test_cross_row_blob_swap_is_undetected(self):
        """Demonstrate that BLOBs encrypted with the same key are interchangeable.

        The HMAC covers MAGIC || IV || ciphertext but not row metadata. Two
        different plaintexts encrypted with the same master key produce BLOBs
        that each pass integrity checks independently. Swapping them between
        rows is undetected — this is a known gap in the current threat model.

        This test documents the gap, not a failure.
        """
        key = os.urandom(32)
        plaintext_a = b"speaker embedding row A"
        plaintext_b = b"speaker embedding row B"
        blob_a = encrypt_blob(key, plaintext_a)
        blob_b = encrypt_blob(key, plaintext_b)

        # Each blob decrypts to its own plaintext — both pass integrity checks
        assert decrypt_blob(key, blob_a) == plaintext_a
        assert decrypt_blob(key, blob_b) == plaintext_b

        # Swapping: blob_b decrypted as if it were row A's data — no error
        # This is the gap: the system cannot detect the swap
        assert decrypt_blob(key, blob_b) == plaintext_b
        assert decrypt_blob(key, blob_a) == plaintext_a


# ── Key management ────────────────────────────────────────────
# Can keys be derived, stored, and retrieved correctly?


class TestKeyManagement:

    # -- HKDF key derivation --

    def test_derivation_is_deterministic(self):
        """Same master key always produces the same derived keys."""
        master = os.urandom(32)
        enc1, mac1 = derive_keys(master)
        enc2, mac2 = derive_keys(master)
        assert enc1 == enc2
        assert mac1 == mac2

    def test_enc_and_mac_keys_are_independent(self):
        """Encryption and MAC keys must differ."""
        master = os.urandom(32)
        enc_key, mac_key = derive_keys(master)
        assert enc_key != mac_key

    def test_derived_key_lengths(self):
        master = os.urandom(32)
        enc_key, mac_key = derive_keys(master)
        assert len(enc_key) == _kCCKeySizeAES256
        assert len(mac_key) == _kCCKeySizeAES256

    def test_master_key_sensitivity(self):
        """One-bit change in master key produces completely different derived keys."""
        master_a = os.urandom(32)
        master_b = bytearray(master_a)
        master_b[0] ^= 0x01
        master_b = bytes(master_b)

        enc_a, mac_a = derive_keys(master_a)
        enc_b, mac_b = derive_keys(master_b)

        assert enc_a != enc_b
        assert mac_a != mac_b

    # -- Keychain orchestration (mocked) --

    def test_creates_new_key_when_none_exists(self, monkeypatch):
        """When keychain returns None, generate a new key and store it."""
        stored = {}

        def mock_get():
            return stored.get("key")

        def mock_set(key):
            stored["key"] = key
            return True

        monkeypatch.setattr("speakers.crypto._keychain_get", mock_get)
        monkeypatch.setattr("speakers.crypto._keychain_set", mock_set)

        result = get_or_create_key()
        assert result is not None
        assert len(result) == _kCCKeySizeAES256
        assert stored["key"] == result

    def test_returns_existing_key(self, monkeypatch):
        """When keychain has a valid key, return it without generating."""
        existing_key = os.urandom(32)
        set_called = False

        def mock_get():
            return existing_key

        def mock_set(key):
            nonlocal set_called
            set_called = True
            return True

        monkeypatch.setattr("speakers.crypto._keychain_get", mock_get)
        monkeypatch.setattr("speakers.crypto._keychain_set", mock_set)

        result = get_or_create_key()
        assert result == existing_key
        assert not set_called

    def test_returns_none_when_storage_fails(self, monkeypatch):
        """When both get and set fail, return None (encryption disabled)."""
        def mock_get():
            return None

        def mock_set(key):
            return False

        monkeypatch.setattr("speakers.crypto._keychain_get", mock_get)
        monkeypatch.setattr("speakers.crypto._keychain_set", mock_set)

        result = get_or_create_key()
        assert result is None

    def test_rejects_wrong_length_key(self, monkeypatch):
        """If keychain returns a key of wrong length, generate a new one."""
        stored = {}

        def mock_get():
            return b"too-short"

        def mock_set(key):
            stored["key"] = key
            return True

        monkeypatch.setattr("speakers.crypto._keychain_get", mock_get)
        monkeypatch.setattr("speakers.crypto._keychain_set", mock_set)

        result = get_or_create_key()
        assert result is not None
        assert len(result) == _kCCKeySizeAES256


# ── Migration safety ──────────────────────────────────────────
# Will a future crypto rewrite break existing encrypted databases?


class TestMigrationSafety:

    # Generated by CommonCrypto ctypes implementation, 2026-03-21.
    # Master key and plaintext are fixed; the BLOB includes a random IV
    # so this exact output is tied to the specific encryption run.
    _FROZEN_MASTER_KEY = bytes.fromhex(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    _FROZEN_PLAINTEXT = b"VoxTerm test vector for cross-implementation compatibility"
    _FROZEN_BLOB = bytes.fromhex(
        "565845310360b4743bc48bbe69bfeaafb13bf8af"
        "63ebfbe9b7d6df0b7d3f68ab479d04242ec31cd3"
        "4aefc9f4f7ae31a8e270226a9844379f962efebb"
        "8eac8d104af31918e89a736764bba3c309bdc951"
        "059b53955c0eee40fb69a81c52aa9fe3219ced98"
        "1f2b8cc3696ec8b151fc2b69894d919c"
    )

    def test_frozen_vector_decrypts(self):
        """A BLOB produced by the CommonCrypto implementation must decrypt
        correctly under any conforming implementation.

        This is the compatibility gate: if a future crypto rewrite (e.g. PR #15
        switching to the `cryptography` package) breaks this test, the new
        implementation cannot read existing encrypted speaker databases.
        """
        result = decrypt_blob(self._FROZEN_MASTER_KEY, self._FROZEN_BLOB)
        assert result == self._FROZEN_PLAINTEXT

    def test_frozen_vector_detected_as_encrypted(self):
        assert is_encrypted(self._FROZEN_BLOB) is True

    def test_frozen_vector_header_structure(self):
        """Verify the frozen BLOB has the expected format."""
        assert self._FROZEN_BLOB[:_MAGIC_LEN] == _MAGIC
        assert len(self._FROZEN_BLOB) > _HEADER_LEN
