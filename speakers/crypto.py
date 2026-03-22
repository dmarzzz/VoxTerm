"""Transparent AES-256-CBC encryption for speaker embedding BLOBs.

On macOS: uses CommonCrypto (ctypes) for AES and Keychain for key storage.
On Linux: uses the `cryptography` library for AES and file-based key storage.

Security properties:
- AES-256-CBC with random IV per BLOB
- HMAC-SHA256 for integrity (encrypt-then-MAC)
- Separate encryption and MAC keys derived via HKDF-SHA256
- Key stored in platform keystore (macOS Keychain) or file with 0600 perms (Linux)
- Encrypted BLOBs prefixed with magic marker (VXE1) for unambiguous detection
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hmac
import hashlib
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ── CommonCrypto constants ──────────────────────────────────

_kCCEncrypt = 0
_kCCDecrypt = 1
_kCCAlgorithmAES = 0
_kCCOptionPKCS7Padding = 1
_kCCKeySizeAES256 = 32
_kCCBlockSizeAES128 = 16
_kCCSuccess = 0

# Load macOS frameworks via ctypes (no subprocess, no pip deps)
_libpath = ctypes.util.find_library("System")
_lib = ctypes.CDLL(_libpath) if _libpath else None

_sec_path = ctypes.util.find_library("Security")
_sec = ctypes.CDLL(_sec_path) if _sec_path else None

# Check if Python cryptography library is available (Linux fallback)
_has_cryptography = False
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as _crypto_padding
    _has_cryptography = True
except ImportError:
    pass

# Keychain service/account identifiers
_KC_SERVICE = b"voxterm-speaker-encryption"
_KC_ACCOUNT = b"voxterm"

# Encrypted BLOB format:
#   MAGIC (4) || IV (16) || HMAC-SHA256 (32) || ciphertext
_MAGIC = b"VXE1"  # VoXterm Encrypted v1
_MAGIC_LEN = 4
_IV_LEN = 16
_HMAC_LEN = 32
_HEADER_LEN = _MAGIC_LEN + _IV_LEN + _HMAC_LEN

# HKDF labels for key derivation
_ENC_LABEL = b"voxterm-enc-v1"
_MAC_LABEL = b"voxterm-mac-v1"


def is_available() -> bool:
    """Check if encryption is available on this platform."""
    return _lib is not None or _has_cryptography


# ── HKDF key derivation ────────────────────────────────────

def _hkdf_expand(master: bytes, label: bytes, length: int = 32) -> bytes:
    """HKDF-Expand (RFC 5869) using HMAC-SHA256.

    Derives a subkey from the master key for a specific purpose.
    """
    # Single-round HKDF-Expand (length <= 32 for SHA256)
    return hmac.new(master, label + b"\x01", hashlib.sha256).digest()[:length]


def derive_keys(master_key: bytes) -> tuple[bytes, bytes]:
    """Derive separate encryption and MAC keys from the master key."""
    enc_key = _hkdf_expand(master_key, _ENC_LABEL, _kCCKeySizeAES256)
    mac_key = _hkdf_expand(master_key, _MAC_LABEL, _kCCKeySizeAES256)
    return enc_key, mac_key


# ── Keychain via Security framework (macOS, native) ─────────

# OSStatus codes
_errSecSuccess = 0
_errSecItemNotFound = -25300
_errSecDuplicateItem = -25299


def _keychain_get() -> bytes | None:
    """Retrieve the encryption key from macOS Keychain."""
    if not _sec:
        return None
    try:
        pw_len = ctypes.c_uint32(0)
        pw_data = ctypes.c_void_p(0)

        status = _sec.SecKeychainFindGenericPassword(
            None,
            ctypes.c_uint32(len(_KC_SERVICE)), _KC_SERVICE,
            ctypes.c_uint32(len(_KC_ACCOUNT)), _KC_ACCOUNT,
            ctypes.byref(pw_len),
            ctypes.byref(pw_data),
            None,
        )
        if status != _errSecSuccess:
            return None

        key = ctypes.string_at(pw_data, pw_len.value)
        _sec.SecKeychainItemFreeContent(None, pw_data)
        return key if len(key) == _kCCKeySizeAES256 else None
    except Exception:
        return None


def _keychain_set(key: bytes) -> bool:
    """Store the encryption key in macOS Keychain."""
    if not _sec:
        return False
    try:
        status = _sec.SecKeychainAddGenericPassword(
            None,
            ctypes.c_uint32(len(_KC_SERVICE)), _KC_SERVICE,
            ctypes.c_uint32(len(_KC_ACCOUNT)), _KC_ACCOUNT,
            ctypes.c_uint32(len(key)), key,
            None,
        )
        if status == _errSecDuplicateItem:
            pw_len = ctypes.c_uint32(0)
            pw_data = ctypes.c_void_p(0)
            item_ref = ctypes.c_void_p(0)

            _sec.SecKeychainFindGenericPassword(
                None,
                ctypes.c_uint32(len(_KC_SERVICE)), _KC_SERVICE,
                ctypes.c_uint32(len(_KC_ACCOUNT)), _KC_ACCOUNT,
                ctypes.byref(pw_len), ctypes.byref(pw_data),
                ctypes.byref(item_ref),
            )
            if pw_data.value:
                _sec.SecKeychainItemFreeContent(None, pw_data)
            if item_ref.value:
                status = _sec.SecKeychainItemModifyContent(
                    item_ref, None,
                    ctypes.c_uint32(len(key)), key,
                )
                cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
                cf.CFRelease(item_ref)
            return status == _errSecSuccess

        return status == _errSecSuccess
    except Exception:
        return False


# ── File-based key storage (Linux) ──────────────────────────

def _file_key_path() -> Path:
    """Key file path for file-based storage on Linux."""
    from paths import DB_DIR
    return DB_DIR / ".keyfile"


def _file_key_get() -> bytes | None:
    """Retrieve the encryption key from a chmod-600 file."""
    path = _file_key_path()
    if not path.exists():
        return None
    try:
        key = path.read_bytes()
        return key if len(key) == _kCCKeySizeAES256 else None
    except Exception:
        return None


def _file_key_set(key: bytes) -> bool:
    """Store the encryption key in a chmod-600 file."""
    path = _file_key_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        path.chmod(0o600)
        return True
    except Exception:
        return False


# ── Platform-aware key management ────────────────────────────

def get_or_create_key() -> bytes | None:
    """Get the master encryption key, creating one if it doesn't exist.

    On macOS: uses Keychain. On Linux: uses file-based storage with 0600 perms.
    Returns 32-byte master key, or None if storage is unavailable.
    """
    # Try platform-appropriate key retrieval
    if _sec:
        key = _keychain_get()
    else:
        key = _file_key_get()

    if key and len(key) == _kCCKeySizeAES256:
        return key

    # Generate a new random master key
    key = os.urandom(_kCCKeySizeAES256)

    if _sec:
        if _keychain_set(key):
            return key
    else:
        if _file_key_set(key):
            return key

    log.warning("Could not store encryption key — encryption disabled")
    return None


# ── AES-256-CBC via CommonCrypto (macOS) ─────────────────────

def _cc_crypt(operation: int, key: bytes, iv: bytes, data: bytes) -> bytes:
    """Low-level CommonCrypto CCCrypt wrapper."""
    if not _lib:
        raise RuntimeError("CommonCrypto not available")

    out_size = len(data) + _kCCBlockSizeAES128
    out_buf = ctypes.create_string_buffer(out_size)
    out_moved = ctypes.c_size_t(0)

    status = _lib.CCCrypt(
        ctypes.c_uint32(operation),
        ctypes.c_uint32(_kCCAlgorithmAES),
        ctypes.c_uint32(_kCCOptionPKCS7Padding),
        key, ctypes.c_size_t(len(key)),
        iv,
        data, ctypes.c_size_t(len(data)),
        out_buf, ctypes.c_size_t(out_size),
        ctypes.byref(out_moved),
    )
    if status != _kCCSuccess:
        raise RuntimeError(f"CCCrypt failed with status {status}")

    return out_buf.raw[: out_moved.value]


# ── AES-256-CBC via cryptography library (Linux) ────────────

def _py_crypt(operation: int, key: bytes, iv: bytes, data: bytes) -> bytes:
    """AES-256-CBC via Python cryptography library."""
    if not _has_cryptography:
        raise RuntimeError("cryptography library not available — pip install cryptography")

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    if operation == _kCCEncrypt:
        padder = _crypto_padding.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        encryptor = cipher.encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    else:
        decryptor = cipher.decryptor()
        padded = decryptor.update(data) + decryptor.finalize()
        unpadder = _crypto_padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()


# Select the appropriate crypto backend
_crypt = _cc_crypt if _lib else _py_crypt


def encrypt_blob(master_key: bytes, plaintext: bytes) -> bytes:
    """Encrypt a BLOB with AES-256-CBC + HMAC-SHA256.

    Returns: MAGIC (4) || IV (16) || HMAC (32) || ciphertext
    """
    if not plaintext:
        return b""

    enc_key, mac_key = derive_keys(master_key)

    iv = os.urandom(_IV_LEN)
    ciphertext = _crypt(_kCCEncrypt, enc_key, iv, plaintext)

    # HMAC over magic + IV + ciphertext for integrity (encrypt-then-MAC)
    mac_data = _MAGIC + iv + ciphertext
    mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()

    return _MAGIC + iv + mac + ciphertext


def decrypt_blob(master_key: bytes, data: bytes) -> bytes:
    """Decrypt a BLOB encrypted by encrypt_blob.

    Raises ValueError on tampered/corrupt data.
    """
    if not data:
        return b""

    if len(data) < _HEADER_LEN + 1:
        raise ValueError("Encrypted BLOB too short")

    if data[:_MAGIC_LEN] != _MAGIC:
        raise ValueError("Invalid BLOB magic — not a VoxTerm encrypted BLOB")

    iv = data[_MAGIC_LEN : _MAGIC_LEN + _IV_LEN]
    stored_mac = data[_MAGIC_LEN + _IV_LEN : _MAGIC_LEN + _IV_LEN + _HMAC_LEN]
    ciphertext = data[_HEADER_LEN:]

    enc_key, mac_key = derive_keys(master_key)

    # Verify HMAC first (constant-time comparison)
    mac_data = _MAGIC + iv + ciphertext
    expected_mac = hmac.new(mac_key, mac_data, hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, expected_mac):
        raise ValueError("BLOB integrity check failed — data may be tampered")

    return _crypt(_kCCDecrypt, enc_key, iv, ciphertext)


def is_encrypted(data: bytes) -> bool:
    """Check if a BLOB has the VoxTerm encryption magic prefix."""
    if not data or len(data) < _HEADER_LEN + 1:
        return False
    return data[:_MAGIC_LEN] == _MAGIC
