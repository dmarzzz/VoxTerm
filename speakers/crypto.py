"""Transparent AES-256-CBC encryption for speaker embedding BLOBs.

Uses the `cryptography` library for encryption and `keyring` for
cross-platform key storage (macOS Keychain, Linux SecretService, etc.).

Security properties:
- AES-256-CBC with random IV per BLOB
- HMAC-SHA256 for integrity (encrypt-then-MAC)
- Separate encryption and MAC keys derived via HKDF-SHA256
- Key stored/retrieved via keyring (native credential store)
- Encrypted BLOBs prefixed with magic marker (VXE1) for unambiguous detection
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import os
import sys

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────

_KEY_SIZE = 32  # AES-256
_IV_LEN = 16
_HMAC_LEN = 32
_MAGIC = b"VXE1"  # VoXterm Encrypted v1
_MAGIC_LEN = 4
_HEADER_LEN = _MAGIC_LEN + _IV_LEN + _HMAC_LEN

# HKDF labels for key derivation (must match original for backward compat)
_ENC_LABEL = b"voxterm-enc-v1"
_MAC_LABEL = b"voxterm-mac-v1"

# Keyring service/account identifiers
_KR_SERVICE = "voxterm-speaker-encryption"
_KR_ACCOUNT = "voxterm"

# ── Availability check ────────────────────────────────────────

_has_cryptography = False
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as crypto_padding
    _has_cryptography = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if encryption is available."""
    return _has_cryptography


# ── HKDF key derivation (same as original for backward compat) ──

def _hkdf_expand(master: bytes, label: bytes, length: int = 32) -> bytes:
    """HKDF-Expand (RFC 5869) using HMAC-SHA256."""
    return hmac.new(master, label + b"\x01", hashlib.sha256).digest()[:length]


def derive_keys(master_key: bytes) -> tuple[bytes, bytes]:
    """Derive separate encryption and MAC keys from the master key."""
    enc_key = _hkdf_expand(master_key, _ENC_LABEL, _KEY_SIZE)
    mac_key = _hkdf_expand(master_key, _MAC_LABEL, _KEY_SIZE)
    return enc_key, mac_key


# ── Key storage via keyring ───────────────────────────────────

def _legacy_keychain_get() -> bytes | None:
    """Retrieve encryption key from macOS Keychain via legacy ctypes API.

    Used only for one-time migration to keyring on macOS.
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        sec_path = ctypes.util.find_library("Security")
        if not sec_path:
            return None
        sec = ctypes.CDLL(sec_path)

        service = b"voxterm-speaker-encryption"
        account = b"voxterm"
        pw_len = ctypes.c_uint32(0)
        pw_data = ctypes.c_void_p(0)

        status = sec.SecKeychainFindGenericPassword(
            None,
            ctypes.c_uint32(len(service)), service,
            ctypes.c_uint32(len(account)), account,
            ctypes.byref(pw_len),
            ctypes.byref(pw_data),
            None,
        )
        if status != 0:
            return None

        key = ctypes.string_at(pw_data, pw_len.value)
        sec.SecKeychainItemFreeContent(None, pw_data)
        return key if len(key) == _KEY_SIZE else None
    except Exception:
        return None


def _file_key_path() -> str:
    """Path for file-based key fallback (headless environments)."""
    from paths import DATA_DIR
    return str(DATA_DIR / ".encryption_key")


def _get_key_from_keyring() -> bytes | None:
    """Try to retrieve the key from keyring."""
    try:
        import keyring
        stored = keyring.get_password(_KR_SERVICE, _KR_ACCOUNT)
        if stored:
            return bytes.fromhex(stored)
    except Exception:
        pass
    return None


def _set_key_in_keyring(key: bytes) -> bool:
    """Try to store the key in keyring."""
    try:
        import keyring
        keyring.set_password(_KR_SERVICE, _KR_ACCOUNT, key.hex())
        return True
    except Exception:
        return False


def _get_key_from_file() -> bytes | None:
    """Read key from file-based fallback."""
    path = _file_key_path()
    try:
        with open(path, "rb") as f:
            key = f.read()
        return key if len(key) == _KEY_SIZE else None
    except (OSError, FileNotFoundError):
        return None


def _set_key_in_file(key: bytes) -> bool:
    """Store key in file with restricted permissions."""
    path = _file_key_path()
    try:
        from paths import DATA_DIR
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Write atomically
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(key)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def get_or_create_key() -> bytes | None:
    """Get the master encryption key, creating one if it doesn't exist.

    Resolution order:
    1. keyring (cross-platform credential store)
    2. Legacy macOS Keychain (one-time migration)
    3. File-based fallback (headless environments)
    4. Generate new key and store it
    """
    # 1. Try keyring
    key = _get_key_from_keyring()
    if key:
        return key

    # 2. Try legacy macOS Keychain (migration)
    key = _legacy_keychain_get()
    if key:
        if _set_key_in_keyring(key):
            log.info("Migrated encryption key from Keychain to keyring")
        return key

    # 3. Try file-based fallback
    key = _get_key_from_file()
    if key:
        return key

    # 4. Generate new key
    key = os.urandom(_KEY_SIZE)
    if _set_key_in_keyring(key):
        return key
    if _set_key_in_file(key):
        log.info("Stored encryption key in file (no keyring available)")
        return key

    log.warning("Could not store encryption key — encryption disabled")
    return None


# ── AES-256-CBC encryption ────────────────────────────────────

def _aes_encrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    """Encrypt data with AES-256-CBC + PKCS7 padding."""
    padder = crypto_padding.PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def _aes_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    """Decrypt AES-256-CBC + PKCS7 padded data."""
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(data) + decryptor.finalize()
    unpadder = crypto_padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def encrypt_blob(master_key: bytes, plaintext: bytes) -> bytes:
    """Encrypt a BLOB with AES-256-CBC + HMAC-SHA256.

    Returns: MAGIC (4) || IV (16) || HMAC (32) || ciphertext
    """
    if not plaintext:
        return b""

    enc_key, mac_key = derive_keys(master_key)

    iv = os.urandom(_IV_LEN)
    ciphertext = _aes_encrypt(enc_key, iv, plaintext)

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

    return _aes_decrypt(enc_key, iv, ciphertext)


def is_encrypted(data: bytes) -> bool:
    """Check if a BLOB has the VoxTerm encryption magic prefix."""
    if not data or len(data) < _HEADER_LEN + 1:
        return False
    return data[:_MAGIC_LEN] == _MAGIC
