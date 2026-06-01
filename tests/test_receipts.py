"""Tests for restraint receipts (network/receipts.py).

The pure-Python tests always run (signing, body format, chain). The
cross-language interop test runs only if the open-opticon `he-receipt` binary is
available (env HE_RECEIPT or on PATH), proving VoxTerm-emitted receipts verify
with that project's stdlib-only Go verifier — without requiring Go in CI.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from network.receipts import (
    RECEIPT_ORIGIN,
    ReceiptEmitter,
    receipt_body,
    receipt_digest,
)


def _new_emitter(session="sess-test", **kw):
    return ReceiptEmitter(ec.generate_private_key(ec.SECP256R1()), session, **kw)


def test_body_is_canonical_seven_lines():
    body = receipt_body("s", 7, b"\x11" * 32, b"\x22" * 32, False, bytes(32))
    lines = body.decode().split("\n")
    assert lines[0] == RECEIPT_ORIGIN
    assert lines[1] == "s" and lines[2] == "7"
    assert lines[3] == "11" * 32 and lines[4] == "22" * 32
    assert lines[5] == "0" and lines[6] == "00" * 32
    assert body.endswith(b"\n") and len(lines) == 8  # trailing newline -> empty last


def test_emit_signs_and_self_verifies():
    em = _new_emitter()
    px, py = em.pub_xy_hex()
    b = em.emit(b"audio window", "transcript text")
    # Reconstruct the signature and verify against the public key.
    sig = bytes.fromhex(b.sig)
    r = int.from_bytes(sig[:32], "big")
    s = int.from_bytes(sig[32:], "big")
    der = utils.encode_dss_signature(r, s)
    pub = ec.EllipticCurvePublicNumbers(int(px, 16), int(py, 16), ec.SECP256R1()).public_key()
    pub.verify(der, b.body.encode(), ec.ECDSA(hashes.SHA256()))  # raises on failure


def test_input_and_output_hashes_commit_the_right_bytes():
    em = _new_emitter()
    b = em.emit(b"the audio", "the text")
    lines = b.body.split("\n")
    assert lines[3] == hashlib.sha256(b"the audio").hexdigest()
    assert lines[4] == hashlib.sha256(b"the text").hexdigest()
    assert lines[5] == "0"  # retained=False by default


def test_chain_links_across_batches():
    em = _new_emitter()
    b1 = em.emit(b"a1", "t1")
    b2 = em.emit(b"a2", "t2")
    # batch 2's prev_digest must equal sha256(batch 1 body).
    assert b2.body.split("\n")[6] == receipt_digest(b1.body.encode()).hex()
    assert b1.body.split("\n")[2] == "1" and b2.body.split("\n")[2] == "2"


def test_retained_flag_surfaces():
    em = _new_emitter(retained=True)
    assert em.emit(b"a", "t").body.split("\n")[5] == "1"


def test_rejects_session_with_newline():
    # A newline in the session would add a line to the canonical body, breaking
    # the 7-line wire format — must be rejected, not silently mis-encoded.
    with pytest.raises(ValueError):
        receipt_body("a\nb", 1, b"\x00" * 32, b"\x00" * 32, False, bytes(32))


def test_rejects_negative_batch():
    with pytest.raises(ValueError):
        receipt_body("s", -1, b"\x00" * 32, b"\x00" * 32, False, bytes(32))


def test_from_key_file_rejects_short_key(tmp_path):
    p = tmp_path / "weak.hex"
    p.write_text("01020304")  # 4 bytes -> a cryptographically weak scalar
    with pytest.raises(ValueError):
        ReceiptEmitter.from_key_file(p, session="s")


def test_from_key_file_generates_and_reloads(tmp_path):
    p = tmp_path / "sub" / "dir" / "key.hex"  # parent dirs do not exist yet
    em1 = ReceiptEmitter.from_key_file(p, session="s")
    em2 = ReceiptEmitter.from_key_file(p, session="s")  # reloads the same key
    assert em1.pub_xy_hex() == em2.pub_xy_hex()


def _he_receipt_bin():
    return os.environ.get("HE_RECEIPT") or shutil.which("he-receipt")


@pytest.mark.skipif(_he_receipt_bin() is None, reason="open-opticon he-receipt not available")
def test_interop_with_open_opticon_he_receipt():
    """A Python-emitted receipt verifies with open-opticon's Go verifier, and a
    suppressed-batch gap is detected — proving the cross-language bridge."""
    he = _he_receipt_bin()
    em = _new_emitter(session="interop")
    px, py = em.pub_xy_hex()
    b1 = em.emit(b"pcm-1", "text-1")
    b2 = em.emit(b"pcm-2", "text-2")
    d1 = receipt_digest(b1.body.encode()).hex()

    def verify(bundle, prev_hex):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(bundle.to_json())
            path = f.name
        try:
            return subprocess.run(
                [he, "verify", "--file", path, "--expect-prev", prev_hex,
                 "--pin-x", px, "--pin-y", py, "--require-not-retained"],
                capture_output=True, text=True,
            ).returncode
        finally:
            os.unlink(path)

    assert verify(b1, "0" * 64) == 0, "genesis receipt must verify"
    assert verify(b2, d1) == 0, "batch 2 must chain onto batch 1"
    assert verify(b2, "0" * 64) == 1, "a suppressed batch must be detected as a gap"
