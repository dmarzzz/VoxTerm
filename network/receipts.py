"""Restraint receipts — make VoxTerm's "audio discarded, only text out" claim
*verifiable*, not just asserted.

Per transcription batch, VoxTerm can emit a signed, hash-chained **restraint
receipt** that commits:

  - ``input_hash``  : SHA-256 of the audio window that was processed and then
    discarded (proves which audio existed without storing it),
  - ``output_hash`` : SHA-256 of the only thing emitted (the transcript text),
  - ``retained``    : 0/1 — did the app keep the raw audio? (VoxTerm's claim: 0),
  - ``prev``        : SHA-256 of the previous receipt body — an append-only hash
    chain, so a silently dropped batch is a detectable gap.

The receipt body is the exact, canonical wire format defined by open-opticon
(``honest-ear/restraint-receipt/v1``); a receipt verifies with that project's
stdlib-only Go verifier (``he-receipt verify``) and its transparency-log/witness
tooling — with **no code merge** between the two projects. The signing key is a
P-256 key; on macOS it can live in the Secure Enclave (future work), on other
platforms it is a file-based key (back it up — losing it un-signs future
receipts).

HONEST SCOPE: a receipt is a tamper-evident, gap-free, signed binding of
input→output under a device key — *accountability*, verifiably stronger than a
bare promise. It does NOT by itself prove which code ran or that no covert exfil
path exists; that needs firmware measurement (a TEE) and/or reproducible builds +
open source. VoxTerm being open-source is the complement.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

RECEIPT_ORIGIN = "honest-ear/restraint-receipt/v1"
_ZERO32 = bytes(32)


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def receipt_body(
    session: str,
    batch: int,
    input_hash: bytes,
    output_hash: bytes,
    retained: bool,
    prev_digest: bytes,
) -> bytes:
    """The exact canonical bytes a receipt signs (and that become a transparency-
    log leaf). Byte-for-byte identical to open-opticon's ``ReceiptBody`` —
    newline-delimited, no CBOR/COSE library needed.
    """
    for name, h in (("input_hash", input_hash), ("output_hash", output_hash), ("prev_digest", prev_digest)):
        if len(h) != 32:
            raise ValueError(f"{name} must be 32 bytes, got {len(h)}")
    if "\n" in session or "\r" in session:
        raise ValueError("session must not contain newlines (it is a line in the canonical body)")
    if batch < 0 or batch >= 2 ** 64:
        raise ValueError("batch must be a uint64")
    return (
        f"{RECEIPT_ORIGIN}\n{session}\n{batch}\n"
        f"{input_hash.hex()}\n{output_hash.hex()}\n{1 if retained else 0}\n{prev_digest.hex()}\n"
    ).encode("utf-8")


def receipt_digest(body: bytes) -> bytes:
    """SHA-256 of the receipt body — its log leaf and the next receipt's ``prev``."""
    return _sha256(body)


@dataclass
class ReceiptBundle:
    """A signed receipt, on the wire (matches open-opticon's ReceiptBundle JSON)."""

    schema: str
    body: str
    sig: str  # 64-byte r||s, hex
    pub_x: str
    pub_y: str

    def to_json(self) -> str:
        return json.dumps(
            {"schema": self.schema, "body": self.body, "sig": self.sig,
             "pub_x": self.pub_x, "pub_y": self.pub_y},
            separators=(",", ":"),
        )


class ReceiptEmitter:
    """Emits signed, hash-chained restraint receipts for a transcription session.

    Hook it where a transcript batch is finalized (e.g. the hivemind sink): call
    :meth:`emit` with the audio window bytes and the emitted text. The emitter
    hashes the audio (it never stores or forwards the PCM), signs the receipt, and
    advances the chain. Optionally POSTs each receipt to ``sink_url``.
    """

    def __init__(self, key: ec.EllipticCurvePrivateKey, session: str,
                 sink_url: Optional[str] = None, retained: bool = False) -> None:
        if not isinstance(key.curve, ec.SECP256R1):
            raise ValueError("receipt key must be P-256 (SECP256R1)")
        self._key = key
        self._session = session
        self._sink_url = sink_url
        self._retained = retained
        self._prev = _ZERO32  # genesis
        self._batch = 0
        nums = key.public_key().public_numbers()
        self._pub_x = nums.x.to_bytes(32, "big")
        self._pub_y = nums.y.to_bytes(32, "big")

    @classmethod
    def from_key_file(cls, path: str | Path, session: str, **kw) -> "ReceiptEmitter":
        """Load a P-256 private scalar (32-byte hex, as open-opticon's ``he-log
        genkey`` prints) and build an emitter. Generates+saves a key if absent."""
        p = Path(path)
        if p.exists():
            hex_str = p.read_text().strip()
            if len(hex_str) != 64:
                raise ValueError("private key must be 32 bytes (64 hex chars)")
            key = ec.derive_private_key(int(hex_str, 16), ec.SECP256R1())
        else:
            key = ec.generate_private_key(ec.SECP256R1())
            d = key.private_numbers().private_value
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{d:064x}\n")
            try:
                p.chmod(0o600)
            except OSError:
                pass
        return cls(key, session, **kw)

    def pub_xy_hex(self) -> tuple[str, str]:
        return self._pub_x.hex(), self._pub_y.hex()

    def emit(self, audio_pcm: bytes, text: str) -> ReceiptBundle:
        """Hash the (then-discarded) audio window + the emitted text, sign a
        chained receipt, advance the chain, and (if configured) POST it."""
        self._batch += 1
        body = receipt_body(
            self._session, self._batch,
            _sha256(audio_pcm), _sha256(text.encode("utf-8")),
            self._retained, self._prev,
        )
        der = self._key.sign(body, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(der)
        sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        bundle = ReceiptBundle(
            schema=RECEIPT_ORIGIN, body=body.decode("utf-8"),
            sig=sig.hex(), pub_x=self._pub_x.hex(), pub_y=self._pub_y.hex(),
        )
        self._prev = receipt_digest(body)  # advance the chain
        if self._sink_url:
            self._post(bundle)
        return bundle

    def _post(self, bundle: ReceiptBundle) -> None:
        req = urllib.request.Request(
            self._sink_url, data=bundle.to_json().encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5).close()
        except Exception:
            # A receipt sink being down must never break transcription.
            pass
