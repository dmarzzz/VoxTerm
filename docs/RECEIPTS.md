# Restraint receipts (optional) — make the privacy claim *verifiable*

VoxTerm's pitch is "audio is processed in real time and **discarded**; only text
is saved." Today that's a trust claim. **Restraint receipts** let a session emit a
signed, hash-chained record that makes it *checkable* — without sending audio
anywhere and without coupling VoxTerm to any other project.

This is **opt-in** and off by default. When off, nothing changes.

## What a receipt commits (per transcription batch)

| field | meaning |
|---|---|
| `input_hash` | SHA-256 of the audio window that was processed **and then discarded** — proves which audio existed without storing it |
| `output_hash` | SHA-256 of the only thing emitted (the transcript text) |
| `retained` | `0`/`1` — did the app keep the raw audio? VoxTerm sets `0` (verified: `audio/buffer.py:get_and_clear()` clears the buffer; PCM is never persisted) |
| `prev` | SHA-256 of the previous receipt body — an append-only hash chain, so a silently dropped batch is a detectable **gap** |

Each receipt is signed with a P-256 key and is a self-describing line-oriented
body (`honest-ear/restraint-receipt/v1`). It interoperates with the open-source
[open-opticon](https://github.com/NubsCarson/open-opticon) verifier
(`he-receipt verify`) and its transparency-log / witness-cosigning tooling — a
VoxTerm-emitted receipt verifies there unchanged (see `tests/test_receipts.py`,
which cross-checks against the Go verifier when present).

## Honest scope (read this)

A receipt is a **tamper-evident, gap-free, signed binding of input→output under a
device key** — accountability, verifiably stronger than a bare promise, and
publicly auditable. It does **not** by itself prove *which code* ran or that no
covert exfiltration path exists; that needs firmware measurement (a TEE) or, on a
Mac, the complement VoxTerm already provides: it's open-source and reproducible.
On macOS the device key can later live in the Secure Enclave (**not yet
implemented**); today it is a file-based key — **back it up**, and treat it like
any signing key.

## Wiring (≈10 lines, for maintainers)

`network/receipts.py` (`ReceiptEmitter`) is standalone and tested. To turn it on,
create the emitter at startup behind two flags and call it where a transcript
batch is finalized:

```python
# tui/app.py main(): parse --receipt-key-file and --receipt-sink-url, then:
from network.receipts import ReceiptEmitter
emitter = (ReceiptEmitter.from_key_file(args.receipt_key_file, session=record_id,
                                        sink_url=args.receipt_sink_url)
           if args.receipt_key_file else None)

# where a batch is finalized (e.g. hivemind add_segment / _build_payload_locked,
# or _on_transcription) — emitter hashes the audio window + the emitted text:
if emitter is not None:
    emitter.emit(audio_window_bytes, batch_text)   # signs, chains, optionally POSTs
```

`emitter.emit()` hashes the PCM and the text — it never stores or forwards the
audio. The receipt JSON can be POSTed to a sink (`--receipt-sink-url`) or handled
inline. Verify a stream with `he-receipt verify --expect-prev <prev> ...`.
