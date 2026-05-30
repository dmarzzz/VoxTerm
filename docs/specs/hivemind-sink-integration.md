---
title: VoxTerm ⇄ Hivemind Sink Integration
description: One client speaking two transcript-sink protocols — legacy (non-TEE) and attested (TEE) — selected by config.
author: shashank
discussions-to: https://github.com/dmarzzz/voxterm-transcript-sink
status: Draft
type: Standards Track
category: Interface
created: 2026-05-30
requires: voxterm-sink/1
---

## Abstract

This proposal specifies how the VoxTerm client publishes session transcripts to
an always-on sink over either of two wire protocols: the existing non-attested
`shape-rotator-hivemind/v1` batch API, and the attested `voxterm-sink/1` TEE API.
The active protocol is chosen by a single configuration value, defaulting to the
attested (`tee`) protocol; the non-attested `legacy` protocol is opt-in.

## Motivation

Two sinks exist with incompatible wire contracts and trust models; the client
speaks only the legacy one. A real VoxTerm batch posted to a TEE sink returns
`404` (`/hivemind/transcripts`) or `400 schema_mismatch` (`/v1/transcript`). A
single client that supports both lets cohorts keep legacy convent boxes while
adopting attested sinks, without a flag day.

## Specification

The key words MUST, MUST NOT, SHOULD, MAY are to be interpreted as in RFC 2119.

### Mode selection

- The client MUST expose `--hivemind-protocol {legacy,tee}` (config key
  `hivemind_protocol`, env `VOXTERM_HIVEMIND_PROTOCOL`).
- It MUST default to `tee` (secure-by-default: the attested protocol unless the
  operator explicitly opts into `legacy`).
- Precedence MUST be CLI > env > config > default. `--hivemind off` MUST disable
  publishing regardless of protocol.
- A client MUST speak exactly one protocol per run.
- In `tee` mode a sink URL (`--hivemind-sink-url`) is REQUIRED; mDNS discovery
  MUST NOT be used (§ Discovery). With no URL the client MUST NOT publish: under
  `--hivemind on` it MUST fail fast with a config error; under `--hivemind auto`
  it MUST enter a no-publish state and log why.

| Client mode | Legacy sink | TEE sink |
|---|---|---|
| `legacy` | ✅ | ❌ |
| `tee` (default) | ❌ | ✅ (URL required) |

### Mode A — Legacy (`shape-rotator-hivemind/v1`)

Current behavior; normative for completeness. The client:

- MUST `POST {base}/hivemind/transcripts`, `application/json`, ≤ 1 MiB/batch.
- MAY discover via mDNS `_sr-hivemind._tcp.local.` or use `--hivemind-sink-url`.
- MUST flush every ~60 s OR 30 segments OR at EOF.
- MUST send `{record_id, batch_index, started_at, ended_at, origin_device,
  location?, segments:[{t, speaker, text}]}` with RFC3339-`Z` timestamps.
- MUST NOT sign or encrypt; `origin_device` is opaque provenance, not identity.

### Mode B — TEE (`voxterm-sink/1`)

The client:

- MUST hold a long-term Ed25519 author keypair; `author` on the wire is the
  32-byte pubkey as 64-hex.
- SHOULD sign every `TranscriptChunk` and finalized `Transcript` with the author
  key (`signature = ed25519(JCS(self \ {signature}))`). This is SHOULD for v1
  because `voxterm-sink/1` treats author signatures as OPTIONAL (sink §7.4,
  §9.1); it becomes MUST when sink write-auth is upgraded (sink roadmap §12.1).
- MUST send `X-Sink-Protocol: voxterm-sink/1` on every request (sink §7.1).
- MUST, before its first push to a sink URL, perform the `voxterm-sink/1` §6
  verification: fetch `/v1/attestation?nonce`, verify the TDX DCAP quote and
  collateral, replay the event log, and confirm channel binding
  (`report_data` ≡ quote `REPORTDATA`). It MUST refuse to push on any failure.
- MUST apply a measurement policy. Both `tofu` (default) and `pinned` MUST
  record/check `sink_sig_pubkey`; `pinned` additionally checks `MRTD/RTMR0..2` +
  `compose_hash` against `measurements.json`. A silent `sink_sig_pubkey` change
  is a trust failure under either policy.
- MUST re-verify at least every 24 h and on any `sink_sig_pubkey` change.
- MUST publish via `POST /v1/transcript/stream` (NDJSON `StreamHeader` +
  monotonic `TranscriptChunk`s, `is_final` to close). This is the required
  baseline for VoxTerm Mode B. The client MAY also support finalized
  `POST /v1/transcript` for export/retry paths, but that is not the primary
  interoperability path.
- After a sink is verified, the client MUST verify the `X-Sink-Signature` header
  on every data-bearing response where the sink provides it, including
  `/v1/info`, transcript reads, `/chunks`, write acknowledgements, and buffered
  stream acknowledgement bodies. It MUST verify against the pinned
  `sink_sig_pubkey`, and MUST reject a response that fails.
  (`/v1/attestation` is exempt: its trust comes from the quote + nonce, not the
  signature — the signature there is over a not-yet-trusted key.) This
  strengthens the sink spec's §7.1 "SHOULD verify" to a MUST for conforming
  VoxTerm clients.
  - **Signed-bytes basis** (the sink spec says `BLAKE3(canonical_response_body)`
    generically; this pins it for the two body types): for `application/json`
    responses the client MUST verify `ed25519` over `BLAKE3(JCS(body))`; for
    `application/x-ndjson` responses (`/chunks`, buffered stream acks) it MUST
    verify over `BLAKE3(raw response bytes)`, since NDJSON has no single-object
    canonical form. This matches the current conforming sink's buffered response
    behavior. A future true streaming ack response MUST provide either per-frame
    signatures or a trailer signature before VoxTerm treats it as verified; until
    then, VoxTerm MUST use the buffered NDJSON ack profile.
- MUST set `session_id` = session start time **in UTC** via
  `strftime("%Y-%m-%d_%H%M%S")`, `sink_id` from the verified sink's
  `/v1/info`, `hivemind_id` (UUID), per-segment `t_start`/`t_end`, and
  `created_at` (RFC3339 UTC).
- On stream reconnect after a partial write or network failure, the client MUST
  resume from an authenticated sink high-water mark: use `ack_seq` values in the
  verified NDJSON acknowledgement body, or query and verify the signed chunk log
  per sink §7.5, then send the next monotonic `seq`. `X-Sink-Seq` is a response
  header and is not covered by `X-Sink-Signature`; the client MAY treat it only
  as an unauthenticated hint. The current PoC sink cannot serve a chunk query for
  an in-progress session before a transcript has been assembled, so
  pre-finalization resume relies on the verified acknowledgement body. The client
  MUST NOT resend a different chunk body for an already acknowledged
  `(session_id, author, seq)`.

#### Author key (client identity)

The author keypair MUST be CSPRNG-generated once (32-byte Ed25519), stored in the
VoxTerm app-support dir (e.g. `author_ed25519.key`, `0600`), and never
transmitted. It is long-term — its pubkey is the wire `author`. v1 does no
automatic rotation; a user-initiated reset yields a new identity (prior records
stay under the old `author`).

#### Attestation verification profile

The verifier MUST use the Phala/Intel DCAP validation path used by
`voxterm-sink/1`: either Phala `dcap-qvl` bindings, the `dcap-qvl` CLI, or a
library with equivalent Intel DCAP quote, PCK chain, TCB info, QE identity, and
CRL validation. The verifier MUST reject:

- invalid quote signatures, invalid or expired collateral, revoked PCK chains,
  and malformed event logs;
- `REPORTDATA` that does not equal the §5.4 recomputation from
  `sink_sig_pubkey`, optional `sink_dh_pubkey`, and the client nonce;
- TCB status worse than `UpToDate` unless the measurement policy explicitly
  allows stale TCB for that sink; stale allowance MUST NOT be the default;
- RTMR3 event-log replay mismatch, and for pinned mode any `MRTD`/`RTMR0..2` or
  `compose_hash` mismatch.

The client MUST fail closed: verification errors produce no write attempt. On
TOFU first contact the client records the verified measurements and surfaces that
the sink is newly trusted; on later contact it compares against the persisted
record.

#### Verified-sink store (TOFU/pinned)

- The **sink identity key** is the attested `sink_sig_pubkey`, NOT the URL
  (`sink_sig` is attestation-bound and stable across restarts/upgrades, sink
  §5.2). The client MUST key its trust store on `sink_sig_pubkey`.
- The client MUST also maintain a URL → expected `sink_sig_pubkey` index. If a
  previously verified URL presents a different `sink_sig_pubkey`, the client
  MUST treat it as a trust failure unless the user explicitly resets trust for
  that URL or accepts an operator-announced migration.
- The client MUST persist, per verified sink: `sink_sig_pubkey`,
  `(MRTD, RTMR0..2, compose_hash, app_id)`, optional informational `RTMR3`,
  `first_seen`, `last_verified`, and the set of URLs it has been reached at.
  Location: the VoxTerm app-support dir alongside the device id (e.g.
  `verified_sinks.json`).
- A **URL change with an unchanged, re-verified `sink_sig_pubkey` + measurements
  is allowed** (the same attested sink may move or be multi-homed); the client
  SHOULD record the new URL.
- A **`sink_sig_pubkey` or measurement change for a known sink MUST be treated as
  a trust failure** (refuse to push, surface it) unless it matches an
  operator-announced upgrade the user accepted.

`verified_sinks.json` MUST have this shape:

```json
{
  "schema_version": 1,
  "url_index": {
    "https://sink.example": "<sink_sig_pubkey>"
  },
  "sinks": {
    "<sink_sig_pubkey>": {
      "sink_sig_pubkey": "<hex ed25519 pubkey>",
      "app_id": "<hex>",
      "mrtd": "<hex>",
      "rtmrs": {
        "0": "<hex>",
        "1": "<hex>",
        "2": "<hex>"
      },
      "rtmr3_replayed": "<hex>",
      "compose_hash": "<hex>",
      "policy": "tofu",
      "allow_stale_tcb": false,
      "first_seen": "2026-05-30T00:00:00Z",
      "last_verified": "2026-05-30T00:00:00Z",
      "urls": ["https://sink.example"]
    }
  }
}
```

For pinned mode, the client MUST consume the sink-published `measurements.json`
shape defined by `voxterm-sink/1` Appendix B. This integration spec does not
define a second schema. A pinned client checks `MRTD/RTMR0..2` against an entry
in `dstack_base_images` and checks the RTMR3-replayed `compose-hash` against the
top-level `compose_hash`, exactly as Appendix B specifies.

TEE-only settings: `--hivemind-sink-measurements {tofu,pinned}`,
`--hivemind-id UUID` (config `hivemind_id`, env `VOXTERM_HIVEMIND_ID`). In `tee`
mode `hivemind_id` is REQUIRED unless a default cohort ID is configured. These
settings MUST be ignored in `legacy` mode.

### Legacy → v1 mapping (bridge)

The bridge is OPTIONAL (and off by default — see Security Considerations). **If
implemented**, the converter (client adapter or sink endpoint) MUST map:

| Legacy | v1 | Rule |
|---|---|---|
| `record_id`, `batch_index`, `ended_at`, `origin_device` | `source` | explicit provenance object (below) |
| `location` | `tags` | if present, add `location:<location>` |
| `started_at` | `session_id`, `created_at` | derive `session_id` via UTC `strftime` |
| `origin_device` (UUID) | `author` | deterministic **pseudo-author** (below) |
| `segments[].t` | `segments[].t_start` | `t_end` = next `t` (or `t+ε` for last) |
| `segments[].speaker` | `segments[].speaker.label` | `local_id` stable per label |
| `segments[].text` | `segments[].text` | verbatim |
| — | `id` | BLAKE3(JCS), computed by the converter |

A UUID is not a valid 64-hex Ed25519 pubkey, so `author` MUST be synthesized
deterministically and treated as a **pseudo-author** (an opaque, **non-signing**
identifier; there is no corresponding private key and bridged records MUST NOT be
considered author-signed):

```text
author = BLAKE3-256("voxterm-legacy-origin-device\0" || origin_device).hex()
```

The converter MUST preserve legacy provenance using this `source` shape:

```json
{
  "tool": "voxterm",
  "legacy_protocol": "shape-rotator-hivemind/v1",
  "record_id": "...",
  "batch_index": 0,
  "ended_at": "...",
  "origin_device": "..."
}
```

### Discovery & auth

- Legacy discovery MAY use mDNS. In `tee` mode `--hivemind-sink-url` is REQUIRED
  (unless a future explicit TEE discovery mechanism exists), TEE sinks are
  verified by attestation, and clients MUST NOT treat network presence as a TEE
  trust signal — legacy mDNS records (`_sr-hivemind._tcp.local.`) MUST NOT be
  used as TEE sink candidates.
- This spec covers **publishing** only. The client needs no auth to push: legacy
  writes are unauthenticated, and `tee` writes are open over the attested channel
  in v1 (the client has already verified the TEE). Reading transcripts back
  (backfill) and its read-tier auth — the `1234` placeholder / `POST /v1/auth`,
  sink §8.3 — are **out of scope** here and deferred to a future sync feature.

## Rationale

`--hivemind-protocol` is orthogonal to the existing `--hivemind auto|on|off`
(enable/discovery) so the two concerns compose. The default is `tee`
(secure-by-default): a client publishes only to an attested sink unless the
operator deliberately downgrades to `legacy`. One protocol per run keeps the
trust model unambiguous.

## Backwards Compatibility

The `tee` default is a **deliberate behavior change**, not byte-compatible with
today's legacy-only client. Consequences:

- Operators who want the current behavior MUST set `--hivemind-protocol legacy`
  (config `hivemind_protocol: legacy`). Legacy remains fully supported, just
  opt-in. No legacy field or route changes.
- Because `tee` requires an explicit sink URL and never uses mDNS, the default
  out-of-the-box behavior becomes **no-publish until an attested sink URL is
  configured** (under `--hivemind auto`), instead of legacy mDNS auto-discovery.
  This is the intended safe default.
- **Migration ordering**: the default flip MUST land together with the Mode B
  client implementation. Until then the client is legacy-only and cannot honor a
  `tee` default; shipping the flip earlier would break publishing. Existing
  legacy tests (`tests/test_hivemind.py`) MUST pin `--hivemind-protocol legacy`.

## Reference Implementation

The current VoxTerm client implements **legacy mode only**
(`network/hivemind.py`); there is no `hivemind_protocol` selector, author key,
or `tee` path in the repo yet. The sink (`voxterm-transcript-sink`,
`voxterm-sink/1`) implements the TEE half. Closing the gap — the selector and
Mode B — is the migration step; see
`voxterm-transcript-sink/DEVELOPMENT.md`.

## Security Considerations

- The legacy↔v1 bridge is **non-attested**: `origin_device` is a UUID, not a
  pubkey, and legacy clients perform no §6 verification and sign nothing. Bridged
  records therefore carry a synthesized, unverifiable `author`. The bridge buys
  storage interop, not TEE guarantees; it MUST default off and MUST NOT be
  presented as conformant attestation.
- In `tee` mode a measurement or `sink_sig_pubkey` change without an announced
  upgrade MUST be treated as compromise. `pinned` is RECOMMENDED for sensitive
  cohorts; `tofu` trusts first contact.

## Copyright

Released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
