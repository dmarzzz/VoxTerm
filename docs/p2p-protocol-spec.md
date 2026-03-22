# P2P LAN Collaborative Transcription Protocol

> Feature Specification for VoxTerm
> Draft: 2026-03-21

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Architecture](#3-architecture)
4. [Discovery Layer](#4-discovery-layer)
5. [Wire Protocol](#5-wire-protocol)
6. [Clock Synchronization](#6-clock-synchronization)
7. [Audio Streaming](#7-audio-streaming)
8. [Transcript Exchange](#8-transcript-exchange)
9. [Transcript Assembly](#9-transcript-assembly)
10. [Session Lifecycle](#10-session-lifecycle)
11. [Security & Privacy](#11-security--privacy)
12. [Testing & Debug](#12-testing--debug)
13. [Risks & Open Questions](#13-risks--open-questions)

---

## 1. Overview

### Problem

VoxTerm runs on a single device with a single microphone. In a room with multiple people, audio quality degrades with distance — speakers far from the mic produce poor transcriptions. Speaker diarization struggles to separate many overlapping voices from one mixed signal.

### Solution

A **peer-to-peer LAN protocol** that connects multiple VoxTerm instances on the same network. Each device contributes its close-mic audio and local transcription to the group. Every node builds its own merged transcript using the best available data from all peers.

### Design Principles

- **Local-first**: All processing happens on-device. No cloud, no central server, no infrastructure.
- **Private**: Raw audio is shared only with peers on the local network, never stored remotely. Each node controls its own participation.
- **Decentralized**: Pure peer-to-peer mesh. No coordinator, no leader election, no consensus. Each node is sovereign over its own output.
- **Simple**: Minimal message types, standard transports (TCP/UDP), zero configuration beyond joining a session.
- **Cross-platform**: macOS and Linux. Discovery via mDNS (Bonjour/Avahi), transport via standard sockets.
- **Eventual consistency not required**: Each node builds its own transcript independently. No reconciliation, no conflict resolution. Different nodes may produce different text. This is acceptable and by design.

---

## 2. Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Discover other VoxTerm instances on the same LAN automatically via mDNS | P0 |
| FR-2 | Join/leave a named session with zero configuration | P0 |
| FR-3 | Stream raw PCM audio to all peers in real-time over UDP | P0 |
| FR-4 | Exchange finalized transcript segments with speaker attribution over TCP | P0 |
| FR-5 | Exchange partial (in-progress) transcript segments for live display | P0 |
| FR-6 | Maintain continuous clock synchronization via heartbeat round-trips | P0 |
| FR-7 | Each node independently assembles a merged transcript from all peer data | P0 |
| FR-8 | Graceful handling of peer join, leave, and crash (no disruption to other peers) | P0 |
| FR-9 | Session code acts as both join credential and encryption key seed | P0 |
| FR-10 | All TCP and UDP traffic encrypted (AES-256-GCM) using session-code-derived key | P0 |
| FR-11 | Browse all VoxTerm instances on the LAN (peer browser, pre-session) | P0 |
| FR-12 | Manual peer entry by IP address as fallback when mDNS is unavailable | P1 |
| FR-13 | Bandwidth adaptation — reduce audio quality or skip audio under constrained networks | P2 |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Discovery latency (time to find peers) | < 2s on LAN |
| NFR-2 | Audio stream latency (mic to peer) | < 50ms on LAN |
| NFR-3 | Transcript segment delivery latency | < 200ms on LAN |
| NFR-4 | Audio bandwidth per peer | ~32 KB/s (16kHz 16-bit mono) |
| NFR-5 | Maximum peers per session | 20 (practical mesh limit) |
| NFR-6 | New dependencies | 1 (`zeroconf` for mDNS) |
| NFR-7 | Graceful degradation if network drops | Continues local-only, reconnects when available |
| NFR-8 | Cross-platform | macOS + Linux minimum |

---

## 3. Architecture

### Network Topology

Full mesh. Every node connects to every other node directly. No relay, no coordinator.

```
        Node A ←——→ Node B
          ↕    ╲  ╱    ↕
          ↕     ╲╱     ↕
          ↕     ╱╲     ↕
          ↕   ╱    ╲   ↕
        Node C ←——→ Node D
```

Mesh is practical up to ~20 peers. Each node maintains N-1 TCP connections and sends/receives N-1 UDP audio streams. At 10 peers, inbound audio is 9 × 32 KB/s = 288 KB/s (~2.3 Mbps) — trivial for WiFi.

### Per-Node Architecture

```
EXISTING VOXTERM PROCESS
├── Main thread (Textual event loop)
│   ├── Audio timer: reads mic + system audio + PEER AUDIO STREAMS
│   ├── Silero VAD
│   ├── UI rendering (now shows multi-peer transcript)
│   └── Transcript assembly (merge local + peer segments)
│
├── Worker thread (transcription)
│   ├── MLX transcription (local mic audio)
│   ├── Optional: transcribe best-of-N peer audio for distant speakers
│   └── Speaker identification
│
NEW: NETWORK MODULE
├── mDNS service (discovery)
│   ├── Advertise _voxterm._tcp.local.
│   └── Browse for peers
│
├── Peer manager
│   ├── Peer table (node_id → connection state, clock offset)
│   ├── Heartbeat loop (clock sync + liveness)
│   └── Connect/disconnect handling
│
├── UDP audio streamer
│   ├── Outbound: local mic PCM → all peers
│   └── Inbound: peer PCM → per-peer audio buffers
│
└── TCP segment exchange
    ├── Outbound: local PARTIAL/FINAL segments → all peers
    └── Inbound: peer segments → transcript assembly
```

### Data Flow

```
 MY MIC                          PEER'S MIC
   │                                │
   ▼                                ▼
 Local VAD                    [on their device]
   │                                │
   ▼                                │
 Local ASR ──→ FINAL ──TCP──→ their transcript
   │                                │
   │            ◄──TCP── FINAL ←── their ASR
   │                                │
   ▼                                ▼
 My transcript              Their transcript
 (assembled                  (assembled
  locally)                    locally)
   │                                │
   ▼                                ▼
 My audio ──UDP──→ their buffers (multi-channel processing)
               ◄──UDP── their audio → my buffers
```

---

## 4. Discovery Layer

### Protocol

mDNS/DNS-SD ([RFC 6762](https://www.rfc-editor.org/rfc/rfc6762), [RFC 6763](https://www.rfc-editor.org/rfc/rfc6763)).

### Service Type

```
_voxterm._tcp.local.
```

### Service Instance Name

```
<display_name>._voxterm._tcp.local.
```

Where `<display_name>` is the user's chosen name (e.g., "halcyon"). Sanitized to DNS-SD label rules (≤63 bytes, UTF-8).

### TXT Record

Published alongside the service registration:

| Key | Value | Description |
|-----|-------|-------------|
| `node_id` | UUID (hex, 32 chars) | Unique node identifier, generated once per install |
| `in_session` | `0` or `1` | Whether this node is currently in a session (no session name leaked) |
| `proto_v` | `1` | Protocol version |
| `tcp_port` | integer | TCP port for segment exchange |
| `udp_port` | integer | UDP port for audio streaming |

Note: The session code is NOT included in the mDNS advertisement. The mDNS layer only reveals that a VoxTerm instance exists and whether it's in a session. The session code is exchanged verbally out-of-band and used during the TCP handshake for authentication + key derivation.

### Implementation

Python `zeroconf` library (pure Python, cross-platform, no system dependencies):

```python
from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo

# Advertise (always — even before joining a session)
info = ServiceInfo(
    "_voxterm._tcp.local.",
    f"{display_name}._voxterm._tcp.local.",
    addresses=[socket.inet_aton(my_ip)],
    port=tcp_port,
    properties={
        "node_id": node_id,
        "in_session": "1" if in_session else "0",
        "proto_v": "1",
        "tcp_port": str(tcp_port),
        "udp_port": str(udp_port),
    },
)
zeroconf.register_service(info)

# Browse all VoxTerm peers on the network (peer browser)
class PeerListener:
    def add_service(self, zc, type, name):
        info = zc.get_service_info(type, name)
        # Add to peer browser UI — visible regardless of session
        update_peer_browser(info)

browser = ServiceBrowser(zeroconf, "_voxterm._tcp.local.", PeerListener())

# When joining a session, connect to peers and authenticate with session code
# The session code is entered by the user, NOT discovered via mDNS
```

### Fallback

If mDNS is unavailable (e.g., AP isolation, corporate firewall), the user can manually enter a peer's IP:port. The UI exposes this as a simple input field.

---

## 5. Wire Protocol

### Transport

| Message Type | Transport | Rationale |
|---|---|---|
| `HELLO` | TCP | Reliable handshake |
| `HEARTBEAT` | TCP | Clock sync requires reliable delivery |
| `HEARTBEAT_ACK` | TCP | Round-trip timing |
| `AUDIO_FRAME` | UDP | Low latency, lossy OK |
| `PARTIAL` | TCP | Must arrive in order |
| `FINAL` | TCP | Must be reliable |
| `BYE` | TCP | Graceful disconnect |

### Message Framing

**TCP**: Length-prefixed, encrypted JSON:

```
[4 bytes: uint32 LE total frame length]
[12 bytes: GCM nonce]
[N bytes: AES-256-GCM ciphertext (JSON payload)]
[16 bytes: GCM authentication tag]
```

Same length-prefix pattern as existing `diarization/ipc.py`, but the payload is encrypted. The 4-byte length covers everything after it (nonce + ciphertext + tag).

**UDP**: Single encrypted datagram per audio frame. Structure defined in AUDIO_FRAME message (§5).

### Message Definitions

#### HELLO

Sent after TCP connection AND encryption handshake (§11) complete. Both peers send simultaneously. This message is already encrypted.

```json
{
    "type": "hello",
    "node_id": "a1b2c3d4...",
    "display_name": "halcyon",
    "proto_v": 1,
    "sample_rate": 16000,
    "channels": 1,
    "encoding": "pcm_s16le"
}
```

No session code or name is included — if decryption succeeded, both sides have the same key, which means they have the same session code. A node MUST disconnect if `proto_v` is unsupported.

#### HEARTBEAT

Sent every 1 second by each peer.

```json
{
    "type": "heartbeat",
    "node_id": "a1b2c3d4...",
    "local_ts": 1713400.123,
    "seq": 1204
}
```

`local_ts` is `time.monotonic()` on the sender. `seq` is a monotonically increasing counter per node.

#### HEARTBEAT_ACK

Sent in response to a HEARTBEAT.

```json
{
    "type": "heartbeat_ack",
    "node_id": "b5e6f7a8...",
    "local_ts": 1713400.187,
    "echo_ts": 1713400.123,
    "echo_node_id": "a1b2c3d4..."
}
```

#### AUDIO_FRAME

Sent over UDP. Binary format (not JSON) for efficiency:

```
Header (20 bytes):
  [4 bytes] magic: 0x564F5854 ("VOXT")
  [16 bytes] node_id (UUID bytes)

Metadata (16 bytes):
  [4 bytes] sequence number (uint32 LE)
  [8 bytes] timestamp (float64 LE, sender's monotonic clock)
  [4 bytes] payload length (uint32 LE)

Payload (variable):
  [N bytes] raw PCM audio (16-bit signed LE, mono, 16kHz)
```

Frame size: 20ms of audio = 640 bytes PCM + 36 bytes header = 676 bytes per datagram. Sent 50 times/second. Well under typical MTU (1500 bytes).

#### PARTIAL

Sent while the local user is speaking. Updates the in-progress transcription. Peers replace the previous partial from this node.

```json
{
    "type": "partial",
    "node_id": "a1b2c3d4...",
    "speaker_name": "halcyon",
    "seq": 48,
    "text": "I think we should use",
    "start_ts": 1340.200
}
```

`seq` ties partials to their eventual FINAL.

#### FINAL

Sent when an utterance is complete (VAD silence detected, ASR finalized).

```json
{
    "type": "final",
    "node_id": "a1b2c3d4...",
    "speaker_name": "halcyon",
    "seq": 48,
    "text": "I think we should use mDNS for discovery",
    "start_ts": 1340.200,
    "end_ts": 1343.800,
    "confidence": 0.94
}
```

#### BYE

Sent before graceful disconnect.

```json
{
    "type": "bye",
    "node_id": "a1b2c3d4...",
    "reason": "user_quit"
}
```

If a peer disappears without sending BYE (crash, network drop), the heartbeat timeout (5 seconds with no heartbeat) triggers removal from the peer table.

---

## 6. Clock Synchronization

### Goal

Align monotonic clocks across peers to within ~10ms. This allows accurate ordering of transcript segments from different nodes.

### Method

NTP-simplified round-trip estimation using HEARTBEAT/HEARTBEAT_ACK:

```
Node A sends HEARTBEAT at local time T1
Node B receives it at local time T2, responds with HEARTBEAT_ACK
Node A receives ACK at local time T3

Round-trip time:  RTT = T3 - T1
One-way latency:  OWL = RTT / 2
Clock offset:     offset_B = T2 - (T1 + OWL)
                           = T2 - T1 - RTT/2
```

To convert a timestamp from Node B to Node A's clock:

```
adjusted_ts = remote_ts - offset_B
```

### Refinement

- Maintain a sliding window of the last 20 offset samples per peer
- Use the median (robust to outlier spikes)
- Heartbeats every 1s means the window covers ~20 seconds
- On LAN, RTT is typically < 1ms, so offset accuracy is ~0.5ms
- Drift is corrected continuously — no single calibration point

### Audio Stream Alignment

For multi-channel audio processing, ~10ms accuracy from heartbeat sync may not be sufficient. Use **cross-correlation of the audio signals** to refine alignment:

1. Take a short window (~100ms) from two peers' audio streams
2. Compute the cross-correlation to find the sample offset that maximizes similarity
3. This gives sub-millisecond alignment from the audio content itself

Cross-correlation is only needed if/when multi-channel processing is implemented. For transcript ordering, heartbeat sync is more than adequate.

---

## 7. Audio Streaming

### Format

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Bit depth | 16-bit signed integer |
| Channels | 1 (mono) |
| Encoding | PCM, little-endian |
| Frame duration | 20ms |
| Frame size | 640 bytes (320 samples × 2 bytes) |
| Frames per second | 50 |
| Bandwidth per peer | 32 KB/s (~256 kbps) |

### Transport

UDP unicast to each peer. Each audio frame is a single datagram. No retransmission — lost frames are treated as silence.

### Sending

The existing audio capture loop (15fps timer in `app.py`) feeds the local mic buffer. The network module reads from the same buffer and packetizes into 20ms frames for UDP transmission.

### Receiving

Each peer has a dedicated receive buffer (ring buffer, ~2 seconds). Incoming UDP frames are written by sequence number. Gaps (lost frames) are filled with silence. The buffer is readable by the transcription pipeline as an additional audio source.

### Flow Control

If a node is overwhelmed (CPU-bound on transcription), it can stop reading peer audio buffers without affecting the protocol. Audio frames are fire-and-forget — no backpressure.

---

## 8. Transcript Exchange

### Segment Lifecycle

```
User speaks → VAD onset → ASR begins
                            │
                            ├→ PARTIAL (every ~300ms while speaking)
                            ├→ PARTIAL (updated text)
                            ├→ PARTIAL (updated text)
                            │
                          VAD offset → ASR finalizes
                            │
                            └→ FINAL (immutable)
```

### Partial Handling

- Partials are ephemeral — each new partial from a node replaces the previous one
- Identified by `(node_id, seq)` — same seq means same utterance
- Displayed in the UI with a visual indicator (e.g., dimmed, italicized)
- Discarded when the corresponding FINAL arrives

### Final Handling

- Finals are permanent entries in the merged transcript
- Inserted into the transcript at the position determined by `start_ts` (adjusted for clock offset)
- A FINAL replaces any existing PARTIAL with the same `(node_id, seq)`

### Ordering

Segments from all peers are ordered by clock-adjusted `start_ts`. When two segments from different nodes have start times within 50ms of each other, they are considered simultaneous — either ordering is acceptable.

---

## 9. Transcript Assembly

Each node independently assembles its transcript. There is no shared state and no consensus.

### Algorithm

```
merged_transcript = []

on FINAL received (local or remote):
    adjust start_ts and end_ts using peer's clock offset
    insert into merged_transcript ordered by adjusted start_ts
    update UI

on PARTIAL received (remote):
    store as pending_partial[node_id] (overwrite previous)
    display after last FINAL in transcript, visually distinguished

on PARTIAL generated (local):
    display at bottom of transcript
    broadcast to all peers
```

### Multi-Channel Audio Processing (Future)

When implemented, each node can optionally use peer audio streams to enhance transcription of distant speakers:

1. For each detected utterance, compare energy levels across all available audio channels (local mic + peer streams)
2. Select the channel with highest SNR for that speaker
3. Optionally combine channels for noise reduction
4. Run ASR on the enhanced signal

This is an optimization layer on top of the basic protocol. The protocol works without it — peer transcripts alone produce a good merged result.

---

## 10. Session Lifecycle

### Creating a Session

```bash
# CLI
python3 app.py --session-create --name "halcyon"

# Or press N in the TUI
```

1. Node generates a random 8-character session code: `VOXJ-7K3M`
2. Code is displayed prominently in the TUI
3. User reads the code aloud to others in the room
4. Node updates its mDNS advertisement to `in_session=1`
5. Node begins listening for incoming TCP connections

### Joining

```bash
# CLI
python3 app.py --session-join VOXJ-7K3M --name "bob"

# Or press J in the TUI, then type the code
```

1. User enters the session code (heard verbally from the creator)
2. Node derives the session key from the code via HKDF
3. Node scans mDNS for peers with `in_session=1`
4. For each peer found, attempts TCP connection → encryption handshake (§11)
5. If handshake succeeds (same session key = same code), the HELLO exchange proceeds
6. If handshake fails (wrong key), disconnect silently — this peer is in a different session
7. Heartbeat exchange begins (clock sync)
8. UDP audio streaming begins (encrypted)
9. Peer appears in the UI with their display name

### Late Join

A peer joining mid-session does NOT receive historical transcript data. They start receiving from the point of connection. Rationale: keeps the protocol simple, avoids state synchronization, and the new peer can ask someone to export the transcript if needed.

### Leaving

- **Graceful**: Node sends BYE, closes connections, deregisters mDNS service
- **Crash**: Heartbeat timeout (5s) triggers removal from peer table. Peer's partial (if any) is discarded. Their finalized segments remain in the transcript.

### Reconnection

If a known peer (same `node_id`) reconnects, it is treated as a new join. No state recovery. The transcript retains their previous segments.

---

## 11. Security & Privacy

### Threat Model

The LAN is semi-trusted. Peers are people in the same room who have agreed to share a session. Threats:

- **Unauthorized join**: Someone on the same WiFi snooping or joining uninvited
- **Eavesdropping**: Raw audio and transcript text are sensitive — must not be readable on the wire
- **Replay/injection**: An attacker replaying captured packets or injecting fake segments

### Session Code (Join + Encryption)

The session code serves dual purpose — it's how you join AND how encryption is bootstrapped. No separate "encryption setup" step.

**UX flow:**

```
Creator:
  1. Press N (new session) in TUI
  2. Enters their display name
  3. Screen shows: "Session code: VOXJ-7K3M"  (8 alphanumeric, displayed large)
  4. Tells others the code verbally

Joiner:
  1. Press J (join session) in TUI
  2. Enters their display name
  3. Types the session code: VOXJ-7K3M
  4. Connected. Encrypted. Done.
```

The code is spoken aloud in the room — this is the out-of-band authentication. If you can hear the code, you're authorized to join.

**Key derivation:**

```
session_code = "VOXJ-7K3M"
salt = sha256(b"voxterm-p2p-v1")  # fixed, protocol-version-scoped
session_key = HKDF(
    algorithm=SHA256,
    length=32,
    salt=salt,
    info=b"voxterm-session-key",
    ikm=session_code.encode()
)
```

This produces a 256-bit symmetric key from the session code. All peers deriving from the same code get the same key.

**Session code format**: 8 characters, alphanumeric, uppercase, with a hyphen for readability: `XXXX-XXXX`. Generated randomly by the creator. Entropy: 36^8 ≈ 2.8 trillion combinations. Sufficient for a LAN session — brute force is impractical within a session's lifetime.

### Encryption

All traffic is encrypted. This is not optional.

**TCP messages**: After the TCP connection is established, both peers immediately perform a key confirmation handshake before any protocol messages:

```
1. Both sides derive session_key from the session code
2. Initiator sends: AES-256-GCM(key=session_key, nonce=random_12, plaintext="voxterm-hello")
3. Responder decrypts. If it fails → wrong session code → disconnect
4. Responder sends: AES-256-GCM(key=session_key, nonce=random_12, plaintext="voxterm-hello-ack")
5. Initiator decrypts. If it fails → disconnect
6. All subsequent TCP messages: [4-byte length][12-byte nonce][encrypted payload][16-byte GCM tag]
```

**UDP audio frames**: Each datagram is encrypted individually:

```
[4 bytes magic: 0x564F5854]
[16 bytes node_id]
[4 bytes sequence number (plaintext, used as part of nonce construction)]
[12 bytes nonce]
[N bytes AES-256-GCM encrypted payload + 16-byte tag]
```

The sequence number is in plaintext (needed to construct the nonce on the receiving side and for packet ordering), but the audio payload is fully encrypted. The GCM tag ensures integrity — tampered or injected packets fail authentication and are dropped.

**Nonce construction**: For UDP, nonces are constructed as `node_id[:4] + seq_number(8 bytes)` to guarantee uniqueness without coordination. For TCP, a simple counter per connection suffices.

### Peer Browser (Pre-Session)

Before joining or creating a session, the TUI shows a **peer browser** — a live list of all VoxTerm instances visible on the LAN via mDNS. This is pre-session, no encryption needed (only the mDNS advertisement is visible, which contains display name and node ID, not audio or transcript data).

```
┌─── VoxTerm Peers on Network ──────────────────┐
│                                                │
│  ● halcyon        192.168.1.42    (no session) │
│  ● marcus         192.168.1.67    session: ●   │
│  ● sarah          192.168.1.103   session: ●   │
│                                                │
│  [N] New session   [J] Join session            │
│  [I] Join by IP    [R] Refresh                 │
└────────────────────────────────────────────────┘
```

Nodes advertising a session show a colored indicator (they're in a session, but you can't see the session name or join without the code). Nodes without a session are just "present on the network."

This is useful for:
- Confirming your device is visible before starting a session
- Seeing who's on the network (testing, debugging)
- Knowing that peers exist before asking for a session code

### Data Residency

- Raw audio from peers is buffered in memory only, never written to disk
- Peer transcript segments are included in the local transcript, which follows existing export/save behavior
- No peer data persists after the session ends unless the user explicitly saves the transcript
- Session keys are held in memory only, never persisted

---

## 12. Testing & Debug

### Running Multiple Instances Locally

For development, run multiple VoxTerm instances on the same machine using different ports and simulated audio:

```bash
# Terminal 1: Node A (uses real mic)
python3 app.py --session-create --name "alice" --p2p-port 9900

# Terminal 2: Node B (uses audio file as fake mic input)
python3 app.py --session-join VOXJ-7K3M --name "bob" --p2p-port 9901 --fake-mic tests/fixtures/bob_audio.wav

# Terminal 3: Node C
python3 app.py --session-join VOXJ-7K3M --name "carol" --p2p-port 9902 --fake-mic tests/fixtures/carol_audio.wav
```

The `--fake-mic` flag replays a WAV file as if it were live mic input — same sample rate, same chunking, real-time pacing. This lets you simulate a multi-person room from one laptop.

### Debug Panel

Press `D` in the TUI during a P2P session to toggle the network debug overlay. Displays:

```
┌─── P2P Debug ───────────────────────────────────────────────┐
│ Session: VOXJ-7K3M  |  Peers: 3/3 connected  |  Proto: v1  │
│                                                             │
│ Peer          Latency   Clock Δ   Audio Loss   State        │
│ bob           0.4ms     -2.1ms    0.1%         streaming    │
│ carol         0.8ms     +0.3ms    0.0%         streaming    │
│ dave          1.2ms     +5.7ms    2.3%         streaming    │
│                                                             │
│ Audio Out: 32.0 KB/s  |  Audio In: 95.8 KB/s (3 peers)     │
│ TCP Out: 0.4 KB/s     |  TCP In: 1.1 KB/s                  │
│ UDP tx: 150/s         |  UDP rx: 447/s  (dropped: 3)       │
│                                                             │
│ Heartbeats: 204 sent / 201 acked                            │
│ Clock sync samples: 20/20 (median offset: -2.1ms bob,      │
│   +0.3ms carol, +5.7ms dave)                                │
│ Segments rx: 47 finals, 312 partials                        │
│ Encryption: AES-256-GCM ✓  |  Key: HKDF(VOXJ-7K3M)        │
│                                                             │
│ Last events:                                                │
│  14:03:22.447  bob FINAL seq=31 "yeah that riff is sick"    │
│  14:03:21.890  carol PARTIAL seq=19 "can we try it in..."   │
│  14:03:20.112  dave connected (clock offset: +5.7ms)        │
└─────────────────────────────────────────────────────────────┘
```

### Debug Logging

All network activity is logged at DEBUG level to the existing VoxTerm log infrastructure:

| Log Category | Examples |
|---|---|
| `p2p.discovery` | mDNS advertisements, peer found/lost events |
| `p2p.peer` | TCP connect/disconnect, HELLO exchange, heartbeat timeouts |
| `p2p.clock` | Offset samples, median updates, drift corrections |
| `p2p.audio` | UDP send/receive rates, packet loss, buffer underruns |
| `p2p.segments` | PARTIAL/FINAL sent/received, transcript insertion |
| `p2p.crypto` | Key derivation, encryption handshake success/failure, GCM auth failures |

### Unit Testing

The network module is testable without a real network by mocking sockets:

| Test Area | Approach |
|---|---|
| Message serialization | Encode → decode round-trip for all 7 message types |
| Clock sync | Simulate heartbeat sequences with known offsets, verify convergence |
| Transcript assembly | Feed segments from multiple fake peers, verify ordering |
| Encryption | Encrypt → decrypt round-trip, verify wrong-key rejection, verify nonce uniqueness |
| Peer lifecycle | Simulate connect → heartbeat → crash (no BYE) → timeout detection |
| Audio framing | Verify 20ms chunking, sequence numbering, silence fill for gaps |

### Integration Testing

Two-instance tests using loopback (`127.0.0.1`):

1. **Discovery**: Instance A advertises, Instance B discovers, verify handshake completes
2. **Audio round-trip**: A sends audio, B receives, verify PCM content matches (minus any lost UDP packets)
3. **Transcript exchange**: A sends FINAL, B receives and inserts in correct position
4. **Encryption**: A and B with same session code connect successfully; A and C with different codes fail at handshake
5. **Crash recovery**: Kill A's process, verify B detects timeout and cleans up

### Simulated Multi-Peer Harness

For stress testing, a `network/test_harness.py` that spins up N virtual peers in-process (using async sockets on loopback), each replaying a different audio file and generating transcript segments. Useful for:

- Testing 10+ peer mesh behavior without 10 laptops
- Measuring bandwidth scaling
- Profiling transcript assembly under load
- Verifying clock sync convergence across many peers

---

## 13. Risks & Open Questions

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| AP isolation blocks all peer traffic | Protocol unusable on that network | Manual IP fallback (FR-9), clear error message |
| Clock sync insufficient for audio alignment | Multi-channel processing produces artifacts | Cross-correlation refinement (§6), transcript-only mode as fallback |
| UDP audio overwhelms WiFi on large sessions | Audio glitches, packet loss | Cap at 20 peers (NFR-5), bandwidth adaptation (FR-11) |
| mDNS not available on minimal Linux installs | Discovery fails | Manual IP fallback, document Avahi setup |
| NAT/firewall between peers on same WiFi | TCP/UDP connections fail | Rare on LAN; manual IP fallback |

### Open Questions

1. **Unregistered speakers** — Someone in the room not running VoxTerm. Their voice bleeds into multiple mics. Should one node "claim" them (highest energy wins)? Or let each node independently transcribe them (duplicates in the merged transcript)?

2. **Multi-channel processing strategy** — What algorithms for combining N audio streams? Candidates: best-channel selection (simple), delay-and-sum beamforming (moderate), mask-based source separation (complex). Needs experimentation.

3. **Audio codec** — Raw PCM is simple but 32KB/s per peer. Opus at comparable quality is ~6KB/s. Worth the added dependency/complexity for larger sessions?

4. **Session history sync** — Currently late joiners get no history. Should there be an optional mechanism to request transcript history from peers? Adds complexity but improves UX.

5. **Speaker identity across peers** — If "halcyon" on Node A tags a speaker as "Marcus", should that propagate to other peers? Currently no — each node's transcript is independent. But sharing speaker names could be a useful P1 feature.

---

## Appendix

### Bandwidth Budget (10 peers)

| Stream | Per Peer | Total (9 peers) |
|--------|----------|-----------------|
| Audio in | 32 KB/s | 288 KB/s |
| Audio out | 32 KB/s | 32 KB/s (same stream to all) |
| Heartbeat | ~0.1 KB/s | ~0.9 KB/s |
| Transcripts | ~0.5 KB/s (bursty) | ~4.5 KB/s |
| **Total inbound** | | **~293 KB/s (~2.3 Mbps)** |
| **Total outbound** | | **~37 KB/s (~0.3 Mbps)** |

### Dependencies

| Package | Purpose | Platform |
|---------|---------|----------|
| `zeroconf` | mDNS discovery | macOS + Linux (pure Python) |

### File Structure (Proposed)

```
network/
├── __init__.py
├── discovery.py       # mDNS service advertisement and browsing
├── peer.py            # Peer connection state, clock offset tracking
├── session.py         # Session manager — peer lifecycle, join/leave
├── audio_stream.py    # UDP audio frame send/receive
├── segments.py        # TCP transcript segment exchange
├── protocol.py        # Message definitions, serialization, framing
├── clock.py           # Clock sync — offset estimation, adjustment
├── crypto.py          # Session code generation, HKDF key derivation, AES-256-GCM encrypt/decrypt
├── debug.py           # Debug overlay data collection, log formatters, stats counters
└── test_harness.py    # Multi-peer simulation for testing without multiple machines

widgets/
├── peer_browser.py    # Pre-session peer browser screen (N/J/I keybindings)
└── p2p_debug.py       # Network debug overlay (extends D key toggle)
```
