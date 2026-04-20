# Hivemind Mode — Decentralized Notes Protocol (Scoping)

> Draft: 2026-04-20
> Status: scoping, not a spec. Open questions flagged inline.

This is the sibling protocol to Party Mode. Where Party Mode is **ephemeral, LAN-only, real-time transcript sharing** in one room, Hivemind Mode is **persistent, cross-location, async sharing** of transcripts, notes, and readouts across trusted groups — "multiple hive minds" a user can belong to simultaneously.

Party mode feeds hivemind. Hivemind is what agents point at.

---

## 1. Use Cases

1. **Auto-dump readouts.** Every VoxTerm party-mode session ends → signed entry auto-published to the relevant hivemind(s). Meeting notes become addressable, legible, and agent-consumable without manual curation.
2. **Multiple hivemind membership.** One user is in N groups (e.g., `shape-rotator`, `the-grove`, `greenpoint-compute`). Entries are tagged with which hivemind(s) they belong to. A user filters by group.
3. **Convent mode.** A group treats one hivemind as the canonical dump site — "push everything into the repo or some folder" — with the whole group readable by each member's local agent.
4. **Eval substrate (V2).** The hivemind becomes the data surface for the meta-cooperative game: measuring recall, cross-project contributions, graph density between members.
5. **Community-aligned agent tuning (V2).** The store is what a community agent (Hermes, or a locally fine-tuned Claude/Qwen) can be reinforced against. The eval scores the community's collaboration; the protocol stores the raw substrate.

---

## 2. Principles → Protocol Consequences

| Principle | Protocol consequence |
|---|---|
| **Sovereignty / exit.** Own what you contribute. Be able to leave. | Each author holds a signed append-only log. Leaving = stop publishing. Past entries persist with whoever already received them (Signal model). Exit is honest about this: you stop contributing, you don't retract what was already shared. |
| **Privacy by default.** | Private hivemind is the default: group-key end-to-end encrypted. Public is opt-in. |
| **Radical self-reliance → communal effort.** | Each node is fully functional offline, holds a complete local copy of the hivemind it joined, syncs opportunistically. |
| **Organisms, not organizations. Emergent.** | No central relay, no coordinator. Mesh gossip. Every member is sovereign over their own copy. |
| **Do-it-together.** | Shared logs scoped per group (not user-to-user DMs). The collective artifact is the point. |
| **Community Aligned Intelligence.** | The local store is agent-legible by design — markdown directory on disk, MCP server interface — so agents can be tuned against a community's actual output. |

---

## 3. Architecture

```
 sources (VoxTerm / editor / agent)
         │  write
         ▼
 per-author signed append-only log  ── each entry tagged with hivemind_id(s)
         │  gossip
         ▼
 p2p replicator (one document per hivemind)
   • discovery: mDNS (LAN) + DHT or relays (WAN) + pinned invites
   • gossip of new entries to group members
   • request/reply backfill when a peer joins or reconnects
         │  read
         ▼
 local indexed store
   • markdown directory export (convent mode — files on disk, human-readable)
   • MCP server for Claude Code / Cursor / Hermes
   • search / filter by hivemind, author, tag, time range
```

### Build vs Buy

**Do not roll the substrate.** Candidates evaluated:

| Candidate | Verdict |
|---|---|
| **Iroh / iroh-docs** (Rust, signed KV + blob store + gossip + LAN+WAN discovery + hole-punching) | **Recommended.** One iroh-doc per hivemind maps cleanly. Embeddable from Python via bindings. |
| libp2p | Flexible, heavier, more glue code. Overkill for v1. |
| Willow | Nice data model, early, less production-tested. |
| Nostr | Simple, but relay-dependent and public-biased. Poor fit for private groups. |
| SSB / Earthstar | Battle-tested but quirky data models, smaller ecosystems. |

---

## 4. Entry Model

```
Entry {
  id:            BLAKE3(canonical_bytes)     # content-addressed
  hivemind_id:   UUID                         # the group this belongs to
  author:        ed25519_pubkey               # signing identity (device-scoped in v1)
  created_at:    RFC3339
  parent_ids:    [Entry.id]                   # threads, edits, replies
  content_type:  "transcript" | "note" | "readout" | "summary"
  tags:          [string]                     # user- or agent-applied
  payload:       bytes                        # encrypted if private hivemind
  signature:     ed25519_sig(canonical_bytes)
}
```

**Immutability.** Entries are immutable once published. "Edits" are new entries with `parent_ids = [old.id]` and `tags: ["edit"]`. "Deletions" are tombstone entries — peers are expected to respect them in views, but bytes may persist on disks that already received the original. This is the honest model for an append-only p2p system.

**Multi-hivemind tagging.** An entry may belong to more than one hivemind (e.g., a meeting readout relevant to both `shape-rotator` and `the-grove`). In practice v1 can ship with single-hivemind entries and add multi-tagging in v2 if demand is there.

---

## 5. Membership & Access Control

### Private hivemind (default)

- Shared symmetric `hivemind_key` (AES-256 class).
- All entry payloads encrypted with `hivemind_key` via AES-256-GCM.
- Membership = possession of `hivemind_key` + a registered author pubkey.
- Invite = out-of-band key share: QR code, airdrop, URL with fragment `voxterm://hivemind/<id>#<key>`.
- Removal = rotate `hivemind_key` for future entries. V1 makes no attempt at post-compromise secrecy; old key is assumed compromised once a member leaves.

### Public hivemind

- No payload encryption.
- Entries are still signed by author.
- Write permission via capability tokens issued by the root author (or by a member with delegate rights).

### V1 vs V2

V1 ships with **static group keys**. MLS-style ratcheting for forward secrecy and post-compromise security is V2 territory.

---

## 6. Discovery & Transport

| Scope | V1 | V2 |
|---|---|---|
| Same LAN | mDNS service `_voxterm-hive._tcp.local.`, hivemind_id hash in TXT | same |
| Known peers | Pinned multiaddrs in `~/.config/voxterm/hivemind/<id>.toml` | same |
| WAN discovery | Bootstrap relay list (Grove-operated initially, document self-hosting) | DHT (iroh/libp2p kademlia) |
| NAT traversal | Iroh hole-punching | same |

---

## 7. VoxTerm Integration

1. New keybinding (`H`) opens hivemind settings: list memberships, set default publish target, generate/accept invites.
2. On Party Mode session end: prompt `publish readout to <hivemind>? [y/N]`, or auto-publish when a default is set.
3. Transcripts already land at `~/Documents/voxterm/`; add `~/Documents/voxterm/hivemind/<id>/*.md` — the agent-legible mirror that is the gossip view on disk.
4. MCP server surface at `localhost:<port>` for Claude Code / Hermes / Cursor to query hivemind content directly.

---

## 8. Out of Scope for V1

- MLS ratcheting / post-compromise secrecy
- Voting / moderation / governance primitives
- The meta-cooperative evals themselves (recall, graph density) — these **consume** the store, they don't live in the protocol
- Fine-tuning pipeline and eval infrastructure — separate project
- Mobile client

---

## 9. Open Questions

1. **Device keys vs person keys.** One keypair per device is simpler but loses state on device loss; per-person with device subkeys is more like Signal but more complex. Lean device-per-key for v1, with a person-level manifest layered on later.
2. **Bootstrap relay operators.** Grove-operated → reliable but centralized; community-run → sovereign but flaky. Start Grove-operated, document self-hosting. Acknowledge the centralization tradeoff explicitly.
3. **Retraction UX.** Tombstones don't recall bytes already sent. Be explicit in UX copy so users don't have false expectations of deletion.
4. **Kick/remove member.** V1: manual key rotation by group lead. V2: voting primitive.
5. **Entry size.** Iroh's blob store handles large objects; do not impose an artificial cap. Chunk only if a specific transport hits limits.
6. **Granola vs VoxTerm as the source.** The publish button is where stickiness lives. Make it trivial to send any transcript — VoxTerm, Granola exports, or manual — into a hivemind. Hivemind is a destination; sources are pluggable.
7. **Speaker identity across hivemind members.** If Alice's VoxTerm has `Marcus → speaker_id_42` and Bob's has `Marcus → speaker_id_99`, should these reconcile? V1: no — each author's entries carry their own speaker labels. V2: consider a shared speaker registry per hivemind.

---

## 10. Suggested V1 Build Order

1. Entry format + local signed append-only log (standalone library, testable in isolation).
2. Iroh-docs integration; one doc per hivemind.
3. Group key derivation and payload encryption.
4. mDNS discovery on LAN.
5. CLI: `hivemind create`, `hivemind invite`, `hivemind accept <url>`, `hivemind ls`, `hivemind tail`.
6. VoxTerm hook: auto-publish readout on party-mode end.
7. Markdown directory export.
8. Bootstrap relay for WAN peers.

---

## 11. Relationship to Other Work

- **Party Mode** (`docs/party-mode-design.md`) — real-time transcript sharing on one LAN. Sessions are ephemeral; readouts from sessions are what gets published into hivemind.
- **P2P Protocol Spec** (`docs/p2p-protocol-spec.md`) — wire protocol for Party Mode. Hivemind reuses the identity and crypto primitives where practical (ed25519 signing, AES-256-GCM, HKDF) but delegates transport/gossip to iroh rather than a custom TCP/UDP mesh.
- **Meta-cooperative game evals** — a separate layer that reads from hivemind storage. Out of scope for this protocol.
