# P2P Shared-AP Isolation Test

Issue #88 asks for two independent parties running at the same time on one
access point without transcript or audio cross-contamination.

## Automated Coverage

`tests/test_p2p_multi_peer.py::TestMultiPeerMesh::test_two_parties_on_same_host_do_not_cross_contaminate`
runs two parties concurrently on one host:

- party alpha: two peers using `alpha-forest-river`
- party beta: two peers using `beta-horse-galaxy`

Each party broadcasts a unique transcript line. The test then deliberately asks
an alpha peer to connect to beta's listener while keeping alpha's session code.
That connection must fail during the encrypted session-code handshake. The
assertions prove:

- alpha peers receive only alpha transcript text
- beta peers receive only beta transcript text
- beta transcript text never appears in alpha's received finals
- alpha transcript text never appears in beta's received finals
- wrong-party TCP connection attempts do not add a peer

This covers the core protocol failure mode behind shared-AP contamination:
same-network peers with different session codes must not decrypt, join, or
receive each other's traffic.

## Manual Shared-AP Trial

The automated test does not replace a physical Wi-Fi trial because it does not
exercise access-point multicast behavior, real mDNS timing, RF loss, or room
audio bleed. For a field check:

1. Put four devices on the same AP.
2. Start party A on two devices and party B on two devices.
3. Record distinct phrases in each party for at least one minute.
4. Confirm party A transcripts contain only party A phrases.
5. Confirm party B transcripts contain only party B phrases.
6. Watch the footer for unexpected joins or party color/name changes.

Failure modes to document if observed:

- a peer auto-joins the wrong party
- a footer shows an unknown peer in the party
- transcript text from one party appears in the other party
- audio merge includes audio from a different party
- discovery flaps between parties and causes repeated join/leave messages

If any of those happen, open a UX follow-up for explicit party selection,
stronger party namespacing, or a "lock party" affordance.
