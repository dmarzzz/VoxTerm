# VoxTerm Fileverse sidecar

A small Node service that wraps [`@fileverse/agents`](https://github.com/fileverse/agents)
so VoxTerm clients can POST transcripts and get on-chain provenance via a
Gnosis Safe registry without speaking JS or holding wallet keys.

The HTTP shape matches the FastAPI collector in `server/` exactly, so the
Python client doesn't change — just point `--remote-upload-url` at this
sidecar.

> **One central instance.** All transcripts land under a single Safe / Pinata
> account / Pimlico key held by this sidecar. Devices authenticate with a
> shared bearer token. Per-device tokens are a v2.

## What it does for each transcript

1. Receives a multipart POST: `metadata` (JSON) + `transcript` (markdown).
2. Calls `agent.create()` (or `agent.update()` if the `session_id` was seen
   before) → pins the markdown to IPFS via Pinata, writes a registry
   transaction on Gnosis (gas sponsored by Pimlico).
3. Records `(session_id → file_id, gnosis_tx, gnosis_block, metadata)` in a
   small local SQLite mirror so list/fetch endpoints can answer without
   touching the chain.
4. Returns `202 {file_id, gnosis_tx, gnosis_block}`.

Audio uploads are accepted but ignored (the SDK is markdown-only). The
client keeps a local WAV next to the transcript on its own disk.

## Set up

You need three accounts (all have free tiers):

| What | Where | Why |
|---|---|---|
| Pinata JWT | https://app.pinata.cloud/developers/api-keys | IPFS pinning |
| Pimlico API key | https://dashboard.pimlico.io/ | Sponsors the Gnosis txns |
| EVM private key | Generate locally: `node -e "console.log(require('viem/accounts').generatePrivateKey())"` | Owns the Safe |

Then:

```bash
cp .env.example .env
# Fill in PRIVATE_KEY, PIMLICO_API_KEY, PINATA_JWT,
# and pick a strong VOXTERM_UPLOAD_TOKEN: openssl rand -hex 32
npm install
npm start
```

The first request triggers `agent.setupStorage(NAMESPACE)`, which creates
a fresh Gnosis Safe for the namespace and persists creds under
`./creds/<NAMESPACE>.json`. **Back that file up** — losing it means
losing access to the Safe.

## Configure clients

On each VoxTerm device:

```bash
voxterm \
  --remote-upload-url https://your-sidecar.example/v1/transcripts \
  --remote-upload-token <VOXTERM_UPLOAD_TOKEN>
```

Both flags are persisted to the device's `~/.state.json` after first use.

## Run with Docker

```bash
docker build -t voxterm-sidecar sidecar/
docker run --rm \
  --env-file sidecar/.env \
  -v sidecar-data:/data \
  -p 8000:8000 \
  voxterm-sidecar
```

The `/data` volume holds the SQLite mirror and the Agent creds; back it up
with the same care as the Pinata JWT.

## Endpoints

All endpoints except `/healthz` require:

```
Authorization: Bearer $VOXTERM_UPLOAD_TOKEN
```

| Method | Path | Notes |
|---|---|---|
| `GET` | `/healthz` | liveness, no auth |
| `POST` | `/v1/transcripts` | multipart: `metadata` (JSON), `transcript` (md). Idempotent on `session_id` (re-POST → `agent.update`, revision bumps). |
| `GET` | `/v1/transcripts?limit=50&offset=0` | recent uploads from the SQLite mirror |
| `GET` | `/v1/transcripts/{sid}` | one record (metadata + file_id + gnosis_tx + revision) |
| `GET` | `/v1/transcripts/{sid}/raw` | streams the latest markdown content via `agent.getFile` |

## Sharing model

The Safe address is the public handle. Anyone with it can:

- Read the on-chain registry on Gnosis to enumerate every recorded transcript
- Fetch each file from any IPFS gateway by CID (no auth at the IPFS layer)

There is **no encryption**. Markdown is plaintext on public IPFS. Only
upload sessions whose participants consented to a public archive.

## Operational notes

- **TLS**: terminate at Caddy / nginx / Cloudflare in front of this service.
  Don't rely on Node for cert handling.
- **Token rotation**: change `VOXTERM_UPLOAD_TOKEN` in env, redeploy,
  redistribute to clients via their state files.
- **Safe key compromise**: an attacker can write garbage manifests under
  the namespace; they cannot delete prior on-chain entries. Mitigation is
  to spin up a new Safe + namespace if it ever happens.
- **Backups**: the `/data` volume holds the SQLite index *and* the Agent
  creds. Both are recoverable from on-chain + Pinata respectively, but it's
  much easier to back up the volume than to reconstruct.

## Tests

```bash
npm test
```

Tests use a fake Agent (no Pinata/Pimlico/Gnosis calls) — see
`test/server.test.js`. They cover: auth required, happy path,
idempotency on `session_id`, validation (bad sid, non-object metadata),
list, fetch, and the `/raw` proxy through the SDK.

## Why not just use the FastAPI collector in `server/`?

Two reasons we picked this:

1. **On-chain provenance** — every transcript is a tamper-evident,
   timestamped registry entry signed by the Safe. The FastAPI collector
   has no such audit trail.
2. **Sharing UX** — give a reviewer the Safe address; they can browse the
   archive without us provisioning auth or running a frontend.

The FastAPI collector remains a valid path if you need privacy
(encryption) or want to avoid the three-vendor footprint
(Pinata + Pimlico + Gnosis).
