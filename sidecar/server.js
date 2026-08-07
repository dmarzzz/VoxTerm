// VoxTerm Fileverse sidecar — HTTP shim that wraps @fileverse/agents so the
// existing Python upload client can POST transcripts and get on-chain
// provenance via a Gnosis Safe registry without speaking JS.
//
// Endpoints (mirror the FastAPI collector contract — client doesn't change):
//   POST /v1/transcripts            multipart: metadata (JSON) + transcript (md)
//   GET  /v1/transcripts            ?limit=&offset=  (from local SQLite mirror)
//   GET  /v1/transcripts/:sid       single record from the mirror
//   GET  /v1/transcripts/:sid/raw   fetches latest content via the SDK
//   GET  /healthz                   liveness, no auth
//
// Auth: every endpoint except /healthz requires
//   Authorization: Bearer $VOXTERM_UPLOAD_TOKEN
// Single shared token across all client devices in v1; per-device tokens
// are a follow-up.

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { TranscriptStore } from './store.js';
import { createOrUpdate, getAgent } from './agents-client.js';

const PORT = Number(process.env.PORT || 8000);
const HOST = process.env.HOST || '127.0.0.1';
const DB_PATH = process.env.DB_PATH || './sidecar.db';

// session_id matches the same restrictive grammar as the FastAPI collector
// (no '.', '..', slashes, control chars, spaces) so callers can't smuggle
// path-like values into the on-chain registry.
const VALID_SESSION_ID = /^[A-Za-z0-9_\-:]{1,128}$/;

export function buildApp({ store, agentFactory } = {}) {
  const app = express();
  store ??= new TranscriptStore(DB_PATH);

  const upload = multer({ storage: multer.memoryStorage() });

  // ── auth ────────────────────────────────────────────────────────
  // Token is read per-request, not at import time, so tests can set the
  // env var after importing the module and the production startup check
  // (in the entry point below) can rotate the token via redeploy.
  app.use((req, res, next) => {
    if (req.path === '/healthz') return next();
    const expected = process.env.VOXTERM_UPLOAD_TOKEN || '';
    if (!expected) {
      return res.status(500).json({
        error:
          'sidecar misconfigured: VOXTERM_UPLOAD_TOKEN not set (refusing to accept unauthenticated traffic)',
      });
    }
    const header = req.get('authorization') || '';
    if (header !== `Bearer ${expected}`) {
      return res.status(401).json({ error: 'unauthorized' });
    }
    next();
  });

  // ── routes ──────────────────────────────────────────────────────
  app.get('/healthz', (_req, res) => res.json({ ok: true }));

  app.post(
    '/v1/transcripts',
    upload.fields([
      { name: 'metadata', maxCount: 1 },
      { name: 'transcript', maxCount: 1 },
      { name: 'audio', maxCount: 1 }, // accepted but ignored — markdown only
    ]),
    async (req, res) => {
      // multer puts text fields under .body when using upload.fields if the
      // client sends metadata as a regular form field; if as a file part,
      // it lands under .files. Handle both.
      let metaRaw = null;
      if (req.files?.metadata?.[0]) {
        metaRaw = req.files.metadata[0].buffer.toString('utf-8');
      } else if (typeof req.body?.metadata === 'string') {
        metaRaw = req.body.metadata;
      }
      if (!metaRaw) {
        return res.status(400).json({ error: 'metadata field is required' });
      }
      let meta;
      try {
        meta = JSON.parse(metaRaw);
      } catch (e) {
        return res.status(400).json({ error: `invalid metadata JSON: ${e.message}` });
      }
      if (!meta || typeof meta !== 'object' || Array.isArray(meta)) {
        return res.status(400).json({ error: 'metadata must be a JSON object' });
      }

      const sid = meta.session_id;
      if (typeof sid !== 'string' || !VALID_SESSION_ID.test(sid)) {
        return res
          .status(400)
          .json({ error: 'metadata.session_id is missing or has illegal characters' });
      }

      const transcriptFile = req.files?.transcript?.[0];
      if (!transcriptFile) {
        return res.status(400).json({ error: 'transcript field is required' });
      }
      const markdown = transcriptFile.buffer.toString('utf-8');

      let entry_count = 0;
      try {
        entry_count = parseInt(meta.entry_count ?? 0, 10);
        if (!Number.isFinite(entry_count)) entry_count = 0;
      } catch {
        return res.status(400).json({ error: 'metadata.entry_count must be an integer' });
      }

      try {
        const result = await createOrUpdate({
          session_id: sid,
          markdown,
          store,
          agentFactory,
        });

        store.upsert({
          session_id: sid,
          file_id: result.file_id,
          gnosis_tx: result.tx_hash,
          gnosis_block: result.block_number,
          hostname: String(meta.hostname || ''),
          model: String(meta.model_name || ''),
          language: String(meta.language || ''),
          entry_count,
          voxterm_version: String(meta.voxterm_version || ''),
        });

        return res.status(202).json({
          ok: true,
          session_id: sid,
          file_id: result.file_id,
          gnosis_tx: result.tx_hash,
          gnosis_block: result.block_number,
        });
      } catch (e) {
        return res.status(502).json({
          error: `fileverse upload failed: ${e.message}`,
        });
      }
    }
  );

  app.get('/v1/transcripts', (req, res) => {
    const limit = Number(req.query.limit ?? 50);
    const offset = Number(req.query.offset ?? 0);
    if (!Number.isInteger(limit) || limit < 1 || limit > 500) {
      return res.status(400).json({ error: 'limit must be an integer 1..500' });
    }
    if (!Number.isInteger(offset) || offset < 0) {
      return res.status(400).json({ error: 'offset must be >= 0' });
    }
    res.json({ items: store.list({ limit, offset }) });
  });

  app.get('/v1/transcripts/:sid', (req, res) => {
    if (!VALID_SESSION_ID.test(req.params.sid)) {
      return res.status(400).json({ error: 'invalid session_id' });
    }
    const row = store.get(req.params.sid);
    if (!row) return res.status(404).json({ error: 'session not found' });
    res.json(row);
  });

  app.get('/v1/transcripts/:sid/raw', async (req, res) => {
    if (!VALID_SESSION_ID.test(req.params.sid)) {
      return res.status(400).json({ error: 'invalid session_id' });
    }
    const row = store.get(req.params.sid);
    if (!row) return res.status(404).json({ error: 'session not found' });
    try {
      const agent = await getAgent({ agentFactory });
      const file = await agent.getFile(row.file_id);
      const content =
        typeof file === 'string'
          ? file
          : file?.content ?? file?.markdown ?? JSON.stringify(file);
      res.type('text/markdown').send(content);
    } catch (e) {
      res.status(502).json({ error: `fileverse fetch failed: ${e.message}` });
    }
  });

  return app;
}

// ── entry point ─────────────────────────────────────────────────────
if (import.meta.url === `file://${process.argv[1]}`) {
  if (!process.env.VOXTERM_UPLOAD_TOKEN) {
    console.error(
      'VoxTerm sidecar refusing to start: VOXTERM_UPLOAD_TOKEN is not set.\n' +
        'See sidecar/.env.example.'
    );
    process.exit(2);
  }
  const app = buildApp();
  app.listen(PORT, HOST, () => {
    console.log(`VoxTerm Fileverse sidecar listening on http://${HOST}:${PORT}`);
  });
}
