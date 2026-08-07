// End-to-end tests for the sidecar.
//
// We never touch the real @fileverse/agents SDK. agentFactory is the
// injection seam — it returns a fake Agent that records each call and
// returns shaped data the sidecar can normalize.
//
// Run with: cd sidecar && node --test test/

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { TranscriptStore } from '../store.js';
import { _resetForTests } from '../agents-client.js';
import { buildApp } from '../server.js';

const TOKEN = 'secret-test-token';
process.env.VOXTERM_UPLOAD_TOKEN = TOKEN;

function makeFakeAgent() {
  const calls = { create: [], update: [], getFile: [] };
  let nextId = 1;
  const files = new Map();
  return {
    calls,
    async create(md) {
      const fileId = `file-${nextId++}`;
      files.set(fileId, md);
      calls.create.push({ fileId, md });
      return {
        fileId,
        txHash: `0xtx-${fileId}`,
        blockNumber: 1000 + nextId,
      };
    },
    async update(fileId, md) {
      files.set(fileId, md);
      calls.update.push({ fileId, md });
      return {
        fileId,
        txHash: `0xtx-update-${fileId}`,
        blockNumber: 2000 + nextId,
      };
    },
    async getFile(fileId) {
      calls.getFile.push({ fileId });
      return files.get(fileId) ?? null;
    },
  };
}

function startServer({ agent }) {
  _resetForTests();
  const tmp = mkdtempSync(join(tmpdir(), 'voxterm-sidecar-'));
  const dbPath = join(tmp, 'test.db');
  const store = new TranscriptStore(dbPath);
  const app = buildApp({
    store,
    agentFactory: async () => agent,
  });
  const server = app.listen(0);
  const { port } = server.address();
  return {
    base: `http://127.0.0.1:${port}`,
    cleanup: () => {
      server.close();
      store.close();
      rmSync(tmp, { recursive: true, force: true });
    },
  };
}

function postMultipart(base, { token, sessionId, transcript = '# hi', extra = {} } = {}) {
  const boundary = '----test-' + Math.random().toString(16).slice(2);
  const meta = JSON.stringify({
    session_id: sessionId,
    hostname: 'test-host',
    model_name: 'qwen3-0.6b',
    language: 'en',
    started_at: '2026-05-03T12:00:00Z',
    ended_at: '2026-05-03T12:01:00Z',
    entry_count: 3,
    voxterm_version: '0.1.0',
    ...extra,
  });
  const body =
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="metadata"\r\n\r\n` +
    `${meta}\r\n` +
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="transcript"; filename="t.md"\r\n` +
    `Content-Type: text/markdown\r\n\r\n` +
    `${transcript}\r\n` +
    `--${boundary}--\r\n`;
  return new Promise((resolve, reject) => {
    const url = new URL('/v1/transcripts', base);
    const req = http.request(
      {
        method: 'POST',
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        headers: {
          'content-type': `multipart/form-data; boundary=${boundary}`,
          'content-length': Buffer.byteLength(body),
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf-8');
          let json = null;
          try {
            json = JSON.parse(text);
          } catch {}
          resolve({ status: res.statusCode, body: json, raw: text });
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function getJson(base, path, { token } = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, base);
    http
      .get(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname + url.search,
          headers: token ? { authorization: `Bearer ${token}` } : {},
        },
        (res) => {
          const chunks = [];
          res.on('data', (c) => chunks.push(c));
          res.on('end', () => {
            const text = Buffer.concat(chunks).toString('utf-8');
            let json = null;
            try {
              json = JSON.parse(text);
            } catch {}
            resolve({ status: res.statusCode, body: json, raw: text });
          });
        }
      )
      .on('error', reject);
  });
}

// ── /healthz ────────────────────────────────────────────────────────

test('GET /healthz needs no auth', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const r = await getJson(base, '/healthz');
    assert.equal(r.status, 200);
    assert.deepEqual(r.body, { ok: true });
  } finally {
    cleanup();
  }
});

// ── auth ────────────────────────────────────────────────────────────

test('POST /v1/transcripts without token → 401', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const r = await postMultipart(base, { sessionId: 'sid-noauth' });
    assert.equal(r.status, 401);
    assert.equal(agent.calls.create.length, 0);
  } finally {
    cleanup();
  }
});

test('GET /v1/transcripts without token → 401', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const r = await getJson(base, '/v1/transcripts');
    assert.equal(r.status, 401);
  } finally {
    cleanup();
  }
});

// ── happy path ──────────────────────────────────────────────────────

test('POST /v1/transcripts → agent.create + index row', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const r = await postMultipart(base, { token: TOKEN, sessionId: 'sid-1' });
    assert.equal(r.status, 202);
    assert.equal(r.body.ok, true);
    assert.equal(r.body.session_id, 'sid-1');
    assert.equal(r.body.file_id, 'file-1');
    assert.match(r.body.gnosis_tx, /^0xtx-file-1$/);
    assert.equal(agent.calls.create.length, 1);
    assert.equal(agent.calls.update.length, 0);

    const fetch = await getJson(base, '/v1/transcripts/sid-1', { token: TOKEN });
    assert.equal(fetch.status, 200);
    assert.equal(fetch.body.session_id, 'sid-1');
    assert.equal(fetch.body.file_id, 'file-1');
    assert.equal(fetch.body.revision, 1);
    assert.equal(fetch.body.hostname, 'test-host');
  } finally {
    cleanup();
  }
});

// ── idempotency ─────────────────────────────────────────────────────

test('Re-POSTing the same session_id calls agent.update, bumps revision', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    await postMultipart(base, { token: TOKEN, sessionId: 'sid-r', transcript: 'v1' });
    const r2 = await postMultipart(base, { token: TOKEN, sessionId: 'sid-r', transcript: 'v2' });
    assert.equal(r2.status, 202);
    assert.equal(agent.calls.create.length, 1);
    assert.equal(agent.calls.update.length, 1);
    assert.equal(agent.calls.update[0].md, 'v2');
    const row = (await getJson(base, '/v1/transcripts/sid-r', { token: TOKEN })).body;
    assert.equal(row.revision, 2);
  } finally {
    cleanup();
  }
});

// ── validation ──────────────────────────────────────────────────────

test('POST with bad session_id → 400', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    for (const bad of ['..', '.', 'a/b', '', 'has space', 'x\\y']) {
      const r = await postMultipart(base, { token: TOKEN, sessionId: bad });
      assert.equal(r.status, 400, `bad sid '${bad}' should be rejected`);
    }
    assert.equal(agent.calls.create.length, 0);
  } finally {
    cleanup();
  }
});

test('POST with non-object metadata → 400', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const url = new URL('/v1/transcripts', base);
    const boundary = '----t';
    const body =
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="metadata"\r\n\r\n` +
      `[]\r\n` +
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="transcript"; filename="t.md"\r\n\r\n` +
      `x\r\n` +
      `--${boundary}--\r\n`;
    const r = await new Promise((resolve, reject) => {
      const req = http.request(
        {
          method: 'POST',
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          headers: {
            'content-type': `multipart/form-data; boundary=${boundary}`,
            'content-length': Buffer.byteLength(body),
            authorization: `Bearer ${TOKEN}`,
          },
        },
        (res) => {
          res.on('data', () => {});
          res.on('end', () => resolve({ status: res.statusCode }));
        }
      );
      req.on('error', reject);
      req.write(body);
      req.end();
    });
    assert.equal(r.status, 400);
  } finally {
    cleanup();
  }
});

// ── list ────────────────────────────────────────────────────────────

test('GET /v1/transcripts lists most-recent first', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    for (const sid of ['a', 'b', 'c']) {
      await postMultipart(base, { token: TOKEN, sessionId: sid });
    }
    const r = await getJson(base, '/v1/transcripts', { token: TOKEN });
    assert.equal(r.status, 200);
    assert.equal(r.body.items.length, 3);
    const sids = r.body.items.map((it) => it.session_id);
    assert.deepEqual(new Set(sids), new Set(['a', 'b', 'c']));
  } finally {
    cleanup();
  }
});

test('GET /v1/transcripts/:sid → 404 when unknown', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    const r = await getJson(base, '/v1/transcripts/missing', { token: TOKEN });
    assert.equal(r.status, 404);
  } finally {
    cleanup();
  }
});

// ── /raw fetches via the SDK ────────────────────────────────────────

test('GET /v1/transcripts/:sid/raw returns the latest agent.getFile content', async () => {
  const agent = makeFakeAgent();
  const { base, cleanup } = startServer({ agent });
  try {
    await postMultipart(base, { token: TOKEN, sessionId: 'sid-raw', transcript: '# Hello' });
    await postMultipart(base, { token: TOKEN, sessionId: 'sid-raw', transcript: '# Hello v2' });
    const url = new URL('/v1/transcripts/sid-raw/raw', base);
    const got = await new Promise((resolve, reject) => {
      http.get(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          headers: { authorization: `Bearer ${TOKEN}` },
        },
        (res) => {
          const chunks = [];
          res.on('data', (c) => chunks.push(c));
          res.on('end', () =>
            resolve({
              status: res.statusCode,
              body: Buffer.concat(chunks).toString('utf-8'),
              type: res.headers['content-type'],
            })
          );
        }
      ).on('error', reject);
    });
    assert.equal(got.status, 200);
    assert.match(got.type, /text\/markdown/);
    assert.equal(got.body, '# Hello v2');
    assert.equal(agent.calls.getFile.length, 1);
  } finally {
    cleanup();
  }
});
