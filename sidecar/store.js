// SQLite mirror for the Fileverse Agent. The Agents SDK doesn't expose
// list/search; we shadow each create/update with a row here so /v1/transcripts
// (list + fetch) is fast.
//
// Every column except session_id is purely informational — it can all be
// reconstructed by reading the on-chain Gnosis registry — but we keep it
// here so the API can answer common questions without hitting the chain.

import Database from 'better-sqlite3';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS transcripts (
  session_id      TEXT PRIMARY KEY,
  file_id         TEXT NOT NULL,
  gnosis_tx       TEXT,
  gnosis_block    INTEGER,
  hostname        TEXT,
  model           TEXT,
  language        TEXT,
  entry_count     INTEGER,
  voxterm_version TEXT,
  uploaded_at     TEXT NOT NULL,
  revision        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_transcripts_uploaded_at
  ON transcripts(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcripts_hostname
  ON transcripts(hostname);
`;

export class TranscriptStore {
  constructor(dbPath) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.exec(SCHEMA);
  }

  /**
   * Insert a fresh row, or bump revision + replace metadata if the session
   * already exists (re-upload). Caller passes the new file_id from the SDK.
   */
  upsert({
    session_id, file_id, gnosis_tx, gnosis_block,
    hostname, model, language, entry_count, voxterm_version,
  }) {
    const stmt = this.db.prepare(`
      INSERT INTO transcripts (
        session_id, file_id, gnosis_tx, gnosis_block,
        hostname, model, language, entry_count, voxterm_version,
        uploaded_at, revision
      ) VALUES (
        @session_id, @file_id, @gnosis_tx, @gnosis_block,
        @hostname, @model, @language, @entry_count, @voxterm_version,
        @uploaded_at, 1
      )
      ON CONFLICT(session_id) DO UPDATE SET
        file_id=excluded.file_id,
        gnosis_tx=excluded.gnosis_tx,
        gnosis_block=excluded.gnosis_block,
        hostname=excluded.hostname,
        model=excluded.model,
        language=excluded.language,
        entry_count=excluded.entry_count,
        voxterm_version=excluded.voxterm_version,
        uploaded_at=excluded.uploaded_at,
        revision=revision + 1
    `);
    stmt.run({
      session_id,
      file_id,
      gnosis_tx: gnosis_tx ?? null,
      gnosis_block: gnosis_block ?? null,
      hostname: hostname ?? '',
      model: model ?? '',
      language: language ?? '',
      entry_count: entry_count ?? 0,
      voxterm_version: voxterm_version ?? '',
      uploaded_at: new Date().toISOString(),
    });
  }

  get(session_id) {
    return this.db
      .prepare('SELECT * FROM transcripts WHERE session_id = ?')
      .get(session_id) || null;
  }

  list({ limit = 50, offset = 0 } = {}) {
    const lim = Math.max(1, Math.min(limit, 500));
    const off = Math.max(0, offset);
    return this.db
      .prepare(
        'SELECT * FROM transcripts ORDER BY uploaded_at DESC LIMIT ? OFFSET ?'
      )
      .all(lim, off);
  }

  close() {
    this.db.close();
  }
}
