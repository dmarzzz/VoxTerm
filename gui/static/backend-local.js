"use strict";
// VoxTerm on-device (phone) backend. Implements the same window.VOX_BACKEND seam as
// backend-remote.js (getJSON / events / authUrl), but instead of talking HTTP to a Python engine
// it drives the native voxasr Tauri plugin (record -> offline Whisper transcribe-at-stop) and
// persists sessions in localStorage. The on-device transcription makes no network calls.
//
// Loaded INSTEAD of backend-remote.js in the on-device GUI copy, BEFORE app.js, so app.js's
// `const BACKEND = window.VOX_BACKEND` (and the desktop-only feature gating keyed off the
// `on-device` class) pick this up. app.js itself needs no changes.

(function () {
  const invoke = (cmd, args) => window.__TAURI__.core.invoke(cmd, args);
  const LS_PREFIX = "voxterm.session.";   // one localStorage entry per session: the agent-json doc
  const MODEL_LABEL = "Whisper base.en";

  const pad2 = (n) => String(n).padStart(2, "0");
  function newStem() {
    const d = new Date();
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}_` +
           `${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
  }
  function hms(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    return `${pad2(Math.floor(sec / 3600))}:${pad2(Math.floor((sec % 3600) / 60))}:${pad2(sec % 60)}`;
  }
  function listStems() {
    const out = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(LS_PREFIX)) out.push(k.slice(LS_PREFIX.length));
    }
    return out.sort().reverse();   // stems are sortable timestamps → newest first
  }
  function loadDoc(stem) {
    try { return JSON.parse(localStorage.getItem(LS_PREFIX + stem) || "null"); } catch (_) { return null; }
  }

  // Build the agent-json doc app.js renders, from the plugin's segments. Offline Whisper has no
  // diarization, so it's a single speaker; each <=30 s decode window is one turn with its start.
  function buildDoc(stem, segments, durationSec) {
    const turns = (segments || []).filter((s) => s && s.text).map((s) => ({
      speaker: "Speaker 1",
      speaker_id: 0,
      text: s.text,
      t_offset: typeof s.start === "number" ? s.start : undefined,
      t_offset_hms: typeof s.start === "number" ? hms(s.start) : "",
    }));
    return {
      session: { id: stem, duration_hms: hms(durationSec), model: MODEL_LABEL },
      turns,
      speakers: [{ id: 0, label: "Speaker 1" }],
    };
  }

  // ---- export formatters (client-side; the phone has no Python export) ----
  function nameOf(turn, renames) {
    const r = renames && renames[turn.speaker_id];
    return (r && r.trim()) || turn.speaker || ("Speaker " + ((turn.speaker_id || 0) + 1));
  }
  function toMarkdown(doc, renames) {
    const out = ["# " + ((doc.session && doc.session.id) || "Transcript"), ""];
    const meta = [doc.session && doc.session.duration_hms, doc.session && doc.session.model].filter(Boolean).join(" · ");
    if (meta) out.push("_" + meta + "_", "");
    for (const t of doc.turns || []) {
      out.push(`**${nameOf(t, renames)}**${t.t_offset_hms ? ` [${t.t_offset_hms}]` : ""}`, t.text || "", "");
    }
    return out.join("\n").trim() + "\n";
  }
  function toJson(doc, renames) {
    const d = JSON.parse(JSON.stringify(doc));
    for (const sp of d.speakers || []) { const r = renames[sp.id]; if (r && r.trim()) sp.label = r.trim(); }
    for (const t of d.turns || []) { const r = renames[t.speaker_id]; if (r && r.trim()) t.speaker = r.trim(); }
    return JSON.stringify(d, null, 2);
  }
  // Subtitle cues need numeric timings. Sanitize text for the SRT/VTT line format (no embedded
  // newlines, no literal "-->" delimiter) and clamp end>start.
  const cueText = (t) => (t || "").replace(/\s*\n\s*/g, " ").replace(/-->/g, "→").trim();
  function cues(doc) {
    const turns = (doc.turns || []).filter((t) => typeof t.t_offset === "number");
    return turns.map((t, i) => {
      const start = t.t_offset;
      let end = i + 1 < turns.length ? turns[i + 1].t_offset : t.t_offset + 30;
      if (end <= start) end = start + 1;
      return { start, end, text: cueText(t.text) };
    });
  }
  const srtTime = (s) => {
    const ms = Math.floor((s % 1) * 1000); s = Math.floor(s);
    return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)},${String(ms).padStart(3, "0")}`;
  };
  function toSrt(doc) {
    const c = cues(doc);
    return c.length ? c.map((x, i) => `${i + 1}\n${srtTime(x.start)} --> ${srtTime(x.end)}\n${x.text}\n`).join("\n") : null;
  }
  function toVtt(doc) {
    const c = cues(doc);
    return c.length ? "WEBVTT\n\n" + c.map((x) => `${srtTime(x.start).replace(",", ".")} --> ${srtTime(x.end).replace(",", ".")}\n${x.text}\n`).join("\n") : null;
  }
  function exportDoc(doc, kind, renames) {
    renames = renames || {};
    if (kind === "json") return toJson(doc, renames);
    if (kind === "srt") return toSrt(doc);
    if (kind === "vtt") return toVtt(doc);
    return toMarkdown(doc, renames);   // md (default; also feeds copy/summarize-for-AI)
  }

  const AUTOSAVE_MS = 20000;   // persist the rough live transcript at least this often while recording

  class LocalBackend {
    constructor() {
      window.VOX_ONDEVICE = true;
      document.documentElement.classList.add("on-device");   // pre-paint: hides Python-only controls
      this._stem = null;
      this._lastPhase = "idle";
      this._sawActive = false;   // observed recording/transcribing for the current take?
      this._done = false;        // terminal (done/error) edge already emitted for this take?
      this._em = null;
      this._timer = null;
      this._lastSave = 0;        // ms timestamp of the last recording autosave (0 = none yet this take)
    }

    // No audio store on-device: return the path as-is. app.js's range-probe fetch 404s → the player
    // hides and the WAV download disables, which is the intended graceful no-audio behavior.
    authUrl(u) { return u; }

    async getJSON(url, opts) {
      const body = opts && opts.body ? JSON.parse(opts.body) : {};
      const path = url.split("?")[0];
      const qs = new URLSearchParams(url.split("?")[1] || "");
      try {
        switch (path) {
          case "/api/options":
            return { models: [MODEL_LABEL], default_model: MODEL_LABEL, languages: { en: "English" }, input_devices: [] };
          case "/api/record/start":
            this._stem = newStem();
            this._lastPhase = "idle"; this._sawActive = false; this._done = false; this._lastSave = 0;
            this._startPoll();                                 // resume polling for this take
            await invoke("plugin:voxasr|start_transcribe");    // pends on the mic-permission prompt
            return { ok: true };
          case "/api/record/stop":
            await invoke("plugin:voxasr|stop_transcribe");
            return { ok: true };
          case "/api/sessions":
            return { sessions: listStems().map((stem) => ({ stem })) };
          case "/api/session": {
            const doc = loadDoc(qs.get("stem"));
            if (!doc) return { ok: false };
            if (qs.get("kind") === "agent_json") return { ok: true, text: JSON.stringify(doc) };
            if (qs.get("kind") === "transcript") return { ok: true, text: toMarkdown(doc, {}) };
            return { ok: false };
          }
          case "/api/session/delete": {
            const key = LS_PREFIX + body.stem;
            const existed = localStorage.getItem(key) !== null;
            localStorage.removeItem(key);
            return { ok: true, deleted: existed ? [body.stem] : [] };
          }
          case "/api/export": {
            const doc = loadDoc(body.stem);
            if (!doc) return { ok: false };
            const text = exportDoc(doc, body.kind, body.renames);
            return text == null ? { ok: false } : { ok: true, text };   // null = no cues for srt/vtt
          }
          case "/api/summarize":
            return { ok: false, error: "on-device summary unavailable" };
          default:
            return { ok: false, error: "unsupported on-device: " + path };
        }
      } catch (e) {
        return { ok: false, error: String((e && e.message) || e) };
      }
    }

    // EventSource-like: app.js assigns .onmessage/.onerror. We poll the plugin and synthesize the
    // status frames app.js's applyStatus() expects (recording → transcribing → done). The poll runs
    // only while a take is active — it stops at rest and restarts on /api/record/start.
    events() {
      this._em = { onmessage: null, onerror: null, close: () => this._stopPoll() };
      this._startPoll();
      return this._em;
    }
    _startPoll() { if (!this._timer) this._timer = setInterval(() => this._poll(), 500); }
    _stopPoll() { if (this._timer) { clearInterval(this._timer); this._timer = null; } }
    async _poll() {
      let st;
      try { st = await invoke("plugin:voxasr|poll_transcript"); } catch (_) { return; }
      const frame = this._frameFor(st);
      if (frame && this._em && this._em.onmessage) this._em.onmessage({ data: JSON.stringify(frame) });
      // Stop once a take's terminal edge is consumed, or when idle at rest — restarts on next record.
      if (this._done || st.phase === "idle") this._stopPoll();
    }

    // Crash insurance: while recording, persist the rough live transcript (finalized windows + the
    // in-progress partial) to localStorage every AUTOSAVE_MS. The authoritative pass at stop overwrites
    // this under the same stem; if the app dies mid-take, this is what survives — losing at most the
    // current ~30 s window instead of the whole session. Throttled and best-effort (storage-full is
    // swallowed; the final pass would surface a real save error).
    _autosave(st) {
      if (!this._stem) return;
      const now = Date.now();
      if (this._lastSave && now - this._lastSave < AUTOSAVE_MS) return;
      const segs = (st.liveLines || []).filter((s) => s && s.text).slice();
      const lp = st.livePartial;
      if (lp && lp.text) segs.push(lp);     // include the in-progress window so the newest words aren't lost
      if (!segs.length) return;             // nothing decoded yet — don't write an empty placeholder
      const doc = buildDoc(this._stem, segs, st.elapsed || 0);
      try { localStorage.setItem(LS_PREFIX + this._stem, JSON.stringify(doc)); this._lastSave = now; } catch (_) {/* storage full — final pass reports it */}
    }

    _frameFor(st) {
      const phase = st.phase || "idle";
      let frame;
      if (phase === "recording") {
        this._sawActive = true;
        // Live preview: map the plugin's finalized windows + in-progress decode into the {active,
        // lines, partial} shape app.js's live view already renders. Whisper re-decodes the whole
        // window each pass, so the partial is all "volatile" (no separately-stable head) until it
        // finalizes into a line.
        const lines = (st.liveLines || []).filter((s) => s && s.text).map((s) => ({ t: hms(s.start), text: s.text }));
        const lp = st.livePartial;
        const partial = lp && lp.text ? { t: hms(lp.start), stable: "", volatile: lp.text } : null;
        frame = {
          recording: true, elapsed: st.elapsed || 0, level: st.level || 0, job: { state: "idle" },
          live: { active: true, lines: lines, partial: partial },
        };
        this._autosave(st);   // periodically persist the rough live transcript so a crash mid-take isn't total loss
      } else if (phase === "transcribing") {
        this._sawActive = true;
        frame = { recording: false, job: { state: "transcribing", frac: 0, msg: "Transcribing…" } };
      } else if (phase === "error" && this._lastPhase !== "error") {
        this._done = true;
        frame = { recording: false, job: { state: "error", error: st.error || "recording failed" } };
      } else if (phase === "done" && this._lastPhase !== "done" && this._sawActive && this._stem) {
        // Build the transcript once, on the transition into done. Require _sawActive so a stale
        // phase='done' left over from a prior take can't persist the old segments under a new stem.
        this._done = true;
        const doc = buildDoc(this._stem, st.segments, st.durationSec);
        if (doc.turns.length === 0) {
          frame = { recording: false, job: { state: "done", n_turns: 0, n_speakers: 1, stem: this._stem } };  // no-speech: nothing to persist
        } else {
          try {
            localStorage.setItem(LS_PREFIX + this._stem, JSON.stringify(doc));
            frame = { recording: false, job: { state: "done", n_turns: doc.turns.length, n_speakers: 1, stem: this._stem } };
          } catch (_) {
            frame = { recording: false, job: { state: "error", error: "storage full — delete old sessions to free space" } };
          }
        }
      } else {
        frame = { recording: false, job: { state: "idle" } };
      }
      this._lastPhase = phase;
      return frame;
    }
  }

  window.VOX_BACKEND = window.VOX_BACKEND || new LocalBackend();
})();
