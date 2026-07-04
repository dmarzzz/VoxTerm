"use strict";
const $ = (id) => document.getElementById(id);
// Muted, desaturated speaker dots — distinct enough to tell speakers apart at a glance
// without breaking the monochrome chrome (no neon). The dot is the only color in a turn row.
const PALETTE = ["#9ba3ad", "#b58f8f", "#b0a98a", "#8fa6b5", "#a397b3", "#8fb3a0", "#b59c8a", "#a98fa9"];
const PEER_COLOR = "#8fa6b5";   // P2P peer turns — a fixed muted slate (not a PALETTE slot, which rotates per speaker)

let OPTS = { models: [], languages: {}, default_model: "", input_devices: [] };
let CUR = null;            // current doc (agent_json parsed) or null for raw-markdown view
let CUR_STEM = null, CUR_DIR = null;   // loaded session id/dir (set even for raw-markdown)
let RENAMES = {};          // speaker_id -> custom name (view + export)
let lastJobState = "idle";
let _levelPeak = 0;          // peak mic level during the current recording (drives the low-signal warning)
let _recStartedAt = 0;       // ms timestamp recording began
let SESSIONS = [];         // last-fetched session list (filtered by the search box for render)
const EMPTY_DEFAULT = $("empty").innerHTML;   // restored after a no-speech/error overwrite

// ---------- helpers ----------
const BACKEND = window.VOX_BACKEND;   // set by backend-remote.js (loaded first)
async function getJSON(url, opts) {
  try {
    return await BACKEND.getJSON(url, opts);
  } catch {
    toast("Network error — is the server running?");
    return { ok: false, error: "network" };
  }
}
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 2200);
}
function fmtClock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(sec / 60), s = sec % 60, h = Math.floor(m / 60);
  return h ? `${h}:${String(m % 60).padStart(2, "0")}:${String(s).padStart(2, "0")}`
           : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function colorFor(sid) { return PALETTE[((sid || 0) % PALETTE.length + PALETTE.length) % PALETTE.length]; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
// localStorage wrappers — private/incognito mode can throw on access, so swallow it.
const LS_MODEL = "voxterm.model", LS_LANG = "voxterm.language", LS_MIC = "voxterm.mic", LS_DIARIZE = "voxterm.diarize", LS_SOURCE = "voxterm.source", LS_SUMMODEL = "voxterm.summodel";
const LS_DIARIZE_OD = "voxterm.diarize_ondevice";   // on-device opt-in speaker diarization (default off — it's slow)
function lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function lsSet(key, val) { try { localStorage.setItem(key, val); } catch { /* private mode */ } }
function setNav(open) {
  document.body.classList.toggle("nav-open", open);
  $("navToggle").setAttribute("aria-expanded", open ? "true" : "false");
}
function nameFor(turn) {
  if (turn.peer) return turn.peer_name ? `${turn.speaker} · ${turn.peer_name}` : turn.speaker;
  if (RENAMES[turn.speaker_id]) return RENAMES[turn.speaker_id];
  return turn.speaker || "(unattributed)";
}

// which content block is showing: 'empty' | 'live' | 'transcript'
function setView(state) {
  $("empty").classList.toggle("hidden", state !== "empty");
  const t = state === "transcript";
  $("tvHead").classList.toggle("hidden", !t);
  if (!t) {
    const p = $("player"); if (p && !p.paused) p.pause();   // never leave audio playing off-screen
    $("summaryBlock").classList.add("hidden"); $("summaryBody").textContent = "";   // summary is per-session
  }
  // VOX_CONV owns the panel-level switch (turns / liveLines / graph / interruptions) for the current
  // mode within this top-level state, and re-analyzes when a derived mode is showing.
  if (window.VOX_CONV) window.VOX_CONV.setTop(state);
}

// The doc the Graph/Interruptions modes analyze: the loaded transcript, or — while recording — a
// single-speaker doc synthesized from the live tail so the derived views update in real time.
let LIVE_DOC = null;
function hmsToSec(s) {
  const p = String(s || "").split(":").map(Number);
  if (!p.length || p.some(Number.isNaN)) return undefined;
  return p.reduce((acc, n) => acc * 60 + n, 0);
}
function buildLiveDoc(lines) {
  return {
    session: { id: "(live)" },
    speakers: [{ id: 0, label: "Speaker 1" }],
    turns: (lines || []).map((l) => ({ speaker: "Speaker 1", speaker_id: 0, text: l.text, t_offset: hmsToSec(l.t), t_offset_hms: l.t })),
  };
}
function activeDoc() { return (CUR && Array.isArray(CUR.turns)) ? CUR : LIVE_DOC; }
// Click-to-play seam used by graph nodes / interruption events.
window.VOX_SEEK = function (sec) {
  const p = $("player"); if (!p || !p.src || typeof sec !== "number") return;
  const go = () => { try { p.currentTime = sec; } catch { /* not seekable yet */ } p.play().catch(() => {}); };
  if (p.readyState >= 1) go(); else { p.addEventListener("loadedmetadata", go, { once: true }); p.load(); }
};

// Live amplitude strip: push each SSE level reading and draw a scrolling bar history.
const WAVE_MAX = 80;
const _wave = [];
function drawWave(level) {
  const c = $("recWave");
  if (!c || !c.getContext) return;
  _wave.push(Math.min(1, Math.max(0, (level || 0) / 0.25)));
  if (_wave.length > WAVE_MAX) _wave.shift();
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth || 600, cssH = c.clientHeight || 28;
  const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
  if (c.width !== bw || c.height !== bh) { c.width = bw; c.height = bh; }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  const lvlColor = (getComputedStyle(document.documentElement).getPropertyValue("--muted") || "#a1a1a8").trim();
  ctx.fillStyle = lvlColor || "#a1a1a8";
  const barW = cssW / WAVE_MAX;
  for (let i = 0; i < _wave.length; i++) {
    const h = Math.max(2, _wave[i] * cssH);
    ctx.fillRect(i * barW + barW * 0.2, (cssH - h) / 2, barW * 0.6, h);
  }
}
function clearWave() {
  _wave.length = 0;
  const c = $("recWave");
  if (c && c.getContext) { const ctx = c.getContext("2d"); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height); }
}

// ---------- native <details> menus: close on outside-click / Escape ----------
function wireMenus() {
  document.addEventListener("click", (e) => {
    document.querySelectorAll("details.menu[open]").forEach((d) => {
      if (!d.contains(e.target)) d.removeAttribute("open");
    });
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const open = document.querySelector("details.menu[open]");
    if (open) { open.removeAttribute("open"); const s = open.querySelector("summary"); if (s) s.focus(); }
  });
}

// ---------- init ----------
async function init() {
  const o = await getJSON("/api/options");
  if (!o || o.ok === false || !Array.isArray(o.models)) {
    toast("Cannot reach the VoxTerm server — reload once it's running.");
    setView("empty");
    return;
  }
  OPTS = { models: o.models || [], languages: o.languages || {}, default_model: o.default_model || "", input_devices: o.input_devices || [] };
  const mSel = $("model"), lSel = $("language"), dSel = $("micDevice");
  OPTS.models.forEach((m) => { const op = document.createElement("option"); op.value = m; op.textContent = m; if (m === OPTS.default_model) op.selected = true; mSel.appendChild(op); });
  Object.entries(OPTS.languages).forEach(([code, name]) => { const op = document.createElement("option"); op.value = code; op.textContent = name; if (code === "en") op.selected = true; lSel.appendChild(op); });
  (OPTS.input_devices.length ? OPTS.input_devices : [{ index: -1, name: "System default" }]).forEach((d) => { const op = document.createElement("option"); op.value = String(d.index); op.textContent = d.name; dSel.appendChild(op); });

  const savedModel = lsGet(LS_MODEL);
  if (savedModel && OPTS.models.includes(savedModel)) mSel.value = savedModel;
  const savedLang = lsGet(LS_LANG);
  if (savedLang && Object.prototype.hasOwnProperty.call(OPTS.languages, savedLang)) lSel.value = savedLang;
  const savedMic = lsGet(LS_MIC);
  if (savedMic != null && [...dSel.options].some((op) => op.value === savedMic)) dSel.value = savedMic;
  mSel.addEventListener("change", () => lsSet(LS_MODEL, mSel.value));
  lSel.addEventListener("change", () => lsSet(LS_LANG, lSel.value));
  dSel.addEventListener("change", () => lsSet(LS_MIC, dSel.value));
  if (lsGet(LS_DIARIZE) === "0") $("diarize").checked = false;
  $("diarize").addEventListener("change", () => lsSet(LS_DIARIZE, $("diarize").checked ? "1" : "0"));
  // On-device speaker diarization: default ON — the Graph and Interruptions modes are built on real
  // speaker turns, and with it off the interruption counter can never move. It adds a slower pass at
  // stop; turning it off is a persisted opt-out.
  const dOd = $("diarizeOndevice");
  if (dOd) {
    dOd.checked = lsGet(LS_DIARIZE_OD) !== "0";
    dOd.addEventListener("change", () => lsSet(LS_DIARIZE_OD, dOd.checked ? "1" : "0"));
  }
  const savedSource = lsGet(LS_SOURCE);
  if (savedSource && [...$("source").options].some((op) => op.value === savedSource)) $("source").value = savedSource;
  $("source").addEventListener("change", () => lsSet(LS_SOURCE, $("source").value));
  const savedSum = lsGet(LS_SUMMODEL); if (savedSum != null) $("sumModel").value = savedSum;
  $("sumModel").addEventListener("change", () => lsSet(LS_SUMMODEL, $("sumModel").value.trim()));

  $("recBtn").addEventListener("click", toggleRecord);
  $("sessionSearch").addEventListener("input", (e) => renderSessions(e.target.value));
  $("navToggle").addEventListener("click", () => setNav(!document.body.classList.contains("nav-open")));
  $("copyAgent").addEventListener("click", copyForAI);
  $("summarizeAi").addEventListener("click", summarizeForAI);
  // Local-LLM summary needs the Python engine — skip it on-device (the button is CSS-hidden there).
  if (!window.VOX_ONDEVICE) $("summarizeLocal").addEventListener("click", summarizeLocal);
  $("summaryClose").addEventListener("click", () => { $("summaryBlock").classList.add("hidden"); $("summaryBody").textContent = ""; });
  $("summaryCopy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("summaryBody").textContent); toast("Summary copied"); }
    catch { toast("Clipboard blocked"); }
  });
  $("dlWav").addEventListener("click", downloadWav);
  $("dlMd").addEventListener("click", async () => { if (!CUR) return; const t = await serverExport("md"); if (t != null) download(t, `${CUR.session.id}-agent.md`, "text/markdown"); });
  $("dlJson").addEventListener("click", async () => { if (!CUR) return; const t = await serverExport("json"); if (t != null) download(t, `${CUR.session.id}-agent.json`, "application/json"); });
  $("dlSrt").addEventListener("click", async () => { if (!CUR) return; const t = await serverExport("srt"); if (t != null) download(t, `${CUR.session.id}.srt`, "application/x-subrip"); });
  $("dlVtt").addEventListener("click", async () => { if (!CUR) return; const t = await serverExport("vtt"); if (t != null) download(t, `${CUR.session.id}.vtt`, "text/vtt"); });
  $("deleteSession").addEventListener("click", () => { if (CUR_STEM) deleteSession(CUR_STEM, CUR_DIR); });
  setExportEnabled(false);
  wireMenus();

  document.addEventListener("click", (e) => {
    if (!document.body.classList.contains("nav-open")) return;
    if ($("sidebar").contains(e.target) || $("navToggle").contains(e.target)) return;
    setNav(false);
  });
  document.addEventListener("keydown", onKeydown);

  if (window.VOX_CONV) { window.VOX_CONV.getDoc = activeDoc; window.VOX_CONV.init(); }

  await loadSessions();
  openEvents();
}

function onKeydown(e) {
  if (e.key === "Escape") { setNav(false); return; }
  const el = document.activeElement, tag = el && el.tagName;
  const typing = tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA" || tag === "AUDIO" || tag === "VIDEO" || (el && el.isContentEditable);
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const isSpace = e.code === "Space" || e.key === " ";
  if (isSpace && el && (tag === "BUTTON" || el.getAttribute("role") === "button")) return;
  if (isSpace || e.key === "r" || e.key === "R") { e.preventDefault(); toggleRecord(); }
}

// Export/copy actions are meaningless without a loaded transcript — disable them outright.
function setExportEnabled(on) {
  ["copyAgent", "summarizeAi", "dlJson", "dlMd", "dlSrt", "dlVtt"].forEach((id) => { $(id).disabled = !on; });
  // WAV + delete + local-summarize only need a loaded session (work even in raw-markdown view)
  const haveSession = !!CUR_STEM;
  $("summarizeLocal").disabled = !haveSession;
  $("dlWav").disabled = !haveSession;
  $("deleteSession").disabled = !haveSession;
}

// ---------- live status (SSE) ----------
function openEvents() {
  const es = BACKEND.events();
  es.onmessage = (e) => { let s; try { s = JSON.parse(e.data); } catch { return; } applyStatus(s); };
  es.onerror = () => {/* browser auto-reconnects */};
}
function applyStatus(s) {
  const job = s.job || { state: "idle" };
  document.body.classList.toggle("recording", !!s.recording);
  $("recBtn").setAttribute("aria-label", s.recording ? "Stop recording" : "Start recording");
  if (s.recording) {
    $("timer").textContent = fmtClock(s.elapsed);
    drawWave(s.level);
    $("model").disabled = $("language").disabled = $("micDevice").disabled = true;
    _levelPeak = Math.max(_levelPeak, s.level || 0);
    const quiet = (Date.now() - _recStartedAt) / 1000 > 3 && _levelPeak < 0.02;
    document.body.classList.toggle("low-signal", quiet);
    $("recState").textContent = quiet ? "Mic very quiet — check input" : "Recording";
  } else {
    document.body.classList.remove("low-signal");
    if (_wave.length) clearWave();
    $("model").disabled = $("language").disabled = $("micDevice").disabled = false;
    if (lastJobState === "idle" && job.state === "idle") $("recState").textContent = "Ready";
  }
  const working = job.state === "transcribing";
  document.body.classList.toggle("working", working);
  if (working) {
    if (lastJobState !== "transcribing" && !CUR) {   // replace the "Recording…" placeholder once
      $("empty").innerHTML = `<p class="rec-ph"><strong>Transcribing…</strong><br>Building your transcript — this takes a few seconds.</p>`;
      setView("empty");
    }
    $("progress").classList.remove("hidden");
    $("progressMsg").textContent = job.msg || "Transcribing…";
    const pct = Math.round((job.frac || 0) * 100);
    $("progressPct").textContent = pct + "%";
    $("barFill").style.width = pct + "%";
    $("recState").textContent = "Transcribing…";
    $("recBtn").disabled = true;
  }
  if (job.state === "done" && lastJobState !== "done") {
    $("progress").classList.add("hidden");
    $("recState").textContent = "Ready";
    $("recBtn").disabled = false;
    if (!job.n_turns) { toast("No speech detected — check your mic level or speak closer."); showNoSpeech(); }
    else { toast(`Done — ${job.n_turns} turns, ${job.n_speakers} speaker(s)` + (job.warning ? ` · ${job.warning}` : "")); if (job.stem) loadSession(job.stem); }
    loadSessions();
  }
  if (job.state === "error" && lastJobState !== "error") {
    $("progress").classList.add("hidden");
    $("recBtn").disabled = false;
    toast("Error: " + (job.error || "transcription failed"));
    $("recState").textContent = "Ready";
    if (!CUR) { $("empty").innerHTML = EMPTY_DEFAULT; setView("empty"); }   // clear any stale live view
  }
  // live transcript document (near-real-time tail of an in-progress recording).
  const live = s.live || { active: false, lines: [], partial: null };
  document.body.classList.toggle("live-on", !!live.active);
  if (live.active && s.recording) {   // only while actually recording — never re-flip a loaded transcript
    const lines = live.lines || [];
    LIVE_DOC = buildLiveDoc(lines);    // feed the Graph/Interruptions modes the live tail
    setView("live");                   // triggers a re-analyze if a derived mode is open (reads LIVE_DOC)
    const el = $("liveLines");
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    let html = lines.map((l) => `<div class="ll"><span class="ll-t">${escapeHtml(l.t)}</span>${escapeHtml(l.text)}</div>`).join("");
    const p = live.partial;
    if (p && (p.stable || p.volatile)) {
      html += `<div class="ll ll-partial"><span class="ll-t">${escapeHtml(p.t || "")}</span>`
        + escapeHtml(p.stable || "") + (p.stable && p.volatile ? " " : "")
        + `<span class="ll-vol">${escapeHtml(p.volatile || "")}</span></div>`;
    }
    el.innerHTML = html || `<div class="ll-empty">listening…</div>`;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }
  lastJobState = job.state;
}

// Shown while recording (no live transcript preview by design — see toggleRecord). The dock keeps
// the pulsing record dot + level meter; the accurate, diarized transcript replaces this on stop.
function showRecording() {
  CUR = null; CUR_STEM = null; CUR_DIR = null; RENAMES = {};
  setExportEnabled(false);
  $("empty").innerHTML = `<p class="rec-ph"><span class="rec-ph-dot"></span><strong>Recording…</strong><br>`
    + `Speak normally — your transcript appears here when you stop.</p>`;
  setView("empty");
}

// Shown when a recording finishes with zero turns — so an empty result is explained, not silent.
function showNoSpeech() {
  CUR = null; CUR_STEM = null; CUR_DIR = null;
  setExportEnabled(false);
  setView("empty");
  $("empty").innerHTML = `<p><strong>No speech detected.</strong><br>That recording came through near-silent — `
    + `your mic may be muted or too quiet. Raise the input level, move closer, or pick a different `
    + `input device (gear menu), then record again. The audio was still saved.</p>`;
}

// ---------- record ----------
async function toggleRecord() {
  const recording = document.body.classList.contains("recording");
  if (!recording) {
    _levelPeak = 0; _recStartedAt = Date.now();
    const device = parseInt($("micDevice").value, 10);
    // On a fresh install the FIRST start blocks on the OS mic-permission prompt. Disable the button
    // and show a clear "Starting…" so a second tap can't desync the start/stop state machine while the
    // dialog is up (the bug behind "pressed three times before it recorded"). Re-enabled below.
    $("recBtn").disabled = true;
    $("recState").textContent = "Starting…";
    const r = await getJSON("/api/record/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device: Number.isNaN(device) ? -1 : device, source: $("source").value }),
    });
    $("recBtn").disabled = false;   // recording is live now (or it failed) — applyStatus owns it from here
    if (!r.ok) { $("recState").textContent = "Ready"; toast(r.error ? "Mic error: " + r.error : "Could not start (mic busy?)"); return; }
    showRecording();   // no live preview — keep recording light; the accurate transcript appears on stop
  } else {
    $("recBtn").disabled = true;     // debounce; applyStatus is the single owner of re-enabling
    // record/stop tears down the live monitor itself (engine.stop_recording -> live_stop),
    // so we don't call /api/live/stop separately — doing so raced two live_stop() calls.
    const r = await getJSON("/api/record/stop", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: $("model").value, language: $("language").value, diarize: (window.VOX_ONDEVICE ? $("diarizeOndevice") : $("diarize")).checked }),
    });
    // On success the SSE job (transcribing -> done/error) re-enables the button; only re-enable
    // here if the stop request itself failed (so the button can't get stuck disabled).
    if (!r || r.ok === false) { $("recBtn").disabled = false; toast(r && r.error ? "Stop failed: " + r.error : "Stop failed"); }
  }
}

// ---------- sessions ----------
async function loadSessions() {
  SESSIONS = (await getJSON("/api/sessions")).sessions || [];
  renderSessions($("sessionSearch") ? $("sessionSearch").value : "");
}
function sessionTitle(s) { return (s.title && s.title.trim()) || prettyStem(s.stem); }
// Derive a title from the loaded doc's first words — used immediately after a recording, before
// the session list has refreshed with the server-derived title (avoids showing the raw date).
function titleFromTurns(doc) {
  const t = (doc.turns || []).map((x) => (x.text || "").trim()).find((s) => s.length >= 2);
  if (!t) return "";
  const s = t.replace(/\s+/g, " ");
  return s.length > 56 ? s.slice(0, 54).trimEnd() + "…" : s;
}
function renderSessions(query) {
  const q = (query || "").trim().toLowerCase();
  const list = q ? SESSIONS.filter((s) => sessionTitle(s).toLowerCase().includes(q) || prettyStem(s.stem).toLowerCase().includes(q) || s.stem.toLowerCase().includes(q)) : SESSIONS;
  const ul = $("sessions"); ul.innerHTML = "";
  if (!SESSIONS.length) { ul.innerHTML = `<li class="sessions-empty">No sessions yet — record one to get started.</li>`; return; }
  if (!list.length) { ul.innerHTML = `<li class="sessions-empty">No sessions match “${escapeHtml(query)}”.</li>`; return; }
  list.forEach((s) => {
    const li = document.createElement("li"); li.className = "session"; li.dataset.stem = s.stem;
    li.tabIndex = 0; li.setAttribute("role", "button");
    li.innerHTML = `<div class="s-main"><div class="s-title">${escapeHtml(sessionTitle(s))}</div>
      <div class="s-sub">${escapeHtml(prettyStem(s.stem))}</div></div>
      <button class="session-del" title="Delete" aria-label="Delete session">✕</button>`;
    li.addEventListener("click", () => loadSession(s.stem, s.dir));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); loadSession(s.stem, s.dir); } });
    const del = li.querySelector(".session-del");
    del.addEventListener("keydown", (e) => { e.stopPropagation(); });
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s.stem, s.dir); });
    ul.appendChild(li);
  });
}
function prettyStem(stem) {
  const m = stem.match(/(\d{4})-?(\d{2})-?(\d{2})[_-]?(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : stem;
}

async function loadSession(stem, dir) {
  const dq = dir ? `&dir=${encodeURIComponent(dir)}` : "";
  let res = await getJSON(`/api/session?stem=${encodeURIComponent(stem)}&kind=agent_json${dq}`);
  if (!res.ok) {
    res = await getJSON(`/api/session?stem=${encodeURIComponent(stem)}&kind=transcript${dq}`);
    if (res.ok) return showRawMarkdown(stem, dir, res.text);
    if (!CUR) { $("empty").innerHTML = EMPTY_DEFAULT; setView("empty"); }   // don't strand "Recording…"
    return toast("Could not load session");
  }
  try { CUR = JSON.parse(res.text); } catch {
    if (!CUR) { $("empty").innerHTML = EMPTY_DEFAULT; setView("empty"); }
    return toast("Bad session JSON");
  }
  CUR._dir = dir || null; CUR_STEM = stem; CUR_DIR = dir || null;
  RENAMES = {};
  render();
  document.querySelectorAll(".session").forEach((el) => el.classList.toggle("active", el.dataset.stem === stem));
  setNav(false);
}

async function deleteSession(stem, dir) {
  if (!confirm("Delete this session's transcript files? (audio is kept)")) return;
  const r = await getJSON("/api/session/delete", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stem: stem, dir: dir || null }),
  });
  if (!r || r.ok === false) return toast("Could not delete session");
  const wasOpen = CUR_STEM === stem;
  toast(r.deleted && r.deleted.length ? `Deleted ${r.deleted.length} file(s)` : "Nothing to delete");
  $("exportMenu").removeAttribute("open");
  await loadSessions();
  if (wasOpen) { CUR = null; CUR_STEM = null; CUR_DIR = null; RENAMES = {}; setExportEnabled(false); $("empty").innerHTML = EMPTY_DEFAULT; setView("empty"); }
}

// ---------- audio player ----------
function audioUrl(stem, dir) {
  return BACKEND.authUrl(`/api/audio?stem=${encodeURIComponent(stem)}${dir ? `&dir=${encodeURIComponent(dir)}` : ""}`);
}
let _playerToken = 0;
async function setupPlayer(stem, dir) {
  const my = ++_playerToken;            // ignore a slow probe that resolves after a newer session loads
  const player = $("player");
  player.classList.add("hidden"); player.removeAttribute("src");
  $("dlWav").disabled = true;     // re-enabled below only if audio actually exists for this session
  const url = audioUrl(stem, dir);
  try {
    // Probe with a 1-byte range (the server has no HEAD); 206/200 => audio exists.
    const r = await fetch(url, { headers: { Range: "bytes=0-0" } });
    if (my !== _playerToken) return;     // a newer session rendered while we were probing
    if (r.ok) { player.src = url; player.classList.remove("hidden"); $("dlWav").disabled = false; }
  } catch { /* no audio — leave hidden + download disabled */ }
}
async function downloadWav() {
  if (!CUR_STEM) return;
  const a = document.createElement("a");
  a.href = audioUrl(CUR_STEM, CUR_DIR); a.download = `${CUR_STEM}.wav`; a.click();
  $("exportMenu").removeAttribute("open");
}

// ---------- render ----------
function render() {
  setView("transcript");
  setExportEnabled(true);
  const s = CUR.session;
  const known = SESSIONS.find((x) => x.stem === s.id);
  $("tvTitle").textContent = (known && sessionTitle(known)) || titleFromTurns(CUR) || prettyStem(s.id);
  $("tvMeta").textContent = `${prettyStem(s.id)} · ${CUR.turns.length} turns · ${CUR.speakers.length} speaker(s) · ${s.duration_hms || ""} · ${s.model || ""}`;
  setupPlayer(s.id, CUR._dir);

  const wrap = $("turns"); wrap.innerHTML = "";
  let prevSid = null;
  CUR.turns.forEach((t) => {
    const same = !t.peer && t.speaker_id === prevSid;
    prevSid = t.peer ? null : t.speaker_id;
    const row = document.createElement("div");
    row.className = "turn" + (t.confidence_uncertain ? " uncertain" : "") + (same ? " same-speaker" : "");
    const c = t.peer ? PEER_COLOR : colorFor(t.speaker_id);
    const mk = (t.markers || []).map((m) => `<span class="mk">${escapeHtml(m)}</span>`).join("");
    // On-device there's one speaker (no diarization), so render the name as plain text — no rename
    // button (it would be a no-op; a CSS-only disable leaves it keyboard/AT-reachable).
    const spk = (t.peer || window.VOX_ONDEVICE)
      ? `<span class="t-spk"><span class="dot" style="background:${c}"></span>${escapeHtml(nameFor(t))}</span>`
      : `<span class="t-spk"><span class="dot" style="background:${c}"></span><button data-sid="${t.speaker_id}">${escapeHtml(nameFor(t))}</button></span>`;
    row.innerHTML = `<div class="t-head">${spk}<span class="t-time">${escapeHtml(t.t_offset_hms || "")}</span>${mk}</div>`
      + `<div class="t-text">${escapeHtml(t.text)}</div>`;
    const btn = row.querySelector("button[data-sid]");
    if (btn) btn.addEventListener("click", () => renameSpeaker(t.speaker_id));
    // click the timestamp to seek the audio player (if a player is loaded)
    if (typeof t.t_offset === "number") {
      const tm = row.querySelector(".t-time"); tm.style.cursor = "pointer"; tm.title = "Play from here";
      tm.addEventListener("click", () => {
        const p = $("player"); if (!p.src) return;
        const seek = () => { try { p.currentTime = t.t_offset; } catch { /* not seekable yet */ } p.play().catch(() => {}); };
        if (p.readyState >= 1) seek();                                   // metadata ready
        else { p.addEventListener("loadedmetadata", seek, { once: true }); p.load(); }  // cold (preload=none)
      });
    }
    wrap.appendChild(row);
  });
  if (window.VOX_CONV) window.VOX_CONV.refresh();   // refresh Graph/Interruptions if one is open
}
function showRawMarkdown(stem, dir, text) {
  CUR = null; LIVE_DOC = null; CUR_STEM = stem; CUR_DIR = dir || null;   // no structured turns → derived modes show empty, not a stale live graph
  setExportEnabled(false);   // no structured JSON behind a raw-markdown view (WAV/delete stay on)
  setView("transcript");
  $("tvTitle").textContent = prettyStem(stem); $("tvMeta").textContent = "(no AI export — raw transcript)";
  setupPlayer(stem, dir);
  $("turns").innerHTML = `<pre style="white-space:pre-wrap;font:inherit;color:var(--muted);margin:0">${escapeHtml(text)}</pre>`;
  document.querySelectorAll(".session").forEach((el) => el.classList.toggle("active", el.dataset.stem === stem));
}
function renameSpeaker(sid) {
  const cur = RENAMES[sid] || (CUR.speakers.find((x) => x.id === sid) || {}).label || `Speaker ${sid}`;
  const name = prompt("Rename speaker (applies to this view + your copy/export):", cur);
  if (name && name.trim()) { RENAMES[sid] = name.trim(); render(); toast("Renamed — included when you copy/export"); }
}

// ---------- export (server-side, single source of truth) ----------
async function serverExport(kind) {
  if (!CUR) return null;
  const r = await getJSON("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stem: CUR.session.id, dir: CUR._dir || null, kind, renames: RENAMES }),
  });
  if (!r || r.ok === false) { toast("Export failed — is the session still on disk?"); return null; }
  return r.text;
}
// Copy `text` to the clipboard, or download it as markdown if the clipboard is blocked.
async function copyOrDownload(text, filename, copiedMsg) {
  try { await navigator.clipboard.writeText(text); toast(copiedMsg); }
  catch { download(text, filename, "text/markdown"); toast("Clipboard blocked — downloaded instead"); }
}
async function copyForAI() {
  if (!CUR) return toast("Load a transcript first");
  const md = await serverExport("md");
  if (md == null) return;
  await copyOrDownload(md, `${CUR.session.id}-agent.md`, "Copied AI transcript to clipboard");
}
async function summaryPrompt() {
  const md = await serverExport("md");
  if (md == null) return null;
  return [
    "## Task", "",
    "You are given a transcript of a recorded conversation (below). Read it in full, then produce:", "",
    "1. **Summary** — a concise overview (3-5 sentences) of what the conversation was about.",
    "2. **Key decisions** — a bullet list of decisions reached, or \"None\" if there were none.",
    "3. **Action items** — a bullet list of follow-ups, each with the owner if one is identifiable.",
    "4. **Per-speaker highlights** — for each speaker, 1-2 bullets on their main points or positions.", "",
    "Stick to what the transcript actually says. Do not invent details. Speaker labels are diarization",
    "clusters or manual renames, not verified identities — treat them as such.", "", "---", "", md,
  ].join("\n");
}
async function summarizeForAI() {
  if (!CUR) return toast("Load a transcript first");
  const text = await summaryPrompt();
  if (text == null) return;
  $("exportMenu").removeAttribute("open");
  await copyOrDownload(text, `${CUR.session.id}-summarize.md`, "Copied summary prompt to clipboard");
}
// Run the summary locally via the engine's summarizer (MLX on Apple Silicon, or an
// `ollama:<model>` backend set in settings). Shows the result inline, or a clear error
// when no backend is available — distinct from "Summarize for AI" which copies a prompt.
async function summarizeLocal() {
  if (!CUR_STEM) return toast("Load a transcript first");
  $("exportMenu").removeAttribute("open");
  const body = $("summaryBody"), block = $("summaryBlock");
  $("summaryTitle").textContent = "Summarizing…";
  body.textContent = ""; block.classList.remove("hidden");
  const r = await getJSON("/api/summarize", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stem: CUR_STEM, dir: CUR_DIR, model: $("sumModel").value.trim() }),
  });
  if (r && r.ok) {
    $("summaryTitle").textContent = "Summary" + (r.template ? ` · ${r.template}` : "");
    body.textContent = r.summary || "(empty summary)";
  } else {
    block.classList.add("hidden");
    toast("Summarize failed: " + ((r && r.error) || "no local LLM backend"));
  }
}
function download(text, filename, mime) {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  $("exportMenu").removeAttribute("open");
}

init().catch((e) => toast("Init failed: " + e));

// PWA: register the offline app-shell service worker (script-src 'self' allows this). The on-device
// bundle ships no SW (it's already fully local, and a SW under the asset origin only risks staleness).
if ("serviceWorker" in navigator && !window.VOX_ONDEVICE) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
}
