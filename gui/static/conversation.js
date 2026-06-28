"use strict";
// Conversation modes: the Graph (topic/argument tree) and Interruptions views, plus the mode
// switcher that owns which content panel is visible. app.js stays in charge of the top-level state
// (empty / live / transcript); this module owns the panel-level switch between Transcript, Graph and
// Interruptions and renders the latter two from window.VOX_ANALYZE.analyze(activeDoc).
//
// Coupling with app.js is deliberately thin:
//   VOX_CONV.getDoc      — app.js sets this to a fn returning the active {turns, speakers, session}
//                          (the loaded transcript, or a synthesized doc while recording live).
//   VOX_CONV.setTop(s)   — app.js calls this from setView('empty'|'live'|'transcript').
//   VOX_CONV.refresh()   — app.js calls this when the active doc's content changes (new turns, live
//                          tail) so Graph/Interruptions re-analyze + redraw if they're showing.
// Seeking reuses window.VOX_SEEK(sec) (set by app.js) so nodes/events are click-to-play.

(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const SVGNS = "http://www.w3.org/2000/svg";

  let top = "empty";                 // 'empty' | 'live' | 'transcript'
  let mode = "transcript";           // 'transcript' | 'graph' | 'interruptions'
  let lastSig = "";                  // content signature — skip redundant re-analysis
  let analyzeTok = 0;                // guards against an out-of-order async analyze resolving late
  let debounce = null;
  // On-device LLM is OPT-IN: the heuristic renders instantly + live (cheap, safe), and the LLM runs
  // only when the user taps "Sharpen" — running the ~0.5 GB model automatically on every render can
  // OOM-crash the app (a native crash JS can't catch). llmCache holds the last LLM result for the doc
  // it was computed for, so it survives mode switches but is dropped when the transcript changes.
  let llmCache = null;               // { sig, data:{graph,interruptions}, status:"partial"|"complete" }
  let llmBusy = false;
  let llmProgress = null;            // { done, total } during a multi-chunk Sharpen run, else null
  let llmToken = null;               // { canceled } for the in-flight run, so a new run/leave can stop it
  // Graph view state (render-only — no re-analysis needed): which slice of utterances to show, and the
  // per-node on-device "insight" results. Insight is OPT-IN like Sharpen and reuses the same model.
  let graphFilter = "all";           // 'all' | 'args' | 'spk:<name>'
  const insightCache = new Map();    // nodeKey -> insight text ("…" while running)
  let insightBusy = null;            // nodeKey currently generating, else null
  let insightTok = 0;
  let lastDocSig = "";               // drop insight cache when the transcript itself changes

  function docSig(doc) {
    const turns = (doc && doc.turns) || [];
    const last = turns[turns.length - 1];
    return `${turns.length}|${last ? (last.text || "").length : 0}|${last ? last.t_offset : 0}`;
  }
  function seek(sec) { if (typeof sec === "number" && window.VOX_SEEK) window.VOX_SEEK(sec); }

  // ---- mode switch -----------------------------------------------------------
  function setTop(state) { top = state; applyMode(); }
  function setMode(m) { if (m === mode) return; mode = m; lastSig = ""; applyMode(); }

  function applyMode() {
    const live = top === "live", loaded = top === "transcript", content = live || loaded;
    $("modeTabs").classList.toggle("hidden", !content);
    // Transcript mode shows the existing turns/liveLines; Graph/Interruptions show their panels.
    $("turns").classList.toggle("hidden", !(content && mode === "transcript" && loaded));
    $("liveLines").classList.toggle("hidden", !(content && mode === "transcript" && live));
    $("graphPanel").classList.toggle("hidden", !(content && mode === "graph"));
    $("interruptPanel").classList.toggle("hidden", !(content && mode === "interruptions"));
    document.querySelectorAll("#modeTabs .mode-tab").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    if (content && mode !== "transcript") scheduleAnalyze(true);
  }

  // Called by app.js when content changes. Only does work if a derived panel is actually showing.
  function refresh() { if ((top === "live" || top === "transcript") && mode !== "transcript") scheduleAnalyze(false); }

  function scheduleAnalyze(immediate) {
    clearTimeout(debounce);
    debounce = setTimeout(runAnalyze, immediate ? 0 : 350);
  }

  async function runAnalyze() {
    const doc = api.getDoc && api.getDoc();
    if (!doc || !(doc.turns || []).length) { renderEmpty(); return; }
    const ds = docSig(doc);
    if (ds !== lastDocSig) { insightCache.clear(); insightBusy = null; lastDocSig = ds; }  // stale insights
    const useLlm = !!(llmCache && llmCache.sig === ds);   // user already sharpened THIS transcript
    const sig = ds + "|" + mode + "|" + (useLlm ? "llm" : "heur") + "|" + (llmBusy ? "busy" : "");
    if (sig === lastSig) return;     // nothing changed since last draw
    lastSig = sig;
    const tok = ++analyzeTok;
    let res, source;
    if (useLlm) { res = llmCache.data; source = "llm"; }
    else {
      try { res = await window.VOX_ANALYZE.analyze(doc); source = "heuristic"; }
      catch (e) { if (tok === analyzeTok) renderError(String((e && e.message) || e)); return; }
    }
    if (tok !== analyzeTok) return;  // a newer analyze superseded this one
    if (mode === "graph") renderGraph(res.graph, source);
    else if (mode === "interruptions") renderInterruptions(res.interruptions, source);
  }

  // Toolbar shown atop the Graph / Interruptions panels: the caption, plus the opt-in on-device-LLM
  // control (Sharpen button → busy state → "on-device LLM" badge). Tapping Sharpen runs the model
  // exactly once for the current transcript; failures fall back to the already-shown heuristic.
  function toolbar(caption, source) {
    let right = "";
    if (llmBusy) {
      const prog = (llmProgress && llmProgress.total > 1)
        ? `Analyzing on-device… chunk ${Math.min(llmProgress.done + 1, llmProgress.total)} / ${llmProgress.total}`
        : `Analyzing on-device… first run loads the model (~10s)`;
      right = `<span class="llm-state" aria-live="polite">${esc(prog)}</span>`;
    }
    else if (source === "llm") {
      const partial = llmCache && llmCache.status === "partial";
      right = `<span class="llm-badge" title="${esc((window.VOX_LLM && window.VOX_LLM.model) || "on-device LLM")}">⚡ on-device LLM${partial ? " · partial" : ""}</span>`;
    }
    else if (window.VOX_LLM && window.VOX_LLM.available) right = `<button class="llm-sharpen" type="button">✨ Sharpen with on-device AI</button>`;
    return `<div class="conv-toolbar"><span class="conv-cap">${caption}</span>${right}</div>`;
  }
  function wireToolbar(root) {
    const b = root.querySelector(".llm-sharpen");
    if (b) b.addEventListener("click", sharpen);
  }

  async function sharpen() {
    const doc = api.getDoc && api.getDoc();
    if (!doc || llmBusy || !(window.VOX_LLM && window.VOX_LLM.available)) return;
    const ds = docSig(doc);
    if (llmToken) llmToken.canceled = true;              // stop any prior run before starting a new one
    const token = llmToken = { canceled: false };
    llmBusy = true; llmProgress = { done: 0, total: 0 }; lastSig = ""; runAnalyze();   // repaint with the busy state
    // Paint each chunk's merged result as it arrives — a 30–60 s run feels responsive instead of a
    // single long spinner. The partial is cached (status:"partial") so it survives mode switches.
    const onProgress = (p) => {
      if (token.canceled) return;
      llmProgress = { done: p.done, total: p.total };
      if (p.phase === "partial" && p.data) { llmCache = { sig: ds, data: p.data, status: "partial" }; }
      lastSig = ""; runAnalyze();
    };
    try {
      const data = await window.VOX_LLM.analyze(doc, onProgress, token);
      if (!token.canceled) llmCache = { sig: ds, data, status: "complete" };
    } catch (e) {
      const m = String((e && e.message) || e);
      if (m === "canceled") return;                      // superseded by a newer run — leave its state alone
      if (!(llmCache && llmCache.sig === ds)) {           // keep any partial we already painted
        llmCache = null;
        const p = mode === "graph" ? $("graphPanel") : $("interruptPanel");
        if (p) { const s = document.createElement("div"); s.className = "conv-hint"; s.textContent = "On-device AI couldn't analyze this one — showing the quick estimate."; p.prepend(s); }
      }
    } finally {
      if (token === llmToken) { llmBusy = false; llmProgress = null; llmToken = null; lastSig = ""; runAnalyze(); }
    }
  }

  function renderEmpty() {
    if (mode === "graph") $("graphPanel").innerHTML = `<p class="conv-empty">No conversation yet — the topic map appears as people talk.</p>`;
    if (mode === "interruptions") $("interruptPanel").innerHTML = `<p class="conv-empty">No interruptions yet — the counter updates live as people talk over each other.</p>`;
  }
  function renderError(msg) {
    const html = `<p class="conv-empty">Couldn't analyze this conversation: ${esc(msg)}</p>`;
    if (mode === "graph") $("graphPanel").innerHTML = html;
    if (mode === "interruptions") $("interruptPanel").innerHTML = html;
  }

  // ---- GRAPH: top-down topic/argument flow (mobile-first, vertical) ----------
  // A vertical outline: each topic is a full-width section header, its utterances are cards beneath it
  // in time order. Argument relationships show as an indent + glyph (↳ rebuts / ↳ supports) rather than
  // horizontal edges, so it reads cleanly on a phone (vertical scroll only). Tapping a card seeks audio;
  // the 💡 surfaces an on-device insight about the topic (opt-in, mobile LLM only).
  const TYPE_LABEL = { statement: "", question: "Q", retort: "↩", counter: "⚔" };
  const speakerOf = (n) => n.speaker || (n.speaker_id != null ? `Speaker ${n.speaker_id}` : null);
  function clip(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  function renderGraph(graph, source) {
    const nodes = graph.nodes || [], edges = graph.edges || [];
    const byId = {}; nodes.forEach((n) => (byId[n.id] = n));
    const topics = nodes.filter((n) => n.type === "topic");
    const inEdge = {};   // incoming utterance→utterance edge kind per utterance (relationship to prev)
    edges.forEach((e) => { if (e.kind === "reply" || e.kind === "rebuts" || e.kind === "supports") inEdge[e.to] = e.kind; });
    const childrenOf = (tid) => edges.filter((e) => e.from === tid && e.kind === "contains").map((e) => byId[e.to]).filter(Boolean);

    const spk = [];   // distinct speakers (for the by-speaker filter)
    nodes.forEach((n) => { const s = speakerOf(n); if (s && !spk.includes(s)) spk.push(s); });
    const inFilter = (n) => {
      if (graphFilter === "args") return n.type === "retort" || n.type === "counter" || n.type === "question" || inEdge[n.id] === "rebuts" || inEdge[n.id] === "supports";
      if (graphFilter.startsWith("spk:")) return speakerOf(n) === graphFilter.slice(4);
      return true;
    };

    const llm = !!(window.VOX_LLM && window.VOX_LLM.available);
    let html = "", shownUtter = 0;
    topics.forEach((tp) => {
      const kids = childrenOf(tp.id).filter(inFilter);
      if (!kids.length) return;
      // The topic header just labels the section; insight lives on the leaves (tap a card's 💡), so the
      // header stays clean. Its panel still renders here if a leaf resolved to this topic's key.
      html += `<section class="gtopic">`
        + `<div class="gtopic-head"><span class="gtopic-label">${esc(tp.label)}</span></div>`
        + insightPanel(tp.id);
      kids.forEach((n) => {
        shownUtter++;
        const rel = inEdge[n.id], badge = TYPE_LABEL[n.type] || "", sname = speakerOf(n);
        const seekAttr = typeof n.t_offset === "number" ? ` data-seek="${n.t_offset}" tabindex="0"` : "";
        html += `<div class="gutter rel-${rel || "none"}">`
          + `<button class="gutter-main type-${n.type}" type="button"${seekAttr}>`
          + (rel === "rebuts" ? `<span class="grel reb">↳ rebuts</span>` : rel === "supports" ? `<span class="grel sup">↳ supports</span>` : "")
          + (badge ? `<span class="gbadge">${esc(badge)}</span>` : "")
          + (sname && spk.length > 1 ? `<span class="gspk">${esc(clip(sname, 16))}</span>` : "")
          + `<span class="gtext">${esc(n.label)}</span></button>`
          + insightBtn(tp.id, n.id, llm)
          + insightPanel(n.id)
          + `</div>`;
      });
      html += `</section>`;
    });
    if (!html) html = `<p class="conv-empty">${topics.length ? "Nothing matches this filter." : "No topics yet — they appear as people talk."}</p>`;

    const cap = `Topic map · ${topics.length} topic${topics.length === 1 ? "" : "s"} · ${shownUtter} shown`;
    $("graphPanel").innerHTML = toolbar(cap, source) + filterBar(spk) + `<div class="graph-flow">${html}</div>`;
    wireSeeks($("graphPanel")); wireToolbar($("graphPanel"));
    wireFilter($("graphPanel")); wireInsights($("graphPanel"), byId, edges);
  }

  // ---- graph filter (All · Arguments · by speaker) ---------------------------
  function filterBar(spk) {
    const seg = (val, lbl) => `<button class="gfilter${graphFilter === val ? " active" : ""}" type="button" data-filter="${val}">${lbl}</button>`;
    let html = `<div class="graph-filter">${seg("all", "All")}${seg("args", "Arguments")}`;
    if (spk.length > 1) {
      const byS = graphFilter.startsWith("spk:");
      html += `<select class="gspk-sel" aria-label="Filter by speaker"><option value="all"${byS ? "" : " selected"}>All speakers</option>`
        + spk.map((s) => `<option value="spk:${esc(s)}"${graphFilter === "spk:" + s ? " selected" : ""}>${esc(clip(s, 22))}</option>`).join("") + `</select>`;
    }
    return html + `</div>`;
  }
  function wireFilter(root) {
    root.querySelectorAll(".gfilter").forEach((b) => b.addEventListener("click", () => { graphFilter = b.dataset.filter; lastSig = ""; runAnalyze(); }));
    const sel = root.querySelector(".gspk-sel");
    if (sel) sel.addEventListener("change", () => { graphFilter = sel.value === "all" ? "all" : sel.value; lastSig = ""; runAnalyze(); });
  }

  // ---- tap-to-insight on a leaf/topic (opt-in, on-device LLM) -----------------
  function insightBtn(topicId, key, llm) {
    if (!llm) return "";
    const busy = insightBusy === key;
    return `<button class="ins-btn${busy ? " busy" : ""}" type="button" data-insight="${esc(topicId)}" data-inskey="${esc(key)}" `
      + `title="Insight about this topic" aria-label="Insight about this topic"${busy ? " disabled" : ""}>${busy ? "…" : "💡"}</button>`;
  }
  function insightPanel(key) {
    if (!insightCache.has(key)) return "";
    return `<div class="ins-panel">${esc(insightCache.get(key))}</div>`;
  }
  function wireInsights(root, byId, edges) {
    root.querySelectorAll(".ins-btn").forEach((b) => b.addEventListener("click", (ev) => {
      ev.stopPropagation();
      requestInsight(b.dataset.insight, b.dataset.inskey, byId, edges);
    }));
  }
  async function requestInsight(topicId, key, byId, edges) {
    if (!(window.VOX_LLM && window.VOX_LLM.insight)) return;
    if (insightCache.has(key)) { insightCache.delete(key); lastSig = ""; runAnalyze(); return; }  // tap again to collapse
    if (insightBusy || llmBusy) return;   // one insight at a time; never compete with a full Sharpen
    const tp = byId[topicId];
    const kids = edges.filter((e) => e.from === topicId && e.kind === "contains").map((e) => byId[e.to]).filter(Boolean);
    const focus = key !== topicId ? byId[key] : null;
    const payload = {
      label: tp ? tp.label : "topic",
      points: kids.map((n) => ({ type: n.type, speaker: speakerOf(n) || "", text: n.label })),
      focus: focus ? focus.label : null,
    };
    insightBusy = key; insightCache.set(key, "…thinking"); lastSig = ""; runAnalyze();
    const myTok = ++insightTok;
    try {
      const txt = await window.VOX_LLM.insight(payload);
      if (myTok === insightTok) insightCache.set(key, (txt || "").trim() || "No insight available.");
    } catch (e) {
      if (myTok === insightTok) insightCache.set(key, "Couldn't generate an insight — try again.");
    } finally {
      if (myTok === insightTok) { insightBusy = null; lastSig = ""; runAnalyze(); }
    }
  }

  // ---- INTERRUPTIONS: counters + timeline + list -----------------------------
  function renderInterruptions(ir, source) {
    const dur = ir.durationSec || 0;
    const cap = source === "llm"
      ? `Speakers + interruptions inferred on-device by the LLM.`
      : ir.multiSpeaker
        ? `Detected from speaker changes + timing across multiple speakers.`
        : `Single-speaker transcription — the quick estimate can't tell speakers apart, so it finds few interruptions. Tap “Sharpen” to let the on-device LLM infer speakers.`;

    const stat = (n, label, cls) => `<div class="stat ${cls}"><div class="stat-n">${n}</div><div class="stat-l">${label}</div></div>`;
    const head = toolbar(esc(cap), source)
      + `<div class="ir-stats">`
      + stat(ir.total, "Interruptions", "total")
      + stat(ir.overlapCount, "Overlaps", "overlap")
      + stat(ir.rapidCount, "Rapid switches", "rapid")
      + `</div>`;

    // merge + sort events for the timeline and list
    const events = ir.overlap.map((e) => ({ ...e, cat: "overlap" }))
      .concat(ir.rapidSwitch.map((e) => ({ ...e, cat: "rapid" })))
      .sort((a, b) => (a.t_offset || 0) - (b.t_offset || 0));

    $("interruptPanel").innerHTML = head + timeline(events, dur) + perMinute(events, dur) + eventList(events);
    wireSeeks($("interruptPanel")); wireToolbar($("interruptPanel"));
  }

  function timeline(events, dur) {
    if (!dur || !events.length) return `<div class="ir-timeline-empty">No interruptions on the timeline yet.</div>`;
    const W = 1000, H = 46, pad = 8, y0 = 26;
    let ticks = "";
    events.forEach((e) => {
      const x = pad + (Math.min(e.t_offset || 0, dur) / dur) * (W - pad * 2);
      const col = e.cat === "overlap" ? "#c47a7a" : "#c4a86a";
      ticks += `<line class="ir-tick" data-seek="${e.t_offset}" x1="${x}" y1="${y0 - 12}" x2="${x}" y2="${y0 + 12}" stroke="${col}" stroke-width="2"><title>${esc(e.t_hms)} · ${e.cat}</title></line>`;
    });
    return `<div class="ir-timeline"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="ir-timeline-svg" role="img" aria-label="Interruption timeline">`
      + `<line x1="${pad}" y1="${y0}" x2="${W - pad}" y2="${y0}" stroke="#34343a" stroke-width="1.5"/>` + ticks + `</svg>`
      + `<div class="ir-axis"><span>00:00</span><span>${esc(hms(dur))}</span></div></div>`;
  }

  // interruptions-over-time histogram (adaptive bin width so short conversations still get bars)
  function perMinute(events, dur) {
    if (!dur || !events.length) return "";
    const bins = Math.min(24, Math.max(8, Math.ceil(dur / 5)));
    const o = new Array(bins).fill(0), r = new Array(bins).fill(0);
    events.forEach((e) => { const b = Math.min(bins - 1, Math.floor(((e.t_offset || 0) / dur) * bins)); (e.cat === "overlap" ? o : r)[b]++; });
    const max = Math.max(1, ...o.map((v, i) => v + r[i]));
    const W = 1000, H = 90, pad = 8, bw = (W - pad * 2) / bins, gap = Math.min(6, bw * 0.2);
    let bars = "";
    for (let i = 0; i < bins; i++) {
      const x = pad + i * bw + gap / 2, w = bw - gap;
      const ho = (o[i] / max) * (H - 16), hr = (r[i] / max) * (H - 16);
      bars += `<rect x="${x}" y="${H - ho}" width="${w}" height="${ho}" fill="#c47a7a"/>`;
      bars += `<rect x="${x}" y="${H - ho - hr}" width="${w}" height="${hr}" fill="#c4a86a"/>`;
    }
    return `<div class="ir-bars"><div class="conv-cap">Interruptions over time <span class="lg"><i class="sw o"></i>overlap <i class="sw r"></i>rapid switch</span></div>`
      + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="ir-bars-svg" role="img" aria-label="Interruptions over time">${bars}</svg></div>`;
  }

  function eventList(events) {
    if (!events.length) return "";
    const rows = events.map((e) => {
      const cat = e.cat === "overlap" ? "Overlap" : "Rapid switch";
      const desc = e.cat === "overlap"
        ? `${e.by || "someone"} ${e.note || "over"} ${e.of || "the previous speaker"}`
        : `${e.gapSec != null ? e.gapSec + "s gap" : "fast handoff"}${e.from ? ` after ${e.from}` : ""}`;
      const seekAttr = typeof e.t_offset === "number" ? ` data-seek="${e.t_offset}" tabindex="0" role="button"` : "";
      return `<li class="ir-ev ${e.cat}"${seekAttr}><span class="ir-t">${esc(e.t_hms || "")}</span>`
        + `<span class="ir-badge">${cat}</span><span class="ir-desc">${esc(desc)}</span></li>`;
    }).join("");
    return `<ul class="ir-list">${rows}</ul>`;
  }

  function hms(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const p = (n) => String(n).padStart(2, "0");
    return `${p(Math.floor(sec / 60))}:${p(sec % 60)}`;
  }

  // click/Enter on any [data-seek] element seeks the audio player
  function wireSeeks(root) {
    root.querySelectorAll("[data-seek]").forEach((el) => {
      const sec = parseFloat(el.getAttribute("data-seek"));
      el.addEventListener("click", () => seek(sec));
      el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); seek(sec); } });
    });
  }

  const api = {
    getDoc: null,
    setTop, setMode, refresh,
    // called by llm-backend.js once a model is detected, so the Sharpen button appears on an open mode
    llmReady() { lastSig = ""; applyMode(); },
    init() {
      document.querySelectorAll("#modeTabs .mode-tab").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    },
  };
  window.VOX_CONV = api;
})();
