"use strict";
// On-device LLM analyzer backend (mobile only). When the native `voxllm` plugin reports a model is
// loaded, this REPLACES window.VOX_ANALYZE (the offline heuristic from analyze.js) with one that runs
// a small local LLM to produce the conversation graph + interruptions — real topic/retort/counter
// structure and speaker-inferred interruptions that the single-speaker heuristic can't.
//
// Division of labour (so the Kotlin plugin stays a generic, reusable LLM runner):
//   Kotlin/native  — "given this prompt, generate text" (start_generate / poll_generate), nothing
//                    domain-specific. Also exposes llm_available.
//   JS (this file) — prompt engineering, JSON extraction/repair, and mapping the model's compact JSON
//                    into the exact {graph, interruptions} contract analyze.js defines. On ANY failure
//                    it falls back to the heuristic, so a flaky model never breaks the modes.
//
// Loaded after analyze.js (so the heuristic exists to wrap) and after the backend script (so
// window.VOX_ONDEVICE is set). No-ops entirely off-device.

(function () {
  if (!window.VOX_ONDEVICE || !(window.__TAURI__ && window.__TAURI__.core)) return;
  const invoke = (cmd, args) => window.__TAURI__.core.invoke(cmd, args);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const CFG = {
    // The bundled Qwen2.5-0.5B .task has an effective KV cache of ~1280 tokens (prompt + output). The
    // schema/instructions are ~320 tok and we reserve output, so ONE chunk's transcript body is capped
    // at ~2000 chars (~600 tok). A long conversation is split into overlapping chunks and the per-chunk
    // results merged — so the model sees the WHOLE thing, not just the last minute.
    chunkChars: 2000,      // transcript body per chunk
    overlapChars: 350,     // trailing context re-included in the next chunk (so a straddling event is seen whole)
    maxChunks: 16,         // hard cap; anything beyond is logged + dropped (never silently truncated)
    dedupSec: 3,           // two interruptions within this many seconds + same type = the same event
    maxTokens: 512,        // advisory generation cap (the native side fixes the model's total window)
    temperature: 0.2,      // near-greedy: we want structure, not creativity
    topK: 40,
    pollMs: 500,
    timeoutMs: 180000,     // first call also loads ~550 MB of weights — allow generous headroom
  };

  // ---- transcript -> chunks --------------------------------------------------
  // One rendered line per turn, carrying its real start time so chunks know their span and the model
  // sees absolute [hh:mm:ss] stamps (not chunk-relative ones).
  function renderLines(turns) {
    return turns.map((t) => ({
      text: `${t.t_offset_hms ? `[${t.t_offset_hms}] ` : ""}${t.speaker ? `${t.speaker}: ` : ""}${(t.text || "").replace(/\s+/g, " ").trim()}`,
      t: typeof t.t_offset === "number" ? t.t_offset : null,
    }));
  }

  // Greedily pack whole lines into ~chunkChars chunks, backing up ~overlapChars between chunks so an
  // interruption near a boundary lands whole in at least one chunk. Always makes forward progress.
  function chunkLines(turns) {
    const lines = renderLines(turns);
    const chunks = [];
    let i = 0;
    while (i < lines.length && chunks.length < CFG.maxChunks) {
      let body = "", j = i;
      while (j < lines.length && (!body || body.length + lines[j].text.length + 1 <= CFG.chunkChars)) {
        body += (body ? "\n" : "") + lines[j].text; j++;
      }
      chunks.push({ body, index: chunks.length });
      if (j >= lines.length) { i = j; break; }
      let back = j, ov = 0;                                  // re-include trailing context in the next chunk
      while (back > i + 1 && ov < CFG.overlapChars) { back--; ov += lines[back].text.length + 1; }
      i = Math.max(i + 1, back);
    }
    if (i < lines.length) console.warn(`voxllm: transcript exceeded ${CFG.maxChunks} chunks — analyzed the first ${i}/${lines.length} turns`);
    return chunks;
  }

  // Compact LINE grammar — a 0.5-1.5B model keeps a flat one-item-per-line format FAR more reliably
  // than nested JSON (and a truncated last line is simply dropped, where a truncated JSON object loses
  // the whole chunk). parseModelLines() turns it back into the same {topics,points} shape toContract()
  // consumes; JSON is still accepted as a fallback if the model emits it anyway.
  function buildPrompt(body, ctx) {
    const seg = (ctx && ctx.total > 1)
      ? `This is segment ${ctx.index + 1} of ${ctx.total} of a longer conversation; report only what is visible in THIS segment.\n`
      : "";
    return [
      "You analyze a conversation transcript. It may carry a single ASR speaker label even when several",
      "people talk — INFER distinct speakers from content and turn-taking.",
      "",
      "Output ONE item per line and NOTHING else (no prose, no JSON, no markdown). Use exactly these forms:",
      "T: <short topic title, up to 6 words>",
      "P: <speaker> | <claim|question|statement|retort|counter> | <mm:ss> | <up to 12 word paraphrase>",
      "I: <overlap|rapid> | <mm:ss> | <who cut in> | <who was cut off> | <up to 8 word note>",
      "",
      "Put each P line under the T line it belongs to. retort = pushes back on the previous point;",
      "counter = a direct counter-argument; question = a question. overlap = spoke over someone; rapid =",
      "a fast back-and-forth handoff. Use the [hh:mm:ss] stamps shown. Do not invent content. Lines only.",
      "",
      seg + 'Transcript:\n"""\n' + body + '\n"""',
    ].join("\n");
  }

  // A focused, single-shot insight about ONE topic — plain text, one short generation, far cheaper than
  // the multi-chunk Sharpen. Backs the graph's tap-to-insight (💡). Graph node labels now carry the FULL
  // utterance text, so each point is clipped HERE — the model's total window is ~1280 tokens and 12
  // unclipped turns would blow straight through it.
  function insightPrompt(payload) {
    const clip = (s, n) => { s = String(s || "").trim(); return s.length > n ? s.slice(0, n - 1) + "…" : s; };
    const pts = (payload.points || []).slice(0, 12)
      .map((p) => `- ${p.speaker ? p.speaker + ": " : ""}${clip(p.text, 160)}`).join("\n");
    return [
      "You are a sharp conversation analyst. In 1-2 sentences, explain what this part of the conversation",
      "is about and why it matters — be concrete and specific to the content, do not restate it verbatim,",
      "and do not add facts that aren't here. You may end with one short follow-up question.",
      payload.focus ? `Pay particular attention to this point: "${clip(payload.focus, 200)}".` : "",
      "",
      `Topic: ${payload.label || "(topic)"}`,
      `Points:\n${pts || "(none)"}`,
      "",
      "Insight:",
    ].filter((x) => x !== "").join("\n");
  }

  // ---- run the native LLM ----------------------------------------------------
  let gen = 0;   // bumped per request so a stale poll loop aborts when a newer analyze starts
  async function generate(prompt, opts) {
    const my = ++gen;
    await invoke("plugin:voxllm|start_generate", {
      prompt,
      maxTokens: (opts && opts.maxTokens) || CFG.maxTokens,
      temperature: (opts && opts.temperature != null) ? opts.temperature : CFG.temperature,
      topK: CFG.topK,
    });
    const t0 = Date.now();
    for (;;) {
      await sleep(CFG.pollMs);
      if (my !== gen) throw new Error("superseded");
      let st;
      try { st = await invoke("plugin:voxllm|poll_generate"); } catch (e) { continue; }
      if (st.phase === "done") return st.text || "";
      if (st.phase === "error") throw new Error(st.error || "llm generation failed");
      if (Date.now() - t0 > CFG.timeoutMs) throw new Error("llm timeout");
    }
  }
  // Abort an in-flight generation (a multi-chunk run can take 30–60 s; the user can leave the mode or
  // start a fresh Sharpen). Bumping `gen` makes the poll loop above throw; cancel_generate stops the
  // native decode so the next chunk/run starts clean.
  function cancelGenerate() {
    gen++;
    invoke("plugin:voxllm|cancel_generate").catch(() => {});
  }

  // ---- model output -> { topics, interruptions } -----------------------------
  const djb2 = (s) => { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0; return h; };

  // Parse the line grammar (what we now prompt for). Tolerant: unknown lines are ignored and a
  // truncated final line just contributes nothing — so a cut-off generation still yields everything
  // before the cut, instead of losing the whole chunk to a JSON parse error.
  function parseModelLines(text) {
    if (!text) return null;
    const topics = [], interruptions = [];
    let cur = null;
    for (const raw of String(text).split(/\r?\n/)) {
      const m = /^\s*([TPI])\s*[:|\-]\s*(.+?)\s*$/.exec(raw);
      if (!m) continue;
      const tag = m[1].toUpperCase(), f = m[2].split("|").map((s) => s.trim());
      if (tag === "T") { cur = { title: (f[0] || "").slice(0, 60), points: [] }; topics.push(cur); }
      else if (tag === "P") {
        if (!cur) { cur = { title: "(topic)", points: [] }; topics.push(cur); }
        cur.points.push({ speaker: f[0] || "", kind: (f[1] || "").toLowerCase(), t: f[2] || "", text: f.slice(3).join(" | ").trim() });
      } else if (tag === "I") {
        interruptions.push({ type: (f[0] || "").toLowerCase(), t: f[1] || "", by: f[2] || "", of: f[3] || "", note: f.slice(4).join(" | ").trim() });
      }
    }
    return (topics.length || interruptions.length) ? { topics, interruptions } : null;
  }

  // JSON fallback (the model may still emit JSON). Tries a clean parse, then a truncation repair that
  // closes unclosed strings/brackets and drops a dangling final element — salvaging the completed prefix.
  function parseModelJson(text) {
    if (!text) return null;
    let s = text.replace(/```json/gi, "").replace(/```/g, "").trim();
    const a = s.indexOf("{");
    if (a < 0) return null;
    s = s.slice(a);
    const b = s.lastIndexOf("}");
    if (b > 0) { try { return JSON.parse(s.slice(0, b + 1).replace(/,\s*([}\]])/g, "$1")); } catch (_) { /* fall through to repair */ } }
    return repairTruncatedJson(s);
  }
  function repairTruncatedJson(s) {
    let inStr = false, esc = false, cut = -1; const stk = [];
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true;
      else if (c === "{" || c === "[") stk.push(c === "{" ? "}" : "]");
      else if (c === "}" || c === "]") { stk.pop(); cut = i; }
      else if (c === "," && stk.length) cut = i - 1;   // last completed element inside a container
    }
    if (cut < 1) return null;
    let body = s.slice(0, cut + 1).replace(/,\s*$/, "");
    inStr = false; esc = false; const st = [];
    for (let i = 0; i < body.length; i++) {
      const c = body[i];
      if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; continue; }
      if (c === '"') inStr = true;
      else if (c === "{" || c === "[") st.push(c === "{" ? "}" : "]");
      else if (c === "}" || c === "]") st.pop();
    }
    if (inStr) body += '"';
    while (st.length) body += st.pop();
    try { return JSON.parse(body.replace(/,\s*([}\]])/g, "$1")); } catch (_) { return null; }
  }

  // One entry point: line grammar first, JSON (with repair) as a fallback.
  function parseModel(text) { return parseModelLines(text) || parseModelJson(text); }

  // ---- compact JSON -> {graph, interruptions} contract -----------------------
  const hmsToSec = (t) => {
    const p = String(t || "").trim().split(":").map(Number);
    if (!p.length || p.some(Number.isNaN)) return undefined;
    return p.reduce((acc, n) => acc * 60 + n, 0);
  };
  const hms = (sec) => {
    sec = Math.max(0, Math.round(sec || 0));
    const p = (n) => String(n).padStart(2, "0");
    return `${p(Math.floor(sec / 3600))}:${p(Math.floor((sec % 3600) / 60))}:${p(sec % 60)}`;
  };
  const KIND = { claim: "statement", statement: "statement", question: "question", retort: "retort", counter: "counter" };

  function toContract(parsed, doc) {
    const turns = doc.turns || [];
    const durationSec = turns.reduce((m, t) => Math.max(m, typeof t.t_offset === "number" ? t.t_offset : 0), 0);
    const speakersSeen = new Set();

    const nodes = [{ id: "root", type: "root", label: "Conversation" }];
    const edges = [];
    const topics = Array.isArray(parsed.topics) ? parsed.topics : [];
    topics.forEach((tp, ti) => {
      const tid = `t${ti}`;
      nodes.push({ id: tid, type: "topic", label: String(tp.title || "(topic)").slice(0, 48) });
      edges.push({ from: "root", to: tid, kind: "topic" });
      let prevId = null, prevSpk = null;
      (Array.isArray(tp.points) ? tp.points : []).forEach((p, pi) => {
        const uid = `u${ti}_${pi}`;
        const type = KIND[String(p.kind || "").toLowerCase()] || "statement";
        const sec = hmsToSec(p.t);
        if (p.speaker) speakersSeen.add(String(p.speaker));
        nodes.push({
          id: uid, type, label: String(p.text || "").slice(0, 120),
          t_offset: sec, speaker: p.speaker || null,
        });
        edges.push({ from: tid, to: uid, kind: "contains" });
        if (prevId) {
          const rebut = type === "retort" || type === "counter" || (p.speaker && prevSpk && p.speaker !== prevSpk);
          edges.push({ from: prevId, to: uid, kind: rebut ? "rebuts" : "reply" });
        }
        prevId = uid; prevSpk = p.speaker || null;
      });
    });

    const overlap = [], rapidSwitch = [];
    (Array.isArray(parsed.interruptions) ? parsed.interruptions : []).forEach((e) => {
      const sec = hmsToSec(e.t);
      const rec = { t_offset: sec, t_hms: sec != null ? hms(sec) : String(e.t || ""), by: e.by, of: e.of, from: e.of, note: e.note };
      if (String(e.type || "").toLowerCase() === "rapid") rapidSwitch.push(rec); else overlap.push(rec);
    });

    return {
      graph: { nodes, edges, topicCount: topics.length },
      interruptions: {
        overlap, rapidSwitch,
        overlapCount: overlap.length, rapidCount: rapidSwitch.length, total: overlap.length + rapidSwitch.length,
        durationSec, multiSpeaker: speakersSeen.size > 1,
      },
    };
  }

  // ---- merge per-chunk model output ------------------------------------------
  const norm = (s) => String(s || "").toLowerCase().replace(/\s+/g, " ").trim();
  // Combine the parsed JSON from every chunk into ONE parsed-shaped object, then run it through the
  // single toContract() below — so all the node/edge id logic stays in one place and chunk-local ids
  // can never collide. Topics dedup by title (points unioned); interruptions dedup by time+type.
  function mergeParsed(list) {
    const topics = [], byTitle = new Map();
    for (const p of list) {
      for (const tp of (Array.isArray(p.topics) ? p.topics : [])) {
        const key = norm(tp.title), pts = Array.isArray(tp.points) ? tp.points : [];
        if (key && byTitle.has(key)) byTitle.get(key).points.push(...pts);
        else { const nt = { title: tp.title, points: pts.slice() }; if (key) byTitle.set(key, nt); topics.push(nt); }
      }
    }
    const kept = [];
    for (const p of list) {
      for (const e of (Array.isArray(p.interruptions) ? p.interruptions : [])) {
        const sec = hmsToSec(e.t), type = norm(e.type) === "rapid" ? "rapid" : "overlap";
        const score = (e.by ? 1 : 0) + (e.of ? 1 : 0) + String(e.note || "").length;
        const dup = kept.find((x) => x._type === type && (
          (x._sec != null && sec != null && Math.abs(x._sec - sec) <= CFG.dedupSec) ||
          (x._sec == null && sec == null && norm(x.by) === norm(e.by) && norm(x.of) === norm(e.of))));
        if (dup) { if (score > dup._score) Object.assign(dup, e, { _type: type, _sec: sec, _score: score }); }
        else kept.push(Object.assign({}, e, { _type: type, _sec: sec, _score: score }));
      }
    }
    const interruptions = kept.map(({ _type, _sec, _score, ...e }) => e);
    return { topics, interruptions };
  }

  // Run the whole transcript through the model in overlapping chunks, merging as each finishes. Reports
  // progress (and an incremental merged result) via onProgress so the UI can paint as it goes, and
  // honors token.canceled so a superseded run stops promptly. Throws only if NOTHING parsed.
  const chunkCache = new Map();   // content-addressed (djb2 of the chunk body) so re-Sharpen after a
                                  // small transcript edit reuses every unchanged chunk's parse.
  async function analyzeChunked(doc, onProgress, token) {
    const turns = (doc && doc.turns) || [];
    if (!turns.length) throw new Error("no transcript");
    const chunks = chunkLines(turns);
    const parsedList = [];
    for (const ch of chunks) {
      if (token && token.canceled) throw new Error("canceled");
      if (onProgress) onProgress({ phase: "chunk", done: ch.index, total: chunks.length });
      const ck = djb2(ch.body);
      let parsed = chunkCache.get(ck) || null;
      if (!parsed) {
        try { parsed = parseModel(await generate(buildPrompt(ch.body, { index: ch.index, total: chunks.length }))); }
        catch (e) {
          const m = String((e && e.message) || e);
          if (m === "canceled" || m === "superseded") throw e;   // abort the whole run
          /* otherwise: a junk/failed chunk — skip it, keep going */
        }
        if (parsed && (Array.isArray(parsed.topics) || Array.isArray(parsed.interruptions))) {
          if (chunkCache.size > 128) chunkCache.clear();
          chunkCache.set(ck, parsed);
        }
      }
      if (parsed && (Array.isArray(parsed.topics) || Array.isArray(parsed.interruptions))) {
        parsedList.push(parsed);
        if (onProgress) onProgress({ phase: "partial", done: ch.index + 1, total: chunks.length, data: toContract(mergeParsed(parsedList), doc) });
      }
    }
    if (!parsedList.length) throw new Error("model returned nothing usable");
    return toContract(mergeParsed(parsedList), doc);
  }

  // ---- expose the on-device LLM analyzer (OPT-IN — the UI runs it on demand) --
  // Crucially this does NOT replace window.VOX_ANALYZE (the heuristic). The heuristic keeps rendering
  // the modes instantly and live; the UI calls window.VOX_LLM.analyze() only when the user taps
  // "Sharpen". Running the ~0.5 GB model on every render could OOM-crash the app — a native crash JS
  // cannot catch — so it must be an explicit, one-shot action, never automatic.
  function installLlm(model) {
    window.VOX_LLM = {
      available: true,
      model: model || "on-device LLM",
      // Throws on failure; the caller keeps the heuristic that's already on screen and shows a notice.
      // onProgress (optional): {phase:"chunk"|"partial", done, total, data?}. token (optional):
      // {canceled} — set canceled=true to stop a long multi-chunk run.
      analyze(doc, onProgress, token) { return analyzeChunked(doc, onProgress, token); },
      // Focused single-shot insight about one topic/leaf (plain text). Throws on failure; the graph's
      // 💡 handler shows a short error and the rest of the view is unaffected. Reuses the same generate()
      // + gen guard, so a Sharpen starting mid-insight cleanly supersedes it.
      async insight(payload) {
        const txt = await generate(insightPrompt(payload || {}), { maxTokens: 160 });
        return String(txt || "").replace(/^\s*insight:\s*/i, "").trim();
      },
      cancel: cancelGenerate,
    };
    if (window.VOX_CONV && window.VOX_CONV.llmReady) window.VOX_CONV.llmReady();   // surface the Sharpen button
  }

  // Probe the plugin once at startup; advertise the LLM only if a model actually bundled.
  (async () => {
    try {
      const s = await invoke("plugin:voxllm|llm_available");
      if (s && s.available) installLlm(s.model);
    } catch (_) { /* plugin absent / model missing → heuristic only */ }
  })();
})();
