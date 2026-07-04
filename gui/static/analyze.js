"use strict";
// Conversation analysis seam. Exposes window.VOX_ANALYZE.analyze(doc) -> Promise<{graph, interruptions}>.
//
// This is the ONE place that turns a transcript doc into the structures the Graph and Interruptions
// modes render. The default implementation is a fully-offline HEURISTIC (no model, no network): it
// segments turns into topics by keyword overlap and flags interruptions from speaker/timing/text
// cues. It is deliberately swappable — an on-device LLM backend can replace window.VOX_ANALYZE with
// an async analyzer that returns the SAME shape, and neither the views nor app.js change.
//
// Output contract (stable — the LLM backend must match it):
//   graph: {
//     nodes: [{ id, type:'root'|'topic'|'statement'|'question'|'retort'|'counter', label,
//               turnIdx?, speaker_id?, t_offset? }],
//     edges: [{ from, to, kind:'topic'|'contains'|'reply'|'rebuts'|'supports' }],
//   }
//   interruptions: {
//     overlap:     [{ turnIdx, t_offset, t_hms, by, of, note }],   // someone spoke over someone
//     rapidSwitch: [{ turnIdx, t_offset, t_hms, gapSec, by, from }],// fast back-and-forth handoff
//     overlapCount, rapidCount, total,
//     durationSec, multiSpeaker,            // context the view uses to caption honestly
//   }
//
// Notes on the heuristic's honesty limits (surfaced in the UI, not hidden):
//  - On-device transcription is single-speaker, so "overlap" can only be inferred from text cut-off
//    cues (trailing dash / unfinished line) — weak. With a real diarized (desktop) doc it uses true
//    speaker changes + timing + any overlap markers, which is much stronger.
//  - Topic/retort/counter typing is shallow without an LLM; the graph is a readable topic tree, and
//    the on-device LLM (phase 2) fills in real argument structure through this same seam.

(function () {
  // ---- tuning ----------------------------------------------------------------
  const CFG = {
    topicOverlap: 0.18,     // min keyword overlap (vs the topic's recent window) to stay on-topic
    topicGapSec: 18,        // a silence longer than this is treated as a topic boundary
    topicMinTurns: 2,       // don't break a topic that's still only this many turns (unless big gap)
    topicLabelWords: 4,     // keywords shown in a topic label
    recentWindow: 3,        // compare a turn against the topic's last N turns, not its whole history
    topicLowStreak: 2,      // hysteresis: this many consecutive off-topic turns before breaking
    topicMinKeywords: 2,    // turns with fewer real keywords (backchannels) don't drive boundaries
    rapidGapSec: 1.2,       // consecutive turns closer than this = rapid handoff (multi-speaker only)
    overlapSlackSec: 0.25,  // next turn starts this much before prev's estimated end = overlap
    wordsPerSec: 2.6,       // rough speaking rate to estimate a turn's end from its start + length
    minEndSec: 0.8,         // floor on an estimated turn duration
  };

  const STOP = new Set(("a an and the of to in on at for is are was were be been being it its this that these " +
    "those i you he she we they them him her his her my your our their me us do does did so but or if then than " +
    "as with about into out up down over under just like really very kind sort gonna wanna yeah yep nope okay ok " +
    "um uh er hmm right well now here there what when where who why how which not no yes can could would should " +
    "will shall may might must have has had get got go going one two also too get").split(" "));

  const words = (s) => String(s || "").toLowerCase().match(/[a-z0-9']+/g) || [];
  const keywords = (s) => words(s).filter((w) => w.length >= 4 && !STOP.has(w));
  const hms = (sec) => {
    sec = Math.max(0, Math.round(sec || 0));
    const p = (n) => String(n).padStart(2, "0");
    return `${p(Math.floor(sec / 3600))}:${p(Math.floor((sec % 3600) / 60))}:${p(sec % 60)}`;
  };
  // ASR usually drops the trailing "?", so treat a short wh-initial turn as a question too (a long
  // wh-initial turn — "what we did then was…" — is usually a statement, so gate on length).
  const WH = /^(what|why|how|when|where|who|which|whose|whom|do|does|did|is|are|can|could|would|should|will)\b/i;
  const isQuestion = (t) => { t = String(t || "").trim(); return /\?\s*$/.test(t) || (WH.test(t) && words(t).length <= 10); };
  // Cheap argument-cue lexicons: agreement → a "supports" tie; disagreement/contrast → a "rebuts" tie
  // (and counter/retort typing). Word-boundary anchored so they don't fire inside other words.
  const AGREE = /\b(exactly|agreed|i agree|agree|true|totally|absolutely|good point|makes sense|fair enough|for sure|right on|same here)\b/i;
  const DISAGREE = /\b(but|however|actually|no,|nah|disagree|wrong|not really|i don't think|that's not|on the contrary|except|whereas|instead)\b/i;
  // A trailing em/en dash or double-hyphen — the conventional ASR signature of a line cut off
  // mid-word. Kept STRICT on purpose: a missing full stop is normal for Whisper segments, so we do
  // NOT treat unpunctuated lines as cut off (that flagged ordinary continuous speech as overlap).
  // This is the only overlap signal available when the doc is single-speaker.
  const looksCutOff = (t) => /[—–]\s*$|--\s*$/.test((t || "").trim());

  function topicLabel(kwCounts) {
    const top = Object.entries(kwCounts).sort((a, b) => b[1] - a[1]).slice(0, CFG.topicLabelWords).map((x) => x[0]);
    return top.length ? top.join(" · ") : "(topic)";
  }

  // Inverse document frequency over the doc's turns, so DISTINCTIVE words drive topic boundaries and
  // words sprinkled through the whole conversation barely count. Smoothed so it never goes negative.
  function idfWeights(turns) {
    const df = {}, N = Math.max(1, turns.length);
    turns.forEach((t) => new Set(keywords(t.text)).forEach((w) => { df[w] = (df[w] || 0) + 1; }));
    const idf = {};
    for (const w in df) idf[w] = Math.max(0.05, Math.log((N + 1) / (df[w] + 0.5)));
    return idf;
  }
  const wsum = (set, idf) => { let s = 0; set.forEach((w) => (s += idf[w] || 1)); return s; };

  // Sequential topic segmentation. A turn stays in the current topic while its keywords overlap the
  // topic's RECENT window — measured as an IDF-weighted overlap coefficient, so a distinctive shared
  // word holds a topic together where common filler wouldn't. A topic breaks only on a STRONG, SUSTAINED
  // shift (hysteresis: two off-topic turns in a row, so one stray line doesn't fragment it) once it has
  // a couple of turns, OR immediately on a long silence. Backchannels ("yeah", "okay") are neutral.
  function segmentTopics(turns) {
    const idf = idfWeights(turns);
    const topics = [];
    let cur = null, prevT = null, lowStreak = 0;
    turns.forEach((t, i) => {
      const kw = keywords(t.text), kwSet = new Set(kw);
      const gap = (prevT != null && typeof t.t_offset === "number") ? t.t_offset - prevT : null;
      const bigGap = gap != null && gap > CFG.topicGapSec;
      let start = !cur;
      if (cur) {
        if (bigGap) { start = true; lowStreak = 0; }
        else if (kw.length >= CFG.topicMinKeywords) {
          const shared = new Set(kw.filter((w) => cur.recentSet.has(w)));
          const denom = Math.min(wsum(kwSet, idf), wsum(cur.recentSet, idf)) || 1;
          const ov = wsum(shared, idf) / denom;
          lowStreak = ov < CFG.topicOverlap ? lowStreak + 1 : 0;
          if (lowStreak >= CFG.topicLowStreak && cur.turnIdxs.length >= CFG.topicMinTurns) { start = true; lowStreak = 0; }
        }
      }
      if (start) { cur = { turnIdxs: [], kwCounts: {}, recentKw: [], recentSet: new Set() }; topics.push(cur); }
      cur.turnIdxs.push(i);
      kw.forEach((w) => { cur.kwCounts[w] = (cur.kwCounts[w] || 0) + 1; });
      cur.recentKw.push(kw);
      if (cur.recentKw.length > CFG.recentWindow) cur.recentKw.shift();
      cur.recentSet = new Set(cur.recentKw.flat());
      if (typeof t.t_offset === "number") prevT = t.t_offset;
    });
    return mergeSingletons(topics, idf);
  }

  // Fold a lone-turn topic into whichever neighbor its keywords overlap more (IDF-weighted) — a single
  // distinctive utterance reading as its own "topic" looks worse than a slightly broader one.
  function mergeSingletons(topics, idf) {
    const setOf = (tp) => new Set(Object.keys(tp.kwCounts));
    const sim = (aSet, b) => { if (!b) return -1; const o = setOf(b); let s = 0; aSet.forEach((w) => { if (o.has(w)) s += idf[w] || 1; }); return s; };
    let i = 0;
    while (i < topics.length) {
      if (topics.length > 1 && topics[i].turnIdxs.length === 1) {
        const me = setOf(topics[i]);
        const left = sim(me, topics[i - 1]), right = sim(me, topics[i + 1]);
        const j = right > left ? i + 1 : (i > 0 ? i - 1 : i + 1);
        if (j >= 0 && j < topics.length && j !== i) {
          const dst = topics[j], src = topics[i];
          dst.turnIdxs = dst.turnIdxs.concat(src.turnIdxs).sort((x, y) => x - y);
          for (const w in src.kwCounts) dst.kwCounts[w] = (dst.kwCounts[w] || 0) + src.kwCounts[w];
          topics.splice(i, 1);
          i = j < i ? j : i;   // re-examine the topic we merged into when it sat before us
          continue;
        }
      }
      i++;
    }
    return topics;
  }

  function buildGraph(turns) {
    const nodes = [{ id: "root", type: "root", label: "Conversation" }];
    const edges = [];
    const topics = segmentTopics(turns);

    topics.forEach((tp, ti) => {
      const tid = `t${ti}`;
      nodes.push({ id: tid, type: "topic", label: topicLabel(tp.kwCounts), turnIdx: tp.turnIdxs[0] });
      edges.push({ from: "root", to: tid, kind: "topic" });

      let prevUtterId = null, prevSid = null;
      tp.turnIdxs.forEach((idx) => {
        const t = turns[idx];
        const uid = `u${idx}`;
        // Type heuristic: a question is a question. Then argument cues — a disagreement cue makes a
        // counter (different speaker) or a self-retort (same speaker pivoting); a different-speaker
        // reply with no cue is a retort; an agreement cue turns the tie into a "supports" edge.
        let type = isQuestion(t.text) ? "question" : "statement";
        const changed = prevSid != null && t.speaker_id !== prevSid;
        const agree = AGREE.test(t.text), disagree = DISAGREE.test(t.text);
        let edgeKind = changed ? "rebuts" : "reply";
        if (type === "statement") {
          if (disagree) { type = changed ? "counter" : "retort"; edgeKind = "rebuts"; }
          else if (changed) { type = "retort"; }
        }
        if (agree && !disagree && prevUtterId) edgeKind = "supports";
        nodes.push({
          id: uid, type, turnIdx: idx, speaker_id: t.speaker_id, t_offset: t.t_offset,
          label: fullText(t.text),
        });
        edges.push({ from: tid, to: uid, kind: "contains" });
        if (prevUtterId) edges.push({ from: prevUtterId, to: uid, kind: edgeKind });
        prevUtterId = uid; prevSid = t.speaker_id;
      });
    });
    return { nodes, edges, topicCount: topics.length };
  }

  // Cards show the WHOLE utterance (whitespace-normalized) — truncating it hid what was actually
  // said, which defeats the point of a topic map you read instead of the transcript.
  const fullText = (s) => String(s || "").replace(/\s+/g, " ").trim();

  function detectInterruptions(turns, durationSec, multiSpeaker) {
    const overlap = [], rapidSwitch = [];
    // The turn "holding the floor" so far: the prior turn with the LATEST end. Comparing only against
    // the immediately-previous turn missed real overlaps — with diarized (possibly overlapping)
    // segments, an interjection can start while a turn from two entries back is still going. The
    // cross-turn floor is built ONLY from the diarizer's real t_end: carrying word-count ESTIMATES
    // forward would let one over-estimated turn (long text, fast speech) flag every speaker change in
    // its inflated window as an overlap. Estimate-only docs keep the old previous-turn-only check.
    let floor = null;   // { end, speaker, speaker_id } — latest REAL t_end among prior turns
    for (let i = 1; i < turns.length; i++) {
      const prev = turns[i - 1], cur = turns[i];
      if (typeof prev.t_end === "number" && (!floor || prev.t_end >= floor.end)) {
        floor = { end: prev.t_end, speaker: prev.speaker, speaker_id: prev.speaker_id };
      }
      // Fallback when prev has no real end (and outlasts any real floor): estimate from word count.
      const ps = typeof prev.t_offset === "number" ? prev.t_offset : null;
      const prevEst = (typeof prev.t_end !== "number" && ps != null)
        ? ps + Math.max(CFG.minEndSec, words(prev.text).length / CFG.wordsPerSec) : null;
      const holder = (prevEst != null && (!floor || prevEst > floor.end))
        ? { end: prevEst, speaker: prev.speaker, speaker_id: prev.speaker_id }
        : floor;
      const cs = typeof cur.t_offset === "number" ? cur.t_offset : null;
      const changed = multiSpeaker && holder && cur.speaker_id !== holder.speaker_id;
      // Prefer the diarizer's silence gap (gap_before) when present; fall back to start-to-start.
      const gap = typeof cur.gap_before === "number" ? cur.gap_before
                : (ps != null && cs != null) ? cs - ps : null;

      // OVERLAP — someone began before the floor-holding turn plausibly finished.
      if (cs != null && holder) {
        const overlapped = cs < holder.end - CFG.overlapSlackSec;
        const marked = /\b(overlap|crosstalk)\b/i.test((prev.markers || []).concat(cur.markers || []).join(" "));
        if (marked || (overlapped && (changed || (!multiSpeaker && looksCutOff(prev.text))))) {
          overlap.push({
            turnIdx: i, t_offset: cs, t_hms: cur.t_offset_hms || hms(cs),
            by: cur.speaker, of: changed ? holder.speaker : prev.speaker,
            note: marked ? "marked overlap" : (changed ? "spoke over" : "cut off mid-line"),
          });
          continue;   // an overlap already implies a fast handoff; don't double-count as rapid
        }
      }

      // RAPID SWITCH — a quick handoff: a speaker change within the gap window. Requires real
      // speakers, so this stays empty for single-speaker on-device docs (a speaker-inferring LLM
      // backend, or a diarized recording, is what populates it there).
      if (multiSpeaker && cur.speaker_id !== prev.speaker_id && gap != null && gap >= 0 && gap < CFG.rapidGapSec) {
        rapidSwitch.push({
          turnIdx: i, t_offset: cs, t_hms: cur.t_offset_hms || hms(cs),
          gapSec: Math.round(gap * 100) / 100, by: cur.speaker, from: prev.speaker,
        });
      }
    }
    return {
      overlap, rapidSwitch,
      overlapCount: overlap.length, rapidCount: rapidSwitch.length,
      total: overlap.length + rapidSwitch.length,
      durationSec: durationSec || 0, multiSpeaker,
    };
  }

  function heuristicAnalyze(doc) {
    const turns = (doc && doc.turns) || [];
    const speakers = (doc && doc.speakers) || [];
    const multiSpeaker = new Set(turns.map((t) => t.speaker_id)).size > 1 || speakers.length > 1;
    const durationSec = turns.reduce((m, t) => Math.max(m,
      typeof t.t_end === "number" ? t.t_end : typeof t.t_offset === "number" ? t.t_offset : 0), 0);
    return {
      graph: buildGraph(turns),
      interruptions: detectInterruptions(turns, durationSec, multiSpeaker),
    };
  }

  // Public seam. analyze() is async so an LLM backend can await native inference; the heuristic
  // resolves immediately. A backend swaps this whole object (keeping the method + shape).
  window.VOX_ANALYZE = window.VOX_ANALYZE || {
    kind: "heuristic",
    async analyze(doc) { return heuristicAnalyze(doc); },
  };
})();
