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
  const isQuestion = (t) => /\?\s*$/.test(t || "");
  // A trailing em/en dash or double-hyphen — the conventional ASR signature of a line cut off
  // mid-word. Kept STRICT on purpose: a missing full stop is normal for Whisper segments, so we do
  // NOT treat unpunctuated lines as cut off (that flagged ordinary continuous speech as overlap).
  // This is the only overlap signal available when the doc is single-speaker.
  const looksCutOff = (t) => /[—–]\s*$|--\s*$/.test((t || "").trim());

  function topicLabel(kwCounts) {
    const top = Object.entries(kwCounts).sort((a, b) => b[1] - a[1]).slice(0, CFG.topicLabelWords).map((x) => x[0]);
    return top.length ? top.join(" · ") : "(topic)";
  }

  // Sequential topic segmentation. A turn stays in the current topic while its keywords overlap the
  // topic's RECENT window (overlap coefficient — forgiving for short utterances). A topic breaks on a
  // strong keyword shift, but only once it has a couple of turns, OR immediately on a long silence
  // (a natural topic boundary). Comparing against a recent window rather than the whole accumulated
  // set avoids the dilution that otherwise turns every turn into its own topic.
  function segmentTopics(turns) {
    const topics = [];
    let cur = null, prevT = null;
    turns.forEach((t, i) => {
      const kw = keywords(t.text), kwSet = new Set(kw);
      const gap = (prevT != null && typeof t.t_offset === "number") ? t.t_offset - prevT : null;
      const bigGap = gap != null && gap > CFG.topicGapSec;
      let start = !cur;
      if (cur) {
        const inter = kw.filter((w) => cur.recentSet.has(w)).length;
        const ov = inter / (Math.min(kwSet.size, cur.recentSet.size) || 1);
        if (ov < CFG.topicOverlap && (bigGap || cur.turnIdxs.length >= CFG.topicMinTurns)) start = true;
      }
      if (start) { cur = { turnIdxs: [], kwCounts: {}, recentKw: [], recentSet: new Set() }; topics.push(cur); }
      cur.turnIdxs.push(i);
      kw.forEach((w) => { cur.kwCounts[w] = (cur.kwCounts[w] || 0) + 1; });
      cur.recentKw.push(kw);
      if (cur.recentKw.length > CFG.recentWindow) cur.recentKw.shift();
      cur.recentSet = new Set(cur.recentKw.flat());
      if (typeof t.t_offset === "number") prevT = t.t_offset;
    });
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
        // Type heuristic: a question is a question; a reply by a *different* speaker to the previous
        // line is a retort (multi-speaker docs only — single-speaker can't distinguish speakers).
        let type = isQuestion(t.text) ? "question" : "statement";
        const changed = prevSid != null && t.speaker_id !== prevSid;
        if (changed && type === "statement") type = "retort";
        nodes.push({
          id: uid, type, turnIdx: idx, speaker_id: t.speaker_id, t_offset: t.t_offset,
          label: snippet(t.text),
        });
        edges.push({ from: tid, to: uid, kind: "contains" });
        if (prevUtterId) edges.push({ from: prevUtterId, to: uid, kind: changed ? "rebuts" : "reply" });
        prevUtterId = uid; prevSid = t.speaker_id;
      });
    });
    return { nodes, edges, topicCount: topics.length };
  }

  function snippet(s, n = 64) {
    s = String(s || "").replace(/\s+/g, " ").trim();
    return s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s;
  }

  function detectInterruptions(turns, durationSec, multiSpeaker) {
    const overlap = [], rapidSwitch = [];
    for (let i = 1; i < turns.length; i++) {
      const prev = turns[i - 1], cur = turns[i];
      const ps = typeof prev.t_offset === "number" ? prev.t_offset : null;
      const cs = typeof cur.t_offset === "number" ? cur.t_offset : null;
      const changed = multiSpeaker && cur.speaker_id !== prev.speaker_id;
      // Prefer the diarizer's REAL end time + silence gap (t_end / gap_before) when present; fall back
      // to the word-count estimate for docs without them (desktop, or single-speaker on-device).
      const realEnd = typeof prev.t_end === "number" ? prev.t_end : null;
      const gap = typeof cur.gap_before === "number" ? cur.gap_before
                : (ps != null && cs != null) ? cs - ps : null;

      // OVERLAP — someone began before the previous turn plausibly finished.
      if (ps != null && cs != null) {
        const estEnd = realEnd != null ? realEnd : ps + Math.max(CFG.minEndSec, words(prev.text).length / CFG.wordsPerSec);
        const overlapped = cs < estEnd - CFG.overlapSlackSec;
        const marked = /\b(overlap|crosstalk)\b/i.test((prev.markers || []).concat(cur.markers || []).join(" "));
        if (marked || (overlapped && (changed || (!multiSpeaker && looksCutOff(prev.text))))) {
          overlap.push({
            turnIdx: i, t_offset: cs, t_hms: cur.t_offset_hms || hms(cs),
            by: cur.speaker, of: prev.speaker,
            note: marked ? "marked overlap" : (changed ? "spoke over" : "cut off mid-line"),
          });
          continue;   // an overlap already implies a fast handoff; don't double-count as rapid
        }
      }

      // RAPID SWITCH — a quick handoff: a speaker change within the gap window. Requires real
      // speakers, so this stays empty for single-speaker on-device docs (a speaker-inferring LLM
      // backend, or a diarized recording, is what populates it there).
      if (changed && gap != null && gap >= 0 && gap < CFG.rapidGapSec) {
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
