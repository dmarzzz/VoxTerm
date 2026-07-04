# Conversation modes: Graph & Interruptions

Two live conversation views in the VoxTerm GUI (`gui/static/`), shared by desktop and the Tauri
mobile bundle. A mode switcher (Transcript · Graph · Interruptions) sits above the content; Graph and
Interruptions render from a single pluggable **analyzer seam** so an on-device LLM can later replace
the heuristic without touching the UI.

## Files

| File | Role |
|------|------|
| `gui/static/analyze.js` | `window.VOX_ANALYZE.analyze(doc) -> Promise<{graph, interruptions}>`. The seam. Default = offline heuristic. |
| `gui/static/conversation.js` | `window.VOX_CONV`: mode switcher + SVG topic-tree renderer + interruptions counters/timeline/list. |
| `gui/static/app.js` | `setView()` delegates panel visibility to `VOX_CONV`; feeds it the active doc (loaded transcript or a live-tail doc) and a `window.VOX_SEEK(sec)` click-to-play hook. |
| `gui/static/index.html` | `#modeTabs`, `#graphPanel`, `#interruptPanel`; loads `analyze.js` + `conversation.js` before `app.js`. |
| `gui/static/style.css` | `.mode-tabs`, `.graph-*`, `.ir-*` (appended at end). |

`scripts/stage-mobile.sh` copies all of `gui/static/` into the mobile bundle and rewrites the script
paths, so both new files ship to Android automatically — **edit `gui/static/`, never `mobile-pair/app/`.**

## Analyzer output contract (stable — the LLM backend must match it)

```js
graph: {
  nodes: [{ id, type:'root'|'topic'|'statement'|'question'|'retort'|'counter', label,
            turnIdx?, speaker_id?, t_offset? }],
  edges: [{ from, to, kind:'topic'|'contains'|'reply'|'rebuts'|'supports' }],
}
interruptions: {
  overlap:     [{ turnIdx, t_offset, t_hms, by, of, note }],
  rapidSwitch: [{ turnIdx, t_offset, t_hms, gapSec, by, from }],
  overlapCount, rapidCount, total, durationSec, multiSpeaker,
}
```

## Heuristic backend (shipping now, fully offline)

- **Topics** — sequential keyword segmentation with an **IDF-weighted** overlap coefficient (distinctive
  words hold a topic together; ubiquitous filler barely counts), **hysteresis** (two off-topic turns in
  a row before a break, so one stray line doesn't fragment a topic; backchannels are neutral), and a
  **singleton-merge** post-pass (a lone-turn topic folds into its more-similar neighbor). A long silence
  (`topicGapSec`) still forces a boundary.
- **Graph** — root → topics → utterances. Argument typing from cheap lexical cues: an agreement cue
  ("exactly", "agree", …) makes the tie a **`supports`** edge; a disagreement/contrast cue ("but",
  "actually", "however", …) makes a **`counter`** (different speaker) or self-**`retort`** and a
  `rebuts` edge; a different-speaker reply with no cue is a `retort`. Questions also catch short
  wh-initial turns, since ASR usually drops the "?".
- **Interruptions** — *overlap*: a turn starts before the **floor-holding** turn (the prior turn with
  the latest known end — the diarizer's real `t_end` when present, else word count × speaking rate)
  plausibly finished, gated on a real speaker change or, single-speaker, a strict trailing-dash cut-off
  cue. Tracking the latest end (not just the immediately-previous turn) catches an interjection that
  starts while a turn from two entries back is still going. *rapidSwitch*: a speaker change within
  `rapidGapSec` (preferring the diarizer's `gap_before` silence measure).

### Rendering — top-down, mobile-first

The Graph renders as a **vertical HTML flow** (`gui/static/conversation.js` `renderGraph`): each topic
is a full-width section header, its utterances are cards beneath it in time order, and argument
relationships show inline as an indent + `↳ rebuts` / `↳ supports` rather than horizontal edges — so it
reads cleanly on a phone in portrait (vertical scroll only, no horizontal pan). A filter row offers
**All · Arguments · by-speaker**. Tapping a card seeks the audio (`data-seek` → `VOX_SEEK`).

### Speaker turns on mobile: the "Speakers" toggle (default ON)

Interruptions only mean something with real speaker turns, so the on-device **Speakers** toggle
(diarization-at-stop: pyannote segmentation + speaker embeddings via sherpa-onnx, in
`tauri-plugin-voxasr`) now defaults **ON** — with it off, every take is single-speaker and the
counter can never move (`rapidSwitch` needs a speaker change; `overlap` falls back to a strict
trailing-dash cue Whisper essentially never emits). Turning it off is a persisted opt-out for users
who want the fastest possible stop. If diarization was requested but fails, the plugin surfaces a
`diarWarning` that the GUI shows in the done toast instead of silently reporting one speaker.
Utterance text is built by assigning each Whisper token to its **nearest** diarization segment (no
dropped words in "silence" gaps, no duplicated words across overlapping segments), and adjacent
same-speaker segments under `DIAR_MERGE_GAP_SEC` apart merge into one turn. Live (mid-recording) the
doc is still single-speaker; counts appear when the take is transcribed at stop.

## Phase 2 — on-device LLM backend (BUILT)

`window.VOX_ANALYZE` is replaced on mobile by an LLM-backed analyzer (`gui/static/llm-backend.js`)
that runs a small local model to produce real argument structure (claims / retorts / counter-arguments)
and speaker-inferred interruptions, through the exact contract above. The UI does not change.

**Engine choice (why MediaPipe, not onnxruntime-genai/llama.cpp).** The Android APK is built blind in
CI (no local Android toolchain), so the engine had to be a plain Maven AAR with a high-level Kotlin API
— not a from-source NDK/CMake build. A verification pass (live-checked, 2026-06) ruled out:
- **onnxruntime-genai** — no Maven artifact; AAR must be built from source (NDK + Rust/Corrosion for
  constrained decoding) and bundles a *second* onnxruntime alongside sherpa-onnx. High blind-build risk.
- **llama.cpp (Llamatik AAR)** — real Maven AAR with GBNF grammar (best JSON), but declares **minSdk 26**
  (the app is minSdk 24) → manifest-merge failure without a minSdk bump. Kept as the documented upgrade.

**Chosen:** **MediaPipe LLM Inference** — `com.google.mediapipe:tasks-genai:0.10.27` (Google Maven).
The AAR declares **minSdk 23** (≤ app 24) and ships a single uniquely-named native lib per ABI
(`libllm_inference_engine_jni.so`) — no `.so` collision with sherpa-onnx / tauri-android. API verified
from the AAR bytecode: `LlmInference.createFromOptions(ctx, opts)` + `generateResponse(String)`, CPU
backend via `setPreferredBackend(LlmInference.Backend.CPU)`.

**Model:** `litert-community/Qwen2.5-0.5B-Instruct` `…_q8_ekv1280.task` — ungated (Apache-2.0,
verified curl-able without auth), ~547 MB, bundled as an Android asset (no runtime download; the app
has no INTERNET permission). Effective KV cache 1280 tokens, so `llm-backend.js` caps the transcript it
sends (`CFG.maxChars`).

**Pieces:**
1. `tauri-plugin-voxllm/` — mirrors `tauri-plugin-voxasr`: Rust bridge (`llm_available` / `start_generate`
   / `poll_generate` / `cancel_generate`) + Kotlin `VoxllmPlugin` (a *generic* prompt→text runner; it
   stages the `.task` from assets to `filesDir` and runs MediaPipe on a worker thread, polled). Registered
   `#[cfg(mobile)]` in `src-tauri/src/lib.rs`; `voxllm:default` added to `capabilities/mobile.json`.
2. `tauri-plugin-voxllm/fetch-deps.sh` — curls the `.task` into the plugin's Android assets (override
   with `VOXLLM_MODEL_URL`, e.g. a 1.5B for higher quality + bigger APK).
3. `gui/static/llm-backend.js` — builds a compact-schema prompt, runs `start/poll_generate`, extracts +
   repairs the JSON, maps it into the `{graph, interruptions}` contract, and **falls back to the
   heuristic on any failure** (so a flaky 0.5B model never breaks the modes). Mobile-only; no-ops on desktop.
4. ProGuard keeps for `com.google.mediapipe.**` / protobuf in `gen/android/app/proguard-rules.pro`
   (the AAR ships no consumer rules and release minifies).

**Output reliability:** MediaPipe has no grammar/constrained decoding, so a 0.5B model emits malformed or
truncated JSON. The prompt now asks for a **flat LINE grammar** instead of nested JSON —
`T: <title>` / `P: <speaker> | <kind> | <mm:ss> | <text>` / `I: <type> | <mm:ss> | <by> | <of> | <note>`
— which a small model keeps consistent far better, and where a truncated final line is simply dropped
rather than losing the whole chunk. `parseModelLines()` turns it back into the same `{topics, points}`
shape `toContract()` consumes. Belt-and-suspenders: JSON is still accepted as a fallback (now with a
**truncation repair** that closes unclosed strings/brackets and drops a dangling element), chunk parses
are **content-addressed cached** (re-Sharpen after a small edit reuses unchanged chunks), and any total
failure **falls back to the heuristic**. For hard-guaranteed JSON, the llama.cpp/GBNF path remains the
documented upgrade.

**Tap-to-insight (💡):** in the Graph, each topic header and leaf card carries a `💡` (only when a model
is loaded). Tapping it runs a **focused single-shot** prompt (`window.VOX_LLM.insight`) over just that
topic's points — "what is this about and why does it matter" — and shows a 1–2 sentence insight inline
(cached per node, tap again to collapse). It's much cheaper than the whole-conversation Sharpen, reuses
the same generic `voxllm` runner (no native change), is disabled while a Sharpen runs, and a new tap
supersedes an in-flight one.

**Build + publish:** `.github/workflows/android-experimental.yml` builds the arm64 APK (Whisper model +
LLM `.task` bundled) and publishes a rolling **`android-experimental`** prerelease — separate from the
stable `android-latest`. APK is large (~0.7 GB, sideload only).

**Watch-outs / follow-ups:**
- 0.5B JSON quality is the main quality lever — step up to Qwen2.5-1.5B `.task` (`VOXLLM_MODEL_URL`)
  if output is weak, or move to the llama.cpp/GBNF engine for guaranteed-valid JSON.
- Generation is debounced client-side; consider analyzing only on record-stop (not every live tail) to
  save battery on long sessions.
- First `generateResponse` also loads ~550 MB of weights (seconds) — the JS poll timeout allows for it.

## Local preview while iterating

The views render from any agent-json doc with `turns[]`. To eyeball them without the Python backend,
make a throwaway HTML in `.context/` that links `../gui/static/style.css` + `analyze.js` +
`conversation.js`, provides the `#modeTabs/#graphPanel/#interruptPanel` shell, sets
`VOX_CONV.getDoc = () => sampleDoc`, and calls `VOX_CONV.setTop('transcript')` + `setMode('graph')`.
Override `body{display:block}` (the app body is a sidebar grid). Serve the repo over `http.server`
(file:// is blocked) and open it.
