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

- **Topics** — sequential keyword segmentation: a turn stays on-topic while its keywords overlap the
  topic's recent window (overlap coefficient); a long silence (`topicGapSec`) forces a boundary.
- **Graph** — root → topics → utterances, with reply/rebut edges; a reply by a different speaker is a
  retort/rebut (multi-speaker docs only).
- **Interruptions** — *overlap*: next turn starts before the previous turn's estimated end (estimated
  from word count × speaking rate), gated on a real speaker change or, single-speaker, a strict
  trailing-dash cut-off cue. *rapidSwitch*: a speaker change within `rapidGapSec`.

### Honest limits on mobile (single-speaker on-device ASR)

Whisper on-device produces one speaker, so `rapidSwitch` is always empty and `overlap` only fires on
explicit dash cut-offs — the Interruptions view mostly shows 0 and says so in its caption. This is by
design: real interruption detection needs speaker inference, which is what the Phase 2 LLM provides.
The Graph still works on-device (topic tree), just without true retort/counter typing.

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

**JSON reliability:** MediaPipe has no grammar/constrained decoding, so a 0.5B model can emit malformed
or truncated JSON. Mitigated by (a) a compact, small-model-friendly schema (topics→points, not raw
node/edge ids — transformed to the contract in JS), (b) tolerant JSON extraction/repair, and (c)
heuristic fallback. For hard-guaranteed JSON, the llama.cpp/GBNF path is the documented upgrade.

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
