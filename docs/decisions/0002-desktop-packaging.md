# ADR 0002 — Desktop/app packaging: browser-first + Tauri shell, not Electron

- **Status:** Proposed (resolves #92 — maintainer to ratify / amend)
- **Date:** 2026-06-05
- **Context issue:** #92 "Decide: should Voxterm be an Electron app?"
- **Reference implementation:** PR #175 (`feat/gui`) — a working web GUI + a Tauri
  desktop shell + an Android on-device plugin, so this decision is backed by a
  spike, not just a paper comparison.

## Context

VoxTerm is a Python engine (capture → faster-whisper/MLX → Silero VAD → 3D-Speaker
diarization → export) with a Textual TUI. The question is how — if at all — to ship
a graphical/desktop app without forking the speech pipeline per platform.

Key constraint that shapes everything: **the engine is Python and does the audio
capture itself** (`audio.capture`, `audio.system_capture`). So the "app" is a thin
control surface over a local engine process, *not* a self-contained native binary
that re-implements capture. Mic permission is therefore an OS-level grant to the
Python process, not a browser `getUserMedia` prompt.

## Alternatives considered

| Option | Bundle / footprint | Reuses 1 UI? | Engine integration | Update story | Verdict |
|---|---|---|---|---|---|
| **Electron** | Chromium + Node, ~120–200 MB, high RAM | yes (web) | spawn Python sidecar | electron-updater | Works, but heaviest; ships a second browser engine we don't need |
| **Tauri** | OS webview, ~3–10 MB, low RAM | yes (web) | Rust `setup()` spawns the Python sidecar; webview loads the GUI same-origin | Tauri updater / OS store | **Chosen shell** — tiny, same web UI, Android path exists |
| **Native (SwiftUI/Qt/…)** | small, best integration | no — per-platform rewrite | direct | per-platform | Rejected: abandons the shared UI; huge effort for a small team |
| **Browser-only (PWA)** | zero install (run server, open tab); installable PWA | yes (web) | same-origin HTTP/SSE to the local engine | service worker | **Chosen default** — lowest-friction, works on phone + desktop today |

## Decision

**No to Electron.** Adopt a **browser-first** architecture with a **Tauri** shell:

1. **Primary surface: the web GUI (PWA).** `python -m gui.server` → open
   `127.0.0.1` (loopback default; LAN behind a printed token). Installable as a
   PWA on desktop/phone. Zero packaging required to use it.
2. **Desktop app: a Tauri shell** (`src-tauri/`) that spawns the existing Python
   engine on a dynamic loopback port + per-launch token and points its webview at
   the same GUI — so there is exactly **one** UI codebase, every server-side
   security gate (CSP, same-origin, token, host allowlist) stays armed, and the
   bundle is a few MB instead of a bundled Chromium.
3. **Mobile: a Tauri Android plugin** (`tauri-plugin-voxasr/`) does on-device
   streaming ASR (sherpa-onnx) — the one place the engine is *not* Python.

Electron is rejected primarily because it bundles a second browser engine for no
benefit here (the OS webview is sufficient for a localhost control surface), at
~20–40× the bundle size and materially higher RAM, on a project that wants to run
"always-on in rooms" (#95) where footprint matters.

## Consequences

- **One UI, three surfaces** (browser / Tauri desktop / Tauri mobile) — fixes land
  once. The web GUI in #175 is that UI.
- **Engine stays Python** and is reused verbatim; no per-platform capture rewrite.
- **Trade-off:** the browser/Tauri surface depends on the local engine process
  running; it is a *controller*, not a standalone binary. The native desktop
  *release bundle* (PyInstaller-frozen sidecar) is not finished yet — `cargo tauri
  dev` works and the PWA covers desktop use today; freezing the sidecar for a
  signed installer is the remaining packaging task (tracked separately).
- **iOS** would need a Mac + Xcode (out of scope until hardware is available).

## Status of the spike

Implemented and tested on the fork (PR #175): web GUI (523 passing tests, 4 skipped, plus a headless-Chrome
e2e), Tauri desktop shell verified via `cargo tauri dev`, Android plugin built to a
green APK. So this ADR documents a decision that already has a working reference,
ready for the maintainer to accept or adjust.
