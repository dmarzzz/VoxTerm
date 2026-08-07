# Decision 0001 — Easier launch for non-technical users

**Status:** Open. This memo presents the options; the actual pick is yours.

**Context.** Shape Rotator is deploying VoxTerm on ~5 always-on devices. Some
of the people who will need to start/stop it, change a setting, or copy a
transcript are not comfortable in a terminal. Today VoxTerm runs as a
Textual TUI launched from `python -m tui.app` (or the `voxterm` shell
script). The user-facing question is: *what's the cheapest path to
"a non-technical user can run this"?*

This memo compares four routes. It does **not** pick one — that's a call
the project lead should make with full context.

## What "easier launch" means

Working definition for ranking the options:

1. A non-technical operator can start the app by clicking an icon
   (no `cd ~/code/voxterm && source .venv/bin/activate && …`).
2. They can stop and restart it without opening a terminal.
3. Settings (model, language, upload URL) can be changed without editing
   `~/.state.json` by hand.
4. When something breaks, the failure mode is "the icon shows an error,"
   not a stack trace in a terminal they closed.

Anything beyond this — mouse-first speaker tagging, embedded media, drag-drop
file import — is out of scope for this decision.

## The options

| Option | Effort | Day-one UX | What you give up |
|---|---|---|---|
| **A. macOS `.app` wrapper around the TUI** in a borderless terminal emulator, packaged as a `.dmg` with the venv baked in | ~1 week | Double-click to launch; still a TUI underneath, so no mouse, but visible as a real app in the Dock | Doesn't fix the gap that "TUI is a barrier" — it just makes launching easier |
| **B. Tauri shell** (Rust + WebView frontend) talking to the existing Python audio/diarization backend over a local socket | ~3–4 weeks | Native-feeling app; ~10MB binary; mouse + keyboard; can keep the existing transcription pipeline untouched | Adds a new tech stack (Rust toolchain, Tauri, frontend framework choice); needs an IPC layer between the GUI and the Python audio worker |
| **C. Electron rewrite** of the frontend (React/TS); backend untouched | ~4–6 weeks | Native-feeling app; mouse-first; rich visuals possible | ~150MB binary; ~300MB RAM baseline (on machines that already need to run MLX + diarization); JS supply chain; macOS notarization story we don't currently have |
| **D. Stay on TUI**, invest the same calendar time in onboarding for the deployed devices: auto-launch on boot, cheat-sheet sticker on each laptop, scripted setup script that pins the venv and runs `voxterm` on login | ~3 days | Operator runs the device; users just talk in the room; nobody clicks anything | Doesn't help anyone who wants to install it on their personal machine |

## Trade space, summarized

- **A is the floor** — cheapest, ships fast, but the underlying experience is
  unchanged. Worth doing only if launch friction is the actual blocker (not
  the keyboard-only interaction model).
- **B is the middle path** — real GUI, smaller binary than Electron, lets the
  audio backend stay where it already works. Costs you the most calendar time
  unless someone on the team already knows Tauri.
- **C is the most "obvious" answer and probably the wrong one** for this
  project's constraints: VoxTerm is already memory-heavy because of the local
  models, and the deployed laptops will resent another 300 MB of RAM for a
  React shell.
- **D is the "right" answer if the deployment is the real goal** and personal
  installs are aspirational. Three days to get auto-launch, a printed
  cheatsheet, and a setup script gets you 80% of the win.

## Recommendation (mine, not yours)

**D first, then A if D isn't enough.** Reasoning:

1. The Shape Rotator concrete deliverable is "5 devices in rooms." For that,
   the operator-runs-it model + auto-launch is plenty.
2. A `.app` wrapper is cheap insurance for the personal-install case and
   stacks on top of D without redoing anything.
3. B (Tauri) and C (Electron) are bets on a much bigger product surface than
   we have committed to today. Don't open that scope until either (a) D + A
   are demonstrably insufficient, or (b) the program lead explicitly
   prioritizes a richer frontend over other roadmap items.

If you disagree with this read — particularly if you think a non-trivial
fraction of users will install personally and the TUI itself (not just the
launch friction) is what's blocking them — pick **B over C**. Tauri's
binary/RAM profile doesn't fight VoxTerm's existing model overhead the way
Electron does.

## Next steps once you pick

- **D:** Write the boot-launch script for macOS (LaunchAgent) and Linux
  (systemd user unit). Print the cheatsheet card. Add `voxterm doctor` for
  self-check.
- **A:** Investigate `briefcase` or `py2app` for the `.dmg`; bundle the
  `.venv`; pick a terminal emulator host (likely `osascript` to launch
  Terminal.app with the right command, or embed via a tool like
  `terminal-notifier`'s pattern).
- **B:** Spike for 2 days: get a Tauri window talking to a Python subprocess
  over a Unix socket and rendering live transcription text. If that works,
  scope the rest.
- **C:** Don't, unless you've ruled out the others on real evidence.

## Out of scope for this memo

- Which onboarding script lives where (D follow-up)
- Auth/TLS for the upload server (tracked in `server/README.md`)
- Twitch-chat reactions, physical emoji input (deprioritized in the original
  Shape Rotator brief)
