# VoxTerm Requirements

## Purpose

VoxTerm is a local-first transcription tool for conversations, meetings, and
room-scale sessions. It records microphone and optional system audio, produces
speaker-attributed transcripts, and keeps the default data path on the user's
machine.

## Target Users

- A person recording their own notes, calls, demos, or conversations.
- Small groups that want a shared transcript without sending room audio to a
  cloud transcription service.
- Always-on room deployments where consent, visible recording state, and local
  recovery matter more than centralized administration.
- Agent or workflow integrations that need structured transcript events and
  exports.

## Core Flows

### Record

- The user can start and stop recording from the TUI, GUI, or packaged shell.
- Recording state must be obvious while active: the interface shows persistent
  recording status, elapsed time, and a clear stop affordance.
- The capture path supports microphone input and, where the platform supports
  it, system audio or mic+system mixed input.
- Audio capture failures must be visible in the UI. A missing, muted, revoked,
  or unavailable input must not fail silently.
- Long recordings must stream to disk or bounded buffers rather than requiring
  the full session to stay in memory.

### Transcribe

- Finished speech is transcribed locally by the selected backend.
- The default backend should prefer the fastest accurate local model available
  for the platform.
- Model loading, transcribing, and error states must be visible so the app does
  not appear frozen.
- The user can choose model and language where the backend supports it.
- Unsupported or malformed audio input should produce a clear error instead of
  an empty success state.

### Diarize

- When speaker detection is enabled, VoxTerm identifies speaker turns and
  displays stable speaker labels in the transcript.
- Users can tag and rename speakers during or after a session.
- Voice profile data is stored locally and treated as sensitive biometric data.
- Diarization failures degrade to transcript text with clear UI feedback rather
  than crashing the session.

### Save And Export

- Transcripts are auto-saved while recording so a crash does not lose the whole
  session.
- The user can save/export finalized transcripts as Markdown.
- GUI exports support agent-ready Markdown and structured formats used by
  downstream tools.
- Successful saves should acknowledge the result with count, file location, and
  a next action.
- Copy/export errors must be visible to the user.

### Upload And Sharing

- Default behavior is local-only. Remote egress must be opt-in and visible.
- P2P party mode shares transcript text over the local network for nearby peers.
- Hivemind transcript streaming is configurable and can be disabled.
- Any remote or LAN surface must have an explicit privacy/security posture:
  loopback-only by default, token-gated when exposed, and clear status in the UI.

## Interface Requirements

- Keyboard control must cover the core TUI workflows: record, tag speakers,
  react, summarize/export, browse saved transcripts, party mode, model/language
  selection, debug, clear, help, and quit.
- GUI controls must be reachable by pointer and keyboard.
- Touch/mobile layouts must keep primary record controls large enough for
  comfortable use and avoid horizontal overflow.
- Empty, loading, transcribing, no-speech, low-signal, error, and success states
  must be explicit.
- Motion is allowed only when it clarifies state. Reduced-motion preferences
  should be respected in browser surfaces.
- Transcript content must remain readable under long sessions, repeated
  speakers, reactions, and copied/exported variants.

## Platform Requirements

- macOS Apple Silicon is the primary local ML target.
- macOS Intel, Linux, and Windows should use cross-platform CPU-capable
  backends where possible.
- Browser/PWA GUI runs against the local Python engine over loopback by default.
- Tauri desktop/mobile shells should reuse the same web UI wherever practical.
- Platform-specific audio features, such as system audio capture, may be
  conditional but must fail visibly when unavailable.

## Privacy And Security Requirements

- No raw audio is uploaded by default.
- Saved transcript text lives under the user's local data/transcript directory.
- Voice embeddings and speaker profile data stay local and must use restrictive
  file permissions where supported.
- LAN-exposed GUI/API surfaces require a token.
- Local loopback GUI surfaces validate Host headers to reduce DNS-rebinding
  risk.
- Content Security Policy should keep the web UI self-contained.
- Event streams and companion integrations must be opt-in or local-file based.

## Hardware Assumptions

- A working microphone is required for live capture.
- Apple Silicon machines can run the preferred MLX/Qwen/Parakeet paths.
- CPU-only systems need smaller or cross-platform transcription backends.
- Always-on room deployments need visible recording indicators, stable power,
  and known input devices.
- External non-speech input devices can integrate by writing reaction JSONL to
  the reaction inbox; custom hardware is not required for the core transcript
  data model.

## Deferred Or Non-Goals

- Cloud transcription is not a default requirement.
- A centralized transcript upload service is not required for local use.
- Full hardware procurement and deployment logistics are outside the core app
  requirements.
- Perfect speaker identification is not guaranteed; the app must make
  uncertainty visible and let users correct labels.
- Browser-only microphone capture is not the primary architecture for the
  desktop GUI; the local engine owns capture.
- Public internet exposure of the local GUI/API is out of scope.
