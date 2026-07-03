# Physical Reaction Pad

VoxTerm's reaction layer is deliberately file/CLI based so a physical input
device does not need a custom VoxTerm plugin. Any button pad that can run a
shell command, emit a hotkey into an automation tool, or write JSONL can add
non-speech reactions to the live transcript.

## Minimal Setup

1. Start VoxTerm normally.
2. Point any external bridge at the same inbox path if you need a custom path:

   ```bash
   export VOXTERM_REACTION_INBOX="$HOME/.voxterm-reactions.jsonl"
   voxterm
   ```

3. Map buttons to `voxterm-react` commands:

   ```bash
   voxterm-react clap --author button-pad
   voxterm-react heart --author button-pad
   voxterm-react question "source?" --author button-pad
   voxterm-react idea "save this" --author button-pad
   ```

The running TUI polls the inbox and folds each event into the transcript
timeline, live markdown, exports, and the JSONL event stream.

For pads that expose one persistent serial/HID stream instead of launching a
command per button, run one bridge process and pipe line commands into
`voxterm-react --stdin`:

```bash
pad-serial-bridge /dev/tty.usbmodem123 | voxterm-react --stdin --author front-pad
```

Accepted line formats:

```text
clap
question "source?"
idea save this
{"kind":"reaction","emoji":"plus","text":"agree","author":"front-pad"}
```

Blank or malformed lines are ignored, so a noisy bridge does not crash the
running transcript.

## Device Options

- Stream Deck, X-keys, Touch Portal, BetterTouchTool, Karabiner, or any macro
  keyboard: configure each button to run one command above.
- QMK/VIA keyboard or DIY USB HID pad: map each key to a host-side automation
  shortcut, then have that automation run `voxterm-react`, or use a small
  serial bridge with `voxterm-react --stdin`.
- Custom bridge process: write one JSON object per line to the inbox:

  ```json
  {"kind":"reaction","emoji":"question","text":"source?","author":"button-pad"}
  ```

Aliases are normalized by VoxTerm: `clap`, `heart`, `laugh`, `question`,
`idea`, and `plus`.

## Hardware Behavior Guidelines

- Debounce buttons in firmware or the bridge so one press writes one event.
- Use physical labels or keycaps that match the preset names.
- Keep the pad silent and local; it should not record audio or send network
  traffic.
- Prefer momentary buttons. Latching switches can create accidental repeated
  reactions if the bridge treats state as presses.
- Give each bridge a stable author name, such as `front-pad` or `macro-pad`,
  so transcript readers know the reaction source.

## Acceptance Check

Run this before using a pad in a room:

```bash
tmp="$(mktemp -d)"
VOXTERM_REACTION_INBOX="$tmp/reactions.jsonl" voxterm-react question "source?" --author macro-pad
cat "$tmp/reactions.jsonl"
```

For a persistent bridge:

```bash
tmp="$(mktemp -d)"
printf 'clap\nquestion "source?"\n' | \
  voxterm-react --stdin --author macro-pad --inbox "$tmp/reactions.jsonl"
cat "$tmp/reactions.jsonl"
```

The file should contain one `kind: "reaction"` JSONL row per input reaction.
When VoxTerm is running with the same `VOXTERM_REACTION_INBOX`, the transcript
should show each reaction as a timeline row and include it in saved Markdown.
