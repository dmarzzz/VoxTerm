#!/usr/bin/env bash
# Fetch the native deps for the on-device ASR plugin (gitignored — large binaries):
#   - the sherpa-onnx Android AAR (statically-linked onnxruntime), version-matched to the
#     desktop engine (sherpa_onnx 1.13.2)
#   - an OFFLINE Whisper int8 model, staged into the plugin's Android assets so the APK
#     transcribes fully offline (no first-run download). The phone records, then transcribes the
#     whole clip at stop with Whisper — the same model family the desktop's faster-whisper uses,
#     so the phone gets full-context, punctuated transcripts (no rough live/streaming output).
#
# Model size, selected with VOXASR_MODEL (default: base.en):
#   whisper-tiny.en   ~75 MB assets  — fastest, roughest
#   whisper-base.en   ~155 MB assets — recommended; desktop fw-base parity, comfortably real-time
#   whisper-small.en  ~360 MB assets — most accurate, ~real-time on a mid-range phone
# Whisper has no joiner (unlike a streaming transducer); the plugin loads it as model_type=whisper.
#
# Run before `cargo tauri android build`, e.g.:
#   ./fetch-deps.sh                               # base.en default
#   VOXASR_MODEL=whisper-small.en ./fetch-deps.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VER="1.13.2"
LIBS="$HERE/android/libs"
ASSETS="$HERE/android/src/main/assets/voxterm-model"
mkdir -p "$LIBS" "$ASSETS"

# `.en` tiers are English-only; the plain tiers are MULTILINGUAL (auto-detect language). LANG_TAG is
# staged into assets/lang.txt and read by the Kotlin recognizer ("en" = force English, "auto" = detect).
case "${VOXASR_MODEL:-whisper-base.en}" in
  whisper-tiny.en)  MODEL="sherpa-onnx-whisper-tiny.en";  LANG_TAG="en" ;;
  whisper-base.en)  MODEL="sherpa-onnx-whisper-base.en";  LANG_TAG="en" ;;
  whisper-small.en) MODEL="sherpa-onnx-whisper-small.en"; LANG_TAG="en" ;;
  whisper-tiny)     MODEL="sherpa-onnx-whisper-tiny";     LANG_TAG="auto" ;;
  whisper-base)     MODEL="sherpa-onnx-whisper-base";     LANG_TAG="auto" ;;
  whisper-small)    MODEL="sherpa-onnx-whisper-small";    LANG_TAG="auto" ;;
  *) echo "unknown VOXASR_MODEL='${VOXASR_MODEL}' (want whisper-{tiny,base,small} or .en variants)" >&2; exit 1 ;;
esac

AAR="$LIBS/sherpa-onnx-$VER.aar"
if [ ! -f "$AAR" ]; then
  echo "fetching sherpa-onnx $VER Android AAR…"
  curl -fSL -o "$AAR" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/v$VER/sherpa-onnx-static-link-onnxruntime-$VER.aar"
fi

# Reuse the desktop model cache if present; otherwise download + extract the model release.
CACHE="$HOME/.cache/voxterm/sherpa/$MODEL"
# glob is intentional (each tier ships exactly one encoder int8)
# shellcheck disable=SC2206
cached=( "$CACHE"/*encoder*.int8.onnx )
if [ ! -e "${cached[0]}" ]; then
  echo "fetching ${MODEL}…"   # brace the var: a bare $MODEL before the multibyte "…" trips bash's
  mkdir -p "$(dirname "$CACHE")"
  curl -fSL -o "/tmp/$MODEL.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$MODEL.tar.bz2"
  tar xjf "/tmp/$MODEL.tar.bz2" -C "$(dirname "$CACHE")"
  rm -f "/tmp/$MODEL.tar.bz2"
fi

# Stage the int8 encoder/decoder + tokens under the names the Kotlin plugin loads. Glob the source
# names (Whisper ships base.en-encoder.int8.onnx etc.); first match wins, like the desktop _pick().
stage() {  # stage <glob-relative-to-CACHE> <dest-name>: copy the matching file into assets
  # word-split + glob intentional; each model ships exactly one match
  # shellcheck disable=SC2206
  local matches=( "$CACHE"/$1 )
  [ -e "${matches[0]}" ] || { echo "missing '$1' in $CACHE" >&2; exit 1; }
  cp "${matches[0]}" "$ASSETS/$2"
}
rm -f "$ASSETS"/*           # drop any stale model from a previously selected tier
stage "*encoder*.int8.onnx" encoder.int8.onnx
stage "*decoder*.int8.onnx" decoder.int8.onnx
stage "*tokens*"            tokens.txt
cp "$CACHE/test_wavs/0.wav" "$ASSETS/test.wav"   # debug self-test clip (offline decode check)
printf '%s' "$LANG_TAG" > "$ASSETS/lang.txt"      # read by the recognizer: "en" (English) or "auto" (detect)

# ---- on-device speaker diarization models (opt-in feature) -------------------
# sherpa-onnx OfflineSpeakerDiarization = pyannote segmentation + a speaker-embedding model + fast
# clustering. Bundled so diarization is fully offline; the Kotlin plugin loads them by these names.
DIAR="$HERE/android/src/main/assets/voxterm-diar"
DIAR_CACHE="$HOME/.cache/voxterm/sherpa/diar"
mkdir -p "$DIAR" "$DIAR_CACHE"
SEG_TAR="sherpa-onnx-pyannote-segmentation-3-0"
EMB_ONNX="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"   # speaker embeddings (language-agnostic)
if [ ! -f "$DIAR_CACHE/$SEG_TAR/model.onnx" ]; then
  echo "fetching ${SEG_TAR}…"
  curl -fSL -o "/tmp/$SEG_TAR.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/$SEG_TAR.tar.bz2"
  tar xjf "/tmp/$SEG_TAR.tar.bz2" -C "$DIAR_CACHE"
  rm -f "/tmp/$SEG_TAR.tar.bz2"
fi
if [ ! -f "$DIAR_CACHE/$EMB_ONNX" ]; then
  echo "fetching ${EMB_ONNX}…"
  curl -fSL -o "$DIAR_CACHE/$EMB_ONNX" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recognition-models/$EMB_ONNX"
fi
rm -f "$DIAR"/*
cp "$DIAR_CACHE/$SEG_TAR/model.onnx" "$DIAR/segmentation.onnx"
cp "$DIAR_CACHE/$EMB_ONNX"           "$DIAR/embedding.onnx"

echo "voxasr native deps ready ($(du -sh "$ASSETS" | cut -f1) asr [$MODEL, lang=$LANG_TAG], $(du -sh "$DIAR" | cut -f1) diar, $(du -h "$AAR" | cut -f1) aar)."
