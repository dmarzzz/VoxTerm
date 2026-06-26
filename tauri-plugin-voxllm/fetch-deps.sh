#!/usr/bin/env bash
# Fetch the native dep for the on-device LLM plugin (gitignored — large binary):
#   - a small instruction-tuned LLM in MediaPipe .task format, staged into the plugin's Android
#     assets so the APK runs the conversation Graph / Interruptions analysis fully offline (no
#     first-run download; the app has no INTERNET permission).
#
# Default model: Qwen2.5-0.5B-Instruct (q8, MediaPipe .task, ~547 MB, Apache-2.0, ungated) from the
# litert-community HuggingFace org — verified directly curl-able (no token/login). Override the URL
# with VOXLLM_MODEL_URL to bundle a different .task (e.g. a 1.5B for higher quality, bigger APK).
#
# Run before `cargo tauri android build`:
#   ./tauri-plugin-voxllm/fetch-deps.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$HERE/android/src/main/assets/voxllm-model"
mkdir -p "$ASSETS"

MODEL_URL="${VOXLLM_MODEL_URL:-https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct/resolve/main/Qwen2.5-0.5B-Instruct_multi-prefill-seq_q8_ekv1280.task}"
# Cache keyed by URL so switching models re-downloads, but rebuilds reuse the cached file.
CACHE_DIR="$HOME/.cache/voxterm/voxllm"
CACHE="$CACHE_DIR/$(echo "$MODEL_URL" | shasum | cut -d' ' -f1).task"
mkdir -p "$CACHE_DIR"

if [ ! -f "$CACHE" ]; then
  echo "fetching on-device LLM model (this is large, ~0.5 GB)…"
  echo "  $MODEL_URL"
  curl -fSL -o "$CACHE.tmp" "$MODEL_URL"   # -L: HF 302-redirects to its CDN
  mv "$CACHE.tmp" "$CACHE"
fi

cp "$CACHE" "$ASSETS/model.task"
echo "voxllm model staged ($(du -h "$ASSETS/model.task" | cut -f1) -> ${ASSETS#"$HERE"/}/model.task)."
