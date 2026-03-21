#!/usr/bin/env python3
"""Render a detailed, in-depth pipeline diagram for VoxTerm's diarization system.
Output: docs/pipeline_detailed.png"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# Cyberpunk palette
BG = "#0a0e14"
CYAN = "#00ffcc"
PINK = "#ff44aa"
GREEN = "#44ff44"
AMBER = "#ffaa00"
LAVENDER = "#aa88ff"
SKY = "#44ddff"
CORAL = "#ff6644"
DIM = "#336666"
GRID = "#0d1a1a"
TEXT = "#c0c0c0"
WHITE = "#ffffff"
BORDER = "#00e5ff"
DARK_PANEL = "#0f1520"

GLOW = lambda c: [pe.withStroke(linewidth=4, foreground=c + "22"),
                   pe.withStroke(linewidth=1, foreground=BG)]
GLOW2 = lambda c: [pe.withStroke(linewidth=2, foreground=c + "44")]


def box(ax, x, y, w, h, label, color, fontsize=12, sublabel=None, sublabel2=None, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.15",
        facecolor=DARK_PANEL, edgecolor=color, linewidth=2.2,
        alpha=alpha, zorder=5))
    dy = 0.18 if sublabel else 0
    if sublabel2:
        dy = 0.28
    ax.text(x, y + dy, label, color=color, fontsize=fontsize,
            ha="center", va="center", fontweight="bold",
            fontfamily="monospace", zorder=6, alpha=alpha,
            path_effects=GLOW(color))
    if sublabel:
        ax.text(x, y - 0.15 + (0.1 if sublabel2 else 0), sublabel,
                color=DIM, fontsize=8, ha="center", va="center",
                fontfamily="monospace", zorder=6, alpha=alpha)
    if sublabel2:
        ax.text(x, y - 0.32, sublabel2, color=DIM, fontsize=7,
                ha="center", va="center", fontfamily="monospace",
                zorder=6, alpha=alpha)


def arrow(ax, x1, y1, x2, y2, color, label=None, style="-|>", lw=2.0, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f"{style},head_width=0.3,head_length=0.15",
                                color=color, lw=lw, linestyle=ls, alpha=0.85),
                zorder=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, color=color, fontsize=7,
                ha="center", fontfamily="monospace", alpha=0.65, zorder=6)


def section_label(ax, x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=10, fontweight="bold",
            fontfamily="monospace", alpha=0.8, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=color, linewidth=1.5, alpha=0.7))


def detail_text(ax, x, y, lines, color=DIM, fontsize=7.5):
    for i, line in enumerate(lines):
        c = color
        if line.startswith("*"):
            line = line[1:]
            c = AMBER
        ax.text(x, y - i * 0.22, line, color=c, fontsize=fontsize,
                fontfamily="monospace", zorder=6, alpha=0.85)


def main():
    fig, ax = plt.subplots(figsize=(28, 20), dpi=150)
    ax.set_xlim(-0.5, 27.5)
    ax.set_ylim(-2.5, 17.5)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

    # Subtle grid
    for gx in np.arange(0, 28, 1):
        ax.axvline(gx, color=GRID, linewidth=0.2, alpha=0.3)
    for gy in np.arange(-2, 18, 1):
        ax.axhline(gy, color=GRID, linewidth=0.2, alpha=0.3)

    # ════════════════════════════════════════════════════════
    # TITLE
    # ════════════════════════════════════════════════════════
    ax.text(13.5, 17.0, "VOXTERM // COMPLETE DIARIZATION PIPELINE",
            color=CYAN, fontsize=20, ha="center", fontweight="bold",
            fontfamily="monospace", path_effects=GLOW(CYAN))
    ax.text(13.5, 16.5, "Audio In → Speaker-Labeled Transcript Out",
            color=DIM, fontsize=11, ha="center", fontfamily="monospace")

    # ════════════════════════════════════════════════════════
    # ROW 1: AUDIO CAPTURE (y=15)
    # ════════════════════════════════════════════════════════
    section_label(ax, 0.5, 15.8, "1. AUDIO CAPTURE", CYAN)

    # Mic
    box(ax, 3.0, 15.0, 3.5, 1.2, "MICROPHONE", CYAN,
        sublabel="sounddevice callback → queue",
        sublabel2="16kHz mono float32, 1024 samples/chunk")

    # System audio
    box(ax, 8.5, 15.0, 3.5, 1.2, "SYSTEM AUDIO", SKY,
        sublabel="Swift ScreenCaptureKit subprocess",
        sublabel2="stdout PCM pipe → reader thread")

    # Waveform
    rng = np.random.RandomState(42)
    t = np.linspace(0, 3, 300)
    env = np.zeros(300)
    for s, e, a in [(0.2, 1.0, 0.7), (1.3, 2.5, 0.85), (2.6, 2.9, 0.5)]:
        m = (t >= s) & (t <= e)
        env[m] = a
    sig = env * (0.5*np.sin(2*np.pi*150*t) + 0.3*np.sin(2*np.pi*300*t) + 0.1*rng.randn(300))
    wx = 12.5 + t * 1.5
    wy = 15.0 + sig * 0.4
    ax.plot(wx, wy, color=CYAN, linewidth=0.8, alpha=0.7, zorder=3)
    ax.fill_between(wx, 15.0, wy, color=CYAN, alpha=0.05)
    ax.text(13.5, 14.2, "mixed audio stream", color=DIM, fontsize=7,
            ha="center", fontfamily="monospace")

    arrow(ax, 5.0, 15.0, 12.3, 15.0, CYAN)
    arrow(ax, 10.5, 15.0, 12.3, 15.0, SKY)

    # ════════════════════════════════════════════════════════
    # ROW 2: VAD + BUFFERING (y=12.5)
    # ════════════════════════════════════════════════════════
    section_label(ax, 0.5, 13.3, "2. SPEECH DETECTION & BUFFERING", AMBER)

    arrow(ax, 13.5, 14.3, 5.0, 12.8, AMBER)

    # Silero VAD
    box(ax, 5.0, 12.0, 4.0, 1.5, "SILERO VAD", AMBER, fontsize=13,
        sublabel="ONNX Runtime · <1ms per 512 samples",
        sublabel2="speech probability [0,1] per 32ms frame")

    detail_text(ax, 1.2, 11.0, [
        "Per-chunk speech/silence decision:",
        "  prob >= 0.5 → speech → buffer",
        "  prob < 0.5  → silence → count",
        "*No PyTorch in main process (ONNX only)",
        "*Prevents MLX/PyTorch C++ conflict",
    ])

    # Speech prob bars
    for i in range(18):
        p = env[int(i * len(env)/18)]
        c = GREEN if p > 0.3 else "#222222"
        bx = 3.3 + i * 0.19
        ax.bar(bx, p * 0.4, width=0.14, bottom=10.6, color=c, alpha=0.6, zorder=3)
    ax.axhline(10.78, xmin=0.10, xmax=0.25, color=AMBER, linewidth=0.7,
               linestyle="--", alpha=0.5)

    arrow(ax, 7.2, 12.0, 11.0, 12.0, GREEN, "speech frames")

    # Audio buffer
    box(ax, 13.5, 12.0, 4.5, 1.5, "AUDIO BUFFER", GREEN, fontsize=13,
        sublabel="accumulate until trigger",
        sublabel2="max 3.0s · silence trigger 0.3s")

    detail_text(ax, 16.2, 12.5, [
        "Trigger conditions:",
        "  1. speech + 0.3s silence + >1s buffer",
        "  2. buffer >= 3.0s (force split)",
        "Output: 1-3s audio chunk for processing",
    ])

    # ════════════════════════════════════════════════════════
    # ROW 3: DUAL PROCESSING (y=9)
    # ════════════════════════════════════════════════════════
    section_label(ax, 0.5, 10.3, "3. PARALLEL PROCESSING", LAVENDER)

    # Fork
    arrow(ax, 13.5, 11.1, 5.5, 9.5, LAVENDER, "audio chunk")
    arrow(ax, 13.5, 11.1, 21.5, 9.5, PINK, "audio chunk (IPC)")

    # Process boundaries
    ax.add_patch(FancyBboxPatch(
        (0.5, 6.0), 10.0, 4.5, boxstyle="round,pad=0.2",
        facecolor="#00000000", edgecolor=LAVENDER, linewidth=1.5,
        linestyle="--", alpha=0.3, zorder=2))
    ax.text(1.0, 10.2, "MAIN PROCESS", color=LAVENDER, fontsize=8,
            fontfamily="monospace", alpha=0.5)

    ax.add_patch(FancyBboxPatch(
        (16.5, 0.5), 10.5, 10.0, boxstyle="round,pad=0.2",
        facecolor="#00000000", edgecolor=PINK, linewidth=1.5,
        linestyle="--", alpha=0.3, zorder=2))
    ax.text(17.0, 10.2, "DIARIZER SUBPROCESS (PyTorch)", color=PINK,
            fontsize=8, fontfamily="monospace", alpha=0.5)

    # ── Transcription (left) ──
    box(ax, 5.5, 8.5, 5.5, 2.0, "QWEN3-ASR", LAVENDER, fontsize=15,
        sublabel="MLX Framework · Metal GPU · ~200ms",
        sublabel2="transcription + word timestamps")

    detail_text(ax, 1.0, 7.2, [
        "Qwen3-ASR-0.6B (default) or 1.7B",
        "MLX inference on Apple Metal GPU",
        "Hallucination filter + deduplication",
        "Outputs: text + word-level timestamps",
        "*Never loads PyTorch (separate process)",
    ])

    # Transcript output
    ax.text(5.5, 6.5, '"What are you guys doing?"',
            color=LAVENDER, fontsize=9, ha="center", fontfamily="monospace",
            fontstyle="italic", alpha=0.8)

    # ── Diarization (right) ──

    # Segmentation model
    box(ax, 21.5, 9.0, 4.5, 1.3, "PYANNOTE SEG-3.0", CORAL, fontsize=11,
        sublabel="ONNX · 6MB · ~10ms/5s chunk",
        sublabel2="3 local speakers + overlap detection")

    detail_text(ax, 24.3, 9.3, [
        "7-class powerset output:",
        "  no_spk, S1, S2, S3,",
        "  S1+S2, S1+S3, S2+S3",
        "17ms frame resolution",
    ])

    # Segmentation visualization
    seg_y = 7.8
    ax.text(17.5, seg_y + 0.5, "per-frame activation →", color=DIM,
            fontsize=7, fontfamily="monospace")
    for i in range(30):
        x = 17.5 + i * 0.2
        # Fake segmentation bars
        if i < 12:
            ax.bar(x, 0.3, width=0.15, bottom=seg_y, color=CYAN, alpha=0.5, zorder=3)
        if 8 < i < 25:
            ax.bar(x, 0.25, width=0.15, bottom=seg_y - 0.35, color=PINK, alpha=0.5, zorder=3)
        if 18 < i:
            ax.bar(x, 0.2, width=0.15, bottom=seg_y - 0.65, color=GREEN, alpha=0.5, zorder=3)
    ax.text(24.2, seg_y + 0.15, "S1", color=CYAN, fontsize=7, fontfamily="monospace")
    ax.text(24.2, seg_y - 0.2, "S2", color=PINK, fontsize=7, fontfamily="monospace")
    ax.text(24.2, seg_y - 0.5, "S3", color=GREEN, fontsize=7, fontfamily="monospace")
    # Overlap region marker
    ax.axvspan(19.1, 19.9, ymin=0.41, ymax=0.47, color=CORAL, alpha=0.15)
    ax.text(19.5, seg_y + 0.55, "overlap", color=CORAL, fontsize=6,
            ha="center", fontfamily="monospace", alpha=0.7)

    arrow(ax, 21.5, 8.2, 21.5, 7.2, CORAL)

    # Overlap-aware embedding
    box(ax, 21.5, 6.5, 4.5, 1.2, "OVERLAP-AWARE\nEMBEDDING", AMBER, fontsize=10,
        sublabel="crop to single-speaker frames only")

    detail_text(ax, 17.0, 6.8, [
        "For each local speaker:",
        "  1. Find solo frames (no overlap)",
        "  2. Crop audio to those frames",
        "  3. Extract clean embedding",
        "*Overlap frames → near-zero weight",
    ])

    arrow(ax, 21.5, 5.8, 21.5, 5.0, AMBER)

    # CAM++ embedding
    box(ax, 21.5, 4.2, 4.5, 1.4, "CAM++", PINK, fontsize=15,
        sublabel="512-dim · 7.18M params · 0.73% EER",
        sublabel2="Fbank(80) → D-TDNN → TSTP → embedding")

    # Embedding visualization
    emb = np.random.RandomState(1).randn(50) * 0.5
    for i, v in enumerate(emb):
        bx = 19.5 + i * 0.08
        c = PINK if v > 0 else CYAN
        h = abs(v) * 0.25
        b = 3.1 if v > 0 else 3.1 - h
        ax.bar(bx, h, width=0.055, bottom=b, color=c, alpha=0.45, zorder=3)
    ax.text(21.5, 2.8, "512-dim speaker embedding", color=DIM,
            fontsize=7, ha="center", fontfamily="monospace")

    arrow(ax, 21.5, 3.4, 21.5, 2.5, PINK)

    # ════════════════════════════════════════════════════════
    # ROW 4: CLUSTERING (y=1.5)
    # ════════════════════════════════════════════════════════
    box(ax, 21.5, 1.5, 8.5, 2.5, "ONLINE CLUSTERING", AMBER, fontsize=14)

    detail_text(ax, 17.8, 2.3, [
        "*Running-sum centroids (direction-stable)",
        "*Dual threshold: match=0.35, new=0.30",
        "*Adaptive freeze (late joiners allowed if sim<0.20)",
        "*Overlap detection: skip blended centroid updates",
        "*Transition detection: check recent embedding history",
        "*Spectral re-clustering (eigengap auto-k) every 8 calls",
        "*Centroid refresh from filtered high-quality embeddings",
        "*Pairwise merge (threshold=0.50) every 3 calls",
    ])

    # Centroid dots
    centroids = [(25.0, 2.5, CYAN, "S1"), (26.0, 1.3, PINK, "S2"),
                 (26.5, 2.1, GREEN, "S3"), (25.5, 0.8, AMBER, "S4")]
    for cx, cy, cc, cl in centroids:
        ax.plot(cx, cy, "s", color=cc, markersize=10, alpha=0.7, zorder=7)
        ax.text(cx, cy - 0.3, cl, color=cc, fontsize=8,
                ha="center", fontfamily="monospace", alpha=0.8, zorder=7)

    # Incoming dot
    ax.plot(25.8, 3.0, "o", color=PINK, markersize=7, alpha=0.8, zorder=7)
    ax.annotate("", xy=(26.0, 1.6), xytext=(25.8, 2.9),
                arrowprops=dict(arrowstyle="->,head_width=0.15",
                                color=PINK, lw=1.2, linestyle="--", alpha=0.5),
                zorder=4)
    ax.text(26.3, 2.6, "match?", color=PINK, fontsize=7,
            fontfamily="monospace", alpha=0.6)

    # ════════════════════════════════════════════════════════
    # ROW 5: OUTPUT (y=-1)
    # ════════════════════════════════════════════════════════
    arrow(ax, 5.5, 6.2, 8.0, 0.5, CYAN)
    arrow(ax, 17.5, 1.0, 14.0, 0.0, AMBER)

    # Transcript output
    ax.add_patch(FancyBboxPatch(
        (1.0, -2.2), 12.0, 3.0, boxstyle="round,pad=0.2",
        facecolor=DARK_PANEL, edgecolor=BORDER, linewidth=2.5, zorder=5))
    ax.text(7.0, 0.5, "TRANSCRIPT // OUTPUT", color=BORDER, fontsize=13,
            ha="center", fontfamily="monospace", fontweight="bold", zorder=6,
            path_effects=GLOW(BORDER))

    lines = [
        (CYAN,   "Speaker 1", "15:04:27", "What are you guys doing over here?"),
        (PINK,   "Speaker 2", "15:04:30", "Just working on the diarization thing."),
        (GREEN,  "Speaker 3", "15:04:33", "Did you remember to call your mother?"),
        (AMBER,  "Speaker 1", "15:04:36", "I thought this was working already!"),
    ]
    for i, (c, spk, ts, txt) in enumerate(lines):
        y = -0.1 - i * 0.48
        ax.text(1.5, y, ts, color=DIM, fontsize=8, fontfamily="monospace", zorder=6)
        ax.text(4.0, y, spk, color=c, fontsize=9, fontfamily="monospace",
                fontweight="bold", zorder=6)
        ax.text(6.5, y, txt, color=TEXT, fontsize=8, fontfamily="monospace", zorder=6)

    # ════════════════════════════════════════════════════════
    # PERSISTENCE (right side)
    # ════════════════════════════════════════════════════════
    section_label(ax, 0.5, 4.5, "4. PERSISTENCE", SKY)

    box(ax, 5.0, 3.5, 4.5, 1.2, "SPEAKER STORE", SKY, fontsize=11,
        sublabel="SQLite WAL · 512-dim centroids",
        sublabel2="cross-session matching + profiles")

    detail_text(ax, 1.0, 2.6, [
        "Cross-session speaker recognition:",
        "  HIGH  (>0.55): auto-assign name",
        "  MEDIUM (>0.35): suggest with '?'",
        "  LOW: unknown speaker",
        "Profile: centroid + 20 exemplars + quality",
        "Auto-migration on embedding dim change",
    ])

    box(ax, 5.0, 1.2, 4.5, 1.0, "LIVE SAVE", GREEN, fontsize=10,
        sublabel="~/Documents/voxterm/.live/*.md")

    # ════════════════════════════════════════════════════════
    # STATS BAR
    # ════════════════════════════════════════════════════════
    stats = [
        "CAM++ 7.18M params · 0.73% EER",
        "Silero VAD 0.97 AUC · pyannote seg-3.0 overlap detection",
        "Spectral eigengap · centroid refresh · adaptive freeze",
        "DER: 22.7% (2spk) · 58.4% (4spk overlap) · 53.6% (35min meeting)",
        "RTF 0.013 = 77x real-time on Apple Silicon CPU",
    ]
    for i, s in enumerate(stats):
        ax.text(13.5, -1.5 - i * 0.25, s, color=DIM, fontsize=7,
                ha="center", fontfamily="monospace", alpha=0.5)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_detailed.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
