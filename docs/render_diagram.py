#!/usr/bin/env python3
"""Render a static pipeline diagram for VoxTerm. Output: docs/pipeline.png"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# Cyberpunk palette
BG = "#0a0e14"
CYAN = "#00ffcc"
PINK = "#ff44aa"
GREEN = "#44ff44"
AMBER = "#ffaa00"
LAVENDER = "#aa88ff"
DIM = "#336666"
GRID = "#0d1a1a"
TEXT = "#c0c0c0"
BORDER = "#00e5ff"

GLOW = lambda c: [pe.withStroke(linewidth=3, foreground=c + "33"),
                   pe.withStroke(linewidth=1, foreground=BG)]


def box(ax, x, y, w, h, label, color, fontsize=13, sublabel=None):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.18",
        facecolor=BG, edgecolor=color, linewidth=2.5, zorder=5))
    ax.text(x, y + (0.15 if sublabel else 0), label, color=color,
            fontsize=fontsize, ha="center", va="center",
            fontweight="bold", fontfamily="monospace", zorder=6,
            path_effects=GLOW(color))
    if sublabel:
        ax.text(x, y - 0.28, sublabel, color=DIM, fontsize=9,
                ha="center", va="center", fontfamily="monospace", zorder=6)


def arrow(ax, x1, y1, x2, y2, color, label=None, curved=False):
    style = f"arc3,rad={'0.15' if curved else '0'}"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.18",
                                color=color, lw=2.2, connectionstyle=style,
                                alpha=0.9), zorder=4)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, label, color=color, fontsize=8,
                ha="center", fontfamily="monospace", alpha=0.7, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(22, 12), dpi=150)
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-1.2, 11.5)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    # Grid
    for gx in np.arange(0, 22, 1):
        ax.axvline(gx, color=GRID, linewidth=0.3, alpha=0.4)
    for gy in np.arange(-1, 12, 1):
        ax.axhline(gy, color=GRID, linewidth=0.3, alpha=0.4)

    # ── Title ──
    ax.text(10.5, 11.0, "VOXTERM // DIARIZATION PIPELINE",
            color=CYAN, fontsize=22, ha="center", fontweight="bold",
            fontfamily="monospace", path_effects=GLOW(CYAN))

    # ── Row 1: Audio Capture ──
    # Waveform
    rng = np.random.RandomState(42)
    t = np.linspace(0, 3.5, 400)
    envelope = np.zeros(400)
    for s, e, a in [(0.2, 1.0, 0.7), (1.3, 2.5, 0.9), (2.7, 3.3, 0.6)]:
        m = (t >= s) & (t <= e)
        envelope[m] = a * (1 + 0.3 * np.sin(2*np.pi*3*t[m]))
    sig = envelope * (0.5*np.sin(2*np.pi*150*t) + 0.3*np.sin(2*np.pi*300*t)
                      + 0.12*rng.randn(400))
    wx = 0.5 + t * 1.3
    wy = 9.2 + sig * 0.55
    ax.plot(wx, wy, color=CYAN, linewidth=1.0, alpha=0.85, zorder=3)
    ax.fill_between(wx, 9.2, wy, color=CYAN, alpha=0.06)
    ax.text(0.5, 10.0, "MIC + SYSTEM AUDIO", color=TEXT, fontsize=10,
            fontfamily="monospace", fontweight="bold")
    ax.text(0.5, 8.45, "16kHz  mono  float32", color=DIM, fontsize=8,
            fontfamily="monospace")

    # Arrow to VAD
    arrow(ax, 5.2, 9.2, 6.3, 9.2, AMBER)

    # ── VAD ──
    box(ax, 8.0, 9.2, 3.0, 1.1, "SILERO VAD", AMBER, sublabel="ONNX · <1ms/frame")

    # VAD probability bars
    for i in range(22):
        idx = int(i * len(envelope) / 22)
        p = envelope[idx]
        c = GREEN if p > 0.3 else "#333333"
        bx = 6.65 + i * 0.12
        ax.bar(bx, p * 0.45, width=0.09, bottom=8.15, color=c, alpha=0.65, zorder=3)
    ax.axhline(8.38, xmin=0.295, xmax=0.44, color=AMBER, linewidth=0.8,
               linestyle="--", alpha=0.5)

    # Arrow to buffer
    arrow(ax, 9.6, 9.2, 11.0, 9.2, GREEN)

    # ── Audio Buffer ──
    box(ax, 12.8, 9.2, 3.2, 1.1, "AUDIO BUFFER", GREEN,
        sublabel="silence trigger · 0.8s")

    # Segment chips
    seg_colors = [CYAN, PINK, GREEN]
    seg_labels = ["seg 1", "seg 2", "seg 3"]
    for i in range(3):
        sx = 11.5 + i * 1.05
        ax.add_patch(FancyBboxPatch(
            (sx, 7.7), 0.85, 0.45, boxstyle="round,pad=0.06",
            facecolor=seg_colors[i], edgecolor=seg_colors[i],
            linewidth=1.5, alpha=0.25, zorder=5))
        ax.text(sx + 0.42, 7.93, seg_labels[i], color=seg_colors[i],
                fontsize=8, ha="center", fontfamily="monospace", fontweight="bold")
    ax.text(12.8, 7.35, "VAD-guided segment splitting", color=DIM,
            fontsize=8, ha="center", fontfamily="monospace")

    # ── Process boundary labels ──
    ax.text(0.3, 10.55, "MAIN PROCESS", color=CYAN, fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=CYAN, linewidth=1.5, alpha=0.8))
    ax.text(17.0, 10.55, "SUBPROCESS", color=PINK, fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=PINK, linewidth=1.5, alpha=0.8))

    # ── Fork arrows ──
    arrow(ax, 12.8, 8.5, 4.5, 6.2, LAVENDER, curved=True)
    arrow(ax, 12.8, 8.5, 17.0, 6.2, PINK, curved=True)

    # ── Qwen3-ASR (left) ──
    box(ax, 4.5, 5.5, 4.0, 1.8, "QWEN3-ASR", LAVENDER, fontsize=14,
        sublabel="MLX · Metal GPU · ~200ms")
    ax.text(4.5, 4.15, '"What are you guys\n doing over here?"',
            color=LAVENDER, fontsize=10, ha="center", fontfamily="monospace",
            fontstyle="italic", alpha=0.85)

    # ── CAM++ (right) ──
    box(ax, 17.0, 5.5, 4.0, 1.8, "CAM++", PINK, fontsize=14,
        sublabel="512-dim · PyTorch CPU · 0.013 RTF")

    # Embedding bars
    emb = np.random.RandomState(1).randn(40) * 0.5
    for i, v in enumerate(emb):
        bx = 15.15 + i * 0.095
        c = PINK if v > 0 else CYAN
        h = abs(v) * 0.35
        b = 4.0 if v > 0 else 4.0 - h
        ax.bar(bx, h, width=0.065, bottom=b, color=c, alpha=0.55, zorder=3)
    ax.text(17.0, 3.5, "speaker embedding (512-dim)", color=DIM,
            fontsize=8, ha="center", fontfamily="monospace")

    # ── Arrows to clustering ──
    arrow(ax, 17.0, 4.4, 10.5, 2.5, AMBER)
    arrow(ax, 4.5, 3.7, 4.5, 1.5, CYAN)

    # ── Online Clustering ──
    box(ax, 10.5, 1.8, 7.0, 2.2, "ONLINE CLUSTERING", AMBER, fontsize=15)

    features = [
        "running-sum centroids",
        "dual threshold  (match 0.55 / new 0.50)",
        "continuity bias + ambiguity detection",
        "spectral re-clustering  (eigengap auto-k)",
        "PLDA-lite whitening + Viterbi smoothing",
        "overlap protection  (skip blended updates)",
    ]
    for i, f in enumerate(features):
        ax.text(7.4, 2.45 - i * 0.3, f"  {f}", color=AMBER,
                fontsize=8, fontfamily="monospace", alpha=0.75, zorder=6)

    # Centroid dots
    centroids = [(14.8, 2.6, CYAN, "S1"), (15.8, 1.5, PINK, "S2"),
                 (16.5, 2.2, GREEN, "S3"), (15.2, 1.2, AMBER, "S4")]
    for cx, cy, cc, cl in centroids:
        ax.plot(cx, cy, "s", color=cc, markersize=12, alpha=0.8, zorder=7)
        ax.text(cx, cy - 0.35, cl, color=cc, fontsize=9,
                ha="center", fontfamily="monospace", fontweight="bold", zorder=7)
    # Incoming embedding dot
    ax.plot(15.5, 3.0, "o", color=PINK, markersize=9, alpha=0.9, zorder=7)
    ax.annotate("", xy=(15.8, 1.8), xytext=(15.5, 2.9),
                arrowprops=dict(arrowstyle="->,head_width=0.2",
                                color=PINK, lw=1.5, linestyle="--", alpha=0.6),
                zorder=4)

    # ── Arrow to output ──
    arrow(ax, 7.0, 1.0, 4.5, 0.0, BORDER)

    # ── Transcript Output ──
    ax.add_patch(FancyBboxPatch(
        (0.5, -1.1), 8.0, 2.4, boxstyle="round,pad=0.18",
        facecolor=BG, edgecolor=BORDER, linewidth=2.5, zorder=5))
    ax.text(4.5, 1.0, "TRANSCRIPT // OUTPUT", color=BORDER, fontsize=12,
            ha="center", fontfamily="monospace", fontweight="bold", zorder=6,
            path_effects=GLOW(BORDER))

    lines = [
        (CYAN,  "Speaker 1", "What are you guys doing over here?"),
        (PINK,  "Speaker 2", "Just working on the diarization thing."),
        (GREEN, "Speaker 3", "Did you remember to call your mother?"),
    ]
    for i, (c, spk, txt) in enumerate(lines):
        y = 0.5 - i * 0.5
        ax.text(1.0, y, spk, color=c, fontsize=10, fontfamily="monospace",
                fontweight="bold", zorder=6)
        ax.text(4.0, y, txt, color=TEXT, fontsize=9, fontfamily="monospace",
                zorder=6)

    # ── Model specs footer ──
    specs = [
        "CAM++ 7.18M params · 0.73% EER · VoxCeleb-trained",
        "Silero VAD 0.97 AUC · Spectral eigengap auto-k · PLDA whitening · VBx Viterbi",
    ]
    for i, s in enumerate(specs):
        ax.text(10.5, -0.7 - i * 0.3, s, color=DIM, fontsize=7,
                ha="center", fontfamily="monospace", alpha=0.6)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
