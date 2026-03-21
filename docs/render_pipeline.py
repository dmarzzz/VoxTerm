#!/usr/bin/env python3
"""Render an animated pipeline diagram for VoxTerm's diarization system.

Generates frames with matplotlib, stitches with ffmpeg.
Output: docs/pipeline.mp4
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── constants ──────────────────────────────────────────────
FPS = 30
DURATION = 22  # seconds
W, H = 1920, 1080
DPI = 100
TOTAL_FRAMES = FPS * DURATION

# Cyberpunk palette (matches VoxTerm's theme)
BG = "#0a0e14"
CYAN = "#00ffcc"
PINK = "#ff44aa"
GREEN = "#44ff44"
AMBER = "#ffaa00"
LAVENDER = "#aa88ff"
DIM = "#1a2a2a"
GRID = "#0d1a1a"
TEXT = "#c0c0c0"
WHITE = "#ffffff"
BORDER = "#00e5ff"

# Timeline: (start_frame, end_frame, stage_name)
STAGES = [
    (0,   90,  "audio_in"),
    (90,  165, "vad"),
    (165, 240, "buffer_split"),
    (240, 330, "transcribe_embed"),
    (330, 420, "clustering"),
    (420, 510, "output"),
    (510, TOTAL_FRAMES, "full_pipeline"),
]


def ease_in_out(t):
    """Smooth easing function [0,1] -> [0,1]."""
    return t * t * (3 - 2 * t)


def stage_progress(frame, start, end):
    """Return 0..1 progress within a stage."""
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return ease_in_out((frame - start) / (end - start))


def draw_box(ax, x, y, w, h, label, color=CYAN, alpha=1.0, fontsize=11):
    """Draw a rounded box with label."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=BG, edgecolor=color,
        linewidth=2, alpha=alpha, zorder=5,
    )
    ax.add_patch(box)
    ax.text(x, y, label, color=color, fontsize=fontsize,
            ha="center", va="center", fontweight="bold",
            alpha=alpha, zorder=6,
            path_effects=[pe.withStroke(linewidth=1, foreground=BG)])


def draw_arrow(ax, x1, y1, x2, y2, color=CYAN, alpha=1.0):
    """Draw a glowing arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.3,head_length=0.15",
                    color=color, lw=2, alpha=alpha,
                ),
                zorder=4)


def make_waveform(n=500, seed=42):
    """Generate a fake speech waveform."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4, n)
    # Simulate speech envelope
    envelope = np.zeros(n)
    # Speech regions
    for start, end, amp in [(0.3, 1.2, 0.7), (1.5, 2.8, 0.9), (3.0, 3.8, 0.6)]:
        mask = (t >= start) & (t <= end)
        envelope[mask] = amp * (1 + 0.3 * np.sin(2 * np.pi * 3 * t[mask]))
    signal = envelope * (
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.15 * rng.randn(n)
    )
    return t, signal, envelope


def make_embedding_viz(n_dims=32, seed=1):
    """Generate fake embedding bar visualization."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_dims) * 0.5


def render_frame(frame_idx, fig, ax):
    """Render a single frame of the animation."""
    ax.clear()
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Subtle grid
    for gx in np.arange(0, 20, 1):
        ax.axvline(gx, color=GRID, linewidth=0.3, alpha=0.5)
    for gy in np.arange(0, 11, 1):
        ax.axhline(gy, color=GRID, linewidth=0.3, alpha=0.5)

    # Title
    ax.text(9.6, 10.2, "VOXTERM // DIARIZATION PIPELINE",
            color=CYAN, fontsize=18, ha="center", va="center",
            fontweight="bold", fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=2, foreground="#003333")])

    # ── STAGE 1: Audio Input ──────────────────────────────
    s1 = stage_progress(frame_idx, *STAGES[0][:2])
    if s1 > 0:
        # Waveform
        t, signal, envelope = make_waveform()
        vis_n = int(len(t) * s1)
        if vis_n > 2:
            wave_x = 1.0 + t[:vis_n] * 2.5
            wave_y = 8.5 + signal[:vis_n] * 0.6
            ax.plot(wave_x, wave_y, color=CYAN, linewidth=1.2, alpha=0.9, zorder=3)
            ax.fill_between(wave_x, 8.5, wave_y, color=CYAN, alpha=0.08)

        ax.text(1.0, 9.3, "MIC + SYSTEM AUDIO", color=DIM if s1 < 0.3 else TEXT,
                fontsize=9, fontfamily="monospace", alpha=min(1, s1 * 3))
        ax.text(1.0, 7.8, "16kHz mono float32", color=DIM,
                fontsize=8, fontfamily="monospace", alpha=min(1, s1 * 2))

    # ── STAGE 2: Silero VAD ───────────────────────────────
    s2 = stage_progress(frame_idx, *STAGES[1][:2])
    if s2 > 0:
        draw_arrow(ax, 3.8, 8.5, 5.0, 8.5, AMBER, alpha=min(1, s2 * 3))
        draw_box(ax, 6.5, 8.5, 2.8, 1.0,
                 "SILERO VAD\n(ONNX)", AMBER, alpha=min(1, s2 * 2), fontsize=10)

        # Show speech probability bars
        if s2 > 0.3:
            t, signal, envelope = make_waveform()
            bar_n = int(20 * min(1, (s2 - 0.3) / 0.7))
            for i in range(bar_n):
                idx = int(i * len(envelope) / 20)
                prob = envelope[idx]
                color = GREEN if prob > 0.3 else PINK
                bx = 5.3 + i * 0.12
                ax.bar(bx, prob * 0.5, width=0.08, bottom=7.6,
                       color=color, alpha=0.7, zorder=3)
            ax.text(5.3, 7.35, "speech probability", color=DIM,
                    fontsize=7, fontfamily="monospace")
            ax.axhline(7.85, xmin=0.27, xmax=0.41, color=AMBER,
                       linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(7.95, 7.82, "threshold=0.5", color=AMBER,
                    fontsize=7, fontfamily="monospace", alpha=0.7)

    # ── STAGE 3: Buffer + Split ───────────────────────────
    s3 = stage_progress(frame_idx, *STAGES[2][:2])
    if s3 > 0:
        draw_arrow(ax, 8.0, 8.5, 9.2, 8.5, GREEN, alpha=min(1, s3 * 3))

        # Buffer box
        draw_box(ax, 10.8, 8.5, 2.8, 1.0,
                 "AUDIO BUFFER\nsilence trigger", GREEN, alpha=min(1, s3 * 2), fontsize=10)

        # Show segments splitting
        if s3 > 0.5:
            seg_alpha = min(1, (s3 - 0.5) * 3)
            colors = [CYAN, PINK, GREEN]
            labels = ["seg 1", "seg 2", "seg 3"]
            for i in range(3):
                sx = 9.6 + i * 0.9
                ax.add_patch(FancyBboxPatch(
                    (sx, 7.2), 0.7, 0.4,
                    boxstyle="round,pad=0.05",
                    facecolor=colors[i], edgecolor=colors[i],
                    linewidth=1, alpha=seg_alpha * 0.3, zorder=5))
                ax.text(sx + 0.35, 7.4, labels[i], color=colors[i],
                        fontsize=7, ha="center", fontfamily="monospace",
                        alpha=seg_alpha)
            ax.text(10.8, 6.9, "VAD-guided splitting", color=DIM,
                    fontsize=7, ha="center", fontfamily="monospace", alpha=seg_alpha)

    # ── STAGE 4: Dual path — Transcribe + Embed ──────────
    s4 = stage_progress(frame_idx, *STAGES[3][:2])
    if s4 > 0:
        # Fork arrows
        draw_arrow(ax, 10.8, 7.9, 5.5, 5.5, LAVENDER, alpha=min(1, s4 * 2))
        draw_arrow(ax, 10.8, 7.9, 14.5, 5.5, PINK, alpha=min(1, s4 * 2))

        # Transcription box (left)
        draw_box(ax, 4.5, 5.0, 3.5, 1.5,
                 "QWEN3-ASR\ntranscription\n(MLX / Metal GPU)",
                 LAVENDER, alpha=min(1, s4 * 2), fontsize=10)

        # Embedding box (right)
        draw_box(ax, 14.5, 5.0, 3.5, 1.5,
                 "CAM++\n512-dim embedding\n(PyTorch / CPU)",
                 PINK, alpha=min(1, s4 * 2), fontsize=10)

        # Show embedding bars
        if s4 > 0.5:
            emb = make_embedding_viz(32, seed=1)
            emb_alpha = min(1, (s4 - 0.5) * 3)
            for i, v in enumerate(emb):
                bx = 13.0 + i * 0.1
                color = PINK if v > 0 else CYAN
                ax.bar(bx, abs(v) * 0.3, width=0.07,
                       bottom=3.8 if v > 0 else 3.8 - abs(v) * 0.3,
                       color=color, alpha=emb_alpha * 0.6, zorder=3)
            ax.text(14.5, 3.3, "speaker embedding (512-dim)",
                    color=DIM, fontsize=7, ha="center", fontfamily="monospace",
                    alpha=emb_alpha)

        # Show transcription text
        if s4 > 0.6:
            txt_alpha = min(1, (s4 - 0.6) * 4)
            ax.text(4.5, 3.8, '"What are you guys\n doing over here?"',
                    color=LAVENDER, fontsize=9, ha="center", va="center",
                    fontfamily="monospace", fontstyle="italic", alpha=txt_alpha)

    # ── STAGE 5: Clustering ───────────────────────────────
    s5 = stage_progress(frame_idx, *STAGES[4][:2])
    if s5 > 0:
        draw_arrow(ax, 14.5, 4.2, 9.6, 2.2, AMBER, alpha=min(1, s5 * 2))

        draw_box(ax, 9.6, 2.0, 5.5, 1.8,
                 "ONLINE CLUSTERING", AMBER,
                 alpha=min(1, s5 * 2), fontsize=12)

        # Clustering details
        if s5 > 0.3:
            det_alpha = min(1, (s5 - 0.3) * 2)
            details = [
                "running-sum centroids",
                "dual threshold (match 0.55 / new 0.50)",
                "continuity bias + conflict resolution",
                "spectral re-clustering (eigengap)",
                "PLDA-lite whitening + Viterbi smoothing",
            ]
            for i, d in enumerate(details):
                ax.text(7.2, 2.55 - i * 0.25, f"  {d}", color=AMBER,
                        fontsize=7, fontfamily="monospace", alpha=det_alpha * 0.8)

        # Centroid visualization
        if s5 > 0.5:
            c_alpha = min(1, (s5 - 0.5) * 3)
            centroids = [
                (13.0, 2.5, CYAN, "S1"),
                (13.8, 1.8, PINK, "S2"),
                (14.5, 2.3, GREEN, "S3"),
            ]
            # New embedding dot moving toward centroid
            t_anim = min(1, (s5 - 0.5) * 2)
            dot_x = 14.5 - t_anim * 0.7
            dot_y = 4.0 - t_anim * 1.7
            ax.plot(dot_x, dot_y, "o", color=PINK, markersize=8,
                    alpha=c_alpha, zorder=7)

            for cx, cy, cc, cl in centroids:
                ax.plot(cx, cy, "s", color=cc, markersize=10,
                        alpha=c_alpha, zorder=7)
                ax.text(cx, cy - 0.35, cl, color=cc, fontsize=8,
                        ha="center", fontfamily="monospace", alpha=c_alpha)

    # ── STAGE 6: Output ───────────────────────────────────
    s6 = stage_progress(frame_idx, *STAGES[5][:2])
    if s6 > 0:
        draw_arrow(ax, 4.5, 3.5, 4.5, 2.2, CYAN, alpha=min(1, s6 * 3))
        draw_arrow(ax, 9.6, 0.9, 4.5, 0.3, AMBER, alpha=min(1, s6 * 2))

        # Output transcript
        if s6 > 0.3:
            out_alpha = min(1, (s6 - 0.3) * 2)
            transcript_lines = [
                (CYAN,  "Speaker 1", "What are you guys doing over here?"),
                (PINK,  "Speaker 2", "Just working on the diarization..."),
                (GREEN, "Speaker 3", "Did you remember to call your mother?"),
            ]
            # Transcript box
            ax.add_patch(FancyBboxPatch(
                (1.0, -0.3), 7.0, 2.2,
                boxstyle="round,pad=0.15",
                facecolor=BG, edgecolor=BORDER,
                linewidth=2, alpha=out_alpha, zorder=5))
            ax.text(4.5, 1.65, "TRANSCRIPT // OUTPUT", color=BORDER,
                    fontsize=10, ha="center", fontfamily="monospace",
                    fontweight="bold", alpha=out_alpha, zorder=6)

            for i, (color, speaker, text) in enumerate(transcript_lines):
                line_alpha = min(1, max(0, (s6 - 0.3 - i * 0.15) * 3))
                if line_alpha > 0:
                    y = 1.2 - i * 0.55
                    ax.text(1.5, y, f"{speaker}", color=color,
                            fontsize=9, fontfamily="monospace",
                            fontweight="bold", alpha=line_alpha * out_alpha,
                            zorder=6)
                    ax.text(3.8, y, text, color=TEXT,
                            fontsize=8, fontfamily="monospace",
                            alpha=line_alpha * out_alpha, zorder=6)

    # ── STAGE 7: Full pipeline labels ─────────────────────
    s7 = stage_progress(frame_idx, *STAGES[6][:2])
    if s7 > 0:
        lab_alpha = min(1, s7 * 2)
        # Process labels
        ax.text(2.5, 9.8, "MAIN PROCESS", color=CYAN, fontsize=8,
                fontfamily="monospace", alpha=lab_alpha * 0.6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                          edgecolor=CYAN, alpha=lab_alpha * 0.3))
        ax.text(14.5, 6.8, "SUBPROCESS", color=PINK, fontsize=8,
                ha="center", fontfamily="monospace", alpha=lab_alpha * 0.6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                          edgecolor=PINK, alpha=lab_alpha * 0.3))

        # Timing annotations
        ax.text(6.5, 6.9, "<1ms/chunk", color=AMBER, fontsize=7,
                ha="center", fontfamily="monospace", alpha=lab_alpha * 0.5)
        ax.text(4.5, 6.0, "~200ms", color=LAVENDER, fontsize=7,
                ha="center", fontfamily="monospace", alpha=lab_alpha * 0.5)
        ax.text(14.5, 6.0, "~13ms RTF", color=PINK, fontsize=7,
                ha="center", fontfamily="monospace", alpha=lab_alpha * 0.5)

    # Frame counter
    ax.text(18.5, 0.3, f"frame {frame_idx}/{TOTAL_FRAMES}",
            color=DIM, fontsize=7, ha="right", fontfamily="monospace", alpha=0.4)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "pipeline.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(1, 1, figsize=(W / DPI, H / DPI), dpi=DPI)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        print(f"Rendering {TOTAL_FRAMES} frames at {FPS}fps...")
        for i in range(TOTAL_FRAMES):
            render_frame(i, fig, ax)
            frame_path = os.path.join(tmp, f"frame_{i:05d}.png")
            fig.savefig(frame_path, dpi=DPI, facecolor=BG,
                        pad_inches=0, bbox_inches="tight")
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{TOTAL_FRAMES} frames rendered")

        plt.close(fig)

        print("Encoding with ffmpeg...")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", os.path.join(tmp, "frame_%05d.png"),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0a0e14",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            out_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg stderr:", result.stderr[-500:])
            result.check_returncode()

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
