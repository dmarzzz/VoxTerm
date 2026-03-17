import math
import numpy as np
from collections import deque
from textual.widget import Widget
from textual.strip import Strip
from rich.segment import Segment
from rich.style import Style

_SR = 16000


class WaveformWidget(Widget):
    """Pixel-shader oscilloscope with pitch-mapped colour.

    Flat/dormant when silent — springs to life on voice detection.
    Colour encodes spectral centroid (pitch proxy):
        low pitch → orange/gold
        mid pitch → cyan
        high pitch → blue/magenta
    Half-block (▀) rendering at 2× vertical resolution with neon glow.
    """

    DEFAULT_CSS = """
    WaveformWidget {
        height: 15;
        border: heavy #6644cc;
        border-title-color: #aa66ff;
        border-title-style: bold;
        border-title-align: left;
        background: #060810;
        margin: 0 1;
    }
    """

    DISPLAY_SECONDS = 3.0
    GLOW_RADIUS = 4.5
    GLOW_POWER = 1.5
    NOISE_GATE = 0.003       # RMS below this → silence

    def __init__(self):
        super().__init__()
        self.border_title = "WAVEFORM // OFFLINE"
        self._tick_count = 0
        self._phase = 0.0
        self._scroll = 0.0
        self._recording = False
        self._ready_glow = 0.0          # animates 0→1 when recording starts
        self._frame: list[Strip] = []

        capacity = int(_SR * self.DISPLAY_SECONDS * 1.5)
        self._samples: deque = deque(maxlen=capacity)
        self._signal_age = 999

        self._style_cache: dict[tuple, Style] = {}
        self._bg = np.array([6, 8, 16], dtype=np.float32)

        # pitch → edge-colour gradient  (9-stop cyberpunk neon)
        #   deep-red → hot-magenta → violet → electric-blue →
        #   neon-cyan → acid-green → neon-yellow → hot-orange → fusion-pink
        self._ps = np.array([0.000, 0.125, 0.250, 0.375,
                             0.500, 0.625, 0.750, 0.875, 1.000])
        self._er = np.array([255.,  255.,  170.,  40.,
                              0.,   57.,  230.,  255.,  255.])
        self._eg = np.array([ 20.,   0.,   30., 120.,
                             255.,  255.,  255.,  120.,   0.])
        self._eb = np.array([ 60.,  170.,  255., 255.,
                             220.,   20.,   0.,   20.,  180.])

    # ── public API ────────────────────────────────────────────

    def set_recording(self, on: bool):
        """Toggle recording state — controls extinguished vs alive visuals."""
        self._recording = on
        if not on:
            self._ready_glow = 0.0
            self.border_title = "WAVEFORM // OFFLINE"
        else:
            self.border_title = "WAVEFORM // LIVE"

    def push_samples(self, chunk: np.ndarray):
        self._samples.extend(chunk.ravel().tolist())
        self._signal_age = 0

    def push_amplitude(self, _): pass
    def push_amplitudes(self, _): pass

    # ── style cache ───────────────────────────────────────────

    def _cached_style(self, fr, fg, fb, br, bg, bb):
        key = (fr >> 2, fg >> 2, fb >> 2, br >> 2, bg >> 2, bb >> 2)
        s = self._style_cache.get(key)
        if s is None:
            fh = f"#{(key[0]<<2):02x}{(key[1]<<2):02x}{(key[2]<<2):02x}"
            bh = f"#{(key[3]<<2):02x}{(key[4]<<2):02x}{(key[5]<<2):02x}"
            s = Style(color=fh, bgcolor=bh)
            self._style_cache[key] = s
        return s

    # ── frame builder ─────────────────────────────────────────

    def _build_frame(self):
        W = self.size.width
        H = self.size.height
        if W <= 0 or H <= 0:
            return

        PH = H * 2
        center_py = PH / 2.0
        # Scale waveform height by ready_glow: thin line when off, full when on
        expansion = 0.03 + 0.97 * self._ready_glow  # 3% → 100%
        max_half_px = (center_py - 1.0) * expansion
        phase = self._phase
        scroll = self._scroll
        has_signal = self._signal_age < 30

        # ── 1. amplitude + pitch per column ───────────────────
        amp = np.zeros(W, dtype=np.float32)
        pitch = np.full(W, 0.5, dtype=np.float32)      # default cyan

        if has_signal and len(self._samples) > 0:
            buf = np.array(self._samples, dtype=np.float32)
            n = len(buf)
            spw = max(1, int(_SR * self.DISPLAY_SECONDS) // W)
            need = spw * W

            if n >= need:
                block = buf[-need:].reshape(W, spw)
            else:
                block = np.zeros((W, spw), dtype=np.float32)
                usable = n - n % spw if n >= spw else 0
                if usable > 0:
                    cols = usable // spw
                    block[-cols:] = buf[-usable:].reshape(cols, spw)

            # RMS per column
            rms = np.sqrt(np.mean(block ** 2, axis=1))

            # noise gate
            rms[rms < self.NOISE_GATE] = 0.0

            # aggressive gain so normal speech fills the display
            amp = np.clip(np.power(np.maximum(rms, 0) * 45.0, 0.45), 0, 1)

            # spectral centroid → pitch (only where signal exists)
            voiced = rms > self.NOISE_GATE
            if np.any(voiced):
                ffts = np.fft.rfft(block[voiced], axis=1)
                mags = np.abs(ffts)
                num_bins = mags.shape[1]
                freqs = np.arange(num_bins, dtype=np.float32)
                centroid = (
                    np.sum(mags * freqs.reshape(1, -1), axis=1)
                    / (np.sum(mags, axis=1) + 1e-10)
                )
                pitch[voiced] = np.clip(centroid / (num_bins * 0.4), 0, 1)

            # spatial smooth — amplitude
            for _ in range(3):
                tmp = np.copy(amp)
                tmp[1:-1] = (amp[:-2] + amp[1:-1] * 2 + amp[2:]) / 4.0
                amp = tmp

            # spatial smooth — pitch
            for _ in range(3):
                tmp = np.copy(pitch)
                tmp[1:-1] = (pitch[:-2] + pitch[1:-1] * 2 + pitch[2:]) / 4.0
                pitch = tmp

        avg_amp = float(amp.mean())

        # ── 2. per-column colours from pitch ──────────────────
        ps = self._ps
        edge_r = np.interp(pitch, ps, self._er)
        edge_g = np.interp(pitch, ps, self._eg)
        edge_b = np.interp(pitch, ps, self._eb)

        glow_r = edge_r * 0.60
        glow_g = edge_g * 0.60
        glow_b = edge_b * 0.65

        int_r = edge_r * 0.18
        int_g = edge_g * 0.20
        int_b = np.maximum(edge_b * 0.45, 25)

        # ── 3. wave shape ─────────────────────────────────────
        x = np.arange(W, dtype=np.float32)
        wave = (
            0.65 * np.sin(x * 0.16 + scroll)
            + 0.22 * np.sin(x * 0.37 + scroll * 1.3 + 0.9)
            + 0.13 * np.sin(x * 0.61 + scroll * 0.8 + 2.3)
        )
        wmax = np.max(np.abs(wave))
        if wmax > 1e-6:
            wave /= wmax

        curve = amp * wave
        curve_y = center_py - curve * max_half_px

        # ── 4. pixel shader ───────────────────────────────────
        py = np.arange(PH, dtype=np.float32).reshape(-1, 1)
        cv = curve_y.reshape(1, -1)

        fill_top = np.minimum(center_py, cv)
        fill_bot = np.maximum(center_py, cv)
        inside = (py >= fill_top) & (py <= fill_bot)

        d_above = fill_top - py
        d_below = py - fill_bot
        dist = np.maximum(0, np.maximum(d_above, d_below))

        gr = (self.GLOW_RADIUS * expansion
              + 0.5 * math.sin(phase * 2.0) * (0.3 + avg_amp))
        glow = np.clip(1.0 - dist / max(gr, 0.1), 0, 1)
        glow = np.power(glow, self.GLOW_POWER)

        dist_from_curve = np.abs(py - cv)
        fill_w = np.abs(cv - center_py)
        depth = np.where(fill_w > 0.5, dist_from_curve / fill_w, 0.0)
        depth = np.clip(depth, 0, 1)
        brightness = np.power(np.clip(1.0 - depth, 0, 1), 0.6)

        # ── 5. colour compositing ─────────────────────────────
        bg_c = self._bg
        r = np.full((PH, W), bg_c[0], dtype=np.float32)
        g = np.full((PH, W), bg_c[1], dtype=np.float32)
        b = np.full((PH, W), bg_c[2], dtype=np.float32)

        # glow (per-column colour)
        outside = ~inside
        go = glow * outside
        r += (glow_r.reshape(1, -1) - bg_c[0]) * go
        g += (glow_g.reshape(1, -1) - bg_c[1]) * go
        b += (glow_b.reshape(1, -1) - bg_c[2]) * go

        # fill (per-column colour, bright at curve edge)
        ec_r = edge_r.reshape(1, -1)
        ec_g = edge_g.reshape(1, -1)
        ec_b = edge_b.reshape(1, -1)
        ic_r = int_r.reshape(1, -1)
        ic_g = int_g.reshape(1, -1)
        ic_b = int_b.reshape(1, -1)

        r_in = ec_r * brightness + ic_r * (1.0 - brightness)
        g_in = ec_g * brightness + ic_g * (1.0 - brightness)
        b_in = ec_b * brightness + ic_b * (1.0 - brightness)

        if avg_amp > 0.4:
            boost = min(1.0, (avg_amp - 0.4) / 0.4)
            r_in += (255 - r_in) * boost * 0.25
            g_in += (255 - g_in) * boost * 0.12
            b_in += (255 - b_in) * boost * 0.08

        r[inside] = r_in[inside]
        g[inside] = g_in[inside]
        b[inside] = b_in[inside]

        # center line — lightsaber ignition effect
        # Off: dark. On: blade extends from center outward with bright tip.
        cl = int(center_py)
        rg = self._ready_glow
        if 0 <= cl < PH:
            mid = W / 2.0
            # blade extends from center; reach = how far (in columns) it's gone
            reach = rg * mid  # 0 → half-width

            for cc in range(W):
                dist_from_mid = abs(cc - mid)

                if dist_from_mid > reach:
                    # outside blade — dark
                    r[cl, cc] = np.clip(r[cl, cc] + 2, 0, 255)
                    g[cl, cc] = np.clip(g[cl, cc] + 3, 0, 255)
                    b[cl, cc] = np.clip(b[cl, cc] + 2, 0, 255)
                    continue

                # how close to the tip (0 = hilt, 1 = tip)
                tip_t = dist_from_mid / max(reach, 1.0)

                # bright core: white-cyan, brighter near tip
                tip_boost = max(0.0, 1.0 - abs(tip_t - 1.0) * 4.0)  # spike at tip
                core = 0.6 + 0.4 * tip_t  # brighter toward tip
                pulse = 0.85 + 0.15 * math.sin(phase * 4.0 + cc * 0.05)

                base_r = 20 * core * pulse + 180 * tip_boost
                base_g = 255 * core * pulse + 55 * tip_boost
                base_b = 230 * core * pulse + 55 * tip_boost

                # add pitch-tinted glow when audio is present
                a = amp[cc] if cc < len(amp) else 0
                base_r += edge_r[cc] * a * 0.15
                base_g += edge_g[cc] * a * 0.20
                base_b += edge_b[cc] * a * 0.18

                r[cl, cc] = np.clip(r[cl, cc] + base_r * rg, 0, 255)
                g[cl, cc] = np.clip(g[cl, cc] + base_g * rg, 0, 255)
                b[cl, cc] = np.clip(b[cl, cc] + base_b * rg, 0, 255)

            # glow on rows adjacent to center line (lightsaber halo)
            if rg > 0.1:
                for offset in [-1, 1]:
                    adj = cl + offset
                    if 0 <= adj < PH:
                        halo_strength = rg * 0.35
                        for cc in range(W):
                            if abs(cc - mid) > reach:
                                continue
                            r[adj, cc] = np.clip(r[adj, cc] + 5 * halo_strength, 0, 255)
                            g[adj, cc] = np.clip(g[adj, cc] + 60 * halo_strength, 0, 255)
                            b[adj, cc] = np.clip(b[adj, cc] + 50 * halo_strength, 0, 255)

        # scanlines
        scanline = np.ones((PH, 1), dtype=np.float32)
        scanline[1::2] = 0.87
        r *= scanline; g *= scanline; b *= scanline

        # background grid
        grid = np.zeros((PH, W), dtype=bool)
        grid[::4, ::6] = True
        gm = grid & outside & (glow < 0.05)
        r[gm] += 8; g[gm] += 12; b[gm] += 10

        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        # ── 6. half-block render ──────────────────────────────
        frame: list[Strip] = []
        for cr in range(H):
            segs: list[Segment] = []
            ty = cr * 2
            by = min(cr * 2 + 1, PH - 1)
            for cc in range(W):
                style = self._cached_style(
                    int(r[ty, cc]), int(g[ty, cc]), int(b[ty, cc]),
                    int(r[by, cc]), int(g[by, cc]), int(b[by, cc]),
                )
                segs.append(Segment("▀", style))
            frame.append(Strip(segs))

        self._frame = frame

    # ── textual rendering ─────────────────────────────────────

    def render_line(self, y: int) -> Strip:
        if y < len(self._frame):
            return self._frame[y]
        return Strip.blank(self.size.width)

    def tick(self):
        self._tick_count += 1
        self._phase += 0.08
        if self._signal_age <= 1:
            self._scroll += 0.35
        else:
            self._scroll += 0.06
        self._signal_age += 1
        # animate ready-glow: lightsaber ignite / retract
        if self._recording:
            self._ready_glow = min(1.0, self._ready_glow + 0.05)  # ~1.3s ignition
        else:
            self._ready_glow *= 0.90  # smooth retraction
        self._build_frame()
        self.refresh()
