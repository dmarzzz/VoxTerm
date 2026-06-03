"""Multi-source audio merger — energy-weighted mixing with jitter buffer.

Merges local audio with incoming peer audio streams for multi-mic
transcription.  Each source is weighted by its RMS energy so that
the mic closest to the active speaker dominates the mix.

When no peers are connected the merger is a zero-latency pass-through.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from audio.buffer import PeerAudioBuffer
from config import (
    P2P_AUDIO_QUALITY_GATE,
    P2P_MERGE_DELAY_MS,
    SAMPLE_RATE,
)
from network.clock import ClockSync


class PeerAudioMixer:
    """Delay-buffer + energy-weighted mixer for multi-mic audio merging.

    Local audio chunks are held for ``merge_delay_ms`` before being mixed
    with time-aligned peer audio and emitted.  When no peers are
    registered the delay is zero and chunks pass through immediately.

    Thread-safety: ``add_local_chunk`` is called from the audio timer
    (main thread).  ``peer_frame`` is called from the UDP receive thread.
    Peer buffers are internally locked; the delay deque is only touched
    by ``add_local_chunk``.
    """

    def __init__(self, merge_delay_ms: int = P2P_MERGE_DELAY_MS):
        self._merge_delay_s = merge_delay_ms / 1000.0
        # Deque of (local_ts, chunk) waiting to be merged
        self._delay_buf: deque[tuple[float, np.ndarray]] = deque()
        # Peer audio buffers keyed by node_id (str, hex-decoded)
        self._peer_buffers: dict[str, PeerAudioBuffer] = {}
        self._peer_clocks: dict[str, ClockSync] = {}
        self._lock = threading.Lock()  # protects _peer_buffers/_peer_clocks

        # Live weight tracking: node_id → rolling average weight (0.0–1.0)
        # "__local__" key for local mic. Updated every merge.
        self._live_weights: dict[str, float] = {}
        self._weight_alpha = 0.15  # EMA smoothing factor

        # Dominant source: which mic has the highest weight right now
        self._dominant_source: str = "__local__"

        # Counters for stats
        self._merge_count = 0
        self._peer_contributions = 0

    # ── public properties ────────────────────────────────────

    @property
    def merge_delay(self) -> float:
        """Current merge delay in seconds (0 when no peers)."""
        with self._lock:
            if not self._peer_buffers:
                return 0.0
        return self._merge_delay_s

    @property
    def dominant_source(self) -> str:
        """Node ID of the source with the highest current weight."""
        with self._lock:
            return self._dominant_source

    @property
    def peer_count(self) -> int:
        with self._lock:
            return len(self._peer_buffers)

    @property
    def active_peers(self) -> int:
        with self._lock:
            return len(self._peer_buffers)

    # ── peer management ──────────────────────────────────────

    def register_peer(self, node_id: str, clock: ClockSync) -> None:
        """Register a new peer for audio merging."""
        with self._lock:
            if node_id not in self._peer_buffers:
                self._peer_buffers[node_id] = PeerAudioBuffer()
                self._peer_clocks[node_id] = clock

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer (disconnected)."""
        with self._lock:
            self._peer_buffers.pop(node_id, None)
            self._peer_clocks.pop(node_id, None)

    def peer_frame(
        self, node_id: str, seq: int, pcm_int16: bytes,
    ) -> None:
        """Called from the UDP receive thread when a peer audio frame arrives."""
        with self._lock:
            buf = self._peer_buffers.get(node_id)
        if buf is not None:
            buf.write_frame(seq, pcm_int16)

    # ── main mixing entry point ──────────────────────────────

    def add_local_chunk(
        self, chunk: np.ndarray, local_ts: float,
    ) -> list[np.ndarray]:
        """Buffer a local audio chunk and return any merged chunks ready to emit.

        Returns a (possibly empty) list of merged chunks.  When no peers
        are connected, returns ``[chunk]`` immediately (zero delay).
        """
        delay = self.merge_delay
        if delay == 0.0:
            # No peers — pass through immediately
            return [chunk]

        # Buffer the chunk
        self._delay_buf.append((local_ts, chunk))

        # Emit any chunks whose delay has expired
        cutoff = local_ts - delay
        merged: list[np.ndarray] = []
        while self._delay_buf and self._delay_buf[0][0] <= cutoff:
            ts, local_chunk = self._delay_buf.popleft()
            merged.append(self._merge_chunk(local_chunk))
        return merged

    def flush(self) -> list[np.ndarray]:
        """Flush all remaining buffered chunks (e.g. on session end)."""
        merged: list[np.ndarray] = []
        while self._delay_buf:
            _, local_chunk = self._delay_buf.popleft()
            merged.append(self._merge_chunk(local_chunk))
        return merged

    # ── stats / debug ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return stats for debug overlay."""
        with self._lock:
            return {
                "peer_count": len(self._peer_buffers),
                "delay_ms": int(self._merge_delay_s * 1000),
                "merge_count": self._merge_count,
                "peer_contributions": self._peer_contributions,
                "buffered_chunks": len(self._delay_buf),
                "live_weights": dict(self._live_weights),
                "peer_frames": {
                    nid: buf.frames_received
                    for nid, buf in self._peer_buffers.items()
                },
                # Per-peer frame loss: sequence gaps filled with silence (buffer.py).
                # This is the real audio-loss signal — exported so the debug overlay
                # can show a gap rate instead of a permanently-zero counter.
                "peer_gaps": {
                    nid: buf.gaps_filled
                    for nid, buf in self._peer_buffers.items()
                },
            }

    def debug_info(self) -> dict:
        """Return merge stats for the debug overlay (legacy compat)."""
        with self._lock:
            peer_ids = list(self._peer_buffers.keys())
            peer_frames = {
                nid: buf.frames_received
                for nid, buf in self._peer_buffers.items()
            }
            peer_gaps = {
                nid: buf.gaps_filled
                for nid, buf in self._peer_buffers.items()
            }
        return {
            "peer_count": len(peer_ids),
            "merge_delay_ms": int(self._merge_delay_s * 1000),
            "buffered_chunks": len(self._delay_buf),
            "peer_frames": peer_frames,
            "peer_gaps": peer_gaps,
            "live_weights": dict(self._live_weights),
        }

    # ── internal mixing ──────────────────────────────────────

    def _merge_chunk(self, local_chunk: np.ndarray) -> np.ndarray:
        """Energy-weighted merge of local chunk with peer audio."""
        chunk_duration = len(local_chunk) / SAMPLE_RATE

        # Collect all sources: (node_id, chunk)
        sources: list[tuple[str, np.ndarray]] = [("__local__", local_chunk)]

        with self._lock:
            for node_id, buf in self._peer_buffers.items():
                peer_chunk = buf.read(chunk_duration)
                if len(peer_chunk) >= len(local_chunk):
                    sources.append((node_id, peer_chunk[:len(local_chunk)]))
                elif len(peer_chunk) > 0:
                    # Pad short peer chunk with silence
                    padded = np.zeros_like(local_chunk)
                    padded[:len(peer_chunk)] = peer_chunk
                    sources.append((node_id, padded))
                else:
                    # No data from this peer — skip
                    continue

        self._merge_count += 1

        if len(sources) == 1:
            self._update_live_weights([("__local__", 1.0)])
            return local_chunk

        self._peer_contributions += 1

        # Compute per-source RMS and weights
        weights = np.empty(len(sources), dtype=np.float32)
        for i, (nid, src) in enumerate(sources):
            rms = float(np.sqrt(np.mean(src ** 2)))
            if rms < P2P_AUDIO_QUALITY_GATE:
                weights[i] = 0.0
            else:
                weights[i] = np.sqrt(rms)

        total_weight = weights.sum()
        if total_weight < 1e-8:
            # All sources are silence — return local as-is
            self._update_live_weights([(nid, 0.0) for nid, _ in sources])
            return local_chunk

        weights /= total_weight

        # Track live weights
        self._update_live_weights(
            [(nid, float(w)) for (nid, _), w in zip(sources, weights)]
        )

        # Weighted average
        mixed = np.zeros_like(local_chunk)
        for i, (nid, src) in enumerate(sources):
            if weights[i] > 0:
                mixed += weights[i] * src

        # Gentle boost to compensate for averaging, then clip
        return np.clip(mixed * 1.2, -1.0, 1.0)

    def _update_live_weights(self, source_weights: list[tuple[str, float]]) -> None:
        """Update EMA-smoothed live weights for each source."""
        a = self._weight_alpha
        seen = set()
        for nid, w in source_weights:
            seen.add(nid)
            prev = self._live_weights.get(nid, 0.0)
            self._live_weights[nid] = prev * (1 - a) + w * a
        # Decay sources not present in this merge
        for nid in list(self._live_weights):
            if nid not in seen:
                self._live_weights[nid] *= (1 - a)
                if self._live_weights[nid] < 0.001:
                    del self._live_weights[nid]
        # Track dominant source
        if self._live_weights:
            self._dominant_source = max(self._live_weights, key=self._live_weights.get)
