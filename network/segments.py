"""Transcript assembly — merges segments from multiple peers.

Each node independently assembles its own transcript.  No consensus,
no reconciliation.  Different nodes may produce different text.

The TranscriptAssembler maintains the merged transcript as an ordered
list of finalized segments, plus a set of in-progress partials (one
per peer at most).
"""

from __future__ import annotations

import bisect
import logging
import threading
from dataclasses import dataclass, field

from network.clock import ClockSync

log = logging.getLogger("p2p.segments")


@dataclass
class MergedSegment:
    """A transcript segment from any peer, with clock-adjusted timestamps."""

    node_id: str
    seq: int
    speaker_name: str
    text: str
    start_ts: float
    end_ts: float
    confidence: float
    is_partial: bool
    adjusted_start_ts: float = 0.0

    def __post_init__(self):
        if self.adjusted_start_ts == 0.0:
            self.adjusted_start_ts = self.start_ts


class TranscriptAssembler:
    """Merges finalized transcript segments from all peers into one timeline."""

    def __init__(self):
        self._finals: list[MergedSegment] = []
        self._partials: dict[str, MergedSegment] = {}  # node_id → latest partial
        self._lock = threading.Lock()

    def on_final(
        self,
        node_id: str,
        seq: int,
        speaker_name: str,
        text: str,
        start_ts: float,
        end_ts: float,
        confidence: float,
        clock_sync: ClockSync | None = None,
    ) -> MergedSegment:
        """Insert a finalized segment, sorted by adjusted start time."""
        adjusted = clock_sync.adjust(start_ts) if clock_sync else start_ts
        adjusted_end = clock_sync.adjust(end_ts) if clock_sync else end_ts

        seg = MergedSegment(
            node_id=node_id,
            seq=seq,
            speaker_name=speaker_name,
            text=text,
            start_ts=start_ts,
            end_ts=end_ts,
            confidence=confidence,
            is_partial=False,
            adjusted_start_ts=adjusted,
        )
        seg.end_ts = adjusted_end

        with self._lock:
            # Binary insert by adjusted_start_ts
            keys = [s.adjusted_start_ts for s in self._finals]
            idx = bisect.bisect_right(keys, adjusted)
            self._finals.insert(idx, seg)

            # Clear any pending partial for this node with same seq
            if node_id in self._partials and self._partials[node_id].seq == seq:
                del self._partials[node_id]

        log.debug("Final segment: node=%s seq=%d speaker=%s, %d chars, adjusted_start=%.3f",
                   node_id[:8], seq, speaker_name, len(text), adjusted)
        return seg

    def on_partial(
        self,
        node_id: str,
        seq: int,
        speaker_name: str,
        text: str,
        start_ts: float,
        clock_sync: ClockSync | None = None,
    ) -> MergedSegment:
        """Store/replace the in-progress partial for a peer."""
        adjusted = clock_sync.adjust(start_ts) if clock_sync else start_ts

        seg = MergedSegment(
            node_id=node_id,
            seq=seq,
            speaker_name=speaker_name,
            text=text,
            start_ts=start_ts,
            end_ts=start_ts,  # no end yet
            confidence=0.0,
            is_partial=True,
            adjusted_start_ts=adjusted,
        )
        with self._lock:
            self._partials[node_id] = seg
        log.debug("Partial segment: node=%s seq=%d speaker=%s, %d chars",
                   node_id[:8], seq, speaker_name, len(text))
        return seg

    def get_finals(self) -> list[MergedSegment]:
        """All finalized segments, ordered by adjusted start time."""
        with self._lock:
            return list(self._finals)

    def get_partials(self) -> list[MergedSegment]:
        """Current in-progress partials from all peers."""
        with self._lock:
            return list(self._partials.values())

    def clear_peer(self, node_id: str) -> None:
        """Remove pending partials for a disconnected peer."""
        with self._lock:
            self._partials.pop(node_id, None)
        log.debug("Cleared partials for peer %s", node_id[:8])

    @property
    def final_count(self) -> int:
        return len(self._finals)

    @property
    def partial_count(self) -> int:
        return len(self._partials)
