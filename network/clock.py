"""NTP-style clock synchronization for P2P peers.

Each peer maintains a ClockSync instance per connected peer.  Heartbeat
round-trips feed offset samples; the median of a sliding window gives
a robust estimate accurate to ~1ms on LAN.
"""

from __future__ import annotations

import logging
import statistics

log = logging.getLogger("p2p.clock")


class ClockSync:
    """Per-peer clock offset estimator using heartbeat round-trips.

    Usage::

        sync = ClockSync()

        # On heartbeat round-trip:
        #   t1 = my send time (monotonic)
        #   t2 = their receive time (their monotonic)
        #   t3 = my receive time (monotonic)
        sync.add_sample(t1, t2, t3)

        # Convert a remote timestamp to local clock:
        local_ts = sync.adjust(remote_ts)
    """

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._offsets: list[float] = []
        self._rtts: list[float] = []

    def add_sample(self, t1: float, t2: float, t3: float) -> None:
        """Record a heartbeat round-trip measurement.

        Args:
            t1: Local send time (time.monotonic on this node).
            t2: Remote receive time (time.monotonic on peer).
            t3: Local receive time (time.monotonic on this node).
        """
        rtt = t3 - t1
        if rtt < 0:
            log.warning("Discarding bogus clock sample: rtt=%.3fs (t1=%.3f t2=%.3f t3=%.3f)", rtt, t1, t2, t3)
            return  # bogus sample
        owl = rtt / 2.0
        offset = t2 - t1 - owl

        if rtt > 0.1:
            log.warning("High RTT clock sample: %.1fms", rtt * 1000)

        self._offsets.append(offset)
        self._rtts.append(rtt)
        log.debug("Clock sample: rtt=%.1fms offset=%.1fms (n=%d)", rtt * 1000, offset * 1000, len(self._offsets))

        # Trim to window
        if len(self._offsets) > self._window_size:
            self._offsets = self._offsets[-self._window_size :]
            self._rtts = self._rtts[-self._window_size :]

    @property
    def offset(self) -> float:
        """Median clock offset (remote - local).  0.0 if no samples."""
        if not self._offsets:
            return 0.0
        return statistics.median(self._offsets)

    @property
    def rtt(self) -> float:
        """Median round-trip time in seconds.  0.0 if no samples."""
        if not self._rtts:
            return 0.0
        return statistics.median(self._rtts)

    @property
    def sample_count(self) -> int:
        return len(self._offsets)

    def adjust(self, remote_ts: float) -> float:
        """Convert a remote peer's timestamp to this node's local clock."""
        return remote_ts - self.offset
