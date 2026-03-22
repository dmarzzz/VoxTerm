"""P2P debug stats collection for the debug overlay."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from network.session import SessionManager


class P2PDebugStats:
    """Collects a snapshot of P2P network stats for display."""

    def snapshot(self, session_mgr: "SessionManager") -> dict:
        """Return all P2P stats as a dict for debug display."""
        peers = session_mgr.peers
        peer_stats = []
        total_udp_rx = 0
        total_udp_tx = 0
        total_tcp_rx = 0
        total_tcp_tx = 0
        total_finals = 0
        total_partials = 0

        now = time.monotonic()

        for nid, peer in peers.items():
            latency_ms = peer.clock.rtt * 1000 if peer.clock.sample_count > 0 else None
            offset_ms = peer.clock.offset * 1000 if peer.clock.sample_count > 0 else None
            age = now - peer.last_heartbeat_recv

            peer_stats.append({
                "node_id": nid[:8],
                "display_name": peer.display_name,
                "state": peer.state,
                "latency_ms": latency_ms,
                "clock_offset_ms": offset_ms,
                "clock_samples": peer.clock.sample_count,
                "heartbeat_age_s": round(age, 1),
                "tcp_tx": peer.stats.tcp_tx,
                "tcp_rx": peer.stats.tcp_rx,
                "udp_tx": peer.stats.udp_tx,
                "udp_rx": peer.stats.udp_rx,
                "udp_dropped": peer.stats.udp_dropped,
                "finals_rx": peer.stats.finals_rx,
                "partials_rx": peer.stats.partials_rx,
            })

            total_tcp_rx += peer.stats.tcp_rx
            total_tcp_tx += peer.stats.tcp_tx
            total_udp_rx += peer.stats.udp_rx
            total_udp_tx += peer.stats.udp_tx
            total_finals += peer.stats.finals_rx
            total_partials += peer.stats.partials_rx

        return {
            "session_code": session_mgr.session_code,
            "in_session": session_mgr.is_in_session,
            "peer_count": len(peers),
            "peers": peer_stats,
            "totals": {
                "tcp_tx": total_tcp_tx,
                "tcp_rx": total_tcp_rx,
                "udp_tx": total_udp_tx,
                "udp_rx": total_udp_rx,
                "finals_rx": total_finals,
                "partials_rx": total_partials,
            },
        }

    def format_debug_text(self, session_mgr: "SessionManager") -> str:
        """Format P2P debug info as text for the transcript panel."""
        snap = self.snapshot(session_mgr)
        if not snap["in_session"]:
            return ""

        lines = [
            f"P2P: {snap['session_code']}  |  {snap['peer_count']} peers",
        ]

        for p in snap["peers"]:
            lat = f"{p['latency_ms']:.1f}ms" if p["latency_ms"] is not None else "?"
            off = f"{p['clock_offset_ms']:+.1f}ms" if p["clock_offset_ms"] is not None else "?"
            lines.append(
                f"  {p['display_name']:<12} lat={lat}  clk={off}  "
                f"rx={p['finals_rx']}F/{p['partials_rx']}P  {p['state']}"
            )

        t = snap["totals"]
        lines.append(f"  tcp tx/rx: {t['tcp_tx']}/{t['tcp_rx']}  finals rx: {t['finals_rx']}")

        return "\n".join(lines)
