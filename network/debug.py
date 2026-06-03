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

    def format_debug_text(
        self, session_mgr: "SessionManager",
        mixer=None, assembler=None, merged_view: bool = False,
    ) -> str:
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

        # Audio merge stats
        if mixer is not None:
            ms = mixer.get_stats()
            lines.append(
                f"  merge: {ms['peer_count']} peers, delay={ms['delay_ms']}ms, "
                f"merged={ms['merge_count']}, peer_contrib={ms['peer_contributions']}"
            )
            # Per-peer audio frame loss (sequence gaps filled with silence). This is
            # the real UDP loss signal — surfaced from the mixer, where it's measured.
            peer_frames = ms.get("peer_frames", {})
            peer_gaps = ms.get("peer_gaps", {})
            for nid, frames in peer_frames.items():
                gaps = peer_gaps.get(nid, 0)
                total = frames + gaps
                rate = (gaps / total * 100.0) if total else 0.0
                name = next(
                    (p["display_name"] for p in snap["peers"] if nid.startswith(p["node_id"])),
                    nid[:8],
                )
                lines.append(f"    {name:<12} audio: {frames} frames, {gaps} gaps ({rate:.1f}%)")
            # Live weight bars — shows which mic is dominant right now
            weights = ms.get("live_weights", {})
            if weights:
                # Build name map: __local__ → local name, node_ids → display names
                peer_info = snap["peers"]
                name_map = {"__local__": "you"}
                for p in peer_info:
                    # Match by prefix since live_weights uses full node_id
                    for nid in weights:
                        if nid != "__local__" and nid.startswith(p["node_id"]):
                            name_map[nid] = p["display_name"]
                            break

                bar_parts = []
                for nid, w in sorted(weights.items(), key=lambda x: -x[1]):
                    name = name_map.get(nid, nid[:8])
                    pct = int(w * 100)
                    bar_len = max(1, int(w * 20))
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    bar_parts.append(f"  {name:<12} {bar} {pct}%")
                lines.append("  ── mic weights ──")
                lines.extend(bar_parts)

        # Transcript assembler stats
        if assembler is not None:
            from network.segments import LOCAL_NODE_ID
            finals = assembler.get_finals()
            local_count = sum(1 for s in finals if s.node_id == LOCAL_NODE_ID)
            peer_count = len(finals) - local_count
            partial_count = assembler.partial_count
            # Per-source breakdown
            sources: dict[str, int] = {}
            for s in finals:
                key = "local" if s.node_id == LOCAL_NODE_ID else s.node_id[:8]
                sources[key] = sources.get(key, 0) + 1
            source_str = ", ".join(f"{k}={v}" for k, v in sources.items())
            view_str = "MERGED" if merged_view else "LOCAL"
            lines.append(
                f"  transcript: {len(finals)} finals ({local_count} local, {peer_count} peer), "
                f"{partial_count} partials, view={view_str}"
            )
            if source_str:
                lines.append(f"  sources: {source_str}")

        return "\n".join(lines)
