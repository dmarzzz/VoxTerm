"""PartyManager — owns all P2P/party-mode state and logic.

Extracted from ``app.py`` to keep the god class focused on audio/transcription.
Communicates with VoxTerm exclusively via callbacks — never reaches into
VoxTerm's internal state.

Lifecycle:
    1. Constructed once at VoxTerm startup.
    2. ``start_passive_discovery()`` — background mDNS to show nearby peers.
    3. ``toggle()`` — enter or leave party mode (bound to [N] key).
    4. ``shutdown()`` — cleanup on app exit (optional, os._exit reclaims all).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from enum import Enum
from typing import Callable, TYPE_CHECKING

from config import ConfigStore

if TYPE_CHECKING:
    from textual.app import App

log = logging.getLogger(__name__)


# ── P2P imports — optional, gracefully degrade ───────────────
try:
    from network.session import SessionManager
    from network.segments import TranscriptAssembler
    from network.crypto import generate_session_code, derive_session_key
    from network.debug import P2PDebugStats
    from network.discovery import PeerDiscovery
    from network.audio_stream import AudioStreamer
    from audio.merger import PeerAudioMixer
    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False


class PartyState(Enum):
    SOLO = "solo"           # not in party mode
    SCANNING = "scanning"   # pressed N, looking for groups
    JOINING = "joining"     # connecting to a group
    IN_PARTY = "in_party"   # connected, transcripts flowing


# Artisanally picked party colors — warm, vibrant, visually distinct.
# Indexed by hash of session code so each party gets its own vibe.
PARTY_COLORS = [
    ("#ff6eb4", "#ff9ed2"),  # hot pink
    ("#ff8c42", "#ffb380"),  # tangerine
    ("#a78bfa", "#c4b5fd"),  # lavender
    ("#34d399", "#6ee7b7"),  # mint
    ("#f472b6", "#f9a8d4"),  # rose
    ("#fbbf24", "#fcd34d"),  # gold
    ("#38bdf8", "#7dd3fc"),  # sky
    ("#fb7185", "#fda4af"),  # coral
    ("#4ade80", "#86efac"),  # lime
    ("#c084fc", "#d8b4fe"),  # violet
    ("#f97316", "#fdba74"),  # amber
    ("#2dd4bf", "#5eead4"),  # teal
]


def _party_color(session_code: str) -> tuple[str, str]:
    """Derive a (primary, light) color pair from the session code."""
    import hashlib
    h = int(hashlib.sha256(session_code.encode()).hexdigest(), 16) % len(PARTY_COLORS)
    return PARTY_COLORS[h]


class PartyManager:
    """Encapsulates all P2P/party-mode state and logic.

    Parameters
    ----------
    app : App
        The Textual app instance — used only for ``@work`` threading
        and ``call_from_thread``.  PartyManager never reads or writes
        app attributes directly; all communication goes through callbacks.
    config : ConfigStore
        Persistent config for display name.

    Callbacks (set by VoxTerm after construction):
        on_state_changed(state: PartyState)
        on_peer_joined(display_name: str)
        on_peer_left(node_id: str, display_name: str)
        on_transcript_received(text: str, speaker: str, peer_display_name: str)
        on_partial_received()
        on_debug(msg: str)
        on_party_color_changed(primary: str, light: str)
        on_party_colors_restored()
        on_peer_bloom()
        on_party_failed(error: str)
        on_peer_audio_frame(node_id_bytes: bytes, seq: int, timestamp: float, pcm_bytes: bytes)
    """

    def __init__(self, app: "App", config: ConfigStore):
        self._app = app
        self._config = config

        # Identity
        self._node_id: str = ""
        self._display_name: str = config.get("p2p_display_name") or os.getlogin()

        # State
        self._state = PartyState.SOLO
        self._color_pri = "#00ffcc"
        self._color_light = "#66ffd9"

        # Networking objects
        self._session_mgr: SessionManager | None = None
        self._discovery: PeerDiscovery | None = None
        self._audio_streamer: AudioStreamer | None = None if P2P_AVAILABLE else None
        self._peer_mixer: PeerAudioMixer | None = None if P2P_AVAILABLE else None
        self._send_queue: queue.Queue | None = None
        self._assembler = TranscriptAssembler() if P2P_AVAILABLE else None
        self._p2p_debug = P2PDebugStats() if P2P_AVAILABLE else None
        self._transcript_seq: int = 0
        self._audio_send_seq: int = 0

        # Callbacks — set by VoxTerm
        self.on_state_changed: Callable[[PartyState], None] | None = None
        self.on_peer_joined: Callable[[str], None] | None = None
        self.on_peer_left: Callable[[str, str], None] | None = None
        self.on_transcript_received: Callable[[str, str, str], None] | None = None
        self.on_partial_received: Callable[[], None] | None = None
        self.on_debug: Callable[[str], None] | None = None
        self.on_party_color_changed: Callable[[str, str], None] | None = None
        self.on_party_colors_restored: Callable[[], None] | None = None
        self.on_peer_bloom: Callable[[], None] | None = None
        self.on_party_failed: Callable[[str], None] | None = None
        self.on_peer_audio_frame: Callable[[bytes, int, float, bytes], None] | None = None

    # ── public properties ────────────────────────────────────────

    @property
    def state(self) -> PartyState:
        return self._state

    @property
    def color_pri(self) -> str:
        return self._color_pri

    @property
    def color_light(self) -> str:
        return self._color_light

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def session_mgr(self) -> "SessionManager | None":
        return self._session_mgr

    @property
    def discovery(self) -> "PeerDiscovery | None":
        return self._discovery

    @property
    def peer_mixer(self) -> "PeerAudioMixer | None":
        return self._peer_mixer

    @property
    def audio_streamer(self) -> "AudioStreamer | None":
        return self._audio_streamer

    @property
    def assembler(self) -> "TranscriptAssembler | None":
        return self._assembler

    @property
    def p2p_debug(self) -> "P2PDebugStats | None":
        return self._p2p_debug

    @property
    def send_queue(self) -> queue.Queue | None:
        return self._send_queue

    @property
    def is_in_party(self) -> bool:
        return self._state == PartyState.IN_PARTY

    @property
    def is_host(self) -> bool:
        return getattr(self, "_is_host", False)

    @property
    def is_available(self) -> bool:
        return P2P_AVAILABLE

    @property
    def transcript_seq(self) -> int:
        return self._transcript_seq

    @transcript_seq.setter
    def transcript_seq(self, value: int) -> None:
        self._transcript_seq = value

    @property
    def audio_send_seq(self) -> int:
        return self._audio_send_seq

    @audio_send_seq.setter
    def audio_send_seq(self, value: int) -> None:
        self._audio_send_seq = value

    # ── identity ─────────────────────────────────────────────────

    def _ensure_identity(self) -> None:
        """Generate a stable node_id and persist display name (once per app run)."""
        if not self._node_id:
            self._node_id = str(uuid.uuid4()).replace("-", "")[:16]
        if not self._config.get("p2p_display_name"):
            self._config.set("p2p_display_name", self._display_name)

    # ── passive discovery (on-mount) ─────────────────────────────

    def start_passive_discovery(self) -> None:
        """Start mDNS discovery on launch -- just show who is on the network.

        Must be called from a worker thread (blocking mDNS operations).
        Uses app.call_from_thread for UI callbacks.
        """
        if self._state != PartyState.SOLO:
            return
        try:
            self._stop_discovery()
            self._ensure_identity()
            self._discovery = PeerDiscovery(
                self._node_id,
                self._display_name or "voxterm",
                tcp_port=0,
                udp_port=0,
            )

            my_id = self._node_id

            def on_found(peer_info):
                if peer_info.node_id == my_id:
                    return
                self._app.call_from_thread(
                    self._fire_debug,
                    f"peer online: {peer_info.display_name} · {peer_info.ip}"
                    + (f" · v{peer_info.app_version}" if peer_info.app_version else "")
                )
                self._app.call_from_thread(self._fire_state_changed)

            def on_lost(node_id):
                self._app.call_from_thread(
                    self._fire_debug,
                    f"peer offline: {node_id[:8]}"
                )
                self._app.call_from_thread(self._fire_state_changed)

            self._discovery.on_peer_found = on_found
            self._discovery.on_peer_lost = on_lost
            self._discovery.start()

        except Exception as exc:
            log.warning("peer discovery failed: %s", exc)

    def _stop_discovery(self) -> None:
        """Stop mDNS discovery and clean up."""
        if self._discovery:
            self._discovery.stop()
            self._discovery = None

    # ── active discovery (with TCP port) ─────────────────────────

    def _start_discovery(self, tcp_port: int, on_peer_found=None) -> None:
        """Start mDNS discovery with the actual TCP port."""
        if self._discovery is not None:
            self._discovery.stop()
        self._ensure_identity()
        self._discovery = PeerDiscovery(
            self._node_id,
            self._display_name or "voxterm",
            tcp_port=tcp_port,
            udp_port=0,
        )
        if on_peer_found is not None:
            self._discovery.on_peer_found = on_peer_found
        self._discovery.start()

    # ── toggle (entry point for [N] key) ─────────────────────────

    def toggle(self) -> None:
        """Toggle party mode: enter if solo, leave if in party."""
        if not P2P_AVAILABLE:
            self._fire_debug("P2P unavailable -- install zeroconf and cryptography")
            return
        if self._state == PartyState.SOLO:
            self._enter_party_mode()
        else:
            self._leave_party_mode()

    # ── enter ─────────────────────────────────────────────────────

    def _enter_party_mode(self) -> None:
        """Start party mode: scan for groups, auto-join or host."""
        self._state = PartyState.SCANNING
        self._ensure_identity()
        self._fire_state_changed()

        groups = self._find_party_groups()
        if len(groups) == 1:
            self._state = PartyState.JOINING
            self._fire_state_changed()
            group = groups[0]
            self._join_party(group["session_code"], group["display_name"], group.get("party_color"))
        elif len(groups) == 0:
            # No groups found yet — schedule a rescan before hosting
            self._app.set_timer(0.8, self._rescan_or_host)
        else:
            groups.sort(key=lambda g: g["peer_count"], reverse=True)
            self._join_best_group(groups)

    def _rescan_or_host(self) -> None:
        """Rescan for party groups after a short delay; host if still none found."""
        if self._state != PartyState.SCANNING:
            return
        groups = self._find_party_groups()
        if groups:
            groups.sort(key=lambda g: g["peer_count"], reverse=True)
            self._join_best_group(groups)
        else:
            self._host_party()

    def _join_best_group(self, groups: list[dict]) -> None:
        """Join the best (largest) group from a list of discovered groups."""
        self._state = PartyState.JOINING
        self._fire_state_changed()
        group = groups[0]
        self._join_party(group["session_code"], group["display_name"], group.get("party_color"))

    def _find_party_groups(self) -> list[dict]:
        """Find active party groups from discovered peers."""
        if not self._discovery:
            return []
        peers = self._discovery.get_visible_peers()
        groups: dict[str, dict] = {}
        for p in peers:
            if not p.in_session or not p.session_code:
                continue
            if p.session_code not in groups:
                groups[p.session_code] = {
                    "session_code": p.session_code,
                    "display_name": p.group_name or p.display_name,
                    "party_color": p.party_color,
                    "peer_count": 1,
                    "ip": p.ip,
                    "tcp_port": p.tcp_port,
                }
            else:
                groups[p.session_code]["peer_count"] += 1
        return list(groups.values())

    def _host_party(self) -> None:
        """Create a new party -- you are the host."""
        code = generate_session_code()
        self._color_pri, self._color_light = _party_color(code)
        self._start_party_session(code, is_creator=True)

    def _join_party(self, session_code: str, group_name: str, peer_color: str | None = None) -> None:
        """Join an existing party by session code (read from mDNS)."""
        if peer_color:
            # Use the exact color the host is broadcasting — guaranteed match
            self._color_pri = peer_color
            self._color_light = peer_color  # close enough for the light variant
        else:
            self._color_pri, self._color_light = _party_color(session_code)
        self._start_party_session(session_code, is_creator=False)

    # ── session setup (runs in worker thread) ────────────────────

    def _start_party_session(self, code: str, is_creator: bool) -> None:
        """Start P2P session in a worker thread to avoid blocking the event loop.

        This method is decorated with @work on the app; we call it
        through the app's worker machinery.
        """
        self._app._party_start_session_worker(code, is_creator)

    def start_session_blocking(self, code: str, is_creator: bool) -> None:
        """The actual blocking session-start logic (called from a @work thread).

        This runs in a background thread and uses call_from_thread for UI updates.
        """
        try:
            # Cancel passive discovery worker (but keep the PeerDiscovery instance)
            self._app.workers.cancel_group(self._app, "p2p_discovery")

            old_mgr = self._session_mgr
            if old_mgr is not None:
                try:
                    old_mgr.leave_session()
                except Exception:
                    pass

            # Start audio streamer for multi-mic merging
            from config import P2P_AUDIO_MERGE_ENABLED
            audio_merge = P2P_AUDIO_MERGE_ENABLED and P2P_AVAILABLE
            node_id_bytes = self._node_id.encode("utf-8")[:16].ljust(16, b"\x00")
            session_key = derive_session_key(code)

            if audio_merge:
                streamer = AudioStreamer(node_id_bytes, session_key, udp_port=0)
                streamer.start()
                udp_audio_port = streamer.local_port

                mixer = PeerAudioMixer()
                self._audio_streamer = streamer
                self._peer_mixer = mixer
                self._audio_send_seq = 0

                def on_audio_frame(nid_bytes, seq, timestamp, pcm_bytes):
                    nid = nid_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
                    mixer.peer_frame(nid, seq, pcm_bytes)

                streamer.on_frame_received = on_audio_frame
            else:
                udp_audio_port = 0
                self._audio_streamer = None
                self._peer_mixer = None

            mgr = SessionManager(
                self._display_name, node_id=self._node_id, tcp_port=0,
                audio_merge=audio_merge, udp_audio_port=udp_audio_port,
            )
            self._session_mgr = mgr
            self._wire_session_callbacks()

            if is_creator:
                mgr.create_session()
                mgr._session_code = code
                mgr._session_key = derive_session_key(code)
            else:
                mgr.join_session(code)
            port = mgr._server_sock.getsockname()[1]

            # Bounded sender thread for P2P broadcast
            self._send_queue = queue.Queue(maxsize=64)
            send_q = self._send_queue

            def _p2p_sender_loop():
                while mgr._running:
                    try:
                        kwargs = send_q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    try:
                        mgr.broadcast_final(**kwargs)
                    except Exception:
                        pass

            threading.Thread(
                target=_p2p_sender_loop, daemon=True, name="p2p-sender"
            ).start()

            my_id = self._node_id

            def on_peer_found(peer_info):
                if peer_info.node_id == my_id:
                    return
                if mgr.has_peer(peer_info.node_id):
                    return
                if not peer_info.in_session:
                    return
                # Only connect to peers in the SAME party (same session code)
                if peer_info.session_code != code:
                    self._app.call_from_thread(
                        self._fire_debug,
                        f"{peer_info.display_name} is in a different party"
                    )
                    return
                self._app.call_from_thread(
                    self._fire_debug,
                    f"found {peer_info.display_name} -- "
                    + ("connecting..." if my_id < peer_info.node_id else "waiting...")
                )
                if my_id < peer_info.node_id:
                    threading.Thread(
                        target=self._try_connect_peer,
                        args=(peer_info,),
                        daemon=True,
                    ).start()

            # Reuse existing discovery if available, otherwise create fresh
            if self._discovery:
                self._discovery.on_peer_found = on_peer_found
                self._discovery.update_port(port)
            else:
                self._start_discovery(port, on_peer_found=on_peer_found)
            # Heal the mesh: a KNOWN peer that re-advertises — flips in_session when it
            # joins the party, or changes port/group — fires an Updated mDNS event, not
            # Added, so on_peer_found never sees it and the event was dropped, leaving a
            # permanent hole in a 3+ peer mesh. Wire the same gated connect logic to it.
            # (A peer that fully restarts comes back with a fresh node_id, i.e. an Added
            # event for on_peer_found; the periodic sweep below covers missed/failed
            # initial connects regardless of ordering.)
            self._discovery.on_peer_updated = on_peer_found
            # Advertise our session code + group name + party color via mDNS
            self._discovery.update_group(
                self._display_name, True, session_code=code,
                party_color=self._color_pri,
            )

            # Connect to any already-visible peers in the SAME party
            for peer_info in self._discovery.get_visible_peers():
                if (peer_info.in_session
                        and peer_info.session_code == code
                        and peer_info.node_id != my_id
                        and my_id < peer_info.node_id):
                    threading.Thread(
                        target=self._try_connect_peer,
                        args=(peer_info,),
                        daemon=True,
                    ).start()

            # Fallback retry after 3 seconds
            def _retry_connect():
                time.sleep(3.0)
                if not mgr._running:
                    return
                with mgr._lock:
                    has_peers = bool(mgr._peers)
                if has_peers:
                    return
                visible = self._discovery.get_visible_peers() if self._discovery else []
                for pi in visible:
                    if pi.in_session and pi.session_code == code and pi.node_id != my_id:
                        with mgr._lock:
                            if pi.node_id in mgr._peers:
                                continue
                        self._try_connect_peer(pi)

            threading.Thread(target=_retry_connect, daemon=True).start()

            # Periodic reconnect sweep. The 3s fallback above bails as soon as we hold
            # ANY peer, so in a 3+ peer mesh it can't recover a single late/dropped
            # peer. This idempotent sweep reconnects any same-party peer we should
            # initiate toward (my_id < theirs) but aren't connected to — and because
            # every node runs it, the lower-id side always initiates each missing edge,
            # so a peer that left and came back is re-meshed within one interval.
            def _reconnect_sweep():
                while mgr._running:
                    time.sleep(10.0)
                    # Capture once: _stop_discovery() can null self._discovery from
                    # another thread during the leave_session() window (mgr._running is
                    # still True until BYE + thread joins finish), so re-reading it would
                    # race. The try/except keeps a transient error from silently killing
                    # this long-lived healer thread.
                    disc = self._discovery
                    if not mgr._running or disc is None:
                        continue
                    try:
                        for pi in disc.get_visible_peers():
                            if (pi.in_session and pi.session_code == code
                                    and pi.node_id != my_id
                                    and my_id < pi.node_id
                                    and not mgr.has_peer(pi.node_id)):
                                self._try_connect_peer(pi)
                    except Exception:
                        log.warning("reconnect sweep iteration failed", exc_info=True)

            threading.Thread(
                target=_reconnect_sweep, daemon=True, name="p2p-reconnect"
            ).start()

            # Update state -- we are in the party
            self._app.call_from_thread(self._party_ready, is_creator)

        except Exception as exc:
            # If we already left (user pressed N again), don't report as failure
            if self._state == PartyState.SOLO:
                return
            self._stop_audio_merge()
            try:
                if self._session_mgr is not None:
                    self._session_mgr.leave_session()
            except Exception:
                pass
            self._session_mgr = None
            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
            log.warning("party session failed: %s", error_msg, exc_info=True)
            self._app.call_from_thread(self._handle_party_failed, error_msg)

    def _party_ready(self, is_host: bool) -> None:
        """Called on main thread when party session is ready."""
        self._state = PartyState.IN_PARTY
        self._is_host = is_host
        self._fire_state_changed()
        if self.on_party_color_changed:
            self.on_party_color_changed(self._color_pri, self._color_light)

    def _handle_party_failed(self, error: str) -> None:
        """Called on main thread when party session fails.

        If we failed trying to JOIN (likely stale mDNS entry), fall back
        to hosting instead of giving up.
        """
        if not self._is_host and self._state != PartyState.SOLO:
            # Tried to join a ghost party — host our own instead
            if self.on_debug:
                self.on_debug(f"join failed ({error}), hosting instead")
            self._host_party()
            return
        self._state = PartyState.SOLO
        if self.on_party_failed:
            self.on_party_failed(error)
        # Mark ourselves as not in session but keep discovery cache
        if self._discovery:
            self._discovery.update_session_status(False)
        else:
            self._app._party_start_passive_discovery_worker()

    # ── leave ─────────────────────────────────────────────────────

    def _leave_party_mode(self) -> None:
        """Leave the party and return to solo mode."""
        self._stop_audio_merge()
        old_mgr = self._session_mgr
        self._session_mgr = None
        self._send_queue = None
        self._state = PartyState.SOLO
        if self.on_party_colors_restored:
            self.on_party_colors_restored()
        self._fire_state_changed()
        # Mark not-in-session but keep discovery cache for fast rejoin
        if self._discovery:
            self._discovery.update_session_status(False)
        # Tear down the session in a background thread (no UI lag)
        if old_mgr:
            threading.Thread(
                target=PartyManager._teardown_session,
                args=(old_mgr,),
                daemon=True,
            ).start()

    @staticmethod
    def _teardown_session(mgr: "SessionManager") -> None:
        """Tear down a session manager in the background."""
        try:
            mgr.leave_session()
        except Exception:
            pass

    # ── peer connection ──────────────────────────────────────────

    def _try_connect_peer(self, peer_info) -> None:
        """Try connecting to a discovered peer (runs in background thread)."""
        mgr = self._session_mgr
        if not mgr or not mgr.is_in_session:
            return
        try:
            success = mgr.join_by_ip(
                peer_info.ip, peer_info.tcp_port,
                mgr.session_code,
            )
            if not success:
                self._app.call_from_thread(
                    self._fire_debug,
                    f"connection to {peer_info.display_name} failed (wrong session or unreachable)"
                )
        except Exception as exc:
            self._app.call_from_thread(
                self._fire_debug,
                f"connection error: {exc}"
            )

    # ── audio merge ──────────────────────────────────────────────

    def _stop_audio_merge(self) -> None:
        """Stop audio streamer and return remaining buffered chunks."""
        if self._audio_streamer:
            self._audio_streamer.stop()
            self._audio_streamer = None
        self._peer_mixer = None

    def flush_mixer(self) -> list:
        """Flush remaining buffered chunks from the peer mixer."""
        if self._peer_mixer:
            return self._peer_mixer.flush()
        return []

    # ── session callbacks wiring ─────────────────────────────────

    def _wire_session_callbacks(self) -> None:
        """Wire SessionManager callbacks so peer events flow through PartyManager."""
        mgr = self._session_mgr

        def _mixer_key(node_id: str) -> str:
            """Truncate node_id to match the 16-byte UDP wire format."""
            return node_id.encode("utf-8")[:16].rstrip(b"\x00").decode("utf-8", errors="replace")

        def on_connected(peer):
            # Register peer in the audio mixer for multi-mic merging
            if self._peer_mixer and peer.audio_merge_capable:
                self._peer_mixer.register_peer(_mixer_key(peer.node_id), peer.clock)
            self._app.call_from_thread(
                self._on_peer_connected, peer.display_name
            )

        def on_disconnected(node_id, display_name):
            # Remove peer from audio mixer
            if self._peer_mixer:
                self._peer_mixer.remove_peer(_mixer_key(node_id))
            if self._assembler:
                self._assembler.clear_peer(node_id)
            self._app.call_from_thread(
                self._on_peer_disconnected, node_id, display_name
            )

        def on_final(node_id, msg):
            if not self._assembler:
                return
            peers = mgr.peers
            peer = peers.get(node_id)
            clock_sync = peer.clock if peer else None
            seg = self._assembler.on_final(
                node_id, msg["seq"], msg["speaker_name"], msg["text"],
                msg["start_ts"], msg["end_ts"], msg["confidence"],
                clock_sync=clock_sync,
            )
            if seg is None:
                return  # duplicate segment, already displayed
            peer_name = msg.get("speaker_name", node_id[:8])
            display_name = peer.display_name if peer else node_id[:8]
            self._app.call_from_thread(
                self._on_final_transcript, msg["text"], peer_name, display_name,
            )

        def on_partial(node_id, msg):
            if not self._assembler:
                return
            peers = mgr.peers
            peer = peers.get(node_id)
            clock_sync = peer.clock if peer else None
            self._assembler.on_partial(
                node_id, msg["seq"], msg["speaker_name"], msg["text"],
                msg["start_ts"], clock_sync=clock_sync,
            )
            self._app.call_from_thread(self._on_partial_transcript)

        mgr.on_peer_connected = on_connected
        mgr.on_peer_disconnected = on_disconnected
        mgr.on_final_received = on_final
        mgr.on_partial_received = on_partial

    def _on_peer_connected(self, display_name: str) -> None:
        """Main-thread handler for peer connected event."""
        if self.on_peer_joined:
            self.on_peer_joined(display_name)
        if self.on_peer_bloom:
            self.on_peer_bloom()
        self._fire_state_changed()

    def _on_peer_disconnected(self, node_id: str, display_name: str) -> None:
        """Main-thread handler for peer disconnected event."""
        if self.on_peer_left:
            self.on_peer_left(node_id, display_name)
        self._fire_state_changed()

    def _on_final_transcript(self, text: str, speaker: str, peer_display_name: str) -> None:
        """Main-thread handler for incoming final transcript."""
        if self.on_transcript_received:
            self.on_transcript_received(text, speaker, peer_display_name)

    def _on_partial_transcript(self) -> None:
        """Main-thread handler for incoming partial transcript."""
        if self.on_partial_received:
            self.on_partial_received()

    # ── helpers ──────────────────────────────────────────────────

    def get_peer_udp_addrs(self) -> list[tuple[str, int]]:
        """Return (ip, udp_audio_port) for all connected peers that support audio merge."""
        mgr = self._session_mgr
        if not mgr:
            return []
        peers = mgr.peers
        return [
            (p.ip, p.udp_audio_port)
            for p in peers.values()
            if p.audio_merge_capable and p.udp_audio_port > 0
        ]

    def get_peer_names(self) -> dict[str, str]:
        """Build node_id -> display_name mapping from current peers."""
        if not self._session_mgr:
            return {}
        return {
            nid: p.display_name
            for nid, p in self._session_mgr.peers.items()
        }

    def get_visible_peer_count(self) -> int:
        """Number of peers visible via passive discovery."""
        if self._discovery:
            return len(self._discovery.get_visible_peers())
        return 0

    def enqueue_transcript(
        self, speaker: str, seq: int, text: str, timestamp: float,
    ) -> None:
        """Enqueue a local transcript segment for broadcast to peers."""
        if not self._send_queue:
            return
        try:
            self._send_queue.put_nowait(dict(
                speaker_name=speaker or self._display_name,
                seq=seq, text=text,
                start_ts=timestamp, end_ts=timestamp,
                confidence=0.9,
            ))
        except Exception:
            pass  # queue full -- drop rather than leak threads

    def track_local_segment(
        self, seq: int, speaker: str, text: str, timestamp: float,
        dominant_mic: str = "",
    ) -> None:
        """Track a local segment in the assembler for merged view."""
        if self._assembler:
            self._assembler.add_local(
                seq, speaker or self._display_name,
                text, timestamp, timestamp, confidence=0.9,
                dominant_mic=dominant_mic,
            )

    def format_debug_text(self, merged_view: bool) -> str:
        """Format P2P debug info for the debug overlay."""
        if self._p2p_debug and self._session_mgr and self._session_mgr.is_in_session:
            return self._p2p_debug.format_debug_text(
                self._session_mgr, mixer=self._peer_mixer,
                assembler=self._assembler, merged_view=merged_view,
            )
        return ""

    # ── telemetry text generation ────────────────────────────────

    _PEER_DOT_COLORS = [
        "#00ffcc", "#ff44aa", "#44ff88", "#ffaa44",
        "#bb88ff", "#ff8844", "#44ddff", "#ccff44",
        "#ff6eb4", "#38bdf8", "#4ade80", "#fbbf24",
    ]

    def telemetry_text(self) -> str:
        """Generate the P2P portion of the telemetry bar text."""
        pc = self._color_pri
        if self._state == PartyState.SCANNING:
            return f"    [{pc}]◌ looking for the party...[/]"
        elif self._state == PartyState.JOINING:
            return f"    [{pc}]joining the party...[/]"
        elif self._state == PartyState.IN_PARTY and self._session_mgr:
            peers = self._session_mgr.peers
            if peers:
                parts = []
                for idx, p in enumerate(peers.values()):
                    dot_color = self._PEER_DOT_COLORS[idx % len(self._PEER_DOT_COLORS)]
                    parts.append(f"[{dot_color}]●[/] {p.display_name}")
                names = "  ".join(parts)
                text = f"    {names}  [{pc}]●[/] you"
            else:
                text = f"    [{pc}]●[/] you"
            text += "    [dim]Leave Party \\[P][/]"
            return text
        elif self._discovery:
            visible = len(self._discovery.get_visible_peers())
            if visible > 0:
                return f"    [dim]{visible} nearby[/]  [#00e5ff]\\[P] Party[/]"
            else:
                return f"    [#00e5ff]\\[P] Party[/]"
        return ""

    # ── callback fire helpers ────────────────────────────────────

    def _fire_state_changed(self) -> None:
        if self.on_state_changed:
            self.on_state_changed(self._state)

    def _fire_debug(self, msg: str) -> None:
        if self.on_debug:
            self.on_debug(msg)

    # ── shutdown ─────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown -- optional, os._exit reclaims everything."""
        self._stop_audio_merge()
        if self._session_mgr:
            try:
                self._session_mgr.leave_session()
            except Exception:
                pass
        self._stop_discovery()
