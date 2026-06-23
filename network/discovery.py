"""mDNS service advertisement and peer browsing via zeroconf.

Advertises the local VoxTerm instance as ``_voxterm._tcp.local.`` and
discovers other instances on the LAN.

The **session code is never broadcast** — it is the secret the AES-256-GCM
party key is derived from, so it must stay out-of-band. What *is* broadcast is
``party_id``: a random, non-secret group identifier the host generates per
party. Peers use it only to recognise who belongs to the same party (for
auto-reconnect/mesh-heal); it reveals nothing about the key. Joining still
requires the out-of-band code.
"""

from __future__ import annotations

import logging
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import Callable

from zeroconf import (
    IPVersion,
    ServiceBrowser,
    ServiceInfo,
    ServiceStateChange,
    Zeroconf,
)

from config import P2P_SERVICE_TYPE

log = logging.getLogger("p2p.discovery")


@dataclass
class PeerInfo:
    """Information about a discovered VoxTerm peer on the LAN."""

    node_id: str
    display_name: str
    ip: str
    tcp_port: int
    udp_port: int
    in_session: bool
    group_name: str = ""
    party_id: str = ""  # non-secret group id (NOT the session code / key material)
    party_color: str = ""
    app_version: str = ""
    proto_v: int = 1


class PeerDiscovery:
    """Manages mDNS advertisement and browsing for VoxTerm peers.

    Usage::

        disc = PeerDiscovery("my-node-id", "halcyon", tcp_port=9900, udp_port=9901)
        disc.on_peer_found = lambda info: print(f"Found {info.display_name}")
        disc.on_peer_lost = lambda node_id: print(f"Lost {node_id}")
        disc.start()
        ...
        disc.stop()
    """

    def __init__(
        self,
        node_id: str,
        display_name: str,
        tcp_port: int,
        udp_port: int,
    ):
        self._node_id = node_id
        self._display_name = display_name
        self._tcp_port = tcp_port
        self._udp_port = udp_port
        self._in_session = False
        self._group_name = ""
        self._party_id = ""
        self._party_color = ""
        # Import version at init time to avoid circular imports in mDNS callbacks
        from config import VERSION
        self._app_version = VERSION

        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None

        self._peers: dict[str, PeerInfo] = {}  # node_id → PeerInfo
        self._lock = threading.Lock()

        # Callbacks
        self.on_peer_found: Callable[[PeerInfo], None] | None = None
        self.on_peer_updated: Callable[[PeerInfo], None] | None = None
        self.on_peer_lost: Callable[[str], None] | None = None

    def start(self) -> None:
        """Register our mDNS service and start browsing for peers."""
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        self._service_info = self._build_service_info()
        self._zeroconf.register_service(self._service_info)
        self._browser = ServiceBrowser(
            self._zeroconf,
            P2P_SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )
        log.info("mDNS started: advertising %s on port %d", self._display_name, self._tcp_port)

    def stop(self) -> None:
        """Unregister service and close zeroconf."""
        if self._zeroconf and self._service_info:
            self._zeroconf.unregister_service(self._service_info)
        if self._browser:
            self._browser.cancel()
            self._browser = None
        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None
        log.info("mDNS stopped")

    def update_group(self, group_name: str, in_session: bool, party_id: str = "", party_color: str = "") -> None:
        """Update mDNS TXT record with group name and session status.

        ``party_id`` is the non-secret group identifier (never the session
        code). It is advertised so peers can recognise the same party.
        """
        self._group_name = group_name
        self._in_session = in_session
        self._party_id = party_id
        self._party_color = party_color
        if self._zeroconf and self._service_info:
            new_info = self._build_service_info()
            try:
                self._zeroconf.update_service(new_info)
                self._service_info = new_info
            except Exception:
                log.debug("Failed to update mDNS service")

    def update_session_status(self, in_session: bool) -> None:
        """Update the mDNS TXT record to reflect session status."""
        self._in_session = in_session
        if self._zeroconf and self._service_info:
            new_info = self._build_service_info()
            try:
                self._zeroconf.update_service(new_info)
                self._service_info = new_info
            except Exception:
                log.debug("Failed to update mDNS service (already unregistered?)")

    def update_port(self, tcp_port: int, udp_port: int = 0) -> None:
        """Update the advertised TCP/UDP ports."""
        self._tcp_port = tcp_port
        if udp_port:
            self._udp_port = udp_port

    def get_visible_peers(self) -> list[PeerInfo]:
        """Return a snapshot of currently visible peers."""
        with self._lock:
            return list(self._peers.values())

    @staticmethod
    def _sanitize_dns_label(name: str) -> str:
        """Sanitize a string for use as a DNS-SD instance name component."""
        # Replace non-alphanumeric chars (except hyphens) with hyphens
        sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", name)
        # Collapse multiple hyphens and strip leading/trailing
        sanitized = re.sub(r"-+", "-", sanitized).strip("-")
        return sanitized[:50] or "voxterm"  # DNS labels max 63 chars, leave room for suffix

    def _build_service_info(self) -> ServiceInfo:
        """Build the ServiceInfo for mDNS registration."""
        local_ip = self._get_local_ip()
        # Include node_id suffix to prevent name collisions if two
        # users pick the same display name
        safe_name = self._sanitize_dns_label(self._display_name)
        instance_name = f"{safe_name}-{self._node_id[:6]}"
        return ServiceInfo(
            P2P_SERVICE_TYPE,
            f"{instance_name}.{P2P_SERVICE_TYPE}",
            server=f"{instance_name}.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self._tcp_port,
            properties={
                "node_id": self._node_id,
                "display_name": self._display_name,
                "group_name": self._group_name,
                # party_id is a non-secret group id; the session code (key
                # material) is intentionally NOT advertised.
                "party_id": self._party_id,
                "party_color": self._party_color,
                "app_version": self._app_version,
                "in_session": "1" if self._in_session else "0",
                "proto_v": "1",
                "tcp_port": str(self._tcp_port),
                "udp_port": str(self._udp_port),
            },
        )

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        """Handle mDNS service state changes."""
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):
            info = zeroconf.get_service_info(service_type, name)
            if info is None:
                return
            peer = self._parse_service_info(info)
            if peer is None or peer.node_id == self._node_id:
                return  # skip self
            with self._lock:
                is_new = peer.node_id not in self._peers
                self._peers[peer.node_id] = peer
            if is_new:
                if self.on_peer_found:
                    log.info("Peer found: %s at %s:%d", peer.display_name, peer.ip, peer.tcp_port)
                    self.on_peer_found(peer)
            else:
                if self.on_peer_updated:
                    log.info("Peer updated: %s at %s:%d", peer.display_name, peer.ip, peer.tcp_port)
                    self.on_peer_updated(peer)

        elif state_change == ServiceStateChange.Removed:
            # Try to figure out which peer was removed
            info = zeroconf.get_service_info(service_type, name)
            node_id = None
            if info and info.properties:
                node_id = info.properties.get(b"node_id", b"").decode("utf-8", errors="replace")
            if not node_id:
                # Fallback: match by mDNS instance name which we control
                # Format: "{display_name}-{node_id[:6]}._voxterm._tcp.local."
                instance = name.split(".")[0]  # e.g. "bob-aabb11"
                with self._lock:
                    for nid, peer in self._peers.items():
                        sanitized_name = self._sanitize_dns_label(peer.display_name)
                        expected = f"{sanitized_name}-{nid[:6]}"
                        if instance == expected:
                            node_id = nid
                            break
            if node_id:
                with self._lock:
                    self._peers.pop(node_id, None)
                if self.on_peer_lost:
                    log.info("Peer lost: %s", node_id)
                    self.on_peer_lost(node_id)

    @staticmethod
    def _parse_service_info(info: ServiceInfo) -> PeerInfo | None:
        """Extract PeerInfo from a zeroconf ServiceInfo."""
        props = info.properties
        if not props:
            return None
        try:
            node_id = props[b"node_id"].decode("utf-8")
            ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
            if not ip:
                return None
            display_name = props.get(b"display_name", b"").decode("utf-8", errors="replace")
            return PeerInfo(
                node_id=node_id,
                display_name=display_name or (info.server.split(".")[0] if info.server else node_id[:8]),
                ip=ip,
                tcp_port=int(props.get(b"tcp_port", str(info.port).encode()).decode()),
                udp_port=int(props.get(b"udp_port", b"0").decode()),
                in_session=props.get(b"in_session", b"0") == b"1",
                group_name=(props.get(b"group_name") or b"").decode("utf-8", errors="replace"),
                party_id=(props.get(b"party_id") or b"").decode("utf-8", errors="replace"),
                party_color=(props.get(b"party_color") or b"").decode("utf-8", errors="replace"),
                app_version=(props.get(b"app_version") or b"").decode("utf-8", errors="replace"),
                proto_v=int(props.get(b"proto_v", b"1").decode()),
            )
        except (KeyError, ValueError, IndexError) as exc:
            log.debug("Failed to parse service info: %s", exc)
            return None

    @staticmethod
    def _get_local_ip() -> str:
        """Get this machine's LAN IP address."""
        # Try route-resolution first (works when there's a default route)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))  # doesn't send data, just resolves route
            return s.getsockname()[0]
        except Exception:
            pass
        finally:
            s.close()

        # Fallback: find the first non-loopback IPv4 address
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if not ip.startswith("127."):
                    return ip
        except Exception:
            pass
        return "127.0.0.1"
