"""mDNS service advertisement and peer browsing via zeroconf.

Advertises the local VoxTerm instance as ``_voxterm._tcp.local.`` and
discovers other instances on the LAN.  The session code is never
broadcast — only ``in_session`` (0/1) is visible via mDNS.
"""

from __future__ import annotations

import logging
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

        self._zeroconf: Zeroconf | None = None
        self._browser: ServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None

        self._peers: dict[str, PeerInfo] = {}  # node_id → PeerInfo
        self._lock = threading.Lock()

        # Callbacks
        self.on_peer_found: Callable[[PeerInfo], None] | None = None
        self.on_peer_lost: Callable[[str], None] | None = None

    def start(self) -> None:
        """Register our mDNS service and start browsing for peers."""
        self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        self._service_info = self._build_service_info()
        try:
            self._zeroconf.register_service(self._service_info)
        except Exception as exc:
            log.error("mDNS service registration failed: %s", exc)
            raise
        self._browser = ServiceBrowser(
            self._zeroconf,
            P2P_SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )
        log.info("mDNS started: advertising %s on port %d", self._display_name, self._tcp_port)

    def stop(self) -> None:
        """Unregister service and close zeroconf."""
        if self._zeroconf and self._service_info:
            try:
                self._zeroconf.unregister_service(self._service_info)
            except Exception as exc:
                log.debug("mDNS unregister error: %s", exc)
        if self._browser:
            self._browser.cancel()
            self._browser = None
        if self._zeroconf:
            self._zeroconf.close()
            self._zeroconf = None
        log.info("mDNS stopped")

    def update_session_status(self, in_session: bool) -> None:
        """Update the mDNS TXT record to reflect session status."""
        self._in_session = in_session
        if self._zeroconf and self._service_info:
            try:
                new_info = self._build_service_info()
                self._zeroconf.update_service(new_info)
                self._service_info = new_info
                log.debug("mDNS session status updated: in_session=%s", in_session)
            except Exception as exc:
                log.warning("mDNS service update failed: %s", exc)

    def get_visible_peers(self) -> list[PeerInfo]:
        """Return a snapshot of currently visible peers."""
        with self._lock:
            return list(self._peers.values())

    def _build_service_info(self) -> ServiceInfo:
        """Build the ServiceInfo for mDNS registration."""
        local_ip = self._get_local_ip()
        # Include node_id suffix to prevent name collisions if two
        # users pick the same display name
        instance_name = f"{self._display_name}-{self._node_id[:6]}"
        return ServiceInfo(
            P2P_SERVICE_TYPE,
            f"{instance_name}._voxterm._tcp.local.",
            server=f"{instance_name}.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self._tcp_port,
            properties={
                "node_id": self._node_id,
                "display_name": self._display_name,
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
            if is_new and self.on_peer_found:
                log.info("Peer found: %s at %s:%d", peer.display_name, peer.ip, peer.tcp_port)
                self.on_peer_found(peer)
            elif not is_new:
                log.debug("Peer updated: %s at %s:%d", peer.display_name, peer.ip, peer.tcp_port)

        elif state_change == ServiceStateChange.Removed:
            # Try to figure out which peer was removed
            info = zeroconf.get_service_info(service_type, name)
            node_id = None
            if info and info.properties:
                node_id = info.properties.get(b"node_id", b"").decode("utf-8", errors="replace")
            if not node_id:
                # Fallback: match by service name
                with self._lock:
                    for nid, peer in self._peers.items():
                        if name.startswith(peer.display_name):
                            node_id = nid
                            log.debug("Peer removal: matched %s by service name prefix", node_id)
                            break
                if not node_id:
                    log.debug("Peer removal: could not identify node_id from service %s", name)
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
            peer = PeerInfo(
                node_id=node_id,
                display_name=display_name or (info.server.split(".")[0] if info.server else node_id[:8]),
                ip=ip,
                tcp_port=int(props.get(b"tcp_port", str(info.port).encode()).decode()),
                udp_port=int(props.get(b"udp_port", b"0").decode()),
                in_session=props.get(b"in_session", b"0") == b"1",
                proto_v=int(props.get(b"proto_v", b"1").decode()),
            )
            log.debug("Parsed peer: %s (%s) at %s:%d proto_v=%d",
                       peer.display_name, peer.node_id[:8], peer.ip, peer.tcp_port, peer.proto_v)
            return peer
        except (KeyError, ValueError, IndexError) as exc:
            log.debug("Failed to parse service info: %s", exc)
            return None

    @staticmethod
    def _get_local_ip() -> str:
        """Get this machine's LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # doesn't send data, just resolves route
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            log.warning("Could not determine LAN IP, falling back to 127.0.0.1")
            return "127.0.0.1"
