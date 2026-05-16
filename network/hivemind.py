"""Hivemind transcript-sink discovery + publisher.

Voxterm pushes transcript batches to a swf-node convent box that
advertises itself on the LAN as ``_sr-hivemind._tcp.local.``. The wire
contract is locked in `SHAPE-ROTATOR-OS-SPEC.md` §4.3 (in
`searxng-wth-frnds` / `shape-rotator-wrld-knwldge-viz`).

Voxterm clients **DO NOT sign or encrypt** — the convent-box sink
resigns the wrapped bundle. The only client-side identifier on the
batch is `origin_device`: a v4 UUID generated on first launch and
persisted to the voxterm data dir. Per spec §3.5 it is opaque
provenance metadata, not a cryptographic identity.

Public surface:

    HIVEMIND_SERVICE_TYPE        – mDNS service type (15-byte name)
    Sink                         – discovered hivemind sink record
    HivemindBrowser              – mDNS browser; tracks active sink
    HivemindClient               – batches segments, POSTs to a sink
    get_or_create_device_id()    – persistent v4 UUID for this voxterm
    HivemindMode                 – CLI flag: auto / on / off
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("voxterm.hivemind")

#: 15-byte mDNS service name (post-amendment per swf-node PR #111).
#: RFC 6335 caps DNS-SD service names at 15 bytes; the original
#: ``shape-rotator-hivemind`` was 22 bytes and zeroconf rejected it.
HIVEMIND_SERVICE_TYPE = "_sr-hivemind._tcp.local."

#: Flush triggers from spec §4.3 — every ~60s OR every 30 segments
#: OR on EOF, whichever comes first.
FLUSH_SECONDS_DEFAULT = 60.0
FLUSH_SEGMENTS_DEFAULT = 30

#: Per-batch HTTP timeout. Generous so a Wi-Fi handoff doesn't drop the
#: batch — we'd rather wait than lose the segments.
POST_TIMEOUT_SECONDS = 10.0

#: Max body size we'll send. Mirrors swf-node's 1 MiB cap on
#: ``/hivemind/transcripts``. Normal transcription doesn't approach
#: this (30 segs × low-kB each) but a runaway producer shouldn't be
#: able to OOM the sink.
MAX_BATCH_BYTES = 1 * 1024 * 1024

#: How long ``HivemindMode.ON`` will wait for mDNS discovery before
#: failing to start. Per task spec.
DISCOVERY_TIMEOUT_SECONDS = 5.0

#: ``HivemindMode.AUTO`` waits briefly for an initial sync discovery
#: so the startup banner has something to print. If no sink lands in
#: this window the browser keeps running async; async discoveries are
#: logged when they fire so users tailing voxterm.log see them.
AUTO_INITIAL_DISCOVERY_WAIT_SECONDS = 2.0

#: Heartbeat cadence: log an INFO summary every Nth successful batch
#: so a user tailing voxterm.log sees ongoing health without being
#: drowned in per-batch lines. Per-batch detail moves to DEBUG.
HEARTBEAT_EVERY_N_BATCHES = 5


class HivemindMode(str, Enum):
    """``--hivemind=auto|on|off``.

    ``AUTO``  Discover via mDNS; fall back to local logging if none found.
    ``ON``    Require a sink (via ``--hivemind-sink-url`` or mDNS in
              ``DISCOVERY_TIMEOUT_SECONDS``); raise otherwise.
    ``OFF``   Never POST; everything stays local.
    """

    AUTO = "auto"
    ON = "on"
    OFF = "off"

    @classmethod
    def parse(cls, value: str | None) -> "HivemindMode":
        if value is None:
            return cls.AUTO
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"invalid hivemind mode {value!r}; "
                f"expected one of {', '.join(m.value for m in cls)}"
            )


@dataclass(frozen=True)
class Sink:
    """A discovered hivemind sink (or one given via CLI flag).

    ``pubkey`` is the stable pin the convent advertises in TXT
    records; we surface it for humans/UIs but voxterm itself doesn't
    verify against it (we don't sign or read bundles).
    """

    host: str
    port: int
    pubkey: str = ""
    node: str = ""
    proto: str = "shape-rotator-hivemind/v1"
    version: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def transcripts_url(self) -> str:
        return f"{self.base_url}/hivemind/transcripts"

    @classmethod
    def from_url(cls, url: str) -> "Sink":
        """Build a Sink from a user-supplied ``--hivemind-sink-url``.

        Accepts ``http://host:port`` or ``http://host:port/anything``.
        We always POST to ``{base}/hivemind/transcripts`` so any path
        in the URL is ignored (and logged).
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(f"invalid hivemind-sink-url: {url!r}")
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"hivemind-sink-url must be http(s): got {parsed.scheme!r}"
            )
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        if parsed.path and parsed.path not in ("", "/"):
            log.debug("ignoring path component of sink url: %s", parsed.path)
        return cls(host=parsed.hostname, port=port, node="manual")


@dataclass
class _PendingSegment:
    t: float
    speaker: str
    text: str


# ── persistent device id ────────────────────────────────────────────────


def _default_device_id_path() -> Path:
    """Where the persistent v4 UUID lives.

    Per the task spec, voxterm should put this in its config dir. We
    re-use the existing voxterm data dir from ``config.py`` so we
    don't introduce a parallel "config dir" concept for one file.
    """
    try:
        from config import DATA_DIR  # type: ignore
        return Path(DATA_DIR) / "device_id"
    except Exception:
        # Fallback — only hit in tests that import this module without
        # the rest of voxterm being importable.
        return Path.home() / ".config" / "voxterm" / "device_id"


def get_or_create_device_id(path: Optional[Path] = None) -> str:
    """Return the persistent v4 UUID for this voxterm install.

    Generated on first call; persisted to ``path`` (or the platform
    default). Subsequent calls return the same value. Per spec §3.5
    this is opaque provenance — not a cryptographic identity.
    """
    target = path or _default_device_id_path()
    try:
        existing = target.read_text(encoding="utf-8").strip()
        # Validate that the file actually contains a parseable UUID
        # so a corrupt/empty file gets regenerated rather than POSTed
        # as garbage to the sink.
        uuid.UUID(existing, version=4)
        return existing
    except (FileNotFoundError, ValueError):
        pass
    except Exception as exc:  # permission denied, IO error, etc.
        log.warning("could not read device_id at %s: %s", target, exc)

    new_id = str(uuid.uuid4())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write so a crash mid-write doesn't leave a partial
        # file we'd later treat as invalid.
        tmp = target.with_suffix(".tmp")
        tmp.write_text(new_id + "\n", encoding="utf-8")
        os.replace(tmp, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        log.info("generated voxterm device_id at %s", target)
    except Exception as exc:
        log.warning("could not persist device_id to %s: %s", target, exc)
    return new_id


# ── mDNS browser ────────────────────────────────────────────────────────


class HivemindBrowser:
    """Browses the LAN for hivemind sinks.

    The active sink is the most-recently-advertised one. UI consumers
    can pass an ``on_change`` callback to be notified from the
    zeroconf thread; they must marshal back to their own event loop.
    """

    def __init__(
        self,
        on_change: Optional[Callable[[list[Sink]], None]] = None,
    ) -> None:
        self._on_change = on_change
        self._sinks: dict[str, Sink] = {}  # service-name → Sink
        # Insertion-ordered (Python 3.7+); the most recent advertisement
        # ends up last and is treated as active.
        self._lock = threading.Lock()
        self._zc = None
        self._browser = None
        self._started = threading.Event()

    def start(self) -> None:
        try:
            from zeroconf import IPVersion, ServiceBrowser, Zeroconf
        except ImportError:
            log.warning("zeroconf unavailable — hivemind discovery disabled")
            return

        try:
            self._zc = Zeroconf(ip_version=IPVersion.V4Only)
            self._browser = ServiceBrowser(
                self._zc,
                HIVEMIND_SERVICE_TYPE,
                handlers=[self._on_state_change],
            )
            self._started.set()
        except OSError as exc:
            log.warning("hivemind browser failed to start: %s", exc)
            self._zc = None
            self._browser = None

    def stop(self) -> None:
        try:
            if self._browser is not None:
                self._browser.cancel()
        except Exception:
            pass
        try:
            if self._zc is not None:
                self._zc.close()
        except Exception:
            pass
        self._zc = None
        self._browser = None
        self._started.clear()

    def sinks(self) -> list[Sink]:
        """Return all currently-known sinks (insertion order)."""
        with self._lock:
            return list(self._sinks.values())

    def active_sink(self) -> Sink | None:
        """Return the most-recently-advertised sink, or None."""
        with self._lock:
            if not self._sinks:
                return None
            # Last insertion in dict ordering = most recent.
            return next(reversed(self._sinks.values()))

    def wait_for_sink(self, timeout: float) -> Sink | None:
        """Block until a sink is discovered or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sink = self.active_sink()
            if sink is not None:
                return sink
            time.sleep(0.1)
        return self.active_sink()

    # ── zeroconf glue ────────────────────────────────────────────────

    def _on_state_change(self, zeroconf, service_type, name, state_change):
        from zeroconf import ServiceStateChange

        if state_change in (
            ServiceStateChange.Added,
            ServiceStateChange.Updated,
        ):
            try:
                info = zeroconf.get_service_info(service_type, name)
            except Exception:
                info = None
            if info is None:
                return
            sink = self._parse(info)
            if sink is None:
                return
            with self._lock:
                # Re-insert to make it last (most recent) regardless of
                # whether this is an Added or Updated event.
                self._sinks.pop(name, None)
                self._sinks[name] = sink
            self._fire_change()
        elif state_change == ServiceStateChange.Removed:
            with self._lock:
                fire = self._sinks.pop(name, None) is not None
            if fire:
                self._fire_change()

    def _fire_change(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(self.sinks())
        except Exception as exc:
            log.warning("hivemind on_change callback raised: %s", exc)

    @staticmethod
    def _parse(info) -> Sink | None:
        """Extract a Sink from a zeroconf ServiceInfo. Returns None on
        a misconfigured advertisement rather than raising into the
        zeroconf thread."""
        try:
            props = info.properties or {}

            def _txt(key: str) -> str:
                v = props.get(key.encode("ascii"))
                if v is None:
                    return ""
                if isinstance(v, bytes):
                    return v.decode("utf-8", errors="replace").strip()
                return str(v).strip()

            addrs = info.addresses or []
            if not addrs:
                return None
            host = socket.inet_ntoa(addrs[0])

            # TXT record `port` overrides the SRV port if present;
            # swf-node advertises both as a defensive duplicate.
            port = int(_txt("port") or info.port or 0)
            if port <= 0:
                return None

            return Sink(
                host=host,
                port=port,
                pubkey=_txt("pubkey"),
                node=_txt("node"),
                proto=_txt("proto") or "shape-rotator-hivemind/v1",
                version=_txt("version"),
            )
        except Exception as exc:
            log.debug("failed to parse hivemind ServiceInfo: %s", exc)
            return None


# ── HTTP poster (extracted for tests) ───────────────────────────────────


def post_batch(sink: Sink, batch: dict, timeout: float = POST_TIMEOUT_SECONDS) -> dict:
    """POST a transcript batch payload to ``sink``.

    Returns the parsed JSON response on 2xx, or raises
    ``urllib.error.HTTPError`` / ``urllib.error.URLError``. The caller
    decides how to surface failures — for ``AUTO`` mode we log and
    drop; for ``ON`` mode we let it propagate so the user sees it.
    """
    body = json.dumps(batch, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BATCH_BYTES:
        raise ValueError(
            f"batch is {len(body)} bytes; sink rejects > {MAX_BATCH_BYTES}"
        )
    req = urllib.request.Request(
        sink.transcripts_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "voxterm/hivemind",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return {"raw": raw[:1024].decode("utf-8", errors="replace")}


# ── client (segment buffering + flush cadence) ──────────────────────────


class HivemindClient:
    """Buffers transcript segments and flushes batches to a sink.

    Hooked into the core dictation/transcription loop via
    ``add_segment(t, speaker, text)``. The flush cadence (60s OR 30
    segments OR EOF) lives here, NOT in the core loop.

    Thread-safety: ``add_segment`` and ``close`` are safe to call from
    any thread; flushes happen synchronously in the calling thread (we
    don't spin up a poster thread — keeps shutdown simple).
    """

    def __init__(
        self,
        *,
        device_id: str,
        record_id: Optional[str] = None,
        location: str = "",
        sink: Sink | None = None,
        browser: HivemindBrowser | None = None,
        mode: HivemindMode = HivemindMode.AUTO,
        flush_seconds: float = FLUSH_SECONDS_DEFAULT,
        flush_segments: int = FLUSH_SEGMENTS_DEFAULT,
        push_enabled: bool = False,
        pinned_sink_pubkey: str = "",
        on_state_change: Callable[[bool, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        poster: Callable[[Sink, dict], dict] | None = None,
    ) -> None:
        self._device_id = device_id
        self._record_id = record_id or self._default_record_id(wall_clock())
        self._location = location
        self._explicit_sink = sink
        self._browser = browser
        self._mode = mode
        self._flush_seconds = flush_seconds
        self._flush_segments = flush_segments
        # Push gate: even when a sink is discovered, batches only POST
        # if the user has enabled push (via `h` menu) or mode=on
        # overrides. Pinning by pubkey means a different sink showing
        # up on the LAN won't get our transcripts by accident.
        self._push_enabled = push_enabled
        self._pinned_sink_pubkey = pinned_sink_pubkey
        # Called whenever (push_enabled, pinned_pubkey) changes so the
        # TUI can persist the choice to ConfigStore without this
        # module depending on the store.
        self._on_state_change = on_state_change
        self._clock = clock
        self._wall_clock = wall_clock
        self._poster = poster or (lambda s, b: post_batch(s, b))

        self._lock = threading.RLock()
        self._segments: list[_PendingSegment] = []
        self._batch_started_wall: datetime | None = None
        self._batch_started_mono: float | None = None
        self._batch_index = 0
        self._closed = False
        # Stats for tests / debug UIs.
        self._batches_sent = 0
        self._batches_dropped = 0
        self._last_error: BaseException | None = None
        # Logging breadcrumbs (so per-batch lines stay at DEBUG and we
        # only emit INFO at meaningful transitions or every Nth batch).
        self._first_batch_logged = False
        self._no_sink_warning_logged = False
        self._push_disabled_drop_logged = False
        self._wrong_pubkey_drop_logged = False

    # ── public API ─────────────────────────────────────────────────

    @property
    def record_id(self) -> str:
        return self._record_id

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def batches_sent(self) -> int:
        return self._batches_sent

    @property
    def batches_dropped(self) -> int:
        return self._batches_dropped

    @property
    def pending_segments(self) -> int:
        with self._lock:
            return len(self._segments)

    @property
    def push_enabled(self) -> bool:
        """Whether the user has opted in to actually POST batches.

        Discovery is independent: the mDNS browser runs regardless of
        this flag, so the UI can show "sink available; press ENTER to
        enable" without anything being sent yet.
        """
        return self._push_enabled

    @property
    def pinned_sink_pubkey(self) -> str:
        return self._pinned_sink_pubkey

    def enable_push(self, sink_pubkey: str = "") -> None:
        """User said yes to a sink. Pin to its pubkey and start POSTing.

        Idempotent. Fires the on_state_change callback so the TUI can
        persist the choice to ConfigStore.
        """
        already = self._push_enabled and self._pinned_sink_pubkey == sink_pubkey
        self._push_enabled = True
        self._pinned_sink_pubkey = sink_pubkey
        # Reset drop-rate-limiters so the next flush logs cleanly.
        self._push_disabled_drop_logged = False
        self._wrong_pubkey_drop_logged = False
        if not already:
            log.info(
                "hivemind: push enabled by user (pinned pubkey=%s)",
                (sink_pubkey[:12] + "...") if sink_pubkey else "(any)",
            )
            if self._on_state_change is not None:
                try:
                    self._on_state_change(True, sink_pubkey)
                except Exception:
                    log.warning("hivemind on_state_change raised", exc_info=True)

    def disable_push(self) -> None:
        """User backed out. Stop POSTing; keep discovery + buffer."""
        if not self._push_enabled:
            return
        self._push_enabled = False
        # Keep _pinned_sink_pubkey so re-enabling without a fresh
        # selection still targets the same sink. The TUI can clear it
        # explicitly if it wants "forget the choice entirely".
        log.info("hivemind: push disabled by user")
        if self._on_state_change is not None:
            try:
                self._on_state_change(False, self._pinned_sink_pubkey)
            except Exception:
                log.warning("hivemind on_state_change raised", exc_info=True)

    def discovered_sinks(self) -> list[Sink]:
        """All currently-known sinks (the `h` menu calls this)."""
        if self._explicit_sink is not None:
            return [self._explicit_sink]
        if self._browser is not None:
            return self._browser.sinks()
        return []

    def active_sink(self) -> Sink | None:
        if self._explicit_sink is not None:
            return self._explicit_sink
        if self._browser is not None:
            return self._browser.active_sink()
        return None

    def add_segment(self, t: float, speaker: str, text: str) -> None:
        """Append a transcript segment. Triggers a flush if cadence hits."""
        if self._closed or self._mode == HivemindMode.OFF:
            return
        text = (text or "").strip()
        if not text:
            return
        seg = _PendingSegment(t=float(t), speaker=str(speaker or ""), text=text)
        with self._lock:
            if self._batch_started_wall is None:
                self._batch_started_wall = self._wall_clock()
                self._batch_started_mono = self._clock()
            self._segments.append(seg)

        # Decide whether to flush. The 30-segment trigger is exact; the
        # 60s trigger is checked at every add_segment so a slow trickle
        # of segments still flushes on time.
        if self._should_flush():
            self.flush_now()

    def maybe_flush(self) -> bool:
        """Flush iff a cadence trigger has fired. Returns whether we did.

        Useful when the producer wants to nudge the cadence on a tick
        timer instead of relying on add_segment to cross the line.
        """
        if not self._should_flush():
            return False
        return self.flush_now()

    def flush_now(self) -> bool:
        """Force a flush. Returns True iff anything was POSTed.

        Empty buffers are a no-op (returns False) — we never POST a
        zero-segment batch.
        """
        with self._lock:
            if not self._segments:
                return False
            payload = self._build_payload_locked()
            self._segments.clear()
            self._batch_started_wall = None
            self._batch_started_mono = None
            self._batch_index += 1

        # Push gate. User hasn't opted in via the `h` menu yet (or has
        # explicitly disabled) → drop the batch silently. Local file
        # logging continues; the buffer just doesn't get POSTed.
        if not self._push_enabled:
            if not self._push_disabled_drop_logged:
                log.info(
                    "hivemind: push not enabled; %d segs buffered but not POSTed "
                    "(press 'h' in the TUI to enable)",
                    len(payload["segments"]),
                )
                self._push_disabled_drop_logged = True
            else:
                log.debug(
                    "hivemind: push disabled; dropping batch (%d segs)",
                    len(payload["segments"]),
                )
            self._batches_dropped += 1
            return False
        self._push_disabled_drop_logged = False

        sink = self.active_sink()
        # Pubkey pinning: if the user enabled push against a specific
        # sink (e.g. their convent box), don't accidentally start
        # POSTing to a different sink that happened to appear on the
        # LAN. An empty pinned_pubkey means "any discovered sink is
        # fine" — that's the just-after-enable transient before we
        # save the pubkey, or the explicit-sink-url path.
        if sink is not None and self._pinned_sink_pubkey:
            if sink.pubkey != self._pinned_sink_pubkey:
                if not self._wrong_pubkey_drop_logged:
                    log.warning(
                        "hivemind: discovered sink %s has pubkey %s but pinned "
                        "is %s; dropping batch",
                        sink.transcripts_url,
                        sink.pubkey[:12] + "..." if sink.pubkey else "(none)",
                        self._pinned_sink_pubkey[:12] + "...",
                    )
                    self._wrong_pubkey_drop_logged = True
                self._batches_dropped += 1
                return False
            self._wrong_pubkey_drop_logged = False

        if sink is None:
            # Rate-limit the "no sink" log: WARNING once on entering the
            # no-sink state, DEBUG thereafter until a sink is found.
            # Without this we'd spam one WARNING per flush while no sink
            # is discovered (every 60s in the no-content path).
            if not self._no_sink_warning_logged:
                log.warning(
                    "hivemind: no sink discovered yet; dropping batch (%d segs). "
                    "mDNS browser still running in the background.",
                    len(payload["segments"]),
                )
                self._no_sink_warning_logged = True
            else:
                log.debug(
                    "hivemind: still no sink; dropping batch (%d segs)",
                    len(payload["segments"]),
                )
            self._batches_dropped += 1
            return False

        # We had a sink; reset the no-sink breadcrumb so a future
        # discovery-loss re-triggers a single WARNING.
        self._no_sink_warning_logged = False

        try:
            self._poster(sink, payload)
            self._batches_sent += 1
            # Per-batch detail is DEBUG; INFO is reserved for state
            # transitions (first batch, every Nth heartbeat) so a user
            # tailing voxterm.log doesn't drown.
            log.debug(
                "hivemind: posted batch %d (%d segs) -> %s",
                payload["batch_index"], len(payload["segments"]), sink.transcripts_url,
            )
            if not self._first_batch_logged:
                log.info(
                    "hivemind: first batch posted to %s (sent=%d dropped=%d)",
                    sink.transcripts_url, self._batches_sent, self._batches_dropped,
                )
                self._first_batch_logged = True
            elif self._batches_sent % HEARTBEAT_EVERY_N_BATCHES == 0:
                log.info(
                    "hivemind: still posting OK -> %s (sent=%d dropped=%d)",
                    sink.transcripts_url, self._batches_sent, self._batches_dropped,
                )
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            self._last_error = exc
            self._batches_dropped += 1
            log.warning("hivemind: POST failed (%s); dropped batch", exc)
            return False

    def close(self) -> bool:
        """Flush any pending batch and stop accepting new segments."""
        if self._closed:
            return False
        flushed = self.flush_now()
        self._closed = True
        return flushed

    # ── internals ─────────────────────────────────────────────────

    def _should_flush(self) -> bool:
        with self._lock:
            n = len(self._segments)
            if n == 0:
                return False
            if n >= self._flush_segments:
                return True
            if self._batch_started_mono is None:
                return False
            elapsed = self._clock() - self._batch_started_mono
            return elapsed >= self._flush_seconds

    def _build_payload_locked(self) -> dict:
        ended = self._wall_clock()
        started = self._batch_started_wall or ended
        payload: dict = {
            "record_id": self._record_id,
            "batch_index": self._batch_index,
            "started_at": _iso(started),
            "ended_at": _iso(ended),
            "origin_device": self._device_id,
            "segments": [
                {"t": s.t, "speaker": s.speaker, "text": s.text}
                for s in self._segments
            ],
        }
        if self._location:
            payload["location"] = self._location
        return payload

    @staticmethod
    def _default_record_id(now: datetime) -> str:
        return "transcript-" + now.strftime("%Y-%m-%d-%H%M-voxterm")


def _iso(dt: datetime) -> str:
    """ISO 8601 with 'Z' suffix, no microseconds (sink doesn't need them)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return s.replace("+00:00", "Z")


# ── facade ──────────────────────────────────────────────────────────────


def configure(
    *,
    mode: HivemindMode | str = HivemindMode.AUTO,
    sink_url: str | None = None,
    location: str = "",
    record_id: str | None = None,
    device_id_path: Path | None = None,
    discovery_timeout: float = DISCOVERY_TIMEOUT_SECONDS,
    push_enabled: bool = False,
    pinned_sink_pubkey: str = "",
    on_state_change: Callable[[bool, str], None] | None = None,
) -> HivemindClient | None:
    """Build a configured ``HivemindClient`` for this voxterm session.

    Returns None when ``mode == HivemindMode.OFF``. When ``mode == ON``
    and no sink can be located (neither ``sink_url`` nor mDNS within
    ``discovery_timeout``), raises ``RuntimeError``.

    Push gating (mode=auto default behavior): discovery runs but
    transcripts are buffered, not POSTed, until ``push_enabled=True``
    (the user opted in via the `h` menu and the TUI saved the choice
    to ConfigStore). ``mode=on`` overrides and always pushes — that's
    the scripted/headless path. ``pinned_sink_pubkey`` (when non-empty)
    constrains POSTs to a specific sink identity so a different sink
    showing up on the LAN doesn't get our transcripts by accident.

    The browser, if started, is owned by the returned client. Call
    ``client.close()`` to flush the last batch; the browser keeps
    running for the rest of the process lifetime (mDNS is cheap, and
    closing it from the dictation loop teardown is fiddly).
    """
    if isinstance(mode, str):
        mode = HivemindMode.parse(mode)

    if mode == HivemindMode.OFF:
        log.info("hivemind: mode=off; never posting")
        return None

    device_id = get_or_create_device_id(device_id_path)

    sink: Sink | None = None
    browser: HivemindBrowser | None = None

    if sink_url:
        sink = Sink.from_url(sink_url)
        log.info("hivemind: using explicit sink %s", sink.transcripts_url)
    else:
        # Async discovery: when a sink shows up later (after configure
        # returns), log INFO once so a user tailing voxterm.log sees
        # the transition without us having to poll.
        _discovery_logged = {"done": False}

        def _on_change(sinks: list[Sink]) -> None:
            if _discovery_logged["done"] or not sinks:
                return
            _discovery_logged["done"] = True
            log.info(
                "hivemind: sink discovered via mDNS -> %s",
                sinks[-1].transcripts_url,
            )

        log.info(
            "hivemind: searching for sink on %s (mDNS, mode=%s)",
            HIVEMIND_SERVICE_TYPE, mode.value,
        )
        browser = HivemindBrowser(on_change=_on_change)
        browser.start()

        if mode == HivemindMode.ON:
            sink = browser.wait_for_sink(discovery_timeout)
            if sink is None:
                browser.stop()
                raise RuntimeError(
                    f"hivemind: mode=on but no sink discovered "
                    f"in {discovery_timeout:.1f}s on {HIVEMIND_SERVICE_TYPE}"
                )
            if not _discovery_logged["done"]:
                log.info("hivemind: discovered sink %s", sink.transcripts_url)
                _discovery_logged["done"] = True
        else:
            # AUTO: brief synchronous wait so the startup banner has
            # something to print. If the sink lands later, the on_change
            # callback logs it; we don't block voxterm boot.
            sink = browser.wait_for_sink(AUTO_INITIAL_DISCOVERY_WAIT_SECONDS)
            if sink is not None:
                if not _discovery_logged["done"]:
                    log.info("hivemind: discovered sink %s", sink.transcripts_url)
                    _discovery_logged["done"] = True
            else:
                log.info(
                    "hivemind: no sink yet; mDNS browser keeps searching "
                    "in the background (will log when a sink appears)"
                )

    # mode=on is the legacy "force push" behavior; mode=auto respects
    # the user's persisted opt-in. Either way, an explicit sink_url
    # implicitly opts in (you typed the URL on the command line, so
    # you clearly want it pushed).
    effective_push_enabled = (
        push_enabled
        or mode == HivemindMode.ON
        or sink_url is not None
    )
    if effective_push_enabled and not push_enabled:
        # Log once at startup when we're auto-enabling push for a
        # reason other than the persisted opt-in.
        log.info(
            "hivemind: push enabled at launch (mode=%s%s)",
            mode.value,
            ", explicit sink_url" if sink_url else "",
        )

    return HivemindClient(
        device_id=device_id,
        record_id=record_id,
        location=location,
        sink=sink,
        browser=browser,
        mode=mode,
        push_enabled=effective_push_enabled,
        pinned_sink_pubkey=pinned_sink_pubkey,
        on_state_change=on_state_change,
    )
