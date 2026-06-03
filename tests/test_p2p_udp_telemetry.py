"""Per-peer audio frame-loss telemetry.

The real UDP audio-loss signal is sequence gaps filled with silence, tracked per
peer in PeerAudioBuffer.gaps_filled. It was measured but never exported, so the
debug overlay couldn't show it. This asserts gaps_filled now flows through the
mixer's get_stats()/debug_info() so the overlay can render a per-peer gap rate.
"""

from __future__ import annotations

import numpy as np

from audio.merger import PeerAudioMixer
from network.clock import ClockSync


def _frame(samples: int = 320) -> bytes:
    return np.zeros(samples, dtype=np.int16).tobytes()


def test_peer_gaps_surface_through_mixer_stats():
    mixer = PeerAudioMixer()
    nid = "peerA"
    mixer.register_peer(nid, ClockSync())

    # 0,1,2 contiguous, then jump to 6 -> a gap of 3 frames (3,4,5 lost).
    for seq in (0, 1, 2):
        mixer.peer_frame(nid, seq, _frame())
    mixer.peer_frame(nid, 6, _frame())

    stats = mixer.get_stats()
    assert "peer_gaps" in stats, "mixer get_stats() must export peer_gaps"
    assert stats["peer_frames"][nid] == 4   # frames actually written: 0,1,2,6
    assert stats["peer_gaps"][nid] == 3     # silence frames filled for the gap

    # The overlay computes gap rate as gaps / (frames + gaps).
    frames, gaps = stats["peer_frames"][nid], stats["peer_gaps"][nid]
    assert round(gaps / (frames + gaps) * 100.0, 1) == round(3 / 7 * 100.0, 1)

    # debug_info() (legacy overlay path) exports it too.
    assert mixer.debug_info()["peer_gaps"][nid] == 3


def test_no_gaps_when_contiguous():
    mixer = PeerAudioMixer()
    nid = "peerB"
    mixer.register_peer(nid, ClockSync())
    for seq in range(5):
        mixer.peer_frame(nid, seq, _frame())
    stats = mixer.get_stats()
    assert stats["peer_frames"][nid] == 5
    assert stats["peer_gaps"][nid] == 0
