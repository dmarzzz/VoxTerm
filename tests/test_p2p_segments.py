"""Tests for P2P transcript assembly."""

import pytest

from network.clock import ClockSync
from network.segments import TranscriptAssembler, MergedSegment


class TestTranscriptAssembler:
    def test_single_final(self):
        asm = TranscriptAssembler()
        seg = asm.on_final("node-a", 1, "alice", "hello", 10.0, 12.0, 0.95)
        assert seg.speaker_name == "alice"
        assert seg.text == "hello"
        assert not seg.is_partial
        assert asm.final_count == 1

    def test_finals_ordered_by_start_ts(self):
        asm = TranscriptAssembler()
        asm.on_final("node-b", 1, "bob", "second", 20.0, 22.0, 0.9)
        asm.on_final("node-a", 1, "alice", "first", 10.0, 12.0, 0.9)
        asm.on_final("node-c", 1, "carol", "third", 30.0, 32.0, 0.9)

        finals = asm.get_finals()
        assert [f.speaker_name for f in finals] == ["alice", "bob", "carol"]
        assert [f.adjusted_start_ts for f in finals] == [10.0, 20.0, 30.0]

    def test_finals_with_clock_sync(self):
        asm = TranscriptAssembler()

        # Bob's clock is 5s ahead of local
        bob_sync = ClockSync()
        bob_sync.add_sample(0.0, 5.5, 1.0)  # offset = 5.0

        # Bob says start_ts=15.0, but adjusted = 15.0 - 5.0 = 10.0
        asm.on_final("node-b", 1, "bob", "from bob", 15.0, 17.0, 0.9, clock_sync=bob_sync)
        # Local segment at ts=12.0 (no sync needed)
        asm.on_final("node-a", 1, "alice", "from alice", 12.0, 14.0, 0.9)

        finals = asm.get_finals()
        assert finals[0].speaker_name == "bob"  # adjusted 10.0 < 12.0
        assert finals[1].speaker_name == "alice"

    def test_partial_stored_and_replaced(self):
        asm = TranscriptAssembler()
        asm.on_partial("node-a", 5, "alice", "hel", 10.0)
        assert asm.partial_count == 1

        asm.on_partial("node-a", 5, "alice", "hello wor", 10.0)
        assert asm.partial_count == 1  # replaced, not duplicated

        partials = asm.get_partials()
        assert partials[0].text == "hello wor"

    def test_final_clears_matching_partial(self):
        asm = TranscriptAssembler()
        asm.on_partial("node-a", 5, "alice", "hello wor", 10.0)
        assert asm.partial_count == 1

        asm.on_final("node-a", 5, "alice", "hello world", 10.0, 12.0, 0.95)
        assert asm.partial_count == 0
        assert asm.final_count == 1

    def test_final_does_not_clear_different_seq_partial(self):
        asm = TranscriptAssembler()
        asm.on_partial("node-a", 6, "alice", "next utterance", 15.0)
        asm.on_final("node-a", 5, "alice", "previous", 10.0, 12.0, 0.9)
        assert asm.partial_count == 1  # seq 6 still pending

    def test_multiple_peers_partials(self):
        asm = TranscriptAssembler()
        asm.on_partial("node-a", 1, "alice", "hi", 10.0)
        asm.on_partial("node-b", 1, "bob", "hey", 10.5)
        assert asm.partial_count == 2

    def test_clear_peer(self):
        asm = TranscriptAssembler()
        asm.on_partial("node-a", 1, "alice", "hi", 10.0)
        asm.on_partial("node-b", 1, "bob", "hey", 10.5)
        asm.clear_peer("node-a")
        assert asm.partial_count == 1
        assert asm.get_partials()[0].node_id == "node-b"

    def test_clear_nonexistent_peer_is_noop(self):
        asm = TranscriptAssembler()
        asm.clear_peer("ghost")  # should not raise

    def test_interleaved_peers(self):
        """Segments from multiple peers interleaved by time."""
        asm = TranscriptAssembler()
        asm.on_final("node-a", 1, "alice", "A1", 1.0, 2.0, 0.9)
        asm.on_final("node-b", 1, "bob", "B1", 1.5, 2.5, 0.9)
        asm.on_final("node-a", 2, "alice", "A2", 3.0, 4.0, 0.9)
        asm.on_final("node-b", 2, "bob", "B2", 3.5, 4.5, 0.9)

        finals = asm.get_finals()
        texts = [f.text for f in finals]
        assert texts == ["A1", "B1", "A2", "B2"]

    def test_simultaneous_segments(self):
        """Segments at the same timestamp are both included."""
        asm = TranscriptAssembler()
        asm.on_final("node-a", 1, "alice", "overlap-a", 10.0, 12.0, 0.9)
        asm.on_final("node-b", 1, "bob", "overlap-b", 10.0, 12.0, 0.9)
        assert asm.final_count == 2

    def test_merged_segment_fields(self):
        asm = TranscriptAssembler()
        seg = asm.on_final("node-x", 99, "xavier", "test", 42.0, 44.0, 0.88)
        assert seg.node_id == "node-x"
        assert seg.seq == 99
        assert seg.speaker_name == "xavier"
        assert seg.confidence == 0.88
        assert not seg.is_partial
        assert seg.adjusted_start_ts == 42.0
