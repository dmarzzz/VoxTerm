"""Tests for P2P clock synchronization."""

import pytest

from network.clock import ClockSync


class TestClockSync:
    def test_no_samples(self):
        sync = ClockSync()
        assert sync.offset == 0.0
        assert sync.rtt == 0.0
        assert sync.sample_count == 0

    def test_single_sample_zero_offset(self):
        sync = ClockSync()
        # Peer has same clock: t1=0, t2=0.5 (their clock), t3=1.0
        # RTT=1.0, OWL=0.5, offset = 0.5 - 0 - 0.5 = 0.0
        sync.add_sample(0.0, 0.5, 1.0)
        assert sync.offset == pytest.approx(0.0)
        assert sync.rtt == pytest.approx(1.0)

    def test_positive_offset(self):
        sync = ClockSync()
        # Peer's clock is 10s ahead
        # t1=0, t2=10.5 (0.5 transit + 10 offset), t3=1.0
        # RTT=1.0, OWL=0.5, offset = 10.5 - 0 - 0.5 = 10.0
        sync.add_sample(0.0, 10.5, 1.0)
        assert sync.offset == pytest.approx(10.0)

    def test_negative_offset(self):
        sync = ClockSync()
        # Peer's clock is 5s behind
        # t1=100, t2=95.5 (100.5 - 5), t3=101
        # RTT=1.0, OWL=0.5, offset = 95.5 - 100 - 0.5 = -5.0
        sync.add_sample(100.0, 95.5, 101.0)
        assert sync.offset == pytest.approx(-5.0)

    def test_adjust(self):
        sync = ClockSync()
        # Peer is 10s ahead
        sync.add_sample(0.0, 10.5, 1.0)
        # Their timestamp 110.0 in local clock = 110.0 - 10.0 = 100.0
        assert sync.adjust(110.0) == pytest.approx(100.0)

    def test_median_robust_to_outliers(self):
        sync = ClockSync()
        # 9 good samples with offset ~5.0 (RTT=1ms, peer 5s ahead)
        for _ in range(9):
            sync.add_sample(100.0, 105.0005, 100.001)
        # 1 outlier (huge RTT spike)
        sync.add_sample(100.0, 105.5, 101.0)
        # Median should still be ~5.0, not pulled by outlier
        assert sync.offset == pytest.approx(5.0, abs=0.01)

    def test_window_trimming(self):
        sync = ClockSync(window_size=5)
        # Fill with offset=1.0
        for _ in range(5):
            sync.add_sample(0.0, 1.5, 1.0)
        assert sync.offset == pytest.approx(1.0)

        # Overwrite with offset=2.0
        for _ in range(5):
            sync.add_sample(0.0, 2.5, 1.0)
        assert sync.offset == pytest.approx(2.0)
        assert sync.sample_count == 5

    def test_bogus_negative_rtt_rejected(self):
        sync = ClockSync()
        sync.add_sample(10.0, 5.0, 9.0)  # t3 < t1 → negative RTT
        assert sync.sample_count == 0

    def test_convergence_with_jitter(self):
        """Simulate realistic LAN jitter and verify convergence."""
        import random
        random.seed(42)

        true_offset = 3.14
        sync = ClockSync(window_size=20)

        for _ in range(30):
            base_rtt = 0.001  # 1ms base RTT
            jitter = random.gauss(0, 0.0005)  # ±0.5ms jitter
            rtt = max(0.0001, base_rtt + jitter)
            t1 = 100.0
            t2 = t1 + rtt / 2 + true_offset
            t3 = t1 + rtt
            sync.add_sample(t1, t2, t3)

        assert sync.offset == pytest.approx(true_offset, abs=0.002)

    def test_zero_rtt(self):
        sync = ClockSync()
        sync.add_sample(0.0, 5.0, 0.0)  # instant round trip
        assert sync.offset == pytest.approx(5.0)
        assert sync.rtt == pytest.approx(0.0)
