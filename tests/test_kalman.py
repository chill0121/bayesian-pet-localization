"""
Tests for the Kalman filter RSSI smoother.

Run with: python -m pytest tests/test_kalman.py -v
Or standalone: python tests/test_kalman.py
"""

import sys
import os
import time

import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from filters.kalman import KalmanFilter, KalmanFilterBank


# ---------------------------------------------------------------------------
# Sample data: real readings from your 1F_Office anchor (InfluxDB, 2026-03-06)
# Beacon was stationary at ~1.5m from anchor
# RSSI ranged from -63.52 to -64.31 dBm (very stable, <1 dB spread)
# ---------------------------------------------------------------------------
STATIONARY_SAMPLES = [
    (-64.18, 0.0),
    (-63.84, 5.0),
    (-63.52, 12.8),
    (-64.00, 15.0),
    (-64.31, 21.8),
    (-63.70, 26.0),
    (-64.10, 31.0),
    (-63.90, 36.0),
    (-64.05, 41.0),
    (-63.75, 46.0),
]

# Simulated movement: beacon walks away from anchor then returns
# RSSI drops as distance increases, then recovers
MOVEMENT_SAMPLES = [
    (-50.0,  0.0),   # right next to anchor
    (-52.0,  3.0),
    (-55.0,  6.0),
    (-60.0,  9.0),   # walking away
    (-67.0, 12.0),
    (-75.0, 15.0),
    (-82.0, 18.0),   # far away
    (-83.0, 21.0),
    (-84.0, 24.0),
    (-80.0, 27.0),   # coming back
    (-72.0, 30.0),
    (-63.0, 33.0),
    (-55.0, 36.0),
    (-50.0, 39.0),   # back at anchor
]

# Noisy stationary data (simulated with Gaussian noise, sigma=3 dBm)
rng = np.random.default_rng(42)
TRUE_RSSI = -65.0
NOISY_STATIONARY = [
    (TRUE_RSSI + rng.normal(0, 3), i * 3.0)
    for i in range(50)
]


class TestKalmanFilter:
    """Tests for the single-anchor KalmanFilter."""

    def test_initialization(self):
        """Filter initializes with correct state dimensions."""
        kf = KalmanFilter(initial_rssi=-65.0)
        assert kf.x.shape == (2,), "State should be 2-element vector [rssi, velocity]"
        assert kf.P.shape == (2, 2), "Covariance should be 2x2"
        assert kf.rssi == -65.0
        assert kf.velocity == 0.0

    def test_build_F(self):
        """State transition matrix has correct structure."""
        kf = KalmanFilter()
        F = kf._build_F(dt=1.0)
        assert F.shape == (2, 2)
        # With dt=1: F should be [[1, 1], [0, 1]]
        expected = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(F, expected)

        # With dt=0.5
        F2 = kf._build_F(dt=0.5)
        expected2 = np.array([[1.0, 0.5], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(F2, expected2)

    def test_build_Q(self):
        """Process noise matrix scales with dt and process_noise."""
        kf = KalmanFilter(process_noise=1.0)
        Q = kf._build_Q(dt=1.0)
        assert Q.shape == (2, 2)
        # Q should be symmetric and positive semi-definite
        np.testing.assert_array_almost_equal(Q, Q.T)
        assert np.all(np.linalg.eigvals(Q) >= 0)

    def test_first_measurement_initializes(self):
        """First call to process() should initialize state to the measurement."""
        kf = KalmanFilter(initial_rssi=-70.0)
        result = kf.process(raw_rssi=-65.0, timestamp=100.0)
        # After first measurement, rssi should be close to -65 (not -70)
        assert abs(result - (-65.0)) < 2.0, f"First measurement should initialize near -65, got {result}"

    def test_stationary_converges(self):
        """Filter converges to true value for stationary readings."""
        kf = KalmanFilter(process_noise=0.5, measurement_noise=1.0)
        results = []
        for rssi, t in STATIONARY_SAMPLES:
            result = kf.process(rssi, timestamp=t)
            results.append(result)

        # After processing all samples, should be very close to mean (~-63.9)
        true_mean = np.mean([r for r, _ in STATIONARY_SAMPLES])
        assert abs(results[-1] - true_mean) < 1.0, (
            f"Should converge near {true_mean:.1f}, got {results[-1]:.1f}"
        )

    def test_smoothing_reduces_variance(self):
        """Filtered output should have lower variance than raw input."""
        kf = KalmanFilter(process_noise=0.5, measurement_noise=9.0)  # R=9 for noisy data
        raw_values = [r for r, _ in NOISY_STATIONARY]
        filtered_values = [kf.process(r, t) for r, t in NOISY_STATIONARY]

        raw_var = np.var(raw_values)
        filtered_var = np.var(filtered_values[10:])  # skip transient
        assert filtered_var < raw_var, (
            f"Filtered variance ({filtered_var:.2f}) should be < raw ({raw_var:.2f})"
        )

    def test_tracks_movement(self):
        """Filter tracks a moving signal (RSSI dropping then recovering)."""
        kf = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
        results = []
        for rssi, t in MOVEMENT_SAMPLES:
            results.append(kf.process(rssi, t))

        # At the midpoint (far away), filtered should be below -70
        mid = len(results) // 2
        assert results[mid] < -70, f"Mid-movement should track below -70, got {results[mid]}"

        # At the end (back at anchor), should recover above -60
        assert results[-1] > -60, f"Should recover above -60, got {results[-1]}"

    def test_velocity_indicates_movement(self):
        """Velocity should be negative when moving away, positive when returning."""
        kf = KalmanFilter(process_noise=1.0, measurement_noise=1.0)
        velocities = []
        for rssi, t in MOVEMENT_SAMPLES:
            kf.process(rssi, t)
            velocities.append(kf.velocity)

        # During retreat (samples 3-7), velocity should be negative
        assert any(v < 0 for v in velocities[3:7]), "Velocity should be negative while moving away"
        # During return (samples 9-13), velocity should be positive
        assert any(v > 0 for v in velocities[9:13]), "Velocity should be positive while returning"

    def test_reset(self):
        """Reset clears state back to defaults."""
        kf = KalmanFilter()
        kf.process(-50.0, 0.0)
        kf.process(-55.0, 3.0)
        kf.reset(rssi=-70.0)
        assert kf.rssi == -70.0
        assert kf.velocity == 0.0
        assert kf._last_time is None


class TestKalmanFilterBank:
    """Tests for the multi-anchor KalmanFilterBank."""

    def test_multi_anchor(self):
        """Bank manages separate filters per anchor."""
        bank = KalmanFilterBank()
        bank.update("office", -65.0, 0.0)
        bank.update("kitchen", -80.0, 0.0)

        smoothed = bank.get_smoothed_rssi()
        assert "office" in smoothed
        assert "kitchen" in smoothed
        assert smoothed["office"] > smoothed["kitchen"]  # office is closer

    def test_stale_anchor_excluded(self):
        """Anchors with no recent updates are excluded from smoothed output."""
        bank = KalmanFilterBank(stale_timeout=10.0)
        bank.update("office", -65.0, 0.0)
        bank.update("kitchen", -80.0, 0.0)

        # office goes stale
        bank.update("kitchen", -79.0, 15.0)

        smoothed = bank.get_smoothed_rssi(current_time=15.0)
        assert "kitchen" in smoothed
        assert "office" not in smoothed, "Stale anchor should be excluded"

    def test_stale_anchor_resets_on_return(self):
        """When a stale anchor reports again, its filter should be reset."""
        bank = KalmanFilterBank(stale_timeout=10.0)
        bank.update("office", -65.0, 0.0)
        # Long gap
        result = bank.update("office", -80.0, 100.0)
        # Should have reset and initialized near -80, not still near -65
        assert abs(result - (-80.0)) < 5.0

    def test_active_anchors(self):
        """active_anchors returns all seen anchor IDs."""
        bank = KalmanFilterBank()
        bank.update("a", -60.0, 0.0)
        bank.update("b", -70.0, 0.0)
        bank.update("c", -80.0, 0.0)
        assert set(bank.active_anchors) == {"a", "b", "c"}

    def test_reset_all(self):
        """reset_all clears all filters."""
        bank = KalmanFilterBank()
        bank.update("a", -60.0, 0.0)
        bank.update("b", -70.0, 0.0)
        bank.reset_all()
        assert bank.active_anchors == []


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Kalman filter tests...\n")
    passed = 0
    failed = 0

    for cls in [TestKalmanFilter, TestKalmanFilterBank]:
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(instance, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except NotImplementedError:
                print(f"  SKIP  {cls.__name__}.{name} (not implemented yet)")
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
