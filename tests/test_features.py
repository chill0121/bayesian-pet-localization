"""Tests for the feature engineering pipeline (services/inference/features.py)."""

import sys
import os
import time

import pytest

# Allow imports from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from features import FeatureEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ANCHOR_POSITIONS = {
    "1F_Office": {"x": 3.75, "y": 6.5, "floor": 1},
    "living_sw": {"x": 2.0, "y": 1.0, "floor": 2},
    "living_center": {"x": 10.0, "y": 14.0, "floor": 2},
    "kitchen_ne": {"x": 15.0, "y": 26.0, "floor": 2},
    "master_bed": {"x": 7.0, "y": 5.0, "floor": 3},
}


@pytest.fixture
def engine():
    return FeatureEngine(anchor_positions=ANCHOR_POSITIONS, window_size=5)


@pytest.fixture
def engine_no_anchors():
    return FeatureEngine(window_size=5)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestFeatureEngineBasic:
    def test_first_update_returns_features(self, engine):
        raw = {"living_sw": -65.0, "living_center": -72.0, "kitchen_ne": -80.0}
        smoothed = {"living_sw": -64.5, "living_center": -71.8, "kitchen_ne": -79.5}
        features = engine.update(raw, smoothed, time.time())
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_features_property_matches_last_update(self, engine):
        raw = {"living_sw": -65.0, "living_center": -72.0}
        smoothed = {"living_sw": -64.5, "living_center": -71.8}
        result = engine.update(raw, smoothed, time.time())
        assert engine.features == result

    def test_empty_rssi(self, engine):
        features = engine.update({}, {}, time.time())
        assert features["delta_mean"] == 0.0
        assert features["n_reporting"] == 0
        assert features["activity_score"] == 0.0

    def test_single_anchor(self, engine):
        raw = {"living_sw": -60.0}
        smoothed = {"living_sw": -59.5}
        features = engine.update(raw, smoothed, time.time())
        assert features["strongest_anchor"] == "living_sw"
        assert features["rssi_gap_1_2"] == 0.0
        assert features["rank_living_sw"] == 1


# ---------------------------------------------------------------------------
# Delta RSSI
# ---------------------------------------------------------------------------

class TestDeltaFeatures:
    def test_no_delta_on_first_call(self, engine):
        raw = {"living_sw": -65.0, "living_center": -72.0}
        smoothed = {"living_sw": -64.5, "living_center": -71.8}
        features = engine.update(raw, smoothed, time.time())
        # No previous step → deltas are 0
        assert features["delta_mean"] == 0.0
        assert features["delta_max"] == 0.0
        assert "delta_rssi_living_sw" not in features

    def test_delta_computed_on_second_call(self, engine):
        t = time.time()
        engine.update(
            {"living_sw": -65.0, "living_center": -72.0},
            {"living_sw": -65.0, "living_center": -72.0},
            t,
        )
        features = engine.update(
            {"living_sw": -62.0, "living_center": -74.0},
            {"living_sw": -62.0, "living_center": -74.0},
            t + 1,
        )
        assert features["delta_rssi_living_sw"] == pytest.approx(3.0)
        assert features["delta_rssi_living_center"] == pytest.approx(-2.0)
        assert features["delta_mean"] == pytest.approx(2.5)
        assert features["delta_max"] == pytest.approx(3.0)

    def test_delta_handles_new_anchor(self, engine):
        t = time.time()
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t)
        # Second call adds a new anchor
        features = engine.update(
            {"living_sw": -63.0, "kitchen_ne": -80.0},
            {"living_sw": -63.0, "kitchen_ne": -80.0},
            t + 1,
        )
        # Delta for living_sw exists, kitchen_ne does not (no previous)
        assert "delta_rssi_living_sw" in features
        assert "delta_rssi_kitchen_ne" not in features


# ---------------------------------------------------------------------------
# Anchor Rankings
# ---------------------------------------------------------------------------

class TestRankingFeatures:
    def test_rankings_sorted_correctly(self, engine):
        raw = {"living_sw": -60.0, "living_center": -70.0, "kitchen_ne": -80.0}
        smoothed = dict(raw)
        features = engine.update(raw, smoothed, time.time())
        assert features["rank_living_sw"] == 1
        assert features["rank_living_center"] == 2
        assert features["rank_kitchen_ne"] == 3
        assert features["strongest_anchor"] == "living_sw"

    def test_rssi_gap(self, engine):
        raw = {"living_sw": -55.0, "living_center": -65.0}
        smoothed = dict(raw)
        features = engine.update(raw, smoothed, time.time())
        assert features["rssi_gap_1_2"] == pytest.approx(10.0)

    def test_rank_changes_detected(self, engine):
        t = time.time()
        engine.update(
            {"living_sw": -60.0, "living_center": -70.0, "kitchen_ne": -80.0},
            {"living_sw": -60.0, "living_center": -70.0, "kitchen_ne": -80.0},
            t,
        )
        # Swap living_sw and kitchen_ne
        features = engine.update(
            {"living_sw": -80.0, "living_center": -70.0, "kitchen_ne": -60.0},
            {"living_sw": -80.0, "living_center": -70.0, "kitchen_ne": -60.0},
            t + 1,
        )
        # living_sw: 1→3, kitchen_ne: 3→1 = 2 changes; living_center stays 2 = 0
        assert features["rank_changes"] == 2

    def test_no_rank_changes_on_first_call(self, engine):
        features = engine.update(
            {"living_sw": -60.0, "living_center": -70.0},
            {"living_sw": -60.0, "living_center": -70.0},
            time.time(),
        )
        assert features["rank_changes"] == 0


# ---------------------------------------------------------------------------
# Rolling Variance
# ---------------------------------------------------------------------------

class TestVarianceFeatures:
    def test_zero_variance_single_sample(self, engine):
        features = engine.update(
            {"living_sw": -65.0}, {"living_sw": -65.0}, time.time()
        )
        assert features["var_living_sw"] == 0.0
        assert features["var_mean"] == 0.0

    def test_variance_grows_with_noisy_data(self, engine):
        t = time.time()
        # Feed alternating high/low values
        for i, rssi in enumerate([-60.0, -70.0, -60.0, -70.0, -60.0]):
            engine.update(
                {"living_sw": rssi}, {"living_sw": rssi}, t + i
            )
        var = engine.features["var_living_sw"]
        assert var > 0, "Variance should be positive for alternating data"
        # Sample variance of [-60, -70, -60, -70, -60]: mean=-64, ss=120, var=30.0
        assert var == pytest.approx(30.0)

    def test_variance_stable_for_constant_data(self, engine):
        t = time.time()
        for i in range(5):
            engine.update(
                {"living_sw": -65.0}, {"living_sw": -65.0}, t + i
            )
        assert engine.features["var_living_sw"] == pytest.approx(0.0)

    def test_var_max_picks_noisiest_anchor(self, engine):
        t = time.time()
        # living_sw constant, living_center noisy
        for i, val in enumerate([0.0, 10.0, 0.0, 10.0, 0.0]):
            engine.update(
                {"living_sw": -65.0, "living_center": -65.0 + val},
                {"living_sw": -65.0, "living_center": -65.0 + val},
                t + i,
            )
        assert engine.features["var_max"] == engine.features["var_living_center"]
        assert engine.features["var_living_sw"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cross-Floor Attenuation
# ---------------------------------------------------------------------------

class TestCrossFloorFeatures:
    def test_per_floor_counts_and_means(self, engine):
        raw = {
            "1F_Office": -60.0,
            "living_sw": -65.0,
            "living_center": -70.0,
            "kitchen_ne": -75.0,
            "master_bed": -85.0,
        }
        features = engine.update(raw, dict(raw), time.time())
        assert features["n_anchors_floor_1"] == 1
        assert features["n_anchors_floor_2"] == 3
        assert features["n_anchors_floor_3"] == 1
        assert features["rssi_mean_floor_1"] == pytest.approx(-60.0)
        assert features["rssi_mean_floor_2"] == pytest.approx(-70.0)
        assert features["rssi_mean_floor_3"] == pytest.approx(-85.0)

    def test_best_floor_is_strongest(self, engine):
        raw = {
            "1F_Office": -80.0,
            "living_sw": -55.0,
            "living_center": -60.0,
            "master_bed": -90.0,
        }
        features = engine.update(raw, dict(raw), time.time())
        assert features["rssi_best_floor"] == 2

    def test_same_vs_cross_ratio(self, engine):
        raw = {
            "1F_Office": -80.0,        # floor 1 mean: -80
            "living_sw": -55.0,         # floor 2 mean: -57.5
            "living_center": -60.0,
            "master_bed": -90.0,        # floor 3 mean: -90
        }
        features = engine.update(raw, dict(raw), time.time())
        # best floor = 2 (mean -57.5)
        # others mean = (-80 + -90) / 2 = -85
        # ratio = -57.5 - (-85) = 27.5
        assert features["same_vs_cross_ratio"] == pytest.approx(27.5)

    def test_no_anchor_positions_skips_cross_floor(self, engine_no_anchors):
        raw = {"living_sw": -65.0, "kitchen_ne": -75.0}
        features = engine_no_anchors.update(raw, dict(raw), time.time())
        assert features["rssi_best_floor"] == 0
        assert features["same_vs_cross_ratio"] == 0.0

    def test_single_floor_ratio_zero(self, engine):
        raw = {"living_sw": -65.0, "living_center": -70.0}
        features = engine.update(raw, dict(raw), time.time())
        assert features["rssi_best_floor"] == 2
        assert features["same_vs_cross_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Composite / Activity
# ---------------------------------------------------------------------------

class TestCompositeFeatures:
    def test_n_reporting(self, engine):
        raw = {"living_sw": -60.0, "living_center": -70.0, "kitchen_ne": -80.0}
        features = engine.update(raw, dict(raw), time.time())
        assert features["n_reporting"] == 3

    def test_displacement_ft_present(self, engine):
        features = engine.update(
            {"living_sw": -60.0}, {"living_sw": -60.0}, time.time()
        )
        assert "displacement_ft" in features

    def test_activity_score_zero_when_still(self, engine):
        t = time.time()
        # All constant data → var, delta, and displacement all 0
        for i in range(3):
            engine.update(
                {"living_sw": -65.0, "living_center": -70.0},
                {"living_sw": -65.0, "living_center": -70.0},
                t + i,
            )
        assert engine.features["activity_score"] == pytest.approx(0.0)

    def test_activity_score_rises_with_movement(self, engine):
        t = time.time()
        # First pass: stable
        for i in range(3):
            engine.update(
                {"living_sw": -65.0}, {"living_sw": -65.0}, t + i
            )
        stable_score = engine.features["activity_score"]

        # Sudden large change
        features = engine.update(
            {"living_sw": -50.0}, {"living_sw": -50.0}, t + 3
        )
        # raw delta_max = 15.0 → saturates to 1.0
        assert features["activity_score"] > stable_score

    def test_activity_score_rises_with_displacement(self, engine):
        t = time.time()
        # Feed same RSSI but different positions → displacement drives score
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t,
                       position=(0.0, 0.0, 1))
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t + 1,
                       position=(3.0, 4.0, 1))  # 5 ft displacement
        score = engine.features["activity_score"]
        # displacement = 5.0, norm = 4.0 → saturates at 1.0
        # 0.50 * 1.0 = 0.50 from displacement alone
        assert score >= 0.5

    def test_activity_score_bounded_0_to_1(self, engine):
        t = time.time()
        # Feed extreme data
        for i, rssi in enumerate([-30, -100, -30, -100, -30]):
            engine.update({"living_sw": rssi}, {"living_sw": rssi}, t + i,
                          position=(float(i * 10), 0.0, 1))
        score = engine.features["activity_score"]
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Window behaviour
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_window_size_respected(self):
        engine = FeatureEngine(window_size=3)
        t = time.time()
        for i in range(10):
            engine.update(
                {"living_sw": -60.0 - i}, {"living_sw": -60.0 - i}, t + i
            )
        # Only last 3 values should be in the buffer
        buf = engine._history["living_sw"]
        assert len(buf) == 3
        assert list(buf) == [-67.0, -68.0, -69.0]

    def test_variance_uses_window_only(self):
        engine = FeatureEngine(window_size=3)
        t = time.time()
        # First 3 constant, then 3 noisy
        for i in range(3):
            engine.update({"a": -65.0}, {"a": -65.0}, t + i)
        assert engine.features["var_a"] == pytest.approx(0.0)

        for i, v in enumerate([-60.0, -70.0, -60.0]):
            engine.update({"a": v}, {"a": v}, t + 3 + i)
        # Window now: [-60, -70, -60].  Var = 33.333... (rounded to 3 dp)
        assert engine.features["var_a"] == pytest.approx(100.0 / 3, abs=0.01)

    def test_stale_repeats_not_stored(self):
        """Repeated identical raw RSSI values (stale buffer reads) are
        deduplicated so the window only contains genuinely new readings."""
        engine = FeatureEngine(window_size=5)
        t = time.time()
        # Simulate 10 inference calls where raw_rssi hasn't changed
        for i in range(10):
            engine.update({"a": -65.0}, {"a": -65.0}, t + i * 0.1)
        # Only 1 genuinely distinct value was ever seen
        assert len(engine._history["a"]) == 1
        assert engine.features["var_a"] == 0.0

    def test_dedup_appends_on_real_change(self):
        """A genuinely new raw RSSI value is always stored."""
        engine = FeatureEngine(window_size=5)
        t = time.time()
        # 3 stale repeats, then a change, then more stale
        for i in range(3):
            engine.update({"a": -65.0}, {"a": -65.0}, t + i)
        engine.update({"a": -70.0}, {"a": -70.0}, t + 3)
        for i in range(3):
            engine.update({"a": -70.0}, {"a": -70.0}, t + 4 + i)
        # Should have exactly 2 entries: -65, -70
        assert list(engine._history["a"]) == [-65.0, -70.0]


# ---------------------------------------------------------------------------
# Dynamic anchor count / dropout
# ---------------------------------------------------------------------------

class TestDynamicAnchors:
    """Verify the feature engine handles varying anchor counts and dropouts."""

    def test_different_anchor_counts_per_floor(self):
        """8 anchors with 2/3/3 distribution produce correct cross-floor stats."""
        anchors = {
            "f1_a": {"x": 1, "y": 1, "floor": 1},
            "f1_b": {"x": 5, "y": 5, "floor": 1},
            "f2_a": {"x": 1, "y": 1, "floor": 2},
            "f2_b": {"x": 5, "y": 5, "floor": 2},
            "f2_c": {"x": 10, "y": 10, "floor": 2},
            "f3_a": {"x": 1, "y": 1, "floor": 3},
            "f3_b": {"x": 5, "y": 5, "floor": 3},
            "f3_c": {"x": 10, "y": 10, "floor": 3},
        }
        engine = FeatureEngine(anchor_positions=anchors, window_size=5)
        raw = {
            "f1_a": -55.0, "f1_b": -58.0,
            "f2_a": -70.0, "f2_b": -72.0, "f2_c": -74.0,
            "f3_a": -85.0, "f3_b": -87.0, "f3_c": -89.0,
        }
        features = engine.update(raw, dict(raw), time.time())

        assert features["n_anchors_floor_1"] == 2
        assert features["n_anchors_floor_2"] == 3
        assert features["n_anchors_floor_3"] == 3
        assert features["rssi_best_floor"] == 1
        assert features["n_reporting"] == 8
        assert features["strongest_anchor"] == "f1_a"
        # 8 rank entries
        assert all(f"rank_{a}" in features for a in raw)

    def test_anchor_dropout_excluded_from_variance(self):
        """When Kalman bank drops a stale anchor from smoothed_rssi,
        its ghost variance must NOT appear in var_mean / var_max."""
        engine = FeatureEngine(
            anchor_positions={
                "alive": {"x": 1, "y": 1, "floor": 1},
                "dying": {"x": 5, "y": 5, "floor": 2},
            },
            window_size=5,
        )
        t = time.time()

        # Both anchors reporting for 5 steps
        for i, v in enumerate([-60, -62, -64, -62, -60]):
            engine.update(
                {"alive": float(v), "dying": float(v - 10)},
                {"alive": float(v), "dying": float(v - 10)},
                t + i,
            )
        both_var_mean = engine.features["var_mean"]
        assert "var_dying" in engine.features

        # "dying" anchor drops off: Kalman bank would exclude it from
        # smoothed_rssi.  raw_rssi still has it (stale rssi_buffer)
        # but smoothed_rssi does not.
        for i, v in enumerate([-58, -56, -54, -56, -58]):
            engine.update(
                {"alive": float(v), "dying": -70.0},  # raw still has stale
                {"alive": float(v)},                    # smoothed excludes it
                t + 5 + i,
            )

        # var_dying should NOT be in aggregate stats
        assert "var_dying" not in engine.features
        assert "var_alive" in engine.features
        # var_mean should reflect only the alive anchor
        assert engine.features["var_mean"] == engine.features["var_alive"]

    def test_anchor_dropout_excluded_from_rankings(self):
        """Dropped anchors should not appear in rankings."""
        engine = FeatureEngine(window_size=5)
        t = time.time()

        # Step 1: both present
        engine.update(
            {"a": -60.0, "b": -70.0},
            {"a": -60.0, "b": -70.0},
            t,
        )
        assert features_has_rank(engine.features, "a")
        assert features_has_rank(engine.features, "b")
        assert engine.features["strongest_anchor"] == "a"

        # Step 2: b drops from smoothed (Kalman stale)
        features = engine.update(
            {"a": -62.0, "b": -70.0},  # raw still has b
            {"a": -62.0},               # smoothed excludes b
            t + 1,
        )
        assert features_has_rank(features, "a")
        assert not features_has_rank(features, "b")
        assert features["strongest_anchor"] == "a"
        assert features["n_reporting"] == 1

    def test_cross_floor_adjusts_to_dropout(self):
        """If all anchors on a floor drop, that floor should have no features."""
        anchors = {
            "f1_a": {"x": 1, "y": 1, "floor": 1},
            "f2_a": {"x": 5, "y": 5, "floor": 2},
        }
        engine = FeatureEngine(anchor_positions=anchors, window_size=5)
        t = time.time()

        # Both floors reporting
        features = engine.update(
            {"f1_a": -60.0, "f2_a": -75.0},
            {"f1_a": -60.0, "f2_a": -75.0},
            t,
        )
        assert features["n_anchors_floor_1"] == 1
        assert features["n_anchors_floor_2"] == 1

        # Floor 2 anchor drops off
        features = engine.update(
            {"f1_a": -58.0, "f2_a": -75.0},
            {"f1_a": -58.0},  # Kalman excluded f2_a
            t + 1,
        )
        assert features["n_anchors_floor_1"] == 1
        assert "n_anchors_floor_2" not in features
        assert features["same_vs_cross_ratio"] == 0.0  # only one floor


def features_has_rank(features: dict, anchor_id: str) -> bool:
    return f"rank_{anchor_id}" in features


# ---------------------------------------------------------------------------
# Displacement-based activity
# ---------------------------------------------------------------------------

class TestDisplacement:
    def test_displacement_zero_without_position(self, engine):
        """No position passed → displacement stays 0."""
        t = time.time()
        for i in range(5):
            engine.update({"living_sw": -60.0 - i}, {"living_sw": -60.0 - i}, t + i)
        assert engine.features["displacement_ft"] == pytest.approx(0.0)

    def test_displacement_zero_stationary(self, engine):
        """Same position every step → displacement = 0."""
        t = time.time()
        for i in range(5):
            engine.update(
                {"living_sw": -65.0}, {"living_sw": -65.0}, t + i,
                position=(5.0, 5.0, 1),
            )
        assert engine.features["displacement_ft"] == pytest.approx(0.0)

    def test_displacement_pythagorean(self, engine):
        """Displacement should be Euclidean distance from oldest to newest."""
        t = time.time()
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t,
                       position=(0.0, 0.0, 2))
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t + 1,
                       position=(3.0, 4.0, 2))
        assert engine.features["displacement_ft"] == pytest.approx(5.0)

    def test_displacement_cross_floor_returns_small_positive(self, engine):
        """Cross-floor displacement returns 1.0 ft (not 0)."""
        t = time.time()
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t,
                       position=(0.0, 0.0, 1))
        engine.update({"living_sw": -65.0}, {"living_sw": -65.0}, t + 1,
                       position=(0.0, 0.0, 2))
        assert engine.features["displacement_ft"] == pytest.approx(1.0)

    def test_displacement_uses_full_history_window(self):
        """Displacement measures oldest-vs-newest in the position buffer."""
        engine = FeatureEngine(anchor_positions=ANCHOR_POSITIONS, window_size=5)
        t = time.time()
        # Walk a path: (0,0) → (1,0) → (2,0) → ... → (5,0)
        for i in range(6):
            engine.update(
                {"living_sw": -65.0 + i}, {"living_sw": -65.0 + i}, t + i,
                position=(float(i), 0.0, 1),
            )
        # Position history holds last POSITION_HISTORY_SIZE entries.
        # Oldest = (0, 0, 1), newest = (5, 0, 1) → displacement = 5.0
        assert engine.features["displacement_ft"] == pytest.approx(5.0)
