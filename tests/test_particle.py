"""
Tests for the Particle Filter localisation engine.

Run with: python -m pytest tests/test_particle.py -v
Or standalone: python tests/test_particle.py
"""

import math
import sys
import os

import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from occupancy import OccupancyGridSet
from filters.particle import (
    ParticleFilter,
    extract_stairways,
    _expected_rssi,
    _log_likelihood,
    _systematic_resample,
    _effective_sample_size,
)

# ---------------------------------------------------------------------------
# Fixtures: build occupancy grids & anchor positions from real layout.json
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)


def _load_grids():
    return OccupancyGridSet.load_from_layout(LAYOUT_PATH, resolution=0.5)


def _load_anchors():
    import json
    with open(LAYOUT_PATH) as f:
        data = json.load(f)
    anchors = {}
    for floor_data in data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {"x": pos[0], "y": pos[1], "floor": floor_num}
    return anchors


def _load_stairways():
    import json
    with open(LAYOUT_PATH) as f:
        data = json.load(f)
    return extract_stairways(data)


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_expected_rssi_decreases_with_distance(self):
        """RSSI drops as distance increases."""
        rssi_near = _expected_rssi(3.0)
        rssi_far = _expected_rssi(30.0)
        assert rssi_near > rssi_far

    def test_expected_rssi_cross_floor_penalty(self):
        """Cross-floor adds effective distance, lowering RSSI."""
        same_floor = _expected_rssi(10.0, floor_diff=0)
        diff_floor = _expected_rssi(10.0, floor_diff=1)
        assert same_floor > diff_floor

    def test_expected_rssi_min_distance_clamp(self):
        """Very small distances are clamped to MIN_DISTANCE_FT."""
        rssi_zero = _expected_rssi(0.0)
        rssi_min = _expected_rssi(1.0)
        assert rssi_zero == rssi_min  # both clamped to 1 ft

    def test_log_likelihood_peaks_at_match(self):
        """Likelihood is highest when observed == expected."""
        ll_match = _log_likelihood(-65.0, -65.0)
        ll_off = _log_likelihood(-65.0, -75.0)
        assert ll_match > ll_off
        assert ll_match == 0.0  # exact match → log-likelihood = 0

    def test_systematic_resample_preserves_count(self):
        """Resampling returns the same number of particles."""
        rng = np.random.default_rng(42)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        indices = _systematic_resample(weights, rng)
        assert len(indices) == len(weights)

    def test_systematic_resample_favours_heavy(self):
        """High-weight particles are chosen more often."""
        rng = np.random.default_rng(0)
        weights = np.array([0.01, 0.01, 0.01, 0.97])
        indices = _systematic_resample(weights, rng)
        # The last particle (weight 0.97) should dominate
        assert np.sum(indices == 3) > len(weights) // 2

    def test_effective_sample_size_uniform(self):
        """N_eff = N when all weights are equal."""
        n = 100
        w = np.full(n, 1.0 / n)
        assert abs(_effective_sample_size(w) - n) < 0.01

    def test_effective_sample_size_degenerate(self):
        """N_eff = 1 when all weight is on one particle."""
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(_effective_sample_size(w) - 1.0) < 0.01


# ===========================================================================
# extract_stairways
# ===========================================================================

class TestExtractStairways:
    def test_stairways_loaded(self):
        """Stairways are extracted from layout.json."""
        stairs = _load_stairways()
        assert len(stairs) > 0
        for s in stairs:
            assert "from_floor" in s
            assert "to_floor" in s
            assert "entry_x" in s
            assert "entry_y" in s

    def test_bidirectional(self):
        """Each floor-to-floor link has a reverse (up/down pair)."""
        stairs = _load_stairways()
        pairs = {(s["from_floor"], s["to_floor"]) for s in stairs}
        for a, b in list(pairs):
            assert (b, a) in pairs, f"Missing reverse stairway {b}→{a}"


# ===========================================================================
# ParticleFilter — initialisation
# ===========================================================================

class TestParticleFilterInit:
    def test_uninitialised_estimate(self):
        """Before initialisation, estimate has zero confidence."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=50, seed=42)
        est = pf.estimate
        assert est["confidence"] == 0.0
        assert est["particle_count"] == 50

    def test_uniform_init_all_floors(self):
        """Uniform init spreads particles across all floors."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=300, seed=42)
        pf.initialise_uniform()
        particles = pf.particles
        assert particles.shape == (300, 3)
        floors_used = set(particles[:, 2].astype(int))
        assert len(floors_used) > 1, "Particles should span multiple floors"

    def test_uniform_init_single_floor(self):
        """Uniform init on floor 2 places all particles there."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=2)
        assert np.all(pf.particles[:, 2] == 2)

    def test_uniform_all_walkable(self):
        """All initialised particles are in walkable cells."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        pf.initialise_uniform()
        for i in range(pf.n):
            f = int(pf._floor[i])
            assert grids.is_walkable(f, pf._x[i], pf._y[i]), (
                f"Particle {i} at ({pf._x[i]:.1f}, {pf._y[i]:.1f}) floor {f} "
                f"is not walkable"
            )

    def test_init_around(self):
        """Particles initialised around a point stay near it."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        # Pick a point known to be walkable on floor 2 (living room area)
        cx, cy = 10.0, 10.0
        pf.initialise_around(cx, cy, floor=2, spread=3.0)
        est = pf.estimate
        assert est["floor"] == 2
        # Mean should be within ~5 ft of the centre
        assert abs(est["x"] - cx) < 5.0
        assert abs(est["y"] - cy) < 5.0

    def test_weights_sum_to_one_after_init(self):
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform()
        assert abs(pf.weights.sum() - 1.0) < 1e-9


# ===========================================================================
# ParticleFilter — predict (motion model)
# ===========================================================================

class TestParticleFilterPredict:
    def test_predict_moves_particles(self):
        """After predict, some particles should have moved."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=2)
        old_x = pf._x.copy()
        old_y = pf._y.copy()
        pf.predict(dt=1.0)
        # At least some particles should have moved
        moved = (pf._x != old_x) | (pf._y != old_y)
        assert np.any(moved), "predict() should move at least some particles"

    def test_predict_respects_walls(self):
        """After predict, all particles should still be walkable."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        pf.initialise_uniform(floor=2)
        for _ in range(10):
            pf.predict(dt=0.5)
        for i in range(pf.n):
            f = int(pf._floor[i])
            assert grids.is_walkable(f, pf._x[i], pf._y[i])

    def test_predict_noop_before_init(self):
        """predict() is a no-op before initialisation."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=50, seed=42)
        pf.predict(dt=1.0)  # should not crash
        assert not pf._initialised

    def test_displacement_scales_with_dt(self):
        """Larger dt allows larger movement spread."""
        grids = _load_grids()
        anchors = _load_anchors()

        pf_short = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        pf_short.initialise_uniform(floor=2)
        old_x = pf_short._x.copy()
        pf_short.predict(dt=0.1)
        disp_short = np.abs(pf_short._x - old_x).mean()

        pf_long = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        pf_long.initialise_uniform(floor=2)
        old_x2 = pf_long._x.copy()
        pf_long.predict(dt=2.0)
        disp_long = np.abs(pf_long._x - old_x2).mean()

        assert disp_long > disp_short, "Longer dt should produce larger displacements"


# ===========================================================================
# ParticleFilter — update (observation model)
# ===========================================================================

class TestParticleFilterUpdate:
    def test_update_changes_weights(self):
        """RSSI observations should change the weight distribution."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=2)
        old_weights = pf.weights.copy()

        # Simulate strong signal from one anchor
        first_anchor = list(anchors.keys())[0]
        pf.update({first_anchor: -50.0})
        new_weights = pf.weights
        assert not np.allclose(old_weights, new_weights), \
            "Weights should change after update"

    def test_update_favours_close_particles(self):
        """Particles near the anchor should get higher weight after update."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=200, seed=42)
        pf.initialise_uniform(floor=2)

        # Find a Floor 2 anchor
        f2_anchors = {k: v for k, v in anchors.items() if v["floor"] == 2}
        if not f2_anchors:
            return  # skip if no F2 anchors
        aid, apos = next(iter(f2_anchors.items()))

        # Strong signal → beacon is right next to this anchor
        pf.update({aid: -45.0})

        # Particles closer to anchor should have higher weight on average
        dx = pf._x - apos["x"]
        dy = pf._y - apos["y"]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        median_dist = np.median(dists)
        close_mask = dists < median_dist
        far_mask = dists >= median_dist
        close_weight = pf.weights[close_mask].mean()
        far_weight = pf.weights[far_mask].mean()
        assert close_weight > far_weight, \
            "Close particles should have higher average weight"

    def test_update_noop_empty_readings(self):
        """Empty RSSI dict should be a no-op."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=50, seed=42)
        pf.initialise_uniform()
        old_w = pf.weights.copy()
        pf.update({})
        assert np.allclose(pf.weights, old_w)

    def test_update_ignores_unknown_anchors(self):
        """Anchors not in anchor_positions should be silently ignored."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=50, seed=42)
        pf.initialise_uniform()
        old_w = pf.weights.copy()
        pf.update({"nonexistent_anchor": -60.0})
        assert np.allclose(pf.weights, old_w)


# ===========================================================================
# ParticleFilter — resampling
# ===========================================================================

class TestParticleFilterResample:
    def test_resample_preserves_count(self):
        """Resampling preserves the particle count."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=2)

        # Force degenerate weights to trigger resampling
        pf._weights[:] = 0
        pf._weights[0] = 1.0
        did_resample = pf.resample_if_needed()
        assert did_resample
        assert len(pf._x) == 100
        assert abs(pf.weights.sum() - 1.0) < 1e-9

    def test_no_resample_with_uniform_weights(self):
        """Uniform weights → no resampling needed."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform()
        did_resample = pf.resample_if_needed()
        assert not did_resample


# ===========================================================================
# ParticleFilter — full step() cycle
# ===========================================================================

class TestParticleFilterStep:
    def test_step_returns_estimate(self):
        """step() returns a dict with expected keys."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=2)
        est = pf.step({"living_sw": -60.0}, dt=0.5)
        assert "x" in est
        assert "y" in est
        assert "floor" in est
        assert "confidence" in est
        assert "n_eff" in est

    def test_convergence_static_beacon(self):
        """Particle filter converges toward a stationary beacon."""
        grids = _load_grids()
        anchors = _load_anchors()
        stairways = _load_stairways()
        pf = ParticleFilter(
            grids, anchors, stairways,
            n_particles=300, seed=42,
        )
        pf.initialise_uniform(floor=2)

        # Beacon is near anchor 'living_sw' at (2.0, 1.0) on floor 2
        beacon_x, beacon_y = 4.0, 3.0

        # Generate synthetic RSSI from beacon position
        f2_anchors = {k: v for k, v in anchors.items() if v["floor"] == 2}
        for step_i in range(40):
            rssi = {}
            for aid, apos in f2_anchors.items():
                dist = math.sqrt(
                    (beacon_x - apos["x"]) ** 2 + (beacon_y - apos["y"]) ** 2
                )
                # Use same path-loss model + noise as simulator
                dist_m = max(dist * 0.3048, 0.3)
                rssi_val = -59.0 - 10 * 2.7 * math.log10(dist_m)
                rssi_val += np.random.default_rng(step_i).normal(0, 3.0)
                rssi[aid] = rssi_val
            pf.step(rssi, dt=0.5)

        est = pf.estimate
        assert est["floor"] == 2
        # After 40 steps, should be within ~8 ft of the true position
        error = math.sqrt((est["x"] - beacon_x) ** 2 + (est["y"] - beacon_y) ** 2)
        assert error < 8.0, f"Position error {error:.1f} ft is too large"
        assert est["confidence"] > 0.1

    def test_multiple_steps_maintain_validity(self):
        """Repeated steps keep all particles in walkable space."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=150, seed=42)
        pf.initialise_uniform(floor=2)

        f2_anchors = {k: v for k, v in anchors.items() if v["floor"] == 2}
        for _ in range(20):
            rssi = {aid: -65.0 for aid in f2_anchors}
            pf.step(rssi, dt=0.5)

        for i in range(pf.n):
            f = int(pf._floor[i])
            assert grids.is_walkable(f, pf._x[i], pf._y[i])


# ===========================================================================
# ParticleFilter — estimate
# ===========================================================================

class TestParticleFilterEstimate:
    def test_estimate_on_correct_floor(self):
        """Estimate floor matches where particles were initialised."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform(floor=1)
        est = pf.estimate
        assert est["floor"] == 1

    def test_confidence_within_bounds(self):
        """Confidence is between 0 and 1."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=100, seed=42)
        pf.initialise_uniform()
        est = pf.estimate
        assert 0 <= est["confidence"] <= 1.0

    def test_repr(self):
        """__repr__ works without error."""
        grids = _load_grids()
        anchors = _load_anchors()
        pf = ParticleFilter(grids, anchors, n_particles=50, seed=42)
        pf.initialise_uniform()
        s = repr(pf)
        assert "ParticleFilter" in s


# ===========================================================================
# Run standalone
# ===========================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
