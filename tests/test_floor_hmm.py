"""
Tests for the Floor Transition HMM.

Run with: python -m pytest tests/test_floor_hmm.py -v
Or standalone: python tests/test_floor_hmm.py
"""

import math
import sys
import os
import json

import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from filters.floor_hmm import (
    FloorTransitionHMM,
    _build_adjacency,
    _parse_stair_entries,
    _expected_rssi,
    _emission_log_likelihood,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)


def _load_layout():
    with open(LAYOUT_PATH) as f:
        return json.load(f)


def _load_anchors():
    data = _load_layout()
    anchors = {}
    for floor_data in data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {"x": pos[0], "y": pos[1], "floor": floor_num}
    return anchors


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    """Tests for module-level helper functions."""

    def test_build_adjacency_from_layout(self):
        """adjacency graph has correct connections."""
        data = _load_layout()
        adj = _build_adjacency(data)
        # Floor 1 connects to 2
        assert 2 in adj[1]
        # Floor 2 connects to 1 and 3
        assert 1 in adj[2]
        assert 3 in adj[2]
        # Floor 3 connects to 2
        assert 2 in adj[3]
        # Floor 1 does NOT directly connect to 3
        assert 3 not in adj[1]
        # Floor 3 does NOT directly connect to 1
        assert 1 not in adj[3]

    def test_parse_stair_entries(self):
        """stair entries are correctly extracted."""
        data = _load_layout()
        entries = _parse_stair_entries(data)
        assert len(entries) >= 3  # 1→2, 2→1, 2→3, 3→2

        # Check that floor 1 has an entry going to floor 2
        f1_to_f2 = [e for e in entries if e["from_floor"] == 1 and e["to_floor"] == 2]
        assert len(f1_to_f2) == 1
        assert f1_to_f2[0]["entry_x"] > 0
        assert f1_to_f2[0]["entry_y"] > 0

    def test_expected_rssi_same_floor_stronger(self):
        """Same-floor RSSI is stronger than cross-floor at same distance."""
        rssi_same = _expected_rssi(10.0, floor_diff=0)
        rssi_cross = _expected_rssi(10.0, floor_diff=1)
        assert rssi_same > rssi_cross

    def test_emission_log_likelihood_favours_correct_floor(self):
        """Emission model assigns higher likelihood to the correct floor."""
        anchors = {
            "a1": {"x": 5.0, "y": 5.0, "floor": 2},
            "a2": {"x": 10.0, "y": 10.0, "floor": 2},
            "a3": {"x": 5.0, "y": 5.0, "floor": 1},
            "a4": {"x": 5.0, "y": 5.0, "floor": 3},
        }
        # Strong RSSI from floor-2 anchors, weak from 1 and 3 → likely on floor 2
        rssi = {"a1": -55.0, "a2": -60.0, "a3": -72.0, "a4": -70.0}
        ll_f2 = _emission_log_likelihood(rssi, anchors, floor_hypothesis=2)
        ll_f1 = _emission_log_likelihood(rssi, anchors, floor_hypothesis=1)
        ll_f3 = _emission_log_likelihood(rssi, anchors, floor_hypothesis=3)
        assert ll_f2 > ll_f1
        assert ll_f2 > ll_f3


# ===========================================================================
# FloorTransitionHMM construction
# ===========================================================================

class TestHMMConstruction:
    """Tests for HMM initialisation and structure."""

    def test_init_with_layout(self):
        """HMM initialises from real layout.json."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        assert hmm.floors == [1, 2, 3]
        assert len(hmm.stair_entries) >= 3

    def test_uniform_prior(self):
        """Default prior is uniform over floors."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        belief = hmm.floor_belief
        for f in [1, 2, 3]:
            assert abs(belief[f] - 1.0 / 3) < 1e-6

    def test_initial_floor_prior(self):
        """initial_floor concentrates belief on one floor."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=2)
        belief = hmm.floor_belief
        assert belief[2] > 0.99
        assert belief[1] < 0.01
        assert belief[3] < 0.01

    def test_transition_matrix_shape(self):
        """Transition matrix is 3×3 row-stochastic."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        T = hmm.transition_matrix
        assert T.shape == (3, 3)
        # Row-stochastic: each row sums to 1
        for i in range(3):
            assert abs(T[i].sum() - 1.0) < 1e-10

    def test_transition_matrix_no_skip(self):
        """Floor 1→3 and 3→1 have zero probability in base matrix."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        T = hmm.transition_matrix
        # Floor indices: 0=F1, 1=F2, 2=F3
        assert T[0, 2] == 0.0  # F1 → F3
        assert T[2, 0] == 0.0  # F3 → F1

    def test_transition_matrix_self_dominant(self):
        """Self-transition (stay on floor) has the highest probability."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        T = hmm.transition_matrix
        for i in range(3):
            assert T[i, i] > 0.9

    def test_repr(self):
        """repr doesn't raise."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)
        r = repr(hmm)
        assert "FloorTransitionHMM" in r


# ===========================================================================
# Forward inference
# ===========================================================================

class TestForwardInference:
    """Tests for predict / update / step."""

    def test_predict_preserves_probability(self):
        """Belief sums to 1.0 after predict."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=1)
        hmm.predict(dt=1.0)
        belief = hmm.floor_belief
        assert abs(sum(belief.values()) - 1.0) < 1e-10

    def test_predict_spreads_belief(self):
        """Predicting from concentrated belief spreads probability to neighbours."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=1)
        hmm.predict(dt=1.0)
        belief = hmm.floor_belief
        # Floor 2 should have gained some probability
        assert belief[2] > 0.0
        # Floor 3 should still be 0 (no direct connection from 1)
        assert belief[3] < 1e-10

    def test_predict_from_floor2_spreads_both_ways(self):
        """From floor 2, predict spreads to both floor 1 and floor 3."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=2)
        hmm.predict(dt=1.0)
        belief = hmm.floor_belief
        assert belief[1] > 0.0
        assert belief[3] > 0.0

    def test_update_with_floor2_evidence(self):
        """Strong floor-2 RSSI pushes belief toward floor 2."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)  # uniform prior

        # Strong signals from floor-2 anchors, weaker cross-floor
        rssi = {
            "2F_Living_Center": -50.0,
            "2F_Kitchen_NE": -55.0,
            "1F_Office": -70.0,
            "3F_Master_Bed": -72.0,
        }
        hmm.update(rssi)
        belief = hmm.floor_belief
        assert belief[2] > belief[1]
        assert belief[2] > belief[3]

    def test_update_with_floor1_evidence(self):
        """Strong floor-1 RSSI pushes belief toward floor 1."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)

        rssi = {
            "1F_Office": -45.0,
            "2F_Living_Center": -62.0,
            "3F_Master_Bed": -78.0,
        }
        hmm.update(rssi)
        belief = hmm.floor_belief
        assert belief[1] > belief[2]
        assert belief[1] > belief[3]

    def test_step_full_cycle(self):
        """step() runs predict+update and returns valid belief."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=2)
        belief = hmm.step({"2F_Living_Center": -55.0}, dt=1.0)
        assert abs(sum(belief.values()) - 1.0) < 1e-10
        assert hmm.most_likely_floor == 2

    def test_repeated_evidence_converges(self):
        """Repeated same-floor evidence drives belief toward that floor."""
        data = _load_layout()
        anchors = _load_anchors()
        # Start on floor 1 (concentrated)
        hmm = FloorTransitionHMM(data, anchors, initial_floor=1)

        # Repeatedly observe strong floor-2 signals with weaker cross-floor
        for _ in range(20):
            hmm.step({
                "2F_Living_Center": -50.0,
                "2F_Kitchen_NE": -52.0,
                "1F_Office": -68.0,
                "3F_Master_Bed": -70.0,
            }, dt=1.0)

        belief = hmm.floor_belief
        assert belief[2] > 0.9
        assert hmm.most_likely_floor == 2

    def test_floor_skip_impossible(self):
        """Starting on floor 1 with no evidence, floor 3 stays unreachable for short time."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=1)
        # One predict step — floor 3 should be negligible
        hmm.predict(dt=1.0)
        belief = hmm.floor_belief
        assert belief[3] < 1e-8

    def test_floor3_reachable_via_floor2(self):
        """After many steps, floor 3 becomes reachable through floor 2."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors, initial_floor=1)
        # Many predict-only steps → probability diffuses through floor 2
        for _ in range(100):
            hmm.predict(dt=1.0)
        belief = hmm.floor_belief
        # Floor 3 should now have some non-negligible probability
        assert belief[3] > 0.01

    def test_stair_proximity_boost(self):
        """Being near a stairway increases transition probability."""
        data = _load_layout()
        anchors = _load_anchors()

        # Without proximity — predict from floor 1
        hmm1 = FloorTransitionHMM(data, anchors, initial_floor=1)
        hmm1.predict(dt=1.0, stair_proximity=None)
        b1 = hmm1.floor_belief

        # With proximity — right at stairway
        hmm2 = FloorTransitionHMM(data, anchors, initial_floor=1)
        hmm2.predict(dt=1.0, stair_proximity={1: 0.5})  # 0.5 ft from stair
        b2 = hmm2.floor_belief

        # Proximity to stairway should increase floor-2 belief
        assert b2[2] > b1[2]


# ===========================================================================
# Stair proximity helper
# ===========================================================================

class TestStairProximity:
    """Tests for stair_proximity_for_position."""

    def test_at_stairway_entry(self):
        """Distance is near zero when standing at a stair entry."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)

        # Floor 1 stair entry is at (1.47, 20.1)
        prox = hmm.stair_proximity_for_position(1.47, 20.1, floor=1)
        assert 1 in prox
        assert prox[1] < 0.1

    def test_far_from_stairway(self):
        """Distance is large when far from any stair entry."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)

        # Far corner of floor 1 office
        prox = hmm.stair_proximity_for_position(5.0, 3.0, floor=1)
        assert 1 in prox
        assert prox[1] > 15.0  # far from stairway

    def test_different_floor_not_included(self):
        """Proximity only computed for stair entries on the given floor."""
        data = _load_layout()
        anchors = _load_anchors()
        hmm = FloorTransitionHMM(data, anchors)

        prox = hmm.stair_proximity_for_position(5.0, 5.0, floor=1)
        # Should not contain floor 2 or 3 entries (those are on other floors)
        assert 2 not in prox
        assert 3 not in prox


# ===========================================================================
# Integration with particle filter
# ===========================================================================

class TestParticleFilterIntegration:
    """Verify HMM integrates with particle filter."""

    def test_particle_filter_accepts_hmm(self):
        """ParticleFilter can be constructed with a floor_hmm."""
        from occupancy import OccupancyGridSet
        from filters.particle import ParticleFilter, extract_stairways

        data = _load_layout()
        grids = OccupancyGridSet.load_from_layout(LAYOUT_PATH, resolution=0.5)
        anchors = _load_anchors()
        stairways = extract_stairways(data)
        hmm = FloorTransitionHMM(data, anchors, initial_floor=2)

        pf = ParticleFilter(
            grids, anchors, stairways=stairways,
            floor_hmm=hmm, n_particles=100, seed=42,
        )
        pf.initialise_uniform(floor=2)

        # Step should work without error
        est = pf.step({"2F_Living_Center": -55.0}, dt=1.0)
        assert "floor" in est
        assert "x" in est

    def test_hmm_driven_by_step(self):
        """ParticleFilter.step() updates the HMM's belief."""
        from occupancy import OccupancyGridSet
        from filters.particle import ParticleFilter, extract_stairways

        data = _load_layout()
        grids = OccupancyGridSet.load_from_layout(LAYOUT_PATH, resolution=0.5)
        anchors = _load_anchors()
        stairways = extract_stairways(data)
        hmm = FloorTransitionHMM(data, anchors, initial_floor=2)

        pf = ParticleFilter(
            grids, anchors, stairways=stairways,
            floor_hmm=hmm, n_particles=100, seed=42,
        )
        pf.initialise_uniform(floor=2)

        initial_belief = dict(hmm.floor_belief)
        pf.step({"2F_Living_Center": -50.0, "2F_Kitchen_NE": -52.0}, dt=1.0)
        updated_belief = hmm.floor_belief

        # Belief should have changed after step
        assert updated_belief != initial_belief


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
