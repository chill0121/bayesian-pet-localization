"""
End-to-end tests for the integrated inference pipeline.

Verifies that Kalman smoothing → Particle filter → Floor HMM work together
in the run_inference() call chain, producing sensible position estimates.

Run with: python -m pytest tests/test_inference_pipeline.py -v
"""

import json
import math
import os
import sys
import time

import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from filters.kalman import KalmanFilterBank
from filters.particle import ParticleFilter, extract_stairways, _expected_rssi
from filters.floor_hmm import FloorTransitionHMM
from filters.constants import FLOOR_ELEVATION_FT
from occupancy import OccupancyGridSet, bounds_to_polygon, _point_in_polygon

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)


def _load_layout():
    with open(LAYOUT_PATH) as f:
        return json.load(f)


def _load_anchors(layout_data):
    anchors = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {
                "x": pos[0], "y": pos[1], "floor": floor_num,
                "height_ft": a.get("height_ft", 0.0),
            }
    return anchors


def _build_room_polygons(layout_data):
    room_polygons = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        room_polygons[floor_num] = [
            (room["name"], bounds_to_polygon(room["bounds"]))
            for room in floor_data.get("rooms", [])
        ]
    return room_polygons


def _build_room_gates(layout_data):
    """Parse zone gate definitions from layout.json."""
    room_gates = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        for room in floor_data.get("rooms", []):
            gates = room.get("gates", [])
            if not gates:
                continue
            parsed = []
            for gate in gates:
                parsed.append({
                    "axis": gate["axis"],
                    "coord": gate["coord"],
                    "above": gate["between"][0],
                    "below": gate["between"][1],
                })
            room_gates.setdefault(floor_num, {})[room["name"]] = parsed
    return room_gates


def _label_room(x, y, floor, room_polygons, room_gates=None):
    for name, poly in room_polygons.get(floor, []):
        if _point_in_polygon(x, y, poly):
            if room_gates:
                gates = room_gates.get(floor, {}).get(name, [])
                for gate in gates:
                    val = y if gate["axis"] == "y" else x
                    if val >= gate["coord"]:
                        return gate["above"]
                    else:
                        return gate["below"]
            return name
    return "unknown"


def _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors, noise_sigma=4.0, rng=None):
    """Generate simulated RSSI readings from all anchors for a known beacon position.

    Uses 3D distance (with floor elevations) to match the particle filter's
    observation model.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    readings = {}
    beacon_z = FLOOR_ELEVATION_FT.get(beacon_floor, 0.0)
    for aid, apos in anchors.items():
        dx = beacon_x - apos["x"]
        dy = beacon_y - apos["y"]
        anchor_z = FLOOR_ELEVATION_FT.get(apos["floor"], 0.0) + apos.get("height_ft", 0.0)
        dz = beacon_z - anchor_z
        dist_ft = math.sqrt(dx * dx + dy * dy + dz * dz)
        floor_diff = abs(beacon_floor - apos["floor"])
        expected = _expected_rssi(dist_ft, floor_diff)
        readings[aid] = expected + rng.normal(0, noise_sigma)
    return readings


# ---------------------------------------------------------------------------
# Pipeline builder (mirrors main.py startup)
# ---------------------------------------------------------------------------

def _build_pipeline(layout_data, anchors):
    grids = OccupancyGridSet.from_layout_data(layout_data, resolution=0.5)
    stairways = extract_stairways(layout_data)
    hmm = FloorTransitionHMM(layout_data, anchors)
    pf = ParticleFilter(
        occupancy_grids=grids,
        anchor_positions=anchors,
        stairways=stairways,
        floor_hmm=hmm,
        n_particles=500,
        seed=42,
    )
    pf.initialise_uniform()
    kb = KalmanFilterBank(process_noise=0.5, measurement_noise=1.0, stale_timeout=30.0)
    return kb, pf, hmm, grids


# ===========================================================================
# Tests
# ===========================================================================

class TestPipelineInitialisation:
    """Verify all pipeline components can be created from real layout.json."""

    def test_all_components_initialise(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, grids = _build_pipeline(layout, anchors)

        assert kb is not None
        assert pf is not None
        assert hmm is not None
        assert grids is not None
        assert len(grids.floors) == 3

    def test_particle_filter_starts_uniform(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        _, pf, _, _ = _build_pipeline(layout, anchors)

        est = pf.estimate
        assert est["confidence"] > 0
        assert est["particle_count"] == 500


class TestStaticBeacon:
    """Place a beacon at a known position and run several inference cycles."""

    def test_converges_to_floor2_living_room(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, _ = _build_pipeline(layout, anchors)
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)
        rng = np.random.default_rng(123)

        # Beacon sits at (7, 10) on floor 2 (living room / couch area)
        beacon_x, beacon_y, beacon_floor = 7.0, 10.0, 2
        t = time.time()

        for step in range(30):
            rssi = _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors,
                                  noise_sigma=4.0, rng=rng)
            # Kalman smoothing
            for aid, val in rssi.items():
                kb.update(aid, val, t + step * 0.5)
            smoothed = kb.get_smoothed_rssi(t + step * 0.5)

            # Particle filter step
            est = pf.step(smoothed, dt=0.5)

        # After 30 steps the estimate should be close to the beacon
        assert est["floor"] == 2, f"Expected floor 2, got {est['floor']}"
        dist = math.sqrt((est["x"] - beacon_x)**2 + (est["y"] - beacon_y)**2)
        assert dist < 15.0, f"Position error {dist:.1f} ft is too large"
        assert est["confidence"] > 0.1

        # Room label should resolve to the living_room zone (Y < 20.73)
        label = _label_room(est["x"], est["y"], est["floor"], room_polys, gates)
        assert label == "living_room", (
            f"Expected 'living_room' zone, got '{label}' "
            f"at ({est['x']:.1f}, {est['y']:.1f})"
        )

    def test_converges_to_floor3_master_bed(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, _ = _build_pipeline(layout, anchors)
        rng = np.random.default_rng(456)

        beacon_x, beacon_y, beacon_floor = 7.0, 5.0, 3
        t = time.time()

        for step in range(40):
            rssi = _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors,
                                  noise_sigma=4.0, rng=rng)
            for aid, val in rssi.items():
                kb.update(aid, val, t + step * 0.5)
            smoothed = kb.get_smoothed_rssi(t + step * 0.5)
            est = pf.step(smoothed, dt=0.5)

        assert est["floor"] == 3, f"Expected floor 3, got {est['floor']}"
        dist = math.sqrt((est["x"] - beacon_x)**2 + (est["y"] - beacon_y)**2)
        assert dist < 15.0, f"Position error {dist:.1f} ft is too large"

    def test_floor1_office(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, _ = _build_pipeline(layout, anchors)
        rng = np.random.default_rng(789)

        beacon_x, beacon_y, beacon_floor = 4.5, 10.0, 1
        t = time.time()

        for step in range(40):
            rssi = _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors,
                                  noise_sigma=4.0, rng=rng)
            for aid, val in rssi.items():
                kb.update(aid, val, t + step * 0.5)
            smoothed = kb.get_smoothed_rssi(t + step * 0.5)
            est = pf.step(smoothed, dt=0.5)

        assert est["floor"] == 1, f"Expected floor 1, got {est['floor']}"


class TestKalmanSmoothing:
    """Verify the Kalman bank smooths noisy RSSI readings."""

    def test_smoothed_has_lower_variance(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        rng = np.random.default_rng(111)
        kb = KalmanFilterBank(process_noise=0.5, measurement_noise=1.0)

        true_rssi = -65.0
        raw_values = []
        smoothed_values = []
        t = time.time()

        for i in range(50):
            raw = true_rssi + rng.normal(0, 4.0)
            raw_values.append(raw)
            smoothed = kb.update("test_anchor", raw, t + i * 0.5)
            smoothed_values.append(smoothed)

        # Smoothed output should have lower variance than raw
        raw_var = np.var(raw_values)
        smooth_var = np.var(smoothed_values[10:])  # skip initial transient
        assert smooth_var < raw_var, (
            f"Smoothed variance {smooth_var:.2f} >= raw variance {raw_var:.2f}"
        )


class TestFloorHMMIntegration:
    """Verify the HMM tracks floor changes as part of the full pipeline."""

    def test_hmm_belief_matches_beacon_floor(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, _ = _build_pipeline(layout, anchors)
        rng = np.random.default_rng(222)

        # Start on floor 2
        beacon_x, beacon_y, beacon_floor = 7.0, 10.0, 2
        t = time.time()
        for step in range(20):
            rssi = _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors,
                                  noise_sigma=4.0, rng=rng)
            for aid, val in rssi.items():
                kb.update(aid, val, t + step * 0.5)
            smoothed = kb.get_smoothed_rssi(t + step * 0.5)
            pf.step(smoothed, dt=0.5)

        belief = hmm.floor_belief
        assert belief[2] > belief[1], "Floor 2 belief should dominate over floor 1"
        assert belief[2] > belief[3], "Floor 2 belief should dominate over floor 3"


class TestRoomLabeling:
    """Verify room labeling from coordinate lookup."""

    def test_known_positions_labeled_correctly(self):
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)

        # Floor 1 office
        assert _label_room(4.5, 10.0, 1, room_polys, gates) == "office"
        # Floor 3 master bedroom
        assert _label_room(7.0, 5.0, 3, room_polys, gates) == "master_bed"

    def test_outside_building_returns_unknown(self):
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)
        assert _label_room(100.0, 100.0, 1, room_polys, gates) == "unknown"

    def test_zone_living_room_south_of_gate(self):
        """Points south of the peninsula gate (Y < 20.73) → living_room."""
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)

        # Couch area — well south of the Y=20.73 gate line
        label = _label_room(7.0, 10.0, 2, room_polys, gates)
        assert label == "living_room", f"Expected 'living_room', got '{label}'"

        # Near the south end of the room
        label = _label_room(10.0, 5.0, 2, room_polys, gates)
        assert label == "living_room", f"Expected 'living_room', got '{label}'"

    def test_zone_kitchen_north_of_gate(self):
        """Points north of the peninsula gate (Y > 20.73) → kitchen."""
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)

        # Kitchen area — north of Y=20.73
        label = _label_room(10.0, 25.0, 2, room_polys, gates)
        assert label == "kitchen", f"Expected 'kitchen', got '{label}'"

    def test_zone_at_gate_boundary(self):
        """Point exactly at the gate line (Y == 20.73) → kitchen (>= convention)."""
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)

        label = _label_room(7.0, 20.73, 2, room_polys, gates)
        assert label == "kitchen", f"Expected 'kitchen' at boundary, got '{label}'"

    def test_powder_room_not_affected_by_zones(self):
        """Rooms without zones still return their room name."""
        layout = _load_layout()
        room_polys = _build_room_polygons(layout)
        gates = _build_room_gates(layout)

        label = _label_room(2.37, 26.86, 2, room_polys, gates)
        assert label == "powder_room", f"Expected 'powder_room', got '{label}'"


class TestWalkingBeacon:
    """Simulate a beacon moving between rooms over time."""

    def test_beacon_walks_across_floor2(self):
        layout = _load_layout()
        anchors = _load_anchors(layout)
        kb, pf, hmm, _ = _build_pipeline(layout, anchors)
        rng = np.random.default_rng(333)

        # Walk from x=5 to x=25 across floor 2 (living → kitchen)
        t = time.time()
        positions_x = np.linspace(5.0, 25.0, 40)
        for step, bx in enumerate(positions_x):
            rssi = _simulate_rssi(bx, 8.0, 2, anchors, noise_sigma=4.0, rng=rng)
            for aid, val in rssi.items():
                kb.update(aid, val, t + step * 0.5)
            smoothed = kb.get_smoothed_rssi(t + step * 0.5)
            est = pf.step(smoothed, dt=0.5)

        # Final estimate should be on floor 2 and roughly in the kitchen area (x > 15)
        assert est["floor"] == 2
        assert est["x"] > 10.0, f"Final x={est['x']:.1f}, expected > 10 (kitchen area)"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
