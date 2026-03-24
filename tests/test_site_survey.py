"""
Tests for the site survey collection script.

Tests grid generation, aggregation, room labeling, progress save/load,
and point-in-polygon filtering. Does NOT test MQTT collection or DB writes
(those require live infrastructure).

Run with: python -m pytest tests/test_site_survey.py -v
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

# Allow importing from scripts/ and services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from occupancy import OccupancyGridSet

from site_survey import (
    aggregate_readings,
    build_room_gates,
    build_room_polygons,
    generate_grid_points,
    label_room,
    load_anchors,
    load_layout,
    load_progress,
    save_progress,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)


def _setup():
    """Load layout data, grids, polygons, and gates."""
    layout = load_layout(LAYOUT_PATH)
    grids = OccupancyGridSet.from_layout_data(layout, resolution=0.5)
    room_polygons = build_room_polygons(layout)
    room_gates = build_room_gates(layout)
    anchors = load_anchors(layout)
    return layout, grids, room_polygons, room_gates, anchors


# ---------------------------------------------------------------------------
# Grid generation tests
# ---------------------------------------------------------------------------

class TestGridGeneration:
    """Test that grid points are generated correctly."""

    def test_all_points_are_walkable(self):
        """Every generated grid point must be in walkable space."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        for floor in [1, 2, 3]:
            points = generate_grid_points(
                layout, grids, room_polygons, room_gates, floor, resolution=2.0
            )
            for p in points:
                assert grids[floor].is_walkable(p["x"], p["y"]), (
                    f"Point ({p['x']}, {p['y']}) on floor {floor} "
                    f"is not walkable"
                )

    def test_points_have_valid_rooms(self):
        """Every point should have a non-'unknown' room label."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        for floor in [1, 2, 3]:
            points = generate_grid_points(
                layout, grids, room_polygons, room_gates, floor, resolution=2.0
            )
            # Allow 'unknown' for a small fraction (edge cases at walls)
            unknown = [p for p in points if p["room"] == "unknown"]
            assert len(unknown) < len(points) * 0.1, (
                f"Too many unknown rooms on floor {floor}: "
                f"{len(unknown)}/{len(points)}"
            )

    def test_grid_point_count_reasonable(self):
        """At 2 ft resolution, should get a reasonable number of points."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        # Floor 2 is the largest open area
        points_f2 = generate_grid_points(
            layout, grids, room_polygons, room_gates, 2, resolution=2.0
        )
        # Should have at least 20 points (open plan + staircase + powder)
        assert len(points_f2) >= 20, (
            f"Too few points on floor 2: {len(points_f2)}"
        )
        # Should not exceed ~200 (sanity ceiling)
        assert len(points_f2) <= 200, (
            f"Too many points on floor 2: {len(points_f2)}"
        )

    def test_coarser_grid_has_fewer_points(self):
        """Larger resolution should produce fewer points."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        points_2ft = generate_grid_points(
            layout, grids, room_polygons, room_gates, 2, resolution=2.0
        )
        points_4ft = generate_grid_points(
            layout, grids, room_polygons, room_gates, 2, resolution=4.0
        )
        assert len(points_4ft) < len(points_2ft)

    def test_all_floors_produce_points(self):
        """Each floor should produce at least some grid points."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        for floor in [1, 2, 3]:
            points = generate_grid_points(
                layout, grids, room_polygons, room_gates, floor, resolution=2.0
            )
            assert len(points) > 0, f"No points generated for floor {floor}"

    def test_no_duplicate_coordinates(self):
        """Grid should not contain duplicate (x, y) positions."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        for floor in [1, 2, 3]:
            points = generate_grid_points(
                layout, grids, room_polygons, room_gates, floor, resolution=2.0
            )
            coords = [(p["x"], p["y"]) for p in points]
            assert len(coords) == len(set(coords)), (
                f"Duplicate coordinates on floor {floor}"
            )

    def test_point_types_present(self):
        """Should generate both 'grid' and special point types."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        points = generate_grid_points(
            layout, grids, room_polygons, room_gates, 2, resolution=2.0
        )
        types = {p["type"] for p in points}
        assert "grid" in types, "No regular grid points"
        # Floor 2 has doorways and stairs
        assert len(types) >= 2, f"Only one point type: {types}"

    def test_required_fields_present(self):
        """Each point should have all required fields."""
        layout, grids, room_polygons, room_gates, _ = _setup()
        points = generate_grid_points(
            layout, grids, room_polygons, room_gates, 2, resolution=2.0
        )
        required = {"x", "y", "floor", "room", "type"}
        for p in points:
            assert required.issubset(p.keys()), (
                f"Missing keys: {required - p.keys()}"
            )


# ---------------------------------------------------------------------------
# Room labeling tests
# ---------------------------------------------------------------------------

class TestRoomLabeling:
    """Test room/zone labeling logic."""

    def test_floor2_kitchen_zone(self):
        """Point north of peninsula gate should be 'kitchen'."""
        _, _, room_polygons, room_gates, _ = _setup()
        label = label_room(10.0, 25.0, 2, room_polygons, room_gates)
        assert label == "kitchen", f"Expected 'kitchen', got '{label}'"

    def test_floor2_living_room_zone(self):
        """Point south of peninsula gate should be 'living_room'."""
        _, _, room_polygons, room_gates, _ = _setup()
        label = label_room(10.0, 10.0, 2, room_polygons, room_gates)
        assert label == "living_room", f"Expected 'living_room', got '{label}'"

    def test_floor1_office(self):
        """Point in the office should be labeled 'office'."""
        _, _, room_polygons, room_gates, _ = _setup()
        label = label_room(3.0, 6.0, 1, room_polygons, room_gates)
        assert label == "office", f"Expected 'office', got '{label}'"

    def test_floor3_master_bed(self):
        """Point in master bedroom area."""
        _, _, room_polygons, room_gates, _ = _setup()
        label = label_room(10.0, 8.0, 3, room_polygons, room_gates)
        assert label == "master_bed", f"Expected 'master_bed', got '{label}'"

    def test_floor2_powder_room(self):
        """Point inside powder room."""
        _, _, room_polygons, room_gates, _ = _setup()
        label = label_room(2.0, 27.0, 2, room_polygons, room_gates)
        assert label == "powder_room", f"Expected 'powder_room', got '{label}'"


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregation:
    """Test RSSI reading aggregation."""

    def test_basic_aggregation(self):
        """Mean and std are computed correctly."""
        readings = {
            "anchor_a": [-60.0, -62.0, -58.0, -61.0, -59.0],
            "anchor_b": [-70.0, -72.0, -68.0, -71.0, -69.0],
        }
        result = aggregate_readings(readings)
        assert result["anchor_count"] == 2
        assert result["n_readings"] == 5
        # Mean of anchor_a should be -60.0
        assert abs(result["rssi_vector"]["anchor_a"] - (-60.0)) < 0.5
        # Mean of anchor_b should be -70.0
        assert abs(result["rssi_vector"]["anchor_b"] - (-70.0)) < 0.5
        # Std should be reasonable
        assert result["rssi_std"]["anchor_a"] > 0
        assert result["rssi_std"]["anchor_b"] > 0

    def test_single_reading(self):
        """Single reading should produce mean = that reading, std = 0."""
        readings = {"anchor_a": [-65.0]}
        result = aggregate_readings(readings)
        assert result["rssi_vector"]["anchor_a"] == -65.0
        assert result["rssi_std"]["anchor_a"] == 0.0
        assert result["n_readings"] == 1

    def test_empty_readings(self):
        """Empty dict should produce empty results."""
        result = aggregate_readings({})
        assert result["anchor_count"] == 0
        assert result["n_readings"] == 0

    def test_high_variance_warning(self):
        """High-variance anchor should generate a warning."""
        # Create readings with high variance
        readings = {
            "anchor_a": [-50.0, -70.0, -45.0, -75.0, -55.0, -65.0],
        }
        result = aggregate_readings(readings)
        assert len(result["warnings"]) > 0
        assert "anchor_a" in result["warnings"][0]

    def test_many_readings_accuracy(self):
        """With many readings, mean should converge to true value."""
        np.random.seed(42)
        true_rssi = -65.0
        noisy = list(true_rssi + np.random.normal(0, 4.0, 200))
        readings = {"anchor_a": noisy}
        result = aggregate_readings(readings)
        assert abs(result["rssi_vector"]["anchor_a"] - true_rssi) < 1.0
        assert result["n_readings"] == 200


# ---------------------------------------------------------------------------
# Progress / resume tests
# ---------------------------------------------------------------------------

class TestProgress:
    """Test progress save/load for survey resumption."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Progress should round-trip through save/load."""
        # Monkeypatch the progress directory
        import site_survey
        monkeypatch.setattr(site_survey, "PROGRESS_DIR", str(tmp_path))

        completed = [0, 2, 5, 7]
        save_progress(floor=2, completed_indices=completed,
                      survey_id="test_001")

        loaded, sid = load_progress(floor=2)
        assert loaded == set(completed)
        assert sid == "test_001"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        """Loading from nonexistent file returns empty set."""
        import site_survey
        monkeypatch.setattr(site_survey, "PROGRESS_DIR", str(tmp_path))

        loaded, sid = load_progress(floor=99)
        assert loaded == set()
        assert sid == ""


# ---------------------------------------------------------------------------
# Anchor loading tests
# ---------------------------------------------------------------------------

class TestAnchors:
    """Test anchor loading from layout."""

    def test_anchors_loaded(self):
        """Should load at least one anchor."""
        layout = load_layout(LAYOUT_PATH)
        anchors = load_anchors(layout)
        assert len(anchors) > 0

    def test_anchor_has_required_fields(self):
        """Each anchor should have x, y, floor."""
        layout = load_layout(LAYOUT_PATH)
        anchors = load_anchors(layout)
        for aid, pos in anchors.items():
            assert "x" in pos, f"Anchor {aid} missing x"
            assert "y" in pos, f"Anchor {aid} missing y"
            assert "floor" in pos, f"Anchor {aid} missing floor"


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
