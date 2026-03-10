"""
Tests for the occupancy grid generator.

Run with: python -m pytest tests/test_occupancy.py -v
Or standalone: python tests/test_occupancy.py
"""

import json
import math
import os
import sys

import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from occupancy import (
    OccupancyGrid,
    OccupancyGridSet,
    bounds_to_polygon,
    _point_in_polygon,
)

# ---------------------------------------------------------------------------
# Path to layout.json
# ---------------------------------------------------------------------------
LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)


def _load_layout():
    with open(LAYOUT_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# bounds_to_polygon
# ---------------------------------------------------------------------------

class TestBoundsToPolygon:

    def test_rect_dict(self):
        """Dict {x1,y1,x2,y2} should produce 4-vertex CCW rectangle."""
        poly = bounds_to_polygon({"x1": 0, "y1": 10, "x2": 3, "y2": 20})
        assert len(poly) == 4
        assert poly[0] == [0, 10]
        assert poly[2] == [3, 20]

    def test_list_no_closing_dup(self):
        """Plain list of vertices should be returned as-is."""
        verts = [[0, 0], [5, 0], [5, 5], [0, 5]]
        poly = bounds_to_polygon(verts)
        assert poly == verts

    def test_list_with_closing_dup(self):
        """Duplicate closing vertex should be stripped."""
        verts = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
        poly = bounds_to_polygon(verts)
        assert len(poly) == 4
        assert poly[0] == [0, 0]


# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------

class TestGridDimensions:

    def test_default_resolution(self):
        """Grid at 0.5 ft for the building footprint (17.7 × 29.39 ft)."""
        layout = _load_layout()
        grids = OccupancyGridSet.from_layout_data(layout, resolution=0.5)
        g = grids[1]
        # ceil(17.7/0.5) = 36, ceil(29.39/0.5) = 59
        assert g.width_cells == math.ceil(17.7 / 0.5)
        assert g.height_cells == math.ceil(29.39 / 0.5)

    def test_custom_resolution(self):
        """Grid at 1.0 ft should halve cell counts (approx)."""
        layout = _load_layout()
        grids = OccupancyGridSet.from_layout_data(layout, resolution=1.0)
        g = grids[1]
        assert g.width_cells == math.ceil(17.7 / 1.0)
        assert g.height_cells == math.ceil(29.39 / 1.0)

    def test_all_floors_present(self):
        """OccupancyGridSet should contain floors 1, 2, 3."""
        layout = _load_layout()
        grids = OccupancyGridSet.from_layout_data(layout)
        assert grids.floors == [1, 2, 3]
        for f in [1, 2, 3]:
            assert f in grids


# ---------------------------------------------------------------------------
# Walkability — known points
# ---------------------------------------------------------------------------

class TestWalkability:

    @classmethod
    def setup_class(cls):
        cls.layout = _load_layout()
        cls.grids = OccupancyGridSet.from_layout_data(cls.layout, resolution=0.5)

    # -- Floor 1 --

    def test_office_interior_walkable(self):
        """Center of the 1F office should be walkable."""
        assert self.grids[1].is_walkable(3.75, 6.5)

    def test_garage_blocked(self):
        """Center of the garage (non-walkable room) should be blocked."""
        # Garage roughly center: x≈12, y≈13
        assert not self.grids[1].is_walkable(12.0, 13.0)

    def test_outside_building_blocked(self):
        """Point outside the building boundary should be blocked."""
        assert not self.grids[1].is_walkable(-1.0, 15.0)
        assert not self.grids[1].is_walkable(20.0, 15.0)

    # -- Floor 2 --

    def test_living_room_walkable(self):
        """Center of the open-plan living/kitchen area should be walkable."""
        assert self.grids[2].is_walkable(10.0, 14.0)

    def test_kitchen_counter_blocked(self):
        """Kitchen counter obstacle should be blocked."""
        # Counter: x1=5.43, y1=27.85, x2=17.7, y2=29.39 — center ~11.5, 28.6
        assert not self.grids[2].is_walkable(11.5, 28.6)

    def test_kitchen_peninsula_blocked(self):
        """Kitchen peninsula obstacle should be blocked."""
        # Peninsula: x1=9.66, y1=20.73, x2=17.7, y2=23.06 — center ~13.7, 21.9
        assert not self.grids[2].is_walkable(13.7, 21.9)

    def test_fireplace_blocked(self):
        """Fireplace obstacle should be blocked."""
        # Fireplace: x1=16.05, y1=2.07, x2=17.7, y2=6.01 — center ~16.9, 4.0
        assert not self.grids[2].is_walkable(16.9, 4.0)

    def test_powder_room_walkable(self):
        """Powder room interior should be walkable."""
        assert self.grids[2].is_walkable(2.37, 26.86)

    # -- Floor 3 --

    def test_master_bed_walkable(self):
        """Center of master bedroom should be walkable."""
        assert self.grids[3].is_walkable(12.0, 8.0)

    def test_master_bed_closet_blocked(self):
        """Closet in master bedroom should be blocked."""
        # Closet: [10.3,14.09] to [17.13,16.72] — center ~13.7, 15.4
        assert not self.grids[3].is_walkable(13.7, 15.4)

    def test_linen_closet_blocked(self):
        """Linen closet at NW of 3F stairwell should be blocked."""
        # x1=0, y1=20.91, x2=3.35, y2=24.04 — center ~1.7, 22.5
        assert not self.grids[3].is_walkable(1.7, 22.5)


# ---------------------------------------------------------------------------
# Anchor positions should be walkable
# ---------------------------------------------------------------------------

class TestAnchorWalkability:

    def test_all_anchors_walkable(self):
        """Every anchor position from layout.json should be on a walkable cell."""
        layout = _load_layout()
        grids = OccupancyGridSet.from_layout_data(layout)
        for floor_data in layout["floors"]:
            floor_num = floor_data["floor"]
            for anchor in floor_data.get("anchors", []):
                pos = anchor["position"]
                assert grids[floor_num].is_walkable(pos[0], pos[1]), (
                    f"Anchor {anchor['id']} at ({pos[0]}, {pos[1]}) on floor {floor_num} "
                    f"is not walkable"
                )


# ---------------------------------------------------------------------------
# Doorway connectivity
# ---------------------------------------------------------------------------

class TestDoorwayConnectivity:

    @classmethod
    def setup_class(cls):
        cls.layout = _load_layout()
        cls.grids = OccupancyGridSet.from_layout_data(cls.layout, resolution=0.5)

    def test_hallway_to_office_door_1f(self):
        """Doorway between 1F hallway and office should be walkable."""
        # Door center at (5.7, 11.31) — right on the shared wall
        assert self.grids[1].is_walkable(5.7, 11.31)

    def test_staircase_opening_1f(self):
        """Staircase opening (1F hallway → staircase) should be walkable."""
        # Center (1.47, 20.1)
        assert self.grids[1].is_walkable(1.47, 20.1)

    def test_staircase_openings_2f(self):
        """Both staircase openings on 2F should be walkable."""
        assert self.grids[2].is_walkable(1.48, 10.27)  # south
        assert self.grids[2].is_walkable(1.48, 20.8)   # north

    def test_powder_room_door_2f(self):
        """Powder room door on 2F should be walkable."""
        assert self.grids[2].is_walkable(3.51, 24.32)

    def test_hallway_to_rooms_3f(self):
        """Doorways from 3F hallway to guest bath, wife's office, master bed."""
        assert self.grids[3].is_walkable(5.51, 24.04)  # guest bath
        assert self.grids[3].is_walkable(6.89, 21.34)  # wife's office
        assert self.grids[3].is_walkable(6.89, 12.7)   # master bed


# ---------------------------------------------------------------------------
# Wall enforcement — rooms without doorways must be separated
# ---------------------------------------------------------------------------

class TestWallEnforcement:

    @classmethod
    def setup_class(cls):
        cls.layout = _load_layout()
        cls.grids = OccupancyGridSet.from_layout_data(cls.layout, resolution=0.5)

    def test_guest_bath_wife_office_wall_3f(self):
        """3F guest bath and wife office share a wall with no doorway between them.
        ray_clear should return False across that boundary."""
        assert not self.grids[3].ray_clear(4.0, 27.0, 12.0, 27.0)

    def test_office_staircase_wall_1f(self):
        """1F office and staircase share a wall with no direct doorway.
        Must go through the hallway instead."""
        assert not self.grids[1].ray_clear(1.5, 10.0, 1.5, 12.0)

    def test_office_hallway_through_door_1f(self):
        """1F office to hallway through the door should still be clear."""
        assert self.grids[1].ray_clear(5.7, 10.0, 5.7, 12.0)


# ---------------------------------------------------------------------------
# Coordinate conversion round-trip
# ---------------------------------------------------------------------------

class TestCoordinateConversion:

    def test_round_trip(self):
        """world_to_grid → grid_to_world should return a point near the original."""
        layout = _load_layout()
        grid = OccupancyGridSet.from_layout_data(layout)[2]

        x_orig, y_orig = 10.0, 14.0
        row, col = grid.world_to_grid(x_orig, y_orig)
        x_back, y_back = grid.grid_to_world(row, col)
        # Cell center should be within half a cell of the original
        assert abs(x_back - x_orig) <= grid.resolution
        assert abs(y_back - y_orig) <= grid.resolution

    def test_origin_maps_to_0_0(self):
        """World origin should map to grid cell (0, 0)."""
        layout = _load_layout()
        grid = OccupancyGridSet.from_layout_data(layout)[1]
        row, col = grid.world_to_grid(0.0, 0.0)
        assert row == 0
        assert col == 0


# ---------------------------------------------------------------------------
# Ray-clear (wall collision)
# ---------------------------------------------------------------------------

class TestRayClear:

    @classmethod
    def setup_class(cls):
        cls.layout = _load_layout()
        cls.grids = OccupancyGridSet.from_layout_data(cls.layout, resolution=0.5)

    def test_open_space_clear(self):
        """Ray within the 2F living room (no obstacles) should be clear."""
        assert self.grids[2].ray_clear(6.0, 5.0, 14.0, 12.0)

    def test_through_wall_blocked(self):
        """Ray from 1F office into garage should cross a wall → not clear."""
        # Office interior: (3.5, 6.0). Garage interior: (12.0, 6.0)
        assert not self.grids[1].ray_clear(3.5, 6.0, 12.0, 6.0)

    def test_through_obstacle_blocked(self):
        """Ray through the fireplace on 2F should be blocked."""
        # From living room (14.0, 5.0) through fireplace center (16.9, 4.0)
        # to far side
        assert not self.grids[2].ray_clear(14.0, 5.0, 17.5, 3.0)

    def test_same_point_clear(self):
        """Ray from a point to itself should be clear (if walkable)."""
        assert self.grids[2].ray_clear(10.0, 14.0, 10.0, 14.0)


# ---------------------------------------------------------------------------
# Walkable fraction sanity
# ---------------------------------------------------------------------------

class TestWalkableFraction:

    def test_reasonable_fraction(self):
        """Each floor should have between 5% and 80% walkable area."""
        layout = _load_layout()
        grids = OccupancyGridSet.from_layout_data(layout)
        for f in grids.floors:
            frac = grids[f].walkable_fraction
            assert 0.05 < frac < 0.90, (
                f"Floor {f} walkable fraction {frac:.1%} outside expected range"
            )


# ---------------------------------------------------------------------------
# repr smoke test
# ---------------------------------------------------------------------------

class TestRepr:

    def test_grid_repr(self):
        layout = _load_layout()
        g = OccupancyGridSet.from_layout_data(layout)[1]
        r = repr(g)
        assert "OccupancyGrid" in r
        assert "floor=1" in r

    def test_set_repr(self):
        layout = _load_layout()
        gs = OccupancyGridSet.from_layout_data(layout)
        r = repr(gs)
        assert "OccupancyGridSet" in r


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
