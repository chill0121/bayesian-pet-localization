"""
Occupancy Grid Generator

Parses layout.json room polygons into per-floor walkable/blocked bitmaps.
The particle filter uses these grids for wall constraints and motion validation.

Grid convention:
    - True  = walkable (room interior, doorway)
    - False = blocked  (wall, obstacle, non-walkable room, outside building)
    - Coordinates: grid[row][col] where row = y-axis, col = x-axis
    - Origin (0,0) in world coords maps to grid[0][0] (bottom-left)

Usage:
    grids = OccupancyGridSet.load_from_layout("config/floorplan/layout.json")
    grids[2].is_walkable(10.0, 14.0)          # True (living room)
    grids[2].ray_clear(3.0, 5.0, 15.0, 25.0)  # check wall crossing
"""

import json
import math

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bounds_to_polygon(bounds):
    """Convert bounds (polygon list or {x1,y1,x2,y2} rect) to [[x,y], ...].

    Handles both formats found in layout.json:
      - List of [x,y] vertices (with optional duplicate closing vertex)
      - Dict with x1, y1, x2, y2 (SW and NE corners)
    """
    if isinstance(bounds, list):
        poly = [list(p) for p in bounds]
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly = poly[:-1]
        return poly
    x1, y1, x2, y2 = bounds['x1'], bounds['y1'], bounds['x2'], bounds['y2']
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test.

    Returns True if (px, py) is inside the polygon defined by a list of
    [x, y] vertices.  Uses the standard even-odd ray-casting rule.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _rasterize_polygon(polygon, grid, resolution, value):
    """Fill grid cells whose centers fall inside *polygon* with *value*.

    Parameters
    ----------
    polygon : list of [x, y]
        Vertices defining the polygon.
    grid : np.ndarray (bool, shape H×W)
        The occupancy grid to modify in-place.
    resolution : float
        Feet per grid cell.
    value : bool
        Value to write into cells inside the polygon.
    """
    rows, cols = grid.shape

    # Compute bounding box in grid coords to avoid scanning the full grid
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    col_min = max(0, int(math.floor(min(xs) / resolution)))
    col_max = min(cols - 1, int(math.floor(max(xs) / resolution)))
    row_min = max(0, int(math.floor(min(ys) / resolution)))
    row_max = min(rows - 1, int(math.floor(max(ys) / resolution)))

    for r in range(row_min, row_max + 1):
        cy = (r + 0.5) * resolution  # cell center y
        for c in range(col_min, col_max + 1):
            cx = (c + 0.5) * resolution  # cell center x
            if _point_in_polygon(cx, cy, polygon):
                grid[r, c] = value


def _carve_doorway(center, width, wall_axis, grid, resolution):
    """Force cells around a doorway to be walkable.

    Carves a rectangular strip perpendicular to the wall so that adjacent
    rooms remain connected even if their polygon boundaries overlap on the
    shared wall segment.

    Parameters
    ----------
    center : list [x, y]
        Midpoint of the doorway.
    width : float
        Width of the doorway opening (feet).
    wall_axis : str
        'x' if the wall runs along the x-axis (horizontal wall, doorway
        opens in the y-direction), 'y' if along the y-axis.
    grid : np.ndarray (bool)
    resolution : float
    """
    cx, cy = center
    half_w = width / 2.0
    # Depth of the carved passage — enough to punch through the wall
    depth = resolution * 2.0

    if wall_axis == 'x':
        # Wall is horizontal: doorway spans x, passage punches through y
        x_lo, x_hi = cx - half_w, cx + half_w
        y_lo, y_hi = cy - depth, cy + depth
    else:
        # Wall is vertical: doorway spans y, passage punches through x
        x_lo, x_hi = cx - depth, cx + depth
        y_lo, y_hi = cy - half_w, cy + half_w

    rows, cols = grid.shape
    col_min = max(0, int(math.floor(x_lo / resolution)))
    col_max = min(cols - 1, int(math.floor(x_hi / resolution)))
    row_min = max(0, int(math.floor(y_lo / resolution)))
    row_max = min(rows - 1, int(math.floor(y_hi / resolution)))

    for r in range(row_min, row_max + 1):
        cy = (r + 0.5) * resolution
        for c in range(col_min, col_max + 1):
            cx = (c + 0.5) * resolution
            # Only carve if cell center falls within the doorway rectangle
            if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
                grid[r, c] = True


# ---------------------------------------------------------------------------
# OccupancyGrid — single floor
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """2-D boolean occupancy grid for a single floor.

    Parameters
    ----------
    floor : int
        Floor number (1, 2, 3, …).
    floor_data : dict
        The floor entry from layout.json (one element of ``floors`` list).
    resolution : float
        Grid cell size in feet.  Default 0.5 ft (≈ 6 in).
    """

    def __init__(self, floor: int, floor_data: dict, resolution: float = 0.5):
        self.floor = floor
        self.resolution = resolution

        # Determine grid dimensions from the outer_boundary
        boundary = floor_data["outer_boundary"]
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

        self.width_ft = self.x_max - self.x_min
        self.height_ft = self.y_max - self.y_min
        self.width_cells = int(math.ceil(self.width_ft / resolution))
        self.height_cells = int(math.ceil(self.height_ft / resolution))

        # Start with everything blocked
        self.grid = np.zeros((self.height_cells, self.width_cells), dtype=bool)

        self._build(floor_data)

    # ----- construction steps ------------------------------------------------

    def _build(self, floor_data: dict):
        """Rasterize rooms, enforce walls, subtract obstacles, carve doorways."""
        rooms = floor_data.get("rooms", [])

        # 1. Mark walkable room interiors
        for room in rooms:
            if not room.get("walkable", True):
                continue
            poly = bounds_to_polygon(room["bounds"])
            _rasterize_polygon(poly, self.grid, self.resolution, True)

        # 2. Subtract obstacles (from all rooms, including walkable ones)
        for room in rooms:
            for obs in room.get("obstacles", []):
                obs_poly = bounds_to_polygon(obs.get("polygon", obs.get("bounds")))
                _rasterize_polygon(obs_poly, self.grid, self.resolution, False)

        # 3. Enforce walls between rooms — block cells at shared room boundaries
        #    so that rooms with zero-thickness shared walls get a cell-wide wall.
        self._enforce_walls(rooms)

        # 4. Carve doorways AFTER wall enforcement to restore connectivity
        for room in rooms:
            for dw in room.get("doorways", []):
                if dw.get('to') == 'exterior':
                    continue  # don't carve beyond building boundary
                _carve_doorway(
                    dw["center"], dw["width"], dw["wall_axis"],
                    self.grid, self.resolution,
                )
            # Also handle stairs_opening (same schema as a doorway)
            so = room.get("stairs_opening")
            if so:
                _carve_doorway(
                    so["center"], so["width"], so["wall_axis"],
                    self.grid, self.resolution,
                )

    def _enforce_walls(self, rooms):
        """Block cells at boundaries between different rooms.

        The layout.json models rooms with shared walls (zero thickness).
        At grid resolution, adjacent room polygons produce contiguous walkable
        cells with no gap.  This method identifies cells at room boundaries
        and blocks them, creating a 1-cell wall between every pair of rooms.
        Doorway carving (run afterwards) punches holes back through at door
        and stair-opening locations.
        """
        walkable_rooms = [room for room in rooms if room.get("walkable", True)]
        if len(walkable_rooms) < 2:
            return

        polys = [bounds_to_polygon(r["bounds"]) for r in walkable_rooms]

        # Build room membership grid: for each walkable cell, which room
        # polygon contains its center?  (-1 = none / obstacle / outside)
        room_id = np.full(self.grid.shape, -1, dtype=np.int8)
        for r in range(self.height_cells):
            for c in range(self.width_cells):
                if not self.grid[r, c]:
                    continue
                x, y = self.grid_to_world(r, c)
                for idx, poly in enumerate(polys):
                    if _point_in_polygon(x, y, poly):
                        room_id[r, c] = idx
                        break

        # Find boundary cells: adjacent walkable cells belonging to different rooms
        to_block = set()
        for r in range(self.height_cells):
            for c in range(self.width_cells):
                rid = room_id[r, c]
                if rid < 0:
                    continue
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if nr >= self.height_cells or nc >= self.width_cells:
                        continue
                    nrid = room_id[nr, nc]
                    if nrid < 0 or rid == nrid:
                        continue
                    # Two different rooms touch — block both boundary cells
                    to_block.add((r, c))
                    to_block.add((nr, nc))

        for r, c in to_block:
            self.grid[r, c] = False

    # ----- coordinate conversion ---------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates (feet) to grid indices (row, col).

        Returns the row and column of the cell containing (x, y).
        Clamps to grid bounds.
        """
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        col = max(0, min(col, self.width_cells - 1))
        row = max(0, min(row, self.height_cells - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y

    # ----- queries -----------------------------------------------------------

    def is_walkable(self, x: float, y: float) -> bool:
        """Return True if the world-coordinate point is in walkable space."""
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        row, col = self.world_to_grid(x, y)
        return bool(self.grid[row, col])

    def is_walkable_grid(self, row: int, col: int) -> bool:
        """Return True if the grid cell is walkable (bounds-safe)."""
        if 0 <= row < self.height_cells and 0 <= col < self.width_cells:
            return bool(self.grid[row, col])
        return False

    def ray_clear(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Return True if all cells on the line from (x1,y1) to (x2,y2) are walkable.

        Uses Bresenham's line algorithm in grid space.  Intended for the
        particle filter to reject moves that cross walls.
        """
        r1, c1 = self.world_to_grid(x1, y1)
        r2, c2 = self.world_to_grid(x2, y2)

        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r2 > r1 else -1
        sc = 1 if c2 > c1 else -1
        err = dc - dr

        r, c = r1, c1
        while True:
            if not self.is_walkable_grid(r, c):
                return False
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
        return True

    @property
    def walkable_fraction(self) -> float:
        """Fraction of grid cells that are walkable (useful for sanity checks)."""
        return float(self.grid.sum()) / self.grid.size

    def __repr__(self):
        return (
            f"OccupancyGrid(floor={self.floor}, "
            f"{self.width_cells}×{self.height_cells} cells, "
            f"res={self.resolution} ft, "
            f"walkable={self.walkable_fraction:.1%})"
        )


# ---------------------------------------------------------------------------
# OccupancyGridSet — all floors
# ---------------------------------------------------------------------------

class OccupancyGridSet:
    """Container for occupancy grids across all floors.

    Parameters
    ----------
    grids : dict[int, OccupancyGrid]
        Mapping of floor number to OccupancyGrid.
    """

    def __init__(self, grids: dict[int, "OccupancyGrid"]):
        self._grids = grids

    @classmethod
    def load_from_layout(cls, layout_path: str, resolution: float = 0.5) -> "OccupancyGridSet":
        """Load layout.json and build occupancy grids for every floor.

        Parameters
        ----------
        layout_path : str
            Path to layout.json.
        resolution : float
            Grid cell size in feet.
        """
        with open(layout_path) as f:
            data = json.load(f)

        grids = {}
        for floor_data in data.get("floors", []):
            floor_num = floor_data["floor"]
            grids[floor_num] = OccupancyGrid(floor_num, floor_data, resolution)
        return cls(grids)

    @classmethod
    def from_layout_data(cls, layout_data: dict, resolution: float = 0.5) -> "OccupancyGridSet":
        """Build grids from already-loaded layout dict (avoids re-reading file)."""
        grids = {}
        for floor_data in layout_data.get("floors", []):
            floor_num = floor_data["floor"]
            grids[floor_num] = OccupancyGrid(floor_num, floor_data, resolution)
        return cls(grids)

    # ----- access ------------------------------------------------------------

    def __getitem__(self, floor: int) -> OccupancyGrid:
        return self._grids[floor]

    def __contains__(self, floor: int) -> bool:
        return floor in self._grids

    def __iter__(self):
        return iter(sorted(self._grids.keys()))

    @property
    def floors(self) -> list[int]:
        """Sorted list of floor numbers."""
        return sorted(self._grids.keys())

    def is_walkable(self, floor: int, x: float, y: float) -> bool:
        """Check walkability across floors."""
        if floor not in self._grids:
            return False
        return self._grids[floor].is_walkable(x, y)

    def __repr__(self):
        parts = [repr(self._grids[f]) for f in self.floors]
        return f"OccupancyGridSet({', '.join(parts)})"
