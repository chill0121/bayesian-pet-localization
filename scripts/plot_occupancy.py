#!/usr/bin/env python3
"""Plot occupancy grids from layout.json — verification overlay.

Generates one PNG per floor showing the binary walkable/blocked grid overlaid
with room outlines and anchor positions from layout.json.

Usage:
    python scripts/plot_occupancy.py              # writes plots/*_floor_occupancy.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from occupancy import OccupancyGridSet, bounds_to_polygon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "floorplan", "layout.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

FLOOR_NAMES = {1: "1st", 2: "2nd", 3: "3rd"}


def draw_room_outlines(ax, rooms, color="#333333", lw=1.2):
    """Draw room polygon outlines on top of the grid."""
    for room in rooms:
        poly = bounds_to_polygon(room["bounds"])
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        walkable = room.get("walkable", True)
        style = "--" if not walkable else "-"
        ax.plot(xs, ys, style, color=color, linewidth=lw, zorder=4)
        # Room label
        cx = np.mean([p[0] for p in poly])
        cy = np.mean([p[1] for p in poly])
        ax.text(cx, cy, room["name"], fontsize=7, ha="center", va="center",
                color=color, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))


def draw_obstacles(ax, rooms):
    """Draw obstacle outlines with hatching."""
    for room in rooms:
        for obs in room.get("obstacles", []):
            poly = bounds_to_polygon(obs.get("polygon", obs.get("bounds")))
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs, ys, facecolor="#ff8888", edgecolor="#cc0000",
                    alpha=0.3, hatch="///", zorder=3)


def draw_anchors(ax, anchors):
    """Draw anchor positions as markers."""
    for anch in anchors:
        pos = anch["position"]
        ax.plot(pos[0], pos[1], "^", color="#0066cc", markersize=10,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)
        ax.annotate(anch["id"], (pos[0], pos[1]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#0066cc", fontweight="bold", zorder=6)


def plot_floor(grid, floor_data, output_path):
    """Generate a single-floor occupancy overlay plot."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 16), dpi=120)

    # Show occupancy grid as an image (origin='lower' so y increases upward)
    # Green = walkable, Red = blocked
    rgb = np.zeros((*grid.grid.shape, 3))
    rgb[grid.grid] = [0.4, 0.85, 0.4]      # walkable cells  — green
    rgb[~grid.grid] = [0.92, 0.75, 0.75]    # blocked cells    — light red

    ax.imshow(rgb, origin="lower", extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
              aspect="equal", zorder=1, interpolation="nearest")

    # Overlay room outlines
    rooms = floor_data.get("rooms", [])
    draw_room_outlines(ax, rooms)
    draw_obstacles(ax, rooms)
    draw_anchors(ax, floor_data.get("anchors", []))

    # Grid lines at cell boundaries
    for c in range(grid.width_cells + 1):
        x = grid.x_min + c * grid.resolution
        ax.axvline(x, color="#cccccc", linewidth=0.2, zorder=2)
    for r in range(grid.height_cells + 1):
        y = grid.y_min + r * grid.resolution
        ax.axhline(y, color="#cccccc", linewidth=0.2, zorder=2)

    ax.set_xlabel("X (feet — East →)")
    ax.set_ylabel("Y (feet — North →)")
    ax.set_title(f"Floor {grid.floor} Occupancy Grid  "
                 f"({grid.width_cells}×{grid.height_cells} cells, "
                 f"{grid.resolution} ft/cell, "
                 f"{grid.walkable_fraction:.0%} walkable)")
    ax.set_xlim(grid.x_min - 0.5, grid.x_max + 0.5)
    ax.set_ylim(grid.y_min - 0.5, grid.y_max + 0.5)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  → {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(LAYOUT_PATH) as f:
        layout = json.load(f)

    grids = OccupancyGridSet.from_layout_data(layout)
    print(f"Loaded grids: {grids}")

    for floor_data in layout["floors"]:
        floor_num = floor_data["floor"]
        grid = grids[floor_num]
        name = FLOOR_NAMES.get(floor_num, str(floor_num))
        output_path = os.path.join(OUTPUT_DIR, f"{name}_floor_occupancy.png")
        plot_floor(grid, floor_data, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
