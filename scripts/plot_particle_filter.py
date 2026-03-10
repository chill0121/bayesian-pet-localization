#!/usr/bin/env python3
"""Visualise particle filter convergence on the floor plan.

Simulates a stationary beacon on Floor 2, runs the particle filter for N
steps, and generates:
  - plots/particle_filter_convergence.png  — 4-panel showing cloud evolution
  - plots/particle_filter_trajectory.png   — simulated walking trajectory

Usage:
    python scripts/plot_particle_filter.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Allow importing from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from occupancy import OccupancyGridSet, bounds_to_polygon
from filters.particle import ParticleFilter, extract_stairways

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "floorplan", "layout.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all():
    with open(LAYOUT_PATH) as f:
        layout = json.load(f)

    grids = OccupancyGridSet.load_from_layout(LAYOUT_PATH, resolution=0.5)

    anchors = {}
    for fd in layout["floors"]:
        for a in fd.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {"x": pos[0], "y": pos[1], "floor": fd["floor"]}

    stairways = extract_stairways(layout)
    return layout, grids, anchors, stairways


def draw_floor_background(ax, grid, floor_data):
    """Draw occupancy grid and room outlines."""
    extent = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
    ax.imshow(
        grid.grid.astype(float),
        origin="lower", extent=extent,
        cmap="Greens", alpha=0.25, vmin=0, vmax=1, zorder=1,
    )
    for room in floor_data.get("rooms", []):
        poly = bounds_to_polygon(room["bounds"])
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, "-", color="#888", linewidth=0.7, zorder=2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")


def draw_anchors(ax, anchors, floor):
    for aid, a in anchors.items():
        if a["floor"] != floor:
            continue
        ax.plot(a["x"], a["y"], "^", color="blue", markersize=6, zorder=6)
        ax.annotate(aid, (a["x"], a["y"]), fontsize=5, color="blue",
                    ha="center", va="bottom", xytext=(0, 4),
                    textcoords="offset points", zorder=6)


def synthetic_rssi(beacon_x, beacon_y, beacon_floor, anchors, rng, noise=4.0):
    """Generate synthetic RSSI readings from a beacon position."""
    rssi = {}
    for aid, a in anchors.items():
        dist = math.sqrt((beacon_x - a["x"]) ** 2 + (beacon_y - a["y"]) ** 2)
        floor_diff = abs(beacon_floor - a["floor"])
        eff = dist + floor_diff * 30.0
        eff = max(eff, 1.0)
        dist_m = eff * 0.3048
        r = -59.0 - 10 * 2.7 * math.log10(dist_m) + rng.normal(0, noise)
        rssi[aid] = r
    return rssi


# ===========================================================================
# Plot 1: Convergence on a static beacon (4-panel snapshots)
# ===========================================================================

def plot_convergence(layout, grids, anchors, stairways):
    floor = 2
    floor_data = [fd for fd in layout["floors"] if fd["floor"] == floor][0]
    grid = grids[floor]
    beacon_x, beacon_y = 10.0, 10.0  # living room area

    pf = ParticleFilter(grids, anchors, stairways, n_particles=500, seed=7)
    pf.initialise_uniform(floor=floor)

    rng = np.random.default_rng(7)
    snapshots = []  # (step, particles, weights, estimate)
    snapshot_steps = [0, 5, 15, 39]

    for step_i in range(40):
        rssi = synthetic_rssi(beacon_x, beacon_y, floor, anchors, rng)
        pf.step(rssi, dt=0.5)
        if step_i in snapshot_steps:
            snapshots.append((
                step_i + 1,
                pf.particles.copy(),
                pf.weights.copy(),
                dict(pf.estimate),
            ))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Particle Filter Convergence — Static Beacon at (10, 10) Floor 2",
                 fontsize=13, fontweight="bold")

    for ax, (step_n, parts, weights, est) in zip(axes, snapshots):
        draw_floor_background(ax, grid, floor_data)
        draw_anchors(ax, anchors, floor)

        # Particles on this floor
        mask = parts[:, 2] == floor
        px, py = parts[mask, 0], parts[mask, 1]
        pw = weights[mask]
        pw_norm = pw / pw.max() if pw.max() > 0 else pw

        ax.scatter(px, py, s=3, c=pw_norm, cmap="hot_r", alpha=0.6, zorder=4)
        ax.plot(beacon_x, beacon_y, "*", color="lime", markersize=14,
                markeredgecolor="black", markeredgewidth=0.8, zorder=8)
        ax.plot(est["x"], est["y"], "x", color="red", markersize=10,
                markeredgewidth=2, zorder=8)
        ax.set_title(f"Step {step_n}  (n_eff={est['n_eff']:.0f}, conf={est['confidence']:.2f})",
                     fontsize=10)
        ax.set_xlabel("x (ft)")
        ax.set_ylabel("y (ft)")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "particle_filter_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ===========================================================================
# Plot 2: Walking trajectory tracking
# ===========================================================================

def plot_trajectory(layout, grids, anchors, stairways):
    floor = 2
    floor_data = [fd for fd in layout["floors"] if fd["floor"] == floor][0]
    grid = grids[floor]

    # Waypoints the beacon walks through on floor 2
    waypoints = [
        (4.0, 4.0),    # near living_sw
        (10.0, 10.0),  # centre living room
        (14.0, 20.0),  # kitchen area
        (10.0, 10.0),  # back to centre
    ]

    pf = ParticleFilter(grids, anchors, stairways, n_particles=400, seed=42)
    pf.initialise_around(waypoints[0][0], waypoints[0][1], floor=floor, spread=3.0)

    rng = np.random.default_rng(42)
    dt = 0.5
    speed_ft_per_step = 1.5  # ft per step

    true_trail = []
    est_trail = []
    bx, by = waypoints[0]
    wp_idx = 1

    for step_i in range(120):
        # Move beacon toward next waypoint
        tx, ty = waypoints[wp_idx]
        dx, dy = tx - bx, ty - by
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < speed_ft_per_step:
            bx, by = tx, ty
            wp_idx = (wp_idx + 1) % len(waypoints)
        else:
            bx += dx / dist * speed_ft_per_step
            by += dy / dist * speed_ft_per_step

        true_trail.append((bx, by))
        rssi = synthetic_rssi(bx, by, floor, anchors, rng)
        est = pf.step(rssi, dt=dt)
        est_trail.append((est["x"], est["y"]))

    true_trail = np.array(true_trail)
    est_trail = np.array(est_trail)

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_floor_background(ax, grid, floor_data)
    draw_anchors(ax, anchors, floor)

    ax.plot(true_trail[:, 0], true_trail[:, 1], "-", color="lime", linewidth=2,
            label="True path", zorder=6)
    ax.plot(est_trail[:, 0], est_trail[:, 1], "--", color="red", linewidth=1.5,
            label="PF estimate", zorder=6)
    ax.plot(true_trail[0, 0], true_trail[0, 1], "o", color="lime",
            markersize=10, markeredgecolor="black", zorder=7, label="Start")
    ax.plot(true_trail[-1, 0], true_trail[-1, 1], "s", color="lime",
            markersize=10, markeredgecolor="black", zorder=7, label="End")

    # Draw final particle cloud
    parts = pf.particles
    mask = parts[:, 2] == floor
    ax.scatter(parts[mask, 0], parts[mask, 1], s=2, c="red", alpha=0.3, zorder=5)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Particle Filter — Walking Trajectory Tracking (Floor 2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "particle_filter_trajectory.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ===========================================================================

def main():
    layout, grids, anchors, stairways = load_all()
    print(f"Loaded {len(anchors)} anchors, {len(stairways)} stairways")
    plot_convergence(layout, grids, anchors, stairways)
    plot_trajectory(layout, grids, anchors, stairways)
    print("Done.")


if __name__ == "__main__":
    main()
