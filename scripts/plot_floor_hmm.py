"""
Floor Transition HMM Visualisation

Produces a multi-panel figure showing:
  1. Transition matrix heatmap (with impossible-skip constraint)
  2. Emission likelihood example for each floor hypothesis
  3. Belief evolution over a simulated floor-change trajectory

Usage:
    python scripts/plot_floor_hmm.py          # saves to plots/
    python scripts/plot_floor_hmm.py --show   # also opens the window
"""

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Allow imports from services/inference/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "inference"))

from filters.floor_hmm import FloorTransitionHMM, _expected_rssi, EMISSION_SIGMA_DBM

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_layout_and_anchors():
    with open(LAYOUT_PATH) as f:
        data = json.load(f)
    anchors = {}
    for floor_data in data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {"x": pos[0], "y": pos[1], "floor": floor_num}
    return data, anchors


def plot_transition_matrix(ax, hmm):
    """Panel 1: base transition matrix heatmap."""
    T = hmm.transition_matrix
    floors = hmm.floors
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(floors)))
    ax.set_xticklabels([f"Floor {f}" for f in floors])
    ax.set_yticks(range(len(floors)))
    ax.set_yticklabels([f"Floor {f}" for f in floors])
    ax.set_xlabel("To floor")
    ax.set_ylabel("From floor")
    ax.set_title("Base Transition Matrix\n(1-s time step, no proximity boost)")

    # Annotate cells
    for i in range(len(floors)):
        for j in range(len(floors)):
            val = T[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="P(transition)")


def plot_emission_example(ax, anchors):
    """Panel 2: expected RSSI and emission likelihood curves per floor hypothesis."""
    # Simulate a pet on floor 2, 8 ft from floor-2 anchors
    distances = np.linspace(1, 50, 200)
    for floor_diff, label, color in [
        (0, "Same floor", "#2196F3"),
        (1, "1 floor away", "#FF9800"),
        (2, "2 floors away", "#F44336"),
    ]:
        expected = [_expected_rssi(d, floor_diff) for d in distances]
        ax.plot(distances, expected, label=label, color=color, linewidth=2)
        # ±1σ band
        upper = [e + EMISSION_SIGMA_DBM for e in expected]
        lower = [e - EMISSION_SIGMA_DBM for e in expected]
        ax.fill_between(distances, lower, upper, alpha=0.15, color=color)

    ax.set_xlabel("Distance (ft)")
    ax.set_ylabel("Expected RSSI (dBm)")
    ax.set_title("Emission Model: Expected RSSI vs Distance\n(shading = ±1σ)")
    ax.legend(loc="upper right")
    ax.set_ylim(-100, -30)
    ax.grid(True, alpha=0.3)


def simulate_floor_trajectory(hmm, anchors, n_steps=120):
    """Simulate a pet moving from floor 2 → stairway → floor 3 → back to floor 2.

    Returns (time, beliefs, true_floors, descriptions).
    """
    times = []
    beliefs = {f: [] for f in hmm.floors}
    true_floors = []
    descriptions = []

    dt = 1.0

    # Phase 1: on floor 2, living room (steps 0–39)
    for t in range(40):
        rssi = {
            "living_center": -52.0 + np.random.normal(0, 3),
            "kitchen_ne": -60.0 + np.random.normal(0, 3),
            "living_sw": -58.0 + np.random.normal(0, 3),
            "1F_Office": -82.0 + np.random.normal(0, 3),
        }
        proximity = {2: 15.0}  # far from stairs
        hmm.step(rssi, dt, proximity)
        times.append(t)
        true_floors.append(2)
        for f in hmm.floors:
            beliefs[f].append(hmm.floor_belief[f])
        descriptions.append("F2 living room")

    # Phase 2: approaching stairway on floor 2, moving toward floor 3 (steps 40–59)
    for t in range(40, 60):
        stair_dist = max(0.5, 15.0 - (t - 40) * 0.75)  # approaching
        rssi = {
            "staircase_mid": -50.0 + np.random.normal(0, 3),
            "living_center": -65.0 + np.random.normal(0, 3),
            "1F_Office": -80.0 + np.random.normal(0, 3),
        }
        proximity = {2: stair_dist}
        hmm.step(rssi, dt, proximity)
        times.append(t)
        true_floors.append(2)
        for f in hmm.floors:
            beliefs[f].append(hmm.floor_belief[f])
        descriptions.append(f"F2 → stairs (d={stair_dist:.1f}ft)")

    # Phase 3: on floor 3 (steps 60–99)
    for t in range(60, 100):
        # No floor-3 anchors, so signals from floor-2 anchors weaken
        rssi = {
            "living_center": -78.0 + np.random.normal(0, 4),
            "kitchen_ne": -82.0 + np.random.normal(0, 4),
            "staircase_mid": -70.0 + np.random.normal(0, 3),
            "1F_Office": -90.0 + np.random.normal(0, 4),
        }
        proximity = {3: 12.0}  # on floor 3, away from stairs
        hmm.step(rssi, dt, proximity)
        times.append(t)
        true_floors.append(3)
        for f in hmm.floors:
            beliefs[f].append(hmm.floor_belief[f])
        descriptions.append("F3 master bed")

    # Phase 4: back to floor 2 (steps 100–119)
    for t in range(100, 120):
        rssi = {
            "living_center": -53.0 + np.random.normal(0, 3),
            "kitchen_ne": -58.0 + np.random.normal(0, 3),
            "staircase_mid": -55.0 + np.random.normal(0, 3),
            "1F_Office": -80.0 + np.random.normal(0, 3),
        }
        proximity = {2: 5.0}
        hmm.step(rssi, dt, proximity)
        times.append(t)
        true_floors.append(2)
        for f in hmm.floors:
            beliefs[f].append(hmm.floor_belief[f])
        descriptions.append("F2 kitchen")

    return times, beliefs, true_floors


def plot_belief_evolution(ax, times, beliefs, true_floors, floors):
    """Panel 3: floor belief over time with ground truth."""
    colors = {1: "#4CAF50", 2: "#2196F3", 3: "#9C27B0"}
    for f in floors:
        ax.plot(times, beliefs[f], label=f"Floor {f}", color=colors[f],
                linewidth=2, alpha=0.9)

    # Ground truth shading
    true_arr = np.array(true_floors)
    for f in floors:
        mask = true_arr == f
        if not np.any(mask):
            continue
        # Find contiguous regions
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        for s, e in zip(starts, ends):
            ax.axvspan(times[s], times[min(e, len(times) - 1)],
                       alpha=0.08, color=colors[f])

    # Phase labels
    ax.axvline(40, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(60, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    ax.text(20, 1.05, "F2 Living", ha="center", fontsize=8, color="gray")
    ax.text(50, 1.05, "→ Stairs", ha="center", fontsize=8, color="gray")
    ax.text(80, 1.05, "F3 Master Bed", ha="center", fontsize=8, color="gray")
    ax.text(110, 1.05, "F2 Kitchen", ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Floor Probability")
    ax.set_title("Floor Belief Evolution — Simulated Floor Change Trajectory")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Floor Transition HMM visualisation")
    parser.add_argument("--show", action="store_true", help="Open plot window")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    data, anchors = load_layout_and_anchors()
    hmm_static = FloorTransitionHMM(data, anchors)
    hmm_sim = FloorTransitionHMM(data, anchors, initial_floor=2)

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    plot_transition_matrix(ax1, hmm_static)
    plot_emission_example(ax2, anchors)

    np.random.seed(42)
    times, beliefs, true_floors = simulate_floor_trajectory(hmm_sim, anchors)
    plot_belief_evolution(ax3, times, beliefs, true_floors, hmm_sim.floors)

    fig.suptitle("Floor Transition HMM — Bayesian Pet Localisation",
                 fontsize=14, fontweight="bold", y=0.98)

    out_path = os.path.join(PLOT_DIR, "floor_hmm_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
