#!/usr/bin/env python3
"""Demo: Active Floor + Timeline tracker visualization.

Generates a static demo image showing what the live dashboard would look like.
Left panel:  Active floor with position estimate, particle cloud, and trail.
Right panel: Timeline strip with floor occupancy, room confidence, and events.

Uses simulated data — no MQTT/inference needed.

Usage:
    python scripts/demo_tracker.py          # writes plots/demo_tracker.png
"""

import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
from datetime import datetime, timedelta


# ── Import floor rendering from plot_layout ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from plot_layout import (
    ROOM_COLORS, draw_floor, bounds_to_polygon,
)

# ── Simulated trajectory ─────────────────────────────────────────────────────
# A plausible 60-minute dog trajectory through the house

TRAJECTORY_WAYPOINTS = [
    # (floor, x, y, room_name, dwell_minutes)
    (2, 10.0, 12.0,  'living_kitchen',  8),   # couch
    (2,  8.0, 18.0,  'living_kitchen',  2),   # walking to kitchen
    (2, 14.0, 26.0,  'living_kitchen',  3),   # water bowl
    (2, 12.0, 22.0,  'living_kitchen',  5),   # kitchen area
    (2,  3.0, 15.5,  'staircase',       1),   # heading downstairs
    (1,  1.5, 15.5,  'staircase',       1),   # 1F landing
    (1,  5.2, 17.0,  'hallway',         2),   # hallway
    (1,  4.0,  7.0,  'office',         12),   # office nap
    (1,  5.2, 17.0,  'hallway',         1),   # back to hallway
    (1,  1.5, 15.5,  'staircase',       1),   # heading up
    (2,  3.0, 15.5,  'staircase',       1),   # 2F staircase
    (2, 10.0, 12.0,  'living_kitchen',  3),   # couch again
    (2,  3.0, 15.5,  'staircase',       1),   # heading to 3F
    (3,  2.0, 15.0,  'staircase',       1),   # 3F landing
    (3,  5.1, 16.0,  'hallway',         1),   # 3F hallway
    (3, 12.0,  8.0,  'master_bed',     10),   # master bed nap
    (3,  5.1, 16.0,  'hallway',         1),   # back to hallway
    (3,  5.1, 22.0,  'hallway',         1),   # toward wife office
    (3, 12.0, 24.0,  'wife_office',     5),   # wife office hang
]


def interpolate_trajectory(waypoints, samples_per_minute=2):
    """Generate a smooth trajectory from waypoints with timestamps."""
    trajectory = []
    t = datetime(2026, 2, 26, 10, 0, 0)
    dt = timedelta(seconds=60 / samples_per_minute)

    for i in range(len(waypoints)):
        floor, x, y, room, dwell = waypoints[i]
        n_samples = max(1, int(dwell * samples_per_minute))

        if i + 1 < len(waypoints):
            nf, nx, ny, nr, _ = waypoints[i + 1]
            # If same floor, interpolate position; otherwise snap
            if floor == nf:
                for j in range(n_samples):
                    frac = j / n_samples
                    px = x + (nx - x) * frac * 0.3  # only drift partially during dwell
                    py = y + (ny - y) * frac * 0.3
                    # Add jitter
                    px += random.gauss(0, 0.3)
                    py += random.gauss(0, 0.3)
                    trajectory.append({
                        'time': t, 'floor': floor,
                        'x': px, 'y': py, 'room': room,
                        'confidence': min(0.95, 0.6 + random.gauss(0.2, 0.08)),
                    })
                    t += dt
            else:
                # Floor transition — dwell on current floor then switch
                for j in range(n_samples):
                    px = x + random.gauss(0, 0.3)
                    py = y + random.gauss(0, 0.3)
                    conf = 0.4 + random.gauss(0, 0.1)  # lower confidence during transition
                    trajectory.append({
                        'time': t, 'floor': floor,
                        'x': px, 'y': py, 'room': room,
                        'confidence': max(0.2, min(0.95, conf)),
                    })
                    t += dt
        else:
            # Last waypoint
            for j in range(n_samples):
                px = x + random.gauss(0, 0.3)
                py = y + random.gauss(0, 0.3)
                trajectory.append({
                    'time': t, 'floor': floor,
                    'x': px, 'y': py, 'room': room,
                    'confidence': min(0.95, 0.7 + random.gauss(0.15, 0.05)),
                })
                t += dt

    return trajectory


def generate_particles(center_x, center_y, confidence, n=60):
    """Generate a particle cloud around the estimated position."""
    spread = 2.5 * (1.0 - confidence) + 0.3
    xs = [center_x + random.gauss(0, spread) for _ in range(n)]
    ys = [center_y + random.gauss(0, spread) for _ in range(n)]
    return xs, ys


# ── Color maps ───────────────────────────────────────────────────────────────

FLOOR_COLORS = {1: '#4a90d9', 2: '#2ecc71', 3: '#e67e22'}
FLOOR_LABELS = {1: '1F', 2: '2F', 3: '3F'}

ROOM_LABEL_MAP = {
    'living_kitchen': 'Living/Kitchen',
    'staircase': 'Staircase',
    'powder_room': 'Powder Room',
    'hallway': 'Hallway',
    'office': 'Office',
    'garage': 'Garage',
    'guest_bath': 'Guest Bath',
    'wife_office': "Wife's Office",
    'master_bed': 'Master Bed',
    'master_bath': 'Master Bath',
}

FLOOR_ROOM_COLORS = {
    'living_kitchen': ROOM_COLORS['living_kitchen'],
    'staircase': ROOM_COLORS['staircase'],
    'powder_room': ROOM_COLORS['powder_room'],
    'hallway': ROOM_COLORS['hallway_1f'],
    'office': ROOM_COLORS['office'],
    'garage': ROOM_COLORS['garage'],
    'guest_bath': ROOM_COLORS['guest_bath'],
    'wife_office': ROOM_COLORS['wife_office'],
    'master_bed': ROOM_COLORS['master_bed'],
    'master_bath': ROOM_COLORS['master_bath'],
}


# ── Drawing functions ────────────────────────────────────────────────────────

def draw_position_overlay(ax, trajectory, current_idx, trail_len=20):
    """Draw position estimate, particle cloud, and recent trail on active floor."""
    cur = trajectory[current_idx]
    cx, cy = cur['x'], cur['y']
    conf = cur['confidence']

    # Trail (fading)
    trail_start = max(0, current_idx - trail_len)
    trail_pts = [trajectory[i] for i in range(trail_start, current_idx + 1)
                 if trajectory[i]['floor'] == cur['floor']]

    if len(trail_pts) > 1:
        xs = [p['x'] for p in trail_pts]
        ys = [p['y'] for p in trail_pts]
        n = len(xs)
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        alphas = np.linspace(0.05, 0.7, n - 1)
        colors = [(0.9, 0.2, 0.2, a) for a in alphas]
        lc = LineCollection(segments, colors=colors, linewidths=2.0, zorder=10)
        ax.add_collection(lc)

    # Particle cloud
    px, py = generate_particles(cx, cy, conf, n=80)
    ax.scatter(px, py, s=8, c='red', alpha=0.15, zorder=9, edgecolors='none')

    # Confidence circle
    radius = 2.5 * (1.0 - conf) + 0.4
    circle = patches.Circle((cx, cy), radius,
                             fill=True, facecolor='red', alpha=0.08,
                             edgecolor='red', linewidth=1.0, linestyle='--',
                             zorder=10)
    ax.add_patch(circle)

    # Position dot
    ax.plot(cx, cy, 'o', color='red', markersize=10, zorder=12,
            markeredgecolor='darkred', markeredgewidth=1.5)
    # Pet label
    ax.annotate('DOG', (cx, cy), xytext=(cx + 0.8, cy + 0.8),
                fontsize=8, fontweight='bold', color='darkred', zorder=13,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffcccc',
                          edgecolor='darkred', alpha=0.9))

    # Confidence text
    ax.annotate(f'{conf:.0%}', (cx, cy),
                xytext=(cx - 0.3, cy - 1.0),
                fontsize=8, color='red', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='red', alpha=0.85),
                zorder=13)


def draw_mini_floors(fig, gs_mini, layout, active_floor, trajectory, current_idx):
    """Draw small inactive floor thumbnails."""
    inactive_floors = [f for f in [1, 2, 3] if f != active_floor]
    floors_data = {f['floor']: f for f in layout['floors']}

    floor_color_maps = {
        1: {'hallway': ROOM_COLORS['hallway_1f'], 'office': ROOM_COLORS['office'],
            'staircase': ROOM_COLORS['staircase'], 'garage': ROOM_COLORS['garage']},
        2: {'living_kitchen': ROOM_COLORS['living_kitchen'],
            'staircase': ROOM_COLORS['staircase'],
            'powder_room': ROOM_COLORS['powder_room']},
        3: {'hallway': ROOM_COLORS['hallway_3f'], 'guest_bath': ROOM_COLORS['guest_bath'],
            'wife_office': ROOM_COLORS['wife_office'], 'master_bed': ROOM_COLORS['master_bed'],
            'master_bath': ROOM_COLORS['master_bath'], 'staircase': ROOM_COLORS['staircase']},
    }

    for i, fl in enumerate(inactive_floors):
        ax = fig.add_subplot(gs_mini[i])
        draw_floor(ax, floors_data[fl], f"Floor {fl}", room_color_map=floor_color_maps[fl],
                   show_vertices=False)
        # Dim inactive floors
        ax.patch.set_alpha(0.4)
        for child in ax.get_children():
            if hasattr(child, 'set_alpha') and not isinstance(child, matplotlib.text.Text):
                try:
                    current_alpha = child.get_alpha()
                    if current_alpha is None:
                        current_alpha = 1.0
                    child.set_alpha(current_alpha * 0.4)
                except:
                    pass
        ax.set_title(f"Floor {fl}", fontsize=9, color='gray')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelsize=6)


def draw_timeline(ax, trajectory, current_idx):
    """Draw the timeline strip: floor bands + room confidence + events."""
    times = [t['time'] for t in trajectory]
    t_start, t_end = times[0], times[-1]
    t_range_min = (t_end - t_start).total_seconds() / 60.0

    # ── Floor occupancy band ──
    # Draw colored rectangles for each floor segment
    prev_floor = trajectory[0]['floor']
    seg_start = 0
    for i in range(1, len(trajectory)):
        if trajectory[i]['floor'] != prev_floor or i == len(trajectory) - 1:
            seg_end = i
            t0 = (times[seg_start] - t_start).total_seconds() / 60.0
            t1 = (times[seg_end] - t_start).total_seconds() / 60.0
            color = FLOOR_COLORS.get(prev_floor, 'gray')
            ax.barh(2.5, t1 - t0, left=t0, height=0.8, color=color, alpha=0.7,
                    edgecolor='white', linewidth=0.5)
            # Label if segment is wide enough
            if t1 - t0 > 3:
                ax.text((t0 + t1) / 2, 2.5, FLOOR_LABELS[prev_floor],
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white')
            prev_floor = trajectory[i]['floor']
            seg_start = i

    # ── Room confidence heatmap ──
    # Collect unique rooms in trajectory order
    all_rooms = []
    seen = set()
    for t in trajectory:
        if t['room'] not in seen:
            all_rooms.append(t['room'])
            seen.add(t['room'])

    n_rooms = len(all_rooms)
    n_times = len(trajectory)
    heatmap = np.zeros((n_rooms, n_times))

    for j, t in enumerate(trajectory):
        for ri, room in enumerate(all_rooms):
            if t['room'] == room:
                heatmap[ri, j] = t['confidence']
            else:
                # Small background probability
                heatmap[ri, j] = random.uniform(0.02, 0.08)

    time_mins = [(t['time'] - t_start).total_seconds() / 60.0 for t in trajectory]
    extent = [time_mins[0], time_mins[-1], -0.5, n_rooms - 0.5]
    ax.imshow(heatmap, aspect='auto', extent=extent,
              cmap='YlOrRd', vmin=0, vmax=1.0,
              origin='lower', interpolation='nearest',
              alpha=0.85, zorder=1)

    # Room labels on y-axis
    ax.set_yticks(list(range(n_rooms)) + [2.5])
    ax.set_yticklabels(
        [ROOM_LABEL_MAP.get(r, r) for r in all_rooms] + ['Floor'],
        fontsize=7)

    # ── Transition event markers ──
    for i in range(1, len(trajectory)):
        if trajectory[i]['floor'] != trajectory[i-1]['floor']:
            t_min = (times[i] - t_start).total_seconds() / 60.0
            ax.axvline(t_min, color='red', linewidth=1.0, linestyle='--', alpha=0.6, zorder=5)
            direction = '↑' if trajectory[i]['floor'] > trajectory[i-1]['floor'] else '↓'
            ax.annotate(f'{direction}{trajectory[i]["floor"]}F',
                        (t_min, n_rooms + 0.2),
                        fontsize=6, color='red', fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='#ffeeee',
                                  edgecolor='red', alpha=0.8))

    # ── Current time marker ──
    cur_t = (times[current_idx] - t_start).total_seconds() / 60.0
    ax.axvline(cur_t, color='black', linewidth=2.0, linestyle='-', zorder=6)
    ax.annotate('NOW', (cur_t, -1.2), fontsize=7, fontweight='bold',
                ha='center', va='top', color='black')

    # Formatting
    ax.set_xlim(time_mins[0] - 0.5, time_mins[-1] + 0.5)
    ax.set_ylim(-0.5, 3.2)
    ax.set_xlabel('Minutes since start', fontsize=9)
    ax.set_title('Position Timeline — Floor Occupancy + Room Confidence',
                 fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Floor band legend
    for fl, color in FLOOR_COLORS.items():
        ax.plot([], [], 's', color=color, markersize=8,
                label=f'Floor {fl}')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    random.seed(42)

    with open('config/floorplan/layout.json') as f:
        layout = json.load(f)

    floors_data = {f['floor']: f for f in layout['floors']}

    # Generate simulated trajectory
    trajectory = interpolate_trajectory(TRAJECTORY_WAYPOINTS, samples_per_minute=2)

    # Pick a "current" moment — find the dog in master_bed
    current_idx = len(trajectory) - 1
    for i, t in enumerate(trajectory):
        if t['room'] == 'master_bed' and t['confidence'] > 0.7:
            current_idx = i + 8  # a few samples into the stay
            break
    current_idx = min(current_idx, len(trajectory) - 1)
    cur = trajectory[current_idx]
    active_floor = cur['floor']

    print(f"Simulated trajectory: {len(trajectory)} points over "
          f"{(trajectory[-1]['time'] - trajectory[0]['time']).total_seconds()/60:.0f} min")
    print(f"Current: Floor {active_floor}, room={cur['room']}, "
          f"pos=({cur['x']:.1f}, {cur['y']:.1f}), conf={cur['confidence']:.0%}")

    # ── Build figure layout ──────────────────────────────────────
    #
    #  ┌──────────────────┬────────────┐
    #  │                  │  Floor N   │
    #  │   Active Floor   ├────────────┤
    #  │   (large)        │  Floor N   │
    #  │                  │  (dimmed)  │
    #  ├──────────────────┴────────────┤
    #  │         Timeline Strip        │
    #  └───────────────────────────────┘

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1],
                           hspace=0.25, wspace=0.15)

    # Active floor (large panel)
    ax_main = fig.add_subplot(gs[0, 0])

    floor_color_maps = {
        1: {'hallway': ROOM_COLORS['hallway_1f'], 'office': ROOM_COLORS['office'],
            'staircase': ROOM_COLORS['staircase'], 'garage': ROOM_COLORS['garage']},
        2: {'living_kitchen': ROOM_COLORS['living_kitchen'],
            'staircase': ROOM_COLORS['staircase'],
            'powder_room': ROOM_COLORS['powder_room']},
        3: {'hallway': ROOM_COLORS['hallway_3f'], 'guest_bath': ROOM_COLORS['guest_bath'],
            'wife_office': ROOM_COLORS['wife_office'], 'master_bed': ROOM_COLORS['master_bed'],
            'master_bath': ROOM_COLORS['master_bath'], 'staircase': ROOM_COLORS['staircase']},
    }

    title = f"Floor {active_floor} — {cur['room'].replace('_', ' ').title()} — {cur['confidence']:.0%} confidence"
    draw_floor(ax_main, floors_data[active_floor], title,
               room_color_map=floor_color_maps[active_floor], show_vertices=False)
    draw_position_overlay(ax_main, trajectory, current_idx, trail_len=30)

    # Inactive floor thumbnails (right column)
    gs_mini = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1],
                                                hspace=0.3)
    draw_mini_floors(fig, gs_mini, layout, active_floor, trajectory, current_idx)

    # Timeline strip (bottom)
    ax_timeline = fig.add_subplot(gs[1, :])
    draw_timeline(ax_timeline, trajectory, current_idx)

    # Super title
    fig.suptitle('Bayesian Pet Localization — Tracker Demo',
                 fontsize=16, fontweight='bold', y=0.98)

    out = 'plots/demo_tracker.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == '__main__':
    main()
