#!/usr/bin/env python3
"""Plot floor plans from layout.json — the canonical visualization script.

Reads config/floorplan/layout.json and generates one PNG per floor showing:
  - Room polygons with fill and outlines
  - Vertex labels (PREFIX.A, PREFIX.B, ...) with coordinates
  - Door hinges, panels, and swing arcs
  - Open staircase entries
  - Obstacles, gates, stairs markers, compass rose

Usage:
    python scripts/plot_layout.py              # writes plots/*_floor_layout.png
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ── Color palette ────────────────────────────────────────────────────────────
ROOM_COLORS = {
    # Floor 1
    'hallway_1f': 'steelblue',
    'office_1f': '#e67e22',
    # Floor 2
    'living_kitchen': 'steelblue',
    'powder_room': '#e67e22',
    # Floor 3
    'hallway_3f': 'steelblue',
    'guest_bath': '#27ae60',
    'office_3f': '#e67e22',
    'master_bed': '#8e44ad',
    'master_bath': '#cc3333',
    # Shared
    'staircase': '#8888bb',
    'garage': '#888888',
}

OBSTACLE_STYLES = {
    'counter':  {'facecolor': '#e8e0d0', 'edgecolor': '#998866', 'hatch': '///'},
    'island':   {'facecolor': '#d4e8d0', 'edgecolor': '#558855', 'hatch': '///'},
    'barrier':  {'facecolor': '#e8d0c0', 'edgecolor': '#cc7744', 'hatch': 'xxx'},
    'closet':   {'facecolor': '#f0e0d0', 'edgecolor': '#996633', 'hatch': '...'},
    'fixture':  {'facecolor': '#cce5ff', 'edgecolor': '#3388cc', 'hatch': '...'},
}

DOOR_COLOR = '#008800'
STAIRS_COLOR = '#884400'


# ── Geometry helpers ─────────────────────────────────────────────────────────

def bounds_to_polygon(bounds):
    """Convert bounds (polygon list or {x1,y1,x2,y2} rect) to [[x,y], ...]."""
    if isinstance(bounds, list):
        poly = [list(p) for p in bounds]
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly = poly[:-1]
        return poly
    x1, y1, x2, y2 = bounds['x1'], bounds['y1'], bounds['x2'], bounds['y2']
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def vertex_label(prefix, index):
    """Return label like 'H.A', 'H.B', ... for a vertex index."""
    return f"{prefix}.{chr(65 + index)}"


# ── Drawing functions ────────────────────────────────────────────────────────

def draw_polygon(ax, poly, color, alpha=0.12, linewidth=2.0, zorder=3):
    """Draw a filled polygon with outline. Returns closed xs, ys."""
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.fill(xs, ys, alpha=alpha, color=color)
    ax.plot(xs, ys, '-', color=color, linewidth=linewidth, zorder=zorder)
    return xs, ys


def draw_room_label(ax, poly, label, color):
    """Draw centered room name."""
    cx = np.mean([p[0] for p in poly])
    cy = np.mean([p[1] for p in poly])
    ax.annotate(label, (cx, cy), fontsize=10, ha='center', va='center',
                color=color, fontweight='bold', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85))


def draw_vertex_labels(ax, poly, prefix, color):
    """Draw vertex dots and labels (PREFIX.A, PREFIX.B, ...) with coordinates."""
    cx = np.mean([p[0] for p in poly])
    cy = np.mean([p[1] for p in poly])
    for i, pt in enumerate(poly):
        ax.plot(pt[0], pt[1], 'o', color=color, markersize=4, zorder=5)
        # Push label outward from centroid
        dx = pt[0] - cx
        dy = pt[1] - cy
        norm = max((dx**2 + dy**2)**0.5, 0.01)
        ox = dx / norm * 0.55
        oy = dy / norm * 0.55
        lbl = vertex_label(prefix, i)
        text = f"{lbl}\n({pt[0]:.1f},{pt[1]:.1f})"
        ax.annotate(text, (pt[0], pt[1]),
                    xytext=(pt[0] + ox, pt[1] + oy),
                    fontsize=5.5, color=color, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor=color, alpha=0.8))


def draw_obstacle(ax, obs):
    """Draw an obstacle polygon with hatching and label."""
    bounds = obs.get('polygon', obs.get('bounds'))
    poly = bounds_to_polygon(bounds)
    style = OBSTACLE_STYLES.get(obs.get('type', ''),
                                 {'facecolor': '#e0e0e0', 'edgecolor': '#888888',
                                  'hatch': '///'})
    patch = patches.Polygon(poly,
                            facecolor=style['facecolor'],
                            edgecolor=style['edgecolor'],
                            linewidth=1.2, hatch=style['hatch'],
                            alpha=0.55, zorder=2)
    ax.add_patch(patch)
    cx = np.mean([p[0] for p in poly])
    cy = np.mean([p[1] for p in poly])
    ax.annotate(obs.get('name', ''), (cx, cy),
                fontsize=6.5, ha='center', va='center',
                color=style['edgecolor'], fontweight='bold', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=style['edgecolor'], alpha=0.85))


def draw_door(ax, door, color=DOOR_COLOR):
    """Draw a door with hinge dot, panel line, swing arc, and white wall gap."""
    cx, cy = door['center']
    width = door.get('width', 2.0)
    wall_axis = door.get('wall_axis', 'x')
    dtype = door.get('type', 'door')
    hinge = door.get('hinge')
    arc_deg = door.get('arc_degrees')

    if dtype == 'open':
        # Open doorway: thick translucent colored line
        if wall_axis == 'x':
            ax.plot([cx - width/2, cx + width/2], [cy, cy],
                    '-', color=STAIRS_COLOR, linewidth=3, alpha=0.5, zorder=4)
        else:
            ax.plot([cx, cx], [cy - width/2, cy + width/2],
                    '-', color=STAIRS_COLOR, linewidth=3, alpha=0.5, zorder=4)
    elif hinge and arc_deg:
        # Door with hinge + arc
        hx, hy = hinge
        # White gap on wall
        if wall_axis == 'x':
            ax.plot([cx - width/2, cx + width/2], [cy, cy],
                    '-', color='white', linewidth=4, zorder=4)
        else:
            ax.plot([cx, cx], [cy - width/2, cy + width/2],
                    '-', color='white', linewidth=4, zorder=4)

        # Panel end = opposite end of door from hinge along wall
        # end_pt = hinge + direction * width
        ex = 2 * cx - hx  # since center = midpoint of hinge and end
        ey = 2 * cy - hy

        # Hinge dot
        ax.plot(hx, hy, 'o', color=color, markersize=4, zorder=7)
        # Panel end dot (smaller)
        ax.plot(ex, ey, 'o', color=color, markersize=2.5, zorder=7)
        # Panel line
        ax.plot([hx, ex], [hy, ey], '-', color=color, linewidth=1.8, zorder=5)

        # Swing arc
        t1, t2 = arc_deg
        arc_patch = patches.Arc(
            (hx, hy), width * 2, width * 2,
            angle=0, theta1=t1, theta2=t2,
            color=color, linewidth=1.0, linestyle='--', zorder=5)
        ax.add_patch(arc_patch)
    else:
        # Fallback: simple marker
        if wall_axis == 'x':
            ax.plot([cx - width/2, cx + width/2], [cy, cy],
                    '-', color='white', linewidth=4, zorder=4)
        else:
            ax.plot([cx, cx], [cy - width/2, cy + width/2],
                    '-', color='white', linewidth=4, zorder=4)
        ax.plot(cx, cy, 's', color=color, markersize=5, zorder=6)

    # Door label
    to = door.get('to', '')
    text = f"\u2192{to}" if to else door.get('label', '')
    ox, oy = (0, 0.5) if wall_axis == 'x' else (0.5, 0)
    ax.annotate(text, (cx + ox, cy + oy), fontsize=5.5, ha='center', va='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#eeffee',
                          edgecolor=color, alpha=0.85))


def draw_gate(ax, gate):
    """Draw a kitchen/living gate line."""
    seg = gate['segment']
    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
            '--', color='#cc6600', linewidth=2.0, zorder=7)
    ax.plot(seg[0][0], seg[0][1], '<', color='#cc6600', markersize=6, zorder=8)
    ax.plot(seg[1][0], seg[1][1], '>', color='#cc6600', markersize=6, zorder=8)
    mid_x = (seg[0][0] + seg[1][0]) / 2
    mid_y = (seg[0][1] + seg[1][1]) / 2
    zones = gate.get('between', [])
    if len(zones) >= 2:
        ax.text(mid_x, mid_y + 3.0, zones[0].upper(),
                fontsize=9, ha='center', va='center', color='#006633',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#006633', alpha=0.85))
        ax.text(mid_x, mid_y - 3.0, zones[1].upper(),
                fontsize=9, ha='center', va='center', color='#cc6600',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#cc6600', alpha=0.85))


def draw_compass(ax):
    """Draw compass rose in upper right."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    arrow_len = min(x_range * 0.06, 0.7)
    text_off = arrow_len + 0.35
    cx = xlim[1] - x_range * 0.1
    cy = ylim[1] - y_range * 0.1
    ax.annotate('', xy=(cx, cy + arrow_len), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    ax.text(cx, cy + text_off, 'N', fontsize=10, fontweight='bold',
            color='darkgreen', ha='center', va='center')
    ax.annotate('', xy=(cx - arrow_len, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(cx - text_off, cy, 'W', fontsize=9, fontweight='bold',
            color='gray', ha='center', va='center')


# ── Main floor drawing ───────────────────────────────────────────────────────

def draw_floor(ax, floor_data, title, room_color_map=None, show_vertices=True):
    """Draw a complete floor from layout.json data.

    Args:
        show_vertices: If True, draw vertex labels with coordinates.
                       Set False for dashboard/tracker views.
    """
    all_xs, all_ys = [], []

    # Outer boundary
    ob = floor_data.get('outer_boundary', [])
    if ob:
        ob_pts = ob + [ob[0]]
        ax.plot([p[0] for p in ob_pts], [p[1] for p in ob_pts],
                '-', color='#222222', linewidth=1.8, zorder=8)
        all_xs.extend(p[0] for p in ob)
        all_ys.extend(p[1] for p in ob)

    # Rooms
    for room in floor_data.get('rooms', []):
        rname = room['name']
        color = (room_color_map or {}).get(rname, 'steelblue')
        walkable = room.get('walkable', True)
        prefix = room.get('vertex_prefix', rname[0].upper())
        poly = bounds_to_polygon(room['bounds'])

        if not walkable:
            patch = patches.Polygon(poly,
                                    facecolor='#e0e0e0', edgecolor='#888888',
                                    linewidth=1.2, hatch='///', alpha=0.55, zorder=2)
            ax.add_patch(patch)
            draw_room_label(ax, poly, room.get('label', rname), '#555555')
            if show_vertices:
                draw_vertex_labels(ax, poly, prefix, '#888888')
            all_xs.extend(p[0] for p in poly)
            all_ys.extend(p[1] for p in poly)
            continue

        draw_polygon(ax, poly, color)
        draw_room_label(ax, poly, room.get('label', rname), color)
        if show_vertices:
            draw_vertex_labels(ax, poly, prefix, color)
        all_xs.extend(p[0] for p in poly)
        all_ys.extend(p[1] for p in poly)

        # Obstacles
        for obs in room.get('obstacles', []):
            draw_obstacle(ax, obs)

        # Doorways
        for door in room.get('doorways', []):
            draw_door(ax, door)

        # Stairs openings
        for stairs_opening in room.get('stairs_openings', []):
            draw_door(ax, stairs_opening, color=STAIRS_COLOR)

        # Gates
        for gate in room.get('gates', []):
            draw_gate(ax, gate)

    # Stairs markers
    for stair in floor_data.get('stairs', []):
        entry = stair['entry']
        direction = stair.get('direction', '')
        to_floor = stair.get('to_floor', '')
        marker = '^' if direction == 'up' else 'v'
        ax.plot(entry[0], entry[1], marker, color=STAIRS_COLOR,
                markersize=8, zorder=9)
        arrow = '\u2191' if direction == 'up' else '\u2193'
        ax.annotate(f"{arrow}{to_floor}F", (entry[0], entry[1]),
                    xytext=(entry[0] - 0.8, entry[1]),
                    fontsize=6, fontweight='bold', color=STAIRS_COLOR,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='#fff0dd',
                              edgecolor=STAIRS_COLOR, alpha=0.9))

    # Anchors (BLE receivers)
    for anchor in floor_data.get('anchors', []):
        pos = anchor['position']
        aid = anchor.get('id', '')
        height = anchor.get('height_ft', 0)
        ax.plot(pos[0], pos[1], 'D', color='#cc0000', markersize=8,
                markeredgecolor='white', markeredgewidth=1.2, zorder=12)
        ax.annotate(f"{aid}\nh={height:.1f}ft",
                    (pos[0], pos[1]),
                    xytext=(pos[0] + 0.6, pos[1] + 0.6),
                    fontsize=6, fontweight='bold', color='#cc0000',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0',
                              edgecolor='#cc0000', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#cc0000',
                                   lw=1.0, shrinkA=0, shrinkB=3),
                    zorder=12)

    # Finalize axes
    pad = 1.5
    ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
    ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)
    draw_compass(ax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (ft)  \u2192  East')
    ax.set_ylabel('Y (ft)  \u2192  North')


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open('config/floorplan/layout.json') as f:
        layout = json.load(f)

    floors = {f['floor']: f for f in layout['floors']}
    version = layout.get('_schema_notes', {}).get('version', '?')

    floor_configs = [
        (1, "1st Floor", {
            'hallway': ROOM_COLORS['hallway_1f'],
            'office': ROOM_COLORS['office_1f'],
            'staircase': ROOM_COLORS['staircase'],
            'garage': ROOM_COLORS['garage'],
        }),
        (2, "2nd Floor", {
            'living_kitchen': ROOM_COLORS['living_kitchen'],
            'staircase': ROOM_COLORS['staircase'],
            'powder_room': ROOM_COLORS['powder_room'],
        }),
        (3, "3rd Floor", {
            'hallway': ROOM_COLORS['hallway_3f'],
            'guest_bath': ROOM_COLORS['guest_bath'],
            'office': ROOM_COLORS['office_3f'],
            'master_bed': ROOM_COLORS['master_bed'],
            'master_bath': ROOM_COLORS['master_bath'],
            'staircase': ROOM_COLORS['staircase'],
        }),
    ]

    ordinal = {1: '1st', 2: '2nd', 3: '3rd'}
    for floor_num, floor_name, color_map in floor_configs:
        fig, ax = plt.subplots(1, 1, figsize=(20, 24))
        title = (f"{floor_name} \u2014 layout.json v{version}\n"
                 f"(0,0) = SW corner  |  x \u2192 East, y \u2192 North")
        draw_floor(ax, floors[floor_num], title, room_color_map=color_map)
        plt.tight_layout()

        out = f"plots/{ordinal[floor_num]}_floor_layout.png"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    print(f"\n\u2713 All {len(floor_configs)} floor plots generated from layout.json v{version}")


if __name__ == '__main__':
    main()
