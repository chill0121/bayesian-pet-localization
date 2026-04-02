"""
Dashboard for Bayesian Pet Localization

Real-time indoor positioning display with particle cloud visualization,
floor belief tracking, and historical timeline.

Framework: Dash + Plotly (callback-based, no full-page reruns)
Theme: Dark (DARKLY bootstrap theme)
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import requests
from dash import Input, Output, Patch, State, clientside_callback, dcc, html
from dash.exceptions import PreventUpdate

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
FLOORPLAN_LOCAL = os.getenv(
    "FLOORPLAN_PATH",
    os.path.join(os.path.dirname(__file__), "floorplan", "layout.json"),
)
DEFAULT_REFRESH_MS = 1000

# -----------------------------------------------------------------------------
# Theme & Colors
# -----------------------------------------------------------------------------

PLOT_BG = "#16213e"
CARD_BG = "#1a1a2e"
GRID_COLOR = "rgba(255,255,255,0.05)"
TEXT_COLOR = "#e0e0e0"
MUTED_TEXT = "#888888"
ACCENT = "#4fc3f7"
DANGER = "#ef5350"
SUCCESS = "#66bb6a"
WARNING = "#ffa726"

FLOOR_COLORS = {1: "#4a90d9", 2: "#2ecc71", 3: "#e67e22"}
FLOOR_LABELS = {1: "1F", 2: "2F", 3: "3F"}

ROOM_PALETTE = {
    "hallway": "#4a90d9",
    "office": "#e67e22",
    "staircase": "#7070aa",
    "garage": "#666666",
    "living_kitchen": "#4a90d9",
    "powder_room": "#e67e22",
    "guest_bath": "#27ae60",
    "master_bed": "#8e44ad",
    "master_bath": "#cc3333",
}

ACTIVITY_ICONS = {
    "sleeping": "😴",
    "stationary": "🐕",
    "moving": "🏃",
    "unknown": "❓",
}

# -----------------------------------------------------------------------------
# API Client
# -----------------------------------------------------------------------------

_floorplan_cache = None


def _api_get(path, params=None, timeout=2):
    """GET helper with error handling."""
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def fetch_position():
    return _api_get("/position")


def fetch_particles():
    return _api_get("/particles")


def fetch_rssi():
    return _api_get("/rssi") or {}


def fetch_rssi_history(limit=200):
    return _api_get("/rssi/history", {"limit": limit}) or []


def fetch_position_history(limit=200, source="memory"):
    return _api_get("/position/history", {"limit": limit, "source": source}) or []


def fetch_health():
    return _api_get("/health") or {
        "status": "offline",
        "mqtt_connected": False,
        "anchors_active": 0,
    }


def fetch_stats():
    return _api_get("/stats") or {}


def fetch_floorplan():
    global _floorplan_cache
    if _floorplan_cache is not None:
        return _floorplan_cache

    data = _api_get("/floorplan", timeout=5)
    if data:
        _floorplan_cache = data
        return data

    # Fallback: load local layout.json
    for path in [FLOORPLAN_LOCAL, "config/floorplan/layout.json"]:
        try:
            with open(path) as f:
                layout = json.load(f)
            _floorplan_cache = {"floorplan": layout, "anchor_coords": {}}
            return _floorplan_cache
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None


# -----------------------------------------------------------------------------
# Geometry Helpers
# -----------------------------------------------------------------------------


def bounds_to_polygon(bounds):
    """Convert bounds (polygon list or {x1,y1,x2,y2} rect) to [[x,y], ...]."""
    if isinstance(bounds, list):
        poly = [list(p) for p in bounds]
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly = poly[:-1]
        return poly
    x1, y1 = bounds["x1"], bounds["y1"]
    x2, y2 = bounds["x2"], bounds["y2"]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def hex_to_rgba(hex_color, alpha):
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _density_colors(px, py, radius=2.0):
    """Return normalised [0,1] density score for each particle.

    For each particle, counts how many other particles fall within
    *radius* (feet) using a simple grid-binning approach (O(N) instead
    of O(N²) pairwise distance).
    """
    if len(px) < 2:
        return [0.5] * len(px)
    inv_r = 1.0 / radius
    # Bin particles into grid cells of size `radius`
    from collections import Counter
    bins = Counter()
    keys = []
    for x, y in zip(px, py):
        k = (int(x * inv_r), int(y * inv_r))
        bins[k] += 1
        keys.append(k)
    # Each particle's density = sum of counts in its cell + 8 neighbours
    counts = []
    for k in keys:
        total = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                total += bins.get((k[0] + dx, k[1] + dy), 0)
        counts.append(total)
    max_c = max(counts)
    min_c = min(counts)
    rng = max_c - min_c
    if rng == 0:
        return [0.5] * len(px)
    return [(c - min_c) / rng for c in counts]


# -----------------------------------------------------------------------------
# Floor Plan Renderer
# -----------------------------------------------------------------------------

_dynamic_start = {}   # {floor_num: index of first dynamic trace}


def render_floor(floor_data, position=None, particles=None, trail=None,
                 is_active=True, show_heatmap=False, show_particles=True,
                 color_density=False):
    """Build a Plotly figure for one floor.

    Args:
        floor_data: One entry from layout.json ``floors`` list.
        position:   Current position dict (x, y, floor, confidence, ...).
        particles:  Particle dict with x/y/floor/weight arrays.
        trail:      List of position-history dicts.
        is_active:  True for the main view, False for dimmed thumbnails.
        show_heatmap: Show density heatmap overlay instead of/alongside particles.
        show_particles: Show individual particle scatter markers.
        color_density: Color particles by spatial density instead of weight.
    """
    fig = go.Figure()
    floor_num = floor_data["floor"]
    rooms = floor_data.get("rooms", [])
    all_x, all_y = [], []

    # ── Outer boundary ──
    ob = floor_data.get("outer_boundary", [])
    if ob:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in ob] + [ob[0][0]],
            y=[p[1] for p in ob] + [ob[0][1]],
            mode="lines",
            line=dict(color="#555" if is_active else "#333", width=2),
            showlegend=False, hoverinfo="skip",
        ))
        all_x.extend(p[0] for p in ob)
        all_y.extend(p[1] for p in ob)

    # ── Room polygons ──
    for room in rooms:
        poly = bounds_to_polygon(room["bounds"])
        color = ROOM_PALETTE.get(room["name"], "#4a80b0")
        walkable = room.get("walkable", True)
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        all_x.extend(p[0] for p in poly)
        all_y.extend(p[1] for p in poly)

        fill_alpha = 0.15 if walkable else 0.06
        line_alpha = 0.9 if is_active else 0.4
        fill_c = hex_to_rgba(color if walkable else "#888888", fill_alpha)
        line_c = hex_to_rgba(color if walkable else "#666666", line_alpha)

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=fill_c,
            line=dict(color=line_c, width=1.5 if is_active else 0.8),
            showlegend=False,
            hoverinfo="text" if is_active else "skip",
            hovertext=room.get("label", room["name"]),
        ))

        # Room label
        cx = np.mean([p[0] for p in poly])
        cy = np.mean([p[1] for p in poly])
        fig.add_annotation(
            x=cx, y=cy,
            text=room.get("label", room["name"]),
            showarrow=False,
            font=dict(
                size=10 if is_active else 7,
                color=hex_to_rgba(color, 0.85 if is_active else 0.5),
            ),
        )

        # ── Obstacles ──
        if is_active:
            for obs in room.get("obstacles", []):
                obs_poly = bounds_to_polygon(
                    obs.get("polygon", obs.get("bounds"))
                )
                ox = [p[0] for p in obs_poly] + [obs_poly[0][0]]
                oy = [p[1] for p in obs_poly] + [obs_poly[0][1]]
                fig.add_trace(go.Scatter(
                    x=ox, y=oy, mode="lines", fill="toself",
                    fillcolor="rgba(180,160,130,0.15)",
                    line=dict(color="#998866", width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))

        # ── Doorways ──
        if is_active:
            for door in room.get("doorways", []):
                _draw_door(fig, door)
            for so in room.get("stairs_openings", []):
                _draw_door(fig, so, color="#ff9800")

        # ── Gates ──
        if is_active:
            for gate in room.get("gates", []):
                seg = gate["segment"]
                fig.add_trace(go.Scatter(
                    x=[seg[0][0], seg[1][0]],
                    y=[seg[0][1], seg[1][1]],
                    mode="lines",
                    line=dict(color="#cc6600", width=2, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))

    # ── Anchors ──
    for anchor in floor_data.get("anchors", []):
        pos = anchor["position"]
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode="markers" + ("+text" if is_active else ""),
            marker=dict(size=8 if is_active else 5,
                        color=ACCENT, symbol="diamond"),
            text=[anchor["id"]] if is_active else None,
            textposition="bottom center",
            textfont=dict(size=7, color=ACCENT),
            showlegend=False,
            hovertext=anchor["id"], hoverinfo="text",
        ))

    # ── Stairs markers ──
    for stair in floor_data.get("stairs", []):
        entry = stair["entry"]
        direction = stair.get("direction", "up")
        sym = "triangle-up" if direction == "up" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[entry[0]], y=[entry[1]], mode="markers",
            marker=dict(size=9 if is_active else 5,
                        color="#ff9800", symbol=sym),
            showlegend=False,
            hovertext=f"Stairs {'↑' if direction == 'up' else '↓'} "
                       f"Floor {stair.get('to_floor', '?')}",
            hoverinfo="text",
        ))

    # ── Active-only dynamic overlays (fixed 9 traces for Patch support) ──
    if is_active:
        _dynamic_start[floor_num] = len(fig.data)

        # ── [+0] Particle cloud ──
        px, py, pw_norm = [], [], []
        p_vis = False
        if particles:
            mask = [i for i, f in enumerate(particles["floor"])
                    if f == floor_num]
            if mask and show_particles:
                px = [particles["x"][i] for i in mask]
                py = [particles["y"][i] for i in mask]
                pw = [particles["weight"][i] for i in mask]
                max_w = max(pw) if pw else 1.0
                pw_norm = (_density_colors(px, py) if color_density
                           else [w / max_w for w in pw])
                p_vis = True
        fig.add_trace(go.Scatter(
            x=px, y=py, mode="markers",
            marker=dict(size=4, color=pw_norm or None,
                        colorscale="Viridis", opacity=0.55,
                        showscale=False),
            showlegend=False, hoverinfo="skip", visible=p_vis,
        ))

        # ── [+1] Density heatmap overlay ──
        h_vis = bool(px and show_heatmap and len(px) > 2)
        fig.add_trace(go.Histogram2dContour(
            x=px if h_vis else [0],
            y=py if h_vis else [0],
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [0.2, "rgba(68,1,84,0.35)"],
                [0.4, "rgba(59,82,139,0.50)"],
                [0.6, "rgba(33,145,140,0.60)"],
                [0.8, "rgba(53,183,121,0.70)"],
                [1.0, "rgba(94,201,98,0.80)"],
            ],
            ncontours=12,
            showscale=False,
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            hoverinfo="skip",
            showlegend=False,
            visible=h_vis,
        ))

        # ── [+2..+6] Position trail (5 fading segments) ──
        same_floor = []
        if trail:
            same_floor = [t for t in trail
                          if t.get("floor") == floor_num][-60:]
        n_trail = len(same_floor)
        for s in range(5):
            if n_trail > 1:
                seg_sz = max(1, n_trail // 5)
                lo = s * seg_sz
                hi = min(lo + seg_sz + 1, n_trail)
                alpha = 0.25 + 0.65 * (s / 5)
                fig.add_trace(go.Scatter(
                    x=[t["x"] for t in same_floor[lo:hi]],
                    y=[t["y"] for t in same_floor[lo:hi]],
                    mode="lines",
                    line=dict(
                        color=f"rgba(255,38,146,{alpha:.2f})",
                        width=2,
                    ),
                    showlegend=False, hoverinfo="skip",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[], y=[], mode="lines",
                    line=dict(color="rgba(255,38,146,0.25)", width=2),
                    showlegend=False, hoverinfo="skip", visible=False,
                ))

        # ── [+7] Confidence circle  ── [+8] Dog marker ──
        if position and position.get("floor") == floor_num:
            x, y = position["x"], position["y"]
            conf = position.get("confidence", 0.5)

            radius = 2.5 * (1 - conf) + 0.4
            theta = np.linspace(0, 2 * np.pi, 60)
            fig.add_trace(go.Scatter(
                x=(x + radius * np.cos(theta)).tolist(),
                y=(y + radius * np.sin(theta)).tolist(),
                mode="lines", fill="toself",
                fillcolor="rgba(187,0,93,0.08)",
                line=dict(color="rgba(187,0,93,0.45)",
                          width=1, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=14, color="#bb005d",
                            line=dict(width=2, color="#80003f")),
                text=["🐕"], textposition="top center",
                textfont=dict(size=16),
                showlegend=False,
                hovertext=(
                    f"{position.get('location_label', '')}<br>"
                    f"({x:.1f}, {y:.1f})<br>"
                    f"Confidence: {conf:.0%}"
                ),
                hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="lines", fill="toself",
                fillcolor="rgba(187,0,93,0.08)",
                line=dict(color="rgba(187,0,93,0.45)",
                          width=1, dash="dash"),
                showlegend=False, hoverinfo="skip", visible=False,
            ))
            fig.add_trace(go.Scatter(
                x=[], y=[], mode="markers+text",
                marker=dict(size=14, color="#bb005d",
                            line=dict(width=2, color="#80003f")),
                text=["🐕"], textposition="top center",
                textfont=dict(size=16),
                showlegend=False, hoverinfo="text", visible=False,
            ))

    # ── Thumbnail particle overlay ──
    if not is_active and particles:
        mask = [i for i, f in enumerate(particles["floor"])
                if f == floor_num]
        if mask:
            px = [particles["x"][i] for i in mask]
            py = [particles["y"][i] for i in mask]
            pw = [particles["weight"][i] for i in mask]
            max_w = max(pw) if pw else 1.0
            pw_norm = (_density_colors(px, py) if color_density
                       else [w / max_w for w in pw])
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers",
                marker=dict(size=2, color=pw_norm,
                            colorscale="Viridis", opacity=0.5,
                            showscale=False),
                showlegend=False, hoverinfo="skip",
            ))

    # ── Layout ──
    pad = 0.5 if is_active else 1.5
    x_range = ([min(all_x) - pad, max(all_x) + pad]
               if all_x else [-1, 20])
    y_range = ([min(all_y) - pad, max(all_y) + pad]
               if all_y else [-1, 20])

    height = 680 if is_active else 220
    fig.update_layout(
        plot_bgcolor=PLOT_BG if is_active else "#111827",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=x_range, showgrid=True, gridcolor=GRID_COLOR,
                   title="ft" if is_active else None,
                   tickfont=dict(size=9, color=MUTED_TEXT),
                   title_font=dict(size=10, color=MUTED_TEXT)),
        yaxis=dict(range=y_range, showgrid=True, gridcolor=GRID_COLOR,
                   scaleanchor="x",
                   title="ft" if is_active else None,
                   tickfont=dict(size=9, color=MUTED_TEXT),
                   title_font=dict(size=10, color=MUTED_TEXT)),
        margin=(dict(l=35, r=10, t=35, b=25) if is_active
                else dict(l=20, r=5, t=30, b=15)),
        height=height,
        showlegend=False,
        title=dict(
            text=floor_data.get("name", f"Floor {floor_num}"),
            font=dict(size=13 if is_active else 10, color=TEXT_COLOR),
        ),
    )
    return fig


def _patch_active_dynamic(n_static, floor_num, position, particles, trail,
                          show_heatmap, show_particles, color_density=False):
    """Return a Patch updating only the 9 dynamic trace slots."""
    p = Patch()

    # ── [+0] Particle cloud ──
    pidx = n_static
    px, py, pw_norm = [], [], []
    p_vis = False
    if particles:
        mask = [i for i, f in enumerate(particles["floor"])
                if f == floor_num]
        if mask and show_particles:
            px = [particles["x"][i] for i in mask]
            py = [particles["y"][i] for i in mask]
            pw = [particles["weight"][i] for i in mask]
            max_w = max(pw) if pw else 1.0
            pw_norm = (_density_colors(px, py) if color_density
                       else [w / max_w for w in pw])
            p_vis = True
    p["data"][pidx]["x"] = px
    p["data"][pidx]["y"] = py
    if p_vis:
        p["data"][pidx]["marker"]["color"] = pw_norm
    p["data"][pidx]["visible"] = p_vis

    # ── [+1] Density heatmap ──
    hidx = n_static + 1
    h_vis = bool(px and show_heatmap and len(px) > 2)
    p["data"][hidx]["x"] = px if h_vis else [0]
    p["data"][hidx]["y"] = py if h_vis else [0]
    p["data"][hidx]["visible"] = h_vis

    # ── [+2..+6] Trail segments ──
    same_floor = []
    if trail:
        same_floor = [t for t in trail
                      if t.get("floor") == floor_num][-60:]
    n_trail = len(same_floor)
    for s in range(5):
        tidx = n_static + 2 + s
        if n_trail > 1:
            seg_sz = max(1, n_trail // 5)
            lo = s * seg_sz
            hi = min(lo + seg_sz + 1, n_trail)
            alpha = 0.25 + 0.65 * (s / 5)
            p["data"][tidx]["x"] = [t["x"] for t in same_floor[lo:hi]]
            p["data"][tidx]["y"] = [t["y"] for t in same_floor[lo:hi]]
            p["data"][tidx]["line"]["color"] = (
                f"rgba(255,38,146,{alpha:.2f})")
            p["data"][tidx]["visible"] = True
        else:
            p["data"][tidx]["x"] = []
            p["data"][tidx]["y"] = []
            p["data"][tidx]["visible"] = False

    # ── [+7] Confidence circle  ── [+8] Dog marker ──
    cidx = n_static + 7
    didx = n_static + 8
    if position and position.get("floor") == floor_num:
        x, y = position["x"], position["y"]
        conf = position.get("confidence", 0.5)
        radius = 2.5 * (1 - conf) + 0.4
        theta = np.linspace(0, 2 * np.pi, 60)
        p["data"][cidx]["x"] = (x + radius * np.cos(theta)).tolist()
        p["data"][cidx]["y"] = (y + radius * np.sin(theta)).tolist()
        p["data"][cidx]["visible"] = True
        p["data"][didx]["x"] = [x]
        p["data"][didx]["y"] = [y]
        p["data"][didx]["hovertext"] = (
            f"{position.get('location_label', '')}<br>"
            f"({x:.1f}, {y:.1f})<br>"
            f"Confidence: {conf:.0%}")
        p["data"][didx]["visible"] = True
    else:
        p["data"][cidx]["x"] = []
        p["data"][cidx]["y"] = []
        p["data"][cidx]["visible"] = False
        p["data"][didx]["x"] = []
        p["data"][didx]["y"] = []
        p["data"][didx]["visible"] = False

    return p


def _draw_door(fig, door, color="#00aa44"):
    """Draw a simplified door indicator on a floor plan."""
    cx, cy = door["center"]
    width = door.get("width", 2.0)
    wall_axis = door.get("wall_axis", "x")

    if wall_axis == "x":
        fig.add_trace(go.Scatter(
            x=[cx - width / 2, cx + width / 2], y=[cy, cy],
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.5, showlegend=False, hoverinfo="skip",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[cx, cx], y=[cy - width / 2, cy + width / 2],
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.5, showlegend=False, hoverinfo="skip",
        ))


# -----------------------------------------------------------------------------
# Chart Builders
# -----------------------------------------------------------------------------


def build_rssi_chart(rssi_data):
    """Horizontal RSSI bar chart, dark-themed."""
    if not rssi_data:
        return _empty_figure("No RSSI data", height=260)

    anchors = sorted(rssi_data.keys())
    values = []
    for a in anchors:
        v = rssi_data[a]
        values.append(v.get("rssi", v) if isinstance(v, dict) else v)

    colors = [
        SUCCESS if v > -60 else (WARNING if v > -75 else DANGER)
        for v in values
    ]
    fig = go.Figure(go.Bar(
        y=anchors, x=values, orientation="h",
        marker_color=colors, text=[f"{v:.0f}" for v in values],
        textposition="outside", textfont=dict(size=9, color=TEXT_COLOR),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-100, -30], title="dBm",
                   tickfont=dict(color=MUTED_TEXT, size=8),
                   title_font=dict(color=MUTED_TEXT, size=9),
                   gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=9),
                   autorange="reversed"),
        margin=dict(l=90, r=40, t=30, b=30),
        height=260,
        title=dict(text="RSSI by Anchor", font=dict(size=11, color=TEXT_COLOR)),
    )
    return fig


def build_floor_belief_chart(floor_belief):
    """Horizontal bar chart showing floor probability distribution."""
    if not floor_belief:
        return _empty_figure("No floor belief data", height=160)

    floors = sorted(floor_belief.keys(), key=lambda f: int(f))
    probs = [floor_belief[f] for f in floors]
    labels = [f"Floor {f}" for f in floors]
    colors = [FLOOR_COLORS.get(int(f), "#888") for f in floors]

    fig = go.Figure(go.Bar(
        y=labels, x=probs, orientation="h",
        marker_color=colors,
        text=[f"{p:.0%}" for p in probs],
        textposition="inside",
        textfont=dict(size=11, color="white", family="monospace"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=10)),
        margin=dict(l=65, r=10, t=5, b=5),
        height=120, bargap=0.3,
    )
    return fig


def build_timeline(history, window_label="1h"):
    """Floor occupancy + room timeline from position history."""
    if not history:
        return _empty_figure("No position history yet", height=200)

    # Parse timestamps
    valid = []
    for h in history:
        ts = h.get("timestamp")
        if not ts:
            continue
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                continue
        valid.append({**h, "_ts": ts})

    if not valid:
        return _empty_figure("No timestamped history", height=200)

    valid.sort(key=lambda h: h["_ts"])
    t0 = valid[0]["_ts"]

    fig = go.Figure()

    # ── Floor occupancy band (top strip) ──
    prev_floor = valid[0].get("floor", 1)
    seg_start = 0
    for i in range(1, len(valid)):
        cur_floor = valid[i].get("floor", prev_floor)
        if cur_floor != prev_floor or i == len(valid) - 1:
            t_start = (valid[seg_start]["_ts"] - t0).total_seconds() / 60
            t_end = (valid[i]["_ts"] - t0).total_seconds() / 60
            color = FLOOR_COLORS.get(prev_floor, "#888")
            fig.add_trace(go.Bar(
                x=[t_end - t_start], y=["Floor"],
                base=[t_start], orientation="h",
                marker_color=color, opacity=0.7,
                showlegend=False, hoverinfo="skip",
                width=0.6,
            ))
            fig.add_annotation(
                x=(t_start + t_end) / 2, y="Floor",
                text=FLOOR_LABELS.get(prev_floor, str(prev_floor)),
                showarrow=False,
                font=dict(size=9, color="white", family="monospace"),
            )
            prev_floor = cur_floor
            seg_start = i

    # ── Room timeline (scatter) ──
    mins = [(h["_ts"] - t0).total_seconds() / 60 for h in valid]
    room_labels = [h.get("location_label", "?") for h in valid]
    confs = [h.get("confidence", 0.5) for h in valid]

    unique_rooms = list(dict.fromkeys(room_labels))
    room_y = {r: idx for idx, r in enumerate(unique_rooms)}

    fig.add_trace(go.Scatter(
        x=mins,
        y=[unique_rooms[room_y[r]] for r in room_labels],
        mode="markers",
        marker=dict(
            size=6, color=confs,
            colorscale="YlOrRd", cmin=0, cmax=1,
            showscale=False, opacity=0.8,
        ),
        showlegend=False,
        hovertemplate="%{y}<br>%{marker.color:.0%} conf<extra></extra>",
    ))

    # ── Transition markers ──
    for i in range(1, len(valid)):
        if valid[i].get("floor") != valid[i - 1].get("floor"):
            t_min = (valid[i]["_ts"] - t0).total_seconds() / 60
            fig.add_vline(
                x=t_min, line_width=1,
                line_dash="dash", line_color=DANGER, opacity=0.5,
            )

    t_max = max(mins) if mins else 60
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[-0.5, t_max + 0.5],
            title=f"Minutes ({window_label} window)",
            tickfont=dict(color=MUTED_TEXT, size=8),
            title_font=dict(color=MUTED_TEXT, size=9),
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=9)),
        margin=dict(l=100, r=10, t=25, b=35),
        height=200,
        title=dict(
            text="Position Timeline",
            font=dict(size=11, color=TEXT_COLOR),
        ),
    )
    return fig


def build_rssi_sparklines(rssi_history):
    """Per-anchor RSSI trend sparklines."""
    if not rssi_history:
        return _empty_figure("No RSSI history", height=300)

    # Each entry is {anchor_id, rssi, distance, timestamp} (one per message)
    series = {}
    for entry in rssi_history:
        aid = entry.get("anchor_id")
        rssi_val = entry.get("rssi")
        if not aid or rssi_val is None:
            # Fallback: old dict-of-anchors format
            for k, v in entry.items():
                if k.startswith("_") or k == "timestamp":
                    continue
                rv = v.get("rssi", v) if isinstance(v, dict) else v
                if isinstance(rv, (int, float)):
                    series.setdefault(k, []).append(rv)
            continue
        series.setdefault(aid, []).append(rssi_val)

    if not series:
        return _empty_figure("No RSSI history", height=300)

    fig = go.Figure()
    colors = ["#4fc3f7", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc",
              "#26c6da", "#ffee58", "#ec407a", "#8d6e63", "#78909c"]
    for ci, (aid, rssi_vals) in enumerate(sorted(series.items())):
        fig.add_trace(go.Scatter(
            x=list(range(len(rssi_vals))), y=rssi_vals,
            mode="lines", name=aid,
            line=dict(color=colors[ci % len(colors)], width=1.5),
        ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Sample", tickfont=dict(color=MUTED_TEXT, size=8),
                   title_font=dict(color=MUTED_TEXT, size=9),
                   gridcolor=GRID_COLOR),
        yaxis=dict(title="RSSI (dBm)", range=[-100, -30],
                   tickfont=dict(color=MUTED_TEXT, size=8),
                   title_font=dict(color=MUTED_TEXT, size=9),
                   gridcolor=GRID_COLOR),
        legend=dict(font=dict(size=8, color=TEXT_COLOR),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=10, t=30, b=35),
        height=300,
        title=dict(text="RSSI Trends (per anchor)",
                   font=dict(size=11, color=TEXT_COLOR)),
    )
    return fig


def build_diagnostics_table(stats):
    """Return a list of dbc.ListGroupItem for pipeline diagnostics."""
    if not stats:
        return [dbc.ListGroupItem("Inference service offline",
                                  color="danger")]
    pipeline = stats.get("pipeline", {})
    items = [
        _diag_item("MQTT",
                    "Connected" if stats.get("mqtt_connected") else "Disconnected",
                    stats.get("mqtt_connected", False)),
        _diag_item("InfluxDB",
                    "Connected" if stats.get("influxdb_connected") else "Disconnected",
                    stats.get("influxdb_connected", False)),
        _diag_item("PostgreSQL",
                    "Connected" if stats.get("postgres_connected") else "Disconnected",
                    stats.get("postgres_connected", False)),
        _diag_item("Kalman Filter",
                    f"Active — {len(pipeline.get('kalman_anchors', []))} anchors"
                    if pipeline.get("kalman_active") else "Inactive",
                    pipeline.get("kalman_active", False)),
        _diag_item("Particle Filter",
                    f"n_eff={pipeline.get('particle_n_eff', 0):.0f} / "
                    f"{pipeline.get('particle_count', 0)}"
                    if pipeline.get("particle_filter_active") else "Inactive",
                    pipeline.get("particle_filter_active", False)),
        _diag_item("Floor HMM",
                    f"Floor {pipeline.get('most_likely_floor', '?')}"
                    if pipeline.get("floor_hmm_active") else "Inactive",
                    pipeline.get("floor_hmm_active", False)),
        _diag_item("RF Classifier",
                    f"Trained — {len(pipeline.get('rf_classes', []))} classes"
                    if pipeline.get("rf_classifier_active") else "Not trained",
                    pipeline.get("rf_classifier_active", False)),
    ]
    items.append(dbc.ListGroupItem([
        html.Small(f"Messages: {stats.get('message_count', 0):,}  |  "
                   f"Devices: {stats.get('devices_seen', 0)}  |  "
                   f"Beacon: {stats.get('beacon_id', '?')}",
                   className="text-muted"),
    ]))
    return items


def _diag_item(label, value, ok):
    color = "success" if ok else "warning"
    return dbc.ListGroupItem(
        [html.Strong(f"{label}: ", className="me-2"),
         html.Span(value)],
        color=color,
        className="py-1 px-3",
    )


def _empty_figure(msg, height=300):
    """Return a placeholder figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color=MUTED_TEXT),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=height, margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# -----------------------------------------------------------------------------
# Dash Application
# -----------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Pet Localization",
    update_title=None,
    suppress_callback_exceptions=True,
)
server = app.server  # WSGI entry-point for gunicorn


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, className="py-3", children=[

    # Interval timer
    dcc.Interval(id="interval", interval=DEFAULT_REFRESH_MS),

    # Hidden stores for cached data
    dcc.Store(id="store-position"),
    dcc.Store(id="store-particles"),
    dcc.Store(id="store-rssi"),
    dcc.Store(id="store-health"),
    dcc.Store(id="store-trail"),
    dcc.Store(id="store-displayed-floor"),

    # ── Header ──
    dbc.Row(className="mb-3 align-items-center", children=[
        dbc.Col(width=7, children=[
            html.H3("🐕 Bayesian Pet Localization",
                     className="mb-0 text-light"),
            html.Small("Real-time indoor positioning system",
                       className="text-muted"),
        ]),
        dbc.Col(width=5, className="text-end", children=[
            html.Div(id="header-badges", children=[
                dbc.Badge("API: ?", id="badge-api", color="danger",
                          className="me-2 px-2 py-1"),
                dbc.Badge("MQTT ✗", id="badge-mqtt", color="danger",
                          className="me-2 px-2 py-1"),
                dbc.Badge("Anchors: 0", id="badge-anchors", color="danger",
                          className="me-2 px-2 py-1"),
                dbc.Badge("🔋 —", id="badge-battery", color="secondary",
                          className="me-2 px-2 py-1"),
            ]),
            html.Div([
                html.Small("Refresh: ", className="text-muted me-1"),
                dbc.Select(
                    id="refresh-select",
                    options=[
                        {"label": "50 ms", "value": "50"},
                        {"label": "100 ms", "value": "100"},
                        {"label": "¼ s", "value": "250"},
                        {"label": "½ s", "value": "500"},
                        {"label": "1 s", "value": "1000"},
                        {"label": "2 s", "value": "2000"},
                        {"label": "5 s", "value": "5000"},
                        {"label": "Off", "value": "0"},
                    ],
                    value="1000",
                    size="sm",
                    style={"width": "80px", "display": "inline-block"},
                ),
            ], className="mt-1"),
        ]),
    ]),

    # ── Summary metrics ──
    dbc.Row(className="mb-3 g-2", children=[
        dbc.Col(width=6, md=3, children=[
            dbc.Card(className="bg-dark border-secondary text-center", children=[
                dbc.CardBody([
                    html.Small("Location", className="text-muted d-block"),
                    html.H5("—", id="metric-location",
                            className="mb-0 text-light"),
                ], className="py-2"),
            ]),
        ]),
        dbc.Col(width=6, md=3, children=[
            dbc.Card(className="bg-dark border-secondary text-center", children=[
                dbc.CardBody([
                    html.Small("Floor", className="text-muted d-block"),
                    html.H5("—", id="metric-floor",
                            className="mb-0 text-light"),
                ], className="py-2"),
            ]),
        ]),
        dbc.Col(width=6, md=3, children=[
            dbc.Card(className="bg-dark border-secondary text-center", children=[
                dbc.CardBody([
                    html.Small("Confidence", className="text-muted d-block"),
                    html.H5("—", id="metric-confidence",
                            className="mb-0 text-light"),
                ], className="py-2"),
            ]),
        ]),
        dbc.Col(width=6, md=3, children=[
            dbc.Card(className="bg-dark border-secondary text-center", children=[
                dbc.CardBody([
                    html.Small("Activity", className="text-muted d-block"),
                    html.H5("—", id="metric-activity",
                            className="mb-0 text-light"),
                ], className="py-2"),
            ]),
        ]),
    ]),

    # ── Tabs ──
    dbc.Tabs(id="tabs", active_tab="live", className="mb-3", children=[

        # ══════════ TAB 1: LIVE TRACKER ══════════
        dbc.Tab(label="Live Tracker", tab_id="live", children=[
            dbc.Row(className="g-3", children=[
                # Active floor (main panel)
                dbc.Col(width=12, lg=8, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="active-floor",
                                      config={"displayModeBar": False}),
                            html.Div(className="d-flex align-items-center mt-1", children=[
                                dbc.Checklist(
                                    id="viz-options",
                                    options=[
                                        {"label": " Particles",
                                         "value": "particles"},
                                        {"label": " Density Heatmap",
                                         "value": "heatmap"},
                                        {"label": " Color by Density",
                                         "value": "color_density"},
                                    ],
                                    value=["particles"],
                                    inline=True,
                                    switch=True,
                                    className="text-muted small",
                                ),
                            ]),
                        ], className="p-2"),
                    ]),
                ]),
                # Side panel
                dbc.Col(width=12, lg=4, children=[
                    # Floor belief
                    dbc.Card(className="bg-dark border-secondary mb-3",
                             children=[
                        dbc.CardHeader("Floor Belief",
                                       className="py-1 px-3 small"),
                        dbc.CardBody([
                            dcc.Graph(id="floor-belief",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                    # Activity
                    dbc.Card(className="bg-dark border-secondary mb-3",
                             children=[
                        dbc.CardHeader("Activity",
                                       className="py-1 px-3 small"),
                        dbc.CardBody(id="activity-card", className="p-3 text-center"),
                    ]),
                    # RSSI bars
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardHeader("Signal Strength",
                                       className="py-1 px-3 small"),
                        dbc.CardBody([
                            dcc.Graph(id="rssi-bars",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
            # Floor thumbnails (non-active floors only, left-aligned)
            dbc.Row(className="g-2 mt-2 mb-1", children=[
                dbc.Col(id="thumb-col-1", width=6, lg=4, children=[
                    dbc.Card(className="bg-dark border-secondary",
                             id="thumb-card-1", children=[
                        dbc.CardBody([
                            dcc.Graph(id="floor-thumb-1",
                                      config={"displayModeBar": False}),
                        ], className="p-1"),
                    ]),
                ]),
                dbc.Col(id="thumb-col-2", width=6, lg=4, children=[
                    dbc.Card(className="bg-dark border-secondary",
                             id="thumb-card-2", children=[
                        dbc.CardBody([
                            dcc.Graph(id="floor-thumb-2",
                                      config={"displayModeBar": False}),
                        ], className="p-1"),
                    ]),
                ]),
                dbc.Col(id="thumb-col-3", width=6, lg=4, children=[
                    dbc.Card(className="bg-dark border-secondary",
                             id="thumb-card-3", children=[
                        dbc.CardBody([
                            dcc.Graph(id="floor-thumb-3",
                                      config={"displayModeBar": False}),
                        ], className="p-1"),
                    ]),
                ]),
            ]),
            # Timeline
            dbc.Row(className="g-2 mt-1", children=[
                dbc.Col(width=12, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="timeline",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
        ]),

        # ══════════ TAB 2: SIGNAL ANALYSIS ══════════
        dbc.Tab(label="Signals", tab_id="signals", children=[
            dbc.Row(className="g-3 mt-1", children=[
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="sig-rssi-bars",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="sig-floor-belief",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
            dbc.Row(className="g-3 mt-1", children=[
                dbc.Col(width=12, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="sig-sparklines",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
        ]),

        # ══════════ TAB 3: HISTORY ══════════
        dbc.Tab(label="History", tab_id="history", children=[
            dbc.Row(className="mt-2 mb-3", children=[
                dbc.Col(width="auto", children=[
                    html.Label("Time window:", className="text-muted me-2"),
                    dbc.RadioItems(
                        id="history-window",
                        options=[
                            {"label": "1 h", "value": "1"},
                            {"label": "3 h", "value": "3"},
                            {"label": "6 h", "value": "6"},
                            {"label": "12 h", "value": "12"},
                            {"label": "24 h", "value": "24"},
                        ],
                        value="1",
                        inline=True,
                        className="d-inline",
                    ),
                ]),
            ]),
            dbc.Row(className="g-3", children=[
                dbc.Col(width=12, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="hist-timeline",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
            dbc.Row(className="g-3 mt-1", children=[
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="hist-dwell",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardBody([
                            dcc.Graph(id="hist-activity",
                                      config={"displayModeBar": False}),
                        ], className="p-2"),
                    ]),
                ]),
            ]),
        ]),

        # ══════════ TAB 4: DIAGNOSTICS ══════════
        dbc.Tab(label="Diagnostics", tab_id="diagnostics", children=[
            dbc.Row(className="g-3 mt-2", children=[
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardHeader("Pipeline Status",
                                       className="py-2 px-3"),
                        dbc.ListGroup(id="diag-list", flush=True),
                    ]),
                ]),
                dbc.Col(width=12, lg=6, children=[
                    dbc.Card(className="bg-dark border-secondary", children=[
                        dbc.CardHeader("Raw Stats",
                                       className="py-2 px-3"),
                        dbc.CardBody([
                            html.Pre(id="diag-raw",
                                     className="text-muted small mb-0",
                                     style={"maxHeight": "400px",
                                            "overflow": "auto"}),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
])


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

# ── Refresh rate control (clientside) ──
app.clientside_callback(
    """
    function(val) {
        const ms = parseInt(val);
        if (ms === 0) return [5000, true];
        return [ms, false];
    }
    """,
    Output("interval", "interval"),
    Output("interval", "disabled"),
    Input("refresh-select", "value"),
)

# ── Header badges (clientside) ──
app.clientside_callback(
    """
    function(health) {
        if (!health) return ["API Offline", "danger", "MQTT ✗", "danger",
                             "Anchors: 0", "danger", "🔋 —", "secondary"];
        const apiOk = health.status === "ok";
        const mqttOk = !!health.mqtt_connected;
        const anchors = health.anchors_active || 0;
        // Battery: CR2032 ranges ~3000mV (full) to ~2000mV (dead)
        let batText = "🔋 —";
        let batColor = "secondary";
        const mv = health.beacon_battery_mv;
        if (mv != null) {
            const pct = Math.min(100, Math.max(0, Math.round((mv - 2000) / 10)));
            batText = "🔋 " + pct + "% (" + mv + "mV)";
            batColor = pct > 40 ? "success" : pct > 15 ? "warning" : "danger";
        }
        return [
            "API: " + (health.status || "?"), apiOk ? "success" : "danger",
            mqttOk ? "MQTT" : "MQTT ✗", mqttOk ? "success" : "danger",
            "Anchors: " + anchors, anchors > 0 ? "success" : "danger",
            batText, batColor,
        ];
    }
    """,
    Output("badge-api", "children"), Output("badge-api", "color"),
    Output("badge-mqtt", "children"), Output("badge-mqtt", "color"),
    Output("badge-anchors", "children"), Output("badge-anchors", "color"),
    Output("badge-battery", "children"), Output("badge-battery", "color"),
    Input("store-health", "data"),
)

# ── Summary metrics (clientside) ──
app.clientside_callback(
    """
    function(pos) {
        const icons = {"sleeping": "😴", "stationary": "🐕",
                        "moving": "🏃", "unknown": "❓"};
        if (!pos) return ["—", "—", "—", "—"];
        const act = pos.activity || "unknown";
        const icon = icons[act] || "";
        const conf = pos.confidence || 0;
        return [
            pos.location_label || "Unknown",
            "Floor " + (pos.floor || "?"),
            Math.round(conf * 100) + "%",
            icon + " " + act.charAt(0).toUpperCase() + act.slice(1),
        ];
    }
    """,
    Output("metric-location", "children"),
    Output("metric-floor", "children"),
    Output("metric-confidence", "children"),
    Output("metric-activity", "children"),
    Input("store-position", "data"),
)


# ── Data fetch (fires on every interval tick) ──
@app.callback(
    Output("store-position", "data"),
    Output("store-particles", "data"),
    Output("store-rssi", "data"),
    Output("store-health", "data"),
    Output("store-trail", "data"),
    Input("interval", "n_intervals"),
)
def fetch_data(_n):
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_pos = ex.submit(fetch_position)
        f_par = ex.submit(fetch_particles)
        f_rssi = ex.submit(fetch_rssi)
        f_health = ex.submit(fetch_health)
        f_trail = ex.submit(fetch_position_history, 120)
    return (f_pos.result(), f_par.result(), f_rssi.result(),
            f_health.result(), f_trail.result())


# ── Live Tracker tab ──
@app.callback(
    Output("active-floor", "figure"),
    Output("floor-thumb-1", "figure"),
    Output("floor-thumb-2", "figure"),
    Output("floor-thumb-3", "figure"),
    Output("floor-belief", "figure"),
    Output("activity-card", "children"),
    Output("rssi-bars", "figure"),
    Output("timeline", "figure"),
    Output("thumb-col-1", "style"),
    Output("thumb-col-2", "style"),
    Output("thumb-col-3", "style"),
    Output("store-displayed-floor", "data"),
    Input("store-position", "data"),
    Input("store-particles", "data"),
    Input("store-rssi", "data"),
    Input("store-trail", "data"),
    State("tabs", "active_tab"),
    State("viz-options", "value"),
    State("interval", "n_intervals"),
    State("active-floor", "figure"),
    State("store-displayed-floor", "data"),
)
def update_live(position, particles, rssi, trail, active_tab, viz_options,
                n_intervals, current_fig, displayed_floor):
    if active_tab != "live":
        raise PreventUpdate

    viz_options = viz_options or []
    show_heatmap = "heatmap" in viz_options
    show_particles = "particles" in viz_options
    color_density = "color_density" in viz_options

    fp = fetch_floorplan()
    floors_list = (fp.get("floorplan", {}).get("floors", [])
                   if fp else [])
    floors_by_num = {f["floor"]: f for f in floors_list}

    active_floor = position.get("floor", 2) if position else 2
    floor_changed = active_floor != displayed_floor

    # Patch is safe only when: the browser has a figure for THIS floor,
    # n_intervals > 2 (not a fresh page load), and we have the trace map.
    n_static = _dynamic_start.get(active_floor)
    can_patch = (not floor_changed
                 and current_fig is not None
                 and n_static is not None
                 and (n_intervals or 0) > 2)

    # ── Active floor figure ──
    if active_floor in floors_by_num:
        if can_patch:
            main_fig = _patch_active_dynamic(
                n_static, active_floor, position, particles, trail,
                show_heatmap, show_particles,
                color_density=color_density,
            )
        else:
            main_fig = render_floor(
                floors_by_num[active_floor], position, particles, trail,
                is_active=True, show_heatmap=show_heatmap,
                show_particles=show_particles,
                color_density=color_density,
            )
    else:
        main_fig = _empty_figure("Floor data unavailable")

    # ── Thumbnails (only non-active floors) ──
    thumbs = []
    for fn in [1, 2, 3]:
        if fn in floors_by_num and fn != active_floor:
            thumbs.append(render_floor(
                floors_by_num[fn], particles=particles,
                is_active=False, color_density=color_density))
        elif fn == active_floor:
            thumbs.append(dash.no_update)
        else:
            thumbs.append(_empty_figure(f"Floor {fn}", height=220))

    # ── Thumbnail visibility (hide active floor, expand others) ──
    thumb_styles = []
    for fn in [1, 2, 3]:
        if fn == active_floor:
            thumb_styles.append({"display": "none"})
        else:
            thumb_styles.append({})

    # Floor belief
    fb = position.get("floor_belief") if position else None
    belief_fig = build_floor_belief_chart(fb)

    # Activity
    if position:
        act = position.get("activity", "unknown")
        icon = ACTIVITY_ICONS.get(act, "❓")
        act_children = [
            html.Span(icon, style={"fontSize": "2.5rem"}),
            html.H5(act.title(), className="mb-0 mt-1 text-light"),
        ]
    else:
        act_children = [html.Span("—", className="text-muted h4")]

    # RSSI
    rssi_fig = build_rssi_chart(rssi)

    # Timeline
    timeline_fig = build_timeline(trail, "recent")

    return (main_fig, *thumbs, belief_fig, act_children, rssi_fig,
            timeline_fig, *thumb_styles, active_floor)


# ── Signals tab ──
@app.callback(
    Output("sig-rssi-bars", "figure"),
    Output("sig-floor-belief", "figure"),
    Output("sig-sparklines", "figure"),
    Input("store-position", "data"),
    Input("store-rssi", "data"),
    Input("interval", "n_intervals"),
    State("tabs", "active_tab"),
)
def update_signals(position, rssi, _n, active_tab):
    if active_tab != "signals":
        raise PreventUpdate

    rssi_fig = build_rssi_chart(rssi)
    fb = position.get("floor_belief") if position else None
    belief_fig = build_floor_belief_chart(fb)
    rssi_hist = fetch_rssi_history(limit=200)
    spark_fig = build_rssi_sparklines(rssi_hist)
    return rssi_fig, belief_fig, spark_fig


# ── History tab ──
@app.callback(
    Output("hist-timeline", "figure"),
    Output("hist-dwell", "figure"),
    Output("hist-activity", "figure"),
    Input("history-window", "value"),
    Input("interval", "n_intervals"),
    State("tabs", "active_tab"),
)
def update_history(window, _n, active_tab):
    if active_tab != "history":
        raise PreventUpdate

    hours = int(window)
    # For longer windows, try database; in-memory for short
    source = "db" if hours > 1 else "memory"
    limit_map = {1: 500, 3: 1500, 6: 3000, 12: 5000, 24: 10000}
    limit = limit_map.get(hours, 500)
    history = fetch_position_history(limit=limit, source=source)

    timeline_fig = build_timeline(history, f"{hours}h")

    # Dwell time breakdown
    if history:
        room_time = {}
        for h in history:
            label = h.get("location_label", "unknown")
            room_time[label] = room_time.get(label, 0) + 1
        rooms = sorted(room_time.keys(), key=lambda r: room_time[r],
                       reverse=True)
        vals = [room_time[r] for r in rooms]
        total = sum(vals) or 1

        dwell_fig = go.Figure(go.Bar(
            x=[v / total for v in vals], y=rooms,
            orientation="h",
            marker_color=[ROOM_PALETTE.get(r.lower().replace(" ", "_"),
                                            ACCENT) for r in rooms],
            text=[f"{v / total:.0%}" for v in vals],
            textposition="outside",
            textfont=dict(size=9, color=TEXT_COLOR),
        ))
        dwell_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Fraction of time", range=[0, 1],
                       tickformat=".0%",
                       tickfont=dict(color=MUTED_TEXT, size=8),
                       title_font=dict(color=MUTED_TEXT, size=9),
                       gridcolor=GRID_COLOR),
            yaxis=dict(tickfont=dict(color=TEXT_COLOR, size=9),
                       autorange="reversed"),
            margin=dict(l=100, r=40, t=30, b=35), height=300,
            title=dict(text="Room Dwell Time",
                       font=dict(size=11, color=TEXT_COLOR)),
        )
    else:
        dwell_fig = _empty_figure("No history data", height=300)

    # Activity breakdown
    if history:
        act_counts = {}
        for h in history:
            a = h.get("activity", "unknown")
            act_counts[a] = act_counts.get(a, 0) + 1
        acts = list(act_counts.keys())
        act_vals = [act_counts[a] for a in acts]
        act_colors = {"sleeping": "#9c27b0", "stationary": "#2196f3",
                      "moving": "#ff9800", "unknown": "#666"}

        act_fig = go.Figure(go.Pie(
            labels=[f"{ACTIVITY_ICONS.get(a, '')} {a.title()}"
                    for a in acts],
            values=act_vals,
            marker_colors=[act_colors.get(a, "#888") for a in acts],
            textinfo="percent+label",
            textfont=dict(size=10, color="white"),
            hole=0.4,
        ))
        act_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10), height=300,
            title=dict(text="Activity Breakdown",
                       font=dict(size=11, color=TEXT_COLOR)),
            legend=dict(font=dict(color=TEXT_COLOR, size=9)),
        )
    else:
        act_fig = _empty_figure("No history data", height=300)

    return timeline_fig, dwell_fig, act_fig


# ── Diagnostics tab ──
@app.callback(
    Output("diag-list", "children"),
    Output("diag-raw", "children"),
    Input("interval", "n_intervals"),
    State("tabs", "active_tab"),
)
def update_diagnostics(_n, active_tab):
    if active_tab != "diagnostics":
        raise PreventUpdate
    stats = fetch_stats()
    items = build_diagnostics_table(stats)
    raw_json = json.dumps(stats, indent=2, default=str) if stats else "{}"
    return items, raw_json


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
