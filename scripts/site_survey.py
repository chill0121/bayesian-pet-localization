#!/usr/bin/env python3
"""
Site Survey Collection Script for Bayesian Pet Localization

CLI tool to record labeled RSSI fingerprints at grid positions across the
floorplan. Subscribes to MQTT, collects readings over a configurable dwell
time at each point, aggregates mean/std per anchor, and stores samples to
PostgreSQL ``fingerprint_samples`` table.

Grid points are auto-generated from layout.json room polygons at a configurable
resolution, filtered to walkable cells via the occupancy grid.

Usage:
    # Survey floor 2 at 2 ft resolution, 45s dwell (default)
    python scripts/site_survey.py --floor 2

    # Dry-run: show grid points without collecting
    python scripts/site_survey.py --floor 2 --dry-run

    # Survey all floors, custom dwell time
    python scripts/site_survey.py --all-floors --dwell 60

    # Resume an interrupted survey
    python scripts/site_survey.py --floor 2 --resume

    # Export to CSV alongside DB writes
    python scripts/site_survey.py --floor 2 --output-csv survey_f2.csv
"""

import argparse
import csv
import json
import math
import os
import select
import sys
import termios
import time
import tty
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

# Allow importing from services/inference/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "services", "inference")
)

from db import Database
from occupancy import OccupancyGridSet, bounds_to_polygon, _point_in_polygon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)

DEFAULT_RESOLUTION = 3.0    # feet between grid points
DEFAULT_DWELL = 30          # seconds per point
DEFAULT_MIN_ANCHORS = 3     # minimum anchors required for a valid sample
HIGH_VARIANCE_THRESHOLD = 8.0  # dBm — flag points above this
POI_DWELL_OCCLUDED = 15     # seconds per POI during occluded pass

# Progress file for resume capability
PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "..", "scratch")


# ---------------------------------------------------------------------------
# Layout + Grid Planning
# ---------------------------------------------------------------------------

def load_layout(path: str = LAYOUT_PATH) -> dict:
    """Load and return the floor plan layout data."""
    with open(path) as f:
        return json.load(f)


def load_anchors(layout_data: dict) -> dict:
    """Extract anchor positions: {anchor_id: {x, y, floor}}."""
    anchors = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {"x": pos[0], "y": pos[1], "floor": floor_num}
    return anchors


def build_room_polygons(layout_data: dict) -> dict:
    """Build {floor: [(room_name, polygon), ...]} for room labeling."""
    room_polygons = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        room_polygons[floor_num] = [
            (room["name"], bounds_to_polygon(room["bounds"]))
            for room in floor_data.get("rooms", [])
        ]
    return room_polygons


def build_room_gates(layout_data: dict) -> dict:
    """Parse zone gate definitions: {floor: {room_name: [gate_defs]}}."""
    room_gates = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        for room in floor_data.get("rooms", []):
            gates = room.get("gates", [])
            if not gates:
                continue
            parsed = []
            for gate in gates:
                between = gate["between"]
                parsed.append({
                    "axis": gate["axis"],
                    "coord": gate["coord"],
                    "above": between[0],
                    "below": between[1],
                })
            room_gates.setdefault(floor_num, {})[room["name"]] = parsed
    return room_gates


def label_room(x: float, y: float, floor: int, room_polygons: dict,
               room_gates: dict) -> str:
    """Determine room/zone label for a point using polygon + gates."""
    _room, zone = label_room_and_zone(x, y, floor, room_polygons, room_gates)
    return zone


def label_room_and_zone(x: float, y: float, floor: int, room_polygons: dict,
                        room_gates: dict) -> tuple[str, str]:
    """Return (parent_room, zone_label) for a point.

    If the point is inside a room with gate-defined sub-zones, returns
    the polygon room name as parent and the gate-resolved name as zone.
    Otherwise parent == zone (degenerate case, no sub-zones).
    """
    for name, poly in room_polygons.get(floor, []):
        if _point_in_polygon(x, y, poly):
            gates = room_gates.get(floor, {}).get(name, [])
            for gate in gates:
                val = y if gate["axis"] == "y" else x
                if val >= gate["coord"]:
                    return name, gate["above"]
                else:
                    return name, gate["below"]
            return name, name
    return "unknown", "unknown"


def generate_grid_points(
    layout_data: dict,
    grids: OccupancyGridSet,
    room_polygons: dict,
    room_gates: dict,
    floor: int,
    resolution: float = DEFAULT_RESOLUTION,
) -> list[dict]:
    """Generate survey grid points for a single floor.

    Returns a list of dicts: {x, y, floor, room, zone, point_type} sorted into
    a serpentine walking path.  ``room`` is the parent polygon name and
    ``zone`` is the gate-resolved sub-zone (equals ``room`` when no gates).
    """
    points = []
    seen = set()  # (round(x,2), round(y,2)) to deduplicate

    floor_data = None
    for fd in layout_data.get("floors", []):
        if fd["floor"] == floor:
            floor_data = fd
            break
    if floor_data is None:
        return []

    grid = grids[floor]

    # 1. Regular grid points within walkable area
    # Compute floor bounding box
    boundary = floor_data["outer_boundary"]
    xs = [p[0] for p in boundary]
    ys = [p[1] for p in boundary]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Offset grid by half-resolution to avoid landing on walls
    x_start = x_min + resolution / 2
    y_start = y_min + resolution / 2

    x = x_start
    row_idx = 0
    while x <= x_max:
        col_points = []
        y = y_start
        while y <= y_max:
            if grid.is_walkable(x, y):
                key = (round(x, 2), round(y, 2))
                if key not in seen:
                    parent_room, zone = label_room_and_zone(
                        x, y, floor, room_polygons, room_gates)
                    col_points.append({
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "floor": floor,
                        "room": parent_room,
                        "zone": zone,
                        "type": "grid",
                    })
                    seen.add(key)
            y += resolution

        # Serpentine: reverse every other column
        if row_idx % 2 == 1:
            col_points.reverse()
        points.extend(col_points)
        row_idx += 1
        x += resolution

    # 2. Mandatory points: doorway midpoints
    for room in floor_data.get("rooms", []):
        for dw in room.get("doorways", []):
            if dw.get("to") == "exterior":
                continue
            cx, cy = dw["center"]
            if grid.is_walkable(cx, cy):
                key = (round(cx, 2), round(cy, 2))
                if key not in seen:
                    parent_room, zone = label_room_and_zone(
                        cx, cy, floor, room_polygons, room_gates)
                    # Classify as "stairs" if either end is a staircase
                    is_stair = (
                        dw.get("to") == "staircase"
                        or room["name"] == "staircase"
                    )
                    points.append({
                        "x": round(cx, 2),
                        "y": round(cy, 2),
                        "floor": floor,
                        "room": parent_room,
                        "zone": zone,
                        "type": "stairs" if is_stair else "doorway",
                    })
                    seen.add(key)
        # Stairs openings
        for so in room.get("stairs_openings", []):
            cx, cy = so["center"]
            if grid.is_walkable(cx, cy):
                key = (round(cx, 2), round(cy, 2))
                if key not in seen:
                    parent_room, zone = label_room_and_zone(
                        cx, cy, floor, room_polygons, room_gates)
                    points.append({
                        "x": round(cx, 2),
                        "y": round(cy, 2),
                        "floor": floor,
                        "room": parent_room,
                        "zone": zone,
                        "type": "stairs",
                    })
                    seen.add(key)

    # 3. Mandatory points: stairway entries
    for stair in floor_data.get("stairs", []):
        cx, cy = stair["entry"]
        if grid.is_walkable(cx, cy):
            key = (round(cx, 2), round(cy, 2))
            if key not in seen:
                parent_room, zone = label_room_and_zone(
                    cx, cy, floor, room_polygons, room_gates)
                points.append({
                    "x": round(cx, 2),
                    "y": round(cy, 2),
                    "floor": floor,
                    "room": parent_room,
                    "zone": zone,
                    "type": "stairs",
                })
                seen.add(key)

    # 4. Mandatory points: POIs
    for room in floor_data.get("rooms", []):
        for poi in room.get("poi", []):
            px, py = poi["position"]
            if grid.is_walkable(px, py):
                key = (round(px, 2), round(py, 2))
                if key not in seen:
                    parent_room, zone = label_room_and_zone(
                        px, py, floor, room_polygons, room_gates)
                    points.append({
                        "x": round(px, 2),
                        "y": round(py, 2),
                        "floor": floor,
                        "room": parent_room,
                        "zone": zone,
                        "type": "poi",
                    })
                    seen.add(key)

    return points


# ---------------------------------------------------------------------------
# MQTT Collector
# ---------------------------------------------------------------------------

class RSSICollector:
    """Collects RSSI readings from MQTT during a dwell period."""

    def __init__(self, host: str, port: int, beacon_id: str):
        self.host = host
        self.port = port
        self.beacon_id = beacon_id
        self._client = None
        self._connected = False
        # Per-anchor list of raw RSSI readings during current dwell
        self._readings: dict[str, list[float]] = defaultdict(list)
        self._collecting = False

    def connect(self) -> bool:
        """Connect to MQTT broker. Returns True on success."""
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(self.host, self.port, 60)
            self._client.loop_start()
            # Wait for connection
            deadline = time.time() + 5.0
            while not self._connected and time.time() < deadline:
                time.sleep(0.1)
            return self._connected
        except Exception as e:
            print(f"\nERROR: Cannot connect to MQTT broker at {self.host}:{self.port}")
            print(f"  {e}")
            print(f"\nMake sure Mosquitto is running:")
            print(f"  docker compose up -d mosquitto")
            return False

    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def start_collection(self):
        """Clear buffer and begin collecting readings."""
        self._readings.clear()
        self._collecting = True

    def stop_collection(self) -> dict[str, list[float]]:
        """Stop collecting and return the buffered readings."""
        self._collecting = False
        return dict(self._readings)

    @property
    def reading_count(self) -> int:
        """Total readings buffered across all anchors."""
        return sum(len(v) for v in self._readings.values())

    @property
    def anchor_count(self) -> int:
        """Number of unique anchors that have reported."""
        return len(self._readings)

    @property
    def current_means(self) -> dict[str, float]:
        """Current per-anchor mean RSSI (for live display)."""
        return {
            a: round(sum(v) / len(v), 1) if v else 0.0
            for a, v in self._readings.items()
        }

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self._connected = rc == 0
        if self._connected:
            # Subscribe to our specific beacon only
            client.subscribe(f"espresense/devices/{self.beacon_id}/+")

    def _on_message(self, client, userdata, msg):
        if not self._collecting:
            return
        try:
            parts = msg.topic.split("/")
            if len(parts) < 4:
                return
            anchor_id = parts[3]
            payload = json.loads(msg.payload.decode())
            rssi = payload.get("rssi")
            if rssi is not None:
                self._readings[anchor_id].append(float(rssi))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_readings(readings: dict[str, list[float]]) -> dict:
    """Compute per-anchor mean/std/count from raw readings.

    Returns {
        "rssi_vector": {anchor: mean, ...},
        "rssi_std": {anchor: std, ...},
        "n_readings": int (mean across anchors),
        "anchor_count": int,
        "warnings": [str, ...],
    }
    """
    rssi_vector = {}
    rssi_std = {}
    counts = []
    warnings = []

    for anchor_id, values in readings.items():
        if len(values) < 2:
            rssi_vector[anchor_id] = round(values[0], 1) if values else -100.0
            rssi_std[anchor_id] = 0.0
            counts.append(len(values))
            continue

        arr = np.array(values)
        mean_rssi = float(np.mean(arr))
        std_rssi = float(np.std(arr, ddof=1))

        rssi_vector[anchor_id] = round(mean_rssi, 2)
        rssi_std[anchor_id] = round(std_rssi, 2)
        counts.append(len(values))

        if std_rssi > HIGH_VARIANCE_THRESHOLD:
            warnings.append(
                f"  ⚠ {anchor_id}: std={std_rssi:.1f} dBm (high variance)"
            )

    avg_readings = int(np.mean(counts)) if counts else 0

    return {
        "rssi_vector": rssi_vector,
        "rssi_std": rssi_std,
        "n_readings": avg_readings,
        "anchor_count": len(readings),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Progress / Resume
# ---------------------------------------------------------------------------

def progress_file_path(floor: int) -> str:
    """Return path to the progress file for a given floor."""
    return os.path.join(PROGRESS_DIR, f"survey_progress_f{floor}.json")


def save_progress(floor: int, completed_indices: list[int], survey_id: str,
                  notes: str = ""):
    """Save survey progress for resumption."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    data = {
        "floor": floor,
        "survey_id": survey_id,
        "completed": sorted(completed_indices),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    path = progress_file_path(floor)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_progress(floor: int) -> tuple[set[int], str]:
    """Load previously completed point indices. Returns (set, survey_id)."""
    path = progress_file_path(floor)
    if not os.path.exists(path):
        return set(), ""
    with open(path) as f:
        data = json.load(f)
    return set(data.get("completed", [])), data.get("survey_id", "")


# ---------------------------------------------------------------------------
# Terminal UI helpers
# ---------------------------------------------------------------------------

def clear_line():
    """Clear current terminal line."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def getch_nonblocking(timeout: float = 0.0) -> str:
    """Non-blocking single character read from stdin.

    Returns the character pressed, or "" if nothing within timeout.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            return ch
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------------------------------------------------------
# Dry-run visualization
# ---------------------------------------------------------------------------

def dry_run_display(points: list[dict], floor: int, resolution: float):
    """Display grid points summary and optionally plot."""
    zone_counts = defaultdict(int)
    room_counts = defaultdict(int)
    type_counts = defaultdict(int)
    for p in points:
        zone_counts[p.get("zone", p["room"])] += 1
        room_counts[p["room"]] += 1
        type_counts[p["type"]] += 1

    print(f"\n{'=' * 60}")
    print(f"  DRY RUN — Floor {floor} | {resolution} ft grid")
    print(f"{'=' * 60}")
    print(f"  Total points: {len(points)}")
    print()
    print("  Points by zone (parent room):")
    for zone, count in sorted(zone_counts.items()):
        # Find parent room for this zone
        parent = next(
            (p["room"] for p in points if p.get("zone") == zone),
            zone,
        )
        suffix = f"  ← {parent}" if parent != zone else ""
        print(f"    {zone:25s}  {count:3d}{suffix}")
    print()
    print("  Points by type:")
    for ptype, count in sorted(type_counts.items()):
        print(f"    {ptype:25s}  {count:3d}")
    print()

    # Occluded pass info
    poi_count = type_counts.get("poi", 0)
    if poi_count:
        print(f"  Occluded POI pass: {poi_count} points × "
              f"{POI_DWELL_OCCLUDED}s dwell")
    else:
        print("  Occluded POI pass: no POI points defined")
    print()

    # Try to generate a matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
        ax.set_aspect("equal")
        ax.set_title(f"Survey Grid — Floor {floor} ({resolution} ft)")
        ax.set_xlabel("X (feet, East →)")
        ax.set_ylabel("Y (feet, North →)")

        # Draw room polygons
        layout = load_layout()
        for fd in layout["floors"]:
            if fd["floor"] != floor:
                continue
            for room in fd.get("rooms", []):
                poly = bounds_to_polygon(room["bounds"])
                poly_closed = poly + [poly[0]]
                xs = [p[0] for p in poly_closed]
                ys = [p[1] for p in poly_closed]
                fill = "#e8e8e8" if room.get("walkable", True) else "#d0d0d0"
                ax.fill(xs, ys, alpha=0.3, color=fill)
                ax.plot(xs, ys, "k-", linewidth=0.8)
                # Room label at centroid
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                ax.text(cx, cy, room["name"], ha="center", va="center",
                        fontsize=7, alpha=0.5)

                # Draw obstacles
                for obs in room.get("obstacles", []):
                    obs_poly = bounds_to_polygon(
                        obs.get("polygon", obs.get("bounds"))
                    )
                    obs_closed = obs_poly + [obs_poly[0]]
                    ax.fill([p[0] for p in obs_closed],
                            [p[1] for p in obs_closed],
                            alpha=0.4, color="#999999")

        # Plot grid points with coordinate labels
        colors = {"grid": "#2196F3", "doorway": "#FF9800",
                  "stairs": "#4CAF50", "poi": "#E91E63"}
        for ptype, color in colors.items():
            pts = [p for p in points if p["type"] == ptype]
            if pts:
                ax.scatter([p["x"] for p in pts], [p["y"] for p in pts],
                           c=color, s=25, label=f"{ptype} ({len(pts)})",
                           zorder=5, edgecolors="white", linewidths=0.5)

        # Number each point and show coordinates
        for i, p in enumerate(points, 1):
            ax.annotate(
                f"{i}\n({p['x']:.1f},{p['y']:.1f})",
                (p["x"], p["y"]),
                textcoords="offset points", xytext=(4, 4),
                fontsize=4.5, color="#333333", zorder=6,
            )

        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

        out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"survey_grid_f{floor}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Grid plot saved to: {out_path}")

    except ImportError:
        print("  (matplotlib not installed — skipping plot)")

    print()


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def init_csv(path: str, anchors: list[str]) -> csv.DictWriter:
    """Create CSV file and return writer."""
    fieldnames = [
        "timestamp", "floor", "grid_x", "grid_y", "room", "zone",
        "point_type", "anchor_count", "n_readings", "duration_seconds",
        "notes",
    ] + [f"rssi_{a}" for a in sorted(anchors)] + [
        f"std_{a}" for a in sorted(anchors)
    ]
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return writer


# ---------------------------------------------------------------------------
# Main survey loop
# ---------------------------------------------------------------------------

def run_survey(
    collector: RSSICollector,
    db_client: Database | None,
    points: list[dict],
    floor: int,
    dwell: int,
    min_anchors: int,
    survey_id: str,
    resume: bool = False,
    csv_writer: csv.DictWriter | None = None,
    known_anchors: list[str] | None = None,
    zone_override: str | None = None,
    skip_occluded: bool = False,
):
    """Main interactive survey loop.

    Iterates through grid points, collecting RSSI at each position via
    keypress-triggered dwell periods.
    """
    completed = set()
    if resume:
        completed, prev_id = load_progress(floor)
        if completed:
            print(f"\n  Resuming survey: {len(completed)} points already "
                  f"completed (survey {prev_id})")
        survey_id = prev_id or survey_id

    total = len(points)
    remaining = [i for i in range(total) if i not in completed]

    if not remaining:
        print("\n  All points already completed! Nothing to do.")
        return

    print(f"\n{'=' * 60}")
    print(f"  SITE SURVEY — Floor {floor}")
    print(f"  {len(remaining)} points remaining of {total}")
    print(f"  Dwell: {dwell}s | Min anchors: {min_anchors}")
    print(f"  Survey ID: {survey_id}")
    print(f"{'=' * 60}")
    print()
    print("  Controls:")
    print("    ENTER  — Start collection at current point")
    print("    s      — Skip this point")
    print("    r      — Redo the previous point")
    print("    n      — Add a note to the next sample")
    print("    q      — Save progress and quit")
    print()

    note_buffer = ""
    last_completed_idx = None
    point_num = 0

    # Apply zone override if specified
    if zone_override:
        for p in points:
            p["zone"] = zone_override
        print(f"  Zone override active: all points labeled as '{zone_override}'")
        print()

    for seq, idx in enumerate(remaining):
        point = points[idx]
        point_num = seq + 1 + len(completed)

        # Show next point prompt
        zone_info = point['zone']
        if point['room'] != point['zone']:
            zone_info = f"{point['zone']} (in {point['room']})"
        print(f"{'─' * 60}")
        print(f"  Point {point_num}/{total}  "
              f"({point['x']:.1f}, {point['y']:.1f})  [{zone_info}]  "
              f"({point['type']})")
        print(f"  Place beacon on stand → ", end="")

        # Wait for user input
        while True:
            print("ENTER=start | s=skip | n=note | q=quit: ", end="")
            sys.stdout.flush()

            # Read a line (cooked mode so user can see what they type)
            try:
                user_input = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_input = "q"

            if user_input in ("", "start"):
                break
            elif user_input == "s":
                print(f"  → Skipped point {point_num}")
                break
            elif user_input == "n":
                note_buffer = input("  Note: ").strip()
                print(f"  → Note saved: \"{note_buffer}\"")
                continue
            elif user_input == "r" and last_completed_idx is not None:
                # Redo last point — mark it as not completed
                completed.discard(last_completed_idx)
                save_progress(floor, sorted(completed), survey_id)
                prev_pt = points[last_completed_idx]
                print(f"  → Redo point at ({prev_pt['x']}, {prev_pt['y']}) "
                      f"[{prev_pt['room']}]")
                print(f"  Place beacon → press ENTER when ready: ", end="")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                _collect_point(
                    collector, db_client, prev_pt, last_completed_idx,
                    floor, dwell, min_anchors, survey_id, note_buffer,
                    csv_writer, known_anchors, completed, total,
                    point_num - 1,
                )
                completed.add(last_completed_idx)
                save_progress(floor, sorted(completed), survey_id)
                note_buffer = ""
                continue
            elif user_input == "q":
                save_progress(floor, sorted(completed), survey_id)
                print(f"\n  Progress saved. {len(completed)}/{total} "
                      f"completed. Use --resume to continue.")
                return
            else:
                print(f"  Unknown command: '{user_input}'")
                continue

        if user_input == "s":
            continue
        if user_input == "q":
            save_progress(floor, sorted(completed), survey_id)
            print(f"\n  Progress saved. {len(completed)}/{total} "
                  f"completed. Use --resume to continue.")
            return

        # Collect at this point
        _collect_point(
            collector, db_client, point, idx, floor, dwell,
            min_anchors, survey_id, note_buffer, csv_writer,
            known_anchors, completed, total, point_num,
        )
        completed.add(idx)
        save_progress(floor, sorted(completed), survey_id)
        last_completed_idx = idx
        note_buffer = ""

    # Survey complete
    save_progress(floor, sorted(completed), survey_id)
    print(f"\n{'═' * 60}")
    print(f"  ✓ Floor {floor} survey COMPLETE — {len(completed)}/{total} points")
    print(f"{'═' * 60}")

    # --- Occluded POI pass ---
    poi_points = [(i, p) for i, p in enumerate(points) if p["type"] == "poi"]
    if poi_points and not skip_occluded:
        print(f"\n{'=' * 60}")
        print(f"  OCCLUDED POI PASS — Floor {floor}")
        print(f"  {len(poi_points)} POI points | {POI_DWELL_OCCLUDED}s dwell each")
        print(f"{'=' * 60}")
        print()
        print("  Place the beacon face-down (or under a light weight)")
        print("  to simulate body occlusion at each POI.")
        print()
        print("  ENTER=start pass | S=skip entire pass: ", end="")
        sys.stdout.flush()
        try:
            start_input = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            start_input = "s"

        if start_input != "s":
            for seq, (idx_poi, poi) in enumerate(poi_points, 1):
                zone_info = poi["zone"]
                if poi["room"] != poi["zone"]:
                    zone_info = f"{poi['zone']} (in {poi['room']})"
                print(f"{'─' * 60}")
                print(f"  Occluded {seq}/{len(poi_points)}  "
                      f"({poi['x']}, {poi['y']})  [{zone_info}]")
                print(f"  Place beacon face-down → ", end="")
                print("ENTER=start | s=skip | q=quit: ", end="")
                sys.stdout.flush()
                try:
                    user_input = input().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    user_input = "q"

                if user_input == "s":
                    print(f"  → Skipped occluded {seq}")
                    continue
                if user_input == "q":
                    print(f"  → Occluded pass stopped early.")
                    break

                occ_note = "occluded"
                if note_buffer:
                    occ_note = f"occluded; {note_buffer}"
                _collect_point(
                    collector, db_client, poi, idx_poi, floor,
                    POI_DWELL_OCCLUDED, min_anchors, survey_id,
                    occ_note, csv_writer, known_anchors,
                    completed, total, point_num,
                )

            print(f"\n  ✓ Occluded POI pass complete.")
        else:
            print(f"  → Skipped occluded pass.")

    # QC summary
    _print_qc_summary(points, completed)


def _collect_point(
    collector: RSSICollector,
    db_client: Database | None,
    point: dict,
    idx: int,
    floor: int,
    dwell: int,
    min_anchors: int,
    survey_id: str,
    note: str,
    csv_writer: csv.DictWriter | None,
    known_anchors: list[str] | None,
    completed: set,
    total: int,
    point_num: int,
):
    """Collect RSSI readings at a single grid point."""
    collector.start_collection()
    start_time = time.time()
    end_time = start_time + dwell

    # Live progress during collection
    try:
        while time.time() < end_time:
            remaining_s = int(end_time - time.time())
            n_readings = collector.reading_count
            n_anchors = collector.anchor_count
            bar_len = 30
            elapsed_frac = (time.time() - start_time) / dwell
            filled = int(bar_len * elapsed_frac)
            bar = "█" * filled + "░" * (bar_len - filled)

            sys.stdout.write(
                f"\r  Collecting... {remaining_s:2d}s  |{bar}|  "
                f"Anchors: {n_anchors}  Readings: {n_readings}"
            )
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Collection interrupted!")

    readings = collector.stop_collection()
    actual_duration = time.time() - start_time

    # Aggregate
    result = aggregate_readings(readings)
    print()  # newline after progress bar

    # QC check
    if result["anchor_count"] < min_anchors:
        print(f"  ✗ REJECTED — only {result['anchor_count']} anchors "
              f"(need {min_anchors})")
        print(f"    Skipping point. Try again or check anchor health.")
        # Terminal bell
        sys.stdout.write("\a")
        sys.stdout.flush()
        return

    # Show warnings
    for w in result["warnings"]:
        print(w)

    # Build labels for classifier
    zone_label = point.get("zone", point["room"])
    parent_room = point["room"]
    location_label = zone_label  # zone is the primary label (no floor prefix)

    # Build notes string
    notes_parts = [f"survey_id={survey_id}", f"type={point['type']}"]
    if note:
        notes_parts.append(note)
    notes_str = "; ".join(notes_parts)

    # Write to database
    if db_client is not None and db_client.connected:
        success = db_client.write_fingerprint(
            location_label=location_label,
            zone_label=zone_label,
            room=parent_room,
            floor=floor,
            grid_x=point["x"],
            grid_y=point["y"],
            rssi_vector=result["rssi_vector"],
            rssi_std=result["rssi_std"],
            duration_seconds=round(actual_duration, 1),
            n_readings=result["n_readings"],
            notes=notes_str,
        )
        if success:
            print(f"  ✓ Saved ({result['anchor_count']} anchors, "
                  f"{result['n_readings']} avg readings)")
        else:
            print(f"  ⚠ DB write failed — data logged to CSV/progress only")
    else:
        print(f"  ✓ Collected ({result['anchor_count']} anchors, "
              f"{result['n_readings']} avg readings) [no DB]")

    # Write to CSV if enabled
    if csv_writer is not None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "floor": floor,
            "grid_x": point["x"],
            "grid_y": point["y"],
            "room": parent_room,
            "zone": zone_label,
            "point_type": point["type"],
            "anchor_count": result["anchor_count"],
            "n_readings": result["n_readings"],
            "duration_seconds": round(actual_duration, 1),
            "notes": notes_str,
        }
        for anchor_id, rssi in result["rssi_vector"].items():
            row[f"rssi_{anchor_id}"] = rssi
        for anchor_id, std in result["rssi_std"].items():
            row[f"std_{anchor_id}"] = std
        csv_writer.writerow(row)

    # Terminal bell to signal completion
    sys.stdout.write("\a")
    sys.stdout.flush()


def _print_qc_summary(points: list[dict], completed: set[int]):
    """Print a quality-control summary of the completed survey."""
    zone_counts = defaultdict(int)
    zone_total = defaultdict(int)
    zone_parent = {}
    for i, p in enumerate(points):
        zone = p.get("zone", p["room"])
        zone_total[zone] += 1
        zone_parent[zone] = p["room"]
        if i in completed:
            zone_counts[zone] += 1

    print("\n  QC Summary — Points per zone:")
    all_zones = sorted(set(list(zone_counts.keys()) + list(zone_total.keys())))
    for zone in all_zones:
        done = zone_counts.get(zone, 0)
        total = zone_total.get(zone, 0)
        status = "✓" if done >= 3 else "⚠ <3 points"
        parent = zone_parent.get(zone, "")
        suffix = f"  (in {parent})" if parent and parent != zone else ""
        print(f"    {zone:25s}  {done:3d}/{total:3d}  {status}{suffix}")
    print()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Site survey RSSI collection for BLE pet localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/site_survey.py --floor 2
  python scripts/site_survey.py --floor 2 --dry-run
  python scripts/site_survey.py --all-floors --dwell 60
  python scripts/site_survey.py --floor 2 --resume
  python scripts/site_survey.py --floor 2 --output-csv survey_f2.csv
        """,
    )
    parser.add_argument(
        "--host", default="localhost",
        help="MQTT broker host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--beacon-id",
        default=os.getenv(
            "BEACON_ID",
            "iBeacon:426c7565-4368-6172-6d42-6561636f6e73-3838-4949",
        ),
        help="Beacon device ID (default: from BEACON_ID env or hardcoded)",
    )
    parser.add_argument(
        "--floor", type=int, default=None,
        help="Floor number to survey (1, 2, or 3)",
    )
    parser.add_argument(
        "--all-floors", action="store_true",
        help="Survey all floors sequentially",
    )
    parser.add_argument(
        "--resolution", type=float, default=DEFAULT_RESOLUTION,
        help=f"Grid spacing in feet (default: {DEFAULT_RESOLUTION})",
    )
    parser.add_argument(
        "--dwell", type=int, default=DEFAULT_DWELL,
        help=f"Seconds per grid point (default: {DEFAULT_DWELL})",
    )
    parser.add_argument(
        "--min-anchors", type=int, default=DEFAULT_MIN_ANCHORS,
        help=f"Minimum anchors required for valid sample (default: {DEFAULT_MIN_ANCHORS})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted survey from saved progress",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show grid points and plot without collecting data",
    )
    parser.add_argument(
        "--zone-label", type=str, default=None,
        help="Override auto-detected zone label for all points (e.g., 'office_dog_bed')",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Optional CSV output file path",
    )
    # DB connection (defaults from env)
    parser.add_argument(
        "--db-host",
        default=os.getenv("POSTGRES_HOST", "localhost"),
    )
    parser.add_argument(
        "--db-port", type=int,
        default=int(os.getenv("POSTGRES_PORT", "5432")),
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("POSTGRES_USER", "localization"),
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("POSTGRES_PASSWORD", ""),
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("POSTGRES_DB", "pet_tracking"),
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip database writes (CSV/console only)",
    )
    parser.add_argument(
        "--no-occluded", action="store_true",
        help="Skip the occluded POI collection pass",
    )

    args = parser.parse_args()

    # Validate floor selection
    if args.floor is None and not args.all_floors:
        parser.error("Specify --floor N or --all-floors")

    floors_to_survey = [1, 2, 3] if args.all_floors else [args.floor]

    # Load layout
    print(f"\n  Loading layout from {LAYOUT_PATH}...")
    layout_data = load_layout()
    grids = OccupancyGridSet.from_layout_data(layout_data, resolution=0.5)
    room_polygons = build_room_polygons(layout_data)
    room_gates = build_room_gates(layout_data)
    anchors = load_anchors(layout_data)

    print(f"  Loaded {len(anchors)} anchors across "
          f"{len(layout_data['floors'])} floors")

    # Generate grid points and handle dry-run
    for floor in floors_to_survey:
        if floor not in grids:
            print(f"\n  WARNING: Floor {floor} not found in layout — skipping")
            continue

        points = generate_grid_points(
            layout_data, grids, room_polygons, room_gates,
            floor, args.resolution,
        )

        if not points:
            print(f"\n  WARNING: No walkable grid points on floor {floor}")
            continue

        if args.dry_run:
            dry_run_display(points, floor, args.resolution)
            continue

        # Survey ID for this run
        survey_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Connect to MQTT
        collector = RSSICollector(args.host, args.port, args.beacon_id)
        print(f"\n  Connecting to MQTT broker at {args.host}:{args.port}...")
        if not collector.connect():
            return

        print(f"  Connected ✓  (beacon: {args.beacon_id})")

        # Connect to DB
        db_client = None
        if not args.no_db:
            db_client = Database(
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                password=args.db_password,
                dbname=args.db_name,
            )
            if db_client.connect():
                print(f"  PostgreSQL connected ✓")
            else:
                print(f"  PostgreSQL unavailable — data will be CSV/console only")
                db_client = None

        # CSV output
        csv_writer = None
        csv_file = None
        if args.output_csv:
            anchor_ids = sorted(anchors.keys())
            csv_file = open(args.output_csv, "w", newline="")
            fieldnames = [
                "timestamp", "floor", "grid_x", "grid_y", "room", "zone",
                "point_type", "anchor_count", "n_readings",
                "duration_seconds", "notes",
            ] + [f"rssi_{a}" for a in anchor_ids] + [
                f"std_{a}" for a in anchor_ids
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            print(f"  CSV output: {args.output_csv}")

        try:
            run_survey(
                collector=collector,
                db_client=db_client,
                points=points,
                floor=floor,
                dwell=args.dwell,
                min_anchors=args.min_anchors,
                survey_id=survey_id,
                resume=args.resume,
                csv_writer=csv_writer,
                known_anchors=sorted(anchors.keys()),
                zone_override=args.zone_label,
                skip_occluded=args.no_occluded,
            )
        except KeyboardInterrupt:
            print(f"\n\n  Survey interrupted (Ctrl+C).")
        finally:
            collector.disconnect()
            if db_client is not None:
                db_client.close()
            if csv_file is not None:
                csv_file.close()

    if args.dry_run:
        all_points = []
        for f in floors_to_survey:
            if f in grids:
                all_points.extend(generate_grid_points(
                    layout_data, grids, room_polygons, room_gates,
                    f, args.resolution,
                ))
        total_pts = len(all_points)
        poi_pts = sum(1 for p in all_points if p["type"] == "poi")
        grid_time = total_pts * (args.dwell + 15) / 60
        occ_time = poi_pts * (POI_DWELL_OCCLUDED + 10) / 60
        print(f"  Estimated time:")
        print(f"    Grid pass:     ~{grid_time:.0f} min  "
              f"({total_pts} pts × {args.dwell + 15}s)")
        print(f"    Occluded pass: ~{occ_time:.0f} min  "
              f"({poi_pts} POI × {POI_DWELL_OCCLUDED + 10}s)")
        print(f"    Total:         ~{grid_time + occ_time:.0f} min")


if __name__ == "__main__":
    main()
