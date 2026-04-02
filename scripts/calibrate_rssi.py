#!/usr/bin/env python3
"""
RSSI Path-Loss Calibration Script

Collects RSSI samples at known distances from anchors to fit the log-distance
path-loss model parameters:  TX_POWER_DBM and PATH_LOSS_N.

Procedure:
  1. Stand at known positions relative to a same-room anchor.
  2. Script collects RSSI for a dwell period, computes mean.
  3. After all positions, least-squares fits:
         RSSI = TX_POWER - 10 * N * log10(d_meters)
  4. Prints fitted constants and residuals.

Can also measure WALL_ATTENUATION_DB by comparing readings from same-room
vs through-wall positions at similar distances.

Usage:
    # Interactive: prompted per measurement point
    python scripts/calibrate_rssi.py

    # Quick: 3ft, 6ft, 10ft from 1F_Office anchor, 30s dwell each
    python scripts/calibrate_rssi.py --anchor 1F_Office --beacon dog_collar --distances 3,6,10 --dwell 30

    # Through-wall measurement for wall attenuation
    python scripts/calibrate_rssi.py --anchor 1F_Office --wall-test --dwell 30
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt

# Allow importing from services/inference/
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "services", "inference")
)

from filters.constants import TX_POWER_DBM, PATH_LOSS_N, WALL_ATTENUATION_DB

def _load_beacon_id_default() -> str:
    """Read BEACON_ID from environment or .env file."""
    val = os.environ.get("BEACON_ID")
    if val:
        return val
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("BEACON_ID="):
                    return line.split("=", 1)[1].strip()
    return "dog_collar"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAYOUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "floorplan", "layout.json"
)
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC = "espresense/devices/+/+"

DEFAULT_BEACON_ID = _load_beacon_id_default()
DEFAULT_DWELL = 30  # seconds per measurement point
FT_TO_M = 0.3048


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def load_anchors() -> dict:
    """Load anchor positions from layout.json → {id: {x, y, floor, mac}}."""
    with open(LAYOUT_PATH) as f:
        data = json.load(f)
    anchors = {}
    for floor_data in data.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [0, 0])
            anchors[a["id"]] = {
                "x": pos[0], "y": pos[1],
                "floor": floor_num,
                "mac": a.get("mac", ""),
            }
    return anchors


# ---------------------------------------------------------------------------
# MQTT Collector
# ---------------------------------------------------------------------------

class RSSICollector:
    """Subscribes to ESPresense MQTT and collects raw RSSI per anchor."""

    def __init__(self, anchor_macs: dict[str, str], beacon_id: str):
        """
        anchor_macs : {anchor_id: mac_address}
            Only topics matching these anchors' node names are captured.
        beacon_id : str
            ESPresense device name to filter on (e.g. "dog_collar").
        """
        self._beacon_id = beacon_id
        self._anchor_ids: set = set()
        for aid in anchor_macs:
            self._anchor_ids.add(aid)

        self._samples: dict[str, list[float]] = defaultdict(list)
        self._collecting = False

        self._client = mqtt.Client()
        self._client.on_message = self._on_message

    def _on_message(self, client, userdata, msg):
        if not self._collecting:
            return
        try:
            payload = json.loads(msg.payload)
            rssi = payload.get("rssi")
            if rssi is None:
                return

            # Topic: espresense/devices/<beacon_id>/<node_name>
            parts = msg.topic.split("/")
            if len(parts) < 4:
                return
            beacon_id = parts[2]
            node_name = parts[3]

            # Only accept messages from the target beacon
            if beacon_id != self._beacon_id:
                return

            # Match node_name to anchor_id (case-insensitive)
            node_lower = node_name.lower().replace("-", "_")
            for aid in self._anchor_ids:
                if aid.lower().replace("-", "_") == node_lower:
                    self._samples[aid].append(float(rssi))
                    break
        except (json.JSONDecodeError, ValueError):
            pass

    def connect(self):
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.subscribe(MQTT_TOPIC)
        self._client.loop_start()

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def collect(self, dwell_seconds: float) -> dict[str, dict]:
        """Collect RSSI for *dwell_seconds*. Returns {anchor_id: {mean, std, n}}."""
        self._samples.clear()
        self._collecting = True
        time.sleep(dwell_seconds)
        self._collecting = False

        results = {}
        for aid, vals in self._samples.items():
            if vals:
                results[aid] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        return results


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_path_loss(measurements: list[dict]) -> dict:
    """Fit TX_POWER and PATH_LOSS_N from calibration measurements.

    Each measurement: {distance_ft: float, rssi_mean: float}

    Model: RSSI = TX_POWER - 10 * N * log10(d_meters)
    Rewrite as:  RSSI = a + b * log10(d_m)
       where a = TX_POWER, b = -10*N

    Uses least-squares.
    """
    if len(measurements) < 2:
        return {"tx_power": TX_POWER_DBM, "path_loss_n": PATH_LOSS_N,
                "error": "Need at least 2 measurements to fit"}

    # Build design matrix
    y = np.array([m["rssi_mean"] for m in measurements])
    x = np.array([math.log10(m["distance_ft"] * FT_TO_M) for m in measurements])

    # y = a + b*x  →  [1, x] @ [a, b]^T = y
    A = np.column_stack([np.ones_like(x), x])
    result, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

    tx_power = float(result[0])
    path_loss_n = float(-result[1] / 10.0)

    # Compute per-point residuals
    predicted = result[0] + result[1] * x
    point_residuals = y - predicted
    rmse = float(np.sqrt(np.mean(point_residuals ** 2)))

    return {
        "tx_power": round(tx_power, 1),
        "path_loss_n": round(path_loss_n, 2),
        "rmse": round(rmse, 2),
        "residuals": [round(float(r), 2) for r in point_residuals],
        "n_points": len(measurements),
    }


def estimate_wall_attenuation(
    same_room: list[dict], through_wall: list[dict]
) -> dict:
    """Estimate wall attenuation by comparing same-room vs through-wall.

    Each entry: {distance_ft, rssi_mean, n_walls}
    At similar distances, the difference in RSSI divided by walls = attenuation.
    """
    if not same_room or not through_wall:
        return {"wall_attenuation_db": WALL_ATTENUATION_DB,
                "error": "Need both same-room and through-wall samples"}

    # For each through-wall reading, predict what RSSI *would be* without
    # wall penalty (using the same-room fitted model), then difference = walls * atten
    attenuations = []
    for tw in through_wall:
        dist_m = tw["distance_ft"] * FT_TO_M
        # Expected RSSI without walls (from model)
        expected_no_wall = TX_POWER_DBM - 10.0 * PATH_LOSS_N * math.log10(max(dist_m, 0.3))
        diff = expected_no_wall - tw["rssi_mean"]
        if tw.get("n_walls", 1) > 0:
            attenuations.append(diff / tw["n_walls"])

    if not attenuations:
        return {"wall_attenuation_db": WALL_ATTENUATION_DB,
                "error": "No valid through-wall measurements"}

    return {
        "wall_attenuation_db": round(float(np.mean(attenuations)), 1),
        "std": round(float(np.std(attenuations)), 1),
        "n_samples": len(attenuations),
    }


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_calibration(args):
    """Walk user through calibration point by point."""
    anchors = load_anchors()
    dwell = args.dwell

    if args.anchor:
        target_anchors = [args.anchor]
    else:
        target_anchors = sorted(anchors.keys())
        print("Available anchors:")
        for i, aid in enumerate(target_anchors, 1):
            a = anchors[aid]
            print(f"  {i}. {aid} — floor {a['floor']} at ({a['x']:.2f}, {a['y']:.2f})")
        sel = input("\nEnter anchor ID (or number): ").strip()
        if sel.isdigit():
            target_anchors = [target_anchors[int(sel) - 1]]
        else:
            target_anchors = [sel]

    anchor_id = target_anchors[0]
    if anchor_id not in anchors:
        print(f"Unknown anchor: {anchor_id}")
        return

    anchor = anchors[anchor_id]
    anchor_macs = {aid: anchors[aid]["mac"] for aid in anchors}

    print(f"\n{'='*60}")
    print(f"Calibrating against: {anchor_id}")
    print(f"  Position: ({anchor['x']:.2f}, {anchor['y']:.2f}) floor {anchor['floor']}")
    print(f"  Dwell time: {dwell}s per point")
    print(f"{'='*60}\n")

    # Parse distances
    if args.distances:
        distances = [float(d) for d in args.distances.split(",")]
    else:
        print("Enter distances in feet from the anchor (comma-separated).")
        print("Recommended: at least 3 points, e.g. 3,6,10,15")
        dist_input = input("Distances (ft): ").strip()
        distances = [float(d) for d in dist_input.split(",")]

    print(f"\nWill measure at {len(distances)} distances: {distances} ft")
    print("Ensure clear line-of-sight to the anchor (same room).\n")

    # Connect MQTT
    beacon_id = args.beacon
    print(f"Beacon device ID: {beacon_id}")
    collector = RSSICollector(anchor_macs, beacon_id)
    try:
        collector.connect()
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        print("Ensure mosquitto is running (docker compose up mosquitto)")
        return

    measurements = []
    try:
        for dist_ft in distances:
            print(f"--- Point: {dist_ft} ft from {anchor_id} ---")
            input(f"  Stand {dist_ft} ft from the anchor and press ENTER...")
            print(f"  Collecting for {dwell}s...", end="", flush=True)

            result = collector.collect(dwell)
            print(" done.")

            if anchor_id in result:
                r = result[anchor_id]
                print(f"  {anchor_id}: mean={r['mean']:.1f} dBm, "
                      f"std={r['std']:.1f}, n={r['n']}")
                measurements.append({
                    "anchor_id": anchor_id,
                    "distance_ft": dist_ft,
                    "rssi_mean": r["mean"],
                    "rssi_std": r["std"],
                    "n_samples": r["n"],
                })
            else:
                print(f"  WARNING: No readings from {anchor_id}!")
                print(f"  Received from: {list(result.keys()) or 'none'}")

            # Show all anchors for reference
            for aid, r in sorted(result.items()):
                if aid != anchor_id:
                    print(f"    (also: {aid}: {r['mean']:.1f} dBm, n={r['n']})")
            print()

    finally:
        collector.disconnect()

    if not measurements:
        print("No measurements collected. Check MQTT connectivity.")
        return

    # Fit
    print(f"\n{'='*60}")
    print("FITTING RESULTS")
    print(f"{'='*60}")

    fit = fit_path_loss(measurements)
    print(f"\nFitted TX_POWER_DBM : {fit['tx_power']} dBm")
    print(f"Fitted PATH_LOSS_N  : {fit['path_loss_n']}")
    print(f"RMSE                : {fit.get('rmse', 'N/A')} dBm")

    print(f"\nCurrent constants:")
    print(f"  TX_POWER_DBM = {TX_POWER_DBM}")
    print(f"  PATH_LOSS_N  = {PATH_LOSS_N}")

    print(f"\nPer-point details:")
    print(f"  {'Dist(ft)':>8}  {'RSSI(obs)':>10}  {'RSSI(fit)':>10}  {'Residual':>10}")
    for i, m in enumerate(measurements):
        d_m = m["distance_ft"] * FT_TO_M
        fitted_rssi = fit["tx_power"] - 10.0 * fit["path_loss_n"] * math.log10(d_m)
        resid = fit["residuals"][i] if "residuals" in fit else "?"
        print(f"  {m['distance_ft']:8.1f}  {m['rssi_mean']:10.1f}  {fitted_rssi:10.1f}  {resid:>10}")

    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        os.path.dirname(__file__), "..", "scratch",
        f"calibration_{anchor_id}_{timestamp}.json"
    )
    cal_data = {
        "anchor_id": anchor_id,
        "anchor_position": {"x": anchor["x"], "y": anchor["y"], "floor": anchor["floor"]},
        "dwell_seconds": dwell,
        "measurements": measurements,
        "fit": fit,
        "current_constants": {
            "TX_POWER_DBM": TX_POWER_DBM,
            "PATH_LOSS_N": PATH_LOSS_N,
            "WALL_ATTENUATION_DB": WALL_ATTENUATION_DB,
        },
        "timestamp": timestamp,
    }
    with open(outfile, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"\nRaw data saved to: {outfile}")

    # Suggest update
    if "error" not in fit:
        print(f"\nTo apply these constants, update constants.py:")
        print(f"  TX_POWER_DBM = {fit['tx_power']}")
        print(f"  PATH_LOSS_N  = {fit['path_loss_n']}")


def wall_test(args):
    """Measure wall attenuation: same-room vs through-wall at similar distances."""
    anchors = load_anchors()
    dwell = args.dwell

    anchor_id = args.anchor
    if not anchor_id:
        print("--anchor required for wall test")
        return
    if anchor_id not in anchors:
        print(f"Unknown anchor: {anchor_id}")
        return

    anchor = anchors[anchor_id]
    anchor_macs = {aid: anchors[aid]["mac"] for aid in anchors}

    print(f"\n{'='*60}")
    print(f"WALL ATTENUATION TEST — {anchor_id}")
    print(f"  Position: ({anchor['x']:.2f}, {anchor['y']:.2f}) floor {anchor['floor']}")
    print(f"{'='*60}\n")

    beacon_id = args.beacon
    print(f"Beacon device ID: {beacon_id}")
    collector = RSSICollector(anchor_macs, beacon_id)
    try:
        collector.connect()
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        return

    same_room = []
    through_wall = []

    try:
        # Same-room measurements
        print("PHASE 1: Same-room measurements (clear line of sight)")
        n_same = int(input("  How many same-room positions? ").strip() or "2")
        for i in range(n_same):
            dist = float(input(f"  Distance (ft) for position {i+1}: ").strip())
            input("  Position yourself and press ENTER...")
            print(f"  Collecting for {dwell}s...", end="", flush=True)
            result = collector.collect(dwell)
            print(" done.")
            if anchor_id in result:
                r = result[anchor_id]
                print(f"  {anchor_id}: mean={r['mean']:.1f} dBm, n={r['n']}")
                same_room.append({
                    "distance_ft": dist,
                    "rssi_mean": r["mean"],
                    "n_walls": 0,
                })
            else:
                print(f"  WARNING: No readings from {anchor_id}")
            print()

        # Through-wall measurements
        print("\nPHASE 2: Through-wall measurements")
        n_wall = int(input("  How many through-wall positions? ").strip() or "2")
        for i in range(n_wall):
            dist = float(input(f"  Distance (ft) for position {i+1}: ").strip())
            walls = int(input(f"  Number of walls between you and anchor: ").strip() or "1")
            input("  Position yourself and press ENTER...")
            print(f"  Collecting for {dwell}s...", end="", flush=True)
            result = collector.collect(dwell)
            print(" done.")
            if anchor_id in result:
                r = result[anchor_id]
                print(f"  {anchor_id}: mean={r['mean']:.1f} dBm, n={r['n']}")
                through_wall.append({
                    "distance_ft": dist,
                    "rssi_mean": r["mean"],
                    "n_walls": walls,
                })
            else:
                print(f"  WARNING: No readings from {anchor_id}")
            print()

    finally:
        collector.disconnect()

    # Estimate
    print(f"\n{'='*60}")
    print("WALL ATTENUATION RESULTS")
    print(f"{'='*60}")

    wa = estimate_wall_attenuation(same_room, through_wall)
    print(f"\nEstimated WALL_ATTENUATION_DB: {wa['wall_attenuation_db']} dBm")
    if "std" in wa:
        print(f"  Std dev: {wa['std']} dBm  (n={wa['n_samples']})")
    print(f"\nCurrent: WALL_ATTENUATION_DB = {WALL_ATTENUATION_DB}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        os.path.dirname(__file__), "..", "scratch",
        f"wall_attenuation_{anchor_id}_{timestamp}.json"
    )
    with open(outfile, "w") as f:
        json.dump({
            "anchor_id": anchor_id,
            "same_room": same_room,
            "through_wall": through_wall,
            "result": wa,
            "timestamp": timestamp,
        }, f, indent=2)
    print(f"Data saved to: {outfile}")


# ---------------------------------------------------------------------------
# Offline mode — fit from previously saved calibration files
# ---------------------------------------------------------------------------

def offline_fit(args):
    """Re-fit constants from saved calibration JSON files."""
    import glob

    pattern = os.path.join(
        os.path.dirname(__file__), "..", "scratch", "calibration_*.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print("No calibration files found in scratch/")
        return

    print(f"Found {len(files)} calibration file(s):")
    all_measurements = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        n = len(data.get("measurements", []))
        print(f"  {os.path.basename(f)}: {data['anchor_id']}, {n} points")
        all_measurements.extend(data["measurements"])

    if not all_measurements:
        print("No measurements found.")
        return

    fit = fit_path_loss(all_measurements)
    print(f"\nCombined fit ({fit['n_points']} points):")
    print(f"  TX_POWER_DBM = {fit['tx_power']}")
    print(f"  PATH_LOSS_N  = {fit['path_loss_n']}")
    print(f"  RMSE         = {fit.get('rmse', 'N/A')} dBm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RSSI Path-Loss Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive calibration against 1F_Office anchor
  python scripts/calibrate_rssi.py --anchor 1F_Office --beacon dog_collar --distances 3,6,10 --dwell 30

  # Wall attenuation test
  python scripts/calibrate_rssi.py --anchor 1F_Office --beacon dog_collar --wall-test --dwell 30

  # Re-fit from saved calibration data
  python scripts/calibrate_rssi.py --offline-fit
        """,
    )
    parser.add_argument("--anchor", type=str, help="Anchor ID to calibrate against")
    parser.add_argument("--beacon", type=str, default=DEFAULT_BEACON_ID,
                        help=f"ESPresense beacon/device ID (default: {DEFAULT_BEACON_ID})")
    parser.add_argument("--distances", type=str,
                        help="Comma-separated distances in feet (e.g. 3,6,10,15)")
    parser.add_argument("--dwell", type=int, default=DEFAULT_DWELL,
                        help=f"Seconds to collect at each point (default: {DEFAULT_DWELL})")
    parser.add_argument("--wall-test", action="store_true",
                        help="Run wall attenuation measurement mode")
    parser.add_argument("--offline-fit", action="store_true",
                        help="Re-fit constants from saved calibration JSON files")

    args = parser.parse_args()

    if args.offline_fit:
        offline_fit(args)
    elif args.wall_test:
        wall_test(args)
    else:
        interactive_calibration(args)


if __name__ == "__main__":
    main()
