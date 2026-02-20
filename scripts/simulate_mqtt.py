#!/usr/bin/env python3
"""
MQTT Simulator for Bayesian Pet Localization

Simulates ESPresense-style MQTT messages so you can test the full pipeline
without any hardware. Publishes realistic RSSI readings from virtual anchors
for a virtual beacon moving between rooms.

Usage:
    python scripts/simulate_mqtt.py                          # defaults: localhost:1883
    python scripts/simulate_mqtt.py --host 192.168.1.100     # Pi IP
    python scripts/simulate_mqtt.py --beacon-id myBeacon     # custom beacon ID
    python scripts/simulate_mqtt.py --mode static            # beacon in fixed position
    python scripts/simulate_mqtt.py --mode walking           # beacon moves between rooms
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Anchor layout (matches config/floorplan/layout.json)
# ---------------------------------------------------------------------------

ANCHORS = {
    # Floor 1
    "office":       {"floor": 1, "x": 4.5, "y": 10.0},
    "hallway_f1":   {"floor": 1, "x": 7.0, "y": 2.0},
    # Floor 2
    "living_nw":    {"floor": 2, "x": 1.0, "y": 14.0},
    "living_sw":    {"floor": 2, "x": 1.0, "y": 1.0},
    "kitchen_ne":   {"floor": 2, "x": 29.0, "y": 14.0},
    "kitchen_se":   {"floor": 2, "x": 29.0, "y": 1.0},
    "powder_room":  {"floor": 2, "x": 28.0, "y": 13.0},
    # Floor 3
    "master_bedroom": {"floor": 3, "x": 7.0, "y": 5.0},
    "master_bath":    {"floor": 3, "x": 11.0, "y": 12.0},
    "wife_office":    {"floor": 3, "x": 5.0, "y": 18.0},
    "hallway_f3":     {"floor": 3, "x": 7.0, "y": 12.0},
}

# Points of interest the simulated beacon can visit
WAYPOINTS = [
    {"name": "office_bed",   "floor": 1, "x": 2.0, "y": 10.0, "dwell": 10},
    {"name": "couch",        "floor": 2, "x": 7.0, "y": 10.0, "dwell": 15},
    {"name": "water_bowl",   "floor": 2, "x": 20.0, "y": 12.0, "dwell": 3},
    {"name": "kitchen",      "floor": 2, "x": 22.0, "y": 7.0, "dwell": 5},
    {"name": "master_bed",   "floor": 3, "x": 7.0, "y": 5.0, "dwell": 20},
    {"name": "wife_office",  "floor": 3, "x": 5.0, "y": 18.0, "dwell": 8},
]


def rssi_from_distance(distance_ft: float, tx_power: float = -59.0, n: float = 2.7) -> float:
    """
    Simulate RSSI from distance using log-distance path loss model.
    
    RSSI = tx_power - 10 * n * log10(d)
    
    Added noise imitates real indoor BLE behavior.
    """
    if distance_ft < 0.3:
        distance_ft = 0.3  # clamp minimum
    # Convert feet to meters for the model
    distance_m = distance_ft * 0.3048
    rssi = tx_power - 10 * n * math.log10(distance_m)
    # Add Gaussian noise (σ ≈ 3-5 dBm is typical indoors)
    rssi += random.gauss(0, 4.0)
    return round(max(min(rssi, -20), -100), 1)


def distance_2d(x1, y1, x2, y2):
    """Euclidean distance in 2D."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def publish_readings(client, beacon_id: str, beacon_x: float, beacon_y: float, beacon_floor: int):
    """Publish simulated RSSI readings from all anchors for the beacon position."""
    for anchor_id, anchor in ANCHORS.items():
        if anchor["floor"] == beacon_floor:
            dist = distance_2d(beacon_x, beacon_y, anchor["x"], anchor["y"])
        else:
            # Cross-floor attenuation: add ~15-25 dB loss per floor
            floor_diff = abs(anchor["floor"] - beacon_floor)
            dist = distance_2d(beacon_x, beacon_y, anchor["x"], anchor["y"])
            dist += floor_diff * 30  # effective extra distance for floor penetration

        rssi = rssi_from_distance(dist)
        distance_m = dist * 0.3048  # approximate

        payload = {
            "id": beacon_id,
            "name": beacon_id,
            "rssi": rssi,
            "distance": round(distance_m, 2),
            "mac": "AA:BB:CC:DD:EE:FF",
        }

        topic = f"espresense/devices/{beacon_id}/{anchor_id}"
        client.publish(topic, json.dumps(payload))


def run_static(client, beacon_id: str, args):
    """Static mode: beacon stays in one place."""
    wp = WAYPOINTS[0]  # default: office bed
    if args.location:
        matching = [w for w in WAYPOINTS if w["name"] == args.location]
        if matching:
            wp = matching[0]
        else:
            print(f"Unknown location '{args.location}'. Available: {[w['name'] for w in WAYPOINTS]}")
            return

    print(f"Static mode: beacon at '{wp['name']}' (floor {wp['floor']}, x={wp['x']}, y={wp['y']})")
    print(f"Publishing every {args.interval}s — Ctrl+C to stop\n")

    count = 0
    while True:
        publish_readings(client, beacon_id, wp["x"], wp["y"], wp["floor"])
        count += 1
        if count % 10 == 0:
            print(f"  Published {count} rounds of readings")
        time.sleep(args.interval)


def run_walking(client, beacon_id: str, args):
    """Walking mode: beacon moves between waypoints."""
    print(f"Walking mode: cycling through {len(WAYPOINTS)} waypoints")
    print(f"Publishing every {args.interval}s — Ctrl+C to stop\n")

    count = 0
    wp_idx = 0
    dwell_remaining = 0

    while True:
        wp = WAYPOINTS[wp_idx]

        if dwell_remaining <= 0:
            print(f"  -> Moved to '{wp['name']}' (floor {wp['floor']})")
            dwell_remaining = wp["dwell"]

        # Add small position jitter (dog shifting around)
        jitter_x = random.gauss(0, 0.3)
        jitter_y = random.gauss(0, 0.3)

        publish_readings(
            client, beacon_id,
            wp["x"] + jitter_x, wp["y"] + jitter_y, wp["floor"]
        )

        count += 1
        dwell_remaining -= args.interval

        if dwell_remaining <= 0:
            wp_idx = (wp_idx + 1) % len(WAYPOINTS)

        if count % 20 == 0:
            print(f"  Published {count} rounds of readings")

        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Simulate ESPresense MQTT messages")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--beacon-id", default="dog_collar", help="Beacon device ID")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between publish rounds")
    parser.add_argument("--mode", choices=["static", "walking"], default="walking",
                        help="static = fixed position, walking = moves between rooms")
    parser.add_argument("--location", default=None,
                        help="For static mode: waypoint name (e.g. couch, water_bowl)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BLE Beacon MQTT Simulator")
    print("=" * 60)
    print(f"  Broker:    {args.host}:{args.port}")
    print(f"  Beacon ID: {args.beacon_id}")
    print(f"  Mode:      {args.mode}")
    print(f"  Interval:  {args.interval}s")
    print(f"  Anchors:   {len(ANCHORS)}")
    print("=" * 60)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(args.host, args.port, 60)
    except Exception as e:
        print(f"\nERROR: Cannot connect to MQTT broker at {args.host}:{args.port}")
        print(f"  {e}")
        print(f"\nMake sure Mosquitto is running:")
        print(f"  docker compose up -d mosquitto")
        return

    client.loop_start()
    print("\nConnected to MQTT broker ✓\n")

    try:
        if args.mode == "static":
            run_static(client, args.beacon_id, args)
        else:
            run_walking(client, args.beacon_id, args)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
