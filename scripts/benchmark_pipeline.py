#!/usr/bin/env python3
"""
Pipeline Benchmark – Step-Level Timing for Bayesian Pet Localization

Runs the full inference pipeline locally and measures every stage in ms.
Supports two modes:

  **Simulated** (default): Generates synthetic RSSI data — no Docker required.
  **Live** (``--live``):    Subscribes to the real MQTT broker, collects RSSI
                           from physical anchors, and benchmarks the pipeline
                           on actual BLE data.  Requires Docker services
                           (Mosquitto at minimum) to be running.

Timed stages
────────────
 0. MQTT message receive     – (live only) broker → client delivery
 1. RSSI generation/collect   – simulate path-loss / parse MQTT payload
 2. Kalman filter update      – per-anchor 1-D Kalman bank
 3. Feature engineering       – ~40 derived features from smoothed RSSI
 4. Particle filter step      – broken into sub-steps:
      4a. Floor HMM step      – emission × forward algorithm
      4b. Teleport check      – HMM vs particle majority floor
      4c. Predict             – motion model + wall constraints
      4d. Update (likelihood) – log-distance path-loss weighting
      4e. Resample            – systematic resampling (if triggered)
      4f. Estimate            – weighted mean position + confidence
 5. Room labeling             – polygon lookup + zone gate check
 6. Activity classification   – threshold on activity_score
 7. Zone classifier (RF)      – optional (skipped if no model)
 8. Full inference cycle       – end-to-end (stages 1–7)

Usage:
    # Simulated (no Docker needed)
    python scripts/benchmark_pipeline.py
    python scripts/benchmark_pipeline.py --cycles 500
    python scripts/benchmark_pipeline.py --particles 1000 --cycles 200
    python scripts/benchmark_pipeline.py --mode walking

    # Live (requires Mosquitto + beacon active)
    python scripts/benchmark_pipeline.py --live
    python scripts/benchmark_pipeline.py --live --cycles 100 --beacon-id dog_collar
    python scripts/benchmark_pipeline.py --live --mqtt-host 192.168.1.100
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
# Resolve project root and add service paths for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "services" / "inference"))

from filters.kalman import KalmanFilterBank
from filters.particle import (
    ParticleFilter,
    extract_stairways,
    extract_stair_runs,
    extract_staircase_bounds,
)
from filters.floor_hmm import FloorTransitionHMM
from occupancy import OccupancyGridSet, bounds_to_polygon, _point_in_polygon
from features import FeatureEngine

try:
    from models.classifier import ZoneClassifier
except ImportError:
    ZoneClassifier = None

# ---------------------------------------------------------------------------
# Simulated beacon waypoints (same as simulate_mqtt.py)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    {"name": "office_bed",   "floor": 1, "x": 3.75, "y": 6.5,   "dwell_cycles": 20},
    {"name": "1f_hallway",   "floor": 1, "x": 7.0,  "y": 2.0,   "dwell_cycles": 5},
    {"name": "couch",        "floor": 2, "x": 10.0, "y": 12.0,  "dwell_cycles": 30},
    {"name": "water_bowl",   "floor": 2, "x": 14.0, "y": 26.0,  "dwell_cycles": 6},
    {"name": "kitchen",      "floor": 2, "x": 12.0, "y": 25.0,  "dwell_cycles": 10},
    {"name": "master_bed",   "floor": 3, "x": 12.0, "y": 8.0,   "dwell_cycles": 30},
    {"name": "3f_hallway",   "floor": 3, "x": 7.0,  "y": 18.0,  "dwell_cycles": 5},
    {"name": "3f_office",    "floor": 3, "x": 12.0, "y": 24.0,  "dwell_cycles": 15},
]

# Path-loss simulation constants (mirrors simulate_mqtt.py)
SIM_TX_POWER = -59.0
SIM_PATH_LOSS_N = 2.7
SIM_NOISE_STD = 4.0
SIM_CROSS_FLOOR_PENALTY_FT = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_layout():
    layout_path = PROJECT_ROOT / "config" / "floorplan" / "layout.json"
    with open(layout_path) as f:
        return json.load(f)


def _extract_anchors(layout):
    anchors = {}
    for floor_data in layout.get("floors", []):
        floor_num = floor_data["floor"]
        for a in floor_data.get("anchors", []):
            pos = a.get("position", [a.get("x", 0), a.get("y", 0)])
            aid = a["id"].lower()
            anchors[aid] = {
                "x": pos[0],
                "y": pos[1],
                "floor": floor_num,
                "height_ft": a.get("height_ft", 0.0),
            }
    return anchors


def _build_room_polygons(layout):
    room_polygons = {}
    room_gates = {}
    for floor_data in layout.get("floors", []):
        floor_num = floor_data["floor"]
        room_polygons[floor_num] = [
            (room["name"], bounds_to_polygon(room["bounds"]))
            for room in floor_data.get("rooms", [])
        ]
        for room in floor_data.get("rooms", []):
            gates = room.get("gates", [])
            if not gates:
                continue
            parsed = []
            for gate in gates:
                parsed.append({
                    "axis": gate["axis"],
                    "coord": gate["coord"],
                    "above": gate["between"][0],
                    "below": gate["between"][1],
                })
            room_gates.setdefault(floor_num, {})[room["name"]] = parsed
    return room_polygons, room_gates


def _label_room(x, y, floor, room_polygons, room_gates):
    for name, poly in room_polygons.get(floor, []):
        if _point_in_polygon(x, y, poly):
            gates = room_gates.get(floor, {}).get(name, [])
            for gate in gates:
                val = y if gate["axis"] == "y" else x
                if val >= gate["coord"]:
                    return gate["above"]
                else:
                    return gate["below"]
            return name
    return "unknown"


def _simulate_rssi(beacon_x, beacon_y, beacon_floor, anchors, rng):
    """Generate synthetic RSSI readings for all anchors."""
    rssi = {}
    for aid, apos in anchors.items():
        dx = beacon_x - apos["x"]
        dy = beacon_y - apos["y"]
        dist = math.sqrt(dx * dx + dy * dy)
        floor_diff = abs(beacon_floor - apos["floor"])
        if floor_diff > 0:
            dist += floor_diff * SIM_CROSS_FLOOR_PENALTY_FT
        dist = max(dist, 0.3)
        dist_m = dist * 0.3048
        raw = SIM_TX_POWER - 10 * SIM_PATH_LOSS_N * math.log10(dist_m)
        raw += rng.gauss(0, SIM_NOISE_STD)
        rssi[aid] = round(max(min(raw, -20), -100), 1)
    return rssi


class Timer:
    """Nanosecond-precision context-manager timer."""
    __slots__ = ("_start", "elapsed_ns")

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *_):
        self.elapsed_ns = time.perf_counter_ns() - self._start

    @property
    def ms(self):
        return self.elapsed_ns / 1_000_000


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_cycles: int, n_particles: int, mode: str, dt: float):
    layout = _load_layout()
    anchors = _extract_anchors(layout)
    room_polygons, room_gates = _build_room_polygons(layout)

    # --- Pipeline component initialisation ---
    occ_grids = OccupancyGridSet.from_layout_data(layout)
    kalman_bank = KalmanFilterBank(process_noise=0.5, measurement_noise=1.0, stale_timeout=30.0)
    floor_hmm = FloorTransitionHMM(layout, anchors)
    stairways = extract_stairways(layout)
    stair_runs = extract_stair_runs(layout)
    staircase_bounds = extract_staircase_bounds(layout)

    pf = ParticleFilter(
        occupancy_grids=occ_grids,
        anchor_positions=anchors,
        stairways=stairways,
        stair_runs=stair_runs,
        staircase_bounds=staircase_bounds,
        floor_hmm=floor_hmm,
        n_particles=n_particles,
        seed=42,
    )
    pf.initialise_uniform()

    feature_engine = FeatureEngine(anchor_positions=anchors, window_size=10)

    # Try to load classifier (will be None if not trained)
    zone_classifier = None
    if ZoneClassifier is not None:
        model_dir = PROJECT_ROOT / "models"
        joblib_files = sorted(model_dir.glob("random_forest_v*.joblib"))
        if joblib_files:
            try:
                zone_classifier = ZoneClassifier.from_file(str(joblib_files[-1]))
            except Exception:
                pass

    # --- Timing accumulators ---
    timings = defaultdict(list)

    # --- Waypoint / beacon position logic ---
    rng = random.Random(123)
    wp_idx = 0
    dwell_left = WAYPOINTS[0]["dwell_cycles"]
    beacon = WAYPOINTS[0]

    print("=" * 72)
    print("  Pipeline Benchmark — Bayesian Pet Localization")
    print("=" * 72)
    print(f"  Cycles:        {n_cycles}")
    print(f"  Particles:     {n_particles}")
    print(f"  Mode:          {mode}")
    print(f"  Simulated dt:  {dt:.3f}s")
    print(f"  Anchors:       {len(anchors)}")
    print(f"  Classifier:    {'loaded' if zone_classifier else 'not available'}")
    print("=" * 72)
    print()

    sim_time = time.time()

    for cycle in range(n_cycles):
        # === Advance beacon position (walking mode) ===
        if mode == "walking":
            dwell_left -= 1
            if dwell_left <= 0:
                wp_idx = (wp_idx + 1) % len(WAYPOINTS)
                beacon = WAYPOINTS[wp_idx]
                dwell_left = beacon["dwell_cycles"]
        else:
            beacon = WAYPOINTS[0]

        jitter_x = rng.gauss(0, 0.3)
        jitter_y = rng.gauss(0, 0.3)
        bx = beacon["x"] + jitter_x
        by = beacon["y"] + jitter_y
        bf = beacon["floor"]

        # ── FULL CYCLE TIMER ─────────────────────────────────────────
        cycle_start = time.perf_counter_ns()

        # ── 1. RSSI generation (simulates beacon TX → anchor RX) ────
        with Timer() as t_rssi_gen:
            rssi_vector = _simulate_rssi(bx, by, bf, anchors, rng)
        timings["1_rssi_generation"].append(t_rssi_gen.ms)

        # ── 2. Kalman filter update ─────────────────────────────────
        sim_time += dt
        with Timer() as t_kalman:
            for anchor_id, raw_rssi in rssi_vector.items():
                kalman_bank.update(anchor_id, raw_rssi, sim_time)
            smoothed_rssi = kalman_bank.get_smoothed_rssi(current_time=sim_time)
        timings["2_kalman_filter"].append(t_kalman.ms)

        # ── 3. Feature engineering ──────────────────────────────────
        with Timer() as t_features:
            computed_features = feature_engine.update(rssi_vector, smoothed_rssi, sim_time)
        timings["3_feature_engineering"].append(t_features.ms)

        # ── 4. Particle filter (broken into sub-steps) ──────────────
        # We call the sub-methods individually instead of pf.step()
        # so we can time each one.
        pf._last_rssi = smoothed_rssi

        # 4a. Floor HMM step
        with Timer() as t_hmm:
            if pf._floor_hmm is not None and pf._initialised:
                est = pf.estimate
                proximity = pf._floor_hmm.stair_proximity_for_position(
                    est["x"], est["y"], est["floor"]
                )
                pf._floor_hmm.step(smoothed_rssi, dt, proximity)
        timings["4a_floor_hmm"].append(t_hmm.ms)

        # 4b. Teleport check
        with Timer() as t_teleport:
            if pf._floor_hmm is not None and pf._initialised:
                pf._maybe_teleport(dt)
        timings["4b_teleport_check"].append(t_teleport.ms)

        # 4c. Predict (motion model)
        with Timer() as t_predict:
            pf.predict(dt)
        timings["4c_predict_motion"].append(t_predict.ms)

        # 4d. Update (observation / likelihood)
        with Timer() as t_update:
            pf.update(smoothed_rssi)
        timings["4d_update_likelihood"].append(t_update.ms)

        # 4e. Resample
        with Timer() as t_resample:
            did_resample = pf.resample_if_needed()
        timings["4e_resample"].append(t_resample.ms)

        # 4f. Estimate
        with Timer() as t_estimate:
            estimate = pf.estimate
        timings["4f_estimate"].append(t_estimate.ms)

        # Total particle filter time
        pf_total = (t_hmm.ms + t_teleport.ms + t_predict.ms
                    + t_update.ms + t_resample.ms + t_estimate.ms)
        timings["4_particle_filter_total"].append(pf_total)

        # ── 5. Room labeling ────────────────────────────────────────
        with Timer() as t_room:
            room_label = _label_room(
                estimate["x"], estimate["y"], estimate["floor"],
                room_polygons, room_gates,
            )
        timings["5_room_labeling"].append(t_room.ms)

        # ── 6. Activity classification ──────────────────────────────
        with Timer() as t_activity:
            activity_score = computed_features.get("activity_score", 0.0)
            if activity_score < 0.15:
                activity_label = "sleeping"
            elif activity_score < 0.45:
                activity_label = "stationary"
            else:
                activity_label = "moving"
        timings["6_activity_classification"].append(t_activity.ms)

        # ── 7. Zone classifier (optional) ───────────────────────────
        with Timer() as t_classifier:
            zone_label = None
            if zone_classifier is not None and zone_classifier.is_trained:
                try:
                    zone_label, _, _ = zone_classifier.predict_for_room(
                        computed_features,
                        smoothed_rssi=smoothed_rssi,
                        room=room_label,
                    )
                except Exception:
                    pass
        timings["7_zone_classifier"].append(t_classifier.ms)

        # ── End of cycle ─────────────────────────────────────────────
        cycle_elapsed = (time.perf_counter_ns() - cycle_start) / 1_000_000
        timings["8_full_inference_cycle"].append(cycle_elapsed)

        # Progress indicator
        if (cycle + 1) % max(1, n_cycles // 10) == 0 or cycle == 0:
            pct = (cycle + 1) / n_cycles * 100
            print(
                f"  [{pct:5.1f}%]  cycle {cycle+1:>5}/{n_cycles}  "
                f"| pos=({estimate['x']:.1f}, {estimate['y']:.1f}) F{estimate['floor']}  "
                f"| room={room_label:<18s}  "
                f"| cycle={cycle_elapsed:.2f}ms"
            )

    # --- Report ---
    print()
    _print_report(timings, n_cycles, n_particles, did_resample, source="simulated")



# ---------------------------------------------------------------------------
# Live (MQTT) benchmark runner
# ---------------------------------------------------------------------------

def run_live_benchmark(
    n_cycles: int,
    n_particles: int,
    mqtt_host: str,
    mqtt_port: int,
    beacon_id: str,
):
    """Subscribe to live MQTT and benchmark the pipeline on real RSSI data."""
    layout = _load_layout()
    anchors = _extract_anchors(layout)
    room_polygons, room_gates = _build_room_polygons(layout)

    # --- Pipeline component initialisation (same as simulated) ---
    occ_grids = OccupancyGridSet.from_layout_data(layout)
    kalman_bank = KalmanFilterBank(process_noise=0.5, measurement_noise=1.0, stale_timeout=30.0)
    floor_hmm = FloorTransitionHMM(layout, anchors)
    stairways = extract_stairways(layout)
    stair_runs = extract_stair_runs(layout)
    staircase_bounds = extract_staircase_bounds(layout)

    pf = ParticleFilter(
        occupancy_grids=occ_grids,
        anchor_positions=anchors,
        stairways=stairways,
        stair_runs=stair_runs,
        staircase_bounds=staircase_bounds,
        floor_hmm=floor_hmm,
        n_particles=n_particles,
        seed=42,
    )
    pf.initialise_uniform()

    feature_engine = FeatureEngine(anchor_positions=anchors, window_size=10)

    zone_classifier = None
    if ZoneClassifier is not None:
        model_dir = PROJECT_ROOT / "models"
        joblib_files = sorted(model_dir.glob("random_forest_v*.joblib"))
        if joblib_files:
            try:
                zone_classifier = ZoneClassifier.from_file(str(joblib_files[-1]))
            except Exception:
                pass

    # --- Shared state between MQTT thread and benchmark loop ---
    rssi_buffer: dict[str, dict] = {}   # anchor_id -> latest reading
    buffer_lock = threading.Lock()
    new_data_event = threading.Event()
    mqtt_connected_event = threading.Event()
    mqtt_msg_timestamps: deque = deque(maxlen=5000)
    mqtt_delivery_latencies: list = []

    # --- Timing accumulators ---
    timings: dict[str, list] = defaultdict(list)

    # --- MQTT callbacks ---
    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe(f"espresense/devices/{beacon_id}/#")
        client.subscribe("espresense/devices/#")
        mqtt_connected_event.set()

    def on_message(client, userdata, msg):
        recv_ns = time.perf_counter_ns()
        try:
            topic_parts = msg.topic.split("/")
            if len(topic_parts) < 4 or topic_parts[0] != "espresense":
                return
            device_id = topic_parts[2]
            anchor_id = topic_parts[3] if len(topic_parts) > 3 else "unknown"

            raw = msg.payload.decode().strip()
            if not raw:
                return
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return

            mqtt_msg_timestamps.append(recv_ns)

            if device_id != beacon_id:
                return

            rssi_val = payload.get("rssi", -100)
            distance_val = payload.get("distance", 0)

            with buffer_lock:
                rssi_buffer[anchor_id] = {
                    "rssi": rssi_val,
                    "distance": distance_val,
                    "recv_ns": recv_ns,
                }
            new_data_event.set()
        except Exception:
            pass

    # --- Connect to MQTT ---
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    print("=" * 72)
    print("  Pipeline Benchmark — LIVE MQTT Mode")
    print("=" * 72)
    print(f"  MQTT Broker:   {mqtt_host}:{mqtt_port}")
    print(f"  Beacon ID:     {beacon_id}")
    print(f"  Cycles:        {n_cycles}")
    print(f"  Particles:     {n_particles}")
    print(f"  Anchors:       {len(anchors)}")
    print(f"  Classifier:    {'loaded' if zone_classifier else 'not available'}")
    print("=" * 72)
    print()

    try:
        client.connect(mqtt_host, mqtt_port, 60)
    except Exception as e:
        print(f"ERROR: Cannot connect to MQTT broker at {mqtt_host}:{mqtt_port}")
        print(f"  {e}")
        print(f"\nMake sure Mosquitto is running:")
        print(f"  docker compose up -d mosquitto")
        return

    client.loop_start()

    if not mqtt_connected_event.wait(timeout=5):
        print("ERROR: MQTT connection timed out after 5s")
        client.loop_stop()
        client.disconnect()
        return

    print("  Connected to MQTT broker ✓")
    print(f"  Waiting for beacon '{beacon_id}' messages...")
    print()

    if not new_data_event.wait(timeout=30):
        print("ERROR: No beacon messages received within 30s")
        print(f"  Is the beacon '{beacon_id}' active and in range of anchors?")
        client.loop_stop()
        client.disconnect()
        return

    print("  Beacon detected ✓ — starting benchmark\n")

    last_inference_time = None
    did_resample = False
    MIN_INFERENCE_INTERVAL = 0.25

    for cycle in range(n_cycles):
        new_data_event.clear()
        if not new_data_event.wait(timeout=5):
            print(f"  WARNING: No new data for 5s at cycle {cycle+1}, stopping early")
            break

        # ── FULL CYCLE TIMER ─────────────────────────────────────────
        cycle_start = time.perf_counter_ns()

        # ── 0. MQTT delivery timing ─────────────────────────────────
        with buffer_lock:
            snapshot = dict(rssi_buffer)
        delivery_latencies = []
        for anchor_id, data in snapshot.items():
            lat = (cycle_start - data["recv_ns"]) / 1_000_000
            delivery_latencies.append(lat)
        if delivery_latencies:
            timings["0_mqtt_delivery"].append(statistics.mean(delivery_latencies))
            mqtt_delivery_latencies.extend(delivery_latencies)
        else:
            timings["0_mqtt_delivery"].append(0.0)

        # ── 1. RSSI collection (parse from buffer) ──────────────────
        with Timer() as t_rssi_gen:
            rssi_vector = {aid: data["rssi"] for aid, data in snapshot.items()}
        timings["1_rssi_collection"].append(t_rssi_gen.ms)

        if not rssi_vector:
            continue

        now = time.time()
        if last_inference_time is not None:
            dt = min(now - last_inference_time, 10.0)
        else:
            dt = 0.5
        last_inference_time = now

        # ── 2. Kalman filter update ─────────────────────────────────
        with Timer() as t_kalman:
            for anchor_id, raw_rssi in rssi_vector.items():
                kalman_bank.update(anchor_id, raw_rssi, now)
            smoothed_rssi = kalman_bank.get_smoothed_rssi(current_time=now)
        timings["2_kalman_filter"].append(t_kalman.ms)

        # ── 3. Feature engineering ──────────────────────────────────
        with Timer() as t_features:
            computed_features = feature_engine.update(rssi_vector, smoothed_rssi, now)
        timings["3_feature_engineering"].append(t_features.ms)

        # ── 4. Particle filter sub-steps ────────────────────────────
        pf._last_rssi = smoothed_rssi

        with Timer() as t_hmm:
            if pf._floor_hmm is not None and pf._initialised:
                est = pf.estimate
                proximity = pf._floor_hmm.stair_proximity_for_position(
                    est["x"], est["y"], est["floor"]
                )
                pf._floor_hmm.step(smoothed_rssi, dt, proximity)
        timings["4a_floor_hmm"].append(t_hmm.ms)

        with Timer() as t_teleport:
            if pf._floor_hmm is not None and pf._initialised:
                pf._maybe_teleport(dt)
        timings["4b_teleport_check"].append(t_teleport.ms)

        with Timer() as t_predict:
            pf.predict(dt)
        timings["4c_predict_motion"].append(t_predict.ms)

        with Timer() as t_update:
            pf.update(smoothed_rssi)
        timings["4d_update_likelihood"].append(t_update.ms)

        with Timer() as t_resample:
            did_resample = pf.resample_if_needed()
        timings["4e_resample"].append(t_resample.ms)

        with Timer() as t_estimate:
            estimate = pf.estimate
        timings["4f_estimate"].append(t_estimate.ms)

        pf_total = (t_hmm.ms + t_teleport.ms + t_predict.ms
                    + t_update.ms + t_resample.ms + t_estimate.ms)
        timings["4_particle_filter_total"].append(pf_total)

        # ── 5. Room labeling ────────────────────────────────────────
        with Timer() as t_room:
            room_label = _label_room(
                estimate["x"], estimate["y"], estimate["floor"],
                room_polygons, room_gates,
            )
        timings["5_room_labeling"].append(t_room.ms)

        # ── 6. Activity classification ──────────────────────────────
        with Timer() as t_activity:
            activity_score = computed_features.get("activity_score", 0.0)
            if activity_score < 0.15:
                activity_label = "sleeping"
            elif activity_score < 0.45:
                activity_label = "stationary"
            else:
                activity_label = "moving"
        timings["6_activity_classification"].append(t_activity.ms)

        # ── 7. Zone classifier (optional) ───────────────────────────
        with Timer() as t_classifier:
            zone_label = None
            if zone_classifier is not None and zone_classifier.is_trained:
                try:
                    zone_label, _, _ = zone_classifier.predict_for_room(
                        computed_features,
                        smoothed_rssi=smoothed_rssi,
                        room=room_label,
                    )
                except Exception:
                    pass
        timings["7_zone_classifier"].append(t_classifier.ms)

        # ── End of cycle ─────────────────────────────────────────────
        cycle_elapsed = (time.perf_counter_ns() - cycle_start) / 1_000_000
        timings["8_full_inference_cycle"].append(cycle_elapsed)

        n_anchors_active = len(rssi_vector)
        if (cycle + 1) % max(1, n_cycles // 10) == 0 or cycle == 0:
            pct = (cycle + 1) / n_cycles * 100
            print(
                f"  [{pct:5.1f}%]  cycle {cycle+1:>5}/{n_cycles}  "
                f"| anchors={n_anchors_active}  "
                f"| pos=({estimate['x']:.1f}, {estimate['y']:.1f}) F{estimate['floor']}  "
                f"| room={room_label:<18s}  "
                f"| dt={dt:.3f}s  "
                f"| cycle={cycle_elapsed:.2f}ms"
            )

        # Pace: don't spin faster than data arrives
        elapsed_s = (time.perf_counter_ns() - cycle_start) / 1e9
        if elapsed_s < MIN_INFERENCE_INTERVAL:
            time.sleep(MIN_INFERENCE_INTERVAL - elapsed_s)

    # Cleanup
    client.loop_stop()
    client.disconnect()

    actual_cycles = len(timings.get("8_full_inference_cycle", []))
    if actual_cycles == 0:
        print("\n  No inference cycles completed.")
        return

    # --- MQTT message rate stats ---
    print()
    ts_list = list(mqtt_msg_timestamps)
    if len(ts_list) >= 2:
        total_duration_s = (ts_list[-1] - ts_list[0]) / 1e9
        if total_duration_s > 0:
            msg_rate = (len(ts_list) - 1) / total_duration_s
            intervals_ms = [
                (ts_list[i+1] - ts_list[i]) / 1_000_000
                for i in range(len(ts_list) - 1)
            ]
            timings["0a_mqtt_msg_interval"] = intervals_ms
            print(f"  MQTT message rate:  {msg_rate:.1f} msg/s  "
                  f"({len(ts_list)} messages in {total_duration_s:.1f}s)")
            print(f"  Inter-message gap:  "
                  f"min={min(intervals_ms):.1f}ms  "
                  f"mean={statistics.mean(intervals_ms):.1f}ms  "
                  f"median={statistics.median(intervals_ms):.1f}ms  "
                  f"max={max(intervals_ms):.1f}ms")
    if mqtt_delivery_latencies:
        print(f"  MQTT → pipeline:    "
              f"min={min(mqtt_delivery_latencies):.2f}ms  "
              f"mean={statistics.mean(mqtt_delivery_latencies):.2f}ms  "
              f"max={max(mqtt_delivery_latencies):.2f}ms")

    # --- Report ---
    print()
    _print_report(timings, actual_cycles, n_particles, did_resample, source="live")


def _percentile(data, p):
    """Compute the p-th percentile (0–100) of a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (data[c] - data[f]) * (k - f)


def _print_report(timings, n_cycles, n_particles, last_resample, source="simulated"):
    """Print a formatted step-timing summary table."""
    is_live = source == "live"

    # Header labels — adapt stage 0/1 name based on source
    labels = {
        "0_mqtt_delivery":          "MQTT Delivery Latency",
        "0a_mqtt_msg_interval":     "MQTT Inter-Message Gap",
        "1_rssi_generation":        "RSSI Generation (sim TX→RX)",
        "1_rssi_collection":        "RSSI Collection (buffer read)",
        "2_kalman_filter":          "Kalman Filter Bank",
        "3_feature_engineering":    "Feature Engineering (~40 feats)",
        "4a_floor_hmm":             "  └ Floor HMM step",
        "4b_teleport_check":        "  └ Teleport check",
        "4c_predict_motion":        "  └ Predict (motion model)",
        "4d_update_likelihood":     "  └ Update (RSSI likelihood)",
        "4e_resample":              "  └ Resample (systematic)",
        "4f_estimate":              "  └ Estimate (weighted mean)",
        "4_particle_filter_total":  "Particle Filter (total)",
        "5_room_labeling":          "Room Labeling (polygon)",
        "6_activity_classification":"Activity Classification",
        "7_zone_classifier":        "Zone Classifier (RF)",
        "8_full_inference_cycle":   "FULL INFERENCE CYCLE",
    }

    # Ordered keys for display — adjust stage 0/1 based on source
    if is_live:
        pre_pf_keys = [
            "0_mqtt_delivery",
            "0a_mqtt_msg_interval",
            "1_rssi_collection",
            "2_kalman_filter",
        ]
    else:
        pre_pf_keys = [
            "1_rssi_generation",
            "2_kalman_filter",
        ]

    display_order = [
        *pre_pf_keys,
        "3_feature_engineering",
        None,  # separator before PF breakdown
        "4a_floor_hmm",
        "4b_teleport_check",
        "4c_predict_motion",
        "4d_update_likelihood",
        "4e_resample",
        "4f_estimate",
        None,  # separator after PF breakdown
        "4_particle_filter_total",
        "5_room_labeling",
        "6_activity_classification",
        "7_zone_classifier",
        None,  # separator before total
        "8_full_inference_cycle",
    ]

    col_w = 36  # label column width
    num_w = 10  # numeric column width

    sep_line = "─" * (col_w + 1 + num_w * 6 + 5 * 3 + 2)

    print("=" * len(sep_line))
    print(f"  PIPELINE STEP TIMING REPORT  ({source.upper()})")
    print(f"  {n_cycles} cycles  |  {n_particles} particles")
    print("=" * len(sep_line))
    print()

    # Table header
    hdr = (
        f"{'Stage':<{col_w}} │"
        f"{'Min':>{num_w}} "
        f"{'Mean':>{num_w}} "
        f"{'Median':>{num_w}} "
        f"{'P95':>{num_w}} "
        f"{'P99':>{num_w}} "
        f"{'Max':>{num_w}}"
    )
    print(hdr)
    print(sep_line)

    for key in display_order:
        if key is None:
            print(sep_line)
            continue

        data = sorted(timings.get(key, [0.0]))
        n = len(data)
        label = labels.get(key, key)

        mn = data[0]
        mx = data[-1]
        mean = statistics.mean(data)
        med = statistics.median(data)
        p95 = _percentile(data, 95)
        p99 = _percentile(data, 99)

        # Format: highlight the full-cycle row
        if key == "8_full_inference_cycle":
            fmt = (
                f"\033[1m{label:<{col_w}}\033[0m │"
                f"\033[1m{mn:>{num_w}.3f}\033[0m "
                f"\033[1m{mean:>{num_w}.3f}\033[0m "
                f"\033[1m{med:>{num_w}.3f}\033[0m "
                f"\033[1m{p95:>{num_w}.3f}\033[0m "
                f"\033[1m{p99:>{num_w}.3f}\033[0m "
                f"\033[1m{mx:>{num_w}.3f}\033[0m"
            )
        else:
            fmt = (
                f"{label:<{col_w}} │"
                f"{mn:>{num_w}.3f} "
                f"{mean:>{num_w}.3f} "
                f"{med:>{num_w}.3f} "
                f"{p95:>{num_w}.3f} "
                f"{p99:>{num_w}.3f} "
                f"{mx:>{num_w}.3f}"
            )
        print(fmt)

    print(sep_line)
    print()

    # Summary stats
    total_data = timings["8_full_inference_cycle"]
    total_time = sum(total_data)
    throughput = n_cycles / (total_time / 1000) if total_time > 0 else 0

    print(f"  Total wall time:    {total_time:.1f} ms  ({total_time/1000:.2f} s)")
    print(f"  Throughput:         {throughput:.1f} inferences/sec")
    print()

    # Breakdown: fraction of time per stage
    print("  Time budget (% of mean cycle):")
    cycle_mean = statistics.mean(total_data)
    if is_live:
        budget_keys = [
            "1_rssi_collection",
            "2_kalman_filter",
            "3_feature_engineering",
            "4_particle_filter_total",
            "5_room_labeling",
            "6_activity_classification",
            "7_zone_classifier",
        ]
    else:
        budget_keys = [
            "1_rssi_generation",
            "2_kalman_filter",
            "3_feature_engineering",
            "4_particle_filter_total",
            "5_room_labeling",
            "6_activity_classification",
            "7_zone_classifier",
        ]
    for key in budget_keys:
        data = timings.get(key, [0.0])
        stage_mean = statistics.mean(data)
        pct = (stage_mean / cycle_mean * 100) if cycle_mean > 0 else 0
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        label = labels.get(key, key)
        print(f"    {label:<36s}  {pct:5.1f}%  {bar}")

    print()

    # PF sub-step breakdown
    pf_mean = statistics.mean(timings["4_particle_filter_total"])
    if pf_mean > 0:
        print("  Particle filter sub-step breakdown (% of PF total):")
        pf_sub_keys = [
            "4a_floor_hmm",
            "4b_teleport_check",
            "4c_predict_motion",
            "4d_update_likelihood",
            "4e_resample",
            "4f_estimate",
        ]
        for key in pf_sub_keys:
            data = timings.get(key, [0.0])
            stage_mean = statistics.mean(data)
            pct = (stage_mean / pf_mean * 100) if pf_mean > 0 else 0
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            label = labels.get(key, key)
            print(f"    {label:<36s}  {pct:5.1f}%  {bar}")
        print()

    # Real-time feasibility check
    print("  Real-time feasibility:")
    cycle_p95 = _percentile(sorted(total_data), 95)
    intervals = [100, 250, 400, 500, 1000]
    for interval in intervals:
        headroom = interval - cycle_p95
        status = "✓" if headroom > 0 else "✗"
        print(f"    {status}  {interval:>5}ms interval  →  "
              f"headroom {headroom:+.1f}ms  (P95 cycle = {cycle_p95:.2f}ms)")
    print()
    print("  All times in milliseconds (ms).")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the inference pipeline step-by-step timing"
    )
    parser.add_argument(
        "--cycles", type=int, default=200,
        help="Number of inference cycles to run (default: 200)",
    )
    parser.add_argument(
        "--particles", type=int, default=500,
        help="Number of particles in the filter (default: 500)",
    )
    parser.add_argument(
        "--mode", choices=["static", "walking"], default="walking",
        help="Beacon movement mode for simulated mode (default: walking)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.4,
        help="Simulated dt between inference cycles in seconds (default: 0.4)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live MQTT data instead of simulated RSSI",
    )
    parser.add_argument(
        "--mqtt-host", default="localhost",
        help="MQTT broker host (default: localhost)",
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--beacon-id", default="dog_collar",
        help="Beacon device ID to track (default: dog_collar)",
    )
    args = parser.parse_args()

    if args.live:
        run_live_benchmark(
            n_cycles=args.cycles,
            n_particles=args.particles,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            beacon_id=args.beacon_id,
        )
    else:
        run_benchmark(
            n_cycles=args.cycles,
            n_particles=args.particles,
            mode=args.mode,
            dt=args.dt,
        )


if __name__ == "__main__":
    main()
