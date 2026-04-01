"""
Inference Service for Bayesian Pet Localization

Subscribes to MQTT for RSSI data, runs ML pipeline, serves position via FastAPI.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from pydantic import BaseModel

from filters.kalman import KalmanFilterBank
from filters.particle import (
    ParticleFilter, extract_stairways, extract_stair_runs, extract_staircase_bounds,
)
from filters.floor_hmm import FloorTransitionHMM
from occupancy import OccupancyGridSet, bounds_to_polygon, _point_in_polygon
from db import Database
from features import FeatureEngine

try:
    from models.classifier import ZoneClassifier
except ImportError:
    ZoneClassifier = None  # type: ignore[misc,assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
BEACON_ID = os.getenv("BEACON_ID", "dog_collar")

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.environ["INFLUXDB_TOKEN"]
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "pet-localization")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "rssi_raw")
FLOORPLAN_PATH = os.getenv("FLOORPLAN_PATH", "config/floorplan/layout.json")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB = os.getenv("POSTGRES_DB", "pet_tracking")

# -----------------------------------------------------------------------------
# Pipeline tuning constants
# -----------------------------------------------------------------------------

# Kalman filter bank
KALMAN_PROCESS_NOISE = 0.5        # process noise variance (trust measurements)
KALMAN_MEASUREMENT_NOISE = 1.0    # measurement noise variance (~0.8 dBm from data)
KALMAN_STALE_TIMEOUT = 30.0       # seconds before anchor is considered stale

# Particle filter
N_PARTICLES = 500                 # number of particles

# Feature engineering
FEATURE_WINDOW_SIZE = 10          # rolling window for per-anchor variance (samples)

# Activity classification thresholds (applied to activity_score 0–1)
ACTIVITY_SLEEPING_THRESHOLD = 0.15    # below → "sleeping"
ACTIVITY_STATIONARY_THRESHOLD = 0.45  # below → "stationary", above → "moving"

# Inference timing
DEFAULT_DT = 0.5                  # assumed dt (seconds) for first inference cycle
MAX_DT = 10.0                     # cap dt to avoid huge jumps after long gaps
INFERENCE_MIN_INTERVAL = 0.25     # minimum seconds between inference cycles
                                  # (dashboard can poll faster for diagnostics;
                                  #  this only gates how often particles move)

# In-memory ring buffer sizes
POSITION_HISTORY_MAXLEN = 500
RSSI_HISTORY_MAXLEN = 1000

# MQTT reconnect
MQTT_RECONNECT_DELAY = 5          # seconds between reconnection attempts
MQTT_KEEPALIVE = 60               # MQTT keepalive interval (seconds)

# -----------------------------------------------------------------------------
# Floor Plan / Anchor Coordinate Lookup
# -----------------------------------------------------------------------------

# Maps anchor_id -> {"x": float, "y": float, "floor": int}
anchor_coords: dict[str, dict] = {}
# Full floor plan data for the /floorplan endpoint
floorplan_data: dict = {}
# Occupancy grids for wall constraints (particle filter)
occupancy_grids: Optional[OccupancyGridSet] = None

# Inference pipeline components
kalman_bank: Optional[KalmanFilterBank] = None
particle_filter: Optional[ParticleFilter] = None
floor_hmm: Optional[FloorTransitionHMM] = None

# Per-floor room polygons for location labeling: {floor: [(name, polygon), ...]}
room_polygons: dict[int, list[tuple[str, list]]] = {}

# Per-floor, per-room gate definitions for zone sub-classification
# {floor: {room_name: [{"axis": "y", "coord": 20.73, "above": "kitchen", "below": "living_room"}, ...]}}
room_gates: dict[int, dict[str, list[dict]]] = {}

# Timing: last inference timestamp for computing dt
_last_inference_time: Optional[float] = None
_last_inference_mono: float = 0.0  # monotonic clock for rate-limiting

# Feature engineering pipeline
feature_engine: Optional[FeatureEngine] = None

# Zone classifier (Random Forest sub-zone prediction)
zone_classifier: Optional[ZoneClassifier] = None

# Path to model artifacts directory
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))

# PostgreSQL client
db: Optional[Database] = None


def load_floorplan():
    """Load anchor coordinates from layout.json."""
    global anchor_coords, floorplan_data
    # Try multiple paths (Docker mount vs local dev)
    search_paths = [
        FLOORPLAN_PATH,
        "/app/floorplan/layout.json",
        "config/floorplan/layout.json",
        os.path.join(os.path.dirname(__file__), "../../config/floorplan/layout.json"),
    ]
    for path in search_paths:
        try:
            with open(path) as f:
                floorplan_data = json.load(f)
            for floor_data in floorplan_data.get("floors", []):
                floor_num = floor_data["floor"]
                for anchor in floor_data.get("anchors", []):
                    pos = anchor.get("position", [anchor.get("x", 0), anchor.get("y", 0)])
                    # Normalise anchor ID to lowercase to match ESPresense
                    # MQTT topic convention (espresense/devices/<dev>/<anchor>)
                    anchor_id = anchor["id"].lower()
                    anchor_coords[anchor_id] = {
                        "x": pos[0],
                        "y": pos[1],
                        "floor": floor_num,
                        "height_ft": anchor.get("height_ft", 0.0),
                    }
            logger.info(f"Loaded {len(anchor_coords)} anchor positions from {path}")
            return
        except FileNotFoundError:
            continue
    logger.warning("Could not find layout.json — position will lack coordinates")

# -----------------------------------------------------------------------------
# InfluxDB Client
# -----------------------------------------------------------------------------

influx_client: Optional[InfluxDBClient] = None
influx_write_api = None
_influxdb_last_attempt: float = 0.0
_INFLUXDB_RETRY_INTERVAL: float = 60.0  # seconds between reconnect attempts


def init_influxdb():
    """Initialize InfluxDB client with retry backoff."""
    global influx_client, influx_write_api, _influxdb_last_attempt
    now = time.time()
    if now - _influxdb_last_attempt < _INFLUXDB_RETRY_INTERVAL:
        return  # skip retry, too soon
    _influxdb_last_attempt = now
    try:
        influx_client = InfluxDBClient(
            url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG
        )
        influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        # Verify connection
        influx_client.ping()
        logger.info("Connected to InfluxDB at %s", INFLUXDB_URL)
    except Exception as e:
        logger.info("InfluxDB not available — RSSI data will not be persisted (retry in %ds)", int(_INFLUXDB_RETRY_INTERVAL))
        influx_client = None
        influx_write_api = None


def write_rssi_to_influxdb(device_id: str, anchor_id: str, rssi: float, distance: float):
    """Write a single RSSI reading to InfluxDB."""
    global influx_write_api
    if influx_write_api is None:
        init_influxdb()
        if influx_write_api is None:
            return
    try:
        point = (
            Point("rssi_reading")
            .tag("device_id", device_id)
            .tag("anchor_id", anchor_id)
            .field("rssi", float(rssi))
            .field("distance", float(distance))
            .time(datetime.now(timezone.utc), WritePrecision.MS)
        )
        influx_write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    except Exception as e:
        logger.warning(f"InfluxDB write failed: {e}")
        influx_write_api = None  # reset so next call triggers backoff retry

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------

# Latest RSSI readings from each anchor
rssi_buffer: dict[str, dict] = defaultdict(dict)

# MQTT connection state
mqtt_connected = False

# Message counter for health monitoring
message_count = 0

# Position history (ring buffer for recent positions)
position_history: deque = deque(maxlen=POSITION_HISTORY_MAXLEN)

# RSSI history (ring buffer for recent readings)
rssi_history: deque = deque(maxlen=RSSI_HISTORY_MAXLEN)

# Current position estimate
current_position = {
    "x": 0.0,
    "y": 0.0,
    "floor": 1,
    "confidence": 0.0,
    "location_label": "unknown",
    "activity": "unknown",
    "timestamp": None,
}

# -----------------------------------------------------------------------------
# MQTT Client
# -----------------------------------------------------------------------------

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker."""
    global mqtt_connected
    logger.info(f"Connected to MQTT broker with result code {rc}")
    mqtt_connected = rc == 0
    # Subscribe to ESPresense device topics
    # ESPresense publishes to: espresense/devices/<device_id>/<anchor_id>
    client.subscribe("espresense/devices/#")
    # Also subscribe to room-level topics for diagnostics
    client.subscribe("espresense/rooms/#")


def on_disconnect(client, userdata, flags, rc, properties=None):
    """Callback when disconnected from MQTT broker."""
    global mqtt_connected
    mqtt_connected = False
    logger.warning(f"Disconnected from MQTT broker (rc={rc})")


def on_message(client, userdata, msg):
    """Callback when MQTT message received."""
    global message_count
    try:
        topic_parts = msg.topic.split("/")
        # Expected format: espresense/devices/<device_id>/<anchor_id>
        if len(topic_parts) >= 4 and topic_parts[0] == "espresense":
            device_id = topic_parts[2]
            anchor_id = topic_parts[3] if len(topic_parts) > 3 else "unknown"

            raw = msg.payload.decode().strip()
            if not raw:
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return
            if not isinstance(payload, dict):
                return

            rssi_val = payload.get("rssi", -100)
            distance_val = payload.get("distance", 0)
            now = datetime.now(timezone.utc).isoformat()
            
            # Store RSSI reading
            rssi_buffer[device_id][anchor_id] = {
                "rssi": rssi_val,
                "distance": distance_val,
                "timestamp": now,
            }
            
            message_count += 1
            
            # Log to RSSI history and InfluxDB for our beacon
            if device_id == BEACON_ID:
                rssi_history.append({
                    "anchor_id": anchor_id,
                    "rssi": rssi_val,
                    "distance": distance_val,
                    "timestamp": now,
                })
                
                # Write to InfluxDB
                write_rssi_to_influxdb(device_id, anchor_id, rssi_val, distance_val)
                
                # Rate-limit inference: only run if enough time has
                # elapsed since the last cycle.  RSSI buffer still
                # accumulates on every message so the next cycle sees
                # the freshest readings from all anchors.
                now_mono = time.monotonic()
                if (_last_inference_time is None
                        or now_mono - _last_inference_mono
                        >= INFERENCE_MIN_INTERVAL):
                    run_inference(device_id)
            
            # Log all devices periodically for discovery
            if message_count % 50 == 0:
                logger.info(
                    f"Messages received: {message_count} | "
                    f"Devices seen: {list(rssi_buffer.keys())} | "
                    f"Beacon anchors: {list(rssi_buffer.get(BEACON_ID, {}).keys())}"
                )
                
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")


def _label_room(x: float, y: float, floor: int) -> str:
    """Determine which room (or zone within a room) a point falls in.

    If the room defines zones separated by gates (e.g. the open-plan
    living_kitchen split into 'kitchen' and 'living_room' by the
    peninsula gate line), the zone name is returned instead of the
    room name.
    """


def _load_active_classifier() -> Optional[ZoneClassifier]:
    """Load the active RF model from disk if one exists.

    Checks the ``models/`` directory for ``random_forest_v*.joblib`` files.
    If a PostgreSQL ``model_versions`` row with ``active=TRUE`` exists and
    points to a valid file, that model is loaded.  Otherwise falls back to
    the newest joblib file in the directory.

    Returns ``None`` if no model is available (expected before first training).
    """
    global zone_classifier
    model_dir = Path(MODEL_DIR)

    # Try DB-driven model path first
    if db is not None and db.connected:
        try:
            from sqlalchemy import select
            from db import model_versions as mv_table
            with db._engine.connect() as conn:
                row = conn.execute(
                    select(mv_table.c.artifact_path).where(
                        mv_table.c.model_type == "random_forest",
                        mv_table.c.active == True,  # noqa: E712
                    )
                ).first()
            if row and row[0]:
                artifact_path = Path(row[0])
                if artifact_path.exists():
                    clf = ZoneClassifier.from_file(str(artifact_path))
                    logger.info("Loaded active RF model from DB: %s", artifact_path)
                    return clf
        except Exception as e:
            logger.debug("Could not load model via DB: %s", e)

    # Fallback: scan models/ directory for newest joblib file
    if model_dir.exists():
        joblib_files = sorted(model_dir.glob("random_forest_v*.joblib"))
        if joblib_files:
            newest = joblib_files[-1]
            try:
                clf = ZoneClassifier.from_file(str(newest))
                logger.info("Loaded RF model from disk: %s", newest)
                return clf
            except Exception as e:
                logger.warning("Failed to load RF model %s: %s", newest, e)

    logger.info("No RF model available — classifier disabled until training (TODO #21)")
    return None


def _label_room(x: float, y: float, floor: int) -> str:
    for name, poly in room_polygons.get(floor, []):
        if _point_in_polygon(x, y, poly):
            # Check if this room has gate-defined zones
            gates = room_gates.get(floor, {}).get(name, [])
            for gate in gates:
                axis = gate["axis"]
                coord = gate["coord"]
                val = y if axis == "y" else x
                if val >= coord:
                    return gate["above"]
                else:
                    return gate["below"]
            return name
    return "unknown"


def run_inference(device_id: str):
    """
    Run the ML inference pipeline.

    Pipeline:
    1. Collect RSSI vector from all anchors
    2. Apply Kalman filter for per-anchor smoothing
    3. Feed smoothed RSSI into particle filter (which drives Floor HMM)
    4. Read position estimate and determine room label
    5. Update current_position
    """
    global current_position, _last_inference_time, _last_inference_mono

    readings = rssi_buffer.get(device_id, {})
    if not readings:
        return

    now = time.time()
    now_mono = time.monotonic()

    # --- 1. Raw RSSI vector ---------------------------------------------------
    rssi_vector = {anchor: data["rssi"] for anchor, data in readings.items()}
    if not rssi_vector:
        return

    # --- 2. Kalman smoothing --------------------------------------------------
    if kalman_bank is not None:
        for anchor_id, raw_rssi in rssi_vector.items():
            kalman_bank.update(anchor_id, raw_rssi, now)
        smoothed_rssi = kalman_bank.get_smoothed_rssi(current_time=now)
    else:
        smoothed_rssi = rssi_vector

    # --- 3. Feature engineering -----------------------------------------------
    if feature_engine is not None:
        computed_features = feature_engine.update(rssi_vector, smoothed_rssi, now)
    else:
        computed_features = {}

    # --- 4. Particle filter step (includes Floor HMM) ------------------------
    if particle_filter is None:
        logger.warning("Particle filter not initialised — skipping inference")
        return

    # Compute dt since last inference
    dt = (now - _last_inference_time) if _last_inference_time is not None else DEFAULT_DT
    dt = min(dt, MAX_DT)  # cap to avoid huge jumps after long gaps

    estimate = particle_filter.step(smoothed_rssi, dt)

    _last_inference_time = now
    _last_inference_mono = now_mono

    # --- 5. Room label (from particle filter position) -------------------------
    polygon_label = _label_room(estimate["x"], estimate["y"], estimate["floor"])

    # --- 6. Floor belief from HMM (if available) ------------------------------
    floor_belief = None
    if floor_hmm is not None:
        floor_belief = floor_hmm.floor_belief

    # --- 7. Activity label from feature engine --------------------------------
    activity_score = computed_features.get("activity_score", 0.0)
    if activity_score < ACTIVITY_SLEEPING_THRESHOLD:
        activity_label = "sleeping"
    elif activity_score < ACTIVITY_STATIONARY_THRESHOLD:
        activity_label = "stationary"
    else:
        activity_label = "moving"

    # --- 8. Hierarchical sub-zone prediction ----------------------------------
    # Particle filter determines the room (polygon lookup).
    # RF classifier refines to a sub-zone within that room.
    # Zone smoother applies activity-adaptive EMA to prevent snapping.
    zone_label = None
    zone_confidence = 0.0
    zone_probabilities = {}

    if zone_classifier is not None and zone_classifier.is_trained:
        try:
            zone_label, zone_confidence, zone_probabilities = (
                zone_classifier.predict_for_room(
                    computed_features,
                    smoothed_rssi=smoothed_rssi,
                    room=polygon_label,
                )
            )
        except Exception as e:
            logger.warning("Zone classifier prediction failed: %s", e)

    # --- 9. Final fused label -------------------------------------------------
    # If sub-zone prediction is available, use it; else fall back to room label.
    if zone_label is not None:
        location_label = zone_label
    else:
        location_label = polygon_label

    # --- 10. Update position state --------------------------------------------
    position_update = {
        "x": round(estimate["x"], 2),
        "y": round(estimate["y"], 2),
        "floor": estimate["floor"],
        "location_label": location_label,
        "confidence": round(estimate["confidence"], 3),
        "activity": activity_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_rssi": rssi_vector,
        "smoothed_rssi": smoothed_rssi,
        "n_eff": estimate.get("n_eff", 0.0),
        "particle_count": estimate.get("particle_count", 0),
        "floor_belief": floor_belief,
        "polygon_label": polygon_label,
        "zone_label": zone_label,
        "zone_confidence": round(zone_confidence, 3),
        "zone_probabilities": zone_probabilities,
    }
    current_position.update(position_update)

    # Store in position history (in-memory ring buffer)
    position_history.append({**current_position})

    # Persist to PostgreSQL
    if db is not None:
        db.write_position(
            beacon_id=device_id,
            x=position_update["x"],
            y=position_update["y"],
            floor=position_update["floor"],
            confidence=position_update["confidence"],
            location_label=location_label,
            activity_state=position_update["activity"],
            raw_rssi=rssi_vector,
            smoothed_rssi=smoothed_rssi if isinstance(smoothed_rssi, dict) else None,
            n_eff=estimate.get("n_eff"),
            particle_count=estimate.get("particle_count"),
            floor_belief=floor_belief,
        )

    logger.debug(
        f"Position: floor {estimate['floor']} "
        f"({estimate['x']:.1f}, {estimate['y']:.1f}) "
        f"room={location_label} conf={estimate['confidence']:.2f} "
        f"n_eff={estimate.get('n_eff', 0):.0f}"
    )


def start_mqtt_client():
    """Start MQTT client in background thread."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    while True:
        try:
            logger.info(f"Connecting to MQTT broker at {MQTT_HOST}:{MQTT_PORT}...")
            client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
            client.loop_forever()
        except Exception as e:
            logger.error(f"MQTT connection failed: {e} — retrying in {MQTT_RECONNECT_DELAY}s...")
            import time
            time.sleep(MQTT_RECONNECT_DELAY)


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Pet Localization API",
    description="Real-time indoor positioning for pet tracking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PositionResponse(BaseModel):
    x: float
    y: float
    floor: int
    confidence: float
    location_label: str
    activity: str
    timestamp: Optional[str]
    raw_rssi: Optional[dict] = None
    smoothed_rssi: Optional[dict] = None
    n_eff: Optional[float] = None
    particle_count: Optional[int] = None
    floor_belief: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    mqtt_connected: bool
    anchors_active: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "pet-localization-inference", "version": "0.1.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if mqtt_connected else "degraded",
        "mqtt_connected": mqtt_connected,
        "anchors_active": len(rssi_buffer.get(BEACON_ID, {})),
    }


@app.get("/position", response_model=PositionResponse)
async def get_position():
    """Get current position estimate."""
    if current_position["timestamp"] is None:
        raise HTTPException(status_code=503, detail="No position data available yet")
    return current_position


@app.get("/rssi")
async def get_rssi():
    """Get raw RSSI readings from all anchors."""
    return rssi_buffer.get(BEACON_ID, {})


@app.get("/rssi/history")
async def get_rssi_history(limit: int = 100):
    """Get recent RSSI history."""
    return list(rssi_history)[-limit:]


@app.get("/position/history")
async def get_position_history(limit: int = 100, source: str = "memory"):
    """Get recent position history.

    Args:
        limit: Max records to return.
        source: 'memory' for in-memory ring buffer, 'db' for PostgreSQL.
    """
    if source == "db" and db is not None and db.connected:
        rows = db.read_position_history(beacon_id=BEACON_ID, limit=limit)
        # Serialize datetimes for JSON
        for row in rows:
            for k, v in row.items():
                if isinstance(v, datetime):
                    row[k] = v.isoformat()
        return rows
    return list(position_history)[-limit:]


@app.get("/devices")
async def get_devices():
    """List all detected BLE devices (helps find your beacon's device ID)."""
    devices = {}
    for device_id, anchors in rssi_buffer.items():
        devices[device_id] = {
            "anchor_count": len(anchors),
            "anchors": {
                aid: {"rssi": data["rssi"], "distance": data["distance"]}
                for aid, data in anchors.items()
            },
        }
    return {
        "configured_beacon_id": BEACON_ID,
        "total_devices": len(devices),
        "devices": devices,
    }


@app.get("/anchors")
async def get_anchors():
    """List all detected anchors for the configured beacon."""
    beacon_data = rssi_buffer.get(BEACON_ID, {})
    return {
        "beacon_id": BEACON_ID,
        "anchor_count": len(beacon_data),
        "anchors": list(beacon_data.keys()),
    }


@app.get("/floorplan")
async def get_floorplan():
    """Get floor plan data including anchor positions."""
    return {
        "floorplan": floorplan_data,
        "anchor_coords": anchor_coords,
    }


@app.get("/fingerprints")
async def get_fingerprints(
    floor: Optional[int] = None,
    location: Optional[str] = None,
    limit: int = 100,
):
    """Get fingerprint samples from PostgreSQL (for site survey / training)."""
    if db is None or not db.connected:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    rows = db.read_fingerprint_samples(
        floor=floor, location_label=location, limit=limit
    )
    for row in rows:
        for k, v in row.items():
            if isinstance(v, datetime):
                row[k] = v.isoformat()
    return rows


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    stats = {
        "mqtt_connected": mqtt_connected,
        "message_count": message_count,
        "beacon_id": BEACON_ID,
        "devices_seen": len(rssi_buffer),
        "anchors_for_beacon": len(rssi_buffer.get(BEACON_ID, {})),
        "position_history_size": len(position_history),
        "rssi_history_size": len(rssi_history),
        "influxdb_connected": influx_client is not None,
        "postgres_connected": db.connected if db else False,
        "pipeline": {
            "kalman_active": kalman_bank is not None,
            "kalman_anchors": kalman_bank.active_anchors if kalman_bank else [],
            "particle_filter_active": particle_filter is not None,
            "floor_hmm_active": floor_hmm is not None,
        },
    }
    if particle_filter is not None:
        est = particle_filter.estimate
        stats["pipeline"]["particle_n_eff"] = est.get("n_eff", 0)
        stats["pipeline"]["particle_count"] = est.get("particle_count", 0)
    if floor_hmm is not None:
        stats["pipeline"]["floor_belief"] = floor_hmm.floor_belief
        stats["pipeline"]["most_likely_floor"] = floor_hmm.most_likely_floor
    if zone_classifier is not None:
        stats["pipeline"]["zone_classifier_active"] = zone_classifier.is_trained
        stats["pipeline"]["zone_classes"] = zone_classifier.classes
        stats["pipeline"]["zone_to_room"] = zone_classifier.zone_to_room
    else:
        stats["pipeline"]["zone_classifier_active"] = False
    return stats


@app.get("/particles")
async def get_particles():
    """Get current particle positions and weights for visualization."""
    if particle_filter is None:
        raise HTTPException(status_code=503, detail="Particle filter not initialized")

    parts = particle_filter.particles
    wts = particle_filter.weights

    return {
        "x": parts[:, 0].tolist(),
        "y": parts[:, 1].tolist(),
        "floor": parts[:, 2].astype(int).tolist(),
        "weight": wts.tolist(),
        "n_particles": len(parts),
        "estimate": particle_filter.estimate,
    }


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Start MQTT client and initialize connections on application startup."""
    global occupancy_grids, kalman_bank, particle_filter, floor_hmm
    global room_polygons, room_gates, feature_engine, zone_classifier, db
    logger.info("Starting inference service...")

    # Load floor plan anchor coordinates
    load_floorplan()

    # Build occupancy grids from loaded floor plan
    if floorplan_data:
        occupancy_grids = OccupancyGridSet.from_layout_data(floorplan_data)
        logger.info(f"Built occupancy grids: {occupancy_grids}")

    # Build room polygon lookup and zone/gate data for location labeling
    for floor_data in floorplan_data.get("floors", []):
        floor_num = floor_data["floor"]
        room_polygons[floor_num] = [
            (room["name"], bounds_to_polygon(room["bounds"]))
            for room in floor_data.get("rooms", [])
        ]
        # Parse zone gates for rooms that have them
        for room in floor_data.get("rooms", []):
            gates = room.get("gates", [])
            if not gates:
                continue
            parsed_gates = []
            for gate in gates:
                # Convention: between[0] = zone where axis value >= coord
                #             between[1] = zone where axis value <  coord
                parsed_gates.append({
                    "axis": gate["axis"],
                    "coord": gate["coord"],
                    "above": gate["between"][0],
                    "below": gate["between"][1],
                })
            room_gates.setdefault(floor_num, {})[room["name"]] = parsed_gates
            logger.info(
                f"Floor {floor_num} room '{room['name']}' has "
                f"{len(parsed_gates)} zone gate(s): "
                f"{[g['above'] + '/' + g['below'] for g in parsed_gates]}"
            )

    # --- Inference pipeline initialisation ---
    # Kalman filter bank (per-anchor RSSI smoothing)
    kalman_bank = KalmanFilterBank(
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        stale_timeout=KALMAN_STALE_TIMEOUT,
    )
    logger.info("Kalman filter bank initialised")

    # Floor Transition HMM
    if floorplan_data and anchor_coords:
        floor_hmm = FloorTransitionHMM(floorplan_data, anchor_coords)
        logger.info(f"Floor HMM initialised: floors {floor_hmm.floors}")

    # Particle filter (needs occupancy grids + anchors + stairways + HMM)
    if occupancy_grids and anchor_coords:
        stairways = extract_stairways(floorplan_data)
        stair_runs = extract_stair_runs(floorplan_data)
        staircase_bounds = extract_staircase_bounds(floorplan_data)
        particle_filter = ParticleFilter(
            occupancy_grids=occupancy_grids,
            anchor_positions=anchor_coords,
            stairways=stairways,
            stair_runs=stair_runs,
            staircase_bounds=staircase_bounds,
            floor_hmm=floor_hmm,
            n_particles=N_PARTICLES,
        )
        particle_filter.initialise_uniform()
        logger.info(f"Particle filter initialised: {particle_filter}")

    # Feature engineering pipeline
    feature_engine = FeatureEngine(
        anchor_positions=anchor_coords,
        window_size=FEATURE_WINDOW_SIZE,
    )
    logger.info("Feature engine initialised")

    # Zone classifier — load active model if one exists
    zone_classifier = _load_active_classifier()

    # Initialize InfluxDB
    init_influxdb()

    # Initialize PostgreSQL
    db = Database(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )
    db.connect()

    # Start MQTT client in background thread
    mqtt_thread = Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()

    logger.info(f"MQTT client connecting to {MQTT_HOST}:{MQTT_PORT}")
    logger.info(f"Tracking beacon ID: {BEACON_ID}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
