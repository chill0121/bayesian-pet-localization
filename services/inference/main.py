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
from threading import Thread
from typing import Optional

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from pydantic import BaseModel

# Local modules (to be implemented)
# from filters.kalman import KalmanFilter
# from models.classifier import LocationClassifier
# from filters.particle import ParticleFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
BEACON_ID = os.getenv("BEACON_ID", "dog_collar")

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "pet-localization")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "rssi_raw")
FLOORPLAN_PATH = os.getenv("FLOORPLAN_PATH", "config/floorplan/layout.json")

# -----------------------------------------------------------------------------
# Floor Plan / Anchor Coordinate Lookup
# -----------------------------------------------------------------------------

# Maps anchor_id -> {"x": float, "y": float, "floor": int}
anchor_coords: dict[str, dict] = {}
# Full floor plan data for the /floorplan endpoint
floorplan_data: dict = {}


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
                    anchor_coords[anchor["id"]] = {
                        "x": anchor["x"],
                        "y": anchor["y"],
                        "floor": floor_num,
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
position_history: deque = deque(maxlen=500)

# RSSI history (ring buffer for recent readings)
rssi_history: deque = deque(maxlen=1000)

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
            
            payload = json.loads(msg.payload.decode())
            
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
                
                # Trigger inference
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


def run_inference(device_id: str):
    """
    Run the ML inference pipeline.
    
    Pipeline:
    1. Collect RSSI vector from all anchors
    2. Apply Kalman filter for smoothing
    3. Extract features
    4. Run classifier for location prediction
    5. Update particle filter for temporal smoothing
    6. Update current_position
    """
    global current_position
    
    readings = rssi_buffer.get(device_id, {})
    if not readings:
        return
    
    # Extract RSSI vector
    rssi_vector = {anchor: data["rssi"] for anchor, data in readings.items()}
    
    # TODO: Implement actual inference pipeline
    # For now, just find the strongest anchor and use its coordinates
    if rssi_vector:
        strongest_anchor = max(rssi_vector, key=rssi_vector.get)
        
        # Look up anchor coordinates from floor plan
        coords = anchor_coords.get(strongest_anchor, {})
        
        position_update = {
            "x": coords.get("x", 0.0),
            "y": coords.get("y", 0.0),
            "floor": coords.get("floor", 1),
            "location_label": strongest_anchor,
            "confidence": 0.5,  # Placeholder
            "activity": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_rssi": rssi_vector,
        }
        current_position.update(position_update)
        
        # Store in position history
        position_history.append({**current_position})
        
        logger.debug(f"Position updated: {strongest_anchor} @ floor {coords.get('floor')}, ({coords.get('x')}, {coords.get('y')})")


def start_mqtt_client():
    """Start MQTT client in background thread."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    while True:
        try:
            logger.info(f"Connecting to MQTT broker at {MQTT_HOST}:{MQTT_PORT}...")
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            logger.error(f"MQTT connection failed: {e} — retrying in 5s...")
            import time
            time.sleep(5)


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
async def get_position_history(limit: int = 100):
    """Get recent position history."""
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


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "mqtt_connected": mqtt_connected,
        "message_count": message_count,
        "beacon_id": BEACON_ID,
        "devices_seen": len(rssi_buffer),
        "anchors_for_beacon": len(rssi_buffer.get(BEACON_ID, {})),
        "position_history_size": len(position_history),
        "rssi_history_size": len(rssi_history),
        "influxdb_connected": influx_client is not None,
    }


# -----------------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Start MQTT client and initialize connections on application startup."""
    logger.info("Starting inference service...")
    
    # Load floor plan anchor coordinates
    load_floorplan()
    
    # Initialize InfluxDB
    init_influxdb()
    
    # Start MQTT client in background thread
    mqtt_thread = Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    logger.info(f"MQTT client connecting to {MQTT_HOST}:{MQTT_PORT}")
    logger.info(f"Tracking beacon ID: {BEACON_ID}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
