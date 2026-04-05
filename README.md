# Bayesian Pet Localization

A real-time indoor positioning system that tracks a dog's location across a 3-story townhouse using BLE signal fingerprinting, Bayesian state estimation, and physics-based filtering.

## Overview

This project combines IoT sensor networks with probabilistic ML to solve indoor localization — a core challenge in robotics and spatial analytics. A BLE beacon on a dog collar broadcasts to fixed ESP32 anchor nodes throughout the house. The received signal strengths (RSSI) feed a multi-stage inference pipeline that produces a continuous position estimate with room-level (and sub-room zone) resolution across all three floors.

**Key capabilities:**
- Sub-room zone resolution (e.g. kitchen vs. living area within an open floor plan)
- Multi-floor positioning with stairway-aware floor transition logic
- Real-time Streamlit dashboard with per-floor visualization
- Activity inference (sleeping, stationary, moving) from signal dynamics
- Full ML pipeline: signal smoothing → feature engineering → classification → Bayesian filtering

## Hardware

### BLE Beacon (Dog Collar Tag)

**BlueCharm BC021** (~$19)
- Motion sensor + button trigger for smart advertising
- 36×36×5.75mm, 8g with CR2032 battery
- Configurable: 100–10,000ms advertising interval, -40 to +4 dBm TX power
- BLE 5.0, supports iBeacon / Eddystone formats

**Project configuration:**
- SLOT 0 (iBeacon): 211.25ms advertising interval, +4 dBm TX power, -55 dBm measured power
- SLOT 2 (TLM (Eddystone)): 10s interval for battery voltage and temperature telemetry
- Motion trigger: OFF (continuous advertising for consistent tracking, activity manually calculated from RSSI delta)

### Anchor Nodes

**M5Stack AtomS3 Lite** × 8 units
- ESP32-S3, built-in 3D antenna, USB-C powered
- Runs [ESPresense](https://espresense.com/) firmware
- Publishes RSSI readings to MQTT topics per device

**Placement across floors:**
| Floor | Anchors | Locations |
|-------|---------|-----------|
| 1st | 2 | Office (near dog bed), Hallway (near stairs) |
| 2nd | 3 | Living SE, Living SW/center, Kitchen NE |
| 3rd | 3 | Master bedroom, Hallway, 3F office |

### Infrastructure

- **Raspberry Pi 4/5 (4GB+):** Hosts all services via Docker Compose
- **Mosquitto:** MQTT broker receiving ESPresense messages
- **InfluxDB 2.7:** Time-series storage for raw RSSI data
- **PostgreSQL 15:** Position history, labeled fingerprints, model registry

## Floorplan

**Coordinate system:** 1 unit = 1 foot. Origin (0, 0) at SW corner of each floor. +X = East, +Y = North.

**Scan accuracy:** ARPlan 3D on Samsung S21 Ultra (ToF depth sensor). Cross-scan door width tolerance: ±0.15 in (±4mm) across 6 shared doorways (~±0.1–0.3% at room scale).

```
Floor 1: Entrance + hallway (14×4 ft), staircase, office (9×8 ft)
Floor 2: Open floor plan (15×30 ft) with kitchen/living zone split, powder room (3×4 ft)
Floor 3: Master bedroom (14×10 ft), master bath (6×5 ft), hallway (14.5×3.5 ft),
         spare bedroom (10×10 ft), guest bathroom (5×5 ft)
```

The floorplan is defined in `config/floorplan/layout.json` with room polygons, doorway connections, obstacle boundaries, anchor coordinates, stairway entries, points of interest, and sub-room zone gates.

## Inference Pipeline

### Data Flow

```
BLE Beacon → ESP32 Anchors (ESPresense) → MQTT → Inference Service → PostgreSQL / InfluxDB
                                                        ↓
                                                  Streamlit Dashboard
```

### Pipeline Stages

Each MQTT message for the tracked beacon triggers a full inference cycle:

```
Raw RSSI → Kalman Filter → Feature Engineering → Particle Filter (+ Floor HMM) → Position Estimate
                                ↓                        ↓
                        Random Forest              Room Label +
                        (if trained)               Activity State
```

**1. Kalman Filtering** — Per-anchor RSSI smoothing via a 1-D Kalman filter bank. State vector `[rssi, rssi_velocity]` tracks signal level and rate of change. Stale anchors (>30s silent) are automatically reset.

**2. Feature Engineering** — The `FeatureEngine` computes ~40 derived features per cycle:
- **Delta RSSI:** Per-anchor smoothed change + aggregate mean/max
- **Anchor Rankings:** Ordinal rank by signal strength, rank change count, top-two RSSI gap
- **Rolling Variance:** Per-anchor raw RSSI variance over a sliding window, plus aggregates
- **Cross-Floor Attenuation:** Per-floor anchor counts, mean RSSI, best-floor indicator, same-vs-cross signal ratio
- **Composite Activity Score:** Weighted blend (60% variance, 40% delta) normalized to [0, 1]

**3. Particle Filter (Sequential Monte Carlo)** — 500 particles distributed across walkable space. Each cycle:
- **Predict:** Gaussian random walk (~1 m/s max), wall-constrained via Bresenham ray-casting against the occupancy grid
- **Update:** Log-distance path-loss likelihood weighting (TX power −59 dBm, path-loss exponent 2.7, σ 5 dBm). Cross-floor anchors incur a +30 ft effective distance penalty.
- **Resample:** Systematic resampling when effective sample size drops below 50%
- **Estimate:** Weighted mean position with floor assignment and confidence metric

**4. Floor Transition HMM** — Discrete hidden Markov model over floor states {1, 2, 3}. Self-stay probability ~0.97, adjacent-floor transitions only (no 1→3 jumps). Stairway proximity (≤6 ft from an entry point) provides a logistic boost to transition probability, preventing the system from locking onto a single floor.

**5. Room Classification** — Primary method: point-in-polygon lookup against floor plan room boundaries, with zone gates for open-plan areas (e.g. kitchen vs. living room split by a Y-axis boundary). Optional: a trained Random Forest classifier can override the polygon label when its confidence exceeds a configurable threshold (default 0.7).

**6. Activity Inference** — Derived from the feature engine's activity score:
| Score Range | State |
|-------------|-------|
| < 0.15 | Sleeping |
| 0.15 – 0.45 | Stationary |
| ≥ 0.45 | Moving |

### Bayesian Formulation

$$P(L_t | S_{1:t}) \propto P(S_t | L_t) \int P(L_t | L_{t-1}) \, P(L_{t-1} | S_{1:t-1}) \, dL_{t-1}$$

- $P(S_t | L_t)$: Observation model — log-distance path-loss likelihood of RSSI given particle position
- $P(L_t | L_{t-1})$: Motion model — Gaussian random walk constrained by walls and floor transitions
- The particle filter approximates this integral via sequential importance sampling with resampling

## Random Forest Room Classifier

A supervised room classifier trained on labeled RSSI fingerprints:

- **Features:** ~48 inputs — per-anchor smoothed RSSI (sentinel −100 dBm for missing anchors) + all derived feature-engine outputs
- **Model:** `sklearn.ensemble.RandomForestClassifier` (200 trees, balanced class weights)
- **Labels:** Floor-qualified room names (e.g. `2F_kitchen`, `3F_master_bed`)
- **Data augmentation:** 5× multiplier with Gaussian RSSI jitter (σ 3 dBm) and random anchor dropout (15%)
- **Evaluation:** Stratified K-fold cross-validation (5 folds, adjusted for small classes)
- **Persistence:** Serialized via joblib; active model tracked in PostgreSQL `model_versions` table

The classifier is optional — the system produces position estimates from the particle filter alone. When a trained model is available, its predictions are fused with the geometric polygon lookup using a confidence gate.

## Occupancy Grids

The `OccupancyGridSet` converts floor plan room polygons into walkable/blocked bitmaps at 0.5 ft resolution:

1. Rasterize room interiors as walkable
2. Subtract obstacle polygons (furniture, fixtures)
3. Enforce walls at room boundaries
4. Carve doorways to restore connectivity between rooms

The particle filter uses this grid for wall-constrained motion (`ray_clear` Bresenham test) and uniform initialization across walkable cells.

## Database Schema

**PostgreSQL** stores structured pipeline outputs and training data:
- `position_history` — Timestamped position estimates with confidence, RSSI vectors, particle stats, floor belief
- `fingerprint_samples` — Labeled training data (location, floor, RSSI vector, features, metadata)
- `model_versions` — Model registry (type, version, artifact path, metrics, active flag)
- `floors` / `anchors` — Floor definitions and anchor positions

**InfluxDB** stores high-frequency raw RSSI as time-series data (`rssi_reading` measurement with device/anchor tags).

## API Endpoints

The inference service exposes a FastAPI interface:

| Endpoint | Description |
|----------|-------------|
| `GET /position` | Current position estimate (x, y, floor, confidence, location, activity) |
| `GET /position/history` | Historical positions (filterable by beacon, floor) |
| `GET /rssi` | Current raw RSSI from all anchors |
| `GET /rssi/history` | Recent RSSI ring buffer |
| `GET /health` | System health (MQTT status, anchor count) |
| `GET /stats` | Pipeline diagnostics (Kalman state, particle ESS, HMM belief, RF status) |
| `GET /devices` | All detected BLE devices |
| `GET /anchors` | Anchors reporting for the tracked beacon |
| `GET /floorplan` | Floor plan data with anchor coordinates |
| `GET /fingerprints` | Labeled RSSI samples from PostgreSQL |

## Dashboard

A Streamlit web UI providing real-time visualization:

- **Status sidebar:** API health, MQTT connection, active anchor count, configurable refresh rate
- **Position metrics:** Location label, floor, confidence %, activity state
- **Multi-floor plans:** Per-floor room polygons with position marker, confidence radius, particle cloud, and recent trail
- **RSSI chart:** Per-anchor bar chart color-coded by signal strength (−100 to −30 dBm)
- **Raw data:** Expandable JSON view of the latest RSSI readings

## Project Structure

```
bayesian-pet-localization/
├── config/
│   ├── floorplan/
│   │   └── layout.json              # Floor plans, rooms, anchors, POIs, zones
│   ├── mosquitto/
│   │   └── mosquitto.conf           # MQTT broker config
│   └── postgres/
│       └── init.sql                 # Database schema
├── models/                           # Trained ML models (.joblib, gitignored)
├── scripts/
│   ├── simulate_mqtt.py             # MQTT simulator (static/walking modes)
│   ├── demo_tracker.py              # Demo visualization generator
│   └── plot_layout.py               # Floor plan renderer
├── services/
│   ├── inference/                   # Python inference service
│   │   ├── main.py                  # FastAPI + MQTT subscriber + pipeline
│   │   ├── db.py                    # PostgreSQL interface
│   │   ├── features.py              # Feature engineering pipeline
│   │   ├── occupancy.py             # Occupancy grid generation
│   │   ├── filters/
│   │   │   ├── constants.py         # Shared physical-model constants
│   │   │   ├── kalman.py            # Kalman filter bank
│   │   │   ├── particle.py          # Particle filter (SMC)
│   │   │   └── floor_hmm.py         # Floor transition HMM
│   │   ├── models/
│   │   │   └── classifier.py        # Random Forest room classifier
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── dashboard/                   # Streamlit visualization
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── tests/                            # 194 tests across 8 test files
├── .env.example                      # Environment variable template
├── docker-compose.yml                # Full stack (5 services)
└── TODO.md                           # Roadmap and task tracking
```

## Quick Start

```bash
# Clone
git clone https://github.com/chill0121/bayesian-pet-localization.git
cd bayesian-pet-localization

# Configure secrets
cp .env.example .env
# Edit .env — set passwords and InfluxDB token

# Start all services
docker compose up -d

# Access
# Dashboard:    http://localhost:8501
# Inference API: http://localhost:8000
# InfluxDB UI:  http://localhost:8086
```

**Services:**
| Service | Port | Description |
|---------|------|-------------|
| Mosquitto | 1883 | MQTT broker (anchors publish here) |
| InfluxDB | 8086 | Time-series database (raw RSSI) |
| PostgreSQL | 5432 | Relational database (positions, fingerprints, models) |
| Inference | 8000 | FastAPI inference engine |
| Dashboard | 8501 | Streamlit visualization |

### Running Without Hardware

The MQTT simulator lets you test the full pipeline without physical devices:

```bash
pip install paho-mqtt
python scripts/simulate_mqtt.py --host localhost --beacon-id dog_collar --mode walking
```

### Running Tests

```bash
pip install -r scripts/requirements.txt
python -m pytest tests/ -v
```

## Configuration

All secrets (database passwords, API tokens) are stored in a `.env` file that is **gitignored**. Docker Compose reads from `.env` automatically. See `.env.example` for the template.

Pipeline tuning constants (path-loss model, filter parameters, activity thresholds) are centralized as named module-level variables in their respective files under `services/inference/`. Shared physical constants live in `filters/constants.py`.

## Project Status

**Current phase:** Core inference pipeline complete. Pending hardware deployment and site survey.

### Completed
- Per-anchor Kalman filter bank with staleness handling
- Occupancy grid generator with wall enforcement and doorway carving
- Feature engineering pipeline (~40 derived features)
- Particle filter with wall-constrained motion and log-distance observation model
- Floor transition HMM with stairway proximity boosting
- Random Forest room classifier infrastructure (training, augmentation, persistence)
- PostgreSQL and InfluxDB integration
- Full pipeline integration in FastAPI service
- Streamlit dashboard with multi-floor visualization
- MQTT simulator for hardware-free testing
- 194 unit and integration tests

### Pending
- Anchor deployment (9 remaining units)
- Site survey collection and fingerprint labeling
- RF classifier training on real data
- Dashboard enhancements (heatmaps, particle cloud, history trail)
- Parameter tuning against live data

See [TODO.md](TODO.md) for the full roadmap.

---

## Technical Highlights

This project demonstrates:

1. **Bayesian State Estimation** — Custom particle filter implementation with systematic resampling, wall-constrained motion, and multi-floor support
2. **Sensor Fusion** — Kalman-filtered RSSI combined with physics-based path-loss models and occupancy grid constraints
3. **Probabilistic Floor Inference** — HMM with stairway-proximity-modulated transition probabilities
4. **Feature Engineering** — 40+ derived features from raw signal data for ML classification
5. **System Architecture** — End-to-end IoT pipeline: hardware → MQTT → real-time ML inference → database → dashboard
6. **Signal Processing** — Handling NLOS interference, cross-floor signal bleed, and noisy indoor RF environments