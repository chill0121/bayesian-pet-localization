# Bayesian Pet Localization

A real-time indoor positioning system that tracks a dog's location across a 3-story townhouse with sub-room (2m) accuracy.

## Overview

This project demonstrates advanced spatial analytics by combining IoT sensor networks with Bayesian state estimation to solve the indoor localization problem, a core challenge in robotics and autonomous systems.

**Key Features:**
- Sub-room resolution tracking (e.g. distinguishes between couch vs. water bowl 2m apart)
- Multi-floor positioning with elevation inference
- Real-time visualization dashboard
- End-to-end ML pipeline from hardware to inference

## Hardware

### BLE Beacon (Dog Collar Tag)

**Selected: BlueCharm BC021** (~$19)
- Motion sensor + button trigger for smart advertising
- Lightest option: 36×36×5.75mm, 8g
- CR2032 battery — ~1.5 months at 100ms intervals
- Configurable: 100-10,000ms advertising interval, -40 to +4 dBm TX power
- BLE 5.0 (backward compatible with 4.0)
- Supports iBeacon, Eddystone UID/URL/TLM formats

**Configuration for this project:**
- Advertising interval: 100-200ms (5-10Hz) for real-time tracking
- TX Power: 0 dBm (reduces floor bleed while maintaining range)
- Motion-triggered advertising: fast when moving, slow when stationary

### Anchor Nodes

**M5Stack AtomS3 Lite** × 11-12 units
- ESP32-S3 with built-in 3D antenna
- Runs [ESPresense](https://espresense.com/) firmware
- USB-C powered, compact form factor

**Placement by floor:**
| Floor | Nodes | Locations |
|-------|-------|-----------|
| 1st | 2 | Office (near dog bed), Hallway (near stairs) |
| 2nd | 5 | 4× corners of open floor plan, 1× powder room |
| 3rd | 4 | Master bedroom, Master bath, Wife's office, Hallway/stairs |

### Infrastructure

- **Raspberry Pi 4/5 (4GB+):** Hosts MQTT broker, databases, inference engine
- **Mosquitto:** MQTT message broker
- **InfluxDB:** Time-series storage for raw RSSI
- **PostgreSQL:** Cleaned state history and labeled training data

## Floorplan

```
Floor 1: Entrance + hallway (14x4 ft), staircase, office (9x8 ft)
Floor 2: Open floor plan (15x30 ft), powder room (3x4 ft), stairway
Floor 3: Master bedroom (14x10 ft), master bath (6x5 ft), hallway (14.5x3.5 ft),
         spare bedroom (10x10 ft), guest bathroom (5x5 ft)
```

## Technical Approach

### Data Flow

```
BLE Beacon -> ESP32 Anchors (ESPresense) -> MQTT -> Python Service -> InfluxDB/PostgreSQL
```

### ML Pipeline

```
Raw RSSI -> Kalman Filter -> Feature Engineering -> Random Forest -> Particle Filter -> Point Estimate (x, y, floor)
```

### Core Algorithms

1. **Kalman Filter:** Signal smoothing to reduce RSSI noise
2. **RSSI Fingerprinting (Random Forest/SVM):** Supervised classification treating signal signatures as location IDs
3. **Particle Filter (Sequential Monte Carlo):** Bayesian state estimation combining ML predictions with motion model
4. **Hidden Markov Model:** Floor transition logic preventing impossible state changes

### Feature Engineering

- **Delta RSSI:** Difference between strongest and second-strongest anchor
- **Anchor rankings:** Vector of anchor IDs sorted by signal strength
- **Rolling variance:** High variance indicates movement or doorways
- **Relative RSSI from other floors:** Cross-floor attenuation patterns

### Tech Stack

- **Data:** MQTT (Mosquitto), InfluxDB, PostgreSQL
- **Processing:** Python (paho-mqtt, FastAPI)
- **ML:** scikit-learn, custom particle filter implementation
- **Visualization:** Streamlit, Plotly

## Site Survey Protocol

### Grid-Based Fingerprinting

- Create 0.5m x 0.5m or 1m x 1m occupancy grid across walkable areas
- For each grid cell: capture 60 seconds of RSSI from all anchors
- Rotate collar 360 degrees during capture (accounts for body shielding)
- Include floor ID and relative RSSI from anchors on other floors

### Labeled Dataset Collection

Place collar at known locations (Couch, Water Bowl, Dog Beds) and log RSSI vectors to build training set.

## ML-Enhanced Bayesian Localization

The "Gold Standard" approach combines fingerprinting with particle filtering:

1. **ML Model (Observer):** Random Forest outputs probability distribution over floor plan
2. **Particle Filter (Smoother):**
   - Maintain 500+ particles (x, y, z)
   - Update particles based on motion model (~1 m/s dog speed)
   - Weight particles using ML model output
   - Resample near high-probability particles

**Bayesian Formulation:**

$$P(L_t | S_{1:t}) \propto P(S_t | L_t) \int P(L_t | L_{t-1}) P(L_{t-1} | S_{1:t-1}) dL_{t-1}$$

- $P(S_t | L_t)$: ML model (likelihood of signals at location)
- $P(L_t | L_{t-1})$: Motion model (transition probability)

## Visualization

- **Framework:** Streamlit + Plotly
- **Floor Plan Mapping:** Map coordinates to SVG/PNG of townhouse
- **Outputs:**
  - Real-time position with confidence
  - 24-hour residence heatmaps
  - Particle cloud visualization
  - Activity inference (Sleeping, Pacing, Eating/Drinking)

## Project Structure

```
bayesian-pet-localization/
├── config/
│   ├── floorplan/
│   │   └── layout.json          # Floor plans, anchor positions, POIs
│   ├── mosquitto/
│   │   └── mosquitto.conf       # MQTT broker config
│   └── postgres/
│       └── init.sql             # Database schema
├── models/                       # Trained ML models (.pkl)
├── services/
│   ├── inference/               # Python inference service
│   │   ├── main.py              # FastAPI + MQTT subscriber
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── dashboard/               # Streamlit visualization
│       ├── app.py
│       ├── requirements.txt
│       └── Dockerfile
└── docker-compose.yml           # Full stack deployment
```

## Quick Start (Raspberry Pi)

```bash
# Clone repository
git clone https://github.com/chill0121/bayesian-pet-localization.git
cd bayesian-pet-localization

# Start all services
docker compose up -d

# View logs
docker compose logs -f inference

# Access dashboard
# Open browser: http://<pi-ip>:8501
```

**Services:**
| Service | Port | Description |
|---------|------|-------------|
| Mosquitto | 1883 | MQTT broker (anchors connect here) |
| InfluxDB | 8086 | Time-series database |
| PostgreSQL | 5432 | Relational database |
| Inference | 8000 | FastAPI (position API) |
| Dashboard | 8501 | Streamlit (visualization) |

## Project Status

**Current Phase:** Hardware Setup & Data Collection

### Hardware Checklist

| Component | Qty Needed | Status |
|-----------|------------|--------|
| BlueCharm BC021 beacon | 2 (1 + backup) | 1 acquired |
| M5Stack AtomS3 Lite | 11-12 | 1 acquired |
| USB-C wall adapters | 12 | To order |
| Raspberry Pi 4/5 | 1 | 1 acquired |
| CR2032 batteries | 4+ | To order |

### Implementation Phases

**Phase 1: Hardware Setup**
- Flash ESPresense firmware on AtomS3 Lite
- Deploy anchors throughout house
- Set up MQTT broker on Raspberry Pi
- Configure beacon advertising interval and TX power

**Phase 2: Data Collection**
- Build data collection script with location labeling
- Execute site survey (grid-based fingerprinting)
- Collect labeled dataset at key locations

**Phase 3: Model Development**
- Implement Kalman filter for signal smoothing
- Train fingerprinting classifier (Random Forest/SVM)
- Build particle filter with ML observation model
- Implement floor transition HMM

**Phase 4: Inference Pipeline**
- Create FastAPI inference endpoint
- Integrate with MQTT stream
- Add map constraints (occupancy grid)

**Phase 5: Visualization**
- Build Streamlit dashboard
- Create floor plan overlays
- Add real-time position display and historical heatmaps

---

## Portfolio Highlights

This project demonstrates data science and ML engineering skills through:

1. **System Architecture:** End-to-end IoT data pipeline with MQTT
2. **Noise Analysis:** Raw vs. Kalman-filtered signal paths
3. **Floor Transition Model:** Probabilistic floor determination with HMM
4. **Signal Bleed Handling:** Non-Line-of-Sight (NLOS) interference in indoor environments
5. **Bayesian State Estimation:** Particle filter implementation for spatial inference

*Showcases data science and ML engineering skills: sensor fusion, probabilistic modeling, real-time inference pipelines, and handling noisy spatial data.*