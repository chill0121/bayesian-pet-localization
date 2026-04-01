-- PostgreSQL initialization for pet tracking
-- Schema version 2.0 — aligned with layout.json v3.0 and inference pipeline

-- ---------------------------------------------------------------------------
-- Floor definitions (from layout.json)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS floors (
    id SERIAL PRIMARY KEY,
    floor_number INTEGER UNIQUE NOT NULL,
    name VARCHAR(100),
    outer_boundary JSONB,                      -- [[x,y], ...] CCW polygon from layout.json
    ceiling_height_ft FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Anchor configuration (BLE receivers)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anchors (
    id SERIAL PRIMARY KEY,
    anchor_id VARCHAR(50) UNIQUE NOT NULL,     -- e.g., '1F_Office', 'living_center'
    floor INTEGER NOT NULL REFERENCES floors(floor_number),
    x FLOAT NOT NULL,                          -- feet from SW origin
    y FLOAT NOT NULL,
    height_ft FLOAT,                           -- mount height
    calibration_offset FLOAT DEFAULT 0,        -- per-anchor RSSI correction (future)
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Position history (pipeline output — written every inference cycle)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS position_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    beacon_id VARCHAR(100) NOT NULL,
    estimated_x FLOAT NOT NULL,
    estimated_y FLOAT NOT NULL,
    estimated_floor INTEGER NOT NULL,
    confidence FLOAT,
    location_label VARCHAR(50),                -- room or zone name (e.g., 'kitchen')
    activity_state VARCHAR(20),                -- 'sleeping', 'moving', 'stationary' (TODO #13)
    raw_rssi JSONB,                            -- {"anchor_id": rssi, ...}
    smoothed_rssi JSONB,                       -- Kalman-filtered RSSI vector
    n_eff FLOAT,                               -- particle filter effective sample size
    particle_count INTEGER,
    floor_belief JSONB                         -- HMM belief distribution {floor: prob}
);

-- Partitioning-friendly index on timestamp (for future table partitioning)
CREATE INDEX idx_position_timestamp ON position_history(timestamp);
CREATE INDEX idx_position_beacon ON position_history(beacon_id);
CREATE INDEX idx_position_floor_time ON position_history(estimated_floor, timestamp);

-- ---------------------------------------------------------------------------
-- Labeled fingerprint samples (site survey — TODO #8, #19)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fingerprint_samples (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    location_label VARCHAR(50) NOT NULL,       -- room or zone name (e.g., 'kitchen')
    zone_label VARCHAR(50),                    -- sub-zone name (e.g., 'office_dog_bed')
    room VARCHAR(50),                          -- parent room name (e.g., 'office')
    floor INTEGER NOT NULL REFERENCES floors(floor_number),
    grid_x FLOAT NOT NULL,                     -- survey grid position (feet)
    grid_y FLOAT NOT NULL,
    rssi_vector JSONB NOT NULL,                -- {"anchor_id": mean_rssi, ...}
    rssi_std JSONB,                            -- {"anchor_id": std_dev, ...}
    features JSONB,                            -- pre-computed feature vector (TODO #3)
    duration_seconds FLOAT,                    -- collection dwell time
    n_readings INTEGER,                        -- number of raw readings averaged
    notes TEXT                                 -- free-form annotation
);

CREATE INDEX idx_fingerprint_location ON fingerprint_samples(location_label);
CREATE INDEX idx_fingerprint_floor ON fingerprint_samples(floor);

-- ---------------------------------------------------------------------------
-- Trained model registry (TODO #6, #21)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,           -- 'random_forest', 'particle_tuned', etc.
    version VARCHAR(20) NOT NULL,
    artifact_path VARCHAR(200),                -- path to serialized model file
    metrics JSONB,                             -- {"accuracy": 0.92, "cv_score": 0.89, ...}
    hyperparameters JSONB,                     -- training config snapshot
    n_training_samples INTEGER,
    trained_at TIMESTAMPTZ DEFAULT NOW(),
    active BOOLEAN DEFAULT FALSE               -- only one active per model_type
);

CREATE UNIQUE INDEX idx_model_active
    ON model_versions(model_type) WHERE active = TRUE;

-- ---------------------------------------------------------------------------
-- Seed floor data from layout.json
-- ---------------------------------------------------------------------------
INSERT INTO floors (floor_number, name, outer_boundary) VALUES
    (1, 'First Floor',  '[[0,29.39],[17.7,29.39],[17.7,2.07],[6.85,2.07],[6.85,0],[0,0]]'),
    (2, 'Second Floor', '[[0,29.39],[17.7,29.39],[17.7,2.07],[6.85,2.07],[6.85,0],[0,0]]'),
    (3, 'Third Floor',  '[[0,29.39],[17.7,29.39],[17.7,2.07],[6.85,2.07],[6.85,0],[0,0]]')
ON CONFLICT (floor_number) DO NOTHING;

-- Seed anchor positions from layout.json
INSERT INTO anchors (anchor_id, floor, x, y, height_ft) VALUES
    ('1f_office',         1,  3.77, 11.31, 1.06),
    ('1f_hallway',        1,  7.01, 18.67, 1.07),
    ('2f_living_sw',      2,  4.64,  0.00, 1.20),
    ('2f_living_center',  2,  3.47, 19.60, 1.01),
    ('2f_kitchen_ne',     2, 17.70, 20.60, 3.71),
    ('3f_master_bed',     3, 10.67, 14.09, 1.14),
    ('3f_hallway',        3,  3.35, 19.84, 1.03),
    ('3f_office',         3, 10.38, 19.48, 1.02)
ON CONFLICT (anchor_id) DO NOTHING;
