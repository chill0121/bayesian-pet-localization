-- PostgreSQL initialization for pet tracking

-- Table for labeled fingerprinting data (site survey)
CREATE TABLE IF NOT EXISTS fingerprint_samples (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    location_label VARCHAR(50) NOT NULL,      -- e.g., 'couch', 'water_bowl', 'office_bed'
    floor INTEGER NOT NULL,
    grid_x FLOAT,                              -- optional grid coordinates
    grid_y FLOAT,
    rssi_vector JSONB NOT NULL,                -- {"office": -67, "hallway": -82, ...}
    features JSONB                             -- pre-computed features
);

-- Table for position state history
CREATE TABLE IF NOT EXISTS position_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    beacon_id VARCHAR(50) NOT NULL,
    estimated_x FLOAT NOT NULL,
    estimated_y FLOAT NOT NULL,
    estimated_floor INTEGER NOT NULL,
    confidence FLOAT,
    location_label VARCHAR(50),                -- classified zone/room
    activity_state VARCHAR(20),                -- 'sleeping', 'moving', 'stationary'
    raw_rssi JSONB,                            -- snapshot of RSSI at this moment
    particle_count INTEGER
);

-- Table for anchor configuration
CREATE TABLE IF NOT EXISTS anchors (
    id SERIAL PRIMARY KEY,
    anchor_id VARCHAR(50) UNIQUE NOT NULL,     -- e.g., 'office', 'kitchen_nw'
    floor INTEGER NOT NULL,
    x FLOAT NOT NULL,                          -- physical coordinates
    y FLOAT NOT NULL,
    description VARCHAR(100),
    calibration_offset FLOAT DEFAULT 0,        -- RSSI calibration
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for floor definitions
CREATE TABLE IF NOT EXISTS floors (
    id SERIAL PRIMARY KEY,
    floor_number INTEGER UNIQUE NOT NULL,
    name VARCHAR(50),
    width_ft FLOAT,
    height_ft FLOAT,
    svg_path VARCHAR(200),                     -- path to floor plan SVG
    occupancy_grid JSONB                       -- walkable areas mask
);

-- Indexes for common queries
CREATE INDEX idx_position_timestamp ON position_history(timestamp);
CREATE INDEX idx_position_beacon ON position_history(beacon_id);
CREATE INDEX idx_fingerprint_location ON fingerprint_samples(location_label);
CREATE INDEX idx_fingerprint_floor ON fingerprint_samples(floor);

-- Insert initial floor data
INSERT INTO floors (floor_number, name, width_ft, height_ft) VALUES
    (1, 'First Floor', 14, 12),
    (2, 'Second Floor', 30, 15),
    (3, 'Third Floor', 14.5, 20)
ON CONFLICT (floor_number) DO NOTHING;
