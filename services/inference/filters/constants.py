"""
Shared physical-model constants for the inference pipeline.

These values are used by multiple modules (particle filter, floor HMM,
features, etc.) and are centralized here to avoid duplication and
ensure consistency when tuning.
"""

# ---------------------------------------------------------------------------
# RSSI path-loss model
# ---------------------------------------------------------------------------
TX_POWER_DBM: float = -60.0          # measured BLE transmit power at 1 m (dBm) — BC021 Pro at +4dBm TX
PATH_LOSS_N: float = 3.5             # path-loss exponent (indoor residential)
RSSI_SIGMA: float = 5.0              # observation noise std-dev (dBm)
CROSS_FLOOR_PENALTY_FT: float = 30.0 # (deprecated) kept for floor_hmm compat
FLOOR_ATTENUATION_DB: float = 10.0   # dB penalty per floor crossed (floor slab attenuation)
MIN_DISTANCE_FT: float = 1.0         # clamp minimum distance to avoid log(0) (ft)
WALL_ATTENUATION_DB: float = 10.0    # RSSI penalty per interior wall crossing (dBm)

# Floor elevations (measured: 14 risers × 7.5" = 8.75 ft per run)
FLOOR_ELEVATION_FT: dict[int, float] = {1: 0.0, 2: 8.75, 3: 17.5}
