"""
Shared physical-model constants for the inference pipeline.

These values are used by multiple modules (particle filter, floor HMM,
features, etc.) and are centralized here to avoid duplication and
ensure consistency when tuning.
"""

# ---------------------------------------------------------------------------
# RSSI path-loss model
# ---------------------------------------------------------------------------
TX_POWER_DBM: float = -59.0          # measured BLE transmit power at 1 m (dBm)
PATH_LOSS_N: float = 2.7             # path-loss exponent (indoor BLE)
RSSI_SIGMA: float = 5.0              # observation noise std-dev (dBm)
CROSS_FLOOR_PENALTY_FT: float = 30.0 # effective extra distance per floor diff (ft)
MIN_DISTANCE_FT: float = 1.0         # clamp minimum distance to avoid log(0) (ft)
