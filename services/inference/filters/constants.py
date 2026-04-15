"""
Shared physical-model constants for the inference pipeline.

These values are used by multiple modules (particle filter, floor HMM,
features, etc.) and are centralized here to avoid duplication and
ensure consistency when tuning.
"""

# ---------------------------------------------------------------------------
# RSSI path-loss model
# ---------------------------------------------------------------------------
TX_POWER_DBM: float = -66.0          # measured BLE transmit power at 1 m (dBm) — site survey fit (2026-04-14)
PATH_LOSS_N: float = 2.4             # path-loss exponent — site survey fit (compact residential)
RSSI_SIGMA: float = 5.0              # observation noise std-dev (dBm) — confirmed by survey residuals
CROSS_FLOOR_PENALTY_FT: float = 30.0 # (deprecated) kept for floor_hmm compat
FLOOR_ATTENUATION_DB: float = 6.0    # dB penalty per floor crossed — survey: 5.4 (1F) to 6.9 (2F avg)
MIN_DISTANCE_FT: float = 1.0         # clamp minimum distance to avoid log(0) (ft)
WALL_ATTENUATION_DB: float = 6.0     # RSSI penalty per interior wall crossing — survey: ~6.4 dB measured

# Floor elevations (measured: 14 risers × 7.5" = 8.75 ft per run)
FLOOR_ELEVATION_FT: dict[int, float] = {1: 0.0, 2: 8.75, 3: 17.5}

# ---------------------------------------------------------------------------
# Zone-likelihood reweighting
# ---------------------------------------------------------------------------
ZONE_LIKELIHOOD_STRENGTH: float = 0.3  # exponent dampening (0=off, 1=full Bayesian)
