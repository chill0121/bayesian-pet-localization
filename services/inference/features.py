"""
Feature Engineering Pipeline for Bayesian Pet Localization

Computes derived features from raw and Kalman-smoothed RSSI streams for
downstream ML models (Random Forest room classifier, activity inference).

Feature groups
--------------
1. **Delta RSSI** — per-anchor change between consecutive smoothed readings,
   plus aggregate statistics (mean, max absolute delta).
2. **Anchor Rankings** — ordinal rank of each anchor by signal strength,
   rank-change count, strongest anchor ID, RSSI gap between top two.
3. **Rolling Variance** — per-anchor raw-RSSI variance over a sliding window,
   plus aggregate stats.  Captures signal stability (stationary) vs turbulence
   (moving).
4. **Cross-Floor Attenuation** — per-floor anchor counts and mean RSSI, best-
   floor indicator, same-vs-cross floor signal ratio.
5. **Composite / Activity** — number of reporting anchors and a 0–1 activity
   score derived from variance and delta features.

Usage::

    engine = FeatureEngine(anchor_positions=anchor_coords, window_size=30)
    # Each inference cycle:
    features = engine.update(raw_rssi, smoothed_rssi, timestamp)
"""

from collections import defaultdict, deque
from typing import Optional

# ---------------------------------------------------------------------------
# Activity-score tuning constants
# ---------------------------------------------------------------------------

# Variance normalisation ceiling: var_mean at this value maps to 1.0
VARIANCE_NORMALISATION_MAX = 20.0   # dBm²
# Delta normalisation ceiling: delta_max at this value maps to 1.0
DELTA_NORMALISATION_MAX = 5.0       # dBm
# Blend weights for the composite activity score (must sum to 1.0)
ACTIVITY_VAR_WEIGHT = 0.6
ACTIVITY_DELTA_WEIGHT = 0.4


class FeatureEngine:
    """Stateful feature extractor for RSSI-based localisation.

    Maintains a rolling window of raw RSSI per anchor to compute
    temporal statistics.  Each call to :meth:`update` ingests the
    latest RSSI vectors and returns a flat dict of named features.

    Parameters
    ----------
    anchor_positions : dict[str, dict]
        anchor_id → {"x": float, "y": float, "floor": int}.
        Used for cross-floor attenuation features.
    window_size : int
        Number of *distinct* raw-RSSI samples to keep per anchor for
        rolling statistics.  Only genuinely new values (i.e. the raw
        value changed from the previous call) are appended, so the
        window corresponds to real ESPresense publishes (~1 Hz per
        anchor).  Default 10 ≈ 10 s of per-anchor data.
    """

    def __init__(
        self,
        anchor_positions: Optional[dict[str, dict]] = None,
        window_size: int = 10,
    ):
        self._anchors = anchor_positions or {}
        self._window_size = window_size

        # Per-anchor rolling buffer of raw RSSI values
        self._history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Per-anchor last raw value — used to detect genuinely new readings
        # vs stale repeats from the rssi_buffer
        self._last_raw: dict[str, float] = {}

        # Previous step state (for delta / ranking change computation)
        self._prev_smoothed: dict[str, float] = {}
        self._prev_rankings: list[str] = []

        # Latest computed features
        self._features: dict[str, float | int | str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_rssi: dict[str, float],
        smoothed_rssi: dict[str, float],
        timestamp: float,
    ) -> dict:
        """Ingest new RSSI data and compute all feature groups.

        Parameters
        ----------
        raw_rssi : dict[str, float]
            anchor_id → raw RSSI (dBm) from the current MQTT batch.
        smoothed_rssi : dict[str, float]
            anchor_id → Kalman-smoothed RSSI (dBm).
        timestamp : float
            Current time (seconds, e.g. ``time.time()``).

        Returns
        -------
        dict
            Flat dictionary of feature name → value.
        """
        # Store raw values in per-anchor rolling window — only when the
        # value actually changed (avoids filling the window with stale
        # repeats from the rssi_buffer, which contains ALL anchors on
        # every call even though only one anchor updated via MQTT).
        for anchor_id, rssi in raw_rssi.items():
            prev = self._last_raw.get(anchor_id)
            if prev is None or rssi != prev:
                self._history[anchor_id].append(rssi)
                self._last_raw[anchor_id] = rssi

        features: dict = {}
        features.update(self._delta_features(smoothed_rssi))
        features.update(self._ranking_features(smoothed_rssi))
        features.update(self._variance_features(smoothed_rssi))
        features.update(self._cross_floor_features(smoothed_rssi))
        features.update(self._composite_features(smoothed_rssi, features))

        # Advance state for next step
        self._prev_smoothed = dict(smoothed_rssi)
        self._features = features
        return features

    @property
    def features(self) -> dict:
        """Most recently computed feature dict (read-only convenience)."""
        return dict(self._features)

    # ------------------------------------------------------------------
    # 1. Delta RSSI
    # ------------------------------------------------------------------

    def _delta_features(self, smoothed: dict[str, float]) -> dict:
        """Per-anchor RSSI delta from previous step + aggregates."""
        deltas: dict[str, float] = {}
        abs_deltas: list[float] = []

        for anchor_id, rssi in smoothed.items():
            prev = self._prev_smoothed.get(anchor_id)
            if prev is not None:
                d = rssi - prev
                deltas[f"delta_rssi_{anchor_id}"] = round(d, 3)
                abs_deltas.append(abs(d))

        out: dict = dict(deltas)
        if abs_deltas:
            out["delta_mean"] = round(sum(abs_deltas) / len(abs_deltas), 3)
            out["delta_max"] = round(max(abs_deltas), 3)
        else:
            out["delta_mean"] = 0.0
            out["delta_max"] = 0.0
        return out

    # ------------------------------------------------------------------
    # 2. Anchor Rankings
    # ------------------------------------------------------------------

    def _ranking_features(self, smoothed: dict[str, float]) -> dict:
        """Ordinal anchor rankings by RSSI strength."""
        if not smoothed:
            return {
                "rank_changes": 0,
                "strongest_anchor": "none",
                "rssi_gap_1_2": 0.0,
            }

        # Sort anchors strongest (highest RSSI) first
        sorted_anchors = sorted(smoothed.keys(), key=lambda a: smoothed[a], reverse=True)

        out: dict = {}
        for rank, anchor_id in enumerate(sorted_anchors, start=1):
            out[f"rank_{anchor_id}"] = rank

        # Rank change count
        if self._prev_rankings:
            prev_set = set(self._prev_rankings)
            curr_set = set(sorted_anchors)
            # Only count anchors present in both steps
            common = prev_set & curr_set
            prev_rank = {a: i for i, a in enumerate(self._prev_rankings)}
            curr_rank = {a: i for i, a in enumerate(sorted_anchors)}
            changes = sum(1 for a in common if prev_rank[a] != curr_rank[a])
            out["rank_changes"] = changes
        else:
            out["rank_changes"] = 0

        out["strongest_anchor"] = sorted_anchors[0]

        # Gap between 1st and 2nd strongest
        if len(sorted_anchors) >= 2:
            out["rssi_gap_1_2"] = round(
                smoothed[sorted_anchors[0]] - smoothed[sorted_anchors[1]], 3
            )
        else:
            out["rssi_gap_1_2"] = 0.0

        self._prev_rankings = sorted_anchors
        return out

    # ------------------------------------------------------------------
    # 3. Rolling Variance
    # ------------------------------------------------------------------

    def _variance_features(self, smoothed: dict[str, float]) -> dict:
        """Per-anchor raw-RSSI variance over the rolling window.

        Only anchors currently present in *smoothed* (i.e. not stale)
        are included.  This prevents dropped anchors from ghosting in
        the aggregate stats with a permanent 0.0 variance.
        """
        out: dict = {}
        variances: list[float] = []

        for anchor_id in smoothed:
            buf = self._history.get(anchor_id)
            if buf is None or len(buf) < 2:
                out[f"var_{anchor_id}"] = 0.0
                continue
            mean = sum(buf) / len(buf)
            var = sum((v - mean) ** 2 for v in buf) / (len(buf) - 1)
            var = round(var, 3)
            out[f"var_{anchor_id}"] = var
            variances.append(var)

        if variances:
            out["var_mean"] = round(sum(variances) / len(variances), 3)
            out["var_max"] = round(max(variances), 3)
        else:
            out["var_mean"] = 0.0
            out["var_max"] = 0.0
        return out

    # ------------------------------------------------------------------
    # 4. Cross-Floor Attenuation
    # ------------------------------------------------------------------

    def _cross_floor_features(self, smoothed: dict[str, float]) -> dict:
        """Per-floor anchor count, mean RSSI, and best-floor indicator."""
        # Group smoothed RSSI by anchor's floor
        floor_rssi: dict[int, list[float]] = defaultdict(list)
        for anchor_id, rssi in smoothed.items():
            anchor = self._anchors.get(anchor_id)
            if anchor is not None:
                floor_rssi[anchor["floor"]].append(rssi)

        out: dict = {}
        floor_means: dict[int, float] = {}

        for floor, values in sorted(floor_rssi.items()):
            out[f"n_anchors_floor_{floor}"] = len(values)
            mean = sum(values) / len(values)
            out[f"rssi_mean_floor_{floor}"] = round(mean, 2)
            floor_means[floor] = mean

        if floor_means:
            best_floor = max(floor_means, key=floor_means.get)
            out["rssi_best_floor"] = best_floor

            # Same-vs-cross ratio: best floor mean minus mean of all other floors
            other_values = [m for f, m in floor_means.items() if f != best_floor]
            if other_values:
                other_mean = sum(other_values) / len(other_values)
                out["same_vs_cross_ratio"] = round(
                    floor_means[best_floor] - other_mean, 2
                )
            else:
                out["same_vs_cross_ratio"] = 0.0
        else:
            out["rssi_best_floor"] = 0
            out["same_vs_cross_ratio"] = 0.0

        return out

    # ------------------------------------------------------------------
    # 5. Composite / Activity
    # ------------------------------------------------------------------

    def _composite_features(
        self, smoothed: dict[str, float], computed: dict
    ) -> dict:
        """Aggregate features: reporting anchor count and activity score."""
        n_reporting = len(smoothed)

        # Activity score: 0 (sleeping) → 1 (moving)
        # Blend of rolling variance and delta magnitude.
        var_mean = computed.get("var_mean", 0.0)
        delta_max = computed.get("delta_max", 0.0)

        # Normalize each component to [0, 1] based on tuning ceilings, then blend
        var_component = min(var_mean / VARIANCE_NORMALISATION_MAX, 1.0)
        delta_component = min(delta_max / DELTA_NORMALISATION_MAX, 1.0)
        # Weighted blend (variance is more reliable for stationary vs moving, so higher weight)
        activity_score = round(
            ACTIVITY_VAR_WEIGHT * var_component + ACTIVITY_DELTA_WEIGHT * delta_component, 3
        )

        return {
            "n_reporting": n_reporting,
            "activity_score": activity_score,
        }
