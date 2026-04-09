"""
Floor Transition Hidden Markov Model

Maintains a Bayesian belief distribution over which floor the pet
occupies, enforcing physical stairway connectivity constraints.

States
------
One discrete state per floor in the building (e.g. floors 1, 2, 3).

Transition model
----------------
A row-stochastic matrix ``T`` where ``T[i, j]`` = P(floor_j | floor_i):

* Self-transition (stay on floor):  dominant probability (~0.97).
* Adjacent-floor transitions:       allowed only between floors
  connected by a stairway in layout.json.
* Non-adjacent floor transitions:   probability 0.
  (e.g. floor 1 → 3 is impossible — must pass through 2.)

When the pet's estimated position is near a stairway entry, the
transition probability toward the connected floor is boosted via
a logistic proximity factor.

Emission (observation) model
----------------------------
For each floor hypothesis *F*, take the **best (strongest) RSSI per
anchor-floor** and fit the expected cross-floor attenuation pattern.
The unknown base signal level (affected by distance + beacon occlusion)
is analytically profiled out, so only *relative* signal differences
between floors matter.  This makes the model robust to global signal
depression (e.g. a sleeping pet occluding the beacon).

Forward algorithm
-----------------
Each ``step()`` call runs:

  belief[t]  =  normalize( emission  ⊙  (Tᵀ @ belief[t−1]) )

Integration
-----------
The particle filter can read ``floor_belief`` to modulate its own
floor-transition probabilities or re-weight particles across floors.

Pipeline::

    hmm = FloorTransitionHMM(layout_data, anchor_positions)
    # each RSSI batch:
    hmm.step(rssi_readings, dt, stair_proximity)
    # read result:
    hmm.floor_belief   → {1: 0.05, 2: 0.90, 3: 0.05}
    hmm.most_likely_floor  → 2
"""

import math

import numpy as np

from .constants import (
    TX_POWER_DBM,
    PATH_LOSS_N,
    FLOOR_ATTENUATION_DB,
    MIN_DISTANCE_FT,
)

# ---------------------------------------------------------------------------
# HMM-specific constants
# ---------------------------------------------------------------------------

# Base transition probability per second for adjacent floors
BASE_TRANSITION_RATE = 0.01      # ~1 % per second when near stairs

# Self-transition floor (per step at dt = 1 s)
BASE_SELF_PROB = 0.97

# Emission model
EMISSION_SIGMA_DBM = 8.0            # per-floor residual noise scale

# Stairway proximity boost
STAIR_PROXIMITY_FT = 6.0            # radius for proximity scaling
STAIR_BOOST_FACTOR = 5.0            # max multiplier on transition prob
                                    # when standing right at the stairway

# Safety cap on per-step transition probability (prevents numerical blow-up)
MAX_TRANSITION_PROB_CAP = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_rssi(distance_ft: float, floor_diff: int = 0) -> float:
    """Log-distance path-loss expected RSSI (mirrors particle.py).

    Uses a direct dB floor attenuation penalty that is independent of
    PATH_LOSS_N, avoiding over-penalising cross-floor anchors when N
    is large.
    """
    distance_ft = max(distance_ft, MIN_DISTANCE_FT)
    distance_m = distance_ft * 0.3048
    return (TX_POWER_DBM
            - 10.0 * PATH_LOSS_N * math.log10(distance_m)
            - floor_diff * FLOOR_ATTENUATION_DB)


def _emission_log_likelihood(
    rssi_readings: dict[str, float],
    anchor_positions: dict[str, dict],
    floor_hypothesis: int,
) -> float:
    """Compute log-likelihood of RSSI observations given a floor hypothesis.

    Uses a per-floor best-signal model that analytically profiles out the
    unknown base signal strength (distance + occlusion), so only the
    *relative* RSSI pattern across floors matters.

    Model
    -----
    best_rssi[g] = base - |f - g| * FLOOR_ATTENUATION_DB + noise

    where *base* is the nuisance parameter (marginalized out) and *f* is
    the hypothesized floor.  The MLE of *base* for hypothesis *f* is::

        base_hat = mean( best_rssi[g] + |f - g| * ATTEN )

    The log-likelihood is then the sum of squared residuals under this
    best-fit base, scaled by ``EMISSION_SIGMA_DBM``.
    """
    # 1. Best (strongest) RSSI per anchor-floor
    best_by_floor: dict[int, float] = {}
    for anchor_id, observed_rssi in rssi_readings.items():
        if anchor_id not in anchor_positions:
            continue
        a_floor = anchor_positions[anchor_id]["floor"]
        if a_floor not in best_by_floor or observed_rssi > best_by_floor[a_floor]:
            best_by_floor[a_floor] = observed_rssi

    if not best_by_floor:
        return 0.0

    n_floors = len(best_by_floor)

    # With only one floor of data, no relative pattern exists — return 0
    # (uninformative) and let the transition model carry the belief.
    if n_floors < 2:
        return 0.0

    # 2. Profile out the unknown base signal
    adjusted = [
        best_by_floor[g] + abs(floor_hypothesis - g) * FLOOR_ATTENUATION_DB
        for g in best_by_floor
    ]
    base_hat = sum(adjusted) / n_floors

    # 3. Sum of squared residuals
    ll = 0.0
    for g, best_rssi in best_by_floor.items():
        floor_diff = abs(floor_hypothesis - g)
        expected = base_hat - floor_diff * FLOOR_ATTENUATION_DB
        residual = best_rssi - expected
        ll += -0.5 * (residual / EMISSION_SIGMA_DBM) ** 2

    return ll


def _build_adjacency(layout_data: dict) -> dict[int, set[int]]:
    """Parse layout.json stairs entries → floor adjacency graph.

    Returns
    -------
    dict  mapping  floor_number → set of directly-connected floor numbers.
    """
    adjacency: dict[int, set[int]] = {}
    for floor_data in layout_data.get("floors", []):
        f = floor_data["floor"]
        adjacency.setdefault(f, set())
        for stair in floor_data.get("stairs", []):
            to_f = stair["to_floor"]
            adjacency[f].add(to_f)
            adjacency.setdefault(to_f, set())
            adjacency[to_f].add(f)
    return adjacency


def _parse_stair_entries(layout_data: dict) -> list[dict]:
    """Extract stair entry points from layout.json.

    Returns list of:
        {"from_floor": int, "to_floor": int,
         "entry_x": float, "entry_y": float}
    """
    entries = []
    for floor_data in layout_data.get("floors", []):
        from_floor = floor_data["floor"]
        for stair in floor_data.get("stairs", []):
            entry = stair.get("entry", [0, 0])
            entries.append({
                "from_floor": from_floor,
                "to_floor": stair["to_floor"],
                "entry_x": entry[0],
                "entry_y": entry[1],
            })
    return entries


# ---------------------------------------------------------------------------
# FloorTransitionHMM
# ---------------------------------------------------------------------------

class FloorTransitionHMM:
    """Hidden Markov Model for floor-level localisation.

    Parameters
    ----------
    layout_data : dict
        Parsed layout.json content (the top-level dict).
    anchor_positions : dict[str, dict]
        anchor_id → {"x": float, "y": float, "floor": int}.
    initial_floor : int | None
        If given, initialise belief concentrated on this floor.
        Otherwise, uniform prior.
    """

    def __init__(
        self,
        layout_data: dict,
        anchor_positions: dict[str, dict],
        initial_floor: int | None = None,
    ):
        self._anchors = anchor_positions
        self._adjacency = _build_adjacency(layout_data)
        self._stair_entries = _parse_stair_entries(layout_data)

        # Ordered list of floor IDs
        self._floors = sorted(self._adjacency.keys())
        self._n = len(self._floors)
        self._floor_idx = {f: i for i, f in enumerate(self._floors)}

        # Belief vector  (index → floor via self._floors[i])
        if initial_floor is not None and initial_floor in self._floor_idx:
            self._belief = np.zeros(self._n)
            self._belief[self._floor_idx[initial_floor]] = 1.0
        else:
            self._belief = np.ones(self._n) / self._n

        # Build base transition matrix (no stairway proximity boost)
        self._base_T = self._build_transition_matrix()

    # ------------------------------------------------------------------
    # Transition matrix construction
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> np.ndarray:
        """Construct the base row-stochastic transition matrix.

        ``T[i, j]`` = P(floor_j at t+1  |  floor_i at t), for dt = 1 s.

        * Self-transition:  BASE_SELF_PROB
        * Adjacent floors:  remaining probability split equally
        * Non-adjacent:     0
        """
        T = np.zeros((self._n, self._n))
        for i, fi in enumerate(self._floors):
            neighbours = self._adjacency.get(fi, set())
            n_neighbours = len(neighbours)
            if n_neighbours == 0:
                T[i, i] = 1.0
            else:
                off_diag = (1.0 - BASE_SELF_PROB) / n_neighbours
                T[i, i] = BASE_SELF_PROB
                for fj in neighbours:
                    j = self._floor_idx[fj]
                    T[i, j] = off_diag
        return T

    def _transition_matrix(
        self, dt: float, stair_proximity: dict[int, float] | None = None
    ) -> np.ndarray:
        """Return transition matrix scaled by dt and stairway proximity.

        Parameters
        ----------
        dt : float
            Seconds since last step.
        stair_proximity : dict[int, float] | None
            floor → minimum distance (ft) from estimated position to the
            nearest stairway entry on that floor.  If provided, proximity
            boosts adjacent-floor transition probability.

        Returns
        -------
        np.ndarray, shape (n_floors, n_floors)
            Row-stochastic transition matrix.
        """
        # Scale base off-diagonal rates by dt
        # P_transition(dt) ≈ 1 - exp(-rate * dt) ≈ rate * dt  for small dt
        rate = BASE_TRANSITION_RATE * dt

        T = np.zeros((self._n, self._n))
        for i, fi in enumerate(self._floors):
            neighbours = self._adjacency.get(fi, set())
            if not neighbours:
                T[i, i] = 1.0
                continue

            # Compute per-neighbour transition probability
            for fj in neighbours:
                j = self._floor_idx[fj]
                p = rate

                # Proximity boost: logistic ramp as distance → 0
                if stair_proximity is not None and fi in stair_proximity:
                    dist = stair_proximity[fi]
                    # boost = STAIR_BOOST_FACTOR when dist=0, → 1.0 far away
                    boost = 1.0 + (STAIR_BOOST_FACTOR - 1.0) / (
                        1.0 + (dist / STAIR_PROXIMITY_FT) ** 2
                    )
                    p *= boost

                T[i, j] = min(p, MAX_TRANSITION_PROB_CAP)

            T[i, i] = 1.0 - T[i].sum()
            # Guard: if numerical error makes self < 0, renormalize
            if T[i, i] < 0:
                T[i] /= T[i].sum()

        return T

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _emission_probs(self, rssi_readings: dict[str, float]) -> np.ndarray:
        """Compute normalized emission probabilities for each floor.

        Returns
        -------
        np.ndarray, shape (n_floors,)
            normalized emission likelihoods.
        """
        log_liks = np.array([
            _emission_log_likelihood(rssi_readings, self._anchors, f)
            for f in self._floors
        ])
        # Numerically stable softmax
        log_liks -= log_liks.max()
        probs = np.exp(log_liks)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(self._n) / self._n

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def predict(
        self, dt: float, stair_proximity: dict[int, float] | None = None
    ) -> None:
        """Forward-predict the belief through the transition model.

        Parameters
        ----------
        dt : float
            Seconds since last step.
        stair_proximity : dict[int, float] | None
            floor → min distance to nearest stairway entry.
        """
        T = self._transition_matrix(dt, stair_proximity)
        # belief_new = T^T @ belief  (column convention)
        self._belief = T.T @ self._belief
        # normalize (guard against numerical drift)
        total = self._belief.sum()
        if total > 0:
            self._belief /= total

    def update(self, rssi_readings: dict[str, float]) -> None:
        """Update belief with RSSI emission evidence.

        Parameters
        ----------
        rssi_readings : dict[str, float]
            anchor_id → RSSI (dBm).
        """
        if not rssi_readings:
            return
        emission = self._emission_probs(rssi_readings)
        self._belief *= emission
        total = self._belief.sum()
        if total > 0:
            self._belief /= total
        else:
            # Complete collapse — reset to uniform
            self._belief[:] = 1.0 / self._n

    def step(
        self,
        rssi_readings: dict[str, float],
        dt: float,
        stair_proximity: dict[int, float] | None = None,
    ) -> dict[int, float]:
        """Full predict → update cycle.

        Parameters
        ----------
        rssi_readings : dict[str, float]
            anchor_id → RSSI (dBm).
        dt : float
            Seconds since last step.
        stair_proximity : dict[int, float] | None
            floor → min distance to nearest stairway entry.

        Returns
        -------
        dict[int, float]
            Floor belief distribution (same as ``floor_belief``).
        """
        self.predict(dt, stair_proximity)
        self.update(rssi_readings)
        return self.floor_belief

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def floor_belief(self) -> dict[int, float]:
        """Current belief distribution over floors.

        Returns
        -------
        dict[int, float]
            floor_number → probability.
        """
        return {f: float(self._belief[i]) for i, f in enumerate(self._floors)}

    @property
    def most_likely_floor(self) -> int:
        """Floor with the highest current probability."""
        return self._floors[int(np.argmax(self._belief))]

    @property
    def floors(self) -> list[int]:
        """Ordered list of floor numbers in the model."""
        return list(self._floors)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Base transition matrix (for inspection / visualisation)."""
        return self._base_T.copy()

    @property
    def stair_entries(self) -> list[dict]:
        """Parsed stairway entry points."""
        return list(self._stair_entries)

    def stair_proximity_for_position(
        self, x: float, y: float, floor: int
    ) -> dict[int, float]:
        """Compute distance from (x, y, floor) to nearest stair entry per floor.

        Useful for the particle filter to pass stair_proximity into ``step()``.

        Parameters
        ----------
        x, y : float
            Position in feet.
        floor : int
            Current floor.

        Returns
        -------
        dict[int, float]
            floor → min distance to a stairway entry on that floor.
            Only includes floors that have stairway entries.
        """
        proximity: dict[int, float] = {}
        for stair in self._stair_entries:
            sf = stair["from_floor"]
            # Only compute for the floor we're actually on
            if sf != floor:
                continue
            dist = math.sqrt(
                (x - stair["entry_x"]) ** 2 + (y - stair["entry_y"]) ** 2
            )
            if sf not in proximity or dist < proximity[sf]:
                proximity[sf] = dist
        return proximity

    def __repr__(self) -> str:
        belief = self.floor_belief
        parts = [f"F{f}={p:.2f}" for f, p in sorted(belief.items())]
        return f"FloorTransitionHMM({', '.join(parts)})"
