"""
Particle Filter for Indoor Pet Localization

Bayesian state estimation using Sequential Monte Carlo (particle filter).
Maintains a cloud of weighted position hypotheses that are propagated through
a motion model and updated with RSSI observations.  Wall constraints from the
occupancy grid prevent particles from passing through walls.

State per particle: (x, y, floor)
    - x, y : position in feet (layout.json coordinate system)
    - floor : integer floor number (1, 2, 3)

Observation model:
    Log-distance path loss converts each particle's distance to each anchor
    into an expected RSSI.  The likelihood of the observed RSSI under a
    Gaussian centred on that expected value weights the particle.  Cross-floor
    anchors incur extra effective distance to model floor/ceiling attenuation.

Motion model:
    Gaussian random walk scaled by dt and a maximum dog speed (~1 m/s ≈ 3.28
    ft/s).  Proposed moves that cross walls (occupancy grid ``ray_clear``)
    or land in blocked cells are rejected (particle stays put).

Floor transitions:
    Particles near a stairway entry point may transition to the connected
    floor with a small probability.  The stairway definitions come from
    layout.json ``stairs`` entries.

Pipeline integration:
    1. Create ``ParticleFilter`` with occupancy grids and anchor positions
    2. On each RSSI batch: ``pf.update(rssi_dict, dt)``
    3. Read estimate:       ``pf.estimate``  →  {x, y, floor, confidence}
"""

import math
from typing import Optional

import numpy as np

from .floor_hmm import FloorTransitionHMM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dog motion budget
DOG_MAX_SPEED_FT = 3.28          # ~1 m/s in feet/s
DOG_SPEED_SIGMA_FT = 1.5         # std-dev of per-axis displacement per second

# RSSI path-loss model (matches simulate_mqtt.py)
TX_POWER_DBM = -59.0             # measured power at 1 m
PATH_LOSS_N = 2.7                # path loss exponent (indoor BLE)
RSSI_SIGMA = 5.0                 # observation noise std-dev (dBm) — loose to
                                 # account for multipath / body shielding
CROSS_FLOOR_PENALTY_FT = 30.0    # effective extra distance per floor difference
MIN_DISTANCE_FT = 1.0            # clamp minimum distance (avoids log(0))

# Resampling
RESAMPLE_THRESHOLD_RATIO = 0.5   # resample when N_eff < N * ratio

# Floor transition
STAIR_PROXIMITY_FT = 4.0         # how close a particle must be to a stair entry
FLOOR_TRANSITION_PROB = 0.02     # base probability of changing floor per step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_rssi(distance_ft: float, floor_diff: int = 0) -> float:
    """Compute expected RSSI from distance using log-distance path loss.

    Parameters
    ----------
    distance_ft : float
        Euclidean distance in feet between particle and anchor.
    floor_diff : int
        Absolute floor difference (0 = same floor).

    Returns
    -------
    float
        Expected RSSI in dBm.
    """
    effective_ft = distance_ft + floor_diff * CROSS_FLOOR_PENALTY_FT
    effective_ft = max(effective_ft, MIN_DISTANCE_FT)
    distance_m = effective_ft * 0.3048
    return TX_POWER_DBM - 10.0 * PATH_LOSS_N * math.log10(distance_m)


def _log_likelihood(observed_rssi: float, expected_rssi: float) -> float:
    """Log-likelihood of observing *observed_rssi* given *expected_rssi*.

    Assumes Gaussian observation noise with std-dev ``RSSI_SIGMA``.
    """
    diff = observed_rssi - expected_rssi
    return -0.5 * (diff / RSSI_SIGMA) ** 2


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling.  Returns an index array of length N.

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Normalised particle weights (sum to 1).
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int, shape (N,)
        Indices of resampled particles.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cum = np.cumsum(weights)
    indices = np.searchsorted(cum, positions)
    return indices


def _effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from normalised weights."""
    return 1.0 / np.sum(weights ** 2)


# ---------------------------------------------------------------------------
# ParticleFilter
# ---------------------------------------------------------------------------

class ParticleFilter:
    """Sequential Monte Carlo filter for 2-D + floor localisation.

    Parameters
    ----------
    occupancy_grids : OccupancyGridSet
        Pre-built occupancy grids (one per floor).
    anchor_positions : dict[str, dict]
        anchor_id → {"x": float, "y": float, "floor": int}.
    stairways : list[dict]
        Stairway connection list.  Each entry:
        {"from_floor": int, "to_floor": int, "entry_x": float, "entry_y": float}.
    floor_hmm : FloorTransitionHMM | None
        Optional floor-level HMM.  When provided, floor transition
        probabilities are modulated by the HMM's belief distribution
        instead of using a fixed probability.
    n_particles : int
        Number of particles.
    seed : int | None
        Random seed for reproducibility (None = random).
    """

    def __init__(
        self,
        occupancy_grids,
        anchor_positions: dict[str, dict],
        stairways: list[dict] | None = None,
        floor_hmm: FloorTransitionHMM | None = None,
        n_particles: int = 500,
        seed: int | None = None,
    ):
        self._grids = occupancy_grids
        self._anchors = anchor_positions
        self._stairways = stairways or []
        self._floor_hmm = floor_hmm
        self.n = n_particles
        self._rng = np.random.default_rng(seed)

        # Particle state arrays
        self._x = np.zeros(self.n)
        self._y = np.zeros(self.n)
        self._floor = np.ones(self.n, dtype=int)
        self._weights = np.full(self.n, 1.0 / self.n)

        self._initialised = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise_uniform(self, floor: int | None = None) -> None:
        """Spread particles uniformly across walkable space.

        Parameters
        ----------
        floor : int, optional
            Restrict initial placement to a single floor.  If None,
            particles are spread evenly across all floors.
        """
        floors = [floor] if floor is not None else list(self._grids.floors)
        placed = 0
        max_attempts = self.n * 50  # safety limit

        # Collect walkable cells across target floors
        walkable_cells: list[tuple[int, float, float]] = []
        for f in floors:
            grid = self._grids[f]
            for r in range(grid.height_cells):
                for c in range(grid.width_cells):
                    if grid.grid[r, c]:
                        x, y = grid.grid_to_world(r, c)
                        walkable_cells.append((f, x, y))

        if not walkable_cells:
            raise ValueError(f"No walkable cells on floor(s) {floors}")

        indices = self._rng.choice(len(walkable_cells), size=self.n)
        for i, idx in enumerate(indices):
            f, x, y = walkable_cells[idx]
            self._floor[i] = f
            self._x[i] = x
            self._y[i] = y

        self._weights[:] = 1.0 / self.n
        self._initialised = True

    def initialise_around(self, x: float, y: float, floor: int,
                          spread: float = 5.0) -> None:
        """Initialise particles in a Gaussian cloud around a known position.

        Parameters
        ----------
        x, y : float
            Centre of the cloud (feet).
        floor : int
            Starting floor.
        spread : float
            Std-dev of the Gaussian (feet).
        """
        self._floor[:] = floor
        placed = 0
        attempts = 0
        max_attempts = self.n * 100
        while placed < self.n and attempts < max_attempts:
            px = x + self._rng.normal(0, spread)
            py = y + self._rng.normal(0, spread)
            if self._grids.is_walkable(floor, px, py):
                self._x[placed] = px
                self._y[placed] = py
                placed += 1
            attempts += 1

        # If we couldn't place enough, duplicate the ones we have
        if placed < self.n and placed > 0:
            for i in range(placed, self.n):
                src = i % placed
                self._x[i] = self._x[src]
                self._y[i] = self._y[src]

        self._weights[:] = 1.0 / self.n
        self._initialised = True

    # ------------------------------------------------------------------
    # Predict (motion model)
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate particles through the motion model.

        Each particle is displaced by Gaussian noise scaled by dt, clamped to
        the maximum dog speed.  Moves that cross a wall or land in a blocked
        cell are rejected (particle keeps its previous position).

        Parameters
        ----------
        dt : float
            Time elapsed since last step (seconds).
        """
        if not self._initialised:
            return

        dt = max(dt, 0.01)  # guard against zero/negative dt
        sigma = DOG_SPEED_SIGMA_FT * math.sqrt(dt)
        max_disp = DOG_MAX_SPEED_FT * dt

        dx = self._rng.normal(0, sigma, self.n)
        dy = self._rng.normal(0, sigma, self.n)

        # Clamp displacement magnitude to max dog speed * dt
        dist = np.sqrt(dx ** 2 + dy ** 2)
        over = dist > max_disp
        if np.any(over):
            scale = max_disp / dist[over]
            dx[over] *= scale
            dy[over] *= scale

        new_x = self._x + dx
        new_y = self._y + dy

        # Accept/reject: check walkability and wall crossing
        for i in range(self.n):
            f = int(self._floor[i])
            if f not in self._grids:
                continue
            grid = self._grids[f]
            nx, ny = float(new_x[i]), float(new_y[i])
            if grid.is_walkable(nx, ny) and grid.ray_clear(
                float(self._x[i]), float(self._y[i]), nx, ny
            ):
                self._x[i] = nx
                self._y[i] = ny
            # else: particle stays put (implicit wall bounce)

        # Floor transitions near stairways
        self._maybe_transition_floors(dt)

    def _maybe_transition_floors(self, dt: float) -> None:
        """Allow particles near stairways to change floor.

        When a ``FloorTransitionHMM`` is attached, the HMM's belief
        for the destination floor is used as the transition probability
        (scaled by proximity).  Otherwise falls back to the fixed
        ``FLOOR_TRANSITION_PROB`` rate.
        """
        if not self._stairways:
            return

        # HMM belief (if available) for destination-floor weighting
        hmm_belief = (
            self._floor_hmm.floor_belief if self._floor_hmm is not None else None
        )

        base_p = min(FLOOR_TRANSITION_PROB * dt, 0.1)

        for stair in self._stairways:
            from_floor = stair["from_floor"]
            to_floor = stair["to_floor"]
            sx, sy = stair["entry_x"], stair["entry_y"]

            # Find particles on the departure floor near the stairway entry
            on_floor = self._floor == from_floor
            if not np.any(on_floor):
                continue

            dx = self._x[on_floor] - sx
            dy = self._y[on_floor] - sy
            dists = np.sqrt(dx ** 2 + dy ** 2)
            close = dists < STAIR_PROXIMITY_FT

            if not np.any(close):
                continue

            # Compute per-particle transition probability
            if hmm_belief is not None:
                # Use HMM belief: higher destination-floor belief → higher
                # transition probability.  Scale so that belief ≥ 0.5 makes
                # transition very likely for nearby particles.
                dest_belief = hmm_belief.get(to_floor, 0.0)
                p_transition = min(dest_belief * 0.5, 0.4)
            else:
                p_transition = base_p

            # Proximity-scaled probability: closer → more likely
            close_dists = dists[close]
            proximity_scale = np.clip(
                1.0 - close_dists / STAIR_PROXIMITY_FT, 0.1, 1.0
            )
            p_per_particle = p_transition * proximity_scale

            # Indices into full arrays
            full_indices = np.where(on_floor)[0][close]
            do_transition = self._rng.random(len(full_indices)) < p_per_particle

            for idx in full_indices[do_transition]:
                # Place transitioned particle at the destination stairway entry
                dest_entry = self._find_stair_entry(to_floor, from_floor)
                if dest_entry is not None:
                    self._floor[idx] = to_floor
                    self._x[idx] = dest_entry[0]
                    self._y[idx] = dest_entry[1]

    def _find_stair_entry(
        self, on_floor: int, coming_from: int
    ) -> Optional[tuple[float, float]]:
        """Find the stairway entry point on *on_floor* that connects to *coming_from*."""
        for stair in self._stairways:
            if stair["from_floor"] == on_floor and stair["to_floor"] == coming_from:
                return (stair["entry_x"], stair["entry_y"])
        return None

    # ------------------------------------------------------------------
    # Update (observation model)
    # ------------------------------------------------------------------

    def update(self, rssi_readings: dict[str, float]) -> None:
        """Weight particles based on observed RSSI vector.

        Parameters
        ----------
        rssi_readings : dict[str, float]
            anchor_id → RSSI (dBm).  Only anchors with known positions
            (in ``self._anchors``) contribute to the likelihood.
        """
        if not self._initialised or not rssi_readings:
            return

        log_weights = np.zeros(self.n)

        for anchor_id, observed_rssi in rssi_readings.items():
            if anchor_id not in self._anchors:
                continue
            ax = self._anchors[anchor_id]["x"]
            ay = self._anchors[anchor_id]["y"]
            a_floor = self._anchors[anchor_id]["floor"]

            for i in range(self.n):
                dist = math.sqrt(
                    (self._x[i] - ax) ** 2 + (self._y[i] - ay) ** 2
                )
                floor_diff = abs(int(self._floor[i]) - a_floor)
                expected = _expected_rssi(dist, floor_diff)
                log_weights[i] += _log_likelihood(observed_rssi, expected)

        # Convert log-weights to normal weights (numerically stable)
        log_weights -= np.max(log_weights)
        raw = np.exp(log_weights) * self._weights
        total = raw.sum()
        if total > 0:
            self._weights = raw / total
        else:
            # All weights collapsed — reinitialise to uniform
            self._weights[:] = 1.0 / self.n

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def resample_if_needed(self) -> bool:
        """Perform systematic resampling when particle diversity is low.

        Returns True if resampling was performed.
        """
        n_eff = _effective_sample_size(self._weights)
        if n_eff >= self.n * RESAMPLE_THRESHOLD_RATIO:
            return False

        indices = _systematic_resample(self._weights, self._rng)
        self._x = self._x[indices]
        self._y = self._y[indices]
        self._floor = self._floor[indices]
        self._weights[:] = 1.0 / self.n
        return True

    # ------------------------------------------------------------------
    # Full step (convenience)
    # ------------------------------------------------------------------

    def step(self, rssi_readings: dict[str, float], dt: float) -> dict:
        """Run a full predict → update → resample cycle.

        If a ``FloorTransitionHMM`` is attached, it is also stepped with
        the same RSSI readings and stairway-proximity information derived
        from the current particle estimate.

        Parameters
        ----------
        rssi_readings : dict[str, float]
            anchor_id → RSSI (dBm).
        dt : float
            Seconds since last step.

        Returns
        -------
        dict
            Current position estimate (same as ``self.estimate``).
        """
        # Drive the floor HMM (before particle predict so that beliefs
        # are available for transition modulation)
        if self._floor_hmm is not None and self._initialised:
            est = self.estimate
            proximity = self._floor_hmm.stair_proximity_for_position(
                est["x"], est["y"], est["floor"]
            )
            self._floor_hmm.step(rssi_readings, dt, proximity)

        self.predict(dt)
        self.update(rssi_readings)
        self.resample_if_needed()
        return self.estimate

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------

    @property
    def estimate(self) -> dict:
        """Weighted position estimate.

        Returns
        -------
        dict
            {"x": float, "y": float, "floor": int, "confidence": float,
             "n_eff": float, "particle_count": int}
        """
        if not self._initialised:
            return {
                "x": 0.0, "y": 0.0, "floor": 1,
                "confidence": 0.0, "n_eff": 0.0, "particle_count": self.n,
            }

        # Majority floor (weighted vote)
        floors = np.unique(self._floor)
        floor_weights = {
            int(f): float(self._weights[self._floor == f].sum()) for f in floors
        }
        best_floor = max(floor_weights, key=floor_weights.get)

        # Weighted mean on majority floor
        mask = self._floor == best_floor
        w = self._weights[mask]
        w_sum = w.sum()
        if w_sum > 0:
            w_norm = w / w_sum
        else:
            w_norm = np.ones_like(w) / len(w)

        x_est = float(np.dot(w_norm, self._x[mask]))
        y_est = float(np.dot(w_norm, self._y[mask]))

        # Confidence: combination of floor weight dominance and spatial spread
        spread_x = float(np.sqrt(np.dot(w_norm, (self._x[mask] - x_est) ** 2)))
        spread_y = float(np.sqrt(np.dot(w_norm, (self._y[mask] - y_est) ** 2)))
        spatial_spread = math.sqrt(spread_x ** 2 + spread_y ** 2)

        # Confidence heuristic: high when floor is dominant and spread is low
        floor_conf = floor_weights[best_floor]
        # Map spread to 0-1: 0 ft → 1.0, >10 ft → ~0
        spread_conf = math.exp(-spatial_spread / 5.0)
        confidence = floor_conf * spread_conf

        n_eff = float(_effective_sample_size(self._weights))

        return {
            "x": x_est,
            "y": y_est,
            "floor": best_floor,
            "confidence": round(confidence, 3),
            "n_eff": round(n_eff, 1),
            "particle_count": self.n,
        }

    # ------------------------------------------------------------------
    # Accessors for visualisation / debugging
    # ------------------------------------------------------------------

    @property
    def particles(self) -> np.ndarray:
        """Return particle states as (N, 3) array [x, y, floor]."""
        return np.column_stack([self._x, self._y, self._floor])

    @property
    def weights(self) -> np.ndarray:
        """Return normalised weight array."""
        return self._weights.copy()

    @property
    def effective_sample_size(self) -> float:
        return _effective_sample_size(self._weights)

    def __repr__(self):
        est = self.estimate
        return (
            f"ParticleFilter(n={self.n}, "
            f"floor={est['floor']}, "
            f"pos=({est['x']:.1f}, {est['y']:.1f}), "
            f"conf={est['confidence']:.2f}, "
            f"n_eff={est['n_eff']:.0f})"
        )


# ---------------------------------------------------------------------------
# Helper: extract stairways from layout.json data
# ---------------------------------------------------------------------------

def extract_stairways(layout_data: dict) -> list[dict]:
    """Parse stairway connections from layout.json into the format
    expected by ``ParticleFilter``.

    Returns a list of dicts:
        {"from_floor": int, "to_floor": int, "entry_x": float, "entry_y": float}
    """
    stairways = []
    for floor_data in layout_data.get("floors", []):
        from_floor = floor_data["floor"]
        for stair in floor_data.get("stairs", []):
            entry = stair.get("entry", [0, 0])
            stairways.append({
                "from_floor": from_floor,
                "to_floor": stair["to_floor"],
                "entry_x": entry[0],
                "entry_y": entry[1],
            })
    return stairways
