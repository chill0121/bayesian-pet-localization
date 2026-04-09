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

from .constants import (
    TX_POWER_DBM,
    PATH_LOSS_N,
    RSSI_SIGMA,
    FLOOR_ATTENUATION_DB,
    MIN_DISTANCE_FT,
    WALL_ATTENUATION_DB,
    FLOOR_ELEVATION_FT,
)
from .floor_hmm import FloorTransitionHMM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dog motion budget
DOG_MAX_SPEED_FT = 3.28          # ~1 m/s in feet/s
DOG_SPEED_SIGMA_FT = 1.5         # std-dev of per-axis displacement per second

# RSSI-gradient drift: pull particles toward the RSSI-weighted centroid
# of same-floor anchors.  Helps guide particles through doorways.
DRIFT_ALPHA = 0.15               # fraction of (target − particle) added per second

# Resampling
RESAMPLE_THRESHOLD_RATIO = 0.5   # resample when N_eff < N * ratio

# Floor transition
STAIR_PROXIMITY_FT = 4.0         # how close a particle must be to a stair entry
FLOOR_TRANSITION_PROB = 0.02     # base probability of changing floor per step
FLOOR_TRANSITION_RATE_HMM = 0.8  # per-second rate when HMM belief indicates
                                 # a different floor (scaled by dt × proximity)
FLOOR_TRANSITION_COOLDOWN = 5.0  # seconds a particle must stay on a floor
                                 # before it can transition again (prevents
                                 # rapid 1→2→3 cascading through staircase)

# Floor teleport: when the HMM strongly disagrees with the particle
# majority floor for several consecutive seconds, reinitialize a
# fraction of particles on the HMM's most-likely floor.
TELEPORT_BELIEF_THRESHOLD = 0.85 # HMM belief for alternate floor must exceed this
TELEPORT_HOLDOFF_SEC = 2.0       # sustained disagreement required before teleport
TELEPORT_FRACTION = 0.3          # fraction of particles to teleport

# Safety guards
MIN_DT = 0.01                    # minimum dt (seconds) to avoid division issues
MAX_INIT_ATTEMPTS_MULTIPLIER = 50  # max_attempts = n_particles × this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_rssi(distance_ft: float, floor_diff: int = 0, n_walls: int = 0) -> float:
    """Compute expected RSSI from distance using log-distance path loss.

    Parameters
    ----------
    distance_ft : float
        Euclidean distance in feet between particle and anchor.
    floor_diff : int
        Absolute floor difference (0 = same floor).
    n_walls : int
        Number of interior walls the signal must pass through (same floor).

    Returns
    -------
    float
        Expected RSSI in dBm.
    """
    effective_ft = max(distance_ft, MIN_DISTANCE_FT)
    distance_m = effective_ft * 0.3048
    return (
        TX_POWER_DBM
        - 10.0 * PATH_LOSS_N * math.log10(distance_m)
        - n_walls * WALL_ATTENUATION_DB
        - floor_diff * FLOOR_ATTENUATION_DB
    )


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
        anchor_id → {"x": float, "y": float, "floor": int, "height_ft": float (opt)}.
    stairways : list[dict]
        Stairway connection list.  Each entry:
        {"from_floor": int, "to_floor": int, "entry_x": float, "entry_y": float}.
    stair_runs : list[dict] | None
        Stair run definitions for height interpolation.  Each entry:
        {"from_floor": int, "to_floor": int, "low_z": float, "high_z": float,
         "entry_y_low": float, "entry_y_high": float}.
    staircase_bounds : dict[int, dict] | None
        Per-floor staircase room bounding box: {floor: {"x_max": float}}.
        Particles with x <= x_max on a run's floor are considered on-stairs.
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
        stair_runs: list[dict] | None = None,
        staircase_bounds: dict[int, dict] | None = None,
        floor_hmm: FloorTransitionHMM | None = None,
        n_particles: int = 500,
        seed: int | None = None,
    ):
        self._grids = occupancy_grids
        self._anchors = anchor_positions
        self._stairways = stairways or []
        self._stair_runs = stair_runs or []
        self._staircase_bounds = staircase_bounds or {}
        self._floor_hmm = floor_hmm
        self.n = n_particles
        self._rng = np.random.default_rng(seed)

        # Snapped anchor positions: wall-mounted anchors are shifted to
        # the nearest walkable cell so that ray-based wall counting
        # correctly identifies which room the anchor belongs to.
        self._snapped_anchors: dict[str, tuple[float, float]] = {}
        for aid, apos in anchor_positions.items():
            a_floor = apos["floor"]
            if a_floor in occupancy_grids:
                sx, sy = occupancy_grids.nearest_walkable(
                    a_floor, apos["x"], apos["y"]
                )
                self._snapped_anchors[aid] = (sx, sy)

        # Pre-compute absolute anchor elevations (floor + mount height)
        self._anchor_z: dict[str, float] = {}
        for aid, apos in anchor_positions.items():
            floor_z = FLOOR_ELEVATION_FT.get(apos["floor"], 0.0)
            self._anchor_z[aid] = floor_z + apos.get("height_ft", 0.0)

        # Build per-floor stair run lookup: {floor: [run, ...]}
        self._floor_stair_runs: dict[int, list[dict]] = {}
        for run in self._stair_runs:
            for f in (run["from_floor"], run["to_floor"]):
                self._floor_stair_runs.setdefault(f, []).append(run)

        # Particle state arrays
        self._x = np.zeros(self.n)
        self._y = np.zeros(self.n)
        self._floor = np.ones(self.n, dtype=int)
        self._weights = np.full(self.n, 1.0 / self.n)

        # Per-particle floor-transition cooldown: time remaining (seconds)
        # before the particle is allowed to change floors again.
        self._transition_cooldown = np.zeros(self.n)

        # Last RSSI readings (stored for drift computation in predict)
        self._last_rssi: dict[str, float] = {}

        # Floor-teleport holdoff timer: tracks when HMM first started
        # disagreeing with the particle majority floor.
        self._teleport_disagreement_start: float | None = None
        self._cumulative_time: float = 0.0   # running clock for holdoff

        self._initialised = False

    # ------------------------------------------------------------------
    # Initialization
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
        max_attempts = self.n * MAX_INIT_ATTEMPTS_MULTIPLIER  # safety limit

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

        Each particle is displaced by an RSSI-gradient drift plus Gaussian
        noise scaled by dt, clamped to the maximum dog speed.  Moves that
        cross a wall or land in a blocked cell are rejected (particle keeps
        its previous position).

        Parameters
        ----------
        dt : float
            Time elapsed since last step (seconds).
        """
        if not self._initialised:
            return

        dt = max(dt, MIN_DT)  # guard against zero/negative dt
        sigma = DOG_SPEED_SIGMA_FT * math.sqrt(dt)
        max_disp = DOG_MAX_SPEED_FT * dt

        # --- RSSI-gradient drift ---
        # Compute a per-floor target from RSSI-weighted anchor centroid.
        # Particles get a gentle pull toward the strongest-signal region,
        # which helps navigate through doorways and corridors.
        drift_targets = self._compute_drift_targets()

        dx = self._rng.normal(0, sigma, self.n)
        dy = self._rng.normal(0, sigma, self.n)

        # Add drift toward RSSI-weighted target (per floor)
        drift_scale = DRIFT_ALPHA * dt
        for floor_num, (tx, ty) in drift_targets.items():
            mask = self._floor == floor_num
            if not np.any(mask):
                continue
            dx[mask] += drift_scale * (tx - self._x[mask])
            dy[mask] += drift_scale * (ty - self._y[mask])

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

    def _compute_drift_targets(self) -> dict[int, tuple[float, float]]:
        """Compute per-floor RSSI-weighted anchor centroid.

        For each floor that has at least two anchors with current RSSI,
        returns a (target_x, target_y) that particles on that floor are
        gently pulled toward.  Uses linear power weighting so that
        stronger (closer) anchors dominate.

        Returns
        -------
        dict[int, tuple[float, float]]
            floor → (target_x, target_y).
        """
        targets: dict[int, tuple[float, float]] = {}
        if not self._last_rssi:
            return targets

        # Group same-floor anchors with their RSSI
        floor_data: dict[int, list[tuple[float, float, float]]] = {}
        for aid, rssi in self._last_rssi.items():
            if aid not in self._anchors:
                continue
            af = self._anchors[aid]["floor"]
            ax = self._anchors[aid]["x"]
            ay = self._anchors[aid]["y"]
            # Convert dBm to linear power for weighting
            # Shift by +100 to keep values positive before exponentiation
            w = 10.0 ** ((rssi + 100.0) / 10.0)
            floor_data.setdefault(af, []).append((ax, ay, w))

        for fnum, entries in floor_data.items():
            if len(entries) < 1:
                continue
            total_w = sum(e[2] for e in entries)
            if total_w <= 0:
                continue
            tx = sum(e[0] * e[2] for e in entries) / total_w
            ty = sum(e[1] * e[2] for e in entries) / total_w
            targets[fnum] = (tx, ty)

        return targets

    def _maybe_transition_floors(self, dt: float) -> None:
        """Allow particles near stairways to change floor.

        When a ``FloorTransitionHMM`` is attached, the HMM's belief
        for the destination floor is used as the transition probability
        (scaled by dt and proximity).  Otherwise falls back to the fixed
        ``FLOOR_TRANSITION_PROB`` rate.

        Guards:
        * Per-particle **cooldown**: after a floor transition, the particle
          cannot transition again for ``FLOOR_TRANSITION_COOLDOWN`` seconds.
          This prevents rapid cascading (e.g. 1→2→3 within a few steps).
        * **Same-step guard**: a particle that transitions in this call is
          excluded from further transitions in the same call (prevents
          instant bounce-back through a shared stairway entry).
        """
        if not self._stairways:
            return

        # Tick down cooldowns
        self._transition_cooldown = np.maximum(
            self._transition_cooldown - dt, 0.0
        )

        # HMM belief (if available) for destination-floor weighting
        hmm_belief = (
            self._floor_hmm.floor_belief if self._floor_hmm is not None else None
        )

        base_p = min(FLOOR_TRANSITION_PROB * dt, 0.1)

        # Track particles that already changed floor this step
        already_transitioned = np.zeros(self.n, dtype=bool)

        for stair in self._stairways:
            from_floor = stair["from_floor"]
            to_floor = stair["to_floor"]
            sx, sy = stair["entry_x"], stair["entry_y"]

            # Find particles on the departure floor near the stairway entry,
            # excluding those on cooldown or already transitioned this step.
            eligible = (
                (self._floor == from_floor)
                & (self._transition_cooldown == 0.0)
                & ~already_transitioned
            )
            if not np.any(eligible):
                continue

            dx = self._x[eligible] - sx
            dy = self._y[eligible] - sy
            dists = np.sqrt(dx ** 2 + dy ** 2)
            close = dists < STAIR_PROXIMITY_FT

            if not np.any(close):
                continue

            # Compute per-particle transition probability
            if hmm_belief is not None:
                dest_belief = hmm_belief.get(to_floor, 0.0)
                # Scale by dt so the per-second rate is consistent
                # regardless of inference frequency.
                p_transition = min(
                    dest_belief * FLOOR_TRANSITION_RATE_HMM * dt,
                    0.4,
                )
            else:
                p_transition = base_p

            # Proximity-scaled probability: closer → more likely
            close_dists = dists[close]
            proximity_scale = np.clip(
                1.0 - close_dists / STAIR_PROXIMITY_FT, 0.1, 1.0
            )
            p_per_particle = p_transition * proximity_scale

            # Indices into full arrays
            full_indices = np.where(eligible)[0][close]
            do_transition = self._rng.random(len(full_indices)) < p_per_particle

            for idx in full_indices[do_transition]:
                # Place transitioned particle at the destination stairway entry
                dest_entry = self._find_stair_entry(to_floor, from_floor)
                if dest_entry is not None:
                    self._floor[idx] = to_floor
                    self._x[idx] = dest_entry[0]
                    self._y[idx] = dest_entry[1]
                    self._transition_cooldown[idx] = FLOOR_TRANSITION_COOLDOWN
                    already_transitioned[idx] = True

    def _find_stair_entry(
        self, on_floor: int, coming_from: int
    ) -> Optional[tuple[float, float]]:
        """Find the stairway entry point on *on_floor* that connects to *coming_from*."""
        for stair in self._stairways:
            if stair["from_floor"] == on_floor and stair["to_floor"] == coming_from:
                return (stair["entry_x"], stair["entry_y"])
        return None

    # ------------------------------------------------------------------
    # Elevation helpers
    # ------------------------------------------------------------------

    def _particle_elevation(self, i: int) -> float:
        """Return absolute elevation (ft) for particle *i*.

        For particles in staircase rooms, elevation is linearly
        interpolated based on Y-position along the stair run.
        For all other particles, elevation is the floor ground level.
        """
        p_floor = int(self._floor[i])
        base_z = FLOOR_ELEVATION_FT.get(p_floor, 0.0)

        # Check if particle is in the staircase column on this floor
        sc = self._staircase_bounds.get(p_floor)
        if sc is None:
            return base_z
        if float(self._x[i]) > sc["x_max"]:
            return base_z

        # Find the stair run for this floor
        runs = self._floor_stair_runs.get(p_floor, [])
        py = float(self._y[i])
        for run in runs:
            y_lo = min(run["entry_y_low"], run["entry_y_high"])
            y_hi = max(run["entry_y_low"], run["entry_y_high"])
            if py < y_lo or py > y_hi:
                continue
            # Linear interpolation: Y decreases as Z increases
            t = (run["entry_y_low"] - py) / (run["entry_y_low"] - run["entry_y_high"])
            t = max(0.0, min(1.0, t))
            return run["low_z"] + t * (run["high_z"] - run["low_z"])

        return base_z

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
            a_z = self._anchor_z.get(anchor_id, 0.0)
            # Use snapped (walkable) position for wall counting so that
            # wall-mounted anchors are attributed to their room.
            snap = self._snapped_anchors.get(anchor_id)

            for i in range(self.n):
                dx = self._x[i] - ax
                dy = self._y[i] - ay
                dist_2d_sq = dx * dx + dy * dy

                # Height-aware distance: staircase particles get
                # interpolated elevation; others use floor ground level.
                p_z = self._particle_elevation(i)
                dz = p_z - a_z
                dist = math.sqrt(dist_2d_sq + dz * dz)

                floor_diff = abs(int(self._floor[i]) - a_floor)
                n_walls = 0
                if floor_diff == 0 and snap is not None:
                    p_floor = int(self._floor[i])
                    if p_floor in self._grids:
                        n_walls = self._grids[p_floor].count_wall_crossings(
                            float(self._x[i]), float(self._y[i]),
                            snap[0], snap[1],
                        )
                expected = _expected_rssi(dist, floor_diff, n_walls)
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

        When the HMM strongly disagrees with the particle majority floor
        for a sustained period (``TELEPORT_HOLDOFF_SEC``), a fraction of
        particles are teleported to the HMM's preferred floor to recover
        from situations where particles are stuck behind walls or on the
        wrong floor.

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
        # Store RSSI for drift computation in predict()
        self._last_rssi = rssi_readings

        # Drive the floor HMM (before particle predict so that beliefs
        # are available for transition modulation)
        if self._floor_hmm is not None and self._initialised:
            est = self.estimate
            proximity = self._floor_hmm.stair_proximity_for_position(
                est["x"], est["y"], est["floor"]
            )
            self._floor_hmm.step(rssi_readings, dt, proximity)

            # --- Floor teleport check ---
            self._maybe_teleport(dt)

        self.predict(dt)
        self.update(rssi_readings)
        self.resample_if_needed()
        return self.estimate

    # ------------------------------------------------------------------
    # Floor teleport
    # ------------------------------------------------------------------

    def _maybe_teleport(self, dt: float) -> None:
        """Teleport particles when HMM strongly disagrees with majority floor.

        If the HMM's most-likely floor differs from the particle majority
        floor *and* the HMM belief exceeds ``TELEPORT_BELIEF_THRESHOLD``
        for at least ``TELEPORT_HOLDOFF_SEC`` cumulative seconds, spawn
        ``TELEPORT_FRACTION`` of particles uniformly on walkable cells of
        the target floor.  This recovers from situations where particles
        are trapped on the wrong floor or stuck behind walls.

        The target floor must be adjacent to the particle majority floor —
        teleporting directly across non-adjacent floors (e.g. 1→3) is not
        allowed.
        """
        if self._floor_hmm is None:
            return

        hmm_belief = self._floor_hmm.floor_belief
        hmm_best = max(hmm_belief, key=hmm_belief.get)
        hmm_conf = hmm_belief[hmm_best]

        # Current particle majority floor
        floors, counts = np.unique(self._floor, return_counts=True)
        particle_best = int(floors[np.argmax(counts)])

        self._cumulative_time += dt

        # Only allow teleport to adjacent floors
        adjacent = self._adjacent_floors()

        if (hmm_best != particle_best
                and hmm_conf >= TELEPORT_BELIEF_THRESHOLD
                and hmm_best in adjacent.get(particle_best, set())):
            if self._teleport_disagreement_start is None:
                self._teleport_disagreement_start = self._cumulative_time
            elif (self._cumulative_time - self._teleport_disagreement_start
                  >= TELEPORT_HOLDOFF_SEC):
                # Teleport!
                self._do_teleport(hmm_best)
                self._teleport_disagreement_start = None
        else:
            # Agreement (or weak disagreement or non-adjacent) — reset holdoff
            self._teleport_disagreement_start = None

    def _adjacent_floors(self) -> dict[int, set[int]]:
        """Build floor adjacency from stairway definitions."""
        adj: dict[int, set[int]] = {}
        for stair in self._stairways:
            f = stair["from_floor"]
            t = stair["to_floor"]
            adj.setdefault(f, set()).add(t)
            adj.setdefault(t, set()).add(f)
        return adj

    def _do_teleport(self, target_floor: int) -> None:
        """Reinitialise a fraction of lowest-weight particles on *target_floor*."""
        n_teleport = max(1, int(self.n * TELEPORT_FRACTION))

        # Pick the lowest-weight particles to replace
        indices = np.argsort(self._weights)[:n_teleport]

        # Collect walkable cells on the target floor
        if target_floor not in self._grids:
            return
        grid = self._grids[target_floor]
        walkable: list[tuple[float, float]] = []
        for r in range(grid.height_cells):
            for c in range(grid.width_cells):
                if grid.grid[r, c]:
                    walkable.append(grid.grid_to_world(r, c))
        if not walkable:
            return

        choices = self._rng.choice(len(walkable), size=n_teleport)
        for i, idx in enumerate(indices):
            wx, wy = walkable[choices[i]]
            self._x[idx] = wx
            self._y[idx] = wy
            self._floor[idx] = target_floor

        # Reset weights to uniform after teleport
        self._weights[:] = 1.0 / self.n

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


def extract_stair_runs(layout_data: dict) -> list[dict]:
    """Parse stair run geometry from ``layout_data["stair_geometry"]["runs"]``.

    Returns a list of dicts suitable for ``ParticleFilter(stair_runs=...)``.
    """
    sg = layout_data.get("stair_geometry", {})
    return sg.get("runs", [])


def extract_staircase_bounds(layout_data: dict) -> dict[int, dict]:
    """Build per-floor staircase bounding-box info.

    Returns {floor: {"x_max": float}} where x_max is the eastern extent
    of the staircase room on that floor.  Used to quickly test whether a
    particle is inside the stairwell column.
    """
    bounds: dict[int, dict] = {}
    for floor_data in layout_data.get("floors", []):
        floor_num = floor_data["floor"]
        for room in floor_data.get("rooms", []):
            if room["name"] != "staircase":
                continue
            b = room["bounds"]
            if isinstance(b, dict):
                x_max = b["x2"]
            else:
                x_max = max(p[0] for p in b)
            bounds[floor_num] = {"x_max": x_max}
    return bounds
