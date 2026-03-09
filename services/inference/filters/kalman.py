"""
Kalman Filter for RSSI Smoothing

Applies a 1-D Kalman filter independently to each anchor's RSSI stream.
This reduces BLE signal noise while preserving real transitions (e.g., the
dog moving from one room to another).

State model:
    x = [rssi, rssi_velocity]   (2×1)
    - rssi:          current smoothed RSSI value (dBm)
    - rssi_velocity: rate of change (dBm/s), captures movement trends

Matrices:
    F (state transition):  Predicts next state from current state + dt
    H (observation):       Maps state to measurement (we only observe rssi)
    Q (process noise):     How much we trust the motion model
    R (measurement noise): How much we trust raw RSSI readings
    P (covariance):        Uncertainty in the current state estimate

Usage in the inference pipeline:
    1. Create a KalmanFilterBank (manages one KalmanFilter per anchor)
    2. On each RSSI reading: bank.update(anchor_id, raw_rssi, timestamp)
    3. Get smoothed values:  bank.get_smoothed_rssi() -> {anchor_id: float}
"""

import numpy as np


class KalmanFilter:
    """
    1-D Kalman filter for a single RSSI stream (one anchor).

    State vector: [rssi, rssi_velocity]
    Observation:  [rssi]

    Parameters
    ----------
    process_noise : float
        Variance of the process noise (Q scaling factor).
        Higher = trust measurements more, lower = smoother output.
        Typical range for BLE RSSI: 0.1 – 2.0
    measurement_noise : float
        Variance of RSSI measurement noise (R).
        Measured from your real data: the InfluxDB readings showed ~0.8 dBm
        spread at a stationary position, so R ≈ 1.0 is a reasonable start.
    initial_rssi : float
        Initial RSSI estimate (dBm). Default -70 is mid-range for BLE.
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        initial_rssi: float = -70.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State vector: [rssi, rssi_velocity]
        self.x = np.array([initial_rssi, 0.0])

        # State covariance matrix (start with high uncertainty)
        self.P = np.array([
            [10.0, 0.0],
            [0.0, 5.0],
        ])

        # Observation matrix: we only measure rssi directly
        self.H = np.array([[1.0, 0.0]])

        # Measurement noise covariance
        self.R = np.array([[self.measurement_noise]])

        # Timestamp of last update (seconds, monotonic or UTC epoch)
        self._last_time: float | None = None

    def _build_F(self, dt: float) -> np.ndarray:
        """
        Build the state transition matrix for the given time delta.

        F = [[1, dt],
             [0,  1]]

        This models: rssi_new = rssi_old + velocity * dt
                     velocity_new = velocity_old  (constant velocity assumption)

        Parameters
        ----------
        dt : float
            Time elapsed since last update, in seconds.

        Returns
        -------
        np.ndarray
            2×2 state transition matrix.
        """
        return np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Build the process noise covariance matrix for the given time delta.

        Use a discrete white noise model scaled by self.process_noise and dt.
        A common formulation for constant-velocity models:

        Q = process_noise * [[dt^3/3, dt^2/2],
                              [dt^2/2, dt    ]]

        This ties the noise to the time step — longer gaps = more uncertainty.

        Parameters
        ----------
        dt : float
            Time elapsed since last update, in seconds.

        Returns
        -------
        np.ndarray
            2×2 process noise covariance matrix.
        """
        return self.process_noise * np.array([
            [dt**3/3, dt**2/2],
            [dt**2/2, dt]
            ])

    def predict(self, dt: float) -> None:
        """
        Prediction step: project state and covariance forward by dt seconds.

        Updates self.x and self.P in place using:
            x_predicted = F @ x
            P_predicted = F @ P @ F^T + Q

        Parameters
        ----------
        dt : float
            Time elapsed since last update, in seconds.
        """
        F = self._build_F(dt)
        self.x = F @ self.x # matrix multiplication to get predicted state
        self.P = F @ self.P @ F.T + self._build_Q(dt)

    def update(self, measurement: float) -> None:
        """
        Update step: incorporate a new RSSI measurement.

        Computes the Kalman gain and corrects the predicted state:
            y = z - H @ x                  (innovation / residual)
            S = H @ P @ H^T + R            (innovation covariance)
            K = P @ H^T @ S^-1             (Kalman gain)
            x = x + K @ y                  (updated state)
            P = (I - K @ H) @ P            (updated covariance)

        Parameters
        ----------
        measurement : float
            Raw RSSI value in dBm.
        """
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def process(self, raw_rssi: float, timestamp: float) -> float:
        """
        Full predict-then-update cycle for a new RSSI reading.

        This is the main entry point. Call this each time a new RSSI value
        arrives from an anchor.

        On first call, initializes the state to the raw measurement.
        On subsequent calls, computes dt from the previous timestamp,
        runs predict(dt), then update(raw_rssi).

        Parameters
        ----------
        raw_rssi : float
            Raw RSSI measurement in dBm (e.g., -67.5).
        timestamp : float
            Time of the measurement in seconds (e.g., time.time()).

        Returns
        -------
        float
            Smoothed RSSI value in dBm.
        """
        if self._last_time is None:
            self.update(raw_rssi)
            self._last_time = timestamp
            return self.rssi
        else:
            dt = timestamp - self._last_time
            self.predict(dt)
            self.update(raw_rssi)
            self._last_time = timestamp
            return self.rssi

    @property
    def rssi(self) -> float:
        """Current smoothed RSSI estimate (dBm)."""
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        """Current RSSI rate of change (dBm/s). Useful for activity detection."""
        return float(self.x[1])

    @property
    def variance(self) -> float:
        """Current uncertainty (variance) of the RSSI estimate."""
        return float(self.P[0, 0])

    def reset(self, rssi: float = -70.0) -> None:
        """Reset filter state (e.g., after a long gap with no readings)."""
        self.x = np.array([rssi, 0.0])
        self.P = np.array([
            [10.0, 0.0],
            [0.0, 5.0],
        ])
        self._last_time = None


class KalmanFilterBank:
    """
    Manages one KalmanFilter per anchor for the tracked beacon.

    The inference pipeline creates one bank per beacon. Each incoming RSSI
    reading is routed to the appropriate per-anchor filter by anchor_id.

    Parameters
    ----------
    process_noise : float
        Passed to each KalmanFilter instance.
    measurement_noise : float
        Passed to each KalmanFilter instance.
    stale_timeout : float
        Seconds after which an anchor with no updates is considered stale.
        Stale anchors are excluded from get_smoothed_rssi() and their
        filters are reset when they next report.
    """

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        stale_timeout: float = 30.0,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.stale_timeout = stale_timeout

        # anchor_id -> KalmanFilter
        self._filters: dict[str, KalmanFilter] = {}
        # anchor_id -> last update timestamp
        self._last_seen: dict[str, float] = {}

    def _get_or_create_filter(self, anchor_id: str, initial_rssi: float) -> KalmanFilter:
        """Get existing filter for anchor, or create a new one."""
        if anchor_id not in self._filters:
            self._filters[anchor_id] = KalmanFilter(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                initial_rssi=initial_rssi,
            )
        return self._filters[anchor_id]

    def update(self, anchor_id: str, raw_rssi: float, timestamp: float) -> float:
        """
        Process a new RSSI reading from an anchor.

        If the anchor hasn't reported in longer than stale_timeout, resets
        its filter before processing.

        Parameters
        ----------
        anchor_id : str
            The anchor that reported this reading (e.g., "1f_office").
        raw_rssi : float
            Raw RSSI in dBm.
        timestamp : float
            Measurement time in seconds (e.g., time.time()).

        Returns
        -------
        float
            Smoothed RSSI value for this anchor.
        """
        if anchor_id in self._last_seen and (timestamp - self._last_seen[anchor_id]) > self.stale_timeout:
            self._filters[anchor_id].reset(rssi=raw_rssi)

        kf = self._get_or_create_filter(anchor_id, initial_rssi=raw_rssi)
        smoothed = kf.process(raw_rssi, timestamp)
        self._last_seen[anchor_id] = timestamp
        return smoothed

    def get_smoothed_rssi(self, current_time: float | None = None) -> dict[str, float]:
        """
        Get smoothed RSSI values for all non-stale anchors.

        Parameters
        ----------
        current_time : float, optional
            Current time in seconds. If provided, excludes stale anchors.

        Returns
        -------
        dict[str, float]
            Mapping of anchor_id -> smoothed RSSI (dBm).
        """
        result = {}
        for anchor_id, kf in self._filters.items():
            if current_time is not None and (current_time - self._last_seen.get(anchor_id, 0)) > self.stale_timeout:
                continue
            result[anchor_id] = kf.rssi
        return result

    def get_velocities(self) -> dict[str, float]:
        """Get RSSI velocity (dBm/s) for each anchor. Useful for activity detection."""
        return {aid: f.velocity for aid, f in self._filters.items()}

    def get_variances(self) -> dict[str, float]:
        """Get RSSI variance for each anchor. High variance = less certain."""
        return {aid: f.variance for aid, f in self._filters.items()}

    @property
    def active_anchors(self) -> list[str]:
        """List of anchor IDs that have been seen."""
        return list(self._filters.keys())

    def reset_all(self) -> None:
        """Reset all filters."""
        self._filters.clear()
        self._last_seen.clear()
