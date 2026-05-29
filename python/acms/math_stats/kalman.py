"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class KalmanFilter:
    """Kalman Filter for dynamic state estimation.

    Implements the standard linear Kalman filter for:
    - Dynamic hedge ratio estimation
    - Price level tracking with noise
    - Trend extraction
    Uses Joseph form for numerical stability.
    """

    def __init__(self, state_dim: int = 1, observation_dim: int = 1,
                 process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        """Initialize Kalman Filter.

        Args:
            state_dim: Dimension of state vector.
            observation_dim: Dimension of observation vector.
            process_noise: Process noise variance.
            measurement_noise: Measurement noise variance.
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(observation_dim) * measurement_noise
        self.H = np.eye(observation_dim, state_dim)
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        """Initialize state and covariance.

        Args:
            x0: Initial state estimate.
            P0: Initial state covariance.
        """
        self.x = x0.copy()
        self.P = P0 if P0 is not None else np.eye(self.state_dim)

    def predict(self, F: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step: project state forward.

        Args:
            F: State transition matrix (default: identity).

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        if F is None:
            F = np.eye(self.state_dim)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step: incorporate observation.

        Uses Joseph form for numerical stability.

        Args:
            z: Observation vector.

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()

    def filter_series(self, observations: np.ndarray,
                      F: Optional[np.ndarray] = None) -> Dict:
        """Run Kalman filter on a series of observations.

        Args:
            observations: Matrix of observations (T x obs_dim).
            F: State transition matrix.

        Returns:
            Dict with filtered states and covariances.
        """
        T = len(observations)
        states = np.zeros((T, self.state_dim))
        covs = np.zeros((T, self.state_dim, self.state_dim))

        for t in range(T):
            self.predict(F)
            self.update(observations[t])
            states[t] = self.x
            covs[t] = self.P

        return {"states": states, "covariances": covs}

    @staticmethod
    def dynamic_hedge_ratio(y: np.ndarray, x: np.ndarray) -> Dict:
        """Estimate dynamic hedge ratio using Kalman filter.

        Models the relationship y_t = alpha_t + beta_t * x_t + epsilon_t
        where alpha and beta follow random walks.

        Args:
            y: Dependent variable series.
            x: Independent variable series.

        Returns:
            Dict with hedge_ratio and intercept series.
        """
        T = len(y)
        if T < 10 or len(x) != T:
            return {"hedge_ratio": np.ones(T), "intercept": np.zeros(T)}

        kf = KalmanFilter(state_dim=2, observation_dim=1,
                          process_noise=1e-5, measurement_noise=1e-2)
        kf.H = np.array([[1.0, x[0]]])
        kf.initialize(np.array([0.0, 1.0]))

        ratios = np.zeros(T)
        intercepts = np.zeros(T)

        for t in range(T):
            kf.predict()
            kf.H = np.array([[1.0, x[t]]])
            kf.update(np.array([y[t]]))
            intercepts[t] = kf.x[0]
            ratios[t] = kf.x[1]

        return {"hedge_ratio": ratios, "intercept": intercepts}

    @staticmethod
    def trend_extraction(data: np.ndarray, noise_var: float = 1e-4,
                         process_var: float = 1e-6) -> Dict:
        """Extract trend from noisy data using Kalman filter.

        Models observation as trend + noise where trend follows
        a local level model (random walk).

        Args:
            data: Input time series.
            noise_var: Observation noise variance.
            process_var: Process (trend) noise variance.

        Returns:
            Dict with trend and innovation series.
        """
        T = len(data)
        if T < 5:
            return {"trend": data.copy(), "innovations": np.zeros(T)}

        kf = KalmanFilter(state_dim=1, observation_dim=1,
                          process_noise=process_var, measurement_noise=noise_var)
        kf.initialize(np.array([data[0]]))

        trend = np.zeros(T)
        innovations = np.zeros(T)

        for t in range(T):
            kf.predict()
            innovations[t] = data[t] - kf.x[0]
            kf.update(np.array([data[t]]))
            trend[t] = kf.x[0]

        return {"trend": trend, "innovations": innovations}

__all__ = ['KalmanFilter']
