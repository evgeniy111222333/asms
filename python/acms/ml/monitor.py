"""Model monitoring and drift detection."""

import numpy as np
from scipy import stats as scipy_stats
from typing import Optional, Dict


class ModelMonitor:
    """Model monitoring for drift detection.

    Monitors:
    - Feature drift using Population Stability Index (PSI) and KS test
    - Prediction drift using KS test
    """

    def __init__(self, reference_features: Optional[np.ndarray] = None,
                 reference_predictions: Optional[np.ndarray] = None,
                 drift_threshold: float = 0.05,
                 psi_threshold: float = 0.2):
        self.reference_features = reference_features
        self.reference_predictions = reference_predictions
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold

    def detect_feature_drift(self, current_features: np.ndarray) -> Dict:
        """Detect feature drift using Kolmogorov-Smirnov test and PSI.

        Args:
            current_features: Current feature matrix.

        Returns:
            Dict with drift detection results per feature including KS test
            and Population Stability Index.
        """
        if self.reference_features is None:
            return {"drift_detected": False, "details": "No reference features"}

        n_features = min(self.reference_features.shape[1], current_features.shape[1])
        drift_results = {}
        any_drift = False

        for j in range(n_features):
            ref = self.reference_features[:, j]
            cur = current_features[:, j]

            # KS test
            ks_stat, p_value = scipy_stats.ks_2samp(ref, cur)
            ks_drifted = p_value < self.drift_threshold

            # PSI
            psi_value = self._compute_psi(ref, cur)
            psi_drifted = psi_value > self.psi_threshold

            drifted = ks_drifted or psi_drifted
            drift_results[f"feature_{j}"] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "ks_drifted": ks_drifted,
                "psi_value": float(psi_value),
                "psi_drifted": psi_drifted,
                "drifted": drifted,
            }
            if drifted:
                any_drift = True

        return {"drift_detected": any_drift, "details": drift_results}

    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict:
        """Detect prediction drift using KS test.

        Args:
            current_predictions: Current model predictions.

        Returns:
            Dict with drift detection results.
        """
        if self.reference_predictions is None:
            return {"drift_detected": False, "details": "No reference predictions"}

        ks_stat, p_value = scipy_stats.ks_2samp(self.reference_predictions, current_predictions)
        return {
            "drift_detected": p_value < self.drift_threshold,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
        }

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray,
                     n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI).

        Args:
            expected: Reference distribution values.
            actual: Current distribution values.
            n_bins: Number of bins for distribution comparison.

        Returns:
            PSI value. Values > 0.2 indicate significant drift.
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)


__all__ = ["ModelMonitor"]
