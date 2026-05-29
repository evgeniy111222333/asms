"""Correlation Risk Monitoring for ACMS."""

import numpy as np
from typing import Optional, Dict, List


class CorrelationRiskMonitor:
    """Dynamic correlation matrix monitoring with eigenvalue decomposition.

    Detects correlation breakdowns, concentration risk from excessive
    correlation, and structural changes via eigenvalue analysis.
    """

    def __init__(self, lookback: int = 60, max_correlation: float = 0.85,
                 breakdown_threshold: float = 0.3):
        """Initialize correlation risk monitor.

        Args:
            lookback: Rolling window for correlation computation.
            max_correlation: Maximum acceptable pairwise correlation.
            breakdown_threshold: Threshold for detecting correlation breakdowns.
        """
        self.lookback = lookback
        self.max_correlation = max_correlation
        self.breakdown_threshold = breakdown_threshold
        self._prev_correlations: Optional[np.ndarray] = None
        self._eigenvalue_history: List[np.ndarray] = []

    def compute_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation matrix from returns.

        Args:
            returns_matrix: Matrix of returns (T x N).

        Returns:
            N x N correlation matrix.
        """
        if returns_matrix.shape[0] < 10:
            return np.eye(returns_matrix.shape[1])
        return np.corrcoef(returns_matrix.T)

    def eigenvalue_decomposition(self, corr_matrix: np.ndarray) -> Dict:
        """Perform eigenvalue decomposition of correlation matrix.

        Eigenvalue analysis reveals the effective dimensionality
        of the portfolio and concentration of correlation risk.

        Args:
            corr_matrix: N x N correlation matrix.

        Returns:
            Dict with eigenvalues, eigenvectors, and concentration metrics.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        n = len(eigenvalues)
        total = np.sum(eigenvalues)
        pct_variance = eigenvalues / total if total > 0 else eigenvalues

        # Effective rank (number of significant eigenvalues)
        effective_rank = float(np.sum(eigenvalues > 1.0))

        # Concentration ratio: share of variance in top eigenvalue
        concentration_ratio = float(pct_variance[0]) if len(pct_variance) > 0 else 1.0

        self._eigenvalue_history.append(eigenvalues)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "pct_variance_explained": pct_variance,
            "effective_rank": effective_rank,
            "concentration_ratio": concentration_ratio,
            "is_concentrated": concentration_ratio > 0.5,
        }

    def detect_correlation_breakdown(self, current_corr: np.ndarray) -> Dict:
        """Detect significant changes in correlation structure.

        Args:
            current_corr: Current correlation matrix.

        Returns:
            Dict with breakdown detection results.
        """
        if self._prev_correlations is None or current_corr.shape != self._prev_correlations.shape:
            self._prev_correlations = current_corr.copy()
            return {"breakdown_detected": False, "max_change": 0.0}

        diff = np.abs(current_corr - self._prev_correlations)
        max_change = float(np.max(diff))
        avg_change = float(np.mean(diff))

        breakdown = max_change > self.breakdown_threshold
        self._prev_correlations = current_corr.copy()

        # Also check eigenvalue stability
        eigen_stability = True
        if len(self._eigenvalue_history) >= 2:
            prev_eig = self._eigenvalue_history[-2] if len(self._eigenvalue_history) >= 2 else self._eigenvalue_history[-1]
            curr_eig = self._eigenvalue_history[-1]
            if len(prev_eig) == len(curr_eig):
                eig_change = np.max(np.abs(curr_eig - prev_eig))
                eigen_stability = eig_change < 0.5

        return {
            "breakdown_detected": breakdown,
            "max_change": max_change,
            "avg_change": avg_change,
            "affected_pairs": int(np.sum(diff > self.breakdown_threshold * 0.5)),
            "eigenvalue_stable": eigen_stability,
        }

    def check_concentration_risk(self, corr_matrix: np.ndarray,
                                 weights: np.ndarray) -> Dict:
        """Check for correlation-driven concentration risk.

        Args:
            corr_matrix: Correlation matrix.
            weights: Portfolio weights.

        Returns:
            Dict with concentration risk metrics.
        """
        n = len(weights)
        high_corr_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > self.max_correlation:
                    high_corr_count += 1

        port_var = weights @ corr_matrix @ weights
        avg_var = np.mean(np.diag(corr_matrix))
        div_ratio = np.sqrt(avg_var / port_var) if port_var > 0 else 1.0

        eigen_data = self.eigenvalue_decomposition(corr_matrix)

        return {
            "high_correlation_pairs": high_corr_count,
            "max_correlation": float(np.max(np.abs(corr_matrix - np.eye(n)))),
            "diversification_ratio": float(div_ratio),
            "concentration_ratio": eigen_data["concentration_ratio"],
            "effective_rank": eigen_data["effective_rank"],
            "risk_level": "high" if high_corr_count > n else "moderate" if high_corr_count > 0 else "low",
        }

__all__ = ['CorrelationRiskMonitor']
