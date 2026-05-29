"""Black-Litterman portfolio model."""

from typing import Optional

import numpy as np


class BlackLitterman:
    """Black-Litterman model for portfolio allocation.

    Combines market equilibrium with investor views to produce
    posterior expected returns and optimal weights.
    """

    def __init__(self, tau: float = 0.05):
        self.tau = tau

    def compute(self, market_weights: np.ndarray, cov_matrix: np.ndarray,
                risk_aversion: float = 2.5, views: Optional[np.ndarray] = None,
                view_confidence: Optional[np.ndarray] = None,
                view_returns: Optional[np.ndarray] = None) -> dict:
        """Compute Black-Litterman posterior returns and optimal weights."""
        pi = risk_aversion * cov_matrix @ market_weights

        if views is None or view_confidence is None or view_returns is None:
            return {"expected_returns": pi, "weights": market_weights}

        omega = np.diag(view_confidence)
        tau_sigma = self.tau * cov_matrix

        m1 = np.linalg.inv(tau_sigma)
        m2 = views.T @ np.linalg.inv(omega) @ views

        m_matrix = np.linalg.inv(m1 + m2)
        posterior_cov = cov_matrix + m_matrix

        m3 = m1 @ pi
        m4 = views.T @ np.linalg.inv(omega) @ view_returns
        posterior_mean = m_matrix @ (m3 + m4)

        weights = np.linalg.inv(risk_aversion * posterior_cov) @ posterior_mean
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)

        return {"expected_returns": posterior_mean, "posterior_cov": posterior_cov, "weights": weights}


__all__ = [
    "BlackLitterman",
]
