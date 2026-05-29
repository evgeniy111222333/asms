"""Maximum diversification portfolio optimization."""

import numpy as np
from scipy.optimize import minimize


class MaximumDiversificationPortfolio:
    """Maximum Diversification Portfolio.

    Maximizes the diversification ratio:
    DR = (w' * sigma) / sqrt(w' * Sigma * w)
    where sigma is the vector of asset volatilities.
    """

    def optimize(self, cov_matrix: np.ndarray) -> dict:
        """Find maximum diversification portfolio weights."""
        n = cov_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "diversification_ratio": 1.0}

        vols = np.sqrt(np.diag(cov_matrix))
        if np.any(vols == 0):
            return {"weights": np.ones(n) / n, "diversification_ratio": 1.0}

        def neg_div_ratio(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return 0.0
            weighted_vol = weights @ vols
            return -(weighted_vol / port_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(neg_div_ratio, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        dr = (weights @ vols) / port_vol if port_vol > 0 else 1.0

        return {"weights": weights, "diversification_ratio": float(dr)}


__all__ = [
    "MaximumDiversificationPortfolio",
]
