"""Risk parity portfolio optimization."""

import numpy as np
from scipy.optimize import minimize


class RiskParityOptimizer:
    """Risk Parity - equal risk contribution from each asset."""

    def optimize(self, cov_matrix: np.ndarray) -> dict:
        """Find risk parity weights."""
        n = cov_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "risk_contributions": np.array([1.0])}

        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return np.zeros(n)
            marginal = cov_matrix @ weights
            return weights * marginal / port_vol

        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.mean(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        rc = risk_contribution(weights)
        return {"weights": weights, "risk_contributions": rc}


__all__ = [
    "RiskParityOptimizer",
]
