"""Mean-variance portfolio optimization."""

import numpy as np
from typing import Optional, List
from scipy.optimize import minimize

from acms.portfolio.config import PortfolioConfig


class MeanVarianceOptimizer:
    """Mean-Variance Optimization (Markowitz).

    Finds optimal portfolio weights to maximize Sharpe ratio
    or minimize volatility for a target return.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                 target_return: Optional[float] = None) -> dict:
        """Find optimal portfolio weights."""
        n = len(expected_returns)
        if n < 2:
            return {"weights": np.array([1.0]), "return": expected_returns[0] if len(expected_returns) > 0 else 0.0,
                    "volatility": 0.0, "sharpe_ratio": 0.0}

        target = target_return or self.config.target_return

        def portfolio_vol(weights):
            return np.sqrt(weights @ cov_matrix @ weights)

        def portfolio_ret(weights):
            return weights @ expected_returns

        def neg_sharpe(weights):
            ret = portfolio_ret(weights)
            vol = portfolio_vol(weights)
            if vol == 0:
                return 0.0
            return -(ret - self.config.risk_free_rate) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target is not None:
            constraints.append({"type": "eq", "fun": lambda w: portfolio_ret(w) - target})

        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe if target is None else portfolio_vol,
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
        )

        weights = result.x
        ret = portfolio_ret(weights)
        vol = portfolio_vol(weights)
        sharpe = (ret - self.config.risk_free_rate) / vol if vol > 0 else 0.0

        return {"weights": weights, "return": float(ret), "volatility": float(vol), "sharpe_ratio": float(sharpe)}

    def efficient_frontier(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                           num_points: int = 50) -> List[dict]:
        """Compute the efficient frontier."""
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        targets = np.linspace(min_ret, max_ret, num_points)
        frontier = []
        for target in targets:
            result = self.optimize(expected_returns, cov_matrix, target)
            if result.get("volatility", float('inf')) < float('inf'):
                frontier.append(result)
        return frontier


__all__ = [
    "MeanVarianceOptimizer",
]
