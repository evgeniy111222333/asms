"""CVaR portfolio optimization and risk budgeting."""

import logging

import numpy as np
from scipy.optimize import minimize, linprog

logger = logging.getLogger(__name__)


class CVaRPortfolioOptimization:
    """CVaR Portfolio Optimization using linear programming.

    Minimizes CVaR subject to return and weight constraints
    using the Rockafellar-Uryasev reformulation.
    """

    def __init__(self, confidence: float = 0.95, min_return: float = None,
                 max_weight: float = 0.40):
        self.confidence = confidence
        self.min_return = min_return
        self.max_weight = max_weight

    def optimize(self, returns_matrix: np.ndarray) -> dict:
        """Optimize portfolio to minimize CVaR using linear programming."""
        T, N = returns_matrix.shape
        if N < 2 or T < 50:
            return {"weights": np.ones(N) / N, "cvar": float('nan'), "var": float('nan')}

        alpha = self.confidence
        c = np.zeros(N + 1 + T)
        c[N] = 1.0
        c[N + 1:] = 1.0 / ((1 - alpha) * T)

        A_ub = np.zeros((T, N + 1 + T))
        for t in range(T):
            A_ub[t, :N] = returns_matrix[t]
            A_ub[t, N] = 1.0
            A_ub[t, N + 1 + t] = 1.0
        b_ub = np.zeros(T)

        A_eq = np.zeros((1, N + 1 + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        if self.min_return is not None:
            mean_returns = np.mean(returns_matrix, axis=0)
            ret_row = np.zeros(N + 1 + T)
            ret_row[:N] = -mean_returns
            A_ub_return = ret_row.reshape(1, -1)
            b_ub_return = np.array([-self.min_return])
            A_ub = np.vstack([A_ub, A_ub_return])
            b_ub = np.concatenate([b_ub, b_ub_return])

        bounds = [(0.01, self.max_weight)] * N + [(None, None)] + [(0, None)] * T

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if result.success:
                weights = result.x[:N]
                var_val = result.x[N]
                cvar_val = result.fun
                if weights.sum() > 0:
                    weights /= weights.sum()
                return {
                    "weights": weights,
                    "cvar": float(cvar_val),
                    "var": float(var_val),
                    "success": True,
                }
        except Exception as e:
            logger.warning("CVaR optimization failed, falling back to equal weights: %s", e)

        return {"weights": np.ones(N) / N, "cvar": float('nan'), "var": float('nan'), "success": False}


class CVaRRiskBudgeting:
    """Risk budgeting with CVaR constraints.

    Allocates risk budget to each asset such that
    the CVaR contribution matches the target.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def optimize(self, returns_matrix: np.ndarray,
                 risk_budget: np.ndarray = None) -> dict:
        """Find CVaR risk budgeting weights."""
        n = returns_matrix.shape[1]
        if n < 2:
            return {"weights": np.array([1.0]), "cvar_contributions": np.array([0.0])}

        if risk_budget is None:
            risk_budget = np.ones(n) / n

        alpha = self.confidence

        def portfolio_cvar(weights):
            port_returns = returns_matrix @ weights
            threshold = np.percentile(port_returns, (1 - alpha) * 100)
            tail = port_returns[port_returns <= threshold]
            if len(tail) == 0:
                return -threshold
            return -np.mean(tail)

        def cvar_contribution(weights):
            cvar = portfolio_cvar(weights)
            delta = 1e-5
            contributions = np.zeros(n)
            for i in range(n):
                w_up = weights.copy()
                w_up[i] += delta
                cvar_up = portfolio_cvar(w_up)
                contributions[i] = (cvar_up - cvar) / delta * weights[i]
            return contributions

        def objective(weights):
            contribs = cvar_contribution(weights)
            total = np.sum(np.abs(contribs))
            if total == 0:
                return 1e10
            pct_contrib = np.abs(contribs) / total
            return np.sum((pct_contrib - risk_budget) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        contribs = cvar_contribution(weights)

        return {"weights": weights, "cvar_contributions": contribs}


__all__ = [
    "CVaRPortfolioOptimization",
    "CVaRRiskBudgeting",
]
