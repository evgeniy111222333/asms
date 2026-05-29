"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class AlmgrenChriss:
    """Almgren-Chriss optimal execution model.

    Computes the optimal trading trajectory that minimizes
    expected cost plus risk (variance of cost).
    """

    def __init__(self, total_shares: float, total_time: int, sigma: float,
                 eta: float = 0.1, gamma: float = 0.1, lambd: float = 0.1):
        """Initialize Almgren-Chriss model.

        Args:
            total_shares: Total shares to execute.
            total_time: Total execution time in periods.
            sigma: Asset volatility.
            eta: Temporary impact coefficient.
            gamma: Permanent impact coefficient.
            lambd: Risk aversion parameter.
        """
        self.X = total_shares
        self.T = total_time
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambd = lambd

    def optimal_trajectory(self, num_steps: int = 100) -> dict:
        """Compute optimal trading trajectory.

        Args:
            num_steps: Number of time steps.

        Returns:
            Dict with trajectory, trades, costs, and kappa.
        """
        kappa = np.sqrt(self.lambd * self.sigma ** 2 / self.eta)
        t = np.linspace(0, self.T, num_steps)
        x = self.X * (np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)) if np.sinh(kappa * self.T) != 0 else np.linspace(self.X, 0, num_steps)
        n = np.diff(x, prepend=self.X)
        n[0] = self.X - x[0]
        permanent_cost = 0.5 * self.gamma * self.X ** 2
        temporary_cost = self.eta * np.sum(n ** 2)
        expected_cost = permanent_cost + temporary_cost
        cost_variance = self.sigma ** 2 * np.sum(x[:-1] ** 2 * np.diff(t))
        return {
            "trajectory": x, "trades": n, "times": t,
            "expected_cost": float(expected_cost),
            "cost_variance": float(cost_variance), "kappa": float(kappa),
        }

__all__ = ['AlmgrenChriss']
