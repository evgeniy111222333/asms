"""Value at Risk for ACMS."""

import numpy as np
from typing import Optional
from scipy import stats


class ValueAtRisk:
    """Value at Risk computation using 3 methods.

    Provides historical, parametric, and Monte Carlo VaR estimation.
    All methods return VaR as a positive number representing potential loss.
    """

    @staticmethod
    def historical(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Historical VaR - percentile of historical returns.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level (e.g. 0.99 for 99% VaR).

        Returns:
            VaR as a positive float representing potential loss.
        """
        if len(returns) < 100:
            return float('nan')
        return float(-np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def parametric(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Parametric VaR - assumes normal distribution.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.

        Returns:
            Parametric VaR as a positive float.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(confidence)
        return float(-(mu - z * sigma))

    @staticmethod
    def monte_carlo(returns: np.ndarray, confidence: float = 0.99,
                    num_simulations: int = 10000, horizon_days: int = 1) -> float:
        """Monte Carlo VaR using Student's t distribution for fat tails.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.
            num_simulations: Number of Monte Carlo simulations.
            horizon_days: Forecast horizon in days.

        Returns:
            Monte Carlo VaR as a positive float.
        """
        if len(returns) < 30:
            return float('nan')
        df, loc, scale = stats.t.fit(returns)
        simulated = stats.t.rvs(df, loc=loc, scale=scale, size=num_simulations) * np.sqrt(horizon_days)
        return float(-np.percentile(simulated, (1 - confidence) * 100))

    @staticmethod
    def cvar(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Conditional VaR (Expected Shortfall) - average of losses beyond VaR.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.

        Returns:
            CVaR as a positive float.
        """
        if len(returns) < 100:
            return float('nan')
        var = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return float(-var)
        return float(-np.mean(tail_returns))

__all__ = ['ValueAtRisk']
