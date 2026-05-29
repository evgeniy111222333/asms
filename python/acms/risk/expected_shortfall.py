"""Expected Shortfall for ACMS."""

import numpy as np
from typing import Dict
from scipy import stats


class ExpectedShortfall:
    """Expected Shortfall (ES) - tail risk measure with decomposition.

    ES measures the average loss in the worst (1-confidence)% of cases.
    Also known as CVaR or AVaR. Provides tail risk decomposition
    showing which positions contribute most to tail risk.
    """

    @staticmethod
    def historical_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES from historical returns.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Expected Shortfall as a positive float.
        """
        if len(returns) < 50:
            return float('nan')
        threshold = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= threshold]
        if len(tail) == 0:
            return float(-threshold)
        return float(-np.mean(tail))

    @staticmethod
    def parametric_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES assuming normal distribution.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Parametric Expected Shortfall.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(confidence)
        phi_z = stats.norm.pdf(z)
        es = -(mu - sigma * phi_z / (1 - confidence))
        return float(es)

    @staticmethod
    def cornish_fisher_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES using Cornish-Fisher expansion for non-normal returns.

        Adjusts for skewness and kurtosis in the return distribution.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Cornish-Fisher adjusted Expected Shortfall.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))

        z = stats.norm.ppf(confidence)
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        phi_z = stats.norm.pdf(z_cf)
        es = -(mu - sigma * phi_z / (1 - confidence))
        return float(es)

    @staticmethod
    def tail_risk_decomposition(returns_matrix: np.ndarray, weights: np.ndarray,
                                confidence: float = 0.975) -> Dict:
        """Decompose tail risk into per-asset contributions.

        Identifies which assets contribute most to portfolio tail risk
        by computing each asset's marginal ES contribution.

        Args:
            returns_matrix: Matrix of asset returns (T x N).
            weights: Portfolio weights (N,).
            confidence: ES confidence level.

        Returns:
            Dict with per-asset ES contributions and percentages.
        """
        if returns_matrix.shape[1] != len(weights) or len(weights) < 2:
            return {"contributions": np.array([]), "pct_contributions": np.array([]),
                    "total_es": float('nan')}

        n = len(weights)
        portfolio_returns = returns_matrix @ weights
        total_es = ExpectedShortfall.historical_es(portfolio_returns, confidence)
        if np.isnan(total_es):
            return {"contributions": np.zeros(n), "pct_contributions": np.zeros(n),
                    "total_es": float('nan')}

        delta = 1e-4
        marginal_es = np.zeros(n)
        for i in range(n):
            w_up = weights.copy()
            w_up[i] += delta
            port_ret_up = returns_matrix @ w_up
            es_up = ExpectedShortfall.historical_es(port_ret_up, confidence)
            if not np.isnan(es_up):
                marginal_es[i] = (es_up - total_es) / delta

        contributions = weights * marginal_es
        total_contrib = np.sum(np.abs(contributions))
        pct_contributions = np.abs(contributions) / total_contrib * 100 if total_contrib > 0 else np.zeros(n)

        return {
            "contributions": contributions,
            "marginal_es": marginal_es,
            "pct_contributions": pct_contributions,
            "total_es": float(total_es),
        }

__all__ = ['ExpectedShortfall']
