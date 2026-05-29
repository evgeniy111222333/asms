"""Portfolio Heat Map for ACMS."""

import numpy as np
from typing import Dict, List
from scipy import stats
from acms.core import Position


class PortfolioHeatMap:
    """Portfolio risk heat map - risk contribution per position.

    Shows each position's contribution to overall portfolio risk,
    including marginal VaR, component VaR, and percentage contribution.
    """

    def compute(self, positions: List[Position], returns_matrix: np.ndarray,
                weights: np.ndarray, confidence: float = 0.99) -> List[Dict]:
        """Compute risk contribution per position.

        Args:
            positions: List of current positions.
            returns_matrix: Historical returns matrix (T x N).
            weights: Current portfolio weights.
            confidence: VaR confidence level.

        Returns:
            List of dicts with risk contribution per position.
        """
        if returns_matrix.shape[1] != len(weights) or len(positions) != len(weights):
            return []

        cov_matrix = np.cov(returns_matrix.T)
        port_var = weights @ cov_matrix @ weights
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10

        z = stats.norm.ppf(confidence)
        marginal_var = (cov_matrix @ weights) / port_vol * z
        component_var = weights * marginal_var

        total_cvar = np.sum(component_var)
        heatmap = []
        for i, pos in enumerate(positions):
            pct_contribution = component_var[i] / total_cvar * 100 if total_cvar != 0 else 0
            heatmap.append({
                "symbol": pos.symbol,
                "weight": float(weights[i]),
                "marginal_var": float(marginal_var[i]),
                "component_var": float(component_var[i]),
                "pct_risk_contribution": float(pct_contribution),
                "position_notional": pos.notional_value,
                "risk_level": "high" if pct_contribution > 30 else "medium" if pct_contribution > 15 else "low",
            })
        return heatmap

__all__ = ['PortfolioHeatMap']
