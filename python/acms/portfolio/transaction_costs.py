"""Transaction cost modeling."""

from typing import Dict

import numpy as np


class TransactionCostModel:
    """Transaction cost model with fixed, proportional, and market impact components.

    Models three components of trading costs:
    - Fixed cost: constant per-trade cost
    - Proportional cost: percentage of trade notional
    - Market impact: square-root model based on participation rate
    """

    def __init__(self, fixed_cost_usd: float = 1.0, proportional_cost_bps: float = 5.0,
                 market_impact_alpha: float = 0.1, avg_daily_volume_usd: float = 1000000.0):
        self.fixed_cost_usd = fixed_cost_usd
        self.proportional_cost_bps = proportional_cost_bps
        self.market_impact_alpha = market_impact_alpha
        self.avg_daily_volume_usd = avg_daily_volume_usd

    def compute_cost(self, trade_notional: float, current_weights: np.ndarray,
                     target_weights: np.ndarray, portfolio_value: float) -> Dict:
        """Compute total transaction cost for a rebalance."""
        n_trades = int(np.sum(np.abs(target_weights - current_weights) > 0.001))
        fixed = self.fixed_cost_usd * n_trades
        proportional = trade_notional * self.proportional_cost_bps / 10000
        participation = trade_notional / self.avg_daily_volume_usd if self.avg_daily_volume_usd > 0 else 0
        market_impact = self.market_impact_alpha * np.sqrt(participation) * trade_notional
        total = fixed + proportional + market_impact

        return {
            "fixed_cost": float(fixed),
            "proportional_cost": float(proportional),
            "market_impact_cost": float(market_impact),
            "total_cost": float(total),
            "total_cost_bps": float(total / portfolio_value * 10000) if portfolio_value > 0 else 0.0,
            "n_trades": n_trades,
        }

    def cost_adjusted_weights(self, current_weights: np.ndarray,
                               target_weights: np.ndarray,
                               portfolio_value: float) -> np.ndarray:
        """Adjust target weights to account for transaction costs."""
        trade_notional = np.sum(np.abs(target_weights - current_weights)) * portfolio_value
        cost_info = self.compute_cost(trade_notional, current_weights, target_weights, portfolio_value)
        total_cost = cost_info["total_cost"]

        cost_fraction = total_cost / portfolio_value if portfolio_value > 0 else 0
        adjusted = target_weights * (1 - cost_fraction)
        if adjusted.sum() > 0:
            adjusted /= adjusted.sum()
        return adjusted


__all__ = [
    "TransactionCostModel",
]
