"""Leverage optimization."""

from typing import Dict

import numpy as np


class LeverageOptimizer:
    """Leverage optimization based on Kelly and risk constraints.

    Finds the optimal leverage level based on:
    - Target volatility
    - Risk tolerance
    - Kelly criterion with drawdown constraint
    """

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 3.0,
                 max_drawdown: float = 0.25):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown

    def volatility_target_leverage(self, current_vol: float) -> float:
        """Compute leverage to achieve target volatility."""
        if current_vol <= 0:
            return 1.0
        leverage = self.target_vol / current_vol
        return max(1.0, min(leverage, self.max_leverage))

    def kelly_leverage(self, expected_return: float, volatility: float,
                       risk_free_rate: float = 0.0) -> float:
        """Compute Kelly-optimal leverage."""
        if volatility <= 0:
            return 1.0
        excess_return = expected_return - risk_free_rate
        leverage = excess_return / (volatility ** 2)
        return max(0.0, min(leverage, self.max_leverage))

    def optimal_leverage(self, expected_return: float, volatility: float,
                         risk_free_rate: float = 0.0,
                         win_rate: float = 0.5) -> Dict:
        """Compute optimal leverage combining Kelly and volatility targeting."""
        kelly_lev = self.kelly_leverage(expected_return, volatility, risk_free_rate)
        vol_lev = self.volatility_target_leverage(volatility)

        half_kelly = kelly_lev * 0.5

        if volatility > 0:
            dd_constrained_lev = np.sqrt(2 * self.max_drawdown) / volatility
        else:
            dd_constrained_lev = self.max_leverage

        optimal = min(half_kelly, vol_lev, dd_constrained_lev, self.max_leverage)
        optimal = max(0.0, optimal)

        return {
            "optimal_leverage": float(optimal),
            "kelly_leverage": float(kelly_lev),
            "half_kelly": float(half_kelly),
            "vol_target_leverage": float(vol_lev),
            "drawdown_constrained_leverage": float(dd_constrained_lev),
        }


__all__ = [
    "LeverageOptimizer",
]
