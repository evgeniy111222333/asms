"""Position sizing with multiple methodologies."""

import numpy as np


class PositionSizer:
    """Position sizing with multiple methodologies.

    Supports:
    - Kelly Criterion (fractional)
    - Risk-based (volatility targeting)
    - Fixed-fractional
    - Fixed-quantity
    """

    def __init__(self, method: str = "risk_based", max_position_pct: float = 0.02,
                 kelly_fraction: float = 0.5, target_volatility: float = 0.15,
                 risk_per_trade_pct: float = 0.01):
        self.method = method
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.target_volatility = target_volatility
        self.risk_per_trade_pct = risk_per_trade_pct

    def compute_size(self, equity: float, price: float, volatility: float = 0.0,
                     win_rate: float = 0.5, avg_win_loss_ratio: float = 1.0,
                     stop_distance_pct: float = 0.02) -> float:
        """Compute position size based on the configured method."""
        if equity <= 0 or price <= 0:
            return 0.0

        if self.method == "kelly":
            return self._kelly_size(equity, price, win_rate, avg_win_loss_ratio)
        elif self.method == "risk_based":
            return self._risk_based_size(equity, price, volatility, stop_distance_pct)
        elif self.method == "fixed_fractional":
            return self._fixed_fractional_size(equity, price)
        elif self.method == "volatility_target":
            return self._volatility_target_size(equity, price, volatility)
        else:
            return self._risk_based_size(equity, price, volatility, stop_distance_pct)

    def _kelly_size(self, equity: float, price: float,
                    win_rate: float, avg_win_loss_ratio: float) -> float:
        """Kelly Criterion position sizing (fractional)."""
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            return 0.0
        kelly_pct = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        if kelly_pct <= 0:
            return 0.0
        position_pct = kelly_pct * self.kelly_fraction
        position_pct = min(position_pct, self.max_position_pct)
        notional = equity * position_pct
        return notional / price

    def _risk_based_size(self, equity: float, price: float,
                         volatility: float, stop_distance_pct: float) -> float:
        """Risk-based position sizing."""
        if stop_distance_pct <= 0:
            stop_distance_pct = max(volatility * 0.25, 0.01) if volatility > 0 else 0.02
        risk_amount = equity * self.risk_per_trade_pct
        size = risk_amount / (price * stop_distance_pct)
        max_notional = equity * self.max_position_pct
        max_size = max_notional / price
        return min(size, max_size)

    def _fixed_fractional_size(self, equity: float, price: float) -> float:
        """Fixed-fractional position sizing."""
        notional = equity * self.max_position_pct
        return notional / price

    def _volatility_target_size(self, equity: float, price: float,
                                volatility: float) -> float:
        """Volatility-targeting position sizing."""
        if volatility <= 0:
            volatility = 0.20
        notional = equity * self.target_volatility / volatility
        max_notional = equity * self.max_position_pct
        notional = min(notional, max_notional)
        return notional / price


__all__ = [
    "PositionSizer",
]
