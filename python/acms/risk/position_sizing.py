"""Position Sizing for ACMS.

Provides standalone position sizing functions:
- Kelly Criterion with drawdown constraint
- Fixed fractional sizing
- Volatility regime sizing
- Dynamic combined sizing
"""

import numpy as np
from typing import Optional


def kelly_size(win_rate: float, avg_win: float, avg_loss: float,
               capital: float, fraction: float = 0.5,
               max_drawdown: float = 0.25) -> float:
    """Kelly Criterion position sizing with drawdown constraint.

    The drawdown constraint reduces Kelly fraction when estimated
    maximum drawdown exceeds the threshold.

    Args:
        win_rate: Historical win rate.
        avg_win: Average winning trade size.
        avg_loss: Average losing trade size.
        capital: Total capital.
        fraction: Fractional Kelly (0.5 = half-Kelly).
        max_drawdown: Maximum acceptable drawdown.

    Returns:
        Position size in currency units.
    """
    if avg_loss == 0 or win_rate == 0:
        return 0.0
    kelly_f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    kelly_f = max(kelly_f, 0.0)

    if kelly_f > 0:
        consecutive_losses = int(np.log(1 - max_drawdown) / np.log(1 - kelly_f)) if kelly_f < 1 else 1
        expected_dd = 1 - (1 - kelly_f) ** max(3, consecutive_losses)
        if expected_dd > max_drawdown:
            kelly_f *= max_drawdown / expected_dd

    return capital * kelly_f * fraction


def fixed_fractional_size(capital: float, risk_pct: float = 0.02,
                          entry_price: float = 0, stop_price: float = 0) -> float:
    """Fixed fractional position sizing.

    Args:
        capital: Total capital.
        risk_pct: Percentage of capital to risk per trade.
        entry_price: Entry price.
        stop_price: Stop loss price.

    Returns:
        Position size in units.
    """
    if entry_price == 0 or stop_price == 0 or entry_price == stop_price:
        return 0.0
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry_price - stop_price)
    return risk_amount / risk_per_unit


def volatility_regime_size(capital: float, base_risk_pct: float = 0.02,
                           current_vol: float = 0.0, target_vol: float = 0.15) -> float:
    """Dynamic position sizing based on volatility regime.

    Uses Kelly criterion with drawdown constraint combined with
    volatility targeting to dynamically adjust position sizes.

    Args:
        capital: Total capital.
        base_risk_pct: Base risk percentage at target volatility.
        current_vol: Current realized volatility (annualized).
        target_vol: Target volatility for base risk.

    Returns:
        Adjusted position notional value.
    """
    if current_vol <= 0 or target_vol <= 0:
        return capital * base_risk_pct
    vol_scalar = target_vol / current_vol
    vol_scalar = max(0.25, min(vol_scalar, 3.0))
    adjusted_risk = base_risk_pct * vol_scalar
    return capital * adjusted_risk


def dynamic_position_size(capital: float, base_risk_pct: float,
                           current_vol: float, target_vol: float,
                           win_rate: float = 0.5, avg_win: float = 0.02,
                           avg_loss: float = 0.01, max_drawdown: float = 0.20) -> float:
    """Dynamic position sizing combining Kelly criterion and volatility targeting.

    Applies Kelly with drawdown constraint, then scales by the
    volatility regime ratio to achieve target volatility.

    Args:
        capital: Total capital.
        base_risk_pct: Base risk percentage.
        current_vol: Current realized volatility (annualized).
        target_vol: Target portfolio volatility.
        win_rate: Strategy win rate for Kelly.
        avg_win: Average win as fraction of position.
        avg_loss: Average loss as fraction of position.
        max_drawdown: Maximum acceptable drawdown.

    Returns:
        Position size in currency units.
    """
    ks = kelly_size(win_rate, avg_win, avg_loss, capital,
                     fraction=0.5, max_drawdown=max_drawdown)
    vs = volatility_regime_size(capital, base_risk_pct,
                                 current_vol, target_vol)
    # Use the more conservative of the two
    return min(ks, vs)


__all__ = ['kelly_size', 'fixed_fractional_size', 'volatility_regime_size', 'dynamic_position_size']
