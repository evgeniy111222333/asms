"""Reporting data models."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List


class DrawdownPeriod:
    """A single drawdown period."""
    peak_date: str
    trough_date: str
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_days: int
    recovery_date: Optional[str] = None
    recovery_days: Optional[int] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report with all metrics computed."""
    period_start: datetime
    period_end: datetime
    starting_capital: float
    ending_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration_hours: float
    best_trade: float
    worst_trade: float
    avg_winning_trade: float
    avg_losing_trade: float
    consecutive_wins: int
    consecutive_losses: int
    var_99: Optional[float] = None
    cvar_99: Optional[float] = None
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    calmar_ratio: float = 0.0
    monthly_returns: Optional[Dict[str, float]] = None
    yearly_returns: Optional[Dict[str, float]] = None
    drawdown_periods: Optional[List[Dict]] = None


@dataclass
class StrategyReport:
    """Strategy-specific performance report."""
    strategy_id: str
    strategy_type: str
    total_trades: int
    win_rate: float
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period: float
    best_trade: float
    worst_trade: float
    profit_factor: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0



__all__ = ["DrawdownPeriod", "PerformanceReport", "StrategyReport"]
