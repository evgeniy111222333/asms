"""Backtest configuration and data classes."""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

from acms.core import Side

logger = logging.getLogger(__name__)


class BacktestMode(str, Enum):
    """Backtest execution mode."""
    SINGLE = "single"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    slippage_model: str = "percentage"  # "percentage", "sqrt", "almgren_chriss"
    position_size_pct: float = 0.02
    max_positions: int = 5
    margin_enabled: bool = False
    max_leverage: float = 1.0
    wf_train_pct: float = 0.7
    wf_test_pct: float = 0.3
    wf_anchored: bool = False
    mc_simulations: int = 1000
    mc_method: str = "bootstrap"
    detect_regimes: bool = True
    regime_lookback: int = 100
    # Fill model
    fill_model: str = "immediate"  # "immediate", "partial", "fok"
    partial_fill_pct: float = 0.7
    # Sensitivity
    sensitivity_params: Optional[Dict[str, List[float]]] = None


@dataclass
class BacktestTrade:
    """Record of a single completed trade in a backtest."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    holding_period_bars: int
    strategy_id: str
    regime: str = "unknown"
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    etd: float = 0.0  # End Trade Drawdown (MFE - final PnL)


@dataclass
class MCStatistics:
    """Monte Carlo simulation statistics.

    Stores the full distribution of simulation results,
    not just point estimates.
    """
    mean_return: float = 0.0
    median_return: float = 0.0
    p5_return: float = 0.0
    p95_return: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    max_drawdown_p5: float = 0.0
    max_drawdown_median: float = 0.0
    sharpe_p5: float = 0.0
    sharpe_median: float = 0.0
    prob_positive: float = 0.0
    num_simulations: int = 0
    simulated_returns: Optional[np.ndarray] = None
    simulated_drawdowns: Optional[np.ndarray] = None
    simulated_sharpes: Optional[np.ndarray] = None


@dataclass
class BacktestResult:
    """Complete backtest result with metrics and trade data."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_bars: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    avg_holding_period: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    mc_statistics: Optional[MCStatistics] = None
    benchmark_return: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    # Benchmark comparison
    buy_hold_return: float = 0.0
    equal_weight_return: float = 0.0
    # Sensitivity results
    sensitivity_results: Optional[Dict] = None
    # Rolling metrics
    rolling_sharpe: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_sortino: np.ndarray = field(default_factory=lambda: np.array([]))
    rolling_max_dd: np.ndarray = field(default_factory=lambda: np.array([]))


__all__ = ["BacktestMode", "BacktestConfig", "BacktestTrade", "MCStatistics", "BacktestResult"]
