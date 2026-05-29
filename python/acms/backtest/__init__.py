"""Backtest Engine - Comprehensive strategy backtesting.

Re-exports all public names from submodules for backward compatibility.
"""

from acms.backtest.config import (
    BacktestMode,
    BacktestConfig,
    BacktestTrade,
    MCStatistics,
    BacktestResult,
)
from acms.backtest.slippage import SlippageModel
from acms.backtest.fill_model import FillModel
from acms.backtest.analytics import TradeAnalytics, RollingMetrics
from acms.backtest.benchmark import BenchmarkComparison, RegimeDetector, SensitivityAnalysis
from acms.backtest.engine import BacktestEngine

__all__ = [
    "BacktestMode",
    "BacktestConfig",
    "BacktestTrade",
    "MCStatistics",
    "BacktestResult",
    "SlippageModel",
    "FillModel",
    "TradeAnalytics",
    "RollingMetrics",
    "BenchmarkComparison",
    "RegimeDetector",
    "SensitivityAnalysis",
    "BacktestEngine",
]
