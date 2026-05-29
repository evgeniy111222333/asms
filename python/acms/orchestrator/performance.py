"""Performance monitoring and equity curve tracking."""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np


class PerformanceMonitor:
    """Monitors strategy performance and auto-disables underperformers.

    Tracks rolling Sharpe ratio and disables strategies that
    consistently underperform.
    """

    def __init__(self, min_sharpe: float = -1.0, lookback_trades: int = 20,
                 auto_disable: bool = True):
        self.min_sharpe = min_sharpe
        self.lookback_trades = lookback_trades
        self.auto_disable = auto_disable
        self._strategy_pnls: Dict[str, List[float]] = defaultdict(list)
        self._disabled_strategies: set = set()

    def record_pnl(self, strategy_id: str, pnl: float) -> None:
        """Record P&L for a strategy."""
        self._strategy_pnls[strategy_id].append(pnl)
        if len(self._strategy_pnls[strategy_id]) > self.lookback_trades * 2:
            self._strategy_pnls[strategy_id] = self._strategy_pnls[strategy_id][-self.lookback_trades:]

    def check_strategy(self, strategy_id: str) -> Dict:
        """Check if a strategy should be auto-disabled."""
        pnls = self._strategy_pnls.get(strategy_id, [])
        if len(pnls) < self.lookback_trades:
            return {"strategy_id": strategy_id, "should_disable": False,
                    "reason": "insufficient_data", "sharpe": 0.0}

        recent = pnls[-self.lookback_trades:]
        mean_pnl = np.mean(recent)
        std_pnl = np.std(recent)
        sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0

        should_disable = self.auto_disable and sharpe < self.min_sharpe
        if should_disable:
            self._disabled_strategies.add(strategy_id)

        return {
            "strategy_id": strategy_id,
            "should_disable": should_disable,
            "sharpe": float(sharpe),
            "mean_pnl": float(mean_pnl),
            "std_pnl": float(std_pnl),
            "total_pnl": float(sum(recent)),
            "win_rate": float(sum(1 for p in recent if p > 0) / len(recent)),
        }

    def is_disabled(self, strategy_id: str) -> bool:
        """Check if a strategy has been auto-disabled."""
        return strategy_id in self._disabled_strategies

    def reenable(self, strategy_id: str) -> None:
        """Re-enable a previously disabled strategy."""
        self._disabled_strategies.discard(strategy_id)


class EquityCurveTracker:
    """Real-time P&L tracking with equity curve."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.equity_history: List[Dict] = []
        self._current_equity = initial_capital

    def update(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """Record equity snapshot."""
        self._current_equity = equity
        self.equity_history.append({
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "equity": equity,
            "pnl": equity - self.initial_capital,
            "pnl_pct": (equity - self.initial_capital) / self.initial_capital,
        })

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def current_pnl(self) -> float:
        return self._current_equity - self.initial_capital

    @property
    def current_pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self._current_equity - self.initial_capital) / self.initial_capital

    def get_equity_array(self) -> np.ndarray:
        """Get equity values as numpy array."""
        return np.array([e["equity"] for e in self.equity_history])

    def get_max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        if len(self.equity_history) < 2:
            return 0.0
        equity = self.get_equity_array()
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))


__all__ = [
    "PerformanceMonitor",
    "EquityCurveTracker",
]
