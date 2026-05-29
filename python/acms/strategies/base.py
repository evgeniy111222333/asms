"""Strategy base class, parameter adaptation, and regime detection helpers."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from acms.core import Signal, SignalDirection, Candle, Position
from acms.signals import MarketRegime, RegimeDetector


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, strategy_id: str, symbol: str):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.is_active = True
        self.position: Optional[Position] = None
        self.signals_generated = 0
        self.trades_executed = 0
        self._state: Dict = {}
        self._regime_detector = RegimeDetector()

    @abstractmethod
    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate market data and return a signal if conditions are met."""
        ...

    @abstractmethod
    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Check if an existing position should be closed."""
        ...

    def reset(self):
        """Reset strategy state."""
        self.position = None
        self.signals_generated = 0
        self.trades_executed = 0
        self._state = {}

    def _detect_regime(self, candles: List[Candle]) -> MarketRegime:
        """Detect current market regime from candles."""
        if len(candles) < 50:
            return MarketRegime.UNKNOWN
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        return self._regime_detector.detect(closes, highs, lows)

    def _adapt_param(self, base_value: float, regime: MarketRegime,
                     trending_mult: float = 1.0, mr_mult: float = 1.0,
                     volatile_mult: float = 0.5, quiet_mult: float = 0.8) -> float:
        """Adapt a parameter based on market regime.

        Args:
            base_value: Base parameter value.
            regime: Current market regime.
            trending_mult: Multiplier for trending regime.
            mr_mult: Multiplier for mean-reverting regime.
            volatile_mult: Multiplier for volatile regime.
            quiet_mult: Multiplier for quiet regime.

        Returns:
            Adapted parameter value.
        """
        multipliers = {
            MarketRegime.TRENDING: trending_mult,
            MarketRegime.MEAN_REVERTING: mr_mult,
            MarketRegime.VOLATILE: volatile_mult,
            MarketRegime.QUIET: quiet_mult,
            MarketRegime.UNKNOWN: 1.0,
        }
        return base_value * multipliers.get(regime, 1.0)


__all__ = [
    "Strategy",
]
