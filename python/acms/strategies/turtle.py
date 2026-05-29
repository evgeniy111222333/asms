"""Turtle trading strategy implementation."""

import numpy as np
from typing import Optional, List
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import ATR
from acms.strategies.base import Strategy


class TurtleTradingStrategy(Strategy):
    """Turtle Trading - Donchian breakout with ATR position sizing and pyramiding.

    Classic trend-following system:
    - System 1: 20-day breakout for entry, 10-day breakout for exit
    - System 2: 55-day breakout for entry, 20-day breakout for exit
    - Position size = 1% of account / (N * Dollar per point) where N = ATR(20)
    - Pyramiding: Add to winning positions at each 0.5N price advance
    - Exit on trailing stop at 2N from most recent entry
    """

    def __init__(self, symbol: str, entry_period: int = 20, exit_period: int = 10,
                 atr_period: int = 20, risk_pct: float = 0.01,
                 account_size: float = 100000.0, max_units: int = 4,
                 pyramid_spacing_atr: float = 0.5):
        super().__init__("turtle", symbol)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct
        self.account_size = account_size
        self.max_units = max_units
        self.pyramid_spacing_atr = pyramid_spacing_atr
        self._current_units = 0
        self._last_breakout_type: Optional[str] = None
        self._last_entry_price: Optional[float] = None
        self._trailing_stop: Optional[float] = None

    def compute_position_size(self, atr: float, price: float) -> float:
        """Compute position size using Turtle N-based sizing."""
        if atr <= 0 or price <= 0:
            return 0.0
        risk_amount = self.account_size * self.risk_pct
        unit_size = risk_amount / atr
        return max(unit_size, 0.0)

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < max(self.entry_period, self.atr_period) + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        regime = self._detect_regime(candles)
        atr = ATR(self.atr_period).compute(highs, lows, closes)
        if np.isnan(atr):
            return None
        max_units = int(self._adapt_param(float(self.max_units), regime,
                                           trending_mult=1.5, volatile_mult=0.5))
        highest = np.max(highs[-self.entry_period - 1:-1])
        lowest = np.min(lows[-self.entry_period - 1:-1])
        current = closes[-1]
        position_size = self.compute_position_size(atr, current)
        pyramid_signal = None
        if self._last_breakout_type == "up" and self._last_entry_price is not None:
            if current >= self._last_entry_price + self.pyramid_spacing_atr * atr:
                if self._current_units < max_units:
                    self._current_units += 1
                    self._last_entry_price = current
                    self._trailing_stop = current - 2 * atr
                    pyramid_signal = Signal(
                        id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                        symbol=self.symbol, direction=SignalDirection.LONG,
                        strength=min(position_size / self.account_size, 1.0) * 0.5,
                        strategy_id=self.strategy_id,
                        indicators={"type": "pyramid", "atr": atr, "units": self._current_units},
                    )
        if pyramid_signal is not None:
            self.signals_generated += 1
            return pyramid_signal
        if current > highest and self._current_units < max_units:
            self._current_units += 1
            self._last_breakout_type = "up"
            self._last_entry_price = current
            self._trailing_stop = current - 2 * atr
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(position_size / self.account_size, 1.0),
                strategy_id=self.strategy_id,
                indicators={"breakout_level": highest, "atr": atr, "units": self._current_units,
                            "position_size": position_size},
            )
        elif current < lowest and self._current_units < max_units:
            self._current_units += 1
            self._last_breakout_type = "down"
            self._last_entry_price = current
            self._trailing_stop = current + 2 * atr
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(position_size / self.account_size, 1.0),
                strategy_id=self.strategy_id,
                indicators={"breakout_level": lowest, "atr": atr, "units": self._current_units,
                            "position_size": position_size},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position = None) -> bool:
        """Check exit conditions: trailing stop or counter-breakout."""
        if len(candles) < self.exit_period + 1:
            return False
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        # Trailing stop exit
        if self._trailing_stop is not None:
            if self._last_breakout_type == "up" and closes[-1] < self._trailing_stop:
                self._current_units = 0
                self._last_breakout_type = None
                return True
            if self._last_breakout_type == "down" and closes[-1] > self._trailing_stop:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        # Counter-breakout exit
        if self._last_breakout_type == "up":
            exit_level = np.min(lows[-self.exit_period - 1:-1])
            if closes[-1] < exit_level:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        elif self._last_breakout_type == "down":
            exit_level = np.max(highs[-self.exit_period - 1:-1])
            if closes[-1] > exit_level:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        return False

    def should_exit_with_position(self, candles: List[Candle], position: Position) -> bool:
        """Override to delegate to the unified should_exit."""
        return self.should_exit(candles)


__all__ = [
    "TurtleTradingStrategy",
]
