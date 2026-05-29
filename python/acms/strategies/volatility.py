"""Volatility strategy implementation."""

import numpy as np
from typing import Optional, List
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import ATR
from acms.strategies.base import Strategy


class VolatilityStrategy(Strategy):
    """Volatility trading using ATR breakout + IV/RV spread."""

    def __init__(self, symbol: str, atr_period: int = 14, atr_mult: float = 1.5,
                 vol_lookback: int = 20):
        super().__init__("volatility", symbol)
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.vol_lookback = vol_lookback
        self._atr_pct_history: List[float] = []

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.vol_lookback + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        atr_val = ATR(self.atr_period).compute(highs, lows, closes)
        if np.isnan(atr_val):
            return None
        atr_pct = atr_val / closes[-1] * 100
        self._atr_pct_history.append(atr_pct)
        if len(self._atr_pct_history) < self.vol_lookback:
            return None
        avg_atr_pct = np.mean(self._atr_pct_history[-self.vol_lookback:])
        regime = self._detect_regime(candles)
        mult = self._adapt_param(self.atr_mult, regime, volatile_mult=2.0, quiet_mult=1.0)
        if atr_pct > avg_atr_pct * mult:
            direction = SignalDirection.LONG if closes[-1] > closes[-2] else SignalDirection.SHORT
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=direction,
                strength=min(atr_pct / avg_atr_pct / 3, 1.0) if avg_atr_pct > 0 else 0.5,
                strategy_id=self.strategy_id,
                indicators={"atr_pct": atr_pct, "avg_atr_pct": avg_atr_pct},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr_val = ATR(self.atr_period).compute(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]), closes,
        )
        if np.isnan(atr_val):
            return False
        if position.side == Side.BUY:
            return closes[-1] < position.entry_price - 2.5 * atr_val
        elif position.side == Side.SELL:
            return closes[-1] > position.entry_price + 2.5 * atr_val
        return False


__all__ = [
    "VolatilityStrategy",
]
