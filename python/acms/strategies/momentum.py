"""Momentum strategy implementations."""

import numpy as np
from typing import Optional, List
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import EMA, ADX, ATR, RSI, MACD, Supertrend
from acms.signals import MarketRegime
from acms.strategies.base import Strategy


class TrendFollowingMomentum(Strategy):
    """Trend following using EMA crossover + ADX filter."""

    def __init__(self, symbol: str, fast_period: int = 20, slow_period: int = 50,
                 adx_threshold: float = 25.0):
        super().__init__("momentum_trend", symbol)
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.adx = ADX(14)
        self.adx_threshold = adx_threshold
        self._prev_fast_above_slow = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 60:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        fast = self.fast_ema.compute(closes)
        slow = self.slow_ema.compute(closes)
        if np.isnan(fast[-1]) or np.isnan(slow[-1]):
            return None
        fast_above = fast[-1] > slow[-1]
        regime = self._detect_regime(candles)
        threshold = self._adapt_param(self.adx_threshold, regime, trending_mult=0.8, mr_mult=1.5)
        adx_val = self.adx.compute(highs, lows, closes)
        if np.isnan(adx_val) or adx_val < threshold:
            self._prev_fast_above_slow = fast_above
            return None
        signal = None
        if self._prev_fast_above_slow is not None:
            if fast_above and not self._prev_fast_above_slow:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=min(adx_val / 50.0, 1.0), strategy_id=self.strategy_id,
                    indicators={"fast_ema": fast[-1], "slow_ema": slow[-1], "adx": adx_val, "regime": regime.value},
                )
            elif not fast_above and self._prev_fast_above_slow:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=min(adx_val / 50.0, 1.0), strategy_id=self.strategy_id,
                    indicators={"fast_ema": fast[-1], "slow_ema": slow[-1], "adx": adx_val, "regime": regime.value},
                )
        self._prev_fast_above_slow = fast_above
        self.signals_generated += 1
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        fast = self.fast_ema.compute(closes)
        slow = self.slow_ema.compute(closes)
        if np.isnan(fast[-1]) or np.isnan(slow[-1]):
            return False
        if position.side == Side.BUY and fast[-1] < slow[-1]:
            return True
        if position.side == Side.SELL and fast[-1] > slow[-1]:
            return True
        return False


class BreakoutMomentum(Strategy):
    """Breakout strategy using Donchian channels + volume confirmation."""

    def __init__(self, symbol: str, channel_period: int = 20, volume_mult: float = 1.5):
        super().__init__("momentum_breakout", symbol)
        self.channel_period = channel_period
        self.volume_mult = volume_mult

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.channel_period + 1:
            return None
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        upper = np.max(highs[-self.channel_period - 1:-1])
        lower = np.min(lows[-self.channel_period - 1:-1])
        avg_vol = np.mean(volumes[-self.channel_period:-1])
        current_close = closes[-1]
        current_vol = volumes[-1]
        vol_mult = self._adapt_param(self.volume_mult, self._detect_regime(candles),
                                      volatile_mult=0.5, quiet_mult=2.0)
        vol_confirm = current_vol > avg_vol * vol_mult if avg_vol > 0 else False
        if current_close > upper and vol_confirm:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG, strength=0.7,
                strategy_id=self.strategy_id,
                indicators={"upper_channel": upper, "volume_ratio": current_vol / avg_vol if avg_vol > 0 else 0},
            )
        elif current_close < lower and vol_confirm:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT, strength=0.7,
                strategy_id=self.strategy_id,
                indicators={"lower_channel": lower, "volume_ratio": current_vol / avg_vol if avg_vol > 0 else 0},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr = ATR(14).compute(np.array([c.high for c in candles]), np.array([c.low for c in candles]), closes)
        if np.isnan(atr):
            return False
        if position.side == Side.BUY:
            return closes[-1] < position.entry_price - 2 * atr
        elif position.side == Side.SELL:
            return closes[-1] > position.entry_price + 2 * atr
        return False


class RSIMomentum(Strategy):
    """RSI momentum - buy on RSI cross above 30, sell on cross below 70."""

    def __init__(self, symbol: str, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("momentum_rsi", symbol)
        self.rsi = RSI(period)
        self.oversold = oversold
        self.overbought = overbought
        self._prev_rsi = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        closes = np.array([c.close for c in candles])
        rsi_val = self.rsi.compute(closes)
        if np.isnan(rsi_val):
            self._prev_rsi = None
            return None
        signal = None
        if self._prev_rsi is not None:
            if self._prev_rsi < self.oversold and rsi_val >= self.oversold:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=(self.oversold - self._prev_rsi) / self.oversold,
                    strategy_id=self.strategy_id, indicators={"rsi": rsi_val},
                )
            elif self._prev_rsi > self.overbought and rsi_val <= self.overbought:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=(self._prev_rsi - self.overbought) / (100 - self.overbought),
                    strategy_id=self.strategy_id, indicators={"rsi": rsi_val},
                )
        self._prev_rsi = rsi_val
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        rsi_val = self.rsi.compute(closes)
        if np.isnan(rsi_val):
            return False
        if position.side == Side.BUY and rsi_val > self.overbought:
            return True
        if position.side == Side.SELL and rsi_val < self.oversold:
            return True
        return False


class MACDMomentum(Strategy):
    """MACD histogram momentum strategy."""

    def __init__(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("momentum_macd", symbol)
        self.macd = MACD(fast, slow, signal)
        self._prev_hist = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        closes = np.array([c.close for c in candles])
        result = self.macd.compute(closes)
        if result is None:
            self._prev_hist = None
            return None
        hist = result["histogram"]
        signal = None
        if self._prev_hist is not None:
            if self._prev_hist < 0 and hist > 0:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=min(abs(hist) / abs(closes[-1]) * 1000, 1.0) if closes[-1] != 0 else 0.5,
                    strategy_id=self.strategy_id,
                    indicators={"macd": result["macd"], "signal": result["signal"], "histogram": hist},
                )
            elif self._prev_hist > 0 and hist < 0:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=min(abs(hist) / abs(closes[-1]) * 1000, 1.0) if closes[-1] != 0 else 0.5,
                    strategy_id=self.strategy_id,
                    indicators={"macd": result["macd"], "signal": result["signal"], "histogram": hist},
                )
        self._prev_hist = hist
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        result = self.macd.compute(closes)
        if result is None:
            return False
        if position.side == Side.BUY and result["histogram"] < 0:
            return True
        if position.side == Side.SELL and result["histogram"] > 0:
            return True
        return False


class SupertrendMomentum(Strategy):
    """Supertrend momentum strategy."""

    def __init__(self, symbol: str, period: int = 10, multiplier: float = 3.0):
        super().__init__("momentum_supertrend", symbol)
        self.supertrend = Supertrend(period, multiplier)
        self._prev_direction = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 20:
            return None
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        result = self.supertrend.compute(highs, lows, closes)
        direction = result["direction"][-1]
        signal = None
        if self._prev_direction is not None:
            if direction == 1 and self._prev_direction == -1:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=0.6, strategy_id=self.strategy_id,
                    indicators={"supertrend": result["supertrend"][-1], "direction": 1},
                )
            elif direction == -1 and self._prev_direction == 1:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=0.6, strategy_id=self.strategy_id,
                    indicators={"supertrend": result["supertrend"][-1], "direction": -1},
                )
        self._prev_direction = direction
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        result = self.supertrend.compute(highs, lows, closes)
        direction = result["direction"][-1]
        if position.side == Side.BUY and direction == -1:
            return True
        if position.side == Side.SELL and direction == 1:
            return True
        return False


__all__ = [
    "TrendFollowingMomentum",
    "BreakoutMomentum",
    "RSIMomentum",
    "MACDMomentum",
    "SupertrendMomentum",
]
