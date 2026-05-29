"""Mean-reversion strategy implementation."""

import numpy as np
from typing import Optional, List
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import BollingerBands, RSI, compute_zscore, compute_hurst_exponent
from acms.signals import MarketRegime
from acms.strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Mean-reversion using Bollinger Bands + RSI + z-score + Hurst confirmation."""

    def __init__(self, symbol: str, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, zscore_threshold: float = 2.0):
        super().__init__("mean_reversion", symbol)
        self.bb = BollingerBands(bb_period, bb_std)
        self.rsi = RSI(rsi_period)
        self.zscore_threshold = zscore_threshold

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 50:
            return None
        closes = np.array([c.close for c in candles])
        bb_result = self.bb.compute(closes)
        rsi_val = self.rsi.compute(closes)
        zscore = compute_zscore(closes[-30:])
        hurst = compute_hurst_exponent(closes[-100:]) if len(closes) >= 100 else 0.5
        if bb_result is None or np.isnan(rsi_val):
            return None
        pct_b = bb_result["percent_b"]
        hurst_confirm = hurst < 0.55
        regime = self._detect_regime(candles)
        z_thresh = self._adapt_param(self.zscore_threshold, regime, mr_mult=0.8, trending_mult=1.5)
        if pct_b < 0.05 and rsi_val < 35 and zscore < -z_thresh:
            strength = 0.8 if hurst_confirm else 0.5
            strength *= 1.2 if regime == MarketRegime.MEAN_REVERTING else 1.0
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(strength, 1.0), strategy_id=self.strategy_id,
                indicators={"bb_pct_b": pct_b, "rsi": rsi_val, "zscore": zscore, "hurst": hurst},
            )
        elif pct_b > 0.95 and rsi_val > 65 and zscore > z_thresh:
            strength = 0.8 if hurst_confirm else 0.5
            strength *= 1.2 if regime == MarketRegime.MEAN_REVERTING else 1.0
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(strength, 1.0), strategy_id=self.strategy_id,
                indicators={"bb_pct_b": pct_b, "rsi": rsi_val, "zscore": zscore, "hurst": hurst},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        bb_result = self.bb.compute(closes)
        if bb_result is None:
            return False
        if position.side == Side.BUY and closes[-1] >= bb_result["middle"]:
            return True
        if position.side == Side.SELL and closes[-1] <= bb_result["middle"]:
            return True
        return False


__all__ = [
    "MeanReversionStrategy",
]
