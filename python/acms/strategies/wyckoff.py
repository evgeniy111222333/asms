"""Wyckoff strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import ATR
from acms.strategies.base import Strategy


class WyckoffStrategy(Strategy):
    """Wyckoff accumulation/distribution detection via volume spread analysis.

    Identifies Wyckoff phases using volume and price action:
    - Accumulation: PS, SC, AR, ST, Spring, SOS, LPS
    - Distribution: PSY, BC, AR, ST, UTAD, SOW, LPSY

    Volume Spread Analysis (VSA) examines the relationship between
    price spread (range) and volume to identify professional activity.
    """

    def __init__(self, symbol: str, lookback: int = 100, volume_threshold: float = 2.0,
                 spring_threshold: float = 0.02):
        super().__init__("wyckoff", symbol)
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.spring_threshold = spring_threshold

    def _vsa_analysis(self, closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, volumes: np.ndarray) -> Dict[str, bool]:
        """Volume Spread Analysis to detect buying/selling climaxes."""
        n = len(closes)
        if n < 3:
            return {"buying_climax": False, "selling_climax": False,
                    "no_demand": False, "no_supply": False}
        avg_vol = np.mean(volumes)
        avg_range = np.mean(highs - lows)
        spread = highs[-1] - lows[-1]
        vol = volumes[-1]
        result = {
            "buying_climax": False,
            "selling_climax": False,
            "no_demand": False,
            "no_supply": False,
        }
        if vol > avg_vol * self.volume_threshold:
            if closes[-1] > closes[-2] and spread > avg_range * 1.5:
                result["buying_climax"] = True
            elif closes[-1] < closes[-2] and spread > avg_range * 1.5:
                result["selling_climax"] = True
        elif vol < avg_vol * 0.5:
            if closes[-1] < closes[-2]:
                result["no_demand"] = True
            else:
                result["no_supply"] = True
        return result

    def detect_accumulation(self, closes: np.ndarray, volumes: np.ndarray,
                            lows: np.ndarray) -> Dict[str, bool]:
        """Detect Wyckoff accumulation phases."""
        phases = {
            "selling_climax": False, "automatic_rally": False,
            "spring": False, "sign_of_strength": False,
        }
        if len(closes) < self.lookback:
            return phases
        recent_closes = closes[-self.lookback:]
        recent_volumes = volumes[-self.lookback:]
        recent_lows = lows[-self.lookback:]
        avg_vol = np.mean(recent_volumes)
        vol_spike = recent_volumes > avg_vol * self.volume_threshold
        price_decline = np.diff(recent_closes) / recent_closes[:-1] < -0.02
        if np.any(vol_spike[1:] & price_decline):
            phases["selling_climax"] = True
        support = np.min(recent_lows[:len(recent_lows) // 2])
        if recent_lows[-1] < support * (1 - self.spring_threshold):
            if closes[-1] > support:
                phases["spring"] = True
        if phases["selling_climax"]:
            sc_idx = np.argmax(vol_spike[1:] & price_decline)
            if sc_idx < len(recent_closes) - 5:
                post_sc = recent_closes[sc_idx + 1:sc_idx + 6]
                if len(post_sc) > 0 and np.mean(np.diff(post_sc)) > 0:
                    phases["automatic_rally"] = True
        recent_rally = closes[-5:]
        recent_vol5 = volumes[-5:]
        if len(recent_rally) >= 5:
            if np.mean(np.diff(recent_rally)) > 0 and np.mean(recent_vol5) > avg_vol * 1.5:
                phases["sign_of_strength"] = True
        return phases

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.lookback:
            return None
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        lows = np.array([c.low for c in candles])
        highs = np.array([c.high for c in candles])
        phases = self.detect_accumulation(closes, volumes, lows)
        vsa = self._vsa_analysis(closes, highs, lows, volumes)
        if phases["spring"] or phases["sign_of_strength"] or vsa["no_supply"]:
            strength = 0.7 if phases["spring"] else 0.5
            if vsa["no_supply"]:
                strength = max(strength, 0.6)
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=strength, strategy_id=self.strategy_id,
                indicators={**phases, **vsa},
            )
        resistance = np.max(highs[:len(highs) // 2])
        if highs[-1] > resistance * (1 + self.spring_threshold) and closes[-1] < resistance:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=0.6, strategy_id=self.strategy_id,
                indicators={"utad": True, **phases, **vsa},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr = ATR(14).compute(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]), closes,
        )
        if np.isnan(atr):
            return False
        if position.side == Side.BUY and closes[-1] < position.entry_price - 3 * atr:
            return True
        if position.side == Side.SELL and closes[-1] > position.entry_price + 3 * atr:
            return True
        return False


__all__ = [
    "WyckoffStrategy",
]
