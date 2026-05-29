"""Market-making strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import ATR
from acms.strategies.base import Strategy


class MarketMakingStrategy(Strategy):
    """Market-making strategy with adverse selection protection.

    Provides continuous two-sided quotes while managing:
    - Inventory risk through skew adjustment
    - Adverse selection through toxic flow detection
    - Spread optimization based on volatility
    - Dynamic spread widening in volatile conditions
    """

    def __init__(self, symbol: str, base_spread_bps: float = 10.0,
                 inventory_limit: float = 5.0, skew_factor: float = 0.5,
                 min_profit_bps: float = 2.0,
                 adverse_selection_threshold: float = 0.7,
                 volatility_spread_mult: float = 2.0):
        super().__init__("market_making", symbol)
        self.base_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.skew_factor = skew_factor
        self.min_profit_bps = min_profit_bps
        self.adverse_selection_threshold = adverse_selection_threshold
        self.volatility_spread_mult = volatility_spread_mult
        self._inventory: float = 0.0
        self._recent_trades: List[Dict] = []
        self._toxic_flow_score: float = 0.0

    def compute_quotes(self, mid_price: float, atr: float, atr_pct: float) -> Dict:
        """Compute bid/ask quotes with inventory skew and volatility adjustment."""
        base_spread = mid_price * self.base_spread_bps / 10000.0
        vol_mult = 1.0
        if atr_pct > 5.0:
            vol_mult = self.volatility_spread_mult
        elif atr_pct > 3.0:
            vol_mult = 1.0 + (atr_pct - 3.0) / 2.0 * (self.volatility_spread_mult - 1.0)
        spread = base_spread * vol_mult
        inventory_ratio = self._inventory / self.inventory_limit if self.inventory_limit > 0 else 0
        skew_bps = self.skew_factor * inventory_ratio * spread
        half_spread = spread / 2.0
        bid = mid_price - half_spread - skew_bps
        ask = mid_price + half_spread - skew_bps
        if (ask - bid) < mid_price * self.min_profit_bps / 10000.0:
            min_half = mid_price * self.min_profit_bps / 20000.0
            bid = mid_price - min_half - skew_bps
            ask = mid_price + min_half - skew_bps
        return {
            "bid": bid,
            "ask": ask,
            "spread_bps": (ask - bid) / mid_price * 10000,
            "skew_bps": skew_bps / mid_price * 10000,
            "vol_mult": vol_mult,
        }

    def detect_adverse_selection(self, trade_side: str, trade_size: float,
                                 avg_trade_size: float, price_impact: float) -> Dict:
        """Detect adverse selection (toxic order flow)."""
        size_score = min(trade_size / max(avg_trade_size, 1e-10), 5.0) / 5.0
        impact_score = min(abs(price_impact) / 0.01, 1.0)
        self._toxic_flow_score = 0.7 * self._toxic_flow_score + 0.3 * (size_score + impact_score) / 2.0
        is_toxic = self._toxic_flow_score > self.adverse_selection_threshold
        action = "cancel" if is_toxic else "widen" if self._toxic_flow_score > 0.5 else "normal"
        return {"is_toxic": is_toxic, "score": self._toxic_flow_score, "action": action}

    def record_trade(self, side: str, size: float, price: float) -> None:
        """Record a trade for inventory and flow analysis."""
        self._recent_trades.append({"side": side, "size": size, "price": price, "time": datetime.utcnow()})
        if side == "buy":
            self._inventory += size
        else:
            self._inventory -= size
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 30:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        atr = ATR(14).compute(highs, lows, closes)
        if np.isnan(atr) or closes[-1] == 0:
            return None
        atr_pct = atr / closes[-1] * 100
        mid_price = closes[-1]
        quotes = self.compute_quotes(mid_price, atr, atr_pct)
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        flow = self.detect_adverse_selection("buy", volumes[-1], avg_vol, 0.0)
        if flow["action"] == "cancel":
            return None
        if self._inventory > self.inventory_limit * 0.8:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=0.3, strategy_id=self.strategy_id,
                indicators={"action": "reduce_inventory", "inventory": self._inventory,
                            "spread_bps": quotes["spread_bps"], "toxic_score": self._toxic_flow_score},
            )
        elif self._inventory < -self.inventory_limit * 0.8:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=0.3, strategy_id=self.strategy_id,
                indicators={"action": "reduce_inventory", "inventory": self._inventory,
                            "spread_bps": quotes["spread_bps"], "toxic_score": self._toxic_flow_score},
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
        if position.side == Side.BUY and closes[-1] < position.entry_price - 2 * atr:
            return True
        if position.side == Side.SELL and closes[-1] > position.entry_price + 2 * atr:
            return True
        return False


__all__ = [
    "MarketMakingStrategy",
]
