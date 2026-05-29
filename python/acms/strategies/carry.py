"""Carry trade strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.strategies.base import Strategy


class CarryStrategy(Strategy):
    """Carry trade strategy with cross-exchange funding rate arbitrage.

    Profits from:
    1. Funding rate differentials (positive/negative rates)
    2. Cross-exchange price discrepancies
    3. Cross-exchange funding rate arbitrage (same asset, different rates)
    """

    def __init__(self, symbol: str, funding_threshold: float = 0.01,
                 position_period_hours: int = 8,
                 arb_threshold_bps: float = 20.0,
                 funding_arb_min_spread: float = 0.005):
        super().__init__("carry", symbol)
        self.funding_threshold = funding_threshold
        self.position_period_hours = position_period_hours
        self.arb_threshold_bps = arb_threshold_bps
        self.funding_arb_min_spread = funding_arb_min_spread
        self._current_funding_rate: Dict[str, float] = {}

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate for a symbol from exchange."""
        return self._current_funding_rate.get(symbol)

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate carry strategy using funding rate data."""
        if not candles:
            return None
        symbol = candles[0].symbol if hasattr(candles[0], 'symbol') else self.symbol
        funding_rate = self._current_funding_rate.get(symbol)
        if funding_rate is None:
            funding_rate = self._fetch_funding_rate(symbol)
        if funding_rate is not None:
            return self.evaluate_funding(funding_rate, 0.0)
        return None

    def evaluate_funding(self, funding_rate: float, predicted_rate: float) -> Optional[Signal]:
        """Evaluate based on current and predicted funding rate."""
        if funding_rate < -self.funding_threshold:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(abs(funding_rate) / 0.1, 1.0),
                strategy_id=self.strategy_id,
                indicators={"funding_rate": funding_rate, "predicted_rate": predicted_rate},
            )
        elif funding_rate > self.funding_threshold:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(abs(funding_rate) / 0.1, 1.0),
                strategy_id=self.strategy_id,
                indicators={"funding_rate": funding_rate, "predicted_rate": predicted_rate},
            )
        return None

    def detect_cross_exchange_arbitrage(self, price_a: float, price_b: float,
                                        fee_bps: float = 10.0) -> Optional[Dict]:
        """Detect cross-exchange price arbitrage opportunity."""
        if price_a <= 0 or price_b <= 0:
            return None
        spread_bps = abs(price_a - price_b) / min(price_a, price_b) * 10000
        if spread_bps > self.arb_threshold_bps + fee_bps:
            buy_exchange = "A" if price_a < price_b else "B"
            sell_exchange = "B" if price_a < price_b else "A"
            return {
                "spread_bps": spread_bps,
                "net_profit_bps": spread_bps - fee_bps,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": min(price_a, price_b),
                "sell_price": max(price_a, price_b),
            }
        return None

    def detect_funding_rate_arbitrage(self, funding_rate_a: float,
                                      funding_rate_b: float,
                                      fee_rate: float = 0.001) -> Optional[Dict]:
        """Detect cross-exchange funding rate arbitrage."""
        spread = funding_rate_a - funding_rate_b
        net_profit = abs(spread) - fee_rate * 2
        if abs(spread) > self.funding_arb_min_spread and net_profit > 0:
            if spread > 0:
                short_exchange = "A"
                long_exchange = "B"
            else:
                short_exchange = "B"
                long_exchange = "A"
            return {
                "funding_spread": abs(spread),
                "net_profit": net_profit,
                "short_exchange": short_exchange,
                "long_exchange": long_exchange,
                "rate_a": funding_rate_a,
                "rate_b": funding_rate_b,
            }
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Check if carry position should be exited."""
        symbol = position.symbol if hasattr(position, 'symbol') else self.symbol
        funding_rate = self._current_funding_rate.get(symbol)
        if funding_rate is not None:
            if position.side == Side.BUY and funding_rate > self.funding_threshold:
                return True
            if position.side == Side.SELL and funding_rate < -self.funding_threshold:
                return True
        return False


__all__ = [
    "CarryStrategy",
]
