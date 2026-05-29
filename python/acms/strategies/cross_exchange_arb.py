"""Cross-exchange arbitrage strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from collections import deque

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.strategies.base import Strategy


class CrossExchangeArbitrageStrategy(Strategy):
    """Cross-Exchange Arbitrage: detect and exploit price differences.

    Monitors the same asset across multiple exchanges and generates
    signals when the price differential exceeds transaction costs
    plus a minimum profit threshold.
    """

    def __init__(self, symbol: str, exchanges: Optional[List[str]] = None,
                 min_profit_bps: float = 5.0, fee_bps: float = 10.0,
                 latency_buffer_bps: float = 3.0):
        super().__init__("cross_exchange_arb", symbol)
        self.exchanges = exchanges or ["exchange_a", "exchange_b"]
        self.min_profit_bps = min_profit_bps
        self.fee_bps = fee_bps
        self.latency_buffer_bps = latency_buffer_bps
        self._price_history: Dict[str, deque] = {ex: deque(maxlen=100) for ex in self.exchanges}
        self._exchange_adapters: Dict[str, object] = {}
        self.min_spread = (self.min_profit_bps + self.fee_bps + self.latency_buffer_bps) / 10000.0
        self.exit_spread = self.min_spread * 0.5

    def update_price(self, exchange: str, price: float) -> None:
        """Update price for an exchange."""
        if exchange in self._price_history:
            self._price_history[exchange].append((datetime.utcnow(), price))

    def detect_arbitrage(self, prices: Dict[str, float]) -> Optional[Dict]:
        """Detect arbitrage opportunities across exchanges."""
        if len(prices) < 2:
            return None
        best_opportunity = None
        best_net_profit = 0.0
        exchanges_list = list(prices.keys())
        for i in range(len(exchanges_list)):
            for j in range(i + 1, len(exchanges_list)):
                ex_a = exchanges_list[i]
                ex_b = exchanges_list[j]
                price_a = prices[ex_a]
                price_b = prices[ex_b]
                if price_a <= 0 or price_b <= 0:
                    continue
                spread_bps = abs(price_a - price_b) / min(price_a, price_b) * 10000
                total_cost = self.fee_bps + self.latency_buffer_bps + self.min_profit_bps
                if spread_bps > total_cost:
                    net_profit = spread_bps - self.fee_bps - self.latency_buffer_bps
                    if net_profit > best_net_profit:
                        best_net_profit = net_profit
                        buy_exchange = ex_a if price_a < price_b else ex_b
                        sell_exchange = ex_b if price_a < price_b else ex_a
                        best_opportunity = {
                            "spread_bps": spread_bps,
                            "net_profit_bps": net_profit,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": min(price_a, price_b),
                            "sell_price": max(price_a, price_b),
                            "total_cost_bps": self.fee_bps + self.latency_buffer_bps,
                        }
        return best_opportunity

    def register_exchange_adapter(self, exchange_name: str, adapter: object) -> None:
        """Register an exchange adapter for cross-exchange price queries."""
        self._exchange_adapters[exchange_name] = adapter

    def _get_current_spread(self, symbol: str) -> Optional[float]:
        """Get current best spread across exchanges for a symbol."""
        prices: Dict[str, float] = {}
        for exchange_name, price_deque in self._price_history.items():
            if price_deque:
                _, price = price_deque[-1]
                prices[exchange_name] = price
        if len(prices) < 2:
            return None
        best_spread = None
        base_price = None
        for ex, p in prices.items():
            if base_price is None:
                base_price = p
                continue
            if base_price > 0:
                spread = (p - base_price) / base_price
                if best_spread is None or abs(spread) > abs(best_spread):
                    best_spread = spread
        return best_spread

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate cross-exchange arbitrage from candles of the primary exchange."""
        if not candles:
            return None
        symbol = candles[0].symbol if hasattr(candles[0], 'symbol') else self.symbol
        primary_price = candles[-1].close
        best_arb = None
        for exchange_name, exchange_adapter in self._exchange_adapters.items():
            try:
                ticker = exchange_adapter.get_ticker(symbol)
                if ticker:
                    other_price = ticker.get("last", 0)
                    if other_price > 0:
                        spread_pct = (other_price - primary_price) / primary_price
                        if best_arb is None or abs(spread_pct) > abs(best_arb[1]):
                            best_arb = (exchange_name, spread_pct)
            except Exception:
                continue
        if best_arb is None:
            current_spread = self._get_current_spread(symbol)
            if current_spread is not None and abs(current_spread) > self.min_spread:
                return Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol,
                    direction=SignalDirection.LONG if current_spread > 0 else SignalDirection.SHORT,
                    strength=min(1.0, abs(current_spread) / self.min_spread),
                    strategy_id=self.strategy_id,
                    indicators={"spread": current_spread, "source": "price_history"},
                )
            return None
        if abs(best_arb[1]) > self.min_spread:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol,
                direction=SignalDirection.LONG if best_arb[1] > 0 else SignalDirection.SHORT,
                strength=min(1.0, abs(best_arb[1]) / self.min_spread),
                strategy_id=self.strategy_id,
                indicators={"target_exchange": best_arb[0], "spread": best_arb[1]},
                metadata={"target_exchange": best_arb[0], "spread": best_arb[1]},
            )
        return None

    def evaluate_multi_exchange(self, prices: Dict[str, float]) -> Optional[Signal]:
        """Evaluate arbitrage opportunity across exchanges."""
        arb = self.detect_arbitrage(prices)
        if arb is None:
            return None
        self.signals_generated += 1
        return Signal(
            id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=self.symbol, direction=SignalDirection.LONG,
            strength=min(arb["net_profit_bps"] / 20.0, 1.0),
            strategy_id=self.strategy_id,
            indicators=arb,
        )

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Check if arbitrage position should be exited based on spread convergence."""
        if position is None:
            return True
        symbol = position.symbol if hasattr(position, 'symbol') else self.symbol
        current_spread = self._get_current_spread(symbol)
        if current_spread is not None:
            if abs(current_spread) < self.exit_spread:
                return True
            if position.side == Side.BUY and current_spread < 0:
                return True
            if position.side == Side.SELL and current_spread > 0:
                return True
        return False


__all__ = [
    "CrossExchangeArbitrageStrategy",
]
