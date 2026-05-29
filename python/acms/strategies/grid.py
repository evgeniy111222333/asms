"""Grid trading strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Side
from acms.indicators import ATR
from acms.strategies.base import Strategy


class GridTradingStrategy(Strategy):
    """Grid Trading with dynamic grid adjustment and inventory management.

    Places buy/sell orders at fixed price intervals (grid levels).
    Profits from price oscillations within a range.
    Grid levels adapt based on ATR. Inventory is managed per level.
    """

    def __init__(self, symbol: str, grid_levels: int = 10, grid_spacing_atr_mult: float = 0.5,
                 grid_atr_period: int = 14, position_per_grid: float = 0.01,
                 max_inventory: float = 1.0, take_profit_atr_mult: float = 1.0):
        super().__init__("grid_trading", symbol)
        self.grid_levels = grid_levels
        self.grid_spacing_atr_mult = grid_spacing_atr_mult
        self.grid_atr_period = grid_atr_period
        self.position_per_grid = position_per_grid
        self.max_inventory = max_inventory
        self.take_profit_atr_mult = take_profit_atr_mult
        self._grid_levels: List[float] = []
        self._center_price: Optional[float] = None
        self._inventory: float = 0.0
        self._filled_levels: Dict[float, Dict] = {}

    def compute_grid(self, current_price: float, atr: float) -> List[float]:
        """Compute dynamic grid levels based on current price and ATR."""
        regime_mult = 1.0
        spacing = atr * self.grid_spacing_atr_mult * regime_mult
        if spacing <= 0:
            spacing = current_price * 0.005
        self._center_price = current_price
        half = self.grid_levels // 2
        levels = [current_price + (i - half) * spacing for i in range(self.grid_levels)]
        self._grid_levels = sorted(levels)
        return self._grid_levels

    def get_grid_orders(self, current_price: float, atr: float) -> List[Dict]:
        """Generate grid orders with inventory management."""
        levels = self.compute_grid(current_price, atr)
        orders = []
        for level in levels:
            if abs(self._inventory) >= self.max_inventory:
                break
            if level < current_price:
                qty = min(self.position_per_grid, self.max_inventory - abs(self._inventory))
                orders.append({"price": level, "side": "buy", "quantity": qty})
                self._inventory += qty
            elif level > current_price:
                qty = min(self.position_per_grid, self.max_inventory - abs(self._inventory))
                orders.append({"price": level, "side": "sell", "quantity": qty})
                self._inventory -= qty
        return orders

    def record_fill(self, level: float, side: str, qty: float, fill_price: float) -> None:
        """Record a grid level fill for profit taking."""
        self._filled_levels[level] = {"side": side, "qty": qty, "entry_price": fill_price}

    def check_take_profit(self, current_price: float, atr: float) -> List[Dict]:
        """Check filled levels for take-profit opportunities."""
        close_orders = []
        tp_distance = atr * self.take_profit_atr_mult
        for level, info in list(self._filled_levels.items()):
            if info["side"] == "buy" and current_price >= level + tp_distance:
                pnl = (current_price - info["entry_price"]) * info["qty"]
                close_orders.append({"level": level, "side": "sell", "qty": info["qty"], "pnl": pnl})
                self._inventory -= info["qty"]
                del self._filled_levels[level]
            elif info["side"] == "sell" and current_price <= level - tp_distance:
                pnl = (info["entry_price"] - current_price) * info["qty"]
                close_orders.append({"level": level, "side": "buy", "qty": info["qty"], "pnl": pnl})
                self._inventory += info["qty"]
                del self._filled_levels[level]
        return close_orders

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.grid_atr_period + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        atr = ATR(self.grid_atr_period).compute(highs, lows, closes)
        if np.isnan(atr):
            return None
        current_price = closes[-1]
        grid = self.compute_grid(current_price, atr)
        buy_levels = [g for g in grid if g < current_price]
        if buy_levels and abs(current_price - buy_levels[-1]) < atr * 0.1:
            if self._inventory < self.max_inventory:
                return Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=0.4, strategy_id=self.strategy_id,
                    indicators={"grid_price": buy_levels[-1], "atr": atr, "inventory": self._inventory},
                )
        sell_levels = [g for g in grid if g > current_price]
        if sell_levels and abs(sell_levels[0] - current_price) < atr * 0.1:
            if self._inventory > -self.max_inventory:
                return Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=0.4, strategy_id=self.strategy_id,
                    indicators={"grid_price": sell_levels[0], "atr": atr, "inventory": self._inventory},
                )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        if not self._grid_levels:
            return False
        if position.side == Side.BUY:
            for level in self._grid_levels:
                if level > position.entry_price and closes[-1] >= level:
                    return True
        elif position.side == Side.SELL:
            for level in self._grid_levels:
                if level < position.entry_price and closes[-1] <= level:
                    return True
        return False


__all__ = [
    "GridTradingStrategy",
]
