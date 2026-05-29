"""Exchange Adapter for ACMS."""

import asyncio
import hashlib
import hmac
import time
import json
import logging
from typing import Optional, Callable, Dict, List, Any
from datetime import datetime

import httpx

from acms.core import (
    Order, Trade, Position, Candle, Tick, Side, OrderType,
    OrderStatus, TimeInForce, ExchangeId,
)
from acms.exchanges.base import (
    ExchangeAdapter, ExchangeCredentials, ExchangeError,
    RateLimitError, NetworkError, RateLimiter, LocalOrderBook,
)

logger = logging.getLogger(__name__)


class PaperTradingAdapter(ExchangeAdapter):
    """Paper trading adapter for simulation.

    Simulates order execution with configurable slippage and latency.
    Maintains local position and balance state.
    """

    def __init__(self, initial_balance: float = 100000.0,
                 slippage_bps: float = 5.0, commission_bps: float = 10.0):
        super().__init__(ExchangeCredentials())
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self.last_prices: dict[str, float] = {}
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self._order_counter = 0
        self._trade_counter = 0

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        return []

    async def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        price = self.last_prices.get(symbol, 50000.0)
        return {
            "bids": [(price - i * 0.1, 1.0) for i in range(depth)],
            "asks": [(price + i * 0.1, 1.0) for i in range(depth)],
        }

    async def place_order(self, order: Order) -> Order:
        price = self.last_prices.get(order.symbol, order.price or 0)
        slippage = price * self.slippage_bps / 10000

        if order.side == Side.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        commission = order.quantity * fill_price * self.commission_bps / 10000

        if order.order_type == OrderType.MARKET or order.order_type == "market":
            order.average_fill_price = fill_price
            order.filled_quantity = order.quantity
            order.status = OrderStatus.FILLED
            order.commission = commission

            self._update_balance_and_position(order, fill_price)

            self._trade_counter += 1
            self.trades.append(Trade(
                id=f"ptrade_{self._trade_counter}", order_id=order.id,
                symbol=order.symbol, side=order.side,
                quantity=order.quantity, price=fill_price,
                commission=commission, timestamp=datetime.utcnow(),
                exchange="paper",
            ))
        else:
            order.status = OrderStatus.SUBMITTED

        self.orders[order.id] = order
        return order

    def _update_balance_and_position(self, order: Order, fill_price: float) -> None:
        """Update balance and position after order fill."""
        notional = order.quantity * fill_price
        commission = order.commission

        if order.side == Side.BUY:
            self.balance -= (notional + commission)
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                new_qty = pos.quantity + order.quantity
                new_entry = (pos.entry_price * pos.quantity + fill_price * order.quantity) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_entry
                pos.mark_price = fill_price
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol, side=Side.BUY,
                    quantity=order.quantity, entry_price=fill_price,
                    mark_price=fill_price, exchange="paper",
                )
        else:
            self.balance += (notional - commission)
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.quantity
                pos.realized_pnl += (fill_price - pos.entry_price) * order.quantity
                pos.mark_price = fill_price
                if pos.quantity <= 1e-10:
                    del self.positions[order.symbol]

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        return self.orders.get(order_id, Order(
            id=order_id, symbol=symbol, side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED, quantity=0,
        ))

    async def get_positions(self) -> list[Position]:
        for symbol, pos in self.positions.items():
            last_price = self.last_prices.get(symbol, pos.mark_price)
            pos.mark_price = last_price
            pos.unrealized_pnl = (last_price - pos.entry_price) * pos.quantity
        return list(self.positions.values())

    async def get_balance(self) -> dict:
        return {"USDT": {"free": self.balance, "locked": 0.0}}

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        pass

    async def subscribe_order_book(self, symbol: str, callback: Callable):
        pass

    def update_price(self, symbol: str, price: float) -> None:
        """Update the simulated price for a symbol.

        Args:
            symbol: Trading pair symbol.
            price: New market price.
        """
        self.last_prices[symbol] = price


# Multi-Exchange Arbitrage Detection

__all__ = ['PaperTradingAdapter']
