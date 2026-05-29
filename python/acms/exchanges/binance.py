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


class BinanceAdapter(ExchangeAdapter):
    """Binance exchange adapter (REST + WebSocket).

    Supports spot and futures trading with full WebSocket streaming.
    """

    REST_URL = "https://api.binance.com"
    WS_URL = "wss://stream.binance.com:9443/ws"
    TESTNET_REST = "https://testnet.binance.vision"
    TESTNET_WS = "wss://testnet.binance.vision/ws"
    FUTURES_REST = "https://fapi.binance.com"
    FUTURES_WS = "wss://fstream.binance.com/ws"

    def __init__(self, credentials: ExchangeCredentials, testnet: bool = False):
        super().__init__(credentials, testnet)
        self.base_url = self.TESTNET_REST if testnet else self.REST_URL
        self.ws_url = self.TESTNET_WS if testnet else self.WS_URL
        self.rate_limiter = RateLimiter(max_requests=20, window_seconds=1.0)

    def _sign(self, params: dict) -> dict:
        """Sign request with HMAC SHA256."""
        if not self.credentials.api_secret:
            return params
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self.credentials.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.credentials.api_key}

    def _map_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        await self.rate_limiter.acquire()
        params = {"symbol": self._map_symbol(symbol), "interval": timeframe, "limit": limit}
        resp = await self.rest_client.get(f"{self.base_url}/api/v3/klines", params=params)
        data = resp.json()
        candles = []
        for k in data:
            candles.append(Candle(
                symbol=symbol, timeframe=timeframe,
                open_time=datetime.fromtimestamp(k[0] / 1000),
                close_time=datetime.fromtimestamp(k[6] / 1000),
                open=float(k[1]), high=float(k[2]), low=float(k[3]),
                close=float(k[4]), volume=float(k[5]),
                quote_volume=float(k[7]), trades=int(k[8]),
                taker_buy_volume=float(k[9]),
                taker_buy_quote_volume=float(k[10]),
            ))
        return candles

    async def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        await self.rate_limiter.acquire()
        params = {"symbol": self._map_symbol(symbol), "limit": depth}
        resp = await self.rest_client.get(f"{self.base_url}/api/v3/depth", params=params)
        data = resp.json()
        return {
            "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
            "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
        }

    async def place_order(self, order: Order) -> Order:
        await self.rate_limiter.acquire()
        params = {
            "symbol": self._map_symbol(order.symbol),
            "side": "BUY" if order.side == Side.BUY else "SELL",
            "type": self._order_type_map(order.order_type),
            "quantity": str(order.quantity),
            "timestamp": int(time.time() * 1000),
        }
        if order.price:
            params["price"] = str(order.price)
        if order.time_in_force:
            params["timeInForce"] = order.time_in_force.value.upper()
        params = self._sign(params)
        resp = await self.rest_client.post(
            f"{self.base_url}/api/v3/order", params=params, headers=self._headers()
        )
        data = resp.json()
        order.exchange_order_id = str(data.get("orderId", ""))
        order.status = OrderStatus.SUBMITTED
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self.rate_limiter.acquire()
        params = {
            "symbol": self._map_symbol(symbol), "orderId": order_id,
            "timestamp": int(time.time() * 1000),
        }
        params = self._sign(params)
        resp = await self.rest_client.delete(
            f"{self.base_url}/api/v3/order", params=params, headers=self._headers()
        )
        return resp.status_code == 200

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        await self.rate_limiter.acquire()
        params = {"symbol": self._map_symbol(symbol), "orderId": order_id,
                  "timestamp": int(time.time() * 1000)}
        params = self._sign(params)
        resp = await self.rest_client.get(
            f"{self.base_url}/api/v3/order", params=params, headers=self._headers()
        )
        data = resp.json()
        return Order(
            id=str(data.get("orderId", "")), symbol=symbol,
            side=Side.BUY if data.get("side") == "BUY" else Side.SELL,
            order_type=OrderType.LIMIT,
            status=self._status_map(data.get("status", "")),
            quantity=float(data.get("origQty", 0)),
            price=float(data.get("price", 0)),
            filled_quantity=float(data.get("executedQty", 0)),
        )

    async def get_positions(self) -> list[Position]:
        return []

    async def get_balance(self) -> dict:
        await self.rate_limiter.acquire()
        params = {"timestamp": int(time.time() * 1000)}
        params = self._sign(params)
        resp = await self.rest_client.get(
            f"{self.base_url}/api/v3/account", params=params, headers=self._headers()
        )
        data = resp.json()
        balances = {}
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            if free > 0 or locked > 0:
                balances[b["asset"]] = {"free": free, "locked": locked}
        return balances

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        stream_name = f"{self._map_symbol(symbol).lower()}@trade"
        url = f"{self.ws_url}/{stream_name}"

        def parse_trade(data: dict):
            return Tick(
                symbol=symbol, exchange="binance",
                price=float(data["p"]), quantity=float(data["q"]),
                side=Side.BUY if data["m"] is False else Side.SELL,
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                trade_id=str(data["t"]),
            )

        await self._ws_connect_with_reconnect(url, callback, parse_trade)

    async def subscribe_order_book(self, symbol: str, callback: Callable):
        stream_name = f"{self._map_symbol(symbol).lower()}@depth20@100ms"
        url = f"{self.ws_url}/{stream_name}"

        def parse_depth(data: dict):
            return {
                "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
                "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
            }

        await self._ws_connect_with_reconnect(url, callback, parse_depth)

    async def subscribe_kline(self, symbol: str, timeframe: str, callback: Callable):
        """Subscribe to kline/candlestick WebSocket stream.

        Args:
            symbol: Trading pair symbol.
            timeframe: Kline interval (1m, 5m, 15m, 1h, 4h, 1d).
            callback: Async callback receiving Candle objects.
        """
        stream_name = f"{self._map_symbol(symbol).lower()}@kline_{timeframe}"
        url = f"{self.ws_url}/{stream_name}"

        def parse_kline(data: dict):
            k = data.get("k", {})
            return Candle(
                symbol=symbol, timeframe=timeframe,
                open_time=datetime.fromtimestamp(k.get("t", 0) / 1000),
                close_time=datetime.fromtimestamp(k.get("T", 0) / 1000),
                open=float(k.get("o", 0)), high=float(k.get("h", 0)),
                low=float(k.get("l", 0)), close=float(k.get("c", 0)),
                volume=float(k.get("v", 0)),
            )

        await self._ws_connect_with_reconnect(url, callback, parse_kline)

    async def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get funding rate from Binance Futures."""
        try:
            params = {"symbol": self._map_symbol(symbol)}
            resp = await self.rest_client.get(
                f"{self.FUTURES_REST}/fapi/v1/premiumIndex", params=params
            )
            data = resp.json()
            if isinstance(data, list) and data:
                data = data[0]
            return {
                "funding_rate": float(data.get("lastFundingRate", 0)),
                "funding_time": datetime.fromtimestamp(data.get("nextFundingTime", 0) / 1000),
                "mark_price": float(data.get("markPrice", 0)),
            }
        except Exception as e:
            logger.warning("Failed to get Binance funding rate: %s", e)
            return None

    @staticmethod
    def _order_type_map(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET", OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP_LOSS", OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.TRAILING_STOP: "TRAILING_STOP_MARKET",
        }
        return mapping.get(order_type, "LIMIT")

    @staticmethod
    def _status_map(status: str) -> OrderStatus:
        mapping = {
            "NEW": OrderStatus.SUBMITTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.CREATED)


# Bybit Adapter

__all__ = ['BinanceAdapter']
