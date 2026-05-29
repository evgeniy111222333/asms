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


class BybitAdapter(ExchangeAdapter):
    """Bybit exchange adapter with full WebSocket support.

    Supports V5 API with public and private WebSocket channels:
    - Public: trades, orderbook, kline
    - Private: order, execution, position, wallet
    """

    REST_URL = "https://api.bybit.com"
    TESTNET_REST = "https://api-testnet.bybit.com"
    WS_PUBLIC_URL = "wss://stream.bybit.com/v5/public/spot"
    WS_PRIVATE_URL = "wss://stream.bybit.com/v5/private"
    TESTNET_WS_PUBLIC = "wss://stream-testnet.bybit.com/v5/public/spot"
    TESTNET_WS_PRIVATE = "wss://stream-testnet.bybit.com/v5/private"

    def __init__(self, credentials: ExchangeCredentials, testnet: bool = False):
        super().__init__(credentials, testnet)
        self.base_url = self.TESTNET_REST if testnet else self.REST_URL
        self.ws_public_url = self.TESTNET_WS_PUBLIC if testnet else self.WS_PUBLIC_URL
        self.ws_private_url = self.TESTNET_WS_PRIVATE if testnet else self.WS_PRIVATE_URL
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

    def _map_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "")

    def _sign(self, params: dict) -> dict:
        if not self.credentials.api_secret:
            return params
        timestamp = str(int(time.time() * 1000))
        param_str = timestamp + self.credentials.api_key + str(params.get("recvWindow", "5000"))
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sign_str = param_str + query
        signature = hmac.new(
            self.credentials.api_secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()
        return {**params, "api_key": self.credentials.api_key, "timestamp": timestamp, "sign": signature}

    def _auth_headers(self) -> dict:
        """Generate authenticated headers for private WebSocket."""
        timestamp = str(int(time.time() * 1000))
        sign_str = f"GET/realtime{timestamp}"
        signature = hmac.new(
            self.credentials.api_secret.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "op": "auth",
            "args": [self.credentials.api_key, timestamp, signature],
        }

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        await self.rate_limiter.acquire()
        params = {"category": "spot", "symbol": self._map_symbol(symbol),
                  "interval": timeframe, "limit": limit}
        resp = await self.rest_client.get(f"{self.base_url}/v5/market/kline", params=params)
        data = resp.json().get("result", {}).get("list", [])
        candles = []
        for k in data:
            candles.append(Candle(
                symbol=symbol, timeframe=timeframe,
                open_time=datetime.fromtimestamp(int(k[0]) / 1000),
                close_time=datetime.fromtimestamp(int(k[0]) / 1000 + 60),
                open=float(k[1]), high=float(k[2]), low=float(k[3]),
                close=float(k[4]), volume=float(k[5]),
            ))
        return candles

    async def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        await self.rate_limiter.acquire()
        params = {"category": "spot", "symbol": self._map_symbol(symbol), "limit": depth}
        resp = await self.rest_client.get(f"{self.base_url}/v5/market/orderbook", params=params)
        data = resp.json().get("result", {})
        return {
            "bids": [(float(b[0]), float(b[1])) for b in data.get("b", [])],
            "asks": [(float(a[0]), float(a[1])) for a in data.get("a", [])],
        }

    async def place_order(self, order: Order) -> Order:
        await self.rate_limiter.acquire()
        params = {
            "category": "spot",
            "symbol": self._map_symbol(order.symbol),
            "side": "Buy" if order.side == Side.BUY else "Sell",
            "orderType": "Market" if order.order_type == OrderType.MARKET else "Limit",
            "qty": str(order.quantity),
        }
        if order.price:
            params["price"] = str(order.price)
        params = self._sign(params)
        resp = await self.rest_client.post(f"{self.base_url}/v5/order/place", json=params)
        data = resp.json()
        order.exchange_order_id = data.get("result", {}).get("orderId", "")
        order.status = OrderStatus.SUBMITTED
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self.rate_limiter.acquire()
        params = {"category": "spot", "symbol": self._map_symbol(symbol), "orderId": order_id}
        params = self._sign(params)
        resp = await self.rest_client.post(f"{self.base_url}/v5/order/cancel", json=params)
        return resp.json().get("retCode") == 0

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        await self.rate_limiter.acquire()
        params = {"category": "spot", "symbol": self._map_symbol(symbol), "orderId": order_id}
        params = self._sign(params)
        resp = await self.rest_client.get(f"{self.base_url}/v5/order/realtime", params=params)
        data = resp.json().get("result", {})
        order_list = data.get("list", [])
        if not order_list:
            return Order(id=order_id, symbol=symbol, side=Side.BUY,
                         order_type=OrderType.LIMIT, status=OrderStatus.CREATED, quantity=0)
        o = order_list[0]
        return Order(
            id=order_id, symbol=symbol,
            side=Side.BUY if o.get("side") == "Buy" else Side.SELL,
            order_type=OrderType.LIMIT,
            status=self._status_map(o.get("orderStatus", "")),
            quantity=float(o.get("qty", 0)),
            price=float(o.get("price", 0)),
            filled_quantity=float(o.get("cumExecQty", 0)),
        )

    async def get_positions(self) -> list[Position]:
        await self.rate_limiter.acquire()
        params = {"category": "linear", "settleCoin": "USDT"}
        params = self._sign(params)
        try:
            resp = await self.rest_client.get(
                f"{self.base_url}/v5/position/list", params=params
            )
            data = resp.json().get("result", {}).get("list", [])
            positions = []
            for p in data:
                positions.append(Position(
                    symbol=p.get("symbol", ""),
                    side=Side.BUY if p.get("side") == "Buy" else Side.SELL,
                    quantity=float(p.get("size", 0)),
                    entry_price=float(p.get("avgPrice", 0)),
                    mark_price=float(p.get("markPrice", 0)),
                    unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                    leverage=float(p.get("leverage", 1)),
                    exchange="bybit",
                ))
            return positions
        except Exception as e:
            logger.error(
                "Failed to get positions from Bybit: %s. "
                "Positions may be stale or unavailable. Check exchange connectivity.",
                e
            )
            return []

    async def get_balance(self) -> dict:
        await self.rate_limiter.acquire()
        params = {"accountType": "UNIFIED"}
        params = self._sign(params)
        resp = await self.rest_client.get(
            f"{self.base_url}/v5/account/wallet-balance", params=params
        )
        data = resp.json()
        balances = {}
        for account in data.get("result", {}).get("list", []):
            for coin in account.get("coin", []):
                free = float(coin.get("availableToWithdraw", 0))
                if free > 0:
                    balances[coin["coin"]] = {"free": free, "locked": float(coin.get("locked", 0))}
        return balances

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        """Subscribe to Bybit public trade WebSocket."""
        import websockets

        url = self.ws_public_url

        async def _run():
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                self.ws_connected = True
                self._ws_reconnect_attempts = 0
                # Subscribe to trades
                sub_msg = {"op": "subscribe", "args": [f"publicTrade.{self._map_symbol(symbol)}"]}
                await ws.send(json.dumps(sub_msg))
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if data.get("topic", "").startswith("publicTrade."):
                            for trade_data in data.get("data", []):
                                tick = Tick(
                                    symbol=symbol, exchange="bybit",
                                    price=float(trade_data["p"]),
                                    quantity=float(trade_data["v"]),
                                    side=Side.BUY if trade_data["S"] == "Buy" else Side.SELL,
                                    timestamp=datetime.fromtimestamp(trade_data["T"] / 1000),
                                    trade_id=str(trade_data["i"]),
                                )
                                await callback(tick)
                    except Exception as e:
                        logger.error("Error processing Bybit trade WS: %s", e)

        await self._ws_connect_with_reconnect(url, callback, lambda d: d)
        # Use direct WS connection for proper subscription
        import websockets
        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [f"publicTrade.{self._map_symbol(symbol)}"]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("topic", "").startswith("publicTrade."):
                            for trade_data in data.get("data", []):
                                tick = Tick(
                                    symbol=symbol, exchange="bybit",
                                    price=float(trade_data["p"]),
                                    quantity=float(trade_data["v"]),
                                    side=Side.BUY if trade_data["S"] == "Buy" else Side.SELL,
                                    timestamp=datetime.fromtimestamp(trade_data["T"] / 1000),
                                    trade_id=str(trade_data["i"]),
                                )
                                await callback(tick)
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.error("Bybit WS error: %s, reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)

    async def subscribe_order_book(self, symbol: str, callback: Callable):
        """Subscribe to Bybit orderbook WebSocket with local book management."""
        import websockets

        url = self.ws_public_url
        mapped_symbol = self._map_symbol(symbol)
        if symbol not in self.order_books:
            self.order_books[symbol] = LocalOrderBook(symbol)

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [f"orderbook.50.{mapped_symbol}"]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        topic = data.get("topic", "")
                        if topic.startswith("orderbook."):
                            ob_data = data.get("data", {})
                            bids = [(float(b[0]), float(b[1])) for b in ob_data.get("b", [])]
                            asks = [(float(a[0]), float(a[1])) for a in ob_data.get("a", [])]
                            self.order_books[symbol].update(bids, asks)
                            await callback(self.order_books[symbol].snapshot())
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.error("Bybit OB WS error: %s, reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)

    async def subscribe_kline(self, symbol: str, timeframe: str, callback: Callable):
        """Subscribe to Bybit kline WebSocket stream."""
        import websockets

        url = self.ws_public_url
        mapped_symbol = self._map_symbol(symbol)

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [f"kline.{timeframe}.{mapped_symbol}"]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("topic", "").startswith("kline."):
                            k_data = data.get("data", [{}])
                            for k in k_data:
                                candle = Candle(
                                    symbol=symbol, timeframe=timeframe,
                                    open_time=datetime.fromtimestamp(k.get("start", 0) / 1000),
                                    close_time=datetime.fromtimestamp(k.get("end", 0) / 1000),
                                    open=float(k.get("open", 0)),
                                    high=float(k.get("high", 0)),
                                    low=float(k.get("low", 0)),
                                    close=float(k.get("close", 0)),
                                    volume=float(k.get("volume", 0)),
                                )
                                await callback(candle)
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                await asyncio.sleep(delay)

    async def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get funding rate from Bybit."""
        try:
            params = {"category": "linear", "symbol": self._map_symbol(symbol)}
            resp = await self.rest_client.get(
                f"{self.base_url}/v5/market/tickers", params=params
            )
            data = resp.json().get("result", {}).get("list", [])
            if data:
                return {
                    "funding_rate": float(data[0].get("fundingRate", 0)),
                    "funding_time": datetime.fromtimestamp(
                        data[0].get("nextFundingTime", 0) / 1000
                    ),
                    "mark_price": float(data[0].get("markPrice", 0)),
                }
        except Exception as e:
            logger.warning("Failed to get Bybit funding rate: %s", e)
        return None

    @staticmethod
    def _status_map(status: str) -> OrderStatus:
        mapping = {
            "Created": OrderStatus.CREATED,
            "New": OrderStatus.SUBMITTED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.CREATED)


# OKX Adapter

__all__ = ['BybitAdapter']
