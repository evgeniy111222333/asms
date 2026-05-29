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


class OKXAdapter(ExchangeAdapter):
    """OKX exchange adapter with full WebSocket support.

    Supports V5 API with public and private WebSocket channels:
    - Public: trades, books, candle
    - Private: orders, account
    """

    REST_URL = "https://www.okx.com"
    WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
    WS_PRIVATE_URL = "wss://ws.okx.com:8443/ws/v5/private"
    TESTNET_REST = "https://www.okx.com"
    TESTNET_WS_PUBLIC = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
    TESTNET_WS_PRIVATE = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

    def __init__(self, credentials: ExchangeCredentials, testnet: bool = False):
        super().__init__(credentials, testnet)
        self.base_url = self.REST_URL
        self.ws_public_url = self.TESTNET_WS_PUBLIC if testnet else self.WS_PUBLIC_URL
        self.ws_private_url = self.TESTNET_WS_PRIVATE if testnet else self.WS_PRIVATE_URL
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)

    def _map_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "-")

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> dict:
        message = timestamp + method + path + body
        signature = hmac.new(
            self.credentials.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "OK-ACCESS-KEY": self.credentials.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.credentials.passphrase,
        }

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        await self.rate_limiter.acquire()
        inst_id = self._map_symbol(symbol)
        params = {"instId": inst_id, "bar": timeframe, "limit": limit}
        resp = await self.rest_client.get(f"{self.base_url}/api/v5/market/candles", params=params)
        data = resp.json().get("data", [])
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
        params = {"instId": self._map_symbol(symbol), "sz": str(depth)}
        resp = await self.rest_client.get(f"{self.base_url}/api/v5/market/books", params=params)
        data = resp.json().get("data", [{}])[0]
        return {
            "bids": [(float(b[0]), float(b[1])) for b in data.get("bids", [])],
            "asks": [(float(a[0]), float(a[1])) for a in data.get("asks", [])],
        }

    async def place_order(self, order: Order) -> Order:
        await self.rate_limiter.acquire()
        timestamp = datetime.utcnow().isoformat()
        body = json.dumps({
            "instId": self._map_symbol(order.symbol),
            "tdMode": "cash",
            "side": "buy" if order.side == Side.BUY else "sell",
            "ordType": "market" if order.order_type == OrderType.MARKET else "limit",
            "sz": str(order.quantity),
        })
        headers = self._sign(timestamp, "POST", "/api/v5/trade/order", body)
        resp = await self.rest_client.post(
            f"{self.base_url}/api/v5/trade/order", data=body, headers=headers
        )
        data = resp.json()
        order.exchange_order_id = data.get("data", [{}])[0].get("ordId", "")
        order.status = OrderStatus.SUBMITTED
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self.rate_limiter.acquire()
        timestamp = datetime.utcnow().isoformat()
        body = json.dumps({"instId": self._map_symbol(symbol), "ordId": order_id})
        headers = self._sign(timestamp, "POST", "/api/v5/trade/cancel-order", body)
        resp = await self.rest_client.post(
            f"{self.base_url}/api/v5/trade/cancel-order", data=body, headers=headers
        )
        return resp.json().get("code") == "0"

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        await self.rate_limiter.acquire()
        timestamp = datetime.utcnow().isoformat()
        path = f"/api/v5/trade/order?instId={self._map_symbol(symbol)}&ordId={order_id}"
        headers = self._sign(timestamp, "GET", path)
        resp = await self.rest_client.get(
            f"{self.base_url}{path}", headers=headers
        )
        data = resp.json().get("data", [])
        if not data:
            return Order(id=order_id, symbol=symbol, side=Side.BUY,
                         order_type=OrderType.LIMIT, status=OrderStatus.CREATED, quantity=0)
        o = data[0]
        return Order(
            id=order_id, symbol=symbol,
            side=Side.BUY if o.get("side") == "buy" else Side.SELL,
            order_type=OrderType.LIMIT,
            status=self._status_map(o.get("state", "")),
            quantity=float(o.get("sz", 0)),
            price=float(o.get("px", 0)),
            filled_quantity=float(o.get("accFillSz", 0)),
        )

    async def get_positions(self) -> list[Position]:
        await self.rate_limiter.acquire()
        timestamp = datetime.utcnow().isoformat()
        path = "/api/v5/account/positions"
        headers = self._sign(timestamp, "GET", path)
        try:
            resp = await self.rest_client.get(f"{self.base_url}{path}", headers=headers)
            data = resp.json().get("data", [])
            positions = []
            for p in data:
                positions.append(Position(
                    symbol=p.get("instId", ""),
                    side=Side.BUY if p.get("posSide") == "long" else Side.SELL,
                    quantity=float(p.get("pos", 0)),
                    entry_price=float(p.get("avgPx", 0)),
                    mark_price=float(p.get("markPx", 0)),
                    unrealized_pnl=float(p.get("upl", 0)),
                    leverage=float(p.get("lever", 1)),
                    exchange="okx",
                ))
            return positions
        except Exception as e:
            logger.error(
                "Failed to get positions from OKX: %s. "
                "Positions may be stale or unavailable. Check exchange connectivity.",
                e
            )
            return []

    async def get_balance(self) -> dict:
        await self.rate_limiter.acquire()
        timestamp = datetime.utcnow().isoformat()
        headers = self._sign(timestamp, "GET", "/api/v5/account/balance")
        resp = await self.rest_client.get(
            f"{self.base_url}/api/v5/account/balance", headers=headers
        )
        data = resp.json().get("data", [{}])[0]
        balances = {}
        for detail in data.get("details", []):
            free = float(detail.get("availBal", 0))
            if free > 0:
                balances[detail["ccy"]] = {"free": free, "locked": float(detail.get("frozenBal", 0))}
        return balances

    async def subscribe_ticks(self, symbol: str, callback: Callable):
        """Subscribe to OKX public trade WebSocket."""
        import websockets

        url = self.ws_public_url
        inst_id = self._map_symbol(symbol)

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [{"channel": "trades", "instId": inst_id}]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("arg", {}).get("channel") == "trades":
                            for trade_data in data.get("data", []):
                                tick = Tick(
                                    symbol=symbol, exchange="okx",
                                    price=float(trade_data["px"]),
                                    quantity=float(trade_data["sz"]),
                                    side=Side.BUY if trade_data["side"] == "buy" else Side.SELL,
                                    timestamp=datetime.fromtimestamp(int(trade_data["ts"]) / 1000),
                                    trade_id=str(trade_data.get("tradeId", "")),
                                )
                                await callback(tick)
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.error("OKX WS error: %s, reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)

    async def subscribe_order_book(self, symbol: str, callback: Callable):
        """Subscribe to OKX books WebSocket with local book management."""
        import websockets

        url = self.ws_public_url
        inst_id = self._map_symbol(symbol)
        if symbol not in self.order_books:
            self.order_books[symbol] = LocalOrderBook(symbol)

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [{"channel": "books5", "instId": inst_id}]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("arg", {}).get("channel", "").startswith("books"):
                            ob_data = data.get("data", [{}])[0]
                            bids = [(float(b[0]), float(b[1])) for b in ob_data.get("bids", [])]
                            asks = [(float(a[0]), float(a[1])) for a in ob_data.get("asks", [])]
                            self.order_books[symbol].update(bids, asks)
                            await callback(self.order_books[symbol].snapshot())
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.error("OKX OB WS error: %s, reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)

    async def subscribe_kline(self, symbol: str, timeframe: str, callback: Callable):
        """Subscribe to OKX candle WebSocket."""
        import websockets

        url = self.ws_public_url
        inst_id = self._map_symbol(symbol)
        # OKX bar format: 1m, 5m, 1H, 1D
        bar = timeframe.upper() if timeframe.endswith("h") else timeframe

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    sub_msg = {"op": "subscribe", "args": [{"channel": f"candle{bar}", "instId": inst_id}]}
                    await ws.send(json.dumps(sub_msg))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("arg", {}).get("channel", "").startswith("candle"):
                            for c_data in data.get("data", []):
                                candle = Candle(
                                    symbol=symbol, timeframe=timeframe,
                                    open_time=datetime.fromtimestamp(int(c_data[0]) / 1000),
                                    close_time=datetime.fromtimestamp(int(c_data[0]) / 1000 + 60),
                                    open=float(c_data[1]), high=float(c_data[2]),
                                    low=float(c_data[3]), close=float(c_data[4]),
                                    volume=float(c_data[5]),
                                )
                                await callback(candle)
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                await asyncio.sleep(delay)

    async def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get funding rate from OKX."""
        try:
            params = {"instId": self._map_symbol(symbol)}
            resp = await self.rest_client.get(
                f"{self.base_url}/api/v5/public/funding-rate", params=params
            )
            data = resp.json().get("data", [])
            if data:
                return {
                    "funding_rate": float(data[0].get("fundingRate", 0)),
                    "funding_time": datetime.fromtimestamp(
                        int(data[0].get("nextFundingTime", 0)) / 1000
                    ),
                    "mark_price": float(data[0].get("markPx", 0)),
                }
        except Exception as e:
            logger.warning("Failed to get OKX funding rate: %s", e)
        return None

    @staticmethod
    def _status_map(status: str) -> OrderStatus:
        mapping = {
            "live": OrderStatus.SUBMITTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
        }
        return mapping.get(status, OrderStatus.CREATED)


# Paper Trading Adapter

__all__ = ['OKXAdapter']
