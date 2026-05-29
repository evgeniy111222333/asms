"""Exchange Adapters - Real WebSocket/REST connections.

Implements adapters for:
- Binance (REST + WebSocket)
- Bybit (REST + WebSocket with auto-reconnect)
- OKX (REST + WebSocket with auto-reconnect)
- Paper Trading (simulation)

Additional features:
- Rate limiting with token bucket algorithm
- Order book depth streaming with local book management
- Funding rate fetching and streaming
- Multi-exchange arbitrage detection
- Unified error handling and retry logic
"""

import asyncio
import hashlib
import hmac
import time
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple, Any
from datetime import datetime
from enum import Enum

import httpx

from acms.core import (
    Order, Trade, Position, Candle, Tick, Side, OrderType,
    OrderStatus, TimeInForce, ExchangeId,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter for exchange API calls.

    Implements a sliding window rate limiter that respects
    per-exchange rate limits.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0,
                 burst_size: Optional[int] = None):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size or max_requests
        self._tokens = float(self.burst_size)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        async with self._lock:
            self._refill_tokens()
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) * (self.window_seconds / self.max_requests)
                await asyncio.sleep(wait_time)
                self._refill_tokens()
            self._tokens -= 1.0

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill = elapsed * (self.max_requests / self.window_seconds)
        self._tokens = min(self.burst_size, self._tokens + refill)
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        self._refill_tokens()
        return self._tokens


# ============================================================================
# Local Order Book Management
# ============================================================================

class LocalOrderBook:
    """Local order book manager for depth streaming.

    Maintains a local copy of the order book updated via
    WebSocket depth stream messages.
    """

    def __init__(self, symbol: str, max_depth: int = 50):
        self.symbol = symbol
        self.max_depth = max_depth
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}
        self.last_update_id: int = 0
        self.updated_at: Optional[datetime] = None

    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
               update_id: int = 0) -> None:
        """Update the local order book with new data.

        Args:
            bids: List of (price, quantity) tuples. Quantity 0 removes level.
            asks: List of (price, quantity) tuples.
            update_id: Sequential update identifier.
        """
        if update_id <= self.last_update_id and update_id != 0:
            return

        for price, qty in bids:
            if qty <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price, qty in asks:
            if qty <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = update_id
        self.updated_at = datetime.utcnow()
        self._trim_depth()

    def _trim_depth(self) -> None:
        """Trim order book to max_depth levels."""
        if len(self.bids) > self.max_depth:
            sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])
            self.bids = dict(sorted_bids[:self.max_depth])
        if len(self.asks) > self.max_depth:
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            self.asks = dict(sorted_asks[:self.max_depth])

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return min(self.asks.keys()) if self.asks else None

    def get_spread(self) -> Optional[float]:
        """Get current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def snapshot(self) -> Dict:
        """Get a snapshot of the order book.

        Returns:
            Dict with sorted bids and asks lists.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        return {
            "symbol": self.symbol,
            "bids": [(p, q) for p, q in sorted_bids],
            "asks": [(p, q) for p, q in sorted_asks],
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============================================================================
# Error Handling
# ============================================================================

class ExchangeError(Exception):
    """Base exception for exchange errors."""

    def __init__(self, message: str, exchange: str = "", code: str = ""):
        super().__init__(message)
        self.exchange = exchange
        self.code = code


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""
    pass


class InsufficientFundsError(ExchangeError):
    """Insufficient funds for order."""
    pass


class OrderNotFoundError(ExchangeError):
    """Order not found on exchange."""
    pass


class NetworkError(ExchangeError):
    """Network connectivity error."""
    pass


def classify_exchange_error(status_code: int, response_data: dict, exchange: str) -> ExchangeError:
    """Classify an exchange error response into a specific error type.

    Args:
        status_code: HTTP status code.
        response_data: Parsed JSON response body.
        exchange: Exchange name.

    Returns:
        Appropriate ExchangeError subclass instance.
    """
    msg = str(response_data)
    code = ""

    if exchange == "binance":
        code = str(response_data.get("code", ""))
        msg = response_data.get("msg", msg)
    elif exchange == "bybit":
        code = str(response_data.get("retCode", ""))
        msg = response_data.get("retMsg", msg)
    elif exchange == "okx":
        code = str(response_data.get("code", ""))
        msg = response_data.get("msg", msg)

    if status_code == 429:
        return RateLimitError(msg, exchange, code)
    if "insufficient" in msg.lower() or code in ("-2019", "130040", "51421"):
        return InsufficientFundsError(msg, exchange, code)
    if "not found" in msg.lower() or "order" in msg.lower():
        return OrderNotFoundError(msg, exchange, code)
    return ExchangeError(msg, exchange, code)


async def retry_with_backoff(func: Callable, max_retries: int = 3,
                              base_delay: float = 1.0, max_delay: float = 30.0,
                              exchange: str = "") -> Any:
    """Retry an async function with exponential backoff.

    Args:
        func: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay between retries.
        exchange: Exchange name for error classification.

    Returns:
        Result of successful function call.

    Raises:
        ExchangeError: After all retries exhausted.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except RateLimitError as e:
            last_error = e
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning("Rate limited on %s, retrying in %.1fs (attempt %d/%d)",
                           exchange, delay, attempt + 1, max_retries)
            await asyncio.sleep(delay)
        except NetworkError as e:
            last_error = e
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning("Network error on %s, retrying in %.1fs (attempt %d/%d)",
                           exchange, delay, attempt + 1, max_retries)
            await asyncio.sleep(delay)
        except ExchangeError as e:
            raise
        except Exception as e:
            last_error = ExchangeError(str(e), exchange)
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)
            else:
                raise last_error
    raise last_error or ExchangeError("Unknown error", exchange)


# ============================================================================
# Exchange Credentials and Base Adapter
# ============================================================================

@dataclass
class ExchangeCredentials:
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""


class ExchangeAdapter(ABC):
    """Abstract base class for exchange adapters."""

    def __init__(self, credentials: ExchangeCredentials, testnet: bool = False):
        self.credentials = credentials
        self.testnet = testnet
        self.rest_client = httpx.AsyncClient(timeout=30.0)
        self.ws_connected = False
        self._ws_callbacks: list[Callable] = []
        self.rate_limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        self.order_books: Dict[str, LocalOrderBook] = {}
        self._ws_reconnect_attempts = 0
        self._max_reconnect_attempts = 10

    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 500) -> list[Candle]:
        ...

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        ...

    @abstractmethod
    async def get_balance(self) -> dict:
        ...

    @abstractmethod
    async def subscribe_ticks(self, symbol: str, callback: Callable):
        ...

    @abstractmethod
    async def subscribe_order_book(self, symbol: str, callback: Callable):
        ...

    async def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get current funding rate for a symbol.

        Returns:
            Dict with 'funding_rate', 'funding_time', 'mark_price' or None.
        """
        return None

    async def close(self) -> None:
        """Close the exchange adapter and release resources."""
        await self.rest_client.aclose()
        self.ws_connected = False

    async def _ws_connect_with_reconnect(self, url: str, callback: Callable,
                                          message_parser: Callable) -> None:
        """Connect to WebSocket with auto-reconnect and exponential backoff.

        Args:
            url: WebSocket URL.
            callback: Callback for parsed messages.
            message_parser: Function to parse raw messages into domain objects.
        """
        import websockets

        while self._ws_reconnect_attempts < self._max_reconnect_attempts:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self.ws_connected = True
                    self._ws_reconnect_attempts = 0
                    logger.info("WebSocket connected to %s", url)
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            parsed = message_parser(data)
                            if parsed is not None:
                                await callback(parsed)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON from WebSocket")
                        except Exception as e:
                            logger.error("Error processing WS message: %s", e)
            except websockets.ConnectionClosed as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.warning("WS disconnected from %s, reconnecting in %ds (attempt %d)",
                               url, delay, self._ws_reconnect_attempts)
                await asyncio.sleep(delay)
            except Exception as e:
                self.ws_connected = False
                self._ws_reconnect_attempts += 1
                delay = min(2 ** self._ws_reconnect_attempts, 60)
                logger.error("WS error: %s, reconnecting in %ds", e, delay)
                await asyncio.sleep(delay)

        logger.error("Max reconnection attempts reached for %s", url)


# ============================================================================
# Binance Adapter
# ============================================================================

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


# ============================================================================
# Bybit Adapter
# ============================================================================

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
        except Exception:
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


# ============================================================================
# OKX Adapter
# ============================================================================

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
        except Exception:
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


# ============================================================================
# Paper Trading Adapter
# ============================================================================

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


# ============================================================================
# Multi-Exchange Arbitrage Detection
# ============================================================================

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    estimated_profit_usd: float
    buy_fee_usd: float
    sell_fee_usd: float
    timestamp: datetime


class ArbitrageDetector:
    """Multi-exchange arbitrage detection engine.

    Monitors price differences across exchanges and identifies
    fee-adjusted profitable opportunities.
    """

    def __init__(self, exchanges: Dict[str, ExchangeAdapter],
                 default_fees_bps: Dict[str, float] = None):
        self.exchanges = exchanges
        self.fees_bps = default_fees_bps or {
            "binance": 10.0, "bybit": 10.0, "okx": 10.0, "paper": 0.0,
        }
        self._latest_prices: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_price(self, exchange: str, symbol: str, price: float) -> None:
        """Update the latest price for an exchange/symbol pair.

        Args:
            exchange: Exchange name.
            symbol: Trading pair symbol.
            price: Latest price.
        """
        self._latest_prices[symbol][exchange] = price

    async def fetch_all_prices(self, symbol: str) -> Dict[str, float]:
        """Fetch current prices from all exchanges for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dict mapping exchange name to price.
        """
        prices = {}
        for name, adapter in self.exchanges.items():
            try:
                ob = await adapter.get_order_book(symbol, depth=1)
                if ob.get("asks") and ob.get("bids"):
                    best_ask = ob["asks"][0][0]
                    best_bid = ob["bids"][0][0]
                    mid = (best_ask + best_bid) / 2.0
                    prices[name] = mid
                    self._latest_prices[symbol][name] = mid
            except Exception as e:
                logger.warning("Failed to fetch price from %s: %s", name, e)
        return prices

    def detect_opportunities(self, symbol: str, trade_size: float = 1.0) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities for a symbol.

        Compares prices across exchanges and identifies fee-adjusted
        profitable opportunities.

        Args:
            symbol: Trading pair symbol.
            trade_size: Size in base currency for profit estimation.

        Returns:
            List of ArbitrageOpportunity sorted by estimated profit.
        """
        exchange_prices = self._latest_prices.get(symbol, {})
        if len(exchange_prices) < 2:
            return []

        opportunities = []
        exchange_names = list(exchange_prices.keys())

        for i in range(len(exchange_names)):
            for j in range(len(exchange_names)):
                if i == j:
                    continue
                buy_exchange = exchange_names[i]
                sell_exchange = exchange_names[j]
                buy_price = exchange_prices[buy_exchange]
                sell_price = exchange_prices[sell_exchange]

                if buy_price <= 0 or sell_price <= 0:
                    continue

                spread_pct = (sell_price - buy_price) / buy_price
                buy_fee = buy_price * trade_size * self.fees_bps.get(buy_exchange, 10) / 10000
                sell_fee = sell_price * trade_size * self.fees_bps.get(sell_exchange, 10) / 10000
                gross_profit = (sell_price - buy_price) * trade_size
                net_profit = gross_profit - buy_fee - sell_fee

                if net_profit > 0:
                    opportunities.append(ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_exchange,
                        sell_exchange=sell_exchange,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        spread_pct=spread_pct * 100,
                        estimated_profit_usd=net_profit,
                        buy_fee_usd=buy_fee,
                        sell_fee_usd=sell_fee,
                        timestamp=datetime.utcnow(),
                    ))

        opportunities.sort(key=lambda x: x.estimated_profit_usd, reverse=True)
        return opportunities


# ============================================================================
# Exchange Factory
# ============================================================================

def create_exchange_adapter(exchange: str, credentials: Optional[ExchangeCredentials] = None,
                            testnet: bool = False) -> ExchangeAdapter:
    """Factory function to create exchange adapters.

    Args:
        exchange: Exchange name ('binance', 'bybit', 'okx', 'paper').
        credentials: Exchange API credentials.
        testnet: Whether to use testnet endpoints.

    Returns:
        ExchangeAdapter instance.

    Raises:
        ValueError: If exchange name is unknown.
    """
    creds = credentials or ExchangeCredentials()
    if exchange == "binance":
        return BinanceAdapter(creds, testnet)
    elif exchange == "bybit":
        return BybitAdapter(creds, testnet)
    elif exchange == "okx":
        return OKXAdapter(creds, testnet)
    elif exchange == "paper":
        return PaperTradingAdapter()
    else:
        raise ValueError(f"Unknown exchange: {exchange}")
