"""Exchange Adapters - Base classes and error handling.

Contains:
- ExchangeAdapter ABC
- ExchangeCredentials
- Error classes (ExchangeError, RateLimitError, etc.)
- Retry logic
- Re-exports from rate_limiter and order_book submodules
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import httpx

from acms.core import (
    Order, Trade, Position, Candle, Tick, Side, OrderType,
    OrderStatus, TimeInForce, ExchangeId,
)
from acms.exchanges.rate_limiter import RateLimiter
from acms.exchanges.order_book import LocalOrderBook

logger = logging.getLogger(__name__)


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


__all__ = [
    'RateLimiter', 'LocalOrderBook', 'ExchangeError', 'RateLimitError',
    'InsufficientFundsError', 'OrderNotFoundError', 'NetworkError',
    'classify_exchange_error', 'retry_with_backoff',
    'ExchangeCredentials', 'ExchangeAdapter',
]
