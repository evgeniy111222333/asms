"""Market data cache with auto-expiry."""

import json
import logging
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class MarketDataCache:
    """Cache for latest market data with auto-expiry.

    Stores latest prices, order book snapshots, and candle data
    with short TTLs appropriate for real-time market data.
    """

    def __init__(self, redis_client=None, prefix: str = "acms:market",
                 price_ttl: int = 30, orderbook_ttl: int = 10):
        self._redis = redis_client
        self.prefix = prefix
        self.price_ttl = price_ttl
        self.orderbook_ttl = orderbook_ttl

    async def set_latest_price(self, symbol: str, price: float,
                                exchange: str = "") -> bool:
        """Cache latest price for a symbol.

        Args:
            symbol: Trading pair symbol.
            price: Latest price.
            exchange: Exchange name.

        Returns:
            True if cached successfully.
        """
        key = f"{self.prefix}:price:{symbol}"
        data = {
            "price": price,
            "exchange": exchange,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            await self._redis.setex(key, self.price_ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.warning("Price cache set error: %s", e)
            return False

    async def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """Get latest cached price for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dict with 'price', 'exchange', 'timestamp' or None.
        """
        key = f"{self.prefix}:price:{symbol}"
        try:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Price cache get error: %s", e)
        return None

    async def set_orderbook(self, symbol: str, orderbook: Dict) -> bool:
        """Cache order book snapshot.

        Args:
            symbol: Trading pair symbol.
            orderbook: Dict with 'bids' and 'asks' lists.

        Returns:
            True if cached successfully.
        """
        key = f"{self.prefix}:orderbook:{symbol}"
        data = {
            "orderbook": orderbook,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            await self._redis.setex(key, self.orderbook_ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.warning("Orderbook cache set error: %s", e)
            return False

    async def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get cached order book for a symbol."""
        key = f"{self.prefix}:orderbook:{symbol}"
        try:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Orderbook cache get error: %s", e)
        return None

    async def set_all_prices(self, prices: Dict[str, float]) -> bool:
        """Cache prices for multiple symbols at once.

        Args:
            prices: Dict mapping symbol to price.

        Returns:
            True if all prices cached.
        """
        try:
            pipe = self._redis.pipeline()
            for symbol, price in prices.items():
                key = f"{self.prefix}:price:{symbol}"
                data = json.dumps({"price": price, "timestamp": datetime.utcnow().isoformat()})
                pipe.setex(key, self.price_ttl, data)
            await pipe.execute()
            return True
        except Exception as e:
            logger.warning("Bulk price cache error: %s", e)
            return False



__all__ = ["MarketDataCache"]
