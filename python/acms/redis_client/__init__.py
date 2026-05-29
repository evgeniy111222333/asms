"""Redis Client Module - Caching, PubSub, Rate Limiting, Sessions.

Implements:
- CacheManager: get/set with TTL, pattern-based invalidation
- PubSubManager: publish/subscribe for real-time events
- RateLimiter: sliding window rate limiting using Redis
- SessionManager: user session storage with TTL
- MarketDataCache: latest prices, orderbooks with auto-expiry
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Awaitable, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RedisConfig:
    """Redis connection configuration."""
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True


def get_redis(config: Optional[RedisConfig] = None):
    """Get a Redis client instance.

    Args:
        config: Redis configuration.

    Returns:
        Redis client or None if redis is not available.
    """
    config = config or RedisConfig()
    try:
        import redis.asyncio as aioredis
        return aioredis.from_url(
            config.url,
            password=config.password,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            retry_on_timeout=config.retry_on_timeout,
            decode_responses=True,
        )
    except ImportError:
        logger.warning("redis.asyncio not installed, using in-memory fallback")
        return _InMemoryRedis()


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """Redis-based cache with TTL and pattern-based invalidation.

    Provides a high-level caching interface with support for
    namespace-based key organization and pattern-based invalidation.
    """

    def __init__(self, redis_client=None, default_ttl: int = 300,
                 prefix: str = "acms:cache"):
        self._redis = redis_client
        self.default_ttl = default_ttl
        self.prefix = prefix

    def _key(self, key: str) -> str:
        """Build namespaced cache key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        full_key = self._key(key)
        try:
            value = await self._redis.get(full_key)
            if value is not None:
                return json.loads(value)
        except Exception as e:
            logger.warning("Cache get error for key '%s': %s", key, e)
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a cached value with TTL.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds. Uses default if not specified.

        Returns:
            True if successfully cached.
        """
        full_key = self._key(key)
        ttl = ttl or self.default_ttl
        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(full_key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning("Cache set error for key '%s': %s", key, e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key to delete.

        Returns:
            True if key was deleted.
        """
        full_key = self._key(key)
        try:
            result = await self._redis.delete(full_key)
            return result > 0
        except Exception as e:
            logger.warning("Cache delete error for key '%s': %s", key, e)
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., 'signals:*', 'prices:BTC*').

        Returns:
            Number of keys invalidated.
        """
        full_pattern = self._key(pattern)
        try:
            keys = []
            async for key in self._redis.scan_iter(match=full_pattern):
                keys.append(key)
            if keys:
                deleted = await self._redis.delete(*keys)
                return deleted
        except Exception as e:
            logger.warning("Cache invalidate error for pattern '%s': %s", pattern, e)
        return 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        full_key = self._key(key)
        try:
            return bool(await self._redis.exists(full_key))
        except Exception:
            return False

    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and cache the value.

        Args:
            key: Cache key.
            factory: Async callable to compute the value if not cached.
            ttl: Time-to-live in seconds.

        Returns:
            Cached or freshly computed value.
        """
        value = await self.get(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value


# ============================================================================
# PubSub Manager
# ============================================================================

class PubSubManager:
    """Redis PubSub for real-time event broadcasting.

    Provides publish/subscribe functionality for real-time
    event distribution across ACMS components.
    """

    def __init__(self, redis_client=None, prefix: str = "acms:pubsub"):
        self._redis = redis_client
        self.prefix = prefix
        self._pubsub = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._running = False

    def _channel(self, channel: str) -> str:
        return f"{self.prefix}:{channel}"

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish a message to a channel.

        Args:
            channel: Channel name.
            message: Message to publish (JSON-serializable).

        Returns:
            True if message was published.
        """
        full_channel = self._channel(channel)
        try:
            serialized = json.dumps(message, default=str)
            await self._redis.publish(full_channel, serialized)
            return True
        except Exception as e:
            logger.warning("PubSub publish error on '%s': %s", channel, e)
            return False

    async def subscribe(self, channel: str, handler: Callable[[Dict], Awaitable[None]]) -> None:
        """Subscribe to a channel with a message handler.

        Args:
            channel: Channel name.
            handler: Async callable receiving message dicts.
        """
        full_channel = self._channel(channel)
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(handler)

        try:
            if self._pubsub is None:
                self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(full_channel)
            logger.info("Subscribed to channel '%s'", channel)
        except Exception as e:
            logger.warning("PubSub subscribe error on '%s': %s", channel, e)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        full_channel = self._channel(channel)
        try:
            if self._pubsub:
                await self._pubsub.unsubscribe(full_channel)
            self._subscriptions.pop(channel, None)
        except Exception as e:
            logger.warning("PubSub unsubscribe error: %s", e)

    async def listen(self) -> None:
        """Start listening for messages on subscribed channels."""
        if not self._pubsub:
            return
        self._running = True
        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                if message["type"] == "message":
                    channel = message["channel"]
                    # Strip prefix
                    if channel.startswith(self._prefix_str()):
                        channel = channel[len(self._prefix_str()):]
                    try:
                        data = json.loads(message["data"])
                        handlers = self._subscriptions.get(channel, [])
                        for handler in handlers:
                            try:
                                await handler(data)
                            except Exception as e:
                                logger.error("Handler error on '%s': %s", channel, e)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in pubsub message")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("PubSub listen error: %s", e)
        finally:
            self._running = False

    def _prefix_str(self) -> str:
        return f"{self.prefix}:"

    async def stop(self) -> None:
        """Stop listening and close pubsub connection."""
        self._running = False
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
                await self._pubsub.close()
            except Exception:
                pass


# ============================================================================
# Rate Limiter (Sliding Window)
# ============================================================================

class RedisRateLimiter:
    """Sliding window rate limiter using Redis.

    Uses a sorted set to track request timestamps within
    the sliding window for accurate rate limiting.
    """

    def __init__(self, redis_client=None, prefix: str = "acms:ratelimit"):
        self._redis = redis_client
        self.prefix = prefix

    def _key(self, identifier: str) -> str:
        return f"{self.prefix}:{identifier}"

    async def is_allowed(self, identifier: str, max_requests: int = 100,
                          window_seconds: int = 60) -> Dict:
        """Check if a request is within rate limits.

        Args:
            identifier: Unique identifier (e.g., user_id, IP address).
            max_requests: Maximum requests allowed in window.
            window_seconds: Sliding window size in seconds.

        Returns:
            Dict with 'allowed', 'remaining', 'reset_at' fields.
        """
        key = self._key(identifier)
        now = time.time()
        window_start = now - window_seconds

        try:
            # Remove old entries outside the window
            await self._redis.zremrangebyscore(key, 0, window_start)

            # Count current entries
            current_count = await self._redis.zcard(key)

            if current_count >= max_requests:
                # Get the oldest entry's time for reset calculation
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                reset_at = oldest[0][1] + window_seconds if oldest else now + window_seconds
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_at": reset_at,
                    "retry_after": reset_at - now,
                }

            # Add current request
            await self._redis.zadd(key, {str(now): now})
            await self._redis.expire(key, window_seconds + 1)

            return {
                "allowed": True,
                "remaining": max_requests - current_count - 1,
                "reset_at": now + window_seconds,
                "retry_after": 0,
            }

        except Exception as e:
            logger.warning("Rate limit check error: %s", e)
            # Allow on error to avoid blocking
            return {"allowed": True, "remaining": max_requests, "reset_at": 0, "retry_after": 0}


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """User session storage using Redis with TTL.

    Stores session data as JSON with configurable expiry.
    """

    def __init__(self, redis_client=None, prefix: str = "acms:session",
                 default_ttl: int = 86400):
        self._redis = redis_client
        self.prefix = prefix
        self.default_ttl = default_ttl

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    async def create_session(self, user_id: str, data: Optional[Dict] = None,
                              ttl: Optional[int] = None) -> str:
        """Create a new user session.

        Args:
            user_id: User identifier.
            data: Optional session data dict.
            ttl: Session TTL in seconds.

        Returns:
            Session ID string.
        """
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "data": data or {},
        }
        key = self._key(session_id)
        try:
            await self._redis.setex(key, ttl or self.default_ttl,
                                     json.dumps(session_data, default=str))
        except Exception as e:
            logger.warning("Session create error: %s", e)
        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data.

        Args:
            session_id: Session identifier.

        Returns:
            Session data dict or None if expired/not found.
        """
        key = self._key(session_id)
        try:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning("Session get error: %s", e)
        return None

    async def update_session(self, session_id: str, data: Dict) -> bool:
        """Update session data.

        Args:
            session_id: Session identifier.
            data: New session data to merge.

        Returns:
            True if session was updated.
        """
        key = self._key(session_id)
        try:
            existing = await self.get_session(session_id)
            if existing:
                existing["data"].update(data)
                existing["updated_at"] = datetime.utcnow().isoformat()
                ttl = await self._redis.ttl(key)
                if ttl > 0:
                    await self._redis.setex(key, ttl,
                                             json.dumps(existing, default=str))
                return True
        except Exception as e:
            logger.warning("Session update error: %s", e)
        return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        key = self._key(session_id)
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Session delete error: %s", e)
            return False


# ============================================================================
# Market Data Cache
# ============================================================================

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


# ============================================================================
# In-Memory Redis Fallback
# ============================================================================

class _InMemoryRedis:
    """Simple in-memory fallback when Redis is not available.

    Implements a subset of the aioredis interface for development
    and testing without a Redis server.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}
        self._channels: Dict[str, List] = {}
        self._sorted_sets: Dict[str, Dict] = {}

    async def get(self, key: str) -> Optional[str]:
        self._check_expiry(key)
        return self._data.get(key)

    async def set(self, key: str, value: str, **kwargs) -> bool:
        self._data[key] = value
        return True

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self._data[key] = value
        self._ttl[key] = time.time() + ttl
        return True

    async def delete(self, *keys) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                self._ttl.pop(key, None)
                count += 1
        return count

    async def exists(self, key: str) -> bool:
        self._check_expiry(key)
        return key in self._data

    async def expire(self, key: str, ttl: int) -> bool:
        if key in self._data:
            self._ttl[key] = time.time() + ttl
            return True
        return False

    async def ttl(self, key: str) -> int:
        if key in self._ttl:
            remaining = self._ttl[key] - time.time()
            return max(0, int(remaining))
        return -1

    async def publish(self, channel: str, message: str) -> int:
        if channel in self._channels:
            for handler in self._channels[channel]:
                try:
                    handler(message)
                except Exception:
                    pass
        return len(self._channels.get(channel, []))

    async def pubsub(self):
        return _InMemoryPubSub(self._channels)

    def pipeline(self):
        return _InMemoryPipeline(self)

    async def zadd(self, key: str, mapping: Dict) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        self._sorted_sets[key].update(mapping)
        return len(mapping)

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        if key not in self._sorted_sets:
            return 0
        before = len(self._sorted_sets[key])
        self._sorted_sets[key] = {
            k: v for k, v in self._sorted_sets[key].items()
            if not (min_score <= v <= max_score)
        }
        return before - len(self._sorted_sets[key])

    async def zcard(self, key: str) -> int:
        return len(self._sorted_sets.get(key, {}))

    async def zrange(self, key: str, start: int, end: int, withscores: bool = False):
        ss = self._sorted_sets.get(key, {})
        items = sorted(ss.items(), key=lambda x: x[1])
        if end == 0:
            end = 1
        return items[start:end]

    async def scan_iter(self, match: str = "*"):
        import fnmatch
        for key in list(self._data.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    def _check_expiry(self, key: str) -> None:
        if key in self._ttl and time.time() > self._ttl[key]:
            self._data.pop(key, None)
            del self._ttl[key]

    async def close(self):
        pass


class _InMemoryPubSub:
    def __init__(self, channels: Dict):
        self._channels = channels
        self._subscribed = []

    async def subscribe(self, channel: str):
        if channel not in self._channels:
            self._channels[channel] = []
        self._subscribed.append(channel)

    async def unsubscribe(self):
        self._subscribed.clear()

    async def close(self):
        pass

    async def listen(self):
        return
        yield  # Make it an async generator


class _InMemoryPipeline:
    def __init__(self, redis: _InMemoryRedis):
        self._redis = redis
        self._commands = []

    def setex(self, key: str, ttl: int, value: str):
        self._commands.append(("setex", key, ttl, value))
        return self

    async def execute(self):
        results = []
        for cmd in self._commands:
            if cmd[0] == "setex":
                _, key, ttl, value = cmd
                await self._redis.setex(key, ttl, value)
                results.append(True)
        self._commands.clear()
        return results
