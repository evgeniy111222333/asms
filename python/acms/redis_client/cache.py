"""Cache manager with TTL and pattern invalidation."""

import json
import logging
import asyncio
from typing import Optional, Any, Callable

logger = logging.getLogger(__name__)


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



__all__ = ["CacheManager"]
