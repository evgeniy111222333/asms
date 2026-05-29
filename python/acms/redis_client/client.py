"""Redis client factory and in-memory fallback."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


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
                except Exception as e:
                    logger.warning("Handler error in local publish: %s", e)
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


__all__ = ["RedisConfig", "get_redis", "_InMemoryRedis", "_InMemoryPubSub", "_InMemoryPipeline"]
