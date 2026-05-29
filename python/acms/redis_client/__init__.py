"""Redis Client Module - Caching, PubSub, Rate Limiting, Sessions.

Re-exports all public names from submodules for backward compatibility.
"""

from acms.redis_client.client import RedisConfig, get_redis, _InMemoryRedis, _InMemoryPubSub, _InMemoryPipeline
from acms.redis_client.cache import CacheManager
from acms.redis_client.pubsub import PubSubManager
from acms.redis_client.rate_limiter import RedisRateLimiter
from acms.redis_client.session import SessionManager
from acms.redis_client.market_data import MarketDataCache

__all__ = [
    "RedisConfig",
    "get_redis",
    "_InMemoryRedis",
    "_InMemoryPubSub",
    "_InMemoryPipeline",
    "CacheManager",
    "PubSubManager",
    "RedisRateLimiter",
    "SessionManager",
    "MarketDataCache",
]
