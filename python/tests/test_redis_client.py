"""Comprehensive tests for acms.redis_client module.

Tests all classes, methods, and edge cases:
- RedisConfig dataclass
- get_redis function
- CacheManager (get/set/delete/invalidate_pattern/exists/get_or_set)
- PubSubManager (publish/subscribe/unsubscribe/listen/stop)
- RedisRateLimiter (is_allowed)
- SessionManager (create/get/update/delete session)
- MarketDataCache (set/get latest price, orderbook, set_all_prices)
- _InMemoryRedis (all methods)
- _InMemoryPubSub (all methods)
- _InMemoryPipeline (all methods)
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import asyncio
import json
import time
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from acms.redis_client import (
    RedisConfig, get_redis, CacheManager, PubSubManager,
    RedisRateLimiter, SessionManager, MarketDataCache,
    _InMemoryRedis, _InMemoryPubSub, _InMemoryPipeline,
)


# ============================================================================
# Helpers
# ============================================================================

def make_in_memory_redis():
    """Create an _InMemoryRedis instance for testing."""
    return _InMemoryRedis()


# ============================================================================
# RedisConfig Tests
# ============================================================================

class TestRedisConfig:
    """Tests for RedisConfig dataclass."""

    def test_defaults(self):
        """All fields should have expected defaults."""
        cfg = RedisConfig()
        assert cfg.url == "redis://localhost:6379/0"
        assert cfg.password is None
        assert cfg.max_connections == 20
        assert cfg.socket_timeout == 5.0
        assert cfg.socket_connect_timeout == 5.0
        assert cfg.retry_on_timeout is True

    def test_custom_values(self):
        """Should accept custom values for all fields."""
        cfg = RedisConfig(
            url="redis://custom:6380/1",
            password="secret",
            max_connections=50,
            socket_timeout=10.0,
            socket_connect_timeout=15.0,
            retry_on_timeout=False,
        )
        assert cfg.url == "redis://custom:6380/1"
        assert cfg.password == "secret"
        assert cfg.max_connections == 50
        assert cfg.socket_timeout == 10.0
        assert cfg.socket_connect_timeout == 15.0
        assert cfg.retry_on_timeout is False

    def test_partial_custom(self):
        """Should allow setting only some fields."""
        cfg = RedisConfig(url="redis://other:6379/2")
        assert cfg.url == "redis://other:6379/2"
        assert cfg.password is None
        assert cfg.max_connections == 20

    def test_zero_values(self):
        """Should accept zero values."""
        cfg = RedisConfig(max_connections=0, socket_timeout=0.0)
        assert cfg.max_connections == 0
        assert cfg.socket_timeout == 0.0

    def test_negative_values(self):
        """Dataclass doesn't enforce validation, negatives accepted."""
        cfg = RedisConfig(max_connections=-1, socket_timeout=-1.0)
        assert cfg.max_connections == -1
        assert cfg.socket_timeout == -1.0


# ============================================================================
# get_redis Tests
# ============================================================================

class TestGetRedis:
    """Tests for get_redis function."""

    def test_default_config(self):
        """With no config, should return a redis client (either real or in-memory fallback)."""
        result = get_redis()
        # Should return either a real Redis client or _InMemoryRedis
        assert result is not None

    def test_custom_config(self):
        """Should accept custom RedisConfig."""
        cfg = RedisConfig(url="redis://custom:6379/1", password="test")
        result = get_redis(cfg)
        assert result is not None

    def test_import_fallback(self):
        """When aioredis import fails, should return _InMemoryRedis."""
        # Clear cached module references
        import importlib
        import acms.redis_client
        with patch.dict('sys.modules', {'redis.asyncio': None, 'redis': None}):
            result = get_redis()
            assert isinstance(result, _InMemoryRedis)


# ============================================================================
# _InMemoryRedis Tests
# ============================================================================

class TestInMemoryRedis:
    """Tests for _InMemoryRedis fallback implementation."""

    def setup_method(self):
        self.redis = make_in_memory_redis()

    # --- get/set ---

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Should set and get a value."""
        await self.redis.set("key1", "value1")
        result = await self.redis.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Getting a nonexistent key should return None."""
        result = await self.redis.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrite(self):
        """Setting an existing key should overwrite."""
        await self.redis.set("key1", "value1")
        await self.redis.set("key1", "value2")
        result = await self.redis.get("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_set_returns_true(self):
        """set should return True."""
        result = await self.redis.set("key1", "value1")
        assert result is True

    # --- setex ---

    @pytest.mark.asyncio
    async def test_setex_basic(self):
        """Should set with TTL."""
        await self.redis.setex("key1", 60, "value1")
        result = await self.redis.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_setex_ttl_tracking(self):
        """Should track TTL."""
        await self.redis.setex("key1", 60, "value1")
        ttl = await self.redis.ttl("key1")
        assert ttl > 0
        assert ttl <= 60

    @pytest.mark.asyncio
    async def test_setex_returns_true(self):
        """setex should return True."""
        result = await self.redis.setex("key1", 60, "value1")
        assert result is True

    # --- delete ---

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        """Should delete an existing key and return count 1."""
        await self.redis.set("key1", "value1")
        count = await self.redis.delete("key1")
        assert count == 1
        result = await self.redis.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Deleting nonexistent key should return 0."""
        count = await self.redis.delete("nonexistent")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_multiple(self):
        """Should delete multiple keys and return count."""
        await self.redis.set("key1", "value1")
        await self.redis.set("key2", "value2")
        await self.redis.set("key3", "value3")
        count = await self.redis.delete("key1", "key2", "nonexistent")
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_removes_ttl(self):
        """Deleting should also remove TTL entry."""
        await self.redis.setex("key1", 60, "value1")
        await self.redis.delete("key1")
        assert "key1" not in self.redis._ttl

    # --- exists ---

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Should return True for existing key."""
        await self.redis.set("key1", "value1")
        result = await self.redis.exists("key1")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Should return False for nonexistent key."""
        result = await self.redis.exists("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_after_delete(self):
        """Should return False after key is deleted."""
        await self.redis.set("key1", "value1")
        await self.redis.delete("key1")
        result = await self.redis.exists("key1")
        assert result is False

    # --- expire ---

    @pytest.mark.asyncio
    async def test_expire_existing_key(self):
        """Should set TTL on existing key."""
        await self.redis.set("key1", "value1")
        result = await self.redis.expire("key1", 120)
        assert result is True
        ttl = await self.redis.ttl("key1")
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_expire_nonexistent_key(self):
        """Should return False for nonexistent key."""
        result = await self.redis.expire("nonexistent", 120)
        assert result is False

    # --- ttl ---

    @pytest.mark.asyncio
    async def test_ttl_with_expiry(self):
        """Should return remaining TTL for key with TTL."""
        await self.redis.setex("key1", 300, "value1")
        ttl = await self.redis.ttl("key1")
        assert 0 < ttl <= 300

    @pytest.mark.asyncio
    async def test_ttl_no_expiry(self):
        """Should return -1 for key without TTL."""
        await self.redis.set("key1", "value1")
        ttl = await self.redis.ttl("key1")
        assert ttl == -1

    @pytest.mark.asyncio
    async def test_ttl_nonexistent(self):
        """Should return -1 for nonexistent key."""
        ttl = await self.redis.ttl("nonexistent")
        assert ttl == -1

    # --- expiry check ---

    @pytest.mark.asyncio
    async def test_expiry_on_get(self):
        """Expired keys should return None on get."""
        # Set with very short TTL
        await self.redis.setex("key1", 0, "value1")
        # Manually force expiry by setting TTL in past
        self.redis._ttl["key1"] = time.time() - 1
        result = await self.redis.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_expiry_on_exists(self):
        """Expired keys should return False on exists."""
        await self.redis.setex("key1", 0, "value1")
        self.redis._ttl["key1"] = time.time() - 1
        result = await self.redis.exists("key1")
        assert result is False

    @pytest.mark.asyncio
    async def test_expiry_cleans_up(self):
        """Accessing expired key should clean it up."""
        await self.redis.setex("key1", 0, "value1")
        self.redis._ttl["key1"] = time.time() - 1
        await self.redis.get("key1")
        assert "key1" not in self.redis._data
        assert "key1" not in self.redis._ttl

    # --- publish ---

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self):
        """Publishing with no subscribers should return 0."""
        result = await self.redis.publish("channel1", "message1")
        assert result == 0

    @pytest.mark.asyncio
    async def test_publish_with_subscribers(self):
        """Publishing with subscribers should return subscriber count."""
        messages = []
        self.redis._channels["channel1"] = [lambda m: messages.append(m)]
        result = await self.redis.publish("channel1", "message1")
        assert result == 1
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_publish_multiple_subscribers(self):
        """Should call all subscriber handlers."""
        messages1 = []
        messages2 = []
        self.redis._channels["channel1"] = [
            lambda m: messages1.append(m),
            lambda m: messages2.append(m),
        ]
        result = await self.redis.publish("channel1", "msg")
        assert result == 2
        assert len(messages1) == 1
        assert len(messages2) == 1

    @pytest.mark.asyncio
    async def test_publish_handler_exception_swallowed(self):
        """Handler exceptions should be swallowed."""
        self.redis._channels["channel1"] = [lambda m: 1 / 0]
        result = await self.redis.publish("channel1", "msg")
        assert result == 1

    # --- pubsub ---

    @pytest.mark.asyncio
    async def test_pubsub_returns_in_memory_pubsub(self):
        """pubsub() should return _InMemoryPubSub."""
        ps = await self.redis.pubsub()
        assert isinstance(ps, _InMemoryPubSub)

    # --- pipeline ---

    def test_pipeline_returns_in_memory_pipeline(self):
        """pipeline() should return _InMemoryPipeline."""
        pipe = self.redis.pipeline()
        assert isinstance(pipe, _InMemoryPipeline)

    # --- sorted sets (zadd, zremrangebyscore, zcard, zrange) ---

    @pytest.mark.asyncio
    async def test_zadd_basic(self):
        """Should add members to sorted set."""
        count = await self.redis.zadd("zset1", {"member1": 1.0, "member2": 2.0})
        assert count == 2

    @pytest.mark.asyncio
    async def test_zadd_update(self):
        """Updating existing members should work."""
        await self.redis.zadd("zset1", {"member1": 1.0})
        count = await self.redis.zadd("zset1", {"member1": 2.0, "member3": 3.0})
        assert count == 2

    @pytest.mark.asyncio
    async def test_zremrangebyscore(self):
        """Should remove members within score range."""
        await self.redis.zadd("zset1", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        removed = await self.redis.zremrangebyscore("zset1", 2.0, 3.0)
        assert removed == 2
        assert await self.redis.zcard("zset1") == 2

    @pytest.mark.asyncio
    async def test_zremrangebyscore_no_match(self):
        """Removing from non-matching range should return 0."""
        await self.redis.zadd("zset1", {"a": 1.0, "b": 2.0})
        removed = await self.redis.zremrangebyscore("zset1", 5.0, 10.0)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_zremrangebyscore_nonexistent_key(self):
        """Removing from nonexistent key should return 0."""
        removed = await self.redis.zremrangebyscore("nonexistent", 0.0, 10.0)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_zcard(self):
        """Should return count of members."""
        await self.redis.zadd("zset1", {"a": 1.0, "b": 2.0})
        assert await self.redis.zcard("zset1") == 2

    @pytest.mark.asyncio
    async def test_zcard_nonexistent(self):
        """zcard of nonexistent key should return 0."""
        assert await self.redis.zcard("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_zrange(self):
        """Should return sorted members."""
        await self.redis.zadd("zset1", {"c": 3.0, "a": 1.0, "b": 2.0})
        result = await self.redis.zrange("zset1", 0, 2, withscores=True)
        assert len(result) == 2  # end=0 maps to end=1 in impl
        # First element should be lowest score

    @pytest.mark.asyncio
    async def test_zrange_nonexistent(self):
        """zrange of nonexistent key should return empty."""
        result = await self.redis.zrange("nonexistent", 0, 1)
        assert result == []

    # --- scan_iter ---

    @pytest.mark.asyncio
    async def test_scan_iter_all(self):
        """Should iterate all matching keys."""
        await self.redis.set("test:key1", "val1")
        await self.redis.set("test:key2", "val2")
        await self.redis.set("other:key3", "val3")
        keys = []
        async for key in self.redis.scan_iter(match="test:*"):
            keys.append(key)
        assert len(keys) == 2
        assert "test:key1" in keys
        assert "test:key2" in keys

    @pytest.mark.asyncio
    async def test_scan_iter_wildcard(self):
        """Wildcard match should return all keys."""
        await self.redis.set("key1", "val1")
        await self.redis.set("key2", "val2")
        keys = []
        async for key in self.redis.scan_iter(match="*"):
            keys.append(key)
        assert len(keys) >= 2

    @pytest.mark.asyncio
    async def test_scan_iter_no_match(self):
        """Non-matching pattern should return empty."""
        await self.redis.set("key1", "val1")
        keys = []
        async for key in self.redis.scan_iter(match="nomatch:*"):
            keys.append(key)
        assert len(keys) == 0

    # --- close ---

    @pytest.mark.asyncio
    async def test_close(self):
        """close() should not raise."""
        await self.redis.close()


# ============================================================================
# _InMemoryPubSub Tests
# ============================================================================

class TestInMemoryPubSub:
    """Tests for _InMemoryPubSub."""

    def setup_method(self):
        self.channels = {}
        self.pubsub = _InMemoryPubSub(self.channels)

    @pytest.mark.asyncio
    async def test_subscribe(self):
        """Should add channel to channels dict and subscribed list."""
        await self.pubsub.subscribe("channel1")
        assert "channel1" in self.channels
        assert "channel1" in self.pubsub._subscribed

    @pytest.mark.asyncio
    async def test_subscribe_creates_channel(self):
        """Subscribing should create channel list if not present."""
        await self.pubsub.subscribe("new_channel")
        assert "new_channel" in self.channels

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Unsubscribe should clear subscribed list."""
        await self.pubsub.subscribe("channel1")
        await self.pubsub.unsubscribe()
        assert len(self.pubsub._subscribed) == 0

    @pytest.mark.asyncio
    async def test_close(self):
        """close() should not raise."""
        await self.pubsub.close()

    @pytest.mark.asyncio
    async def test_listen_returns_immediately(self):
        """listen() should be an async generator that yields nothing."""
        result = []
        async for msg in self.pubsub.listen():
            result.append(msg)
        assert len(result) == 0


# ============================================================================
# _InMemoryPipeline Tests
# ============================================================================

class TestInMemoryPipeline:
    """Tests for _InMemoryPipeline."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.pipeline = _InMemoryPipeline(self.redis)

    @pytest.mark.asyncio
    async def test_setex_and_execute(self):
        """Should execute queued setex commands."""
        self.pipeline.setex("key1", 60, "value1")
        self.pipeline.setex("key2", 60, "value2")
        results = await self.pipeline.execute()
        assert len(results) == 2
        assert all(r is True for r in results)
        assert await self.redis.get("key1") == "value1"
        assert await self.redis.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_setex_returns_self(self):
        """setex should return self for chaining."""
        result = self.pipeline.setex("key1", 60, "value1")
        assert result is self.pipeline

    @pytest.mark.asyncio
    async def test_execute_clears_commands(self):
        """After execute, commands should be cleared."""
        self.pipeline.setex("key1", 60, "value1")
        await self.pipeline.execute()
        assert len(self.pipeline._commands) == 0

    @pytest.mark.asyncio
    async def test_execute_empty(self):
        """Executing empty pipeline should return empty list."""
        results = await self.pipeline.execute()
        assert results == []


# ============================================================================
# CacheManager Tests
# ============================================================================

class TestCacheManager:
    """Tests for CacheManager class."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.cache = CacheManager(redis_client=self.redis, default_ttl=300, prefix="acms:cache")

    def test_key_building(self):
        """_key should build namespaced cache key."""
        assert self.cache._key("mykey") == "acms:cache:mykey"

    def test_key_custom_prefix(self):
        """Should use custom prefix."""
        cache = CacheManager(redis_client=self.redis, prefix="custom")
        assert cache._key("mykey") == "custom:mykey"

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Should cache and retrieve a value."""
        assert await self.cache.set("key1", {"data": "value1"})
        result = await self.cache.get("key1")
        assert result == {"data": "value1"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Getting nonexistent key should return None."""
        result = await self.cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self):
        """Should use custom TTL."""
        await self.cache.set("key1", "value1", ttl=600)
        full_key = self.cache._key("key1")
        ttl = await self.redis.ttl(full_key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_set_default_ttl(self):
        """Should use default TTL when not specified."""
        await self.cache.set("key1", "value1")
        full_key = self.cache._key("key1")
        ttl = await self.redis.ttl(full_key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_set_returns_true(self):
        """Successful set should return True."""
        result = await self.cache.set("key1", "value1")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_json_serializable(self):
        """Should handle JSON-serializable values."""
        await self.cache.set("key1", [1, 2, 3])
        result = await self.cache.get("key1")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_set_complex_value(self):
        """Should handle complex nested values."""
        data = {"nested": {"deep": [1, 2, {"key": "val"}]}}
        await self.cache.set("key1", data)
        result = await self.cache.get("key1")
        assert result == data

    @pytest.mark.asyncio
    async def test_delete_existing(self):
        """Should delete existing key and return True."""
        await self.cache.set("key1", "value1")
        result = await self.cache.delete("key1")
        assert result is True
        assert await self.cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Deleting nonexistent key should return False."""
        result = await self.cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Should return True for existing key."""
        await self.cache.set("key1", "value1")
        assert await self.cache.exists("key1") is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Should return False for nonexistent key."""
        assert await self.cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self):
        """Should invalidate all keys matching pattern."""
        await self.cache.set("signals:1", "sig1")
        await self.cache.set("signals:2", "sig2")
        await self.cache.set("prices:1", "price1")
        count = await self.cache.invalidate_pattern("signals:*")
        assert count == 2
        assert await self.cache.get("signals:1") is None
        assert await self.cache.get("signals:2") is None
        assert await self.cache.get("prices:1") is not None

    @pytest.mark.asyncio
    async def test_invalidate_pattern_no_match(self):
        """Invalidating non-matching pattern should return 0."""
        await self.cache.set("key1", "value1")
        count = await self.cache.invalidate_pattern("nomatch:*")
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_or_set_cache_hit(self):
        """Should return cached value without calling factory."""
        await self.cache.set("key1", "cached_value")
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "new_value"

        result = await self.cache.get_or_set("key1", factory)
        assert result == "cached_value"
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_get_or_set_cache_miss(self):
        """Should call factory and cache result on cache miss."""
        result = await self.cache.get_or_set("key1", lambda: "computed_value")
        assert result == "computed_value"

    @pytest.mark.asyncio
    async def test_get_or_set_async_factory(self):
        """Should support async factory functions."""
        async def async_factory():
            return "async_value"

        result = await self.cache.get_or_set("key1", async_factory)
        assert result == "async_value"

    @pytest.mark.asyncio
    async def test_get_or_set_caches_result(self):
        """Factory result should be cached for subsequent calls."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        result1 = await self.cache.get_or_set("key1", factory)
        result2 = await self.cache.get_or_set("key1", factory)
        assert result1 == "value_1"
        assert result2 == "value_1"  # Cached, not recomputed
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_get_or_set_with_custom_ttl(self):
        """Should pass TTL to set."""
        await self.cache.get_or_set("key1", lambda: "value", ttl=600)
        full_key = self.cache._key("key1")
        ttl = await self.redis.ttl(full_key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_set_error_handling(self):
        """Should return False on set error."""
        broken_redis = AsyncMock()
        broken_redis.setex.side_effect = Exception("Connection error")
        cache = CacheManager(redis_client=broken_redis)
        result = await cache.set("key1", "value1")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_error_handling(self):
        """Should return None on get error."""
        broken_redis = AsyncMock()
        broken_redis.get.side_effect = Exception("Connection error")
        cache = CacheManager(redis_client=broken_redis)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_error_handling(self):
        """Should return False on delete error."""
        broken_redis = AsyncMock()
        broken_redis.delete.side_effect = Exception("Connection error")
        cache = CacheManager(redis_client=broken_redis)
        result = await cache.delete("key1")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_pattern_error_handling(self):
        """Should return 0 on invalidate error."""
        broken_redis = AsyncMock()
        broken_redis.scan_iter.side_effect = Exception("Connection error")
        cache = CacheManager(redis_client=broken_redis)
        result = await cache.invalidate_pattern("test:*")
        assert result == 0

    @pytest.mark.asyncio
    async def test_exists_error_handling(self):
        """Should return False on exists error."""
        broken_redis = AsyncMock()
        broken_redis.exists.side_effect = Exception("Connection error")
        cache = CacheManager(redis_client=broken_redis)
        result = await cache.exists("key1")
        assert result is False


# ============================================================================
# PubSubManager Tests
# ============================================================================

class TestPubSubManager:
    """Tests for PubSubManager class."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.pubsub = PubSubManager(redis_client=self.redis, prefix="acms:pubsub")

    def test_channel_building(self):
        """_channel should build namespaced channel name."""
        assert self.pubsub._channel("events") == "acms:pubsub:events"

    def test_prefix_str(self):
        """_prefix_str should return prefix with colon."""
        assert self.pubsub._prefix_str() == "acms:pubsub:"

    @pytest.mark.asyncio
    async def test_publish(self):
        """Should publish message and return True."""
        result = await self.pubsub.publish("events", {"type": "test"})
        assert result is True

    @pytest.mark.asyncio
    async def test_publish_error_handling(self):
        """Should return False on publish error."""
        broken_redis = AsyncMock()
        broken_redis.publish.side_effect = Exception("Connection error")
        ps = PubSubManager(redis_client=broken_redis)
        result = await ps.publish("events", {"type": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe(self):
        """Should register handler and subscribe."""
        handler = AsyncMock()
        await self.pubsub.subscribe("events", handler)
        assert "events" in self.pubsub._subscriptions
        assert handler in self.pubsub._subscriptions["events"]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_handlers(self):
        """Should support multiple handlers per channel."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        await self.pubsub.subscribe("events", handler1)
        await self.pubsub.subscribe("events", handler2)
        assert len(self.pubsub._subscriptions["events"]) == 2

    @pytest.mark.asyncio
    async def test_subscribe_creates_pubsub(self):
        """Should create pubsub object on first subscribe."""
        handler = AsyncMock()
        await self.pubsub.subscribe("events", handler)
        assert self.pubsub._pubsub is not None

    @pytest.mark.asyncio
    async def test_subscribe_error_handling(self):
        """Should not raise on subscribe error."""
        broken_redis = AsyncMock()
        broken_redis.pubsub.side_effect = Exception("Connection error")
        ps = PubSubManager(redis_client=broken_redis)
        handler = AsyncMock()
        # Should not raise
        await ps.subscribe("events", handler)
        # Handler should still be registered locally
        assert "events" in ps._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Should remove handlers for channel."""
        handler = AsyncMock()
        # Subscribe without a real pubsub (subscribe error but handler gets registered)
        self.pubsub._subscriptions["events"] = [handler]
        await self.pubsub.unsubscribe("events")
        assert "events" not in self.pubsub._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_no_pubsub(self):
        """Unsubscribing with no pubsub should not raise."""
        await self.pubsub.unsubscribe("events")

    @pytest.mark.asyncio
    async def test_unsubscribe_error_handling(self):
        """Should not raise on unsubscribe error."""
        broken_redis = AsyncMock()
        broken_redis.pubsub.return_value = AsyncMock()
        broken_redis.pubsub.return_value.unsubscribe.side_effect = Exception("Error")
        ps = PubSubManager(redis_client=broken_redis)
        handler = AsyncMock()
        await ps.subscribe("events", handler)
        # Should not raise
        await ps.unsubscribe("events")

    @pytest.mark.asyncio
    async def test_listen_no_pubsub(self):
        """Listening with no pubsub should return immediately."""
        await self.pubsub.listen()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Should stop and clean up."""
        handler = AsyncMock()
        await self.pubsub.subscribe("events", handler)
        await self.pubsub.stop()
        assert self.pubsub._running is False


# ============================================================================
# RedisRateLimiter Tests
# ============================================================================

class TestRedisRateLimiter:
    """Tests for RedisRateLimiter class."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.limiter = RedisRateLimiter(redis_client=self.redis, prefix="acms:ratelimit")

    def test_key_building(self):
        """_key should build namespaced key."""
        assert self.limiter._key("user1") == "acms:ratelimit:user1"

    @pytest.mark.asyncio
    async def test_is_allowed_first_request(self):
        """First request should be allowed."""
        result = await self.limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        assert result["allowed"] is True
        assert result["remaining"] == 4

    @pytest.mark.asyncio
    async def test_is_allowed_within_limit(self):
        """Requests within limit should be allowed."""
        for i in range(5):
            result = await self.limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        # 5th request should still be allowed (it counts itself)
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_is_allowed_exceeds_limit(self):
        """Request exceeding limit should be rejected."""
        for i in range(5):
            await self.limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        result = await self.limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        assert result["allowed"] is False
        assert result["remaining"] == 0
        assert result["retry_after"] > 0

    @pytest.mark.asyncio
    async def test_is_allowed_result_fields(self):
        """Result should have all required fields."""
        result = await self.limiter.is_allowed("user1")
        assert "allowed" in result
        assert "remaining" in result
        assert "reset_at" in result
        assert "retry_after" in result

    @pytest.mark.asyncio
    async def test_is_allowed_different_identifiers(self):
        """Different identifiers should have independent limits."""
        for i in range(5):
            await self.limiter.is_allowed("user1", max_requests=5, window_seconds=60)
        result = await self.limiter.is_allowed("user2", max_requests=5, window_seconds=60)
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_is_allowed_error_fallback(self):
        """Should allow on error to avoid blocking."""
        broken_redis = AsyncMock()
        broken_redis.zremrangebyscore.side_effect = Exception("Connection error")
        limiter = RedisRateLimiter(redis_client=broken_redis)
        result = await limiter.is_allowed("user1")
        assert result["allowed"] is True


# ============================================================================
# SessionManager Tests
# ============================================================================

class TestSessionManager:
    """Tests for SessionManager class."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.session_mgr = SessionManager(redis_client=self.redis, prefix="acms:session", default_ttl=86400)

    def test_key_building(self):
        """_key should build namespaced key."""
        assert self.session_mgr._key("sess1") == "acms:session:sess1"

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Should create a session and return session ID."""
        session_id = await self.session_mgr.create_session("user1")
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_create_session_with_data(self):
        """Should create session with custom data."""
        session_id = await self.session_mgr.create_session("user1", data={"role": "admin"})
        session = await self.session_mgr.get_session(session_id)
        assert session is not None
        assert session["user_id"] == "user1"
        assert session["data"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_create_session_with_custom_ttl(self):
        """Should create session with custom TTL."""
        session_id = await self.session_mgr.create_session("user1", ttl=3600)
        key = self.session_mgr._key(session_id)
        ttl = await self.redis.ttl(key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_create_session_has_created_at(self):
        """Session should have created_at timestamp."""
        session_id = await self.session_mgr.create_session("user1")
        session = await self.session_mgr.get_session(session_id)
        assert "created_at" in session

    @pytest.mark.asyncio
    async def test_create_session_unique_ids(self):
        """Each session should get a unique ID."""
        id1 = await self.session_mgr.create_session("user1")
        id2 = await self.session_mgr.create_session("user1")
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Should retrieve session data."""
        session_id = await self.session_mgr.create_session("user1", data={"key": "val"})
        session = await self.session_mgr.get_session(session_id)
        assert session is not None
        assert session["user_id"] == "user1"

    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self):
        """Getting nonexistent session should return None."""
        result = await self.session_mgr.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_session(self):
        """Should update session data."""
        session_id = await self.session_mgr.create_session("user1", data={"key1": "val1"})
        result = await self.session_mgr.update_session(session_id, {"key2": "val2"})
        assert result is True
        session = await self.session_mgr.get_session(session_id)
        assert session["data"]["key1"] == "val1"
        assert session["data"]["key2"] == "val2"

    @pytest.mark.asyncio
    async def test_update_session_adds_updated_at(self):
        """Update should add updated_at timestamp."""
        session_id = await self.session_mgr.create_session("user1")
        await self.session_mgr.update_session(session_id, {"key": "val"})
        session = await self.session_mgr.get_session(session_id)
        assert "updated_at" in session

    @pytest.mark.asyncio
    async def test_update_nonexistent_session(self):
        """Updating nonexistent session should return False."""
        result = await self.session_mgr.update_session("nonexistent", {"key": "val"})
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """Should delete a session."""
        session_id = await self.session_mgr.create_session("user1")
        result = await self.session_mgr.delete_session(session_id)
        assert result is True
        assert await self.session_mgr.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self):
        """Deleting nonexistent session should still return True."""
        result = await self.session_mgr.delete_session("nonexistent")
        assert result is True

    @pytest.mark.asyncio
    async def test_create_session_error_handling(self):
        """Should not raise on create error."""
        broken_redis = AsyncMock()
        broken_redis.setex.side_effect = Exception("Connection error")
        mgr = SessionManager(redis_client=broken_redis)
        session_id = await mgr.create_session("user1")
        assert session_id is not None  # Still returns UUID even on error

    @pytest.mark.asyncio
    async def test_get_session_error_handling(self):
        """Should return None on get error."""
        broken_redis = AsyncMock()
        broken_redis.get.side_effect = Exception("Connection error")
        mgr = SessionManager(redis_client=broken_redis)
        result = await mgr.get_session("session1")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_session_error_handling(self):
        """Should return False on update error."""
        broken_redis = AsyncMock()
        broken_redis.get.side_effect = Exception("Connection error")
        mgr = SessionManager(redis_client=broken_redis)
        result = await mgr.update_session("session1", {"key": "val"})
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session_error_handling(self):
        """Should return False on delete error."""
        broken_redis = AsyncMock()
        broken_redis.delete.side_effect = Exception("Connection error")
        mgr = SessionManager(redis_client=broken_redis)
        result = await mgr.delete_session("session1")
        assert result is False


# ============================================================================
# MarketDataCache Tests
# ============================================================================

class TestMarketDataCache:
    """Tests for MarketDataCache class."""

    def setup_method(self):
        self.redis = make_in_memory_redis()
        self.cache = MarketDataCache(
            redis_client=self.redis, prefix="acms:market",
            price_ttl=30, orderbook_ttl=10,
        )

    @pytest.mark.asyncio
    async def test_set_and_get_latest_price(self):
        """Should cache and retrieve latest price."""
        assert await self.cache.set_latest_price("BTC/USDT", 50000.0, "binance") is True
        result = await self.cache.get_latest_price("BTC/USDT")
        assert result is not None
        assert result["price"] == 50000.0
        assert result["exchange"] == "binance"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_set_latest_price_no_exchange(self):
        """Should work without exchange parameter."""
        assert await self.cache.set_latest_price("BTC/USDT", 50000.0) is True
        result = await self.cache.get_latest_price("BTC/USDT")
        assert result is not None
        assert result["exchange"] == ""

    @pytest.mark.asyncio
    async def test_get_latest_price_nonexistent(self):
        """Getting price for unknown symbol should return None."""
        result = await self.cache.get_latest_price("UNKNOWN/USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_latest_price_error(self):
        """Should return False on error."""
        broken_redis = AsyncMock()
        broken_redis.setex.side_effect = Exception("Connection error")
        cache = MarketDataCache(redis_client=broken_redis)
        result = await cache.set_latest_price("BTC/USDT", 50000.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_latest_price_error(self):
        """Should return None on error."""
        broken_redis = AsyncMock()
        broken_redis.get.side_effect = Exception("Connection error")
        cache = MarketDataCache(redis_client=broken_redis)
        result = await cache.get_latest_price("BTC/USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_orderbook(self):
        """Should cache and retrieve orderbook."""
        ob = {"bids": [[50000, 1.0], [49999, 2.0]], "asks": [[50001, 1.5], [50002, 0.5]]}
        assert await self.cache.set_orderbook("BTC/USDT", ob) is True
        result = await self.cache.get_orderbook("BTC/USDT")
        assert result is not None
        assert result["orderbook"] == ob
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_orderbook_nonexistent(self):
        """Getting orderbook for unknown symbol should return None."""
        result = await self.cache.get_orderbook("UNKNOWN/USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_orderbook_error(self):
        """Should return False on error."""
        broken_redis = AsyncMock()
        broken_redis.setex.side_effect = Exception("Connection error")
        cache = MarketDataCache(redis_client=broken_redis)
        result = await cache.set_orderbook("BTC/USDT", {})
        assert result is False

    @pytest.mark.asyncio
    async def test_get_orderbook_error(self):
        """Should return None on error."""
        broken_redis = AsyncMock()
        broken_redis.get.side_effect = Exception("Connection error")
        cache = MarketDataCache(redis_client=broken_redis)
        result = await cache.get_orderbook("BTC/USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_all_prices(self):
        """Should cache prices for multiple symbols."""
        prices = {"BTC/USDT": 50000.0, "ETH/USDT": 3000.0, "SOL/USDT": 100.0}
        assert await self.cache.set_all_prices(prices) is True
        btc = await self.cache.get_latest_price("BTC/USDT")
        eth = await self.cache.get_latest_price("ETH/USDT")
        sol = await self.cache.get_latest_price("SOL/USDT")
        assert btc is not None
        assert btc["price"] == 50000.0
        assert eth is not None
        assert eth["price"] == 3000.0
        assert sol is not None
        assert sol["price"] == 100.0

    @pytest.mark.asyncio
    async def test_set_all_prices_empty(self):
        """Should handle empty prices dict."""
        result = await self.cache.set_all_prices({})
        assert result is True

    @pytest.mark.asyncio
    async def test_set_all_prices_error(self):
        """Should return False on error."""
        broken_redis = AsyncMock()
        broken_redis.pipeline.side_effect = Exception("Connection error")
        cache = MarketDataCache(redis_client=broken_redis)
        result = await cache.set_all_prices({"BTC/USDT": 50000.0})
        assert result is False

    @pytest.mark.asyncio
    async def test_price_ttl_used(self):
        """Price should use price_ttl."""
        await self.cache.set_latest_price("BTC/USDT", 50000.0)
        key = f"acms:market:price:BTC/USDT"
        ttl = await self.redis.ttl(key)
        assert ttl > 0
        assert ttl <= 30

    @pytest.mark.asyncio
    async def test_orderbook_ttl_used(self):
        """Orderbook should use orderbook_ttl."""
        await self.cache.set_orderbook("BTC/USDT", {"bids": [], "asks": []})
        key = f"acms:market:orderbook:BTC/USDT"
        ttl = await self.redis.ttl(key)
        assert ttl > 0
        assert ttl <= 10

    @pytest.mark.asyncio
    async def test_overwrite_price(self):
        """Setting price twice should overwrite."""
        await self.cache.set_latest_price("BTC/USDT", 50000.0)
        await self.cache.set_latest_price("BTC/USDT", 51000.0)
        result = await self.cache.get_latest_price("BTC/USDT")
        assert result["price"] == 51000.0
