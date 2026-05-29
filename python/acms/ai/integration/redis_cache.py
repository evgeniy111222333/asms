"""Redis Model Cache for ACMS AI Pipeline.

Implements:
- ModelCache: Redis-backed model artifact cache with TTL management
- FeatureCache: Precomputed feature caching for low-latency inference
- PredictionCache: Prediction result caching with invalidation strategies
- DistributedCacheCoordinator: Multi-instance cache coordination
- Cache warming on startup
- TTL management with configurable expiry policies
- Cache invalidation strategies (time-based, event-based, version-based)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Configuration
# ============================================================================

class InvalidationStrategy(str, Enum):
    """Cache invalidation strategy."""
    TTL = "ttl"
    VERSION = "version"
    EVENT = "event"
    HYBRID = "hybrid"


@dataclass
class CacheConfig:
    """Redis cache configuration.

    Attributes:
        redis_url: Redis connection URL.
        prefix: Key prefix namespace.
        default_ttl: Default TTL in seconds.
        model_ttl: TTL for model artifacts.
        feature_ttl: TTL for feature caches.
        prediction_ttl: TTL for prediction caches.
        max_serialized_size_mb: Maximum serialized value size in MB.
        enable_compression: Whether to compress values.
        invalidation_strategy: Cache invalidation strategy.
    """
    redis_url: str = "redis://localhost:6379/1"
    prefix: str = "acms:ai:cache"
    default_ttl: int = 3600
    model_ttl: int = 86400
    feature_ttl: int = 300
    prediction_ttl: int = 60
    max_serialized_size_mb: int = 50
    enable_compression: bool = False
    invalidation_strategy: InvalidationStrategy = InvalidationStrategy.HYBRID


# ============================================================================
# Model Cache
# ============================================================================

class ModelCache:
    """Redis-backed model artifact cache with version management.

    Caches serialized model artifacts (weights, configs, metadata)
    with version-based invalidation and TTL management.
    """

    def __init__(self, redis_client: Any = None, config: Optional[CacheConfig] = None):
        """Initialize the model cache.

        Args:
            redis_client: Redis client instance (async).
            config: Cache configuration.
        """
        self._redis = redis_client
        self.config = config or CacheConfig()
        self._prefix = f"{self.config.prefix}:model"
        self._version_registry: Dict[str, str] = {}  # model_id -> version_hash
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _key(self, model_id: str, version: Optional[str] = None) -> str:
        """Build cache key for a model.

        Args:
            model_id: Model identifier.
            version: Model version (uses current if None).

        Returns:
            Namespaced cache key.
        """
        ver = version or self._version_registry.get(model_id, "latest")
        return f"{self._prefix}:{model_id}:{ver}"

    async def get(self, model_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a cached model artifact.

        Args:
            model_id: Model identifier.
            version: Model version.

        Returns:
            Cached model data dict or None if not found.
        """
        key = self._key(model_id, version)
        try:
            data = await self._redis.get(key)
            if data is not None:
                self._hit_count += 1
                return json.loads(data)
            self._miss_count += 1
        except Exception as e:
            logger.warning("Model cache get error for '%s': %s", model_id, e)
            self._miss_count += 1
        return None

    async def set(self, model_id: str, data: Dict[str, Any],
                  version: Optional[str] = None, ttl: Optional[int] = None) -> bool:
        """Cache a model artifact.

        Args:
            model_id: Model identifier.
            data: Model data to cache.
            version: Model version string.
            ttl: Time-to-live in seconds.

        Returns:
            True if cached successfully.
        """
        version = version or data.get("version", "latest")
        key = self._key(model_id, version)
        ttl = ttl or self.config.model_ttl

        # Update version registry
        self._version_registry[model_id] = version

        try:
            serialized = json.dumps(data, default=str)
            # Size check
            size_mb = len(serialized) / (1024 * 1024)
            if size_mb > self.config.max_serialized_size_mb:
                logger.warning("Model '%s' too large to cache (%.1f MB)",
                               model_id, size_mb)
                return False

            await self._redis.setex(key, ttl, serialized)
            logger.debug("Model '%s' v%s cached (%.1f KB)",
                         model_id, version, len(serialized) / 1024)
            return True
        except Exception as e:
            logger.warning("Model cache set error for '%s': %s", model_id, e)
            return False

    async def delete(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete a cached model.

        Args:
            model_id: Model identifier.
            version: Model version (deletes all versions if None).

        Returns:
            True if deletion was successful.
        """
        try:
            if version:
                key = self._key(model_id, version)
                result = await self._redis.delete(key)
                return result > 0
            else:
                # Delete all versions
                pattern = f"{self._prefix}:{model_id}:*"
                keys = []
                async for key in self._redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    deleted = await self._redis.delete(*keys)
                    self._version_registry.pop(model_id, None)
                    return deleted > 0
                return False
        except Exception as e:
            logger.warning("Model cache delete error for '%s': %s", model_id, e)
            return False

    async def invalidate_version(self, model_id: str, new_version: str) -> int:
        """Invalidate all cached versions older than new_version.

        Args:
            model_id: Model identifier.
            new_version: New version string.

        Returns:
            Number of keys invalidated.
        """
        old_version = self._version_registry.get(model_id)
        self._version_registry[model_id] = new_version

        if old_version and old_version != new_version:
            old_key = self._key(model_id, old_version)
            try:
                result = await self._redis.delete(old_key)
                logger.info("Invalidated model '%s' version '%s'",
                            model_id, old_version)
                return result
            except Exception as e:
                logger.warning("Model cache invalidation error: %s", e)
        return 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as fraction."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self.hit_rate, 4),
            "tracked_models": len(self._version_registry),
        }


# ============================================================================
# Feature Cache
# ============================================================================

class FeatureCache:
    """Precomputed feature cache for low-latency inference.

    Caches feature vectors keyed by symbol + timeframe,
    with short TTLs appropriate for real-time serving.
    """

    def __init__(self, redis_client: Any = None, config: Optional[CacheConfig] = None):
        """Initialize the feature cache.

        Args:
            redis_client: Redis client instance.
            config: Cache configuration.
        """
        self._redis = redis_client
        self.config = config or CacheConfig()
        self._prefix = f"{self.config.prefix}:feature"
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _key(self, symbol: str, timeframe: str, feature_set: str = "default") -> str:
        """Build cache key for features.

        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe string.
            feature_set: Feature set identifier.

        Returns:
            Namespaced cache key.
        """
        return f"{self._prefix}:{symbol}:{timeframe}:{feature_set}"

    async def get(self, symbol: str, timeframe: str,
                  feature_set: str = "default") -> Optional[Dict[str, Any]]:
        """Get cached features.

        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe string.
            feature_set: Feature set identifier.

        Returns:
            Feature data dict or None.
        """
        key = self._key(symbol, timeframe, feature_set)
        try:
            data = await self._redis.get(key)
            if data is not None:
                self._hit_count += 1
                return json.loads(data)
            self._miss_count += 1
        except Exception as e:
            logger.warning("Feature cache get error: %s", e)
            self._miss_count += 1
        return None

    async def set(self, symbol: str, timeframe: str,
                  features: Dict[str, Any], feature_set: str = "default",
                  ttl: Optional[int] = None) -> bool:
        """Cache feature data.

        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe string.
            features: Feature data to cache.
            feature_set: Feature set identifier.
            ttl: Time-to-live in seconds.

        Returns:
            True if cached successfully.
        """
        key = self._key(symbol, timeframe, feature_set)
        ttl = ttl or self.config.feature_ttl
        try:
            serialized = json.dumps(features, default=str)
            await self._redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning("Feature cache set error: %s", e)
            return False

    async def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cached features for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"{self._prefix}:{symbol}:*"
        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await self._redis.delete(*keys)
        except Exception as e:
            logger.warning("Feature cache invalidation error: %s", e)
        return 0

    async def warm(self, symbols: List[str], timeframes: List[str],
                   feature_factory: Callable[[str, str], Awaitable[Dict[str, Any]]]) -> int:
        """Warm the feature cache for specified symbols and timeframes.

        Args:
            symbols: List of symbols to warm.
            timeframes: List of timeframes to warm.
            feature_factory: Async callable to compute features.

        Returns:
            Number of entries cached.
        """
        count = 0
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    features = await feature_factory(symbol, timeframe)
                    if features:
                        await self.set(symbol, timeframe, features)
                        count += 1
                except Exception as e:
                    logger.warning("Feature cache warm error for %s/%s: %s",
                                   symbol, timeframe, e)
        logger.info("Feature cache warmed: %d entries", count)
        return count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0 else 0.0, 4
            ),
        }


# ============================================================================
# Prediction Cache
# ============================================================================

class PredictionCache:
    """Prediction result cache for low-latency serving.

    Caches model predictions keyed by input hash, enabling
    instant responses for repeated queries.
    """

    def __init__(self, redis_client: Any = None, config: Optional[CacheConfig] = None):
        """Initialize the prediction cache.

        Args:
            redis_client: Redis client instance.
            config: Cache configuration.
        """
        self._redis = redis_client
        self.config = config or CacheConfig()
        self._prefix = f"{self.config.prefix}:prediction"
        self._hit_count: int = 0
        self._miss_count: int = 0

    @staticmethod
    def _hash_input(model_id: str, input_data: Dict[str, Any]) -> str:
        """Create a deterministic hash for model input.

        Args:
            model_id: Model identifier.
            input_data: Model input dict.

        Returns:
            SHA-256 hash string.
        """
        canonical = json.dumps({"model_id": model_id, **input_data}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _key(self, model_id: str, input_hash: str) -> str:
        """Build cache key for a prediction."""
        return f"{self._prefix}:{model_id}:{input_hash}"

    async def get(self, model_id: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a cached prediction.

        Args:
            model_id: Model identifier.
            input_data: Model input (used to derive cache key).

        Returns:
            Cached prediction dict or None.
        """
        input_hash = self._hash_input(model_id, input_data)
        key = self._key(model_id, input_hash)
        try:
            data = await self._redis.get(key)
            if data is not None:
                self._hit_count += 1
                return json.loads(data)
            self._miss_count += 1
        except Exception as e:
            logger.warning("Prediction cache get error: %s", e)
            self._miss_count += 1
        return None

    async def set(self, model_id: str, input_data: Dict[str, Any],
                  prediction: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache a prediction result.

        Args:
            model_id: Model identifier.
            input_data: Model input.
            prediction: Prediction result to cache.
            ttl: Time-to-live in seconds.

        Returns:
            True if cached successfully.
        """
        input_hash = self._hash_input(model_id, input_data)
        key = self._key(model_id, input_hash)
        ttl = ttl or self.config.prediction_ttl
        try:
            serialized = json.dumps(prediction, default=str)
            await self._redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.warning("Prediction cache set error: %s", e)
            return False

    async def invalidate_model(self, model_id: str) -> int:
        """Invalidate all cached predictions for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"{self._prefix}:{model_id}:*"
        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await self._redis.delete(*keys)
        except Exception as e:
            logger.warning("Prediction cache invalidation error: %s", e)
        return 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0 else 0.0, 4
            ),
        }


# ============================================================================
# Distributed Cache Coordinator
# ============================================================================

class DistributedCacheCoordinator:
    """Coordinates cache invalidation across multiple service instances.

    Uses Redis Pub/Sub to broadcast invalidation events so that
    all instances invalidate their local caches when data changes.
    """

    def __init__(self, redis_client: Any = None,
                 channel: str = "acms:ai:cache:invalidation",
                 instance_id: Optional[str] = None):
        """Initialize the cache coordinator.

        Args:
            redis_client: Redis client instance.
            channel: Pub/Sub channel for invalidation events.
            instance_id: Unique instance identifier (auto-generated if None).
        """
        self._redis = redis_client
        self._channel = channel
        self.instance_id = instance_id or f"ai-instance-{id(self)}"
        self._pubsub = None
        self._invalidation_handlers: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start listening for invalidation events."""
        if self._redis is None:
            return

        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self._channel)
            self._running = True
            self._task = asyncio.create_task(self._listen_loop())
            logger.info("Cache coordinator started (instance: %s)", self.instance_id)
        except Exception as e:
            logger.warning("Cache coordinator start error: %s", e)

    async def stop(self) -> None:
        """Stop listening for invalidation events."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Cache coordinator task cancelled during stop")
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.close()
            except Exception as e:
                logger.warning("Error closing Redis pubsub: %s", e)

    def on_invalidation(self, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a handler for invalidation events.

        Args:
            handler: Async callable receiving invalidation event data.
        """
        self._invalidation_handlers.append(handler)

    async def broadcast_invalidation(self, cache_type: str, key: str,
                                      reason: str = "update") -> bool:
        """Broadcast an invalidation event to all instances.

        Args:
            cache_type: Type of cache ('model', 'feature', 'prediction').
            key: Key that was invalidated.
            reason: Reason for invalidation.

        Returns:
            True if broadcast was successful.
        """
        event = {
            "cache_type": cache_type,
            "key": key,
            "reason": reason,
            "source_instance": self.instance_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            await self._redis.publish(self._channel, json.dumps(event))
            return True
        except Exception as e:
            logger.warning("Invalidation broadcast error: %s", e)
            return False

    async def _listen_loop(self) -> None:
        """Listen for invalidation events from other instances."""
        if not self._pubsub:
            return
        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                if message["type"] == "message":
                    try:
                        event = json.loads(message["data"])
                        # Skip own invalidations
                        if event.get("source_instance") == self.instance_id:
                            continue
                        for handler in self._invalidation_handlers:
                            try:
                                await handler(event)
                            except Exception as e:
                                logger.error("Invalidation handler error: %s", e)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Invalid invalidation event received")
        except asyncio.CancelledError:
            logger.debug("Cache coordinator listen cancelled")
        except Exception as e:
            logger.error("Cache coordinator listen error: %s", e)
