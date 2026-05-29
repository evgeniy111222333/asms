"""Sliding window rate limiter using Redis."""

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


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


__all__ = ["RedisRateLimiter"]
