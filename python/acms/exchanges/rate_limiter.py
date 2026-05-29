"""Rate Limiter for Exchange API calls."""

import asyncio
import time
from typing import Optional


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


__all__ = ['RateLimiter']
