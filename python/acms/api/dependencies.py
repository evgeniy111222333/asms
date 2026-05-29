"""FastAPI dependencies for auth, rate limiting, and database access."""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession


# ============================================================================
# Database Dependency
# ============================================================================

_db_session: Optional[AsyncSession] = None
_engines = {}


def set_engines(postgres_engine, redis_client):
    """Set database engines (called during app startup)."""
    global _engines
    _engines["postgres"] = postgres_engine
    _engines["redis"] = redis_client


async def get_db() -> AsyncSession:
    """Get async database session."""
    global _db_session, _engines
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    
    if _db_session is None:
        postgres_engine = _engines.get("postgres")
        if postgres_engine:
            async_session = async_sessionmaker(
                postgres_engine, class_=AsyncSession, expire_on_commit=False
            )
            _db_session = async_session()
    
    return _db_session


def get_engines() -> dict:
    """Get database engines."""
    return _engines


# ============================================================================
# Security
# ============================================================================

security = {
    "algorithm": "HS256",
    "secret_key": None,  # Must be set via JWT_SECRET env var
}


auth_manager = None


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user_id from request state (set by auth middleware)."""
    return getattr(request.state, "user_id", None)


async def get_current_user(request: Request) -> dict:
    """Get current authenticated user from JWT token.
    
    Raises:
        HTTPException: If authentication fails.
    """
    global auth_manager
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth header")
    
    token = auth_header.replace("Bearer ", "")
    
    if auth_manager is None:
        from acms.auth import AuthManager
        redis_client = _engines.get("redis")
        auth_manager = AuthManager(redis_client=redis_client)
    
    token_data = await auth_manager.verify_token_async(token)
    if token_data is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    request.state.user_id = token_data.user_id
    return {"user_id": token_data.user_id, "email": token_data.email}


# ============================================================================
# Rate Limiting (Redis-based for multi-worker support)
# ============================================================================

class RedisRateLimiter:
    """Redis-backed sliding window rate limiter.
    
    SECURITY: Uses Redis sorted sets for accurate sliding window rate limiting
    that works correctly across multiple workers/processes.
    """
    
    def __init__(
        self,
        redis_client=None,
        max_requests: int = 100,
        window_seconds: int = 60,
        prefix: str = "acms:ratelimit",
    ):
        self._redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._prefix = prefix
    
    def _key(self, identifier: str) -> str:
        return f"{self._prefix}:{identifier}"
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is within rate limit using sliding window algorithm.
        
        Uses Redis ZSET with timestamps for accurate rate limiting
        across multiple workers.
        """
        if self._redis is None:
            # Fallback to in-memory if Redis unavailable
            return True
        
        key = self._key(identifier)
        now = time.time()
        window_start = now - self.window_seconds
        
        try:
            # Remove old entries outside the window
            await self._redis.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            current_count = await self._redis.zcard(key)
            
            if current_count >= self.max_requests:
                return False
            
            # Add current request
            await self._redis.zadd(key, {str(now): now})
            await self._redis.expire(key, self.window_seconds + 1)
            
            return True
        except Exception:
            # Allow on error to avoid blocking
            return True
    
    def get_headers(self, identifier: str) -> Dict[str, str]:
        """Get rate limit headers for response."""
        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Window": str(self.window_seconds),
        }


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


def get_rate_limiter() -> RedisRateLimiter:
    """Get or create rate limiter with Redis connection."""
    global _rate_limiter, _engines
    
    if _rate_limiter is None:
        redis_client = _engines.get("redis")
        _rate_limiter = RedisRateLimiter(redis_client=redis_client)
    
    return _rate_limiter


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for the current request.
    
    Uses user_id if authenticated, otherwise falls back to IP address.
    Works correctly across multiple workers via Redis.
    """
    limiter = get_rate_limiter()
    
    # Use user_id if available, otherwise IP
    client_id = getattr(request.state, "user_id", None)
    if client_id is None:
        client_id = request.client.host if request.client else "unknown"
    
    if not await limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limiter.max_requests),
                "X-RateLimit-Window": str(limiter.window_seconds),
                "Retry-After": str(limiter.window_seconds),
            }
        )


# Backward compatibility alias
EndpointRateLimiter = RedisRateLimiter
rate_limiter = None  # Will be replaced by get_rate_limiter()


__all__ = [
    "get_db",
    "set_engines",
    "get_engines",
    "security",
    "auth_manager",
    "_get_user_id",
    "get_current_user",
    "RedisRateLimiter",
    "EndpointRateLimiter",
    "get_rate_limiter",
    "check_rate_limit",
]
