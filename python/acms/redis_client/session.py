"""User session storage using Redis with TTL."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)


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


__all__ = ["SessionManager"]
