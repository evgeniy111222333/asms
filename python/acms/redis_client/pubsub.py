"""PubSub manager for real-time event broadcasting."""

import json
import logging
import asyncio
from typing import Dict, List, Any, Callable, Awaitable

logger = logging.getLogger(__name__)


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
            logger.debug("PubSub listen cancelled")
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
            except Exception as e:
                logger.warning("Error closing pubsub connection: %s", e)



__all__ = ["PubSubManager"]
