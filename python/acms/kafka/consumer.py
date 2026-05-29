"""Kafka consumer with handler registration and graceful shutdown."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    """Kafka consumer configuration."""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    group_id: str = "acms-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 100


class KafkaConsumer:
    """Async Kafka consumer for processing messages from Redpanda.

    Supports:
    - Consumer groups for horizontal scaling
    - Multiple topic subscription
    - Message deserialization and validation
    - Graceful shutdown
    """

    def __init__(self, config: Optional[ConsumerConfig] = None):
        self.config = config or ConsumerConfig()
        self._consumer = None
        self._handlers: Dict[str, Callable] = {}
        self._running = False
        self._consumed_count = 0
        self._task: Optional[asyncio.Task] = None

    async def start(self, topics: List[str]) -> None:
        """Start the Kafka consumer.

        Args:
            topics: List of topic names to subscribe to.
        """
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=",".join(self.config.bootstrap_servers),
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                max_poll_records=self.config.max_poll_records,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            await self._consumer.start()
            self._running = True
            self._task = asyncio.create_task(self._consume_loop())
            logger.info("Kafka consumer started, subscribed to %s", topics)
        except ImportError:
            logger.warning("aiokafka not installed, consumer disabled")
            self._running = False
        except Exception as e:
            logger.error("Failed to start Kafka consumer: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the Kafka consumer gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Kafka consumer task cancelled during stop")
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.warning("Error stopping Kafka consumer: %s", e)

    def register_handler(self, topic: str, handler: Callable[[Dict], Awaitable[None]]) -> None:
        """Register a message handler for a specific topic.

        Args:
            topic: Topic name.
            handler: Async callable receiving the message dict.
        """
        self._handlers[topic] = handler

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        if not self._consumer:
            return

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                try:
                    topic = message.topic
                    value = message.value
                    handler = self._handlers.get(topic)
                    if handler:
                        await handler(value)
                    self._consumed_count += 1
                except Exception as e:
                    logger.error("Error processing message from '%s': %s",
                                 message.topic, e)
        except asyncio.CancelledError:
            logger.debug("Kafka consume loop cancelled")
        except Exception as e:
            logger.error("Consumer loop error: %s", e)

    @property
    def stats(self) -> Dict:
        """Get consumer statistics."""
        return {
            "consumed_count": self._consumed_count,
            "running": self._running,
            "subscribed_handlers": list(self._handlers.keys()),
        }


__all__ = [
    "ConsumerConfig",
    "KafkaConsumer",
]
