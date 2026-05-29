"""Kafka producer with schema validation and dead letter queue."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable

from acms.kafka.topics import Topic, validate_message
from acms.kafka.router import KeyRouter

logger = logging.getLogger(__name__)


@dataclass
class ProducerConfig:
    """Kafka producer configuration."""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    acks: str = "all"
    compression: str = "snappy"
    max_retries: int = 3
    retry_backoff_ms: int = 100
    batch_size: int = 16384
    linger_ms: int = 5


class _MockProducer:
    """Mock producer that logs messages instead of sending to Kafka."""

    def __init__(self):
        self._messages: List[Dict] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_and_wait(self, topic: str, value: Dict, key: Optional[str] = None):
        self._messages.append({"topic": topic, "value": value, "key": key})
        logger.debug("Mock producer: topic=%s key=%s", topic, key)


class KafkaProducer:
    """Async Kafka producer for publishing messages to Redpanda.

    Supports:
    - Message validation against topic schemas
    - Key-based partition routing
    - Automatic retries with backoff
    - Dead letter queue for failed messages
    """

    def __init__(self, config: Optional[ProducerConfig] = None):
        self.config = config or ProducerConfig()
        self._producer = None
        self._key_router = KeyRouter()
        self._dlq_handler: Optional[Callable] = None
        self._sent_count = 0
        self._error_count = 0
        self._running = False

    async def start(self) -> None:
        """Start the Kafka producer.

        Creates an aiokafka producer instance with the configured settings.
        """
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=",".join(self.config.bootstrap_servers),
                acks=self.config.acks,
                compression_type=self.config.compression,
                retry_backoff_ms=self.config.retry_backoff_ms,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            await self._producer.start()
            self._running = True
            logger.info("Kafka producer started, connected to %s", self.config.bootstrap_servers)
        except ImportError:
            logger.warning("aiokafka not installed, using mock producer")
            self._producer = _MockProducer()
            self._running = True
        except Exception as e:
            logger.error("Failed to start Kafka producer: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the Kafka producer gracefully."""
        if self._producer:
            try:
                await self._producer.stop()
            except Exception as e:
                logger.warning("Error stopping producer: %s", e)
        self._running = False

    async def publish(self, topic: str, message: Dict, key: Optional[str] = None,
                       validate: bool = True) -> bool:
        """Publish a message to a Kafka topic.

        Args:
            topic: Target topic name.
            message: Message payload dict.
            key: Optional message key for partition routing.
            validate: Whether to validate against topic schema.

        Returns:
            True if message was published successfully.
        """
        if not self._running:
            logger.error("Producer not running")
            return False

        # Validate message
        if validate and not validate_message(topic, message):
            logger.warning("Invalid message for topic '%s', sending to DLQ", topic)
            await self._send_to_dlq(topic, message, "validation_failed")
            return False

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        # Route key
        if key is None:
            key = message.get("symbol", message.get("strategy_id", ""))

        try:
            await self._producer.send_and_wait(topic, value=message, key=key)
            self._sent_count += 1
            return True
        except Exception as e:
            self._error_count += 1
            logger.error("Failed to publish to '%s': %s", topic, e)
            await self._send_to_dlq(topic, message, str(e))
            return False

    async def publish_batch(self, topic: str, messages: List[Dict],
                             keys: Optional[List[str]] = None) -> int:
        """Publish a batch of messages.

        Args:
            topic: Target topic.
            messages: List of message dicts.
            keys: Optional list of message keys.

        Returns:
            Number of successfully published messages.
        """
        success_count = 0
        for i, message in enumerate(messages):
            key = keys[i] if keys and i < len(keys) else None
            if await self.publish(topic, message, key):
                success_count += 1
        return success_count

    async def _send_to_dlq(self, original_topic: str, message: Dict,
                            reason: str) -> None:
        """Send a failed message to the dead letter queue.

        Args:
            original_topic: The topic the message was intended for.
            message: The original message payload.
            reason: Reason for failure.
        """
        dlq_message = {
            "original_topic": original_topic,
            "original_message": message,
            "reason": reason,
            "failed_at": datetime.utcnow().isoformat(),
        }

        if self._dlq_handler:
            try:
                await self._dlq_handler(dlq_message)
            except Exception as e:
                logger.error("DLQ handler failed: %s", e)

        # Also try to send to DLQ topic
        try:
            if self._producer and self._running:
                await self._producer.send_and_wait(Topic.DLQ, value=dlq_message)
        except Exception as e:
            logger.error("Failed to send to DLQ topic: %s", e)

    def set_dlq_handler(self, handler: Callable) -> None:
        """Set a custom dead letter queue handler.

        Args:
            handler: Async callable receiving DLQ messages.
        """
        self._dlq_handler = handler

    @property
    def stats(self) -> Dict:
        """Get producer statistics."""
        return {
            "sent_count": self._sent_count,
            "error_count": self._error_count,
            "running": self._running,
        }


__all__ = [
    "ProducerConfig",
    "KafkaProducer",
    "_MockProducer",
]
