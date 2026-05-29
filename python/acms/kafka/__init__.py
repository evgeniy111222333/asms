"""Kafka Module - Async message bus integration with Redpanda.

Implements:
- KafkaProducer: async publish to Redpanda topics
- KafkaConsumer: async consume from topics with consumer groups
- Topic management: create, delete, list topics
- Schema registration: JSON schema for message validation
- Dead letter queue handling
- Message key routing for partition ordering

Topics:
- market-data: Real-time market data (ticks, candles, orderbooks)
- signals: Trading signals from strategy engines
- orders: Order submission and update events
- risk-events: Risk alerts and circuit breaker events
- trades: Trade execution confirmations
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable, Awaitable, Union
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Topic(str, Enum):
    """Standard ACMS Kafka topics."""
    MARKET_DATA = "market-data"
    SIGNALS = "signals"
    ORDERS = "orders"
    RISK_EVENTS = "risk-events"
    TRADES = "trades"
    DLQ = "dead-letter-queue"


# ============================================================================
# Message Schemas (JSON Schema validation)
# ============================================================================

SCHEMAS: Dict[str, Dict] = {
    Topic.MARKET_DATA: {
        "type": "object",
        "required": ["symbol", "exchange", "price", "timestamp", "data_type"],
        "properties": {
            "symbol": {"type": "string"},
            "exchange": {"type": "string"},
            "price": {"type": "number"},
            "quantity": {"type": "number"},
            "timestamp": {"type": "string"},
            "data_type": {"type": "string", "enum": ["tick", "candle", "orderbook"]},
            "metadata": {"type": "object"},
        },
    },
    Topic.SIGNALS: {
        "type": "object",
        "required": ["signal_id", "symbol", "direction", "strength", "strategy_id", "timestamp"],
        "properties": {
            "signal_id": {"type": "string"},
            "symbol": {"type": "string"},
            "direction": {"type": "string", "enum": ["long", "short", "neutral"]},
            "strength": {"type": "number", "minimum": 0, "maximum": 1},
            "strategy_id": {"type": "string"},
            "indicators": {"type": "object"},
            "timestamp": {"type": "string"},
        },
    },
    Topic.ORDERS: {
        "type": "object",
        "required": ["order_id", "symbol", "side", "order_type", "quantity", "action", "timestamp"],
        "properties": {
            "order_id": {"type": "string"},
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "order_type": {"type": "string"},
            "quantity": {"type": "number"},
            "price": {"type": "number"},
            "action": {"type": "string", "enum": ["submit", "cancel", "update"]},
            "strategy_id": {"type": "string"},
            "timestamp": {"type": "string"},
        },
    },
    Topic.RISK_EVENTS: {
        "type": "object",
        "required": ["event_type", "severity", "timestamp"],
        "properties": {
            "event_type": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
            "details": {"type": "object"},
            "timestamp": {"type": "string"},
        },
    },
    Topic.TRADES: {
        "type": "object",
        "required": ["trade_id", "order_id", "symbol", "side", "quantity", "price", "timestamp"],
        "properties": {
            "trade_id": {"type": "string"},
            "order_id": {"type": "string"},
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "quantity": {"type": "number"},
            "price": {"type": "number"},
            "commission": {"type": "number"},
            "exchange": {"type": "string"},
            "timestamp": {"type": "string"},
        },
    },
}


def validate_message(topic: str, message: Dict) -> bool:
    """Validate a message against its topic's JSON schema.

    Args:
        topic: Topic name.
        message: Message dict to validate.

    Returns:
        True if valid, False otherwise.
    """
    schema = SCHEMAS.get(topic)
    if schema is None:
        return True  # No schema to validate against

    required = schema.get("required", [])
    for field_name in required:
        if field_name not in message:
            logger.warning("Message missing required field '%s' for topic '%s'", field_name, topic)
            return False

    properties = schema.get("properties", {})
    for key, value in message.items():
        if key in properties:
            prop_schema = properties[key]
            expected_type = prop_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False
            elif expected_type == "object" and not isinstance(value, dict):
                return False

            # Check enum values
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                logger.warning("Invalid enum value '%s' for field '%s'", value, key)
                return False

            # Check numeric ranges
            if isinstance(value, (int, float)):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    return False
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    return False

    return True


# ============================================================================
# Message Key Routing
# ============================================================================

class KeyRouter:
    """Routes messages to partitions based on message keys.

    Ensures that messages for the same symbol/strategy are
    processed in order by mapping them to the same partition.
    """

    def __init__(self, num_partitions: int = 6):
        self.num_partitions = num_partitions

    def get_partition(self, key: str) -> int:
        """Get partition number for a message key.

        Args:
            key: Message key (symbol, strategy_id, etc.).

        Returns:
            Partition number.
        """
        return hash(key) % self.num_partitions

    def get_key_for_symbol(self, symbol: str) -> str:
        """Get routing key for a symbol."""
        return f"symbol:{symbol}"

    def get_key_for_strategy(self, strategy_id: str) -> str:
        """Get routing key for a strategy."""
        return f"strategy:{strategy_id}"

    def get_key_for_order(self, order_id: str) -> str:
        """Get routing key for an order."""
        return f"order:{order_id}"


# ============================================================================
# Kafka Producer
# ============================================================================

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


# ============================================================================
# Kafka Consumer
# ============================================================================

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
                pass
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception:
                pass

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
            pass
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


# ============================================================================
# Topic Management
# ============================================================================

class TopicManager:
    """Kafka topic management operations.

    Provides create, delete, list, and describe operations
    for Kafka/Redpanda topics.
    """

    def __init__(self, bootstrap_servers: List[str] = None):
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self._admin_client = None

    async def connect(self) -> None:
        """Connect to the Kafka admin client."""
        try:
            from aiokafka import AIOKafkaAdminClient
            self._admin_client = AIOKafkaAdminClient(
                bootstrap_servers=",".join(self.bootstrap_servers)
            )
            await self._admin_client.start()
        except ImportError:
            logger.warning("aiokafka not installed, topic management disabled")
        except Exception as e:
            logger.error("Failed to connect admin client: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect the admin client."""
        if self._admin_client:
            await self._admin_client.close()

    async def create_topic(self, topic_name: str, num_partitions: int = 6,
                            replication_factor: int = 1,
                            config: Optional[Dict] = None) -> bool:
        """Create a new Kafka topic.

        Args:
            topic_name: Topic name.
            num_partitions: Number of partitions.
            replication_factor: Replication factor.
            config: Optional topic configuration.

        Returns:
            True if topic was created successfully.
        """
        if not self._admin_client:
            logger.warning("Admin client not connected")
            return False

        try:
            from kafka.admin import NewTopic
            topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                topic_configs=config or {},
            )
            await self._admin_client.create_topics([topic])
            logger.info("Created topic '%s'", topic_name)
            return True
        except Exception as e:
            logger.error("Failed to create topic '%s': %s", topic_name, e)
            return False

    async def delete_topic(self, topic_name: str) -> bool:
        """Delete a Kafka topic.

        Args:
            topic_name: Topic to delete.

        Returns:
            True if deletion was successful.
        """
        if not self._admin_client:
            return False
        try:
            await self._admin_client.delete_topics([topic_name])
            logger.info("Deleted topic '%s'", topic_name)
            return True
        except Exception as e:
            logger.error("Failed to delete topic '%s': %s", topic_name, e)
            return False

    async def list_topics(self) -> List[str]:
        """List all Kafka topics.

        Returns:
            List of topic names.
        """
        if not self._admin_client:
            return []
        try:
            topics = await self._admin_client.list_topics()
            return [t.topic for t in topics if not t.topic.startswith("__")]
        except Exception as e:
            logger.error("Failed to list topics: %s", e)
            return []

    async def ensure_default_topics(self) -> Dict[str, bool]:
        """Create all default ACMS topics if they don't exist.

        Returns:
            Dict mapping topic name to creation success.
        """
        results = {}
        existing = await self.list_topics()

        for topic in Topic:
            if topic.value not in existing:
                success = await self.create_topic(
                    topic.value, num_partitions=6, replication_factor=1
                )
                results[topic.value] = success
            else:
                results[topic.value] = True  # Already exists

        return results


# ============================================================================
# Mock Producer (for when aiokafka is not available)
# ============================================================================

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
