"""Kafka topic definitions and management."""

import logging
from enum import Enum
from typing import Dict, List, Optional

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


__all__ = [
    "Topic",
    "SCHEMAS",
    "validate_message",
    "TopicManager",
]
