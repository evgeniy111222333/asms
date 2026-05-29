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

from acms.kafka.topics import Topic, SCHEMAS, validate_message, TopicManager
from acms.kafka.producer import ProducerConfig, KafkaProducer, _MockProducer
from acms.kafka.consumer import ConsumerConfig, KafkaConsumer
from acms.kafka.router import KeyRouter

__all__ = [
    "Topic",
    "SCHEMAS",
    "validate_message",
    "TopicManager",
    "ProducerConfig",
    "KafkaProducer",
    "_MockProducer",
    "ConsumerConfig",
    "KafkaConsumer",
    "KeyRouter",
]
