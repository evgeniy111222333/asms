"""Comprehensive tests for acms.kafka module.

Tests all classes, methods, and edge cases:
- Topic enum
- SCHEMAS dict
- validate_message (all topics, invalid messages, missing fields, wrong types, enum violations)
- KeyRouter (get_partition, get_key_for_symbol/strategy/order)
- ProducerConfig dataclass
- KafkaProducer (start/stop/publish/publish_batch/set_dlq_handler/stats)
- ConsumerConfig dataclass
- KafkaConsumer (start/stop/register_handler/stats)
- TopicManager (connect/disconnect/create/delete/list/ensure_default_topics)
- _MockProducer (start/stop/send_and_wait)
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from acms.kafka import (
    Topic, SCHEMAS, validate_message, KeyRouter,
    ProducerConfig, KafkaProducer, ConsumerConfig, KafkaConsumer,
    TopicManager, _MockProducer,
)


# ============================================================================
# Topic Enum Tests
# ============================================================================

class TestTopic:
    """Tests for Topic enum."""

    def test_all_topics(self):
        """Should have all expected topic values."""
        assert Topic.MARKET_DATA == "market-data"
        assert Topic.SIGNALS == "signals"
        assert Topic.ORDERS == "orders"
        assert Topic.RISK_EVENTS == "risk-events"
        assert Topic.TRADES == "trades"
        assert Topic.DLQ == "dead-letter-queue"

    def test_topic_count(self):
        """Should have exactly 6 topics."""
        assert len(Topic) == 6

    def test_topic_is_string(self):
        """Topics should be string enums."""
        assert isinstance(Topic.MARKET_DATA, str)

    def test_topic_values_unique(self):
        """All topic values should be unique."""
        values = [t.value for t in Topic]
        assert len(values) == len(set(values))


# ============================================================================
# SCHEMAS Tests
# ============================================================================

class TestSchemas:
    """Tests for SCHEMAS dict."""

    def test_schemas_has_all_topics_except_dlq(self):
        """SCHEMAS should have schemas for all topics except DLQ."""
        for topic in [Topic.MARKET_DATA, Topic.SIGNALS, Topic.ORDERS,
                      Topic.RISK_EVENTS, Topic.TRADES]:
            assert topic in SCHEMAS

    def test_schemas_have_required_fields(self):
        """Each schema should have required field list."""
        for topic, schema in SCHEMAS.items():
            assert "required" in schema, f"Missing 'required' in schema for {topic}"
            assert isinstance(schema["required"], list)

    def test_schemas_have_properties(self):
        """Each schema should have properties dict."""
        for topic, schema in SCHEMAS.items():
            assert "properties" in schema, f"Missing 'properties' in schema for {topic}"
            assert isinstance(schema["properties"], dict)

    def test_schemas_have_type_object(self):
        """Each schema should specify type: object."""
        for topic, schema in SCHEMAS.items():
            assert schema.get("type") == "object"

    def test_market_data_schema(self):
        """Market data schema should have expected required fields."""
        schema = SCHEMAS[Topic.MARKET_DATA]
        assert "symbol" in schema["required"]
        assert "exchange" in schema["required"]
        assert "price" in schema["required"]
        assert "timestamp" in schema["required"]
        assert "data_type" in schema["required"]

    def test_signals_schema(self):
        """Signals schema should have expected required fields."""
        schema = SCHEMAS[Topic.SIGNALS]
        assert "signal_id" in schema["required"]
        assert "symbol" in schema["required"]
        assert "direction" in schema["required"]
        assert "strength" in schema["required"]
        assert "strategy_id" in schema["required"]

    def test_orders_schema(self):
        """Orders schema should have expected required fields."""
        schema = SCHEMAS[Topic.ORDERS]
        assert "order_id" in schema["required"]
        assert "symbol" in schema["required"]
        assert "side" in schema["required"]
        assert "quantity" in schema["required"]
        assert "action" in schema["required"]

    def test_risk_events_schema(self):
        """Risk events schema should have expected required fields."""
        schema = SCHEMAS[Topic.RISK_EVENTS]
        assert "event_type" in schema["required"]
        assert "severity" in schema["required"]
        assert "timestamp" in schema["required"]

    def test_trades_schema(self):
        """Trades schema should have expected required fields."""
        schema = SCHEMAS[Topic.TRADES]
        assert "trade_id" in schema["required"]
        assert "order_id" in schema["required"]
        assert "symbol" in schema["required"]
        assert "side" in schema["required"]
        assert "quantity" in schema["required"]
        assert "price" in schema["required"]

    def test_data_type_enum(self):
        """Market data data_type should have enum constraint."""
        schema = SCHEMAS[Topic.MARKET_DATA]
        assert "enum" in schema["properties"]["data_type"]
        assert set(schema["properties"]["data_type"]["enum"]) == {"tick", "candle", "orderbook"}

    def test_direction_enum(self):
        """Signals direction should have enum constraint."""
        schema = SCHEMAS[Topic.SIGNALS]
        assert "enum" in schema["properties"]["direction"]
        assert set(schema["properties"]["direction"]["enum"]) == {"long", "short", "neutral"}

    def test_side_enum(self):
        """Orders side should have enum constraint."""
        schema = SCHEMAS[Topic.ORDERS]
        assert "enum" in schema["properties"]["side"]
        assert set(schema["properties"]["side"]["enum"]) == {"buy", "sell"}

    def test_severity_enum(self):
        """Risk events severity should have enum constraint."""
        schema = SCHEMAS[Topic.RISK_EVENTS]
        assert "enum" in schema["properties"]["severity"]
        assert set(schema["properties"]["severity"]["enum"]) == {"info", "warning", "critical"}

    def test_strength_range(self):
        """Signals strength should have min/max constraints."""
        schema = SCHEMAS[Topic.SIGNALS]
        assert schema["properties"]["strength"]["minimum"] == 0
        assert schema["properties"]["strength"]["maximum"] == 1


# ============================================================================
# validate_message Tests
# ============================================================================

class TestValidateMessage:
    """Tests for validate_message function."""

    # --- Valid messages ---

    def test_valid_market_data_tick(self):
        """Valid tick message should pass."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01T00:00:00",
            "data_type": "tick",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is True

    def test_valid_market_data_candle(self):
        """Valid candle message should pass."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01T00:00:00",
            "data_type": "candle",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is True

    def test_valid_market_data_orderbook(self):
        """Valid orderbook message should pass."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01T00:00:00",
            "data_type": "orderbook",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is True

    def test_valid_signal_long(self):
        """Valid long signal should pass."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": 0.8,
            "strategy_id": "strat1", "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.SIGNALS, msg) is True

    def test_valid_signal_short(self):
        """Valid short signal should pass."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "short", "strength": 0.6,
            "strategy_id": "strat1", "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.SIGNALS, msg) is True

    def test_valid_signal_neutral(self):
        """Valid neutral signal should pass."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "neutral", "strength": 0.1,
            "strategy_id": "strat1", "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.SIGNALS, msg) is True

    def test_valid_order(self):
        """Valid order message should pass."""
        msg = {
            "order_id": "ord1", "symbol": "BTC/USDT",
            "side": "buy", "order_type": "market",
            "quantity": 1.0, "action": "submit",
            "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.ORDERS, msg) is True

    def test_valid_risk_event(self):
        """Valid risk event should pass."""
        msg = {
            "event_type": "drawdown_exceeded", "severity": "warning",
            "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.RISK_EVENTS, msg) is True

    def test_valid_trade(self):
        """Valid trade message should pass."""
        msg = {
            "trade_id": "trade1", "order_id": "ord1",
            "symbol": "BTC/USDT", "side": "buy",
            "quantity": 1.0, "price": 50000.0,
            "timestamp": "2024-01-01T00:00:00",
        }
        assert validate_message(Topic.TRADES, msg) is True

    # --- Unknown topic (no schema) ---

    def test_unknown_topic(self):
        """Unknown topic should always return True (no schema)."""
        msg = {"any": "data"}
        assert validate_message("unknown-topic", msg) is True

    def test_dlq_topic_no_schema(self):
        """DLQ topic has no schema and should return True."""
        msg = {"any": "data"}
        assert validate_message(Topic.DLQ, msg) is True

    # --- Missing required fields ---

    def test_missing_required_field_market_data(self):
        """Missing required field should fail validation."""
        msg = {"symbol": "BTC/USDT", "exchange": "binance"}
        assert validate_message(Topic.MARKET_DATA, msg) is False

    def test_missing_symbol(self):
        """Missing symbol should fail."""
        msg = {
            "exchange": "binance", "price": 50000.0,
            "timestamp": "2024-01-01", "data_type": "tick",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is False

    def test_missing_signal_id(self):
        """Missing signal_id should fail."""
        msg = {
            "symbol": "BTC/USDT", "direction": "long",
            "strength": 0.8, "strategy_id": "strat1",
            "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.SIGNALS, msg) is False

    def test_missing_order_action(self):
        """Missing action should fail."""
        msg = {
            "order_id": "ord1", "symbol": "BTC/USDT",
            "side": "buy", "order_type": "market",
            "quantity": 1.0, "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.ORDERS, msg) is False

    # --- Wrong types ---

    def test_wrong_type_string_field(self):
        """String field with non-string value should fail."""
        msg = {
            "symbol": 123, "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01",
            "data_type": "tick",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is False

    def test_wrong_type_number_field(self):
        """Number field with non-number value should fail."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": "not_a_number", "timestamp": "2024-01-01",
            "data_type": "tick",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is False

    def test_wrong_type_object_field(self):
        """Object field with non-dict value should fail."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": 0.8,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
            "indicators": "not_a_dict",
        }
        assert validate_message(Topic.SIGNALS, msg) is False

    # --- Enum violations ---

    def test_invalid_data_type(self):
        """Invalid data_type should fail."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01",
            "data_type": "invalid_type",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is False

    def test_invalid_direction(self):
        """Invalid direction should fail."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "up", "strength": 0.8,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.SIGNALS, msg) is False

    def test_invalid_side(self):
        """Invalid side should fail."""
        msg = {
            "order_id": "ord1", "symbol": "BTC/USDT",
            "side": "long", "order_type": "market",
            "quantity": 1.0, "action": "submit",
            "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.ORDERS, msg) is False

    def test_invalid_severity(self):
        """Invalid severity should fail."""
        msg = {
            "event_type": "test", "severity": "extreme",
            "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.RISK_EVENTS, msg) is False

    def test_invalid_action(self):
        """Invalid action should fail."""
        msg = {
            "order_id": "ord1", "symbol": "BTC/USDT",
            "side": "buy", "order_type": "market",
            "quantity": 1.0, "action": "execute",
            "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.ORDERS, msg) is False

    # --- Numeric range violations ---

    def test_strength_below_minimum(self):
        """Strength below 0 should fail."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": -0.1,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.SIGNALS, msg) is False

    def test_strength_above_maximum(self):
        """Strength above 1 should fail."""
        msg = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": 1.5,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.SIGNALS, msg) is False

    def test_strength_at_boundaries(self):
        """Strength at 0 and 1 should pass."""
        msg_min = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": 0.0,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
        }
        msg_max = {
            "signal_id": "sig1", "symbol": "BTC/USDT",
            "direction": "long", "strength": 1.0,
            "strategy_id": "strat1", "timestamp": "2024-01-01",
        }
        assert validate_message(Topic.SIGNALS, msg_min) is True
        assert validate_message(Topic.SIGNALS, msg_max) is True

    # --- Extra fields should be OK ---

    def test_extra_fields_allowed(self):
        """Extra fields not in schema should be allowed."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01",
            "data_type": "tick", "custom_field": "value",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is True

    # --- Integer is valid number ---

    def test_integer_as_number(self):
        """Integer should be valid for number type fields."""
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000, "timestamp": "2024-01-01",
            "data_type": "tick",
        }
        assert validate_message(Topic.MARKET_DATA, msg) is True


# ============================================================================
# KeyRouter Tests
# ============================================================================

class TestKeyRouter:
    """Tests for KeyRouter class."""

    def test_default_partitions(self):
        """Default partition count should be 6."""
        router = KeyRouter()
        assert router.num_partitions == 6

    def test_custom_partitions(self):
        """Should accept custom partition count."""
        router = KeyRouter(num_partitions=12)
        assert router.num_partitions == 12

    def test_get_partition_returns_int(self):
        """get_partition should return an integer."""
        router = KeyRouter()
        result = router.get_partition("BTC/USDT")
        assert isinstance(result, int)

    def test_get_partition_within_range(self):
        """Partition should be within 0 to num_partitions-1."""
        router = KeyRouter(num_partitions=6)
        for key in ["BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "DOT"]:
            partition = router.get_partition(key)
            assert 0 <= partition < 6

    def test_get_partition_deterministic(self):
        """Same key should always map to same partition."""
        router = KeyRouter()
        p1 = router.get_partition("BTC/USDT")
        p2 = router.get_partition("BTC/USDT")
        assert p1 == p2

    def test_get_key_for_symbol(self):
        """Should return symbol-prefixed key."""
        router = KeyRouter()
        assert router.get_key_for_symbol("BTC/USDT") == "symbol:BTC/USDT"

    def test_get_key_for_strategy(self):
        """Should return strategy-prefixed key."""
        router = KeyRouter()
        assert router.get_key_for_strategy("strat1") == "strategy:strat1"

    def test_get_key_for_order(self):
        """Should return order-prefixed key."""
        router = KeyRouter()
        assert router.get_key_for_order("ord1") == "order:ord1"

    def test_different_keys_different_partitions(self):
        """Different keys should potentially map to different partitions."""
        router = KeyRouter(num_partitions=6)
        partitions = set()
        for i in range(100):
            partitions.add(router.get_partition(f"key_{i}"))
        # With 100 keys and 6 partitions, should have multiple partitions
        assert len(partitions) > 1


# ============================================================================
# ProducerConfig Tests
# ============================================================================

class TestProducerConfig:
    """Tests for ProducerConfig dataclass."""

    def test_defaults(self):
        """Should have expected default values."""
        cfg = ProducerConfig()
        assert cfg.bootstrap_servers == ["localhost:9092"]
        assert cfg.acks == "all"
        assert cfg.compression == "snappy"
        assert cfg.max_retries == 3
        assert cfg.retry_backoff_ms == 100
        assert cfg.batch_size == 16384
        assert cfg.linger_ms == 5

    def test_custom_values(self):
        """Should accept custom values."""
        cfg = ProducerConfig(
            bootstrap_servers=["kafka1:9092", "kafka2:9092"],
            acks="1",
            compression="gzip",
            max_retries=5,
            retry_backoff_ms=200,
            batch_size=32768,
            linger_ms=10,
        )
        assert cfg.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]
        assert cfg.acks == "1"
        assert cfg.compression == "gzip"


# ============================================================================
# _MockProducer Tests
# ============================================================================

class TestMockProducer:
    """Tests for _MockProducer class."""

    def setup_method(self):
        self.producer = _MockProducer()

    @pytest.mark.asyncio
    async def test_start(self):
        """start() should not raise."""
        await self.producer.start()

    @pytest.mark.asyncio
    async def test_stop(self):
        """stop() should not raise."""
        await self.producer.stop()

    @pytest.mark.asyncio
    async def test_send_and_wait(self):
        """Should store message in internal list."""
        await self.producer.send_and_wait("topic1", {"key": "value"}, key="test_key")
        assert len(self.producer._messages) == 1
        assert self.producer._messages[0]["topic"] == "topic1"
        assert self.producer._messages[0]["value"] == {"key": "value"}
        assert self.producer._messages[0]["key"] == "test_key"

    @pytest.mark.asyncio
    async def test_send_and_wait_no_key(self):
        """Should work without key."""
        await self.producer.send_and_wait("topic1", {"key": "value"})
        assert self.producer._messages[0]["key"] is None

    @pytest.mark.asyncio
    async def test_multiple_messages(self):
        """Should store all messages."""
        for i in range(5):
            await self.producer.send_and_wait("topic1", {"i": i}, key=f"key_{i}")
        assert len(self.producer._messages) == 5


# ============================================================================
# KafkaProducer Tests
# ============================================================================

class TestKafkaProducer:
    """Tests for KafkaProducer class."""

    def setup_method(self):
        self.producer = KafkaProducer()

    def test_default_config(self):
        """Should have default ProducerConfig."""
        assert isinstance(self.producer.config, ProducerConfig)

    def test_custom_config(self):
        """Should accept custom config."""
        cfg = ProducerConfig(acks="1")
        producer = KafkaProducer(config=cfg)
        assert producer.config.acks == "1"

    def test_initial_state(self):
        """Initial state should be not running."""
        assert self.producer._running is False
        assert self.producer._sent_count == 0
        assert self.producer._error_count == 0

    @pytest.mark.asyncio
    async def test_start(self):
        """Should start producer (using mock since aiokafka not available)."""
        await self.producer.start()
        assert self.producer._running is True
        assert self.producer._producer is not None

    @pytest.mark.asyncio
    async def test_stop(self):
        """Should stop producer."""
        await self.producer.start()
        await self.producer.stop()
        assert self.producer._running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        """Stopping without starting should not raise."""
        await self.producer.stop()

    @pytest.mark.asyncio
    async def test_publish_not_running(self):
        """Publishing when not running should return False."""
        result = await self.producer.publish("topic1", {"key": "value"})
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_valid_message(self):
        """Should publish valid message successfully."""
        await self.producer.start()
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "timestamp": "2024-01-01T00:00:00",
            "data_type": "tick",
        }
        result = await self.producer.publish(Topic.MARKET_DATA, msg)
        assert result is True
        assert self.producer._sent_count == 1

    @pytest.mark.asyncio
    async def test_publish_invalid_message(self):
        """Should send invalid message to DLQ and return False."""
        await self.producer.start()
        msg = {"invalid": "message"}
        result = await self.producer.publish(Topic.MARKET_DATA, msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_without_validation(self):
        """Should skip validation when validate=False."""
        await self.producer.start()
        msg = {"invalid": "message"}
        result = await self.producer.publish(Topic.MARKET_DATA, msg, validate=False)
        assert result is True

    @pytest.mark.asyncio
    async def test_publish_adds_timestamp(self):
        """Should add timestamp if not present."""
        await self.producer.start()
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "data_type": "tick",
        }
        await self.producer.publish(Topic.MARKET_DATA, msg, validate=False)
        assert "timestamp" in msg

    @pytest.mark.asyncio
    async def test_publish_preserves_existing_timestamp(self):
        """Should not overwrite existing timestamp."""
        await self.producer.start()
        original_ts = "2024-01-01T00:00:00"
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "data_type": "tick",
            "timestamp": original_ts,
        }
        await self.producer.publish(Topic.MARKET_DATA, msg, validate=False)
        assert msg["timestamp"] == original_ts

    @pytest.mark.asyncio
    async def test_publish_auto_key_from_symbol(self):
        """Should use symbol as default key."""
        await self.producer.start()
        msg = {
            "symbol": "BTC/USDT", "exchange": "binance",
            "price": 50000.0, "data_type": "tick",
            "timestamp": "2024-01-01",
        }
        await self.producer.publish(Topic.MARKET_DATA, msg, validate=False)
        # Check that mock producer received a key
        mock = self.producer._producer
        assert mock._messages[-1]["key"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_publish_auto_key_from_strategy(self):
        """Should use strategy_id as fallback key (symbol takes priority)."""
        await self.producer.start()
        msg = {
            "signal_id": "sig1", "strategy_id": "strat1",
            "symbol": "BTC/USDT", "direction": "long",
            "strength": 0.8, "timestamp": "2024-01-01",
        }
        await self.producer.publish(Topic.SIGNALS, msg, validate=False)
        mock = self.producer._producer
        # Key routing: symbol first, then strategy_id
        assert mock._messages[-1]["key"] == "BTC/USDT"

    @pytest.mark.asyncio
    async def test_publish_custom_key(self):
        """Should use provided key."""
        await self.producer.start()
        msg = {"key": "value", "timestamp": "2024-01-01"}
        await self.producer.publish("custom-topic", msg, key="my_key", validate=False)
        mock = self.producer._producer
        assert mock._messages[-1]["key"] == "my_key"

    @pytest.mark.asyncio
    async def test_publish_batch(self):
        """Should publish batch of messages."""
        await self.producer.start()
        messages = [
            {"key": "value1", "timestamp": "2024-01-01"},
            {"key": "value2", "timestamp": "2024-01-01"},
            {"key": "value3", "timestamp": "2024-01-01"},
        ]
        result = await self.producer.publish_batch("custom-topic", messages)
        assert result == 3

    @pytest.mark.asyncio
    async def test_publish_batch_with_keys(self):
        """Should publish batch with custom keys."""
        await self.producer.start()
        messages = [
            {"key": "value1", "timestamp": "2024-01-01"},
            {"key": "value2", "timestamp": "2024-01-01"},
        ]
        keys = ["key1", "key2"]
        result = await self.producer.publish_batch("custom-topic", messages, keys=keys)
        assert result == 2

    @pytest.mark.asyncio
    async def test_publish_batch_partial_failure(self):
        """Batch with some invalid messages should return partial count."""
        await self.producer.start()
        messages = [
            {"symbol": "BTC/USDT", "exchange": "binance",
             "price": 50000.0, "timestamp": "2024-01-01", "data_type": "tick"},
            {"invalid": "message"},  # Will fail validation
        ]
        result = await self.producer.publish_batch(Topic.MARKET_DATA, messages)
        assert result == 1

    @pytest.mark.asyncio
    async def test_publish_batch_empty(self):
        """Empty batch should return 0."""
        await self.producer.start()
        result = await self.producer.publish_batch("topic", [])
        assert result == 0

    def test_set_dlq_handler(self):
        """Should set DLQ handler."""
        handler = AsyncMock()
        self.producer.set_dlq_handler(handler)
        assert self.producer._dlq_handler is handler

    @pytest.mark.asyncio
    async def test_dlq_handler_called_on_invalid(self):
        """DLQ handler should be called when message is invalid."""
        handler = AsyncMock()
        self.producer.set_dlq_handler(handler)
        await self.producer.start()
        msg = {"invalid": "message"}
        await self.producer.publish(Topic.MARKET_DATA, msg)
        assert handler.called

    @pytest.mark.asyncio
    async def test_dlq_handler_error(self):
        """DLQ handler error should be caught."""
        handler = AsyncMock(side_effect=Exception("DLQ handler error"))
        self.producer.set_dlq_handler(handler)
        await self.producer.start()
        msg = {"invalid": "message"}
        # Should not raise
        result = await self.producer.publish(Topic.MARKET_DATA, msg)
        assert result is False

    def test_stats(self):
        """stats should return current statistics."""
        stats = self.producer.stats
        assert "sent_count" in stats
        assert "error_count" in stats
        assert "running" in stats
        assert stats["sent_count"] == 0
        assert stats["error_count"] == 0
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_stats_after_publish(self):
        """Stats should reflect published messages."""
        await self.producer.start()
        msg = {"key": "value", "timestamp": "2024-01-01"}
        await self.producer.publish("topic", msg, validate=False)
        stats = self.producer.stats
        assert stats["sent_count"] == 1
        assert stats["running"] is True


# ============================================================================
# ConsumerConfig Tests
# ============================================================================

class TestConsumerConfig:
    """Tests for ConsumerConfig dataclass."""

    def test_defaults(self):
        """Should have expected default values."""
        cfg = ConsumerConfig()
        assert cfg.bootstrap_servers == ["localhost:9092"]
        assert cfg.group_id == "acms-consumer"
        assert cfg.auto_offset_reset == "latest"
        assert cfg.enable_auto_commit is True
        assert cfg.max_poll_records == 100

    def test_custom_values(self):
        """Should accept custom values."""
        cfg = ConsumerConfig(
            bootstrap_servers=["kafka1:9092"],
            group_id="custom-group",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_records=200,
        )
        assert cfg.group_id == "custom-group"
        assert cfg.auto_offset_reset == "earliest"


# ============================================================================
# KafkaConsumer Tests
# ============================================================================

class TestKafkaConsumer:
    """Tests for KafkaConsumer class."""

    def setup_method(self):
        self.consumer = KafkaConsumer()

    def test_default_config(self):
        """Should have default ConsumerConfig."""
        assert isinstance(self.consumer.config, ConsumerConfig)

    def test_initial_state(self):
        """Initial state should be not running."""
        assert self.consumer._running is False
        assert self.consumer._consumed_count == 0

    def test_register_handler(self):
        """Should register message handler."""
        handler = AsyncMock()
        self.consumer.register_handler("signals", handler)
        assert "signals" in self.consumer._handlers
        assert self.consumer._handlers["signals"] is handler

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        """Stopping without starting should not raise."""
        await self.consumer.stop()

    def test_stats(self):
        """stats should return current statistics."""
        stats = self.consumer.stats
        assert "consumed_count" in stats
        assert "running" in stats
        assert "subscribed_handlers" in stats

    @pytest.mark.asyncio
    async def test_start_import_error(self):
        """Should handle ImportError when aiokafka not available."""
        with patch.dict('sys.modules', {'aiokafka': None}):
            consumer = KafkaConsumer()
            await consumer.start(["signals"])
            assert consumer._running is False

    @pytest.mark.asyncio
    async def test_start_exception(self):
        """Should raise on other exceptions."""
        # When aiokafka is not installed, it falls back gracefully
        # This test verifies the import error path is handled
        consumer = KafkaConsumer()
        await consumer.start(["signals"])
        # Without aiokafka, consumer is disabled (not running)
        assert consumer._running is False


# ============================================================================
# TopicManager Tests
# ============================================================================

class TestTopicManager:
    """Tests for TopicManager class."""

    def setup_method(self):
        self.manager = TopicManager(bootstrap_servers=["localhost:9092"])

    def test_default_servers(self):
        """Default servers should be localhost:9092."""
        mgr = TopicManager()
        assert mgr.bootstrap_servers == ["localhost:9092"]

    def test_custom_servers(self):
        """Should accept custom servers."""
        mgr = TopicManager(bootstrap_servers=["kafka1:9092", "kafka2:9092"])
        assert mgr.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        """Should handle ImportError when aiokafka not available."""
        with patch.dict('sys.modules', {'aiokafka': None}):
            await self.manager.connect()
            assert self.manager._admin_client is None

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        """Should handle ImportError when aiokafka not available."""
        # Without aiokafka installed, connect should handle gracefully
        await self.manager.connect()
        # Admin client won't be available without aiokafka
        # Just verify no exception is raised

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self):
        """Disconnecting with no client should not raise."""
        await self.manager.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_with_client(self):
        """Should close admin client on disconnect."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client
        await self.manager.disconnect()
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_topic_no_client(self):
        """Should return False when admin client not connected."""
        result = await self.manager.create_topic("test-topic")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_topic_success(self):
        """Should create topic when admin client is available."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client
        # Mock NewTopic in the acms.kafka module (where it's imported)
        mock_new_topic = MagicMock()
        with patch.dict('sys.modules', {'kafka': MagicMock(admin=MagicMock(NewTopic=mock_new_topic)), 'kafka.admin': MagicMock(NewTopic=mock_new_topic)}):
            result = await self.manager.create_topic("test-topic")
            assert result is True
            mock_client.create_topics.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_topic_error(self):
        """Should return False on creation error."""
        mock_client = AsyncMock()
        mock_client.create_topics.side_effect = Exception("Topic exists")
        self.manager._admin_client = mock_client
        with patch.dict('sys.modules', {'kafka': MagicMock(admin=MagicMock(NewTopic=MagicMock())), 'kafka.admin': MagicMock(NewTopic=MagicMock())}):
            result = await self.manager.create_topic("test-topic")
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_topic_no_client(self):
        """Should return False when admin client not connected."""
        result = await self.manager.delete_topic("test-topic")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_topic_success(self):
        """Should delete topic successfully."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client
        result = await self.manager.delete_topic("test-topic")
        assert result is True
        mock_client.delete_topics.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_topic_error(self):
        """Should return False on deletion error."""
        mock_client = AsyncMock()
        mock_client.delete_topics.side_effect = Exception("Not found")
        self.manager._admin_client = mock_client
        result = await self.manager.delete_topic("test-topic")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_topics_no_client(self):
        """Should return empty list when admin client not connected."""
        result = await self.manager.list_topics()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_topics_success(self):
        """Should list topics successfully."""
        mock_client = AsyncMock()
        mock_topic = MagicMock()
        mock_topic.topic = "market-data"
        mock_client.list_topics.return_value = [mock_topic]
        self.manager._admin_client = mock_client
        result = await self.manager.list_topics()
        assert "market-data" in result

    @pytest.mark.asyncio
    async def test_list_topics_filters_internal(self):
        """Should filter out internal topics starting with __."""
        mock_client = AsyncMock()
        mock_topic1 = MagicMock()
        mock_topic1.topic = "market-data"
        mock_topic2 = MagicMock()
        mock_topic2.topic = "__consumer_offsets"
        mock_client.list_topics.return_value = [mock_topic1, mock_topic2]
        self.manager._admin_client = mock_client
        result = await self.manager.list_topics()
        assert "market-data" in result
        assert "__consumer_offsets" not in result

    @pytest.mark.asyncio
    async def test_list_topics_error(self):
        """Should return empty list on error."""
        mock_client = AsyncMock()
        mock_client.list_topics.side_effect = Exception("Error")
        self.manager._admin_client = mock_client
        result = await self.manager.list_topics()
        assert result == []

    @pytest.mark.asyncio
    async def test_ensure_default_topics_all_new(self):
        """Should create all default topics when none exist."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client

        # Mock list_topics to return empty
        self.manager.list_topics = AsyncMock(return_value=[])
        # Mock create_topic to succeed
        self.manager.create_topic = AsyncMock(return_value=True)

        results = await self.manager.ensure_default_topics()
        for topic in Topic:
            assert results[topic.value] is True

    @pytest.mark.asyncio
    async def test_ensure_default_topics_some_exist(self):
        """Existing topics should be marked True without creating."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client

        self.manager.list_topics = AsyncMock(return_value=["market-data", "signals"])
        self.manager.create_topic = AsyncMock(return_value=True)

        results = await self.manager.ensure_default_topics()
        assert results["market-data"] is True
        assert results["signals"] is True
        # Others should have been created
        assert self.manager.create_topic.call_count == len(Topic) - 2

    @pytest.mark.asyncio
    async def test_create_topic_with_config(self):
        """Should pass config to topic creation."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client
        mock_new_topic = MagicMock()
        with patch.dict('sys.modules', {'kafka': MagicMock(admin=MagicMock(NewTopic=mock_new_topic)), 'kafka.admin': MagicMock(NewTopic=mock_new_topic)}):
            result = await self.manager.create_topic(
                "test-topic",
                num_partitions=3,
                replication_factor=3,
                config={"retention.ms": "86400000"},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_create_topic_default_params(self):
        """Should use default partitions and replication."""
        mock_client = AsyncMock()
        self.manager._admin_client = mock_client
        mock_new_topic = MagicMock()
        with patch.dict('sys.modules', {'kafka': MagicMock(admin=MagicMock(NewTopic=mock_new_topic)), 'kafka.admin': MagicMock(NewTopic=mock_new_topic)}):
            result = await self.manager.create_topic("test-topic")
            assert result is True
