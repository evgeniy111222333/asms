"""Kafka Consumer Integration for ACMS AI Pipeline.

Implements:
- AIKafkaConsumer: Base consumer for AI training data from Redpanda
- MarketDataConsumer: Consumes trades, orderbooks, candles from Redpanda
- SignalConsumer: Consumes trading signals from signal engine
- TrainingDataBuffer: Time-based flushing buffer for training data
- Data quality filtering on consumption
- Backpressure handling with configurable thresholds
- Consumer group management
- Offset management for exactly-once processing
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Awaitable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AIConsumerConfig:
    """Configuration for AI Kafka consumers.

    Attributes:
        bootstrap_servers: Redpanda broker addresses.
        group_id: Consumer group identifier.
        auto_offset_reset: Offset reset strategy ('earliest' or 'latest').
        enable_auto_commit: Whether to auto-commit offsets.
        max_poll_records: Maximum records per poll.
        session_timeout_ms: Consumer session timeout.
        max_poll_interval_ms: Maximum time between polls.
        fetch_max_bytes: Maximum bytes per fetch.
        backpressure_threshold: Queue depth triggering backpressure.
        flush_interval_seconds: Buffer flush interval.
        buffer_max_size: Maximum buffer size before forced flush.
        quality_filter_enabled: Whether to filter low-quality data.
    """
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    group_id: str = "acms-ai-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    max_poll_interval_ms: int = 300000
    fetch_max_bytes: int = 1048576
    backpressure_threshold: int = 10000
    flush_interval_seconds: float = 5.0
    buffer_max_size: int = 10000
    quality_filter_enabled: bool = True


class DataQuality(str, Enum):
    """Data quality assessment levels."""
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"


# ============================================================================
# Data Quality Filter
# ============================================================================

class DataQualityFilter:
    """Filters incoming data based on quality checks.

    Applies configurable rules to reject or flag suspicious data:
    - Price validity (positive, non-zero, reasonable range)
    - Volume validity (non-negative, reasonable magnitude)
    - Timestamp freshness (not too old or future-dated)
    - Schema validation (required fields present)
    """

    def __init__(self, max_price_deviation_pct: float = 50.0,
                 max_age_seconds: float = 300.0,
                 min_volume: float = 0.0,
                 max_volume: float = 1e12):
        """Initialize the quality filter.

        Args:
            max_price_deviation_pct: Maximum price change from reference (percent).
            max_age_seconds: Maximum data age in seconds.
            min_volume: Minimum valid volume.
            max_volume: Maximum valid volume.
        """
        self.max_price_deviation_pct = max_price_deviation_pct
        self.max_age_seconds = max_age_seconds
        self.min_volume = min_volume
        self.max_volume = max_volume
        self._reference_prices: Dict[str, float] = {}
        self._filter_stats: Dict[str, int] = defaultdict(int)

    def assess_market_data(self, data: Dict[str, Any]) -> DataQuality:
        """Assess quality of market data message.

        Args:
            data: Market data message dict.

        Returns:
            DataQuality assessment.
        """
        # Check required fields
        required_fields = ["symbol", "price", "timestamp"]
        for field_name in required_fields:
            if field_name not in data:
                self._filter_stats["missing_field"] += 1
                return DataQuality.INVALID

        # Price validation
        price = data.get("price", 0)
        if not isinstance(price, (int, float)) or price <= 0:
            self._filter_stats["invalid_price"] += 1
            return DataQuality.INVALID

        # Price deviation check
        symbol = data.get("symbol", "")
        ref_price = self._reference_prices.get(symbol)
        if ref_price is not None and ref_price > 0:
            deviation_pct = abs(price - ref_price) / ref_price * 100
            if deviation_pct > self.max_price_deviation_pct:
                self._filter_stats["price_deviation"] += 1
                return DataQuality.SUSPICIOUS

        # Update reference price
        self._reference_prices[symbol] = price

        # Volume validation
        volume = data.get("volume", 0)
        if isinstance(volume, (int, float)):
            if volume < self.min_volume or volume > self.max_volume:
                self._filter_stats["invalid_volume"] += 1
                return DataQuality.SUSPICIOUS

        # Timestamp freshness
        timestamp_str = data.get("timestamp", "")
        if timestamp_str:
            try:
                msg_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                age = (datetime.utcnow() - msg_time.replace(tzinfo=None)).total_seconds()
                if abs(age) > self.max_age_seconds:
                    self._filter_stats["stale_data"] += 1
                    return DataQuality.SUSPICIOUS
            except (ValueError, TypeError):
                pass  # Can't parse timestamp, accept data

        self._filter_stats["valid"] += 1
        return DataQuality.VALID

    def assess_signal(self, data: Dict[str, Any]) -> DataQuality:
        """Assess quality of a signal message.

        Args:
            data: Signal message dict.

        Returns:
            DataQuality assessment.
        """
        required_fields = ["signal_id", "symbol", "direction", "strength"]
        for field_name in required_fields:
            if field_name not in data:
                self._filter_stats["signal_missing_field"] += 1
                return DataQuality.INVALID

        # Strength range check
        strength = data.get("strength", 0)
        if not isinstance(strength, (int, float)) or strength < 0 or strength > 1:
            self._filter_stats["invalid_strength"] += 1
            return DataQuality.INVALID

        self._filter_stats["signal_valid"] += 1
        return DataQuality.VALID

    @property
    def stats(self) -> Dict[str, int]:
        """Get filter statistics."""
        return dict(self._filter_stats)


# ============================================================================
# Training Data Buffer
# ============================================================================

class TrainingDataBuffer:
    """Time-based flushing buffer for training data.

    Accumulates consumed data and flushes in batches based on
    time intervals or buffer size thresholds. Supports per-symbol
    buffering and configurable flush callbacks.
    """

    def __init__(self, flush_interval_seconds: float = 5.0,
                 max_buffer_size: int = 10000,
                 per_symbol_buffer: bool = True):
        """Initialize the training data buffer.

        Args:
            flush_interval_seconds: Seconds between automatic flushes.
            max_buffer_size: Maximum items before forced flush.
            per_symbol_buffer: Whether to maintain per-symbol buffers.
        """
        self.flush_interval_seconds = flush_interval_seconds
        self.max_buffer_size = max_buffer_size
        self.per_symbol_buffer = per_symbol_buffer

        self._buffer: List[Dict[str, Any]] = []
        self._symbol_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._flush_callbacks: List[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = []
        self._last_flush_time: float = time.time()
        self._total_buffered: int = 0
        self._total_flushed: int = 0
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

    def add_flush_callback(self, callback: Callable[[List[Dict[str, Any]]], Awaitable[None]]) -> None:
        """Register a callback for when data is flushed.

        Args:
            callback: Async callable receiving the flushed data list.
        """
        self._flush_callbacks.append(callback)

    async def start(self) -> None:
        """Start the periodic flush loop."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("TrainingDataBuffer started (flush interval: %.1fs, max size: %d)",
                     self.flush_interval_seconds, self.max_buffer_size)

    async def stop(self) -> None:
        """Stop the buffer and flush remaining data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self.flush()

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                if self._should_flush():
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Flush loop error: %s", e)

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        time_exceeded = time.time() - self._last_flush_time >= self.flush_interval_seconds
        size_exceeded = self._total_buffered >= self.max_buffer_size
        return time_exceeded or size_exceeded

    def add(self, data: Dict[str, Any]) -> None:
        """Add data to the buffer.

        Args:
            data: Data dict to buffer.
        """
        self._buffer.append(data)
        self._total_buffered += 1

        if self.per_symbol_buffer:
            symbol = data.get("symbol", "unknown")
            self._symbol_buffers[symbol].append(data)

        # Force flush if buffer is full
        if self._total_buffered >= self.max_buffer_size:
            asyncio.create_task(self.flush())

    async def flush(self) -> int:
        """Flush the buffer, calling all registered callbacks.

        Returns:
            Number of items flushed.
        """
        if not self._buffer:
            return 0

        # Get data to flush
        data_to_flush = self._buffer[:]

        # Clear buffers
        self._buffer = []
        self._symbol_buffers = defaultdict(list)
        n_flushed = self._total_buffered
        self._total_buffered = 0
        self._total_flushed += n_flushed
        self._last_flush_time = time.time()

        # Call callbacks
        for callback in self._flush_callbacks:
            try:
                await callback(data_to_flush)
            except Exception as e:
                logger.error("Flush callback error: %s", e)

        logger.debug("Flushed %d items from buffer", n_flushed)
        return n_flushed

    @property
    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "current_buffer_size": len(self._buffer),
            "total_buffered": self._total_buffered,
            "total_flushed": self._total_flushed,
            "last_flush_time": self._last_flush_time,
            "symbol_count": len(self._symbol_buffers),
        }


# ============================================================================
# AI Kafka Consumer (Base)
# ============================================================================

class AIKafkaConsumer:
    """Base Kafka consumer for AI pipeline data consumption.

    Provides:
    - Consumer group management
    - Offset management for exactly-once processing
    - Backpressure handling
    - Data quality filtering
    - Automatic reconnection
    """

    def __init__(self, config: Optional[AIConsumerConfig] = None):
        """Initialize the AI Kafka consumer.

        Args:
            config: Consumer configuration.
        """
        self.config = config or AIConsumerConfig()
        self._consumer = None
        self._quality_filter = DataQualityFilter() if self.config.quality_filter_enabled else None
        self._buffer = TrainingDataBuffer(
            flush_interval_seconds=self.config.flush_interval_seconds,
            max_buffer_size=self.config.buffer_max_size,
        )
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self._offsets: Dict[Tuple[str, int], int] = {}  # (topic, partition) -> offset
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consumed_count: int = 0
        self._filtered_count: int = 0
        self._error_count: int = 0
        self._backpressure_active: bool = False

    async def start(self, topics: List[str]) -> None:
        """Start consuming from specified topics.

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
                session_timeout_ms=self.config.session_timeout_ms,
                max_poll_interval_ms=self.config.max_poll_interval_ms,
                fetch_max_bytes=self.config.fetch_max_bytes,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            await self._consumer.start()
            self._running = True
            await self._buffer.start()
            self._task = asyncio.create_task(self._consume_loop())
            logger.info("AIKafkaConsumer started, topics: %s, group: %s",
                         topics, self.config.group_id)
        except ImportError:
            logger.warning("aiokafka not installed, AI consumer disabled")
            self._running = False
        except Exception as e:
            logger.error("Failed to start AI Kafka consumer: %s", e)
            raise

    async def stop(self) -> None:
        """Stop consuming and flush buffers."""
        self._running = False
        await self._buffer.stop()
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
        logger.info("AIKafkaConsumer stopped (consumed: %d, filtered: %d, errors: %d)",
                     self._consumed_count, self._filtered_count, self._error_count)

    def register_handler(self, topic: str,
                          handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register a message handler for a topic.

        Args:
            topic: Topic name.
            handler: Async callable for processing messages.
        """
        self._handlers[topic] = handler

    async def _consume_loop(self) -> None:
        """Main consumption loop with backpressure and quality filtering."""
        if not self._consumer:
            return

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                # Backpressure check
                if self._buffer._total_buffered >= self.config.backpressure_threshold:
                    self._backpressure_active = True
                    logger.warning("Backpressure active: buffer at %d items",
                                   self._buffer._total_buffered)
                    await asyncio.sleep(0.1)
                    continue
                else:
                    self._backpressure_active = False

                try:
                    topic = message.topic
                    value = message.value

                    # Quality filtering
                    if self._quality_filter is not None:
                        quality = self._assess_quality(topic, value)
                        if quality == DataQuality.INVALID:
                            self._filtered_count += 1
                            continue
                        value["_quality"] = quality.value

                    # Buffer for training
                    self._buffer.add(value)

                    # Call topic handler
                    handler = self._handlers.get(topic)
                    if handler:
                        await handler(value)

                    # Track offset for exactly-once processing
                    self._offsets[(message.topic, message.partition)] = message.offset
                    self._consumed_count += 1

                    # Manual commit after successful processing
                    if not self.config.enable_auto_commit and self._consumed_count % 100 == 0:
                        await self._commit_offsets()

                except Exception as e:
                    self._error_count += 1
                    logger.error("Error processing message from '%s': %s", message.topic, e)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Consumer loop error: %s", e)

    def _assess_quality(self, topic: str, data: Dict[str, Any]) -> DataQuality:
        """Assess data quality based on topic type.

        Args:
            topic: Kafka topic name.
            data: Message data.

        Returns:
            DataQuality assessment.
        """
        if self._quality_filter is None:
            return DataQuality.VALID

        if "market-data" in topic:
            return self._quality_filter.assess_market_data(data)
        elif "signal" in topic:
            return self._quality_filter.assess_signal(data)
        return DataQuality.VALID

    async def _commit_offsets(self) -> None:
        """Manually commit offsets for exactly-once processing."""
        if not self._consumer or not self._offsets:
            return
        try:
            await self._consumer.commit()
            logger.debug("Offsets committed")
        except Exception as e:
            logger.warning("Failed to commit offsets: %s", e)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            "consumed_count": self._consumed_count,
            "filtered_count": self._filtered_count,
            "error_count": self._error_count,
            "backpressure_active": self._backpressure_active,
            "buffer_stats": self._buffer.stats,
            "quality_filter_stats": self._quality_filter.stats if self._quality_filter else {},
            "running": self._running,
            "tracked_offsets": len(self._offsets),
        }


# ============================================================================
# Market Data Consumer
# ============================================================================

class MarketDataConsumer:
    """Consumes market data (trades, orderbooks, candles) from Redpanda.

    Subscribes to market data topics and routes messages to
    appropriate handlers based on data_type (tick, candle, orderbook).
    """

    def __init__(self, config: Optional[AIConsumerConfig] = None,
                 symbols: Optional[List[str]] = None):
        """Initialize the market data consumer.

        Args:
            config: Consumer configuration.
            symbols: List of symbols to filter (None = all).
        """
        self.config = config or AIConsumerConfig()
        self.config.group_id = f"{self.config.group_id}-market-data"
        self.symbols = set(symbols) if symbols else None

        self._consumer = AIKafkaConsumer(self.config)
        self._tick_handlers: List[Callable] = []
        self._candle_handlers: List[Callable] = []
        self._orderbook_handlers: List[Callable] = []
        self._latest_prices: Dict[str, float] = {}
        self._latest_candles: Dict[str, Dict] = {}

    async def start(self) -> None:
        """Start consuming market data."""
        topics = ["market-data"]
        self._consumer.register_handler("market-data", self._handle_market_data)
        await self._consumer.start(topics)
        logger.info("MarketDataConsumer started for symbols: %s",
                     self.symbols or "all")

    async def stop(self) -> None:
        """Stop consuming market data."""
        await self._consumer.stop()

    def on_tick(self, handler: Callable) -> None:
        """Register handler for tick data."""
        self._tick_handlers.append(handler)

    def on_candle(self, handler: Callable) -> None:
        """Register handler for candle data."""
        self._candle_handlers.append(handler)

    def on_orderbook(self, handler: Callable) -> None:
        """Register handler for orderbook data."""
        self._orderbook_handlers.append(handler)

    async def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """Route market data to appropriate handlers.

        Args:
            data: Market data message.
        """
        symbol = data.get("symbol", "")
        if self.symbols and symbol not in self.symbols:
            return

        data_type = data.get("data_type", "")

        if data_type == "tick":
            self._latest_prices[symbol] = data.get("price", 0)
            for handler in self._tick_handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error("Tick handler error: %s", e)

        elif data_type == "candle":
            self._latest_candles[symbol] = data
            for handler in self._candle_handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error("Candle handler error: %s", e)

        elif data_type == "orderbook":
            for handler in self._orderbook_handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error("Orderbook handler error: %s", e)

    @property
    def latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all tracked symbols."""
        return dict(self._latest_prices)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            "consumer_stats": self._consumer.stats,
            "tracked_symbols": len(self._latest_prices),
            "latest_candle_symbols": len(self._latest_candles),
        }


# ============================================================================
# Signal Consumer
# ============================================================================

class SignalConsumer:
    """Consumes trading signals from the signal engine via Kafka.

    Processes signals for ML model training (label generation)
    and real-time model evaluation feedback.
    """

    def __init__(self, config: Optional[AIConsumerConfig] = None,
                 symbols: Optional[List[str]] = None):
        """Initialize the signal consumer.

        Args:
            config: Consumer configuration.
            symbols: List of symbols to filter.
        """
        self.config = config or AIConsumerConfig()
        self.config.group_id = f"{self.config.group_id}-signals"
        self.symbols = set(symbols) if symbols else None

        self._consumer = AIKafkaConsumer(self.config)
        self._signal_handlers: List[Callable] = []
        self._signal_buffer: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self._signal_counts: Dict[str, int] = defaultdict(int)

    async def start(self) -> None:
        """Start consuming signals."""
        self._consumer.register_handler("signals", self._handle_signal)
        await self._consumer.start(["signals"])
        logger.info("SignalConsumer started")

    async def stop(self) -> None:
        """Stop consuming signals."""
        await self._consumer.stop()

    def on_signal(self, handler: Callable) -> None:
        """Register handler for signal data."""
        self._signal_handlers.append(handler)

    async def _handle_signal(self, data: Dict[str, Any]) -> None:
        """Process incoming signal.

        Args:
            data: Signal message.
        """
        symbol = data.get("symbol", "")
        if self.symbols and symbol not in self.symbols:
            return

        direction = data.get("direction", "neutral")
        self._signal_counts[direction] += 1
        self._signal_buffer.append(data)

        for handler in self._signal_handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error("Signal handler error: %s", e)

    def get_recent_signals(self, limit: int = 100,
                            symbol: Optional[str] = None,
                            direction: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent signals from buffer.

        Args:
            limit: Maximum signals to return.
            symbol: Optional symbol filter.
            direction: Optional direction filter.

        Returns:
            List of signal dicts.
        """
        signals = list(self._signal_buffer)
        if symbol:
            signals = [s for s in signals if s.get("symbol") == symbol]
        if direction:
            signals = [s for s in signals if s.get("direction") == direction]
        return signals[-limit:]

    @property
    def stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            "consumer_stats": self._consumer.stats,
            "signal_counts": dict(self._signal_counts),
            "buffer_size": len(self._signal_buffer),
        }
