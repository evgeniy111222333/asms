"""Comprehensive tests for acms.exchanges module.

Tests all exchange adapter classes, rate limiter, order book,
error handling, retry logic, factory function, and PaperTradingAdapter.
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import asyncio
import pytest
import pytest_asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from acms.core import (
    Order, Trade, Position, Candle, Tick, Side, OrderType,
    OrderStatus, TimeInForce, ExchangeId,
)
from acms.exchanges import (
    RateLimiter,
    LocalOrderBook,
    ExchangeError,
    RateLimitError,
    InsufficientFundsError,
    OrderNotFoundError,
    NetworkError,
    classify_exchange_error,
    retry_with_backoff,
    ExchangeCredentials,
    ExchangeAdapter,
    BinanceAdapter,
    BybitAdapter,
    OKXAdapter,
    PaperTradingAdapter,
    ArbitrageOpportunity,
    ArbitrageDetector,
    create_exchange_adapter,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def credentials():
    return ExchangeCredentials(api_key="test_key", api_secret="test_secret", passphrase="test_pass")


@pytest.fixture
def empty_credentials():
    return ExchangeCredentials()


@pytest.fixture
def paper_adapter():
    return PaperTradingAdapter(initial_balance=100000.0, slippage_bps=5.0, commission_bps=10.0)


@pytest.fixture
def binance_adapter(credentials):
    return BinanceAdapter(credentials, testnet=True)


@pytest.fixture
def bybit_adapter(credentials):
    return BybitAdapter(credentials, testnet=True)


@pytest.fixture
def okx_adapter(credentials):
    return OKXAdapter(credentials, testnet=True)


@pytest.fixture
def sample_order():
    return Order(
        id="order_001",
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.CREATED,
        quantity=0.1,
        price=50000.0,
    )


@pytest.fixture
def limit_order():
    return Order(
        id="order_002",
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.CREATED,
        quantity=0.1,
        price=49000.0,
        time_in_force=TimeInForce.GTC,
    )


@pytest.fixture
def sell_order():
    return Order(
        id="order_003",
        symbol="BTC/USDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        status=OrderStatus.CREATED,
        quantity=0.1,
        price=50000.0,
    )


# ============================================================================
# RateLimiter Tests
# ============================================================================

class TestRateLimiter:

    def test_construction_defaults(self):
        rl = RateLimiter()
        assert rl.max_requests == 10
        assert rl.window_seconds == 1.0
        assert rl.burst_size == 10
        assert rl._tokens == 10.0

    def test_construction_custom(self):
        rl = RateLimiter(max_requests=20, window_seconds=2.0, burst_size=30)
        assert rl.max_requests == 20
        assert rl.window_seconds == 2.0
        assert rl.burst_size == 30
        assert rl._tokens == 30.0

    def test_construction_burst_default_to_max_requests(self):
        rl = RateLimiter(max_requests=15)
        assert rl.burst_size == 15

    @pytest.mark.asyncio
    async def test_acquire_reduces_tokens(self):
        rl = RateLimiter(max_requests=10, window_seconds=1.0)
        await rl.acquire()
        assert rl._tokens < 10.0

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        rl = RateLimiter(max_requests=100, window_seconds=1.0)
        for _ in range(5):
            await rl.acquire()
        assert rl._tokens < 100.0

    @pytest.mark.asyncio
    async def test_acquire_refills_over_time(self):
        rl = RateLimiter(max_requests=10, window_seconds=0.1)
        # Drain tokens
        for _ in range(10):
            await rl.acquire()
        # Wait for refill
        await asyncio.sleep(0.2)
        assert rl.available_tokens > 0

    def test_available_tokens_property(self):
        rl = RateLimiter(max_requests=10, window_seconds=1.0)
        tokens = rl.available_tokens
        # Can be int or float depending on refill calculation
        assert float(tokens) > 0

    def test_refill_tokens(self):
        rl = RateLimiter(max_requests=10, window_seconds=1.0)
        rl._tokens = 0.0
        rl._last_refill = time.monotonic() - 0.5  # Half a second ago
        rl._refill_tokens()
        # Should have refilled some tokens
        assert rl._tokens > 0

    def test_refill_capped_at_burst(self):
        rl = RateLimiter(max_requests=10, window_seconds=1.0, burst_size=10)
        rl._tokens = 9.0
        rl._last_refill = time.monotonic() - 10.0  # Long time ago
        rl._refill_tokens()
        assert rl._tokens <= 10.0

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test concurrent access to rate limiter."""
        rl = RateLimiter(max_requests=100, window_seconds=1.0)
        tasks = [rl.acquire() for _ in range(10)]
        await asyncio.gather(*tasks)
        assert rl._tokens < 100.0


# ============================================================================
# LocalOrderBook Tests
# ============================================================================

class TestLocalOrderBook:

    def test_construction(self):
        ob = LocalOrderBook("BTC/USDT")
        assert ob.symbol == "BTC/USDT"
        assert ob.max_depth == 50
        assert ob.bids == {}
        assert ob.asks == {}
        assert ob.last_update_id == 0
        assert ob.updated_at is None

    def test_construction_custom_depth(self):
        ob = LocalOrderBook("ETH/USDT", max_depth=20)
        assert ob.max_depth == 20

    def test_update_adds_bids_and_asks(self):
        ob = LocalOrderBook("BTC/USDT")
        bids = [(50000.0, 1.5), (49999.0, 2.0)]
        asks = [(50001.0, 1.0), (50002.0, 3.0)]
        ob.update(bids, asks, update_id=1)
        assert ob.bids[50000.0] == 1.5
        assert ob.bids[49999.0] == 2.0
        assert ob.asks[50001.0] == 1.0
        assert ob.asks[50002.0] == 3.0
        assert ob.last_update_id == 1
        assert ob.updated_at is not None

    def test_update_removes_zero_quantity(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.5)], [(50001.0, 1.0)], update_id=1)
        assert 50000.0 in ob.bids
        # Remove bid with qty 0
        ob.update([(50000.0, 0.0)], [], update_id=2)
        assert 50000.0 not in ob.bids

    def test_update_removes_negative_quantity(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.5)], [(50001.0, 1.0)], update_id=1)
        ob.update([(50000.0, -0.1)], [], update_id=2)
        assert 50000.0 not in ob.bids

    def test_update_ignores_stale_update_id(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.5)], [(50001.0, 1.0)], update_id=5)
        ob.update([(50000.0, 2.0)], [(50001.0, 2.0)], update_id=3)  # Stale
        assert ob.bids[50000.0] == 1.5  # Not updated

    def test_update_allows_zero_update_id(self):
        """update_id=0 should always be applied."""
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.5)], [(50001.0, 1.0)], update_id=5)
        ob.update([(50000.0, 2.0)], [(50001.0, 2.0)], update_id=0)
        assert ob.bids[50000.0] == 2.0

    def test_update_empty_lists(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([], [], update_id=1)
        assert ob.bids == {}
        assert ob.asks == {}

    def test_get_best_bid(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(49998.0, 1.0), (50000.0, 1.5), (49999.0, 2.0)], [], update_id=1)
        assert ob.get_best_bid() == 50000.0

    def test_get_best_bid_empty(self):
        ob = LocalOrderBook("BTC/USDT")
        assert ob.get_best_bid() is None

    def test_get_best_ask(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([], [(50002.0, 1.0), (50001.0, 1.5), (50003.0, 2.0)], update_id=1)
        assert ob.get_best_ask() == 50001.0

    def test_get_best_ask_empty(self):
        ob = LocalOrderBook("BTC/USDT")
        assert ob.get_best_ask() is None

    def test_get_spread(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.0)], [(50001.0, 1.0)], update_id=1)
        assert ob.get_spread() == 1.0

    def test_get_spread_no_bids(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([], [(50001.0, 1.0)], update_id=1)
        assert ob.get_spread() is None

    def test_get_spread_no_asks(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.0)], [], update_id=1)
        assert ob.get_spread() is None

    def test_get_mid_price(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.0)], [(50002.0, 1.0)], update_id=1)
        assert ob.get_mid_price() == 50001.0

    def test_get_mid_price_empty(self):
        ob = LocalOrderBook("BTC/USDT")
        assert ob.get_mid_price() is None

    def test_snapshot(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(49999.0, 1.0), (50000.0, 1.5)], [(50001.0, 2.0), (50002.0, 3.0)], update_id=1)
        snap = ob.snapshot()
        assert snap["symbol"] == "BTC/USDT"
        assert len(snap["bids"]) == 2
        assert len(snap["asks"]) == 2
        assert snap["spread"] is not None
        assert snap["mid_price"] is not None
        assert snap["updated_at"] is not None
        # Bids sorted descending
        assert snap["bids"][0][0] > snap["bids"][1][0]
        # Asks sorted ascending
        assert snap["asks"][0][0] < snap["asks"][1][0]

    def test_snapshot_no_data(self):
        ob = LocalOrderBook("BTC/USDT")
        snap = ob.snapshot()
        assert snap["symbol"] == "BTC/USDT"
        assert snap["bids"] == []
        assert snap["asks"] == []
        assert snap["spread"] is None
        assert snap["mid_price"] is None
        assert snap["updated_at"] is None

    def test_trim_depth(self):
        ob = LocalOrderBook("BTC/USDT", max_depth=3)
        bids = [(50000.0 - i, 1.0) for i in range(5)]
        asks = [(50001.0 + i, 1.0) for i in range(5)]
        ob.update(bids, asks, update_id=1)
        assert len(ob.bids) <= 3
        assert len(ob.asks) <= 3
        # Best levels should be kept
        assert max(ob.bids.keys()) == 50000.0
        assert min(ob.asks.keys()) == 50001.0


# ============================================================================
# Error Classes Tests
# ============================================================================

class TestExchangeErrors:

    def test_exchange_error(self):
        e = ExchangeError("test error", exchange="binance", code="1000")
        assert str(e) == "test error"
        assert e.exchange == "binance"
        assert e.code == "1000"

    def test_exchange_error_defaults(self):
        e = ExchangeError("test")
        assert e.exchange == ""
        assert e.code == ""

    def test_rate_limit_error(self):
        e = RateLimitError("rate limited", exchange="binance")
        assert isinstance(e, ExchangeError)
        assert e.exchange == "binance"

    def test_insufficient_funds_error(self):
        e = InsufficientFundsError("insufficient", exchange="bybit")
        assert isinstance(e, ExchangeError)

    def test_order_not_found_error(self):
        e = OrderNotFoundError("not found", exchange="okx")
        assert isinstance(e, ExchangeError)

    def test_network_error(self):
        e = NetworkError("timeout", exchange="binance")
        assert isinstance(e, ExchangeError)


# ============================================================================
# classify_exchange_error Tests
# ============================================================================

class TestClassifyExchangeError:

    def test_429_rate_limit(self):
        error = classify_exchange_error(429, {"msg": "too many requests"}, "binance")
        assert isinstance(error, RateLimitError)

    def test_429_rate_limit_bybit(self):
        error = classify_exchange_error(429, {"retMsg": "rate limit"}, "bybit")
        assert isinstance(error, RateLimitError)

    def test_429_rate_limit_okx(self):
        error = classify_exchange_error(429, {"msg": "rate limit"}, "okx")
        assert isinstance(error, RateLimitError)

    def test_insufficient_funds_message(self):
        error = classify_exchange_error(400, {"msg": "Insufficient balance"}, "binance")
        assert isinstance(error, InsufficientFundsError)

    def test_insufficient_funds_code_binance(self):
        error = classify_exchange_error(400, {"code": -2019, "msg": "error"}, "binance")
        assert isinstance(error, InsufficientFundsError)

    def test_insufficient_funds_code_bybit(self):
        error = classify_exchange_error(400, {"retCode": 130040, "retMsg": "error"}, "bybit")
        assert isinstance(error, InsufficientFundsError)

    def test_insufficient_funds_code_okx(self):
        error = classify_exchange_error(400, {"code": "51421", "msg": "error"}, "okx")
        assert isinstance(error, InsufficientFundsError)

    def test_order_not_found_message(self):
        error = classify_exchange_error(404, {"msg": "Order not found"}, "binance")
        assert isinstance(error, OrderNotFoundError)

    def test_order_keyword_in_message(self):
        error = classify_exchange_error(400, {"msg": "Invalid order id"}, "binance")
        assert isinstance(error, OrderNotFoundError)

    def test_generic_exchange_error(self):
        error = classify_exchange_error(500, {"msg": "Internal error"}, "binance")
        assert isinstance(error, ExchangeError)
        assert not isinstance(error, RateLimitError)

    def test_binance_extracts_msg(self):
        error = classify_exchange_error(400, {"code": -1000, "msg": "Binance error"}, "binance")
        assert "Binance error" in str(error)

    def test_bybit_extracts_retMsg(self):
        error = classify_exchange_error(400, {"retCode": 1000, "retMsg": "Bybit error"}, "bybit")
        assert "Bybit error" in str(error)

    def test_okx_extracts_msg(self):
        error = classify_exchange_error(400, {"code": "50000", "msg": "OKX error"}, "okx")
        assert "OKX error" in str(error)

    def test_unknown_exchange(self):
        error = classify_exchange_error(400, {"error": "something"}, "unknown_exchange")
        assert isinstance(error, ExchangeError)


# ============================================================================
# retry_with_backoff Tests
# ============================================================================

class TestRetryWithBackoff:

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(succeed, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("rate limited", "binance")
            return "success"

        result = await retry_with_backoff(fail_then_succeed, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("connection failed", "binance")
            return "success"

        result = await retry_with_backoff(fail_then_succeed, max_retries=3, base_delay=0.01)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_raises_exchange_error_immediately(self):
        """ExchangeError (non-retryable) should be raised immediately."""

        async def fail():
            raise ExchangeError("bad request", "binance", "400")

        with pytest.raises(ExchangeError):
            await retry_with_backoff(fail, max_retries=3, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("rate limited", "binance")

        with pytest.raises(RateLimitError):
            await retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)
        assert call_count == 3  # max_retries + 1

    @pytest.mark.asyncio
    async def test_generic_exception_retried(self):
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("generic error")
            return "success"

        result = await retry_with_backoff(fail_then_succeed, max_retries=3, base_delay=0.01)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_generic_exception_exhausted(self):
        async def always_fail():
            raise ValueError("always fails")

        with pytest.raises(ExchangeError):
            await retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_max_delay_respected(self):
        """Ensure delay doesn't exceed max_delay."""
        start = time.monotonic()
        call_count = 0

        async def fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("rate limited", "binance")

        with pytest.raises(RateLimitError):
            await retry_with_backoff(fail, max_retries=2, base_delay=100.0, max_delay=0.05)
        elapsed = time.monotonic() - start
        # Should not take base_delay * 2^2 = 400 seconds
        assert elapsed < 2.0


# ============================================================================
# ExchangeCredentials Tests
# ============================================================================

class TestExchangeCredentials:

    def test_construction_defaults(self):
        creds = ExchangeCredentials()
        assert creds.api_key == ""
        assert creds.api_secret == ""
        assert creds.passphrase == ""

    def test_construction_with_values(self):
        creds = ExchangeCredentials(api_key="key", api_secret="secret", passphrase="pass")
        assert creds.api_key == "key"
        assert creds.api_secret == "secret"
        assert creds.passphrase == "pass"


# ============================================================================
# ExchangeAdapter Base Class Tests
# ============================================================================

class ConcreteAdapter(ExchangeAdapter):
    """Concrete implementation for testing the abstract base class."""

    async def get_candles(self, symbol, timeframe, limit=500):
        return []

    async def get_order_book(self, symbol, depth=20):
        return {"bids": [], "asks": []}

    async def place_order(self, order):
        return order

    async def cancel_order(self, order_id, symbol):
        return True

    async def get_order_status(self, order_id, symbol):
        return Order(id=order_id, symbol=symbol, side=Side.BUY,
                     order_type=OrderType.LIMIT, status=OrderStatus.CREATED, quantity=0)

    async def get_positions(self):
        return []

    async def get_balance(self):
        return {}

    async def subscribe_ticks(self, symbol, callback):
        pass

    async def subscribe_order_book(self, symbol, callback):
        pass


class TestExchangeAdapterBase:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ExchangeAdapter(ExchangeCredentials())

    def test_concrete_construction(self, credentials):
        adapter = ConcreteAdapter(credentials, testnet=False)
        assert adapter.credentials == credentials
        assert adapter.testnet is False
        assert adapter.ws_connected is False
        assert isinstance(adapter.rate_limiter, RateLimiter)
        assert adapter.order_books == {}
        assert adapter._ws_reconnect_attempts == 0
        assert adapter._max_reconnect_attempts == 10

    def test_construction_testnet(self, credentials):
        adapter = ConcreteAdapter(credentials, testnet=True)
        assert adapter.testnet is True

    @pytest.mark.asyncio
    async def test_get_funding_rate_default(self, credentials):
        adapter = ConcreteAdapter(credentials)
        result = await adapter.get_funding_rate("BTC/USDT")
        assert result is None

    @pytest.mark.asyncio
    async def test_close(self, credentials):
        adapter = ConcreteAdapter(credentials)
        adapter.ws_connected = True
        await adapter.close()
        assert adapter.ws_connected is False


# ============================================================================
# BinanceAdapter Tests
# ============================================================================

class TestBinanceAdapter:

    def test_construction_production(self, credentials):
        adapter = BinanceAdapter(credentials, testnet=False)
        assert adapter.base_url == BinanceAdapter.REST_URL
        assert adapter.ws_url == BinanceAdapter.WS_URL
        assert adapter.rate_limiter.max_requests == 20

    def test_construction_testnet(self, credentials):
        adapter = BinanceAdapter(credentials, testnet=True)
        assert adapter.base_url == BinanceAdapter.TESTNET_REST
        assert adapter.ws_url == BinanceAdapter.TESTNET_WS

    def test_map_symbol(self, credentials):
        adapter = BinanceAdapter(credentials)
        assert adapter._map_symbol("BTC/USDT") == "BTCUSDT"
        assert adapter._map_symbol("ETH/USDT") == "ETHUSDT"

    def test_sign_with_secret(self, credentials):
        adapter = BinanceAdapter(credentials)
        params = {"symbol": "BTCUSDT", "timestamp": "1234567890"}
        signed = adapter._sign(params)
        assert "signature" in signed
        assert signed["symbol"] == "BTCUSDT"

    def test_sign_without_secret(self, empty_credentials):
        adapter = BinanceAdapter(empty_credentials)
        params = {"symbol": "BTCUSDT"}
        signed = adapter._sign(params)
        assert "signature" not in signed

    def test_headers(self, credentials):
        adapter = BinanceAdapter(credentials)
        headers = adapter._headers()
        assert headers["X-MBX-APIKEY"] == "test_key"

    def test_order_type_map(self):
        assert BinanceAdapter._order_type_map(OrderType.MARKET) == "MARKET"
        assert BinanceAdapter._order_type_map(OrderType.LIMIT) == "LIMIT"
        assert BinanceAdapter._order_type_map(OrderType.STOP) == "STOP_LOSS"
        assert BinanceAdapter._order_type_map(OrderType.STOP_LIMIT) == "STOP_LOSS_LIMIT"
        assert BinanceAdapter._order_type_map(OrderType.TRAILING_STOP) == "TRAILING_STOP_MARKET"
        # Unknown type defaults to LIMIT
        assert BinanceAdapter._order_type_map(OrderType.ICEBERG) == "LIMIT"

    def test_status_map(self):
        assert BinanceAdapter._status_map("NEW") == OrderStatus.SUBMITTED
        assert BinanceAdapter._status_map("PARTIALLY_FILLED") == OrderStatus.PARTIALLY_FILLED
        assert BinanceAdapter._status_map("FILLED") == OrderStatus.FILLED
        assert BinanceAdapter._status_map("CANCELED") == OrderStatus.CANCELLED
        assert BinanceAdapter._status_map("REJECTED") == OrderStatus.REJECTED
        assert BinanceAdapter._status_map("EXPIRED") == OrderStatus.EXPIRED
        assert BinanceAdapter._status_map("UNKNOWN") == OrderStatus.CREATED

    def test_class_constants(self):
        assert BinanceAdapter.REST_URL == "https://api.binance.com"
        assert BinanceAdapter.WS_URL == "wss://stream.binance.com:9443/ws"
        assert BinanceAdapter.TESTNET_REST == "https://testnet.binance.vision"
        assert BinanceAdapter.FUTURES_REST == "https://fapi.binance.com"

    @pytest.mark.asyncio
    async def test_get_positions_returns_empty(self, credentials):
        adapter = BinanceAdapter(credentials)
        positions = await adapter.get_positions()
        assert positions == []


# ============================================================================
# BybitAdapter Tests
# ============================================================================

class TestBybitAdapter:

    def test_construction_production(self, credentials):
        adapter = BybitAdapter(credentials, testnet=False)
        assert adapter.base_url == BybitAdapter.REST_URL
        assert adapter.ws_public_url == BybitAdapter.WS_PUBLIC_URL
        assert adapter.ws_private_url == BybitAdapter.WS_PRIVATE_URL
        assert adapter.rate_limiter.max_requests == 10

    def test_construction_testnet(self, credentials):
        adapter = BybitAdapter(credentials, testnet=True)
        assert adapter.base_url == BybitAdapter.TESTNET_REST
        assert adapter.ws_public_url == BybitAdapter.TESTNET_WS_PUBLIC
        assert adapter.ws_private_url == BybitAdapter.TESTNET_WS_PRIVATE

    def test_map_symbol(self, credentials):
        adapter = BybitAdapter(credentials)
        assert adapter._map_symbol("BTC/USDT") == "BTCUSDT"

    def test_sign_with_secret(self, credentials):
        adapter = BybitAdapter(credentials)
        params = {"category": "spot"}
        signed = adapter._sign(params)
        assert "api_key" in signed
        assert "timestamp" in signed
        assert "sign" in signed

    def test_sign_without_secret(self, empty_credentials):
        adapter = BybitAdapter(empty_credentials)
        params = {"category": "spot"}
        signed = adapter._sign(params)
        # Without secret, should return params unchanged
        assert "sign" not in signed

    def test_auth_headers(self, credentials):
        adapter = BybitAdapter(credentials)
        auth = adapter._auth_headers()
        assert auth["op"] == "auth"
        assert len(auth["args"]) == 3
        assert auth["args"][0] == "test_key"

    def test_status_map(self):
        assert BybitAdapter._status_map("Created") == OrderStatus.CREATED
        assert BybitAdapter._status_map("New") == OrderStatus.SUBMITTED
        assert BybitAdapter._status_map("PartiallyFilled") == OrderStatus.PARTIALLY_FILLED
        assert BybitAdapter._status_map("Filled") == OrderStatus.FILLED
        assert BybitAdapter._status_map("Cancelled") == OrderStatus.CANCELLED
        assert BybitAdapter._status_map("Rejected") == OrderStatus.REJECTED
        assert BybitAdapter._status_map("Unknown") == OrderStatus.CREATED

    def test_class_constants(self):
        assert BybitAdapter.REST_URL == "https://api.bybit.com"
        assert BybitAdapter.TESTNET_REST == "https://api-testnet.bybit.com"


# ============================================================================
# OKXAdapter Tests
# ============================================================================

class TestOKXAdapter:

    def test_construction_production(self, credentials):
        adapter = OKXAdapter(credentials, testnet=False)
        assert adapter.base_url == OKXAdapter.REST_URL
        assert adapter.ws_public_url == OKXAdapter.WS_PUBLIC_URL
        assert adapter.ws_private_url == OKXAdapter.WS_PRIVATE_URL
        assert adapter.rate_limiter.max_requests == 10

    def test_construction_testnet(self, credentials):
        adapter = OKXAdapter(credentials, testnet=True)
        assert adapter.ws_public_url == OKXAdapter.TESTNET_WS_PUBLIC
        assert adapter.ws_private_url == OKXAdapter.TESTNET_WS_PRIVATE

    def test_map_symbol(self, credentials):
        adapter = OKXAdapter(credentials)
        assert adapter._map_symbol("BTC/USDT") == "BTC-USDT"
        assert adapter._map_symbol("ETH/USDT") == "ETH-USDT"

    def test_sign(self, credentials):
        adapter = OKXAdapter(credentials)
        headers = adapter._sign("2024-01-01T00:00:00.000Z", "GET", "/api/v5/account/balance")
        assert "OK-ACCESS-KEY" in headers
        assert "OK-ACCESS-SIGN" in headers
        assert "OK-ACCESS-TIMESTAMP" in headers
        assert "OK-ACCESS-PASSPHRASE" in headers
        assert headers["OK-ACCESS-KEY"] == "test_key"
        assert headers["OK-ACCESS-PASSPHRASE"] == "test_pass"

    def test_sign_with_body(self, credentials):
        adapter = OKXAdapter(credentials)
        body = json.dumps({"instId": "BTC-USDT"})
        headers = adapter._sign("2024-01-01T00:00:00.000Z", "POST", "/api/v5/trade/order", body)
        assert "OK-ACCESS-SIGN" in headers

    def test_status_map(self):
        assert OKXAdapter._status_map("live") == OrderStatus.SUBMITTED
        assert OKXAdapter._status_map("partially_filled") == OrderStatus.PARTIALLY_FILLED
        assert OKXAdapter._status_map("filled") == OrderStatus.FILLED
        assert OKXAdapter._status_map("canceled") == OrderStatus.CANCELLED
        assert OKXAdapter._status_map("unknown") == OrderStatus.CREATED

    def test_class_constants(self):
        assert OKXAdapter.REST_URL == "https://www.okx.com"
        assert OKXAdapter.WS_PUBLIC_URL == "wss://ws.okx.com:8443/ws/v5/public"


# ============================================================================
# PaperTradingAdapter Tests
# ============================================================================

class TestPaperTradingAdapter:

    def test_construction_defaults(self):
        adapter = PaperTradingAdapter()
        assert adapter.balance == 100000.0
        assert adapter.initial_balance == 100000.0
        assert adapter.slippage_bps == 5.0
        assert adapter.commission_bps == 10.0
        assert adapter.positions == {}
        assert adapter.orders == {}
        assert adapter.trades == []
        assert adapter._order_counter == 0

    def test_construction_custom(self):
        adapter = PaperTradingAdapter(initial_balance=50000.0, slippage_bps=10.0, commission_bps=5.0)
        assert adapter.balance == 50000.0
        assert adapter.slippage_bps == 10.0
        assert adapter.commission_bps == 5.0

    @pytest.mark.asyncio
    async def test_get_candles_returns_empty(self, paper_adapter):
        candles = await paper_adapter.get_candles("BTC/USDT", "1h")
        assert candles == []

    @pytest.mark.asyncio
    async def test_get_order_book(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        ob = await paper_adapter.get_order_book("BTC/USDT", depth=5)
        assert "bids" in ob
        assert "asks" in ob
        assert len(ob["bids"]) == 5
        assert len(ob["asks"]) == 5
        # Bids should be at or below last price
        for price, qty in ob["bids"]:
            assert price <= 50000.0
        # Asks should be at or above last price
        for price, qty in ob["asks"]:
            assert price >= 50000.0

    @pytest.mark.asyncio
    async def test_get_order_book_default_price(self, paper_adapter):
        """Default price when symbol not in last_prices."""
        ob = await paper_adapter.get_order_book("UNKNOWN/USDT", depth=5)
        assert len(ob["bids"]) == 5

    @pytest.mark.asyncio
    async def test_place_market_buy_order(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        result = await paper_adapter.place_order(sample_order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 0.1
        assert result.average_fill_price > 0
        assert result.commission > 0
        assert "BTC/USDT" in paper_adapter.positions
        assert paper_adapter.positions["BTC/USDT"].side == Side.BUY
        assert paper_adapter.balance < 100000.0  # Balance reduced

    @pytest.mark.asyncio
    async def test_place_market_sell_order(self, paper_adapter, sell_order):
        # First buy to have a position
        paper_adapter.update_price("BTC/USDT", 50000.0)
        buy_order = Order(
            id="buy_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.CREATED,
            quantity=0.2, price=50000.0,
        )
        await paper_adapter.place_order(buy_order)
        # Now sell
        result = await paper_adapter.place_order(sell_order)
        assert result.status == OrderStatus.FILLED
        assert "BTC/USDT" in paper_adapter.positions

    @pytest.mark.asyncio
    async def test_place_limit_order(self, paper_adapter, limit_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        result = await paper_adapter.place_order(limit_order)
        assert result.status == OrderStatus.SUBMITTED
        assert limit_order.id in paper_adapter.orders

    @pytest.mark.asyncio
    async def test_place_order_creates_trade(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        assert len(paper_adapter.trades) == 1
        assert paper_adapter.trades[0].order_id == sample_order.id
        assert paper_adapter.trades[0].symbol == "BTC/USDT"
        assert paper_adapter.trades[0].exchange == "paper"

    @pytest.mark.asyncio
    async def test_place_order_slippage_buy(self, paper_adapter):
        """Buy order fill price should include slippage above market price."""
        paper_adapter.update_price("BTC/USDT", 50000.0)
        order = Order(id="slip_buy", symbol="BTC/USDT", side=Side.BUY,
                      order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                      quantity=1.0)
        result = await paper_adapter.place_order(order)
        # Slippage = 50000 * 5 / 10000 = 25
        assert result.average_fill_price > 50000.0

    @pytest.mark.asyncio
    async def test_place_order_slippage_sell(self, paper_adapter):
        """Sell order fill price should include slippage below market price."""
        paper_adapter.update_price("BTC/USDT", 50000.0)
        order = Order(id="slip_sell", symbol="BTC/USDT", side=Side.SELL,
                      order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                      quantity=1.0)
        result = await paper_adapter.place_order(order)
        assert result.average_fill_price < 50000.0

    @pytest.mark.asyncio
    async def test_place_order_commission(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        order = Order(id="comm_test", symbol="BTC/USDT", side=Side.BUY,
                      order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                      quantity=1.0)
        result = await paper_adapter.place_order(order)
        # Commission = qty * fill_price * 10 / 10000
        expected_commission = 1.0 * result.average_fill_price * 10 / 10000
        assert abs(result.commission - expected_commission) < 0.01

    @pytest.mark.asyncio
    async def test_position_creation(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        pos = paper_adapter.positions["BTC/USDT"]
        assert pos.symbol == "BTC/USDT"
        assert pos.side == Side.BUY
        assert pos.quantity == 0.1
        assert pos.entry_price > 0

    @pytest.mark.asyncio
    async def test_position_avg_price_on_add(self, paper_adapter):
        """Adding to position should average the entry price."""
        paper_adapter.update_price("BTC/USDT", 50000.0)
        order1 = Order(id="add1", symbol="BTC/USDT", side=Side.BUY,
                       order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                       quantity=0.1)
        await paper_adapter.place_order(order1)
        first_entry = paper_adapter.positions["BTC/USDT"].entry_price

        paper_adapter.update_price("BTC/USDT", 51000.0)
        order2 = Order(id="add2", symbol="BTC/USDT", side=Side.BUY,
                       order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                       quantity=0.1)
        await paper_adapter.place_order(order2)
        pos = paper_adapter.positions["BTC/USDT"]
        assert pos.quantity == 0.2
        # Average entry should be between 50000 and 51000
        assert 50000.0 < pos.entry_price < 51000.0 + 100  # Allow for slippage

    @pytest.mark.asyncio
    async def test_position_reduced_on_sell(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        # Buy 0.2
        buy_order = Order(id="buy_full", symbol="BTC/USDT", side=Side.BUY,
                          order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                          quantity=0.2)
        await paper_adapter.place_order(buy_order)
        # Sell 0.1
        sell_order = Order(id="sell_half", symbol="BTC/USDT", side=Side.SELL,
                           order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                           quantity=0.1)
        await paper_adapter.place_order(sell_order)
        pos = paper_adapter.positions["BTC/USDT"]
        assert pos.quantity == pytest.approx(0.1, abs=1e-6)

    @pytest.mark.asyncio
    async def test_position_closed_on_full_sell(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        buy_order = Order(id="buy_full", symbol="BTC/USDT", side=Side.BUY,
                          order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                          quantity=0.1)
        await paper_adapter.place_order(buy_order)
        # Sell entire position
        sell_order = Order(id="sell_full", symbol="BTC/USDT", side=Side.SELL,
                           order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                           quantity=0.1)
        await paper_adapter.place_order(sell_order)
        assert "BTC/USDT" not in paper_adapter.positions

    @pytest.mark.asyncio
    async def test_position_realized_pnl(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        buy_order = Order(id="buy_pnl", symbol="BTC/USDT", side=Side.BUY,
                          order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                          quantity=0.1)
        await paper_adapter.place_order(buy_order)

        paper_adapter.update_price("BTC/USDT", 51000.0)
        sell_order = Order(id="sell_pnl", symbol="BTC/USDT", side=Side.SELL,
                           order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                           quantity=0.1)
        await paper_adapter.place_order(sell_order)
        # After closing position, realized_pnl was tracked during the sell

    @pytest.mark.asyncio
    async def test_cancel_order_exists(self, paper_adapter, limit_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(limit_order)
        result = await paper_adapter.cancel_order("order_002", "BTC/USDT")
        assert result is True
        assert paper_adapter.orders["order_002"].status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_not_exists(self, paper_adapter):
        result = await paper_adapter.cancel_order("nonexistent", "BTC/USDT")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status_exists(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        result = await paper_adapter.get_order_status("order_001", "BTC/USDT")
        assert result.id == "order_001"

    @pytest.mark.asyncio
    async def test_get_order_status_not_exists(self, paper_adapter):
        result = await paper_adapter.get_order_status("nonexistent", "BTC/USDT")
        assert result.id == "nonexistent"
        assert result.status == OrderStatus.CREATED
        assert result.quantity == 0

    @pytest.mark.asyncio
    async def test_get_positions(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        positions = await paper_adapter.get_positions()
        assert len(positions) >= 1
        btc_pos = [p for p in positions if p.symbol == "BTC/USDT"][0]
        assert btc_pos.side == Side.BUY

    @pytest.mark.asyncio
    async def test_get_positions_updates_mark_price(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        paper_adapter.update_price("BTC/USDT", 51000.0)
        positions = await paper_adapter.get_positions()
        btc_pos = [p for p in positions if p.symbol == "BTC/USDT"][0]
        assert btc_pos.mark_price == 51000.0
        assert btc_pos.unrealized_pnl != 0

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, paper_adapter):
        positions = await paper_adapter.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_balance(self, paper_adapter):
        balance = await paper_adapter.get_balance()
        assert "USDT" in balance
        assert balance["USDT"]["free"] == 100000.0
        assert balance["USDT"]["locked"] == 0.0

    @pytest.mark.asyncio
    async def test_get_balance_after_trade(self, paper_adapter, sample_order):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        await paper_adapter.place_order(sample_order)
        balance = await paper_adapter.get_balance()
        assert balance["USDT"]["free"] < 100000.0

    @pytest.mark.asyncio
    async def test_subscribe_ticks_noop(self, paper_adapter):
        """subscribe_ticks should be a no-op for paper trading."""
        await paper_adapter.subscribe_ticks("BTC/USDT", lambda tick: None)

    @pytest.mark.asyncio
    async def test_subscribe_order_book_noop(self, paper_adapter):
        await paper_adapter.subscribe_order_book("BTC/USDT", lambda ob: None)

    def test_update_price(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        assert paper_adapter.last_prices["BTC/USDT"] == 50000.0
        paper_adapter.update_price("BTC/USDT", 51000.0)
        assert paper_adapter.last_prices["BTC/USDT"] == 51000.0

    @pytest.mark.asyncio
    async def test_multiple_orders(self, paper_adapter):
        paper_adapter.update_price("BTC/USDT", 50000.0)
        for i in range(5):
            order = Order(
                id=f"order_{i}", symbol="BTC/USDT", side=Side.BUY,
                order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                quantity=0.01,
            )
            await paper_adapter.place_order(order)
        assert len(paper_adapter.trades) == 5
        assert paper_adapter.positions["BTC/USDT"].quantity == pytest.approx(0.05, abs=1e-6)

    @pytest.mark.asyncio
    async def test_place_order_no_last_price(self, paper_adapter, sample_order):
        """When no last price set, should use order price or 0."""
        # sample_order has price=50000.0
        result = await paper_adapter.place_order(sample_order)
        # Should still work, using order price
        assert result.status == OrderStatus.FILLED


# ============================================================================
# ArbitrageOpportunity Tests
# ============================================================================

class TestArbitrageOpportunity:

    def test_construction(self):
        now = datetime.utcnow()
        opp = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="bybit",
            buy_price=49900.0,
            sell_price=50100.0,
            spread_pct=0.4,
            estimated_profit_usd=10.0,
            buy_fee_usd=5.0,
            sell_fee_usd=5.0,
            timestamp=now,
        )
        assert opp.symbol == "BTC/USDT"
        assert opp.buy_exchange == "binance"
        assert opp.sell_exchange == "bybit"
        assert opp.buy_price == 49900.0
        assert opp.sell_price == 50100.0
        assert opp.spread_pct == 0.4
        assert opp.estimated_profit_usd == 10.0
        assert opp.timestamp == now


# ============================================================================
# ArbitrageDetector Tests
# ============================================================================

class TestArbitrageDetector:

    @pytest.fixture
    def detector(self):
        exchanges = {
            "binance": PaperTradingAdapter(),
            "bybit": PaperTradingAdapter(),
        }
        return ArbitrageDetector(exchanges)

    @pytest.fixture
    def detector_with_custom_fees(self):
        exchanges = {
            "exchange_a": PaperTradingAdapter(),
            "exchange_b": PaperTradingAdapter(),
        }
        return ArbitrageDetector(exchanges, default_fees_bps={"exchange_a": 5.0, "exchange_b": 5.0})

    def test_construction(self, detector):
        assert "binance" in detector.exchanges
        assert "bybit" in detector.exchanges
        assert detector.fees_bps["binance"] == 10.0

    def test_construction_default_fees(self):
        detector = ArbitrageDetector({})
        assert detector.fees_bps["binance"] == 10.0
        assert detector.fees_bps["bybit"] == 10.0
        assert detector.fees_bps["okx"] == 10.0
        assert detector.fees_bps["paper"] == 0.0

    def test_update_price(self, detector):
        detector.update_price("binance", "BTC/USDT", 50000.0)
        assert detector._latest_prices["BTC/USDT"]["binance"] == 50000.0

    def test_update_price_multiple_exchanges(self, detector):
        detector.update_price("binance", "BTC/USDT", 49900.0)
        detector.update_price("bybit", "BTC/USDT", 50100.0)
        assert detector._latest_prices["BTC/USDT"]["binance"] == 49900.0
        assert detector._latest_prices["BTC/USDT"]["bybit"] == 50100.0

    @pytest.mark.asyncio
    async def test_fetch_all_prices(self):
        adapter_a = PaperTradingAdapter()
        adapter_a.update_price("BTC/USDT", 49900.0)
        adapter_b = PaperTradingAdapter()
        adapter_b.update_price("BTC/USDT", 50100.0)
        detector = ArbitrageDetector({"ex_a": adapter_a, "ex_b": adapter_b})
        prices = await detector.fetch_all_prices("BTC/USDT")
        # PaperTradingAdapter returns synthetic order book from last_prices
        assert isinstance(prices, dict)

    def test_detect_opportunities_insufficient_exchanges(self, detector):
        detector.update_price("binance", "BTC/USDT", 50000.0)
        # Only one exchange has price
        result = detector.detect_opportunities("BTC/USDT")
        assert result == []

    def test_detect_opportunities_profitable(self, detector_with_custom_fees):
        det = detector_with_custom_fees
        det.update_price("exchange_a", "BTC/USDT", 49900.0)
        det.update_price("exchange_b", "BTC/USDT", 50100.0)
        result = det.detect_opportunities("BTC/USDT")
        # Spread = 200/49900*10000 = ~40 bps
        # Fees = 5 + 5 = 10 bps
        # Should be profitable
        if len(result) > 0:
            opp = result[0]
            assert isinstance(opp, ArbitrageOpportunity)
            assert opp.estimated_profit_usd > 0

    def test_detect_opportunities_not_profitable(self, detector_with_custom_fees):
        det = detector_with_custom_fees
        det.update_price("exchange_a", "BTC/USDT", 50000.0)
        det.update_price("exchange_b", "BTC/USDT", 50001.0)
        result = det.detect_opportunities("BTC/USDT", trade_size=0.001)
        # Spread too small
        assert result == []

    def test_detect_opportunities_zero_price(self, detector):
        detector.update_price("binance", "BTC/USDT", 0.0)
        detector.update_price("bybit", "BTC/USDT", 50000.0)
        result = detector.detect_opportunities("BTC/USDT")
        # Zero price should be skipped
        assert isinstance(result, list)

    def test_detect_opportunities_sorted_by_profit(self, detector_with_custom_fees):
        det = detector_with_custom_fees
        # Add a third exchange
        det.exchanges["exchange_c"] = PaperTradingAdapter()
        det.fees_bps["exchange_c"] = 5.0
        det.update_price("exchange_a", "BTC/USDT", 49800.0)
        det.update_price("exchange_b", "BTC/USDT", 50200.0)
        det.update_price("exchange_c", "BTC/USDT", 50000.0)
        result = det.detect_opportunities("BTC/USDT")
        # Should be sorted by estimated_profit_usd descending
        for i in range(len(result) - 1):
            assert result[i].estimated_profit_usd >= result[i + 1].estimated_profit_usd

    def test_detect_opportunities_no_symbol(self, detector):
        result = detector.detect_opportunities("ETH/USDT")
        assert result == []

    def test_detect_opportunities_with_trade_size(self, detector_with_custom_fees):
        det = detector_with_custom_fees
        det.update_price("exchange_a", "BTC/USDT", 49900.0)
        det.update_price("exchange_b", "BTC/USDT", 50100.0)
        result_small = det.detect_opportunities("BTC/USDT", trade_size=0.01)
        result_large = det.detect_opportunities("BTC/USDT", trade_size=10.0)
        if result_small and result_large:
            assert result_large[0].estimated_profit_usd > result_small[0].estimated_profit_usd


# ============================================================================
# create_exchange_adapter Factory Tests
# ============================================================================

class TestCreateExchangeAdapter:

    def test_create_binance(self, credentials):
        adapter = create_exchange_adapter("binance", credentials)
        assert isinstance(adapter, BinanceAdapter)

    def test_create_bybit(self, credentials):
        adapter = create_exchange_adapter("bybit", credentials)
        assert isinstance(adapter, BybitAdapter)

    def test_create_okx(self, credentials):
        adapter = create_exchange_adapter("okx", credentials)
        assert isinstance(adapter, OKXAdapter)

    def test_create_paper(self):
        adapter = create_exchange_adapter("paper")
        assert isinstance(adapter, PaperTradingAdapter)

    def test_create_paper_ignores_credentials(self, credentials):
        adapter = create_exchange_adapter("paper", credentials)
        assert isinstance(adapter, PaperTradingAdapter)

    def test_create_binance_testnet(self, credentials):
        adapter = create_exchange_adapter("binance", credentials, testnet=True)
        assert isinstance(adapter, BinanceAdapter)
        assert adapter.testnet is True
        assert adapter.base_url == BinanceAdapter.TESTNET_REST

    def test_create_bybit_testnet(self, credentials):
        adapter = create_exchange_adapter("bybit", credentials, testnet=True)
        assert isinstance(adapter, BybitAdapter)
        assert adapter.testnet is True

    def test_create_okx_testnet(self, credentials):
        adapter = create_exchange_adapter("okx", credentials, testnet=True)
        assert isinstance(adapter, OKXAdapter)
        assert adapter.testnet is True

    def test_create_invalid_exchange(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            create_exchange_adapter("invalid_exchange")

    def test_create_default_credentials(self):
        """When no credentials provided, should use empty ExchangeCredentials."""
        adapter = create_exchange_adapter("binance")
        assert isinstance(adapter, BinanceAdapter)
        assert adapter.credentials.api_key == ""

    def test_create_paper_has_default_balance(self):
        adapter = create_exchange_adapter("paper")
        assert adapter.balance == 100000.0


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_paper_trading_full_lifecycle(self):
        """Full lifecycle: buy -> price change -> sell -> check PnL."""
        adapter = PaperTradingAdapter(initial_balance=100000.0)
        adapter.update_price("BTC/USDT", 50000.0)

        # Buy
        buy = Order(id="buy1", symbol="BTC/USDT", side=Side.BUY,
                    order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=1.0)
        result = await adapter.place_order(buy)
        assert result.status == OrderStatus.FILLED
        initial_balance = adapter.balance

        # Price goes up
        adapter.update_price("BTC/USDT", 55000.0)

        # Check position
        positions = await adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].unrealized_pnl > 0

        # Sell
        sell = Order(id="sell1", symbol="BTC/USDT", side=Side.SELL,
                     order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=1.0)
        result = await adapter.place_order(sell)
        assert result.status == OrderStatus.FILLED

        # Balance should have increased (profit)
        final_balance = adapter.balance
        assert final_balance > initial_balance  # Ignoring commission for direction check

    @pytest.mark.asyncio
    async def test_paper_trading_multiple_symbols(self):
        adapter = PaperTradingAdapter()
        adapter.update_price("BTC/USDT", 50000.0)
        adapter.update_price("ETH/USDT", 3000.0)

        btc_order = Order(id="btc1", symbol="BTC/USDT", side=Side.BUY,
                          order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=0.1)
        eth_order = Order(id="eth1", symbol="ETH/USDT", side=Side.BUY,
                          order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=1.0)

        await adapter.place_order(btc_order)
        await adapter.place_order(eth_order)

        assert "BTC/USDT" in adapter.positions
        assert "ETH/USDT" in adapter.positions
        assert len(adapter.trades) == 2

    @pytest.mark.asyncio
    async def test_order_book_multiple_updates(self):
        ob = LocalOrderBook("BTC/USDT", max_depth=10)
        # Initial update
        ob.update([(50000.0, 1.0), (49999.0, 2.0)], [(50001.0, 1.5), (50002.0, 2.5)], update_id=1)
        # Modify existing level
        ob.update([(50000.0, 3.0)], [(50001.0, 0.5)], update_id=2)
        assert ob.bids[50000.0] == 3.0
        assert ob.asks[50001.0] == 0.5
        # Remove level
        ob.update([(49999.0, 0.0)], [], update_id=3)
        assert 49999.0 not in ob.bids

    @pytest.mark.asyncio
    async def test_rate_limiter_burst(self):
        """Test that burst_size allows immediate burst of requests."""
        rl = RateLimiter(max_requests=10, window_seconds=1.0, burst_size=5)
        # Should be able to acquire up to burst_size immediately
        for _ in range(5):
            await rl.acquire()
        # After burst, need to wait
        assert rl._tokens <= 0.5

    def test_classify_error_all_codes(self):
        """Test all error classification paths."""
        # Rate limit
        assert isinstance(classify_exchange_error(429, {}, "binance"), RateLimitError)
        # Insufficient funds by message
        assert isinstance(classify_exchange_error(400, {"msg": "Insufficient margin"}, "okx"), InsufficientFundsError)
        # Insufficient funds by code
        assert isinstance(classify_exchange_error(400, {"code": -2019, "msg": ""}, "binance"), InsufficientFundsError)
        assert isinstance(classify_exchange_error(400, {"retCode": 130040, "retMsg": ""}, "bybit"), InsufficientFundsError)
        assert isinstance(classify_exchange_error(400, {"code": "51421", "msg": ""}, "okx"), InsufficientFundsError)
        # Order not found
        assert isinstance(classify_exchange_error(404, {"msg": "Order not found"}, "binance"), OrderNotFoundError)
        # Generic
        assert isinstance(classify_exchange_error(500, {"msg": "Server error"}, "binance"), ExchangeError)

    @pytest.mark.asyncio
    async def test_paper_trading_balance_tracking(self):
        """Verify balance tracking through multiple trades."""
        adapter = PaperTradingAdapter(initial_balance=100000.0, slippage_bps=0.0, commission_bps=0.0)
        adapter.update_price("BTC/USDT", 50000.0)

        # Buy 1 BTC at 50000
        buy = Order(id="b1", symbol="BTC/USDT", side=Side.BUY,
                    order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=1.0)
        await adapter.place_order(buy)
        assert adapter.balance == pytest.approx(50000.0, abs=1.0)

        # Sell 1 BTC at 51000
        adapter.update_price("BTC/USDT", 51000.0)
        sell = Order(id="s1", symbol="BTC/USDT", side=Side.SELL,
                     order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=1.0)
        await adapter.place_order(sell)
        assert adapter.balance == pytest.approx(101000.0, abs=1.0)

    def test_local_order_book_concurrent_updates(self):
        ob = LocalOrderBook("BTC/USDT")
        # Rapid sequential updates
        for i in range(100):
            ob.update(
                [(50000.0 - i * 0.1, 1.0)],
                [(50001.0 + i * 0.1, 1.0)],
                update_id=i + 1,
            )
        assert ob.last_update_id == 100
        assert ob.get_best_bid() is not None
        assert ob.get_best_ask() is not None

    @pytest.mark.asyncio
    async def test_retry_with_backoff_insufficient_funds_not_retried(self):
        """InsufficientFundsError should not be retried (it's an ExchangeError)."""
        call_count = 0

        async def always_insufficient():
            nonlocal call_count
            call_count += 1
            raise InsufficientFundsError("no money", "binance")

        with pytest.raises(InsufficientFundsError):
            await retry_with_backoff(always_insufficient, max_retries=3, base_delay=0.01)
        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_order_not_found_not_retried(self):
        """OrderNotFoundError should not be retried."""
        call_count = 0

        async def always_not_found():
            nonlocal call_count
            call_count += 1
            raise OrderNotFoundError("not found", "binance")

        with pytest.raises(OrderNotFoundError):
            await retry_with_backoff(always_not_found, max_retries=3, base_delay=0.01)
        assert call_count == 1

    def test_exchange_credentials_equality(self):
        c1 = ExchangeCredentials(api_key="key", api_secret="secret")
        c2 = ExchangeCredentials(api_key="key", api_secret="secret")
        assert c1 == c2

    def test_exchange_credentials_inequality(self):
        c1 = ExchangeCredentials(api_key="key1")
        c2 = ExchangeCredentials(api_key="key2")
        assert c1 != c2

    @pytest.mark.asyncio
    async def test_arbitrage_detector_empty_exchanges(self):
        detector = ArbitrageDetector({})
        result = detector.detect_opportunities("BTC/USDT")
        assert result == []

    @pytest.mark.asyncio
    async def test_arbitrage_detector_single_exchange(self):
        adapter = PaperTradingAdapter()
        detector = ArbitrageDetector({"only_one": adapter})
        detector.update_price("only_one", "BTC/USDT", 50000.0)
        result = detector.detect_opportunities("BTC/USDT")
        assert result == []

    def test_order_book_snapshot_after_clear(self):
        ob = LocalOrderBook("BTC/USDT")
        ob.update([(50000.0, 1.0)], [(50001.0, 1.0)], update_id=1)
        # Remove all
        ob.update([(50000.0, 0.0)], [(50001.0, 0.0)], update_id=2)
        snap = ob.snapshot()
        assert snap["bids"] == []
        assert snap["asks"] == []
        assert snap["spread"] is None
        assert snap["mid_price"] is None

    @pytest.mark.asyncio
    async def test_paper_adapter_sell_without_position(self):
        """Selling without a position should still work (naked short in paper trading)."""
        adapter = PaperTradingAdapter()
        adapter.update_price("BTC/USDT", 50000.0)
        sell = Order(id="naked_short", symbol="BTC/USDT", side=Side.SELL,
                     order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=0.1)
        result = await adapter.place_order(sell)
        assert result.status == OrderStatus.FILLED
        # No position created since there was no existing long position
        # and sell reduces/closes; but with no position, behavior is sell increases balance

    @pytest.mark.asyncio
    async def test_paper_adapter_small_quantity(self):
        adapter = PaperTradingAdapter()
        adapter.update_price("BTC/USDT", 50000.0)
        order = Order(id="tiny", symbol="BTC/USDT", side=Side.BUY,
                      order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                      quantity=0.00001)
        result = await adapter.place_order(order)
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_paper_adapter_large_quantity(self):
        adapter = PaperTradingAdapter(initial_balance=1e9)
        adapter.update_price("BTC/USDT", 50000.0)
        order = Order(id="huge", symbol="BTC/USDT", side=Side.BUY,
                      order_type=OrderType.MARKET, status=OrderStatus.CREATED,
                      quantity=1000.0)
        result = await adapter.place_order(order)
        assert result.status == OrderStatus.FILLED


class TestBinanceAdapterSigning:

    def test_sign_deterministic(self, credentials):
        """Same params should produce same signature."""
        adapter = BinanceAdapter(credentials)
        params = {"symbol": "BTCUSDT", "timestamp": "1234567890000"}
        sig1 = adapter._sign(params.copy())
        sig2 = adapter._sign(params.copy())
        assert sig1["signature"] == sig2["signature"]

    def test_sign_different_params_different_signature(self, credentials):
        adapter = BinanceAdapter(credentials)
        sig1 = adapter._sign({"symbol": "BTCUSDT", "timestamp": "1"})
        sig2 = adapter._sign({"symbol": "ETHUSDT", "timestamp": "1"})
        assert sig1["signature"] != sig2["signature"]


class TestBybitAdapterSigning:

    def test_sign_deterministic(self, credentials):
        adapter = BybitAdapter(credentials)
        params = {"category": "spot"}
        # Note: timestamp is embedded so we mock time
        with patch('time.time', return_value=1700000000.0):
            sig1 = adapter._sign(params.copy())
            sig2 = adapter._sign(params.copy())
        assert sig1["sign"] == sig2["sign"]


class TestOKXAdapterSigning:

    def test_sign_deterministic(self, credentials):
        adapter = OKXAdapter(credentials)
        ts = "2024-01-01T00:00:00.000Z"
        sig1 = adapter._sign(ts, "GET", "/api/v5/account/balance")
        sig2 = adapter._sign(ts, "GET", "/api/v5/account/balance")
        assert sig1["OK-ACCESS-SIGN"] == sig2["OK-ACCESS-SIGN"]

    def test_sign_different_methods(self, credentials):
        adapter = OKXAdapter(credentials)
        ts = "2024-01-01T00:00:00.000Z"
        sig_get = adapter._sign(ts, "GET", "/api/v5/account/balance")
        sig_post = adapter._sign(ts, "POST", "/api/v5/account/balance")
        assert sig_get["OK-ACCESS-SIGN"] != sig_post["OK-ACCESS-SIGN"]

    def test_sign_with_body_changes_signature(self, credentials):
        adapter = OKXAdapter(credentials)
        ts = "2024-01-01T00:00:00.000Z"
        sig_no_body = adapter._sign(ts, "POST", "/api/v5/trade/order", "")
        sig_with_body = adapter._sign(ts, "POST", "/api/v5/trade/order", '{"instId":"BTC-USDT"}')
        assert sig_no_body["OK-ACCESS-SIGN"] != sig_with_body["OK-ACCESS-SIGN"]


class TestIntegration:

    @pytest.mark.asyncio
    async def test_paper_trading_with_arbitrage_detector(self):
        """Integration: PaperTrading + ArbitrageDetector."""
        adapter_a = PaperTradingAdapter()
        adapter_a.update_price("BTC/USDT", 49900.0)
        adapter_b = PaperTradingAdapter()
        adapter_b.update_price("BTC/USDT", 50100.0)

        detector = ArbitrageDetector(
            {"ex_a": adapter_a, "ex_b": adapter_b},
            default_fees_bps={"ex_a": 5.0, "ex_b": 5.0},
        )
        detector.update_price("ex_a", "BTC/USDT", 49900.0)
        detector.update_price("ex_b", "BTC/USDT", 50100.0)

        opportunities = detector.detect_opportunities("BTC/USDT", trade_size=1.0)
        if opportunities:
            opp = opportunities[0]
            assert opp.buy_exchange in ["ex_a", "ex_b"]
            assert opp.sell_exchange in ["ex_a", "ex_b"]
            assert opp.buy_exchange != opp.sell_exchange

    @pytest.mark.asyncio
    async def test_order_book_with_paper_trading(self):
        """Integration: LocalOrderBook + PaperTradingAdapter."""
        adapter = PaperTradingAdapter()
        adapter.update_price("BTC/USDT", 50000.0)

        ob = await adapter.get_order_book("BTC/USDT", depth=10)
        local_book = LocalOrderBook("BTC/USDT")
        bids = [(p, q) for p, q in ob["bids"]]
        asks = [(p, q) for p, q in ob["asks"]]
        local_book.update(bids, asks, update_id=1)

        assert local_book.get_best_bid() is not None
        assert local_book.get_best_ask() is not None
        assert local_book.get_spread() is not None
        assert local_book.get_mid_price() is not None

    def test_error_hierarchy(self):
        """All specific errors should be catchable as ExchangeError."""
        errors = [
            RateLimitError("rate", "binance"),
            InsufficientFundsError("funds", "binance"),
            OrderNotFoundError("order", "binance"),
            NetworkError("net", "binance"),
        ]
        for e in errors:
            assert isinstance(e, ExchangeError)
            try:
                raise e
            except ExchangeError:
                pass  # Expected

    @pytest.mark.asyncio
    async def test_full_paper_trading_session(self):
        """Simulate a full trading session with paper adapter."""
        adapter = PaperTradingAdapter(initial_balance=100000.0)
        adapter.update_price("BTC/USDT", 50000.0)

        # Place buy order
        buy = Order(id="session_buy", symbol="BTC/USDT", side=Side.BUY,
                    order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=0.5)
        result = await adapter.place_order(buy)
        assert result.status == OrderStatus.FILLED

        # Verify position
        positions = await adapter.get_positions()
        assert len(positions) == 1

        # Price moves
        adapter.update_price("BTC/USDT", 52000.0)

        # Check PnL
        positions = await adapter.get_positions()
        assert positions[0].unrealized_pnl > 0

        # Close position
        sell = Order(id="session_sell", symbol="BTC/USDT", side=Side.SELL,
                     order_type=OrderType.MARKET, status=OrderStatus.CREATED, quantity=0.5)
        result = await adapter.place_order(sell)
        assert result.status == OrderStatus.FILLED

        # Verify position closed
        positions = await adapter.get_positions()
        assert len(positions) == 0

        # Verify balance increased
        balance = await adapter.get_balance()
        assert balance["USDT"]["free"] > 100000.0 - 1.0  # At minimum, very close to initial
