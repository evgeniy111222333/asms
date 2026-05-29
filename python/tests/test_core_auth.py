"""Exhaustive tests for acms.core and acms.auth modules.

Tests every class, method, property, edge case, boundary condition, and error path
in acms/core/__init__.py and acms/auth/__init__.py.
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import hashlib
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from jose import jwt

from acms.core import (
    Side,
    OrderType,
    OrderStatus,
    TimeInForce,
    ExchangeId,
    Timeframe,
    SignalDirection,
    RiskDecision,
    Symbol,
    Candle,
    Tick,
    Signal,
    Position,
    Order,
    Trade,
    PortfolioSnapshot,
    RiskCheckResult,
    ExecutionReport,
    ACMSConfig,
)
from acms.auth import TokenData, AuthManager


# ============================================================================
# Helper fixtures
# ============================================================================

@pytest.fixture
def now():
    """Return a fixed datetime for testing."""
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def sample_symbol():
    """Return a sample Symbol."""
    return Symbol(base="BTC", quote="USDT")


@pytest.fixture
def sample_candle(now):
    """Return a sample bullish Candle."""
    return Candle(
        symbol="BTC/USDT",
        timeframe="1h",
        open_time=now,
        close_time=now + timedelta(hours=1),
        open=40000.0,
        high=41000.0,
        low=39500.0,
        close=40800.0,
        volume=100.0,
    )


@pytest.fixture
def sample_tick(now):
    """Return a sample Tick."""
    return Tick(
        symbol="BTC/USDT",
        exchange="binance",
        price=40000.0,
        quantity=0.5,
        side=Side.BUY,
        timestamp=now,
    )


@pytest.fixture
def sample_signal(now):
    """Return a sample Signal."""
    return Signal(
        id="sig_001",
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
        strength=0.85,
        strategy_id="strat_ma_cross",
        timestamp=now,
    )


@pytest.fixture
def sample_position():
    """Return a sample Position."""
    return Position(
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=1.0,
        entry_price=40000.0,
        mark_price=41000.0,
    )


@pytest.fixture
def sample_order(now):
    """Return a sample Order."""
    return Order(
        id="ord_001",
        symbol="BTC/USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.CREATED,
        quantity=1.0,
        price=40000.0,
        created_at=now,
    )


@pytest.fixture
def sample_trade(now):
    """Return a sample Trade."""
    return Trade(
        id="trade_001",
        order_id="ord_001",
        symbol="BTC/USDT",
        side=Side.BUY,
        quantity=0.5,
        price=40000.0,
        commission=10.0,
        timestamp=now,
    )


@pytest.fixture
def auth_manager():
    """Return an AuthManager with default params."""
    return AuthManager()


@pytest.fixture
def custom_auth_manager():
    """Return an AuthManager with custom params."""
    return AuthManager(secret_key="my-custom-secret", algorithm="HS384", expiry_hours=48)


# ============================================================================
# Enum Tests: Side
# ============================================================================

class TestSideEnum:
    """Tests for the Side enum."""

    def test_side_buy_value(self):
        assert Side.BUY.value == "buy"

    def test_side_sell_value(self):
        assert Side.SELL.value == "sell"

    def test_side_is_str(self):
        assert isinstance(Side.BUY, str)
        assert isinstance(Side.SELL, str)

    def test_side_string_comparison(self):
        assert Side.BUY == "buy"
        assert Side.SELL == "sell"

    def test_side_iteration(self):
        values = list(Side)
        assert len(values) == 2
        assert Side.BUY in values
        assert Side.SELL in values

    def test_side_from_value(self):
        assert Side("buy") is Side.BUY
        assert Side("sell") is Side.SELL

    def test_side_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Side("invalid")

    def test_side_name(self):
        assert Side.BUY.name == "BUY"
        assert Side.SELL.name == "SELL"

    def test_side_members(self):
        members = Side.__members__
        assert "BUY" in members
        assert "SELL" in members
        assert len(members) == 2


# ============================================================================
# Enum Tests: OrderType
# ============================================================================

class TestOrderTypeEnum:
    """Tests for the OrderType enum."""

    def test_market_value(self):
        assert OrderType.MARKET.value == "market"

    def test_limit_value(self):
        assert OrderType.LIMIT.value == "limit"

    def test_stop_value(self):
        assert OrderType.STOP.value == "stop"

    def test_stop_limit_value(self):
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_trailing_stop_value(self):
        assert OrderType.TRAILING_STOP.value == "trailing_stop"

    def test_iceberg_value(self):
        assert OrderType.ICEBERG.value == "iceberg"

    def test_twap_value(self):
        assert OrderType.TWAP.value == "twap"

    def test_vwap_value(self):
        assert OrderType.VWAP.value == "vwap"

    def test_order_type_count(self):
        assert len(list(OrderType)) == 8

    def test_order_type_is_str(self):
        for ot in OrderType:
            assert isinstance(ot, str)

    def test_order_type_iteration(self):
        values = list(OrderType)
        assert len(values) == 8

    def test_order_type_from_value(self):
        assert OrderType("market") is OrderType.MARKET
        assert OrderType("limit") is OrderType.LIMIT

    def test_order_type_invalid_value(self):
        with pytest.raises(ValueError):
            OrderType("invalid_type")

    def test_order_type_names(self):
        expected_names = {"MARKET", "LIMIT", "STOP", "STOP_LIMIT",
                          "TRAILING_STOP", "ICEBERG", "TWAP", "VWAP"}
        actual_names = {m.name for m in OrderType}
        assert actual_names == expected_names


# ============================================================================
# Enum Tests: OrderStatus
# ============================================================================

class TestOrderStatusEnum:
    """Tests for the OrderStatus enum."""

    def test_created_value(self):
        assert OrderStatus.CREATED.value == "created"

    def test_validated_value(self):
        assert OrderStatus.VALIDATED.value == "validated"

    def test_submitted_value(self):
        assert OrderStatus.SUBMITTED.value == "submitted"

    def test_partially_filled_value(self):
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"

    def test_filled_value(self):
        assert OrderStatus.FILLED.value == "filled"

    def test_cancelled_value(self):
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_rejected_value(self):
        assert OrderStatus.REJECTED.value == "rejected"

    def test_expired_value(self):
        assert OrderStatus.EXPIRED.value == "expired"

    def test_order_status_count(self):
        assert len(list(OrderStatus)) == 8

    def test_order_status_from_value(self):
        assert OrderStatus("created") is OrderStatus.CREATED
        assert OrderStatus("filled") is OrderStatus.FILLED

    def test_order_status_invalid_value(self):
        with pytest.raises(ValueError):
            OrderStatus("pending")

    def test_order_status_is_str(self):
        for os in OrderStatus:
            assert isinstance(os, str)


# ============================================================================
# Enum Tests: TimeInForce
# ============================================================================

class TestTimeInForceEnum:
    """Tests for the TimeInForce enum."""

    def test_gtc_value(self):
        assert TimeInForce.GTC.value == "gtc"

    def test_ioc_value(self):
        assert TimeInForce.IOC.value == "ioc"

    def test_fok_value(self):
        assert TimeInForce.FOK.value == "fok"

    def test_gtd_value(self):
        assert TimeInForce.GTD.value == "gtd"

    def test_day_value(self):
        assert TimeInForce.DAY.value == "day"

    def test_time_in_force_count(self):
        assert len(list(TimeInForce)) == 5

    def test_time_in_force_from_value(self):
        assert TimeInForce("gtc") is TimeInForce.GTC

    def test_time_in_force_invalid_value(self):
        with pytest.raises(ValueError):
            TimeInForce("invalid")


# ============================================================================
# Enum Tests: ExchangeId
# ============================================================================

class TestExchangeIdEnum:
    """Tests for the ExchangeId enum."""

    def test_binance_value(self):
        assert ExchangeId.BINANCE.value == "binance"

    def test_bybit_value(self):
        assert ExchangeId.BYBIT.value == "bybit"

    def test_okx_value(self):
        assert ExchangeId.OKX.value == "okx"

    def test_paper_value(self):
        assert ExchangeId.PAPER.value == "paper"

    def test_exchange_id_count(self):
        assert len(list(ExchangeId)) == 4

    def test_exchange_id_from_value(self):
        assert ExchangeId("binance") is ExchangeId.BINANCE

    def test_exchange_id_invalid_value(self):
        with pytest.raises(ValueError):
            ExchangeId("coinbase")


# ============================================================================
# Enum Tests: Timeframe
# ============================================================================

class TestTimeframeEnum:
    """Tests for the Timeframe enum."""

    def test_s1_value(self):
        assert Timeframe.S1.value == "1s"

    def test_s5_value(self):
        assert Timeframe.S5.value == "5s"

    def test_s15_value(self):
        assert Timeframe.S15.value == "15s"

    def test_s30_value(self):
        assert Timeframe.S30.value == "30s"

    def test_m1_value(self):
        assert Timeframe.M1.value == "1m"

    def test_m5_value(self):
        assert Timeframe.M5.value == "5m"

    def test_m15_value(self):
        assert Timeframe.M15.value == "15m"

    def test_m30_value(self):
        assert Timeframe.M30.value == "30m"

    def test_h1_value(self):
        assert Timeframe.H1.value == "1h"

    def test_h4_value(self):
        assert Timeframe.H4.value == "4h"

    def test_d1_value(self):
        assert Timeframe.D1.value == "1d"

    def test_w1_value(self):
        assert Timeframe.W1.value == "1w"

    def test_timeframe_count(self):
        assert len(list(Timeframe)) == 12

    def test_timeframe_from_value(self):
        assert Timeframe("1h") is Timeframe.H1

    def test_timeframe_invalid_value(self):
        with pytest.raises(ValueError):
            Timeframe("2h")

    def test_timeframe_is_str(self):
        for tf in Timeframe:
            assert isinstance(tf, str)


# ============================================================================
# Enum Tests: SignalDirection
# ============================================================================

class TestSignalDirectionEnum:
    """Tests for the SignalDirection enum."""

    def test_long_value(self):
        assert SignalDirection.LONG.value == "long"

    def test_short_value(self):
        assert SignalDirection.SHORT.value == "short"

    def test_neutral_value(self):
        assert SignalDirection.NEUTRAL.value == "neutral"

    def test_signal_direction_count(self):
        assert len(list(SignalDirection)) == 3

    def test_signal_direction_from_value(self):
        assert SignalDirection("long") is SignalDirection.LONG

    def test_signal_direction_invalid_value(self):
        with pytest.raises(ValueError):
            SignalDirection("up")


# ============================================================================
# Enum Tests: RiskDecision
# ============================================================================

class TestRiskDecisionEnum:
    """Tests for the RiskDecision enum."""

    def test_allow_value(self):
        assert RiskDecision.ALLOW.value == "allow"

    def test_reject_value(self):
        assert RiskDecision.REJECT.value == "reject"

    def test_throttle_value(self):
        assert RiskDecision.THROTTLE.value == "throttle"

    def test_risk_decision_count(self):
        assert len(list(RiskDecision)) == 3

    def test_risk_decision_from_value(self):
        assert RiskDecision("allow") is RiskDecision.ALLOW

    def test_risk_decision_invalid_value(self):
        with pytest.raises(ValueError):
            RiskDecision("deny")


# ============================================================================
# Dataclass Tests: Symbol
# ============================================================================

class TestSymbol:
    """Tests for the Symbol dataclass."""

    def test_construction_with_base_and_quote(self):
        s = Symbol(base="ETH", quote="BTC")
        assert s.base == "ETH"
        assert s.quote == "BTC"

    def test_default_quote_is_usdt(self):
        s = Symbol(base="BTC")
        assert s.quote == "USDT"

    def test_pair_property(self):
        s = Symbol(base="BTC", quote="USDT")
        assert s.pair == "BTC/USDT"

    def test_pair_property_custom_quote(self):
        s = Symbol(base="ETH", quote="BTC")
        assert s.pair == "ETH/BTC"

    def test_str_returns_pair(self):
        s = Symbol(base="BTC", quote="USDT")
        assert str(s) == "BTC/USDT"

    def test_hash_is_based_on_pair(self):
        s1 = Symbol(base="BTC", quote="USDT")
        s2 = Symbol(base="BTC", quote="USDT")
        assert hash(s1) == hash(s2)

    def test_hash_different_symbols(self):
        s1 = Symbol(base="BTC", quote="USDT")
        s2 = Symbol(base="ETH", quote="USDT")
        # Different symbols should (generally) have different hashes
        assert hash(s1) != hash(s2)

    def test_hash_usable_in_set(self):
        s1 = Symbol(base="BTC", quote="USDT")
        s2 = Symbol(base="BTC", quote="USDT")
        s3 = Symbol(base="ETH", quote="USDT")
        symbol_set = {s1, s2, s3}
        assert len(symbol_set) == 2

    def test_hash_usable_as_dict_key(self):
        s = Symbol(base="BTC", quote="USDT")
        d = {s: 100.0}
        assert d[s] == 100.0

    def test_equality_same_pair(self):
        s1 = Symbol(base="BTC", quote="USDT")
        s2 = Symbol(base="BTC", quote="USDT")
        assert s1 == s2

    def test_equality_different_pair(self):
        s1 = Symbol(base="BTC", quote="USDT")
        s2 = Symbol(base="ETH", quote="USDT")
        assert s1 != s2

    def test_empty_base(self):
        s = Symbol(base="", quote="USDT")
        assert s.pair == "/USDT"

    def test_empty_quote(self):
        s = Symbol(base="BTC", quote="")
        assert s.pair == "BTC/"

    def test_symbol_with_special_chars(self):
        s = Symbol(base="BTC-PERP", quote="USD")
        assert s.pair == "BTC-PERP/USD"


# ============================================================================
# Dataclass Tests: Candle
# ============================================================================

class TestCandle:
    """Tests for the Candle dataclass."""

    def test_construction_basic(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.symbol == "BTC/USDT"
        assert c.timeframe == "1h"
        assert c.open == 40000.0
        assert c.high == 41000.0
        assert c.low == 39500.0
        assert c.close == 40800.0
        assert c.volume == 100.0

    def test_default_quote_volume(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.quote_volume == 0.0

    def test_default_trades(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.trades == 0

    def test_default_taker_buy_volume(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.taker_buy_volume == 0.0

    def test_default_taker_buy_quote_volume(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.taker_buy_quote_volume == 0.0

    def test_typical_price(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        expected = (41000.0 + 39500.0 + 40800.0) / 3.0
        assert c.typical_price == pytest.approx(expected)

    def test_typical_price_zero_values(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=0.0, high=0.0, low=0.0, close=0.0,
            volume=0.0,
        )
        assert c.typical_price == 0.0

    def test_range_bullish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.range == pytest.approx(1500.0)

    def test_range_zero_when_high_equals_low(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=40000.0, low=40000.0, close=40000.0,
            volume=100.0,
        )
        assert c.range == 0.0

    def test_body_bullish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.body == pytest.approx(800.0)

    def test_body_bearish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40800.0, high=41000.0, low=39500.0, close=40000.0,
            volume=100.0,
        )
        assert c.body == pytest.approx(800.0)

    def test_body_doji(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=40100.0, low=39900.0, close=40000.0,
            volume=100.0,
        )
        assert c.body == 0.0

    def test_is_bullish_true(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        assert c.is_bullish is True

    def test_is_bullish_false_bearish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40800.0, high=41000.0, low=39500.0, close=40000.0,
            volume=100.0,
        )
        assert c.is_bullish is False

    def test_is_bullish_false_doji(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=40100.0, low=39900.0, close=40000.0,
            volume=100.0,
        )
        assert c.is_bullish is False

    def test_upper_wick_bullish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        # max(open, close) = 40800, upper_wick = 41000 - 40800 = 200
        assert c.upper_wick == pytest.approx(200.0)

    def test_upper_wick_bearish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40800.0, high=41000.0, low=39500.0, close=40000.0,
            volume=100.0,
        )
        # max(open, close) = 40800, upper_wick = 41000 - 40800 = 200
        assert c.upper_wick == pytest.approx(200.0)

    def test_upper_wick_zero(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=41000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        # max(open, close) = 41000, upper_wick = 41000 - 41000 = 0
        assert c.upper_wick == 0.0

    def test_lower_wick_bullish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        # min(open, close) = 40000, lower_wick = 40000 - 39500 = 500
        assert c.lower_wick == pytest.approx(500.0)

    def test_lower_wick_bearish(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40800.0, high=41000.0, low=39500.0, close=40000.0,
            volume=100.0,
        )
        # min(open, close) = 40000, lower_wick = 40000 - 39500 = 500
        assert c.lower_wick == pytest.approx(500.0)

    def test_lower_wick_zero(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=39500.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0,
        )
        # min(open, close) = 39500, lower_wick = 39500 - 39500 = 0
        assert c.lower_wick == 0.0

    def test_candle_equal_ohlc(self, now):
        """All OHLC equal - doji with no wicks."""
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=40000.0, low=40000.0, close=40000.0,
            volume=0.0,
        )
        assert c.typical_price == 40000.0
        assert c.range == 0.0
        assert c.body == 0.0
        assert c.is_bullish is False
        assert c.upper_wick == 0.0
        assert c.lower_wick == 0.0

    def test_candle_with_all_optional_fields(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=100.0, quote_volume=4050000.0, trades=500,
            taker_buy_volume=60.0, taker_buy_quote_volume=2430000.0,
        )
        assert c.quote_volume == 4050000.0
        assert c.trades == 500
        assert c.taker_buy_volume == 60.0
        assert c.taker_buy_quote_volume == 2430000.0

    def test_candle_negative_volume(self, now):
        """Edge case: negative volume (malformed data)."""
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=41000.0, low=39500.0, close=40800.0,
            volume=-100.0,
        )
        assert c.volume == -100.0

    def test_upper_wick_when_close_equals_high(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=39500.0, high=40800.0, low=39400.0, close=40800.0,
            volume=100.0,
        )
        # max(39500, 40800) = 40800, upper_wick = 40800 - 40800 = 0
        assert c.upper_wick == 0.0

    def test_lower_wick_when_open_equals_low(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=39500.0, high=40800.0, low=39500.0, close=40000.0,
            volume=100.0,
        )
        # min(39500, 40000) = 39500, lower_wick = 39500 - 39500 = 0
        assert c.lower_wick == 0.0

    def test_very_small_price_difference(self, now):
        c = Candle(
            symbol="BTC/USDT", timeframe="1h",
            open_time=now, close_time=now + timedelta(hours=1),
            open=40000.0, high=40000.0001, low=39999.9999, close=40000.00005,
            volume=1.0,
        )
        assert c.is_bullish is True
        assert c.range == pytest.approx(0.0002)
        assert c.body == pytest.approx(0.00005)


# ============================================================================
# Dataclass Tests: Tick
# ============================================================================

class TestTick:
    """Tests for the Tick dataclass."""

    def test_construction(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="binance",
            price=40000.0, quantity=0.5, side=Side.BUY, timestamp=now,
        )
        assert t.symbol == "BTC/USDT"
        assert t.exchange == "binance"
        assert t.price == 40000.0
        assert t.quantity == 0.5
        assert t.side == Side.BUY
        assert t.timestamp == now

    def test_default_trade_id(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="binance",
            price=40000.0, quantity=0.5, side=Side.BUY, timestamp=now,
        )
        assert t.trade_id == ""

    def test_custom_trade_id(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="binance",
            price=40000.0, quantity=0.5, side=Side.BUY,
            timestamp=now, trade_id="trade_123",
        )
        assert t.trade_id == "trade_123"

    def test_sell_side(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="bybit",
            price=40000.0, quantity=1.0, side=Side.SELL, timestamp=now,
        )
        assert t.side == Side.SELL

    def test_zero_quantity(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="binance",
            price=40000.0, quantity=0.0, side=Side.BUY, timestamp=now,
        )
        assert t.quantity == 0.0

    def test_zero_price(self, now):
        t = Tick(
            symbol="BTC/USDT", exchange="binance",
            price=0.0, quantity=1.0, side=Side.BUY, timestamp=now,
        )
        assert t.price == 0.0


# ============================================================================
# Dataclass Tests: Signal
# ============================================================================

class TestSignal:
    """Tests for the Signal dataclass."""

    def test_construction(self, now):
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross", timestamp=now,
        )
        assert s.id == "sig_001"
        assert s.symbol == "BTC/USDT"
        assert s.direction == SignalDirection.LONG
        assert s.strength == 0.85
        assert s.strategy_id == "strat_ma_cross"

    def test_default_indicators(self, now):
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross", timestamp=now,
        )
        assert s.indicators == {}

    def test_custom_indicators(self, now):
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross", timestamp=now,
            indicators={"rsi": 70.5, "macd": 0.3},
        )
        assert s.indicators == {"rsi": 70.5, "macd": 0.3}

    def test_default_timestamp(self):
        before = datetime.utcnow()
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross",
        )
        after = datetime.utcnow()
        assert before <= s.timestamp <= after

    def test_default_metadata(self, now):
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross", timestamp=now,
        )
        assert s.metadata == {}

    def test_custom_metadata(self, now):
        s = Signal(
            id="sig_001", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.85,
            strategy_id="strat_ma_cross", timestamp=now,
            metadata={"source": "ml_model", "version": "2.0"},
        )
        assert s.metadata == {"source": "ml_model", "version": "2.0"}

    def test_short_direction(self, now):
        s = Signal(
            id="sig_002", symbol="BTC/USDT",
            direction=SignalDirection.SHORT, strength=0.7,
            strategy_id="strat_rsi", timestamp=now,
        )
        assert s.direction == SignalDirection.SHORT

    def test_neutral_direction(self, now):
        s = Signal(
            id="sig_003", symbol="BTC/USDT",
            direction=SignalDirection.NEUTRAL, strength=0.3,
            strategy_id="strat_trend", timestamp=now,
        )
        assert s.direction == SignalDirection.NEUTRAL

    def test_strength_boundary_zero(self, now):
        s = Signal(
            id="sig_004", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=0.0,
            strategy_id="strat_test", timestamp=now,
        )
        assert s.strength == 0.0

    def test_strength_boundary_one(self, now):
        s = Signal(
            id="sig_005", symbol="BTC/USDT",
            direction=SignalDirection.LONG, strength=1.0,
            strategy_id="strat_test", timestamp=now,
        )
        assert s.strength == 1.0

    def test_metadata_independent_instances(self, now):
        """Ensure default_factory creates independent dicts."""
        s1 = Signal(id="s1", symbol="A", direction=SignalDirection.LONG,
                     strength=0.5, strategy_id="test", timestamp=now)
        s2 = Signal(id="s2", symbol="B", direction=SignalDirection.SHORT,
                     strength=0.5, strategy_id="test", timestamp=now)
        s1.metadata["key"] = "val"
        assert "key" not in s2.metadata

    def test_indicators_independent_instances(self, now):
        """Ensure default_factory creates independent dicts."""
        s1 = Signal(id="s1", symbol="A", direction=SignalDirection.LONG,
                     strength=0.5, strategy_id="test", timestamp=now)
        s2 = Signal(id="s2", symbol="B", direction=SignalDirection.SHORT,
                     strength=0.5, strategy_id="test", timestamp=now)
        s1.indicators["rsi"] = 50.0
        assert "rsi" not in s2.indicators


# ============================================================================
# Dataclass Tests: Position
# ============================================================================

class TestPosition:
    """Tests for the Position dataclass."""

    def test_construction_basic(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.symbol == "BTC/USDT"
        assert p.side == Side.BUY
        assert p.quantity == 1.0
        assert p.entry_price == 40000.0
        assert p.mark_price == 41000.0

    def test_default_unrealized_pnl(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.unrealized_pnl == 0.0

    def test_default_realized_pnl(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.realized_pnl == 0.0

    def test_default_leverage(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.leverage == 1.0

    def test_default_exchange(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.exchange == "paper"

    def test_notional_value_positive(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=2.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.notional_value == pytest.approx(82000.0)

    def test_notional_value_with_negative_quantity(self):
        p = Position(
            symbol="BTC/USDT", side=Side.SELL, quantity=-2.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.notional_value == pytest.approx(82000.0)

    def test_notional_value_zero_quantity(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=0.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.notional_value == 0.0

    def test_notional_value_zero_price(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=0.0,
        )
        assert p.notional_value == 0.0

    def test_margin_used_normal(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=10.0,
        )
        assert p.margin_used == pytest.approx(4100.0)

    def test_margin_used_leverage_one(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=1.0,
        )
        assert p.margin_used == pytest.approx(41000.0)

    def test_margin_used_zero_leverage_returns_zero(self):
        """Edge case: zero leverage returns 0.0."""
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=0.0,
        )
        assert p.margin_used == 0.0

    def test_margin_used_negative_leverage_returns_zero(self):
        """Edge case: negative leverage returns 0.0."""
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=-5.0,
        )
        assert p.margin_used == 0.0

    def test_margin_used_high_leverage(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=100.0,
        )
        assert p.margin_used == pytest.approx(410.0)

    def test_custom_exchange(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, exchange="binance",
        )
        assert p.exchange == "binance"

    def test_custom_leverage(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=5.0,
        )
        assert p.leverage == 5.0

    def test_sell_side(self):
        p = Position(
            symbol="BTC/USDT", side=Side.SELL, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        assert p.side == Side.SELL

    def test_very_small_leverage(self):
        p = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0, leverage=0.001,
        )
        assert p.margin_used == pytest.approx(41000.0 / 0.001)


# ============================================================================
# Dataclass Tests: Order
# ============================================================================

class TestOrder:
    """Tests for the Order dataclass."""

    def test_construction_basic(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.id == "ord_001"
        assert o.symbol == "BTC/USDT"
        assert o.side == Side.BUY
        assert o.order_type == OrderType.LIMIT
        assert o.status == OrderStatus.CREATED
        assert o.quantity == 1.0
        assert o.price == 40000.0

    def test_default_stop_price(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.stop_price is None

    def test_default_time_in_force(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.time_in_force == TimeInForce.GTC

    def test_default_filled_quantity(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.filled_quantity == 0.0

    def test_default_average_fill_price(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.average_fill_price == 0.0

    def test_default_commission(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.commission == 0.0

    def test_default_exchange(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.exchange == "paper"

    def test_default_strategy_id(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.strategy_id is None

    def test_remaining_quantity_unfilled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, filled_quantity=0.0, created_at=now,
        )
        assert o.remaining_quantity == 1.0

    def test_remaining_quantity_partially_filled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.PARTIALLY_FILLED,
            quantity=1.0, price=40000.0, filled_quantity=0.6, created_at=now,
        )
        assert o.remaining_quantity == pytest.approx(0.4)

    def test_remaining_quantity_fully_filled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.FILLED,
            quantity=1.0, price=40000.0, filled_quantity=1.0, created_at=now,
        )
        assert o.remaining_quantity == 0.0

    def test_is_active_created(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is True

    def test_is_active_validated(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.VALIDATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is True

    def test_is_active_submitted(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.SUBMITTED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is True

    def test_is_active_partially_filled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.PARTIALLY_FILLED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is True

    def test_is_active_filled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.FILLED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is False

    def test_is_active_cancelled(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CANCELLED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is False

    def test_is_active_rejected(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.REJECTED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is False

    def test_is_active_expired(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.EXPIRED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.is_active is False

    def test_notional_value_with_price(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=2.0, price=40000.0, created_at=now,
        )
        assert o.notional_value == pytest.approx(80000.0)

    def test_notional_value_without_price(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.CREATED,
            quantity=1.0, price=None, created_at=now,
        )
        assert o.notional_value == 0.0

    def test_notional_value_zero_quantity(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=0.0, price=40000.0, created_at=now,
        )
        assert o.notional_value == 0.0

    def test_market_order_no_price(self, now):
        o = Order(
            id="ord_002", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.SUBMITTED,
            quantity=1.0, created_at=now,
        )
        assert o.price is None
        assert o.notional_value == 0.0

    def test_stop_order_with_stop_price(self, now):
        o = Order(
            id="ord_003", symbol="BTC/USDT", side=Side.SELL,
            order_type=OrderType.STOP, status=OrderStatus.CREATED,
            quantity=1.0, price=39000.0, stop_price=39500.0, created_at=now,
        )
        assert o.stop_price == 39500.0

    def test_default_created_at(self):
        before = datetime.utcnow()
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0,
        )
        after = datetime.utcnow()
        assert before <= o.created_at <= after

    def test_custom_strategy_id(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, strategy_id="strat_001",
            created_at=now,
        )
        assert o.strategy_id == "strat_001"

    def test_custom_time_in_force(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, time_in_force=TimeInForce.IOC,
            created_at=now,
        )
        assert o.time_in_force == TimeInForce.IOC

    def test_sell_side(self, now):
        o = Order(
            id="ord_001", symbol="BTC/USDT", side=Side.SELL,
            order_type=OrderType.LIMIT, status=OrderStatus.CREATED,
            quantity=1.0, price=40000.0, created_at=now,
        )
        assert o.side == Side.SELL


# ============================================================================
# Dataclass Tests: Trade
# ============================================================================

class TestTrade:
    """Tests for the Trade dataclass."""

    def test_construction(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now,
        )
        assert t.id == "trade_001"
        assert t.order_id == "ord_001"
        assert t.symbol == "BTC/USDT"
        assert t.side == Side.BUY
        assert t.quantity == 0.5
        assert t.price == 40000.0
        assert t.commission == 10.0
        assert t.timestamp == now

    def test_default_exchange(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now,
        )
        assert t.exchange == "paper"

    def test_custom_exchange(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now, exchange="binance",
        )
        assert t.exchange == "binance"

    def test_default_is_maker(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now,
        )
        assert t.is_maker is False

    def test_custom_is_maker(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now, is_maker=True,
        )
        assert t.is_maker is True

    def test_default_slippage(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now,
        )
        assert t.slippage == 0.0

    def test_custom_slippage(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now, slippage=5.0,
        )
        assert t.slippage == 5.0

    def test_sell_side(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.SELL, quantity=0.5, price=40000.0,
            commission=10.0, timestamp=now,
        )
        assert t.side == Side.SELL

    def test_zero_commission(self, now):
        t = Trade(
            id="trade_001", order_id="ord_001", symbol="BTC/USDT",
            side=Side.BUY, quantity=0.5, price=40000.0,
            commission=0.0, timestamp=now,
        )
        assert t.commission == 0.0


# ============================================================================
# Dataclass Tests: PortfolioSnapshot
# ============================================================================

class TestPortfolioSnapshot:
    """Tests for the PortfolioSnapshot dataclass."""

    def test_construction(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0,
        )
        assert ps.timestamp == now
        assert ps.total_value == 100000.0
        assert ps.available_balance == 50000.0
        assert ps.unrealized_pnl == 1000.0
        assert ps.realized_pnl == 5000.0

    def test_default_positions(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0,
        )
        assert ps.positions == []

    def test_custom_positions(self, now):
        pos = Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        )
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0, positions=[pos],
        )
        assert len(ps.positions) == 1
        assert ps.positions[0].symbol == "BTC/USDT"

    def test_default_margin_used(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0,
        )
        assert ps.margin_used == 0.0

    def test_default_leverage(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0,
        )
        assert ps.leverage == 1.0

    def test_custom_margin_used(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0, margin_used=20000.0,
        )
        assert ps.margin_used == 20000.0

    def test_custom_leverage(self, now):
        ps = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0, leverage=5.0,
        )
        assert ps.leverage == 5.0

    def test_positions_independent_instances(self, now):
        """Ensure default_factory creates independent lists."""
        ps1 = PortfolioSnapshot(
            timestamp=now, total_value=100000.0,
            available_balance=50000.0, unrealized_pnl=1000.0,
            realized_pnl=5000.0,
        )
        ps2 = PortfolioSnapshot(
            timestamp=now, total_value=200000.0,
            available_balance=100000.0, unrealized_pnl=2000.0,
            realized_pnl=10000.0,
        )
        ps1.positions.append(Position(
            symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
            entry_price=40000.0, mark_price=41000.0,
        ))
        assert len(ps2.positions) == 0

    def test_negative_values(self, now):
        """Portfolio can have negative PnL."""
        ps = PortfolioSnapshot(
            timestamp=now, total_value=90000.0,
            available_balance=50000.0, unrealized_pnl=-5000.0,
            realized_pnl=-2000.0,
        )
        assert ps.unrealized_pnl == -5000.0
        assert ps.realized_pnl == -2000.0


# ============================================================================
# Dataclass Tests: RiskCheckResult
# ============================================================================

class TestRiskCheckResult:
    """Tests for the RiskCheckResult dataclass."""

    def test_construction(self, now):
        r = RiskCheckResult(
            decision=RiskDecision.ALLOW, check_name="max_position",
            reason="Within limits", current_value=50000.0,
            limit_value=100000.0, timestamp=now,
        )
        assert r.decision == RiskDecision.ALLOW
        assert r.check_name == "max_position"
        assert r.reason == "Within limits"
        assert r.current_value == 50000.0
        assert r.limit_value == 100000.0
        assert r.timestamp == now

    def test_default_timestamp(self):
        before = datetime.utcnow()
        r = RiskCheckResult(
            decision=RiskDecision.REJECT, check_name="drawdown",
            reason="Exceeded", current_value=0.25, limit_value=0.20,
        )
        after = datetime.utcnow()
        assert before <= r.timestamp <= after

    def test_reject_decision(self, now):
        r = RiskCheckResult(
            decision=RiskDecision.REJECT, check_name="drawdown",
            reason="Exceeded limit", current_value=0.25,
            limit_value=0.20, timestamp=now,
        )
        assert r.decision == RiskDecision.REJECT

    def test_throttle_decision(self, now):
        r = RiskCheckResult(
            decision=RiskDecision.THROTTLE, check_name="rate_limit",
            reason="Too many orders", current_value=15,
            limit_value=10, timestamp=now,
        )
        assert r.decision == RiskDecision.THROTTLE

    def test_current_equals_limit(self, now):
        r = RiskCheckResult(
            decision=RiskDecision.ALLOW, check_name="position",
            reason="At limit", current_value=100000.0,
            limit_value=100000.0, timestamp=now,
        )
        assert r.current_value == r.limit_value


# ============================================================================
# Dataclass Tests: ExecutionReport
# ============================================================================

class TestExecutionReport:
    """Tests for the ExecutionReport dataclass."""

    def test_construction(self, now):
        er = ExecutionReport(
            order_id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.FILLED,
            quantity=1.0, filled_quantity=1.0, average_price=40000.0,
            commission=10.0, slippage=5.0, latency_us=500,
            exchange="binance", timestamp=now,
        )
        assert er.order_id == "ord_001"
        assert er.symbol == "BTC/USDT"
        assert er.side == Side.BUY
        assert er.order_type == OrderType.LIMIT
        assert er.status == OrderStatus.FILLED
        assert er.quantity == 1.0
        assert er.filled_quantity == 1.0
        assert er.average_price == 40000.0
        assert er.commission == 10.0
        assert er.slippage == 5.0
        assert er.latency_us == 500
        assert er.exchange == "binance"
        assert er.timestamp == now

    def test_default_timestamp(self):
        before = datetime.utcnow()
        er = ExecutionReport(
            order_id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.FILLED,
            quantity=1.0, filled_quantity=1.0, average_price=40000.0,
            commission=10.0, slippage=0.0, latency_us=100,
            exchange="paper",
        )
        after = datetime.utcnow()
        assert before <= er.timestamp <= after

    def test_zero_latency(self, now):
        er = ExecutionReport(
            order_id="ord_001", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, status=OrderStatus.FILLED,
            quantity=1.0, filled_quantity=1.0, average_price=40000.0,
            commission=10.0, slippage=0.0, latency_us=0,
            exchange="paper", timestamp=now,
        )
        assert er.latency_us == 0

    def test_sell_side_report(self, now):
        er = ExecutionReport(
            order_id="ord_002", symbol="ETH/USDT", side=Side.SELL,
            order_type=OrderType.MARKET, status=OrderStatus.FILLED,
            quantity=5.0, filled_quantity=5.0, average_price=3000.0,
            commission=5.0, slippage=1.0, latency_us=200,
            exchange="bybit", timestamp=now,
        )
        assert er.side == Side.SELL

    def test_partial_fill_report(self, now):
        er = ExecutionReport(
            order_id="ord_003", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, status=OrderStatus.PARTIALLY_FILLED,
            quantity=1.0, filled_quantity=0.5, average_price=40000.0,
            commission=5.0, slippage=0.0, latency_us=300,
            exchange="okx", timestamp=now,
        )
        assert er.status == OrderStatus.PARTIALLY_FILLED
        assert er.filled_quantity == 0.5


# ============================================================================
# Dataclass Tests: ACMSConfig
# ============================================================================

class TestACMSConfig:
    """Tests for the ACMSConfig dataclass - all default values and overrides."""

    def test_default_db_url(self):
        cfg = ACMSConfig()
        assert cfg.db_url == "postgresql://acms:acms@localhost:5432/acms"

    def test_default_redis_url(self):
        cfg = ACMSConfig()
        assert cfg.redis_url == "redis://localhost:6379/0"

    def test_default_redpanda_brokers(self):
        cfg = ACMSConfig()
        assert cfg.redpanda_brokers == ["localhost:9092"]

    def test_default_api_host(self):
        cfg = ACMSConfig()
        assert cfg.api_host == "0.0.0.0"

    def test_default_api_port(self):
        cfg = ACMSConfig()
        assert cfg.api_port == 8000

    def test_default_api_workers(self):
        cfg = ACMSConfig()
        assert cfg.api_workers == 4

    def test_default_jwt_secret(self):
        cfg = ACMSConfig()
        assert cfg.jwt_secret == "change-me-in-production"

    def test_default_jwt_expiry_hours(self):
        cfg = ACMSConfig()
        assert cfg.jwt_expiry_hours == 24

    def test_default_api_key_length(self):
        cfg = ACMSConfig()
        assert cfg.api_key_length == 32

    def test_default_max_position_per_symbol(self):
        cfg = ACMSConfig()
        assert cfg.max_position_per_symbol == 100000.0

    def test_default_max_total_position(self):
        cfg = ACMSConfig()
        assert cfg.max_total_position == 1000000.0

    def test_default_max_order_notional(self):
        cfg = ACMSConfig()
        assert cfg.max_order_notional == 50000.0

    def test_default_max_daily_drawdown(self):
        cfg = ACMSConfig()
        assert cfg.max_daily_drawdown == 0.05

    def test_default_max_drawdown(self):
        cfg = ACMSConfig()
        assert cfg.max_drawdown == 0.20

    def test_default_max_orders_per_second(self):
        cfg = ACMSConfig()
        assert cfg.max_orders_per_second == 10

    def test_default_max_orders_per_minute(self):
        cfg = ACMSConfig()
        assert cfg.max_orders_per_minute == 100

    def test_default_binance_api_key(self):
        cfg = ACMSConfig()
        assert cfg.binance_api_key == ""

    def test_default_binance_api_secret(self):
        cfg = ACMSConfig()
        assert cfg.binance_api_secret == ""

    def test_default_bybit_api_key(self):
        cfg = ACMSConfig()
        assert cfg.bybit_api_key == ""

    def test_default_bybit_api_secret(self):
        cfg = ACMSConfig()
        assert cfg.bybit_api_secret == ""

    def test_default_okx_api_key(self):
        cfg = ACMSConfig()
        assert cfg.okx_api_key == ""

    def test_default_okx_api_secret(self):
        cfg = ACMSConfig()
        assert cfg.okx_api_secret == ""

    def test_default_okx_passphrase(self):
        cfg = ACMSConfig()
        assert cfg.okx_passphrase == ""

    def test_default_data_dir(self):
        cfg = ACMSConfig()
        assert cfg.data_dir == "/data/acms"

    def test_default_parquet_dir(self):
        cfg = ACMSConfig()
        assert cfg.parquet_dir == "/data/acms/parquet"

    def test_default_ml_model_dir(self):
        cfg = ACMSConfig()
        assert cfg.ml_model_dir == "/data/acms/models"

    def test_default_ml_training_enabled(self):
        cfg = ACMSConfig()
        assert cfg.ml_training_enabled is True

    def test_default_log_level(self):
        cfg = ACMSConfig()
        assert cfg.log_level == "INFO"

    def test_default_log_file(self):
        cfg = ACMSConfig()
        assert cfg.log_file == "/data/acms/logs/acms.log"

    def test_override_db_url(self):
        cfg = ACMSConfig(db_url="postgresql://user:pass@db:5432/mydb")
        assert cfg.db_url == "postgresql://user:pass@db:5432/mydb"

    def test_override_redis_url(self):
        cfg = ACMSConfig(redis_url="redis://redis:6379/1")
        assert cfg.redis_url == "redis://redis:6379/1"

    def test_override_redpanda_brokers(self):
        cfg = ACMSConfig(redpanda_brokers=["broker1:9092", "broker2:9092"])
        assert cfg.redpanda_brokers == ["broker1:9092", "broker2:9092"]

    def test_override_api_host(self):
        cfg = ACMSConfig(api_host="127.0.0.1")
        assert cfg.api_host == "127.0.0.1"

    def test_override_api_port(self):
        cfg = ACMSConfig(api_port=9000)
        assert cfg.api_port == 9000

    def test_override_api_workers(self):
        cfg = ACMSConfig(api_workers=8)
        assert cfg.api_workers == 8

    def test_override_jwt_secret(self):
        cfg = ACMSConfig(jwt_secret="my-super-secret")
        assert cfg.jwt_secret == "my-super-secret"

    def test_override_jwt_expiry_hours(self):
        cfg = ACMSConfig(jwt_expiry_hours=48)
        assert cfg.jwt_expiry_hours == 48

    def test_override_api_key_length(self):
        cfg = ACMSConfig(api_key_length=64)
        assert cfg.api_key_length == 64

    def test_override_max_position_per_symbol(self):
        cfg = ACMSConfig(max_position_per_symbol=50000.0)
        assert cfg.max_position_per_symbol == 50000.0

    def test_override_max_total_position(self):
        cfg = ACMSConfig(max_total_position=500000.0)
        assert cfg.max_total_position == 500000.0

    def test_override_max_order_notional(self):
        cfg = ACMSConfig(max_order_notional=25000.0)
        assert cfg.max_order_notional == 25000.0

    def test_override_max_daily_drawdown(self):
        cfg = ACMSConfig(max_daily_drawdown=0.03)
        assert cfg.max_daily_drawdown == 0.03

    def test_override_max_drawdown(self):
        cfg = ACMSConfig(max_drawdown=0.15)
        assert cfg.max_drawdown == 0.15

    def test_override_max_orders_per_second(self):
        cfg = ACMSConfig(max_orders_per_second=20)
        assert cfg.max_orders_per_second == 20

    def test_override_max_orders_per_minute(self):
        cfg = ACMSConfig(max_orders_per_minute=200)
        assert cfg.max_orders_per_minute == 200

    def test_override_exchange_credentials(self):
        cfg = ACMSConfig(
            binance_api_key="bn_key",
            binance_api_secret="bn_secret",
            bybit_api_key="bb_key",
            bybit_api_secret="bb_secret",
            okx_api_key="okx_key",
            okx_api_secret="okx_secret",
            okx_passphrase="okx_pass",
        )
        assert cfg.binance_api_key == "bn_key"
        assert cfg.binance_api_secret == "bn_secret"
        assert cfg.bybit_api_key == "bb_key"
        assert cfg.bybit_api_secret == "bb_secret"
        assert cfg.okx_api_key == "okx_key"
        assert cfg.okx_api_secret == "okx_secret"
        assert cfg.okx_passphrase == "okx_pass"

    def test_override_data_dirs(self):
        cfg = ACMSConfig(
            data_dir="/custom/data",
            parquet_dir="/custom/data/parquet",
            ml_model_dir="/custom/models",
        )
        assert cfg.data_dir == "/custom/data"
        assert cfg.parquet_dir == "/custom/data/parquet"
        assert cfg.ml_model_dir == "/custom/models"

    def test_override_ml_training_enabled(self):
        cfg = ACMSConfig(ml_training_enabled=False)
        assert cfg.ml_training_enabled is False

    def test_override_log_settings(self):
        cfg = ACMSConfig(log_level="DEBUG", log_file="/var/log/acms.log")
        assert cfg.log_level == "DEBUG"
        assert cfg.log_file == "/var/log/acms.log"

    def test_redpanda_brokers_independent_instances(self):
        """Ensure default_factory creates independent lists."""
        cfg1 = ACMSConfig()
        cfg2 = ACMSConfig()
        cfg1.redpanda_brokers.append("extra:9092")
        assert len(cfg2.redpanda_brokers) == 1

    def test_multiple_overrides(self):
        cfg = ACMSConfig(
            db_url="postgresql://test:test@db:5432/testdb",
            redis_url="redis://redis:6379/2",
            api_port=9000,
            jwt_secret="prod-secret",
            max_drawdown=0.10,
            log_level="WARNING",
        )
        assert cfg.db_url == "postgresql://test:test@db:5432/testdb"
        assert cfg.redis_url == "redis://redis:6379/2"
        assert cfg.api_port == 9000
        assert cfg.jwt_secret == "prod-secret"
        assert cfg.max_drawdown == 0.10
        assert cfg.log_level == "WARNING"


# ============================================================================
# Auth Tests: TokenData
# ============================================================================

class TestTokenData:
    """Tests for the TokenData dataclass."""

    def test_construction(self):
        td = TokenData(user_id="user_001", email="test@example.com")
        assert td.user_id == "user_001"
        assert td.email == "test@example.com"

    def test_equality(self):
        td1 = TokenData(user_id="user_001", email="test@example.com")
        td2 = TokenData(user_id="user_001", email="test@example.com")
        assert td1 == td2

    def test_inequality_different_user_id(self):
        td1 = TokenData(user_id="user_001", email="test@example.com")
        td2 = TokenData(user_id="user_002", email="test@example.com")
        assert td1 != td2

    def test_inequality_different_email(self):
        td1 = TokenData(user_id="user_001", email="a@example.com")
        td2 = TokenData(user_id="user_001", email="b@example.com")
        assert td1 != td2

    def test_empty_user_id(self):
        td = TokenData(user_id="", email="test@example.com")
        assert td.user_id == ""

    def test_empty_email(self):
        td = TokenData(user_id="user_001", email="")
        assert td.email == ""


# ============================================================================
# Auth Tests: AuthManager.__init__
# ============================================================================

class TestAuthManagerInit:
    """Tests for AuthManager initialization."""

    def test_default_secret_key(self):
        am = AuthManager()
        assert am.secret_key == "change-me-in-production"

    def test_default_algorithm(self):
        am = AuthManager()
        assert am.algorithm == "HS256"

    def test_default_expiry_hours(self):
        am = AuthManager()
        assert am.expiry_hours == 24

    def test_custom_secret_key(self):
        am = AuthManager(secret_key="my-secret")
        assert am.secret_key == "my-secret"

    def test_custom_algorithm(self):
        am = AuthManager(algorithm="HS384")
        assert am.algorithm == "HS384"

    def test_custom_expiry_hours(self):
        am = AuthManager(expiry_hours=48)
        assert am.expiry_hours == 48

    def test_all_custom_params(self):
        am = AuthManager(secret_key="s", algorithm="HS512", expiry_hours=12)
        assert am.secret_key == "s"
        assert am.algorithm == "HS512"
        assert am.expiry_hours == 12


# ============================================================================
# Auth Tests: AuthManager.create_token
# ============================================================================

class TestAuthManagerCreateToken:
    """Tests for AuthManager.create_token."""

    def test_returns_string(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        assert isinstance(token, str)

    def test_token_is_valid_jwt(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        # Should not raise
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert "sub" in payload

    def test_token_contains_user_id(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert payload["sub"] == "user_001"

    def test_token_contains_email(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert payload["email"] == "test@example.com"

    def test_token_contains_exp(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert "exp" in payload

    def test_token_contains_iat(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert "iat" in payload

    def test_token_expiry_is_correct(self, auth_manager):
        before = datetime.utcnow()
        token = auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        after = datetime.utcnow()
        # JWT exp is a Unix timestamp truncated to seconds
        expected_exp_min = int((before + timedelta(hours=24)).timestamp())
        expected_exp_max = int((after + timedelta(hours=24)).timestamp()) + 1
        assert expected_exp_min <= payload["exp"] <= expected_exp_max

    def test_token_with_custom_expiry(self, custom_auth_manager):
        token = custom_auth_manager.create_token("user_001", "test@example.com")
        payload = jwt.decode(
            token, custom_auth_manager.secret_key,
            algorithms=[custom_auth_manager.algorithm],
        )
        # Custom manager has 48h expiry
        assert "exp" in payload

    def test_different_user_ids_produce_different_tokens(self, auth_manager):
        token1 = auth_manager.create_token("user_001", "a@example.com")
        token2 = auth_manager.create_token("user_002", "a@example.com")
        assert token1 != token2

    def test_different_emails_produce_different_tokens(self, auth_manager):
        token1 = auth_manager.create_token("user_001", "a@example.com")
        token2 = auth_manager.create_token("user_001", "b@example.com")
        assert token1 != token2

    def test_token_with_empty_user_id(self, auth_manager):
        token = auth_manager.create_token("", "test@example.com")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert payload["sub"] == ""

    def test_token_with_empty_email(self, auth_manager):
        token = auth_manager.create_token("user_001", "")
        payload = jwt.decode(
            token, auth_manager.secret_key,
            algorithms=[auth_manager.algorithm],
        )
        assert payload["email"] == ""


# ============================================================================
# Auth Tests: AuthManager.verify_token
# ============================================================================

class TestAuthManagerVerifyToken:
    """Tests for AuthManager.verify_token."""

    def test_valid_token_returns_token_data(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        result = auth_manager.verify_token(token)
        assert result is not None
        assert isinstance(result, TokenData)

    def test_valid_token_contains_correct_user_id(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        result = auth_manager.verify_token(token)
        assert result.user_id == "user_001"

    def test_valid_token_contains_correct_email(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        result = auth_manager.verify_token(token)
        assert result.email == "test@example.com"

    def test_expired_token_returns_none(self):
        am = AuthManager(expiry_hours=0)
        # Create token with 0-hour expiry - may already be expired
        token = am.create_token("user_001", "test@example.com")
        # Manually create an expired token for a reliable test
        expired_payload = {
            "sub": "user_001",
            "email": "test@example.com",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2),
        }
        expired_token = jwt.encode(
            expired_payload, am.secret_key, algorithm=am.algorithm,
        )
        result = am.verify_token(expired_token)
        assert result is None

    def test_malformed_token_returns_none(self, auth_manager):
        result = auth_manager.verify_token("not.a.valid.token")
        assert result is None

    def test_empty_string_token_returns_none(self, auth_manager):
        result = auth_manager.verify_token("")
        assert result is None

    def test_random_string_token_returns_none(self, auth_manager):
        result = auth_manager.verify_token("abc123xyz789")
        assert result is None

    def test_wrong_secret_returns_none(self):
        am1 = AuthManager(secret_key="secret1")
        am2 = AuthManager(secret_key="secret2")
        token = am1.create_token("user_001", "test@example.com")
        result = am2.verify_token(token)
        assert result is None

    def test_wrong_algorithm_returns_none(self):
        am1 = AuthManager(algorithm="HS256")
        am2 = AuthManager(algorithm="HS384")
        token = am1.create_token("user_001", "test@example.com")
        result = am2.verify_token(token)
        assert result is None

    def test_token_missing_sub_returns_none(self, auth_manager):
        """Token with no 'sub' claim should return None."""
        payload = {
            "email": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(
            payload, auth_manager.secret_key,
            algorithm=auth_manager.algorithm,
        )
        result = auth_manager.verify_token(token)
        assert result is None

    def test_token_missing_email_still_returns_token_data(self, auth_manager):
        """Token with no 'email' claim - email will be None in TokenData."""
        payload = {
            "sub": "user_001",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
        }
        token = jwt.encode(
            payload, auth_manager.secret_key,
            algorithm=auth_manager.algorithm,
        )
        result = auth_manager.verify_token(token)
        assert result is not None
        assert result.user_id == "user_001"
        assert result.email is None

    def test_token_tampered_signature_returns_none(self, auth_manager):
        token = auth_manager.create_token("user_001", "test@example.com")
        # Tamper with the token by changing a character
        tampered = token[:-5] + "XXXXX"
        result = auth_manager.verify_token(tampered)
        assert result is None


# ============================================================================
# Auth Tests: AuthManager.hash_password
# ============================================================================

class TestAuthManagerHashPassword:
    """Tests for AuthManager.hash_password."""

    def test_returns_string(self, auth_manager):
        result = auth_manager.hash_password("mypassword")
        assert isinstance(result, str)

    def test_hash_contains_colon_separator(self, auth_manager):
        result = auth_manager.hash_password("mypassword")
        assert ":" in result

    def test_hash_has_two_parts(self, auth_manager):
        result = auth_manager.hash_password("mypassword")
        parts = result.split(":")
        assert len(parts) == 2

    def test_salt_is_hex_string(self, auth_manager):
        result = auth_manager.hash_password("mypassword")
        salt, _ = result.split(":")
        # salt should be 32 hex chars (16 bytes = 32 hex chars)
        assert len(salt) == 32
        int(salt, 16)  # Should not raise - valid hex

    def test_hash_is_hex_string(self, auth_manager):
        result = auth_manager.hash_password("mypassword")
        _, hashed = result.split(":")
        # SHA-256 produces 64 hex chars
        assert len(hashed) == 64
        int(hashed, 16)  # Should not raise - valid hex

    def test_different_calls_produce_different_salts(self, auth_manager):
        """Each call should generate a new random salt."""
        h1 = auth_manager.hash_password("mypassword")
        h2 = auth_manager.hash_password("mypassword")
        salt1 = h1.split(":")[0]
        salt2 = h2.split(":")[0]
        assert salt1 != salt2

    def test_different_calls_produce_different_hashes(self, auth_manager):
        """Same password, different salt = different full hash string."""
        h1 = auth_manager.hash_password("mypassword")
        h2 = auth_manager.hash_password("mypassword")
        assert h1 != h2

    def test_hash_computation_is_correct(self, auth_manager):
        """Verify the hash computation manually."""
        result = auth_manager.hash_password("testpw")
        salt, stored_hash = result.split(":")
        expected = hashlib.sha256((salt + "testpw").encode()).hexdigest()
        assert stored_hash == expected

    def test_empty_password(self, auth_manager):
        result = auth_manager.hash_password("")
        assert ":" in result
        salt, stored_hash = result.split(":")
        expected = hashlib.sha256((salt + "").encode()).hexdigest()
        assert stored_hash == expected

    def test_long_password(self, auth_manager):
        long_pw = "a" * 1000
        result = auth_manager.hash_password(long_pw)
        assert ":" in result
        salt, stored_hash = result.split(":")
        expected = hashlib.sha256((salt + long_pw).encode()).hexdigest()
        assert stored_hash == expected


# ============================================================================
# Auth Tests: AuthManager.verify_password
# ============================================================================

class TestAuthManagerVerifyPassword:
    """Tests for AuthManager.verify_password."""

    def test_correct_password(self, auth_manager):
        hashed = auth_manager.hash_password("mypassword")
        assert auth_manager.verify_password("mypassword", hashed) is True

    def test_wrong_password(self, auth_manager):
        hashed = auth_manager.hash_password("mypassword")
        assert auth_manager.verify_password("wrongpassword", hashed) is False

    def test_empty_password_correct(self, auth_manager):
        hashed = auth_manager.hash_password("")
        assert auth_manager.verify_password("", hashed) is True

    def test_empty_password_wrong(self, auth_manager):
        hashed = auth_manager.hash_password("")
        assert auth_manager.verify_password("something", hashed) is False

    def test_malformed_hash_no_colon(self, auth_manager):
        result = auth_manager.verify_password("mypassword", "no-colon-here")
        assert result is False

    def test_malformed_hash_empty_string(self, auth_manager):
        result = auth_manager.verify_password("mypassword", "")
        assert result is False

    def test_malformed_hash_only_colon(self, auth_manager):
        result = auth_manager.verify_password("mypassword", ":")
        assert result is False

    def test_malformed_hash_multiple_colons(self, auth_manager):
        """Multiple colons - split gives more than 2 parts, should raise ValueError."""
        result = auth_manager.verify_password("mypassword", "a:b:c")
        # The split(":") with no maxsplit gives 3 parts, unpacking fails -> ValueError -> False
        assert result is False

    def test_case_sensitive_password(self, auth_manager):
        hashed = auth_manager.hash_password("MyPassword")
        assert auth_manager.verify_password("mypassword", hashed) is False
        assert auth_manager.verify_password("MyPassword", hashed) is True

    def test_password_with_special_chars(self, auth_manager):
        pw = "p@$$w0rd!#%^&*()"
        hashed = auth_manager.hash_password(pw)
        assert auth_manager.verify_password(pw, hashed) is True

    def test_password_with_unicode(self, auth_manager):
        pw = "пароль123"
        hashed = auth_manager.hash_password(pw)
        assert auth_manager.verify_password(pw, hashed) is True

    def test_manually_constructed_hash(self, auth_manager):
        """Verify against a manually constructed hash."""
        salt = "abcdef0123456789abcdef0123456789"
        password = "test123"
        computed_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        stored = f"{salt}:{computed_hash}"
        assert auth_manager.verify_password(password, stored) is True

    def test_manually_constructed_wrong_hash(self, auth_manager):
        salt = "abcdef0123456789abcdef0123456789"
        stored = f"{salt}:0" * 64  # Wrong hash
        # This won't have the right format since we're doing f-string wrong
        # Let's construct it properly
        wrong_hash = "0" * 64
        stored = f"{salt}:{wrong_hash}"
        assert auth_manager.verify_password("test123", stored) is False


# ============================================================================
# Auth Tests: AuthManager.generate_api_key
# ============================================================================

class TestAuthManagerGenerateApiKey:
    """Tests for AuthManager.generate_api_key."""

    def test_returns_tuple(self, auth_manager):
        result = auth_manager.generate_api_key()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_key_starts_with_acms_prefix(self, auth_manager):
        raw_key, _ = auth_manager.generate_api_key()
        assert raw_key.startswith("acms_")

    def test_key_has_correct_length(self, auth_manager):
        """acms_ (5 chars) + 48 hex chars (24 bytes) = 53 chars total."""
        raw_key, _ = auth_manager.generate_api_key()
        assert len(raw_key) == 5 + 48  # "acms_" + 24 bytes as hex (48 chars)

    def test_hash_is_sha256_hex(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        assert len(hashed_key) == 64  # SHA-256 produces 64 hex chars
        int(hashed_key, 16)  # Should be valid hex

    def test_hash_matches_key(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert hashed_key == expected_hash

    def test_different_calls_produce_different_keys(self, auth_manager):
        key1, _ = auth_manager.generate_api_key()
        key2, _ = auth_manager.generate_api_key()
        assert key1 != key2

    def test_different_calls_produce_different_hashes(self, auth_manager):
        _, hash1 = auth_manager.generate_api_key()
        _, hash2 = auth_manager.generate_api_key()
        assert hash1 != hash2


# ============================================================================
# Auth Tests: AuthManager.verify_api_key
# ============================================================================

class TestAuthManagerVerifyApiKey:
    """Tests for AuthManager.verify_api_key."""

    def test_correct_key(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        assert auth_manager.verify_api_key(raw_key, hashed_key) is True

    def test_wrong_key(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        assert auth_manager.verify_api_key("acms_wrongkey", hashed_key) is False

    def test_empty_key(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        assert auth_manager.verify_api_key("", hashed_key) is False

    def test_empty_hash(self, auth_manager):
        raw_key, _ = auth_manager.generate_api_key()
        assert auth_manager.verify_api_key(raw_key, "") is False

    def test_manually_computed_hash(self, auth_manager):
        raw_key = "acms_testkey123"
        hashed = hashlib.sha256(raw_key.encode()).hexdigest()
        assert auth_manager.verify_api_key(raw_key, hashed) is True

    def test_case_sensitivity(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        upper_key = raw_key.upper()
        assert auth_manager.verify_api_key(upper_key, hashed_key) is False

    def test_key_with_extra_whitespace(self, auth_manager):
        raw_key, hashed_key = auth_manager.generate_api_key()
        padded_key = raw_key + " "
        assert auth_manager.verify_api_key(padded_key, hashed_key) is False

    def test_cross_verify_different_keys(self, auth_manager):
        """One key's hash should not verify another key."""
        key1, hash1 = auth_manager.generate_api_key()
        key2, hash2 = auth_manager.generate_api_key()
        assert auth_manager.verify_api_key(key1, hash2) is False
        assert auth_manager.verify_api_key(key2, hash1) is False


# ============================================================================
# Auth Tests: AuthManager.authenticate_user
# ============================================================================

class TestAuthManagerAuthenticateUser:
    """Tests for AuthManager.authenticate_user (placeholder implementation)."""

    def test_returns_dict(self, auth_manager):
        result = auth_manager.authenticate_user("test@example.com", "password")
        assert isinstance(result, dict)

    def test_returns_user_id(self, auth_manager):
        result = auth_manager.authenticate_user("test@example.com", "password")
        assert "id" in result
        assert result["id"] == "user_dev_001"

    def test_returns_email(self, auth_manager):
        result = auth_manager.authenticate_user("test@example.com", "password")
        assert "email" in result
        assert result["email"] == "test@example.com"

    def test_accepts_any_email(self, auth_manager):
        result = auth_manager.authenticate_user("anyone@anywhere.com", "password")
        assert result["email"] == "anyone@anywhere.com"

    def test_accepts_any_password(self, auth_manager):
        """Placeholder accepts any password."""
        result = auth_manager.authenticate_user("test@example.com", "anything123")
        assert result is not None

    def test_accepts_empty_credentials(self, auth_manager):
        """Placeholder accepts empty credentials."""
        result = auth_manager.authenticate_user("", "")
        assert result is not None
        assert result["email"] == ""

    def test_returns_not_none(self, auth_manager):
        """Placeholder always returns a user dict, never None."""
        result = auth_manager.authenticate_user("x", "y")
        assert result is not None

    def test_result_has_exactly_two_keys(self, auth_manager):
        result = auth_manager.authenticate_user("test@example.com", "password")
        assert len(result) == 2
        assert set(result.keys()) == {"id", "email"}
