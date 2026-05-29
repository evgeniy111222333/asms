"""Core types and configuration for ACMS Python layer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from datetime import datetime
from decimal import Decimal


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTD = "gtd"
    DAY = "day"


class ExchangeId(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"
    PAPER = "paper"


class Timeframe(str, Enum):
    S1 = "1s"
    S5 = "5s"
    S15 = "15s"
    S30 = "30s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    THROTTLE = "throttle"


@dataclass
class Symbol:
    base: str
    quote: str = "USDT"

    @property
    def pair(self) -> str:
        return f"{self.base}/{self.quote}"

    def __str__(self) -> str:
        return self.pair

    def __hash__(self) -> int:
        return hash(self.pair)


@dataclass
class Candle:
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class Tick:
    symbol: str
    exchange: str
    price: float
    quantity: float
    side: Side
    timestamp: datetime
    trade_id: str = ""


@dataclass
class Signal:
    id: str
    symbol: str
    direction: SignalDirection
    strength: float  # 0.0 - 1.0
    strategy_id: str
    indicators: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    exchange: str = "paper"

    @property
    def notional_value(self) -> float:
        return abs(self.quantity * self.mark_price)

    @property
    def margin_used(self) -> float:
        return self.notional_value / self.leverage if self.leverage > 0 else 0.0


@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    exchange: str = "paper"
    strategy_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.CREATED, OrderStatus.VALIDATED,
            OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def notional_value(self) -> float:
        return self.quantity * (self.price or 0.0)


@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    exchange: str = "paper"
    is_maker: bool = False
    slippage: float = 0.0


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    available_balance: float
    unrealized_pnl: float
    realized_pnl: float
    positions: list[Position] = field(default_factory=list)
    margin_used: float = 0.0
    leverage: float = 1.0


@dataclass
class RiskCheckResult:
    decision: RiskDecision
    check_name: str
    reason: str
    current_value: float
    limit_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float
    average_price: float
    commission: float
    slippage: float
    latency_us: int
    exchange: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ACMSConfig:
    """Main ACMS configuration."""
    # Database
    db_url: str = "postgresql://acms:acms@localhost:5432/acms"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Redpanda
    redpanda_brokers: list[str] = field(default_factory=lambda: ["localhost:9092"])

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24
    api_key_length: int = 32

    # Risk
    max_position_per_symbol: float = 100000.0
    max_total_position: float = 1000000.0
    max_order_notional: float = 50000.0
    max_daily_drawdown: float = 0.05
    max_drawdown: float = 0.20
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100

    # Exchanges
    binance_api_key: str = ""
    binance_api_secret: str = ""
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""

    # Data
    data_dir: str = "/data/acms"
    parquet_dir: str = "/data/acms/parquet"

    # ML
    ml_model_dir: str = "/data/acms/models"
    ml_training_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "/data/acms/logs/acms.log"
