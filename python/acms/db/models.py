"""ORM models and field whitelists for the ACMS database."""

from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, JSON,
    ForeignKey, Text, Numeric,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# ============================================================================
# Field Whitelists for Update Operations
# ============================================================================

ALLOWED_ORDER_FIELDS = {"status", "filled_quantity", "average_fill_price", "commission", "updated_at"}
ALLOWED_STRATEGY_FIELDS = {"name", "type", "config", "is_active", "updated_at"}


# ============================================================================
# ORM Models
# ============================================================================

class User(Base):
    """User account model."""
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    api_key = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ApiKey(Base):
    """API key model for programmatic access."""
    __tablename__ = "api_keys"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    permissions = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Strategy(Base):
    """Strategy configuration model."""
    __tablename__ = "strategies"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    config = Column(JSON, default=dict)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OrderRecord(Base):
    """Order record model."""
    __tablename__ = "orders"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    order_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)
    stop_price = Column(Numeric(20, 8), nullable=True)
    filled_quantity = Column(Numeric(20, 8), default=0)
    average_fill_price = Column(Numeric(20, 8), default=0)
    commission = Column(Numeric(20, 8), default=0)
    exchange = Column(String, nullable=False)
    exchange_order_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeRecord(Base):
    """Trade execution record model."""
    __tablename__ = "trades"
    id = Column(String, primary_key=True)
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    commission = Column(Numeric(20, 8), default=0)
    commission_asset = Column(String, default="")
    exchange = Column(String, nullable=False)
    exchange_trade_id = Column(String, nullable=True)
    is_maker = Column(Boolean, default=False)
    slippage = Column(Numeric(20, 8), default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)


class TradeArchiveRecord(Base):
    """Archived trade records model."""
    __tablename__ = "trades_archive"
    id = Column(String, primary_key=True)
    order_id = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    commission = Column(Numeric(20, 8), default=0)
    commission_asset = Column(String, default="")
    exchange = Column(String, nullable=False)
    exchange_trade_id = Column(String, nullable=True)
    is_maker = Column(Boolean, default=False)
    slippage = Column(Numeric(20, 8), default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    archived_at = Column(DateTime, default=datetime.utcnow)


class PositionRecord(Base):
    """Position record model."""
    __tablename__ = "positions"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    mark_price = Column(Numeric(20, 8), default=0)
    unrealized_pnl = Column(Numeric(20, 8), default=0)
    realized_pnl = Column(Numeric(20, 8), default=0)
    leverage = Column(Numeric(10, 2), default=1)
    exchange = Column(String, nullable=False)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SignalRecord(Base):
    """Signal record model."""
    __tablename__ = "signals"
    id = Column(String, primary_key=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    strength = Column(Numeric(10, 4), default=0)
    indicators = Column(JSON, default=dict)
    signal_metadata = Column('metadata', JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)


class CandleRecord(Base):
    """OHLCV candle record model."""
    __tablename__ = "candles"
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)
    quote_volume = Column(Numeric(20, 8), default=0)
    trades = Column(Integer, default=0)
    exchange = Column(String, nullable=False)


class RiskEvent(Base):
    """Risk event record model."""
    __tablename__ = "risk_events"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    event_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    """Backtest result record model."""
    __tablename__ = "backtest_results"
    id = Column(String, primary_key=True)
    strategy_id = Column(String, ForeignKey("strategies.id"), nullable=False)
    config = Column(JSON, default=dict)
    total_return = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    win_rate = Column(Numeric(10, 4))
    total_trades = Column(Integer)
    results_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class PortfolioSnapshotRecord(Base):
    """Portfolio snapshot record model."""
    __tablename__ = "portfolio_snapshots"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    total_value = Column(Numeric(20, 8), nullable=False)
    available_balance = Column(Numeric(20, 8), default=0)
    unrealized_pnl = Column(Numeric(20, 8), default=0)
    realized_pnl = Column(Numeric(20, 8), default=0)
    margin_used = Column(Numeric(20, 8), default=0)
    leverage = Column(Numeric(10, 2), default=1)
    positions_json = Column(JSON, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ExchangeCredential(Base):
    """Encrypted exchange credential model."""
    __tablename__ = "exchange_credentials"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    exchange = Column(String, nullable=False)
    api_key_encrypted = Column(Text, nullable=False)
    api_secret_encrypted = Column(Text, nullable=False)
    passphrase_encrypted = Column(Text, nullable=True)
    is_testnet = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


__all__ = [
    "Base",
    "ALLOWED_ORDER_FIELDS",
    "ALLOWED_STRATEGY_FIELDS",
    "User",
    "ApiKey",
    "Strategy",
    "OrderRecord",
    "TradeRecord",
    "TradeArchiveRecord",
    "PositionRecord",
    "SignalRecord",
    "CandleRecord",
    "RiskEvent",
    "BacktestResult",
    "PortfolioSnapshotRecord",
    "ExchangeCredential",
]
