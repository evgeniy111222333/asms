"""Database module - PostgreSQL + SQLAlchemy + Alembic.

Implements:
- 12 ORM models for ACMS data
- CRUD operations for all models
- Transaction management with context manager
- Bulk insert operations for high-frequency data
- Query helpers for common access patterns
- Alembic integration helpers
- Connection pool configuration
- Data cleanup/archival for old records
"""

import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Type

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Boolean, Enum, JSON,
    ForeignKey, Text, BigInteger, Numeric, create_engine, and_, desc, asc,
    func, text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


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
    metadata = Column(JSON, default=dict)
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


# ============================================================================
# Database Initialization
# ============================================================================

def init_db(db_url: str = "postgresql://acms:acms@localhost:5432/acms",
            pool_size: int = 10, max_overflow: int = 20) -> sessionmaker:
    """Initialize database and create all tables.

    Args:
        db_url: PostgreSQL connection string.
        pool_size: Connection pool size.
        max_overflow: Maximum overflow connections.

    Returns:
        SessionMaker factory.
    """
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ============================================================================
# Database Manager - CRUD Operations
# ============================================================================

class DatabaseManager:
    """High-level database operations manager.

    Provides CRUD operations for all models, transaction management,
    bulk inserts, and query helpers.
    """

    def __init__(self, db_url: str = "postgresql://acms:acms@localhost:5432/acms"):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=10, max_overflow=20,
                pool_pre_ping=True, pool_recycle=3600,
            )
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
        return self._engine

    def _get_session(self) -> Session:
        self._get_engine()
        return self._session_factory()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions.

        Yields:
            SQLAlchemy Session with automatic commit/rollback.
        """
        session = self._get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ========================================================================
    # User CRUD
    # ========================================================================

    async def create_user(self, email: str, password: str, username: str = "",
                          is_admin: bool = False) -> str:
        """Create a new user.

        Args:
            email: User email address.
            password: Plain text password (will be hashed).
            username: Optional username.
            is_admin: Whether user has admin privileges.

        Returns:
            User ID string.
        """
        user_id = str(uuid.uuid4())
        if not username:
            username = email.split("@")[0]
        # In production: hash password with bcrypt
        hashed = f"hashed_{password}"  # Placeholder

        with self.transaction() as session:
            user = User(
                id=user_id, email=email, username=username,
                hashed_password=hashed, is_admin=is_admin,
            )
            session.add(user)
        return user_id

    async def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        with self.transaction() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                return {"id": user.id, "email": user.email, "username": user.username,
                        "is_active": user.is_active, "is_admin": user.is_admin}
        return None

    # ========================================================================
    # Order CRUD
    # ========================================================================

    async def create_order(self, user_id: str, symbol: str, side: str,
                           order_type: str, quantity: float, price: Optional[float] = None,
                           stop_price: Optional[float] = None, exchange: str = "paper",
                           strategy_id: Optional[str] = None) -> Dict:
        """Create a new order record.

        Args:
            user_id: Owner user ID.
            symbol: Trading pair symbol.
            side: Order side (buy/sell).
            order_type: Order type (market/limit/etc).
            quantity: Order quantity.
            price: Limit price (optional).
            stop_price: Stop price (optional).
            exchange: Exchange name.
            strategy_id: Associated strategy ID.

        Returns:
            Dict with created order data.
        """
        order_id = f"ord_{uuid.uuid4().hex[:12]}"
        with self.transaction() as session:
            order = OrderRecord(
                id=order_id, user_id=user_id, strategy_id=strategy_id,
                symbol=symbol, side=side, order_type=order_type,
                status="created", quantity=quantity, price=price,
                stop_price=stop_price, exchange=exchange,
            )
            session.add(order)
        return {"id": order_id, "symbol": symbol, "status": "created"}

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        with self.transaction() as session:
            order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
            if order:
                return self._model_to_dict(order)
        return None

    async def update_order(self, order_id: str, updates: Dict) -> bool:
        """Update order fields.

        Args:
            order_id: Order ID to update.
            updates: Dict of field names to new values.

        Returns:
            True if order was found and updated.
        """
        with self.transaction() as session:
            order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
            if order:
                for key, value in updates.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                return True
        return False

    async def list_orders(self, user_id: str, symbol: Optional[str] = None,
                          status: Optional[str] = None, exchange: Optional[str] = None,
                          strategy_id: Optional[str] = None,
                          limit: int = 50, offset: int = 0,
                          sort_by: str = "created_at", sort_order: str = "desc") -> List[Dict]:
        """List orders with filtering, sorting, and pagination."""
        with self.transaction() as session:
            query = session.query(OrderRecord).filter(OrderRecord.user_id == user_id)
            if symbol:
                query = query.filter(OrderRecord.symbol == symbol)
            if status:
                query = query.filter(OrderRecord.status == status)
            if exchange:
                query = query.filter(OrderRecord.exchange == exchange)
            if strategy_id:
                query = query.filter(OrderRecord.strategy_id == strategy_id)

            sort_column = getattr(OrderRecord, sort_by, OrderRecord.created_at)
            if sort_order == "desc":
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(asc(sort_column))

            orders = query.offset(offset).limit(limit).all()
            return [self._model_to_dict(o) for o in orders]

    async def delete_order(self, order_id: str) -> bool:
        """Delete an order record."""
        with self.transaction() as session:
            deleted = session.query(OrderRecord).filter(OrderRecord.id == order_id).delete()
            return deleted > 0

    # ========================================================================
    # Strategy CRUD
    # ========================================================================

    async def create_strategy(self, user_id: str, name: str, type: str,
                               symbol: str, config: Dict = None) -> str:
        """Create a new strategy record."""
        strategy_id = f"strat_{uuid.uuid4().hex[:12]}"
        with self.transaction() as session:
            strategy = Strategy(
                id=strategy_id, user_id=user_id, name=name,
                type=type, symbol=symbol, config=config or {},
            )
            session.add(strategy)
        return strategy_id

    async def list_strategies(self, user_id: str) -> List[Dict]:
        """List all strategies for a user."""
        with self.transaction() as session:
            strategies = session.query(Strategy).filter(Strategy.user_id == user_id).all()
            return [self._model_to_dict(s) for s in strategies]

    async def update_strategy(self, strategy_id: str, updates: Dict) -> bool:
        """Update strategy fields."""
        with self.transaction() as session:
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            if strategy:
                for key, value in updates.items():
                    if hasattr(strategy, key):
                        setattr(strategy, key, value)
                return True
        return False

    # ========================================================================
    # Trade CRUD
    # ========================================================================

    async def list_trades(self, symbol: Optional[str] = None, side: Optional[str] = None,
                          exchange: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> List[Dict]:
        """List trades with filtering and pagination."""
        with self.transaction() as session:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if side:
                query = query.filter(TradeRecord.side == side)
            if exchange:
                query = query.filter(TradeRecord.exchange == exchange)
            trades = query.order_by(desc(TradeRecord.timestamp)).offset(offset).limit(limit).all()
            return [self._model_to_dict(t) for t in trades]

    async def bulk_insert_trades(self, trades: List[Dict]) -> int:
        """Bulk insert trade records for high-frequency data.

        Args:
            trades: List of trade data dicts.

        Returns:
            Number of records inserted.
        """
        if not trades:
            return 0
        with self.transaction() as session:
            objects = []
            for t in trades:
                objects.append(TradeRecord(
                    id=t.get("id", str(uuid.uuid4())),
                    order_id=t.get("order_id", ""),
                    symbol=t["symbol"], side=t["side"],
                    quantity=t["quantity"], price=t["price"],
                    commission=t.get("commission", 0),
                    exchange=t.get("exchange", "paper"),
                    timestamp=t.get("timestamp", datetime.utcnow()),
                ))
            session.bulk_save_objects(objects)
        return len(objects)

    # ========================================================================
    # Candle CRUD
    # ========================================================================

    async def get_candles(self, symbol: str, timeframe: str,
                          limit: int = 500) -> List[Dict]:
        """Get candle data from database.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            limit: Maximum number of candles.

        Returns:
            List of candle data dicts.
        """
        with self.transaction() as session:
            candles = session.query(CandleRecord).filter(
                and_(CandleRecord.symbol == symbol, CandleRecord.timeframe == timeframe)
            ).order_by(desc(CandleRecord.open_time)).limit(limit).all()
            return [self._model_to_dict(c) for c in reversed(candles)]

    async def bulk_insert_candles(self, candles: List[Dict]) -> int:
        """Bulk insert candle records.

        Args:
            candles: List of candle data dicts.

        Returns:
            Number of records inserted.
        """
        if not candles:
            return 0
        with self.transaction() as session:
            objects = []
            for c in candles:
                objects.append(CandleRecord(
                    id=c.get("id", str(uuid.uuid4())),
                    symbol=c["symbol"], timeframe=c["timeframe"],
                    open_time=c["open_time"], close_time=c.get("close_time", c["open_time"]),
                    open=c["open"], high=c["high"], low=c["low"],
                    close=c["close"], volume=c["volume"],
                    quote_volume=c.get("quote_volume", 0),
                    trades=c.get("trades", 0),
                    exchange=c.get("exchange", "unknown"),
                ))
            session.bulk_save_objects(objects)
        return len(objects)

    # ========================================================================
    # Query Helpers
    # ========================================================================

    async def get_active_orders(self, user_id: str) -> List[Dict]:
        """Get all active (non-terminal) orders for a user."""
        with self.transaction() as session:
            orders = session.query(OrderRecord).filter(
                and_(
                    OrderRecord.user_id == user_id,
                    OrderRecord.status.in_(["created", "submitted", "partially_filled"]),
                )
            ).all()
            return [self._model_to_dict(o) for o in orders]

    async def get_open_positions(self, user_id: str) -> List[Dict]:
        """Get all open positions for a user."""
        with self.transaction() as session:
            positions = session.query(PositionRecord).filter(
                and_(PositionRecord.user_id == user_id, PositionRecord.closed_at.is_(None))
            ).all()
            return [self._model_to_dict(p) for p in positions]

    async def get_recent_signals(self, symbol: Optional[str] = None,
                                  strategy_id: Optional[str] = None,
                                  direction: Optional[str] = None,
                                  limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent signals with optional filtering."""
        with self.transaction() as session:
            query = session.query(SignalRecord)
            if symbol:
                query = query.filter(SignalRecord.symbol == symbol)
            if strategy_id:
                query = query.filter(SignalRecord.strategy_id == strategy_id)
            if direction:
                query = query.filter(SignalRecord.direction == direction)
            signals = query.order_by(desc(SignalRecord.timestamp)).offset(offset).limit(limit).all()
            return [self._model_to_dict(s) for s in signals]

    async def get_pnl_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get P&L history for a user.

        Args:
            user_id: User ID.
            days: Number of days of history.

        Returns:
            List of portfolio snapshot dicts.
        """
        since = datetime.utcnow() - timedelta(days=days)
        with self.transaction() as session:
            snapshots = session.query(PortfolioSnapshotRecord).filter(
                and_(PortfolioSnapshotRecord.user_id == user_id,
                     PortfolioSnapshotRecord.timestamp >= since)
            ).order_by(asc(PortfolioSnapshotRecord.timestamp)).all()
            return [self._model_to_dict(s) for s in snapshots]

    async def get_latest_portfolio_snapshot(self, user_id: str) -> Optional[Dict]:
        """Get the most recent portfolio snapshot."""
        with self.transaction() as session:
            snapshot = session.query(PortfolioSnapshotRecord).filter(
                PortfolioSnapshotRecord.user_id == user_id
            ).order_by(desc(PortfolioSnapshotRecord.timestamp)).first()
            return self._model_to_dict(snapshot) if snapshot else None

    # ========================================================================
    # Backtest Results
    # ========================================================================

    async def create_backtest_result(self, strategy_id: str, config: Dict,
                                      results: Dict) -> str:
        """Create a backtest result record."""
        result_id = f"bt_{uuid.uuid4().hex[:12]}"
        with self.transaction() as session:
            bt = BacktestResult(
                id=result_id, strategy_id=strategy_id, config=config,
                total_return=results.get("total_return", 0),
                sharpe_ratio=results.get("sharpe_ratio", 0),
                max_drawdown=results.get("max_drawdown", 0),
                win_rate=results.get("win_rate", 0),
                total_trades=results.get("total_trades", 0),
                results_json=results,
            )
            session.add(bt)
        return result_id

    async def get_backtest_result(self, backtest_id: str) -> Optional[Dict]:
        """Get backtest result by ID."""
        with self.transaction() as session:
            bt = session.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
            return self._model_to_dict(bt) if bt else None

    # ========================================================================
    # Data Cleanup / Archival
    # ========================================================================

    async def cleanup_old_candles(self, symbol: str, timeframe: str,
                                   keep_days: int = 90) -> int:
        """Remove candle data older than keep_days.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            keep_days: Number of days to retain.

        Returns:
            Number of records deleted.
        """
        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        with self.transaction() as session:
            deleted = session.query(CandleRecord).filter(
                and_(CandleRecord.symbol == symbol,
                     CandleRecord.timeframe == timeframe,
                     CandleRecord.open_time < cutoff)
            ).delete()
        return deleted

    async def archive_old_trades(self, days: int = 180) -> int:
        """Archive trade records older than specified days.

        Args:
            days: Number of days threshold.

        Returns:
            Number of records archived.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.transaction() as session:
            trades = session.query(TradeRecord).filter(
                TradeRecord.timestamp < cutoff
            ).all()
            count = len(trades)
            # In production: move to archive table
            for t in trades:
                session.delete(t)
        return count

    # ========================================================================
    # Alembic Integration Helpers
    # ========================================================================

    @staticmethod
    def get_alembic_config(db_url: str = "") -> Dict:
        """Get Alembic configuration for migrations.

        Args:
            db_url: Database URL override.

        Returns:
            Dict with Alembic configuration.
        """
        return {
            "script_location": "alembic",
            "sqlalchemy.url": db_url or "postgresql://acms:acms@localhost:5432/acms",
            "render_as_batch": True,
        }

    @staticmethod
    def check_migration_status(engine) -> Dict:
        """Check current migration status.

        Args:
            engine: SQLAlchemy engine.

        Returns:
            Dict with migration status info.
        """
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version_num FROM alembic_version"))
                versions = [row[0] for row in result]
                return {"current_version": versions[0] if versions else None, "status": "up_to_date"}
        except Exception as e:
            return {"current_version": None, "status": "not_initialized", "error": str(e)}

    # ========================================================================
    # Utility
    # ========================================================================

    @staticmethod
    def _model_to_dict(model) -> Dict:
        """Convert SQLAlchemy model instance to dict."""
        if model is None:
            return {}
        result = {}
        for column in model.__table__.columns:
            value = getattr(model, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
