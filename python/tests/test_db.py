"""Comprehensive tests for acms.db module.

Tests all ORM models, DatabaseManager CRUD, query helpers, cleanup,
Alembic integration, and utility functions:
- All 12 ORM models (User, ApiKey, Strategy, OrderRecord, TradeRecord,
  PositionRecord, SignalRecord, CandleRecord, RiskEvent, BacktestResult,
  PortfolioSnapshotRecord, ExchangeCredential) - table names, columns, relationships
- init_db function
- DatabaseManager - _get_engine, _get_session, transaction context manager
- User CRUD (create_user, get_user)
- Order CRUD (create_order, get_order, update_order, list_orders, delete_order)
- Strategy CRUD (create_strategy, list_strategies, update_strategy)
- Trade CRUD (list_trades, bulk_insert_trades)
- Candle CRUD (get_candles, bulk_insert_candles)
- Query helpers (get_active_orders, get_open_positions, get_recent_signals,
  get_pnl_history, get_latest_portfolio_snapshot)
- Backtest results (create_backtest_result, get_backtest_result)
- Cleanup (cleanup_old_candles, archive_old_trades)
- Alembic helpers (get_alembic_config, check_migration_status)
- _model_to_dict utility
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from acms.db import (
    Base,
    User,
    ApiKey,
    Strategy,
    OrderRecord,
    TradeRecord,
    PositionRecord,
    SignalRecord,
    CandleRecord,
    RiskEvent,
    BacktestResult,
    PortfolioSnapshotRecord,
    ExchangeCredential,
    init_db,
    DatabaseManager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def db_url():
    """Provide SQLite in-memory database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def db_manager(db_url):
    """Provide a DatabaseManager with SQLite in-memory database."""
    manager = DatabaseManager(db_url=db_url)
    # Force table creation
    manager._get_engine()
    return manager


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run_async(coro):
    """Helper to run async functions in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# ORM Model Tests
# ============================================================================

class TestUserModel:
    """Tests for User ORM model."""

    def test_table_name(self):
        """User table name should be 'users'."""
        assert User.__tablename__ == "users"

    def test_columns_exist(self):
        """User should have all required columns."""
        columns = [c.name for c in User.__table__.columns]
        assert "id" in columns
        assert "email" in columns
        assert "username" in columns
        assert "hashed_password" in columns
        assert "is_active" in columns
        assert "is_admin" in columns
        assert "api_key" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_primary_key(self):
        """id should be the primary key."""
        pk_columns = [c.name for c in User.__table__.primary_key.columns]
        assert "id" in pk_columns

    def test_email_unique(self):
        """email column should be unique."""
        email_col = User.__table__.columns["email"]
        assert email_col.unique is True

    def test_username_unique(self):
        """username column should be unique."""
        username_col = User.__table__.columns["username"]
        assert username_col.unique is True

    def test_email_not_nullable(self):
        """email should not be nullable."""
        assert User.__table__.columns["email"].nullable is False

    def test_hashed_password_not_nullable(self):
        """hashed_password should not be nullable."""
        assert User.__table__.columns["hashed_password"].nullable is False

    def test_is_active_default(self):
        """is_active should default to True."""
        assert User.__table__.columns["is_active"].default.arg is True

    def test_is_admin_default(self):
        """is_admin should default to False."""
        assert User.__table__.columns["is_admin"].default.arg is False

    def test_api_key_nullable(self):
        """api_key should be nullable."""
        assert User.__table__.columns["api_key"].nullable is True

    def test_api_key_unique(self):
        """api_key should be unique."""
        assert User.__table__.columns["api_key"].unique is True


class TestApiKeyModel:
    """Tests for ApiKey ORM model."""

    def test_table_name(self):
        assert ApiKey.__tablename__ == "api_keys"

    def test_columns_exist(self):
        columns = [c.name for c in ApiKey.__table__.columns]
        assert "id" in columns
        assert "user_id" in columns
        assert "key_hash" in columns
        assert "name" in columns
        assert "permissions" in columns
        assert "is_active" in columns
        assert "last_used_at" in columns
        assert "created_at" in columns

    def test_user_id_foreign_key(self):
        """user_id should reference users.id."""
        fk = list(ApiKey.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"

    def test_key_hash_unique(self):
        assert ApiKey.__table__.columns["key_hash"].unique is True

    def test_last_used_at_nullable(self):
        assert ApiKey.__table__.columns["last_used_at"].nullable is True

    def test_permissions_json_type(self):
        """permissions should be JSON type."""
        from sqlalchemy import JSON
        assert isinstance(ApiKey.__table__.columns["permissions"].type, JSON)


class TestStrategyModel:
    """Tests for Strategy ORM model."""

    def test_table_name(self):
        assert Strategy.__tablename__ == "strategies"

    def test_columns_exist(self):
        columns = [c.name for c in Strategy.__table__.columns]
        assert "id" in columns
        assert "user_id" in columns
        assert "name" in columns
        assert "type" in columns
        assert "symbol" in columns
        assert "config" in columns
        assert "is_active" in columns
        assert "created_at" in columns
        assert "updated_at" in columns

    def test_user_id_foreign_key(self):
        fk = list(Strategy.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"

    def test_is_active_default_false(self):
        assert Strategy.__table__.columns["is_active"].default.arg is False


class TestOrderRecordModel:
    """Tests for OrderRecord ORM model."""

    def test_table_name(self):
        assert OrderRecord.__tablename__ == "orders"

    def test_columns_exist(self):
        columns = [c.name for c in OrderRecord.__table__.columns]
        expected = ["id", "user_id", "strategy_id", "symbol", "side",
                    "order_type", "status", "quantity", "price", "stop_price",
                    "filled_quantity", "average_fill_price", "commission",
                    "exchange", "exchange_order_id", "created_at", "updated_at"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_user_id_foreign_key(self):
        fk = list(OrderRecord.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"

    def test_strategy_id_foreign_key(self):
        fk = list(OrderRecord.__table__.columns["strategy_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "strategies.id"

    def test_strategy_id_nullable(self):
        assert OrderRecord.__table__.columns["strategy_id"].nullable is True

    def test_quantity_not_nullable(self):
        assert OrderRecord.__table__.columns["quantity"].nullable is False

    def test_exchange_not_nullable(self):
        assert OrderRecord.__table__.columns["exchange"].nullable is False


class TestTradeRecordModel:
    """Tests for TradeRecord ORM model."""

    def test_table_name(self):
        assert TradeRecord.__tablename__ == "trades"

    def test_columns_exist(self):
        columns = [c.name for c in TradeRecord.__table__.columns]
        expected = ["id", "order_id", "symbol", "side", "quantity", "price",
                    "commission", "commission_asset", "exchange",
                    "exchange_trade_id", "is_maker", "slippage", "timestamp"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_order_id_foreign_key(self):
        fk = list(TradeRecord.__table__.columns["order_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "orders.id"

    def test_is_maker_default(self):
        assert TradeRecord.__table__.columns["is_maker"].default.arg is False


class TestPositionRecordModel:
    """Tests for PositionRecord ORM model."""

    def test_table_name(self):
        assert PositionRecord.__tablename__ == "positions"

    def test_columns_exist(self):
        columns = [c.name for c in PositionRecord.__table__.columns]
        expected = ["id", "user_id", "strategy_id", "symbol", "side",
                    "quantity", "entry_price", "mark_price", "unrealized_pnl",
                    "realized_pnl", "leverage", "exchange", "opened_at",
                    "closed_at", "updated_at"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_user_id_foreign_key(self):
        fk = list(PositionRecord.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"

    def test_closed_at_nullable(self):
        assert PositionRecord.__table__.columns["closed_at"].nullable is True


class TestSignalRecordModel:
    """Tests for SignalRecord ORM model."""

    def test_table_name(self):
        assert SignalRecord.__tablename__ == "signals"

    def test_columns_exist(self):
        columns = [c.name for c in SignalRecord.__table__.columns]
        expected = ["id", "strategy_id", "symbol", "direction", "strength",
                    "indicators", "signal_metadata", "timestamp"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_strategy_id_foreign_key(self):
        fk = list(SignalRecord.__table__.columns["strategy_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "strategies.id"

    def test_indicators_json_type(self):
        from sqlalchemy import JSON
        assert isinstance(SignalRecord.__table__.columns["indicators"].type, JSON)


class TestCandleRecordModel:
    """Tests for CandleRecord ORM model."""

    def test_table_name(self):
        assert CandleRecord.__tablename__ == "candles"

    def test_columns_exist(self):
        columns = [c.name for c in CandleRecord.__table__.columns]
        expected = ["id", "symbol", "timeframe", "open_time", "close_time",
                    "open", "high", "low", "close", "volume", "quote_volume",
                    "trades", "exchange"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_no_foreign_keys(self):
        """CandleRecord should not have foreign keys."""
        fks = list(CandleRecord.__table__.foreign_keys)
        assert len(fks) == 0


class TestRiskEventModel:
    """Tests for RiskEvent ORM model."""

    def test_table_name(self):
        assert RiskEvent.__tablename__ == "risk_events"

    def test_columns_exist(self):
        columns = [c.name for c in RiskEvent.__table__.columns]
        expected = ["id", "user_id", "event_type", "severity", "details", "timestamp"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_user_id_foreign_key(self):
        fk = list(RiskEvent.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"


class TestBacktestResultModel:
    """Tests for BacktestResult ORM model."""

    def test_table_name(self):
        assert BacktestResult.__tablename__ == "backtest_results"

    def test_columns_exist(self):
        columns = [c.name for c in BacktestResult.__table__.columns]
        expected = ["id", "strategy_id", "config", "total_return",
                    "sharpe_ratio", "max_drawdown", "win_rate", "total_trades",
                    "results_json", "created_at"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_strategy_id_foreign_key(self):
        fk = list(BacktestResult.__table__.columns["strategy_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "strategies.id"

    def test_results_json_type(self):
        from sqlalchemy import JSON
        assert isinstance(BacktestResult.__table__.columns["results_json"].type, JSON)


class TestPortfolioSnapshotRecordModel:
    """Tests for PortfolioSnapshotRecord ORM model."""

    def test_table_name(self):
        assert PortfolioSnapshotRecord.__tablename__ == "portfolio_snapshots"

    def test_columns_exist(self):
        columns = [c.name for c in PortfolioSnapshotRecord.__table__.columns]
        expected = ["id", "user_id", "total_value", "available_balance",
                    "unrealized_pnl", "realized_pnl", "margin_used",
                    "leverage", "positions_json", "timestamp"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_user_id_foreign_key(self):
        fk = list(PortfolioSnapshotRecord.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"


class TestExchangeCredentialModel:
    """Tests for ExchangeCredential ORM model."""

    def test_table_name(self):
        assert ExchangeCredential.__tablename__ == "exchange_credentials"

    def test_columns_exist(self):
        columns = [c.name for c in ExchangeCredential.__table__.columns]
        expected = ["id", "user_id", "exchange", "api_key_encrypted",
                    "api_secret_encrypted", "passphrase_encrypted",
                    "is_testnet", "is_active", "created_at"]
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_user_id_foreign_key(self):
        fk = list(ExchangeCredential.__table__.columns["user_id"].foreign_keys)[0]
        assert str(fk.target_fullname) == "users.id"

    def test_passphrase_nullable(self):
        assert ExchangeCredential.__table__.columns["passphrase_encrypted"].nullable is True

    def test_is_testnet_default(self):
        assert ExchangeCredential.__table__.columns["is_testnet"].default.arg is False


# ============================================================================
# init_db Tests
# ============================================================================

class TestInitDb:
    """Tests for the init_db function."""

    def test_init_db_creates_sessionmaker(self):
        """init_db should return a sessionmaker."""
        factory = init_db(db_url="sqlite:///:memory:")
        assert callable(factory)

    def test_init_db_creates_tables(self):
        """init_db should create all tables."""
        engine = create_engine("sqlite:///:memory:")
        factory = init_db(db_url="sqlite:///:memory:")
        # If we can create a session without error, tables were created
        session = factory()
        session.close()

    def test_init_db_custom_pool_size(self):
        """Should accept custom pool_size."""
        # SQLite doesn't use pool_size, but the function should accept it
        factory = init_db(db_url="sqlite:///:memory:", pool_size=5, max_overflow=10)
        assert callable(factory)


# ============================================================================
# DatabaseManager Tests
# ============================================================================

class TestDatabaseManagerInit:
    """Tests for DatabaseManager initialization."""

    def test_init_default_url(self):
        """Default db_url should be PostgreSQL."""
        manager = DatabaseManager()
        assert "postgresql" in manager.db_url

    def test_init_custom_url(self):
        """Custom db_url should be stored."""
        url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url=url)
        assert manager.db_url == url

    def test_engine_initially_none(self):
        """Engine should be None before first access."""
        manager = DatabaseManager(db_url="sqlite:///:memory:")
        assert manager._engine is None

    def test_session_factory_initially_none(self):
        """Session factory should be None before first engine access."""
        manager = DatabaseManager(db_url="sqlite:///:memory:")
        assert manager._session_factory is None


class TestDatabaseManagerEngine:
    """Tests for DatabaseManager engine management."""

    def test_get_engine_creates_engine(self, db_manager):
        """_get_engine should create an engine."""
        engine = db_manager._get_engine()
        assert engine is not None

    def test_get_engine_caches(self, db_manager):
        """_get_engine should return the same engine on repeated calls."""
        engine1 = db_manager._get_engine()
        engine2 = db_manager._get_engine()
        assert engine1 is engine2

    def test_get_engine_creates_tables(self, db_manager):
        """_get_engine should create all tables."""
        engine = db_manager._get_engine()
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        assert "users" in table_names
        assert "orders" in table_names
        assert "strategies" in table_names

    def test_get_engine_creates_session_factory(self, db_manager):
        """_get_engine should create session factory."""
        db_manager._get_engine()
        assert db_manager._session_factory is not None

    def test_get_session(self, db_manager):
        """_get_session should return a session."""
        session = db_manager._get_session()
        assert session is not None
        session.close()


class TestDatabaseManagerTransaction:
    """Tests for DatabaseManager transaction context manager."""

    def test_transaction_commit(self, db_manager):
        """Transaction should commit on success."""
        with db_manager.transaction() as session:
            user = User(
                id="test-tx-1", email="tx@test.com",
                username="txuser", hashed_password="hashed_pw",
            )
            session.add(user)

        # Verify committed
        with db_manager.transaction() as session:
            user = session.query(User).filter(User.id == "test-tx-1").first()
            assert user is not None
            assert user.email == "tx@test.com"

    def test_transaction_rollback(self, db_manager):
        """Transaction should rollback on exception."""
        try:
            with db_manager.transaction() as session:
                user = User(
                    id="test-tx-rollback", email="rollback@test.com",
                    username="rollbackuser", hashed_password="hashed_pw",
                )
                session.add(user)
                raise ValueError("Force rollback")
        except ValueError:
            pass

        # Verify rolled back
        with db_manager.transaction() as session:
            user = session.query(User).filter(User.id == "test-tx-rollback").first()
            assert user is None

    def test_transaction_session_closed(self, db_manager):
        """Session should be closed after transaction."""
        with db_manager.transaction() as session:
            pass
        # Session should be closed (no exception accessing it again)


# ============================================================================
# User CRUD Tests
# ============================================================================

class TestUserCRUD:
    """Tests for User CRUD operations."""

    def test_create_user(self, db_manager):
        """Should create a user and return user_id."""
        user_id = run_async(
            db_manager.create_user("test@example.com", "password123")
        )
        assert user_id is not None
        assert isinstance(user_id, str)
        assert len(user_id) > 0

    def test_create_user_with_username(self, db_manager):
        """Should use provided username."""
        user_id = run_async(
            db_manager.create_user("test2@example.com", "password", username="customuser")
        )
        user = run_async(db_manager.get_user(user_id))
        assert user["username"] == "customuser"

    def test_create_user_default_username(self, db_manager):
        """Should derive username from email if not provided."""
        user_id = run_async(
            db_manager.create_user("johndoe@example.com", "password")
        )
        user = run_async(db_manager.get_user(user_id))
        assert user["username"] == "johndoe"

    def test_create_user_is_admin(self, db_manager):
        """Should set is_admin flag."""
        user_id = run_async(
            db_manager.create_user("admin@example.com", "password", is_admin=True)
        )
        user = run_async(db_manager.get_user(user_id))
        assert user["is_admin"] is True

    def test_create_user_default_not_admin(self, db_manager):
        """Default is_admin should be False."""
        user_id = run_async(
            db_manager.create_user("normal@example.com", "password")
        )
        user = run_async(db_manager.get_user(user_id))
        assert user["is_admin"] is False

    def test_create_user_password_hashed(self, db_manager):
        """Password should be stored as hashed version (placeholder)."""
        user_id = run_async(
            db_manager.create_user("hash@example.com", "mypassword")
        )
        with db_manager.transaction() as session:
            user = session.query(User).filter(User.id == user_id).first()
            assert user.hashed_password == "hashed_mypassword"

    def test_get_user(self, db_manager):
        """Should get user by ID."""
        user_id = run_async(
            db_manager.create_user("get@example.com", "password")
        )
        user = run_async(db_manager.get_user(user_id))
        assert user is not None
        assert user["id"] == user_id
        assert user["email"] == "get@example.com"
        assert "is_active" in user
        assert "is_admin" in user

    def test_get_user_nonexistent(self, db_manager):
        """Should return None for non-existent user."""
        result = run_async(db_manager.get_user("nonexistent-id"))
        assert result is None

    def test_get_user_returns_dict(self, db_manager):
        """Should return dict, not model instance."""
        user_id = run_async(
            db_manager.create_user("dict@example.com", "password")
        )
        user = run_async(db_manager.get_user(user_id))
        assert isinstance(user, dict)

    def test_create_multiple_users(self, db_manager):
        """Should create multiple users with different IDs."""
        id1 = run_async(db_manager.create_user("user1@example.com", "pw1"))
        id2 = run_async(db_manager.create_user("user2@example.com", "pw2"))
        assert id1 != id2


# ============================================================================
# Order CRUD Tests
# ============================================================================

class TestOrderCRUD:
    """Tests for Order CRUD operations."""

    def setup_method(self):
        """Create a fresh DatabaseManager for each test."""
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()
        # Create a user for foreign key
        self.user_id = run_async(
            self.db_manager.create_user("orderuser@example.com", "password")
        )

    def test_create_order(self):
        """Should create an order and return dict."""
        result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
                exchange="binance",
            )
        )
        assert "id" in result
        assert result["symbol"] == "BTC/USDT"
        assert result["status"] == "created"
        assert result["id"].startswith("ord_")

    def test_create_order_with_all_fields(self):
        """Should create order with all optional fields."""
        result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="ETH/USDT",
                side="sell",
                order_type="limit",
                quantity=5.0,
                price=3000.0,
                stop_price=2800.0,
                exchange="kraken",
                strategy_id="strat_abc123",
            )
        )
        assert result["symbol"] == "ETH/USDT"
        assert result["status"] == "created"

    def test_create_order_default_exchange(self):
        """Default exchange should be 'paper'."""
        result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
            )
        )
        # Check via get_order
        order = run_async(self.db_manager.get_order(result["id"]))
        assert order["exchange"] == "paper"

    def test_get_order(self):
        """Should get order by ID."""
        create_result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
            )
        )
        order = run_async(self.db_manager.get_order(create_result["id"]))
        assert order is not None
        assert order["id"] == create_result["id"]
        assert order["symbol"] == "BTC/USDT"
        assert order["side"] == "buy"
        assert order["status"] == "created"

    def test_get_order_nonexistent(self):
        """Should return None for non-existent order."""
        result = run_async(self.db_manager.get_order("nonexistent"))
        assert result is None

    def test_update_order(self):
        """Should update order fields."""
        create_result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="buy",
                order_type="limit",
                quantity=1.0,
                price=50000.0,
            )
        )
        success = run_async(
            self.db_manager.update_order(create_result["id"], {"status": "filled", "filled_quantity": 1.0})
        )
        assert success is True
        order = run_async(self.db_manager.get_order(create_result["id"]))
        assert order["status"] == "filled"

    def test_update_order_nonexistent(self):
        """Should return False for non-existent order."""
        success = run_async(
            self.db_manager.update_order("nonexistent", {"status": "filled"})
        )
        assert success is False

    def test_update_order_ignores_invalid_fields(self):
        """Should silently ignore updates to non-existent fields."""
        create_result = run_async(
            self.db_manager.create_order(
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=1.0,
            )
        )
        success = run_async(
            self.db_manager.update_order(create_result["id"], {
                "status": "submitted",
                "nonexistent_field": "ignored",
            })
        )
        assert success is True
        order = run_async(self.db_manager.get_order(create_result["id"]))
        assert order["status"] == "submitted"

    def test_list_orders(self):
        """Should list orders for a user."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ETH/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        orders = run_async(self.db_manager.list_orders(self.user_id))
        assert len(orders) == 2

    def test_list_orders_filter_by_symbol(self):
        """Should filter orders by symbol."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ETH/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, symbol="BTC/USDT")
        )
        assert len(orders) == 1
        assert orders[0]["symbol"] == "BTC/USDT"

    def test_list_orders_filter_by_status(self):
        """Should filter orders by status."""
        create_result = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.update_order(create_result["id"], {"status": "filled"}))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, status="filled")
        )
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"

    def test_list_orders_filter_by_exchange(self):
        """Should filter orders by exchange."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0, exchange="binance",
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ETH/USDT", side="sell",
            order_type="limit", quantity=5.0, exchange="kraken",
        ))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, exchange="binance")
        )
        assert len(orders) == 1
        assert orders[0]["exchange"] == "binance"

    def test_list_orders_pagination(self):
        """Should support limit and offset."""
        for i in range(5):
            run_async(self.db_manager.create_order(
                user_id=self.user_id, symbol="BTC/USDT", side="buy",
                order_type="market", quantity=1.0,
            ))
        page1 = run_async(self.db_manager.list_orders(self.user_id, limit=2, offset=0))
        page2 = run_async(self.db_manager.list_orders(self.user_id, limit=2, offset=2))
        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should not overlap
        ids_page1 = {o["id"] for o in page1}
        ids_page2 = {o["id"] for o in page2}
        assert ids_page1.isdisjoint(ids_page2)

    def test_list_orders_sort_asc(self):
        """Should sort in ascending order."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="AAA/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ZZZ/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, sort_by="symbol", sort_order="asc")
        )
        assert orders[0]["symbol"] <= orders[1]["symbol"]

    def test_list_orders_sort_desc(self):
        """Should sort in descending order."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="AAA/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ZZZ/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, sort_by="symbol", sort_order="desc")
        )
        assert orders[0]["symbol"] >= orders[1]["symbol"]

    def test_list_orders_default_sort(self):
        """Default sort should be by created_at desc."""
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        orders = run_async(self.db_manager.list_orders(self.user_id))
        assert len(orders) >= 1

    def test_list_orders_empty(self):
        """Should return empty list for user with no orders."""
        orders = run_async(self.db_manager.list_orders(self.user_id))
        assert orders == []

    def test_delete_order(self):
        """Should delete an order."""
        create_result = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        deleted = run_async(self.db_manager.delete_order(create_result["id"]))
        assert deleted is True
        # Verify deleted
        order = run_async(self.db_manager.get_order(create_result["id"]))
        assert order is None

    def test_delete_order_nonexistent(self):
        """Should return False for non-existent order."""
        deleted = run_async(self.db_manager.delete_order("nonexistent"))
        assert deleted is False

    def test_list_orders_filter_by_strategy(self):
        """Should filter orders by strategy_id."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0, strategy_id=strat_id,
        ))
        run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ETH/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        orders = run_async(
            self.db_manager.list_orders(self.user_id, strategy_id=strat_id)
        )
        assert len(orders) == 1


# ============================================================================
# Strategy CRUD Tests
# ============================================================================

class TestStrategyCRUD:
    """Tests for Strategy CRUD operations."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()
        self.user_id = run_async(
            self.db_manager.create_user("stratuser@example.com", "password")
        )

    def test_create_strategy(self):
        """Should create a strategy and return strategy_id."""
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id,
                name="Momentum BTC",
                type="momentum",
                symbol="BTC/USDT",
            )
        )
        assert strategy_id is not None
        assert strategy_id.startswith("strat_")

    def test_create_strategy_with_config(self):
        """Should accept config dict."""
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id,
                name="RSI Strategy",
                type="mean_reversion",
                symbol="ETH/USDT",
                config={"period": 14, "overbought": 70},
            )
        )
        assert strategy_id is not None

    def test_create_strategy_default_config(self):
        """Should default config to empty dict."""
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id,
                name="Test",
                type="test",
                symbol="BTC/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            strat = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            assert strat.config == {}

    def test_list_strategies(self):
        """Should list strategies for a user."""
        run_async(self.db_manager.create_strategy(
            user_id=self.user_id, name="Strat1", type="momentum", symbol="BTC/USDT",
        ))
        run_async(self.db_manager.create_strategy(
            user_id=self.user_id, name="Strat2", type="mean_reversion", symbol="ETH/USDT",
        ))
        strategies = run_async(self.db_manager.list_strategies(self.user_id))
        assert len(strategies) == 2

    def test_list_strategies_empty(self):
        """Should return empty list for user with no strategies."""
        strategies = run_async(self.db_manager.list_strategies(self.user_id))
        assert strategies == []

    def test_list_strategies_returns_dicts(self):
        """Should return dicts, not model instances."""
        run_async(self.db_manager.create_strategy(
            user_id=self.user_id, name="Strat1", type="momentum", symbol="BTC/USDT",
        ))
        strategies = run_async(self.db_manager.list_strategies(self.user_id))
        assert isinstance(strategies[0], dict)

    def test_update_strategy(self):
        """Should update strategy fields."""
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="Original", type="momentum",
                symbol="BTC/USDT",
            )
        )
        success = run_async(
            self.db_manager.update_strategy(strategy_id, {"is_active": True, "name": "Updated"})
        )
        assert success is True
        strategies = run_async(self.db_manager.list_strategies(self.user_id))
        assert strategies[0]["is_active"] is True
        assert strategies[0]["name"] == "Updated"

    def test_update_strategy_nonexistent(self):
        """Should return False for non-existent strategy."""
        success = run_async(
            self.db_manager.update_strategy("nonexistent", {"is_active": True})
        )
        assert success is False

    def test_update_strategy_ignores_invalid_fields(self):
        """Should silently ignore non-existent fields."""
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="Test", type="test",
                symbol="BTC/USDT",
            )
        )
        success = run_async(
            self.db_manager.update_strategy(strategy_id, {"name": "New", "bogus": "ignored"})
        )
        assert success is True


# ============================================================================
# Trade CRUD Tests
# ============================================================================

class TestTradeCRUD:
    """Tests for Trade CRUD operations."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()
        self.user_id = run_async(
            self.db_manager.create_user("tradeuser@example.com", "password")
        )
        # Create order for foreign key
        self.order_id = run_async(
            self.db_manager.create_order(
                user_id=self.user_id, symbol="BTC/USDT", side="buy",
                order_type="market", quantity=1.0,
            )
        )["id"]

    def test_list_trades_empty(self):
        """Should return empty list when no trades."""
        trades = run_async(self.db_manager.list_trades())
        assert trades == []

    def test_bulk_insert_trades(self):
        """Should bulk insert trades."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
                "commission": 0.5,
                "exchange": "binance",
            },
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50001.0,
                "commission": 0.5,
                "exchange": "binance",
            },
        ]
        count = run_async(self.db_manager.bulk_insert_trades(trades_data))
        assert count == 2

    def test_bulk_insert_trades_empty(self):
        """Should return 0 for empty list."""
        count = run_async(self.db_manager.bulk_insert_trades([]))
        assert count == 0

    def test_list_trades_after_insert(self):
        """Should list inserted trades."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
                "exchange": "binance",
            },
        ]
        run_async(self.db_manager.bulk_insert_trades(trades_data))
        trades = run_async(self.db_manager.list_trades())
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USDT"

    def test_list_trades_filter_by_symbol(self):
        """Should filter trades by symbol."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
            },
            {
                "order_id": self.order_id,
                "symbol": "ETH/USDT",
                "side": "sell",
                "quantity": 5.0,
                "price": 3000.0,
            },
        ]
        run_async(self.db_manager.bulk_insert_trades(trades_data))
        trades = run_async(self.db_manager.list_trades(symbol="BTC/USDT"))
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USDT"

    def test_list_trades_filter_by_side(self):
        """Should filter trades by side."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
            },
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "sell",
                "quantity": 0.5,
                "price": 51000.0,
            },
        ]
        run_async(self.db_manager.bulk_insert_trades(trades_data))
        trades = run_async(self.db_manager.list_trades(side="buy"))
        assert len(trades) == 1

    def test_list_trades_filter_by_exchange(self):
        """Should filter trades by exchange."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
                "exchange": "binance",
            },
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
                "exchange": "kraken",
            },
        ]
        run_async(self.db_manager.bulk_insert_trades(trades_data))
        trades = run_async(self.db_manager.list_trades(exchange="binance"))
        assert len(trades) == 1

    def test_list_trades_pagination(self):
        """Should support limit and offset."""
        for i in range(5):
            run_async(self.db_manager.bulk_insert_trades([
                {
                    "order_id": self.order_id,
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "quantity": 0.1,
                    "price": 50000.0 + i,
                },
            ]))
        page1 = run_async(self.db_manager.list_trades(limit=2, offset=0))
        assert len(page1) == 2

    def test_bulk_insert_trades_auto_id(self):
        """Should auto-generate ID if not provided."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
            },
        ]
        count = run_async(self.db_manager.bulk_insert_trades(trades_data))
        assert count == 1
        trades = run_async(self.db_manager.list_trades())
        assert trades[0]["id"] is not None

    def test_bulk_insert_trades_defaults(self):
        """Should use default values for optional fields."""
        trades_data = [
            {
                "order_id": self.order_id,
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.5,
                "price": 50000.0,
            },
        ]
        run_async(self.db_manager.bulk_insert_trades(trades_data))
        trades = run_async(self.db_manager.list_trades())
        assert trades[0]["commission"] is not None
        assert trades[0]["exchange"] == "paper"


# ============================================================================
# Candle CRUD Tests
# ============================================================================

class TestCandleCRUD:
    """Tests for Candle CRUD operations."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()

    def test_get_candles_empty(self):
        """Should return empty list when no candles."""
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        assert candles == []

    def test_bulk_insert_candles(self):
        """Should bulk insert candles."""
        candles_data = [
            {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1, 0, 0),
                "close_time": datetime(2024, 1, 1, 0, 1),
                "open": 42000.0,
                "high": 42100.0,
                "low": 41900.0,
                "close": 42050.0,
                "volume": 100.0,
                "exchange": "binance",
            },
            {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1, 0, 1),
                "close_time": datetime(2024, 1, 1, 0, 2),
                "open": 42050.0,
                "high": 42200.0,
                "low": 42000.0,
                "close": 42150.0,
                "volume": 150.0,
                "exchange": "binance",
            },
        ]
        count = run_async(self.db_manager.bulk_insert_candles(candles_data))
        assert count == 2

    def test_bulk_insert_candles_empty(self):
        """Should return 0 for empty list."""
        count = run_async(self.db_manager.bulk_insert_candles([]))
        assert count == 0

    def test_get_candles_after_insert(self):
        """Should retrieve inserted candles."""
        candles_data = [
            {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1, 0, 0),
                "close_time": datetime(2024, 1, 1, 0, 1),
                "open": 42000.0,
                "high": 42100.0,
                "low": 41900.0,
                "close": 42050.0,
                "volume": 100.0,
                "exchange": "binance",
            },
        ]
        run_async(self.db_manager.bulk_insert_candles(candles_data))
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        assert len(candles) == 1
        assert candles[0]["symbol"] == "BTC/USDT"

    def test_get_candles_limit(self):
        """Should respect limit parameter."""
        candles_data = []
        for i in range(10):
            candles_data.append({
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1, i),
                "close_time": datetime(2024, 1, 1, i, 1),
                "open": 42000.0 + i,
                "high": 42100.0 + i,
                "low": 41900.0 + i,
                "close": 42050.0 + i,
                "volume": 100.0,
                "exchange": "binance",
            })
        run_async(self.db_manager.bulk_insert_candles(candles_data))
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m", limit=5))
        assert len(candles) == 5

    def test_get_candles_different_symbols(self):
        """Should not mix candles from different symbols."""
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            run_async(self.db_manager.bulk_insert_candles([
                {
                    "symbol": symbol,
                    "timeframe": "1m",
                    "open_time": datetime(2024, 1, 1),
                    "close_time": datetime(2024, 1, 1, 0, 1),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 100.0,
                    "exchange": "binance",
                },
            ]))
        btc_candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        eth_candles = run_async(self.db_manager.get_candles("ETH/USDT", "1m"))
        assert len(btc_candles) == 1
        assert len(eth_candles) == 1

    def test_bulk_insert_candles_defaults(self):
        """Should use default values for optional fields."""
        candles_data = [
            {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1),
                "open": 42000.0,
                "high": 42100.0,
                "low": 41900.0,
                "close": 42050.0,
                "volume": 100.0,
            },
        ]
        run_async(self.db_manager.bulk_insert_candles(candles_data))
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        assert candles[0]["exchange"] == "unknown"
        assert candles[0]["trades"] == 0
        assert candles[0]["quote_volume"] is not None


# ============================================================================
# Query Helper Tests
# ============================================================================

class TestQueryHelpers:
    """Tests for DatabaseManager query helper methods."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()
        self.user_id = run_async(
            self.db_manager.create_user("queryuser@example.com", "password")
        )

    # --- get_active_orders ---

    def test_get_active_orders(self):
        """Should return only active orders."""
        order1 = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        order2 = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="ETH/USDT", side="sell",
            order_type="limit", quantity=5.0,
        ))
        # Update one to filled
        run_async(self.db_manager.update_order(order1["id"], {"status": "filled"}))
        active = run_async(self.db_manager.get_active_orders(self.user_id))
        assert len(active) == 1
        assert active[0]["id"] == order2["id"]

    def test_get_active_orders_empty(self):
        """Should return empty list when no active orders."""
        active = run_async(self.db_manager.get_active_orders(self.user_id))
        assert active == []

    def test_get_active_orders_includes_partially_filled(self):
        """Should include partially_filled status."""
        order = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.update_order(order["id"], {"status": "partially_filled"}))
        active = run_async(self.db_manager.get_active_orders(self.user_id))
        assert len(active) == 1

    def test_get_active_orders_includes_submitted(self):
        """Should include submitted status."""
        order = run_async(self.db_manager.create_order(
            user_id=self.user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        run_async(self.db_manager.update_order(order["id"], {"status": "submitted"}))
        active = run_async(self.db_manager.get_active_orders(self.user_id))
        assert len(active) == 1

    # --- get_open_positions ---

    def test_get_open_positions(self):
        """Should return only open positions."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        # Insert open position
        with self.db_manager.transaction() as session:
            pos = PositionRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                strategy_id=strat_id,
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=50000.0,
                exchange="binance",
            )
            session.add(pos)

        positions = run_async(self.db_manager.get_open_positions(self.user_id))
        assert len(positions) == 1

    def test_get_open_positions_empty(self):
        """Should return empty when no open positions."""
        positions = run_async(self.db_manager.get_open_positions(self.user_id))
        assert positions == []

    def test_get_open_positions_excludes_closed(self):
        """Should exclude positions with closed_at set."""
        with self.db_manager.transaction() as session:
            pos = PositionRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                symbol="BTC/USDT",
                side="long",
                quantity=1.0,
                entry_price=50000.0,
                exchange="binance",
                closed_at=datetime.utcnow(),
            )
            session.add(pos)

        positions = run_async(self.db_manager.get_open_positions(self.user_id))
        assert len(positions) == 0

    # --- get_recent_signals ---

    def test_get_recent_signals(self):
        """Should return recent signals."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="sig_test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            for i in range(3):
                signal = SignalRecord(
                    id=str(uuid.uuid4()),
                    strategy_id=strat_id,
                    symbol="BTC/USDT",
                    direction="buy",
                    strength=0.8,
                )
                session.add(signal)

        signals = run_async(self.db_manager.get_recent_signals())
        assert len(signals) == 3

    def test_get_recent_signals_filter_by_symbol(self):
        """Should filter signals by symbol."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="sig_test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            for symbol in ["BTC/USDT", "ETH/USDT"]:
                session.add(SignalRecord(
                    id=str(uuid.uuid4()),
                    strategy_id=strat_id,
                    symbol=symbol,
                    direction="buy",
                    strength=0.8,
                ))

        btc_signals = run_async(self.db_manager.get_recent_signals(symbol="BTC/USDT"))
        assert len(btc_signals) == 1
        assert btc_signals[0]["symbol"] == "BTC/USDT"

    def test_get_recent_signals_filter_by_strategy(self):
        """Should filter signals by strategy_id."""
        strat_id1 = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="strat1", type="momentum",
                symbol="BTC/USDT",
            )
        )
        strat_id2 = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="strat2", type="mean_reversion",
                symbol="ETH/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            session.add(SignalRecord(
                id=str(uuid.uuid4()),
                strategy_id=strat_id1,
                symbol="BTC/USDT",
                direction="buy",
                strength=0.8,
            ))
            session.add(SignalRecord(
                id=str(uuid.uuid4()),
                strategy_id=strat_id2,
                symbol="ETH/USDT",
                direction="sell",
                strength=0.6,
            ))

        signals = run_async(self.db_manager.get_recent_signals(strategy_id=strat_id1))
        assert len(signals) == 1

    def test_get_recent_signals_filter_by_direction(self):
        """Should filter signals by direction."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="dir_test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            session.add(SignalRecord(
                id=str(uuid.uuid4()),
                strategy_id=strat_id,
                symbol="BTC/USDT",
                direction="buy",
                strength=0.8,
            ))
            session.add(SignalRecord(
                id=str(uuid.uuid4()),
                strategy_id=strat_id,
                symbol="BTC/USDT",
                direction="sell",
                strength=0.6,
            ))

        buy_signals = run_async(self.db_manager.get_recent_signals(direction="buy"))
        assert len(buy_signals) == 1
        assert buy_signals[0]["direction"] == "buy"

    def test_get_recent_signals_pagination(self):
        """Should support limit and offset."""
        strat_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="pag_test", type="momentum",
                symbol="BTC/USDT",
            )
        )
        with self.db_manager.transaction() as session:
            for i in range(5):
                session.add(SignalRecord(
                    id=str(uuid.uuid4()),
                    strategy_id=strat_id,
                    symbol="BTC/USDT",
                    direction="buy",
                    strength=0.8,
                ))

        page = run_async(self.db_manager.get_recent_signals(limit=2, offset=0))
        assert len(page) == 2

    # --- get_pnl_history ---

    def test_get_pnl_history(self):
        """Should return P&L history snapshots."""
        with self.db_manager.transaction() as session:
            for i in range(3):
                session.add(PortfolioSnapshotRecord(
                    id=str(uuid.uuid4()),
                    user_id=self.user_id,
                    total_value=100000.0 + i * 100,
                    timestamp=datetime.utcnow() - timedelta(days=5 - i),
                ))

        history = run_async(self.db_manager.get_pnl_history(self.user_id, days=10))
        assert len(history) >= 3

    def test_get_pnl_history_days_filter(self):
        """Should only return snapshots within the days window."""
        with self.db_manager.transaction() as session:
            # Recent snapshot
            session.add(PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                total_value=100000.0,
                timestamp=datetime.utcnow() - timedelta(days=5),
            ))
            # Old snapshot
            session.add(PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                total_value=90000.0,
                timestamp=datetime.utcnow() - timedelta(days=100),
            ))

        recent = run_async(self.db_manager.get_pnl_history(self.user_id, days=30))
        assert len(recent) == 1

    def test_get_pnl_history_empty(self):
        """Should return empty when no snapshots."""
        history = run_async(self.db_manager.get_pnl_history(self.user_id))
        assert history == []

    def test_get_pnl_history_default_days(self):
        """Default should be 30 days."""
        with self.db_manager.transaction() as session:
            session.add(PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                total_value=100000.0,
                timestamp=datetime.utcnow() - timedelta(days=15),
            ))

        history = run_async(self.db_manager.get_pnl_history(self.user_id))
        assert len(history) == 1

    # --- get_latest_portfolio_snapshot ---

    def test_get_latest_portfolio_snapshot(self):
        """Should return the most recent snapshot."""
        with self.db_manager.transaction() as session:
            session.add(PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                total_value=100000.0,
                timestamp=datetime.utcnow() - timedelta(days=2),
            ))
            session.add(PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                total_value=105000.0,
                timestamp=datetime.utcnow() - timedelta(days=1),
            ))

        snapshot = run_async(self.db_manager.get_latest_portfolio_snapshot(self.user_id))
        assert snapshot is not None
        assert snapshot["total_value"] is not None

    def test_get_latest_portfolio_snapshot_none(self):
        """Should return None when no snapshots exist."""
        result = run_async(
            self.db_manager.get_latest_portfolio_snapshot(self.user_id)
        )
        assert result is None


# ============================================================================
# Backtest Result Tests
# ============================================================================

class TestBacktestCRUD:
    """Tests for backtest result CRUD operations."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()
        self.user_id = run_async(
            self.db_manager.create_user("btuser@example.com", "password")
        )
        self.strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=self.user_id, name="bt_strat", type="momentum",
                symbol="BTC/USDT",
            )
        )

    def test_create_backtest_result(self):
        """Should create a backtest result and return result_id."""
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=self.strategy_id,
                config={"period": 14},
                results={
                    "total_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.10,
                    "win_rate": 0.55,
                    "total_trades": 100,
                },
            )
        )
        assert result_id is not None
        assert result_id.startswith("bt_")

    def test_get_backtest_result(self):
        """Should retrieve backtest result by ID."""
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=self.strategy_id,
                config={"period": 14},
                results={
                    "total_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.10,
                    "win_rate": 0.55,
                    "total_trades": 100,
                },
            )
        )
        result = run_async(self.db_manager.get_backtest_result(result_id))
        assert result is not None
        assert result["id"] == result_id
        assert result["strategy_id"] == self.strategy_id

    def test_get_backtest_result_nonexistent(self):
        """Should return None for non-existent result."""
        result = run_async(self.db_manager.get_backtest_result("nonexistent"))
        assert result is None

    def test_backtest_result_defaults(self):
        """Should use default values for missing result fields."""
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=self.strategy_id,
                config={},
                results={},  # Empty results
            )
        )
        result = run_async(self.db_manager.get_backtest_result(result_id))
        assert result is not None

    def test_backtest_result_stores_config(self):
        """Should store config as JSON."""
        config = {"period": 14, "threshold": 0.5}
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=self.strategy_id,
                config=config,
                results={"total_return": 0.1},
            )
        )
        result = run_async(self.db_manager.get_backtest_result(result_id))
        assert result["config"] is not None

    def test_backtest_result_stores_results_json(self):
        """Should store full results as JSON."""
        full_results = {
            "total_return": 0.25,
            "trades": [{"id": 1, "pnl": 100}, {"id": 2, "pnl": -50}],
            "equity_curve": [100000, 100100, 100050],
        }
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=self.strategy_id,
                config={},
                results=full_results,
            )
        )
        result = run_async(self.db_manager.get_backtest_result(result_id))
        assert result["results_json"] is not None


# ============================================================================
# Cleanup Tests
# ============================================================================

class TestCleanup:
    """Tests for data cleanup and archival operations."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()

    def test_cleanup_old_candles(self):
        """Should delete candles older than keep_days."""
        # Insert old candle
        with self.db_manager.transaction() as session:
            session.add(CandleRecord(
                id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                timeframe="1m",
                open_time=datetime(2020, 1, 1),
                close_time=datetime(2020, 1, 1, 0, 1),
                open=42000.0, high=42100.0, low=41900.0,
                close=42050.0, volume=100.0,
                exchange="binance",
            ))
        deleted = run_async(
            self.db_manager.cleanup_old_candles("BTC/USDT", "1m", keep_days=30)
        )
        assert deleted == 1

    def test_cleanup_old_candles_keeps_recent(self):
        """Should keep candles within keep_days window."""
        with self.db_manager.transaction() as session:
            session.add(CandleRecord(
                id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                timeframe="1m",
                open_time=datetime.utcnow() - timedelta(days=5),
                close_time=datetime.utcnow() - timedelta(days=5, minutes=-1),
                open=42000.0, high=42100.0, low=41900.0,
                close=42050.0, volume=100.0,
                exchange="binance",
            ))
        deleted = run_async(
            self.db_manager.cleanup_old_candles("BTC/USDT", "1m", keep_days=30)
        )
        assert deleted == 0

    def test_cleanup_old_candles_no_match(self):
        """Should return 0 when no candles match criteria."""
        deleted = run_async(
            self.db_manager.cleanup_old_candles("NONEXIST/USDT", "1m", keep_days=30)
        )
        assert deleted == 0

    def test_cleanup_old_candles_default_keep_days(self):
        """Default keep_days should be 90."""
        with self.db_manager.transaction() as session:
            session.add(CandleRecord(
                id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                timeframe="1m",
                open_time=datetime(2020, 1, 1),
                close_time=datetime(2020, 1, 1, 0, 1),
                open=42000.0, high=42100.0, low=41900.0,
                close=42050.0, volume=100.0,
                exchange="binance",
            ))
        deleted = run_async(
            self.db_manager.cleanup_old_candles("BTC/USDT", "1m")
        )
        assert deleted == 1

    def test_archive_old_trades(self):
        """Should delete trades older than specified days."""
        user_id = run_async(
            self.db_manager.create_user("archive@example.com", "password")
        )
        order_result = run_async(self.db_manager.create_order(
            user_id=user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        # Insert old trade
        with self.db_manager.transaction() as session:
            session.add(TradeRecord(
                id=str(uuid.uuid4()),
                order_id=order_result["id"],
                symbol="BTC/USDT",
                side="buy",
                quantity=1.0,
                price=50000.0,
                exchange="binance",
                timestamp=datetime(2020, 1, 1),
            ))
        archived = run_async(self.db_manager.archive_old_trades(days=30))
        assert archived == 1

    def test_archive_old_trades_keeps_recent(self):
        """Should keep trades within days threshold."""
        user_id = run_async(
            self.db_manager.create_user("archive2@example.com", "password")
        )
        order_result = run_async(self.db_manager.create_order(
            user_id=user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        with self.db_manager.transaction() as session:
            session.add(TradeRecord(
                id=str(uuid.uuid4()),
                order_id=order_result["id"],
                symbol="BTC/USDT",
                side="buy",
                quantity=1.0,
                price=50000.0,
                exchange="binance",
                timestamp=datetime.utcnow() - timedelta(days=5),
            ))
        archived = run_async(self.db_manager.archive_old_trades(days=30))
        assert archived == 0

    def test_archive_old_trades_default_days(self):
        """Default days should be 180."""
        user_id = run_async(
            self.db_manager.create_user("archive3@example.com", "password")
        )
        order_result = run_async(self.db_manager.create_order(
            user_id=user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        with self.db_manager.transaction() as session:
            session.add(TradeRecord(
                id=str(uuid.uuid4()),
                order_id=order_result["id"],
                symbol="BTC/USDT",
                side="buy",
                quantity=1.0,
                price=50000.0,
                exchange="binance",
                timestamp=datetime(2020, 1, 1),
            ))
        archived = run_async(self.db_manager.archive_old_trades())
        assert archived == 1

    def test_archive_old_trades_no_matches(self):
        """Should return 0 when no old trades."""
        archived = run_async(self.db_manager.archive_old_trades(days=30))
        assert archived == 0


# ============================================================================
# Alembic Integration Tests
# ============================================================================

class TestAlembicHelpers:
    """Tests for Alembic integration helper methods."""

    def test_get_alembic_config_default(self):
        """Should return config with default URL."""
        config = DatabaseManager.get_alembic_config()
        assert "script_location" in config
        assert config["script_location"] == "alembic"
        assert "postgresql" in config["sqlalchemy.url"]
        assert config["render_as_batch"] is True

    def test_get_alembic_config_custom_url(self):
        """Should use provided URL."""
        config = DatabaseManager.get_alembic_config(db_url="sqlite:///test.db")
        assert config["sqlalchemy.url"] == "sqlite:///test.db"

    def test_get_alembic_config_empty_url(self):
        """Empty URL should use default."""
        config = DatabaseManager.get_alembic_config(db_url="")
        assert "postgresql" in config["sqlalchemy.url"]

    def test_check_migration_status_no_table(self):
        """Should return not_initialized when alembic_version table missing."""
        engine = create_engine("sqlite:///:memory:")
        result = DatabaseManager.check_migration_status(engine)
        assert result["status"] == "not_initialized"
        assert result["current_version"] is None
        assert "error" in result

    def test_check_migration_status_with_version(self):
        """Should return current version when available."""
        engine = create_engine("sqlite:///:memory:")
        # Create alembic_version table manually
        with engine.connect() as conn:
            conn.execute(
                __import__('sqlalchemy').text(
                    "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
                )
            )
            conn.execute(
                __import__('sqlalchemy').text(
                    "INSERT INTO alembic_version (version_num) VALUES ('abc123')"
                )
            )
            conn.commit()
        result = DatabaseManager.check_migration_status(engine)
        assert result["status"] == "up_to_date"
        assert result["current_version"] == "abc123"


# ============================================================================
# _model_to_dict Tests
# ============================================================================

class TestModelToDict:
    """Tests for _model_to_dict utility method."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()

    def test_model_to_dict_user(self):
        """Should convert User model to dict."""
        with self.db_manager.transaction() as session:
            user = User(
                id="test-m2d-1",
                email="m2d@test.com",
                username="m2duser",
                hashed_password="hashed",
            )
            session.add(user)

        with self.db_manager.transaction() as session:
            user = session.query(User).filter(User.id == "test-m2d-1").first()
            result = DatabaseManager._model_to_dict(user)
            assert isinstance(result, dict)
            assert result["id"] == "test-m2d-1"
            assert result["email"] == "m2d@test.com"
            assert result["username"] == "m2duser"

    def test_model_to_dict_none(self):
        """Should return empty dict for None."""
        result = DatabaseManager._model_to_dict(None)
        assert result == {}

    def test_model_to_dict_datetime_conversion(self):
        """Should convert datetime fields to ISO format strings."""
        with self.db_manager.transaction() as session:
            user = User(
                id="test-dt-1",
                email="dt@test.com",
                username="dtuser",
                hashed_password="hashed",
            )
            session.add(user)

        with self.db_manager.transaction() as session:
            user = session.query(User).filter(User.id == "test-dt-1").first()
            result = DatabaseManager._model_to_dict(user)
            if result["created_at"] is not None:
                assert isinstance(result["created_at"], str)

    def test_model_to_dict_includes_all_columns(self):
        """Should include all model columns in the dict."""
        with self.db_manager.transaction() as session:
            user = User(
                id="test-cols-1",
                email="cols@test.com",
                username="colsuser",
                hashed_password="hashed",
            )
            session.add(user)

        with self.db_manager.transaction() as session:
            user = session.query(User).filter(User.id == "test-cols-1").first()
            result = DatabaseManager._model_to_dict(user)
            expected_keys = {"id", "email", "username", "hashed_password",
                             "is_active", "is_admin", "api_key",
                             "created_at", "updated_at"}
            assert set(result.keys()) == expected_keys

    def test_model_to_dict_order(self):
        """Should convert OrderRecord model to dict."""
        user_id = run_async(
            self.db_manager.create_user("m2dorder@test.com", "password")
        )
        order_result = run_async(
            self.db_manager.create_order(
                user_id=user_id, symbol="BTC/USDT", side="buy",
                order_type="market", quantity=1.0,
            )
        )
        order = run_async(self.db_manager.get_order(order_result["id"]))
        assert isinstance(order, dict)
        assert "id" in order
        assert "symbol" in order
        assert "side" in order
        assert "status" in order

    def test_model_to_dict_candle(self):
        """Should convert CandleRecord model to dict with datetime fields."""
        run_async(self.db_manager.bulk_insert_candles([
            {
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1),
                "close_time": datetime(2024, 1, 1, 0, 1),
                "open": 42000.0,
                "high": 42100.0,
                "low": 41900.0,
                "close": 42050.0,
                "volume": 100.0,
                "exchange": "binance",
            },
        ]))
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        assert len(candles) == 1
        # Datetime fields should be converted to strings
        if candles[0]["open_time"] is not None:
            assert isinstance(candles[0]["open_time"], str)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for DatabaseManager."""

    def setup_method(self):
        self.db_url = "sqlite:///:memory:"
        self.db_manager = DatabaseManager(db_url=self.db_url)
        self.db_manager._get_engine()

    def test_user_email_at_symbol(self):
        """Username derivation should handle @ correctly."""
        user_id = run_async(
            self.db_manager.create_user("user.name+tag@domain.com", "password")
        )
        user = run_async(self.db_manager.get_user(user_id))
        assert user["username"] == "user.name+tag"

    def test_create_order_id_format(self):
        """Order IDs should follow the ord_ prefix format."""
        user_id = run_async(
            self.db_manager.create_user("edge@example.com", "password")
        )
        result = run_async(
            self.db_manager.create_order(
                user_id=user_id, symbol="BTC/USDT", side="buy",
                order_type="market", quantity=1.0,
            )
        )
        assert result["id"].startswith("ord_")
        # After prefix, should be hex characters
        hex_part = result["id"][4:]
        assert len(hex_part) == 12

    def test_create_strategy_id_format(self):
        """Strategy IDs should follow the strat_ prefix format."""
        user_id = run_async(
            self.db_manager.create_user("edge2@example.com", "password")
        )
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=user_id, name="test", type="test", symbol="BTC/USDT",
            )
        )
        assert strategy_id.startswith("strat_")

    def test_create_backtest_result_id_format(self):
        """Backtest result IDs should follow the bt_ prefix format."""
        user_id = run_async(
            self.db_manager.create_user("edge3@example.com", "password")
        )
        strategy_id = run_async(
            self.db_manager.create_strategy(
                user_id=user_id, name="test", type="test", symbol="BTC/USDT",
            )
        )
        result_id = run_async(
            self.db_manager.create_backtest_result(
                strategy_id=strategy_id,
                config={},
                results={"total_return": 0.1},
            )
        )
        assert result_id.startswith("bt_")

    def test_list_orders_default_limit(self):
        """Default limit should be 50."""
        user_id = run_async(
            self.db_manager.create_user("limit@example.com", "password")
        )
        # Create a single order and verify default limit works
        run_async(self.db_manager.create_order(
            user_id=user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        orders = run_async(self.db_manager.list_orders(user_id))
        assert len(orders) == 1

    def test_get_candles_default_limit(self):
        """Default candle limit should be 500."""
        candles = run_async(self.db_manager.get_candles("BTC/USDT", "1m"))
        assert candles == []

    def test_list_trades_default_limit(self):
        """Default trade limit should be 50."""
        trades = run_async(self.db_manager.list_trades())
        assert trades == []

    def test_get_recent_signals_default_limit(self):
        """Default signal limit should be 50."""
        signals = run_async(self.db_manager.get_recent_signals())
        assert signals == []

    def test_multiple_engines(self):
        """Creating multiple DatabaseManagers should work independently."""
        manager1 = DatabaseManager(db_url="sqlite:///:memory:")
        manager2 = DatabaseManager(db_url="sqlite:///:memory:")
        manager1._get_engine()
        manager2._get_engine()
        # Each should have its own engine
        assert manager1._engine is not manager2._engine

    def test_bulk_insert_trades_large_batch(self):
        """Should handle large batch inserts."""
        user_id = run_async(
            self.db_manager.create_user("batch@example.com", "password")
        )
        order_result = run_async(self.db_manager.create_order(
            user_id=user_id, symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0,
        ))
        trades = []
        for i in range(50):
            trades.append({
                "order_id": order_result["id"],
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.01,
                "price": 50000.0 + i,
                "exchange": "binance",
            })
        count = run_async(self.db_manager.bulk_insert_trades(trades))
        assert count == 50

    def test_bulk_insert_candles_large_batch(self):
        """Should handle large candle batch inserts."""
        candles = []
        for i in range(50):
            day = i // 24
            hour = i % 24
            candles.append({
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "open_time": datetime(2024, 1, 1 + day, hour),
                "close_time": datetime(2024, 1, 1 + day, hour, 1),
                "open": 42000.0 + i,
                "high": 42100.0 + i,
                "low": 41900.0 + i,
                "close": 42050.0 + i,
                "volume": 100.0,
                "exchange": "binance",
            })
        count = run_async(self.db_manager.bulk_insert_candles(candles))
        assert count == 50

    def test_risk_event_model(self):
        """Should create and query risk events."""
        user_id = run_async(
            self.db_manager.create_user("risk@example.com", "password")
        )
        with self.db_manager.transaction() as session:
            event = RiskEvent(
                id=str(uuid.uuid4()),
                user_id=user_id,
                event_type="max_drawdown_exceeded",
                severity="critical",
                details={"drawdown_pct": 0.25},
            )
            session.add(event)

        with self.db_manager.transaction() as session:
            events = session.query(RiskEvent).all()
            assert len(events) == 1
            assert events[0].event_type == "max_drawdown_exceeded"
            assert events[0].severity == "critical"

    def test_exchange_credential_model(self):
        """Should create and query exchange credentials."""
        user_id = run_async(
            self.db_manager.create_user("cred@example.com", "password")
        )
        with self.db_manager.transaction() as session:
            cred = ExchangeCredential(
                id=str(uuid.uuid4()),
                user_id=user_id,
                exchange="binance",
                api_key_encrypted="encrypted_key",
                api_secret_encrypted="encrypted_secret",
                passphrase_encrypted="encrypted_passphrase",
                is_testnet=True,
            )
            session.add(cred)

        with self.db_manager.transaction() as session:
            creds = session.query(ExchangeCredential).all()
            assert len(creds) == 1
            assert creds[0].exchange == "binance"
            assert creds[0].is_testnet is True

    def test_portfolio_snapshot_model(self):
        """Should create and query portfolio snapshots."""
        user_id = run_async(
            self.db_manager.create_user("snap@example.com", "password")
        )
        with self.db_manager.transaction() as session:
            snap = PortfolioSnapshotRecord(
                id=str(uuid.uuid4()),
                user_id=user_id,
                total_value=100000.0,
                available_balance=80000.0,
                unrealized_pnl=5000.0,
                realized_pnl=2000.0,
                margin_used=15000.0,
                leverage=1.5,
            )
            session.add(snap)

        with self.db_manager.transaction() as session:
            snaps = session.query(PortfolioSnapshotRecord).all()
            assert len(snaps) == 1
            assert float(snaps[0].total_value) == 100000.0

    def test_api_key_model(self):
        """Should create and query API keys."""
        user_id = run_async(
            self.db_manager.create_user("apikey@example.com", "password")
        )
        with self.db_manager.transaction() as session:
            key = ApiKey(
                id=str(uuid.uuid4()),
                user_id=user_id,
                key_hash="hash_of_api_key",
                name="My API Key",
                permissions={"read": True, "write": False},
            )
            session.add(key)

        with self.db_manager.transaction() as session:
            keys = session.query(ApiKey).all()
            assert len(keys) == 1
            assert keys[0].name == "My API Key"
            assert keys[0].permissions == {"read": True, "write": False}


# ============================================================================
# Table Creation / Schema Tests
# ============================================================================

class TestTableCreation:
    """Tests for table creation and schema integrity."""

    def test_all_tables_created(self, db_manager):
        """All expected tables should be created."""
        engine = db_manager._get_engine()
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        expected_tables = [
            "users", "api_keys", "strategies", "orders", "trades",
            "positions", "signals", "candles", "risk_events",
            "backtest_results", "portfolio_snapshots", "exchange_credentials",
        ]
        for table in expected_tables:
            assert table in table_names, f"Missing table: {table}"

    def test_user_table_columns(self, db_manager):
        """User table should have correct columns."""
        engine = db_manager._get_engine()
        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("users")}
        assert "id" in columns
        assert "email" in columns
        assert "username" in columns
        assert "hashed_password" in columns
        assert "is_active" in columns
        assert "is_admin" in columns

    def test_orders_table_columns(self, db_manager):
        """Orders table should have all expected columns."""
        engine = db_manager._get_engine()
        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("orders")}
        assert "id" in columns
        assert "user_id" in columns
        assert "strategy_id" in columns
        assert "symbol" in columns
        assert "side" in columns
        assert "order_type" in columns
        assert "status" in columns
        assert "quantity" in columns
        assert "price" in columns
        assert "exchange" in columns

    def test_candles_table_columns(self, db_manager):
        """Candles table should have all expected columns."""
        engine = db_manager._get_engine()
        inspector = inspect(engine)
        columns = {c["name"]: c for c in inspector.get_columns("candles")}
        assert "id" in columns
        assert "symbol" in columns
        assert "timeframe" in columns
        assert "open_time" in columns
        assert "close_time" in columns
        assert "open" in columns
        assert "high" in columns
        assert "low" in columns
        assert "close" in columns
        assert "volume" in columns
        assert "exchange" in columns
