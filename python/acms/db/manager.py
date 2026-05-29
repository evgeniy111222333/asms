"""Database manager - CRUD operations for all ACMS models."""

import uuid
import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from sqlalchemy import create_engine, and_, desc, asc, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import (
    Base,
    ALLOWED_ORDER_FIELDS,
    ALLOWED_STRATEGY_FIELDS,
    User,
    ApiKey,
    Strategy,
    OrderRecord,
    TradeRecord,
    TradeArchiveRecord,
    PositionRecord,
    SignalRecord,
    CandleRecord,
    RiskEvent,
    BacktestResult,
    PortfolioSnapshotRecord,
    ExchangeCredential,
)
from .encryption import CredentialEncryptor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """High-level database operations manager.

    Provides CRUD operations for all models, transaction management,
    bulk inserts, and query helpers.
    """

    def __init__(self, db_url: str = "postgresql://acms:acms@localhost:5432/acms",
                 encryption_key: str = None):
        self.db_url = db_url
        self._engine = None
        self._session_factory = None
        self._credential_encryptor = CredentialEncryptor(key=encryption_key)

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

    def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous DB operation in a thread pool.

        This helper allows async methods to properly offload blocking
        synchronous SQLAlchemy operations to a thread pool, preventing
        event loop blocking.

        Args:
            func: Synchronous function to run.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.
        """
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: func(*args, **kwargs))

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
        """Create a new user with properly hashed password.

        Args:
            email: User email address.
            password: Plain text password (will be hashed with bcrypt).
            username: Optional username.
            is_admin: Whether user has admin privileges.

        Returns:
            User ID string.
        """
        from acms.auth import AuthManager
        auth = AuthManager()

        user_id = str(uuid.uuid4())
        if not username:
            username = email.split("@")[0]
        hashed = auth.hash_password(password)

        def _create():
            with self.transaction() as session:
                user = User(
                    id=user_id, email=email, username=username,
                    hashed_password=hashed, is_admin=is_admin,
                )
                session.add(user)
            return user_id

        return await self._run_sync(_create)

    async def get_user(self, user_id: str = None, email: str = None) -> Optional[Dict]:
        """Get user by ID or email.

        Args:
            user_id: User ID to look up.
            email: Email address to look up.

        Returns:
            User dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                query = session.query(User)
                if user_id:
                    user = query.filter(User.id == user_id).first()
                elif email:
                    user = query.filter(User.email == email).first()
                else:
                    return None
                if user:
                    return {"id": user.id, "email": user.email, "username": user.username,
                            "is_active": user.is_active, "is_admin": user.is_admin,
                            "hashed_password": user.hashed_password}
            return None

        return await self._run_sync(_get)

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email address, including hashed password for auth.

        Args:
            email: Email address to look up.

        Returns:
            User dict if found, None otherwise.
        """
        return await self.get_user(email=email)

    async def update_user(self, user_id: str, updates: Dict) -> bool:
        """Update user fields.

        Args:
            user_id: User ID to update.
            updates: Dict of field names to new values.

        Returns:
            True if user was found and updated.
        """
        allowed_fields = {"username", "email", "is_active", "is_admin", "updated_at"}

        def _update():
            with self.transaction() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user:
                    for key, value in updates.items():
                        if key in allowed_fields and hasattr(user, key):
                            setattr(user, key, value)
                    return True
            return False

        return await self._run_sync(_update)

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user record.

        Args:
            user_id: User ID to delete.

        Returns:
            True if user was found and deleted.
        """
        def _delete():
            with self.transaction() as session:
                deleted = session.query(User).filter(User.id == user_id).delete()
                return deleted > 0

        return await self._run_sync(_delete)

    # ========================================================================
    # ApiKey CRUD
    # ========================================================================

    async def create_api_key(self, user_id: str, name: str, key_hash: str,
                              permissions: Dict = None) -> str:
        """Create a new API key record.

        Args:
            user_id: Owner user ID.
            name: Descriptive name for the key.
            key_hash: SHA-256 hash of the raw key.
            permissions: Optional permissions dict.

        Returns:
            API key ID string.
        """
        key_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                api_key = ApiKey(
                    id=key_id, user_id=user_id, key_hash=key_hash,
                    name=name, permissions=permissions or {},
                )
                session.add(api_key)
            return key_id

        return await self._run_sync(_create)

    async def get_api_key(self, key_id: str) -> Optional[Dict]:
        """Get API key by ID.

        Args:
            key_id: API key ID.

        Returns:
            API key dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
                return self._model_to_dict(api_key) if api_key else None

        return await self._run_sync(_get)

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict]:
        """Get API key by its hash for verification.

        Args:
            key_hash: SHA-256 hash of the raw API key.

        Returns:
            API key dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                api_key = session.query(ApiKey).filter(
                    and_(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
                ).first()
                return self._model_to_dict(api_key) if api_key else None

        return await self._run_sync(_get)

    async def list_api_keys(self, user_id: str) -> List[Dict]:
        """List all API keys for a user.

        Args:
            user_id: Owner user ID.

        Returns:
            List of API key dicts.
        """
        def _list():
            with self.transaction() as session:
                keys = session.query(ApiKey).filter(ApiKey.user_id == user_id).all()
                return [self._model_to_dict(k) for k in keys]

        return await self._run_sync(_list)

    async def deactivate_api_key(self, key_id: str) -> bool:
        """Deactivate an API key.

        Args:
            key_id: API key ID to deactivate.

        Returns:
            True if key was found and deactivated.
        """
        def _deactivate():
            with self.transaction() as session:
                api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
                if api_key:
                    api_key.is_active = False
                    return True
            return False

        return await self._run_sync(_deactivate)

    async def update_api_key_last_used(self, key_id: str) -> None:
        """Update the last_used_at timestamp for an API key.

        Args:
            key_id: API key ID.
        """
        def _update():
            with self.transaction() as session:
                api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
                if api_key:
                    api_key.last_used_at = datetime.utcnow()

        await self._run_sync(_update)

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

        def _create():
            with self.transaction() as session:
                order = OrderRecord(
                    id=order_id, user_id=user_id, strategy_id=strategy_id,
                    symbol=symbol, side=side, order_type=order_type,
                    status="created", quantity=quantity, price=price,
                    stop_price=stop_price, exchange=exchange,
                )
                session.add(order)
            return {"id": order_id, "symbol": symbol, "status": "created"}

        return await self._run_sync(_create)

    async def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        def _get():
            with self.transaction() as session:
                order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
                if order:
                    return self._model_to_dict(order)
            return None

        return await self._run_sync(_get)

    async def update_order(self, order_id: str, updates: Dict) -> bool:
        """Update order fields with whitelist protection.

        Only fields in ALLOWED_ORDER_FIELDS can be updated to prevent
        unauthorized modification of critical order data (e.g., user_id, symbol).

        Args:
            order_id: Order ID to update.
            updates: Dict of field names to new values.

        Returns:
            True if order was found and updated.
        """
        filtered_updates = {k: v for k, v in updates.items() if k in ALLOWED_ORDER_FIELDS}
        if not filtered_updates:
            return False

        def _update():
            with self.transaction() as session:
                order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
                if order:
                    for key, value in filtered_updates.items():
                        setattr(order, key, value)
                    return True
            return False

        return await self._run_sync(_update)

    async def list_orders(self, user_id: str, symbol: Optional[str] = None,
                          status: Optional[str] = None, exchange: Optional[str] = None,
                          strategy_id: Optional[str] = None,
                          limit: int = 50, offset: int = 0,
                          sort_by: str = "created_at", sort_order: str = "desc") -> List[Dict]:
        """List orders with filtering, sorting, and pagination."""
        def _list():
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

        return await self._run_sync(_list)

    async def delete_order(self, order_id: str) -> bool:
        """Delete an order record."""
        def _delete():
            with self.transaction() as session:
                deleted = session.query(OrderRecord).filter(OrderRecord.id == order_id).delete()
                return deleted > 0

        return await self._run_sync(_delete)

    # ========================================================================
    # Strategy CRUD
    # ========================================================================

    async def create_strategy(self, user_id: str, name: str, type: str,
                               symbol: str, config: Dict = None) -> str:
        """Create a new strategy record."""
        strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

        def _create():
            with self.transaction() as session:
                strategy = Strategy(
                    id=strategy_id, user_id=user_id, name=name,
                    type=type, symbol=symbol, config=config or {},
                )
                session.add(strategy)
            return strategy_id

        return await self._run_sync(_create)

    async def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get strategy by ID.

        Args:
            strategy_id: Strategy ID.

        Returns:
            Strategy dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
                return self._model_to_dict(strategy) if strategy else None

        return await self._run_sync(_get)

    async def list_strategies(self, user_id: str) -> List[Dict]:
        """List all strategies for a user."""
        def _list():
            with self.transaction() as session:
                strategies = session.query(Strategy).filter(Strategy.user_id == user_id).all()
                return [self._model_to_dict(s) for s in strategies]

        return await self._run_sync(_list)

    async def update_strategy(self, strategy_id: str, updates: Dict) -> bool:
        """Update strategy fields with whitelist protection.

        Only fields in ALLOWED_STRATEGY_FIELDS can be updated to prevent
        unauthorized modification of critical strategy data (e.g., user_id).

        Args:
            strategy_id: Strategy ID to update.
            updates: Dict of field names to new values.

        Returns:
            True if strategy was found and updated.
        """
        filtered_updates = {k: v for k, v in updates.items() if k in ALLOWED_STRATEGY_FIELDS}
        if not filtered_updates:
            return False

        def _update():
            with self.transaction() as session:
                strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
                if strategy:
                    for key, value in filtered_updates.items():
                        setattr(strategy, key, value)
                    return True
            return False

        return await self._run_sync(_update)

    async def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy record.

        Args:
            strategy_id: Strategy ID to delete.

        Returns:
            True if strategy was found and deleted.
        """
        def _delete():
            with self.transaction() as session:
                deleted = session.query(Strategy).filter(Strategy.id == strategy_id).delete()
                return deleted > 0

        return await self._run_sync(_delete)

    # ========================================================================
    # Trade CRUD
    # ========================================================================

    async def create_trade(self, order_id: str, symbol: str, side: str,
                           quantity: float, price: float, exchange: str = "paper",
                           commission: float = 0, commission_asset: str = "",
                           exchange_trade_id: str = None, is_maker: bool = False,
                           slippage: float = 0) -> str:
        """Create a single trade record.

        Args:
            order_id: Associated order ID.
            symbol: Trading pair symbol.
            side: Trade side (buy/sell).
            quantity: Trade quantity.
            price: Trade price.
            exchange: Exchange name.
            commission: Trade commission.
            commission_asset: Commission currency.
            exchange_trade_id: Exchange-specific trade ID.
            is_maker: Whether this was a maker trade.
            slippage: Execution slippage.

        Returns:
            Trade ID string.
        """
        trade_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                trade = TradeRecord(
                    id=trade_id, order_id=order_id, symbol=symbol, side=side,
                    quantity=quantity, price=price, commission=commission,
                    commission_asset=commission_asset, exchange=exchange,
                    exchange_trade_id=exchange_trade_id, is_maker=is_maker,
                    slippage=slippage,
                )
                session.add(trade)
            return trade_id

        return await self._run_sync(_create)

    async def list_trades(self, symbol: Optional[str] = None, side: Optional[str] = None,
                          exchange: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> List[Dict]:
        """List trades with filtering and pagination."""
        def _list():
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

        return await self._run_sync(_list)

    async def bulk_insert_trades(self, trades: List[Dict]) -> int:
        """Bulk insert trade records for high-frequency data.

        Args:
            trades: List of trade data dicts.

        Returns:
            Number of records inserted.
        """
        if not trades:
            return 0

        def _bulk():
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

        return await self._run_sync(_bulk)

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
        def _get():
            with self.transaction() as session:
                candles = session.query(CandleRecord).filter(
                    and_(CandleRecord.symbol == symbol, CandleRecord.timeframe == timeframe)
                ).order_by(desc(CandleRecord.open_time)).limit(limit).all()
                return [self._model_to_dict(c) for c in reversed(candles)]

        return await self._run_sync(_get)

    async def bulk_insert_candles(self, candles: List[Dict]) -> int:
        """Bulk insert candle records.

        Args:
            candles: List of candle data dicts.

        Returns:
            Number of records inserted.
        """
        if not candles:
            return 0

        def _bulk():
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

        return await self._run_sync(_bulk)

    # ========================================================================
    # Position CRUD
    # ========================================================================

    async def create_position(self, user_id: str, symbol: str, side: str,
                               quantity: float, entry_price: float, exchange: str = "paper",
                               strategy_id: Optional[str] = None,
                               leverage: float = 1) -> str:
        """Create a new position record.

        Args:
            user_id: Owner user ID.
            symbol: Trading pair symbol.
            side: Position side (long/short).
            quantity: Position quantity.
            entry_price: Entry price.
            exchange: Exchange name.
            strategy_id: Associated strategy ID.
            leverage: Position leverage.

        Returns:
            Position ID string.
        """
        position_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                position = PositionRecord(
                    id=position_id, user_id=user_id, strategy_id=strategy_id,
                    symbol=symbol, side=side, quantity=quantity,
                    entry_price=entry_price, exchange=exchange, leverage=leverage,
                )
                session.add(position)
            return position_id

        return await self._run_sync(_create)

    async def update_position(self, position_id: str, updates: Dict) -> bool:
        """Update position fields.

        Args:
            position_id: Position ID to update.
            updates: Dict of field names to new values.

        Returns:
            True if position was found and updated.
        """
        allowed_position_fields = {
            "quantity", "mark_price", "unrealized_pnl", "realized_pnl",
            "leverage", "closed_at", "updated_at",
        }
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_position_fields}
        if not filtered_updates:
            return False

        def _update():
            with self.transaction() as session:
                position = session.query(PositionRecord).filter(
                    PositionRecord.id == position_id
                ).first()
                if position:
                    for key, value in filtered_updates.items():
                        setattr(position, key, value)
                    return True
            return False

        return await self._run_sync(_update)

    async def close_position(self, position_id: str, realized_pnl: float = 0) -> bool:
        """Close a position by setting closed_at timestamp.

        Args:
            position_id: Position ID to close.
            realized_pnl: Final realized P&L.

        Returns:
            True if position was found and closed.
        """
        def _close():
            with self.transaction() as session:
                position = session.query(PositionRecord).filter(
                    PositionRecord.id == position_id
                ).first()
                if position:
                    position.closed_at = datetime.utcnow()
                    position.realized_pnl = realized_pnl
                    return True
            return False

        return await self._run_sync(_close)

    # ========================================================================
    # Signal CRUD
    # ========================================================================

    async def create_signal(self, strategy_id: str, symbol: str, direction: str,
                             strength: float = 0, indicators: Dict = None,
                             metadata: Dict = None) -> str:
        """Create a new signal record.

        Args:
            strategy_id: Associated strategy ID.
            symbol: Trading pair symbol.
            direction: Signal direction (buy/sell/neutral).
            strength: Signal strength (0-1).
            indicators: Indicator values at signal time.
            metadata: Additional metadata.

        Returns:
            Signal ID string.
        """
        signal_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                signal = SignalRecord(
                    id=signal_id, strategy_id=strategy_id, symbol=symbol,
                    direction=direction, strength=strength,
                    indicators=indicators or {}, signal_metadata=metadata or {},
                )
                session.add(signal)
            return signal_id

        return await self._run_sync(_create)

    async def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get signal by ID.

        Args:
            signal_id: Signal ID.

        Returns:
            Signal dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                signal = session.query(SignalRecord).filter(SignalRecord.id == signal_id).first()
                return self._model_to_dict(signal) if signal else None

        return await self._run_sync(_get)

    # ========================================================================
    # RiskEvent CRUD
    # ========================================================================

    async def create_risk_event(self, user_id: str, event_type: str, severity: str,
                                 details: Dict = None) -> str:
        """Create a new risk event record.

        Args:
            user_id: User ID that triggered the event.
            event_type: Type of risk event (e.g., 'max_drawdown', 'position_limit').
            severity: Severity level ('info', 'warning', 'critical').
            details: Additional event details.

        Returns:
            Risk event ID string.
        """
        event_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                event = RiskEvent(
                    id=event_id, user_id=user_id, event_type=event_type,
                    severity=severity, details=details or {},
                )
                session.add(event)
            return event_id

        return await self._run_sync(_create)

    async def list_risk_events(self, user_id: str, severity: Optional[str] = None,
                                event_type: Optional[str] = None,
                                limit: int = 50, offset: int = 0) -> List[Dict]:
        """List risk events for a user with optional filtering.

        Args:
            user_id: User ID.
            severity: Filter by severity level.
            event_type: Filter by event type.
            limit: Maximum number of records.
            offset: Pagination offset.

        Returns:
            List of risk event dicts.
        """
        def _list():
            with self.transaction() as session:
                query = session.query(RiskEvent).filter(RiskEvent.user_id == user_id)
                if severity:
                    query = query.filter(RiskEvent.severity == severity)
                if event_type:
                    query = query.filter(RiskEvent.event_type == event_type)
                events = query.order_by(desc(RiskEvent.timestamp)).offset(offset).limit(limit).all()
                return [self._model_to_dict(e) for e in events]

        return await self._run_sync(_list)

    async def get_risk_event(self, event_id: str) -> Optional[Dict]:
        """Get a risk event by ID.

        Args:
            event_id: Risk event ID.

        Returns:
            Risk event dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                event = session.query(RiskEvent).filter(RiskEvent.id == event_id).first()
                return self._model_to_dict(event) if event else None

        return await self._run_sync(_get)

    # ========================================================================
    # ExchangeCredential CRUD
    # ========================================================================

    async def create_exchange_credential(self, user_id: str, exchange: str,
                                          api_key: str, api_secret: str,
                                          passphrase: Optional[str] = None,
                                          is_testnet: bool = False) -> str:
        """Create a new encrypted exchange credential record.

        Args:
            user_id: Owner user ID.
            exchange: Exchange name (e.g., 'binance', 'okx').
            api_key: Plain text API key (will be encrypted).
            api_secret: Plain text API secret (will be encrypted).
            passphrase: Optional passphrase (will be encrypted).
            is_testnet: Whether this is a testnet credential.

        Returns:
            Credential ID string.
        """
        cred_id = str(uuid.uuid4())
        encryptor = self._credential_encryptor

        api_key_encrypted = encryptor.encrypt(api_key)
        api_secret_encrypted = encryptor.encrypt(api_secret)
        passphrase_encrypted = encryptor.encrypt(passphrase) if passphrase else None

        def _create():
            with self.transaction() as session:
                cred = ExchangeCredential(
                    id=cred_id, user_id=user_id, exchange=exchange,
                    api_key_encrypted=api_key_encrypted,
                    api_secret_encrypted=api_secret_encrypted,
                    passphrase_encrypted=passphrase_encrypted,
                    is_testnet=is_testnet,
                )
                session.add(cred)
            return cred_id

        return await self._run_sync(_create)

    async def get_exchange_credential(self, cred_id: str, decrypt: bool = False) -> Optional[Dict]:
        """Get exchange credential by ID.

        Args:
            cred_id: Credential ID.
            decrypt: Whether to decrypt the credential values.

        Returns:
            Credential dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                cred = session.query(ExchangeCredential).filter(
                    ExchangeCredential.id == cred_id
                ).first()
                if cred:
                    result = self._model_to_dict(cred)
                    if decrypt:
                        encryptor = self._credential_encryptor
                        try:
                            result["api_key"] = encryptor.decrypt(cred.api_key_encrypted)
                            result["api_secret"] = encryptor.decrypt(cred.api_secret_encrypted)
                            if cred.passphrase_encrypted:
                                result["passphrase"] = encryptor.decrypt(cred.passphrase_encrypted)
                        except Exception:
                            logger.error("Failed to decrypt exchange credentials for id=%s", cred_id)
                            result["api_key"] = None
                            result["api_secret"] = None
                            result["passphrase"] = None
                    return result
            return None

        return await self._run_sync(_get)

    async def list_exchange_credentials(self, user_id: str, exchange: Optional[str] = None,
                                          active_only: bool = True) -> List[Dict]:
        """List exchange credentials for a user.

        Args:
            user_id: Owner user ID.
            exchange: Filter by exchange name.
            active_only: Only return active credentials.

        Returns:
            List of credential dicts (without decrypted values).
        """
        def _list():
            with self.transaction() as session:
                query = session.query(ExchangeCredential).filter(
                    ExchangeCredential.user_id == user_id
                )
                if exchange:
                    query = query.filter(ExchangeCredential.exchange == exchange)
                if active_only:
                    query = query.filter(ExchangeCredential.is_active == True)
                creds = query.all()
                return [self._model_to_dict(c) for c in creds]

        return await self._run_sync(_list)

    async def deactivate_exchange_credential(self, cred_id: str) -> bool:
        """Deactivate an exchange credential.

        Args:
            cred_id: Credential ID to deactivate.

        Returns:
            True if credential was found and deactivated.
        """
        def _deactivate():
            with self.transaction() as session:
                cred = session.query(ExchangeCredential).filter(
                    ExchangeCredential.id == cred_id
                ).first()
                if cred:
                    cred.is_active = False
                    return True
            return False

        return await self._run_sync(_deactivate)

    # ========================================================================
    # PortfolioSnapshot CRUD
    # ========================================================================

    async def create_portfolio_snapshot(self, user_id: str, total_value: float,
                                         available_balance: float = 0,
                                         unrealized_pnl: float = 0,
                                         realized_pnl: float = 0,
                                         margin_used: float = 0,
                                         leverage: float = 1,
                                         positions: List = None) -> str:
        """Create a new portfolio snapshot record.

        Args:
            user_id: Owner user ID.
            total_value: Total portfolio value.
            available_balance: Available balance.
            unrealized_pnl: Unrealized P&L.
            realized_pnl: Realized P&L.
            margin_used: Margin used.
            leverage: Current leverage.
            positions: List of position data.

        Returns:
            Snapshot ID string.
        """
        snapshot_id = str(uuid.uuid4())

        def _create():
            with self.transaction() as session:
                snapshot = PortfolioSnapshotRecord(
                    id=snapshot_id, user_id=user_id, total_value=total_value,
                    available_balance=available_balance, unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl, margin_used=margin_used,
                    leverage=leverage, positions_json=positions or [],
                )
                session.add(snapshot)
            return snapshot_id

        return await self._run_sync(_create)

    async def get_portfolio_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Get a portfolio snapshot by ID.

        Args:
            snapshot_id: Snapshot ID.

        Returns:
            Snapshot dict if found, None otherwise.
        """
        def _get():
            with self.transaction() as session:
                snapshot = session.query(PortfolioSnapshotRecord).filter(
                    PortfolioSnapshotRecord.id == snapshot_id
                ).first()
                return self._model_to_dict(snapshot) if snapshot else None

        return await self._run_sync(_get)

    # ========================================================================
    # Query Helpers
    # ========================================================================

    async def get_active_orders(self, user_id: str) -> List[Dict]:
        """Get all active (non-terminal) orders for a user."""
        def _get():
            with self.transaction() as session:
                orders = session.query(OrderRecord).filter(
                    and_(
                        OrderRecord.user_id == user_id,
                        OrderRecord.status.in_(["created", "submitted", "partially_filled"]),
                    )
                ).all()
                return [self._model_to_dict(o) for o in orders]

        return await self._run_sync(_get)

    async def get_open_positions(self, user_id: str) -> List[Dict]:
        """Get all open positions for a user."""
        def _get():
            with self.transaction() as session:
                positions = session.query(PositionRecord).filter(
                    and_(PositionRecord.user_id == user_id, PositionRecord.closed_at.is_(None))
                ).all()
                return [self._model_to_dict(p) for p in positions]

        return await self._run_sync(_get)

    async def get_recent_signals(self, symbol: Optional[str] = None,
                                  strategy_id: Optional[str] = None,
                                  direction: Optional[str] = None,
                                  limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent signals with optional filtering."""
        def _get():
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

        return await self._run_sync(_get)

    async def get_pnl_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get P&L history for a user.

        Args:
            user_id: User ID.
            days: Number of days of history.

        Returns:
            List of portfolio snapshot dicts.
        """
        since = datetime.utcnow() - timedelta(days=days)

        def _get():
            with self.transaction() as session:
                snapshots = session.query(PortfolioSnapshotRecord).filter(
                    and_(PortfolioSnapshotRecord.user_id == user_id,
                         PortfolioSnapshotRecord.timestamp >= since)
                ).order_by(asc(PortfolioSnapshotRecord.timestamp)).all()
                return [self._model_to_dict(s) for s in snapshots]

        return await self._run_sync(_get)

    async def get_latest_portfolio_snapshot(self, user_id: str) -> Optional[Dict]:
        """Get the most recent portfolio snapshot."""
        def _get():
            with self.transaction() as session:
                snapshot = session.query(PortfolioSnapshotRecord).filter(
                    PortfolioSnapshotRecord.user_id == user_id
                ).order_by(desc(PortfolioSnapshotRecord.timestamp)).first()
                return self._model_to_dict(snapshot) if snapshot else None

        return await self._run_sync(_get)

    # ========================================================================
    # Backtest Results
    # ========================================================================

    async def create_backtest_result(self, strategy_id: str, config: Dict,
                                      results: Dict) -> str:
        """Create a backtest result record."""
        result_id = f"bt_{uuid.uuid4().hex[:12]}"

        def _create():
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

        return await self._run_sync(_create)

    async def get_backtest_result(self, backtest_id: str) -> Optional[Dict]:
        """Get backtest result by ID."""
        def _get():
            with self.transaction() as session:
                bt = session.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
                return self._model_to_dict(bt) if bt else None

        return await self._run_sync(_get)

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

        def _cleanup():
            with self.transaction() as session:
                deleted = session.query(CandleRecord).filter(
                    and_(CandleRecord.symbol == symbol,
                         CandleRecord.timeframe == timeframe,
                         CandleRecord.open_time < cutoff)
                ).delete()
            return deleted

        return await self._run_sync(_cleanup)

    async def archive_old_trades(self, days: int = 180) -> int:
        """Move old trades to archive table instead of deleting them.

        Trades older than the specified number of days are first copied
        to the trades_archive table, then deleted from the active trades
        table. This preserves historical data for auditing and analytics.

        Args:
            days: Number of days threshold.

        Returns:
            Number of records archived.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        def _archive():
            with self.transaction() as session:
                # First insert into archive table
                session.execute(
                    text("""
                        INSERT INTO trades_archive (id, order_id, symbol, side, quantity, price,
                            commission, commission_asset, exchange, exchange_trade_id,
                            is_maker, slippage, timestamp, archived_at)
                        SELECT id, order_id, symbol, side, quantity, price,
                            commission, commission_asset, exchange, exchange_trade_id,
                            is_maker, slippage, timestamp, :now
                        FROM trades WHERE timestamp < :cutoff
                    """),
                    {"cutoff": cutoff, "now": datetime.utcnow()}
                )
                # Then delete from original
                result = session.execute(
                    text("DELETE FROM trades WHERE timestamp < :cutoff"),
                    {"cutoff": cutoff}
                )
                return result.rowcount

        return await self._run_sync(_archive)

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


__all__ = [
    "DatabaseManager",
]
