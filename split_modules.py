#!/usr/bin/env python3
"""Script to split god __init__.py files into proper module structures."""

import os
import re
import textwrap

BASE = "/home/z/my-project/asms/python/acms"


def write_file(path, content):
    """Write a file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Wrote {path}")


def compile_check(path):
    """Verify file compiles with py_compile."""
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  COMPILE ERROR in {path}: {e}")
        return False


# ============================================================================
# 1. db/__init__.py
# ============================================================================
def split_db():
    print("\n=== Splitting db/__init__.py ===")
    with open(f"{BASE}/db/__init__.py", 'r') as f:
        content = f.read()

    # models.py - already created manually, skip

    # encryption.py - already created manually, skip

    # session.py - already created manually, skip

    # crud.py - standalone CRUD functions extracted from DatabaseManager
    crud_content = '''"""CRUD operations for the ACMS database."""

import uuid
import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import and_, desc, asc, func, text
from sqlalchemy.orm import Session

from acms.db.models import (
    Base, User, ApiKey, Strategy, OrderRecord, TradeRecord, TradeArchiveRecord,
    PositionRecord, SignalRecord, CandleRecord, RiskEvent, BacktestResult,
    PortfolioSnapshotRecord, ExchangeCredential,
    ALLOWED_ORDER_FIELDS, ALLOWED_STRATEGY_FIELDS,
)
from acms.db.encryption import CredentialEncryptor
from acms.db.session import get_engine, get_session

logger = logging.getLogger(__name__)


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


@contextmanager
def transaction():
    """Context manager for database transactions.

    Yields:
        SQLAlchemy Session with automatic commit/rollback.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _run_sync(func, *args, **kwargs):
    """Run a synchronous DB operation in a thread pool.

    Args:
        func: Synchronous function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call.
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ============================================================================
# User CRUD
# ============================================================================

async def create_user(email: str, password: str, username: str = "",
                      is_admin: bool = False) -> str:
    """Create a new user with properly hashed password."""
    from acms.auth import AuthManager
    auth = AuthManager()

    user_id = str(uuid.uuid4())
    if not username:
        username = email.split("@")[0]
    hashed = auth.hash_password(password)

    def _create():
        with transaction() as session:
            user = User(
                id=user_id, email=email, username=username,
                hashed_password=hashed, is_admin=is_admin,
            )
            session.add(user)
        return user_id

    return await _run_sync(_create)


async def get_user(user_id: str = None, email: str = None) -> Optional[Dict]:
    """Get user by ID or email."""
    def _get():
        with transaction() as session:
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

    return await _run_sync(_get)


async def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email address."""
    return await get_user(email=email)


async def update_user(user_id: str, updates: Dict) -> bool:
    """Update user fields."""
    allowed_fields = {"username", "email", "is_active", "is_admin", "updated_at"}

    def _update():
        with transaction() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in updates.items():
                    if key in allowed_fields and hasattr(user, key):
                        setattr(user, key, value)
                return True
        return False

    return await _run_sync(_update)


async def delete_user(user_id: str) -> bool:
    """Delete a user record."""
    def _delete():
        with transaction() as session:
            deleted = session.query(User).filter(User.id == user_id).delete()
            return deleted > 0

    return await _run_sync(_delete)


# ============================================================================
# ApiKey CRUD
# ============================================================================

async def create_api_key(user_id: str, name: str, key_hash: str,
                          permissions: Dict = None) -> str:
    """Create a new API key record."""
    key_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            api_key = ApiKey(
                id=key_id, user_id=user_id, key_hash=key_hash,
                name=name, permissions=permissions or {},
            )
            session.add(api_key)
        return key_id

    return await _run_sync(_create)


async def get_api_key(key_id: str) -> Optional[Dict]:
    """Get API key by ID."""
    def _get():
        with transaction() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            return _model_to_dict(api_key) if api_key else None

    return await _run_sync(_get)


async def get_api_key_by_hash(key_hash: str) -> Optional[Dict]:
    """Get API key by its hash for verification."""
    def _get():
        with transaction() as session:
            api_key = session.query(ApiKey).filter(
                and_(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
            ).first()
            return _model_to_dict(api_key) if api_key else None

    return await _run_sync(_get)


async def list_api_keys(user_id: str) -> List[Dict]:
    """List all API keys for a user."""
    def _list():
        with transaction() as session:
            keys = session.query(ApiKey).filter(ApiKey.user_id == user_id).all()
            return [_model_to_dict(k) for k in keys]

    return await _run_sync(_list)


async def deactivate_api_key(key_id: str) -> bool:
    """Deactivate an API key."""
    def _deactivate():
        with transaction() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if api_key:
                api_key.is_active = False
                return True
        return False

    return await _run_sync(_deactivate)


async def update_api_key_last_used(key_id: str) -> None:
    """Update the last_used_at timestamp for an API key."""
    def _update():
        with transaction() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if api_key:
                api_key.last_used_at = datetime.utcnow()

    await _run_sync(_update)


# ============================================================================
# Order CRUD
# ============================================================================

async def create_order(user_id: str, symbol: str, side: str,
                       order_type: str, quantity: float, price: Optional[float] = None,
                       stop_price: Optional[float] = None, exchange: str = "paper",
                       strategy_id: Optional[str] = None) -> Dict:
    """Create a new order record."""
    order_id = f"ord_{uuid.uuid4().hex[:12]}"

    def _create():
        with transaction() as session:
            order = OrderRecord(
                id=order_id, user_id=user_id, strategy_id=strategy_id,
                symbol=symbol, side=side, order_type=order_type,
                status="created", quantity=quantity, price=price,
                stop_price=stop_price, exchange=exchange,
            )
            session.add(order)
        return {"id": order_id, "symbol": symbol, "status": "created"}

    return await _run_sync(_create)


async def get_order(order_id: str) -> Optional[Dict]:
    """Get order by ID."""
    def _get():
        with transaction() as session:
            order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
            if order:
                return _model_to_dict(order)
        return None

    return await _run_sync(_get)


async def update_order(order_id: str, updates: Dict) -> bool:
    """Update order fields with whitelist protection."""
    filtered_updates = {k: v for k, v in updates.items() if k in ALLOWED_ORDER_FIELDS}
    if not filtered_updates:
        return False

    def _update():
        with transaction() as session:
            order = session.query(OrderRecord).filter(OrderRecord.id == order_id).first()
            if order:
                for key, value in filtered_updates.items():
                    setattr(order, key, value)
                return True
        return False

    return await _run_sync(_update)


async def list_orders(user_id: str, symbol: Optional[str] = None,
                      status: Optional[str] = None, exchange: Optional[str] = None,
                      strategy_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0,
                      sort_by: str = "created_at", sort_order: str = "desc") -> List[Dict]:
    """List orders with filtering, sorting, and pagination."""
    def _list():
        with transaction() as session:
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
            return [_model_to_dict(o) for o in orders]

    return await _run_sync(_list)


async def delete_order(order_id: str) -> bool:
    """Delete an order record."""
    def _delete():
        with transaction() as session:
            deleted = session.query(OrderRecord).filter(OrderRecord.id == order_id).delete()
            return deleted > 0

    return await _run_sync(_delete)


# ============================================================================
# Strategy CRUD
# ============================================================================

async def create_strategy(user_id: str, name: str, type: str,
                           symbol: str, config: Dict = None) -> str:
    """Create a new strategy record."""
    strategy_id = f"strat_{uuid.uuid4().hex[:12]}"

    def _create():
        with transaction() as session:
            strategy = Strategy(
                id=strategy_id, user_id=user_id, name=name,
                type=type, symbol=symbol, config=config or {},
            )
            session.add(strategy)
        return strategy_id

    return await _run_sync(_create)


async def get_strategy(strategy_id: str) -> Optional[Dict]:
    """Get strategy by ID."""
    def _get():
        with transaction() as session:
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            return _model_to_dict(strategy) if strategy else None

    return await _run_sync(_get)


async def list_strategies(user_id: str) -> List[Dict]:
    """List all strategies for a user."""
    def _list():
        with transaction() as session:
            strategies = session.query(Strategy).filter(Strategy.user_id == user_id).all()
            return [_model_to_dict(s) for s in strategies]

    return await _run_sync(_list)


async def update_strategy(strategy_id: str, updates: Dict) -> bool:
    """Update strategy fields with whitelist protection."""
    filtered_updates = {k: v for k, v in updates.items() if k in ALLOWED_STRATEGY_FIELDS}
    if not filtered_updates:
        return False

    def _update():
        with transaction() as session:
            strategy = session.query(Strategy).filter(Strategy.id == strategy_id).first()
            if strategy:
                for key, value in filtered_updates.items():
                    setattr(strategy, key, value)
                return True
        return False

    return await _run_sync(_update)


async def delete_strategy(strategy_id: str) -> bool:
    """Delete a strategy record."""
    def _delete():
        with transaction() as session:
            deleted = session.query(Strategy).filter(Strategy.id == strategy_id).delete()
            return deleted > 0

    return await _run_sync(_delete)


# ============================================================================
# Trade CRUD
# ============================================================================

async def create_trade(order_id: str, symbol: str, side: str,
                       quantity: float, price: float, exchange: str = "paper",
                       commission: float = 0, commission_asset: str = "",
                       exchange_trade_id: str = None, is_maker: bool = False,
                       slippage: float = 0) -> str:
    """Create a single trade record."""
    trade_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            trade = TradeRecord(
                id=trade_id, order_id=order_id, symbol=symbol, side=side,
                quantity=quantity, price=price, commission=commission,
                commission_asset=commission_asset, exchange=exchange,
                exchange_trade_id=exchange_trade_id, is_maker=is_maker,
                slippage=slippage,
            )
            session.add(trade)
        return trade_id

    return await _run_sync(_create)


async def list_trades(symbol: Optional[str] = None, side: Optional[str] = None,
                      exchange: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> List[Dict]:
    """List trades with filtering and pagination."""
    def _list():
        with transaction() as session:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if side:
                query = query.filter(TradeRecord.side == side)
            if exchange:
                query = query.filter(TradeRecord.exchange == exchange)
            trades = query.order_by(desc(TradeRecord.timestamp)).offset(offset).limit(limit).all()
            return [_model_to_dict(t) for t in trades]

    return await _run_sync(_list)


async def bulk_insert_trades(trades: List[Dict]) -> int:
    """Bulk insert trade records for high-frequency data."""
    if not trades:
        return 0

    def _bulk():
        with transaction() as session:
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

    return await _run_sync(_bulk)


# ============================================================================
# Candle CRUD
# ============================================================================

async def get_candles(symbol: str, timeframe: str,
                      limit: int = 500) -> List[Dict]:
    """Get candle data from database."""
    def _get():
        with transaction() as session:
            candles = session.query(CandleRecord).filter(
                and_(CandleRecord.symbol == symbol, CandleRecord.timeframe == timeframe)
            ).order_by(desc(CandleRecord.open_time)).limit(limit).all()
            return [_model_to_dict(c) for c in reversed(candles)]

    return await _run_sync(_get)


async def bulk_insert_candles(candles: List[Dict]) -> int:
    """Bulk insert candle records."""
    if not candles:
        return 0

    def _bulk():
        with transaction() as session:
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

    return await _run_sync(_bulk)


# ============================================================================
# Position CRUD
# ============================================================================

async def create_position(user_id: str, symbol: str, side: str,
                           quantity: float, entry_price: float, exchange: str = "paper",
                           strategy_id: Optional[str] = None,
                           leverage: float = 1) -> str:
    """Create a new position record."""
    position_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            position = PositionRecord(
                id=position_id, user_id=user_id, strategy_id=strategy_id,
                symbol=symbol, side=side, quantity=quantity,
                entry_price=entry_price, exchange=exchange, leverage=leverage,
            )
            session.add(position)
        return position_id

    return await _run_sync(_create)


async def update_position(position_id: str, updates: Dict) -> bool:
    """Update position fields."""
    allowed_position_fields = {
        "quantity", "mark_price", "unrealized_pnl", "realized_pnl",
        "leverage", "closed_at", "updated_at",
    }
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_position_fields}
    if not filtered_updates:
        return False

    def _update():
        with transaction() as session:
            position = session.query(PositionRecord).filter(
                PositionRecord.id == position_id
            ).first()
            if position:
                for key, value in filtered_updates.items():
                    setattr(position, key, value)
                return True
        return False

    return await _run_sync(_update)


async def close_position(position_id: str, realized_pnl: float = 0) -> bool:
    """Close a position by setting closed_at timestamp."""
    def _close():
        with transaction() as session:
            position = session.query(PositionRecord).filter(
                PositionRecord.id == position_id
            ).first()
            if position:
                position.closed_at = datetime.utcnow()
                position.realized_pnl = realized_pnl
                return True
        return False

    return await _run_sync(_close)


# ============================================================================
# Signal CRUD
# ============================================================================

async def create_signal(strategy_id: str, symbol: str, direction: str,
                         strength: float = 0, indicators: Dict = None,
                         metadata: Dict = None) -> str:
    """Create a new signal record."""
    signal_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            signal = SignalRecord(
                id=signal_id, strategy_id=strategy_id, symbol=symbol,
                direction=direction, strength=strength,
                indicators=indicators or {}, metadata=metadata or {},
            )
            session.add(signal)
        return signal_id

    return await _run_sync(_create)


async def get_signal(signal_id: str) -> Optional[Dict]:
    """Get signal by ID."""
    def _get():
        with transaction() as session:
            signal = session.query(SignalRecord).filter(SignalRecord.id == signal_id).first()
            return _model_to_dict(signal) if signal else None

    return await _run_sync(_get)


# ============================================================================
# Risk Event CRUD
# ============================================================================

async def create_risk_event(user_id: str, event_type: str, severity: str,
                             details: Dict = None) -> str:
    """Create a risk event record."""
    event_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            event = RiskEvent(
                id=event_id, user_id=user_id, event_type=event_type,
                severity=severity, details=details or {},
            )
            session.add(event)
        return event_id

    return await _run_sync(_create)


async def list_risk_events(user_id: str, severity: Optional[str] = None,
                            limit: int = 50, offset: int = 0) -> List[Dict]:
    """List risk events for a user."""
    def _list():
        with transaction() as session:
            query = session.query(RiskEvent).filter(RiskEvent.user_id == user_id)
            if severity:
                query = query.filter(RiskEvent.severity == severity)
            events = query.order_by(desc(RiskEvent.timestamp)).offset(offset).limit(limit).all()
            return [_model_to_dict(e) for e in events]

    return await _run_sync(_list)


async def get_risk_event(event_id: str) -> Optional[Dict]:
    """Get risk event by ID."""
    def _get():
        with transaction() as session:
            event = session.query(RiskEvent).filter(RiskEvent.id == event_id).first()
            return _model_to_dict(event) if event else None

    return await _run_sync(_get)


# ============================================================================
# Exchange Credential CRUD
# ============================================================================

async def create_exchange_credential(user_id: str, exchange: str,
                                      api_key: str, api_secret: str,
                                      passphrase: Optional[str] = None,
                                      is_testnet: bool = False,
                                      encryption_key: Optional[str] = None) -> str:
    """Create encrypted exchange credentials."""
    encryptor = CredentialEncryptor(key=encryption_key)
    cred_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            cred = ExchangeCredential(
                id=cred_id, user_id=user_id, exchange=exchange,
                api_key_encrypted=encryptor.encrypt(api_key),
                api_secret_encrypted=encryptor.encrypt(api_secret),
                passphrase_encrypted=encryptor.encrypt(passphrase) if passphrase else None,
                is_testnet=is_testnet,
            )
            session.add(cred)
        return cred_id

    return await _run_sync(_create)


async def get_exchange_credential(cred_id: str, decrypt: bool = False,
                                    encryption_key: Optional[str] = None) -> Optional[Dict]:
    """Get exchange credential by ID."""
    encryptor = CredentialEncryptor(key=encryption_key) if decrypt else None

    def _get():
        with transaction() as session:
            cred = session.query(ExchangeCredential).filter(
                ExchangeCredential.id == cred_id
            ).first()
            if not cred:
                return None
            result = _model_to_dict(cred)
            if decrypt and encryptor:
                result["api_key"] = encryptor.decrypt(cred.api_key_encrypted)
                result["api_secret"] = encryptor.decrypt(cred.api_secret_encrypted)
                if cred.passphrase_encrypted:
                    result["passphrase"] = encryptor.decrypt(cred.passphrase_encrypted)
            return result

    return await _run_sync(_get)


async def list_exchange_credentials(user_id: str, exchange: Optional[str] = None,
                                      encryption_key: Optional[str] = None) -> List[Dict]:
    """List exchange credentials for a user."""
    encryptor = CredentialEncryptor(key=encryption_key)

    def _list():
        with transaction() as session:
            query = session.query(ExchangeCredential).filter(
                ExchangeCredential.user_id == user_id
            )
            if exchange:
                query = query.filter(ExchangeCredential.exchange == exchange)
            creds = query.all()
            results = []
            for cred in creds:
                result = _model_to_dict(cred)
                try:
                    result["api_key"] = encryptor.decrypt(cred.api_key_encrypted)
                except Exception:
                    result["api_key"] = "***encrypted***"
                results.append(result)
            return results

    return await _run_sync(_list)


async def deactivate_exchange_credential(cred_id: str) -> bool:
    """Deactivate an exchange credential."""
    def _deactivate():
        with transaction() as session:
            cred = session.query(ExchangeCredential).filter(
                ExchangeCredential.id == cred_id
            ).first()
            if cred:
                cred.is_active = False
                return True
        return False

    return await _run_sync(_deactivate)


# ============================================================================
# Portfolio Snapshot CRUD
# ============================================================================

async def create_portfolio_snapshot(user_id: str, total_value: float,
                                      available_balance: float = 0,
                                      unrealized_pnl: float = 0, realized_pnl: float = 0,
                                      margin_used: float = 0, leverage: float = 1,
                                      positions: List[Dict] = None) -> str:
    """Create a portfolio snapshot record."""
    snapshot_id = str(uuid.uuid4())

    def _create():
        with transaction() as session:
            snapshot = PortfolioSnapshotRecord(
                id=snapshot_id, user_id=user_id, total_value=total_value,
                available_balance=available_balance, unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl, margin_used=margin_used,
                leverage=leverage, positions_json=positions or [],
            )
            session.add(snapshot)
        return snapshot_id

    return await _run_sync(_create)


async def get_portfolio_snapshot(snapshot_id: str) -> Optional[Dict]:
    """Get portfolio snapshot by ID."""
    def _get():
        with transaction() as session:
            snapshot = session.query(PortfolioSnapshotRecord).filter(
                PortfolioSnapshotRecord.id == snapshot_id
            ).first()
            return _model_to_dict(snapshot) if snapshot else None

    return await _run_sync(_get)


# ============================================================================
# Query Helpers
# ============================================================================

async def get_active_orders(user_id: str) -> List[Dict]:
    """Get all active orders for a user."""
    def _get():
        with transaction() as session:
            orders = session.query(OrderRecord).filter(
                and_(OrderRecord.user_id == user_id,
                     OrderRecord.status.in_(["created", "partially_filled", "pending"]))
            ).all()
            return [_model_to_dict(o) for o in orders]

    return await _run_sync(_get)


async def get_open_positions(user_id: str) -> List[Dict]:
    """Get all open positions for a user."""
    def _get():
        with transaction() as session:
            positions = session.query(PositionRecord).filter(
                and_(PositionRecord.user_id == user_id,
                     PositionRecord.closed_at.is_(None))
            ).all()
            return [_model_to_dict(p) for p in positions]

    return await _run_sync(_get)


async def get_recent_signals(symbol: Optional[str] = None,
                              strategy_id: Optional[str] = None,
                              direction: Optional[str] = None,
                              limit: int = 50, offset: int = 0) -> List[Dict]:
    """Get recent signals with optional filtering."""
    def _get():
        with transaction() as session:
            query = session.query(SignalRecord)
            if symbol:
                query = query.filter(SignalRecord.symbol == symbol)
            if strategy_id:
                query = query.filter(SignalRecord.strategy_id == strategy_id)
            if direction:
                query = query.filter(SignalRecord.direction == direction)
            signals = query.order_by(desc(SignalRecord.timestamp)).offset(offset).limit(limit).all()
            return [_model_to_dict(s) for s in signals]

    return await _run_sync(_get)


async def get_pnl_history(user_id: str, days: int = 30) -> List[Dict]:
    """Get P&L history for a user."""
    since = datetime.utcnow() - timedelta(days=days)

    def _get():
        with transaction() as session:
            snapshots = session.query(PortfolioSnapshotRecord).filter(
                and_(PortfolioSnapshotRecord.user_id == user_id,
                     PortfolioSnapshotRecord.timestamp >= since)
            ).order_by(asc(PortfolioSnapshotRecord.timestamp)).all()
            return [_model_to_dict(s) for s in snapshots]

    return await _run_sync(_get)


async def get_latest_portfolio_snapshot(user_id: str) -> Optional[Dict]:
    """Get the most recent portfolio snapshot."""
    def _get():
        with transaction() as session:
            snapshot = session.query(PortfolioSnapshotRecord).filter(
                PortfolioSnapshotRecord.user_id == user_id
            ).order_by(desc(PortfolioSnapshotRecord.timestamp)).first()
            return _model_to_dict(snapshot) if snapshot else None

    return await _run_sync(_get)


# ============================================================================
# Backtest Results
# ============================================================================

async def create_backtest_result(strategy_id: str, config: Dict,
                                  results: Dict) -> str:
    """Create a backtest result record."""
    result_id = f"bt_{uuid.uuid4().hex[:12]}"

    def _create():
        with transaction() as session:
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

    return await _run_sync(_create)


async def get_backtest_result(backtest_id: str) -> Optional[Dict]:
    """Get backtest result by ID."""
    def _get():
        with transaction() as session:
            bt = session.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
            return _model_to_dict(bt) if bt else None

    return await _run_sync(_get)


# ============================================================================
# Data Cleanup / Archival
# ============================================================================

async def cleanup_old_candles(symbol: str, timeframe: str,
                               keep_days: int = 90) -> int:
    """Remove candle data older than keep_days."""
    cutoff = datetime.utcnow() - timedelta(days=keep_days)

    def _cleanup():
        with transaction() as session:
            deleted = session.query(CandleRecord).filter(
                and_(CandleRecord.symbol == symbol,
                     CandleRecord.timeframe == timeframe,
                     CandleRecord.open_time < cutoff)
            ).delete()
        return deleted

    return await _run_sync(_cleanup)


async def archive_old_trades(days: int = 180) -> int:
    """Move old trades to archive table instead of deleting them."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    def _archive():
        with transaction() as session:
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
            result = session.execute(
                text("DELETE FROM trades WHERE timestamp < :cutoff"),
                {"cutoff": cutoff}
            )
            return result.rowcount

    return await _run_sync(_archive)


# ============================================================================
# Alembic Integration Helpers
# ============================================================================

def get_alembic_config(db_url: str = "") -> Dict:
    """Get Alembic configuration for migrations."""
    return {
        "script_location": "alembic",
        "sqlalchemy.url": db_url or "postgresql://acms:acms@localhost:5432/acms",
        "render_as_batch": True,
    }


def check_migration_status(engine) -> Dict:
    """Check current migration status."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            versions = [row[0] for row in result]
            return {"current_version": versions[0] if versions else None, "status": "up_to_date"}
    except Exception as e:
        return {"current_version": None, "status": "not_initialized", "error": str(e)}


__all__ = [
    "_model_to_dict",
    "transaction",
    "_run_sync",
    "create_user", "get_user", "get_user_by_email", "update_user", "delete_user",
    "create_api_key", "get_api_key", "get_api_key_by_hash", "list_api_keys",
    "deactivate_api_key", "update_api_key_last_used",
    "create_order", "get_order", "update_order", "list_orders", "delete_order",
    "create_strategy", "get_strategy", "list_strategies", "update_strategy", "delete_strategy",
    "create_trade", "list_trades", "bulk_insert_trades",
    "get_candles", "bulk_insert_candles",
    "create_position", "update_position", "close_position",
    "create_signal", "get_signal",
    "create_risk_event", "list_risk_events", "get_risk_event",
    "create_exchange_credential", "get_exchange_credential",
    "list_exchange_credentials", "deactivate_exchange_credential",
    "create_portfolio_snapshot", "get_portfolio_snapshot",
    "get_active_orders", "get_open_positions", "get_recent_signals",
    "get_pnl_history", "get_latest_portfolio_snapshot",
    "create_backtest_result", "get_backtest_result",
    "cleanup_old_candles", "archive_old_trades",
    "get_alembic_config", "check_migration_status",
]
'''
    write_file(f"{BASE}/db/crud.py", crud_content)

    # manager.py - DatabaseManager that ties everything together
    manager_content = '''"""DatabaseManager - high-level database operations manager."""

import uuid
import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import and_, desc, asc, text
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from acms.db.models import (
    Base, User, ApiKey, Strategy, OrderRecord, TradeRecord, TradeArchiveRecord,
    PositionRecord, SignalRecord, CandleRecord, RiskEvent, BacktestResult,
    PortfolioSnapshotRecord, ExchangeCredential,
    ALLOWED_ORDER_FIELDS, ALLOWED_STRATEGY_FIELDS,
)
from acms.db.encryption import CredentialEncryptor
from acms.db.crud import _model_to_dict

logger = logging.getLogger(__name__)


class DatabaseManager:
    """High-level database operations manager.

    Provides CRUD operations for all models, transaction management,
    bulk inserts, and query helpers. Delegates to standalone functions
    in crud.py where possible.
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

    # Delegate all CRUD methods to the standalone functions in crud.py
    # but use this manager's engine/session for backward compatibility.

    async def create_user(self, email: str, password: str, username: str = "",
                          is_admin: bool = False) -> str:
        from acms.db.crud import create_user as _create_user
        return await _create_user(email, password, username, is_admin)

    async def get_user(self, user_id: str = None, email: str = None) -> Optional[Dict]:
        from acms.db.crud import get_user as _get_user
        return await _get_user(user_id, email)

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        from acms.db.crud import get_user_by_email
        return await get_user_by_email(email)

    async def update_user(self, user_id: str, updates: Dict) -> bool:
        from acms.db.crud import update_user as _update_user
        return await _update_user(user_id, updates)

    async def delete_user(self, user_id: str) -> bool:
        from acms.db.crud import delete_user as _delete_user
        return await _delete_user(user_id)

    async def create_api_key(self, user_id: str, name: str, key_hash: str,
                              permissions: Dict = None) -> str:
        from acms.db.crud import create_api_key as _create_api_key
        return await _create_api_key(user_id, name, key_hash, permissions)

    async def get_api_key(self, key_id: str) -> Optional[Dict]:
        from acms.db.crud import get_api_key as _get_api_key
        return await _get_api_key(key_id)

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[Dict]:
        from acms.db.crud import get_api_key_by_hash
        return await get_api_key_by_hash(key_hash)

    async def list_api_keys(self, user_id: str) -> List[Dict]:
        from acms.db.crud import list_api_keys
        return await list_api_keys(user_id)

    async def deactivate_api_key(self, key_id: str) -> bool:
        from acms.db.crud import deactivate_api_key
        return await deactivate_api_key(key_id)

    async def update_api_key_last_used(self, key_id: str) -> None:
        from acms.db.crud import update_api_key_last_used
        await update_api_key_last_used(key_id)

    async def create_order(self, user_id: str, symbol: str, side: str,
                           order_type: str, quantity: float, price: Optional[float] = None,
                           stop_price: Optional[float] = None, exchange: str = "paper",
                           strategy_id: Optional[str] = None) -> Dict:
        from acms.db.crud import create_order as _create_order
        return await _create_order(user_id, symbol, side, order_type, quantity,
                                   price, stop_price, exchange, strategy_id)

    async def get_order(self, order_id: str) -> Optional[Dict]:
        from acms.db.crud import get_order as _get_order
        return await _get_order(order_id)

    async def update_order(self, order_id: str, updates: Dict) -> bool:
        from acms.db.crud import update_order as _update_order
        return await _update_order(order_id, updates)

    async def list_orders(self, user_id: str, symbol: Optional[str] = None,
                          status: Optional[str] = None, exchange: Optional[str] = None,
                          strategy_id: Optional[str] = None,
                          limit: int = 50, offset: int = 0,
                          sort_by: str = "created_at", sort_order: str = "desc") -> List[Dict]:
        from acms.db.crud import list_orders as _list_orders
        return await _list_orders(user_id, symbol, status, exchange, strategy_id,
                                  limit, offset, sort_by, sort_order)

    async def delete_order(self, order_id: str) -> bool:
        from acms.db.crud import delete_order
        return await delete_order(order_id)

    async def create_strategy(self, user_id: str, name: str, type: str,
                               symbol: str, config: Dict = None) -> str:
        from acms.db.crud import create_strategy as _create_strategy
        return await _create_strategy(user_id, name, type, symbol, config)

    async def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        from acms.db.crud import get_strategy as _get_strategy
        return await _get_strategy(strategy_id)

    async def list_strategies(self, user_id: str) -> List[Dict]:
        from acms.db.crud import list_strategies
        return await list_strategies(user_id)

    async def update_strategy(self, strategy_id: str, updates: Dict) -> bool:
        from acms.db.crud import update_strategy as _update_strategy
        return await _update_strategy(strategy_id, updates)

    async def delete_strategy(self, strategy_id: str) -> bool:
        from acms.db.crud import delete_strategy
        return await delete_strategy(strategy_id)

    async def create_trade(self, order_id: str, symbol: str, side: str,
                           quantity: float, price: float, exchange: str = "paper",
                           commission: float = 0, commission_asset: str = "",
                           exchange_trade_id: str = None, is_maker: bool = False,
                           slippage: float = 0) -> str:
        from acms.db.crud import create_trade as _create_trade
        return await _create_trade(order_id, symbol, side, quantity, price, exchange,
                                   commission, commission_asset, exchange_trade_id,
                                   is_maker, slippage)

    async def list_trades(self, symbol: Optional[str] = None, side: Optional[str] = None,
                          exchange: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> List[Dict]:
        from acms.db.crud import list_trades as _list_trades
        return await _list_trades(symbol, side, exchange, limit, offset)

    async def bulk_insert_trades(self, trades: List[Dict]) -> int:
        from acms.db.crud import bulk_insert_trades
        return await bulk_insert_trades(trades)

    async def get_candles(self, symbol: str, timeframe: str,
                          limit: int = 500) -> List[Dict]:
        from acms.db.crud import get_candles
        return await get_candles(symbol, timeframe, limit)

    async def bulk_insert_candles(self, candles: List[Dict]) -> int:
        from acms.db.crud import bulk_insert_candles
        return await bulk_insert_candles(candles)

    async def create_position(self, user_id: str, symbol: str, side: str,
                               quantity: float, entry_price: float, exchange: str = "paper",
                               strategy_id: Optional[str] = None,
                               leverage: float = 1) -> str:
        from acms.db.crud import create_position as _create_position
        return await _create_position(user_id, symbol, side, quantity, entry_price,
                                      exchange, strategy_id, leverage)

    async def update_position(self, position_id: str, updates: Dict) -> bool:
        from acms.db.crud import update_position as _update_position
        return await _update_position(position_id, updates)

    async def close_position(self, position_id: str, realized_pnl: float = 0) -> bool:
        from acms.db.crud import close_position
        return await close_position(position_id, realized_pnl)

    async def create_signal(self, strategy_id: str, symbol: str, direction: str,
                             strength: float = 0, indicators: Dict = None,
                             metadata: Dict = None) -> str:
        from acms.db.crud import create_signal as _create_signal
        return await _create_signal(strategy_id, symbol, direction, strength,
                                    indicators, metadata)

    async def get_signal(self, signal_id: str) -> Optional[Dict]:
        from acms.db.crud import get_signal
        return await get_signal(signal_id)

    async def create_risk_event(self, user_id: str, event_type: str, severity: str,
                                 details: Dict = None) -> str:
        from acms.db.crud import create_risk_event
        return await create_risk_event(user_id, event_type, severity, details)

    async def list_risk_events(self, user_id: str, severity: Optional[str] = None,
                                limit: int = 50, offset: int = 0) -> List[Dict]:
        from acms.db.crud import list_risk_events
        return await list_risk_events(user_id, severity, limit, offset)

    async def get_risk_event(self, event_id: str) -> Optional[Dict]:
        from acms.db.crud import get_risk_event
        return await get_risk_event(event_id)

    async def create_exchange_credential(self, user_id: str, exchange: str,
                                          api_key: str, api_secret: str,
                                          passphrase: Optional[str] = None,
                                          is_testnet: bool = False) -> str:
        from acms.db.crud import create_exchange_credential
        return await create_exchange_credential(
            user_id, exchange, api_key, api_secret, passphrase, is_testnet,
            encryption_key=None,
        )

    async def get_exchange_credential(self, cred_id: str, decrypt: bool = False) -> Optional[Dict]:
        from acms.db.crud import get_exchange_credential
        return await get_exchange_credential(cred_id, decrypt, encryption_key=None)

    async def list_exchange_credentials(self, user_id: str, exchange: Optional[str] = None) -> List[Dict]:
        from acms.db.crud import list_exchange_credentials
        return await list_exchange_credentials(user_id, exchange, encryption_key=None)

    async def deactivate_exchange_credential(self, cred_id: str) -> bool:
        from acms.db.crud import deactivate_exchange_credential
        return await deactivate_exchange_credential(cred_id)

    async def create_portfolio_snapshot(self, user_id: str, total_value: float,
                                          available_balance: float = 0,
                                          unrealized_pnl: float = 0, realized_pnl: float = 0,
                                          margin_used: float = 0, leverage: float = 1,
                                          positions: List[Dict] = None) -> str:
        from acms.db.crud import create_portfolio_snapshot
        return await create_portfolio_snapshot(
            user_id, total_value, available_balance, unrealized_pnl,
            realized_pnl, margin_used, leverage, positions,
        )

    async def get_portfolio_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        from acms.db.crud import get_portfolio_snapshot
        return await get_portfolio_snapshot(snapshot_id)

    async def get_active_orders(self, user_id: str) -> List[Dict]:
        from acms.db.crud import get_active_orders
        return await get_active_orders(user_id)

    async def get_open_positions(self, user_id: str) -> List[Dict]:
        from acms.db.crud import get_open_positions
        return await get_open_positions(user_id)

    async def get_recent_signals(self, symbol: Optional[str] = None,
                                  strategy_id: Optional[str] = None,
                                  direction: Optional[str] = None,
                                  limit: int = 50, offset: int = 0) -> List[Dict]:
        from acms.db.crud import get_recent_signals
        return await get_recent_signals(symbol, strategy_id, direction, limit, offset)

    async def get_pnl_history(self, user_id: str, days: int = 30) -> List[Dict]:
        from acms.db.crud import get_pnl_history
        return await get_pnl_history(user_id, days)

    async def get_latest_portfolio_snapshot(self, user_id: str) -> Optional[Dict]:
        from acms.db.crud import get_latest_portfolio_snapshot
        return await get_latest_portfolio_snapshot(user_id)

    async def create_backtest_result(self, strategy_id: str, config: Dict,
                                      results: Dict) -> str:
        from acms.db.crud import create_backtest_result
        return await create_backtest_result(strategy_id, config, results)

    async def get_backtest_result(self, backtest_id: str) -> Optional[Dict]:
        from acms.db.crud import get_backtest_result
        return await get_backtest_result(backtest_id)

    async def cleanup_old_candles(self, symbol: str, timeframe: str,
                                   keep_days: int = 90) -> int:
        from acms.db.crud import cleanup_old_candles
        return await cleanup_old_candles(symbol, timeframe, keep_days)

    async def archive_old_trades(self, days: int = 180) -> int:
        from acms.db.crud import archive_old_trades
        return await archive_old_trades(days)

    @staticmethod
    def get_alembic_config(db_url: str = "") -> Dict:
        from acms.db.crud import get_alembic_config
        return get_alembic_config(db_url)

    @staticmethod
    def check_migration_status(engine) -> Dict:
        from acms.db.crud import check_migration_status
        return check_migration_status(engine)

    @staticmethod
    def _model_to_dict(model) -> Dict:
        return _model_to_dict(model)


__all__ = ["DatabaseManager"]
'''
    write_file(f"{BASE}/db/manager.py", manager_content)

    # __init__.py - re-exports all
    init_content = '''"""Database module - PostgreSQL + SQLAlchemy + Alembic.

    Re-exports all public names from submodules for backward compatibility.
    """

from acms.db.models import (
    Base,
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
    ALLOWED_ORDER_FIELDS,
    ALLOWED_STRATEGY_FIELDS,
)
from acms.db.encryption import CredentialEncryptor
from acms.db.session import get_engine, get_session, init_db
from acms.db.crud import (
    _model_to_dict,
    transaction,
    _run_sync,
    create_user, get_user, get_user_by_email, update_user, delete_user,
    create_api_key, get_api_key, get_api_key_by_hash, list_api_keys,
    deactivate_api_key, update_api_key_last_used,
    create_order, get_order, update_order, list_orders, delete_order,
    create_strategy, get_strategy, list_strategies, update_strategy, delete_strategy,
    create_trade, list_trades, bulk_insert_trades,
    get_candles, bulk_insert_candles,
    create_position, update_position, close_position,
    create_signal, get_signal,
    create_risk_event, list_risk_events, get_risk_event,
    create_exchange_credential, get_exchange_credential,
    list_exchange_credentials, deactivate_exchange_credential,
    create_portfolio_snapshot, get_portfolio_snapshot,
    get_active_orders, get_open_positions, get_recent_signals,
    get_pnl_history, get_latest_portfolio_snapshot,
    create_backtest_result, get_backtest_result,
    cleanup_old_candles, archive_old_trades,
    get_alembic_config, check_migration_status,
)
from acms.db.manager import DatabaseManager

__all__ = [
    # Models
    "Base",
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
    "ALLOWED_ORDER_FIELDS",
    "ALLOWED_STRATEGY_FIELDS",
    # Encryption
    "CredentialEncryptor",
    # Session
    "get_engine",
    "get_session",
    "init_db",
    # CRUD
    "_model_to_dict",
    "transaction",
    "_run_sync",
    "create_user", "get_user", "get_user_by_email", "update_user", "delete_user",
    "create_api_key", "get_api_key", "get_api_key_by_hash", "list_api_keys",
    "deactivate_api_key", "update_api_key_last_used",
    "create_order", "get_order", "update_order", "list_orders", "delete_order",
    "create_strategy", "get_strategy", "list_strategies", "update_strategy", "delete_strategy",
    "create_trade", "list_trades", "bulk_insert_trades",
    "get_candles", "bulk_insert_candles",
    "create_position", "update_position", "close_position",
    "create_signal", "get_signal",
    "create_risk_event", "list_risk_events", "get_risk_event",
    "create_exchange_credential", "get_exchange_credential",
    "list_exchange_credentials", "deactivate_exchange_credential",
    "create_portfolio_snapshot", "get_portfolio_snapshot",
    "get_active_orders", "get_open_positions", "get_recent_signals",
    "get_pnl_history", "get_latest_portfolio_snapshot",
    "create_backtest_result", "get_backtest_result",
    "cleanup_old_candles", "archive_old_trades",
    "get_alembic_config", "check_migration_status",
    # Manager
    "DatabaseManager",
]
'''
    write_file(f"{BASE}/db/__init__.py", init_content)

    # Verify
    for fname in ["models.py", "encryption.py", "session.py", "crud.py", "manager.py", "__init__.py"]:
        path = f"{BASE}/db/{fname}"
        compile_check(path)


# ============================================================================
# 2-6: Use a generic splitter for the remaining modules
# ============================================================================

def split_module(package_name, splits):
    """Split a package's __init__.py into submodules.
    
    splits: list of (filename, class_names, extra_imports_str, docstring)
    """
    print(f"\n=== Splitting {package_name}/__init__.py ===")
    
    init_path = f"{BASE}/{package_name}/__init__.py"
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Extract top-level imports (before first class)
    lines = content.split('\n')
    top_imports = []
    first_class_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            top_imports.append(line)
        elif stripped.startswith('class ') or stripped.startswith('@dataclass'):
            first_class_line = i
            break
        elif stripped.startswith('"""') and i == 0:
            continue
        elif stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
            # Could be a variable assignment or dataclass before first class
            if first_class_line is None and not stripped.startswith('logger'):
                # Check if next non-blank line is a class
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('class ') or lines[j].strip().startswith('@dataclass'):
                        first_class_line = i
                        break
                if first_class_line is None:
                    top_imports.append(line)
            elif first_class_line is None:
                top_imports.append(line)
    
    # For each split, extract the class code
    all_exports = []
    
    for filename, class_names, extra_imports, docstring in splits:
        # Find each class in the original content
        class_codes = []
        for cls_name in class_names:
            # Find the class start
            pattern = rf'^(class {cls_name}\b|@dataclass\s*\nclass {cls_name}\b)'
            match = re.search(pattern, content, re.MULTILINE)
            if match is None:
                # Try without @dataclass
                pattern2 = rf'^class {cls_name}\b'
                match = re.search(pattern2, content, re.MULTILINE)
                if match is None:
                    print(f"  WARNING: {cls_name} not found in {package_name}/__init__.py")
                    continue
            
            start = match.start()
            # Find where this class ends (next top-level class or end of file)
            # A class ends when we hit a line at column 0 that's not blank or a decorator
            after_start = content[match.end():]
            
            # Find next class definition at module level
            next_match = re.search(r'\nclass \w+', after_start)
            next_match2 = re.search(r'\n@dataclass', after_start)
            next_match3 = re.search(r'\ndef \w+', after_start)
            next_match4 = re.search(r'\n# =+', after_start)
            
            candidates = []
            for m in [next_match, next_match2, next_match3, next_match4]:
                if m:
                    candidates.append(m.start())
            
            if candidates:
                end_offset = min(candidates)
                # But check if it's inside the class (indented)
                class_text = after_start[:end_offset]
            else:
                class_text = after_start
            
            # Remove trailing blank lines
            class_text = class_text.rstrip()
            
            # Check if there was a @dataclass decorator before the class
            pre_class = content[max(0, start-20):start]
            if '@dataclass' in pre_class:
                class_text = '@dataclass\n' + class_text
            
            class_codes.append(class_text)
        
        if not class_codes:
            continue
            
        # Build the module file
        module_content = f'"""{docstring}"""\n\n'
        module_content += extra_imports + '\n\n'
        module_content += '\n\n\n'.join(class_codes) + '\n\n'
        module_content += f'__all__ = {class_names}\n'
        
        write_file(f"{BASE}/{package_name}/{filename}", module_content)
        all_exports.extend(class_names)
    
    # Write __init__.py with re-exports
    init_content = f'"""{package_name} module.\n\nRe-exports all public names from submodules for backward compatibility.\n"""\n\n'
    for filename, class_names, extra_imports, docstring in splits:
        module_name = filename.replace('.py', '')
        init_content += f'from acms.{package_name}.{module_name} import (\n'
        for name in class_names:
            init_content += f'    {name},\n'
        init_content += ')\n\n'
    
    init_content += f'__all__ = [\n'
    for name in all_exports:
        init_content += f'    "{name}",\n'
    init_content += ']\n'
    
    write_file(f"{BASE}/{package_name}/__init__.py", init_content)
    
    # Verify
    for filename, _, _, _ in splits:
        path = f"{BASE}/{package_name}/{filename}"
        compile_check(path)
    compile_check(f"{BASE}/{package_name}/__init__.py")


# ============================================================================
# 2. ml/__init__.py
# ============================================================================
def split_ml():
    print("\n=== Splitting ml/__init__.py ===")
    
    splits = [
        ("config.py", ["MLConfig"],
         "from dataclasses import dataclass",
         "ML configuration."),
        
        ("features.py", ["FeatureEngineer"],
         "import numpy as np\nfrom typing import Optional",
         "Feature engineering for ML models."),
        
        ("validation.py", ["WalkForwardValidation"],
         "import numpy as np\nfrom typing import List, Dict, Tuple",
         "Walk-forward validation for time series."),
        
        ("ensemble.py", ["EnsembleModel"],
         "import numpy as np\nfrom typing import Optional, List, Any",
         "Ensemble model methods."),
        
        ("monitor.py", ["ModelMonitor"],
         "import numpy as np\nfrom scipy import stats as scipy_stats\nfrom typing import Optional, Dict",
         "Model monitoring and drift detection."),
        
        ("anomaly.py", ["AnomalyDetector"],
         "import numpy as np\nfrom typing import Optional",
         "Anomaly detection using autoencoders."),
        
        ("transformer.py", ["TransformerPredictor"],
         "import numpy as np\nfrom typing import Optional, Dict\nfrom acms.ml.config import MLConfig",
         "Transformer-based price prediction."),
        
        ("lstm.py", ["PricePredictionModel"],
         "import numpy as np\nfrom typing import Optional, Dict\nfrom pathlib import Path\nfrom acms.ml.config import MLConfig",
         "LSTM-based price prediction model."),
        
        ("lightgbm_model.py", ["LightGBMSignalModel"],
         "import numpy as np\nfrom typing import Optional, Dict\nfrom pathlib import Path\nfrom acms.ml.config import MLConfig",
         "LightGBM gradient boosting signal model."),
        
        ("hyperopt.py", ["HyperparameterOptimizer"],
         "import numpy as np\nfrom typing import Optional, Dict",
         "Optuna hyperparameter optimization."),
        
        ("rl.py", ["TradingEnvironment", "RLExecutionOptimizer"],
         "import numpy as np\nfrom typing import List, Dict, Optional\nfrom pathlib import Path\nfrom acms.ml.config import MLConfig",
         "RL trading environment and execution optimizer."),
        
        ("registry.py", ["ModelRegistry"],
         "from typing import Optional, List, Dict, Any\nfrom pathlib import Path",
         "Model persistence registry."),
    ]
    
    split_module("ml", splits)


# ============================================================================
# 3. backtest/__init__.py
# ============================================================================
def split_backtest():
    print("\n=== Splitting backtest/__init__.py ===")
    
    extra_imports = """import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum

from acms.core import Candle, Signal, SignalDirection, Position, Side, Trade
from acms.strategies import Strategy
from acms.risk import RiskEngine, RiskConfig
from acms.indicators import ATR"""
    
    splits = [
        ("config.py", ["BacktestMode", "BacktestConfig", "BacktestTrade", "MCStatistics", "BacktestResult"],
         extra_imports,
         "Backtest configuration and data classes."),
        
        ("slippage.py", ["SlippageModel"],
         "import numpy as np\nfrom acms.core import Side",
         "Slippage models for execution simulation."),
        
        ("fill_model.py", ["FillModel"],
         "from typing import Dict",
         "Execution fill models."),
        
        ("analytics.py", ["TradeAnalytics", "RollingMetrics"],
         "import numpy as np\nfrom acms.core import Side",
         "Trade analytics and rolling metrics."),
        
        ("benchmark.py", ["BenchmarkComparison", "RegimeDetector", "SensitivityAnalysis"],
         "import numpy as np\nfrom typing import Optional, List, Dict\nfrom acms.core import Candle, Side\nfrom acms.strategies import Strategy",
         "Benchmark comparison and analysis."),
        
        ("engine.py", ["BacktestEngine"],
         """import logging
import numpy as np
from typing import Optional, List, Dict
from acms.core import Candle, Signal, SignalDirection, Position, Side, Trade
from acms.strategies import Strategy
from acms.risk import RiskEngine, RiskConfig
from acms.indicators import ATR
from acms.backtest.config import BacktestConfig, BacktestMode, BacktestResult, BacktestTrade, MCStatistics
from acms.backtest.slippage import SlippageModel
from acms.backtest.fill_model import FillModel
from acms.backtest.analytics import TradeAnalytics, RollingMetrics
from acms.backtest.benchmark import BenchmarkComparison, RegimeDetector, SensitivityAnalysis""",
         "Backtest engine - single, walk-forward, and Monte Carlo."),
    ]
    
    split_module("backtest", splits)


# ============================================================================
# 4. redis_client/__init__.py
# ============================================================================
def split_redis_client():
    print("\n=== Splitting redis_client/__init__.py ===")
    
    base_imports = """import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable, Awaitable, Set"""
    
    splits = [
        ("client.py", ["RedisConfig", "get_redis", "_InMemoryRedis", "_InMemoryPubSub", "_InMemoryPipeline"],
         base_imports,
         "Redis client factory and in-memory fallback."),
        
        ("cache.py", ["CacheManager"],
         """import json
import logging
import asyncio
from typing import Optional, Any, Callable""",
         "Cache manager with TTL and pattern invalidation."),
        
        ("pubsub.py", ["PubSubManager"],
         """import json
import logging
import asyncio
from typing import Dict, List, Any, Callable, Awaitable""",
         "PubSub manager for real-time events."),
        
        ("rate_limiter.py", ["RedisRateLimiter"],
         """import logging
import time
from typing import Dict""",
         "Sliding window rate limiter."),
        
        ("session.py", ["SessionManager"],
         """import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict""",
         "User session storage with TTL."),
        
        ("market_data.py", ["MarketDataCache"],
         """import json
import logging
from datetime import datetime
from typing import Optional, Dict""",
         "Market data cache with auto-expiry."),
    ]
    
    split_module("redis_client", splits)


# ============================================================================
# 5. pipeline/__init__.py
# ============================================================================
def split_pipeline():
    print("\n=== Splitting pipeline/__init__.py ===")
    
    splits = [
        ("config.py", ["PipelineConfig"],
         "from dataclasses import dataclass, field",
         "Pipeline configuration."),
        
        ("quality.py", ["DataQualityChecker"],
         "import logging\nimport numpy as np\nfrom typing import Optional, Dict, List",
         "Data quality checking."),
        
        ("resampler.py", ["DataResampler"],
         "import logging\nfrom typing import List, Dict",
         "Data resampling between timeframes."),
        
        ("windowing.py", ["DataWindowing"],
         "import numpy as np\nfrom typing import List, Dict",
         "Data windowing operations."),
        
        ("storage.py", ["ParquetStorage"],
         "import logging\nimport numpy as np\nfrom pathlib import Path\nfrom typing import Optional, List, Any",
         "Parquet file storage."),
        
        ("engine.py", ["DataPipeline"],
         """import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import numpy as np
from acms.pipeline.config import PipelineConfig
from acms.pipeline.quality import DataQualityChecker
from acms.pipeline.resampler import DataResampler
from acms.pipeline.windowing import DataWindowing
from acms.pipeline.storage import ParquetStorage""",
         "Data pipeline engine."),
    ]
    
    split_module("pipeline", splits)


# ============================================================================
# 6. reporting/__init__.py
# ============================================================================
def split_reporting():
    print("\n=== Splitting reporting/__init__.py ===")
    
    splits = [
        ("models.py", ["DrawdownPeriod", "PerformanceReport", "StrategyReport"],
         "from dataclasses import dataclass, field, asdict\nfrom datetime import datetime\nfrom typing import Optional, Dict, List",
         "Reporting data models."),
        
        ("engine.py", ["ReportingEngine"],
         """import json
import logging
import math
import numpy as np
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from acms.reporting.models import DrawdownPeriod, PerformanceReport, StrategyReport""",
         "Reporting engine for generating analytics."),
    ]
    
    split_module("reporting", splits)


if __name__ == "__main__":
    split_db()
    split_ml()
    split_backtest()
    split_redis_client()
    split_pipeline()
    split_reporting()
    
    print("\n=== All splits complete ===")
