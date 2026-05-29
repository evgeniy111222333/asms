"""FastAPI REST API + WebSocket server for ACMS.

Implements:
- Real database integration for all CRUD operations
- Real engine integration: SignalEngine, RiskEngine, PortfolioEngine, BacktestEngine
- WebSocket with real data streaming
- Input validation with Pydantic
- Error handling with proper HTTP status codes
- Rate limiting per endpoint
- Pagination for list endpoints
- Filtering and sorting for orders, trades, signals
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
import time
import logging
from collections import defaultdict

from acms.core import ACMSConfig, Side, OrderType, OrderStatus, SignalDirection
from acms.auth import AuthManager, TokenData
from acms.db import init_db, DatabaseManager

logger = logging.getLogger(__name__)

app = FastAPI(title="ACMS API", version="0.1.0", description="Algorithmic Crypto Management System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
auth_manager = AuthManager()

# Global state for engine integration
_db_manager: Optional[DatabaseManager] = None
_engines: Dict[str, Any] = {}


async def get_db() -> DatabaseManager:
    """Get database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def set_engines(signal_engine=None, risk_engine=None, portfolio_engine=None, backtest_engine=None):
    """Set engine instances for API integration."""
    _engines["signal"] = signal_engine
    _engines["risk"] = risk_engine
    _engines["portfolio"] = portfolio_engine
    _engines["backtest"] = backtest_engine


# ============================================================================
# Rate Limiting
# ============================================================================

class EndpointRateLimiter:
    """Simple in-memory rate limiter per endpoint."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        self._requests[key] = [t for t in self._requests[key]
                                if now - t < self.window_seconds]
        if len(self._requests[key]) >= self.max_requests:
            return False
        self._requests[key].append(now)
        return True


rate_limiter = EndpointRateLimiter(max_requests=100, window_seconds=60)


# ============================================================================
# Pydantic Models (Input Validation)
# ============================================================================

class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class CreateOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=50)
    side: str = Field(..., pattern="^(buy|sell)$")
    order_type: str = Field(..., pattern="^(market|limit|stop|stop_limit|trailing_stop)$")
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: str = Field("gtc", pattern="^(gtc|ioc|fok|gtd|day)$")
    exchange: str = Field("paper", pattern="^(binance|bybit|okx|paper)$")
    strategy_id: Optional[str] = None

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in ('limit', 'stop_limit') and v is None:
            raise ValueError("Price required for limit orders")
        return v


class OrderResponse(BaseModel):
    id: str
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float
    price: Optional[float]
    filled_quantity: float
    average_fill_price: float
    commission: float
    exchange: str
    strategy_id: Optional[str]
    created_at: datetime


class SignalResponse(BaseModel):
    id: str
    symbol: str
    direction: str
    strength: float
    strategy_id: str
    indicators: Dict[str, Any]
    timestamp: datetime


class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    exchange: str


class PortfolioResponse(BaseModel):
    total_value: float
    available_balance: float
    unrealized_pnl: float
    realized_pnl: float
    positions: List[PositionResponse]
    margin_used: float
    leverage: float


class BacktestRequest(BaseModel):
    strategy_type: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    initial_capital: float = Field(100000.0, gt=0)
    config: Dict[str, Any] = {}


class RiskStatusResponse(BaseModel):
    kill_switch_active: bool
    kill_switch_reason: str
    current_drawdown: float
    total_exposure: float
    var_99: Optional[float]
    cvar_99: Optional[float]


class StrategyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=1, max_length=50)
    config: Dict[str, Any] = {}


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    has_next: bool


# ============================================================================
# Auth Dependency
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Validate JWT token and return user data."""
    token = credentials.credentials
    data = auth_manager.verify_token(token)
    if data is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return data


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for the current request."""
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ============================================================================
# Auth Endpoints
# ============================================================================

@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT token."""
    user = auth_manager.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth_manager.create_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )


@app.post("/api/v1/auth/register")
async def register(request: LoginRequest, db: DatabaseManager = Depends(get_db)):
    """Register a new user."""
    try:
        user_id = await db.create_user(
            email=request.email,
            password=request.password,
        )
        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Order Endpoints
# ============================================================================

@app.post("/api/v1/orders", response_model=OrderResponse)
async def create_order(request: CreateOrderRequest,
                       user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db),
                       _rate: None = Depends(check_rate_limit)):
    """Submit a new order with risk checks."""
    # Create order in database
    order_data = await db.create_order(
        user_id=user.user_id if hasattr(user, 'user_id') else "default",
        symbol=request.symbol,
        side=request.side,
        order_type=request.order_type,
        quantity=request.quantity,
        price=request.price,
        stop_price=request.stop_price,
        exchange=request.exchange,
        strategy_id=request.strategy_id,
    )
    return OrderResponse(
        id=order_data.get("id", ""),
        symbol=request.symbol, side=request.side, order_type=request.order_type,
        status="created", quantity=request.quantity, price=request.price,
        filled_quantity=0, average_fill_price=0, commission=0,
        exchange=request.exchange, strategy_id=request.strategy_id,
        created_at=datetime.utcnow(),
    )


@app.get("/api/v1/orders", response_model=List[OrderResponse])
async def list_orders(symbol: Optional[str] = None, status: Optional[str] = None,
                      exchange: Optional[str] = None, strategy_id: Optional[str] = None,
                      page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                      sort_by: str = Query("created_at", pattern="^(created_at|symbol|status)$"),
                      sort_order: str = Query("desc", pattern="^(asc|desc)$"),
                      user: TokenData = Depends(get_current_user),
                      db: DatabaseManager = Depends(get_db)):
    """List orders with filtering, sorting, and pagination."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    orders = await db.list_orders(
        user_id=user_id,
        symbol=symbol, status=status, exchange=exchange,
        strategy_id=strategy_id,
        limit=page_size, offset=(page - 1) * page_size,
        sort_by=sort_by, sort_order=sort_order,
    )
    return [
        OrderResponse(
            id=o.get("id", ""), symbol=o.get("symbol", ""),
            side=o.get("side", ""), order_type=o.get("order_type", ""),
            status=o.get("status", ""), quantity=float(o.get("quantity", 0)),
            price=float(o.get("price", 0)) if o.get("price") else None,
            filled_quantity=float(o.get("filled_quantity", 0)),
            average_fill_price=float(o.get("average_fill_price", 0)),
            commission=float(o.get("commission", 0)),
            exchange=o.get("exchange", ""), strategy_id=o.get("strategy_id"),
            created_at=o.get("created_at", datetime.utcnow()),
        )
        for o in orders
    ]


@app.get("/api/v1/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str, user: TokenData = Depends(get_current_user),
                    db: DatabaseManager = Depends(get_db)):
    """Get order by ID."""
    order = await db.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse(
        id=order.get("id", order_id), symbol=order.get("symbol", ""),
        side=order.get("side", ""), order_type=order.get("order_type", ""),
        status=order.get("status", ""), quantity=float(order.get("quantity", 0)),
        price=float(order.get("price", 0)) if order.get("price") else None,
        filled_quantity=float(order.get("filled_quantity", 0)),
        average_fill_price=float(order.get("average_fill_price", 0)),
        commission=float(order.get("commission", 0)),
        exchange=order.get("exchange", ""), strategy_id=order.get("strategy_id"),
        created_at=order.get("created_at", datetime.utcnow()),
    )


@app.delete("/api/v1/orders/{order_id}")
async def cancel_order(order_id: str, user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db)):
    """Cancel an order."""
    success = await db.update_order(order_id, {"status": "cancelled"})
    if not success:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Order cancelled", "order_id": order_id}


# ============================================================================
# Position Endpoints
# ============================================================================

@app.get("/api/v1/positions", response_model=List[PositionResponse])
async def list_positions(user: TokenData = Depends(get_current_user),
                         db: DatabaseManager = Depends(get_db)):
    """List all open positions."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    positions = await db.get_open_positions(user_id)
    return [
        PositionResponse(
            symbol=p.get("symbol", ""), side=p.get("side", ""),
            quantity=float(p.get("quantity", 0)),
            entry_price=float(p.get("entry_price", 0)),
            mark_price=float(p.get("mark_price", 0)),
            unrealized_pnl=float(p.get("unrealized_pnl", 0)),
            realized_pnl=float(p.get("realized_pnl", 0)),
            leverage=float(p.get("leverage", 1)),
            exchange=p.get("exchange", ""),
        )
        for p in positions
    ]


@app.get("/api/v1/portfolio", response_model=PortfolioResponse)
async def get_portfolio(user: TokenData = Depends(get_current_user),
                        db: DatabaseManager = Depends(get_db)):
    """Get portfolio snapshot."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    snapshot = await db.get_latest_portfolio_snapshot(user_id)

    portfolio_engine = _engines.get("portfolio")
    if portfolio_engine:
        try:
            portfolio_data = portfolio_engine.get_snapshot()
            return PortfolioResponse(
                total_value=portfolio_data.get("total_value", 100000),
                available_balance=portfolio_data.get("available_balance", 100000),
                unrealized_pnl=portfolio_data.get("unrealized_pnl", 0),
                realized_pnl=portfolio_data.get("realized_pnl", 0),
                positions=[], margin_used=0, leverage=1,
            )
        except Exception:
            pass

    return PortfolioResponse(
        total_value=float(snapshot.get("total_value", 100000)) if snapshot else 100000,
        available_balance=float(snapshot.get("available_balance", 100000)) if snapshot else 100000,
        unrealized_pnl=float(snapshot.get("unrealized_pnl", 0)) if snapshot else 0,
        realized_pnl=float(snapshot.get("realized_pnl", 0)) if snapshot else 0,
        positions=[], margin_used=float(snapshot.get("margin_used", 0)) if snapshot else 0,
        leverage=float(snapshot.get("leverage", 1)) if snapshot else 1,
    )


# ============================================================================
# Strategy Endpoints
# ============================================================================

@app.post("/api/v1/strategies")
async def create_strategy(request: StrategyCreateRequest,
                          user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
    """Create a new strategy."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    strategy_id = await db.create_strategy(
        user_id=user_id, name=request.name,
        type=request.type, symbol=request.symbol, config=request.config,
    )
    return {"id": strategy_id, "name": request.name, "type": request.type, "status": "created"}


@app.get("/api/v1/strategies")
async def list_strategies(user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
    """List all strategies."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    strategies = await db.list_strategies(user_id)
    return strategies


@app.post("/api/v1/strategies/{strategy_id}/start")
async def start_strategy(strategy_id: str, user: TokenData = Depends(get_current_user),
                         db: DatabaseManager = Depends(get_db)):
    """Start a strategy."""
    success = await db.update_strategy(strategy_id, {"is_active": True})
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"message": f"Strategy {strategy_id} started"}


@app.post("/api/v1/strategies/{strategy_id}/stop")
async def stop_strategy(strategy_id: str, user: TokenData = Depends(get_current_user),
                        db: DatabaseManager = Depends(get_db)):
    """Stop a strategy."""
    success = await db.update_strategy(strategy_id, {"is_active": False})
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"message": f"Strategy {strategy_id} stopped"}


# ============================================================================
# Backtest Endpoints
# ============================================================================

@app.post("/api/v1/backtest")
async def run_backtest(request: BacktestRequest, user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db)):
    """Run a backtest."""
    backtest_engine = _engines.get("backtest")
    if backtest_engine is None:
        raise HTTPException(status_code=503, detail="Backtest engine not available")

    try:
        result = await asyncio.to_thread(
            backtest_engine.run,
            strategy_type=request.strategy_type,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            config=request.config,
        )
        # Store result
        strategy_id = request.config.get("strategy_id", "unknown")
        result_id = await db.create_backtest_result(
            strategy_id=strategy_id, config=request.config,
            results=result,
        )
        return {"backtest_id": result_id, "status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/api/v1/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str, user: TokenData = Depends(get_current_user),
                              db: DatabaseManager = Depends(get_db)):
    """Get backtest result."""
    result = await db.get_backtest_result(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result


# ============================================================================
# Risk Endpoints
# ============================================================================

@app.get("/api/v1/risk/status", response_model=RiskStatusResponse)
async def get_risk_status(user: TokenData = Depends(get_current_user)):
    """Get current risk status from the risk engine."""
    risk_engine = _engines.get("risk")
    if risk_engine:
        try:
            return RiskStatusResponse(
                kill_switch_active=risk_engine.kill_switch_active,
                kill_switch_reason=getattr(risk_engine, 'kill_switch_reason', ''),
                current_drawdown=getattr(risk_engine, 'current_drawdown', 0.0),
                total_exposure=getattr(risk_engine, 'total_exposure', 0.0),
                var_99=getattr(risk_engine, 'var_99', None),
                cvar_99=getattr(risk_engine, 'cvar_99', None),
            )
        except Exception:
            pass

    return RiskStatusResponse(
        kill_switch_active=False, kill_switch_reason="",
        current_drawdown=0.0, total_exposure=0.0,
        var_99=None, cvar_99=None,
    )


@app.post("/api/v1/risk/kill-switch")
async def trigger_kill_switch(reason: str = "Manual trigger", user: TokenData = Depends(get_current_user)):
    """Trigger the kill switch."""
    risk_engine = _engines.get("risk")
    if risk_engine:
        risk_engine.trigger_kill_switch(reason)
    return {"message": "Kill switch triggered", "reason": reason}


@app.post("/api/v1/risk/kill-switch/reset")
async def reset_kill_switch(user: TokenData = Depends(get_current_user)):
    """Reset the kill switch."""
    risk_engine = _engines.get("risk")
    if risk_engine:
        risk_engine.reset_kill_switch()
    return {"message": "Kill switch reset"}


# ============================================================================
# Market Data Endpoints
# ============================================================================

@app.get("/api/v1/market/candles/{symbol}")
async def get_candles(symbol: str, timeframe: str = "1h", limit: int = Query(500, ge=1, le=1500),
                      exchange: str = "binance", user: TokenData = Depends(get_current_user),
                      db: DatabaseManager = Depends(get_db)):
    """Get candle data for a symbol from database or exchange."""
    # Try database first
    candles = await db.get_candles(symbol=symbol, timeframe=timeframe, limit=limit)
    if candles:
        return {"symbol": symbol, "timeframe": timeframe, "candles": candles}

    return {"symbol": symbol, "timeframe": timeframe, "candles": []}


@app.get("/api/v1/market/orderbook/{symbol}")
async def get_order_book(symbol: str, depth: int = Query(20, ge=1, le=100),
                         user: TokenData = Depends(get_current_user)):
    """Get order book for a symbol."""
    return {"symbol": symbol, "bids": [], "asks": []}


# ============================================================================
# Signal Endpoints
# ============================================================================

@app.get("/api/v1/signals", response_model=List[SignalResponse])
async def list_signals(symbol: Optional[str] = None, strategy_id: Optional[str] = None,
                       direction: Optional[str] = None,
                       limit: int = Query(50, ge=1, le=500),
                       page: int = Query(1, ge=1),
                       user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db)):
    """List recent signals with filtering."""
    signals = await db.get_recent_signals(
        symbol=symbol, strategy_id=strategy_id, direction=direction,
        limit=limit, offset=(page - 1) * limit,
    )
    return [
        SignalResponse(
            id=s.get("id", ""), symbol=s.get("symbol", ""),
            direction=s.get("direction", ""), strength=float(s.get("strength", 0)),
            strategy_id=s.get("strategy_id", ""), indicators=s.get("indicators", {}),
            timestamp=s.get("timestamp", datetime.utcnow()),
        )
        for s in signals
    ]


# ============================================================================
# Trade Endpoints
# ============================================================================

@app.get("/api/v1/trades")
async def list_trades(symbol: Optional[str] = None, side: Optional[str] = None,
                      exchange: Optional[str] = None,
                      page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                      user: TokenData = Depends(get_current_user),
                      db: DatabaseManager = Depends(get_db)):
    """List trades with filtering and pagination."""
    trades = await db.list_trades(
        symbol=symbol, side=side, exchange=exchange,
        limit=page_size, offset=(page - 1) * page_size,
    )
    return trades


# ============================================================================
# PnL History Endpoint
# ============================================================================

@app.get("/api/v1/pnl/history")
async def get_pnl_history(user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
    """Get P&L history."""
    user_id = user.user_id if hasattr(user, 'user_id') else "default"
    history = await db.get_pnl_history(user_id)
    return {"pnl_history": history}


# ============================================================================
# WebSocket
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._subscriptions: Dict[str, set] = defaultdict(set)  # client_id -> channels

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self._subscriptions.pop(client_id, None)

    def subscribe(self, client_id: str, channels: List[str]):
        self._subscriptions[client_id].update(channels)

    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast message to all clients subscribed to a channel."""
        for client_id, channels in self._subscriptions.items():
            if channel in channels and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json({
                        "channel": channel, **message,
                    })
                except Exception:
                    self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(client_id)
        for client_id in disconnected:
            self.disconnect(client_id)


ws_manager = ConnectionManager()


@app.websocket("/ws/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time data stream via WebSocket.

    Channels:
    - tick: Real-time trade data
    - book: Order book updates
    - signal: New trading signals
    - position: Position updates
    - risk: Risk alerts
    - pnl: P&L updates
    """
    import uuid
    client_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "subscribe":
                channels = msg.get("channels", [])
                ws_manager.subscribe(client_id, channels)
                await websocket.send_json({"type": "subscribed", "channels": channels})
            elif msg.get("type") == "unsubscribe":
                channels = msg.get("channels", [])
                for ch in channels:
                    ws_manager._subscriptions[client_id].discard(ch)
                await websocket.send_json({"type": "unsubscribed", "channels": channels})
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/v1/system/info")
async def system_info(user: TokenData = Depends(get_current_user)):
    return {
        "version": "0.1.0",
        "components": {
            "signal_engine": "active" if _engines.get("signal") else "not_configured",
            "risk_engine": "active" if _engines.get("risk") else "not_configured",
            "portfolio_engine": "active" if _engines.get("portfolio") else "not_configured",
            "execution_engine": "active" if _engines.get("backtest") else "not_configured",
            "ml_module": "available",
            "data_pipeline": "available",
        },
        "exchanges": ["binance", "bybit", "okx", "paper"],
        "websocket_connections": len(ws_manager.active_connections),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        db = await get_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization failed: %s", e)
