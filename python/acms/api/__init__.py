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

from acms.api.app import app, create_app
from acms.api.schemas import (
    LoginRequest, TokenResponse, CreateOrderRequest, OrderResponse,
    SignalResponse, PositionResponse, PortfolioResponse, BacktestRequest,
    RiskStatusResponse, StrategyCreateRequest, PaginatedResponse,
)
from acms.api.dependencies import (
    get_db, set_engines, get_engines, get_current_user,
    EndpointRateLimiter, rate_limiter, check_rate_limit,
)
from acms.api.routes.websocket import ConnectionManager, ws_manager, set_redis_client

__all__ = [
    "app",
    "create_app",
    "LoginRequest",
    "TokenResponse",
    "CreateOrderRequest",
    "OrderResponse",
    "SignalResponse",
    "PositionResponse",
    "PortfolioResponse",
    "BacktestRequest",
    "RiskStatusResponse",
    "StrategyCreateRequest",
    "PaginatedResponse",
    "get_db",
    "set_engines",
    "get_engines",
    "get_current_user",
    "EndpointRateLimiter",
    "rate_limiter",
    "check_rate_limit",
    "ConnectionManager",
    "ws_manager",
    "set_redis_client",
]
