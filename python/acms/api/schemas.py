"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


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


__all__ = [
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
]
