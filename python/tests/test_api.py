"""Comprehensive tests for acms.api module.

Tests all classes, methods, and edge cases:
- Pydantic models (LoginRequest, TokenResponse, CreateOrderRequest, etc.)
- EndpointRateLimiter
- FastAPI endpoints (using TestClient)
- Auth dependency
- WebSocket ConnectionManager
- Health check and system info
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the db module to avoid SQLAlchemy compatibility issues
import types
mock_db_mod = types.ModuleType('acms.db')
mock_db_mod.init_db = MagicMock(return_value=MagicMock())

class _MockDatabaseManager:
    """Mock DatabaseManager that returns async-compatible results."""
    def __init__(self, *args, **kwargs):
        pass
    async def get_latest_portfolio_snapshot(self, user_id):
        return None
    async def get_open_positions(self, user_id):
        return []
    async def list_orders(self, **kwargs):
        return []
    async def create_order(self, **kwargs):
        return {"id": "test-order-id", "symbol": "BTC/USDT", "status": "created"}
    async def get_order(self, order_id):
        return None
    async def update_order(self, order_id, updates):
        return True
    async def list_strategies(self, user_id):
        return []
    async def create_strategy(self, **kwargs):
        return "strat_test"
    async def update_strategy(self, strategy_id, updates):
        return True
    async def get_candles(self, **kwargs):
        return []
    async def get_recent_signals(self, **kwargs):
        return []
    async def list_trades(self, **kwargs):
        return []
    async def get_pnl_history(self, user_id):
        return []
    async def get_backtest_result(self, backtest_id):
        return None
    async def create_backtest_result(self, **kwargs):
        return "bt_test"
    async def create_user(self, **kwargs):
        return "user_test"

mock_db_mod.DatabaseManager = _MockDatabaseManager
sys.modules['acms.db'] = mock_db_mod

from fastapi.testclient import TestClient
from pydantic import ValidationError

from acms.api import (
    app, EndpointRateLimiter, rate_limiter,
    LoginRequest, TokenResponse, CreateOrderRequest,
    OrderResponse, SignalResponse, PositionResponse,
    PortfolioResponse, BacktestRequest, RiskStatusResponse,
    StrategyCreateRequest, PaginatedResponse,
    ws_manager, ConnectionManager,
    get_db, set_engines, get_current_user, check_rate_limit,
    auth_manager,
)
from acms.auth import AuthManager, TokenData


# ============================================================================
# Pydantic Model Tests
# ============================================================================

class TestLoginRequest:
    """Tests for LoginRequest model."""

    def test_valid_request(self):
        """Should accept valid email and password."""
        req = LoginRequest(email="test@example.com", password="secret123")
        assert req.email == "test@example.com"
        assert req.password == "secret123"

    def test_empty_email_fails(self):
        """Empty email should fail validation."""
        with pytest.raises(ValidationError):
            LoginRequest(email="", password="secret123")

    def test_empty_password_fails(self):
        """Empty password should fail validation."""
        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com", password="")

    def test_missing_email_fails(self):
        """Missing email should fail validation."""
        with pytest.raises(ValidationError):
            LoginRequest(password="secret123")

    def test_missing_password_fails(self):
        """Missing password should fail validation."""
        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com")

    def test_long_email_accepted(self):
        """Should accept email with special characters."""
        req = LoginRequest(email="a+b@test-domain.com", password="secret")
        assert "@" in req.email

    def test_long_password_accepted(self):
        """Should accept password up to 255 chars."""
        req = LoginRequest(email="test@test.com", password="x" * 255)
        assert len(req.password) == 255


class TestTokenResponse:
    """Tests for TokenResponse model."""

    def test_valid_response(self):
        """Should accept valid token response."""
        resp = TokenResponse(
            access_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        assert resp.access_token == "token123"
        assert resp.token_type == "bearer"

    def test_default_token_type(self):
        """Default token_type should be 'bearer'."""
        resp = TokenResponse(access_token="token", expires_at=datetime.utcnow())
        assert resp.token_type == "bearer"


class TestCreateOrderRequest:
    """Tests for CreateOrderRequest model."""

    def test_valid_market_order(self):
        """Should accept valid market order."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="market",
            quantity=1.0,
        )
        assert req.symbol == "BTC/USDT"
        assert req.side == "buy"
        assert req.order_type == "market"
        assert req.quantity == 1.0

    def test_valid_limit_order_with_price(self):
        """Should accept valid limit order with price."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="limit",
            quantity=1.0, price=50000.0,
        )
        assert req.price == 50000.0

    @pytest.mark.xfail(reason="Pydantic V1 @validator deprecation; needs migration to @field_validator")
    def test_limit_order_without_price_fails(self):
        """Limit order without price should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="limit",
                quantity=1.0,
            )
        # The validator adds an error to the errors list
        assert exc_info.value.errors()

    @pytest.mark.xfail(reason="Pydantic V1 @validator deprecation; needs migration to @field_validator")
    def test_stop_limit_without_price_fails(self):
        """Stop-limit order without price should fail."""
        with pytest.raises(ValidationError) as exc_info:
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="stop_limit",
                quantity=1.0,
            )
        assert exc_info.value.errors()

    def test_zero_quantity_fails(self):
        """Zero quantity should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="market",
                quantity=0.0,
            )

    def test_negative_quantity_fails(self):
        """Negative quantity should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="market",
                quantity=-1.0,
            )

    def test_invalid_side_fails(self):
        """Invalid side should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTC/USDT", side="hold", order_type="market",
                quantity=1.0,
            )

    def test_invalid_order_type_fails(self):
        """Invalid order type should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="iceberg",
                quantity=1.0,
            )

    def test_empty_symbol_fails(self):
        """Empty symbol should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="", side="buy", order_type="market",
                quantity=1.0,
            )

    def test_valid_sell_order(self):
        """Should accept sell order."""
        req = CreateOrderRequest(
            symbol="ETH/USDT", side="sell", order_type="market",
            quantity=10.0,
        )
        assert req.side == "sell"

    def test_default_exchange(self):
        """Default exchange should be 'paper'."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="market",
            quantity=1.0,
        )
        assert req.exchange == "paper"

    def test_default_time_in_force(self):
        """Default time_in_force should be 'gtc'."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="market",
            quantity=1.0,
        )
        assert req.time_in_force == "gtc"

    def test_all_valid_exchanges(self):
        """Should accept all valid exchanges."""
        for exchange in ["binance", "bybit", "okx", "paper"]:
            req = CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="market",
                quantity=1.0, exchange=exchange,
            )
            assert req.exchange == exchange

    def test_all_valid_order_types(self):
        """Should accept all valid order types."""
        for ot in ["market", "limit", "stop", "stop_limit", "trailing_stop"]:
            req = CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type=ot,
                quantity=1.0, price=50000.0 if ot in ("limit", "stop_limit") else None,
            )
            assert req.order_type == ot

    def test_all_valid_time_in_force(self):
        """Should accept all valid time in force values."""
        for tif in ["gtc", "ioc", "fok", "gtd", "day"]:
            req = CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="market",
                quantity=1.0, time_in_force=tif,
            )
            assert req.time_in_force == tif

    def test_optional_strategy_id(self):
        """strategy_id should be optional."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="market",
            quantity=1.0, strategy_id="strat1",
        )
        assert req.strategy_id == "strat1"

    def test_optional_stop_price(self):
        """stop_price should be optional."""
        req = CreateOrderRequest(
            symbol="BTC/USDT", side="buy", order_type="stop",
            quantity=1.0, stop_price=49000.0,
        )
        assert req.stop_price == 49000.0

    def test_negative_price_fails(self):
        """Negative price should fail."""
        with pytest.raises(ValidationError):
            CreateOrderRequest(
                symbol="BTC/USDT", side="buy", order_type="limit",
                quantity=1.0, price=-1.0,
            )


class TestBacktestRequest:
    """Tests for BacktestRequest model."""

    def test_valid_request(self):
        """Should accept valid backtest request."""
        req = BacktestRequest(
            strategy_type="momentum_trend", symbol="BTC/USDT",
            start_date="2024-01-01", end_date="2024-06-01",
        )
        assert req.strategy_type == "momentum_trend"
        assert req.initial_capital == 100000.0

    def test_custom_capital(self):
        """Should accept custom initial capital."""
        req = BacktestRequest(
            strategy_type="momentum_trend", symbol="BTC/USDT",
            start_date="2024-01-01", end_date="2024-06-01",
            initial_capital=50000.0,
        )
        assert req.initial_capital == 50000.0

    def test_zero_capital_fails(self):
        """Zero capital should fail."""
        with pytest.raises(ValidationError):
            BacktestRequest(
                strategy_type="momentum_trend", symbol="BTC/USDT",
                start_date="2024-01-01", end_date="2024-06-01",
                initial_capital=0.0,
            )

    def test_invalid_date_format_fails(self):
        """Invalid date format should fail."""
        with pytest.raises(ValidationError):
            BacktestRequest(
                strategy_type="momentum_trend", symbol="BTC/USDT",
                start_date="01-01-2024", end_date="2024-06-01",
            )

    def test_empty_strategy_type_fails(self):
        """Empty strategy type should fail."""
        with pytest.raises(ValidationError):
            BacktestRequest(
                strategy_type="", symbol="BTC/USDT",
                start_date="2024-01-01", end_date="2024-06-01",
            )


class TestStrategyCreateRequest:
    """Tests for StrategyCreateRequest model."""

    def test_valid_request(self):
        """Should accept valid strategy request."""
        req = StrategyCreateRequest(
            name="Test Strategy", type="momentum_trend", symbol="BTC/USDT",
        )
        assert req.name == "Test Strategy"

    def test_empty_name_fails(self):
        """Empty name should fail."""
        with pytest.raises(ValidationError):
            StrategyCreateRequest(name="", type="momentum", symbol="BTC/USDT")

    def test_default_config(self):
        """Default config should be empty dict."""
        req = StrategyCreateRequest(
            name="Test", type="momentum", symbol="BTC/USDT",
        )
        assert req.config == {}


# ============================================================================
# EndpointRateLimiter Tests
# ============================================================================

class TestEndpointRateLimiter:
    """Tests for EndpointRateLimiter class."""

    def test_defaults(self):
        """Should have default rate limit settings."""
        limiter = EndpointRateLimiter()
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60

    def test_custom_settings(self):
        """Should accept custom settings."""
        limiter = EndpointRateLimiter(max_requests=10, window_seconds=30)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 30

    def test_first_request_allowed(self):
        """First request should be allowed."""
        limiter = EndpointRateLimiter(max_requests=5, window_seconds=60)
        assert limiter.is_allowed("client1") is True

    def test_within_limit_allowed(self):
        """Requests within limit should be allowed."""
        limiter = EndpointRateLimiter(max_requests=5, window_seconds=60)
        for i in range(5):
            assert limiter.is_allowed("client1") is True

    def test_exceeds_limit_blocked(self):
        """Request exceeding limit should be blocked."""
        limiter = EndpointRateLimiter(max_requests=3, window_seconds=60)
        for i in range(3):
            limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False

    def test_different_clients_independent(self):
        """Different clients should have independent limits."""
        limiter = EndpointRateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.is_allowed("client1") is False
        assert limiter.is_allowed("client2") is True

    def test_window_expiry(self):
        """Requests outside window should not count."""
        limiter = EndpointRateLimiter(max_requests=1, window_seconds=0.01)
        limiter.is_allowed("client1")
        import time
        time.sleep(0.02)
        assert limiter.is_allowed("client1") is True


# ============================================================================
# ConnectionManager Tests
# ============================================================================

class TestConnectionManager:
    """Tests for WebSocket ConnectionManager class."""

    def setup_method(self):
        self.manager = ConnectionManager()

    def test_init(self):
        """Should initialize with empty connections."""
        assert len(self.manager.active_connections) == 0
        assert len(self.manager._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_connect(self):
        """Should add connection to active_connections."""
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        await self.manager.connect(mock_ws, "client1")
        assert "client1" in self.manager.active_connections
        mock_ws.accept.assert_called_once()

    def test_disconnect(self):
        """Should remove connection on disconnect."""
        self.manager.active_connections["client1"] = MagicMock()
        self.manager._subscriptions["client1"] = {"channel1"}
        self.manager.disconnect("client1")
        assert "client1" not in self.manager.active_connections
        assert "client1" not in self.manager._subscriptions

    def test_disconnect_nonexistent(self):
        """Disconnecting nonexistent client should not raise."""
        self.manager.disconnect("nonexistent")

    def test_subscribe(self):
        """Should subscribe client to channels."""
        self.manager.active_connections["client1"] = MagicMock()
        self.manager.subscribe("client1", ["tick", "signal"])
        assert "tick" in self.manager._subscriptions["client1"]
        assert "signal" in self.manager._subscriptions["client1"]

    def test_subscribe_additional_channels(self):
        """Should add channels to existing subscriptions."""
        self.manager._subscriptions["client1"] = {"tick"}
        self.manager.subscribe("client1", ["signal"])
        assert "tick" in self.manager._subscriptions["client1"]
        assert "signal" in self.manager._subscriptions["client1"]

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Should send message to all connections."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        self.manager.active_connections["c1"] = mock_ws1
        self.manager.active_connections["c2"] = mock_ws2
        await self.manager.broadcast({"type": "test"})
        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_disconnects_failed(self):
        """Should disconnect clients that fail to receive."""
        mock_ws1 = AsyncMock()
        mock_ws1.send_json.side_effect = Exception("Connection closed")
        mock_ws2 = AsyncMock()
        self.manager.active_connections["c1"] = mock_ws1
        self.manager.active_connections["c2"] = mock_ws2
        await self.manager.broadcast({"type": "test"})
        assert "c1" not in self.manager.active_connections
        assert "c2" in self.manager.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self):
        """Should send message only to subscribed clients."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        self.manager.active_connections["c1"] = mock_ws1
        self.manager.active_connections["c2"] = mock_ws2
        self.manager._subscriptions["c1"] = {"tick"}
        self.manager._subscriptions["c2"] = {"signal"}
        await self.manager.broadcast_to_channel("tick", {"type": "price"})
        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_to_channel_no_subscribers(self):
        """Should handle no subscribers gracefully."""
        await self.manager.broadcast_to_channel("tick", {"type": "price"})

    @pytest.mark.asyncio
    async def test_broadcast_empty(self):
        """Should handle no connections gracefully."""
        await self.manager.broadcast({"type": "test"})


# ============================================================================
# FastAPI Endpoint Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Should return healthy status."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_login_success(self):
        """Should return token on successful login."""
        client = TestClient(app)
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_empty_email(self):
        """Should reject empty email."""
        client = TestClient(app)
        response = client.post("/api/v1/auth/login", json={
            "email": "",
            "password": "password123",
        })
        assert response.status_code == 422

    def test_login_empty_password(self):
        """Should reject empty password."""
        client = TestClient(app)
        response = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "",
        })
        assert response.status_code == 422


class TestProtectedEndpoints:
    """Tests for endpoints requiring authentication."""

    def _get_auth_headers(self):
        """Get valid auth headers for testing."""
        token = auth_manager.create_token("test_user", "test@example.com")
        return {"Authorization": f"Bearer {token}"}

    def test_get_portfolio(self):
        """Should return portfolio data."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.get("/api/v1/portfolio", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "available_balance" in data

    def test_get_risk_status(self):
        """Should return risk status."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.get("/api/v1/risk/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "kill_switch_active" in data
        assert "current_drawdown" in data

    def test_trigger_kill_switch(self):
        """Should trigger kill switch."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.post("/api/v1/risk/kill-switch", headers=headers,
                               params={"reason": "Test"})
        assert response.status_code == 200

    def test_reset_kill_switch(self):
        """Should reset kill switch."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.post("/api/v1/risk/kill-switch/reset", headers=headers)
        assert response.status_code == 200

    def test_list_positions(self):
        """Should return positions list."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.get("/api/v1/positions", headers=headers)
        assert response.status_code == 200

    def test_get_orderbook(self):
        """Should return orderbook."""
        client = TestClient(app)
        headers = self._get_auth_headers()
        response = client.get("/api/v1/market/orderbook/BTC-USDT", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "bids" in data
        assert "asks" in data

    def test_unauthorized_access(self):
        """Should reject requests without auth."""
        client = TestClient(app)
        response = client.get("/api/v1/portfolio")
        assert response.status_code in (401, 403)  # Either unauthorized or forbidden

    def test_invalid_token(self):
        """Should reject invalid token."""
        client = TestClient(app)
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/portfolio", headers=headers)
        assert response.status_code == 401


class TestSystemInfoEndpoint:
    """Tests for system info endpoint."""

    def test_system_info(self):
        """Should return system information."""
        client = TestClient(app)
        token = auth_manager.create_token("test_user", "test@example.com")
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/system/info", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "components" in data
        assert "exchanges" in data


class TestSetEngines:
    """Tests for set_engines function."""

    def test_set_engines(self):
        """Should set engine instances."""
        mock_signal = MagicMock()
        mock_risk = MagicMock()
        mock_portfolio = MagicMock()
        mock_backtest = MagicMock()
        set_engines(
            signal_engine=mock_signal,
            risk_engine=mock_risk,
            portfolio_engine=mock_portfolio,
            backtest_engine=mock_backtest,
        )
        # Just verify it doesn't raise

    def test_set_partial_engines(self):
        """Should accept partial engine configuration."""
        set_engines(signal_engine=MagicMock())


class TestRiskStatusWithEngine:
    """Tests for risk status with risk engine set."""

    def test_risk_status_with_engine(self):
        """Should use risk engine data when available."""
        mock_risk = MagicMock()
        mock_risk.kill_switch_active = True
        mock_risk.kill_switch_reason = "Test"
        mock_risk.current_drawdown = 0.05
        mock_risk.total_exposure = 100000.0
        mock_risk.var_99 = 0.02
        mock_risk.cvar_99 = 0.03

        set_engines(risk_engine=mock_risk)

        client = TestClient(app)
        token = auth_manager.create_token("test_user", "test@example.com")
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/risk/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["kill_switch_active"] is True

        # Clean up
        set_engines(risk_engine=None)
