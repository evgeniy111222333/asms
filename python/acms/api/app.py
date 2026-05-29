"""FastAPI app factory and lifespan management."""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI

from acms.api.dependencies import get_db, get_engines
from acms.api.routes.auth import router as auth_router
from acms.api.routes.orders import router as orders_router
from acms.api.routes.positions import router as positions_router
from acms.api.routes.strategies import router as strategies_router
from acms.api.routes.backtest import router as backtest_router
from acms.api.routes.risk import router as risk_router
from acms.api.routes.market import router as market_router
from acms.api.routes.signals import router as signals_router
from acms.api.routes.websocket import router as websocket_router
from acms.api.routes.websocket import ws_manager

logger = logging.getLogger(__name__)

# Configurable CORS origins from environment
_CORS_ORIGINS = os.environ.get("ACMS_CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    try:
        db = await get_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization failed: %s", e)
    yield
    # Shutdown
    # Clean up WebSocket connections
    for client_id in list(ws_manager.active_connections.keys()):
        ws_manager.disconnect(client_id)
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from fastapi.middleware.cors import CORSMiddleware

    application = FastAPI(
        title="ACMS API",
        version="0.1.0",
        description="Algorithmic Crypto Management System",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    application.include_router(auth_router)
    application.include_router(orders_router)
    application.include_router(positions_router)
    application.include_router(strategies_router)
    application.include_router(backtest_router)
    application.include_router(risk_router)
    application.include_router(market_router)
    application.include_router(signals_router)
    application.include_router(websocket_router)

    # Health check endpoints
    @application.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    @application.get("/api/v1/system/info")
    async def system_info():
        _engines = get_engines()
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

    # Trade list endpoint (keeping at app level for backward compat)
    from acms.api.dependencies import get_current_user, _get_user_id
    from acms.db import DatabaseManager
    from fastapi import Depends, Query
    from acms.auth import TokenData

    @application.get("/api/v1/trades")
    async def list_trades(symbol: str = None, side: str = None,
                          exchange: str = None,
                          page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                          user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
        """List trades with filtering and pagination."""
        trades = await db.list_trades(
            symbol=symbol, side=side, exchange=exchange,
            limit=page_size, offset=(page - 1) * page_size,
        )
        return trades

    @application.get("/api/v1/pnl/history")
    async def get_pnl_history(user: TokenData = Depends(get_current_user),
                              db: DatabaseManager = Depends(get_db)):
        """Get P&L history."""
        user_id = _get_user_id(user)
        history = await db.get_pnl_history(user_id)
        return {"pnl_history": history}

    return application


# Create the default app instance
app = create_app()


__all__ = [
    "app",
    "create_app",
    "lifespan",
]
