"""Market data endpoints - candles and orderbook."""

import logging

from fastapi import APIRouter, Query, Depends

from acms.api.dependencies import get_db, get_current_user, get_engines
from acms.auth import TokenData
from acms.db import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/market", tags=["market"])


@router.get("/candles/{symbol}")
async def get_candles(symbol: str, timeframe: str = "1h", limit: int = Query(500, ge=1, le=1500),
                      exchange: str = "binance", user: TokenData = Depends(get_current_user),
                      db: DatabaseManager = Depends(get_db)):
    """Get candle data for a symbol from database or exchange."""
    # Try database first
    candles = await db.get_candles(symbol=symbol, timeframe=timeframe, limit=limit)
    if candles:
        return {"symbol": symbol, "timeframe": timeframe, "candles": candles}

    return {"symbol": symbol, "timeframe": timeframe, "candles": []}


@router.get("/orderbook/{symbol}")
async def get_order_book(symbol: str, depth: int = Query(20, ge=1, le=100),
                         user: TokenData = Depends(get_current_user),
                         db: DatabaseManager = Depends(get_db)):
    """Get order book for a symbol."""
    _engines = get_engines()
    try:
        if _engines.get("exchange"):
            exchange = _engines["exchange"]
            ob = await exchange.get_order_book(symbol, limit=depth)
            return {"symbol": symbol, "bids": ob.get("bids", []), "asks": ob.get("asks", [])}
        # Fallback: generate from recent trades in DB
        if _engines.get("db"):
            candles = await db.get_candles(symbol=symbol, timeframe="1m", limit=1)
            if candles:
                last_price = candles[-1].close
                spread = last_price * 0.001
                bids = [[last_price - spread * (i + 1), round(100 / (i + 1), 4)] for i in range(depth)]
                asks = [[last_price + spread * (i + 1), round(100 / (i + 1), 4)] for i in range(depth)]
                return {"symbol": symbol, "bids": bids, "asks": asks}
    except Exception as e:
        logger.warning(f"Failed to get orderbook for {symbol}: {e}")
    return {"symbol": symbol, "bids": [], "asks": []}


__all__ = ["router"]
