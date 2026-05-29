"""Signal query endpoints."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query, Depends

from acms.api.schemas import SignalResponse
from acms.api.dependencies import get_db, get_current_user
from acms.auth import TokenData
from acms.db import DatabaseManager

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


@router.get("", response_model=list[SignalResponse])
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


__all__ = ["router"]
