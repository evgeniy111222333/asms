"""Backtest endpoints."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Depends

from acms.api.schemas import BacktestRequest
from acms.api.dependencies import get_db, get_current_user, get_engines
from acms.auth import TokenData
from acms.db import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])


@router.post("")
async def run_backtest(request: BacktestRequest, user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db)):
    """Run a backtest."""
    _engines = get_engines()
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


@router.get("/{backtest_id}")
async def get_backtest_result(backtest_id: str, user: TokenData = Depends(get_current_user),
                              db: DatabaseManager = Depends(get_db)):
    """Get backtest result."""
    result = await db.get_backtest_result(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result


__all__ = ["router"]
