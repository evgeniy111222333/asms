"""Position and portfolio query endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends

from acms.api.schemas import PositionResponse, PortfolioResponse
from acms.api.dependencies import get_db, get_current_user, get_engines, _get_user_id
from acms.auth import TokenData
from acms.db import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["positions"])


@router.get("/api/v1/positions", response_model=list[PositionResponse])
async def list_positions(user: TokenData = Depends(get_current_user),
                         db: DatabaseManager = Depends(get_db)):
    """List all open positions."""
    user_id = _get_user_id(user)
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


@router.get("/api/v1/portfolio", response_model=PortfolioResponse)
async def get_portfolio(user: TokenData = Depends(get_current_user),
                        db: DatabaseManager = Depends(get_db)):
    """Get portfolio snapshot."""
    user_id = _get_user_id(user)
    snapshot = await db.get_latest_portfolio_snapshot(user_id)

    _engines = get_engines()
    portfolio_engine = _engines.get("portfolio")
    if portfolio_engine:
        try:
            portfolio_data = portfolio_engine.get_snapshot()
            return PortfolioResponse(
                total_value=portfolio_data.get("total_value", 0),
                available_balance=portfolio_data.get("available_balance", 0),
                unrealized_pnl=portfolio_data.get("unrealized_pnl", 0),
                realized_pnl=portfolio_data.get("realized_pnl", 0),
                positions=[], margin_used=0, leverage=1,
            )
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            raise HTTPException(status_code=500, detail="Portfolio data unavailable")

    if not snapshot:
        raise HTTPException(status_code=404, detail="No portfolio data found for user")

    return PortfolioResponse(
        total_value=float(snapshot.get("total_value", 0)),
        available_balance=float(snapshot.get("available_balance", 0)),
        unrealized_pnl=float(snapshot.get("unrealized_pnl", 0)),
        realized_pnl=float(snapshot.get("realized_pnl", 0)),
        positions=[], margin_used=float(snapshot.get("margin_used", 0)),
        leverage=float(snapshot.get("leverage", 1)),
    )


__all__ = ["router"]
