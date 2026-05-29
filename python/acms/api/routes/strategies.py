"""Strategy CRUD endpoints."""

from fastapi import APIRouter, HTTPException, Depends

from acms.api.schemas import StrategyCreateRequest
from acms.api.dependencies import get_db, get_current_user, _get_user_id
from acms.auth import TokenData
from acms.db import DatabaseManager

router = APIRouter(prefix="/api/v1/strategies", tags=["strategies"])


@router.post("")
async def create_strategy(request: StrategyCreateRequest,
                          user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
    """Create a new strategy."""
    user_id = _get_user_id(user)
    strategy_id = await db.create_strategy(
        user_id=user_id, name=request.name,
        type=request.type, symbol=request.symbol, config=request.config,
    )
    return {"id": strategy_id, "name": request.name, "type": request.type, "status": "created"}


@router.get("")
async def list_strategies(user: TokenData = Depends(get_current_user),
                          db: DatabaseManager = Depends(get_db)):
    """List all strategies."""
    user_id = _get_user_id(user)
    strategies = await db.list_strategies(user_id)
    return strategies


@router.post("/{strategy_id}/start")
async def start_strategy(strategy_id: str, user: TokenData = Depends(get_current_user),
                         db: DatabaseManager = Depends(get_db)):
    """Start a strategy."""
    success = await db.update_strategy(strategy_id, {"is_active": True})
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"message": f"Strategy {strategy_id} started"}


@router.post("/{strategy_id}/stop")
async def stop_strategy(strategy_id: str, user: TokenData = Depends(get_current_user),
                        db: DatabaseManager = Depends(get_db)):
    """Stop a strategy."""
    success = await db.update_strategy(strategy_id, {"is_active": False})
    if not success:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"message": f"Strategy {strategy_id} stopped"}


__all__ = ["router"]
