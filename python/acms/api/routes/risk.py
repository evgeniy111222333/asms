"""Risk status endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Depends

from acms.api.schemas import RiskStatusResponse
from acms.api.dependencies import get_current_user, get_engines
from acms.auth import TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


@router.get("/status", response_model=RiskStatusResponse)
async def get_risk_status(user: TokenData = Depends(get_current_user)):
    """Get current risk status from the risk engine."""
    _engines = get_engines()
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
        except Exception as e:
            logger.warning(f"Risk engine unavailable: {e}")
            raise HTTPException(status_code=503, detail="Risk engine temporarily unavailable")

    raise HTTPException(status_code=503, detail="Risk engine not configured")


@router.post("/kill-switch")
async def trigger_kill_switch(reason: str = "Manual trigger", user: TokenData = Depends(get_current_user)):
    """Trigger the kill switch."""
    _engines = get_engines()
    risk_engine = _engines.get("risk")
    if risk_engine:
        risk_engine.trigger_kill_switch(reason)
    return {"message": "Kill switch triggered", "reason": reason}


@router.post("/kill-switch/reset")
async def reset_kill_switch(user: TokenData = Depends(get_current_user)):
    """Reset the kill switch."""
    _engines = get_engines()
    risk_engine = _engines.get("risk")
    if risk_engine:
        risk_engine.reset_kill_switch()
    return {"message": "Kill switch reset"}


__all__ = ["router"]
