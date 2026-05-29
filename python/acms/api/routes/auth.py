"""Auth endpoints - login and register."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from acms.db import DatabaseManager
from acms.api.schemas import LoginRequest, TokenResponse
from acms.api.dependencies import get_db, auth_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT token."""
    user = auth_manager.authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth_manager.create_token(user["id"], user["email"])
    return TokenResponse(
        access_token=token,
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )


@router.post("/register")
async def register(request: LoginRequest, db: DatabaseManager = Depends(get_db)):
    """Register a new user."""
    try:
        user_id = await db.create_user(
            email=request.email,
            password=request.password,
        )
        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


__all__ = ["router"]
