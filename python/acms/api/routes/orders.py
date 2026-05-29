"""Order CRUD endpoints."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from acms.api.schemas import CreateOrderRequest, OrderResponse
from acms.api.dependencies import (
    get_db, get_current_user, get_engines, check_rate_limit, _get_user_id,
)
from acms.auth import TokenData
from acms.db import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])


@router.post("", response_model=OrderResponse)
async def create_order(request: CreateOrderRequest,
                       user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db),
                       _rate: None = Depends(check_rate_limit)):
    """Submit a new order with risk checks."""
    user_id = _get_user_id(user)
    _engines = get_engines()

    # Risk check
    risk_engine = _engines.get("risk")
    if risk_engine:
        try:
            risk_result = risk_engine.pre_trade_check(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                price=request.price or 0,
                order_type=request.order_type,
                current_positions=_engines.get("positions", {}),
                portfolio_value=_engines.get("portfolio_value", 0),
                account_equity=_engines.get("equity", 0),
            )
            if not risk_result.get("approved", True):
                raise HTTPException(status_code=400, detail=f"Risk check failed: {risk_result.get('reason', 'Unknown')}")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Risk check error: {e}")

    # Create order in database
    order_data = await db.create_order(
        user_id=user_id,
        symbol=request.symbol,
        side=request.side,
        order_type=request.order_type,
        quantity=request.quantity,
        price=request.price,
        stop_price=request.stop_price,
        exchange=request.exchange,
        strategy_id=request.strategy_id,
    )
    return OrderResponse(
        id=order_data.get("id", ""),
        symbol=request.symbol, side=request.side, order_type=request.order_type,
        status="created", quantity=request.quantity, price=request.price,
        filled_quantity=0, average_fill_price=0, commission=0,
        exchange=request.exchange, strategy_id=request.strategy_id,
        created_at=datetime.utcnow(),
    )


@router.get("", response_model=list[OrderResponse])
async def list_orders(symbol: Optional[str] = None, status: Optional[str] = None,
                      exchange: Optional[str] = None, strategy_id: Optional[str] = None,
                      page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                      sort_by: str = Query("created_at", pattern="^(created_at|symbol|status)$"),
                      sort_order: str = Query("desc", pattern="^(asc|desc)$"),
                      user: TokenData = Depends(get_current_user),
                      db: DatabaseManager = Depends(get_db)):
    """List orders with filtering, sorting, and pagination."""
    user_id = _get_user_id(user)
    orders = await db.list_orders(
        user_id=user_id,
        symbol=symbol, status=status, exchange=exchange,
        strategy_id=strategy_id,
        limit=page_size, offset=(page - 1) * page_size,
        sort_by=sort_by, sort_order=sort_order,
    )
    return [
        OrderResponse(
            id=o.get("id", ""), symbol=o.get("symbol", ""),
            side=o.get("side", ""), order_type=o.get("order_type", ""),
            status=o.get("status", ""), quantity=float(o.get("quantity", 0)),
            price=float(o.get("price", 0)) if o.get("price") else None,
            filled_quantity=float(o.get("filled_quantity", 0)),
            average_fill_price=float(o.get("average_fill_price", 0)),
            commission=float(o.get("commission", 0)),
            exchange=o.get("exchange", ""), strategy_id=o.get("strategy_id"),
            created_at=o.get("created_at", datetime.utcnow()),
        )
        for o in orders
    ]


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str, user: TokenData = Depends(get_current_user),
                    db: DatabaseManager = Depends(get_db)):
    """Get order by ID."""
    order = await db.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderResponse(
        id=order.get("id", order_id), symbol=order.get("symbol", ""),
        side=order.get("side", ""), order_type=order.get("order_type", ""),
        status=order.get("status", ""), quantity=float(order.get("quantity", 0)),
        price=float(order.get("price", 0)) if order.get("price") else None,
        filled_quantity=float(order.get("filled_quantity", 0)),
        average_fill_price=float(order.get("average_fill_price", 0)),
        commission=float(order.get("commission", 0)),
        exchange=order.get("exchange", ""), strategy_id=order.get("strategy_id"),
        created_at=order.get("created_at", datetime.utcnow()),
    )


@router.delete("/{order_id}")
async def cancel_order(order_id: str, user: TokenData = Depends(get_current_user),
                       db: DatabaseManager = Depends(get_db)):
    """Cancel an order."""
    success = await db.update_order(order_id, {"status": "cancelled"})
    if not success:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"message": "Order cancelled", "order_id": order_id}


__all__ = ["router"]
