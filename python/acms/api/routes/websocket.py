"""WebSocket endpoint and ConnectionManager."""

import json
import asyncio
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections and broadcasting with Redis PubSub support."""

    def __init__(self, redis_client=None):
        self.active_connections: Dict[str, WebSocket] = {}
        self._subscriptions: Dict[str, set] = defaultdict(set)  # client_id -> channels
        self._redis = redis_client
        self._redis_task: Optional[asyncio.Task] = None
        self._pubsub = None

    async def start_redis_subscriber(self):
        """Start Redis PubSub listener for real-time data push."""
        if self._redis is None:
            logger.info("No Redis client; WebSocket will use local broadcast only")
            return
        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe("acms:tick", "acms:signal", "acms:risk", "acms:pnl")
            self._redis_task = asyncio.create_task(self._redis_listener())
            logger.info("Redis PubSub subscriber started")
        except Exception as e:
            logger.warning(f"Failed to start Redis PubSub subscriber: {e}")

    async def _redis_listener(self):
        """Listen for Redis PubSub messages and forward to WebSocket clients."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    try:
                        payload = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        payload = {"data": data}
                    # Map Redis channel to WS channel
                    ws_channel = channel.replace("acms:", "")
                    await self.broadcast_to_channel(ws_channel, payload)
        except asyncio.CancelledError:
            logger.debug("Redis listener task cancelled")
        except Exception as e:
            logger.warning(f"Redis listener error: {e}")

    async def stop_redis_subscriber(self):
        """Stop Redis PubSub listener."""
        if self._redis_task:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except asyncio.CancelledError:
                logger.debug("Redis subscriber task cancelled during stop")
        if self._pubsub:
            try:
                await self._pubsub.unsubscribe()
            except Exception as e:
                logger.warning("Error closing Redis pubsub: %s", e)

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self._subscriptions.pop(client_id, None)

    def subscribe(self, client_id: str, channels: List[str]):
        self._subscriptions[client_id].update(channels)

    async def broadcast_to_channel(self, channel: str, message: dict):
        """Broadcast message to all clients subscribed to a channel."""
        for client_id, channels in list(self._subscriptions.items()):
            if channel in channels and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json({
                        "channel": channel, **message,
                    })
                except Exception:
                    self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for client_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(client_id)
        for client_id in disconnected:
            self.disconnect(client_id)


ws_manager = ConnectionManager(redis_client=None)  # Redis client injected at startup if available


def set_redis_client(redis_client):
    """Set Redis client for WebSocket manager."""
    ws_manager._redis = redis_client


@router.websocket("/ws/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time data stream via WebSocket.

    Channels:
    - tick: Real-time trade data
    - book: Order book updates
    - signal: New trading signals
    - position: Position updates
    - risk: Risk alerts
    - pnl: P&L updates
    """
    import uuid
    client_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "subscribe":
                channels = msg.get("channels", [])
                ws_manager.subscribe(client_id, channels)
                await websocket.send_json({"type": "subscribed", "channels": channels})
            elif msg.get("type") == "unsubscribe":
                channels = msg.get("channels", [])
                for ch in channels:
                    ws_manager._subscriptions[client_id].discard(ch)
                await websocket.send_json({"type": "unsubscribed", "channels": channels})
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)


__all__ = [
    "router",
    "ConnectionManager",
    "ws_manager",
    "set_redis_client",
]
