"""FastAPI Routes for ACMS AI Endpoints.

Implements:
- /api/v1/ai/predict - Get predictions from AI models
- /api/v1/ai/models - List and manage AI models
- /api/v1/ai/train - Trigger model training
- /api/v1/ai/features - Feature store API
- /api/v1/ai/monitoring - AI metrics and health
- /api/v1/ai/explain - Model explanations (SHAP/feature importance)
- WebSocket endpoint for real-time AI predictions
- Auth integration with existing AuthManager
- Rate limiting for AI endpoints
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from acms.auth import AuthManager, TokenData

logger = logging.getLogger(__name__)


# ============================================================================
# Router Setup
# ============================================================================

security = HTTPBearer()
auth_manager = AuthManager()

# Global references set by the orchestrator
_ai_orchestrator: Optional[Any] = None
_metrics_collector: Optional[Any] = None
_model_monitor: Optional[Any] = None
_gpu_monitor: Optional[Any] = None
_prediction_cache: Optional[Any] = None
_feature_cache: Optional[Any] = None


def set_ai_components(orchestrator=None, metrics=None, monitor=None,
                      gpu_monitor=None, prediction_cache=None, feature_cache=None):
    """Set AI component references for API integration."""
    global _ai_orchestrator, _metrics_collector, _model_monitor
    global _gpu_monitor, _prediction_cache, _feature_cache
    _ai_orchestrator = orchestrator
    _metrics_collector = metrics
    _model_monitor = monitor
    _gpu_monitor = gpu_monitor
    _prediction_cache = prediction_cache
    _feature_cache = feature_cache


# ============================================================================
# Rate Limiting
# ============================================================================

class AIRateLimiter:
    """Rate limiter for AI endpoints with per-user and per-endpoint limits."""

    def __init__(self, default_limit: int = 60, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str, limit: Optional[int] = None) -> bool:
        """Check if request is within rate limit."""
        max_requests = limit or self.default_limit
        now = time.time()
        self._requests[key] = [
            t for t in self._requests[key] if now - t < self.window_seconds
        ]
        if len(self._requests[key]) >= max_requests:
            return False
        self._requests[key].append(now)
        return True


ai_rate_limiter = AIRateLimiter(default_limit=60, window_seconds=60)


# ============================================================================
# Auth Dependencies
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Validate JWT token and return user data."""
    token = credentials.credentials
    data = auth_manager.verify_token(token)
    if data is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return data


async def check_ai_rate_limit(request: Any = None, user: TokenData = Depends(get_current_user)) -> None:
    """Check rate limit for AI endpoints."""
    client_id = user.user_id if hasattr(user, 'user_id') else "anonymous"
    if not ai_rate_limiter.is_allowed(f"ai:{client_id}"):
        raise HTTPException(status_code=429, detail="AI endpoint rate limit exceeded")


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictRequest(BaseModel):
    """Request body for AI predictions."""
    model_id: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=50)
    timeframe: str = Field("1h", max_length=10)
    features: Optional[Dict[str, Any]] = None
    include_confidence: bool = True
    include_explanation: bool = False


class PredictResponse(BaseModel):
    """Response for AI predictions."""
    prediction_id: str
    model_id: str
    symbol: str
    direction: str  # "long", "short", "neutral"
    confidence: float
    predicted_return: float
    probability_distribution: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    cached: bool = False
    latency_ms: float
    timestamp: str


class ModelInfo(BaseModel):
    """AI model information."""
    model_id: str
    model_type: str
    version: str
    status: str
    accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    last_trained: Optional[str] = None
    features_count: int = 0
    training_samples: int = 0


class TrainRequest(BaseModel):
    """Request body for triggering model training."""
    model_id: str = Field(..., min_length=1, max_length=100)
    model_type: str = Field("lstm", pattern="^(lstm|transformer|lightgbm|ensemble|rl)$")
    symbol: str = Field(..., min_length=1, max_length=50)
    timeframe: str = Field("1h")
    config: Dict[str, Any] = {}
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$")
    force_retrain: bool = False


class TrainResponse(BaseModel):
    """Response for training trigger."""
    job_id: str
    model_id: str
    status: str
    queue_position: int
    estimated_duration_minutes: float
    message: str


class FeatureStoreRequest(BaseModel):
    """Request body for feature store."""
    symbol: str = Field(..., min_length=1, max_length=50)
    timeframe: str = Field("1h")
    feature_set: str = Field("default")
    refresh: bool = False


class FeatureStoreResponse(BaseModel):
    """Response for feature store."""
    symbol: str
    timeframe: str
    feature_set: str
    features: Dict[str, Any]
    feature_count: int
    computed_at: str
    cached: bool


class ExplainRequest(BaseModel):
    """Request body for model explanations."""
    model_id: str = Field(..., min_length=1, max_length=100)
    symbol: str = Field(..., min_length=1, max_length=50)
    method: str = Field("feature_importance", pattern="^(feature_importance|shap|permutation)$")
    n_features: int = Field(20, ge=1, le=100)


class MonitoringResponse(BaseModel):
    """Response for monitoring endpoints."""
    status: str
    timestamp: str
    data: Dict[str, Any]


# ============================================================================
# Router Definition
# ============================================================================

def create_ai_router() -> APIRouter:
    """Create and return the AI API router with all endpoints.

    Returns:
        FastAPI APIRouter with all AI endpoints.
    """
    router = APIRouter(prefix="/api/v1/ai", tags=["AI"])

    # ========================================================================
    # Prediction Endpoints
    # ========================================================================

    @router.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest,
                      user: TokenData = Depends(get_current_user),
                      _rate: None = Depends(check_ai_rate_limit)):
        """Get a prediction from an AI model.

        Supports caching for low-latency responses on repeated queries.
        Returns prediction direction, confidence, and optional explanations.
        """
        start_time = time.time()
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # Check prediction cache
        cached = False
        prediction = None
        if _prediction_cache is not None:
            try:
                cache_data = {
                    "model_id": request.model_id,
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                }
                prediction = await _prediction_cache.get(request.model_id, cache_data)
                if prediction is not None:
                    cached = True
            except Exception as e:
                logger.warning("Prediction cache read error: %s", e)

        if prediction is None:
            # Generate prediction (via orchestrator)
            if _ai_orchestrator is not None:
                try:
                    prediction = await _ai_orchestrator.predict(
                        model_id=request.model_id,
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        features=request.features,
                    )
                except Exception as e:
                    logger.error("Prediction error for model '%s': %s", request.model_id, e)
                    raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
            else:
                # Fallback: return a neutral prediction
                prediction = {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "predicted_return": 0.0,
                    "probability_distribution": {"down": 0.33, "neutral": 0.34, "up": 0.33},
                }

            # Cache the prediction
            if _prediction_cache is not None and prediction is not None:
                try:
                    cache_data = {
                        "model_id": request.model_id,
                        "symbol": request.symbol,
                        "timeframe": request.timeframe,
                    }
                    await _prediction_cache.set(request.model_id, cache_data, prediction)
                except Exception as e:
                    logger.warning("Prediction cache write error: %s", e)

        latency_ms = (time.time() - start_time) * 1000

        # Record inference metric
        if _metrics_collector is not None:
            _metrics_collector.record_inference(
                model_id=request.model_id,
                latency_ms=latency_ms,
                cache_hit=cached,
            )

        explanation = None
        if request.include_explanation and prediction:
            explanation = prediction.get("explanation", {"note": "explanation not available"})

        return PredictResponse(
            prediction_id=prediction_id,
            model_id=request.model_id,
            symbol=request.symbol,
            direction=prediction.get("direction", "neutral"),
            confidence=prediction.get("confidence", 0.0),
            predicted_return=prediction.get("predicted_return", 0.0),
            probability_distribution=prediction.get("probability_distribution"),
            explanation=explanation,
            cached=cached,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow().isoformat(),
        )

    # ========================================================================
    # Model Management Endpoints
    # ========================================================================

    @router.get("/models", response_model=List[ModelInfo])
    async def list_models(status: Optional[str] = None,
                          model_type: Optional[str] = None,
                          user: TokenData = Depends(get_current_user),
                          _rate: None = Depends(check_ai_rate_limit)):
        """List all AI models with optional filtering."""
        if _ai_orchestrator is not None:
            try:
                models = _ai_orchestrator.list_models()
                if status:
                    models = [m for m in models if m.get("status") == status]
                if model_type:
                    models = [m for m in models if m.get("model_type") == model_type]
                return [
                    ModelInfo(
                        model_id=m.get("model_id", ""),
                        model_type=m.get("model_type", "unknown"),
                        version=m.get("version", "0.0.0"),
                        status=m.get("status", "unknown"),
                        accuracy=m.get("accuracy"),
                        sharpe_ratio=m.get("sharpe_ratio"),
                        last_trained=m.get("last_trained"),
                        features_count=m.get("features_count", 0),
                        training_samples=m.get("training_samples", 0),
                    )
                    for m in models
                ]
            except Exception as e:
                logger.error("List models error: %s", e)

        return []

    @router.get("/models/{model_id}")
    async def get_model(model_id: str, user: TokenData = Depends(get_current_user),
                        _rate: None = Depends(check_ai_rate_limit)):
        """Get detailed information about a specific model."""
        if _ai_orchestrator is not None:
            try:
                model_info = _ai_orchestrator.get_model_info(model_id)
                if model_info:
                    return model_info
            except Exception as e:
                logger.error("Get model error: %s", e)

        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    @router.delete("/models/{model_id}")
    async def delete_model(model_id: str, user: TokenData = Depends(get_current_user),
                           _rate: None = Depends(check_ai_rate_limit)):
        """Delete an AI model."""
        if _ai_orchestrator is not None:
            try:
                success = _ai_orchestrator.delete_model(model_id)
                if success:
                    return {"message": f"Model '{model_id}' deleted", "model_id": model_id}
            except Exception as e:
                logger.error("Delete model error: %s", e)

        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # ========================================================================
    # Training Endpoints
    # ========================================================================

    @router.post("/train", response_model=TrainResponse)
    async def trigger_training(request: TrainRequest,
                               user: TokenData = Depends(get_current_user),
                               _rate: None = Depends(check_ai_rate_limit)):
        """Trigger a model training job.

        Queues a training job with the specified configuration.
        Supports priority-based scheduling.
        """
        if _gpu_monitor is None:
            raise HTTPException(status_code=503, detail="GPU monitor not available")

        # Submit to training queue
        priority_map = {"low": "low", "normal": "normal", "high": "high", "urgent": "urgent"}
        from acms.ai.monitoring.gpu_monitor import JobPriority
        priority = JobPriority(priority_map.get(request.priority, "normal"))

        try:
            job_id = _gpu_monitor.job_queue.submit(
                model_id=request.model_id,
                config={
                    "model_type": request.model_type,
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    **request.config,
                },
                priority=priority,
            )

            queue_status = _gpu_monitor.job_queue.get_queue_status()
            queue_depth = queue_status.get("queue_depth", 0)

            return TrainResponse(
                job_id=job_id,
                model_id=request.model_id,
                status="queued",
                queue_position=queue_depth,
                estimated_duration_minutes=request.config.get("estimated_minutes", 30.0),
                message=f"Training job queued with priority '{request.priority}'",
            )
        except Exception as e:
            logger.error("Training submission error: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to submit training job: {str(e)}")

    @router.get("/train/{job_id}")
    async def get_training_status(job_id: str, user: TokenData = Depends(get_current_user)):
        """Get status of a training job."""
        if _gpu_monitor is None:
            raise HTTPException(status_code=503, detail="GPU monitor not available")

        job = _gpu_monitor.job_queue._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")

        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status.value,
            "priority": job.priority.value,
            "gpu_device_id": job.gpu_device_id,
            "progress_pct": job.progress_pct,
            "submitted_at": job.submitted_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "wait_time_seconds": job.wait_time_seconds,
            "runtime_seconds": job.runtime_seconds,
            "error_message": job.error_message,
        }

    @router.post("/train/{job_id}/cancel")
    async def cancel_training(job_id: str, user: TokenData = Depends(get_current_user)):
        """Cancel a training job."""
        if _gpu_monitor is None:
            raise HTTPException(status_code=503, detail="GPU monitor not available")

        success = _gpu_monitor.job_queue.cancel_job(job_id)
        if success:
            return {"message": f"Training job '{job_id}' cancelled", "job_id": job_id}
        raise HTTPException(status_code=404, detail="Training job not found or cannot be cancelled")

    # ========================================================================
    # Feature Store Endpoints
    # ========================================================================

    @router.post("/features", response_model=FeatureStoreResponse)
    async def get_features(request: FeatureStoreRequest,
                           user: TokenData = Depends(get_current_user),
                           _rate: None = Depends(check_ai_rate_limit)):
        """Get features from the feature store.

        Returns precomputed features for a symbol/timeframe,
        computing them on-demand if not cached.
        """
        cached = False
        features = None

        # Try cache first
        if _feature_cache is not None and not request.refresh:
            try:
                features = await _feature_cache.get(
                    request.symbol, request.timeframe, request.feature_set
                )
                if features is not None:
                    cached = True
            except Exception as e:
                logger.warning("Feature cache read error: %s", e)

        # Compute features if not cached
        if features is None:
            if _ai_orchestrator is not None:
                try:
                    features = await _ai_orchestrator.compute_features(
                        symbol=request.symbol,
                        timeframe=request.timeframe,
                        feature_set=request.feature_set,
                    )
                except Exception as e:
                    logger.error("Feature computation error: %s", e)
                    raise HTTPException(status_code=500, detail=f"Feature computation failed: {str(e)}")
            else:
                features = {"error": "Feature computation not available"}

            # Cache the computed features
            if _feature_cache is not None and features:
                try:
                    await _feature_cache.set(
                        request.symbol, request.timeframe, features, request.feature_set
                    )
                except Exception as e:
                    logger.warning("Feature cache write error: %s", e)

        feature_count = len(features) if isinstance(features, dict) else 0

        return FeatureStoreResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            feature_set=request.feature_set,
            features=features or {},
            feature_count=feature_count,
            computed_at=datetime.utcnow().isoformat(),
            cached=cached,
        )

    # ========================================================================
    # Monitoring Endpoints
    # ========================================================================

    @router.get("/monitoring")
    async def get_ai_monitoring(user: TokenData = Depends(get_current_user)):
        """Get comprehensive AI monitoring data."""
        result: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "active",
        }

        if _metrics_collector is not None:
            result["metrics"] = _metrics_collector.get_summary()

        if _model_monitor is not None:
            result["model_health"] = _model_monitor.get_model_status(
                list(_model_monitor._model_versions.keys())[0]
            ) if _model_monitor._model_versions else {}

        if _gpu_monitor is not None:
            result["gpu"] = _gpu_monitor.get_status()

        return result

    @router.get("/monitoring/models/{model_id}")
    async def get_model_monitoring(model_id: str,
                                    user: TokenData = Depends(get_current_user)):
        """Get monitoring data for a specific model."""
        if _model_monitor is not None:
            try:
                status = _model_monitor.get_model_status(model_id)
                return status
            except Exception as e:
                logger.error("Model monitoring error: %s", e)

        raise HTTPException(status_code=404, detail=f"Monitoring data for model '{model_id}' not found")

    @router.get("/monitoring/gpu")
    async def get_gpu_monitoring(user: TokenData = Depends(get_current_user)):
        """Get GPU monitoring data."""
        if _gpu_monitor is not None:
            return _gpu_monitor.get_status()
        return {"status": "gpu_monitoring_not_available"}

    @router.get("/monitoring/metrics")
    async def get_ai_metrics(model_id: Optional[str] = None,
                              window: str = Query("5m", pattern="^(1m|5m|15m|1h)$"),
                              user: TokenData = Depends(get_current_user)):
        """Get AI metrics with configurable aggregation window."""
        if _metrics_collector is not None:
            if model_id:
                inf = _metrics_collector.get_inference_metrics(model_id, window)
                train = _metrics_collector.get_training_metrics(model_id, window)
                perf = _metrics_collector.get_latest_performance(model_id)
                return {
                    "model_id": model_id,
                    "window": window,
                    "inference": inf.to_dict(),
                    "training": train.to_dict(),
                    "performance": perf.to_dict() if perf else None,
                }
            return _metrics_collector.get_summary()
        return {"status": "metrics_not_available"}

    # ========================================================================
    # Explanation Endpoints
    # ========================================================================

    @router.post("/explain")
    async def explain_prediction(request: ExplainRequest,
                                  user: TokenData = Depends(get_current_user),
                                  _rate: None = Depends(check_ai_rate_limit)):
        """Get model explanations (feature importance, SHAP values)."""
        if _ai_orchestrator is not None:
            try:
                explanation = await _ai_orchestrator.explain(
                    model_id=request.model_id,
                    symbol=request.symbol,
                    method=request.method,
                    n_features=request.n_features,
                )
                return explanation or {"message": "Explanation not available"}
            except Exception as e:
                logger.error("Explanation error: %s", e)
                raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

        return {"message": "AI orchestrator not available", "method": request.method}

    # ========================================================================
    # WebSocket for Real-time AI Predictions
    # ========================================================================

    class AIConnectionManager:
        """Manages WebSocket connections for real-time AI predictions."""

        def __init__(self):
            self.active_connections: Dict[str, WebSocket] = {}
            self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

        async def connect(self, websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.active_connections[client_id] = websocket

        def disconnect(self, client_id: str):
            self.active_connections.pop(client_id, None)
            self._subscriptions.pop(client_id, None)

        def subscribe(self, client_id: str, model_ids: List[str]):
            self._subscriptions[client_id].update(model_ids)

        async def broadcast_prediction(self, model_id: str, prediction: Dict):
            """Broadcast a prediction to subscribed clients."""
            for client_id, model_ids in self._subscriptions.items():
                if model_id in model_ids and client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_json({
                            "type": "prediction",
                            "model_id": model_id,
                            **prediction,
                        })
                    except Exception:
                        self.disconnect(client_id)

    ai_ws_manager = AIConnectionManager()

    @router.websocket("/ws/predictions")
    async def ai_predictions_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time AI predictions.

        Subscribe to specific model IDs to receive predictions as they
        are generated. Messages:
        - subscribe: {"type": "subscribe", "model_ids": ["lstm_btc", "transformer_eth"]}
        - unsubscribe: {"type": "unsubscribe", "model_ids": ["lstm_btc"]}
        """
        client_id = str(uuid.uuid4())
        await ai_ws_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "subscribe":
                    model_ids = msg.get("model_ids", [])
                    ai_ws_manager.subscribe(client_id, model_ids)
                    await websocket.send_json({
                        "type": "subscribed",
                        "model_ids": model_ids,
                    })
                elif msg.get("type") == "unsubscribe":
                    model_ids = msg.get("model_ids", [])
                    for mid in model_ids:
                        ai_ws_manager._subscriptions[client_id].discard(mid)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "model_ids": model_ids,
                    })
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ai_ws_manager.disconnect(client_id)
        except Exception as e:
            logger.error("AI WebSocket error: %s", e)
            ai_ws_manager.disconnect(client_id)

    # ========================================================================
    # Health Check
    # ========================================================================

    @router.get("/health")
    async def ai_health():
        """AI service health check."""
        components = {
            "orchestrator": _ai_orchestrator is not None,
            "metrics_collector": _metrics_collector is not None,
            "model_monitor": _model_monitor is not None,
            "gpu_monitor": _gpu_monitor is not None,
            "prediction_cache": _prediction_cache is not None,
            "feature_cache": _feature_cache is not None,
        }
        all_healthy = all(components.values())
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components,
            "websocket_connections": len(ai_ws_manager.active_connections),
            "timestamp": datetime.utcnow().isoformat(),
        }

    return router
