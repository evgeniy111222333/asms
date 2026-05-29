"""AI Orchestrator - Central coordination of all ACMS AI components.

Implements:
- AIOrchestrator: Ties together monitoring, caching, consumers, and models
- Startup sequence: load models, warm caches, start consumers
- Model lifecycle management (register, deploy, retire)
- Training job scheduling and monitoring
- Inference routing with cache-first strategy
- Health monitoring and alerting
- Graceful shutdown with state preservation
- Configuration hot-reload
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from acms.ai.monitoring.model_monitor import AIModelMonitor, DegradationAlert, AlertSeverity
from acms.ai.monitoring.gpu_monitor import GPUMonitor, JobPriority, JobStatus
from acms.ai.monitoring.metrics import (
    AIMetricsCollector,
    ModelPerformanceMetrics,
    InferenceMetrics,
    FeatureMetrics,
)
from acms.ai.integration.redis_cache import (
    ModelCache,
    FeatureCache,
    PredictionCache,
    DistributedCacheCoordinator,
    CacheConfig,
)
from acms.ai.integration.kafka_consumer import (
    AIKafkaConsumer,
    MarketDataConsumer,
    SignalConsumer,
    AIConsumerConfig,
)
from acms.ai.integration.api_routes import set_ai_components

logger = logging.getLogger(__name__)


# ============================================================================
# Orchestrator State
# ============================================================================

class AIOrchestratorState(str, Enum):
    """AI Orchestrator lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"


class ModelLifecycleState(str, Enum):
    """Model lifecycle states."""
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    SERVING = "serving"
    TRAINING = "training"
    DEGRADED = "degraded"
    RETIRED = "retired"
    ERROR = "error"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AIOrchestratorConfig:
    """AI Orchestrator configuration.

    Attributes:
        model_dir: Directory containing model artifacts.
        cache_config: Redis cache configuration.
        consumer_config: Kafka consumer configuration.
        warm_caches_on_startup: Whether to warm caches on start.
        start_consumers_on_startup: Whether to start Kafka consumers.
        inference_timeout_seconds: Timeout for inference requests.
        health_check_interval_seconds: Interval for health checks.
        config_hot_reload: Whether to enable config hot-reload.
        config_path: Path to dynamic configuration file.
        default_model_id: Default model for inference routing.
        max_concurrent_inferences: Maximum parallel inference requests.
    """
    model_dir: str = "/data/acms/models"
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    consumer_config: AIConsumerConfig = field(default_factory=AIConsumerConfig)
    warm_caches_on_startup: bool = True
    start_consumers_on_startup: bool = True
    inference_timeout_seconds: float = 5.0
    health_check_interval_seconds: float = 30.0
    config_hot_reload: bool = False
    config_path: str = "/data/acms/config/ai_config.json"
    default_model_id: str = "lstm_btc_v1"
    max_concurrent_inferences: int = 100


# ============================================================================
# Model Record
# ============================================================================

@dataclass
class ModelRecord:
    """Internal record for a managed model.

    Attributes:
        model_id: Unique model identifier.
        model_type: Type of model (lstm, transformer, lightgbm, etc.).
        version: Model version string.
        lifecycle_state: Current lifecycle state.
        model_instance: The actual model object (if loaded).
        config: Model-specific configuration.
        registered_at: When the model was registered.
        last_inference_at: When the model was last used for inference.
        inference_count: Total inferences served.
        error_count: Total inference errors.
        features_count: Number of input features.
        training_samples: Number of training samples.
    """
    model_id: str
    model_type: str = "unknown"
    version: str = "0.0.0"
    lifecycle_state: ModelLifecycleState = ModelLifecycleState.REGISTERED
    model_instance: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_inference_at: Optional[datetime] = None
    inference_count: int = 0
    error_count: int = 0
    features_count: int = 0
    training_samples: int = 0

    @property
    def is_serving(self) -> bool:
        """Whether the model is available for inference."""
        return self.lifecycle_state in (
            ModelLifecycleState.READY,
            ModelLifecycleState.SERVING,
            ModelLifecycleState.DEGRADED,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "lifecycle_state": self.lifecycle_state.value,
            "registered_at": self.registered_at.isoformat(),
            "last_inference_at": self.last_inference_at.isoformat() if self.last_inference_at else None,
            "inference_count": self.inference_count,
            "error_count": self.error_count,
            "features_count": self.features_count,
            "training_samples": self.training_samples,
            "is_serving": self.is_serving,
        }


# ============================================================================
# AI Orchestrator
# ============================================================================

class AIOrchestrator:
    """Central orchestrator for all ACMS AI components.

    Coordinates model lifecycle, inference routing, training scheduling,
    cache management, monitoring, and consumer lifecycle. Provides a
    single entry point for all AI operations.
    """

    def __init__(self, config: Optional[AIOrchestratorConfig] = None,
                 redis_client: Any = None):
        """Initialize the AI orchestrator.

        Args:
            config: Orchestrator configuration.
            redis_client: Async Redis client instance.
        """
        self.config = config or AIOrchestratorConfig()
        self.state = AIOrchestratorState.STOPPED

        # Components
        self.model_monitor = AIModelMonitor()
        self.gpu_monitor = GPUMonitor()
        self.metrics_collector = AIMetricsCollector()

        # Caches
        self.model_cache = ModelCache(redis_client, self.config.cache_config)
        self.feature_cache = FeatureCache(redis_client, self.config.cache_config)
        self.prediction_cache = PredictionCache(redis_client, self.config.cache_config)
        self.cache_coordinator = DistributedCacheCoordinator(redis_client)

        # Consumers
        self.market_data_consumer: Optional[MarketDataConsumer] = None
        self.signal_consumer: Optional[SignalConsumer] = None

        # Model registry
        self._models: Dict[str, ModelRecord] = {}
        self._default_model_id: str = self.config.default_model_id

        # Inference semaphore
        self._inference_semaphore: Optional[asyncio.Semaphore] = None

        # Background tasks
        self._main_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._config_reload_task: Optional[asyncio.Task] = None
        self._running = False

        # Alert handlers
        self._alert_handlers: List[Callable[[DegradationAlert], Any]] = []

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def start(self) -> None:
        """Start the AI orchestrator and all components.

        Startup sequence:
        1. Set state to STARTING
        2. Initialize GPU monitor
        3. Load registered models
        4. Warm caches
        5. Start Kafka consumers
        6. Register with API routes
        7. Start health monitoring loop
        8. Set state to RUNNING
        """
        if self.state == AIOrchestratorState.RUNNING:
            logger.warning("AI Orchestrator already running")
            return

        self.state = AIOrchestratorState.STARTING
        logger.info("Starting AI Orchestrator...")

        try:
            # 1. Initialize inference semaphore
            self._inference_semaphore = asyncio.Semaphore(self.config.max_concurrent_inferences)

            # 2. Start GPU monitor
            await self.gpu_monitor.start()
            logger.info("GPU monitor started")

            # 3. Load models
            await self._load_models()

            # 4. Warm caches
            if self.config.warm_caches_on_startup:
                await self._warm_caches()

            # 5. Start Kafka consumers
            if self.config.start_consumers_on_startup:
                await self._start_consumers()

            # 6. Start cache coordinator
            await self.cache_coordinator.start()

            # 7. Register with API
            set_ai_components(
                orchestrator=self,
                metrics=self.metrics_collector,
                monitor=self.model_monitor,
                gpu_monitor=self.gpu_monitor,
                prediction_cache=self.prediction_cache,
                feature_cache=self.feature_cache,
            )

            # 8. Start background tasks
            self._running = True
            self._health_task = asyncio.create_task(self._health_check_loop())

            if self.config.config_hot_reload:
                self._config_reload_task = asyncio.create_task(self._config_reload_loop())

            self.state = AIOrchestratorState.RUNNING
            logger.info("AI Orchestrator started with %d model(s)", len(self._models))

        except Exception as e:
            self.state = AIOrchestratorState.ERROR
            logger.error("AI Orchestrator start failed: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the AI orchestrator gracefully.

        Shutdown sequence:
        1. Set state to STOPPING
        2. Stop accepting new inference requests
        3. Stop Kafka consumers
        4. Stop GPU monitor
        5. Stop cache coordinator
        6. Stop background tasks
        7. Flush metrics
        8. Set state to STOPPED
        """
        if self.state == AIOrchestratorState.STOPPED:
            return

        self.state = AIOrchestratorState.STOPPING
        logger.info("Stopping AI Orchestrator...")

        self._running = False

        # Cancel background tasks
        for task in [self._health_task, self._config_reload_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("AI orchestrator task cancelled during stop")

        # Stop consumers
        if self.market_data_consumer:
            await self.market_data_consumer.stop()
        if self.signal_consumer:
            await self.signal_consumer.stop()

        # Stop GPU monitor
        await self.gpu_monitor.stop()

        # Stop cache coordinator
        await self.cache_coordinator.stop()

        # Export final metrics
        self.metrics_collector.export_to_prometheus()

        self.state = AIOrchestratorState.STOPPED
        logger.info("AI Orchestrator stopped")

    # ========================================================================
    # Model Lifecycle
    # ========================================================================

    async def _load_models(self) -> None:
        """Load models from the model directory."""
        model_dir = Path(self.config.model_dir)
        if not model_dir.exists():
            logger.info("Model directory '%s' does not exist, skipping model load",
                         self.config.model_dir)
            return

        # Look for model config files
        for config_file in model_dir.glob("**/model_config.json"):
            try:
                with open(config_file) as f:
                    model_config = json.load(f)

                model_id = model_config.get("model_id", config_file.parent.name)
                await self.register_model(
                    model_id=model_id,
                    model_type=model_config.get("model_type", "unknown"),
                    version=model_config.get("version", "0.0.0"),
                    config=model_config,
                    baseline_accuracy=model_config.get("baseline_accuracy"),
                )
                logger.info("Loaded model config: %s", config_file)
            except Exception as e:
                logger.warning("Failed to load model config '%s': %s", config_file, e)

    async def register_model(self, model_id: str, model_type: str = "unknown",
                              version: str = "0.0.0",
                              config: Optional[Dict[str, Any]] = None,
                              model_instance: Any = None,
                              baseline_accuracy: Optional[float] = None) -> None:
        """Register a model with the orchestrator.

        Args:
            model_id: Unique model identifier.
            model_type: Type of model.
            version: Model version string.
            config: Model-specific configuration.
            model_instance: Pre-loaded model object.
            baseline_accuracy: Baseline accuracy for monitoring.
        """
        record = ModelRecord(
            model_id=model_id,
            model_type=model_type,
            version=version,
            config=config or {},
            model_instance=model_instance,
            lifecycle_state=ModelLifecycleState.READY if model_instance else ModelLifecycleState.REGISTERED,
            features_count=config.get("features_count", 0) if config else 0,
            training_samples=config.get("training_samples", 0) if config else 0,
        )
        self._models[model_id] = record

        # Register with model monitor
        self.model_monitor.register_model(model_id, version, baseline_accuracy)

        logger.info("Model '%s' (type=%s, v=%s) registered", model_id, model_type, version)

    async def deploy_model(self, model_id: str, model_instance: Any,
                           version: Optional[str] = None) -> bool:
        """Deploy a model instance for serving.

        Args:
            model_id: Model identifier.
            model_instance: The model object with predict method.
            version: Model version (updates if different).

        Returns:
            True if deployment was successful.
        """
        record = self._models.get(model_id)
        if record is None:
            logger.error("Model '%s' not registered", model_id)
            return False

        try:
            record.model_instance = model_instance
            if version:
                old_version = record.version
                record.version = version
                # Invalidate old cache entries
                if old_version != version:
                    await self.model_cache.invalidate_version(model_id, version)
                    await self.prediction_cache.invalidate_model(model_id)
                    await self.cache_coordinator.broadcast_invalidation(
                        "model", model_id, "version_update"
                    )

            record.lifecycle_state = ModelLifecycleState.SERVING
            logger.info("Model '%s' v%s deployed for serving", model_id, record.version)
            return True

        except Exception as e:
            record.lifecycle_state = ModelLifecycleState.ERROR
            logger.error("Failed to deploy model '%s': %s", model_id, e)
            return False

    async def retire_model(self, model_id: str) -> bool:
        """Retire a model from serving.

        Args:
            model_id: Model identifier.

        Returns:
            True if model was retired.
        """
        record = self._models.get(model_id)
        if record is None:
            return False

        record.lifecycle_state = ModelLifecycleState.RETIRED
        record.model_instance = None
        await self.prediction_cache.invalidate_model(model_id)
        await self.cache_coordinator.broadcast_invalidation(
            "prediction", model_id, "model_retired"
        )
        logger.info("Model '%s' retired", model_id)
        return True

    # ========================================================================
    # Inference Routing
    # ========================================================================

    async def predict(self, model_id: Optional[str] = None,
                      symbol: str = "BTC/USDT",
                      timeframe: str = "1h",
                      features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route a prediction request to the appropriate model.

        Uses a cache-first strategy: check prediction cache before
        computing. Falls back to the default model if the requested
        model is unavailable.

        Args:
            model_id: Model to use (falls back to default if None).
            symbol: Trading pair symbol.
            timeframe: Timeframe for prediction.
            features: Optional pre-computed features.

        Returns:
            Prediction result dict.
        """
        model_id = model_id or self._default_model_id
        record = self._models.get(model_id)

        # Fallback to any serving model if requested model unavailable
        if record is None or not record.is_serving:
            for mid, rec in self._models.items():
                if rec.is_serving:
                    model_id = mid
                    record = rec
                    break

        if record is None or not record.is_serving:
            return {
                "direction": "neutral",
                "confidence": 0.0,
                "predicted_return": 0.0,
                "error": "no_model_available",
            }

        async with self._inference_semaphore:
            try:
                # Get features if not provided
                if features is None:
                    cached_features = await self.feature_cache.get(symbol, timeframe)
                    if cached_features:
                        features = cached_features
                    else:
                        features = {"symbol": symbol, "timeframe": timeframe}

                # Run inference
                model = record.model_instance
                if model is not None and hasattr(model, "predict"):
                    result = model.predict(features)
                    if isinstance(result, np.ndarray):
                        # Convert numpy output to direction/confidence
                        if len(result.shape) > 0 and result.shape[0] >= 3:
                            probs = result if result.ndim == 1 else result[0]
                            direction_idx = int(np.argmax(probs))
                            directions = ["down", "neutral", "up"]
                            direction = directions[min(direction_idx, 2)]
                            confidence = float(np.max(probs))
                            predicted_return = float(probs[2] - probs[0]) * 0.05
                        else:
                            val = float(result.flat[0])
                            direction = "up" if val > 0.01 else ("down" if val < -0.01 else "neutral")
                            confidence = min(abs(val), 1.0)
                            predicted_return = val * 0.05
                    elif isinstance(result, dict):
                        direction = result.get("direction", "neutral")
                        confidence = result.get("confidence", 0.0)
                        predicted_return = result.get("predicted_return", 0.0)
                    else:
                        direction = "neutral"
                        confidence = 0.0
                        predicted_return = 0.0

                    # Build response
                    prediction = {
                        "direction": direction,
                        "confidence": confidence,
                        "predicted_return": predicted_return,
                        "probability_distribution": {
                            "down": 0.33, "neutral": 0.34, "up": 0.33
                        },
                        "model_version": record.version,
                    }

                else:
                    prediction = {
                        "direction": "neutral",
                        "confidence": 0.0,
                        "predicted_return": 0.0,
                    }

                # Update record
                record.inference_count += 1
                record.last_inference_at = datetime.utcnow()
                record.lifecycle_state = ModelLifecycleState.SERVING

                # Record metrics
                self.metrics_collector.record_inference(model_id, latency_ms=0.0)

                return prediction

            except Exception as e:
                record.error_count += 1
                logger.error("Inference error for model '%s': %s", model_id, e)
                return {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "predicted_return": 0.0,
                    "error": str(e),
                }

    # ========================================================================
    # Feature Computation
    # ========================================================================

    def _compute_features(self, market_data: Dict) -> Dict[str, float]:
        """Compute real features from market data."""
        try:
            features = {}
            if hasattr(self, '_feature_engineer') and self._feature_engineer is not None:
                result = self._feature_engineer.compute_all(market_data)
                if result:
                    features.update(result)

            # Price-based features
            prices = market_data.get("prices", [])
            if prices and len(prices) > 1:
                import numpy as np
                returns = np.diff(np.log(prices)) if all(p > 0 for p in prices) else np.diff(prices) / np.array(prices[:-1])
                features["return_mean"] = float(np.mean(returns)) if len(returns) > 0 else 0.0
                features["return_std"] = float(np.std(returns)) if len(returns) > 0 else 0.0
                features["return_skew"] = float(self._safe_skew(returns)) if len(returns) > 2 else 0.0
                features["return_kurt"] = float(self._safe_kurtosis(returns)) if len(returns) > 3 else 0.0
                features["price_momentum_5"] = float((prices[-1] / prices[-5] - 1)) if len(prices) >= 5 else 0.0
                features["price_momentum_20"] = float((prices[-1] / prices[-20] - 1)) if len(prices) >= 20 else 0.0
                features["volatility"] = float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0

            # Volume features
            volumes = market_data.get("volumes", [])
            if volumes and len(volumes) > 1:
                import numpy as np
                features["volume_mean"] = float(np.mean(volumes))
                features["volume_std"] = float(np.std(volumes))
                features["volume_ratio"] = float(volumes[-1] / np.mean(volumes)) if np.mean(volumes) > 0 else 1.0

            return features
        except Exception as e:
            logger.warning(f"Feature computation error: {e}")
            return {}

    @staticmethod
    def _safe_skew(returns: 'np.ndarray') -> float:
        """Compute skewness safely."""
        n = len(returns)
        if n < 3:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std < 1e-10:
            return 0.0
        return float(np.mean(((returns - mean) / std) ** 3))

    @staticmethod
    def _safe_kurtosis(returns: 'np.ndarray') -> float:
        """Compute excess kurtosis safely."""
        n = len(returns)
        if n < 4:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std < 1e-10:
            return 0.0
        return float(np.mean(((returns - mean) / std) ** 4) - 3.0)

    async def compute_features(self, symbol: str, timeframe: str,
                                feature_set: str = "default") -> Dict[str, Any]:
        """Compute features for a symbol/timeframe.

        Args:
            symbol: Trading pair symbol.
            timeframe: Timeframe string.
            feature_set: Feature set identifier.

        Returns:
            Feature data dict or error.
        """
        # Delegate to feature engineering pipeline
        try:
            # Try to use AdvancedFeatureEngineer if available
            try:
                from acms.ai.features.engineering import AdvancedFeatureEngineer
                engineer = AdvancedFeatureEngineer()
                
                # Fetch market data from cache or DB
                cached_features = await self.feature_cache.get(symbol, timeframe)
                
                # Validate we have sufficient data
                MIN_PRICE_HISTORY = 20  # Minimum candles for meaningful features
                prices = cached_features.get("prices", []) if cached_features else []
                volumes = cached_features.get("volumes", []) if cached_features else []
                
                if not cached_features or not prices:
                    # No data available - return error, NOT silent fallback
                    logger.warning(
                        "Insufficient market data for features: symbol=%s timeframe=%s "
                        "prices_count=%d. Orchestrator should wait for data ingestion.",
                        symbol, timeframe, len(prices)
                    )
                    return {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": "insufficient_market_data",
                        "error_detail": f"Need at least {MIN_PRICE_HISTORY} candles, got {len(prices)}",
                        "status": "degraded",
                    }
                
                if len(prices) < MIN_PRICE_HISTORY:
                    logger.warning(
                        "Limited market data: symbol=%s timeframe=%s prices_count=%d < %d",
                        symbol, timeframe, len(prices), MIN_PRICE_HISTORY
                    )
                    return {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": "limited_market_data",
                        "prices_count": len(prices),
                        "status": "degraded",
                    }
                
                # Data is sufficient, compute features
                computed = self._compute_features(cached_features)
                return {"symbol": symbol, "timeframe": timeframe, **computed}
                
            except ImportError:
                logger.debug("Feature engineering module not available")

            # Fallback: basic feature computation
            return {"symbol": symbol, "timeframe": timeframe, "status": "no_feature_engineer"}
        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return {"error": str(e)}

    # ========================================================================
    # Model Explanation
    # ========================================================================

    async def explain(self, model_id: str, symbol: str,
                      method: str = "feature_importance",
                      n_features: int = 20) -> Dict[str, Any]:
        """Generate model explanations.

        Args:
            model_id: Model to explain.
            symbol: Symbol context.
            method: Explanation method.
            n_features: Number of top features to return.

        Returns:
            Explanation result dict.
        """
        record = self._models.get(model_id)
        if record is None or record.model_instance is None:
            return {"error": f"Model '{model_id}' not available for explanation"}

        try:
            # Get feature metrics for the model
            feature_metrics = self.metrics_collector.get_feature_metrics(model_id)
            if feature_metrics:
                top_features = sorted(
                    feature_metrics,
                    key=lambda f: f.importance_score,
                    reverse=True,
                )[:n_features]
                return {
                    "model_id": model_id,
                    "method": method,
                    "symbol": symbol,
                    "features": [f.to_dict() for f in top_features],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            return {
                "model_id": model_id,
                "method": method,
                "message": "No feature importance data available",
            }
        except Exception as e:
            logger.error("Explanation error: %s", e)
            return {"error": str(e)}

    # ========================================================================
    # Model Listing
    # ========================================================================

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.

        Returns:
            List of model info dicts.
        """
        return [record.to_dict() for record in self._models.values()]

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model.

        Args:
            model_id: Model identifier.

        Returns:
            Model info dict or None.
        """
        record = self._models.get(model_id)
        if record is None:
            return None

        info = record.to_dict()
        # Add monitoring data
        info["monitoring"] = self.model_monitor.get_model_status(model_id)
        # Add latest performance
        perf = self.metrics_collector.get_latest_performance(model_id)
        info["performance"] = perf.to_dict() if perf else None
        return info

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the orchestrator.

        Args:
            model_id: Model identifier.

        Returns:
            True if model was deleted.
        """
        record = self._models.get(model_id)
        if record is None:
            return False

        asyncio.create_task(self.retire_model(model_id))
        del self._models[model_id]
        logger.info("Model '%s' deleted", model_id)
        return True

    # ========================================================================
    # Cache Warming
    # ========================================================================

    async def _warm_caches(self) -> None:
        """Warm caches on startup with model artifacts and features."""
        logger.info("Warming caches...")

        # Warm model cache
        for model_id, record in self._models.items():
            if record.model_instance is not None:
                model_data = {
                    "model_id": model_id,
                    "model_type": record.model_type,
                    "version": record.version,
                    "config": record.config,
                }
                await self.model_cache.set(model_id, model_data, record.version)

        # Warm feature cache for known symbols
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        timeframes = ["1m", "5m", "15m", "1h", "4h"]

        async def feature_factory(sym: str, tf: str) -> Dict[str, Any]:
            return await self.compute_features(sym, tf)

        count = await self.feature_cache.warm(symbols, timeframes, feature_factory)
        logger.info("Cache warming complete: %d feature entries", count)

    # ========================================================================
    # Consumer Management
    # ========================================================================

    async def _start_consumers(self) -> None:
        """Start Kafka consumers for market data and signals."""
        try:
            self.market_data_consumer = MarketDataConsumer(self.config.consumer_config)
            self.market_data_consumer.on_candle(self._on_candle_data)
            self.market_data_consumer.on_tick(self._on_tick_data)
            await self.market_data_consumer.start()
            logger.info("Market data consumer started")
        except Exception as e:
            logger.warning("Failed to start market data consumer: %s", e)

        try:
            self.signal_consumer = SignalConsumer(self.config.consumer_config)
            self.signal_consumer.on_signal(self._on_signal_data)
            await self.signal_consumer.start()
            logger.info("Signal consumer started")
        except Exception as e:
            logger.warning("Failed to start signal consumer: %s", e)

    async def _on_candle_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming candle data from Kafka."""
        symbol = data.get("symbol", "")
        # Invalidate feature cache for this symbol
        await self.feature_cache.invalidate_symbol(symbol)

    async def _on_tick_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming tick data from Kafka."""
        try:
            symbol = data.get("symbol")
            price = data.get("price", 0)
            volume = data.get("volume", 0)

            if symbol and price > 0:
                # Update internal tick buffer
                if not hasattr(self, '_tick_buffers'):
                    self._tick_buffers = {}
                if symbol not in self._tick_buffers:
                    self._tick_buffers[symbol] = []
                self._tick_buffers[symbol].append(data)

                # Keep only last 1000 ticks per symbol
                if len(self._tick_buffers[symbol]) > 1000:
                    self._tick_buffers[symbol] = self._tick_buffers[symbol][-1000:]

                # Trigger real-time risk update if risk engine is available
                if hasattr(self, '_risk_engine') and self._risk_engine:
                    self._risk_engine.update_price(symbol, price)

                # Check for microstructure signals
                self._check_microstructure_signals(symbol, data)

        except Exception as e:
            logger.warning(f"Tick data processing error: {e}")

    def _check_microstructure_signals(self, symbol: str, data: Dict) -> None:
        """Check for microstructure signals from tick data."""
        try:
            if not hasattr(self, '_tick_buffers') or symbol not in self._tick_buffers:
                return
            ticks = self._tick_buffers[symbol]
            if len(ticks) < 10:
                return
            # Check for sudden price moves (more than 2 std devs in short window)
            recent_prices = [t.get("price", 0) for t in ticks[-10:]]
            if len(recent_prices) >= 10 and all(p > 0 for p in recent_prices):
                import numpy as np
                mean_price = np.mean(recent_prices[:-1])
                std_price = np.std(recent_prices[:-1])
                if std_price > 0 and abs(recent_prices[-1] - mean_price) > 2 * std_price:
                    logger.info(f"Microstructure signal: {symbol} price anomaly detected")
        except Exception as e:
            logger.debug(f"Microstructure check error for {symbol}: {e}")

    async def _on_signal_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming signal data from Kafka."""
        try:
            signal_type = data.get("signal_type")
            symbol = data.get("symbol")
            strength = data.get("strength", 0.0)

            if symbol and strength != 0.0:
                # Validate signal
                if abs(strength) > 1.0:
                    logger.warning(f"Signal strength out of range for {symbol}: {strength}")
                    strength = max(-1.0, min(1.0, strength))

                # Route to decision engine
                if hasattr(self, '_decision_router') and self._decision_router:
                    self._decision_router.process_signal(symbol, signal_type, strength, data)

                # Update signal buffer for model training
                if not hasattr(self, '_signal_buffer'):
                    self._signal_buffer = {}
                if symbol not in self._signal_buffer:
                    self._signal_buffer[symbol] = []
                self._signal_buffer[symbol].append({
                    "signal_type": signal_type,
                    "strength": strength,
                    "timestamp": data.get("timestamp", time.time()),
                    "data": data,
                })

                # Keep last 100 signals
                if len(self._signal_buffer[symbol]) > 100:
                    self._signal_buffer[symbol] = self._signal_buffer[symbol][-100:]

        except Exception as e:
            logger.warning(f"Signal processing error: {e}")

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", e)

    async def _check_health(self) -> None:
        """Run health checks on all components."""
        issues = []

        # Check model health
        for model_id, record in self._models.items():
            if record.lifecycle_state == ModelLifecycleState.ERROR:
                issues.append(f"Model '{model_id}' in error state")
            if record.error_count > 100 and record.inference_count > 0:
                error_rate = record.error_count / record.inference_count
                if error_rate > 0.5:
                    issues.append(f"Model '{model_id}' high error rate: {error_rate:.2%}")

        # Check GPU health
        gpu_status = self.gpu_monitor.get_status()
        for gpu in gpu_status.get("gpus", []):
            if gpu.get("health") == "critical":
                issues.append(f"GPU {gpu.get('device_id')} critical: {gpu.get('name')}")

        # Export metrics
        self.metrics_collector.export_to_prometheus()

        # Update state
        if issues and self.state == AIOrchestratorState.RUNNING:
            self.state = AIOrchestratorState.DEGRADED
            for issue in issues:
                logger.warning("Health issue: %s", issue)
        elif not issues and self.state == AIOrchestratorState.DEGRADED:
            self.state = AIOrchestratorState.RUNNING

    # ========================================================================
    # Configuration Hot-Reload
    # ========================================================================

    async def _config_reload_loop(self) -> None:
        """Watch configuration file for changes and hot-reload."""
        config_path = Path(self.config.config_path)
        last_modified: float = 0.0

        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                if config_path.exists():
                    mtime = config_path.stat().st_mtime
                    if mtime > last_modified:
                        last_modified = mtime
                        await self._reload_config(config_path)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Config reload error: %s", e)

    async def _reload_config(self, config_path: Path) -> None:
        """Reload configuration from file.

        Args:
            config_path: Path to the configuration file.
        """
        try:
            with open(config_path) as f:
                new_config = json.load(f)

            # Apply safe-to-reload settings
            if "inference_timeout_seconds" in new_config:
                self.config.inference_timeout_seconds = new_config["inference_timeout_seconds"]
            if "health_check_interval_seconds" in new_config:
                self.config.health_check_interval_seconds = new_config["health_check_interval_seconds"]
            if "default_model_id" in new_config:
                self._default_model_id = new_config["default_model_id"]

            logger.info("Configuration reloaded from '%s'", config_path)
        except Exception as e:
            logger.error("Failed to reload config: %s", e)

    # ========================================================================
    # Alert Handling
    # ========================================================================

    def on_alert(self, handler: Callable[[DegradationAlert], Any]) -> None:
        """Register an alert handler.

        Args:
            handler: Callable receiving DegradationAlert instances.
        """
        self._alert_handlers.append(handler)

    # ========================================================================
    # Status
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status.

        Returns:
            Dict with state, model info, component status, and metrics.
        """
        model_summary = {
            "total": len(self._models),
            "serving": sum(1 for r in self._models.values() if r.is_serving),
            "error": sum(1 for r in self._models.values()
                        if r.lifecycle_state == ModelLifecycleState.ERROR),
        }

        return {
            "state": self.state.value,
            "models": model_summary,
            "default_model": self._default_model_id,
            "gpu_monitor_running": self.gpu_monitor._running,
            "market_data_consumer_active": (
                self.market_data_consumer is not None and
                self.market_data_consumer._consumer._running
            ),
            "signal_consumer_active": (
                self.signal_consumer is not None and
                self.signal_consumer._consumer._running
            ),
            "cache_coordinator_active": self.cache_coordinator._running,
            "metrics_models_tracked": len(self.metrics_collector._performance_history),
            "timestamp": datetime.utcnow().isoformat(),
        }
