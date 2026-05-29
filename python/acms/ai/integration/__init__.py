"""AI Integration Module - Kafka consumers, Redis caching, API routes, and orchestration.

Implements:
- AIKafkaConsumer: Kafka integration for AI training data consumption
- MarketDataConsumer: Real-time market data from Redpanda
- SignalConsumer: Trading signal consumption from signal engine
- TrainingDataBuffer: Time-based training data buffering
- ModelCache: Redis-backed model cache with TTL management
- FeatureCache: Precomputed feature caching for low-latency serving
- PredictionCache: Prediction result caching
- API routes for AI endpoints (predict, models, train, features, monitoring, explain)
- AIOrchestrator: Central coordination of all AI components
"""

from acms.ai.integration.kafka_consumer import (
    AIKafkaConsumer,
    MarketDataConsumer,
    SignalConsumer,
    TrainingDataBuffer,
)
from acms.ai.integration.redis_cache import (
    ModelCache,
    FeatureCache,
    PredictionCache,
    DistributedCacheCoordinator,
)
from acms.ai.integration.api_routes import create_ai_router
from acms.ai.integration.orchestrator import AIOrchestrator

__all__ = [
    "AIKafkaConsumer",
    "MarketDataConsumer",
    "SignalConsumer",
    "TrainingDataBuffer",
    "ModelCache",
    "FeatureCache",
    "PredictionCache",
    "DistributedCacheCoordinator",
    "create_ai_router",
    "AIOrchestrator",
]
