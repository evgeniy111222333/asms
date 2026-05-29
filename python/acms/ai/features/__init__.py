"""
ACMS AI Features Module
=======================

Feature store, engineering, and drift detection for algorithmic crypto trading.

Submodules
----------
store : Feature store with Redis backend and real-time computation
engineering : Advanced feature engineering for crypto market data
drift : Feature and concept drift detection

Key Components
--------------
- FeatureStore: Redis-backed feature storage with versioning
- RealTimeFeatureComputer: On-demand feature computation
- AdvancedFeatureEngineer: Market microstructure, cross-asset, temporal features
- FeatureDriftMonitor: Multi-dimensional drift detection
"""

from acms.ai.features.store import (
    FeatureStore,
    RealTimeFeatureComputer,
    FeatureGroup,
    FeatureStatistics,
    FeatureFreshnessMonitor,
    FeatureVersion,
    FeatureQualityScorer,
    FeatureDependencyGraph,
)
from acms.ai.features.engineering import (
    AdvancedFeatureEngineer,
    MarketMicrostructureFeatures,
    CrossAssetFeatures,
    TemporalFeatures,
    RegimeFeatures,
    SentimentFeatures,
    OnChainFeatures,
    FeatureStabilityScorer,
    FeatureInteractionDetector,
)
from acms.ai.features.drift import (
    FeatureDriftMonitor,
    DriftResult,
    DriftType,
    DriftAlert,
    DriftVisualizer,
    RetrainingTrigger,
)

__all__ = [
    # Store
    "FeatureStore",
    "RealTimeFeatureComputer",
    "FeatureGroup",
    "FeatureStatistics",
    "FeatureFreshnessMonitor",
    "FeatureVersion",
    "FeatureQualityScorer",
    "FeatureDependencyGraph",
    # Engineering
    "AdvancedFeatureEngineer",
    "MarketMicrostructureFeatures",
    "CrossAssetFeatures",
    "TemporalFeatures",
    "RegimeFeatures",
    "SentimentFeatures",
    "OnChainFeatures",
    "FeatureStabilityScorer",
    "FeatureInteractionDetector",
    # Drift
    "FeatureDriftMonitor",
    "DriftResult",
    "DriftType",
    "DriftAlert",
    "DriftVisualizer",
    "RetrainingTrigger",
]

__version__ = "1.0.0"
