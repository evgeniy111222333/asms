"""ACMS AI Module - GPU-ready, self-learning AI system for cryptocurrency trading.

This package provides a comprehensive AI infrastructure for the Algorithmic
Crypto Management System (ACMS), designed for production-grade cryptocurrency
trading with GPU acceleration and self-learning capabilities.

Architecture Overview
---------------------
The AI module is organized into the following sub-packages:

    ai/
    ├── core/           # Core abstractions, configuration, GPU management
    │   ├── config      # Hierarchical configuration system (GPU, Training, Registry, ...)
    │   ├── gpu_manager # Singleton GPU device manager with AMP, OOM prevention
    │   ├── base_models # BaseModel abstract class, Registry, Server, Checkpoint
    │   ├── tensor_utils# Tensor operations, datasets, augmentation, normalization
    │   └── types       # AI-specific types (signals, regimes, risk, predictions)
    ├── models/         # Concrete model implementations
    │   ├── price       # Price prediction models (LSTM, Transformer)
    │   ├── regime      # Regime detection (HMM, clustering)
    │   ├── volatility  # Volatility forecasting (GARCH-NN, Deep Vol)
    │   └── execution   # Execution optimization (RL agents)
    ├── features/       # Feature engineering and feature store
    ├── training/       # Training pipelines, schedulers, callbacks
    ├── inference/      # Real-time inference engine, model serving
    └── monitoring/     # Model monitoring, drift detection, alerting

Key Design Principles
---------------------
1. **GPU-First**: All compute-intensive operations leverage CUDA when available,
   with automatic CPU fallback for development and testing.

2. **Graceful Degradation**: Optional dependencies (PyTorch, ONNX, Redis) are
   handled with lazy imports and meaningful fallbacks. The system never crashes
   due to a missing optional dependency.

3. **Self-Learning**: Automated retraining triggers based on drift detection,
   performance degradation, and scheduled intervals.

4. **Production-Ready**: Comprehensive logging, error handling, memory
   management, and monitoring throughout.

5. **Type-Safe**: Full type annotations with strict typing for all public APIs.

Quick Start
-----------
    from acms.ai import AIConfig, GPUManager, BaseModelRegistry

    # Initialize with configuration
    config = AIConfig.from_env()
    config.validate()
    config.ensure_directories()

    # Initialize GPU
    gpu = GPUManager()
    gpu.initialize(config.gpu)

    # Use model registry
    registry = BaseModelRegistry(config.registry)

Thread Safety
-------------
- GPUManager: Singleton with thread-safe initialization
- BaseModelRegistry: RLock-protected for concurrent access
- BaseModelServer: RLock-protected for concurrent predictions
- TensorCache: RLock-protected LRU cache

Performance Considerations
--------------------------
- Mixed precision (AMP) is enabled by default when GPU is available
- Gradient accumulation simulates larger batch sizes on limited VRAM
- Memory-efficient attention reduces VRAM usage for transformer models
- Tensor caching with LRU eviction prevents redundant computation
- Streaming data loader enables training on datasets larger than RAM
"""

__version__ = "0.1.0"
__author__ = "ACMS Team"

# ============================================================================
# Core sub-package imports
# ============================================================================

from acms.ai.core.config import (
    AIConfig,
    DistributedConfig,
    FeatureStoreConfig,
    GPUConfig,
    InferenceConfig,
    ModelRegistryConfig,
    TrainingConfig,
    load_config,
)

from acms.ai.core.gpu_manager import (
    GPUDeviceInfo,
    GPUMemoryInfo,
    GPUManager,
    device,
    empty_cache,
    get_gpu_manager,
    is_gpu_available,
    memory_info,
    to_device,
)

from acms.ai.core.base_models import (
    BaseModel,
    BaseModelRegistry,
    BaseModelServer,
    BatchPredictionResult,
    ModelCheckpoint,
    ModelMetadata,
    ModelVersion,
    PredictionResult,
)

from acms.ai.core.tensor_utils import (
    MinMaxScaler,
    RobustScaler,
    SlidingWindowDataset,
    StandardScaler,
    StreamingDataLoader,
    TensorCache,
    TensorDataset,
    TimeSeriesAugmentor,
    collate_batch,
    collate_to_torch,
    create_attention_mask,
    create_tensor,
    move_to_device,
    pad_sequences,
    temporal_split,
    walk_forward_splits,
)

from acms.ai.core.types import (
    EvaluationResult,
    ExplanationResult,
    FeatureImportance,
    MarketRegime,
    MarketStateVector,
    ModelInput,
    ModelOutput,
    ModelPerformanceMetrics,
    ModelTask,
    PredictionType,
    PredictionWithUncertainty,
    PositionRecommendation,
    RegimePrediction,
    RiskAssessment,
    RiskLevel,
    SignalPrediction,
    SignalStrength,
    TrainingPhase,
    TrainingState,
    UncertaintyMethod,
)

# ============================================================================
# Type aliases re-exported at top level
# ============================================================================

from acms.ai.core.types import (
    FeatureTensor,
    MarketTensor,
    PredictionTensor,
)

# ============================================================================
# Package-level convenience
# ============================================================================


def get_version() -> str:
    """Return the AI module version string."""
    return __version__


def is_available() -> bool:
    """Check if the AI module's core dependencies are available.

    Returns:
        True if numpy is available (minimum requirement).
    """
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def is_gpu_ready() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CUDA + PyTorch are available.
    """
    return is_gpu_available()


def get_system_info() -> dict:
    """Get a summary of the AI system's capabilities.

    Returns:
        Dictionary with availability flags and device information.
    """
    gpu_manager = get_gpu_manager()
    return {
        "ai_version": __version__,
        "numpy_available": True,
        "torch_available": _check_torch(),
        "cuda_available": is_gpu_available(),
        "gpu_manager_status": gpu_manager.get_status(),
    }


def _check_torch() -> bool:
    """Check if PyTorch is installed."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = [
    # Version
    "__version__",
    "get_version",
    # Convenience
    "is_available",
    "is_gpu_ready",
    "get_system_info",
    # Config
    "AIConfig",
    "GPUConfig",
    "TrainingConfig",
    "ModelRegistryConfig",
    "InferenceConfig",
    "FeatureStoreConfig",
    "DistributedConfig",
    "load_config",
    # GPU
    "GPUManager",
    "GPUMemoryInfo",
    "GPUDeviceInfo",
    "get_gpu_manager",
    "device",
    "empty_cache",
    "is_gpu_available",
    "memory_info",
    "to_device",
    # Models
    "BaseModel",
    "BaseModelRegistry",
    "BaseModelServer",
    "ModelCheckpoint",
    "ModelMetadata",
    "ModelVersion",
    "PredictionResult",
    "BatchPredictionResult",
    # Tensor utils
    "TensorDataset",
    "StreamingDataLoader",
    "TensorCache",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "SlidingWindowDataset",
    "TimeSeriesAugmentor",
    "create_tensor",
    "move_to_device",
    "pad_sequences",
    "create_attention_mask",
    "temporal_split",
    "walk_forward_splits",
    "collate_batch",
    "collate_to_torch",
    # Types
    "MarketTensor",
    "FeatureTensor",
    "PredictionTensor",
    "ModelInput",
    "ModelOutput",
    "TrainingState",
    "EvaluationResult",
    "PredictionWithUncertainty",
    "ModelPerformanceMetrics",
    "FeatureImportance",
    "ExplanationResult",
    "RegimePrediction",
    "SignalPrediction",
    "PositionRecommendation",
    "RiskAssessment",
    "MarketStateVector",
    # Enums
    "ModelTask",
    "PredictionType",
    "MarketRegime",
    "SignalStrength",
    "RiskLevel",
    "UncertaintyMethod",
    "TrainingPhase",
]
