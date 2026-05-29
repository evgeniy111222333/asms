"""ACMS AI Core - Foundation layer for the AI subsystem.

This sub-package provides the foundational abstractions and utilities
upon which all higher-level AI functionality is built:

Configuration
-------------
- **AIConfig**: Top-level configuration aggregating all sub-configs
- **GPUConfig**: GPU device selection, memory, mixed precision
- **TrainingConfig**: Training hyperparameters, optimizers, schedulers
- **ModelRegistryConfig**: Model versioning, storage, retention
- **InferenceConfig**: Serving mode, latency targets, caching
- **FeatureStoreConfig**: Feature caching backend, TTL, precomputation
- **DistributedConfig**: DDP/FSDP multi-GPU settings

GPU Management
--------------
- **GPUManager**: Singleton GPU device manager with:
  - Auto device detection and selection
  - CUDA context management
  - Mixed precision (AMP) context managers
  - Memory tracking and OOM prevention
  - Multi-GPU orchestration
  - Automatic CPU fallback

Model Abstractions
------------------
- **BaseModel**: Abstract base class for all AI models with:
  - Standard lifecycle (fit, predict, evaluate, save, load)
  - Version tracking and metadata
  - ONNX export capability
  - Model warmup
- **BaseModelRegistry**: Model versioning, promotion, and retrieval
- **BaseModelServer**: Inference serving with latency tracking
- **ModelCheckpoint**: Checkpoint with metadata and integrity
- **ModelVersion**: Semantic version tracking
- **PredictionResult / BatchPredictionResult**: Structured outputs

Tensor Utilities
----------------
- **TensorDataset**: Lazy-loading dataset from Parquet
- **StreamingDataLoader**: Memory-efficient streaming for large data
- **TensorCache**: LRU tensor cache with configurable eviction
- **StandardScaler / MinMaxScaler / RobustScaler**: Normalization
- **pad_sequences / create_attention_mask**: Sequence utilities
- **temporal_split / walk_forward_splits**: Time series splitting
- **SlidingWindowDataset**: Sliding window generator
- **TimeSeriesAugmentor**: Data augmentation for time series
- **collate_batch / collate_to_torch**: Efficient batch collation

Types
-----
- **MarketTensor / FeatureTensor / PredictionTensor**: Type aliases
- **ModelInput / ModelOutput**: Structured I/O dataclasses
- **TrainingState / EvaluationResult**: Training lifecycle types
- **PredictionWithUncertainty**: Uncertainty-quantified predictions
- **FeatureImportance / ExplanationResult**: Interpretability types
- **RegimePrediction / SignalPrediction**: Trading signal types
- **PositionRecommendation**: Position sizing and management
- **RiskAssessment**: Comprehensive risk evaluation
- **MarketStateVector**: Canonical market state representation
"""

# ============================================================================
# Configuration
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

# ============================================================================
# GPU Management
# ============================================================================

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

# ============================================================================
# Model Abstractions
# ============================================================================

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

# ============================================================================
# Tensor Utilities
# ============================================================================

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

# ============================================================================
# Types
# ============================================================================

from acms.ai.core.types import (
    EvaluationResult,
    ExplanationResult,
    FeatureImportance,
    FeatureTensor,
    MarketRegime,
    MarketStateVector,
    MarketTensor,
    ModelInput,
    ModelOutput,
    ModelPerformanceMetrics,
    ModelTask,
    PredictionTensor,
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

__all__ = [
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
    # Tensor
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
    "ModelTask",
    "PredictionType",
    "MarketRegime",
    "SignalStrength",
    "RiskLevel",
    "UncertaintyMethod",
    "TrainingPhase",
]
