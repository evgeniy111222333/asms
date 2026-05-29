"""AI configuration system for the ACMS AI module.

Provides a comprehensive, hierarchical configuration system covering all
aspects of the AI pipeline:

- GPUConfig: Device selection, memory management, mixed precision settings
- TrainingConfig: Batch size, epochs, learning rates, schedulers
- ModelRegistryConfig: Model versioning, storage paths, retention
- InferenceConfig: Batch vs real-time, latency targets, concurrency
- FeatureStoreConfig: Redis backend, TTL, feature groups
- DistributedConfig: DDP settings, multi-node orchestration
- AIConfig: Top-level configuration aggregating all sub-configs

All configurations are dataclasses with sensible defaults, supporting
serialization to/from dictionaries and JSON for persistence and transport.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# GPU Configuration
# ============================================================================


@dataclass
class GPUConfig:
    """GPU device and memory management configuration.

    Controls GPU device selection, memory allocation strategy,
    and mixed-precision training settings.

    Attributes:
        device: Target device string ('cuda', 'cuda:0', 'cpu', 'auto').
            'auto' selects CUDA if available, otherwise CPU.
        device_ids: Specific GPU device IDs to use. Empty list means use all.
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0).
            Used for multi-tenant GPU sharing.
        enable_mixed_precision: Whether to use Automatic Mixed Precision (AMP).
        amp_dtype: Data type for AMP ('float16' or 'bfloat16').
        memory_pool_size_mb: Size of CUDA memory pool in MB. 0 = default.
        enable_cudnn_benchmark: Enable cuDNN auto-tuner for fixed-size inputs.
        enable_cudnn_deterministic: Force deterministic cuDNN for reproducibility.
        oom_retry_threshold_mb: Memory threshold below which to retry on OOM.
        enable_memory_efficient_attention: Use memory-efficient attention kernels.
        gradient_checkpointing: Enable gradient checkpointing to save memory.
        pin_memory: Pin memory for faster CPU-GPU transfers in DataLoader.
        max_memory_allocated_ratio: Abort if memory usage exceeds this ratio.
    """

    device: str = "auto"
    device_ids: List[int] = field(default_factory=list)
    memory_fraction: float = 0.9
    enable_mixed_precision: bool = True
    amp_dtype: str = "float16"
    memory_pool_size_mb: int = 0
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    oom_retry_threshold_mb: int = 256
    enable_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    pin_memory: bool = True
    max_memory_allocated_ratio: float = 0.95

    def __post_init__(self) -> None:
        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError(f"memory_fraction must be in (0, 1], got {self.memory_fraction}")
        if self.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {self.amp_dtype}")
        if self.max_memory_allocated_ratio <= 0 or self.max_memory_allocated_ratio > 1.0:
            raise ValueError(
                f"max_memory_allocated_ratio must be in (0, 1], "
                f"got {self.max_memory_allocated_ratio}"
            )

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device string.

        Returns:
            'cuda' if CUDA is available and device is 'auto',
            otherwise 'cpu'.
        """
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            logger.info("CUDA not available, falling back to CPU")
            return "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")
            return "cpu"

    def get_device_ids(self) -> List[int]:
        """Resolve device IDs, auto-detecting if not specified.

        Returns:
            List of available GPU device IDs.
        """
        if self.device_ids:
            return self.device_ids

        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
        except ImportError:
            pass
        return []


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Model training hyperparameters and schedule configuration.

    Controls all aspects of the training loop including optimization,
    learning rate scheduling, regularization, and early stopping.

    Attributes:
        batch_size: Training batch size per GPU.
        eval_batch_size: Batch size during evaluation (can be larger).
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate.
        min_learning_rate: Minimum learning rate for schedulers.
        weight_decay: L2 regularization coefficient.
        optimizer: Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop').
        momentum: Momentum factor for SGD/RMSProp.
        betas: Beta coefficients for Adam/AdamW.
        lr_scheduler: Learning rate scheduler type.
        warmup_steps: Number of warmup steps for LR scheduler.
        warmup_ratio: Fraction of total steps for warmup (alternative to warmup_steps).
        cosine_t_max: T_max for cosine annealing scheduler.
        gradient_clip_norm: Max gradient norm for clipping (0 = disabled).
        gradient_clip_value: Max gradient value for clipping (0 = disabled).
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_metric: Metric to monitor for early stopping.
        early_stopping_delta: Minimum change to qualify as improvement.
        dropout_rate: Dropout probability for regularization.
        label_smoothing: Label smoothing factor for classification.
        accumulation_steps: Gradient accumulation steps for effective larger batch.
        max_grad_norm: Maximum gradient norm (alias for gradient_clip_norm).
        scheduler_step_on_epoch: Whether scheduler steps per epoch (vs per batch).
        num_workers: DataLoader worker count.
        prefetch_factor: DataLoader prefetch factor.
        seed: Random seed for reproducibility.
        deterministic: Force deterministic operations.
        log_interval: Log training metrics every N steps.
        eval_interval: Evaluate every N steps (0 = every epoch).
        save_interval: Save checkpoint every N steps (0 = every epoch).
        save_best_only: Only save checkpoint when best metric improves.
    """

    batch_size: int = 64
    eval_batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    cosine_t_max: Optional[int] = None
    gradient_clip_norm: float = 1.0
    gradient_clip_value: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_delta: float = 1e-4
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler_step_on_epoch: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42
    deterministic: bool = False
    log_interval: int = 50
    eval_interval: int = 0
    save_interval: int = 0
    save_best_only: bool = False

    def __post_init__(self) -> None:
        valid_optimizers = {"adam", "adamw", "sgd", "rmsprop", "lion", "adafactor"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{self.optimizer}'"
            )
        valid_schedulers = {
            "cosine", "linear", "constant", "constant_with_warmup",
            "cosine_with_restarts", "polynomial", "onecycle", "exponential",
            "reduce_on_plateau", "none",
        }
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got '{self.lr_scheduler}'"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size including gradient accumulation.

        Returns:
            Effective batch size = batch_size * accumulation_steps.
        """
        return self.batch_size * self.accumulation_steps

    def get_warmup_steps(self, total_steps: int) -> int:
        """Calculate the number of warmup steps.

        If warmup_steps is explicitly set, use that.
        Otherwise, derive from warmup_ratio * total_steps.

        Args:
            total_steps: Total number of training steps.

        Returns:
            Number of warmup steps.
        """
        if self.warmup_steps > 0:
            return self.warmup_steps
        return max(1, int(total_steps * self.warmup_ratio))


# ============================================================================
# Model Registry Configuration
# ============================================================================


@dataclass
class ModelRegistryConfig:
    """Model versioning and storage configuration.

    Controls how models are versioned, stored, and retrieved from
    the model registry.

    Attributes:
        registry_root: Root directory for model storage.
        max_versions_per_model: Maximum number of versions to retain per model.
        auto_prune: Automatically prune old versions beyond max_versions_per_model.
        compression: Compression algorithm for saved models ('gzip', 'none').
        save_optimizer_state: Include optimizer state in checkpoints.
        save_training_state: Include training state (epoch, scheduler) in checkpoints.
        metadata_backend: Where to store metadata ('file', 'redis', 'database').
        metadata_ttl_seconds: TTL for metadata in Redis (0 = no expiry).
        version_format: Version string format ('semver', 'timestamp', 'incremental').
        export_formats: Supported export formats for models.
        onnx_opset_version: ONNX opset version for ONNX exports.
        validate_on_save: Run validation after saving a model.
        compute_checksum: Compute and store SHA256 checksum for integrity.
    """

    registry_root: str = "/data/acms/ai/models"
    max_versions_per_model: int = 10
    auto_prune: bool = True
    compression: str = "gzip"
    save_optimizer_state: bool = True
    save_training_state: bool = True
    metadata_backend: str = "file"
    metadata_ttl_seconds: int = 0
    version_format: str = "incremental"
    export_formats: List[str] = field(default_factory=lambda: ["pytorch", "onnx"])
    onnx_opset_version: int = 17
    validate_on_save: bool = True
    compute_checksum: bool = True

    def __post_init__(self) -> None:
        valid_compressions = {"gzip", "none", "lz4"}
        if self.compression not in valid_compressions:
            raise ValueError(
                f"compression must be one of {valid_compressions}, got '{self.compression}'"
            )
        valid_backends = {"file", "redis", "database"}
        if self.metadata_backend not in valid_backends:
            raise ValueError(
                f"metadata_backend must be one of {valid_backends}, "
                f"got '{self.metadata_backend}'"
            )
        valid_formats = {"semver", "timestamp", "incremental"}
        if self.version_format not in valid_formats:
            raise ValueError(
                f"version_format must be one of {valid_formats}, "
                f"got '{self.version_format}'"
            )

    @property
    def models_dir(self) -> Path:
        """Path to the models directory."""
        return Path(self.registry_root) / "models"

    @property
    def checkpoints_dir(self) -> Path:
        """Path to the checkpoints directory."""
        return Path(self.registry_root) / "checkpoints"

    @property
    def metadata_dir(self) -> Path:
        """Path to the metadata directory."""
        return Path(self.registry_root) / "metadata"

    def ensure_directories(self) -> None:
        """Create registry directories if they don't exist."""
        for dir_path in (self.models_dir, self.checkpoints_dir, self.metadata_dir):
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dir_path)


# ============================================================================
# Inference Configuration
# ============================================================================


@dataclass
class InferenceConfig:
    """Model inference serving configuration.

    Controls inference mode, concurrency, latency targets,
    and output processing.

    Attributes:
        mode: Inference mode ('batch' for offline, 'realtime' for serving).
        max_batch_size: Maximum batch size for batched inference.
        batch_timeout_ms: Maximum time to wait for batch completion (ms).
        latency_target_ms: Target inference latency in milliseconds.
        max_latency_ms: Maximum acceptable latency before alerting.
        concurrency: Maximum concurrent inference requests.
        timeout_seconds: Request timeout in seconds.
        enable_caching: Cache inference results for identical inputs.
        cache_ttl_seconds: TTL for cached inference results.
        cache_max_size: Maximum number of cached results.
        warmup_iterations: Number of warmup iterations before serving.
        enable_profiling: Enable inference profiling.
        precision: Inference precision ('fp32', 'fp16', 'bf16', 'int8').
        enable_tracing: Enable distributed tracing for inference.
        output_postprocessing: Enable post-processing of model outputs.
        confidence_threshold: Minimum confidence to include prediction.
        uncertainty_estimation: Enable uncertainty estimation.
        mc_dropout_samples: Number of MC dropout forward passes.
        return_probabilities: Return class probabilities with predictions.
    """

    mode: str = "realtime"
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    latency_target_ms: float = 10.0
    max_latency_ms: float = 100.0
    concurrency: int = 4
    timeout_seconds: float = 30.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 10000
    warmup_iterations: int = 5
    enable_profiling: bool = False
    precision: str = "fp16"
    enable_tracing: bool = False
    output_postprocessing: bool = True
    confidence_threshold: float = 0.5
    uncertainty_estimation: bool = False
    mc_dropout_samples: int = 30
    return_probabilities: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"batch", "realtime"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")
        valid_precisions = {"fp32", "fp16", "bf16", "int8"}
        if self.precision not in valid_precisions:
            raise ValueError(
                f"precision must be one of {valid_precisions}, got '{self.precision}'"
            )
        if self.latency_target_ms <= 0:
            raise ValueError(
                f"latency_target_ms must be positive, got {self.latency_target_ms}"
            )
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {self.concurrency}")


# ============================================================================
# Feature Store Configuration
# ============================================================================


@dataclass
class FeatureStoreConfig:
    """Feature store backend configuration.

    Controls how computed features are stored, cached, and retrieved
    for both training and inference pipelines.

    Attributes:
        backend: Storage backend ('redis', 'memory', 'file').
        redis_url: Redis connection URL.
        redis_key_prefix: Key prefix for all feature store entries.
        default_ttl_seconds: Default TTL for cached features.
        max_memory_mb: Maximum memory usage for in-memory backend.
        feature_groups: Named feature group definitions.
        compute_on_demand: Compute features on cache miss.
        precompute_schedule: Cron schedule for batch precomputation.
        precompute_symbols: Symbols to precompute features for.
        precompute_timeframes: Timeframes to precompute.
        enable_versioning: Track feature computation versions.
        consistency_check: Verify feature consistency across sources.
        compression: Compress stored features ('none', 'lz4', 'zstd').
        max_feature_vector_size: Maximum size of a single feature vector.
        batch_fetch_size: Number of keys to fetch in a single Redis pipeline.
        connection_pool_size: Redis connection pool size.
        connection_timeout_ms: Redis connection timeout.
    """

    backend: str = "redis"
    redis_url: str = "redis://localhost:6379/1"
    redis_key_prefix: str = "acms:features"
    default_ttl_seconds: int = 3600
    max_memory_mb: int = 4096
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    compute_on_demand: bool = True
    precompute_schedule: str = ""
    precompute_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    precompute_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])
    enable_versioning: bool = True
    consistency_check: bool = True
    compression: str = "lz4"
    max_feature_vector_size: int = 10000
    batch_fetch_size: int = 100
    connection_pool_size: int = 10
    connection_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        valid_backends = {"redis", "memory", "file"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{self.backend}'"
            )
        valid_compressions = {"none", "lz4", "zstd"}
        if self.compression not in valid_compressions:
            raise ValueError(
                f"compression must be one of {valid_compressions}, got '{self.compression}'"
            )

    def get_redis_key(self, symbol: str, timeframe: str, group: str = "default") -> str:
        """Construct a Redis key for a feature entry.

        Args:
            symbol: Trading pair symbol.
            timeframe: Data timeframe.
            group: Feature group name.

        Returns:
            Fully qualified Redis key string.
        """
        return f"{self.redis_key_prefix}:{symbol}:{timeframe}:{group}"


# ============================================================================
# Distributed Configuration
# ============================================================================


@dataclass
class DistributedConfig:
    """Distributed training configuration (DDP, FSDP).

    Controls distributed data-parallel training across multiple
    GPUs and nodes.

    Attributes:
        backend: Distributed backend ('nccl', 'gloo', 'mpi').
        world_size: Total number of processes across all nodes.
        rank: Global rank of this process.
        local_rank: Local rank on this node.
        master_addr: Address of the master node.
        master_port: Port of the master node.
        init_method: URL for rendezvous ('env://', 'tcp://...', 'file://...').
        timeout_minutes: Timeout for distributed operations.
        enable_ddp: Enable DistributedDataParallel.
        enable_fsdp: Enable FullyShardedDataParallel (overrides DDP).
        fsdp_wrap_min_params: Minimum parameters to create an FSDP unit.
        fsdp_sharding_strategy: FSDP sharding strategy.
        gradient_as_bucket_view: Use gradient bucket view for memory efficiency.
        find_unused_parameters: Find unused parameters in forward pass.
        bucket_cap_mb: DDP gradient bucket size in MB.
        sync_batchnorm: Synchronize batch normalization across GPUs.
        overlap_communication: Overlap gradient communication with computation.
        enable_pipeline_parallelism: Enable pipeline parallelism.
        pipeline_chunks: Number of micro-batches for pipeline parallelism.
    """

    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    init_method: str = "env://"
    timeout_minutes: int = 30
    enable_ddp: bool = False
    enable_fsdp: bool = False
    fsdp_wrap_min_params: int = 1_000_000
    fsdp_sharding_strategy: str = "full"
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    sync_batchnorm: bool = False
    overlap_communication: bool = True
    enable_pipeline_parallelism: bool = False
    pipeline_chunks: int = 1

    def __post_init__(self) -> None:
        valid_backends = {"nccl", "gloo", "mpi"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{self.backend}'"
            )
        valid_sharding = {"full", "shard_grad_op", "no_shard"}
        if self.fsdp_sharding_strategy not in valid_sharding:
            raise ValueError(
                f"fsdp_sharding_strategy must be one of {valid_sharding}, "
                f"got '{self.fsdp_sharding_strategy}'"
            )
        if self.enable_fsdp and self.enable_ddp:
            logger.warning(
                "Both FSDP and DDP enabled; FSDP will take precedence"
            )

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is enabled."""
        return self.enable_ddp or self.enable_fsdp or self.world_size > 1

    @property
    def is_master(self) -> bool:
        """Whether this is the master (rank 0) process."""
        return self.rank == 0

    def get_init_url(self) -> str:
        """Construct the rendezvous URL for distributed init.

        Returns:
            Init URL string for torch.distributed.init_process_group.
        """
        if self.init_method != "env://":
            return self.init_method
        return f"tcp://{self.master_addr}:{self.master_port}"


# ============================================================================
# Top-Level AI Configuration
# ============================================================================


@dataclass
class AIConfig:
    """Top-level configuration for the ACMS AI subsystem.

    Aggregates all sub-configurations and provides global settings
    for the AI pipeline.

    Attributes:
        enabled: Whether the AI subsystem is enabled.
        environment: Deployment environment ('development', 'staging', 'production').
        project_name: Project name for logging and tracking.
        experiment_tracker: Experiment tracking backend ('mlflow', 'wandb', 'none').
        experiment_tracking_uri: URI for the experiment tracking server.
        gpu: GPU configuration.
        training: Training configuration.
        registry: Model registry configuration.
        inference: Inference configuration.
        feature_store: Feature store configuration.
        distributed: Distributed training configuration.
        data_dir: Root directory for AI data (datasets, features).
        log_dir: Directory for AI-specific logs.
        cache_dir: Directory for model and data caches.
        enable_auto_retraining: Enable automatic model retraining.
        retraining_interval_hours: Hours between automatic retraining checks.
        enable_model_monitoring: Enable model performance monitoring.
        monitoring_interval_minutes: Minutes between monitoring checks.
        drift_detection_threshold: Threshold for model drift detection.
        enable_ab_testing: Enable model A/B testing.
        ab_traffic_split: Traffic split ratio for A/B testing.
        max_concurrent_training_jobs: Maximum concurrent training jobs.
        enable_checkpoint_recovery: Resume training from last checkpoint.
        slack_alert_webhook: Webhook URL for training/alert notifications.
        tags: Arbitrary tags for configuration grouping.
    """

    enabled: bool = True
    environment: str = "development"
    project_name: str = "acms-ai"
    experiment_tracker: str = "none"
    experiment_tracking_uri: str = ""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data_dir: str = "/data/acms/ai"
    log_dir: str = "/data/acms/ai/logs"
    cache_dir: str = "/data/acms/ai/cache"
    enable_auto_retraining: bool = False
    retraining_interval_hours: int = 24
    enable_model_monitoring: bool = True
    monitoring_interval_minutes: int = 15
    drift_detection_threshold: float = 0.05
    enable_ab_testing: bool = False
    ab_traffic_split: float = 0.5
    max_concurrent_training_jobs: int = 2
    enable_checkpoint_recovery: bool = True
    slack_alert_webhook: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_environments = {"development", "staging", "production"}
        if self.environment not in valid_environments:
            raise ValueError(
                f"environment must be one of {valid_environments}, "
                f"got '{self.environment}'"
            )
        valid_trackers = {"mlflow", "wandb", "none", "tensorboard"}
        if self.experiment_tracker not in valid_trackers:
            raise ValueError(
                f"experiment_tracker must be one of {valid_trackers}, "
                f"got '{self.experiment_tracker}'"
            )
        if not 0.0 < self.ab_traffic_split < 1.0:
            raise ValueError(
                f"ab_traffic_split must be in (0, 1), got {self.ab_traffic_split}"
            )

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a nested dictionary.

        Returns:
            Dictionary representation of the entire configuration.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "to_dict"):
                result[f.name] = value.to_dict()
            elif dataclasses_is_dataclass(value):
                result[f.name] = asdict(value)
            else:
                result[f.name] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIConfig":
        """Create an AIConfig from a dictionary.

        Handles nested sub-configurations by passing them to their
        respective constructors.

        Args:
            data: Dictionary with configuration values.

        Returns:
            AIConfig instance.
        """
        sub_configs = {
            "gpu": GPUConfig,
            "training": TrainingConfig,
            "registry": ModelRegistryConfig,
            "inference": InferenceConfig,
            "feature_store": FeatureStoreConfig,
            "distributed": DistributedConfig,
        }

        kwargs = {}
        for key, value in data.items():
            if key in sub_configs and isinstance(value, dict):
                kwargs[key] = sub_configs[key](**value)
            else:
                kwargs[key] = value

        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "AIConfig":
        """Create an AIConfig from a JSON string.

        Args:
            json_str: JSON string with configuration values.

        Returns:
            AIConfig instance.
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AIConfig":
        """Load AIConfig from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            AIConfig instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to write the configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Configuration saved to %s", path)

    # ----------------------------------------------------------------
    # Environment Overrides
    # ----------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "AIConfig":
        """Create an AIConfig with values from environment variables.

        Environment variables take the form ACMS_AI_<SECTION>_<KEY>
        (e.g., ACMS_AI_GPU_DEVICE=cuda:0, ACMS_AI_TRAINING_BATCH_SIZE=128).

        Returns:
            AIConfig with environment overrides applied.
        """
        config = cls()
        prefix = "ACMS_AI_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            parts = env_key[len(prefix):].lower().split("_", 1)
            if len(parts) == 2:
                section, key = parts
                sub_config_map = {
                    "gpu": "gpu",
                    "training": "training",
                    "registry": "registry",
                    "inference": "inference",
                    "feature_store": "feature_store",
                    "distributed": "distributed",
                }
                if section in sub_config_map:
                    sub_config = getattr(config, sub_config_map[section])
                    if hasattr(sub_config, key):
                        current = getattr(sub_config, key)
                        try:
                            converted = type(current)(env_value)
                            setattr(sub_config, key, converted)
                            logger.debug(
                                "Override from env: %s = %s", env_key, env_value
                            )
                        except (ValueError, TypeError):
                            logger.warning(
                                "Could not convert env %s=%s to %s",
                                env_key, env_value, type(current).__name__,
                            )
            elif len(parts) == 1:
                key = parts[0]
                if hasattr(config, key):
                    current = getattr(config, key)
                    try:
                        setattr(config, key, type(current)(env_value))
                    except (ValueError, TypeError):
                        logger.warning(
                            "Could not convert env %s=%s", env_key, env_value
                        )

        return config

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the configuration for consistency.

        Returns:
            List of validation warning/error messages. Empty if valid.
        """
        issues: List[str] = []

        # GPU + Distributed consistency
        if self.distributed.is_distributed and self.gpu.resolve_device() == "cpu":
            issues.append(
                "Distributed training enabled but no GPU available; "
                "training will run on CPU which is extremely slow."
            )

        # Mixed precision on CPU
        if self.gpu.enable_mixed_precision and self.gpu.resolve_device() == "cpu":
            issues.append(
                "Mixed precision enabled but running on CPU; "
                "AMP has no benefit on CPU. Consider disabling."
            )

        # Training batch size vs GPU memory
        if self.training.batch_size > 512 and self.gpu.memory_fraction < 0.5:
            issues.append(
                f"Large batch_size ({self.training.batch_size}) with low "
                f"memory_fraction ({self.gpu.memory_fraction}) may cause OOM."
            )

        # Feature store + distributed
        if self.distributed.is_distributed and self.feature_store.backend == "memory":
            issues.append(
                "Using in-memory feature store with distributed training; "
                "each process will have its own copy. Consider using Redis."
            )

        # Inference latency targets
        if self.inference.uncertainty_estimation and self.inference.mc_dropout_samples > 50:
            if self.inference.latency_target_ms < 100:
                issues.append(
                    f"MC dropout with {self.inference.mc_dropout_samples} samples "
                    f"unlikely to meet {self.inference.latency_target_ms}ms latency target."
                )

        # Model registry path
        if self.environment == "production" and self.registry.registry_root.startswith(
            "/tmp"
        ):
            issues.append(
                "Production environment using /tmp for model registry; "
                "data may be lost on restart."
            )

        # Auto retraining without monitoring
        if self.enable_auto_retraining and not self.enable_model_monitoring:
            issues.append(
                "Auto retraining enabled without model monitoring; "
                "retraining triggers will not have drift context."
            )

        if issues:
            for issue in issues:
                logger.warning("Config validation: %s", issue)
        else:
            logger.info("Configuration validation passed")

        return issues

    def ensure_directories(self) -> None:
        """Create all required directories for the AI subsystem."""
        for dir_path in (self.data_dir, self.log_dir, self.cache_dir):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", dir_path)
        self.registry.ensure_directories()


# ============================================================================
# Helpers
# ============================================================================


def dataclasses_is_dataclass(obj: Any) -> bool:
    """Check if an object is a dataclass instance.

    Args:
        obj: Object to check.

    Returns:
        True if obj is a dataclass instance.
    """
    try:
        from dataclasses import is_dataclass
        return is_dataclass(obj) and not isinstance(obj, type)
    except ImportError:
        return False


def load_config(path: Optional[Union[str, Path]] = None) -> AIConfig:
    """Load AIConfig from file or environment.

    Resolution order:
    1. Explicit path provided
    2. ACMS_AI_CONFIG environment variable
    3. ./acms_ai_config.json in current directory
    4. Default configuration

    Args:
        path: Optional explicit path to configuration file.

    Returns:
        Loaded AIConfig instance.
    """
    if path is not None:
        logger.info("Loading AI config from explicit path: %s", path)
        return AIConfig.from_file(path)

    env_path = os.environ.get("ACMS_AI_CONFIG")
    if env_path:
        logger.info("Loading AI config from env ACMS_AI_CONFIG=%s", env_path)
        return AIConfig.from_file(env_path)

    local_path = Path("acms_ai_config.json")
    if local_path.exists():
        logger.info("Loading AI config from local file: %s", local_path)
        return AIConfig.from_file(local_path)

    logger.info("No config file found, using defaults + env overrides")
    return AIConfig.from_env()
