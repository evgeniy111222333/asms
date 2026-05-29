"""Base model abstractions for the ACMS AI module.

Provides the foundational model lifecycle management layer including:
- BaseModel: Abstract base class with save/load/version/serialize
- BaseModelRegistry: Model versioning and retrieval with metadata
- ModelCheckpoint: Checkpoint with metadata and training state
- ModelVersion: Semantic version tracking
- BaseModelServer: Inference serving with batching and warmup
- PredictionResult / BatchPredictionResult: Structured prediction outputs
- ModelMetadata: Training history, performance metrics, provenance
- ModelInput / ModelOutput protocols
- Automatic model serialization to/from disk
- Model warming/preloading
- ONNX export capability

All model implementations should inherit from BaseModel and implement
the required abstract methods.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import os
import shutil
import time
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from .config import AIConfig, InferenceConfig, ModelRegistryConfig
from .gpu_manager import GPUManager, get_gpu_manager
from .types import (
    EvaluationResult,
    ModelInput,
    ModelOutput,
    ModelPerformanceMetrics,
    ModelTask,
    PredictionType,
    PredictionWithUncertainty,
    TrainingState,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")
ModelType = TypeVar("ModelType", bound="BaseModel")


# ============================================================================
# Model Version
# ============================================================================


@dataclass
class ModelVersion:
    """Semantic version tracking for AI models.

    Attributes:
        major: Major version (breaking changes).
        minor: Minor version (new features, backward compatible).
        patch: Patch version (bug fixes).
        pre_release: Pre-release tag (e.g., 'alpha', 'beta', 'rc1').
        build_metadata: Build metadata (e.g., git hash).
        created_at: When this version was created.
        description: Human-readable description of changes.
    """

    major: int = 0
    minor: int = 1
    patch: int = 0
    pre_release: str = ""
    build_metadata: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    def __lt__(self, other: "ModelVersion") -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    @classmethod
    def parse(cls, version_str: str) -> "ModelVersion":
        """Parse a version string like '1.2.3-beta+abc123'.

        Args:
            version_str: Version string to parse.

        Returns:
            ModelVersion instance.

        Raises:
            ValueError: If version string is malformed.
        """
        build_metadata = ""
        if "+" in version_str:
            version_str, build_metadata = version_str.split("+", 1)

        pre_release = ""
        if "-" in version_str:
            version_str, pre_release = version_str.split("-", 1)

        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")

        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            pre_release=pre_release,
            build_metadata=build_metadata,
        )

    def bump_major(self) -> "ModelVersion":
        """Create a new version with major version incremented."""
        return ModelVersion(
            major=self.major + 1,
            minor=0,
            patch=0,
            created_at=datetime.utcnow(),
        )

    def bump_minor(self) -> "ModelVersion":
        """Create a new version with minor version incremented."""
        return ModelVersion(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            created_at=datetime.utcnow(),
        )

    def bump_patch(self) -> "ModelVersion":
        """Create a new version with patch version incremented."""
        return ModelVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            created_at=datetime.utcnow(),
        )


# ============================================================================
# Model Checkpoint
# ============================================================================


@dataclass
class ModelCheckpoint:
    """Model checkpoint with metadata.

    Represents a saved snapshot of model state including optimizer
    state, training progress, and associated metadata.

    Attributes:
        model_id: Unique identifier for the model.
        version: Model version at checkpoint time.
        path: Filesystem path to the checkpoint file.
        epoch: Training epoch at checkpoint time.
        global_step: Global optimizer step at checkpoint time.
        train_loss: Training loss at checkpoint time.
        val_loss: Validation loss at checkpoint time.
        metrics: Additional metrics at checkpoint time.
        training_state: Serialized training state.
        is_best: Whether this is the best checkpoint so far.
        file_size_mb: Size of the checkpoint file in MB.
        checksum: SHA256 checksum for integrity verification.
        created_at: When the checkpoint was created.
        tags: Arbitrary tags for filtering/grouping.
    """

    model_id: str = ""
    version: str = "0.1.0"
    path: str = ""
    epoch: int = 0
    global_step: int = 0
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    metrics: Dict[str, float] = field(default_factory=dict)
    training_state: Optional[Dict[str, Any]] = None
    is_best: bool = False
    file_size_mb: float = 0.0
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the checkpoint file exists and checksum matches."""
        if not self.path or not Path(self.path).exists():
            return False
        if self.checksum:
            return self._compute_checksum() == self.checksum
        return True

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of the checkpoint file."""
        if not self.path or not Path(self.path).exists():
            return ""
        sha256 = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "path": self.path,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "metrics": self.metrics,
            "is_best": self.is_best,
            "file_size_mb": self.file_size_mb,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


# ============================================================================
# Model Metadata
# ============================================================================


@dataclass
class ModelMetadata:
    """Comprehensive model metadata.

    Stores training provenance, performance history, and operational
    metadata for model lifecycle management.

    Attributes:
        model_id: Unique model identifier.
        model_name: Human-readable model name.
        version: Current model version.
        task: Model task type.
        description: Model description.
        architecture: Architecture description (e.g., 'LSTM-2x128-Attention').
        input_schema: Description of expected input format.
        output_schema: Description of output format.
        framework: ML framework used ('pytorch', 'sklearn', 'lightgbm').
        training_history: List of training run summaries.
        performance: Latest performance metrics.
        hyperparameters: Model hyperparameters.
        feature_names: Names of input features in order.
        training_data_range: Date range of training data.
        training_samples: Number of training samples.
        evaluation_samples: Number of evaluation samples.
        tags: Arbitrary tags for filtering.
        created_at: Model creation time.
        updated_at: Last update time.
        created_by: User/system that created the model.
        parent_model_id: ID of the parent model (for fine-tuned models).
        onnx_export_path: Path to ONNX export if available.
    """

    model_id: str = ""
    model_name: str = ""
    version: str = "0.1.0"
    task: ModelTask = ModelTask.PRICE_PREDICTION
    description: str = ""
    architecture: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    framework: str = "pytorch"
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    performance: Optional[ModelPerformanceMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    training_data_range: Tuple[Optional[str], Optional[str]] = (None, None)
    training_samples: int = 0
    evaluation_samples: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    parent_model_id: str = ""
    onnx_export_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "task": self.task.value,
            "description": self.description,
            "architecture": self.architecture,
            "framework": self.framework,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "training_data_range": list(self.training_data_range),
            "training_samples": self.training_samples,
            "evaluation_samples": self.evaluation_samples,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "parent_model_id": self.parent_model_id,
        }
        if self.performance is not None:
            result["performance"] = self.performance.to_dict()
        return result

    def record_training_run(self, training_state: TrainingState) -> None:
        """Record a training run in the history.

        Args:
            training_state: Final training state to record.
        """
        self.training_history.append({
            "epoch": training_state.epoch,
            "global_step": training_state.global_step,
            "best_val_loss": training_state.best_val_loss,
            "best_epoch": training_state.best_epoch,
            "elapsed_seconds": training_state.elapsed_seconds,
            "final_train_loss": training_state.train_loss,
            "final_val_loss": training_state.val_loss,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow()


# ============================================================================
# Prediction Result Types
# ============================================================================


@dataclass
class PredictionResult:
    """Result of a single model prediction.

    Attributes:
        model_id: Source model identifier.
        model_version: Model version used.
        predictions: Raw prediction array.
        prediction_type: Type of prediction.
        confidence: Prediction confidence score [0, 1].
        uncertainty: Optional uncertainty estimation.
        latency_ms: Inference latency in milliseconds.
        metadata: Additional result metadata.
        timestamp: When the prediction was made.
    """

    model_id: str = ""
    model_version: str = ""
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_type: PredictionType = PredictionType.REGRESSION
    confidence: float = 0.0
    uncertainty: Optional[PredictionWithUncertainty] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of the predictions array."""
        if len(self.predictions) > 0:
            return self.predictions.shape
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excludes numpy arrays)."""
        result = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_type": self.prediction_type.value,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if len(self.predictions) > 0:
            result["predictions_shape"] = list(self.predictions.shape)
        return result


@dataclass
class BatchPredictionResult:
    """Result of batch model predictions.

    Attributes:
        model_id: Source model identifier.
        model_version: Model version used.
        results: List of individual prediction results.
        total_latency_ms: Total batch processing latency.
        avg_latency_ms: Average latency per prediction.
        batch_size: Number of predictions in the batch.
        throughput: Predictions per second.
        metadata: Additional batch-level metadata.
    """

    model_id: str = ""
    model_version: str = ""
    results: List[PredictionResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.batch_size = len(self.results)
        if self.batch_size > 0 and self.total_latency_ms > 0:
            self.avg_latency_ms = self.total_latency_ms / self.batch_size
            self.throughput = self.batch_size / (self.total_latency_ms / 1000.0)

    def aggregate_predictions(self) -> np.ndarray:
        """Aggregate all predictions into a single array.

        Returns:
            Stacked prediction array.
        """
        if not self.results:
            return np.array([])
        return np.stack([r.predictions for r in self.results])

    def aggregate_confidences(self) -> np.ndarray:
        """Aggregate all confidence scores.

        Returns:
            Array of confidence scores.
        """
        return np.array([r.confidence for r in self.results])


# ============================================================================
# Protocols
# ============================================================================


class ModelInputProtocol(Protocol):
    """Protocol for model input objects."""

    features: np.ndarray
    symbol: str
    timestamp: datetime


class ModelOutputProtocol(Protocol):
    """Protocol for model output objects."""

    prediction: np.ndarray
    confidence: float
    model_id: str


# ============================================================================
# Base Model
# ============================================================================


class BaseModel(abc.ABC):
    """Abstract base class for all ACMS AI models.

    Provides the standard model lifecycle interface:
    - Training (fit)
    - Inference (predict, predict_batch)
    - Persistence (save, load)
    - Versioning (version, metadata)
    - Export (export_onnx)
    - Evaluation (evaluate)
    - Warmup (warmup)

    Subclasses must implement the abstract methods. Default
    implementations are provided for optional features.
    """

    def __init__(
        self,
        model_id: str,
        task: ModelTask = ModelTask.PRICE_PREDICTION,
        version: Optional[ModelVersion] = None,
        config: Optional[AIConfig] = None,
    ) -> None:
        """Initialize the base model.

        Args:
            model_id: Unique model identifier.
            task: Model task type.
            version: Initial model version.
            config: AI configuration.
        """
        self._model_id = model_id
        self._task = task
        self._version = version or ModelVersion()
        self._config = config or AIConfig()
        self._metadata = ModelMetadata(
            model_id=model_id,
            task=task,
            version=str(self._version),
        )
        self._is_fitted = False
        self._device = self._config.gpu.resolve_device()
        self._gpu_manager = get_gpu_manager()

        logger.info(
            "Initialized %s model '%s' v%s on %s",
            task.value, model_id, self._version, self._device,
        )

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Unique model identifier."""
        return self._model_id

    @property
    def task(self) -> ModelTask:
        """Model task type."""
        return self._task

    @property
    def version(self) -> ModelVersion:
        """Current model version."""
        return self._version

    @property
    def metadata(self) -> ModelMetadata:
        """Model metadata."""
        return self._metadata

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    @property
    def device(self) -> str:
        """Current compute device."""
        return self._device

    # ----------------------------------------------------------------
    # Abstract Methods
    # ----------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> TrainingState:
        """Train the model on the given data.

        Args:
            X: Training features of shape (n_samples, ...) or
                (n_samples, seq_len, n_features) for sequence models.
            y: Training targets.
            validation_data: Optional (X_val, y_val) tuple.
            **kwargs: Additional training arguments.

        Returns:
            Final TrainingState after training completes.
        """
        ...

    @abc.abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> PredictionResult:
        """Make a prediction on a single input or batch.

        Args:
            X: Input features. Shape depends on model architecture.
            **kwargs: Additional prediction arguments.

        Returns:
            PredictionResult with predictions and metadata.
        """
        ...

    @abc.abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get the model's state for serialization.

        Returns:
            Dictionary containing all model state needed for
            reconstruction. Must be JSON-serializable or contain
            numpy arrays.
        """
        ...

    @abc.abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore the model's state from a serialized dictionary.

        Args:
            state: State dictionary as returned by _get_model_state().
        """
        ...

    # ----------------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------------

    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        include_optimizer: bool = True,
        include_training_state: bool = True,
    ) -> ModelCheckpoint:
        """Save the model to disk.

        Creates a checkpoint file with model state and metadata.

        Args:
            path: Custom save path. Uses registry default if None.
            include_optimizer: Include optimizer state.
            include_training_state: Include training state.

        Returns:
            ModelCheckpoint with save details.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self._model_id}' has not been fitted. "
                "Call fit() before save()."
            )

        registry_config = self._config.registry
        if path is None:
            version_dir = Path(registry_config.models_dir) / self._model_id / str(self._version)
            version_dir.mkdir(parents=True, exist_ok=True)
            path = version_dir / "model.pt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self._get_model_state(),
            "model_id": self._model_id,
            "version": str(self._version),
            "task": self._task.value,
            "metadata": self._metadata.to_dict(),
            "is_fitted": self._is_fitted,
        }

        if include_optimizer:
            state["optimizer_state"] = self._get_optimizer_state()

        if include_training_state:
            state["training_state"] = self._get_training_state()

        # Save using PyTorch or pickle
        save_result = self._serialize_state(state, path)

        # Compute checksum
        checksum = ""
        if registry_config.compute_checksum:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()

        file_size_mb = path.stat().st_size / (1024 * 1024)

        # Save metadata sidecar
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(self._metadata.to_dict(), f, indent=2, default=str)

        checkpoint = ModelCheckpoint(
            model_id=self._model_id,
            version=str(self._version),
            path=str(path),
            file_size_mb=file_size_mb,
            checksum=checksum,
            is_best=False,
            created_at=datetime.utcnow(),
        )

        logger.info(
            "Saved model '%s' v%s to %s (%.1f MB)",
            self._model_id, self._version, path, file_size_mb,
        )

        return checkpoint

    @classmethod
    def load(
        cls: type[ModelType],
        path: Union[str, Path],
        config: Optional[AIConfig] = None,
    ) -> ModelType:
        """Load a model from disk.

        Args:
            path: Path to the saved model file.
            config: AI configuration for the loaded model.

        Returns:
            Loaded model instance.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = cls._deserialize_state(path)
        config = config or AIConfig()

        model_id = state.get("model_id", "unknown")
        task_str = state.get("task", "price_prediction")
        task = ModelTask(task_str)

        instance = cls(model_id=model_id, task=task, config=config)
        instance._version = ModelVersion.parse(state.get("version", "0.1.0"))
        instance._is_fitted = state.get("is_fitted", True)

        if "model_state" in state:
            instance._set_model_state(state["model_state"])

        if "optimizer_state" in state:
            instance._set_optimizer_state(state["optimizer_state"])

        if "training_state" in state:
            instance._set_training_state(state["training_state"])

        if "metadata" in state:
            # Restore metadata from saved dict
            meta_dict = state["metadata"]
            instance._metadata.model_name = meta_dict.get("model_name", "")
            instance._metadata.architecture = meta_dict.get("architecture", "")
            instance._metadata.framework = meta_dict.get("framework", "pytorch")
            instance._metadata.hyperparameters = meta_dict.get("hyperparameters", {})
            instance._metadata.feature_names = meta_dict.get("feature_names", [])

        logger.info(
            "Loaded model '%s' v%s from %s",
            model_id, instance._version, path,
        )

        return instance

    def _serialize_state(self, state: Dict[str, Any], path: Path) -> bool:
        """Serialize state dictionary to disk.

        Uses PyTorch's save if available, otherwise pickle.

        Args:
            state: State dictionary to serialize.
            path: Target file path.

        Returns:
            True if serialization succeeded.
        """
        try:
            import torch
            torch.save(state, str(path))
            return True
        except ImportError:
            logger.debug("PyTorch not available for state serialization, falling back to pickle")

        import pickle
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    @classmethod
    def _deserialize_state(cls, path: Path) -> Dict[str, Any]:
        """Deserialize state dictionary from disk.

        Args:
            path: Path to the serialized state file.

        Returns:
            State dictionary.
        """
        try:
            import torch
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except ImportError:
            logger.debug("PyTorch not available for state deserialization, falling back to pickle")

        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def _get_optimizer_state(self) -> Optional[Dict[str, Any]]:
        """Get optimizer state for serialization. Override in subclasses."""
        return None

    def _set_optimizer_state(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state. Override in subclasses."""
        pass

    def _get_training_state(self) -> Optional[Dict[str, Any]]:
        """Get training state for serialization. Override in subclasses."""
        return None

    def _set_training_state(self, state: Dict[str, Any]) -> None:
        """Restore training state. Override in subclasses."""
        pass

    # ----------------------------------------------------------------
    # Batch Prediction
    # ----------------------------------------------------------------

    def predict_batch(
        self,
        X_batch: List[np.ndarray],
        **kwargs: Any,
    ) -> BatchPredictionResult:
        """Make predictions on a batch of inputs.

        Processes inputs sequentially by default. Override for
        efficient batched inference.

        Args:
            X_batch: List of input feature arrays.
            **kwargs: Additional prediction arguments.

        Returns:
            BatchPredictionResult with all predictions.
        """
        start_time = time.perf_counter()
        results: List[PredictionResult] = []

        for X in X_batch:
            result = self.predict(X, **kwargs)
            results.append(result)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return BatchPredictionResult(
            model_id=self._model_id,
            model_version=str(self._version),
            results=results,
            total_latency_ms=total_latency_ms,
        )

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate the model on a dataset.

        Args:
            X: Evaluation features.
            y: Ground truth targets.
            metrics: Optional list of metrics to compute.

        Returns:
            EvaluationResult with computed metrics.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        start_time = time.perf_counter()
        prediction_result = self.predict(X)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        predictions = prediction_result.predictions
        computed_metrics: Dict[str, float] = {}

        # Regression metrics
        if len(predictions) > 0 and len(y) > 0:
            min_len = min(len(predictions), len(y))
            pred_flat = predictions.ravel()[:min_len]
            y_flat = y.ravel()[:min_len]

            mse = float(np.mean((pred_flat - y_flat) ** 2))
            mae = float(np.mean(np.abs(pred_flat - y_flat)))
            computed_metrics["mse"] = mse
            computed_metrics["mae"] = mae
            computed_metrics["rmse"] = float(np.sqrt(mse))

            # R-squared
            ss_res = float(np.sum((pred_flat - y_flat) ** 2))
            ss_tot = float(np.sum((y_flat - np.mean(y_flat)) ** 2))
            computed_metrics["r2"] = 1.0 - ss_res / (ss_tot + 1e-10)

            # Direction accuracy (for price prediction)
            if len(pred_flat) > 1:
                pred_dir = np.sign(np.diff(pred_flat))
                true_dir = np.sign(np.diff(y_flat))
                computed_metrics["direction_accuracy"] = float(
                    np.mean(pred_dir == true_dir)
                )

        computed_metrics["latency_ms"] = elapsed_ms

        return EvaluationResult(
            model_id=self._model_id,
            task=self._task,
            metrics=computed_metrics,
            predictions=predictions,
            targets=y,
        )

    # ----------------------------------------------------------------
    # Warmup
    # ----------------------------------------------------------------

    def warmup(self, iterations: int = 5) -> None:
        """Warm up the model for inference.

        Runs dummy forward passes to:
        - Pre-load model to GPU
        - Warm up CUDA kernels
        - Stabilize memory allocation

        Args:
            iterations: Number of warmup iterations.
        """
        if not self._is_fitted:
            logger.warning("Cannot warmup unfitted model '%s'", self._model_id)
            return

        logger.info("Warming up model '%s' (%d iterations)", self._model_id, iterations)

        # Create dummy input based on input schema
        dummy_input = self._create_dummy_input()
        if dummy_input is None:
            logger.warning("No dummy input available for warmup")
            return

        for i in range(iterations):
            try:
                self.predict(dummy_input)
            except Exception as e:
                logger.warning("Warmup iteration %d failed: %s", i, e)

        # Synchronize GPU
        if self._device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except (ImportError, RuntimeError) as e:
                logger.debug("Could not synchronize GPU: %s", e)

        logger.info("Model '%s' warmup complete", self._model_id)

    def _create_dummy_input(self) -> Optional[np.ndarray]:
        """Create a dummy input for warmup. Override in subclasses.

        Returns:
            Dummy numpy array matching model input shape, or None.
        """
        if self._metadata.input_schema and "shape" in self._metadata.input_schema:
            shape = self._metadata.input_schema["shape"]
            return np.random.randn(*shape).astype(np.float32)
        return None

    # ----------------------------------------------------------------
    # ONNX Export
    # ----------------------------------------------------------------

    def export_onnx(
        self,
        path: Optional[Union[str, Path]] = None,
        opset_version: int = 17,
        dynamic_batch: bool = True,
        **kwargs: Any,
    ) -> str:
        """Export the model to ONNX format.

        Args:
            path: Export file path. Uses registry default if None.
            opset_version: ONNX opset version.
            dynamic_batch: Whether to use dynamic batch dimension.
            **kwargs: Additional export arguments.

        Returns:
            Path to the exported ONNX file.

        Raises:
            RuntimeError: If the model is not fitted or export fails.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before ONNX export")

        registry_config = self._config.registry
        if path is None:
            export_dir = Path(registry_config.models_dir) / self._model_id / str(self._version)
            export_dir.mkdir(parents=True, exist_ok=True)
            path = export_dir / "model.onnx"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torch

            # Subclasses that use PyTorch should implement _get_torch_model
            torch_model = self._get_torch_model_for_export()
            if torch_model is None:
                raise RuntimeError(
                    "Model does not support ONNX export. "
                    "Implement _get_torch_model_for_export()."
                )

            dummy_input = self._create_dummy_input()
            if dummy_input is not None:
                dummy_tensor = torch.from_numpy(dummy_input).to(self._device)

                dynamic_axes = None
                if dynamic_batch:
                    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

                torch.onnx.export(
                    torch_model,
                    dummy_tensor,
                    str(path),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                )

                self._metadata.onnx_export_path = str(path)
                logger.info("Exported model '%s' to ONNX: %s", self._model_id, path)
                return str(path)
            else:
                raise RuntimeError("Cannot create dummy input for ONNX export")

        except ImportError:
            raise RuntimeError("PyTorch is required for ONNX export")
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}") from e

    def _get_torch_model_for_export(self) -> Any:
        """Get the underlying PyTorch model for ONNX export.

        Override in subclasses that use PyTorch.

        Returns:
            torch.nn.Module instance, or None if not applicable.
        """
        return None

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def to_device(self, device: Optional[str] = None) -> "BaseModel":
        """Move the model to a specific device.

        Args:
            device: Target device. Uses current device if None.

        Returns:
            Self for chaining.
        """
        self._device = device or self._device
        return self

    def summary(self) -> str:
        """Generate a human-readable model summary.

        Returns:
            Formatted model summary string.
        """
        lines = [
            f"Model: {self._model_id}",
            f"Version: {self._version}",
            f"Task: {self._task.value}",
            f"Framework: {self._metadata.framework}",
            f"Architecture: {self._metadata.architecture}",
            f"Fitted: {self._is_fitted}",
            f"Device: {self._device}",
        ]
        if self._metadata.hyperparameters:
            lines.append("Hyperparameters:")
            for k, v in self._metadata.hyperparameters.items():
                lines.append(f"  {k}: {v}")
        if self._metadata.feature_names:
            lines.append(f"Features ({len(self._metadata.feature_names)}): "
                         f"{', '.join(self._metadata.feature_names[:5])}...")
        return "\n".join(lines)


# ============================================================================
# Model Registry
# ============================================================================


class BaseModelRegistry:
    """Model versioning and retrieval registry.

    Manages the lifecycle of registered models including:
    - Model registration with versioning
    - Checkpoint management and pruning
    - Model retrieval by ID, version, or task
    - Metadata storage and querying
    - Model promotion (staging -> production)

    Thread-safe for concurrent access.
    """

    def __init__(self, config: Optional[ModelRegistryConfig] = None) -> None:
        """Initialize the model registry.

        Args:
            config: Registry configuration.
        """
        self._config = config or ModelRegistryConfig()
        self._models: Dict[str, Dict[str, ModelMetadata]] = {}  # model_id -> {version -> metadata}
        self._checkpoints: Dict[str, List[ModelCheckpoint]] = {}  # model_id -> checkpoints
        self._production_versions: Dict[str, str] = {}  # model_id -> version
        self._lock = threading.RLock()
        self._config.ensure_directories()

    def register(
        self,
        model: BaseModel,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a model in the registry.

        Args:
            model: Model instance to register.
            tags: Optional tags for categorization.

        Returns:
            Registration key (model_id + version).
        """
        with self._lock:
            model_id = model.model_id
            version_str = str(model.version)

            if model_id not in self._models:
                self._models[model_id] = {}

            metadata = copy.deepcopy(model.metadata)
            if tags:
                metadata.tags.extend(tags)

            self._models[model_id][version_str] = metadata

            logger.info(
                "Registered model '%s' v%s", model_id, version_str
            )
            return f"{model_id}:{version_str}"

    def register_checkpoint(self, checkpoint: ModelCheckpoint) -> None:
        """Register a model checkpoint.

        Args:
            checkpoint: ModelCheckpoint to register.
        """
        with self._lock:
            model_id = checkpoint.model_id
            if model_id not in self._checkpoints:
                self._checkpoints[model_id] = []
            self._checkpoints[model_id].append(checkpoint)

            # Prune old checkpoints
            if self._config.auto_prune:
                self._prune_checkpoints(model_id)

    def get_metadata(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_id: Model identifier.
            version: Version string. Uses latest if None.

        Returns:
            ModelMetadata if found, None otherwise.
        """
        with self._lock:
            if model_id not in self._models:
                return None

            versions = self._models[model_id]
            if not versions:
                return None

            if version is not None:
                return versions.get(version)

            # Return latest version
            latest = max(versions.keys(), key=lambda v: ModelVersion.parse(v))
            return versions.get(latest)

    def get_latest_version(self, model_id: str) -> Optional[str]:
        """Get the latest version string for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Latest version string, or None if model not found.
        """
        with self._lock:
            if model_id not in self._models or not self._models[model_id]:
                return None
            return max(
                self._models[model_id].keys(),
                key=lambda v: ModelVersion.parse(v),
            )

    def list_models(self, task: Optional[ModelTask] = None) -> List[str]:
        """List registered model IDs.

        Args:
            task: Filter by task type. Returns all if None.

        Returns:
            List of model IDs.
        """
        with self._lock:
            if task is None:
                return list(self._models.keys())

            return [
                model_id
                for model_id, versions in self._models.items()
                if any(m.task == task for m in versions.values())
            ]

    def list_versions(self, model_id: str) -> List[str]:
        """List all versions of a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of version strings, sorted latest first.
        """
        with self._lock:
            if model_id not in self._models:
                return []
            versions = list(self._models[model_id].keys())
            versions.sort(key=lambda v: ModelVersion.parse(v), reverse=True)
            return versions

    def promote(
        self,
        model_id: str,
        version: str,
        stage: str = "production",
    ) -> None:
        """Promote a model version to a deployment stage.

        Args:
            model_id: Model identifier.
            version: Version to promote.
            stage: Target stage ('production', 'staging', 'canary').
        """
        with self._lock:
            if model_id not in self._models:
                raise ValueError(f"Model '{model_id}' not found in registry")
            if version not in self._models[model_id]:
                raise ValueError(
                    f"Version '{version}' not found for model '{model_id}'"
                )

            if stage == "production":
                self._production_versions[model_id] = version
                logger.info(
                    "Promoted model '%s' v%s to PRODUCTION",
                    model_id, version,
                )
            else:
                logger.info(
                    "Promoted model '%s' v%s to %s",
                    model_id, version, stage,
                )

    def get_production_version(self, model_id: str) -> Optional[str]:
        """Get the current production version for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Production version string, or None.
        """
        return self._production_versions.get(model_id)

    def get_checkpoints(
        self,
        model_id: str,
        best_only: bool = False,
    ) -> List[ModelCheckpoint]:
        """Get checkpoints for a model.

        Args:
            model_id: Model identifier.
            best_only: Return only the best checkpoint.

        Returns:
            List of ModelCheckpoint instances.
        """
        with self._lock:
            checkpoints = self._checkpoints.get(model_id, [])
            if best_only:
                best = [cp for cp in checkpoints if cp.is_best]
                return best if best else checkpoints
            return sorted(checkpoints, key=lambda cp: cp.epoch, reverse=True)

    def delete_version(self, model_id: str, version: str) -> bool:
        """Delete a specific model version.

        Args:
            model_id: Model identifier.
            version: Version to delete.

        Returns:
            True if the version was found and deleted.
        """
        with self._lock:
            if model_id not in self._models:
                return False
            if version not in self._models[model_id]:
                return False

            del self._models[model_id][version]

            # Also remove checkpoint files
            version_dir = (
                Path(self._config.models_dir) / model_id / version
            )
            if version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)

            logger.info("Deleted model '%s' v%s", model_id, version)
            return True

    def _prune_checkpoints(self, model_id: str) -> None:
        """Prune old checkpoints beyond the configured limit.

        Args:
            model_id: Model identifier.
        """
        max_versions = self._config.max_versions_per_model
        checkpoints = self._checkpoints.get(model_id, [])

        if len(checkpoints) <= max_versions:
            return

        # Sort by creation time, keep the most recent + best
        checkpoints.sort(key=lambda cp: cp.created_at, reverse=True)
        keep = set()

        # Always keep the best checkpoint
        for cp in checkpoints:
            if cp.is_best:
                keep.add(id(cp))

        # Keep most recent up to limit
        for cp in checkpoints[:max_versions]:
            keep.add(id(cp))

        # Remove the rest
        to_remove = [cp for cp in checkpoints if id(cp) not in keep]
        for cp in to_remove:
            if cp.path and Path(cp.path).exists():
                Path(cp.path).unlink(missing_ok=True)
                logger.debug("Pruned checkpoint: %s", cp.path)

        self._checkpoints[model_id] = [
            cp for cp in checkpoints if id(cp) in keep
        ]

    def save_metadata(self, model_id: str, version: str) -> None:
        """Persist metadata for a model version to disk.

        Args:
            model_id: Model identifier.
            version: Version string.
        """
        metadata = self.get_metadata(model_id, version)
        if metadata is None:
            return

        meta_path = (
            Path(self._config.metadata_dir) / model_id / f"{version}.json"
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

    def load_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Load metadata for a model version from disk.

        Args:
            model_id: Model identifier.
            version: Version string.

        Returns:
            ModelMetadata if found, None otherwise.
        """
        meta_path = (
            Path(self._config.metadata_dir) / model_id / f"{version}.json"
        )
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            data = json.load(f)

        metadata = ModelMetadata(
            model_id=data.get("model_id", model_id),
            model_name=data.get("model_name", ""),
            version=data.get("version", version),
            task=ModelTask(data.get("task", "price_prediction")),
            description=data.get("description", ""),
            architecture=data.get("architecture", ""),
            framework=data.get("framework", "pytorch"),
            hyperparameters=data.get("hyperparameters", {}),
            feature_names=data.get("feature_names", []),
            tags=data.get("tags", []),
        )
        return metadata


# ============================================================================
# Model Server
# ============================================================================


class BaseModelServer:
    """Inference server for serving AI model predictions.

    Provides:
    - Single and batch prediction endpoints
    - Model warmup and preloading
    - Latency tracking
    - Thread-safe prediction serving
    - Automatic batching of concurrent requests

    Example:
        server = BaseModelServer(config=inference_config)
        server.load_model("price_predictor", model_instance)

        result = server.predict("price_predictor", input_data)
    """

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        """Initialize the model server.

        Args:
            config: Inference configuration.
        """
        self._config = config or InferenceConfig()
        self._models: Dict[str, BaseModel] = {}
        self._lock = threading.RLock()
        self._latency_history: Dict[str, List[float]] = {}
        self._request_count: Dict[str, int] = {}

    def load_model(
        self,
        model_id: str,
        model: BaseModel,
        warmup: bool = True,
    ) -> None:
        """Load a model into the serving infrastructure.

        Args:
            model_id: Serving identifier for the model.
            model: Model instance to serve.
            warmup: Whether to warm up the model after loading.
        """
        with self._lock:
            self._models[model_id] = model
            self._latency_history[model_id] = []
            self._request_count[model_id] = 0

        if warmup and model.is_fitted:
            model.warmup(iterations=self._config.warmup_iterations)

        logger.info(
            "Loaded model '%s' for serving (warmup=%s)", model_id, warmup
        )

    def unload_model(self, model_id: str) -> None:
        """Unload a model from the serving infrastructure.

        Args:
            model_id: Serving identifier of the model to unload.
        """
        with self._lock:
            self._models.pop(model_id, None)
            self._latency_history.pop(model_id, None)
            self._request_count.pop(model_id, None)

        logger.info("Unloaded model '%s' from serving", model_id)

    def predict(
        self,
        model_id: str,
        X: np.ndarray,
        **kwargs: Any,
    ) -> PredictionResult:
        """Serve a single prediction request.

        Args:
            model_id: Serving identifier of the model.
            X: Input features.
            **kwargs: Additional prediction arguments.

        Returns:
            PredictionResult.

        Raises:
            KeyError: If the model is not loaded.
        """
        model = self._get_model(model_id)
        start_time = time.perf_counter()

        result = model.predict(X, **kwargs)

        latency_ms = (time.perf_counter() - start_time) * 1000
        result.latency_ms = latency_ms

        self._record_latency(model_id, latency_ms)

        # Check latency target
        if latency_ms > self._config.max_latency_ms:
            logger.warning(
                "Model '%s' latency %.1fms exceeds target %.1fms",
                model_id, latency_ms, self._config.max_latency_ms,
            )

        return result

    def predict_batch(
        self,
        model_id: str,
        X_batch: List[np.ndarray],
        **kwargs: Any,
    ) -> BatchPredictionResult:
        """Serve a batch prediction request.

        Args:
            model_id: Serving identifier of the model.
            X_batch: List of input feature arrays.
            **kwargs: Additional prediction arguments.

        Returns:
            BatchPredictionResult.
        """
        model = self._get_model(model_id)
        return model.predict_batch(X_batch, **kwargs)

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded for serving.

        Args:
            model_id: Serving identifier.

        Returns:
            True if the model is loaded.
        """
        return model_id in self._models

    def list_served_models(self) -> List[str]:
        """List all models currently being served.

        Returns:
            List of model serving identifiers.
        """
        return list(self._models.keys())

    def get_latency_stats(self, model_id: str) -> Dict[str, float]:
        """Get latency statistics for a served model.

        Args:
            model_id: Serving identifier.

        Returns:
            Dictionary with p50, p95, p99 latency and request count.
        """
        latencies = self._latency_history.get(model_id, [])
        if not latencies:
            return {"count": 0}

        lat = np.array(latencies)
        return {
            "count": len(lat),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "mean": float(np.mean(lat)),
            "max": float(np.max(lat)),
        }

    def _get_model(self, model_id: str) -> BaseModel:
        """Get a loaded model by ID.

        Args:
            model_id: Serving identifier.

        Returns:
            BaseModel instance.

        Raises:
            KeyError: If model is not loaded.
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(
                    f"Model '{model_id}' not loaded. "
                    f"Available: {list(self._models.keys())}"
                )
            return self._models[model_id]

    def _record_latency(self, model_id: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            model_id: Model identifier.
            latency_ms: Latency in milliseconds.
        """
        with self._lock:
            if model_id not in self._latency_history:
                self._latency_history[model_id] = []
            self._latency_history[model_id].append(latency_ms)
            self._request_count[model_id] = self._request_count.get(model_id, 0) + 1

            # Keep only last 10000 measurements
            if len(self._latency_history[model_id]) > 10000:
                self._latency_history[model_id] = self._latency_history[model_id][-10000:]
