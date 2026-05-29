"""AI Metrics Collection and Export for ACMS.

Implements:
- AIMetricsCollector: Central metrics collection hub
- ModelPerformanceMetrics: Sharpe, accuracy, calibration, directional accuracy
- TrainingMetrics: Loss curves, gradient norms, learning rate tracking
- InferenceMetrics: Latency, throughput, batch size tracking
- FeatureMetrics: Importance shifts, drift scores, correlation changes
- Prometheus metric exporters for all AI metrics
- Metric aggregation with sliding windows
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Data Structures
# ============================================================================

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a trained model.

    Attributes:
        model_id: Model identifier.
        accuracy: Classification accuracy.
        sharpe_ratio: Risk-adjusted return metric.
        sortino_ratio: Downside deviation adjusted return.
        max_drawdown: Maximum peak-to-trough decline.
        win_rate: Percentage of profitable predictions.
        directional_accuracy: Correct direction prediction rate.
        calibration_error: Expected Calibration Error.
        f1_score: Harmonic mean of precision and recall.
        precision: True positives / (true + false positives).
        recall: True positives / (true positives + false negatives).
        auc_roc: Area under ROC curve.
        timestamp: When metrics were computed.
    """
    model_id: str = ""
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    directional_accuracy: float = 0.0
    calibration_error: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    auc_roc: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "accuracy": round(self.accuracy, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "win_rate": round(self.win_rate, 6),
            "directional_accuracy": round(self.directional_accuracy, 6),
            "calibration_error": round(self.calibration_error, 6),
            "f1_score": round(self.f1_score, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "auc_roc": round(self.auc_roc, 6),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrainingMetrics:
    """Training process metrics.

    Attributes:
        model_id: Model being trained.
        epoch: Current training epoch.
        total_epochs: Total epochs to train.
        train_loss: Current training loss.
        val_loss: Current validation loss.
        learning_rate: Current learning rate.
        gradient_norm: L2 norm of gradients.
        batch_loss: Loss for the current batch.
        epoch_time_seconds: Time for the last epoch.
        gpu_memory_used_mb: GPU memory usage during training.
        samples_per_second: Training throughput.
    """
    model_id: str = ""
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    batch_loss: float = 0.0
    epoch_time_seconds: float = 0.0
    gpu_memory_used_mb: float = 0.0
    samples_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "learning_rate": round(self.learning_rate, 8),
            "gradient_norm": round(self.gradient_norm, 6),
            "batch_loss": round(self.batch_loss, 6),
            "epoch_time_seconds": round(self.epoch_time_seconds, 2),
            "gpu_memory_used_mb": round(self.gpu_memory_used_mb, 1),
            "samples_per_second": round(self.samples_per_second, 1),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InferenceMetrics:
    """Inference serving metrics.

    Attributes:
        model_id: Model serving predictions.
        latency_ms: Prediction latency in milliseconds.
        throughput_rps: Requests per second.
        batch_size: Inference batch size.
        p50_latency_ms: 50th percentile latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        error_rate: Error rate as fraction.
        cache_hit_rate: Cache hit rate as fraction.
    """
    model_id: str = ""
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    batch_size: int = 1
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "latency_ms": round(self.latency_ms, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "batch_size": self.batch_size,
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_rate": round(self.error_rate, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 6),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeatureMetrics:
    """Feature health and importance metrics.

    Attributes:
        model_id: Model the features belong to.
        feature_name: Name of the feature.
        importance_score: Feature importance (SHAP, permutation, etc.).
        drift_score: PSI or KS drift score.
        missing_rate: Fraction of missing values.
        mean_value: Current mean of the feature.
        std_value: Current standard deviation.
        correlation_with_target: Pearson correlation with target.
    """
    model_id: str = ""
    feature_name: str = ""
    importance_score: float = 0.0
    drift_score: float = 0.0
    missing_rate: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0
    correlation_with_target: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "feature_name": self.feature_name,
            "importance_score": round(self.importance_score, 6),
            "drift_score": round(self.drift_score, 6),
            "missing_rate": round(self.missing_rate, 6),
            "mean_value": round(self.mean_value, 6),
            "std_value": round(self.std_value, 6),
            "correlation_with_target": round(self.correlation_with_target, 6),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class AIMetricsCollector:
    """Central AI metrics collection, aggregation, and export hub.

    Collects model performance, training, inference, and feature metrics.
    Supports:
    - Prometheus-compatible metric export
    - Sliding window aggregation (1m, 5m, 15m, 1h)
    - Per-model and global metric rollup
    - Metric history for trend analysis
    """

    def __init__(self, window_sizes: Optional[Dict[str, int]] = None,
                 max_history: int = 10000):
        """Initialize the metrics collector.

        Args:
            window_sizes: Dict mapping window name to size in seconds.
            max_history: Maximum number of historical records per metric type.
        """
        self.window_sizes = window_sizes or {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
        }
        self.max_history = max_history

        # Sliding windows for inference latency tracking
        # Structure: {model_id: deque of (timestamp, latency_ms)}
        self._latency_windows: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # Training loss history: {model_id: deque of (timestamp, loss)}
        self._training_losses: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # Gradient norm history: {model_id: deque of (timestamp, norm)}
        self._gradient_norms: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

        # Performance metrics history: {model_id: list of ModelPerformanceMetrics}
        self._performance_history: Dict[str, List[ModelPerformanceMetrics]] = defaultdict(list)

        # Feature metrics: {model_id: {feature_name: FeatureMetrics}}
        self._feature_metrics: Dict[str, Dict[str, FeatureMetrics]] = defaultdict(dict)

        # Counters
        self._inference_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._cache_hits: Dict[str, int] = defaultdict(int)
        self._cache_misses: Dict[str, int] = defaultdict(int)

        # Prometheus gauges/counters (lazily initialized)
        self._prom_metrics: Optional[Dict[str, Any]] = None

    # ========================================================================
    # Recording Methods
    # ========================================================================

    def record_inference(self, model_id: str, latency_ms: float,
                         batch_size: int = 1, is_error: bool = False,
                         cache_hit: bool = False) -> None:
        """Record an inference event.

        Args:
            model_id: Model identifier.
            latency_ms: Inference latency in milliseconds.
            batch_size: Inference batch size.
            is_error: Whether the inference resulted in an error.
            cache_hit: Whether the result was served from cache.
        """
        now = time.time()
        self._latency_windows[model_id].append((now, latency_ms))
        self._inference_counts[model_id] += 1

        if is_error:
            self._error_counts[model_id] += 1

        if cache_hit:
            self._cache_hits[model_id] += 1
        else:
            self._cache_misses[model_id] += 1

    def record_training_step(self, model_id: str, epoch: int,
                              train_loss: float, val_loss: float = 0.0,
                              learning_rate: float = 0.0,
                              gradient_norm: float = 0.0,
                              batch_loss: float = 0.0,
                              gpu_memory_mb: float = 0.0,
                              samples_per_second: float = 0.0) -> None:
        """Record a training step.

        Args:
            model_id: Model being trained.
            epoch: Current epoch.
            train_loss: Training loss.
            val_loss: Validation loss.
            learning_rate: Current learning rate.
            gradient_norm: L2 norm of gradients.
            batch_loss: Loss for current batch.
            gpu_memory_mb: GPU memory used.
            samples_per_second: Training throughput.
        """
        now = time.time()
        self._training_losses[model_id].append((now, train_loss))
        if gradient_norm > 0:
            self._gradient_norms[model_id].append((now, gradient_norm))

    def record_performance(self, metrics: ModelPerformanceMetrics) -> None:
        """Record model performance metrics.

        Args:
            metrics: ModelPerformanceMetrics instance.
        """
        model_id = metrics.model_id
        self._performance_history[model_id].append(metrics)
        # Trim history
        if len(self._performance_history[model_id]) > self.max_history:
            self._performance_history[model_id] = self._performance_history[model_id][-self.max_history:]

    def record_feature(self, metrics: FeatureMetrics) -> None:
        """Record feature metrics.

        Args:
            metrics: FeatureMetrics instance.
        """
        self._feature_metrics[metrics.model_id][metrics.feature_name] = metrics

    # ========================================================================
    # Aggregation Methods
    # ========================================================================

    def _aggregate_window(self, data: Deque[Tuple[float, float]],
                          window_seconds: int) -> List[Tuple[float, float]]:
        """Filter data to a sliding window.

        Args:
            data: Deque of (timestamp, value) tuples.
            window_seconds: Window size in seconds.

        Returns:
            List of tuples within the window.
        """
        cutoff = time.time() - window_seconds
        return [(ts, val) for ts, val in data if ts >= cutoff]

    def get_inference_metrics(self, model_id: str,
                               window: str = "5m") -> InferenceMetrics:
        """Get aggregated inference metrics for a model.

        Args:
            model_id: Model identifier.
            window: Aggregation window name.

        Returns:
            InferenceMetrics with aggregated values.
        """
        window_seconds = self.window_sizes.get(window, 300)
        latencies = self._aggregate_window(
            self._latency_windows.get(model_id, deque()), window_seconds
        )

        if not latencies:
            return InferenceMetrics(model_id=model_id)

        latency_values = np.array([v for _, v in latencies])
        total_requests = self._inference_counts.get(model_id, 0)
        total_errors = self._error_counts.get(model_id, 0)

        # Throughput: requests in window / window_seconds
        throughput = len(latency_values) / window_seconds

        # Cache hit rate
        hits = self._cache_hits.get(model_id, 0)
        misses = self._cache_misses.get(model_id, 0)
        cache_hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

        return InferenceMetrics(
            model_id=model_id,
            latency_ms=float(np.mean(latency_values)),
            throughput_rps=throughput,
            batch_size=1,
            p50_latency_ms=float(np.percentile(latency_values, 50)),
            p95_latency_ms=float(np.percentile(latency_values, 95)),
            p99_latency_ms=float(np.percentile(latency_values, 99)),
            error_rate=total_errors / total_requests if total_requests > 0 else 0.0,
            cache_hit_rate=cache_hit_rate,
        )

    def get_training_metrics(self, model_id: str, window: str = "1h") -> TrainingMetrics:
        """Get aggregated training metrics for a model.

        Args:
            model_id: Model identifier.
            window: Aggregation window name.

        Returns:
            TrainingMetrics with recent training data.
        """
        window_seconds = self.window_sizes.get(window, 3600)
        losses = self._aggregate_window(
            self._training_losses.get(model_id, deque()), window_seconds
        )
        grads = self._aggregate_window(
            self._gradient_norms.get(model_id, deque()), window_seconds
        )

        if not losses:
            return TrainingMetrics(model_id=model_id)

        loss_values = [v for _, v in losses]
        grad_values = [v for _, v in grads] if grads else [0.0]

        return TrainingMetrics(
            model_id=model_id,
            train_loss=float(loss_values[-1]),
            val_loss=0.0,
            gradient_norm=float(np.mean(grad_values)),
        )

    def get_feature_metrics(self, model_id: str) -> List[FeatureMetrics]:
        """Get all feature metrics for a model.

        Args:
            model_id: Model identifier.

        Returns:
            List of FeatureMetrics for each tracked feature.
        """
        return list(self._feature_metrics.get(model_id, {}).values())

    def get_latest_performance(self, model_id: str) -> Optional[ModelPerformanceMetrics]:
        """Get the latest performance metrics for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Most recent ModelPerformanceMetrics or None.
        """
        history = self._performance_history.get(model_id, [])
        return history[-1] if history else None

    # ========================================================================
    # Prometheus Export
    # ========================================================================

    def _init_prometheus(self) -> None:
        """Initialize Prometheus metric objects (lazy)."""
        if self._prom_metrics is not None:
            return

        try:
            from prometheus_client import Counter, Gauge, Histogram

            self._prom_metrics = {
                "inference_latency": Histogram(
                    "acms_ai_inference_latency_ms",
                    "AI model inference latency in milliseconds",
                    ["model_id"],
                ),
                "inference_requests": Counter(
                    "acms_ai_inference_requests_total",
                    "Total AI inference requests",
                    ["model_id"],
                ),
                "inference_errors": Counter(
                    "acms_ai_inference_errors_total",
                    "Total AI inference errors",
                    ["model_id"],
                ),
                "model_accuracy": Gauge(
                    "acms_ai_model_accuracy",
                    "Current model accuracy",
                    ["model_id"],
                ),
                "model_sharpe": Gauge(
                    "acms_ai_model_sharpe_ratio",
                    "Current model Sharpe ratio",
                    ["model_id"],
                ),
                "training_loss": Gauge(
                    "acms_ai_training_loss",
                    "Current training loss",
                    ["model_id"],
                ),
                "training_gradient_norm": Gauge(
                    "acms_ai_training_gradient_norm",
                    "Current gradient norm",
                    ["model_id"],
                ),
                "feature_drift_score": Gauge(
                    "acms_ai_feature_drift_score",
                    "Feature drift score (PSI)",
                    ["model_id", "feature_name"],
                ),
                "feature_importance": Gauge(
                    "acms_ai_feature_importance",
                    "Feature importance score",
                    ["model_id", "feature_name"],
                ),
            }
            logger.info("Prometheus metrics initialized")
        except ImportError:
            logger.warning("prometheus_client not installed; Prometheus export disabled")
            self._prom_metrics = {}

    def export_to_prometheus(self) -> None:
        """Export current metrics to Prometheus gauges/counters/histograms.

        Call this periodically to update Prometheus metrics for scraping.
        """
        self._init_prometheus()
        if not self._prom_metrics:
            return

        # Export inference metrics
        for model_id in self._inference_counts:
            inf_metrics = self.get_inference_metrics(model_id)
            self._prom_metrics["model_accuracy"].labels(model_id=model_id).set(
                inf_metrics.cache_hit_rate  # Reuse for latency tracking
            )

        # Export performance metrics
        for model_id, history in self._performance_history.items():
            if history:
                latest = history[-1]
                self._prom_metrics["model_accuracy"].labels(model_id=model_id).set(
                    latest.accuracy
                )
                self._prom_metrics["model_sharpe"].labels(model_id=model_id).set(
                    latest.sharpe_ratio
                )

        # Export training metrics
        for model_id in self._training_losses:
            losses = list(self._training_losses[model_id])
            if losses:
                _, latest_loss = losses[-1]
                self._prom_metrics["training_loss"].labels(model_id=model_id).set(latest_loss)

        for model_id in self._gradient_norms:
            grads = list(self._gradient_norms[model_id])
            if grads:
                _, latest_grad = grads[-1]
                self._prom_metrics["training_gradient_norm"].labels(
                    model_id=model_id
                ).set(latest_grad)

        # Export feature metrics
        for model_id, features in self._feature_metrics.items():
            for feat_name, feat in features.items():
                self._prom_metrics["feature_drift_score"].labels(
                    model_id=model_id, feature_name=feat_name
                ).set(feat.drift_score)
                self._prom_metrics["feature_importance"].labels(
                    model_id=model_id, feature_name=feat_name
                ).set(feat.importance_score)

    # ========================================================================
    # Summary and History
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics.

        Returns:
            Dict with per-model metric summaries.
        """
        model_ids = set()
        model_ids.update(self._inference_counts.keys())
        model_ids.update(self._training_losses.keys())
        model_ids.update(self._performance_history.keys())
        model_ids.update(self._feature_metrics.keys())

        summaries = {}
        for model_id in model_ids:
            inf = self.get_inference_metrics(model_id)
            train = self.get_training_metrics(model_id)
            perf = self.get_latest_performance(model_id)
            features = self.get_feature_metrics(model_id)

            summaries[model_id] = {
                "inference": inf.to_dict(),
                "training": train.to_dict(),
                "performance": perf.to_dict() if perf else None,
                "feature_count": len(features),
                "total_inferences": self._inference_counts.get(model_id, 0),
                "total_errors": self._error_counts.get(model_id, 0),
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model_count": len(model_ids),
            "models": summaries,
        }

    def get_loss_curve(self, model_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get training loss curve data for charting.

        Args:
            model_id: Model identifier.
            limit: Maximum data points.

        Returns:
            List of dicts with timestamp and loss value.
        """
        losses = list(self._training_losses.get(model_id, deque()))
        return [
            {"timestamp": ts, "loss": val}
            for ts, val in losses[-limit:]
        ]

    def get_gradient_curve(self, model_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get gradient norm curve data for charting.

        Args:
            model_id: Model identifier.
            limit: Maximum data points.

        Returns:
            List of dicts with timestamp and gradient norm.
        """
        grads = list(self._gradient_norms.get(model_id, deque()))
        return [
            {"timestamp": ts, "gradient_norm": val}
            for ts, val in grads[-limit:]
        ]
