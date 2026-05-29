"""
Online / Continuous Learning for ACMS
======================================

Production-grade online learning system that enables models to adapt
continuously to new market data without full retraining.

Features
--------
- OnlineLearner that incrementally updates models with streaming data
- ExperienceReplayBuffer with reservoir sampling to prevent catastrophic forgetting
- ConceptDriftDetector with DDM, EDDM, and ADWIN algorithms
- Multiple model update strategies (full retrain, incremental, ensemble update)
- Data quality filtering for online updates
- Flexible update scheduling (periodic, drift-triggered, performance-triggered)
- Model rollback mechanism when updates degrade performance
- A/B testing framework for evaluating model updates before deployment
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class DriftDetectionMethod(str, Enum):
    """Supported concept drift detection algorithms."""

    DDM = "ddm"
    EDDM = "eddm"
    ADWIN = "adwin"


class UpdateStrategy(str, Enum):
    """Model update strategies for online learning."""

    FULL_RETRAIN = "full_retrain"
    INCREMENTAL = "incremental"
    ENSEMBLE_UPDATE = "ensemble_update"
    FINE_TUNE = "fine_tune"


class UpdateTrigger(str, Enum):
    """Triggers for model updates."""

    PERIODIC = "periodic"
    DRIFT_TRIGGERED = "drift_triggered"
    PERFORMANCE_TRIGGERED = "performance_triggered"
    HYBRID = "hybrid"


@dataclass
class ModelUpdateResult:
    """Result of a model update operation.

    Attributes
    ----------
    success : bool
        Whether the update was applied successfully.
    strategy : UpdateStrategy
        The update strategy used.
    trigger : UpdateTrigger
        What triggered the update.
    val_loss_before : float
        Validation loss before update.
    val_loss_after : float
        Validation loss after update.
    improvement : float
        Relative improvement (positive = better).
    samples_used : int
        Number of samples used for the update.
    update_time_s : float
        Time taken for the update in seconds.
    rolled_back : bool
        Whether the update was rolled back.
    drift_detected : bool
        Whether concept drift was detected.
    metadata : Dict[str, Any]
        Additional metadata about the update.
    """

    success: bool = False
    strategy: UpdateStrategy = UpdateStrategy.INCREMENTAL
    trigger: UpdateTrigger = UpdateTrigger.PERIODIC
    val_loss_before: float = float("inf")
    val_loss_after: float = float("inf")
    improvement: float = 0.0
    samples_used: int = 0
    update_time_s: float = 0.0
    rolled_back: bool = False
    drift_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------


class ExperienceReplayBuffer:
    """Fixed-size replay buffer using reservoir sampling.

    Maintains a representative sample of past experiences to prevent
    catastrophic forgetting during online updates.

    Parameters
    ----------
    capacity : int
        Maximum number of samples to store.
    input_shape : Tuple[int, ...]
        Shape of input tensors.
    target_shape : Tuple[int, ...]
        Shape of target tensors.
    priority_sampling : bool
        Whether to use priority-based sampling (recency-weighted).
    """

    def __init__(
        self,
        capacity: int = 10000,
        input_shape: Optional[Tuple[int, ...]] = None,
        target_shape: Optional[Tuple[int, ...]] = None,
        priority_sampling: bool = True,
    ) -> None:
        self.capacity = capacity
        self.priority_sampling = priority_sampling
        self._inputs: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._priorities: List[float] = []
        self._count: int = 0
        self._input_shape = input_shape
        self._target_shape = target_shape

    @property
    def size(self) -> int:
        """Current number of samples in the buffer."""
        return len(self._inputs)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return len(self._inputs) >= self.capacity

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a single sample using reservoir sampling.

        When the buffer is full, new samples replace existing ones with
        probability proportional to 1/count, ensuring every sample has
        equal probability of being included.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Target tensor.
        """
        self._count += 1
        priority = 1.0 / self._count  # Newer samples have higher priority

        if len(self._inputs) < self.capacity:
            self._inputs.append(x.detach().cpu())
            self._targets.append(y.detach().cpu())
            self._priorities.append(priority)
        else:
            # Reservoir sampling: replace with probability capacity/count
            j = random.randint(0, self._count - 1)
            if j < self.capacity:
                self._inputs[j] = x.detach().cpu()
                self._targets[j] = y.detach().cpu()
                self._priorities[j] = priority

    def add_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Add a batch of samples.

        Parameters
        ----------
        x_batch : torch.Tensor
            Batch of input tensors.
        y_batch : torch.Tensor
            Batch of target tensors.
        """
        for i in range(x_batch.shape[0]):
            self.add(x_batch[i], y_batch[i])

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.
        device : Optional[torch.device]
            Device to move tensors to.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (inputs, targets) batch.
        """
        if len(self._inputs) == 0:
            raise ValueError("Replay buffer is empty")

        actual_size = min(batch_size, len(self._inputs))

        if self.priority_sampling and sum(self._priorities) > 0:
            # Priority sampling with recency bias
            total = sum(self._priorities)
            probs = [p / total for p in self._priorities]
            indices = np.random.choice(
                len(self._inputs), size=actual_size, replace=False, p=probs
            )
        else:
            indices = random.sample(range(len(self._inputs)), actual_size)

        x = torch.stack([self._inputs[i] for i in indices])
        y = torch.stack([self._targets[i] for i in indices])

        if device:
            x = x.to(device)
            y = y.to(device)

        return x, y

    def get_all(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all samples from the buffer."""
        if not self._inputs:
            raise ValueError("Replay buffer is empty")
        x = torch.stack(self._inputs)
        y = torch.stack(self._targets)
        if device:
            x, y = x.to(device), y.to(device)
        return x, y

    def clear(self) -> None:
        """Remove all samples from the buffer."""
        self._inputs.clear()
        self._targets.clear()
        self._priorities.clear()
        self._count = 0

    def get_dataset(self) -> TensorDataset:
        """Create a TensorDataset from the buffer contents."""
        if not self._inputs:
            raise ValueError("Replay buffer is empty")
        return TensorDataset(torch.stack(self._inputs), torch.stack(self._targets))


# ---------------------------------------------------------------------------
# Concept Drift Detectors
# ---------------------------------------------------------------------------


class BaseDriftDetector(ABC):
    """Abstract base class for concept drift detection algorithms."""

    def __init__(self) -> None:
        self.in_drift: bool = False
        self.in_warning: bool = False
        self.drift_count: int = 0
        self._sample_count: int = 0

    @abstractmethod
    def update(self, prediction_error: float) -> Tuple[bool, bool]:
        """Process a new prediction error and check for drift.

        Parameters
        ----------
        prediction_error : float
            Error value (0 = perfect, 1 = worst) for the current prediction.

        Returns
        -------
        Tuple[bool, bool]
            (drift_detected, warning_detected) flags.
        """

    def reset(self) -> None:
        """Reset the detector state."""
        self.in_drift = False
        self.in_warning = False
        self.drift_count = 0
        self._sample_count = 0


class DDMDetector(BaseDriftDetector):
    """DDM (Drift Detection Method) for concept drift detection.

    Monitors the error rate of a learning model and detects drift by
    tracking the minimum error rate (p_min) and its standard deviation
    (s_min). Drift is signaled when p + s > p_min + 2 * s_min, and
    warning when p + s > p_min + s_min.

    Reference: Gama et al., "Learning with Drift Detection" (2004)

    Parameters
    ----------
    warning_level : float
        Number of standard deviations for warning zone (default: 2).
    drift_level : float
        Number of standard deviations for drift detection (default: 3).
    min_samples : int
        Minimum samples before detection starts.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30,
    ) -> None:
        super().__init__()
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        self._p: float = 0.0
        self._s: float = 0.0
        self._p_min: float = float("inf")
        self._s_min: float = float("inf")

    def update(self, prediction_error: float) -> Tuple[bool, bool]:
        """Update DDM with a new prediction error."""
        self._sample_count += 1
        self._p += (prediction_error - self._p) / self._sample_count
        self._s = math.sqrt(self._p * (1 - self._p) / max(1, self._sample_count))

        if self._sample_count < self.min_samples:
            return False, False

        if self._p + self._s < self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s

        drift_detected = False
        warning_detected = False

        if self._p + self._s > self._p_min + self.drift_level * self._s_min:
            drift_detected = True
            self.in_drift = True
            self.drift_count += 1
            self.reset_stats()
        elif self._p + self._s > self._p_min + self.warning_level * self._s_min:
            warning_detected = True
            self.in_warning = True
        else:
            self.in_warning = False

        self.in_drift = drift_detected
        return drift_detected, warning_detected

    def reset_stats(self) -> None:
        """Reset running statistics (called after drift detection)."""
        self._p = 0.0
        self._s = 0.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._sample_count = 0

    def reset(self) -> None:
        super().reset()
        self.reset_stats()


class EDDMDetector(BaseDriftDetector):
    """EDDM (Early Drift Detection Method) for concept drift detection.

    Instead of monitoring error rate directly, EDDM monitors the distance
    between classification errors. It detects drift when the distance
    between errors decreases significantly.

    Reference: Baena-García et al., "Early Drift Detection Method" (2006)

    Parameters
    ----------
    alpha : float
        Weight for updating running averages (default: 0.2).
    beta : float
        Number of standard deviations for warning (default: 0.95).
    gamma : float
        Number of standard deviations for drift (default: 0.90).
    min_samples : int
        Minimum error pairs before detection starts.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.95,
        gamma: float = 0.90,
        min_samples: int = 30,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_samples = min_samples
        self._n_errors: int = 0
        self._last_error_idx: int = 0
        self._mean_dist: float = 0.0
        self._var_dist: float = 0.0
        self._max_metric: float = 0.0
        self._sample_idx: int = 0

    def update(self, prediction_error: float) -> Tuple[bool, bool]:
        """Update EDDM with a new prediction error."""
        self._sample_idx += 1
        is_error = prediction_error > 0.5

        if not is_error:
            return False, False

        if self._n_errors == 0:
            self._n_errors = 1
            self._last_error_idx = self._sample_idx
            return False, False

        distance = self._sample_idx - self._last_error_idx
        self._last_error_idx = self._sample_idx
        self._n_errors += 1

        # Update running mean and variance of distance
        old_mean = self._mean_dist
        self._mean_dist += self.alpha * (distance - self._mean_dist)
        self._var_dist += self.alpha * (distance - old_mean) * (distance - self._mean_dist)

        if self._n_errors < self.min_samples:
            return False, False

        std_dist = math.sqrt(max(0, self._var_dist))
        metric = self._mean_dist + 2 * std_dist

        if metric > self._max_metric:
            self._max_metric = metric

        drift_detected = False
        warning_detected = False

        if metric < self._max_metric * self.gamma:
            drift_detected = True
            self.in_drift = True
            self.drift_count += 1
            self._reset_eddm()
        elif metric < self._max_metric * self.beta:
            warning_detected = True
            self.in_warning = True
        else:
            self.in_warning = False

        self.in_drift = drift_detected
        return drift_detected, warning_detected

    def _reset_eddm(self) -> None:
        """Reset EDDM statistics after drift detection."""
        self._n_errors = 0
        self._mean_dist = 0.0
        self._var_dist = 0.0
        self._max_metric = 0.0

    def reset(self) -> None:
        super().reset()
        self._reset_eddm()
        self._sample_idx = 0


class ADWINDetector(BaseDriftDetector):
    """ADWIN (ADaptive WINdowing) for concept drift detection.

    Maintains a variable-length window of recent error values and
    detects drift when two sub-windows have significantly different means.

    Reference: Bifet & Gavaldà, "Learning from Time-Changing Data" (2007)

    Parameters
    ----------
    delta : float
        Confidence parameter (lower = less sensitive to drift).
    max_buckets : int
        Maximum number of buckets per row in the histogram.
    min_window : int
        Minimum window size before drift can be detected.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 5,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self._window: Deque[float] = deque()
        self._total: float = 0.0
        self._total_sq: float = 0.0
        self._variance: float = 0.0

    def update(self, prediction_error: float) -> Tuple[bool, bool]:
        """Update ADWIN with a new prediction error."""
        self._sample_count += 1
        self._window.append(prediction_error)
        self._total += prediction_error
        self._total_sq += prediction_error ** 2

        n = len(self._window)
        if n > 1:
            self._variance = (self._total_sq - self._total ** 2 / n) / (n - 1)

        if n < self.min_window:
            return False, False

        # Check for drift by testing sub-window splits
        drift_detected = self._check_drift()

        warning_detected = False
        if drift_detected:
            self.in_drift = True
            self.drift_count += 1
        elif n >= self.min_window * 2:
            # Check warning with relaxed threshold
            warning_detected = self._check_warning()

        self.in_warning = warning_detected and not drift_detected
        return drift_detected, warning_detected

    def _check_drift(self) -> bool:
        """Check if any sub-window split indicates drift."""
        n = len(self._window)
        if n < self.min_window * 2:
            return False

        # Evaluate split points at regular intervals
        step = max(1, n // 20)
        for split in range(self.min_window, n - self.min_window + 1, step):
            left = list(self._window)[:split]
            right = list(self._window)[split:]

            n0, n1 = len(left), len(right)
            u0 = sum(left) / n0
            u1 = sum(right) / n1

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt((2.0 / m) * math.log(2.0 / self.delta) * self._variance)

            if abs(u0 - u1) >= eps:
                # Remove the older sub-window
                for _ in range(split):
                    val = self._window.popleft()
                    self._total -= val
                    self._total_sq -= val ** 2
                return True

        return False

    def _check_warning(self) -> bool:
        """Check for warning zone with a relaxed delta."""
        n = len(self._window)
        step = max(1, n // 20)
        warning_delta = self.delta * 10  # Relaxed threshold

        for split in range(self.min_window, n - self.min_window + 1, step):
            left = list(self._window)[:split]
            right = list(self._window)[split:]

            n0, n1 = len(left), len(right)
            u0 = sum(left) / n0
            u1 = sum(right) / n1

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt((2.0 / m) * math.log(2.0 / warning_delta) * self._variance)

            if abs(u0 - u1) >= eps:
                return True

        return False

    def reset(self) -> None:
        super().reset()
        self._window.clear()
        self._total = 0.0
        self._total_sq = 0.0
        self._variance = 0.0


class ConceptDriftDetector:
    """Unified concept drift detection interface.

    Wraps one or more drift detection algorithms and provides a single
    interface for monitoring prediction errors.

    Parameters
    ----------
    methods : List[DriftDetectionMethod]
        Detection methods to use.
    consensus : str
        How to combine results from multiple detectors:
        - 'any': drift if any detector signals
        - 'majority': drift if more than half signal
        - 'all': drift only if all signal
    **kwargs
        Parameters forwarded to individual detectors.
    """

    def __init__(
        self,
        methods: Optional[List[DriftDetectionMethod]] = None,
        consensus: str = "any",
        **kwargs: Any,
    ) -> None:
        methods = methods or [DriftDetectionMethod.DDM]
        self.consensus = consensus
        self._detectors: Dict[str, BaseDriftDetector] = {}

        for method in methods:
            if method == DriftDetectionMethod.DDM:
                self._detectors["ddm"] = DDMDetector(**kwargs)
            elif method == DriftDetectionMethod.EDDM:
                self._detectors["eddm"] = EDDMDetector(**kwargs)
            elif method == DriftDetectionMethod.ADWIN:
                self._detectors["adwin"] = ADWINDetector(**kwargs)

    def update(self, prediction_error: float) -> Tuple[bool, bool]:
        """Feed a new prediction error to all detectors.

        Returns
        -------
        Tuple[bool, bool]
            (drift_detected, warning_detected) based on consensus.
        """
        drifts = []
        warnings = []

        for name, detector in self._detectors.items():
            d, w = detector.update(prediction_error)
            drifts.append(d)
            warnings.append(w)
            if d:
                logger.info(f"Drift detected by {name}")

        if self.consensus == "any":
            return any(drifts), any(warnings)
        elif self.consensus == "majority":
            return (
                sum(drifts) > len(drifts) / 2,
                sum(warnings) > len(warnings) / 2,
            )
        elif self.consensus == "all":
            return all(drifts), all(warnings)
        else:
            raise ValueError(f"Unknown consensus mode: {self.consensus}")

    @property
    def drift_count(self) -> int:
        """Total drift detections across all methods."""
        return sum(d.drift_count for d in self._detectors.values())

    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self._detectors.values():
            detector.reset()


# ---------------------------------------------------------------------------
# Data Quality Filter
# ---------------------------------------------------------------------------


class DataQualityFilter:
    """Filters incoming data for quality before using it for online updates.

    Ensures that only clean, reasonable data is used for model updates,
    preventing corruption from bad data sources, outliers, or anomalies.

    Parameters
    ----------
    max_value : float
        Maximum absolute value allowed in features/targets.
    min_value : float
        Minimum absolute value (below is considered near-zero noise).
    nan_threshold : float
        Maximum fraction of NaN values allowed in a sample.
    outlier_std : float
        Number of standard deviations beyond which values are considered outliers.
    max_gap_ratio : float
        Maximum allowed ratio of consecutive value gaps.
    """

    def __init__(
        self,
        max_value: float = 1e6,
        min_value: float = -1e6,
        nan_threshold: float = 0.1,
        outlier_std: float = 5.0,
        max_gap_ratio: float = 0.5,
    ) -> None:
        self.max_value = max_value
        self.min_value = min_value
        self.nan_threshold = nan_threshold
        self.outlier_std = outlier_std
        self.max_gap_ratio = max_gap_ratio
        self._running_mean: float = 0.0
        self._running_var: float = 1.0
        self._count: int = 0
        self._rejected: int = 0

    @property
    def rejection_rate(self) -> float:
        """Fraction of samples rejected so far."""
        return self._rejected / max(1, self._count)

    def is_valid(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """Check whether a single sample passes quality filters.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        y : torch.Tensor
            Target values.

        Returns
        -------
        bool
            True if the sample is valid.
        """
        self._count += 1

        # NaN check
        nan_frac_x = torch.isnan(x).float().mean().item()
        nan_frac_y = torch.isnan(y).float().mean().item()
        if nan_frac_x > self.nan_threshold or nan_frac_y > self.nan_threshold:
            self._rejected += 1
            return False

        # Value range check
        if x.abs().max().item() > self.max_value or y.abs().max().item() > self.max_value:
            self._rejected += 1
            return False
        if x.min().item() < self.min_value or y.min().item() < self.min_value:
            self._rejected += 1
            return False

        # Outlier check using running statistics
        self._update_stats(x)
        if self._count > 30:
            z_score = (x.mean().item() - self._running_mean) / max(math.sqrt(self._running_var), 1e-8)
            if abs(z_score) > self.outlier_std:
                self._rejected += 1
                return False

        return True

    def filter_batch(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter a batch of samples, keeping only valid ones.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Filtered (inputs, targets).
        """
        valid_indices = []
        for i in range(x_batch.shape[0]):
            if self.is_valid(x_batch[i], y_batch[i]):
                valid_indices.append(i)

        if not valid_indices:
            logger.warning("All samples in batch rejected by quality filter")
            return x_batch[:0], y_batch[:0]

        return x_batch[valid_indices], y_batch[valid_indices]

    def _update_stats(self, x: torch.Tensor) -> None:
        """Update running mean and variance."""
        val = x.mean().item()
        self._count += 1
        old_mean = self._running_mean
        self._running_mean += (val - old_mean) / self._count
        self._running_var += (val - old_mean) * (val - self._running_mean)


# ---------------------------------------------------------------------------
# Online Learner
# ---------------------------------------------------------------------------


class OnlineLearner:
    """Continuous learning system that updates models with new streaming data.

    Integrates experience replay, drift detection, quality filtering,
    and multiple update strategies with rollback support and A/B testing.

    Parameters
    ----------
    model : nn.Module
        The model to update online.
    loss_fn : Callable
        Loss function for training.
    optimizer_class : Type[torch.optim.Optimizer]
        Optimizer class to instantiate.
    optimizer_kwargs : Dict[str, Any]
        Keyword arguments for the optimizer.
    device : str
        Compute device.
    replay_buffer_capacity : int
        Size of the experience replay buffer.
    drift_detector : Optional[ConceptDriftDetector]
        Drift detector instance; None creates a default one.
    update_strategy : UpdateStrategy
        Default update strategy.
    update_trigger : UpdateTrigger
        Default update trigger mode.
    update_interval : int
        Steps between periodic updates.
    performance_threshold : float
        Validation loss threshold for performance-triggered updates.
    performance_patience : int
        Steps of degradation before triggering update.
    rollback_threshold : float
        Maximum allowed performance degradation before rollback.
    ab_test_enabled : bool
        Whether to A/B test model updates before applying.
    ab_test_samples : int
        Number of samples for A/B testing.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        replay_buffer_capacity: int = 10000,
        drift_detector: Optional[ConceptDriftDetector] = None,
        update_strategy: UpdateStrategy = UpdateStrategy.INCREMENTAL,
        update_trigger: UpdateTrigger = UpdateTrigger.HYBRID,
        update_interval: int = 100,
        performance_threshold: float = 0.1,
        performance_patience: int = 50,
        rollback_threshold: float = 0.2,
        ab_test_enabled: bool = False,
        ab_test_samples: int = 500,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-5}
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        self.update_strategy = update_strategy
        self.update_trigger = update_trigger
        self.update_interval = update_interval
        self.performance_threshold = performance_threshold
        self.performance_patience = performance_patience
        self.rollback_threshold = rollback_threshold

        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_capacity)

        # Drift detection
        self.drift_detector = drift_detector or ConceptDriftDetector(
            methods=[DriftDetectionMethod.DDM, DriftDetectionMethod.ADWIN],
            consensus="any",
        )

        # Quality filter
        self.quality_filter = DataQualityFilter()

        # A/B testing
        self.ab_test_enabled = ab_test_enabled
        self.ab_test_samples = ab_test_samples
        self._shadow_model: Optional[nn.Module] = None

        # State tracking
        self._step_count: int = 0
        self._steps_since_update: int = 0
        self._steps_since_improvement: int = 0
        self._best_val_loss: float = float("inf")
        self._current_val_loss: float = float("inf")
        self._update_history: List[ModelUpdateResult] = []
        self._checkpoint_stack: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve the target compute device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def process_sample(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        val_loader: Optional[DataLoader] = None,
    ) -> Optional[ModelUpdateResult]:
        """Process a single new data sample.

        Adds it to the replay buffer, checks for drift, and triggers
        model updates based on the configured strategy.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        y : torch.Tensor
            Target values.
        val_loader : Optional[DataLoader]
            Validation loader for checking model performance.

        Returns
        -------
        Optional[ModelUpdateResult]
            Result of any update that was triggered, or None.
        """
        self._step_count += 1
        self._steps_since_update += 1

        # Quality filter
        if not self.quality_filter.is_valid(x, y):
            return None

        # Add to replay buffer
        self.replay_buffer.add(x, y)

        # Compute prediction error for drift detection
        with torch.no_grad():
            self.model.eval()
            x_dev = x.unsqueeze(0).to(self.device) if x.dim() == 1 else x.to(self.device)
            y_dev = y.unsqueeze(0).to(self.device) if y.dim() == 0 else y.to(self.device)
            pred = self.model(x_dev)
            error = self.loss_fn(pred, y_dev).item()

        # Normalize error to [0, 1] for drift detector
        normalized_error = min(1.0, error / max(self._current_val_loss, 1e-8))
        drift_detected, warning = self.drift_detector.update(normalized_error)

        # Determine if update should be triggered
        should_update = False
        trigger = UpdateTrigger.PERIODIC

        if self.update_trigger == UpdateTrigger.PERIODIC:
            if self._steps_since_update >= self.update_interval:
                should_update = True
                trigger = UpdateTrigger.PERIODIC

        elif self.update_trigger == UpdateTrigger.DRIFT_TRIGGERED:
            if drift_detected:
                should_update = True
                trigger = UpdateTrigger.DRIFT_TRIGGERED

        elif self.update_trigger == UpdateTrigger.PERFORMANCE_TRIGGERED:
            if self._steps_since_improvement >= self.performance_patience:
                should_update = True
                trigger = UpdateTrigger.PERFORMANCE_TRIGGERED

        elif self.update_trigger == UpdateTrigger.HYBRID:
            if drift_detected:
                should_update = True
                trigger = UpdateTrigger.DRIFT_TRIGGERED
            elif self._steps_since_update >= self.update_interval:
                should_update = True
                trigger = UpdateTrigger.PERIODIC
            elif self._steps_since_improvement >= self.performance_patience:
                should_update = True
                trigger = UpdateTrigger.PERFORMANCE_TRIGGERED

        if should_update and val_loader is not None:
            result = self._perform_update(val_loader, trigger, drift_detected)
            return result

        return None

    def process_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        val_loader: Optional[DataLoader] = None,
    ) -> Optional[ModelUpdateResult]:
        """Process a batch of new data samples."""
        x_filtered, y_filtered = self.quality_filter.filter_batch(x_batch, y_batch)

        if x_filtered.shape[0] == 0:
            return None

        self.replay_buffer.add_batch(x_filtered, y_filtered)

        # Compute average error for drift detection
        with torch.no_grad():
            self.model.eval()
            x_dev = x_filtered.to(self.device)
            y_dev = y_filtered.to(self.device)
            pred = self.model(x_dev)
            batch_error = self.loss_fn(pred, y_dev).item()

        normalized_error = min(1.0, batch_error / max(self._current_val_loss, 1e-8))
        drift_detected, _ = self.drift_detector.update(normalized_error)

        self._step_count += x_filtered.shape[0]
        self._steps_since_update += x_filtered.shape[0]

        should_update = False
        trigger = UpdateTrigger.PERIODIC

        if self.update_trigger in (UpdateTrigger.PERIODIC, UpdateTrigger.HYBRID):
            if self._steps_since_update >= self.update_interval:
                should_update = True
                trigger = UpdateTrigger.PERIODIC

        if self.update_trigger in (UpdateTrigger.DRIFT_TRIGGERED, UpdateTrigger.HYBRID):
            if drift_detected:
                should_update = True
                trigger = UpdateTrigger.DRIFT_TRIGGERED

        if should_update and val_loader is not None:
            return self._perform_update(val_loader, trigger, drift_detected)

        return None

    def _perform_update(
        self,
        val_loader: DataLoader,
        trigger: UpdateTrigger,
        drift_detected: bool,
    ) -> ModelUpdateResult:
        """Execute a model update with the configured strategy.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data for evaluating the update.
        trigger : UpdateTrigger
            What triggered this update.
        drift_detected : bool
            Whether drift was detected.

        Returns
        -------
        ModelUpdateResult
            Detailed result of the update operation.
        """
        start_time = time.time()
        result = ModelUpdateResult(
            strategy=self.update_strategy,
            trigger=trigger,
            drift_detected=drift_detected,
        )

        # Measure validation loss before update
        result.val_loss_before = self._evaluate(val_loader)

        # Save checkpoint for potential rollback
        self._save_rollback_checkpoint()

        # Perform the update based on strategy
        samples_used = 0
        try:
            if self.update_strategy == UpdateStrategy.FULL_RETRAIN:
                samples_used = self._full_retrain(val_loader)
            elif self.update_strategy == UpdateStrategy.INCREMENTAL:
                samples_used = self._incremental_update()
            elif self.update_strategy == UpdateStrategy.ENSEMBLE_UPDATE:
                samples_used = self._ensemble_update()
            elif self.update_strategy == UpdateStrategy.FINE_TUNE:
                samples_used = self._fine_tune_update()

            result.success = True
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            result.success = False
            result.metadata["error"] = str(e)

        # Evaluate after update
        result.val_loss_after = self._evaluate(val_loader) if result.success else result.val_loss_before
        result.improvement = result.val_loss_before - result.val_loss_after
        result.samples_used = samples_used
        result.update_time_s = time.time() - start_time

        # Track improvement
        if result.improvement > 0:
            self._steps_since_improvement = 0
            self._best_val_loss = min(self._best_val_loss, result.val_loss_after)
        else:
            self._steps_since_improvement += 1

        self._current_val_loss = result.val_loss_after

        # A/B test if enabled
        if self.ab_test_enabled and result.success:
            ab_passed = self._ab_test_update()
            if not ab_passed:
                logger.info("A/B test failed — rolling back update")
                self._rollback()
                result.rolled_back = True
                result.val_loss_after = result.val_loss_before
                result.improvement = 0.0
        else:
            # Check rollback condition
            if result.success and result.improvement < -self.rollback_threshold:
                logger.warning(
                    f"Update degraded performance by {abs(result.improvement):.4f} "
                    f"(threshold: {self.rollback_threshold}). Rolling back."
                )
                self._rollback()
                result.rolled_back = True
                result.val_loss_after = result.val_loss_before
                result.improvement = 0.0

        self._steps_since_update = 0
        self._update_history.append(result)

        logger.info(
            f"Model update: strategy={result.strategy.value}, "
            f"trigger={result.trigger.value}, "
            f"improvement={result.improvement:.6f}, "
            f"rolled_back={result.rolled_back}"
        )

        return result

    def _incremental_update(self, n_steps: int = 10, batch_size: int = 32) -> int:
        """Perform an incremental update using recent and replay data."""
        self.model.train()
        samples_used = 0

        for _ in range(n_steps):
            if self.replay_buffer.size < batch_size:
                break

            x, y = self.replay_buffer.sample(batch_size, device=self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            samples_used += x.shape[0]

        return samples_used

    def _full_retrain(self, val_loader: DataLoader, epochs: int = 5) -> int:
        """Full retrain on all replay buffer data."""
        if self.replay_buffer.size < 10:
            return 0

        dataset = self.replay_buffer.get_dataset()
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        samples_used = 0
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
                samples_used += x.shape[0]

        return samples_used

    def _ensemble_update(self) -> int:
        """Update an ensemble member with recent data."""
        # Simple ensemble update: fine-tune the current model on replay data
        return self._fine_tune_update(learning_rate=self.optimizer_kwargs.get("lr", 1e-4) * 0.5)

    def _fine_tune_update(self, n_steps: int = 5, learning_rate: Optional[float] = None) -> int:
        """Fine-tune with a lower learning rate."""
        if learning_rate:
            for pg in self.optimizer.param_groups:
                old_lr = pg["lr"]
                pg["lr"] = learning_rate

        result = self._incremental_update(n_steps=n_steps)

        if learning_rate:
            for pg in self.optimizer.param_groups:
                pg["lr"] = old_lr

        return result

    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x, y = batch["input"].to(self.device), batch["target"].to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

        return total_loss / max(total_samples, 1)

    def _save_rollback_checkpoint(self) -> None:
        """Save model state for potential rollback."""
        checkpoint = {
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "val_loss": self._current_val_loss,
            "step_count": self._step_count,
        }
        self._checkpoint_stack.append(checkpoint)
        # Keep only the last 5 checkpoints
        if len(self._checkpoint_stack) > 5:
            self._checkpoint_stack.pop(0)

    def _rollback(self) -> None:
        """Rollback to the previous checkpoint."""
        if not self._checkpoint_stack:
            logger.warning("No checkpoint available for rollback")
            return

        checkpoint = self._checkpoint_stack.pop()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_val_loss = checkpoint["val_loss"]
        logger.info("Model rolled back to previous checkpoint")

    def _ab_test_update(self) -> bool:
        """A/B test the updated model against the previous version.

        Returns
        -------
        bool
            True if the update should be kept (new model is at least as good).
        """
        if not self._checkpoint_stack:
            return True

        # Create shadow model from previous checkpoint
        old_state = self._checkpoint_stack[-1]["model_state_dict"]
        self._shadow_model = copy.deepcopy(self.model)
        self._shadow_model.load_state_dict(old_state)
        self._shadow_model.to(self.device)
        self._shadow_model.eval()

        self.model.eval()

        # Compare on recent replay data
        if self.replay_buffer.size < 50:
            return True

        x, y = self.replay_buffer.sample(min(self.ab_test_samples, self.replay_buffer.size), device=self.device)

        with torch.no_grad():
            new_loss = self.loss_fn(self.model(x), y).item()
            old_loss = self.loss_fn(self._shadow_model(x), y).item()

        # Allow up to 5% degradation (statistical noise)
        passed = new_loss <= old_loss * 1.05
        logger.info(
            f"A/B test: new_loss={new_loss:.6f}, old_loss={old_loss:.6f}, "
            f"passed={passed}"
        )
        return passed

    @property
    def update_history(self) -> List[ModelUpdateResult]:
        """History of all model updates."""
        return self._update_history

    def get_stats(self) -> Dict[str, Any]:
        """Get online learning statistics."""
        return {
            "step_count": self._step_count,
            "steps_since_update": self._steps_since_update,
            "current_val_loss": self._current_val_loss,
            "best_val_loss": self._best_val_loss,
            "replay_buffer_size": self.replay_buffer.size,
            "drift_count": self.drift_detector.drift_count,
            "update_count": len(self._update_history),
            "rollback_count": sum(1 for r in self._update_history if r.rolled_back),
            "data_rejection_rate": self.quality_filter.rejection_rate,
        }
