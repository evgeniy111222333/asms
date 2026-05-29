"""
Walk-Forward Training Pipeline for ACMS
=========================================

Proper time-series model evaluation using walk-forward analysis, which
prevents look-ahead bias and provides realistic out-of-sample performance
estimates.

Features
--------
- WalkForwardTrainer for rigorous time-series model evaluation
- Expanding window and sliding window strategies
- Multiple retraining triggers (time-based, performance-based, drift-based)
- WalkForwardResult with comprehensive out-of-sample metrics
- Model stability analysis across windows
- Automatic hyperparameter tuning per window
- Overlap handling and gap management
"""

from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class WindowStrategy(str, Enum):
    """Window strategy for walk-forward analysis."""

    EXPANDING = "expanding"      # Train window grows each step
    SLIDING = "sliding"          # Fixed-size train window slides forward
    ANCHORED = "anchored"        # Start point fixed, end point slides


class RetrainingTrigger(str, Enum):
    """Triggers for model retraining between windows."""

    ALWAYS = "always"                           # Retrain every window
    TIME_BASED = "time_based"                   # Retrain after N time steps
    PERFORMANCE_BASED = "performance_based"     # Retrain if performance drops
    DRIFT_BASED = "drift_based"                 # Retrain on concept drift
    ADAPTIVE = "adaptive"                       # Combine performance + drift


@dataclass
class WindowMetrics:
    """Metrics for a single walk-forward window.

    Attributes
    ----------
    window_index : int
        Zero-based window index.
    train_start : int
        Start index of training data.
    train_end : int
        End index of training data.
    test_start : int
        Start index of test data.
    test_end : int
        End index of test data.
    train_samples : int
        Number of training samples.
    test_samples : int
        Number of test samples.
    train_loss : float
        Training loss in this window.
    test_loss : float
        Out-of-sample test loss.
    test_metrics : Dict[str, float]
        Additional test metrics (MAE, RMSE, directional accuracy, etc.).
    retrained : bool
        Whether the model was retrained in this window.
    training_time_s : float
        Time taken for training/evaluation.
    """

    window_index: int = 0
    train_start: int = 0
    train_end: int = 0
    test_start: int = 0
    test_end: int = 0
    train_samples: int = 0
    test_samples: int = 0
    train_loss: float = 0.0
    test_loss: float = 0.0
    test_metrics: Dict[str, float] = field(default_factory=dict)
    retrained: bool = True
    training_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "window_index": self.window_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "test_metrics": self.test_metrics,
            "retrained": self.retrained,
            "training_time_s": self.training_time_s,
        }


@dataclass
class WalkForwardResult:
    """Comprehensive results from a walk-forward analysis.

    Attributes
    ----------
    window_metrics : List[WindowMetrics]
        Metrics for each window.
    oos_loss : float
        Average out-of-sample loss across all windows.
    oos_metrics : Dict[str, float]
        Average out-of-sample metrics across all windows.
    total_train_time_s : float
        Total training time across all windows.
    n_windows : int
        Number of walk-forward windows.
    n_retrains : int
        Number of actual retrainings performed.
    stability_score : float
        Model stability across windows (lower variance = higher score).
    best_window : int
        Index of the best-performing window.
    worst_window : int
        Index of the worst-performing window.
    model_params_per_window : List[Dict[str, Any]]
        Best hyperparameters found per window (if auto_tuning enabled).
    """

    window_metrics: List[WindowMetrics] = field(default_factory=list)
    oos_loss: float = 0.0
    oos_metrics: Dict[str, float] = field(default_factory=dict)
    total_train_time_s: float = 0.0
    n_windows: int = 0
    n_retrains: int = 0
    stability_score: float = 0.0
    best_window: int = 0
    worst_window: int = 0
    model_params_per_window: List[Dict[str, Any]] = field(default_factory=list)

    def compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics from window results."""
        if not self.window_metrics:
            return

        self.n_windows = len(self.window_metrics)
        self.n_retrains = sum(1 for w in self.window_metrics if w.retrained)

        # Average OOS loss
        losses = [w.test_loss for w in self.window_metrics]
        self.oos_loss = float(np.mean(losses))

        # Average OOS metrics
        all_metrics: Dict[str, List[float]] = {}
        for w in self.window_metrics:
            for k, v in w.test_metrics.items():
                all_metrics.setdefault(k, []).append(v)

        self.oos_metrics = {
            k: float(np.mean(v)) for k, v in all_metrics.items()
        }

        # Stability: inverse of coefficient of variation
        if self.oos_loss > 0:
            cv = float(np.std(losses) / max(self.oos_loss, 1e-8))
            self.stability_score = max(0.0, 1.0 - cv)
        else:
            self.stability_score = 1.0

        # Best and worst windows
        self.best_window = int(np.argmin(losses))
        self.worst_window = int(np.argmax(losses))

        # Total training time
        self.total_train_time_s = sum(w.training_time_s for w in self.window_metrics)

    def get_rolling_oos_loss(self) -> List[float]:
        """Get rolling cumulative average OOS loss."""
        losses = [w.test_loss for w in self.window_metrics]
        rolling = []
        for i in range(1, len(losses) + 1):
            rolling.append(float(np.mean(losses[:i])))
        return rolling

    def get_loss_trend(self) -> str:
        """Determine if OOS loss is improving, degrading, or stable."""
        if len(self.window_metrics) < 3:
            return "insufficient_data"

        losses = [w.test_loss for w in self.window_metrics]
        mid = len(losses) // 2
        first_half = float(np.mean(losses[:mid]))
        second_half = float(np.mean(losses[mid:]))

        relative_change = (second_half - first_half) / max(abs(first_half), 1e-8)

        if relative_change < -0.05:
            return "improving"
        elif relative_change > 0.05:
            return "degrading"
        else:
            return "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "oos_loss": self.oos_loss,
            "oos_metrics": self.oos_metrics,
            "total_train_time_s": self.total_train_time_s,
            "n_windows": self.n_windows,
            "n_retrains": self.n_retrains,
            "stability_score": self.stability_score,
            "best_window": self.best_window,
            "worst_window": self.worst_window,
            "loss_trend": self.get_loss_trend(),
            "window_metrics": [w.to_dict() for w in self.window_metrics],
        }


# ---------------------------------------------------------------------------
# Window Splitter
# ---------------------------------------------------------------------------


class WindowSplitter:
    """Generates train/test index splits for walk-forward analysis.

    Parameters
    ----------
    strategy : WindowStrategy
        Windowing strategy.
    initial_train_size : int
        Size of the initial training window.
    test_size : int
        Size of each test window.
    step_size : int
        Number of samples to advance between windows.
    max_train_size : Optional[int]
        Maximum training window size for sliding/anchored strategies.
    gap : int
        Gap between train and test sets to prevent data leakage.
    min_train_size : int
        Minimum training samples required.
    """

    def __init__(
        self,
        strategy: WindowStrategy = WindowStrategy.EXPANDING,
        initial_train_size: int = 500,
        test_size: int = 100,
        step_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        gap: int = 0,
        min_train_size: int = 100,
    ) -> None:
        self.strategy = strategy
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.max_train_size = max_train_size
        self.gap = gap
        self.min_train_size = min_train_size

    def split(self, n_samples: int) -> List[Tuple[range, range]]:
        """Generate train/test index ranges for walk-forward analysis.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        List[Tuple[range, range]]
            List of (train_range, test_range) tuples.
        """
        splits = []
        train_end = self.initial_train_size

        while train_end + self.gap + self.test_size <= n_samples:
            # Determine training window
            if self.strategy == WindowStrategy.EXPANDING:
                train_start = 0
                current_train_end = train_end
            elif self.strategy == WindowStrategy.SLIDING:
                window_size = self.max_train_size or self.initial_train_size
                train_start = max(0, train_end - window_size)
                current_train_end = train_end
            elif self.strategy == WindowStrategy.ANCHORED:
                train_start = 0
                current_train_end = train_end
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Enforce minimum training size
            if current_train_end - train_start < self.min_train_size:
                train_end += self.step_size
                continue

            # Apply max train size for expanding/anchored
            if self.max_train_size and self.strategy in (
                WindowStrategy.EXPANDING, WindowStrategy.ANCHORED
            ):
                train_start = max(train_start, current_train_end - self.max_train_size)

            # Test window (with gap)
            test_start = current_train_end + self.gap
            test_end = min(test_start + self.test_size, n_samples)

            if test_end <= test_start:
                break

            train_range = range(train_start, current_train_end)
            test_range = range(test_start, test_end)

            splits.append((train_range, test_range))

            train_end += self.step_size

        logger.info(
            f"Walk-forward split: {len(splits)} windows from {n_samples} samples "
            f"(strategy={self.strategy.value}, train={self.initial_train_size}, "
            f"test={self.test_size}, step={self.step_size}, gap={self.gap})"
        )

        return splits

    def get_n_windows(self, n_samples: int) -> int:
        """Get the number of windows without generating splits."""
        return len(self.split(n_samples))


# ---------------------------------------------------------------------------
# Walk-Forward Trainer
# ---------------------------------------------------------------------------


class WalkForwardTrainer:
    """Walk-forward training and evaluation pipeline for time-series models.

    Provides rigorous out-of-sample evaluation by training on historical data
    and testing on future data, advancing through time in a manner that
    prevents look-ahead bias.

    Parameters
    ----------
    model_fn : Callable
        A factory function that returns a new model instance.
        Signature: model_fn(**kwargs) -> nn.Module
    loss_fn : Callable
        Loss function for training.
    dataset : Dataset
        Full time-series dataset (must be ordered chronologically).
    window_strategy : WindowStrategy
        Windowing strategy (expanding, sliding, anchored).
    initial_train_size : int
        Size of the initial training window.
    test_size : int
        Size of each test window.
    step_size : Optional[int]
        Step size between windows; defaults to test_size.
    max_train_size : Optional[int]
        Maximum training window size for sliding windows.
    gap : int
        Gap between train and test to prevent leakage.
    retraining_trigger : RetrainingTrigger
        When to retrain the model.
    retrain_interval : int
        Windows between retraining for TIME_BASED trigger.
    performance_degradation_threshold : float
        Relative loss increase that triggers retraining.
    drift_threshold : float
        Loss change threshold for drift-based retraining.
    epochs_per_window : int
        Training epochs per window when retraining.
    learning_rate : float
        Training learning rate.
    batch_size : int
        Training batch size.
    device : str
        Compute device.
    auto_tune_hyperparams : bool
        Whether to tune hyperparameters per window.
    auto_tune_interval : int
        Tune hyperparameters every N windows.
    auto_tune_trials : int
        Number of Optuna trials per tuning session.
    checkpoint_dir : Optional[str]
        Directory to save model checkpoints per window.
    verbose : bool
        Whether to log detailed progress.
    """

    def __init__(
        self,
        model_fn: Callable[..., nn.Module],
        loss_fn: Optional[Callable] = None,
        dataset: Optional[Dataset] = None,
        window_strategy: WindowStrategy = WindowStrategy.EXPANDING,
        initial_train_size: int = 500,
        test_size: int = 100,
        step_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        gap: int = 0,
        retraining_trigger: RetrainingTrigger = RetrainingTrigger.ALWAYS,
        retrain_interval: int = 3,
        performance_degradation_threshold: float = 0.1,
        drift_threshold: float = 0.15,
        epochs_per_window: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "auto",
        auto_tune_hyperparams: bool = False,
        auto_tune_interval: int = 5,
        auto_tune_trials: int = 20,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.model_fn = model_fn
        self.loss_fn = loss_fn or nn.MSELoss()
        self.dataset = dataset
        self.window_strategy = window_strategy
        self.epochs_per_window = epochs_per_window
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.retraining_trigger = retraining_trigger
        self.retrain_interval = retrain_interval
        self.performance_degradation_threshold = performance_degradation_threshold
        self.drift_threshold = drift_threshold
        self.auto_tune_hyperparams = auto_tune_hyperparams
        self.auto_tune_interval = auto_tune_interval
        self.auto_tune_trials = auto_tune_trials
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.device = self._resolve_device(device)

        # Window splitter
        self.window_splitter = WindowSplitter(
            strategy=window_strategy,
            initial_train_size=initial_train_size,
            test_size=test_size,
            step_size=step_size,
            max_train_size=max_train_size,
            gap=gap,
        )

        # Current model and optimizer state
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._prev_test_loss: Optional[float] = None
        self._windows_since_retrain: int = 0
        self._best_hyperparams: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve the target compute device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def run(self, dataset: Optional[Dataset] = None) -> WalkForwardResult:
        """Execute the full walk-forward analysis.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Dataset override; uses the constructor dataset if None.

        Returns
        -------
        WalkForwardResult
            Comprehensive walk-forward results.
        """
        dataset = dataset or self.dataset
        if dataset is None:
            raise ValueError("No dataset provided for walk-forward analysis")

        # Generate window splits
        splits = self.window_splitter.split(len(dataset))
        if not splits:
            logger.warning("No valid walk-forward windows generated")
            return WalkForwardResult()

        result = WalkForwardResult()
        self._prev_test_loss = None
        self._windows_since_retrain = 0

        logger.info(
            f"Starting walk-forward analysis: {len(splits)} windows, "
            f"strategy={self.window_strategy.value}"
        )

        for window_idx, (train_range, test_range) in enumerate(splits):
            window_start = time.time()

            # Determine whether to retrain
            should_retrain = self._should_retrain(window_idx, result)

            if should_retrain or self._model is None:
                # Initialize or retrain model
                self._train_window(dataset, train_range, window_idx)
                self._windows_since_retrain = 0
            else:
                self._windows_since_retrain += 1

            # Evaluate on test window
            test_loss, test_metrics = self._evaluate_window(
                dataset, test_range
            )

            # Training loss (on last epoch)
            train_loss = self._get_train_loss(dataset, train_range)

            window_metrics = WindowMetrics(
                window_index=window_idx,
                train_start=train_range.start,
                train_end=train_range.stop,
                test_start=test_range.start,
                test_end=test_range.stop,
                train_samples=len(train_range),
                test_samples=len(test_range),
                train_loss=train_loss,
                test_loss=test_loss,
                test_metrics=test_metrics,
                retrained=should_retrain or self._model is not None,
                training_time_s=time.time() - window_start,
            )

            result.window_metrics.append(window_metrics)
            self._prev_test_loss = test_loss

            # Checkpoint
            if self.checkpoint_dir and should_retrain:
                self._save_window_checkpoint(window_idx)

            if self.verbose:
                logger.info(
                    f"Window {window_idx:3d} | "
                    f"train={len(train_range)} test={len(test_range)} | "
                    f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} | "
                    f"retrained={window_metrics.retrained}"
                )

        # Compute aggregate results
        result.compute_aggregate_metrics()

        logger.info(
            f"Walk-forward complete: {result.n_windows} windows, "
            f"OOS loss={result.oos_loss:.6f}, "
            f"stability={result.stability_score:.4f}, "
            f"trend={result.get_loss_trend()}"
        )

        return result

    def _should_retrain(
        self, window_idx: int, result: WalkForwardResult
    ) -> bool:
        """Determine whether the model should be retrained for this window."""
        if self.retraining_trigger == RetrainingTrigger.ALWAYS:
            return True

        if self.retraining_trigger == RetrainingTrigger.TIME_BASED:
            return self._windows_since_retrain >= self.retrain_interval

        if self.retraining_trigger == RetrainingTrigger.PERFORMANCE_BASED:
            if self._prev_test_loss is None:
                return True
            if len(result.window_metrics) > 0:
                recent_losses = [
                    w.test_loss for w in result.window_metrics[-3:]
                ]
                if len(recent_losses) >= 2:
                    avg_recent = float(np.mean(recent_losses[-2:]))
                    avg_prev = float(np.mean(recent_losses[:-2])) if len(recent_losses) > 2 else recent_losses[0]
                    degradation = (avg_recent - avg_prev) / max(abs(avg_prev), 1e-8)
                    if degradation > self.performance_degradation_threshold:
                        return True
            return False

        if self.retraining_trigger == RetrainingTrigger.DRIFT_BASED:
            if self._prev_test_loss is None:
                return True
            if len(result.window_metrics) >= 2:
                recent = result.window_metrics[-1].test_loss
                prev = result.window_metrics[-2].test_loss
                change = abs(recent - prev) / max(abs(prev), 1e-8)
                if change > self.drift_threshold:
                    return True
            return False

        if self.retraining_trigger == RetrainingTrigger.ADAPTIVE:
            # Combine performance and drift triggers
            should = False
            if self._prev_test_loss is None:
                should = True
            elif self._windows_since_retrain >= self.retrain_interval * 2:
                should = True
            elif len(result.window_metrics) >= 2:
                recent = result.window_metrics[-1].test_loss
                prev = result.window_metrics[-2].test_loss
                change = (recent - prev) / max(abs(prev), 1e-8)
                if change > self.performance_degradation_threshold:
                    should = True
            return should

        return True

    def _train_window(
        self,
        dataset: Dataset,
        train_range: range,
        window_idx: int,
    ) -> None:
        """Train the model on a window's training data.

        Parameters
        ----------
        dataset : Dataset
            Full dataset.
        train_range : range
            Index range for training data.
        window_idx : int
            Current window index.
        """
        # Create new model for first window, else reuse
        if self._model is None:
            self._model = self.model_fn().to(self.device)

        # Auto-tune hyperparameters periodically
        if (
            self.auto_tune_hyperparams
            and window_idx > 0
            and window_idx % self.auto_tune_interval == 0
        ):
            self._auto_tune(dataset, train_range)

        # Create optimizer with current learning rate
        lr = self._best_hyperparams.get("learning_rate", self.learning_rate) if self._best_hyperparams else self.learning_rate
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        # Create data loader
        train_subset = Subset(dataset, list(train_range))
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        # Training loop
        self._model.train()
        for epoch in range(self.epochs_per_window):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x, y = self._unpack_batch(batch)
                self._optimizer.zero_grad()
                pred = self._model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

    def _evaluate_window(
        self,
        dataset: Dataset,
        test_range: range,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on a window's test data.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            Test loss and additional metrics.
        """
        test_subset = Subset(dataset, list(test_range))
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )

        self._model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = self._unpack_batch(batch)
                pred = self._model(x)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        avg_loss = total_loss / max(total_samples, 1)
        metrics: Dict[str, float] = {}

        if all_preds:
            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            metrics["mae"] = torch.nn.functional.l1_loss(preds, targets).item()
            metrics["rmse"] = float(np.sqrt(avg_loss))

            # Directional accuracy
            if preds.numel() > 1:
                pred_dir = torch.sign(preds[1:] - preds[:-1])
                true_dir = torch.sign(targets[1:] - targets[:-1])
                metrics["directional_accuracy"] = (
                    (pred_dir == true_dir).float().mean().item()
                )

        return avg_loss, metrics

    def _get_train_loss(self, dataset: Dataset, train_range: range) -> float:
        """Compute training loss for the window."""
        train_subset = Subset(dataset, list(train_range))
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size * 2,
            shuffle=False,
        )

        self._model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                x, y = self._unpack_batch(batch)
                pred = self._model(x)
                loss = self.loss_fn(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

        return total_loss / max(total_samples, 1)

    def _auto_tune(self, dataset: Dataset, train_range: range) -> None:
        """Run hyperparameter tuning for the current window.

        Uses a simple grid search or random search since we don't want
        to pull in Optuna dependency here (the hyperopt module handles that).
        """
        logger.info(f"Auto-tuning hyperparameters at window {len(train_range)} samples")

        # Simple random search over learning rate
        best_lr = self.learning_rate
        best_loss = float("inf")

        train_subset = Subset(dataset, list(train_range))
        # Use last 20% as validation
        n_val = max(1, len(train_range) // 5)
        val_indices = list(range(len(train_range) - n_val, len(train_range)))
        train_indices = list(range(len(train_range) - n_val))

        train_sub = Subset(train_subset, train_indices)
        val_sub = Subset(train_subset, val_indices)

        for trial_lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            trial_model = self.model_fn().to(self.device)
            trial_optimizer = torch.optim.AdamW(trial_model.parameters(), lr=trial_lr)

            train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=self.batch_size * 2, shuffle=False)

            # Quick training
            trial_model.train()
            for _ in range(min(3, self.epochs_per_window)):
                for batch in train_loader:
                    x, y = self._unpack_batch(batch)
                    trial_optimizer.zero_grad()
                    pred = trial_model(x)
                    loss = self.loss_fn(pred, y)
                    loss.backward()
                    trial_optimizer.step()

            # Evaluate
            trial_model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = self._unpack_batch(batch)
                    pred = trial_model(x)
                    val_loss += self.loss_fn(pred, y).item() * x.shape[0]
                    n += x.shape[0]

            avg_val_loss = val_loss / max(n, 1)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_lr = trial_lr

        self._best_hyperparams = {"learning_rate": best_lr}
        logger.info(f"Auto-tune complete: best_lr={best_lr}, best_val_loss={best_loss:.6f}")

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch into (inputs, targets) on the correct device."""
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["input"], batch["target"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _save_window_checkpoint(self, window_idx: int) -> None:
        """Save a checkpoint for the current window."""
        if not self.checkpoint_dir or self._model is None:
            return
        path = Path(self.checkpoint_dir) / f"window_{window_idx:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "window_idx": window_idx,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
                "hyperparams": self._best_hyperparams,
            },
            path,
        )


# ---------------------------------------------------------------------------
# Model Stability Analysis
# ---------------------------------------------------------------------------


class ModelStabilityAnalyzer:
    """Analyzes model stability across walk-forward windows.

    Provides statistical tests and metrics to determine whether a model's
    performance is consistent or highly variable across time windows.
    """

    @staticmethod
    def compute_stability(result: WalkForwardResult) -> Dict[str, float]:
        """Compute comprehensive stability metrics.

        Parameters
        ----------
        result : WalkForwardResult
            Walk-forward results to analyze.

        Returns
        -------
        Dict[str, float]
            Stability metrics.
        """
        if not result.window_metrics:
            return {}

        losses = [w.test_loss for w in result.window_metrics]
        arr = np.array(losses)

        metrics: Dict[str, float] = {
            "mean_loss": float(np.mean(arr)),
            "std_loss": float(np.std(arr)),
            "min_loss": float(np.min(arr)),
            "max_loss": float(np.max(arr)),
            "cv": float(np.std(arr) / max(np.mean(arr), 1e-8)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "stability_score": result.stability_score,
        }

        # Autocorrelation of losses (high autocorrelation = persistent regimes)
        if len(arr) > 2:
            autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            metrics["loss_autocorrelation"] = float(autocorr)

        # Trend test (Mann-Kendall-like)
        if len(arr) > 3:
            n_inversions = 0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if arr[j] > arr[i]:
                        n_inversions += 1
            total_pairs = len(arr) * (len(arr) - 1) / 2
            metrics["trend_statistic"] = float(n_inversions / total_pairs)

        return metrics

    @staticmethod
    def detect_regime_changes(
        result: WalkForwardResult, threshold: float = 2.0
    ) -> List[int]:
        """Detect windows where model performance changed significantly.

        Parameters
        ----------
        result : WalkForwardResult
            Walk-forward results.
        threshold : float
            Number of standard deviations for change detection.

        Returns
        -------
        List[int]
            Window indices where regime changes were detected.
        """
        if len(result.window_metrics) < 3:
            return []

        losses = [w.test_loss for w in result.window_metrics]
        arr = np.array(losses)
        mean_loss = np.mean(arr)
        std_loss = np.std(arr)

        if std_loss < 1e-8:
            return []

        regime_changes = []
        for i in range(1, len(arr)):
            z_score = abs(arr[i] - arr[i - 1]) / std_loss
            if z_score > threshold:
                regime_changes.append(i)

        return regime_changes

    @staticmethod
    def compare_models(
        results: Dict[str, WalkForwardResult],
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models' walk-forward results.

        Parameters
        ----------
        results : Dict[str, WalkForwardResult]
            Model name → walk-forward results.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Comparison metrics per model.
        """
        comparison: Dict[str, Dict[str, float]] = {}
        for name, result in results.items():
            comparison[name] = {
                "oos_loss": result.oos_loss,
                "stability_score": result.stability_score,
                "n_retrains": float(result.n_retrains),
                "total_time_s": result.total_train_time_s,
                "loss_trend": result.get_loss_trend(),
            }

        return comparison
