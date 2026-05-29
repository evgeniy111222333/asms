"""
Core Training Engine for ACMS
==============================

Production-grade training engine with GPU support, mixed precision training,
gradient accumulation, learning rate scheduling, early stopping, checkpointing,
and comprehensive profiling.

Features
--------
- Automatic mixed precision (AMP) training
- Gradient accumulation for effective large batch sizes
- Multiple learning rate schedulers (cosine annealing, warm restart, one-cycle)
- Gradient clipping and NaN detection
- Early stopping with configurable patience
- Checkpoint saving (best model, periodic saves, training resumption)
- Metrics logging to TensorBoard and/or Weights & Biases
- TrainingState management for resumability
- Epoch-level and step-level callback system
- Automatic batch size tuning for maximum GPU utilization
- Training profiling and bottleneck detection
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class LRSchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    COSINE_ANNEALING = "cosine_annealing"
    WARM_RESTART = "warm_restart"
    ONE_CYCLE = "one_cycle"
    STEP_LR = "step_lr"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"


@dataclass
class TrainingConfig:
    """Configuration for the training engine.

    Attributes
    ----------
    epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization coefficient.
    device : str
        Compute device ('cuda', 'cpu', or 'auto').
    amp : bool
        Enable automatic mixed precision.
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients before updating weights.
    max_grad_norm : float
        Maximum gradient norm for clipping; 0 disables clipping.
    early_stopping_patience : int
        Epochs to wait before early stopping; 0 disables.
    early_stopping_metric : str
        Metric name for early stopping decisions.
    early_stopping_mode : str
        'min' or 'max' — whether lower or higher metric is better.
    checkpoint_dir : str
        Directory for saving checkpoints.
    save_best_only : bool
        Only save when best metric improves.
    save_period : int
        Save checkpoint every N epochs; 0 disables periodic saving.
    lr_scheduler : LRSchedulerType
        Learning rate scheduler type.
    lr_scheduler_kwargs : dict
        Additional keyword arguments for the scheduler.
    warmup_steps : int
        Number of linear warmup steps.
    batch_size : int
        Training batch size; 0 triggers auto-tuning.
    val_batch_size : int
        Validation batch size.
    num_workers : int
        DataLoader worker count.
    pin_memory : bool
        Pin memory for faster GPU transfer.
    log_interval : int
        Log metrics every N steps.
    profile : bool
        Enable training profiling.
    tensorboard_dir : Optional[str]
        TensorBoard log directory; None disables.
    wandb_project : Optional[str]
        W&B project name; None disables W&B.
    wandb_entity : Optional[str]
        W&B entity (team) name.
    seed : int
        Random seed for reproducibility.
    """

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "auto"
    amp: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = False
    save_period: int = 5
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE_ANNEALING
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    log_interval: int = 10
    profile: bool = False
    tensorboard_dir: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    seed: int = 42


@dataclass
class TrainingState:
    """Mutable state tracked during training for resumability.

    Attributes
    ----------
    epoch : int
        Current epoch number.
    global_step : int
        Total steps across all epochs.
    best_metric : float
        Best metric value observed.
    best_epoch : int
        Epoch at which best metric occurred.
    epochs_without_improvement : int
        Consecutive epochs without metric improvement.
    learning_rate : float
        Current learning rate.
    train_loss : float
        Most recent training loss.
    val_loss : float
        Most recent validation loss.
    start_time : float
        Timestamp when training started.
    total_training_time : float
        Cumulative training time in seconds.
    nan_detected : bool
        Whether NaN was detected in gradients.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    learning_rate: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    start_time: float = 0.0
    total_training_time: float = 0.0
    nan_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a dictionary."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "learning_rate": self.learning_rate,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "total_training_time": self.total_training_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Deserialize state from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StepMetrics:
    """Metrics recorded at each training step."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    step_time_ms: float
    memory_mb: float = 0.0
    throughput: float = 0.0


@dataclass
class EpochMetrics:
    """Metrics recorded at each epoch boundary."""

    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    epoch_time_s: float = 0.0
    learning_rate: float = 0.0


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    @abstractmethod
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called once at the start of training."""

    @abstractmethod
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once at the end of training."""

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: EpochMetrics) -> None:
        """Called at the end of each epoch."""

    def on_step_begin(self, trainer: "Trainer", step: int) -> None:
        """Called at the beginning of each training step."""

    def on_step_end(self, trainer: "Trainer", step: int, metrics: StepMetrics) -> None:
        """Called at the end of each training step."""


class EpochCallback(TrainingCallback):
    """Convenience callback that only hooks into epoch events."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: EpochMetrics) -> None:
        """Called at the end of each epoch."""


class StepCallback(TrainingCallback):
    """Convenience callback that only hooks into step events."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass

    @abstractmethod
    def on_step_end(self, trainer: "Trainer", step: int, metrics: StepMetrics) -> None:
        """Called at the end of each step."""


class EarlyStoppingCallback(TrainingCallback):
    """Stops training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement.
    metric : str
        Metric name to monitor.
    mode : str
        'min' if lower is better, 'max' if higher is better.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "val_loss",
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0
        self._stopped_epoch: int = 0
        self.should_stop: bool = False

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._counter = 0
        self.should_stop = False

    def on_train_end(self, trainer: "Trainer") -> None:
        if self._stopped_epoch > 0:
            logger.info(f"Early stopping triggered at epoch {self._stopped_epoch}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: EpochMetrics) -> None:
        current = self._get_metric_value(metrics)
        if current is None:
            return

        if self._is_improvement(current):
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._stopped_epoch = epoch
                self.should_stop = True
                logger.info(
                    f"Early stopping: no improvement in '{self.metric}' for "
                    f"{self.patience} epochs (best={self._best:.6f})"
                )

    def _get_metric_value(self, metrics: EpochMetrics) -> Optional[float]:
        """Extract the monitored metric value."""
        if self.metric == "train_loss":
            return metrics.train_loss
        elif self.metric == "val_loss":
            return metrics.val_loss
        elif self.metric in metrics.val_metrics:
            return metrics.val_metrics[self.metric]
        elif self.metric in metrics.train_metrics:
            return metrics.train_metrics[self.metric]
        return None

    def _is_improvement(self, value: float) -> bool:
        """Check whether the new value is an improvement."""
        if self.mode == "min":
            return value < self._best - self.min_delta
        else:
            return value > self._best + self.min_delta


class CheckpointCallback(TrainingCallback):
    """Saves model checkpoints during training.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints.
    save_best_only : bool
        Only save when best metric improves.
    save_period : int
        Save every N epochs (0 disables periodic saves).
    metric : str
        Metric to monitor for best model.
    mode : str
        'min' or 'max'.
    max_keep : int
        Maximum number of periodic checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = False,
        save_period: int = 5,
        metric: str = "val_loss",
        mode: str = "min",
        max_keep: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.save_period = save_period
        self.metric = metric
        self.mode = mode
        self.max_keep = max_keep
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._periodic_checkpoints: List[Path] = []

    def on_train_begin(self, trainer: "Trainer") -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._periodic_checkpoints = []

    def on_train_end(self, trainer: "Trainer") -> None:
        logger.info(f"Best model saved at {self.checkpoint_dir / 'best_model.pt'}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: EpochMetrics) -> None:
        current = self._get_metric_value(metrics)

        # Save best model
        if current is not None and self._is_improvement(current):
            self._best = current
            self._save_checkpoint(trainer, "best_model.pt", epoch, metrics)

        # Periodic save
        if not self.save_best_only and self.save_period > 0:
            if (epoch + 1) % self.save_period == 0:
                path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
                self._save_checkpoint(trainer, path.name, epoch, metrics)
                self._periodic_checkpoints.append(path)
                self._cleanup_old_checkpoints()

    def _get_metric_value(self, metrics: EpochMetrics) -> Optional[float]:
        if self.metric == "train_loss":
            return metrics.train_loss
        elif self.metric == "val_loss":
            return metrics.val_loss
        elif self.metric in metrics.val_metrics:
            return metrics.val_metrics[self.metric]
        return None

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best
        else:
            return value > self._best

    def _save_checkpoint(
        self, trainer: "Trainer", filename: str, epoch: int, metrics: EpochMetrics
    ) -> None:
        """Save a full training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "training_state": trainer.state.to_dict(),
            "epoch_metrics": {
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
            },
            "config": {
                "learning_rate": trainer.config.learning_rate,
                "weight_decay": trainer.config.weight_decay,
            },
        }
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        if trainer.scaler is not None:
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest periodic checkpoints exceeding max_keep."""
        while len(self._periodic_checkpoints) > self.max_keep:
            old = self._periodic_checkpoints.pop(0)
            if old.exists():
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old}")


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------


class LRSchedulerFactory:
    """Factory for creating learning rate schedulers."""

    @staticmethod
    def create(
        scheduler_type: LRSchedulerType,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        epochs: int,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create a learning rate scheduler.

        Parameters
        ----------
        scheduler_type : LRSchedulerType
            Type of scheduler to create.
        optimizer : torch.optim.Optimizer
            The optimizer to schedule.
        total_steps : int
            Total training steps across all epochs.
        epochs : int
            Total number of training epochs.
        kwargs : dict, optional
            Extra keyword arguments for the scheduler.

        Returns
        -------
        Optional[torch.optim.lr_scheduler._LRScheduler]
            The instantiated scheduler, or None for constant LR.
        """
        kwargs = kwargs or {}

        if scheduler_type == LRSchedulerType.COSINE_ANNEALING:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", epochs),
                eta_min=kwargs.get("eta_min", 1e-6),
            )
        elif scheduler_type == LRSchedulerType.WARM_RESTART:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
                eta_min=kwargs.get("eta_min", 1e-6),
            )
        elif scheduler_type == LRSchedulerType.ONE_CYCLE:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get("max_lr", optimizer.param_groups[0]["lr"]),
                total_steps=total_steps,
                pct_start=kwargs.get("pct_start", 0.3),
                anneal_strategy=kwargs.get("anneal_strategy", "cos"),
                div_factor=kwargs.get("div_factor", 25.0),
                final_div_factor=kwargs.get("final_div_factor", 1e4),
            )
        elif scheduler_type == LRSchedulerType.STEP_LR:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_type == LRSchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.95),
            )
        elif scheduler_type == LRSchedulerType.REDUCE_ON_PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                min_lr=kwargs.get("min_lr", 1e-6),
            )
        elif scheduler_type == LRSchedulerType.CONSTANT:
            return None
        elif scheduler_type == LRSchedulerType.LINEAR_WARMUP:
            warmup_steps = kwargs.get("warmup_steps", 1000)
            return _LinearWarmupScheduler(
                optimizer, warmup_steps=warmup_steps, total_steps=total_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class _LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            alpha = self._step_count / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self._step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return [base_lr * decay for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training Profiler
# ---------------------------------------------------------------------------


class TrainingProfiler:
    """Lightweight profiler for detecting training bottlenecks.

    Tracks timing for data loading, forward pass, backward pass, optimizer
    step, and overall throughput. Reports bottlenecks when requested.
    """

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = {
            "data_load": [],
            "forward": [],
            "backward": [],
            "optimizer_step": [],
            "total_step": [],
        }
        self._active: Dict[str, float] = {}

    def start(self, phase: str) -> None:
        """Mark the start of a phase."""
        self._active[phase] = time.perf_counter()

    def end(self, phase: str) -> None:
        """Mark the end of a phase and record duration."""
        if phase in self._active:
            duration_ms = (time.perf_counter() - self._active.pop(phase)) * 1000.0
            self._timings.setdefault(phase, []).append(duration_ms)

    def report(self) -> Dict[str, Dict[str, float]]:
        """Generate a profiling report.

        Returns
        -------
        dict
            Per-phase statistics: mean, std, min, max, total, count.
        """
        report: Dict[str, Dict[str, float]] = {}
        for phase, durations in self._timings.items():
            if not durations:
                continue
            arr = durations
            report[phase] = {
                "mean_ms": sum(arr) / len(arr),
                "min_ms": min(arr),
                "max_ms": max(arr),
                "total_ms": sum(arr),
                "count": len(arr),
            }
        return report

    def detect_bottleneck(self) -> Optional[str]:
        """Identify the most time-consuming phase.

        Returns
        -------
        Optional[str]
            Name of the bottleneck phase, or None if insufficient data.
        """
        report = self.report()
        if not report:
            return None
        return max(report, key=lambda k: report[k].get("total_ms", 0))

    def reset(self) -> None:
        """Clear all recorded timings."""
        for key in self._timings:
            self._timings[key] = []
        self._active.clear()


# ---------------------------------------------------------------------------
# Automatic Batch Size Tuner
# ---------------------------------------------------------------------------


class AutoBatchSizeTuner:
    """Find the maximum batch size that fits in GPU memory.

    Uses a binary search approach: start with a large batch size and halve
    it whenever an out-of-memory error occurs.

    Parameters
    ----------
    model : nn.Module
        The model to test.
    device : torch.device
        Target device.
    start_batch_size : int
        Initial batch size to try.
    max_batch_size : int
        Upper bound on batch size.
    sample_input_shape : Tuple[int, ...]
        Shape of a single sample (excluding batch dimension).
    amp : bool
        Use AMP during sizing (reduces memory).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        start_batch_size: int = 256,
        max_batch_size: int = 4096,
        sample_input_shape: Tuple[int, ...] = (100,),
        amp: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.start_batch_size = start_batch_size
        self.max_batch_size = max_batch_size
        self.sample_input_shape = sample_input_shape
        self.amp = amp

    def find_batch_size(self) -> int:
        """Run binary search to find maximum viable batch size.

        Returns
        -------
        int
            The largest batch size that fits in GPU memory.
        """
        if self.device.type != "cuda":
            logger.info("Auto batch size tuning only applies to CUDA; using default.")
            return self.start_batch_size

        logger.info("Starting automatic batch size tuning...")
        low, high = 1, self.max_batch_size
        best = low

        self.model.to(self.device)
        self.model.train()

        while low <= high:
            mid = (low + high) // 2
            try:
                self._try_batch(mid)
                best = mid
                low = mid + 1
                logger.debug(f"Batch size {mid} succeeded")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    torch.cuda.empty_cache()
                    logger.debug(f"Batch size {mid} OOM")
                else:
                    raise

        logger.info(f"Auto batch size tuning complete: max batch size = {best}")
        torch.cuda.empty_cache()
        return best

    def _try_batch(self, batch_size: int) -> None:
        """Attempt a forward + backward pass with the given batch size."""
        x = torch.randn(batch_size, *self.sample_input_shape, device=self.device)
        self.model.zero_grad()
        with autocast(enabled=self.amp):
            output = self.model(x)
            if isinstance(output, tuple):
                loss = output[0].sum()
            else:
                loss = output.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# Metrics Logger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Unified interface for logging training metrics.

    Supports TensorBoard and Weights & Biases backends simultaneously.
    """

    def __init__(
        self,
        tensorboard_dir: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tb_writer = None
        self._wandb_run = None

        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=tensorboard_dir)
                logger.info(f"TensorBoard logging to {tensorboard_dir}")
            except ImportError:
                warnings.warn("tensorboard not installed; skipping TensorBoard logging.")

        if wandb_project:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config or {},
                    reinit=True,
                )
                logger.info(f"W&B logging to project '{wandb_project}'")
            except ImportError:
                warnings.warn("wandb not installed; skipping W&B logging.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._tb_writer:
            self._tb_writer.add_scalar(tag, value, step)
        if self._wandb_run:
            import wandb

            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars under a main tag."""
        if self._tb_writer:
            self._tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        if self._wandb_run:
            import wandb

            wandb.log(
                {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}, step=step
            )

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram."""
        if self._tb_writer:
            self._tb_writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text."""
        if self._tb_writer:
            self._tb_writer.add_text(tag, text, step)

    def close(self) -> None:
        """Flush and close all logging backends."""
        if self._tb_writer:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._wandb_run:
            import wandb

            wandb.finish()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


class TrainingLoop:
    """Manages the inner training loop for a single epoch.

    Handles gradient accumulation, AMP, gradient clipping, NaN detection,
    and step-level profiling.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer.
    loss_fn : Callable
        Loss function that takes (y_pred, y_true) and returns a scalar.
    device : torch.device
        Target device.
    scaler : Optional[GradScaler]
        AMP gradient scaler.
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    profiler : Optional[TrainingProfiler]
        Profiler instance for timing.
    callbacks : List[TrainingCallback]
        Step-level callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        scaler: Optional[GradScaler] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        profiler: Optional[TrainingProfiler] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.profiler = profiler
        self.callbacks = callbacks or []

    def run_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        state: TrainingState,
    ) -> Tuple[float, Dict[str, float]]:
        """Execute one full epoch of training.

        Parameters
        ----------
        dataloader : DataLoader
            Training data loader.
        epoch : int
            Current epoch number.
        state : TrainingState
            Mutable training state.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            Average loss and auxiliary metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        nan_count = 0
        grad_norms: List[float] = []

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.perf_counter()

            # Data loading
            if self.profiler:
                self.profiler.start("data_load")

            x, y = self._unpack_batch(batch)
            batch_size = x.shape[0]

            if self.profiler:
                self.profiler.end("data_load")

            # Forward pass
            if self.profiler:
                self.profiler.start("forward")

            use_amp = self.scaler is not None
            with autocast(enabled=use_amp):
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss = loss / self.gradient_accumulation_steps

            if self.profiler:
                self.profiler.end("forward")

            # Backward pass
            if self.profiler:
                self.profiler.start("backward")

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.profiler:
                self.profiler.end("backward")

            # Gradient accumulation boundary
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping and NaN detection
                grad_norm = self._clip_and_check_gradients()
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    nan_count += 1
                    state.nan_detected = True
                    logger.warning(
                        f"NaN/Inf gradient detected at step {state.global_step} "
                        f"(grad_norm={grad_norm:.4f}). Skipping optimizer step."
                    )
                    if self.scaler:
                        self.scaler.update()
                    state.global_step += 1
                    continue

                grad_norms.append(grad_norm)

                # Optimizer step
                if self.profiler:
                    self.profiler.start("optimizer_step")

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.profiler:
                    self.profiler.end("optimizer_step")

            step_time_ms = (time.perf_counter() - step_start) * 1000.0

            total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size

            # Step metrics
            step_metrics = StepMetrics(
                step=state.global_step,
                loss=loss.item() * self.gradient_accumulation_steps,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                grad_norm=grad_norms[-1] if grad_norms else 0.0,
                step_time_ms=step_time_ms,
                memory_mb=self._get_gpu_memory_mb(),
                throughput=batch_size / (step_time_ms / 1000.0) if step_time_ms > 0 else 0,
            )

            # Step callbacks
            for cb in self.callbacks:
                cb.on_step_end(self, state.global_step, step_metrics)

            state.global_step += 1

        avg_loss = total_loss / max(total_samples, 1)
        aux_metrics = {
            "nan_count": float(nan_count),
            "avg_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
            "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
        }
        return avg_loss, aux_metrics

    def _unpack_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch into (inputs, targets) and move to device."""
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["input"], batch["target"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _clip_and_check_gradients(self) -> float:
        """Clip gradients and return the gradient norm.

        Returns
        -------
        float
            Total gradient norm before clipping.
        """
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        return total_norm

    @staticmethod
    def _get_gpu_memory_mb() -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Main training orchestrator for ACMS models.

    Coordinates the training loop, validation, learning rate scheduling,
    early stopping, checkpointing, metrics logging, and profiling.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    config : TrainingConfig
        Training configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : Optional[DataLoader]
        Validation data loader.
    loss_fn : Optional[Callable]
        Loss function; defaults to MSELoss.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer; defaults to AdamW.
    callbacks : Optional[List[TrainingCallback]]
        Training callbacks.
    resume_from : Optional[str]
        Path to a checkpoint to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.MSELoss()

        # Device setup
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # AMP scaler
        self.scaler: Optional[GradScaler] = None
        if config.amp and self.device.type == "cuda":
            self.scaler = GradScaler()

        # Learning rate scheduler (initialized in fit())
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        # Training state
        self.state = TrainingState(learning_rate=config.learning_rate)

        # Callbacks
        self.callbacks: List[TrainingCallback] = list(callbacks or [])
        self._add_default_callbacks()

        # Profiler
        self.profiler = TrainingProfiler() if config.profile else None

        # Metrics logger
        self.metrics_logger = MetricsLogger(
            tensorboard_dir=config.tensorboard_dir,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            wandb_config={"config": config.__dict__},
        )

        # Training loop
        self.training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            scaler=self.scaler,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            profiler=self.profiler,
            callbacks=[cb for cb in self.callbacks if isinstance(cb, StepCallback)],
        )

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve the target compute device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available; using CPU")
        else:
            device = torch.device(device_str)
        return device

    def _add_default_callbacks(self) -> None:
        """Add built-in callbacks based on configuration."""
        # Early stopping
        if self.config.early_stopping_patience > 0:
            self.callbacks.append(
                EarlyStoppingCallback(
                    patience=self.config.early_stopping_patience,
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                )
            )

        # Checkpointing
        self.callbacks.append(
            CheckpointCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                save_best_only=self.config.save_best_only,
                save_period=self.config.save_period,
                metric=self.config.early_stopping_metric,
                mode=self.config.early_stopping_mode,
            )
        )

    def _setup_scheduler(self) -> None:
        """Initialize the learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs // max(
            1, self.config.gradient_accumulation_steps
        )
        self.scheduler = LRSchedulerFactory.create(
            scheduler_type=self.config.lr_scheduler,
            optimizer=self.optimizer,
            total_steps=total_steps,
            epochs=self.config.epochs,
            kwargs=self.config.lr_scheduler_kwargs,
        )

    def fit(self) -> TrainingState:
        """Execute the full training loop.

        Returns
        -------
        TrainingState
            Final training state.
        """
        self._setup_scheduler()
        self._set_seed()

        self.state.start_time = time.time()
        logger.info(
            f"Training started: {self.config.epochs} epochs, "
            f"device={self.device}, AMP={'on' if self.scaler else 'off'}, "
            f"grad_accum={self.config.gradient_accumulation_steps}"
        )

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            # Epoch begin callbacks
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            # Train one epoch
            train_loss, train_aux = self.training_loop.run_epoch(
                self.train_loader, epoch, self.state
            )

            # Validation
            val_loss, val_metrics = self._validate() if self.val_loader else (0.0, {})

            # Update scheduler
            self._step_scheduler(val_loss)

            # Update state
            self.state.train_loss = train_loss
            self.state.val_loss = val_loss
            self.state.learning_rate = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_aux,
                val_metrics=val_metrics,
                epoch_time_s=epoch_time,
                learning_rate=self.state.learning_rate,
            )

            # Log metrics
            self._log_epoch_metrics(epoch_metrics)

            # Epoch end callbacks
            should_stop = False
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, epoch_metrics)
                if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                    should_stop = True

            # Print summary
            self._print_epoch_summary(epoch_metrics)

            if should_stop:
                logger.info(f"Training stopped early at epoch {epoch}")
                break

        # Finalize
        self.state.total_training_time = time.time() - self.state.start_time
        for cb in self.callbacks:
            cb.on_train_end(self)

        self._print_training_summary()
        self.metrics_logger.close()

        return self.state

    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation and return loss and metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = self.training_loop._unpack_batch(batch)
                use_amp = self.scaler is not None
                with autocast(enabled=use_amp):
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)

                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]
                all_preds.append(y_pred.detach())
                all_targets.append(y.detach())

        avg_loss = total_loss / max(total_samples, 1)

        # Compute additional metrics
        metrics: Dict[str, float] = {}
        if all_preds:
            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            metrics["mae"] = torch.nn.functional.l1_loss(preds, targets).item()
            metrics["rmse"] = math.sqrt(avg_loss)

            # Directional accuracy for financial data
            if preds.numel() > 1:
                pred_dir = torch.sign(preds[1:] - preds[:-1])
                true_dir = torch.sign(targets[1:] - targets[:-1])
                metrics["directional_accuracy"] = (
                    (pred_dir == true_dir).float().mean().item()
                )

        self.model.train()
        return avg_loss, metrics

    def _step_scheduler(self, val_loss: float) -> None:
        """Step the learning rate scheduler."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.scheduler.step()
        else:
            self.scheduler.step()

    def _log_epoch_metrics(self, metrics: EpochMetrics) -> None:
        """Log epoch metrics to configured backends."""
        step = metrics.epoch
        self.metrics_logger.log_scalar("train/loss", metrics.train_loss, step)
        self.metrics_logger.log_scalar("val/loss", metrics.val_loss, step)
        self.metrics_logger.log_scalar("train/lr", metrics.learning_rate, step)

        for k, v in metrics.train_metrics.items():
            self.metrics_logger.log_scalar(f"train/{k}", v, step)
        for k, v in metrics.val_metrics.items():
            self.metrics_logger.log_scalar(f"val/{k}", v, step)

    def _print_epoch_summary(self, metrics: EpochMetrics) -> None:
        """Print a concise epoch summary to the logger."""
        msg = (
            f"Epoch {metrics.epoch:4d} | "
            f"train_loss={metrics.train_loss:.6f} | "
            f"val_loss={metrics.val_loss:.6f} | "
            f"lr={metrics.learning_rate:.2e} | "
            f"time={metrics.epoch_time_s:.1f}s"
        )
        if metrics.val_metrics:
            extra = " | ".join(f"{k}={v:.4f}" for k, v in metrics.val_metrics.items())
            msg += f" | {extra}"
        logger.info(msg)

    def _print_training_summary(self) -> None:
        """Print a final training summary."""
        logger.info(
            f"Training complete: {self.state.epoch + 1} epochs, "
            f"best_val_metric={self.state.best_metric:.6f} at epoch {self.state.best_epoch}, "
            f"total_time={self.state.total_training_time:.1f}s"
        )
        if self.profiler:
            bottleneck = self.profiler.detect_bottleneck()
            report = self.profiler.report()
            logger.info(f"Profiling bottleneck: {bottleneck}")
            for phase, stats in report.items():
                logger.info(
                    f"  {phase}: mean={stats['mean_ms']:.1f}ms, "
                    f"total={stats['total_ms']:.0f}ms, count={stats['count']}"
                )

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state = TrainingState.from_dict(checkpoint["training_state"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Resumed training from epoch {self.state.epoch}, "
            f"step {self.state.global_step}"
        )

    @staticmethod
    def auto_batch_size(
        model: nn.Module,
        device: torch.device,
        sample_input_shape: Tuple[int, ...] = (100,),
        amp: bool = True,
        start_batch_size: int = 256,
    ) -> int:
        """Convenience method to find maximum batch size for GPU.

        Parameters
        ----------
        model : nn.Module
            The model to size.
        device : torch.device
            Target GPU device.
        sample_input_shape : Tuple[int, ...]
            Shape of one input sample (no batch dim).
        amp : bool
            Whether AMP will be used.
        start_batch_size : int
            Initial batch size to try.

        Returns
        -------
        int
            Maximum viable batch size.
        """
        tuner = AutoBatchSizeTuner(
            model=model,
            device=device,
            start_batch_size=start_batch_size,
            sample_input_shape=sample_input_shape,
            amp=amp,
        )
        return tuner.find_batch_size()
