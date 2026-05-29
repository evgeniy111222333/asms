"""
Meta-Learning for Fast Adaptation to New Market Conditions
===========================================================

Implements meta-learning algorithms that enable rapid adaptation when
market conditions shift or when deploying to new assets/markets:

- MAML: Model-Agnostic Meta-Learning for quick adaptation
- Reptile: Efficient meta-training via weight averaging
- TaskSampler: Market regime-based task sampling
- MarketRegimeTask: Task representation for market regimes
- MetaLearner: Unified interface for meta-learning

All models support GPU training with graceful CPU fallback.

Typical usage:
    >>> base_model = nn.Linear(10, 3)
    >>> maml = MAML(base_model, inner_lr=0.01, num_inner_steps=5)
    >>> adapted_model = maml.adapt(task_data)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Market Regime Task
# ---------------------------------------------------------------------------

@dataclass
class MarketRegimeTask:
    """A meta-learning task representing a specific market regime.

    Each task encapsulates data from a particular market condition
    (e.g., bull market, high volatility, crisis) for inner-loop training.

    Attributes:
        name: Human-readable task name (e.g., 'btc_bull_2024').
        regime_id: Integer regime identifier.
        support_data: Support set for inner-loop adaptation.
        query_data: Query set for outer-loop evaluation.
        asset_id: Target asset identifier.
        start_time: Start timestamp of the regime period.
        end_time: End timestamp of the regime period.
        metadata: Additional task metadata.
    """

    name: str
    regime_id: int
    support_data: Dict[str, Tensor]
    query_data: Dict[str, Tensor]
    asset_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def support_size(self) -> int:
        """Number of support examples."""
        for v in self.support_data.values():
            if isinstance(v, Tensor):
                return v.shape[0]
        return 0

    @property
    def query_size(self) -> int:
        """Number of query examples."""
        for v in self.query_data.values():
            if isinstance(v, Tensor):
                return v.shape[0]
        return 0


# ---------------------------------------------------------------------------
# Task Sampler
# ---------------------------------------------------------------------------

class TaskSampler:
    """Samples meta-learning tasks based on market regimes.

    Constructs tasks from historical data by:
    1. Detecting market regimes (via clustering or thresholding)
    2. Splitting each regime into support/query sets
    3. Optionally weighting tasks by regime recency or severity

    Args:
        features: Historical feature array of shape (T, F).
        labels: Historical label array of shape (T,).
        regime_labels: Array of regime assignments of shape (T,).
        n_support: Number of support examples per task.
        n_query: Number of query examples per task.
        device: Torch device.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        regime_labels: np.ndarray,
        n_support: int = 32,
        n_query: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.regime_labels = regime_labels
        self.n_support = n_support
        self.n_query = n_query
        self.device = device or _get_device()

        # Pre-group indices by regime
        self._regime_indices: Dict[int, List[int]] = {}
        for idx, regime in enumerate(regime_labels):
            if regime not in self._regime_indices:
                self._regime_indices[regime] = []
            self._regime_indices[regime].append(idx)

    @property
    def num_regimes(self) -> int:
        """Number of unique regimes."""
        return len(self._regime_indices)

    def sample_task(self, regime_id: Optional[int] = None) -> MarketRegimeTask:
        """Sample a single meta-learning task.

        Args:
            regime_id: Specific regime to sample from; random if None.

        Returns:
            MarketRegimeTask with support and query data.
        """
        if regime_id is None:
            regime_id = np.random.choice(list(self._regime_indices.keys()))

        indices = self._regime_indices.get(regime_id, [])
        if len(indices) < self.n_support + self.n_query:
            # Not enough data; sample with replacement
            sampled = np.random.choice(indices, self.n_support + self.n_query, replace=True)
        else:
            sampled = np.random.choice(indices, self.n_support + self.n_query, replace=False)

        support_idx = sampled[: self.n_support]
        query_idx = sampled[self.n_support :]

        support_features = torch.tensor(
            self.features[support_idx], dtype=torch.float32, device=self.device
        )
        support_labels = torch.tensor(
            self.labels[support_idx], dtype=torch.float32, device=self.device
        )
        query_features = torch.tensor(
            self.features[query_idx], dtype=torch.float32, device=self.device
        )
        query_labels = torch.tensor(
            self.labels[query_idx], dtype=torch.float32, device=self.device
        )

        return MarketRegimeTask(
            name=f"regime_{regime_id}",
            regime_id=regime_id,
            support_data={"features": support_features, "labels": support_labels},
            query_data={"features": query_features, "labels": query_labels},
        )

    def sample_batch(
        self, batch_size: int
    ) -> List[MarketRegimeTask]:
        """Sample a batch of tasks for meta-training.

        Args:
            batch_size: Number of tasks.

        Returns:
            List of MarketRegimeTask.
        """
        return [self.sample_task() for _ in range(batch_size)]

    def get_regime_weights(self) -> Dict[int, float]:
        """Compute inverse-frequency weights for regime balancing.

        Returns:
            Dict mapping regime_id to normalised weight.
        """
        counts = {k: len(v) for k, v in self._regime_indices.items()}
        total = sum(counts.values())
        weights = {k: total / (len(counts) * v) for k, v in counts.items()}
        return weights


# ---------------------------------------------------------------------------
# MAML (Model-Agnostic Meta-Learning)
# ---------------------------------------------------------------------------

class MAML:
    """Model-Agnostic Meta-Learning (Finn et al., 2017) for market adaptation.

    Learns initialisation parameters θ* such that a few gradient steps on a
    new task produce good performance. This enables rapid adaptation when
    market conditions shift.

    Algorithm:
        1. Sample task T_i
        2. Inner loop: θ_i = θ - α ∇_θ L_Ti(θ)  (k steps)
        3. Outer loop: θ ← θ - β ∇_θ Σ_i L_Ti(θ_i)

    Args:
        model: Base PyTorch model to meta-learn.
        inner_lr: Learning rate for inner-loop adaptation.
        num_inner_steps: Number of inner-loop gradient steps.
        first_order: If True, use first-order approximation (faster).
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        first_order: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.device = device or _get_device()
        self.model.to(self.device)

    def _clone_model(self) -> nn.Module:
        """Create a deep copy of the model for inner-loop adaptation."""
        return copy.deepcopy(self.model)

    def _compute_loss(
        self,
        model: nn.Module,
        data: Dict[str, Tensor],
    ) -> Tensor:
        """Compute task-specific loss.

        Default: MSE loss for regression tasks.
        Override by subclassing or providing a custom loss_fn.

        Args:
            model: Model to evaluate.
            data: Dict with 'features' and 'labels' tensors.

        Returns:
            Scalar loss tensor.
        """
        predictions = model(data["features"])
        return F.mse_loss(predictions.squeeze(), data["labels"])

    def inner_loop(
        self,
        task: MarketRegimeTask,
        loss_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """Perform inner-loop adaptation on a single task.

        Args:
            task: MarketRegimeTask with support data.
            loss_fn: Optional custom loss function(model, data) → loss.

        Returns:
            Adapted model copy.
        """
        adapted_model = self._clone_model()
        compute_loss = loss_fn or self._compute_loss

        for _ in range(self.num_inner_steps):
            loss = compute_loss(adapted_model, task.support_data)

            if self.first_order:
                # First-order: detach gradients for efficiency
                grads = torch.autograd.grad(
                    loss,
                    adapted_model.parameters(),
                    create_graph=False,
                )
            else:
                # Second-order: keep computation graph for meta-gradient
                grads = torch.autograd.grad(
                    loss,
                    adapted_model.parameters(),
                    create_graph=True,
                )

            # Manual gradient step
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad

        return adapted_model

    def meta_train_step(
        self,
        tasks: List[MarketRegimeTask],
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Perform a single meta-training outer-loop step.

        Args:
            tasks: Batch of MarketRegimeTask instances.
            optimizer: Meta-optimizer for outer-loop updates.
            loss_fn: Optional custom loss function.

        Returns:
            Dict with meta-loss and per-task losses.
        """
        meta_loss = torch.tensor(0.0, device=self.device)
        task_losses: List[float] = []

        for task in tasks:
            # Inner-loop adaptation
            adapted_model = self.inner_loop(task, loss_fn)
            compute_loss = loss_fn or self._compute_loss

            # Evaluate on query set
            query_loss = compute_loss(adapted_model, task.query_data)
            meta_loss = meta_loss + query_loss
            task_losses.append(query_loss.item())

        meta_loss = meta_loss / len(tasks)

        # Outer-loop gradient step
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "task_losses": task_losses,
            "mean_task_loss": np.mean(task_losses),
        }

    def adapt(
        self,
        task: MarketRegimeTask,
        loss_fn: Optional[Callable] = None,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt the meta-learned model to a new task.

        This is the fast adaptation step used at deployment time.

        Args:
            task: New task with support data.
            loss_fn: Optional custom loss function.
            num_steps: Override number of inner steps.

        Returns:
            Adapted model ready for inference on the new task.
        """
        original_steps = self.num_inner_steps
        if num_steps is not None:
            self.num_inner_steps = num_steps
        adapted = self.inner_loop(task, loss_fn)
        self.num_inner_steps = original_steps
        return adapted

    def meta_train(
        self,
        task_sampler: TaskSampler,
        num_iterations: int = 1000,
        meta_lr: float = 0.001,
        tasks_per_iteration: int = 4,
        loss_fn: Optional[Callable] = None,
        eval_interval: int = 100,
    ) -> List[Dict[str, float]]:
        """Run the full meta-training loop.

        Args:
            task_sampler: TaskSampler for generating tasks.
            num_iterations: Number of outer-loop iterations.
            meta_lr: Meta learning rate.
            tasks_per_iteration: Number of tasks per outer step.
            loss_fn: Optional custom loss function.
            eval_interval: Iterations between evaluations.

        Returns:
            List of metric dicts from each evaluation point.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        metrics_log: List[Dict[str, float]] = []

        for iteration in range(num_iterations):
            tasks = task_sampler.sample_batch(tasks_per_iteration)
            metrics = self.meta_train_step(tasks, optimizer, loss_fn)
            metrics["iteration"] = iteration

            if (iteration + 1) % eval_interval == 0:
                metrics_log.append(metrics)

        return metrics_log


# ---------------------------------------------------------------------------
# Reptile
# ---------------------------------------------------------------------------

class Reptile:
    """Reptile meta-learning algorithm (Nichol et al., 2018).

    A simpler alternative to MAML that meta-trains by:
    1. Sampling a task
    2. Training k steps on that task from current weights
    3. Moving initialisation toward the task-adapted weights

    θ ← θ + ε (θ_task - θ)

    This is equivalent to maximising the inner product of gradients across
    tasks, encouraging convergence to a point near all task solutions.

    Args:
        model: Base PyTorch model.
        inner_lr: Learning rate for inner-loop training.
        num_inner_steps: Number of SGD steps per task.
        meta_lr: Meta step size (ε in the update rule).
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        meta_lr: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_lr = meta_lr
        self.device = device or _get_device()
        self.model.to(self.device)

    def _compute_loss(
        self, model: nn.Module, data: Dict[str, Tensor]
    ) -> Tensor:
        """Default MSE loss for regression tasks."""
        predictions = model(data["features"])
        return F.mse_loss(predictions.squeeze(), data["labels"])

    def _inner_train(
        self,
        task: MarketRegimeTask,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, Tensor]:
        """Train on a single task and return weight difference.

        Args:
            task: MarketRegimeTask with support data.
            loss_fn: Optional custom loss.

        Returns:
            Dict mapping parameter names to θ_task - θ_init differences.
        """
        # Save initial weights
        init_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        compute_loss = loss_fn or self._compute_loss
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.inner_lr
        )

        for _ in range(self.num_inner_steps):
            loss = compute_loss(self.model, task.support_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute weight differences
        weight_diff = {}
        for name, param in self.model.named_parameters():
            weight_diff[name] = param.data - init_weights[name]

        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data = init_weights[name]

        return weight_diff

    def meta_train_step(
        self,
        tasks: List[MarketRegimeTask],
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Perform a single Reptile meta-update.

        Args:
            tasks: Batch of tasks.
            loss_fn: Optional custom loss.

        Returns:
            Dict with average weight change magnitude.
        """
        total_diff: Dict[str, Tensor] = {}

        for task in tasks:
            diff = self._inner_train(task, loss_fn)
            for name, delta in diff.items():
                if name not in total_diff:
                    total_diff[name] = delta.clone()
                else:
                    total_diff[name] += delta

        # Average the weight differences and update
        num_tasks = len(tasks)
        change_magnitude = 0.0
        for name, param in self.model.named_parameters():
            if name in total_diff:
                avg_diff = total_diff[name] / num_tasks
                param.data = param.data + self.meta_lr * avg_diff
                change_magnitude += avg_diff.norm().item()

        return {
            "weight_change_magnitude": change_magnitude,
            "num_tasks": num_tasks,
        }

    def adapt(
        self,
        task: MarketRegimeTask,
        loss_fn: Optional[Callable] = None,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """Fast adaptation to a new task using the meta-learned initialisation.

        Args:
            task: New task with support data.
            loss_fn: Optional custom loss.
            num_steps: Override inner steps.

        Returns:
            Adapted model copy.
        """
        adapted = copy.deepcopy(self.model)
        compute_loss = loss_fn or self._compute_loss
        optimizer = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        steps = num_steps or self.num_inner_steps

        for _ in range(steps):
            loss = compute_loss(adapted, task.support_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted

    def meta_train(
        self,
        task_sampler: TaskSampler,
        num_iterations: int = 1000,
        tasks_per_iteration: int = 4,
        loss_fn: Optional[Callable] = None,
        eval_interval: int = 100,
    ) -> List[Dict[str, float]]:
        """Run the full Reptile meta-training loop.

        Args:
            task_sampler: TaskSampler for generating tasks.
            num_iterations: Number of meta-updates.
            tasks_per_iteration: Tasks per meta-update.
            loss_fn: Optional custom loss.
            eval_interval: Iterations between logging.

        Returns:
            List of metric dicts.
        """
        metrics_log: List[Dict[str, float]] = []

        for iteration in range(num_iterations):
            tasks = task_sampler.sample_batch(tasks_per_iteration)
            metrics = self.meta_train_step(tasks, loss_fn)
            metrics["iteration"] = iteration

            if (iteration + 1) % eval_interval == 0:
                metrics_log.append(metrics)

        return metrics_log


# ---------------------------------------------------------------------------
# Meta-Learner (Unified Interface)
# ---------------------------------------------------------------------------

class MetaLearner:
    """Unified interface for meta-learning in the ACMS.

    Wraps either MAML or Reptile with:
    - Task management and sampling
    - Periodic evaluation on held-out regimes
    - Model checkpointing
    - Performance tracking

    Args:
        model: Base model to meta-learn.
        algorithm: 'maml' or 'reptile'.
        inner_lr: Inner-loop learning rate.
        num_inner_steps: Inner-loop gradient steps.
        meta_lr: Meta (outer) learning rate.
        first_order: Use first-order MAML (ignored for Reptile).
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: str = "maml",
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        meta_lr: float = 0.001,
        first_order: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or _get_device()
        self.model = model.to(self.device)
        self.algorithm = algorithm.lower()

        if self.algorithm == "maml":
            self._engine = MAML(
                model,
                inner_lr=inner_lr,
                num_inner_steps=num_inner_steps,
                first_order=first_order,
                device=self.device,
            )
        elif self.algorithm == "reptile":
            self._engine = Reptile(
                model,
                inner_lr=inner_lr,
                num_inner_steps=num_inner_steps,
                meta_lr=meta_lr,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'maml' or 'reptile'.")

        self._metrics_history: List[Dict[str, float]] = []

    def train(
        self,
        task_sampler: TaskSampler,
        num_iterations: int = 1000,
        tasks_per_iteration: int = 4,
        loss_fn: Optional[Callable] = None,
        eval_interval: int = 100,
    ) -> List[Dict[str, float]]:
        """Run meta-training.

        Args:
            task_sampler: TaskSampler instance.
            num_iterations: Number of meta-updates.
            tasks_per_iteration: Tasks per meta-step.
            loss_fn: Optional custom loss.
            eval_interval: Logging interval.

        Returns:
            List of metric dicts.
        """
        metrics = self._engine.meta_train(
            task_sampler=task_sampler,
            num_iterations=num_iterations,
            tasks_per_iteration=tasks_per_iteration,
            loss_fn=loss_fn,
            eval_interval=eval_interval,
        )
        self._metrics_history.extend(metrics)
        return metrics

    def adapt(
        self,
        task: MarketRegimeTask,
        loss_fn: Optional[Callable] = None,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt to a new task.

        Args:
            task: New task with support data.
            loss_fn: Optional custom loss.
            num_steps: Override inner steps.

        Returns:
            Adapted model.
        """
        return self._engine.adapt(task, loss_fn, num_steps)

    def evaluate(
        self,
        tasks: List[MarketRegimeTask],
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluate meta-learning performance on held-out tasks.

        For each task, adapts and measures query-set loss.

        Args:
            tasks: Evaluation tasks.
            loss_fn: Optional custom loss.

        Returns:
            Dict with mean and per-task adaptation losses.
        """
        compute_loss = loss_fn or self._engine._compute_loss
        pre_adapt_losses: List[float] = []
        post_adapt_losses: List[float] = []

        for task in tasks:
            # Pre-adaptation loss
            with torch.no_grad():
                pre_loss = compute_loss(self.model, task.query_data)
                pre_adapt_losses.append(pre_loss.item())

            # Adapt
            adapted = self.adapt(task, loss_fn)

            # Post-adaptation loss
            with torch.no_grad():
                post_loss = compute_loss(adapted, task.query_data)
                post_adapt_losses.append(post_loss.item())

        return {
            "mean_pre_adapt_loss": np.mean(pre_adapt_losses),
            "mean_post_adapt_loss": np.mean(post_adapt_losses),
            "improvement": np.mean(pre_adapt_losses) - np.mean(post_adapt_losses),
            "per_task_pre": pre_adapt_losses,
            "per_task_post": post_adapt_losses,
        }

    def save(self, path: str) -> None:
        """Save the meta-learned model.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "algorithm": self.algorithm,
                "metrics_history": self._metrics_history,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a meta-learned model from checkpoint.

        Args:
            path: File path of the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._metrics_history = checkpoint.get("metrics_history", [])

    @property
    def metrics_history(self) -> List[Dict[str, float]]:
        """Return the full metrics history."""
        return self._metrics_history
