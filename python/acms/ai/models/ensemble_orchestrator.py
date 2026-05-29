"""
Dynamic Model Ensemble Orchestration
======================================

Implements intelligent ensemble methods that dynamically combine predictions
from multiple models for robust crypto trading signals:

- ModelWrapper: Standardised interface for wrapping any model
- DynamicWeightedEnsemble: Performance-based weight assignment
- AdaptiveEnsemble: Real-time weight adjustment via online learning
- StackingEnsemble: Meta-learner over base model predictions
- EnsembleDiversityTracker: Correlation-based diversity monitoring

All components support GPU with graceful CPU fallback.

Typical usage:
    >>> ensemble = DynamicWeightedEnsemble(models=[model_a, model_b, model_c])
    >>> prediction = ensemble.predict(inputs)
    >>> ensemble.update_weights(targets)
    >>> print(ensemble.get_weights())
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """Standardised wrapper for any model in the ensemble.

    Provides a uniform predict() interface and tracks performance metrics
    regardless of the underlying model type (PyTorch, sklearn, etc.).

    Args:
        model: The underlying model object.
        name: Human-readable model name.
        predict_fn: Callable(model, inputs) → predictions.
        device: Torch device (for PyTorch models).
        weight: Initial ensemble weight.
    """

    def __init__(
        self,
        model: Any,
        name: str = "unnamed",
        predict_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        weight: float = 1.0,
    ) -> None:
        self.model = model
        self.name = name
        self.predict_fn = predict_fn
        self.device = device or _get_device()
        self.weight = weight

        # Performance tracking
        self._losses: Deque[float] = deque(maxlen=1000)
        self._predictions: Deque[float] = deque(maxlen=500)
        self._sharpe_values: Deque[float] = deque(maxlen=252)
        self._total_predictions = 0

    def predict(self, inputs: Any) -> Any:
        """Generate a prediction from the wrapped model.

        Args:
            inputs: Model-specific input.

        Returns:
            Model prediction.
        """
        if self.predict_fn is not None:
            return self.predict_fn(self.model, inputs)
        # Default: call model directly
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                if isinstance(inputs, Tensor):
                    inputs = inputs.to(self.device)
                return self.model(inputs)
        return self.model(inputs)

    def record_loss(self, loss: float) -> None:
        """Record a loss value for performance tracking.

        Args:
            loss: Scalar loss value.
        """
        self._losses.append(loss)

    def record_prediction(self, pred: float) -> None:
        """Record a prediction value.

        Args:
            pred: Scalar prediction value.
        """
        self._predictions.append(pred)
        self._total_predictions += 1

    def record_sharpe(self, sharpe: float) -> None:
        """Record a Sharpe ratio value.

        Args:
            sharpe: Sharpe ratio for a period.
        """
        self._sharpe_values.append(sharpe)

    @property
    def mean_loss(self) -> float:
        """Return the mean of recent losses."""
        if not self._losses:
            return float("inf")
        return np.mean(list(self._losses))

    @property
    def mean_sharpe(self) -> float:
        """Return the mean of recent Sharpe ratios."""
        if not self._sharpe_values:
            return 0.0
        return np.mean(list(self._sharpe_values))

    @property
    def recent_accuracy(self) -> float:
        """Return a rough accuracy metric based on recent losses."""
        if not self._losses:
            return 0.0
        # Use exponential of negative mean loss as a proxy
        return math.exp(-self.mean_loss)


# ---------------------------------------------------------------------------
# Dynamic Weighted Ensemble
# ---------------------------------------------------------------------------

class DynamicWeightedEnsemble:
    """Ensemble with performance-based dynamic weighting.

    Weights are assigned based on each model's recent performance
    (lower loss → higher weight). Supports:
    - Exponential performance weighting
    - Recency bias (recent performance matters more)
    - Minimum weight floor (no model is completely excluded)
    - Weight normalisation

    Args:
        models: List of ModelWrapper instances.
        weight_update_freq: Steps between weight updates.
        temperature: Softmax temperature for weight computation.
        min_weight: Minimum weight for any model.
        recency_bias: Exponential decay for older losses (0 = uniform).
    """

    def __init__(
        self,
        models: List[ModelWrapper],
        weight_update_freq: int = 10,
        temperature: float = 1.0,
        min_weight: float = 0.05,
        recency_bias: float = 0.95,
    ) -> None:
        self.models = models
        self.weight_update_freq = weight_update_freq
        self.temperature = temperature
        self.min_weight = min_weight
        self.recency_bias = recency_bias
        self._step_count = 0
        self._weight_history: List[Dict[str, float]] = []

        # Initialise weights uniformly
        n = len(models)
        for m in models:
            m.weight = 1.0 / n
        self._normalise_weights()

    def predict(self, inputs: Any) -> Any:
        """Generate a weighted ensemble prediction.

        For Tensor outputs, computes a weighted average.
        For other types, returns the prediction from the highest-weighted model.

        Args:
            inputs: Model inputs.

        Returns:
            Weighted ensemble prediction.
        """
        predictions: List[Tuple[Any, float]] = []

        for wrapper in self.models:
            pred = wrapper.predict(inputs)
            predictions.append((pred, wrapper.weight))

        # Check if predictions are tensors
        if all(isinstance(p, Tensor) for p, _ in predictions):
            weighted_sum = torch.zeros_like(predictions[0][0])
            for pred, weight in predictions:
                weighted_sum = weighted_sum + pred * weight
            return weighted_sum
        elif all(isinstance(p, np.ndarray) for p, _ in predictions):
            weighted_sum = np.zeros_like(predictions[0][0])
            for pred, weight in predictions:
                weighted_sum = weighted_sum + pred * weight
            return weighted_sum
        else:
            # Return prediction from best model
            best_idx = max(range(len(predictions)), key=lambda i: predictions[i][1])
            return predictions[best_idx][0]

    def update_weights(self, targets: Any) -> Dict[str, float]:
        """Update model weights based on recent performance.

        Computes per-model loss against targets and adjusts weights
        using softmax over inverse losses.

        Args:
            targets: Ground truth for loss computation.

        Returns:
            Dict of model_name → weight.
        """
        self._step_count += 1
        if self._step_count % self.weight_update_freq != 0:
            return self.get_weights()

        # Compute inverse-loss scores
        scores: List[float] = []
        for wrapper in self.models:
            loss = wrapper.mean_loss
            if loss <= 0 or math.isinf(loss):
                score = 1e6
            else:
                score = 1.0 / (loss + 1e-8)
            scores.append(score)

        # Softmax with temperature
        scores_arr = np.array(scores)
        exp_scores = np.exp(scores_arr / self.temperature)
        weights = exp_scores / exp_scores.sum()

        # Apply minimum weight floor
        weights = np.maximum(weights, self.min_weight)
        weights = weights / weights.sum()

        # Update model weights
        for i, wrapper in enumerate(self.models):
            wrapper.weight = weights[i]

        self._weight_history.append(self.get_weights())
        return self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Return current model weights.

        Returns:
            Dict mapping model names to weights.
        """
        return {m.name: m.weight for m in self.models}

    def _normalise_weights(self) -> None:
        """Ensure weights sum to 1."""
        total = sum(m.weight for m in self.models)
        if total > 0:
            for m in self.models:
                m.weight /= total

    @property
    def weight_history(self) -> List[Dict[str, float]]:
        """Return the history of weight assignments."""
        return self._weight_history

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-model performance summary.

        Returns:
            Dict mapping model name to {weight, mean_loss, mean_sharpe, accuracy}.
        """
        summary = {}
        for m in self.models:
            summary[m.name] = {
                "weight": m.weight,
                "mean_loss": m.mean_loss,
                "mean_sharpe": m.mean_sharpe,
                "accuracy": m.recent_accuracy,
            }
        return summary


# ---------------------------------------------------------------------------
# Adaptive Ensemble
# ---------------------------------------------------------------------------

class AdaptiveEnsemble:
    """Ensemble with real-time weight adaptation via online learning.

    Uses the Hedge / exponentiated gradient algorithm to adapt weights
    based on per-step feedback, enabling rapid response to changing
    market conditions.

    The weight update rule is:
        w_i ← w_i × exp(-η × loss_i)
        w ← w / sum(w)

    Args:
        models: List of ModelWrapper instances.
        learning_rate: Hedge learning rate (η).
        decay: Weight decay factor per step (prevents weight collapse).
        min_weight: Minimum weight for any model.
    """

    def __init__(
        self,
        models: List[ModelWrapper],
        learning_rate: float = 0.1,
        decay: float = 0.999,
        min_weight: float = 0.01,
    ) -> None:
        self.models = models
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_weight = min_weight
        self._step_count = 0
        self._weight_history: List[Dict[str, float]] = []

        # Initialise weights
        n = len(models)
        for m in models:
            m.weight = 1.0 / n

    def predict(self, inputs: Any) -> Any:
        """Generate weighted ensemble prediction.

        Args:
            inputs: Model inputs.

        Returns:
            Weighted prediction.
        """
        predictions = []
        for wrapper in self.models:
            pred = wrapper.predict(inputs)
            predictions.append((pred, wrapper.weight))

        if all(isinstance(p, Tensor) for p, _ in predictions):
            weighted_sum = torch.zeros_like(predictions[0][0])
            for pred, weight in predictions:
                weighted_sum = weighted_sum + pred * weight
            return weighted_sum
        elif all(isinstance(p, np.ndarray) for p, _ in predictions):
            weighted_sum = np.zeros_like(predictions[0][0])
            for pred, weight in predictions:
                weighted_sum = weighted_sum + pred * weight
            return weighted_sum
        else:
            best_idx = max(range(len(predictions)), key=lambda i: predictions[i][1])
            return predictions[best_idx][0]

    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        """Update weights online using the Hedge algorithm.

        Args:
            losses: Dict mapping model names to their current loss.

        Returns:
            Updated weights dict.
        """
        self._step_count += 1

        # Hedge update
        for wrapper in self.models:
            loss = losses.get(wrapper.name, 0.0)
            wrapper.weight = wrapper.weight * math.exp(-self.learning_rate * loss)
            # Apply decay to prevent extreme weight values
            wrapper.weight = max(wrapper.weight, self.min_weight)

        # Normalise
        total = sum(m.weight for m in self.models)
        for m in self.models:
            m.weight /= total

        self._weight_history.append(self.get_weights())
        return self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Return current weights."""
        return {m.name: m.weight for m in self.models}

    @property
    def weight_history(self) -> List[Dict[str, float]]:
        """Return weight adaptation history."""
        return self._weight_history


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble(nn.Module):
    """Stacking ensemble with a meta-learner over base model predictions.

    Architecture:
        Base models → predictions → Meta-learner → final prediction

    The meta-learner is a small neural network that learns to optimally
    combine base model predictions, potentially capturing non-linear
    interactions between models.

    Args:
        input_dim: Dimensionality of each base model's prediction.
        num_models: Number of base models.
        hidden_dim: Meta-learner hidden dimension.
        output_dim: Final output dimension.
        num_layers: Number of meta-learner layers.
        dropout: Dropout rate.
        device: Torch device.
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_models: int = 3,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device_ = device or _get_device()
        self.num_models = num_models
        self.input_dim = input_dim

        # Meta-learner network
        layers: List[nn.Module] = []
        prev_dim = num_models * input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.meta_learner = nn.Sequential(*layers)
        self.base_models: List[ModelWrapper] = []

        self.to(self.device_)

    def set_base_models(self, models: List[ModelWrapper]) -> None:
        """Set the base models for the stacking ensemble.

        Args:
            models: List of ModelWrapper instances.
        """
        self.base_models = models

    def forward(self, base_predictions: Tensor) -> Tensor:
        """Produce the final stacked prediction.

        Args:
            base_predictions: (batch, num_models, input_dim) tensor of
                             base model predictions.

        Returns:
            Final prediction (batch, output_dim).
        """
        # Flatten base predictions
        x = base_predictions.reshape(base_predictions.shape[0], -1)
        return self.meta_learner(x)

    def predict(self, inputs: Any) -> Tensor:
        """End-to-end prediction: run base models then stack.

        Args:
            inputs: Model inputs (same for all base models).

        Returns:
            Final stacked prediction tensor.
        """
        base_preds = []
        for wrapper in self.base_models:
            pred = wrapper.predict(inputs)
            if isinstance(pred, Tensor):
                base_preds.append(pred)
            else:
                base_preds.append(torch.tensor(pred, dtype=torch.float32, device=self.device_))

        # Stack: (batch, num_models, input_dim)
        base_tensor = torch.stack(base_preds, dim=1)
        return self.forward(base_tensor)

    def train_step(
        self,
        inputs: Any,
        targets: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train the meta-learner.

        Args:
            inputs: Inputs for base models.
            targets: Ground truth targets.
            optimizer: Meta-learner optimizer.

        Returns:
            Dict with loss value.
        """
        self.train()
        predictions = self.predict(inputs)
        loss = F.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"meta_loss": loss.item()}


# ---------------------------------------------------------------------------
# Ensemble Diversity Tracker
# ---------------------------------------------------------------------------

class EnsembleDiversityTracker:
    """Monitors and optimises diversity among ensemble members.

    Ensemble diversity is crucial for robust predictions. This tracker:
    - Computes pairwise prediction correlations
    - Tracks diversity metrics over time
    - Can penalise weights of correlated models
    - Supports correlation-based diversity optimisation

    Diversity metrics:
    - Disagreement: fraction of predictions that differ
    - Correlation: average pairwise correlation
    - Entropy: entropy of the prediction distribution

    Args:
        models: List of ModelWrapper instances.
        correlation_threshold: Above this, models are considered too correlated.
        diversity_weight: Weight for diversity penalty in weight updates.
        history_length: Number of prediction sets to track.
    """

    def __init__(
        self,
        models: List[ModelWrapper],
        correlation_threshold: float = 0.9,
        diversity_weight: float = 0.1,
        history_length: int = 100,
    ) -> None:
        self.models = models
        self.correlation_threshold = correlation_threshold
        self.diversity_weight = diversity_weight
        self.num_models = len(models)
        self._prediction_history: Deque[np.ndarray] = deque(maxlen=history_length)
        self._correlation_matrix: Optional[np.ndarray] = None
        self._diversity_history: List[Dict[str, float]] = []

    def record_predictions(self, predictions: List[float]) -> None:
        """Record a set of model predictions for diversity analysis.

        Args:
            predictions: List of scalar predictions, one per model.
        """
        self._prediction_history.append(np.array(predictions))

    def compute_correlation_matrix(self) -> np.ndarray:
        """Compute pairwise correlation matrix from prediction history.

        Returns:
            (num_models, num_models) correlation matrix.
        """
        if len(self._prediction_history) < 10:
            return np.eye(self.num_models)

        history = np.array(list(self._prediction_history))  # (T, num_models)
        self._correlation_matrix = np.corrcoef(history.T)  # (M, M)

        # Handle NaN (constant predictions)
        self._correlation_matrix = np.nan_to_num(
            self._correlation_matrix, nan=0.0
        )
        return self._correlation_matrix

    def compute_diversity_metrics(self) -> Dict[str, float]:
        """Compute comprehensive diversity metrics.

        Returns:
            Dict with:
              - mean_correlation: Average pairwise correlation
              - max_correlation: Maximum pairwise correlation (excluding self)
              - disagreement: Average pairwise disagreement
              - entropy: Prediction entropy
        """
        corr = self.compute_correlation_matrix()

        # Exclude diagonal
        mask = ~np.eye(self.num_models, dtype=bool)
        off_diag = corr[mask]

        mean_corr = float(off_diag.mean()) if len(off_diag) > 0 else 0.0
        max_corr = float(off_diag.max()) if len(off_diag) > 0 else 0.0

        # Disagreement: fraction of pairs with different prediction signs
        disagreement = 0.0
        if len(self._prediction_history) >= 2:
            history = np.array(list(self._prediction_history))
            count = 0
            total = 0
            for i in range(self.num_models):
                for j in range(i + 1, self.num_models):
                    disagree = np.mean(
                        np.sign(history[:, i]) != np.sign(history[:, j])
                    )
                    disagreement += disagree
                    total += 1
            disagreement = disagreement / max(total, 1)

        # Entropy of prediction distribution
        entropy = 0.0
        if len(self._prediction_history) > 0:
            history = np.array(list(self._prediction_history))
            mean_preds = history.mean(axis=0)
            # Normalise to probabilities
            probs = np.abs(mean_preds)
            probs = probs / (probs.sum() + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))

        metrics = {
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
            "disagreement": disagreement,
            "entropy": float(entropy),
        }
        self._diversity_history.append(metrics)
        return metrics

    def adjust_weights_for_diversity(
        self, ensemble: Any
    ) -> Dict[str, float]:
        """Penalise weights of highly correlated models.

        If two models have correlation above the threshold, the lower-
        performing model gets a diversity penalty.

        Args:
            ensemble: DynamicWeightedEnsemble or AdaptiveEnsemble instance.

        Returns:
            Dict of model_name → diversity_penalty.
        """
        corr = self.compute_correlation_matrix()
        penalties: Dict[str, float] = {}

        for i in range(self.num_models):
            penalty = 0.0
            for j in range(self.num_models):
                if i != j and corr[i, j] > self.correlation_threshold:
                    # Penalise the worse-performing model
                    if self.models[i].mean_loss > self.models[j].mean_loss:
                        penalty += (corr[i, j] - self.correlation_threshold)

            penalty *= self.diversity_weight
            penalties[self.models[i].name] = penalty

            # Apply penalty to ensemble weights
            if hasattr(ensemble, "models"):
                for m in ensemble.models:
                    if m.name == self.models[i].name:
                        m.weight = max(m.weight * (1 - penalty), 0.01)

        # Renormalise
        if hasattr(ensemble, "_normalise_weights"):
            ensemble._normalise_weights()
        elif hasattr(ensemble, "models"):
            total = sum(m.weight for m in ensemble.models)
            if total > 0:
                for m in ensemble.models:
                    m.weight /= total

        return penalties

    def is_diverse_enough(self) -> bool:
        """Check if the ensemble has sufficient diversity.

        Returns:
            True if the maximum pairwise correlation is below threshold.
        """
        metrics = self.compute_diversity_metrics()
        return metrics["max_correlation"] < self.correlation_threshold

    @property
    def diversity_history(self) -> List[Dict[str, float]]:
        """Return the history of diversity metrics."""
        return self._diversity_history

    @property
    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Return the latest correlation matrix."""
        return self._correlation_matrix
