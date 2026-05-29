"""AI-specific types for the ACMS AI module.

Defines all core type aliases, dataclasses, and enums used throughout
the AI subsystem. These types provide a unified vocabulary for:
- Tensor representations of market data
- Model inputs, outputs, and predictions
- Training and evaluation state
- Uncertainty quantification
- Model performance and interpretability
- Trading signal and position recommendations
- Risk assessment
- Market regime and state representations

All types are designed to be serializable and framework-agnostic where
possible, with PyTorch-specific types clearly annotated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Type Aliases for Tensor Representations
# ============================================================================

# Market data tensors: shape (batch, seq_len, features) for multi-feature
# time series, or (batch, seq_len) for single feature (e.g., close prices).
# Typically float32 for GPU compatibility.
MarketTensor = np.ndarray

# Feature tensor: processed/engineered features ready for model input.
# Shape: (batch, n_features) for tabular models, (batch, seq_len, n_features)
# for sequence models.
FeatureTensor = np.ndarray

# Prediction tensor: raw model output before post-processing.
# Shape depends on task: (batch,) for regression, (batch, n_classes) for
# classification, (batch, seq_len, n_classes) for sequence prediction.
PredictionTensor = np.ndarray

# ============================================================================
# Enums
# ============================================================================


class ModelTask(str, Enum):
    """Supported AI model task types."""

    PRICE_PREDICTION = "price_prediction"
    DIRECTION_CLASSIFICATION = "direction_classification"
    REGIME_DETECTION = "regime_detection"
    VOLATILITY_FORECAST = "volatility_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    SIGNAL_GENERATION = "signal_generation"
    OPTIMAL_EXECUTION = "optimal_execution"
    RISK_ESTIMATION = "risk_estimation"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    FEATURE_EMBEDDING = "feature_embedding"


class PredictionType(str, Enum):
    """Type of prediction output."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    PROBABILISTIC = "probabilistic"
    SEQUENCE = "sequence"


class MarketRegime(str, Enum):
    """Market regime classifications."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class SignalStrength(str, Enum):
    """Trading signal strength levels."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class UncertaintyMethod(str, Enum):
    """Methods for uncertainty estimation."""

    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"
    QUANTILE = "quantile"
    CONFORMAL = "conformal"
    DEEP_ENSEMBLE = "deep_ensemble"


class TrainingPhase(str, Enum):
    """Training lifecycle phases."""

    INITIALIZATION = "initialization"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


# ============================================================================
# Model Input / Output Dataclasses
# ============================================================================


@dataclass
class ModelInput:
    """Structured model input for the AI pipeline.

    Encapsulates all data needed for a single model inference call,
    including raw features, metadata, and optional conditioning
    information.

    Attributes:
        features: Primary feature array of shape (n_features,) or
            (seq_len, n_features).
        symbol: Trading pair symbol (e.g., 'BTC/USDT').
        timeframe: Data timeframe (e.g., '1h', '5m').
        timestamp: Point-in-time for the input data.
        attention_mask: Optional binary mask for variable-length sequences.
            Shape matches features' seq_len dimension.
        conditioning: Optional conditioning variables for conditional
            generation (e.g., market regime, volatility bucket).
        metadata: Additional context information.
    """

    features: np.ndarray
    symbol: str = ""
    timeframe: str = ""
    timestamp: Optional[datetime] = None
    attention_mask: Optional[np.ndarray] = None
    conditioning: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.conditioning is None:
            self.conditioning = {}


@dataclass
class ModelOutput:
    """Structured model output from the AI pipeline.

    Contains the raw prediction, post-processed result, and
    associated metadata for downstream consumption.

    Attributes:
        prediction: Raw model prediction array.
        prediction_type: Type of prediction (regression, classification, etc.).
        symbol: Trading pair symbol this prediction is for.
        timestamp: Time of prediction generation.
        model_id: Identifier of the model that produced this output.
        model_version: Version string of the model.
        latency_ms: Inference latency in milliseconds.
        confidence: Model confidence score in [0, 1].
        metadata: Additional output metadata.
    """

    prediction: np.ndarray
    prediction_type: PredictionType = PredictionType.REGRESSION
    symbol: str = ""
    timestamp: Optional[datetime] = None
    model_id: str = ""
    model_version: str = ""
    latency_ms: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


# ============================================================================
# Training and Evaluation Types
# ============================================================================


@dataclass
class TrainingState:
    """Current state of a model training run.

    Tracks all relevant state for training lifecycle management,
    including optimization state, learning rate, and performance history.

    Attributes:
        epoch: Current epoch number.
        global_step: Total number of optimizer steps taken.
        learning_rate: Current learning rate.
        train_loss: Most recent training loss value.
        val_loss: Most recent validation loss value.
        best_val_loss: Best validation loss observed.
        best_epoch: Epoch at which best validation loss was achieved.
        patience_counter: Number of epochs without improvement.
        phase: Current training phase.
        gradient_norm: L2 norm of gradients at current step.
        elapsed_seconds: Total training wall-clock time.
        epochs_history: Loss history per epoch.
        lr_history: Learning rate history per epoch.
    """

    epoch: int = 0
    global_step: int = 0
    learning_rate: float = 0.001
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    patience_counter: int = 0
    phase: TrainingPhase = TrainingPhase.INITIALIZATION
    gradient_norm: float = 0.0
    elapsed_seconds: float = 0.0
    epochs_history: List[Dict[str, float]] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)

    def record_epoch(self, metrics: Dict[str, float]) -> None:
        """Record metrics for the current epoch.

        Args:
            metrics: Dictionary of metric name -> value for this epoch.
        """
        self.epochs_history.append({"epoch": self.epoch, **metrics})
        self.lr_history.append(self.learning_rate)

        if "val_loss" in metrics and metrics["val_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            self.best_epoch = self.epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize training state to a dictionary."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "learning_rate": self.learning_rate,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "phase": self.phase.value,
            "gradient_norm": self.gradient_norm,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class EvaluationResult:
    """Result of model evaluation on a dataset.

    Comprehensive evaluation metrics for classification and regression
    tasks, with support for time-series-specific metrics.

    Attributes:
        model_id: Identifier of the evaluated model.
        task: Model task type.
        metrics: Dictionary of metric name -> value.
        predictions: Raw prediction array on evaluation set.
        targets: Ground truth targets.
        confusion_matrix: Optional confusion matrix for classification.
        per_class_metrics: Optional per-class metrics for classification.
        calibration_error: Expected calibration error for probabilistic outputs.
        timestamp: When the evaluation was performed.
    """

    model_id: str = ""
    task: ModelTask = ModelTask.PRICE_PREDICTION
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    calibration_error: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def accuracy(self) -> Optional[float]:
        """Return accuracy if this is a classification result."""
        return self.metrics.get("accuracy")

    @property
    def f1_score(self) -> Optional[float]:
        """Return macro F1 score if available."""
        return self.metrics.get("f1_macro")

    @property
    def mse(self) -> Optional[float]:
        """Return mean squared error if this is a regression result."""
        return self.metrics.get("mse")

    @property
    def mae(self) -> Optional[float]:
        """Return mean absolute error if available."""
        return self.metrics.get("mae")

    @property
    def sharpe_ratio(self) -> Optional[float]:
        """Return Sharpe ratio of prediction-based strategy if available."""
        return self.metrics.get("sharpe_ratio")

    def summary(self) -> str:
        """Generate a human-readable summary of evaluation results."""
        lines = [f"Evaluation Result for {self.model_id}", "-" * 40]
        lines.append(f"Task: {self.task.value}")
        lines.append(f"Timestamp: {self.timestamp.isoformat()}")
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.6f}")
        if self.calibration_error is not None:
            lines.append(f"  calibration_error: {self.calibration_error:.6f}")
        return "\n".join(lines)


# ============================================================================
# Uncertainty and Probabilistic Types
# ============================================================================


@dataclass
class PredictionWithUncertainty:
    """Prediction with uncertainty bounds.

    Wraps a point prediction with estimated uncertainty, supporting
    multiple uncertainty quantification methods.

    Attributes:
        point_estimate: Central prediction value(s).
        lower_bound: Lower bound of prediction interval.
        upper_bound: Upper bound of prediction interval.
        confidence_level: Confidence level of the interval (e.g., 0.95).
        std_deviation: Estimated standard deviation of the prediction.
        method: Method used for uncertainty estimation.
        n_samples: Number of forward passes (for MC dropout / ensemble).
        raw_samples: Raw prediction samples (optional, for diagnostics).
    """

    point_estimate: np.ndarray
    lower_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_level: float = 0.95
    std_deviation: Optional[np.ndarray] = None
    method: UncertaintyMethod = UncertaintyMethod.MC_DROPOUT
    n_samples: int = 0
    raw_samples: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # If bounds are empty, derive from std_deviation if available
        if len(self.lower_bound) == 0 and self.std_deviation is not None:
            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(
                self.confidence_level, 1.96
            )
            self.lower_bound = self.point_estimate - z_score * self.std_deviation
            self.upper_bound = self.point_estimate + z_score * self.std_deviation

    @property
    def interval_width(self) -> Optional[np.ndarray]:
        """Width of the prediction interval."""
        if len(self.upper_bound) > 0 and len(self.lower_bound) > 0:
            return self.upper_bound - self.lower_bound
        return None

    @property
    def relative_uncertainty(self) -> Optional[np.ndarray]:
        """Relative uncertainty as interval_width / |point_estimate|."""
        width = self.interval_width
        if width is not None:
            denominator = np.abs(self.point_estimate) + 1e-10
            return width / denominator
        return None


# ============================================================================
# Model Performance and Interpretability Types
# ============================================================================


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance tracking.

    Aggregates multiple dimensions of model quality including
    predictive accuracy, latency, and financial performance.

    Attributes:
        model_id: Identifier of the model.
        model_version: Version string.
        task: Model task type.
        accuracy_metrics: Classification/regression accuracy metrics.
        financial_metrics: Trading-specific metrics (Sharpe, max DD, etc.).
        latency_metrics: Inference latency statistics.
        stability_metrics: Prediction stability over time.
        data_range: Date range of the evaluation data.
        n_samples: Number of samples used for evaluation.
        last_updated: When metrics were last computed.
    """

    model_id: str = ""
    model_version: str = ""
    task: ModelTask = ModelTask.PRICE_PREDICTION
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    financial_metrics: Dict[str, float] = field(default_factory=dict)
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    data_range: Tuple[Optional[str], Optional[str]] = (None, None)
    n_samples: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def is_degraded(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """Check if model performance has degraded beyond thresholds.

        Args:
            thresholds: Optional dict of metric_name -> threshold_value.
                If not provided, uses default thresholds.

        Returns:
            True if any metric exceeds its degradation threshold.
        """
        defaults = {
            "accuracy": 0.5,
            "f1_macro": 0.4,
            "sharpe_ratio": 0.0,
        }
        if thresholds:
            defaults.update(thresholds)

        all_metrics = {**self.accuracy_metrics, **self.financial_metrics}
        for metric_name, threshold in defaults.items():
            if metric_name in all_metrics and all_metrics[metric_name] < threshold:
                logger.warning(
                    "Model %s degraded on %s: %.4f < %.4f",
                    self.model_id, metric_name, all_metrics[metric_name], threshold,
                )
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "task": self.task.value,
            "accuracy_metrics": self.accuracy_metrics,
            "financial_metrics": self.financial_metrics,
            "latency_metrics": self.latency_metrics,
            "stability_metrics": self.stability_metrics,
            "n_samples": self.n_samples,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class FeatureImportance:
    """Feature importance scores for model interpretability.

    Supports multiple importance calculation methods and provides
    both global and per-feature importance rankings.

    Attributes:
        feature_names: Names of features in order.
        importance_scores: Raw importance scores aligned with feature_names.
        method: Method used to compute importance.
        model_id: Source model identifier.
        std_errors: Optional standard errors for importance scores.
        permutation_delta: Optional change in metric when feature is permuted.
    """

    feature_names: List[str]
    importance_scores: np.ndarray
    method: str = "integrated_gradients"
    model_id: str = ""
    std_errors: Optional[np.ndarray] = None
    permutation_delta: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.importance_scores):
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must match "
                f"importance_scores length ({len(self.importance_scores)})"
            )

    @property
    def ranked_features(self) -> List[Tuple[str, float]]:
        """Return features ranked by importance (descending)."""
        indices = np.argsort(self.importance_scores)[::-1]
        return [
            (self.feature_names[i], float(self.importance_scores[i]))
            for i in indices
        ]

    @property
    def top_k(self) -> List[Tuple[str, float]]:
        """Return top features (positive importance only)."""
        return [
            (name, score) for name, score in self.ranked_features
            if score > 0
        ]

    def normalize(self) -> "FeatureImportance":
        """Return a copy with scores normalized to [0, 1] range.

        Returns:
            New FeatureImportance instance with normalized scores.
        """
        min_val = np.min(self.importance_scores)
        max_val = np.max(self.importance_scores)
        range_val = max_val - min_val
        if range_val < 1e-10:
            normalized = np.zeros_like(self.importance_scores)
        else:
            normalized = (self.importance_scores - min_val) / range_val
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=normalized,
            method=self.method,
            model_id=self.model_id,
            std_errors=self.std_errors,
            permutation_delta=self.permutation_delta,
        )


@dataclass
class ExplanationResult:
    """Model interpretability explanation result.

    Contains attribution scores and contextual information
    explaining why a model produced a particular prediction.

    Attributes:
        model_id: Source model identifier.
        input_summary: Summary of the input that was explained.
        attribution_scores: Per-feature attribution scores.
            Positive = contributes to prediction, negative = against.
        base_value: Model's base prediction (without features).
        predicted_value: Model's actual prediction.
        method: Explanation method used (e.g., 'shap', 'integrated_gradients').
        interaction_values: Optional pairwise feature interaction values.
        text_explanation: Human-readable explanation string.
        confidence_in_explanation: Confidence score for the explanation itself.
    """

    model_id: str = ""
    input_summary: Dict[str, Any] = field(default_factory=dict)
    attribution_scores: Dict[str, float] = field(default_factory=dict)
    base_value: float = 0.0
    predicted_value: float = 0.0
    method: str = "shap"
    interaction_values: Optional[Dict[Tuple[str, str], float]] = None
    text_explanation: str = ""
    confidence_in_explanation: float = 1.0

    @property
    def top_positive_features(self) -> List[Tuple[str, float]]:
        """Features with highest positive attribution, sorted descending."""
        positive = {k: v for k, v in self.attribution_scores.items() if v > 0}
        return sorted(positive.items(), key=lambda x: x[1], reverse=True)

    @property
    def top_negative_features(self) -> List[Tuple[str, float]]:
        """Features with highest negative attribution, sorted ascending."""
        negative = {k: v for k, v in self.attribution_scores.items() if v < 0}
        return sorted(negative.items(), key=lambda x: x[1])

    def generate_text(self) -> str:
        """Generate a human-readable explanation from attribution scores."""
        parts = [f"Model {self.model_id} predicted {self.predicted_value:.4f}"]
        parts.append(f"(base value: {self.base_value:.4f})")

        if self.top_positive_features:
            top_pos = self.top_positive_features[:3]
            pos_str = ", ".join(
                f"{name} (+{score:.4f})" for name, score in top_pos
            )
            parts.append(f"Top positive contributors: {pos_str}")

        if self.top_negative_features:
            top_neg = self.top_negative_features[:3]
            neg_str = ", ".join(
                f"{name} ({score:.4f})" for name, score in top_neg
            )
            parts.append(f"Top negative contributors: {neg_str}")

        return ". ".join(parts) + "."


# ============================================================================
# Trading Signal and Position Types
# ============================================================================


@dataclass
class RegimePrediction:
    """Market regime prediction with confidence and transition probabilities.

    Attributes:
        current_regime: Predicted current market regime.
        confidence: Confidence in the regime classification [0, 1].
        regime_probabilities: Probability distribution over all regimes.
        transition_probabilities: Probability of transitioning to each regime
            in the next period.
        regime_duration_estimate: Estimated number of periods until regime change.
        timestamp: Time of prediction.
        symbol: Trading pair symbol.
        model_id: Source model identifier.
    """

    current_regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    transition_probabilities: Dict[str, float] = field(default_factory=dict)
    regime_duration_estimate: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    model_id: str = ""

    def most_likely_next_regime(self) -> MarketRegime:
        """Return the most likely next regime based on transition probabilities.

        Returns:
            MarketRegime enum member with highest transition probability.
        """
        if not self.transition_probabilities:
            return self.current_regime
        best = max(self.transition_probabilities.items(), key=lambda x: x[1])
        try:
            return MarketRegime(best[0])
        except ValueError:
            return MarketRegime.UNKNOWN


@dataclass
class SignalPrediction:
    """AI-generated trading signal with uncertainty.

    Combines a directional signal with uncertainty estimates
    and optional regime context.

    Attributes:
        symbol: Trading pair symbol.
        direction: Predicted direction (1=long, -1=short, 0=neutral).
        strength: Signal strength in [0, 1].
        signal_strength: Categorical signal strength level.
        confidence: Model confidence in the signal [0, 1].
        uncertainty: PredictionWithUncertainty if available.
        timeframe: Signal timeframe.
        target_horizon: Number of bars forward the signal targets.
        regime_context: Current regime prediction for context.
        model_id: Source model identifier.
        timestamp: Time of signal generation.
        features_used: Names of features that contributed most.
        metadata: Additional signal metadata.
    """

    symbol: str = ""
    direction: int = 0
    strength: float = 0.0
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0
    uncertainty: Optional[PredictionWithUncertainty] = None
    timeframe: str = ""
    target_horizon: int = 1
    regime_context: Optional[RegimePrediction] = None
    model_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Whether the signal is strong enough to act on."""
        return self.confidence >= 0.6 and abs(self.direction) > 0 and self.strength >= 0.3

    def to_signal_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for the signal engine."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "timeframe": self.timeframe,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PositionRecommendation:
    """AI-generated position recommendation.

    Combines signal, risk, and regime information into an actionable
    position recommendation for the portfolio engine.

    Attributes:
        symbol: Trading pair symbol.
        side: Recommended side ('buy', 'sell', 'close', 'hold').
        size_fraction: Recommended position size as fraction of portfolio [0, 1].
        entry_price: Recommended entry price.
        stop_loss: Stop loss price.
        take_profit: Take profit price(s).
        confidence: Overall confidence in the recommendation.
        expected_return: Expected return estimate.
        expected_risk: Expected risk (std dev of return).
        risk_reward_ratio: Expected risk/reward ratio.
        max_drawdown_estimate: Estimated maximum drawdown.
        regime: Current market regime context.
        signal: Source signal prediction.
        urgency: How quickly the position should be entered (0=none, 1=critical).
        time_horizon: Recommended holding period in bars.
        model_id: Source model identifier.
        timestamp: Time of recommendation.
    """

    symbol: str = ""
    side: str = "hold"
    size_fraction: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: List[float] = field(default_factory=list)
    confidence: float = 0.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    risk_reward_ratio: float = 0.0
    max_drawdown_estimate: float = 0.0
    regime: Optional[RegimePrediction] = None
    signal: Optional[SignalPrediction] = None
    urgency: float = 0.0
    time_horizon: int = 0
    model_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_valid(self) -> bool:
        """Check if the recommendation is internally consistent."""
        if self.side not in ("buy", "sell", "close", "hold"):
            return False
        if not 0.0 <= self.size_fraction <= 1.0:
            return False
        if not 0.0 <= self.confidence <= 1.0:
            return False
        if self.side == "buy" and self.stop_loss is not None and self.entry_price is not None:
            if self.stop_loss >= self.entry_price:
                return False
        if self.side == "sell" and self.stop_loss is not None and self.entry_price is not None:
            if self.stop_loss <= self.entry_price:
                return False
        return True


# ============================================================================
# Risk Assessment Types
# ============================================================================


@dataclass
class RiskAssessment:
    """AI-generated risk assessment for a position or portfolio.

    Provides a comprehensive risk evaluation combining model-based
    estimates with traditional risk metrics.

    Attributes:
        overall_risk_level: Aggregate risk level classification.
        risk_score: Numerical risk score in [0, 1] (0=safe, 1=extreme).
        var_estimate: Value at Risk estimate.
        cvar_estimate: Conditional VaR (Expected Shortfall) estimate.
        max_drawdown_estimate: Estimated maximum drawdown.
        volatility_estimate: Estimated volatility.
        correlation_risk: Portfolio correlation risk score [0, 1].
        liquidity_risk: Liquidity risk score [0, 1].
        regime_risk: Risk due to potential regime change.
        concentration_risk: Concentration risk score [0, 1].
        tail_risk: Tail risk score [0, 1].
        model_uncertainty: Risk from model prediction uncertainty.
        factors: Dictionary of risk factor name -> contribution.
        recommendations: List of risk mitigation recommendations.
        timestamp: Time of assessment.
        model_id: Source model identifier.
    """

    overall_risk_level: RiskLevel = RiskLevel.MODERATE
    risk_score: float = 0.5
    var_estimate: float = 0.0
    cvar_estimate: float = 0.0
    max_drawdown_estimate: float = 0.0
    volatility_estimate: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0
    regime_risk: float = 0.0
    concentration_risk: float = 0.0
    tail_risk: float = 0.0
    model_uncertainty: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_id: str = ""

    @property
    def is_high_risk(self) -> bool:
        """Whether the overall risk level is high or above."""
        high_levels = {RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME}
        return self.overall_risk_level in high_levels

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_risk_level": self.overall_risk_level.value,
            "risk_score": self.risk_score,
            "var_estimate": self.var_estimate,
            "cvar_estimate": self.cvar_estimate,
            "max_drawdown_estimate": self.max_drawdown_estimate,
            "volatility_estimate": self.volatility_estimate,
            "correlation_risk": self.correlation_risk,
            "liquidity_risk": self.liquidity_risk,
            "regime_risk": self.regime_risk,
            "concentration_risk": self.concentration_risk,
            "tail_risk": self.tail_risk,
            "model_uncertainty": self.model_uncertainty,
            "factors": self.factors,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Market State Vector
# ============================================================================


@dataclass
class MarketStateVector:
    """Complete market state representation for AI models.

    Encapsulates all relevant market information at a point in time,
    serving as the canonical input format for the AI pipeline.

    Attributes:
        symbol: Trading pair symbol.
        timestamp: Point in time for this state.
        prices: OHLCV price data (open, high, low, close, volume).
        technical_indicators: Computed technical indicator values.
        orderbook_features: Order book derived features (imbalance, depth, etc.).
        cross_asset_features: Features from correlated assets.
        macro_features: Macro/market-wide features (funding rate, OI, etc.).
        sentiment_features: Sentiment and alternative data features.
        regime: Current regime classification.
        volatility_regime: Volatility regime (low/normal/high).
        session: Trading session identifier.
        features_vector: Pre-computed concatenated feature vector.
        feature_names: Names aligned with features_vector.
    """

    symbol: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    prices: Dict[str, float] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    orderbook_features: Dict[str, float] = field(default_factory=dict)
    cross_asset_features: Dict[str, float] = field(default_factory=dict)
    macro_features: Dict[str, float] = field(default_factory=dict)
    sentiment_features: Dict[str, float] = field(default_factory=dict)
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility_regime: str = "normal"
    session: str = ""
    features_vector: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)

    @property
    def total_feature_count(self) -> int:
        """Count total number of scalar features across all categories."""
        count = 0
        for feat_dict in (
            self.technical_indicators,
            self.orderbook_features,
            self.cross_asset_features,
            self.macro_features,
            self.sentiment_features,
        ):
            count += len(feat_dict)
        return count

    def to_feature_array(self) -> np.ndarray:
        """Concatenate all features into a single numpy array.

        Returns:
            1-D float32 array of all features in canonical order.
        """
        if self.features_vector is not None:
            return self.features_vector

        parts: List[float] = []
        names: List[str] = []

        for category, prefix in [
            (self.technical_indicators, "ti"),
            (self.orderbook_features, "ob"),
            (self.cross_asset_features, "xa"),
            (self.macro_features, "macro"),
            (self.sentiment_features, "sent"),
        ]:
            for name, value in sorted(category.items()):
                parts.append(float(value))
                names.append(f"{prefix}_{name}")

        self.feature_names = names
        self.features_vector = np.array(parts, dtype=np.float32)
        return self.features_vector

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excludes numpy arrays)."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "prices": self.prices,
            "technical_indicators": self.technical_indicators,
            "orderbook_features": self.orderbook_features,
            "cross_asset_features": self.cross_asset_features,
            "macro_features": self.macro_features,
            "sentiment_features": self.sentiment_features,
            "regime": self.regime.value,
            "volatility_regime": self.volatility_regime,
            "session": self.session,
            "total_feature_count": self.total_feature_count,
        }
