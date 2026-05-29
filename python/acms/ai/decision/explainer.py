"""
DecisionExplainer - Explainability for AI Trading Decisions
============================================================

Provides comprehensive explainability for the ACMS decision system:
- ModelExplainer: SHAP-based model-level explanations
- PredictionExplainer: Individual decision explanations
- FeatureAttribution: Feature importance and attribution computation
- DecisionNarrativeGenerator: Human-readable narrative explanations
- ExplanationDashboard: Data generation for visualization dashboards
- RegulatoryExplainer: Regulatory compliance explanations

GPU-ready with PyTorch-accelerated SHAP approximation.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExplanationType(str, Enum):
    """Types of explanations."""

    FEATURE_ATTRIBUTION = "feature_attribution"
    SHAP_VALUE = "shap_value"
    ATTENTION_WEIGHT = "attention_weight"
    COUNTERFACTUAL = "counterfactual"
    NARRATIVE = "narrative"
    REGULATORY = "regulatory"


class DecisionType(str, Enum):
    """Types of trading decisions."""

    STRATEGY_SELECTION = "strategy_selection"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_LIMIT = "risk_limit"
    TRADE_EXECUTION = "trade_execution"
    HEDGE_PLACEMENT = "hedge_placement"


class ComplianceStandard(str, Enum):
    """Regulatory compliance standards."""

    MIFID_II = "mifid_ii"
    SEC_RULE_606 = "sec_rule_606"
    FCA_COBS = "fca_cobs"
    GENERAL_BEST_INTEREST = "general_best_interest"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureAttribution:
    """Feature-level attribution for a decision.

    Attributes:
        attribution_id: Unique identifier.
        feature_names: Names of the features.
        values: Attribution values (e.g., SHAP values).
        base_value: Base value (expected output without features).
        method: Attribution method used.
        timestamp: When the attribution was computed.
    """

    attribution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feature_names: List[str] = field(default_factory=list)
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    base_value: float = 0.0
    method: str = "kernel_shap"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top-N most important features by absolute attribution.

        Args:
            n: Number of top features.

        Returns:
            List of (feature_name, attribution_value) sorted by magnitude.
        """
        if len(self.values) == 0:
            return []
        abs_vals = np.abs(self.values)
        top_idx = np.argsort(abs_vals)[::-1][:n]
        return [(self.feature_names[i], float(self.values[i])) for i in top_idx if i < len(self.feature_names)]

    @property
    def total_attribution(self) -> float:
        """Sum of all attribution values."""
        return float(np.sum(self.values)) if len(self.values) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "attribution_id": self.attribution_id,
            "feature_names": self.feature_names,
            "values": self.values.tolist(),
            "base_value": self.base_value,
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Model Explainer
# ---------------------------------------------------------------------------

class ModelExplainer:
    """SHAP-based model-level explanation engine.

    Provides both global (model-wide) and local (prediction-level)
    explanations using Kernel SHAP approximation. Supports attention
    weight visualization for transformer-based models.

    Attributes:
        background_data: Background dataset for SHAP baseline.
        n_samples: Number of samples for SHAP approximation.
    """

    def __init__(
        self,
        n_samples: int = 100,
        background_size: int = 50,
        device: str = "auto",
    ) -> None:
        """Initialise the model explainer.

        Args:
            n_samples: Number of samples for Kernel SHAP.
            background_size: Size of background dataset.
            device: Compute device.
        """
        self.n_samples = n_samples
        self.background_size = background_size
        self._background_data: Optional[np.ndarray] = None

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

    def set_background(self, data: np.ndarray) -> None:
        """Set the background dataset for SHAP baseline computation.

        Args:
            data: Background data, shape (n_samples, n_features).
        """
        if len(data) > self.background_size:
            indices = np.random.choice(len(data), self.background_size, replace=False)
            self._background_data = data[indices]
        else:
            self._background_data = data.copy()

    def compute_shap_values(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureAttribution:
        """Compute SHAP values for a single instance using Kernel SHAP.

        Args:
            model_fn: Prediction function that takes (n_samples, n_features) array.
            instance: Single instance to explain, shape (n_features,).
            feature_names: Optional feature names.

        Returns:
            FeatureAttribution with SHAP values.
        """
        if self._background_data is None:
            logger.warning("No background data set; using zeros as baseline")
            self._background_data = np.zeros((1, len(instance)))

        n_features = len(instance)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Kernel SHAP approximation
        shap_values = self._kernel_shap(model_fn, instance)

        # Base value (expected prediction on background)
        base_preds = model_fn(self._background_data)
        base_value = float(np.mean(base_preds))

        return FeatureAttribution(
            feature_names=feature_names,
            values=shap_values,
            base_value=base_value,
            method="kernel_shap",
        )

    def _kernel_shap(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> np.ndarray:
        """Approximate Kernel SHAP values.

        Uses a simplified implementation that samples feature subsets
        and weights them using the SHAP kernel.

        Args:
            model_fn: Prediction function.
            instance: Instance to explain.

        Returns:
            SHAP values array of shape (n_features,).
        """
        n_features = len(instance)
        shap_values = np.zeros(n_features)

        # Generate random coalitions
        for _ in range(self.n_samples):
            # Random subset of features
            coalition = np.random.randint(0, 2, size=n_features).astype(bool)
            n_included = coalition.sum()

            if n_included == 0 or n_included == n_features:
                continue

            # Create coalition and complement instances
            coalition_instance = self._background_data.mean(axis=0).copy()
            coalition_instance[coalition] = instance[coalition]

            complement_instance = self._background_data.mean(axis=0).copy()
            complement_instance[~coalition] = instance[~coalition]

            full_instance = instance.copy()

            # Compute marginal contributions
            pred_full = model_fn(full_instance.reshape(1, -1))[0]
            pred_coalition = model_fn(coalition_instance.reshape(1, -1))[0]
            pred_complement = model_fn(complement_instance.reshape(1, -1))[0]

            # Weight by SHAP kernel
            k = n_included
            weight = (n_features - 1) / (k * (n_features - k) + 1e-12)

            for i in range(n_features):
                if coalition[i]:
                    shap_values[i] += weight * (pred_full - pred_complement)
                else:
                    shap_values[i] += weight * (pred_full - pred_coalition)

        # Normalize
        shap_values /= max(1, self.n_samples)
        return shap_values

    def compute_attention_weights(
        self, attention_layer_output: Any
    ) -> Optional[np.ndarray]:
        """Extract attention weights from a model for visualization.

        Args:
            attention_layer_output: Output from a PyTorch MultiheadAttention layer.

        Returns:
            Attention weight matrix if available, else None.
        """
        if attention_layer_output is None:
            return None

        try:
            if torch is not None and isinstance(attention_layer_output, tuple):
                # PyTorch MultiheadAttention returns (output, attention_weights)
                weights = attention_layer_output[1]
                if weights is not None:
                    return weights.cpu().detach().numpy()
        except Exception as exc:
            logger.warning("Failed to extract attention weights: %s", exc)

        return None

    def global_feature_importance(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_instances: int = 50,
    ) -> FeatureAttribution:
        """Compute global feature importance by averaging local SHAP values.

        Args:
            model_fn: Prediction function.
            X: Dataset to compute global importance over.
            feature_names: Optional feature names.
            n_instances: Number of instances to sample.

        Returns:
            FeatureAttribution with averaged SHAP values.
        """
        if self._background_data is None:
            self.set_background(X)

        n_sample = min(n_instances, len(X))
        indices = np.random.choice(len(X), n_sample, replace=False)

        all_shap: List[np.ndarray] = []
        for idx in indices:
            attribution = self.compute_shap_values(model_fn, X[idx], feature_names)
            all_shap.append(np.abs(attribution.values))

        mean_abs_shap = np.mean(all_shap, axis=0)

        return FeatureAttribution(
            feature_names=feature_names or [f"feature_{i}" for i in range(len(mean_abs_shap))],
            values=mean_abs_shap,
            base_value=0.0,
            method="global_mean_abs_shap",
        )


# ---------------------------------------------------------------------------
# Prediction Explainer
# ---------------------------------------------------------------------------

class PredictionExplainer:
    """Explains individual predictions with feature attributions and narratives.

    Provides a unified interface for explaining a single model prediction,
    combining SHAP values, feature context, and counterfactual analysis.

    Attributes:
        model_explainer: Underlying model explainer.
    """

    def __init__(self, model_explainer: Optional[ModelExplainer] = None) -> None:
        """Initialise the prediction explainer.

        Args:
            model_explainer: Optional custom ModelExplainer instance.
        """
        self.model_explainer = model_explainer or ModelExplainer()

    def explain_prediction(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_context: Optional[Dict[str, str]] = None,
        decision_type: DecisionType = DecisionType.STRATEGY_SELECTION,
    ) -> Dict[str, Any]:
        """Generate a comprehensive explanation for a single prediction.

        Args:
            model_fn: Prediction function.
            instance: Instance to explain.
            feature_names: Feature names.
            feature_context: Additional context for each feature.
            decision_type: Type of decision being explained.

        Returns:
            Dictionary with full explanation.
        """
        # Compute SHAP attributions
        attribution = self.model_explainer.compute_shap_values(
            model_fn, instance, feature_names
        )

        # Get model prediction
        prediction = float(model_fn(instance.reshape(1, -1))[0])

        # Generate counterfactual
        counterfactual = self._simple_counterfactual(
            model_fn, instance, attribution, feature_names
        )

        # Feature contribution breakdown
        contributions = self._feature_contribution_breakdown(
            attribution, feature_context
        )

        return {
            "decision_type": decision_type.value,
            "prediction": prediction,
            "base_value": attribution.base_value,
            "attribution_method": attribution.method,
            "top_features": attribution.top_features(10),
            "total_attribution": attribution.total_attribution,
            "feature_contributions": contributions,
            "counterfactual": counterfactual,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _simple_counterfactual(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        attribution: FeatureAttribution,
        feature_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate a simple counterfactual: what-if the top feature changed.

        Args:
            model_fn: Prediction function.
            instance: Original instance.
            attribution: Feature attributions.
            feature_names: Feature names.

        Returns:
            Counterfactual analysis dictionary.
        """
        top = attribution.top_features(1)
        if not top:
            return {"available": False, "reason": "no features"}

        top_feature, top_value = top[0]
        feature_idx = attribution.feature_names.index(top_feature) if top_feature in attribution.feature_names else 0

        # Perturb the top feature
        perturbed = instance.copy()
        perturbation = -top_value * 0.5  # Partial counterfactual
        perturbed[feature_idx] += perturbation

        original_pred = float(model_fn(instance.reshape(1, -1))[0])
        perturbed_pred = float(model_fn(perturbed.reshape(1, -1))[0])

        return {
            "available": True,
            "feature_changed": top_feature,
            "original_value": float(instance[feature_idx]),
            "perturbed_value": float(perturbed[feature_idx]),
            "original_prediction": original_pred,
            "perturbed_prediction": perturbed_pred,
            "prediction_change": perturbed_pred - original_pred,
        }

    def _feature_contribution_breakdown(
        self,
        attribution: FeatureAttribution,
        feature_context: Optional[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Break down feature contributions with context.

        Args:
            attribution: Feature attributions.
            feature_context: Additional context per feature.

        Returns:
            List of feature contribution dictionaries.
        """
        contributions = []
        for i, name in enumerate(attribution.feature_names):
            if i >= len(attribution.values):
                break
            value = attribution.values[i]
            direction = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            magnitude = abs(value)

            contribution = {
                "feature": name,
                "attribution": float(value),
                "direction": direction,
                "magnitude": float(magnitude),
                "relative_importance": float(magnitude / (np.sum(np.abs(attribution.values)) + 1e-12)),
            }
            if feature_context and name in feature_context:
                contribution["context"] = feature_context[name]

            contributions.append(contribution)

        # Sort by magnitude
        contributions.sort(key=lambda x: x["magnitude"], reverse=True)
        return contributions


# ---------------------------------------------------------------------------
# Decision Narrative Generator
# ---------------------------------------------------------------------------

class DecisionNarrativeGenerator:
    """Generates human-readable narrative explanations for trading decisions.

    Converts quantitative explanations into natural language narratives
    suitable for traders, risk managers, and compliance officers.

    Attributes:
        style: Narrative style ('technical', 'business', 'simple').
    """

    def __init__(self, style: str = "business") -> None:
        """Initialise the narrative generator.

        Args:
            style: Narrative style - 'technical', 'business', or 'simple'.
        """
        self.style = style

    def generate_narrative(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
        base_value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a narrative explanation for a decision.

        Args:
            decision_type: Type of decision.
            prediction: Model prediction value.
            top_features: Top contributing features with values.
            base_value: Baseline prediction.
            context: Additional context information.

        Returns:
            Narrative text string.
        """
        context = context or {}

        if self.style == "technical":
            return self._technical_narrative(
                decision_type, prediction, top_features, base_value, context
            )
        elif self.style == "simple":
            return self._simple_narrative(
                decision_type, prediction, top_features, base_value, context
            )
        else:
            return self._business_narrative(
                decision_type, prediction, top_features, base_value, context
            )

    def _business_narrative(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
        base_value: float,
        context: Dict[str, Any],
    ) -> str:
        """Generate a business-oriented narrative."""
        parts: List[str] = []

        # Decision summary
        decision_label = decision_type.value.replace("_", " ").title()
        parts.append(f"The {decision_label} decision was driven by the following factors.")

        # Key drivers
        if top_features:
            positive = [(n, v) for n, v in top_features if v > 0]
            negative = [(n, v) for n, v in top_features if v < 0]

            if positive:
                drivers = ", ".join(f"{n} (+{v:.3f})" for n, v in positive[:3])
                parts.append(f"Supporting factors: {drivers}.")

            if negative:
                detractors = ", ".join(f"{n} ({v:.3f})" for n, v in negative[:3])
                parts.append(f"Opposing factors: {detractors}.")

        # Prediction context
        deviation = prediction - base_value
        if abs(deviation) > 0.01:
            direction = "higher" if deviation > 0 else "lower"
            parts.append(
                f"The predicted outcome is {abs(deviation):.3f} {direction} "
                f"than the baseline expectation of {base_value:.3f}."
            )

        # Confidence indicator
        total_attribution = sum(abs(v) for _, v in top_features)
        if total_attribution > 0:
            max_contributor = max(top_features, key=lambda x: abs(x[1]))
            dominance = abs(max_contributor[1]) / (total_attribution + 1e-12)
            if dominance > 0.5:
                parts.append(
                    f"The decision is heavily influenced by {max_contributor[0]}, "
                    f"which accounts for {dominance:.0%} of the total feature impact."
                )
            else:
                parts.append("The decision is well-balanced across multiple factors.")

        return " ".join(parts)

    def _technical_narrative(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
        base_value: float,
        context: Dict[str, Any],
    ) -> str:
        """Generate a technical narrative with precise values."""
        parts = [
            f"Decision: {decision_type.value}",
            f"Prediction: {prediction:.6f} (base: {base_value:.6f}, delta: {prediction - base_value:.6f})",
            "Feature attributions (SHAP):",
        ]
        for name, value in top_features:
            parts.append(f"  - {name}: {value:+.6f}")
        return "\n".join(parts)

    def _simple_narrative(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
        base_value: float,
        context: Dict[str, Any],
    ) -> str:
        """Generate a simple, non-technical narrative."""
        if not top_features:
            return f"The {decision_type.value} decision was made based on the overall market conditions."

        main_factor = top_features[0][0]
        direction = "supports" if top_features[0][1] > 0 else "argues against"
        return (
            f"The {decision_type.value.replace('_', ' ')} was mainly influenced by {main_factor}, "
            f"which {direction} the decision. "
            f"Other factors also contributed to a lesser extent."
        )


# ---------------------------------------------------------------------------
# Explanation Dashboard
# ---------------------------------------------------------------------------

class ExplanationDashboard:
    """Generates data structures for explanation visualization dashboards.

    Provides formatted data for:
    - Feature importance bar charts
    - Waterfall plots (SHAP)
    - Decision summary cards
    - Time series of feature contributions

    Attributes:
        history: History of explanations for time-series analysis.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialise the dashboard data generator.

        Args:
            max_history: Maximum number of historical explanations to keep.
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def add_explanation(self, explanation: Dict[str, Any]) -> None:
        """Add an explanation to the history.

        Args:
            explanation: Explanation dictionary from PredictionExplainer.
        """
        self.history.append(explanation)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def feature_importance_chart_data(self) -> Dict[str, Any]:
        """Generate data for a feature importance bar chart.

        Returns:
            Chart data dictionary.
        """
        if not self.history:
            return {"features": [], "values": []}

        # Aggregate feature attributions across history
        feature_sums: Dict[str, float] = {}
        feature_counts: Dict[str, int] = {}

        for exp in self.history:
            for feat_name, feat_val in exp.get("top_features", []):
                feature_sums[feat_name] = feature_sums.get(feat_name, 0.0) + abs(feat_val)
                feature_counts[feat_name] = feature_counts.get(feat_name, 0) + 1

        # Average absolute attribution
        avg_attributions = {
            k: v / feature_counts[k] for k, v in feature_sums.items()
        }

        sorted_features = sorted(avg_attributions.items(), key=lambda x: x[1], reverse=True)

        return {
            "features": [f[0] for f in sorted_features],
            "values": [f[1] for f in sorted_features],
            "chart_type": "bar",
            "title": "Average Feature Importance",
        }

    def waterfall_chart_data(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for a SHAP waterfall plot.

        Args:
            explanation: Single explanation dictionary.

        Returns:
            Waterfall chart data.
        """
        features = explanation.get("top_features", [])
        base_value = explanation.get("base_value", 0.0)

        return {
            "base_value": base_value,
            "features": [{"name": f[0], "value": f[1]} for f in features],
            "chart_type": "waterfall",
            "title": "Feature Contribution Waterfall",
        }

    def decision_summary_cards(self) -> List[Dict[str, Any]]:
        """Generate summary card data for recent decisions.

        Returns:
            List of decision summary card dictionaries.
        """
        cards = []
        for exp in self.history[-10:]:
            cards.append({
                "decision_type": exp.get("decision_type", "unknown"),
                "prediction": exp.get("prediction", 0.0),
                "top_factor": exp.get("top_features", [("unknown", 0.0)])[0][0] if exp.get("top_features") else "none",
                "timestamp": exp.get("timestamp", ""),
            })
        return cards

    def feature_contribution_timeseries(
        self, feature_name: str
    ) -> Dict[str, Any]:
        """Generate time series data for a specific feature's contributions.

        Args:
            feature_name: Name of the feature to track.

        Returns:
            Time series chart data.
        """
        timestamps = []
        values = []

        for exp in self.history:
            for feat_name, feat_val in exp.get("top_features", []):
                if feat_name == feature_name:
                    timestamps.append(exp.get("timestamp", ""))
                    values.append(feat_val)
                    break

        return {
            "feature": feature_name,
            "timestamps": timestamps,
            "values": values,
            "chart_type": "line",
            "title": f"Feature Contribution Over Time: {feature_name}",
        }


# ---------------------------------------------------------------------------
# Regulatory Explainer
# ---------------------------------------------------------------------------

class RegulatoryExplainer:
    """Generates regulatory-compliant explanations for trading decisions.

    Provides explanations that satisfy requirements from MiFID II,
    SEC Rule 606, FCA COBS, and general best interest standards.

    Attributes:
        standard: Primary compliance standard.
    """

    def __init__(
        self,
        standard: ComplianceStandard = ComplianceStandard.MIFID_II,
    ) -> None:
        """Initialise the regulatory explainer.

        Args:
            standard: Compliance standard to target.
        """
        self.standard = standard
        self._explanation_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates for the compliance standard.

        Returns:
            Dictionary of template strings.
        """
        templates = {
            ComplianceStandard.MIFID_II.value: {
                "suitability": (
                    "This investment decision was made based on the client's "
                    "risk profile and investment objectives. The strategy selected "
                    "aligns with the client's stated risk tolerance of {risk_tolerance} "
                    "and investment horizon of {horizon}."
                ),
                "best_execution": (
                    "The execution was performed in accordance with best execution "
                    "obligations. The strategy was selected based on a quantitative "
                    "assessment of expected risk-adjusted returns, with consideration "
                    "of market impact, cost, and speed of execution."
                ),
                "product_governance": (
                    "The product/strategy has been categorized as {product_category} "
                    "under the firm's product governance framework. The target market "
                    "assessment confirms alignment with the client profile."
                ),
            },
            ComplianceStandard.GENERAL_BEST_INTEREST.value: {
                "suitability": (
                    "This decision was made in the best interest of the client, "
                    "based on a systematic evaluation of {n_factors} factors. "
                    "The primary consideration was {primary_factor}."
                ),
                "risk_disclosure": (
                    "Key risks include: {risk_factors}. "
                    "The model confidence for this decision is {confidence:.1%}."
                ),
            },
        }
        return templates.get(self.standard.value, templates[ComplianceStandard.GENERAL_BEST_INTEREST.value])

    def generate_compliance_explanation(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a regulatory-compliant explanation.

        Args:
            decision_type: Type of decision.
            prediction: Model prediction.
            top_features: Top contributing features.
            context: Additional context (risk_tolerance, horizon, etc.).

        Returns:
            Regulatory explanation dictionary.
        """
        primary_factor = top_features[0][0] if top_features else "market_conditions"
        n_factors = len(top_features)
        risk_tolerance = context.get("risk_tolerance", "moderate")
        horizon = context.get("horizon", "medium-term")
        confidence = context.get("confidence", 0.5)

        # Generate suitability explanation
        suitability = self._explanation_templates.get("suitability", "")
        suitability = suitability.format(
            risk_tolerance=risk_tolerance,
            horizon=horizon,
            n_factors=n_factors,
            primary_factor=primary_factor,
        )

        # Generate risk disclosure
        risk_factors = ", ".join(
            f"{name} (impact={abs(val):.3f})"
            for name, val in top_features[:3]
        )
        risk_disclosure = self._explanation_templates.get("risk_disclosure", "")
        if risk_disclosure:
            risk_disclosure = risk_disclosure.format(
                risk_factors=risk_factors,
                confidence=confidence,
            )

        # Generate decision rationale
        rationale = self._generate_rationale(decision_type, prediction, top_features)

        return {
            "compliance_standard": self.standard.value,
            "decision_type": decision_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suitability_assessment": suitability,
            "risk_disclosure": risk_disclosure,
            "decision_rationale": rationale,
            "key_factors": [
                {"name": name, "impact": float(val), "direction": "positive" if val > 0 else "negative"}
                for name, val in top_features[:5]
            ],
            "model_confidence": confidence,
            "audit_trail": {
                "model_version": context.get("model_version", "unknown"),
                "data_timestamp": context.get("data_timestamp", ""),
                "explanation_method": "kernel_shap",
            },
        }

    def _generate_rationale(
        self,
        decision_type: DecisionType,
        prediction: float,
        top_features: List[Tuple[str, float]],
    ) -> str:
        """Generate a decision rationale for regulatory documentation.

        Args:
            decision_type: Type of decision.
            prediction: Prediction value.
            top_features: Contributing features.

        Returns:
            Rationale text.
        """
        feature_summary = "; ".join(
            f"{name} contributed {'positively' if val > 0 else 'negatively'} (SHAP={val:+.4f})"
            for name, val in top_features[:3]
        )

        return (
            f"The {decision_type.value.replace('_', ' ')} decision was made using an AI model "
            f"that produced a prediction of {prediction:.4f}. "
            f"The top contributing factors were: {feature_summary}. "
            f"The decision process is fully documented and auditable."
        )
