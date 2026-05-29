"""
AIRiskManager - AI-Driven Risk Management for Crypto Trading
==============================================================

Implements comprehensive AI-powered risk management with:
- NeuralVaR: Neural conditional quantile regression for VaR estimation
- StressScenarioGenerator: GAN-based stress scenario generation
- RealTimeRiskPredictor: Real-time risk prediction and monitoring
- AdaptiveRiskLimiter: Adaptive risk limit adjustment
- TailRiskHedger: Tail risk hedging recommendations
- RiskExplainer: Risk explainability and attribution
- AIRiskManager: Unified risk management interface

GPU-ready with PyTorch-based neural models.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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

class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class VaRMethod(str, Enum):
    """Value at Risk computation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    NEURAL = "neural"


class StressScenarioType(str, Enum):
    """Types of stress scenarios."""

    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_SPIKE = "volatility_spike"
    FLASH_CRASH = "flash_crash"
    DEPEG_EVENT = "depeg_event"
    REGULATORY_SHOCK = "regulatory_shock"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics snapshot.

    Attributes:
        timestamp: When the metrics were computed.
        var_95: Value at Risk at 95% confidence.
        var_99: Value at Risk at 99% confidence.
        cvar_95: Conditional VaR (Expected Shortfall) at 95%.
        cvar_99: Conditional VaR at 99%.
        max_drawdown: Current maximum drawdown.
        volatility: Annualized volatility.
        sharpe: Sharpe ratio.
        sortino: Sortino ratio.
        beta: Portfolio beta.
        tracking_error: Tracking error vs benchmark.
        risk_level: Overall risk level classification.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "beta": self.beta,
            "risk_level": self.risk_level.value,
        }


@dataclass
class StressTestResult:
    """Result of a stress test scenario.

    Attributes:
        scenario_type: Type of stress scenario.
        scenario_name: Descriptive name.
        portfolio_loss: Estimated portfolio loss under the scenario.
        affected_assets: Assets most impacted.
        probability: Estimated probability of the scenario.
        recovery_time_hours: Estimated recovery time.
        recommendations: Hedging or risk mitigation recommendations.
    """

    scenario_type: StressScenarioType = StressScenarioType.MARKET_CRASH
    scenario_name: str = ""
    portfolio_loss: float = 0.0
    affected_assets: List[str] = field(default_factory=list)
    probability: float = 0.0
    recovery_time_hours: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "scenario_type": self.scenario_type.value,
            "scenario_name": self.scenario_name,
            "portfolio_loss": self.portfolio_loss,
            "affected_assets": self.affected_assets,
            "probability": self.probability,
            "recovery_time_hours": self.recovery_time_hours,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Neural VaR
# ---------------------------------------------------------------------------

class NeuralVaR:
    """Neural conditional quantile regression for VaR estimation.

    Uses quantile regression neural networks to estimate VaR at
    arbitrary confidence levels. Unlike parametric methods, this
    captures non-linear dependencies and fat tails.

    Attributes:
        n_features: Number of input features.
        quantiles: List of quantiles to estimate.
    """

    def __init__(
        self,
        n_features: int = 20,
        quantiles: Optional[List[float]] = None,
        hidden_dim: int = 64,
        device: str = "auto",
    ) -> None:
        """Initialise the Neural VaR estimator.

        Args:
            n_features: Number of input features for VaR prediction.
            quantiles: List of quantiles to estimate (e.g., [0.05, 0.01]).
            hidden_dim: Hidden layer dimension.
            device: Compute device.
        """
        self.n_features = n_features
        self.quantiles = quantiles or [0.05, 0.01, 0.025]
        self.hidden_dim = hidden_dim

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        if torch is not None and nn is not None:
            # Quantile regression network
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, len(self.quantiles)),
            ).to(self._device)
        else:
            self.net = None

    def predict(
        self, features: np.ndarray, method: VaRMethod = VaRMethod.NEURAL
    ) -> Dict[float, float]:
        """Predict VaR at specified quantile levels.

        Args:
            features: Input features, shape (n_features,) or (batch, n_features).
            method: VaR computation method.

        Returns:
            Dictionary mapping quantile level to VaR estimate.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if method == VaRMethod.NEURAL and self.net is not None and torch is not None:
            return self._predict_neural(features)
        elif method == VaRMethod.HISTORICAL:
            return self._predict_historical(features)
        elif method == VaRMethod.PARAMETRIC:
            return self._predict_parametric(features)
        else:
            return self._predict_historical(features)

    def _predict_neural(self, features: np.ndarray) -> Dict[float, float]:
        """Neural quantile regression prediction."""
        x = torch.tensor(features, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            quantile_values = self.net(x).cpu().numpy()

        results: Dict[float, float] = {}
        for i, q in enumerate(self.quantiles):
            # Negative because VaR is typically reported as a positive loss
            results[q] = float(-quantile_values[0, i])
        return results

    def _predict_historical(self, features: np.ndarray) -> Dict[float, float]:
        """Historical simulation VaR."""
        # Use the last column as returns (common convention)
        returns = features[:, -1] if features.shape[1] > 0 else features.flatten()
        results: Dict[float, float] = {}
        for q in self.quantiles:
            var = float(-np.percentile(returns, q * 100))
            results[q] = var
        return results

    def _predict_parametric(self, features: np.ndarray) -> Dict[float, float]:
        """Parametric (Gaussian) VaR."""
        returns = features[:, -1] if features.shape[1] > 0 else features.flatten()
        mu = np.mean(returns)
        sigma = np.std(returns) + 1e-8
        results: Dict[float, float] = {}
        for q in self.quantiles:
            from scipy.stats import norm
            var = float(-(mu + sigma * norm.ppf(q)))
            results[q] = var
        return results

    def compute_cvar(
        self, returns: np.ndarray, var_level: float = 0.05
    ) -> float:
        """Compute Conditional VaR (Expected Shortfall).

        Args:
            returns: Array of portfolio returns.
            var_level: VaR confidence level (e.g., 0.05 for 95% VaR).

        Returns:
            CVaR value.
        """
        var_threshold = np.percentile(returns, var_level * 100)
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return float(var_threshold)
        return float(-np.mean(tail_returns))


# ---------------------------------------------------------------------------
# Stress Scenario Generator
# ---------------------------------------------------------------------------

class StressScenarioGenerator:
    """Parametric stress scenario generation for risk testing.

    Generates extreme market scenarios using fat-tailed distributions
    and scenario-specific transformations. The neural network generator
    is available for trained model-based generation but falls back
    to parametric methods when not trained.

    Attributes:
        n_assets: Number of assets in the portfolio.
        scenario_types: Types of stress scenarios to generate.
    """

    def __init__(
        self,
        n_assets: int = 10,
        scenario_types: Optional[List[StressScenarioType]] = None,
        device: str = "auto",
    ) -> None:
        """Initialise the stress scenario generator.

        Args:
            n_assets: Number of portfolio assets.
            scenario_types: Stress scenario types to generate.
            device: Compute device.
        """
        self.n_assets = n_assets
        self.scenario_types = scenario_types or list(StressScenarioType)
        self._device = "cuda" if (device == "auto" and GPU_AVAILABLE) else "cpu"
        self._use_neural = False  # Disabled until generator is properly trained

        if torch is not None and nn is not None:
            # Scenario generator network (Generator-like)
            # NOTE: This network requires training before use.
            # Until trained, parametric methods are used for generation.
            self.generator = nn.Sequential(
                nn.Linear(32 + len(self.scenario_types), 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_assets),
                nn.Tanh(),  # Bounded returns
            ).to(self._device)
            self.generator.eval()  # Keep in eval mode until trained
        else:
            self.generator = None

    def generate_scenarios(
        self,
        base_returns: np.ndarray,
        base_covariance: np.ndarray,
        n_scenarios: int = 100,
        severity: float = 1.0,
    ) -> List[StressTestResult]:
        """Generate stress test scenarios.

        Args:
            base_returns: Base expected returns.
            base_covariance: Base covariance matrix.
            n_scenarios: Number of scenarios per type.
            severity: Severity multiplier [0.5, 3.0].

        Returns:
            List of StressTestResult objects.
        """
        results: List[StressTestResult] = []

        for scenario_type in self.scenario_types:
            stressed_returns = self._apply_stress(
                base_returns, base_covariance, scenario_type, severity
            )
            result = self._evaluate_stress(
                scenario_type, stressed_returns, base_covariance
            )
            results.append(result)

        return results

    def _apply_stress(
        self,
        base_returns: np.ndarray,
        base_covariance: np.ndarray,
        scenario_type: StressScenarioType,
        severity: float,
    ) -> np.ndarray:
        """Apply stress transformation to base returns.

        Args:
            base_returns: Base expected returns.
            base_covariance: Base covariance matrix.
            scenario_type: Type of stress to apply.
            severity: Severity multiplier.

        Returns:
            Stressed returns array.
        """
        stressed = base_returns.copy()

        if scenario_type == StressScenarioType.MARKET_CRASH:
            # Uniform negative shock with fat tails
            shock = -0.10 * severity + np.random.standard_t(df=3, size=len(base_returns)) * 0.05 * severity
            stressed += shock

        elif scenario_type == StressScenarioType.LIQUIDITY_CRISIS:
            # Increased volatility, wider spreads
            stressed *= (1 - 0.3 * severity)
            stressed += np.random.normal(0, 0.15 * severity, size=len(base_returns))

        elif scenario_type == StressScenarioType.CORRELATION_BREAKDOWN:
            # Correlations spike to 1 (everything moves together)
            common_factor = np.random.normal(-0.05 * severity, 0.10 * severity)
            stressed = stressed * (1 - 0.5 * severity) + common_factor

        elif scenario_type == StressScenarioType.VOLATILITY_SPIKE:
            # Volatility increases 3-5x
            stressed += np.random.normal(0, 0.30 * severity, size=len(base_returns))

        elif scenario_type == StressScenarioType.FLASH_CRASH:
            # Sudden extreme drop followed by partial recovery
            crash_depth = -0.20 * severity
            stressed += crash_depth + np.random.exponential(0.05 * severity, size=len(base_returns))

        elif scenario_type == StressScenarioType.DEPEG_EVENT:
            # Stablecoin or pegged asset depegs
            stressed += np.random.choice([-0.15, 0.0], size=len(base_returns), p=[0.3, 0.7]) * severity

        elif scenario_type == StressScenarioType.REGULATORY_SHOCK:
            # Selective negative shock to specific assets
            affected = np.random.choice(len(base_returns), size=max(1, len(base_returns) // 3), replace=False)
            stressed[affected] -= 0.15 * severity

        return stressed

    def _evaluate_stress(
        self,
        scenario_type: StressScenarioType,
        stressed_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> StressTestResult:
        """Evaluate the impact of a stress scenario.

        Args:
            scenario_type: Type of stress scenario.
            stressed_returns: Returns under stress.
            covariance: Covariance matrix.

        Returns:
            StressTestResult with impact assessment.
        """
        # Equal weight portfolio for estimation
        n = len(stressed_returns)
        weights = np.ones(n) / n
        portfolio_loss = float(-np.dot(weights, stressed_returns))

        # Find most affected assets
        sorted_idx = np.argsort(stressed_returns)
        affected = [f"asset_{i}" for i in sorted_idx[:max(1, n // 3)]]

        # Estimate probability based on severity
        probability = float(np.clip(0.1 / (1 + portfolio_loss), 0.001, 0.5))

        # Estimate recovery time
        recovery_hours = float(24 + portfolio_loss * 100)

        # Generate recommendations
        recommendations = self._generate_recommendations(scenario_type, portfolio_loss)

        return StressTestResult(
            scenario_type=scenario_type,
            scenario_name=f"{scenario_type.value}_stress_test",
            portfolio_loss=portfolio_loss,
            affected_assets=affected,
            probability=probability,
            recovery_time_hours=recovery_hours,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self, scenario_type: StressScenarioType, loss: float
    ) -> List[str]:
        """Generate risk mitigation recommendations.

        Args:
            scenario_type: Type of stress scenario.
            loss: Estimated portfolio loss.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if loss > 0.10:
            recs.append("Consider reducing overall portfolio exposure by 20-30%")

        if scenario_type == StressScenarioType.MARKET_CRASH:
            recs.append("Increase cash allocation")
            recs.append("Add put options or inverse positions for tail protection")
        elif scenario_type == StressScenarioType.LIQUIDITY_CRISIS:
            recs.append("Reduce position sizes to improve liquidity")
            recs.append("Set wider stop-losses to avoid forced liquidation")
        elif scenario_type == StressScenarioType.VOLATILITY_SPIKE:
            recs.append("Reduce leverage and position sizes")
            recs.append("Consider volatility selling strategies if over-hedged")
        elif scenario_type == StressScenarioType.DEPEG_EVENT:
            recs.append("Exit stablecoin positions in affected assets")
            recs.append("Diversify across multiple stablecoins")
        elif scenario_type == StressScenarioType.CORRELATION_BREAKDOWN:
            recs.append("Diversification benefits reduced; consider alternative hedges")

        return recs


# ---------------------------------------------------------------------------
# Real-Time Risk Predictor
# ---------------------------------------------------------------------------

class RealTimeRiskPredictor:
    """Real-time risk prediction and monitoring system.

    Continuously updates risk estimates based on incoming market data
    and provides alerts when risk thresholds are breached.

    Attributes:
        risk_thresholds: Risk thresholds for alert generation.
        alert_history: History of generated alerts.
    """

    def __init__(
        self,
        var_limit_95: float = 0.05,
        max_drawdown_limit: float = 0.15,
        volatility_limit: float = 0.40,
        lookback: int = 100,
    ) -> None:
        """Initialise the real-time risk predictor.

        Args:
            var_limit_95: Maximum acceptable 95% VaR.
            max_drawdown_limit: Maximum acceptable drawdown.
            volatility_limit: Maximum acceptable annualized volatility.
            lookback: Lookback window for rolling calculations.
        """
        self.risk_thresholds = {
            "var_95": var_limit_95,
            "max_drawdown": max_drawdown_limit,
            "volatility": volatility_limit,
        }
        self.lookback = lookback
        self.alert_history: List[Dict[str, Any]] = []

        # Exponentially weighted tracking
        self._ewma_vol: Optional[float] = None
        self._ewma_var: Optional[float] = None
        self._span = 20  # EWMA span

        # Rolling return history for proper VaR/CVaR computation
        self._return_history: List[float] = []
        self._max_history = lookback

    def update(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        asset_returns: np.ndarray,
    ) -> RiskMetrics:
        """Update risk estimates with new return observation.

        Args:
            portfolio_return: Latest portfolio return.
            weights: Current portfolio weights.
            asset_returns: Latest asset-level returns.

        Returns:
            Updated RiskMetrics.
        """
        # Store return in rolling history
        self._return_history.append(portfolio_return)
        if len(self._return_history) > self._max_history:
            self._return_history = self._return_history[-self._max_history:]

        # EWMA volatility update
        if self._ewma_vol is None:
            self._ewma_vol = abs(portfolio_return)
        else:
            alpha = 2.0 / (self._span + 1)
            self._ewma_vol = np.sqrt(
                alpha * portfolio_return ** 2 + (1 - alpha) * self._ewma_vol ** 2
            )

        # Annualize
        annualized_vol = self._ewma_vol * np.sqrt(252)

        # Compute VaR and CVaR properly from rolling history
        returns_arr = np.array(self._return_history)

        if len(returns_arr) >= 30:
            sorted_returns = np.sort(returns_arr)
            n = len(sorted_returns)

            # VaR_95: 5th percentile of historical returns
            var_95 = float(sorted_returns[int(0.05 * n)])
            # VaR_99: 1st percentile of historical returns
            var_99 = float(sorted_returns[int(0.01 * n)])

            # CVaR_95: mean of tail below VaR_95
            tail_95 = sorted_returns[sorted_returns <= var_95]
            cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

            # CVaR_99: mean of tail below VaR_99
            tail_99 = sorted_returns[sorted_returns <= var_99]
            cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

            # Use absolute values for risk metrics (VaR is typically positive)
            var_95 = abs(var_95)
            var_99 = abs(var_99)
            cvar_95 = abs(cvar_95)
            cvar_99 = abs(cvar_99)
        else:
            # Not enough history for historical VaR: use EWMA-based parametric estimate
            var_95 = 1.645 * self._ewma_vol if self._ewma_vol else 0.0
            var_99 = 2.326 * self._ewma_vol if self._ewma_vol else 0.0
            cvar_95 = var_95 * 1.2  # Conservative estimate
            cvar_99 = var_99 * 1.3

        # Risk level classification
        risk_level = self._classify_risk(annualized_vol, var_95)

        metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=annualized_vol,
            risk_level=risk_level,
        )

        # Check for threshold breaches
        self._check_thresholds(metrics)

        return metrics

    def _classify_risk(self, volatility: float, var: float) -> RiskLevel:
        """Classify the current risk level.

        Args:
            volatility: Annualized volatility.
            var: Current VaR estimate.

        Returns:
            RiskLevel classification.
        """
        if volatility > self.risk_thresholds["volatility"] * 2 or var > self.risk_thresholds["var_95"] * 2:
            return RiskLevel.CRITICAL
        elif volatility > self.risk_thresholds["volatility"] or var > self.risk_thresholds["var_95"]:
            return RiskLevel.HIGH
        elif volatility > self.risk_thresholds["volatility"] * 0.5:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _check_thresholds(self, metrics: RiskMetrics) -> None:
        """Check if any risk thresholds are breached and generate alerts.

        Args:
            metrics: Current risk metrics.
        """
        alerts: List[str] = []

        if metrics.var_95 > self.risk_thresholds["var_95"]:
            alerts.append(f"VaR 95% breach: {metrics.var_95:.4f} > {self.risk_thresholds['var_95']:.4f}")

        if metrics.volatility > self.risk_thresholds["volatility"]:
            alerts.append(f"Volatility breach: {metrics.volatility:.4f} > {self.risk_thresholds['volatility']:.4f}")

        if alerts:
            alert_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_level": metrics.risk_level.value,
                "alerts": alerts,
                "metrics": metrics.to_dict(),
            }
            self.alert_history.append(alert_record)
            logger.warning("Risk alert: %s", "; ".join(alerts))


# ---------------------------------------------------------------------------
# Adaptive Risk Limiter
# ---------------------------------------------------------------------------

class AdaptiveRiskLimiter:
    """Dynamically adjusts risk limits based on market conditions.

    Adapts position limits, leverage, and VaR thresholds based on:
    - Current market regime
    - Recent risk metric trends
    - Portfolio performance
    - Market-wide stress indicators

    Attributes:
        base_limits: Base risk limits.
        current_limits: Current (adjusted) risk limits.
    """

    def __init__(
        self,
        base_var_limit: float = 0.05,
        base_leverage_limit: float = 2.0,
        base_position_limit: float = 0.20,
    ) -> None:
        """Initialise the adaptive risk limiter.

        Args:
            base_var_limit: Base VaR limit (95%).
            base_leverage_limit: Base maximum leverage.
            base_position_limit: Base maximum position size (as fraction).
        """
        self.base_limits = {
            "var_95": base_var_limit,
            "leverage": base_leverage_limit,
            "position_size": base_position_limit,
        }
        self.current_limits = self.base_limits.copy()
        self._adjustment_history: List[Dict[str, Any]] = []

    def adjust_limits(
        self,
        risk_metrics: RiskMetrics,
        regime: Optional[str] = None,
        market_stress: float = 0.0,
    ) -> Dict[str, float]:
        """Adapt risk limits based on current conditions.

        Args:
            risk_metrics: Current risk metrics.
            regime: Current market regime.
            market_stress: Market stress indicator [0, 1].

        Returns:
            Adjusted risk limits.
        """
        # Regime-based multiplier
        regime_multiplier = 1.0
        if regime in ("crisis", "volatile"):
            regime_multiplier = 0.5
        elif regime in ("ranging",):
            regime_multiplier = 0.8
        elif regime in ("trending_up", "recovery"):
            regime_multiplier = 1.0

        # Stress-based reduction
        stress_multiplier = 1.0 - 0.5 * market_stress

        # Performance-based adjustment
        perf_multiplier = 1.0
        if risk_metrics.sharpe < -0.5:
            perf_multiplier = 0.6
        elif risk_metrics.sharpe < 0:
            perf_multiplier = 0.8

        # Combined multiplier
        combined = regime_multiplier * stress_multiplier * perf_multiplier

        # Apply adjustments
        self.current_limits = {
            "var_95": self.base_limits["var_95"] * combined,
            "leverage": self.base_limits["leverage"] * combined,
            "position_size": self.base_limits["position_size"] * combined,
        }

        # Record adjustment
        self._adjustment_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "market_stress": market_stress,
            "combined_multiplier": combined,
            "limits": self.current_limits.copy(),
        })

        logger.info(
            "Risk limits adjusted: multiplier=%.2f, limits=%s",
            combined,
            self.current_limits,
        )
        return self.current_limits


# ---------------------------------------------------------------------------
# Tail Risk Hedger
# ---------------------------------------------------------------------------

class TailRiskHedger:
    """Generates tail risk hedging recommendations.

    Analyzes the portfolio's tail risk exposure and recommends
    hedging strategies including options, inverse positions,
    and correlation-based hedges.

    Attributes:
        tail_threshold: Quantile threshold for tail definition.
    """

    def __init__(
        self,
        tail_threshold: float = 0.05,
        hedge_cost_tolerance: float = 0.02,
    ) -> None:
        """Initialise the tail risk hedger.

        Args:
            tail_threshold: Quantile for tail event definition.
            hedge_cost_tolerance: Maximum acceptable hedge cost as % of portfolio.
        """
        self.tail_threshold = tail_threshold
        self.hedge_cost_tolerance = hedge_cost_tolerance

    def analyze_tail_risk(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze portfolio tail risk and recommend hedges.

        Args:
            returns: Historical returns matrix (n_periods, n_assets).
            weights: Current portfolio weights.
            asset_names: Optional asset names.

        Returns:
            Tail risk analysis with hedging recommendations.
        """
        portfolio_returns = returns @ weights

        # Compute tail statistics
        var = float(np.percentile(portfolio_returns, self.tail_threshold * 100))
        cvar = float(np.mean(portfolio_returns[portfolio_returns <= var]))

        # Tail contribution per asset
        tail_mask = portfolio_returns <= var
        tail_returns = returns[tail_mask]
        asset_tail_contributions = np.abs(tail_returns.mean(axis=0) * weights)

        # Find tail risk drivers
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(len(weights))]

        tail_drivers = sorted(
            zip(asset_names, asset_tail_contributions),
            key=lambda x: x[1],
            reverse=True,
        )

        # Generate hedging recommendations
        recommendations = self._recommend_hedges(tail_drivers, cvar)

        return {
            "var": -var,
            "cvar": -cvar,
            "tail_probability": float(tail_mask.mean()),
            "tail_drivers": [(name, float(contrib)) for name, contrib in tail_drivers[:5]],
            "recommendations": recommendations,
            "estimated_hedge_cost": float(min(abs(cvar) * 0.3, self.hedge_cost_tolerance)),
        }

    def _recommend_hedges(
        self,
        tail_drivers: List[Tuple[str, float]],
        cvar: float,
    ) -> List[Dict[str, Any]]:
        """Generate specific hedging recommendations.

        Args:
            tail_drivers: Assets contributing most to tail risk.
            cvar: Portfolio CVaR.

        Returns:
            List of hedge recommendation dictionaries.
        """
        recommendations: List[Dict[str, Any]] = []

        for asset_name, contribution in tail_drivers[:3]:
            if contribution > 0.01:  # Significant contributor
                recommendations.append({
                    "type": "inverse_position",
                    "asset": asset_name,
                    "suggested_size": float(min(contribution * 2, 0.15)),
                    "rationale": f"Reduces tail risk contribution from {asset_name}",
                    "estimated_cost": float(contribution * 0.1),
                })

        # Portfolio-level hedge
        if abs(cvar) > 0.05:
            recommendations.append({
                "type": "put_options",
                "asset": "portfolio",
                "suggested_size": float(min(abs(cvar) * 0.5, 0.10)),
                "rationale": "Protects against extreme downside scenarios",
                "estimated_cost": float(abs(cvar) * 0.05),
            })

        return recommendations


# ---------------------------------------------------------------------------
# Risk Explainer
# ---------------------------------------------------------------------------

class RiskExplainer:
    """Risk explainability and attribution engine.

    Decomposes portfolio risk into contributing factors and
    generates human-readable explanations.

    Attributes:
        factor_names: Names of risk factors.
    """

    def __init__(self, factor_names: Optional[List[str]] = None) -> None:
        """Initialise the risk explainer.

        Args:
            factor_names: Names of risk factors for attribution.
        """
        self.factor_names = factor_names or [
            "market_beta", "size", "momentum", "volatility",
            "liquidity", "correlation", "tail_risk",
        ]

    def explain_risk(
        self,
        risk_metrics: RiskMetrics,
        weights: np.ndarray,
        asset_names: List[str],
        factor_exposures: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive risk explanation.

        Args:
            risk_metrics: Current risk metrics.
            weights: Portfolio weights.
            asset_names: Asset identifiers.
            factor_exposures: Factor exposure matrix (n_assets, n_factors).

        Returns:
            Risk explanation dictionary.
        """
        # Concentration analysis
        hhi = float(np.sum(weights ** 2))
        max_weight = float(np.max(weights))
        max_asset = asset_names[np.argmax(weights)] if len(asset_names) > 0 else "unknown"

        # Factor attribution
        factor_attribution: Dict[str, float] = {}
        if factor_exposures is not None:
            portfolio_exposures = weights @ factor_exposures
            for i, name in enumerate(self.factor_names[:factor_exposures.shape[1]]):
                factor_attribution[name] = float(portfolio_exposures[i])

        # Risk level narrative
        narrative = self._generate_narrative(
            risk_metrics, max_asset, max_weight, hhi, factor_attribution
        )

        return {
            "risk_level": risk_metrics.risk_level.value,
            "var_95": risk_metrics.var_95,
            "cvar_95": risk_metrics.cvar_95,
            "concentration": {
                "hhi": hhi,
                "max_weight": max_weight,
                "max_asset": max_asset,
                "effective_n": 1.0 / (hhi + 1e-12),
            },
            "factor_attribution": factor_attribution,
            "narrative": narrative,
        }

    def _generate_narrative(
        self,
        metrics: RiskMetrics,
        max_asset: str,
        max_weight: float,
        hhi: float,
        factors: Dict[str, float],
    ) -> str:
        """Generate a human-readable risk narrative.

        Args:
            metrics: Risk metrics.
            max_asset: Largest holding asset.
            max_weight: Largest holding weight.
            hhi: Herfindahl-Hirschman Index.
            factors: Factor attributions.

        Returns:
            Narrative text.
        """
        parts: List[str] = []

        # Risk level
        parts.append(f"Current portfolio risk level is {metrics.risk_level.value}.")

        # VaR
        parts.append(
            f"The 95% Value at Risk is {metrics.var_95:.2%}, meaning "
            f"the portfolio is expected to lose no more than {metrics.var_95:.2%} "
            f"on 95% of trading days."
        )

        # Concentration
        if hhi > 0.3:
            parts.append(
                f"Portfolio is highly concentrated (HHI={hhi:.2f}), "
                f"with {max_asset} comprising {max_weight:.1%} of holdings."
            )
        elif hhi > 0.15:
            parts.append(
                f"Portfolio has moderate concentration (HHI={hhi:.2f})."
            )
        else:
            parts.append("Portfolio is well-diversified across positions.")

        # Dominant factors
        if factors:
            dominant = max(factors, key=factors.get)  # type: ignore[arg-type]
            parts.append(
                f"The dominant risk factor is {dominant} "
                f"(exposure={factors[dominant]:.3f})."
            )

        return " ".join(parts)


# ---------------------------------------------------------------------------
# AI Risk Manager (Main Interface)
# ---------------------------------------------------------------------------

class AIRiskManager:
    """Unified AI-driven risk management system.

    Coordinates all risk management components:
    - Neural VaR estimation
    - Stress scenario generation
    - Real-time risk monitoring
    - Adaptive risk limits
    - Tail risk hedging
    - Risk explainability

    Attributes:
        manager_id: Unique identifier.
        neural_var: Neural VaR estimator.
        stress_generator: Stress scenario generator.
        real_time_predictor: Real-time risk predictor.
        adaptive_limiter: Adaptive risk limiter.
        tail_hedger: Tail risk hedger.
        risk_explainer: Risk explainability engine.
    """

    def __init__(
        self,
        manager_id: str = "default",
        n_assets: int = 10,
        n_features: int = 20,
        var_limit_95: float = 0.05,
        max_drawdown_limit: float = 0.15,
        device: str = "auto",
        redis_client: Any = None,
        postgres_client: Any = None,
    ) -> None:
        """Initialise the AI risk manager.

        Args:
            manager_id: Unique identifier.
            n_assets: Number of portfolio assets.
            n_features: Number of features for VaR prediction.
            var_limit_95: 95% VaR limit.
            max_drawdown_limit: Maximum drawdown limit.
            device: Compute device.
            redis_client: Optional Redis client.
            postgres_client: Optional Postgres client.
        """
        self.manager_id = manager_id
        self._redis = redis_client
        self._postgres = postgres_client

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        self.neural_var = NeuralVaR(
            n_features=n_features, device=self._device
        )
        self.stress_generator = StressScenarioGenerator(
            n_assets=n_assets, device=self._device
        )
        self.real_time_predictor = RealTimeRiskPredictor(
            var_limit_95=var_limit_95,
            max_drawdown_limit=max_drawdown_limit,
        )
        self.adaptive_limiter = AdaptiveRiskLimiter(
            base_var_limit=var_limit_95,
        )
        self.tail_hedger = TailRiskHedger()
        self.risk_explainer = RiskExplainer()

        self._risk_history: List[RiskMetrics] = []

        logger.info(
            "AIRiskManager initialised [id=%s, device=%s]",
            manager_id,
            self._device,
        )

    def compute_risk_metrics(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        asset_returns: np.ndarray,
        features: Optional[np.ndarray] = None,
        historical_returns: Optional[np.ndarray] = None,
    ) -> RiskMetrics:
        """Compute comprehensive risk metrics.

        Args:
            portfolio_return: Latest portfolio return.
            weights: Current portfolio weights.
            asset_returns: Latest asset-level returns.
            features: Optional features for neural VaR.
            historical_returns: Optional historical returns for VaR.

        Returns:
            Updated RiskMetrics.
        """
        # Real-time risk update
        metrics = self.real_time_predictor.update(
            portfolio_return, weights, asset_returns
        )

        # Enhance with neural VaR if features available
        if features is not None:
            var_estimates = self.neural_var.predict(features)
            metrics.var_95 = var_estimates.get(0.05, metrics.var_95)
            metrics.var_99 = var_estimates.get(0.01, metrics.var_99)

        # CVaR from historical returns
        if historical_returns is not None:
            portfolio_hist = historical_returns @ weights
            metrics.cvar_95 = self.neural_var.compute_cvar(portfolio_hist, 0.05)
            metrics.cvar_99 = self.neural_var.compute_cvar(portfolio_hist, 0.01)

        self._risk_history.append(metrics)

        logger.debug(
            "Risk metrics: VaR95=%.4f, CVaR95=%.4f, vol=%.2f%%, level=%s",
            metrics.var_95,
            metrics.cvar_95,
            metrics.volatility * 100,
            metrics.risk_level.value,
        )
        return metrics

    def run_stress_tests(
        self,
        base_returns: np.ndarray,
        base_covariance: np.ndarray,
        severity: float = 1.0,
    ) -> List[StressTestResult]:
        """Run comprehensive stress tests.

        Args:
            base_returns: Base expected returns.
            base_covariance: Base covariance matrix.
            severity: Stress severity multiplier.

        Returns:
            List of stress test results.
        """
        results = self.stress_generator.generate_scenarios(
            base_returns, base_covariance, severity=severity
        )

        logger.info(
            "Stress tests completed: %d scenarios, max loss=%.2f%%",
            len(results),
            max(r.portfolio_loss for r in results) * 100 if results else 0,
        )
        return results

    def get_risk_limits(
        self,
        risk_metrics: Optional[RiskMetrics] = None,
        regime: Optional[str] = None,
        market_stress: float = 0.0,
    ) -> Dict[str, float]:
        """Get current adaptive risk limits.

        Args:
            risk_metrics: Current risk metrics.
            regime: Market regime.
            market_stress: Market stress indicator.

        Returns:
            Dictionary of risk limits.
        """
        if risk_metrics is None and self._risk_history:
            risk_metrics = self._risk_history[-1]
        elif risk_metrics is None:
            risk_metrics = RiskMetrics()

        return self.adaptive_limiter.adjust_limits(
            risk_metrics, regime, market_stress
        )

    def analyze_tail_risk(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze tail risk and generate hedging recommendations.

        Args:
            returns: Historical returns matrix.
            weights: Portfolio weights.
            asset_names: Asset identifiers.

        Returns:
            Tail risk analysis.
        """
        return self.tail_hedger.analyze_tail_risk(returns, weights, asset_names)

    def explain_risk(
        self,
        weights: np.ndarray,
        asset_names: List[str],
        factor_exposures: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive risk explanation.

        Args:
            weights: Portfolio weights.
            asset_names: Asset identifiers.
            factor_exposures: Factor exposure matrix.

        Returns:
            Risk explanation.
        """
        metrics = self._risk_history[-1] if self._risk_history else RiskMetrics()
        return self.risk_explainer.explain_risk(
            metrics, weights, asset_names, factor_exposures
        )

    def risk_summary(self) -> Dict[str, Any]:
        """Get a summary of risk management activity.

        Returns:
            Summary dictionary.
        """
        current_metrics = self._risk_history[-1] if self._risk_history else None

        return {
            "manager_id": self.manager_id,
            "total_risk_updates": len(self._risk_history),
            "current_risk_level": current_metrics.risk_level.value if current_metrics else None,
            "current_var_95": current_metrics.var_95 if current_metrics else None,
            "alert_count": len(self.real_time_predictor.alert_history),
            "current_limits": self.adaptive_limiter.current_limits,
            "device": self._device,
        }

    async def save_to_postgres(self) -> int:
        """Persist risk data to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for metrics in self._risk_history:
                    await self._postgres.execute(
                        """
                        INSERT INTO risk_metrics (manager_id, timestamp, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                        """,
                        self.manager_id,
                        metrics.timestamp.isoformat(),
                        json.dumps(metrics.to_dict()),
                    )
                    count += 1
            logger.info("Saved %d risk metrics to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save risk metrics: %s", exc)
        return count
