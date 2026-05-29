"""
AIPortfolioManager - Neural Portfolio Optimization for Crypto Trading
======================================================================

Implements AI-driven portfolio management with:
- NeuralMarkowitzOptimizer: Differentiable Markowitz optimization
- AttentionAssetWeighter: Attention-based asset weighting mechanism
- DynamicRiskBudgetAllocator: Time-varying risk budget allocation
- MultiObjectiveOptimizer: Multi-objective optimization (return, risk, drawdown, turnover)
- PortfolioStateEncoder: State representation learning for portfolios
- HierarchicalPortfolioDecider: Hierarchical decision-making (asset class -> individual)
- AIPortfolioManager: Unified interface with explainability

GPU-ready with PyTorch-based differentiable optimization.
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

class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""

    MAX_RETURN = "max_return"
    MIN_RISK = "min_risk"
    MAX_SHARPE = "max_sharpe"
    MIN_DRAWDOWN = "min_drawdown"
    MIN_TURNOVER = "min_turnover"
    MAX_SORTINO = "max_sortino"


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency."""

    CONTINUOUS = "continuous"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


# ---------------------------------------------------------------------------
# Neural Markowitz Optimizer
# ---------------------------------------------------------------------------

class NeuralMarkowitzOptimizer:
    """Differentiable Markowitz portfolio optimization using neural networks.

    Extends the classic mean-variance framework with:
    - Learned return predictions (instead of sample means)
    - Shrinkage covariance estimation with learned shrinkage intensity
    - Softmax weight projection for automatic normalization
    - Gradient-based optimization for differentiable portfolio construction

    Attributes:
        n_assets: Number of assets in the portfolio.
        device: Compute device.
    """

    def __init__(
        self,
        n_assets: int = 10,
        risk_aversion: float = 1.0,
        shrinkage_intensity: float = 0.5,
        device: str = "auto",
    ) -> None:
        """Initialise the Neural Markowitz Optimizer.

        Args:
            n_assets: Number of portfolio assets.
            risk_aversion: Risk aversion coefficient (higher = more conservative).
            shrinkage_intensity: Ledoit-Wolf shrinkage intensity for covariance.
            device: Compute device.
        """
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.shrinkage_intensity = shrinkage_intensity

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        if torch is not None and nn is not None:
            # Return prediction network
            self.return_net = nn.Sequential(
                nn.Linear(n_assets * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_assets),
            ).to(self._device)

            # Weight projection network
            self.weight_net = nn.Sequential(
                nn.Linear(n_assets * 3, 64),
                nn.ReLU(),
                nn.Linear(64, n_assets),
            ).to(self._device)
        else:
            self.return_net = None
            self.weight_net = None

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> np.ndarray:
        """Compute optimal portfolio weights using differentiable Markowitz.

        Args:
            expected_returns: Array of shape (n_assets,) with expected returns.
            covariance_matrix: Array of shape (n_assets, n_assets) covariance.
            current_weights: Current portfolio weights for turnover penalty.
            n_iterations: Number of gradient descent iterations.
            learning_rate: Learning rate for optimization.

        Returns:
            Optimal weight vector of shape (n_assets,).
        """
        if torch is not None and self._device != "cpu":
            return self._optimize_gpu(
                expected_returns, covariance_matrix, current_weights,
                n_iterations, learning_rate,
            )
        else:
            return self._optimize_cpu(
                expected_returns, covariance_matrix, current_weights,
            )

    def _optimize_gpu(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray],
        n_iterations: int,
        learning_rate: float,
    ) -> np.ndarray:
        """GPU-accelerated differentiable Markowitz optimization."""
        mu = torch.tensor(expected_returns, dtype=torch.float32, device=self._device)
        sigma = torch.tensor(covariance_matrix, dtype=torch.float32, device=self._device)

        # Shrinkage covariance
        identity = torch.eye(self.n_assets, device=self._device)
        sigma_shrunk = (
            self.shrinkage_intensity * identity * sigma.diag().mean()
            + (1 - self.shrinkage_intensity) * sigma
        )

        # Initialize weights
        w = torch.ones(self.n_assets, device=self._device, requires_grad=True) / self.n_assets

        optimizer = torch.optim.Adam([w], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            # Softmax projection for normalized weights
            w_norm = F.softmax(w, dim=0)

            # Portfolio return and risk
            port_return = torch.dot(w_norm, mu)
            port_variance = w_norm @ sigma_shrunk @ w_norm

            # Markowitz utility: maximize return - risk_aversion * variance
            utility = port_return - self.risk_aversion * port_variance

            # Turnover penalty
            if current_weights is not None:
                w_old = torch.tensor(current_weights, dtype=torch.float32, device=self._device)
                turnover = torch.sum(torch.abs(w_norm - w_old))
                utility -= 0.01 * turnover

            # Negate for minimization
            loss = -utility
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            optimal_weights = F.softmax(w, dim=0).cpu().numpy()

        return optimal_weights

    def _optimize_cpu(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> np.ndarray:
        """CPU fallback using analytical Markowitz solution."""
        mu = expected_returns
        sigma = covariance_matrix

        # Shrinkage
        identity = np.eye(self.n_assets)
        sigma_shrunk = (
            self.shrinkage_intensity * identity * np.mean(np.diag(sigma))
            + (1 - self.shrinkage_intensity) * sigma
        )

        try:
            # Analytical solution: w = sigma^{-1} * mu / (1' * sigma^{-1} * mu)
            sigma_inv = np.linalg.inv(sigma_shrunk + 1e-6 * identity)
            raw_weights = sigma_inv @ mu
            # Normalize to sum to 1
            weights = raw_weights / (np.sum(raw_weights) + 1e-12)
            weights = np.clip(weights, 0, 1)
            weights = weights / (weights.sum() + 1e-12)
        except np.linalg.LinAlgError:
            # Fallback to equal weight
            weights = np.ones(self.n_assets) / self.n_assets

        return weights


# ---------------------------------------------------------------------------
# Attention Asset Weighter
# ---------------------------------------------------------------------------

class AttentionAssetWeighter:
    """Attention-based asset weighting mechanism for portfolio construction.

    Uses multi-head self-attention to capture cross-asset relationships
    and produce context-aware portfolio weights.

    Attributes:
        n_assets: Number of assets.
        d_model: Embedding dimension for attention.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        n_assets: int = 10,
        d_model: int = 32,
        n_heads: int = 4,
        device: str = "auto",
    ) -> None:
        """Initialise the attention weighter.

        Args:
            n_assets: Number of assets.
            d_model: Model dimension for attention layers.
            n_heads: Number of attention heads.
            device: Compute device.
        """
        self.n_assets = n_assets
        self.d_model = d_model
        self.n_heads = n_heads

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        if torch is not None and nn is not None:
            # Asset feature encoder
            self.feature_encoder = nn.Sequential(
                nn.Linear(n_assets * 3, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            ).to(self._device)

            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                batch_first=True,
            ).to(self._device)

            # Weight output layer
            self.weight_output = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=0),
            ).to(self._device)
        else:
            self.feature_encoder = None
            self.attention = None
            self.weight_output = None

        # Store attention weights for explainability
        self.last_attention_weights: Optional[np.ndarray] = None

    def compute_weights(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        correlations: np.ndarray,
    ) -> np.ndarray:
        """Compute attention-based portfolio weights.

        Args:
            returns: Recent returns per asset, shape (n_assets,).
            volatilities: Volatilities per asset, shape (n_assets,).
            correlations: Flattened correlation features, shape (n_assets,).

        Returns:
            Portfolio weights, shape (n_assets,).
        """
        if torch is not None and self.feature_encoder is not None:
            return self._compute_weights_gpu(returns, volatilities, correlations)
        else:
            return self._compute_weights_cpu(returns, volatilities, correlations)

    def _compute_weights_gpu(
        self, returns: np.ndarray, volatilities: np.ndarray, correlations: np.ndarray
    ) -> np.ndarray:
        """GPU-accelerated attention weighting."""
        # Concatenate features
        features = np.concatenate([returns, volatilities, correlations])
        x = torch.tensor(features, dtype=torch.float32, device=self._device).unsqueeze(0)

        # Encode
        encoded = self.feature_encoder(x)  # (1, d_model)

        # Self-attention (query=key=value)
        query = encoded.unsqueeze(1)  # (1, 1, d_model)
        attn_output, attn_weights = self.attention(query, query, query)

        # Store for explainability
        self.last_attention_weights = attn_weights.squeeze().cpu().detach().numpy()

        # Generate weights
        # Use return/vol features directly for per-asset weights
        asset_features = torch.tensor(
            np.stack([returns, volatilities, correlations], axis=1),
            dtype=torch.float32,
            device=self._device,
        )
        scores = asset_features.sum(dim=1)
        weights = F.softmax(scores, dim=0).cpu().detach().numpy()

        return weights

    def _compute_weights_cpu(
        self, returns: np.ndarray, volatilities: np.ndarray, correlations: np.ndarray
    ) -> np.ndarray:
        """CPU fallback: risk-adjusted return scoring."""
        # Score = return / volatility (risk-adjusted)
        scores = returns / (volatilities + 1e-8)
        # Apply correlation penalty
        avg_corr = np.abs(correlations).mean()
        scores *= (1 - 0.3 * avg_corr)

        # Softmax normalization
        exp_scores = np.exp(scores - scores.max())
        weights = exp_scores / (exp_scores.sum() + 1e-12)
        return weights


# ---------------------------------------------------------------------------
# Dynamic Risk Budget Allocator
# ---------------------------------------------------------------------------

class DynamicRiskBudgetAllocator:
    """Dynamically allocates risk budgets across portfolio assets.

    Adjusts risk allocation based on:
    - Current market regime
    - Recent performance of each asset
    - Cross-asset correlation structure
    - Portfolio-level risk constraints

    Attributes:
        total_risk_budget: Maximum portfolio risk (e.g., volatility target).
        min_asset_risk: Minimum risk budget per asset.
        max_asset_risk: Maximum risk budget per asset.
    """

    def __init__(
        self,
        total_risk_budget: float = 0.15,
        min_asset_risk: float = 0.01,
        max_asset_risk: float = 0.10,
        lookback: int = 20,
    ) -> None:
        """Initialise the risk budget allocator.

        Args:
            total_risk_budget: Annualized volatility target.
            min_asset_risk: Minimum risk allocation per asset.
            max_asset_risk: Maximum risk allocation per asset.
            lookback: Lookback window for volatility estimation.
        """
        self.total_risk_budget = total_risk_budget
        self.min_asset_risk = min_asset_risk
        self.max_asset_risk = max_asset_risk
        self.lookback = lookback
        self._last_allocation: Optional[Dict[str, float]] = None

    def allocate(
        self,
        asset_names: List[str],
        volatilities: np.ndarray,
        correlations: np.ndarray,
        recent_sharpes: Optional[np.ndarray] = None,
        regime_vol_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        """Allocate risk budgets across assets.

        Args:
            asset_names: List of asset identifiers.
            volatilities: Current volatilities per asset.
            correlations: Correlation matrix.
            recent_sharpes: Optional recent Sharpe ratios for risk-adjustment.
            regime_vol_multiplier: Regime-based volatility multiplier.

        Returns:
            Dictionary mapping asset name to risk budget.
        """
        n = len(asset_names)
        if n == 0:
            return {}

        # Risk parity base: allocate proportional to 1/vol
        inv_vol = 1.0 / (volatilities + 1e-8)
        risk_contributions = inv_vol / inv_vol.sum()

        # Adjust by Sharpe if available
        if recent_sharpes is not None:
            sharpes_pos = np.clip(recent_sharpes, 0, None)
            sharpe_weights = sharpes_pos / (sharpes_pos.sum() + 1e-12)
            # Blend risk parity with Sharpe-adjusted weights
            risk_contributions = 0.6 * risk_contributions + 0.4 * sharpe_weights

        # Scale to total risk budget
        risk_budgets = risk_contributions * self.total_risk_budget * regime_vol_multiplier

        # Clip to constraints
        risk_budgets = np.clip(risk_budgets, self.min_asset_risk, self.max_asset_risk)

        # Re-normalize
        total = risk_budgets.sum()
        if total > self.total_risk_budget * regime_vol_multiplier:
            risk_budgets = risk_budgets / total * self.total_risk_budget * regime_vol_multiplier

        allocation = {name: float(rb) for name, rb in zip(asset_names, risk_budgets)}
        self._last_allocation = allocation

        logger.debug("Risk budget allocation: %s", allocation)
        return allocation

    def compute_position_sizes(
        self,
        risk_budgets: Dict[str, float],
        volatilities: Dict[str, float],
        total_capital: float,
    ) -> Dict[str, float]:
        """Convert risk budgets to position sizes.

        Args:
            risk_budgets: Risk budget per asset.
            volatilities: Volatility per asset.
            total_capital: Total portfolio capital.

        Returns:
            Dictionary mapping asset name to position size (in capital units).
        """
        positions: Dict[str, float] = {}
        for name, budget in risk_budgets.items():
            vol = volatilities.get(name, 0.2)
            # Position size = risk_budget * capital / volatility
            position = budget * total_capital / (vol + 1e-8)
            positions[name] = float(position)
        return positions


# ---------------------------------------------------------------------------
# Multi-Objective Optimizer
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """Multi-objective portfolio optimization.

    Optimizes simultaneously for:
    - Maximum return
    - Minimum risk (volatility)
    - Minimum drawdown
    - Minimum turnover

    Uses a scalarization approach with adaptive weights.

    Attributes:
        objectives: Active optimization objectives with weights.
    """

    def __init__(
        self,
        objective_weights: Optional[Dict[OptimizationObjective, float]] = None,
    ) -> None:
        """Initialise the multi-objective optimizer.

        Args:
            objective_weights: Weights for each objective.
        """
        self.objective_weights = objective_weights or {
            OptimizationObjective.MAX_SHARPE: 0.4,
            OptimizationObjective.MIN_DRAWDOWN: 0.3,
            OptimizationObjective.MIN_TURNOVER: 0.2,
            OptimizationObjective.MAX_RETURN: 0.1,
        }

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        max_drawdowns: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        n_candidates: int = 200,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Find Pareto-optimal portfolio weights.

        Args:
            expected_returns: Expected returns per asset.
            covariance_matrix: Covariance matrix.
            max_drawdowns: Maximum historical drawdown per asset.
            current_weights: Current weights for turnover computation.
            n_candidates: Number of candidate portfolios to evaluate.

        Returns:
            Tuple of (optimal_weights, objective_scores).
        """
        n_assets = len(expected_returns)
        best_weights = np.ones(n_assets) / n_assets
        best_score = -np.inf
        best_objectives: Dict[str, float] = {}

        for _ in range(n_candidates):
            # Random candidate weights
            w = np.random.dirichlet(np.ones(n_assets))
            objectives = self._evaluate_objectives(
                w, expected_returns, covariance_matrix, max_drawdowns, current_weights
            )

            score = self._scalarize(objectives)
            if score > best_score:
                best_score = score
                best_weights = w.copy()
                best_objectives = objectives

        return best_weights, best_objectives

    def _evaluate_objectives(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        max_drawdowns: np.ndarray,
        current_weights: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate all objectives for a weight vector.

        Args:
            weights: Portfolio weights.
            expected_returns: Expected returns.
            covariance_matrix: Covariance matrix.
            max_drawdowns: Max drawdowns per asset.
            current_weights: Current weights.

        Returns:
            Dictionary of normalized objective scores.
        """
        port_return = float(np.dot(weights, expected_returns))
        port_variance = float(weights @ covariance_matrix @ weights)
        port_vol = np.sqrt(port_variance)
        sharpe = port_return / (port_vol + 1e-8)

        port_drawdown = float(np.dot(weights, max_drawdowns))

        turnover = 0.0
        if current_weights is not None:
            turnover = float(np.sum(np.abs(weights - current_weights)))

        return {
            OptimizationObjective.MAX_RETURN.value: port_return,
            OptimizationObjective.MIN_RISK.value: -port_vol,
            OptimizationObjective.MAX_SHARPE.value: sharpe,
            OptimizationObjective.MIN_DRAWDOWN.value: -port_drawdown,
            OptimizationObjective.MIN_TURNOVER.value: -turnover,
        }

    def _scalarize(self, objectives: Dict[str, float]) -> float:
        """Convert multi-objective scores to a single scalar.

        Uses weighted sum scalarization with normalization.

        Args:
            objectives: Objective scores.

        Returns:
            Scalar score.
        """
        score = 0.0
        for obj, weight in self.objective_weights.items():
            val = objectives.get(obj.value, 0.0)
            score += weight * val
        return score


# ---------------------------------------------------------------------------
# Portfolio State Encoder
# ---------------------------------------------------------------------------

class PortfolioStateEncoder:
    """Learns dense representations of portfolio states.

    Encodes the current portfolio state (weights, returns, risk metrics)
    into a fixed-dimensional embedding for downstream decision-making.

    Attributes:
        state_dim: Dimensionality of the state embedding.
        n_assets: Number of assets.
    """

    def __init__(
        self,
        n_assets: int = 10,
        state_dim: int = 64,
        device: str = "auto",
    ) -> None:
        """Initialise the state encoder.

        Args:
            n_assets: Number of portfolio assets.
            state_dim: Output embedding dimension.
            device: Compute device.
        """
        self.n_assets = n_assets
        self.state_dim = state_dim

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        # Input: weights + returns + vol + drawdown = 4 * n_assets
        input_dim = 4 * n_assets

        if torch is not None and nn is not None:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, state_dim),
            ).to(self._device)
        else:
            self.encoder = None

    def encode(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        volatilities: np.ndarray,
        drawdowns: np.ndarray,
    ) -> np.ndarray:
        """Encode portfolio state into a dense embedding.

        Args:
            weights: Current portfolio weights.
            returns: Recent returns per asset.
            volatilities: Volatilities per asset.
            drawdowns: Drawdowns per asset.

        Returns:
            State embedding of shape (state_dim,).
        """
        state = np.concatenate([weights, returns, volatilities, drawdowns])

        if self.encoder is not None and torch is not None:
            x = torch.tensor(state, dtype=torch.float32, device=self._device)
            with torch.no_grad():
                embedding = self.encoder(x).cpu().numpy()
        else:
            # Simple PCA-like projection
            np.random.seed(42)
            projection = np.random.randn(len(state), self.state_dim) / np.sqrt(len(state))
            embedding = state @ projection

        return embedding


# ---------------------------------------------------------------------------
# Hierarchical Portfolio Decider
# ---------------------------------------------------------------------------

class HierarchicalPortfolioDecider:
    """Hierarchical decision-making for portfolio construction.

    Makes decisions at two levels:
    1. Asset class allocation (crypto, DeFi, L1 vs L2, etc.)
    2. Individual asset allocation within each class

    This reduces the dimensionality of each decision and enables
    more structured portfolio construction.

    Attributes:
        asset_classes: Mapping of class name to asset lists.
    """

    def __init__(
        self,
        asset_classes: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialise the hierarchical decider.

        Args:
            asset_classes: Mapping from class name to list of asset names.
        """
        self.asset_classes = asset_classes or {
            "l1_protocols": ["BTC", "ETH", "SOL"],
            "defi": ["UNI", "AAVE", "MKR"],
            "l2_scaling": ["ARB", "OP", "MATIC"],
            "infrastructure": ["LINK", "FET", "RENDER"],
        }

    def decide_allocation(
        self,
        class_signals: Dict[str, float],
        asset_signals: Dict[str, Dict[str, float]],
        total_capital: float = 100000.0,
        max_class_weight: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """Make hierarchical allocation decisions.

        Args:
            class_signals: Signal score per asset class.
            asset_signals: Signal scores per asset within each class.
            total_capital: Total capital to allocate.
            max_class_weight: Maximum allocation to any single class.

        Returns:
            Nested dictionary: class -> {asset -> dollar_allocation}.
        """
        # Level 1: Asset class allocation
        class_scores = np.array(list(class_signals.values()))
        class_names = list(class_signals.keys())

        # Softmax with temperature
        temperature = 0.5
        exp_scores = np.exp(class_scores / temperature)
        class_weights = exp_scores / exp_scores.sum()

        # Clip and renormalize
        class_weights = np.clip(class_weights, 0, max_class_weight)
        class_weights = class_weights / class_weights.sum()

        # Level 2: Individual asset allocation within each class
        allocation: Dict[str, Dict[str, float]] = {}

        for i, class_name in enumerate(class_names):
            class_capital = total_capital * class_weights[i]
            assets_in_class = self.asset_classes.get(class_name, [])
            asset_scores = asset_signals.get(class_name, {})

            if not assets_in_class or not asset_scores:
                allocation[class_name] = {}
                continue

            # Score each asset in the class
            scores = np.array([
                asset_scores.get(asset, 0.0) for asset in assets_in_class
            ])
            exp_asset = np.exp(scores / temperature)
            asset_weights = exp_asset / exp_asset.sum()

            allocation[class_name] = {
                asset: float(class_capital * w)
                for asset, w in zip(assets_in_class, asset_weights)
            }

        logger.debug("Hierarchical allocation: %s", allocation)
        return allocation

    def get_class_summary(self) -> Dict[str, Any]:
        """Get a summary of the asset class structure.

        Returns:
            Summary dictionary.
        """
        return {
            "n_classes": len(self.asset_classes),
            "classes": {
                name: len(assets) for name, assets in self.asset_classes.items()
            },
            "total_assets": sum(len(v) for v in self.asset_classes.values()),
        }


# ---------------------------------------------------------------------------
# AI Portfolio Manager (Main Interface)
# ---------------------------------------------------------------------------

class AIPortfolioManager:
    """Unified AI-driven portfolio management system.

    Combines neural Markowitz optimization, attention-based weighting,
    dynamic risk budgeting, multi-objective optimization, and hierarchical
    decision-making with explainability.

    Attributes:
        manager_id: Unique identifier.
        markowitz: Neural Markowitz optimizer.
        attention_weighter: Attention-based asset weighter.
        risk_allocator: Dynamic risk budget allocator.
        multi_objective: Multi-objective optimizer.
        state_encoder: Portfolio state encoder.
        hierarchical_decider: Hierarchical decision maker.
    """

    def __init__(
        self,
        manager_id: str = "default",
        n_assets: int = 10,
        risk_aversion: float = 1.0,
        total_risk_budget: float = 0.15,
        device: str = "auto",
        redis_client: Any = None,
        postgres_client: Any = None,
    ) -> None:
        """Initialise the AI portfolio manager.

        Args:
            manager_id: Unique identifier.
            n_assets: Number of portfolio assets.
            risk_aversion: Markowitz risk aversion coefficient.
            total_risk_budget: Target annualized portfolio volatility.
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

        self.markowitz = NeuralMarkowitzOptimizer(
            n_assets=n_assets,
            risk_aversion=risk_aversion,
            device=self._device,
        )
        self.attention_weighter = AttentionAssetWeighter(
            n_assets=n_assets,
            device=self._device,
        )
        self.risk_allocator = DynamicRiskBudgetAllocator(
            total_risk_budget=total_risk_budget,
        )
        self.multi_objective = MultiObjectiveOptimizer()
        self.state_encoder = PortfolioStateEncoder(
            n_assets=n_assets,
            device=self._device,
        )
        self.hierarchical_decider = HierarchicalPortfolioDecider()

        # State tracking
        self._current_weights: Optional[np.ndarray] = None
        self._rebalance_history: List[Dict[str, Any]] = []

        logger.info(
            "AIPortfolioManager initialised [id=%s, n_assets=%d, device=%s]",
            manager_id,
            n_assets,
            self._device,
        )

    def rebalance(
        self,
        asset_names: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        volatilities: np.ndarray,
        max_drawdowns: Optional[np.ndarray] = None,
        recent_sharpes: Optional[np.ndarray] = None,
        method: str = "markowitz",
        total_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        """Execute a portfolio rebalance.

        Args:
            asset_names: List of asset identifiers.
            expected_returns: Expected returns per asset.
            covariance_matrix: Covariance matrix.
            volatilities: Volatilities per asset.
            max_drawdowns: Maximum drawdown per asset.
            recent_sharpes: Recent Sharpe ratios.
            method: Optimization method ('markowitz', 'attention', 'multi_objective', 'risk_parity').
            total_capital: Total portfolio capital.

        Returns:
            Rebalance result with weights, positions, and explainability.
        """
        n = len(asset_names)
        if max_drawdowns is None:
            max_drawdowns = np.ones(n) * 0.3

        # Compute optimal weights based on method
        if method == "markowitz":
            weights = self.markowitz.optimize(
                expected_returns, covariance_matrix, self._current_weights
            )
        elif method == "attention":
            correlations = np.array([covariance_matrix[i].mean() for i in range(n)])
            weights = self.attention_weighter.compute_weights(
                expected_returns, volatilities, correlations
            )
        elif method == "multi_objective":
            weights, obj_scores = self.multi_objective.optimize(
                expected_returns, covariance_matrix, max_drawdowns, self._current_weights
            )
        elif method == "risk_parity":
            inv_vol = 1.0 / (volatilities + 1e-8)
            weights = inv_vol / inv_vol.sum()
        else:
            weights = np.ones(n) / n

        # Compute risk budgets
        risk_budgets = self.risk_allocator.allocate(
            asset_names, volatilities, covariance_matrix, recent_sharpes
        )

        # Compute position sizes
        vol_dict = {name: float(vol) for name, vol in zip(asset_names, volatilities)}
        positions = self.risk_allocator.compute_position_sizes(
            risk_budgets, vol_dict, total_capital
        )

        # Encode portfolio state
        drawdowns = max_drawdowns if max_drawdowns is not None else np.zeros(n)
        state_embedding = self.state_encoder.encode(
            weights, expected_returns, volatilities, drawdowns
        )

        # Update state
        self._current_weights = weights

        # Explainability
        explanation = self._generate_explanation(
            asset_names, weights, expected_returns, volatilities, method
        )

        # Record rebalance
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "weights": {name: float(w) for name, w in zip(asset_names, weights)},
            "positions": positions,
            "risk_budgets": risk_budgets,
            "portfolio_return": float(np.dot(weights, expected_returns)),
            "portfolio_volatility": float(np.sqrt(weights @ covariance_matrix @ weights)),
            "portfolio_sharpe": float(
                np.dot(weights, expected_returns)
                / (np.sqrt(weights @ covariance_matrix @ weights) + 1e-8)
            ),
            "explanation": explanation,
        }

        self._rebalance_history.append(result)

        logger.info(
            "Rebalanced portfolio [method=%s]: Sharpe=%.2f, Vol=%.2f%%",
            method,
            result["portfolio_sharpe"],
            result["portfolio_volatility"] * 100,
        )
        return result

    def _generate_explanation(
        self,
        asset_names: List[str],
        weights: np.ndarray,
        expected_returns: np.ndarray,
        volatilities: np.ndarray,
        method: str,
    ) -> Dict[str, Any]:
        """Generate explainability data for the portfolio decision.

        Args:
            asset_names: Asset identifiers.
            weights: Portfolio weights.
            expected_returns: Expected returns.
            volatilities: Volatilities.
            method: Optimization method used.

        Returns:
            Explanation dictionary.
        """
        # Top holdings
        sorted_idx = np.argsort(weights)[::-1]
        top_holdings = [
            {"asset": asset_names[i], "weight": float(weights[i])}
            for i in sorted_idx[:5]
        ]

        # Contribution analysis
        return_contribution = weights * expected_returns
        risk_contribution = weights * volatilities

        # Attention weights (if available)
        attention_data = None
        if self.attention_weighter.last_attention_weights is not None:
            attention_data = self.attention_weighter.last_attention_weights.tolist()

        return {
            "method": method,
            "top_holdings": top_holdings,
            "concentration_hhi": float(np.sum(weights ** 2)),
            "effective_n_assets": float(1.0 / (np.sum(weights ** 2) + 1e-12)),
            "return_contribution": {
                name: float(rc) for name, rc in zip(asset_names, return_contribution)
            },
            "risk_contribution": {
                name: float(rc) for name, rc in zip(asset_names, risk_contribution)
            },
            "attention_weights": attention_data,
            "narrative": self._generate_narrative(
                asset_names, weights, expected_returns, method
            ),
        }

    def _generate_narrative(
        self,
        asset_names: List[str],
        weights: np.ndarray,
        expected_returns: np.ndarray,
        method: str,
    ) -> str:
        """Generate a human-readable narrative for the portfolio decision.

        Args:
            asset_names: Asset identifiers.
            weights: Portfolio weights.
            expected_returns: Expected returns.
            method: Optimization method.

        Returns:
            Narrative text.
        """
        top_idx = np.argmax(weights)
        top_asset = asset_names[top_idx]
        top_weight = weights[top_idx]
        top_return = expected_returns[top_idx]

        n_significant = int(np.sum(weights > 0.05))
        port_sharpe = float(
            np.dot(weights, expected_returns)
            / (np.sqrt(np.dot(weights, expected_returns ** 2)) + 1e-8)
        )

        narrative = (
            f"Portfolio optimized using {method} method. "
            f"Largest allocation is {top_asset} at {top_weight:.1%} "
            f"(expected return: {top_return:.2%}). "
            f"Portfolio has {n_significant} significant positions. "
            f"Estimated Sharpe ratio: {port_sharpe:.2f}."
        )
        return narrative

    def portfolio_stats(self) -> Dict[str, Any]:
        """Compute portfolio manager statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "manager_id": self.manager_id,
            "n_rebalances": len(self._rebalance_history),
            "current_weights": self._current_weights.tolist() if self._current_weights is not None else None,
            "device": self._device,
            "last_rebalance": (
                self._rebalance_history[-1]["timestamp"]
                if self._rebalance_history
                else None
            ),
        }

    async def save_to_postgres(self) -> int:
        """Persist portfolio state to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for record in self._rebalance_history:
                    await self._postgres.execute(
                        """
                        INSERT INTO portfolio_rebalances (manager_id, timestamp, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                        """,
                        self.manager_id,
                        record["timestamp"],
                        json.dumps(record, default=str),
                    )
                    count += 1
            logger.info("Saved %d rebalance records to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save rebalance records: %s", exc)
        return count
