"""
AIStrategySelector - Multi-Armed Bandit Strategy Selection for Crypto Trading
==============================================================================

Implements intelligent strategy selection using:
- Thompson Sampling for Bayesian exploration/exploitation
- Multi-armed bandit framework for strategy arms
- Regime-strategy mapping with confidence scores
- Strategy performance prediction using Bayesian updates
- Dynamic strategy allocation based on market conditions
- Strategy combination optimization for ensemble approaches

GPU-ready with PyTorch-based Bayesian inference and optimization.
"""

from __future__ import annotations

import json
import logging
import math
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
    from torch.distributions import Beta, Normal

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Beta = None  # type: ignore[assignment]
    Normal = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RegimeType(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class StrategyCategory(str, Enum):
    """Category of trading strategy."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    STATISTICAL_ARB = "statistical_arb"
    MARKET_MAKING = "market_making"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    SENTIMENT = "sentiment"
    ML_BASED = "ml_based"


# ---------------------------------------------------------------------------
# Strategy Arm (Multi-Armed Bandit)
# ---------------------------------------------------------------------------

@dataclass
class StrategyArm:
    """A strategy represented as a multi-armed bandit arm.

    Uses Beta distribution posterior for Thompson Sampling,
    tracking successes and failures based on strategy outcomes.

    Attributes:
        strategy_id: Unique identifier.
        name: Human-readable name.
        category: Strategy category.
        alpha: Beta distribution alpha parameter (successes + 1).
        beta: Beta distribution beta parameter (failures + 1).
        total_plays: Number of times this strategy has been selected.
        total_reward: Cumulative reward.
        avg_reward: Average reward per play.
        regimes_played: Count of plays per regime.
        regimes_reward: Cumulative reward per regime.
        last_played: Timestamp of last selection.
        is_active: Whether the strategy is available for selection.
    """

    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: StrategyCategory = StrategyCategory.MOMENTUM
    alpha: float = 1.0
    beta: float = 1.0
    total_plays: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    regimes_played: Dict[str, int] = field(default_factory=dict)
    regimes_reward: Dict[str, float] = field(default_factory=dict)
    last_played: Optional[datetime] = None
    is_active: bool = True

    @property
    def expected_reward(self) -> float:
        """Expected reward under the Beta posterior."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """Uncertainty (standard deviation) of the Beta posterior."""
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def update(self, reward: float, regime: str = "") -> None:
        """Update the arm's posterior with a new observation.

        Args:
            reward: Observed reward in [0, 1] (1 = success, 0 = failure).
            regime: Market regime when this play occurred.
        """
        # Bayesian update of Beta posterior
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.total_plays += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.total_plays
        self.last_played = datetime.now(timezone.utc)

        if regime:
            self.regimes_played[regime] = self.regimes_played.get(regime, 0) + 1
            self.regimes_reward[regime] = self.regimes_reward.get(regime, 0.0) + reward

    def regime_expected_reward(self, regime: str) -> float:
        """Expected reward for a specific regime.

        Uses a regime-specific Beta posterior if enough data exists,
        otherwise falls back to the global posterior.

        Args:
            regime: Market regime.

        Returns:
            Expected reward for the regime.
        """
        plays = self.regimes_played.get(regime, 0)
        if plays < 3:
            return self.expected_reward  # Not enough data, use global

        regime_reward = self.regimes_reward.get(regime, 0.0)
        # Regime-specific posterior with prior from global
        prior_weight = 2.0
        regime_alpha = regime_reward + prior_weight * (self.alpha / (self.alpha + self.beta))
        regime_beta = (plays - regime_reward) + prior_weight * (self.beta / (self.alpha + self.beta))
        return regime_alpha / (regime_alpha + regime_beta)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "category": self.category.value,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_plays": self.total_plays,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "regimes_played": self.regimes_played,
            "regimes_reward": self.regimes_reward,
            "is_active": self.is_active,
        }


# ---------------------------------------------------------------------------
# Thompson Sampler
# ---------------------------------------------------------------------------

class ThompsonSampler:
    """Thompson Sampling for Bayesian strategy selection.

    Samples from Beta posteriors to balance exploration and
    exploitation. GPU-accelerated when available.

    Attributes:
        device: Compute device for sampling.
    """

    def __init__(self, device: str = "auto") -> None:
        """Initialise the Thompson sampler.

        Args:
            device: Compute device ('auto', 'cuda', 'cpu').
        """
        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

    def sample(self, arms: List[StrategyArm], regime: Optional[str] = None) -> int:
        """Select an arm using Thompson Sampling.

        Draws one sample from each arm's posterior and selects
        the arm with the highest sample value.

        Args:
            arms: List of strategy arms.
            regime: Optional regime context for regime-aware sampling.

        Returns:
            Index of the selected arm.
        """
        if not arms:
            raise ValueError("No arms provided for sampling")

        active_arms = [(i, arm) for i, arm in enumerate(arms) if arm.is_active]
        if not active_arms:
            raise ValueError("No active arms available for sampling")

        if GPU_AVAILABLE and torch is not None and Beta is not None:
            samples = []
            for _, arm in active_arms:
                if regime:
                    a = arm.alpha * arm.regimes_played.get(regime, 1)
                    b = arm.beta * max(0.1, arm.regimes_played.get(regime, 1) - arm.regimes_reward.get(regime, 0))
                    a = max(0.01, a)
                    b = max(0.01, b)
                else:
                    a = max(0.01, arm.alpha)
                    b = max(0.01, arm.beta)
                dist = Beta(torch.tensor(a), torch.tensor(b))
                samples.append(dist.sample().item())
        else:
            samples = []
            for _, arm in active_arms:
                if regime:
                    expected = arm.regime_expected_reward(regime)
                else:
                    expected = arm.expected_reward
                # Numpy fallback for sampling
                a = max(0.01, arm.alpha)
                b = max(0.01, arm.beta)
                sample = np.random.beta(a, b)
                if regime and arm.regimes_played.get(regime, 0) >= 3:
                    # Blend with regime-specific expectation
                    sample = 0.7 * sample + 0.3 * expected
                samples.append(sample)

        best_local_idx = int(np.argmax(samples))
        return active_arms[best_local_idx][0]

    def batch_sample(
        self, arms: List[StrategyArm], n_samples: int = 100, regime: Optional[str] = None
    ) -> np.ndarray:
        """Generate batch samples for analysis.

        Args:
            arms: List of strategy arms.
            n_samples: Number of samples per arm.
            regime: Optional regime context.

        Returns:
            Array of shape (n_arms, n_samples) with sampled rewards.
        """
        samples = np.zeros((len(arms), n_samples))
        for i, arm in enumerate(arms):
            a = max(0.01, arm.alpha)
            b = max(0.01, arm.beta)
            samples[i] = np.random.beta(a, b, size=n_samples)
        return samples


# ---------------------------------------------------------------------------
# Regime-Strategy Mapper
# ---------------------------------------------------------------------------

class RegimeStrategyMapper:
    """Maps market regimes to suitable strategies with confidence scores.

    Maintains a learned mapping from regime types to strategy categories,
    updated based on observed outcomes. Provides confidence-weighted
    recommendations.

    Attributes:
        mapping: Regime -> [(strategy_category, confidence)] mappings.
        decay_rate: How quickly old evidence decays.
    """

    def __init__(self, decay_rate: float = 0.05) -> None:
        """Initialise the regime-strategy mapper.

        Args:
            decay_rate: Exponential decay rate for old evidence.
        """
        self.decay_rate = decay_rate
        self.mapping: Dict[str, Dict[str, float]] = {}
        self._evidence_count: Dict[str, Dict[str, int]] = {}

        # Initialize with prior knowledge
        self._init_priors()

    def _init_priors(self) -> None:
        """Set prior regime-strategy mappings based on domain knowledge."""
        priors = {
            RegimeType.TRENDING_UP.value: {
                StrategyCategory.TREND_FOLLOWING.value: 0.8,
                StrategyCategory.MOMENTUM.value: 0.7,
                StrategyCategory.BREAKOUT.value: 0.6,
            },
            RegimeType.TRENDING_DOWN.value: {
                StrategyCategory.TREND_FOLLOWING.value: 0.7,
                StrategyCategory.MOMENTUM.value: 0.6,
                StrategyCategory.MEAN_REVERSION.value: 0.3,
            },
            RegimeType.RANGING.value: {
                StrategyCategory.MEAN_REVERSION.value: 0.8,
                StrategyCategory.MARKET_MAKING.value: 0.7,
                StrategyCategory.STATISTICAL_ARB.value: 0.6,
            },
            RegimeType.VOLATILE.value: {
                StrategyCategory.STATISTICAL_ARB.value: 0.5,
                StrategyCategory.MARKET_MAKING.value: 0.3,
                StrategyCategory.MEAN_REVERSION.value: 0.4,
            },
            RegimeType.CRISIS.value: {
                StrategyCategory.MEAN_REVERSION.value: 0.3,
                StrategyCategory.STATISTICAL_ARB.value: 0.2,
                StrategyCategory.MARKET_MAKING.value: 0.1,
            },
            RegimeType.RECOVERY.value: {
                StrategyCategory.MOMENTUM.value: 0.6,
                StrategyCategory.TREND_FOLLOWING.value: 0.7,
                StrategyCategory.BREAKOUT.value: 0.5,
            },
        }

        for regime, strategies in priors.items():
            self.mapping[regime] = strategies.copy()
            self._evidence_count[regime] = {s: 1 for s in strategies}

    def update(
        self, regime: RegimeType, strategy_category: StrategyCategory, reward: float
    ) -> None:
        """Update the regime-strategy mapping with a new observation.

        Args:
            regime: Observed market regime.
            strategy_category: Strategy category used.
            reward: Observed reward in [0, 1].
        """
        r = regime.value
        s = strategy_category.value

        self.mapping.setdefault(r, {})
        self._evidence_count.setdefault(r, {})

        current_conf = self.mapping[r].get(s, 0.5)
        count = self._evidence_count[r].get(s, 0)

        # Bayesian update with evidence weighting
        evidence_weight = min(1.0, count / 20.0)  # More evidence -> slower updates
        alpha = 0.2 * (1 - evidence_weight * 0.5)
        updated_conf = alpha * reward + (1 - alpha) * current_conf

        self.mapping[r][s] = float(np.clip(updated_conf, 0.0, 1.0))
        self._evidence_count[r][s] = count + 1

    def get_recommendations(
        self, regime: RegimeType, top_k: int = 3, min_confidence: float = 0.3
    ) -> List[Tuple[StrategyCategory, float]]:
        """Get strategy recommendations for a given regime.

        Args:
            regime: Current market regime.
            top_k: Number of recommendations.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of (strategy_category, confidence) pairs.
        """
        regime_strategies = self.mapping.get(regime.value, {})
        ranked = sorted(regime_strategies.items(), key=lambda x: x[1], reverse=True)
        return [
            (StrategyCategory(cat), conf)
            for cat, conf in ranked[:top_k]
            if conf >= min_confidence
        ]

    def get_best_strategy(self, regime: RegimeType) -> Optional[Tuple[StrategyCategory, float]]:
        """Get the single best strategy for a regime.

        Args:
            regime: Current market regime.

        Returns:
            Tuple of (strategy_category, confidence) or None.
        """
        recs = self.get_recommendations(regime, top_k=1)
        return recs[0] if recs else None


# ---------------------------------------------------------------------------
# Strategy Combination Optimizer
# ---------------------------------------------------------------------------

class StrategyCombinationOptimizer:
    """Optimizes the combination of multiple strategies for ensemble trading.

    Uses gradient-free optimization (CMA-ES style) to find optimal
    weight allocations across strategies. Supports constraints on
    minimum/maximum weights and turnover penalties.

    Attributes:
        n_strategies: Number of strategies in the ensemble.
        max_weight: Maximum weight per strategy.
        min_weight: Minimum weight per strategy.
        turnover_penalty: Penalty for weight changes between periods.
    """

    def __init__(
        self,
        n_strategies: int = 5,
        max_weight: float = 0.5,
        min_weight: float = 0.0,
        turnover_penalty: float = 0.01,
        n_candidates: int = 64,
    ) -> None:
        """Initialise the combination optimizer.

        Args:
            n_strategies: Number of strategies to combine.
            max_weight: Maximum allocation to any single strategy.
            min_weight: Minimum allocation (0 = can be excluded).
            turnover_penalty: Regularization for weight changes.
            n_candidates: Number of candidate weight vectors per optimization step.
        """
        self.n_strategies = n_strategies
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.turnover_penalty = turnover_penalty
        self.n_candidates = n_candidates
        self.current_weights = np.ones(n_strategies) / n_strategies

    def optimize(
        self,
        reward_matrix: np.ndarray,
        correlation_matrix: Optional[np.ndarray] = None,
        previous_weights: Optional[np.ndarray] = None,
        n_iterations: int = 50,
    ) -> np.ndarray:
        """Find optimal strategy weights.

        Args:
            reward_matrix: Array of shape (n_periods, n_strategies) with historical rewards.
            correlation_matrix: Optional strategy correlation matrix (n_strategies, n_strategies).
            previous_weights: Weights from the previous period.
            n_iterations: Number of optimization iterations.

        Returns:
            Optimal weight vector of shape (n_strategies,).
        """
        if previous_weights is None:
            previous_weights = self.current_weights

        mean_rewards = reward_matrix.mean(axis=0)
        std_rewards = reward_matrix.std(axis=0) + 1e-8

        best_weights = self.current_weights.copy()
        best_score = -np.inf

        for _ in range(n_iterations):
            # Generate candidate weight vectors
            candidates = np.random.dirichlet(
                np.ones(self.n_strategies), size=self.n_candidates
            )

            # Clip to constraints
            candidates = np.clip(candidates, self.min_weight, self.max_weight)
            # Re-normalize
            candidates = candidates / (candidates.sum(axis=1, keepdims=True) + 1e-12)

            # Score each candidate
            for c in candidates:
                score = self._score_weights(
                    c, mean_rewards, std_rewards, correlation_matrix, previous_weights
                )
                if score > best_score:
                    best_score = score
                    best_weights = c.copy()

        self.current_weights = best_weights

        logger.debug(
            "Optimized strategy weights: %s (score=%.4f)",
            best_weights,
            best_score,
        )
        return best_weights

    def _score_weights(
        self,
        weights: np.ndarray,
        mean_rewards: np.ndarray,
        std_rewards: np.ndarray,
        correlation_matrix: Optional[np.ndarray],
        previous_weights: np.ndarray,
    ) -> float:
        """Score a weight vector using risk-adjusted return with penalties.

        Args:
            weights: Candidate weight vector.
            mean_rewards: Mean reward per strategy.
            std_rewards: Std of rewards per strategy.
            correlation_matrix: Strategy correlation matrix.
            previous_weights: Previous period weights.

        Returns:
            Score (higher is better).
        """
        # Portfolio expected return
        port_return = np.dot(weights, mean_rewards)

        # Portfolio risk (variance)
        if correlation_matrix is not None:
            port_variance = weights @ correlation_matrix @ weights
        else:
            port_variance = np.dot(weights ** 2, std_rewards ** 2)

        # Sharpe-like score
        sharpe_score = port_return / (np.sqrt(port_variance) + 1e-8)

        # Diversification bonus
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        diversification_bonus = 0.1 * entropy

        # Turnover penalty
        turnover = np.sum(np.abs(weights - previous_weights))
        turnover_cost = self.turnover_penalty * turnover

        return sharpe_score + diversification_bonus - turnover_cost

    def get_allocation_report(self) -> Dict[str, Any]:
        """Generate a report on the current strategy allocation.

        Returns:
            Dictionary with allocation details.
        """
        return {
            "weights": self.current_weights.tolist(),
            "max_weight": float(self.current_weights.max()),
            "min_weight": float(self.current_weights.min()),
            "concentration": float(np.sum(self.current_weights ** 2)),  # HHI
            "effective_n": float(1.0 / (np.sum(self.current_weights ** 2) + 1e-12)),
        }


# ---------------------------------------------------------------------------
# AI Strategy Selector (Main Interface)
# ---------------------------------------------------------------------------

class AIStrategySelector:
    """Intelligent strategy selection using multi-armed bandits and Thompson Sampling.

    Provides regime-aware strategy selection with:
    - Thompson Sampling for exploration/exploitation balance
    - Regime-strategy confidence mapping
    - Dynamic strategy allocation
    - Strategy combination optimization for ensembles
    - Performance prediction for upcoming periods

    Attributes:
        selector_id: Unique identifier.
        arms: Registered strategy arms.
        sampler: Thompson Sampling engine.
        regime_mapper: Regime-strategy mapping engine.
        combination_optimizer: Strategy weight optimizer.
    """

    def __init__(
        self,
        selector_id: str = "default",
        device: str = "auto",
        exploration_bonus: float = 0.1,
        regime_aware: bool = True,
        redis_client: Any = None,
        postgres_client: Any = None,
    ) -> None:
        """Initialise the strategy selector.

        Args:
            selector_id: Unique identifier.
            device: Compute device.
            exploration_bonus: Bonus added to uncertain arms for exploration.
            regime_aware: Whether to use regime context in selection.
            redis_client: Optional Redis client.
            postgres_client: Optional Postgres client.
        """
        self.selector_id = selector_id
        self.exploration_bonus = exploration_bonus
        self.regime_aware = regime_aware
        self._redis = redis_client
        self._postgres = postgres_client

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        self.arms: Dict[str, StrategyArm] = {}
        self.sampler = ThompsonSampler(device=self._device)
        self.regime_mapper = RegimeStrategyMapper()
        self.combination_optimizer = StrategyCombinationOptimizer()

        self._selection_history: List[Dict[str, Any]] = []

        logger.info(
            "AIStrategySelector initialised [id=%s, device=%s, regime_aware=%s]",
            selector_id,
            self._device,
            regime_aware,
        )

    # -- Arm Registration --------------------------------------------------

    def register_strategy(
        self,
        name: str,
        category: StrategyCategory = StrategyCategory.MOMENTUM,
        strategy_id: Optional[str] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> str:
        """Register a new strategy as a bandit arm.

        Args:
            name: Human-readable strategy name.
            category: Strategy category.
            strategy_id: Optional custom ID.
            prior_alpha: Prior alpha for Beta distribution.
            prior_beta: Prior beta for Beta distribution.

        Returns:
            The strategy_id.
        """
        arm = StrategyArm(
            strategy_id=strategy_id or str(uuid.uuid4()),
            name=name,
            category=category,
            alpha=prior_alpha,
            beta=prior_beta,
        )
        self.arms[arm.strategy_id] = arm

        # Update combination optimizer
        self.combination_optimizer = StrategyCombinationOptimizer(
            n_strategies=len(self.arms),
            max_weight=self.combination_optimizer.max_weight,
            min_weight=self.combination_optimizer.min_weight,
            turnover_penalty=self.combination_optimizer.turnover_penalty,
        )

        logger.info("Registered strategy arm: %s (%s)", name, category.value)
        return arm.strategy_id

    def deregister_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy arm.

        Args:
            strategy_id: ID of the strategy to remove.

        Returns:
            True if the strategy was found and removed.
        """
        if strategy_id in self.arms:
            del self.arms[strategy_id]
            return True
        return False

    # -- Strategy Selection ------------------------------------------------

    def select_strategy(
        self,
        regime: Optional[RegimeType] = None,
        exclude: Optional[List[str]] = None,
        temperature: float = 1.0,
    ) -> Tuple[str, float]:
        """Select the best strategy for the current market conditions.

        Uses Thompson Sampling with optional regime awareness.

        Args:
            regime: Current market regime (used if regime_aware=True).
            exclude: List of strategy IDs to exclude.
            temperature: Sampling temperature (higher = more exploration).

        Returns:
            Tuple of (strategy_id, sampled_reward).
        """
        if not self.arms:
            raise ValueError("No strategies registered")

        available_arms = [
            arm for arm in self.arms.values()
            if arm.is_active and (exclude is None or arm.strategy_id not in exclude)
        ]

        if not available_arms:
            raise ValueError("No active strategies available for selection")

        regime_str = regime.value if regime and self.regime_aware else None
        selected_idx = self.sampler.sample(available_arms, regime=regime_str)
        selected_arm = available_arms[selected_idx]

        # Compute sampled reward for reporting
        sampled_reward = np.random.beta(
            max(0.01, selected_arm.alpha),
            max(0.01, selected_arm.beta),
        )

        # Record selection
        self._selection_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_id": selected_arm.strategy_id,
            "regime": regime.value if regime else None,
            "sampled_reward": float(sampled_reward),
        })

        logger.info(
            "Selected strategy: %s (regime=%s, sampled_reward=%.3f)",
            selected_arm.name,
            regime.value if regime else "none",
            sampled_reward,
        )
        return selected_arm.strategy_id, float(sampled_reward)

    def select_top_k_strategies(
        self, regime: Optional[RegimeType] = None, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Select the top-k strategies using batch Thompson Sampling.

        Args:
            regime: Current market regime.
            k: Number of strategies to select.

        Returns:
            List of (strategy_id, expected_reward) pairs.
        """
        active_arms = [arm for arm in self.arms.values() if arm.is_active]
        if not active_arms:
            return []

        regime_str = regime.value if regime and self.regime_aware else None

        # Sample multiple times and count selections
        selection_counts: Dict[str, int] = {}
        n_trials = k * 100

        for _ in range(n_trials):
            idx = self.sampler.sample(active_arms, regime=regime_str)
            sid = active_arms[idx].strategy_id
            selection_counts[sid] = selection_counts.get(sid, 0) + 1

        # Rank by selection frequency
        ranked = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
        results: List[Tuple[str, float]] = []
        for sid, count in ranked[:k]:
            arm = self.arms[sid]
            expected = arm.regime_expected_reward(regime_str) if regime_str else arm.expected_reward
            results.append((sid, expected))

        return results

    # -- Feedback ----------------------------------------------------------

    def update_outcome(
        self,
        strategy_id: str,
        reward: float,
        pnl: float = 0.0,
        regime: Optional[RegimeType] = None,
    ) -> None:
        """Update strategy arms with observed outcomes.

        Args:
            strategy_id: ID of the strategy that was used.
            reward: Normalized reward in [0, 1].
            pnl: Raw PnL value.
            regime: Market regime during the outcome.
        """
        arm = self.arms.get(strategy_id)
        if arm is None:
            logger.warning("Unknown strategy_id: %s", strategy_id)
            return

        regime_str = regime.value if regime else ""
        arm.update(reward, regime=regime_str)

        # Update regime mapper
        if regime:
            self.regime_mapper.update(regime, arm.category, reward)

        logger.debug(
            "Updated strategy %s: reward=%.3f, alpha=%.1f, beta=%.1f",
            strategy_id,
            reward,
            arm.alpha,
            arm.beta,
        )

    # -- Dynamic Allocation ------------------------------------------------

    def compute_dynamic_allocation(
        self,
        regime: Optional[RegimeType] = None,
        reward_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, float]:
        """Compute optimal strategy allocation weights.

        Args:
            regime: Current market regime.
            reward_history: Historical rewards per strategy.

        Returns:
            Dictionary mapping strategy_id to weight.
        """
        active_ids = [sid for sid, arm in self.arms.items() if arm.is_active]
        n = len(active_ids)
        if n == 0:
            return {}

        # Build reward matrix if history provided
        if reward_history:
            max_len = max(len(v) for v in reward_history.values()) if reward_history else 0
            reward_matrix = np.zeros((max(1, max_len), n))
            for j, sid in enumerate(active_ids):
                history = reward_history.get(sid, [0.0])
                for i, r in enumerate(history):
                    if i < max_len:
                        reward_matrix[i, j] = r
        else:
            # Use expected rewards as proxy
            expected = np.array([self.arms[sid].expected_reward for sid in active_ids])
            reward_matrix = np.tile(expected, (20, 1))
            # Add noise for variation
            reward_matrix += np.random.normal(0, 0.05, reward_matrix.shape)

        weights = self.combination_optimizer.optimize(reward_matrix)

        allocation = {
            sid: float(w) for sid, w in zip(active_ids, weights) if w > 0.01
        }

        # Normalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}

        logger.info("Dynamic allocation: %s", allocation)
        return allocation

    # -- Performance Prediction --------------------------------------------

    def predict_performance(
        self, strategy_id: str, regime: Optional[RegimeType] = None, horizon: int = 10
    ) -> Dict[str, Any]:
        """Predict future performance of a strategy.

        Uses the Beta posterior to generate a predictive distribution
        for the next `horizon` periods.

        Args:
            strategy_id: Strategy to predict.
            regime: Expected regime.
            horizon: Number of periods to predict.

        Returns:
            Dictionary with prediction statistics.
        """
        arm = self.arms.get(strategy_id)
        if arm is None:
            return {"error": f"Unknown strategy: {strategy_id}"}

        # Sample from posterior
        n_simulations = 1000
        a = max(0.01, arm.alpha)
        b = max(0.01, arm.beta)

        samples = np.random.beta(a, b, size=(n_simulations, horizon))
        cumulative = samples.cumsum(axis=1)

        regime_str = regime.value if regime else None
        if regime_str and regime_str in arm.regimes_played:
            # Adjust with regime-specific expectation
            regime_exp = arm.regime_expected_reward(regime_str)
            global_exp = arm.expected_reward
            adjustment = regime_exp - global_exp
            samples = np.clip(samples + adjustment * 0.3, 0, 1)
            cumulative = samples.cumsum(axis=1)

        return {
            "strategy_id": strategy_id,
            "regime": regime.value if regime else None,
            "horizon": horizon,
            "expected_reward": float(samples.mean()),
            "std_reward": float(samples.std()),
            "expected_cumulative": float(cumulative[:, -1].mean()),
            "p5_cumulative": float(np.percentile(cumulative[:, -1], 5)),
            "p95_cumulative": float(np.percentile(cumulative[:, -1], 95)),
            "sharpe_estimate": float(samples.mean() / (samples.std() + 1e-8)),
        }

    # -- Statistics --------------------------------------------------------

    def selector_stats(self) -> Dict[str, Any]:
        """Compute comprehensive selector statistics.

        Returns:
            Statistics dictionary.
        """
        if not self.arms:
            return {"selector_id": self.selector_id, "total_arms": 0}

        arm_stats = []
        for arm in self.arms.values():
            arm_stats.append({
                "strategy_id": arm.strategy_id,
                "name": arm.name,
                "category": arm.category.value,
                "expected_reward": round(arm.expected_reward, 4),
                "uncertainty": round(arm.uncertainty, 4),
                "total_plays": arm.total_plays,
                "avg_reward": round(arm.avg_reward, 4),
                "is_active": arm.is_active,
            })

        active_count = sum(1 for a in self.arms.values() if a.is_active)

        return {
            "selector_id": self.selector_id,
            "total_arms": len(self.arms),
            "active_arms": active_count,
            "regime_aware": self.regime_aware,
            "total_selections": len(self._selection_history),
            "arms": arm_stats,
            "allocation": self.combination_optimizer.get_allocation_report(),
            "device": self._device,
        }

    # -- Persistence -------------------------------------------------------

    async def save_to_postgres(self) -> int:
        """Persist selector state to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for arm in self.arms.values():
                    await self._postgres.execute(
                        """
                        INSERT INTO strategy_arms (strategy_id, selector_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (strategy_id) DO UPDATE SET data = $3
                        """,
                        arm.strategy_id,
                        self.selector_id,
                        json.dumps(arm.to_dict()),
                    )
                    count += 1
            logger.info("Saved %d strategy arms to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save strategy arms: %s", exc)
        return count
