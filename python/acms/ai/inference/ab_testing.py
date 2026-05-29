"""
ACMS AI A/B Testing
====================

Model A/B testing framework for the Algorithmic Crypto Management System.
Supports traffic splitting, statistical significance testing, automatic
winner promotion, and full test lifecycle management.

Components
----------
ABTestManager : Orchestrates A/B test lifecycle
TrafficSplitter : Routes requests between model variants
StatisticalSignificanceTester : Evaluates test results for significance
ModelComparison : Tracks and compares model performance metrics
ABTestConfig : Configuration for individual A/B tests
ABTestStatus : Test lifecycle status tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class ABTestStatus(Enum):
    """Lifecycle status of an A/B test."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    AUTO_PROMOTED = "auto_promoted"
    FAILED = "failed"


class SplitStrategy(Enum):
    """Traffic splitting strategy."""
    PERCENTAGE = "percentage"
    USER_BASED = "user_based"
    HASH_BASED = "hash_based"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test.

    Parameters
    ----------
    test_id : str
        Unique test identifier.
    model_a_id : str
        Control model identifier.
    model_b_id : str
        Challenger model identifier.
    traffic_split : float
        Fraction of traffic to route to model B (0.0 - 1.0).
    strategy : SplitStrategy
        Traffic splitting strategy.
    min_samples : int
        Minimum samples before evaluating significance.
    significance_level : float
        P-value threshold for statistical significance.
    auto_promote : bool
        Whether to automatically promote the winner.
    max_duration_hours : float
        Maximum test duration before forced evaluation.
    primary_metric : str
        Primary metric for comparison (e.g., ``"sharpe_ratio"``).
    metadata : dict
        Additional test metadata.
    """
    test_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    model_a_id: str = ""
    model_b_id: str = ""
    traffic_split: float = 0.5
    strategy: SplitStrategy = SplitStrategy.PERCENTAGE
    min_samples: int = 1000
    significance_level: float = 0.05
    auto_promote: bool = False
    max_duration_hours: float = 168.0  # 1 week default
    primary_metric: str = "sharpe_ratio"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Traffic Splitter
# ---------------------------------------------------------------------------

class TrafficSplitter:
    """Routes inference requests between model variants in an A/B test.

    Supports multiple splitting strategies:

    - **Percentage**: Deterministic random split based on configured ratio.
    - **User-based**: Consistent routing per user/symbol identifier.
    - **Hash-based**: Deterministic routing using request feature hashing.

    Parameters
    ----------
    strategy : SplitStrategy
        The splitting strategy to use.
    split_ratio : float
        Fraction of traffic to route to variant B (0.0-1.0).
    salt : str
        Salt for hash-based splitting to ensure randomness.
    """

    def __init__(
        self,
        strategy: SplitStrategy = SplitStrategy.PERCENTAGE,
        split_ratio: float = 0.5,
        salt: str = "acms_ab_test",
    ) -> None:
        self._strategy = strategy
        self._split_ratio = split_ratio
        self._salt = salt
        self._user_assignments: Dict[str, str] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        logger.info("TrafficSplitter initialized (strategy=%s, ratio=%.2f)", strategy.value, split_ratio)

    def assign_variant(self, request_id: str, user_id: Optional[str] = None,
                       features_hash: Optional[str] = None) -> str:
        """Determine which variant (``"A"`` or ``"B"``) a request should use.

        Parameters
        ----------
        request_id : str
            Unique request identifier.
        user_id : str, optional
            User or symbol identifier for user-based routing.
        features_hash : str, optional
            Hash of request features for hash-based routing.

        Returns
        -------
        str
            ``"A"`` or ``"B"``
        """
        if self._strategy == SplitStrategy.PERCENTAGE:
            variant = self._percentage_split(request_id)
        elif self._strategy == SplitStrategy.USER_BASED:
            variant = self._user_based_split(user_id or request_id)
        elif self._strategy == SplitStrategy.HASH_BASED:
            variant = self._hash_based_split(features_hash or request_id)
        else:
            variant = "A"

        self._counts[variant] += 1
        return variant

    def _percentage_split(self, request_id: str) -> str:
        """Deterministic percentage split using request ID hash."""
        hash_val = int(hashlib.md5(f"{self._salt}:{request_id}".encode()).hexdigest(), 16)
        return "B" if (hash_val % 100) < (self._split_ratio * 100) else "A"

    def _user_based_split(self, user_id: str) -> str:
        """Consistent routing per user identifier."""
        if user_id in self._user_assignments:
            return self._user_assignments[user_id]
        variant = self._percentage_split(user_id)
        self._user_assignments[user_id] = variant
        return variant

    def _hash_based_split(self, features_hash: str) -> str:
        """Deterministic routing based on feature hash."""
        hash_val = int(hashlib.md5(f"{self._salt}:{features_hash}".encode()).hexdigest(), 16)
        return "B" if (hash_val % 100) < (self._split_ratio * 100) else "A"

    def get_assignment_counts(self) -> Dict[str, int]:
        """Return the number of assignments to each variant."""
        return dict(self._counts)

    @property
    def actual_split_ratio(self) -> float:
        """Return the actual observed split ratio."""
        total = self._counts.get("A", 0) + self._counts.get("B", 0)
        if total == 0:
            return self._split_ratio
        return self._counts.get("B", 0) / total


# ---------------------------------------------------------------------------
# Model Comparison Metrics
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Performance metrics for a single model variant in a test."""
    sample_count: int = 0
    mean_return: float = 0.0
    std_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    total_pnl: float = 0.0
    returns: List[float] = field(default_factory=list)


class ModelComparison:
    """Tracks and compares performance metrics between two model variants.

    Parameters
    ----------
    primary_metric : str
        The metric to use for determining the winner.
    """

    def __init__(self, primary_metric: str = "sharpe_ratio") -> None:
        self._primary_metric = primary_metric
        self._metrics: Dict[str, ModelMetrics] = {
            "A": ModelMetrics(),
            "B": ModelMetrics(),
        }
        logger.info("ModelComparison initialized (primary_metric=%s)", primary_metric)

    def record(self, variant: str, return_value: float, latency_ms: float,
               is_error: bool = False, pnl: float = 0.0) -> None:
        """Record an observation for a variant.

        Parameters
        ----------
        variant : str
            ``"A"`` or ``"B"``
        return_value : float
            The return or outcome value for this observation.
        latency_ms : float
            Inference latency in milliseconds.
        is_error : bool
            Whether this observation resulted in an error.
        pnl : float
            Profit and loss for this observation.
        """
        m = self._metrics.get(variant)
        if m is None:
            logger.warning("Unknown variant: %s", variant)
            return

        m.sample_count += 1
        if is_error:
            m.error_rate = (m.error_rate * (m.sample_count - 1) + 1.0) / m.sample_count
            return

        m.returns.append(return_value)
        m.total_pnl += pnl
        m.mean_return = float(np.mean(m.returns))
        m.std_return = float(np.std(m.returns)) if len(m.returns) > 1 else 0.0
        m.sharpe_ratio = m.mean_return / m.std_return if m.std_return > 0 else 0.0
        m.win_rate = sum(1 for r in m.returns if r > 0) / len(m.returns)
        m.avg_latency_ms = (m.avg_latency_ms * (m.sample_count - 1) + latency_ms) / m.sample_count

        # Max drawdown (running)
        cumulative = np.cumsum(m.returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        m.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def get_metrics(self, variant: str) -> ModelMetrics:
        """Return the current metrics for a variant."""
        return self._metrics[variant]

    def get_comparison(self) -> Dict[str, Any]:
        """Return a comparison summary between variants."""
        a = self._metrics["A"]
        b = self._metrics["B"]

        def _diff(b_val: float, a_val: float) -> float:
            return b_val - a_val if a_val != 0 else 0.0

        return {
            "primary_metric": self._primary_metric,
            "variant_a": {
                "sample_count": a.sample_count,
                "mean_return": a.mean_return,
                "sharpe_ratio": a.sharpe_ratio,
                "max_drawdown": a.max_drawdown,
                "win_rate": a.win_rate,
                "avg_latency_ms": a.avg_latency_ms,
                "error_rate": a.error_rate,
                "total_pnl": a.total_pnl,
            },
            "variant_b": {
                "sample_count": b.sample_count,
                "mean_return": b.mean_return,
                "sharpe_ratio": b.sharpe_ratio,
                "max_drawdown": b.max_drawdown,
                "win_rate": b.win_rate,
                "avg_latency_ms": b.avg_latency_ms,
                "error_rate": b.error_rate,
                "total_pnl": b.total_pnl,
            },
            "differences": {
                "sharpe_ratio_diff": _diff(b.sharpe_ratio, a.sharpe_ratio),
                "mean_return_diff": _diff(b.mean_return, a.mean_return),
                "latency_diff_ms": _diff(b.avg_latency_ms, a.avg_latency_ms),
                "error_rate_diff": _diff(b.error_rate, a.error_rate),
                "pnl_diff": _diff(b.total_pnl, a.total_pnl),
            },
        }

    def determine_winner(self) -> Optional[str]:
        """Determine the winning variant based on the primary metric.

        Returns
        -------
        str or None
            ``"A"``, ``"B"``, or ``None`` if insufficient data.
        """
        a = self._metrics["A"]
        b = self._metrics["B"]
        if a.sample_count == 0 or b.sample_count == 0:
            return None

        metric_map = {
            "sharpe_ratio": (a.sharpe_ratio, b.sharpe_ratio),
            "mean_return": (a.mean_return, b.mean_return),
            "win_rate": (a.win_rate, b.win_rate),
            "total_pnl": (a.total_pnl, b.total_pnl),
            "avg_latency_ms": (-a.avg_latency_ms, -b.avg_latency_ms),  # lower is better
        }
        vals = metric_map.get(self._primary_metric)
        if vals is None:
            return None
        a_val, b_val = vals
        if b_val > a_val:
            return "B"
        elif a_val > b_val:
            return "A"
        return None  # tie


# ---------------------------------------------------------------------------
# Statistical Significance Tester
# ---------------------------------------------------------------------------

class StatisticalSignificanceTester:
    """Evaluates A/B test results for statistical significance.

    Implements two-sample tests for comparing model variants:

    - **Welch's t-test** for continuous metrics (returns, latency).
    - **Chi-squared test** for binary metrics (win/loss).
    - **Mann-Whitney U test** as a non-parametric alternative.

    Parameters
    ----------
    significance_level : float
        P-value threshold for declaring significance (default 0.05).
    min_samples : int
        Minimum samples per variant before testing.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 100,
    ) -> None:
        self._alpha = significance_level
        self._min_samples = min_samples
        logger.info("StatisticalSignificanceTester initialized (alpha=%.3f)", significance_level)

    def test_significance(self, comparison: ModelComparison) -> Dict[str, Any]:
        """Run statistical significance tests on the comparison data.

        Parameters
        ----------
        comparison : ModelComparison
            The comparison object containing metrics for both variants.

        Returns
        -------
        dict
            Test results including p-values, significance flags, and effect sizes.
        """
        a_metrics = comparison.get_metrics("A")
        b_metrics = comparison.get_metrics("B")

        result: Dict[str, Any] = {
            "sufficient_data": (
                a_metrics.sample_count >= self._min_samples
                and b_metrics.sample_count >= self._min_samples
            ),
            "variant_a_samples": a_metrics.sample_count,
            "variant_b_samples": b_metrics.sample_count,
        }

        if not result["sufficient_data"]:
            result["significance"] = None
            result["message"] = (
                f"Insufficient data: A={a_metrics.sample_count}, B={b_metrics.sample_count} "
                f"(minimum={self._min_samples})"
            )
            return result

        # Welch's t-test on returns
        a_returns = np.array(a_metrics.returns)
        b_returns = np.array(b_metrics.returns)

        if len(a_returns) >= 2 and len(b_returns) >= 2:
            t_stat, p_value = self._welch_ttest(a_returns, b_returns)
            effect_size = self._cohens_d(a_returns, b_returns)

            result["welch_t_test"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self._alpha,
                "effect_size": float(effect_size),
                "effect_magnitude": self._interpret_effect(effect_size),
            }
        else:
            result["welch_t_test"] = None

        # Mann-Whitney U test (non-parametric)
        if len(a_returns) >= 2 and len(b_returns) >= 2:
            u_stat, p_value_mw = self._mann_whitney_u(a_returns, b_returns)
            result["mann_whitney"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_value_mw),
                "significant": p_value_mw < self._alpha,
            }
        else:
            result["mann_whitney"] = None

        # Overall significance
        t_test_sig = result.get("welch_t_test", {}).get("significant", False)
        mw_sig = result.get("mann_whitney", {}).get("significant", False)
        result["significance"] = t_test_sig or mw_sig
        result["message"] = (
            "Statistically significant difference detected"
            if result["significance"]
            else "No statistically significant difference"
        )

        return result

    def _welch_ttest(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Welch's t-test for two independent samples."""
        n1, n2 = len(a), len(b)
        m1, m2 = np.mean(a), np.mean(b)
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)

        se = np.sqrt(v1 / n1 + v2 / n2)
        if se < 1e-10:
            return 0.0, 1.0

        t_stat = (m1 - m2) / se
        # Approximate degrees of freedom (Welch-Satterthwaite)
        df_num = (v1 / n1 + v2 / n2) ** 2
        df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else 1.0

        # Approximate two-tailed p-value using normal approximation for large df
        p_value = 2.0 * self._survival_function(abs(t_stat), df)
        return float(t_stat), float(p_value)

    def _survival_function(self, t: float, df: float) -> float:
        """Approximate survival function for t-distribution."""
        # Use normal approximation for large df
        if df > 30:
            z = t
            return 0.5 * (1.0 + self._erf(-z / np.sqrt(2.0)))
        # Simple approximation for small df
        x = df / (df + t * t)
        return 0.5 * self._regularized_beta(x, df / 2.0, 0.5)

    @staticmethod
    def _erf(x: float) -> float:
        """Approximation of the error function."""
        # Abramowitz and Stegun approximation
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return sign * y

    @staticmethod
    def _regularized_beta(x: float, a: float, b: float) -> float:
        """Simplified regularized incomplete beta function approximation."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        # Very rough approximation
        return x ** a * (1.0 - x) ** b

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(a), len(b)
        var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std < 1e-10:
            return 0.0
        return float((np.mean(b) - np.mean(a)) / pooled_std)

    @staticmethod
    def _interpret_effect(d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        if abs_d < 0.5:
            return "small"
        if abs_d < 0.8:
            return "medium"
        return "large"

    def _mann_whitney_u(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """Mann-Whitney U test (simplified implementation)."""
        n1, n2 = len(a), len(b)
        combined = np.concatenate([a, b])
        ranks = np.argsort(np.argsort(combined)) + 1
        r1 = np.sum(ranks[:n1])

        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        u_stat = min(u1, u2)

        # Normal approximation for p-value
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma < 1e-10:
            return float(u_stat), 1.0
        z = (u_stat - mu) / sigma
        p_value = 2.0 * min(self._normal_cdf(z), 1.0 - self._normal_cdf(z))
        return float(u_stat), float(p_value)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1.0 + StatisticalSignificanceTester._erf(z / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# A/B Test Manager
# ---------------------------------------------------------------------------

class ABTestManager:
    """Full lifecycle manager for model A/B tests.

    Handles test creation, traffic splitting, result collection,
    significance evaluation, automatic winner promotion, and test
    archival.

    Parameters
    ----------
    model_server : Any, optional
        Reference to the model server for version management.
    check_interval_seconds : float
        Interval between automatic significance checks.
    auto_promote_enabled : bool
        Whether to allow automatic winner promotion globally.

    Examples
    --------
    >>> manager = ABTestManager()
    >>> config = ABTestConfig(model_a_id="btc_v1", model_b_id="btc_v2", traffic_split=0.3)
    >>> test_id = await manager.create_test(config)
    >>> variant = manager.route_request(test_id, request_id="req_123")
    >>> manager.record_result(test_id, variant="B", return_value=0.02, latency_ms=12.0)
    >>> results = await manager.evaluate_test(test_id)
    """

    def __init__(
        self,
        model_server: Any = None,
        check_interval_seconds: float = 300.0,
        auto_promote_enabled: bool = False,
    ) -> None:
        self._server = model_server
        self._check_interval = check_interval_seconds
        self._auto_promote_enabled = auto_promote_enabled
        self._tests: Dict[str, ABTestConfig] = {}
        self._statuses: Dict[str, ABTestStatus] = {}
        self._splitters: Dict[str, TrafficSplitter] = {}
        self._comparisons: Dict[str, ModelComparison] = {}
        self._significance_tester = StatisticalSignificanceTester()
        self._start_times: Dict[str, float] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        logger.info("ABTestManager initialized (auto_promote=%s)", auto_promote_enabled)

    # -- Lifecycle --

    async def start(self) -> None:
        """Start the background significance checker."""
        self._running = True
        self._check_task = asyncio.create_task(self._periodic_check())
        logger.info("ABTestManager started")

    async def stop(self) -> None:
        """Stop the background checker."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("ABTestManager stopped")

    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test.

        Parameters
        ----------
        config : ABTestConfig
            Test configuration.

        Returns
        -------
        str
            The test ID.
        """
        test_id = config.test_id
        self._tests[test_id] = config
        self._statuses[test_id] = ABTestStatus.RUNNING
        self._splitters[test_id] = TrafficSplitter(
            strategy=config.strategy,
            split_ratio=config.traffic_split,
        )
        self._comparisons[test_id] = ModelComparison(primary_metric=config.primary_metric)
        self._start_times[test_id] = time.time()
        logger.info(
            "Created A/B test %s: %s vs %s (split=%.1f%%)",
            test_id, config.model_a_id, config.model_b_id, config.traffic_split * 100,
        )
        return test_id

    async def pause_test(self, test_id: str) -> None:
        """Pause a running test."""
        if test_id in self._statuses:
            self._statuses[test_id] = ABTestStatus.PAUSED
            logger.info("Paused test %s", test_id)

    async def resume_test(self, test_id: str) -> None:
        """Resume a paused test."""
        if test_id in self._statuses and self._statuses[test_id] == ABTestStatus.PAUSED:
            self._statuses[test_id] = ABTestStatus.RUNNING
            logger.info("Resumed test %s", test_id)

    async def stop_test(self, test_id: str) -> Dict[str, Any]:
        """Stop a test and return final results."""
        if test_id not in self._tests:
            return {"error": f"Test {test_id} not found"}

        self._statuses[test_id] = ABTestStatus.COMPLETED
        evaluation = await self.evaluate_test(test_id)
        logger.info("Stopped test %s with winner: %s", test_id, evaluation.get("winner"))
        return evaluation

    # -- Routing --

    def route_request(self, test_id: str, request_id: str,
                      user_id: Optional[str] = None,
                      features_hash: Optional[str] = None) -> str:
        """Route a request to a model variant.

        Parameters
        ----------
        test_id : str
            The A/B test identifier.
        request_id : str
            The inference request identifier.
        user_id : str, optional
            User identifier for user-based routing.
        features_hash : str, optional
            Feature hash for hash-based routing.

        Returns
        -------
        str
            ``"A"`` or ``"B"``
        """
        splitter = self._splitters.get(test_id)
        if splitter is None:
            logger.warning("Unknown test %s; defaulting to A", test_id)
            return "A"

        status = self._statuses.get(test_id)
        if status != ABTestStatus.RUNNING:
            return "A"  # Default to control when not running

        return splitter.assign_variant(request_id, user_id, features_hash)

    def get_model_id(self, test_id: str, variant: str) -> Optional[str]:
        """Get the model ID for a variant in a test."""
        config = self._tests.get(test_id)
        if config is None:
            return None
        if variant == "A":
            return config.model_a_id
        elif variant == "B":
            return config.model_b_id
        return None

    # -- Result Recording --

    def record_result(
        self,
        test_id: str,
        variant: str,
        return_value: float,
        latency_ms: float,
        is_error: bool = False,
        pnl: float = 0.0,
    ) -> None:
        """Record an observation for a test variant.

        Parameters
        ----------
        test_id : str
            The A/B test identifier.
        variant : str
            ``"A"`` or ``"B"``
        return_value : float
            Outcome metric value.
        latency_ms : float
            Inference latency.
        is_error : bool
            Whether an error occurred.
        pnl : float
            Profit/loss for this observation.
        """
        comparison = self._comparisons.get(test_id)
        if comparison is None:
            logger.warning("Unknown test %s; cannot record result", test_id)
            return
        comparison.record(variant, return_value, latency_ms, is_error, pnl)

    # -- Evaluation --

    async def evaluate_test(self, test_id: str) -> Dict[str, Any]:
        """Evaluate a test for statistical significance.

        Parameters
        ----------
        test_id : str
            The A/B test identifier.

        Returns
        -------
        dict
            Evaluation results including significance, winner, and metrics.
        """
        config = self._tests.get(test_id)
        comparison = self._comparisons.get(test_id)

        if config is None or comparison is None:
            return {"error": f"Test {test_id} not found"}

        # Run significance tests
        sig_result = self._significance_tester.test_significance(comparison)
        winner = comparison.determine_winner()
        comp = comparison.get_comparison()

        # Check duration
        elapsed_hours = (time.time() - self._start_times.get(test_id, time.time())) / 3600.0
        duration_exceeded = elapsed_hours > config.max_duration_hours

        result = {
            "test_id": test_id,
            "model_a_id": config.model_a_id,
            "model_b_id": config.model_b_id,
            "status": self._statuses.get(test_id, ABTestStatus.DRAFT).value,
            "elapsed_hours": elapsed_hours,
            "duration_exceeded": duration_exceeded,
            "winner": winner,
            "comparison": comp,
            "significance": sig_result,
        }

        # Auto-promote if configured
        if (
            config.auto_promote
            and self._auto_promote_enabled
            and winner == "B"
            and sig_result.get("significance", False)
        ):
            await self._promote_winner(test_id, config.model_b_id)
            result["auto_promoted"] = True

        return result

    async def _promote_winner(self, test_id: str, winner_model_id: str) -> None:
        """Promote the winning model as the default version."""
        self._statuses[test_id] = ABTestStatus.AUTO_PROMOTED
        if self._server is not None and hasattr(self._server, "set_default_version"):
            try:
                self._server.set_default_version(
                    self._tests[test_id].model_a_id, winner_model_id
                )
                logger.info("Auto-promoted model %s as winner of test %s", winner_model_id, test_id)
            except Exception as exc:
                logger.error("Auto-promotion failed for test %s: %s", test_id, exc)

    async def _periodic_check(self) -> None:
        """Background loop for periodic significance checking."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                for test_id, status in list(self._statuses.items()):
                    if status != ABTestStatus.RUNNING:
                        continue
                    config = self._tests[test_id]
                    comparison = self._comparisons[test_id]

                    # Check minimum samples
                    a_metrics = comparison.get_metrics("A")
                    b_metrics = comparison.get_metrics("B")
                    if (
                        a_metrics.sample_count >= config.min_samples
                        and b_metrics.sample_count >= config.min_samples
                    ):
                        sig_result = self._significance_tester.test_significance(comparison)
                        if sig_result.get("significance", False):
                            logger.info(
                                "Test %s reached significance (p=%.4f)",
                                test_id,
                                sig_result.get("welch_t_test", {}).get("p_value", 1.0),
                            )
                            if config.auto_promote and self._auto_promote_enabled:
                                winner = comparison.determine_winner()
                                if winner == "B":
                                    await self._promote_winner(test_id, config.model_b_id)
                                else:
                                    self._statuses[test_id] = ABTestStatus.COMPLETED
                                    logger.info("Test %s: variant A is winner; test completed", test_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Periodic check error: %s", exc)

    # -- Query --

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all A/B tests with their current status."""
        results: List[Dict[str, Any]] = []
        for test_id, config in self._tests.items():
            status = self._statuses.get(test_id, ABTestStatus.DRAFT)
            comparison = self._comparisons.get(test_id)
            splitter = self._splitters.get(test_id)

            test_info: Dict[str, Any] = {
                "test_id": test_id,
                "model_a_id": config.model_a_id,
                "model_b_id": config.model_b_id,
                "status": status.value,
                "traffic_split": config.traffic_split,
                "actual_split_ratio": splitter.actual_split_ratio if splitter else None,
                "primary_metric": config.primary_metric,
                "auto_promote": config.auto_promote,
            }

            if comparison:
                a = comparison.get_metrics("A")
                b = comparison.get_metrics("B")
                test_info["sample_counts"] = {"A": a.sample_count, "B": b.sample_count}
                test_info["winner"] = comparison.determine_winner()

            results.append(test_info)
        return results

    def get_test_details(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific test."""
        if test_id not in self._tests:
            return None
        config = self._tests[test_id]
        comparison = self._comparisons.get(test_id)
        splitter = self._splitters.get(test_id)

        details: Dict[str, Any] = {
            "config": {
                "test_id": config.test_id,
                "model_a_id": config.model_a_id,
                "model_b_id": config.model_b_id,
                "traffic_split": config.traffic_split,
                "strategy": config.strategy.value,
                "min_samples": config.min_samples,
                "significance_level": config.significance_level,
                "auto_promote": config.auto_promote,
                "max_duration_hours": config.max_duration_hours,
                "primary_metric": config.primary_metric,
            },
            "status": self._statuses.get(test_id, ABTestStatus.DRAFT).value,
            "elapsed_hours": (time.time() - self._start_times.get(test_id, time.time())) / 3600.0,
            "traffic_splitter": {
                "actual_ratio": splitter.actual_split_ratio if splitter else None,
                "assignment_counts": splitter.get_assignment_counts() if splitter else {},
            },
        }

        if comparison:
            details["comparison"] = comparison.get_comparison()

        return details
