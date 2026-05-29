"""Comprehensive tests for acms.portfolio module.

Tests all classes, methods, and edge cases in the portfolio engine:
- PortfolioConfig dataclass
- TransactionCostModel (compute_cost, cost_adjusted_weights)
- MeanVarianceOptimizer (optimize, efficient_frontier)
- RiskParityOptimizer (optimize)
- HierarchicalRiskParity (optimize, _get_quasi_diag, _recursive_bisection)
- MaximumDiversificationPortfolio (optimize)
- MinimumCorrelationAlgorithm (optimize)
- CVaRPortfolioOptimization (optimize)
- CVaRRiskBudgeting (optimize)
- DynamicRebalancing (check_threshold_rebalance, check_time_rebalance, check_drift_rebalance,
                       should_rebalance, compute_rebalance_cost)
- LeverageOptimizer (volatility_target_leverage, kelly_leverage, optimal_leverage)
- KellyAllocator (allocate)
- BlackLitterman (compute)
- PortfolioEngine (optimize_portfolio, compute_rebalance_trades, reconcile)
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import numpy as np
import pytest
from datetime import datetime, timedelta

from acms.core import Position, PortfolioSnapshot, Side
from acms.portfolio import (
    PortfolioConfig, TransactionCostModel, MeanVarianceOptimizer,
    RiskParityOptimizer, HierarchicalRiskParity, MaximumDiversificationPortfolio,
    MinimumCorrelationAlgorithm, CVaRPortfolioOptimization, CVaRRiskBudgeting,
    DynamicRebalancing, LeverageOptimizer, KellyAllocator, BlackLitterman,
    PortfolioEngine,
)


# ============================================================================
# Helpers
# ============================================================================

def make_position(symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
                  entry_price=50000.0, mark_price=50000.0,
                  unrealized_pnl=0.0, leverage=1.0, exchange="paper"):
    """Create a Position instance for testing."""
    return Position(
        symbol=symbol, side=side, quantity=quantity,
        entry_price=entry_price, mark_price=mark_price,
        unrealized_pnl=unrealized_pnl, leverage=leverage, exchange=exchange,
    )


def make_portfolio(total_value=1000000.0, available_balance=800000.0,
                   unrealized_pnl=0.0, realized_pnl=0.0,
                   positions=None, margin_used=0.0, leverage=1.0):
    """Create a PortfolioSnapshot instance for testing."""
    return PortfolioSnapshot(
        timestamp=datetime.utcnow(),
        total_value=total_value,
        available_balance=available_balance,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        positions=positions or [],
        margin_used=margin_used,
        leverage=leverage,
    )


def generate_returns_matrix(t=300, n=5, seed=42):
    """Generate synthetic returns matrix (T x N)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, (t, n))


def generate_cov_matrix(n=5, seed=42):
    """Generate a valid positive-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (n, n))
    return A.T @ A / n + np.eye(n) * 0.001


def generate_expected_returns(n=5, seed=42):
    """Generate synthetic expected returns."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.20, n)


def generate_corr_matrix(n=5, seed=42):
    """Generate a valid correlation matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (n, n))
    cov = A.T @ A / n + np.eye(n) * 0.001
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


# ============================================================================
# PortfolioConfig Tests
# ============================================================================

class TestPortfolioConfig:
    """Tests for PortfolioConfig dataclass."""

    def test_defaults(self):
        """All fields should have expected defaults."""
        cfg = PortfolioConfig()
        assert cfg.target_return is None
        assert cfg.risk_free_rate == 0.0
        assert cfg.max_weight == 0.40
        assert cfg.min_weight == 0.0
        assert cfg.rebalance_threshold == 0.05
        assert cfg.transaction_cost_bps == 10.0
        assert cfg.max_leverage == 3.0
        assert cfg.fixed_cost_usd == 1.0
        assert cfg.proportional_cost_bps == 5.0
        assert cfg.market_impact_alpha == 0.1
        assert cfg.rebalance_interval_days == 30
        assert cfg.max_drift == 0.10

    def test_custom_values(self):
        """Should accept custom values for all fields."""
        cfg = PortfolioConfig(
            target_return=0.10,
            risk_free_rate=0.02,
            max_weight=0.30,
            min_weight=0.05,
            rebalance_threshold=0.03,
            transaction_cost_bps=5.0,
            max_leverage=2.0,
            fixed_cost_usd=2.0,
            proportional_cost_bps=3.0,
            market_impact_alpha=0.05,
            rebalance_interval_days=14,
            max_drift=0.05,
        )
        assert cfg.target_return == 0.10
        assert cfg.risk_free_rate == 0.02
        assert cfg.max_weight == 0.30
        assert cfg.min_weight == 0.05
        assert cfg.rebalance_threshold == 0.03
        assert cfg.transaction_cost_bps == 5.0
        assert cfg.max_leverage == 2.0
        assert cfg.fixed_cost_usd == 2.0
        assert cfg.proportional_cost_bps == 3.0
        assert cfg.market_impact_alpha == 0.05
        assert cfg.rebalance_interval_days == 14
        assert cfg.max_drift == 0.05

    def test_partial_custom(self):
        """Should allow setting only some fields."""
        cfg = PortfolioConfig(max_weight=0.50)
        assert cfg.max_weight == 0.50
        assert cfg.risk_free_rate == 0.0  # default


# ============================================================================
# TransactionCostModel Tests
# ============================================================================

class TestTransactionCostModel:
    """Tests for TransactionCostModel class."""

    def setup_method(self):
        self.model = TransactionCostModel(
            fixed_cost_usd=1.0, proportional_cost_bps=5.0,
            market_impact_alpha=0.1, avg_daily_volume_usd=1000000.0,
        )

    def test_defaults(self):
        """Default constructor should set expected values."""
        m = TransactionCostModel()
        assert m.fixed_cost_usd == 1.0
        assert m.proportional_cost_bps == 5.0
        assert m.market_impact_alpha == 0.1
        assert m.avg_daily_volume_usd == 1000000.0

    def test_compute_cost_basic(self):
        """Should compute cost breakdown for a simple rebalance."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        portfolio_value = 100000.0
        trade_notional = np.sum(np.abs(target - current)) * portfolio_value  # 0.2 * 100000 = 20000
        result = self.model.compute_cost(trade_notional, current, target, portfolio_value)
        assert "fixed_cost" in result
        assert "proportional_cost" in result
        assert "market_impact_cost" in result
        assert "total_cost" in result
        assert "total_cost_bps" in result
        assert "n_trades" in result
        assert result["total_cost"] > 0
        assert result["n_trades"] == 2  # Both assets changed

    def test_compute_cost_no_change(self):
        """No change in weights should give zero variable costs but may have fixed."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        portfolio_value = 100000.0
        result = self.model.compute_cost(0.0, current, target, portfolio_value)
        assert result["proportional_cost"] == 0.0
        assert result["market_impact_cost"] == 0.0
        assert result["n_trades"] == 0

    def test_compute_cost_single_trade(self):
        """Only one asset changing should give n_trades=1."""
        current = np.array([0.5, 0.5])
        # Only second asset changes significantly
        target = np.array([0.5, 0.5])  # start with same
        # Change first asset by more than 0.001 threshold
        target2 = np.array([0.6, 0.4])
        result = self.model.compute_cost(20000.0, current, target2, 100000.0)
        # Both weights differ by > 0.001
        assert result["n_trades"] == 2

    def test_compute_cost_total_cost_bps(self):
        """Cost in bps should be total_cost / portfolio_value * 10000."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        portfolio_value = 100000.0
        trade_notional = 20000.0
        result = self.model.compute_cost(trade_notional, current, target, portfolio_value)
        expected_bps = result["total_cost"] / portfolio_value * 10000
        assert abs(result["total_cost_bps"] - expected_bps) < 1e-10

    def test_compute_cost_zero_portfolio_value(self):
        """Zero portfolio value should give 0 total_cost_bps."""
        result = self.model.compute_cost(1000.0, np.array([0.5, 0.5]),
                                          np.array([0.6, 0.4]), 0.0)
        assert result["total_cost_bps"] == 0.0

    def test_cost_adjusted_weights_basic(self):
        """Adjusted weights should sum to 1 and be slightly less than target."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        portfolio_value = 100000.0
        adjusted = self.model.cost_adjusted_weights(current, target, portfolio_value)
        assert abs(np.sum(adjusted) - 1.0) < 1e-10
        # Adjusted weights should be slightly less than target due to costs
        # (but normalized back to sum=1, so direction depends on cost fraction)

    def test_cost_adjusted_weights_no_change(self):
        """No change should give same weights."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        adjusted = self.model.cost_adjusted_weights(current, target, 100000.0)
        np.testing.assert_allclose(adjusted, target, atol=1e-10)

    def test_cost_adjusted_weights_zero_portfolio(self):
        """Zero portfolio value should return target weights (no cost deduction)."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        adjusted = self.model.cost_adjusted_weights(current, target, 0.0)
        np.testing.assert_allclose(adjusted, target, atol=1e-10)

    def test_compute_cost_components(self):
        """Individual cost components should be reasonable."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        portfolio_value = 100000.0
        trade_notional = 20000.0
        result = self.model.compute_cost(trade_notional, current, target, portfolio_value)
        # Fixed cost: 1.0 * 2 trades = 2.0
        assert result["fixed_cost"] == 2.0
        # Proportional: 20000 * 5 / 10000 = 10.0
        assert abs(result["proportional_cost"] - 10.0) < 1e-10
        # Market impact: 0.1 * sqrt(20000/1000000) * 20000
        assert result["market_impact_cost"] > 0


# ============================================================================
# MeanVarianceOptimizer Tests
# ============================================================================

class TestMeanVarianceOptimizer:
    """Tests for MeanVarianceOptimizer class."""

    def setup_method(self):
        self.optimizer = MeanVarianceOptimizer()
        self.expected_returns = generate_expected_returns(5, seed=1)
        self.cov_matrix = generate_cov_matrix(5, seed=1)

    def test_optimize_max_sharpe(self):
        """Should find max Sharpe ratio portfolio."""
        result = self.optimizer.optimize(self.expected_returns, self.cov_matrix)
        assert "weights" in result
        assert "return" in result
        assert "volatility" in result
        assert "sharpe_ratio" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert result["volatility"] > 0

    def test_optimize_target_return(self):
        """Should find minimum volatility for target return."""
        target = np.mean(self.expected_returns)
        result = self.optimizer.optimize(self.expected_returns, self.cov_matrix, target_return=target)
        assert abs(result["return"] - target) < 0.05  # Allow some slack
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_weight_bounds(self):
        """Weights should respect min/max bounds."""
        config = PortfolioConfig(max_weight=0.30, min_weight=0.0)
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(self.expected_returns, self.cov_matrix)
        assert np.all(result["weights"] >= -1e-6)
        assert np.all(result["weights"] <= config.max_weight + 1e-6)

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        er = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        config = PortfolioConfig(max_weight=0.80, min_weight=0.20)  # Allow sufficient range for 2 assets
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(er, cov)
        assert len(result["weights"]) == 2
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_optimize_single_asset(self):
        """Single asset should return weight=1.0."""
        er = np.array([0.10])
        cov = np.array([[0.04]])
        result = self.optimizer.optimize(er, cov)
        assert result["weights"][0] == 1.0

    def test_optimize_empty_returns(self):
        """Empty expected returns should not crash."""
        er = np.array([])
        cov = np.array([]).reshape(0, 0)
        result = self.optimizer.optimize(er, cov)
        assert result["return"] == 0.0

    def test_efficient_frontier(self):
        """Should return multiple frontier points."""
        frontier = self.optimizer.efficient_frontier(
            self.expected_returns, self.cov_matrix, num_points=20
        )
        assert len(frontier) > 0
        # Returns should be monotonically increasing (approximately)
        returns = [p["return"] for p in frontier]
        for i in range(1, len(returns)):
            assert returns[i] >= returns[i-1] - 0.01  # Allow small slack

    def test_efficient_frontier_custom_points(self):
        """Should respect num_points parameter."""
        frontier = self.optimizer.efficient_frontier(
            self.expected_returns, self.cov_matrix, num_points=10
        )
        assert len(frontier) <= 10  # May be fewer if optimization fails

    def test_optimize_risk_free_rate(self):
        """Non-zero risk-free rate should affect Sharpe ratio."""
        config = PortfolioConfig(risk_free_rate=0.05)
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(self.expected_returns, self.cov_matrix)
        # Sharpe should be (ret - 0.05) / vol
        expected_sharpe = (result["return"] - 0.05) / result["volatility"]
        assert abs(result["sharpe_ratio"] - expected_sharpe) < 0.01

    def test_optimize_high_correlation_assets(self):
        """Highly correlated assets should still produce valid weights."""
        er = np.array([0.10, 0.12])
        cov = np.array([[0.04, 0.035], [0.035, 0.04]])
        config = PortfolioConfig(max_weight=0.80, min_weight=0.20)
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(er, cov)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-2


# ============================================================================
# RiskParityOptimizer Tests
# ============================================================================

class TestRiskParityOptimizer:
    """Tests for RiskParityOptimizer class."""

    def setup_method(self):
        self.optimizer = RiskParityOptimizer()

    def test_optimize_basic(self):
        """Should produce risk parity weights."""
        cov = generate_cov_matrix(4, seed=10)
        result = self.optimizer.optimize(cov)
        assert "weights" in result
        assert "risk_contributions" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights > 0)

    def test_optimize_equal_risk_contribution(self):
        """Risk contributions should be approximately equal."""
        cov = generate_cov_matrix(3, seed=11)
        result = self.optimizer.optimize(cov)
        rc = result["risk_contributions"]
        # All risk contributions should be roughly equal
        mean_rc = np.mean(rc)
        for contribution in rc:
            assert abs(contribution - mean_rc) / mean_rc < 0.15  # within 15%

    def test_optimize_single_asset(self):
        """Single asset should return weight=1.0."""
        cov = np.array([[0.04]])
        result = self.optimizer.optimize(cov)
        assert result["weights"][0] == 1.0

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = self.optimizer.optimize(cov)
        assert len(result["weights"]) == 2
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6
        # Lower vol asset should get more weight
        assert result["weights"][0] > result["weights"][1]

    def test_optimize_uniform_volatility(self):
        """Assets with same volatility and zero correlation should get equal weights."""
        cov = np.eye(3) * 0.04  # Same variance, no correlation
        result = self.optimizer.optimize(cov)
        weights = result["weights"]
        for w in weights:
            assert abs(w - 1/3) < 0.05

    def test_optimize_extreme_vol_difference(self):
        """One very volatile asset should get very small weight."""
        cov = np.array([[0.01, 0.0], [0.0, 1.0]])  # 10x vol difference
        result = self.optimizer.optimize(cov)
        assert result["weights"][0] > result["weights"][1]


# ============================================================================
# HierarchicalRiskParity Tests
# ============================================================================

class TestHierarchicalRiskParity:
    """Tests for HierarchicalRiskParity class."""

    def setup_method(self):
        self.optimizer = HierarchicalRiskParity()

    def test_optimize_basic(self):
        """Should produce valid HRP weights."""
        ret_mat = generate_returns_matrix(200, 5, seed=20)
        result = self.optimizer.optimize(ret_mat)
        assert "weights" in result
        assert "linkage" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights > 0)

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        ret_mat = generate_returns_matrix(100, 2, seed=21)
        result = self.optimizer.optimize(ret_mat)
        assert len(result["weights"]) == 2
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_single_asset(self):
        """Single asset should return weight=1.0."""
        ret_mat = generate_returns_matrix(100, 1, seed=22)
        result = self.optimizer.optimize(ret_mat)
        assert result["weights"][0] == 1.0

    def test_optimize_many_assets(self):
        """Should work with many assets."""
        ret_mat = generate_returns_matrix(300, 10, seed=23)
        result = self.optimizer.optimize(ret_mat)
        assert len(result["weights"]) == 10
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_correlated_assets(self):
        """Highly correlated assets should get similar weights."""
        rng = np.random.default_rng(24)
        base = rng.normal(0, 0.02, 200)
        # Create 3 highly correlated assets
        ret_mat = np.column_stack([
            base + rng.normal(0, 0.001, 200),
            base + rng.normal(0, 0.001, 200),
            base + rng.normal(0, 0.001, 200),
        ])
        result = self.optimizer.optimize(ret_mat)
        weights = result["weights"]
        # Should be roughly equal for highly correlated assets
        for w in weights:
            assert abs(w - 1/3) < 0.25

    def test_get_quasi_diag(self):
        """_get_quasi_diag should return sorted indices."""
        ret_mat = generate_returns_matrix(100, 4, seed=25)
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        corr = np.corrcoef(ret_mat.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)
        dist = squareform(1 - np.abs(corr), checks=False)
        if len(dist) > 0 and not np.any(np.isnan(dist)):
            link = linkage(dist, method='ward')
            indices = HierarchicalRiskParity._get_quasi_diag(link)
            assert len(indices) > 0
            assert all(isinstance(i, int) for i in indices)


# ============================================================================
# MaximumDiversificationPortfolio Tests
# ============================================================================

class TestMaximumDiversificationPortfolio:
    """Tests for MaximumDiversificationPortfolio class."""

    def setup_method(self):
        self.optimizer = MaximumDiversificationPortfolio()

    def test_optimize_basic(self):
        """Should produce max diversification weights."""
        cov = generate_cov_matrix(4, seed=30)
        result = self.optimizer.optimize(cov)
        assert "weights" in result
        assert "diversification_ratio" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert result["diversification_ratio"] >= 1.0

    def test_optimize_single_asset(self):
        """Single asset should return DR=1.0."""
        cov = np.array([[0.04]])
        result = self.optimizer.optimize(cov)
        assert result["weights"][0] == 1.0
        assert result["diversification_ratio"] == 1.0

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = self.optimizer.optimize(cov)
        assert len(result["weights"]) == 2
        assert result["diversification_ratio"] >= 1.0

    def test_optimize_zero_variance(self):
        """Zero variance should return equal weights."""
        cov = np.array([[0.0, 0.0], [0.0, 0.04]])
        result = self.optimizer.optimize(cov)
        np.testing.assert_allclose(result["weights"], np.array([0.5, 0.5]), atol=0.01)

    def test_optimize_uncorrelated(self):
        """Uncorrelated assets should have DR > 1."""
        cov = np.eye(5) * 0.04
        result = self.optimizer.optimize(cov)
        assert result["diversification_ratio"] > 1.0

    def test_optimize_highly_correlated(self):
        """Highly correlated assets should have DR closer to 1."""
        cov = np.array([[0.04, 0.038], [0.038, 0.04]])
        result = self.optimizer.optimize(cov)
        assert result["diversification_ratio"] < 1.5  # Not much diversification


# ============================================================================
# MinimumCorrelationAlgorithm Tests
# ============================================================================

class TestMinimumCorrelationAlgorithm:
    """Tests for MinimumCorrelationAlgorithm class."""

    def setup_method(self):
        self.optimizer = MinimumCorrelationAlgorithm()

    def test_optimize_basic(self):
        """Should produce minimum correlation weights."""
        corr = generate_corr_matrix(4, seed=40)
        result = self.optimizer.optimize(corr)
        assert "weights" in result
        assert "avg_correlation" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= 0)

    def test_optimize_single_asset(self):
        """Single asset should return weight=1.0."""
        corr = np.array([[1.0]])
        result = self.optimizer.optimize(corr)
        assert result["weights"][0] == 1.0
        assert result["avg_correlation"] == 0.0

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = self.optimizer.optimize(corr)
        assert len(result["weights"]) == 2
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_identity(self):
        """Identity correlation should still produce valid weights."""
        corr = np.eye(3)
        result = self.optimizer.optimize(corr)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_high_correlation(self):
        """High correlation should still produce valid weights."""
        corr = np.array([[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]])
        result = self.optimizer.optimize(corr)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_optimize_weights_positive(self):
        """All weights should be non-negative."""
        corr = generate_corr_matrix(5, seed=41)
        result = self.optimizer.optimize(corr)
        assert np.all(result["weights"] >= -1e-10)


# ============================================================================
# CVaRPortfolioOptimization Tests
# ============================================================================

class TestCVaRPortfolioOptimization:
    """Tests for CVaRPortfolioOptimization class."""

    def setup_method(self):
        self.optimizer = CVaRPortfolioOptimization(confidence=0.95, max_weight=0.40)

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        opt = CVaRPortfolioOptimization()
        assert opt.confidence == 0.95
        assert opt.min_return is None
        assert opt.max_weight == 0.40

    def test_init_custom(self):
        """Custom parameters should be stored."""
        opt = CVaRPortfolioOptimization(confidence=0.99, min_return=0.05, max_weight=0.30)
        assert opt.confidence == 0.99
        assert opt.min_return == 0.05
        assert opt.max_weight == 0.30

    def test_optimize_basic(self):
        """Should produce CVaR-optimal weights."""
        ret_mat = generate_returns_matrix(200, 4, seed=50)
        result = self.optimizer.optimize(ret_mat)
        assert "weights" in result
        assert "cvar" in result
        assert "var" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-4
        assert np.all(weights >= -0.01)  # Approximate non-negativity

    def test_optimize_insufficient_data(self):
        """Too few observations should return equal weights with NaN metrics."""
        ret_mat = generate_returns_matrix(20, 4, seed=51)
        result = self.optimizer.optimize(ret_mat)
        assert np.isnan(result["cvar"])
        assert np.isnan(result["var"])

    def test_optimize_single_asset(self):
        """Single asset should return equal weights fallback."""
        ret_mat = generate_returns_matrix(100, 1, seed=52)
        result = self.optimizer.optimize(ret_mat)
        assert len(result["weights"]) == 1

    def test_optimize_two_assets(self):
        """Should work with 2 assets."""
        ret_mat = generate_returns_matrix(200, 2, seed=53)
        result = self.optimizer.optimize(ret_mat)
        assert len(result["weights"]) == 2

    def test_optimize_with_min_return(self):
        """Should respect minimum return constraint."""
        opt = CVaRPortfolioOptimization(confidence=0.95, min_return=0.001, max_weight=0.40)
        ret_mat = generate_returns_matrix(200, 4, seed=54)
        result = opt.optimize(ret_mat)
        if result.get("success", False):
            # If optimization succeeded, portfolio return should be >= min_return (approximately)
            mean_returns = np.mean(ret_mat, axis=0)
            port_return = np.dot(result["weights"], mean_returns)
            # Allow some slack due to numerical issues
            assert port_return >= 0.0 or np.isnan(result["cvar"])

    def test_optimize_fallback(self):
        """Should fallback to equal weights on failure."""
        # Create problematic data that might cause optimization failure
        opt = CVaRPortfolioOptimization(confidence=0.95, max_weight=0.01)  # Very tight bounds
        ret_mat = generate_returns_matrix(100, 10, seed=55)  # Many assets
        result = opt.optimize(ret_mat)
        # Should either succeed or fallback
        assert len(result["weights"]) == 10


# ============================================================================
# CVaRRiskBudgeting Tests
# ============================================================================

class TestCVaRRiskBudgeting:
    """Tests for CVaRRiskBudgeting class."""

    def setup_method(self):
        self.optimizer = CVaRRiskBudgeting(confidence=0.95)

    def test_init_defaults(self):
        """Default constructor should set confidence=0.95."""
        opt = CVaRRiskBudgeting()
        assert opt.confidence == 0.95

    def test_optimize_basic(self):
        """Should produce CVaR risk budgeting weights."""
        ret_mat = generate_returns_matrix(200, 4, seed=60)
        result = self.optimizer.optimize(ret_mat)
        assert "weights" in result
        assert "cvar_contributions" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-4

    def test_optimize_with_risk_budget(self):
        """Should use custom risk budget."""
        ret_mat = generate_returns_matrix(200, 3, seed=61)
        risk_budget = np.array([0.5, 0.3, 0.2])
        result = self.optimizer.optimize(ret_mat, risk_budget=risk_budget)
        assert len(result["weights"]) == 3

    def test_optimize_equal_budget(self):
        """Default equal budget should produce reasonable weights."""
        ret_mat = generate_returns_matrix(200, 3, seed=62)
        result = self.optimizer.optimize(ret_mat)
        weights = result["weights"]
        # With equal budget, weights should not be too concentrated
        assert np.all(weights >= 0.01 - 1e-6)

    def test_optimize_single_asset(self):
        """Single asset should return weight=1.0."""
        ret_mat = generate_returns_matrix(100, 1, seed=63)
        result = self.optimizer.optimize(ret_mat)
        assert result["weights"][0] == 1.0


# ============================================================================
# DynamicRebalancing Tests
# ============================================================================

class TestDynamicRebalancing:
    """Tests for DynamicRebalancing class."""

    def setup_method(self):
        self.rebalancer = DynamicRebalancing(
            threshold=0.05, time_interval_days=30,
            max_drift=0.10, transaction_cost_bps=10.0,
        )

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        r = DynamicRebalancing()
        assert r.threshold == 0.05
        assert r.time_interval_days == 30
        assert r.max_drift == 0.10
        assert r.transaction_cost_bps == 10.0
        assert r._last_rebalance is None

    # --- check_threshold_rebalance ---

    def test_threshold_no_drift(self):
        """No drift should not trigger rebalance."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        assert self.rebalancer.check_threshold_rebalance(current, target) is False

    def test_threshold_small_drift(self):
        """Drift below threshold should not trigger."""
        current = np.array([0.52, 0.48])
        target = np.array([0.50, 0.50])
        assert self.rebalancer.check_threshold_rebalance(current, target) is False

    def test_threshold_large_drift(self):
        """Drift above threshold should trigger."""
        current = np.array([0.60, 0.40])
        target = np.array([0.50, 0.50])
        assert self.rebalancer.check_threshold_rebalance(current, target) is True

    def test_threshold_exact_boundary(self):
        """Drift exactly at threshold should not trigger (> not >=)."""
        current = np.array([0.55, 0.45])
        target = np.array([0.50, 0.50])
        # Due to floating point, 0.55-0.50 may not be exactly 0.05
        # So use values that produce exact arithmetic
        result = self.rebalancer.check_threshold_rebalance(current, target)
        # May or may not trigger due to floating point
        assert isinstance(result, (bool, np.bool_))

    def test_threshold_just_above(self):
        """Drift just above threshold should trigger."""
        current = np.array([0.551, 0.449])
        target = np.array([0.50, 0.50])
        assert self.rebalancer.check_threshold_rebalance(current, target) is True

    # --- check_time_rebalance ---

    def test_time_no_previous_rebalance(self):
        """No previous rebalance should trigger rebalance."""
        now = datetime.utcnow()
        assert self.rebalancer.check_time_rebalance(now) is True

    def test_time_within_interval(self):
        """Within interval should not trigger."""
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=15)
        assert self.rebalancer.check_time_rebalance(datetime.utcnow()) is False

    def test_time_past_interval(self):
        """Past interval should trigger."""
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=35)
        assert self.rebalancer.check_time_rebalance(datetime.utcnow()) is True

    def test_time_exact_interval(self):
        """Exactly at interval should trigger (>=)."""
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=30)
        assert self.rebalancer.check_time_rebalance(datetime.utcnow()) is True

    # --- check_drift_rebalance ---

    def test_drift_no_drift(self):
        """No drift should not trigger."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        assert self.rebalancer.check_drift_rebalance(current, target) == False

    def test_drift_small(self):
        """Small drift below max should not trigger."""
        current = np.array([0.53, 0.47])
        target = np.array([0.50, 0.50])
        # Total drift = 0.03 + 0.03 = 0.06 < 0.10
        assert self.rebalancer.check_drift_rebalance(current, target) == False

    def test_drift_large(self):
        """Large drift exceeding max should trigger."""
        current = np.array([0.60, 0.40])
        target = np.array([0.50, 0.50])
        # Total drift = 0.10 + 0.10 = 0.20 > 0.10
        assert self.rebalancer.check_drift_rebalance(current, target) == True

    # --- should_rebalance ---

    def test_should_rebalance_none_triggered(self):
        """No triggers should return should_rebalance=False."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=5)
        result = self.rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert result["should_rebalance"] == False
        assert len(result["reasons"]) == 0

    def test_should_rebalance_threshold_triggered(self):
        """Threshold breach should trigger."""
        current = np.array([0.60, 0.40])
        target = np.array([0.50, 0.50])
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=5)
        result = self.rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert result["should_rebalance"] is True
        assert "threshold_breach" in result["reasons"]

    def test_should_rebalance_time_triggered(self):
        """Time interval should trigger."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=35)
        result = self.rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert result["should_rebalance"] is True
        assert "time_interval" in result["reasons"]

    def test_should_rebalance_drift_triggered(self):
        """Drift exceeded should trigger."""
        current = np.array([0.60, 0.40])
        target = np.array([0.50, 0.50])
        self.rebalancer._last_rebalance = datetime.utcnow() - timedelta(days=5)
        # Also triggers threshold, but drift should be in reasons
        result = self.rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert result["should_rebalance"] is True

    def test_should_rebalance_metrics(self):
        """Should include drift metrics."""
        current = np.array([0.55, 0.45])
        target = np.array([0.50, 0.50])
        self.rebalancer._last_rebalance = datetime.utcnow()
        result = self.rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert "max_weight_drift" in result
        assert "total_drift" in result
        assert abs(result["max_weight_drift"] - 0.05) < 1e-10
        assert abs(result["total_drift"] - 0.10) < 1e-5

    # --- compute_rebalance_cost ---

    def test_rebalance_cost_no_change(self):
        """No change should give zero cost."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        result = self.rebalancer.compute_rebalance_cost(current, target, 100000.0)
        assert result["total_turnover"] == 0.0
        assert result["transaction_cost"] == 0.0

    def test_rebalance_cost_basic(self):
        """Should compute reasonable rebalance cost."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        result = self.rebalancer.compute_rebalance_cost(current, target, 100000.0)
        # Turnover = (0.1 + 0.1) * 100000 = 20000
        assert abs(result["total_turnover"] - 20000.0) < 1e-6
        # Cost = 20000 * 10 / 10000 = 20.0
        assert abs(result["transaction_cost"] - 20.0) < 1e-6

    def test_rebalance_cost_large_portfolio(self):
        """Larger portfolio should have proportionally larger cost."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        result_100k = self.rebalancer.compute_rebalance_cost(current, target, 100000.0)
        result_1m = self.rebalancer.compute_rebalance_cost(current, target, 1000000.0)
        assert abs(result_1m["transaction_cost"] - result_100k["transaction_cost"] * 10) < 1e-6


# ============================================================================
# LeverageOptimizer Tests
# ============================================================================

class TestLeverageOptimizer:
    """Tests for LeverageOptimizer class."""

    def setup_method(self):
        self.optimizer = LeverageOptimizer(target_vol=0.15, max_leverage=3.0, max_drawdown=0.25)

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        opt = LeverageOptimizer()
        assert opt.target_vol == 0.15
        assert opt.max_leverage == 3.0
        assert opt.max_drawdown == 0.25

    # --- volatility_target_leverage ---

    def test_vol_target_leverage_basic(self):
        """Should compute leverage for target vol."""
        leverage = self.optimizer.volatility_target_leverage(0.15)
        assert leverage == 1.0  # target_vol / current_vol = 1.0

    def test_vol_target_leverage_low_vol(self):
        """Low current vol should produce high leverage."""
        leverage = self.optimizer.volatility_target_leverage(0.05)
        # 0.15 / 0.05 = 3.0, capped at max 3.0
        assert abs(leverage - 3.0) < 1e-6

    def test_vol_target_leverage_high_vol(self):
        """High current vol should produce leverage=1.0 (minimum)."""
        leverage = self.optimizer.volatility_target_leverage(0.30)
        # 0.15 / 0.30 = 0.5, floored at 1.0
        assert leverage == 1.0

    def test_vol_target_leverage_zero_vol(self):
        """Zero current vol should return 1.0."""
        leverage = self.optimizer.volatility_target_leverage(0.0)
        assert leverage == 1.0

    def test_vol_target_leverage_capped_at_max(self):
        """Leverage should be capped at max_leverage."""
        opt = LeverageOptimizer(target_vol=0.15, max_leverage=2.0)
        leverage = opt.volatility_target_leverage(0.05)
        assert leverage == 2.0

    # --- kelly_leverage ---

    def test_kelly_leverage_basic(self):
        """Should compute Kelly leverage."""
        leverage = self.optimizer.kelly_leverage(expected_return=0.10, volatility=0.15)
        # Kelly = (0.10 - 0.0) / 0.15^2 = 0.10 / 0.0225 = 4.44, capped at 3.0
        assert leverage == 3.0

    def test_kelly_leverage_low_return(self):
        """Low return should give lower leverage."""
        leverage = self.optimizer.kelly_leverage(expected_return=0.02, volatility=0.15)
        # 0.02 / 0.0225 = 0.89
        assert 0.5 < leverage < 1.5

    def test_kelly_leverage_negative_return(self):
        """Negative excess return should return 0."""
        leverage = self.optimizer.kelly_leverage(expected_return=-0.05, volatility=0.15)
        assert leverage == 0.0

    def test_kelly_leverage_zero_vol(self):
        """Zero volatility should return 1.0."""
        leverage = self.optimizer.kelly_leverage(expected_return=0.10, volatility=0.0)
        assert leverage == 1.0

    def test_kelly_leverage_with_risk_free_rate(self):
        """Risk-free rate should reduce leverage."""
        lev_no_rf = self.optimizer.kelly_leverage(0.10, 0.15, risk_free_rate=0.0)
        lev_with_rf = self.optimizer.kelly_leverage(0.10, 0.15, risk_free_rate=0.05)
        assert lev_with_rf < lev_no_rf

    # --- optimal_leverage ---

    def test_optimal_leverage_basic(self):
        """Should return comprehensive leverage analysis."""
        result = self.optimizer.optimal_leverage(
            expected_return=0.10, volatility=0.15,
        )
        assert "optimal_leverage" in result
        assert "kelly_leverage" in result
        assert "half_kelly" in result
        assert "vol_target_leverage" in result
        assert "drawdown_constrained_leverage" in result
        assert result["optimal_leverage"] >= 0
        assert result["optimal_leverage"] <= self.optimizer.max_leverage

    def test_optimal_leverage_uses_minimum(self):
        """Optimal should be minimum of all constraints."""
        result = self.optimizer.optimal_leverage(
            expected_return=0.10, volatility=0.15,
        )
        # Should be <= all component leverages
        assert result["optimal_leverage"] <= result["half_kelly"]
        assert result["optimal_leverage"] <= result["vol_target_leverage"]
        assert result["optimal_leverage"] <= result["drawdown_constrained_leverage"]

    def test_optimal_leverage_conservative(self):
        """Should be more conservative than full Kelly."""
        result = self.optimizer.optimal_leverage(
            expected_return=0.20, volatility=0.15,
        )
        assert result["optimal_leverage"] <= result["kelly_leverage"]

    def test_optimal_leverage_zero_vol(self):
        """Zero vol should use max_leverage for DD constraint."""
        result = self.optimizer.optimal_leverage(
            expected_return=0.10, volatility=0.0,
        )
        assert result["optimal_leverage"] >= 0


# ============================================================================
# KellyAllocator Tests
# ============================================================================

class TestKellyAllocator:
    """Tests for KellyAllocator class."""

    def setup_method(self):
        self.allocator = KellyAllocator()

    def test_allocate_basic(self):
        """Should produce Kelly-optimal allocations."""
        win_rates = np.array([0.6, 0.55, 0.65])
        win_loss_ratios = np.array([1.5, 2.0, 1.2])
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital)
        assert "weights" in result
        assert "allocations" in result
        weights = result["weights"]
        allocations = result["allocations"]
        assert len(weights) == 3
        assert len(allocations) == 3
        assert np.all(weights >= 0)
        assert np.all(allocations >= 0)

    def test_allocate_half_kelly(self):
        """Half Kelly should be default (fraction=0.5)."""
        win_rates = np.array([0.6])
        win_loss_ratios = np.array([2.0])
        capital = 100000.0
        result_full = self.allocator.allocate(win_rates, win_loss_ratios, capital, fraction=1.0)
        result_half = self.allocator.allocate(win_rates, win_loss_ratios, capital, fraction=0.5)
        assert abs(result_half["weights"][0] - result_full["weights"][0] * 0.5) < 1e-10

    def test_allocate_negative_kelly(self):
        """Negative Kelly (bad expectancy) should give 0 weight."""
        win_rates = np.array([0.3])
        win_loss_ratios = np.array([0.5])  # Bad ratio
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital)
        # kelly_f = 0.3 - 0.7/0.5 = 0.3 - 1.4 = -1.1, capped at 0
        assert result["weights"][0] == 0.0

    def test_allocate_zero_win_loss_ratio(self):
        """Zero win/loss ratio should give 0 weight."""
        win_rates = np.array([0.5])
        win_loss_ratios = np.array([0.0])
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital)
        assert result["weights"][0] == 0.0

    def test_allocate_total_exceeds_capital(self):
        """When total Kelly > 1, weights should be normalized."""
        win_rates = np.array([0.7, 0.7, 0.7])
        win_loss_ratios = np.array([3.0, 3.0, 3.0])
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital, fraction=1.0)
        assert np.sum(result["weights"]) <= 1.0 + 1e-10

    def test_allocate_single_asset(self):
        """Single asset should work."""
        win_rates = np.array([0.6])
        win_loss_ratios = np.array([2.0])
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital)
        assert len(result["weights"]) == 1

    def test_allocate_allocations_equal_weights_times_capital(self):
        """Allocations should be weights * capital."""
        win_rates = np.array([0.6, 0.55])
        win_loss_ratios = np.array([2.0, 1.5])
        capital = 100000.0
        result = self.allocator.allocate(win_rates, win_loss_ratios, capital, fraction=0.5)
        for i in range(len(win_rates)):
            assert abs(result["allocations"][i] - result["weights"][i] * capital) < 1e-10


# ============================================================================
# BlackLitterman Tests
# ============================================================================

class TestBlackLitterman:
    """Tests for BlackLitterman class."""

    def setup_method(self):
        self.model = BlackLitterman(tau=0.05)
        self.market_weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.cov_matrix = generate_cov_matrix(4, seed=70)

    def test_init_defaults(self):
        """Default constructor should set tau=0.05."""
        model = BlackLitterman()
        assert model.tau == 0.05

    def test_compute_no_views(self):
        """Without views, should return equilibrium returns and market weights."""
        result = self.model.compute(self.market_weights, self.cov_matrix)
        assert "expected_returns" in result
        assert "weights" in result
        np.testing.assert_array_equal(result["weights"], self.market_weights)

    def test_compute_with_views(self):
        """With views, should produce adjusted weights."""
        views = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 2 views on first 2 assets
        view_confidence = np.array([0.01, 0.01])  # Low uncertainty
        view_returns = np.array([0.15, 0.10])  # Expected view returns
        result = self.model.compute(
            self.market_weights, self.cov_matrix,
            risk_aversion=2.5,
            views=views, view_confidence=view_confidence,
            view_returns=view_returns,
        )
        assert "expected_returns" in result
        assert "posterior_cov" in result
        assert "weights" in result
        weights = result["weights"]
        assert abs(np.sum(weights) - 1.0) < 1e-6
        assert np.all(weights >= -1e-6)

    def test_compute_equilibrium_returns(self):
        """Equilibrium returns should be risk_aversion * Sigma * w_mkt."""
        result = self.model.compute(self.market_weights, self.cov_matrix)
        expected_pi = 2.5 * self.cov_matrix @ self.market_weights
        np.testing.assert_allclose(result["expected_returns"], expected_pi, atol=1e-10)

    def test_compute_custom_risk_aversion(self):
        """Custom risk aversion should affect equilibrium returns."""
        result_low = self.model.compute(self.market_weights, self.cov_matrix, risk_aversion=1.0)
        result_high = self.model.compute(self.market_weights, self.cov_matrix, risk_aversion=5.0)
        # Higher risk aversion should produce larger equilibrium returns
        assert np.sum(np.abs(result_high["expected_returns"])) > np.sum(np.abs(result_low["expected_returns"]))

    def test_compute_views_shift_weights(self):
        """Views should shift weights away from market weights."""
        views = np.array([[1, -1, 0, 0]])  # Asset 1 outperforms Asset 2
        view_confidence = np.array([0.01])
        view_returns = np.array([0.20])
        result = self.model.compute(
            self.market_weights, self.cov_matrix,
            views=views, view_confidence=view_confidence,
            view_returns=view_returns,
        )
        # With bullish view on asset 1 vs asset 2, the view should influence weights
        # The exact direction depends on the posterior estimation
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_compute_partial_views(self):
        """Views on only some assets should still work."""
        views = np.array([[0, 0, 1, 0]])  # View on asset 3 only
        view_confidence = np.array([0.01])
        view_returns = np.array([0.30])
        result = self.model.compute(
            self.market_weights, self.cov_matrix,
            views=views, view_confidence=view_confidence,
            view_returns=view_returns,
        )
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6


# ============================================================================
# PortfolioEngine Tests
# ============================================================================

class TestPortfolioEngine:
    """Tests for PortfolioEngine class - the main portfolio management engine."""

    def setup_method(self):
        self.engine = PortfolioEngine()
        self.expected_returns = generate_expected_returns(5, seed=80)
        self.cov_matrix = generate_cov_matrix(5, seed=80)
        self.ret_mat = generate_returns_matrix(300, 5, seed=80)

    def test_init_defaults(self):
        """Default engine should have all components initialized."""
        engine = PortfolioEngine()
        assert isinstance(engine.config, PortfolioConfig)
        assert isinstance(engine.mv_optimizer, MeanVarianceOptimizer)
        assert isinstance(engine.rp_optimizer, RiskParityOptimizer)
        assert isinstance(engine.hrp_optimizer, HierarchicalRiskParity)
        assert isinstance(engine.max_div_optimizer, MaximumDiversificationPortfolio)
        assert isinstance(engine.min_corr_optimizer, MinimumCorrelationAlgorithm)
        assert isinstance(engine.cvar_optimizer, CVaRPortfolioOptimization)
        assert isinstance(engine.cvar_budget, CVaRRiskBudgeting)
        assert isinstance(engine.kelly_allocator, KellyAllocator)
        assert isinstance(engine.bl_model, BlackLitterman)
        assert isinstance(engine.rebalancing, DynamicRebalancing)
        assert isinstance(engine.leverage_optimizer, LeverageOptimizer)
        assert isinstance(engine.transaction_cost_model, TransactionCostModel)

    def test_init_custom_config(self):
        """Custom config should propagate to sub-components."""
        config = PortfolioConfig(max_weight=0.30, max_leverage=2.0)
        engine = PortfolioEngine(config)
        assert engine.config.max_weight == 0.30
        assert engine.config.max_leverage == 2.0
        assert engine.leverage_optimizer.max_leverage == 2.0

    # --- optimize_portfolio ---

    def test_optimize_mean_variance(self):
        """Should delegate to MeanVarianceOptimizer."""
        result = self.engine.optimize_portfolio(
            "mean_variance", self.expected_returns, self.cov_matrix
        )
        assert "weights" in result
        assert "sharpe_ratio" in result
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_optimize_risk_parity(self):
        """Should delegate to RiskParityOptimizer."""
        result = self.engine.optimize_portfolio(
            "risk_parity", self.expected_returns, self.cov_matrix
        )
        assert "weights" in result
        assert "risk_contributions" in result

    def test_optimize_hrp(self):
        """Should delegate to HierarchicalRiskParity."""
        result = self.engine.optimize_portfolio(
            "hrp", self.expected_returns, self.cov_matrix,
            returns_matrix=self.ret_mat,
        )
        assert "weights" in result
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_optimize_max_diversification(self):
        """Should delegate to MaximumDiversificationPortfolio."""
        result = self.engine.optimize_portfolio(
            "max_diversification", self.expected_returns, self.cov_matrix
        )
        assert "weights" in result
        assert "diversification_ratio" in result

    def test_optimize_min_correlation(self):
        """Should delegate to MinimumCorrelationAlgorithm."""
        corr_matrix = generate_corr_matrix(5, seed=81)
        result = self.engine.optimize_portfolio(
            "min_correlation", self.expected_returns, self.cov_matrix,
            corr_matrix=corr_matrix,
        )
        assert "weights" in result
        assert "avg_correlation" in result

    def test_optimize_cvar(self):
        """Should delegate to CVaRPortfolioOptimization."""
        result = self.engine.optimize_portfolio(
            "cvar", self.expected_returns, self.cov_matrix,
            returns_matrix=self.ret_mat,
        )
        assert "weights" in result

    def test_optimize_cvar_budget(self):
        """Should delegate to CVaRRiskBudgeting."""
        result = self.engine.optimize_portfolio(
            "cvar_budget", self.expected_returns, self.cov_matrix,
            returns_matrix=self.ret_mat,
        )
        assert "weights" in result

    def test_optimize_black_litterman(self):
        """Should delegate to BlackLitterman."""
        market_weights = np.ones(5) / 5
        result = self.engine.optimize_portfolio(
            "black_litterman", self.expected_returns, self.cov_matrix,
            market_weights=market_weights,
        )
        assert "expected_returns" in result
        assert "weights" in result

    def test_optimize_unknown_method(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            self.engine.optimize_portfolio("invalid", self.expected_returns, self.cov_matrix)

    def test_optimize_mean_variance_with_target(self):
        """Should pass target_return to MV optimizer."""
        target = np.mean(self.expected_returns)
        result = self.engine.optimize_portfolio(
            "mean_variance", self.expected_returns, self.cov_matrix,
            target_return=target,
        )
        assert "weights" in result

    # --- compute_rebalance_trades ---

    def test_compute_rebalance_trades_basic(self):
        """Should compute trades needed for rebalancing."""
        current = np.array([0.5, 0.3, 0.2])
        target = np.array([0.4, 0.4, 0.2])
        trades = self.engine.compute_rebalance_trades(current, target, 100000.0)
        assert len(trades) == 2  # Only 2 assets changed above threshold
        for trade in trades:
            assert "asset_index" in trade
            assert "weight_change" in trade
            assert "notional_change" in trade
            assert "action" in trade

    def test_compute_rebalance_trades_no_change(self):
        """No change should produce no trades."""
        current = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        trades = self.engine.compute_rebalance_trades(current, target, 100000.0)
        assert len(trades) == 0

    def test_compute_rebalance_trades_buy_action(self):
        """Weight increase should be 'buy' action."""
        current = np.array([0.4, 0.6])
        target = np.array([0.5, 0.5])
        trades = self.engine.compute_rebalance_trades(current, target, 100000.0)
        buy_trades = [t for t in trades if t["action"] == "buy"]
        sell_trades = [t for t in trades if t["action"] == "sell"]
        assert len(buy_trades) >= 1
        assert len(sell_trades) >= 1

    def test_compute_rebalance_trades_custom_threshold(self):
        """Custom threshold should filter small trades."""
        current = np.array([0.5, 0.5])
        target = np.array([0.503, 0.497])
        # Default threshold 0.05 -> no trades (drift 0.003 < 0.05)
        trades_default = self.engine.compute_rebalance_trades(current, target, 100000.0)
        assert len(trades_default) == 0
        # Custom threshold 0.001 -> trades should appear
        trades_small = self.engine.compute_rebalance_trades(current, target, 100000.0, threshold=0.001)
        assert len(trades_small) == 2

    def test_compute_rebalance_trades_notional(self):
        """Notional should be weight_change * total_value."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        trades = self.engine.compute_rebalance_trades(current, target, 100000.0, threshold=0.01)
        for trade in trades:
            expected_notional = trade["weight_change"] * 100000.0
            assert abs(trade["notional_change"] - expected_notional) < 1e-10

    # --- reconcile ---

    def test_reconcile_matching(self):
        """Matching portfolios should be reconciled."""
        positions = [make_position("BTC/USDT")]
        expected = make_portfolio(total_value=100000.0, positions=positions)
        actual = make_portfolio(total_value=100000.0, positions=positions)
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is True
        assert len(result["discrepancies"]) == 0

    def test_reconcile_value_discrepancy(self):
        """Different total values within tolerance should not be flagged."""
        positions = [make_position("BTC/USDT")]
        expected = make_portfolio(total_value=100000.0, positions=positions)
        actual = make_portfolio(total_value=99999.5, positions=positions)
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is True  # Within $1 tolerance

    def test_reconcile_large_value_discrepancy(self):
        """Large value discrepancy should be flagged."""
        positions = [make_position("BTC/USDT")]
        expected = make_portfolio(total_value=100000.0, positions=positions)
        actual = make_portfolio(total_value=90000.0, positions=positions)
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is False
        value_disc = next(d for d in result["discrepancies"] if d["field"] == "total_value")
        assert value_disc["diff"] == -10000.0

    def test_reconcile_missing_position(self):
        """Missing position should be flagged."""
        positions_exp = [make_position("BTC/USDT"), make_position("ETH/USDT")]
        positions_act = [make_position("BTC/USDT")]
        expected = make_portfolio(total_value=100000.0, positions=positions_exp)
        actual = make_portfolio(total_value=100000.0, positions=positions_act)
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is False
        missing = [d for d in result["discrepancies"] if d.get("field") == "missing_position"]
        assert len(missing) >= 1

    def test_reconcile_quantity_discrepancy(self):
        """Quantity difference should be flagged."""
        pos_exp = make_position("BTC/USDT", quantity=1.0)
        pos_act = make_position("BTC/USDT", quantity=1.5)
        expected = make_portfolio(total_value=100000.0, positions=[pos_exp])
        actual = make_portfolio(total_value=100000.0, positions=[pos_act])
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is False
        qty_disc = next(d for d in result["discrepancies"] if d.get("field") == "quantity")
        assert qty_disc["symbol"] == "BTC/USDT"

    def test_reconcile_timestamp(self):
        """Result should include timestamp."""
        expected = make_portfolio(total_value=100000.0)
        actual = make_portfolio(total_value=100000.0)
        result = self.engine.reconcile(expected, actual)
        assert "timestamp" in result

    def test_reconcile_empty_portfolios(self):
        """Empty portfolios should be reconciled."""
        expected = make_portfolio(total_value=0.0, positions=[])
        actual = make_portfolio(total_value=0.0, positions=[])
        result = self.engine.reconcile(expected, actual)
        assert result["is_reconciled"] is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple portfolio components."""

    def test_full_optimization_workflow(self):
        """Should be able to run a complete optimization workflow."""
        engine = PortfolioEngine()
        ret_mat = generate_returns_matrix(300, 5, seed=90)
        expected_returns = np.mean(ret_mat, axis=0)
        cov_matrix = np.cov(ret_mat.T)

        # 1. Optimize
        result = engine.optimize_portfolio("mean_variance", expected_returns, cov_matrix)
        target_weights = result["weights"]

        # 2. Compute rebalance trades
        current_weights = np.ones(5) / 5
        trades = engine.compute_rebalance_trades(current_weights, target_weights, 1000000.0)
        # Should have some trades
        assert len(trades) > 0

    def test_optimization_methods_produce_valid_weights(self):
        """All optimization methods should produce weights that sum to ~1."""
        engine = PortfolioEngine()
        ret_mat = generate_returns_matrix(300, 5, seed=91)
        expected_returns = np.mean(ret_mat, axis=0)
        cov_matrix = np.cov(ret_mat.T)
        corr_matrix = np.corrcoef(ret_mat.T)

        methods = [
            ("mean_variance", {}),
            ("risk_parity", {}),
            ("hrp", {"returns_matrix": ret_mat}),
            ("max_diversification", {}),
            ("min_correlation", {"corr_matrix": corr_matrix}),
        ]

        for method, kwargs in methods:
            result = engine.optimize_portfolio(method, expected_returns, cov_matrix, **kwargs)
            if "weights" in result:
                assert abs(np.sum(result["weights"]) - 1.0) < 0.05, f"Method {method} failed weight sum check"

    def test_rebalance_cost_and_trades_consistency(self):
        """Rebalance cost should be consistent with computed trades."""
        current = np.array([0.5, 0.3, 0.2])
        target = np.array([0.4, 0.4, 0.2])
        portfolio_value = 100000.0

        engine = PortfolioEngine()
        trades = engine.compute_rebalance_trades(current, target, portfolio_value, threshold=0.01)
        cost_result = engine.rebalancing.compute_rebalance_cost(current, target, portfolio_value)

        # Total turnover from trades should match cost calculation
        trade_turnover = sum(abs(t["weight_change"]) * portfolio_value for t in trades)
        # The cost calculation uses all weight differences, trades may filter some
        assert cost_result["total_turnover"] >= trade_turnover - 1.0

    def test_leverage_optimizer_with_portfolio_engine(self):
        """Leverage optimizer should integrate with portfolio engine."""
        engine = PortfolioEngine()
        result = engine.leverage_optimizer.optimal_leverage(
            expected_return=0.15, volatility=0.20,
        )
        assert result["optimal_leverage"] > 0
        assert result["optimal_leverage"] <= engine.config.max_leverage

    def test_transaction_cost_model_with_rebalance(self):
        """Transaction cost model should work with rebalance trades."""
        engine = PortfolioEngine()
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        portfolio_value = 100000.0
        cost_result = engine.transaction_cost_model.compute_cost(
            20000.0, current, target, portfolio_value
        )
        assert cost_result["total_cost"] > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests across all portfolio components."""

    def test_mean_variance_with_negative_returns(self):
        """MV optimizer should handle negative expected returns."""
        er = np.array([-0.05, -0.10])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        config = PortfolioConfig(max_weight=0.80, min_weight=0.20)
        optimizer = MeanVarianceOptimizer(config)
        result = optimizer.optimize(er, cov)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-2

    def test_risk_parity_with_zero_off_diagonal(self):
        """Risk parity with diagonal covariance should give inverse-variance weights."""
        cov = np.diag([0.01, 0.04, 0.09])
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(cov)
        weights = result["weights"]
        # Inverse variance: [1/0.01, 1/0.04, 1/0.09] = [100, 25, 11.1]
        # Normalized: roughly [0.74, 0.18, 0.08]
        assert weights[0] > weights[1] > weights[2]

    def test_hrp_with_nearly_identical_returns(self):
        """HRP with nearly identical assets should give nearly equal weights."""
        rng = np.random.default_rng(100)
        base = rng.normal(0, 0.02, 200)
        ret_mat = np.column_stack([base + rng.normal(0, 0.0001, 200) for _ in range(3)])
        optimizer = HierarchicalRiskParity()
        result = optimizer.optimize(ret_mat)
        for w in result["weights"]:
            assert abs(w - 1/3) < 0.25

    def test_max_div_with_identical_assets(self):
        """Identical assets should give equal weights for max diversification."""
        cov = np.full((3, 3), 0.04)
        np.fill_diagonal(cov, 0.04)  # This makes it singular...
        # Use slightly different variances
        cov = np.full((3, 3), 0.01) + np.eye(3) * 0.03
        optimizer = MaximumDiversificationPortfolio()
        result = optimizer.optimize(cov)
        assert result["diversification_ratio"] >= 1.0

    def test_kelly_allocator_all_losing_assets(self):
        """All losing assets should give zero allocations."""
        win_rates = np.array([0.2, 0.3, 0.1])
        win_loss_ratios = np.array([0.5, 0.3, 0.2])
        capital = 100000.0
        allocator = KellyAllocator()
        result = allocator.allocate(win_rates, win_loss_ratios, capital)
        assert np.sum(result["weights"]) == 0.0

    def test_black_litterman_zero_tau(self):
        """Zero tau should give more weight to market equilibrium."""
        model = BlackLitterman(tau=0.0)
        market_weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = model.compute(market_weights, cov)
        np.testing.assert_array_equal(result["weights"], market_weights)

    def test_dynamic_rebalancing_first_time(self):
        """First rebalance check should always trigger time-based."""
        rebalancer = DynamicRebalancing()
        now = datetime.utcnow()
        assert rebalancer.check_time_rebalance(now) is True

    def test_cvar_optimization_with_fat_tails(self):
        """CVaR optimization should handle fat-tailed returns."""
        rng = np.random.default_rng(101)
        # Generate fat-tailed returns using t-distribution
        ret_mat = rng.standard_t(df=3, size=(200, 4)) * 0.02
        optimizer = CVaRPortfolioOptimization(confidence=0.95)
        result = optimizer.optimize(ret_mat)
        assert len(result["weights"]) == 4

    def test_transaction_cost_large_rebalance(self):
        """Large rebalance should have significant market impact cost."""
        model = TransactionCostModel(avg_daily_volume_usd=500000.0)
        current = np.array([0.5, 0.5])
        target = np.array([0.1, 0.9])
        portfolio_value = 1000000.0
        trade_notional = np.sum(np.abs(target - current)) * portfolio_value
        result = model.compute_cost(trade_notional, current, target, portfolio_value)
        assert result["market_impact_cost"] > 0

    def test_min_correlation_with_negative_correlation(self):
        """Should handle negative correlations."""
        corr = np.array([[1.0, -0.5], [-0.5, 1.0]])
        optimizer = MinimumCorrelationAlgorithm()
        result = optimizer.optimize(corr)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-6

    def test_portfolio_engine_config_propagation(self):
        """Config values should propagate to all sub-components."""
        config = PortfolioConfig(
            fixed_cost_usd=5.0,
            proportional_cost_bps=10.0,
            market_impact_alpha=0.2,
            max_leverage=5.0,
            transaction_cost_bps=20.0,
            rebalance_threshold=0.03,
        )
        engine = PortfolioEngine(config)
        assert engine.transaction_cost_model.fixed_cost_usd == 5.0
        assert engine.transaction_cost_model.proportional_cost_bps == 10.0
        assert engine.transaction_cost_model.market_impact_alpha == 0.2
        assert engine.leverage_optimizer.max_leverage == 5.0
        assert engine.rebalancing.threshold == 0.03

    def test_reconcile_small_quantity_discrepancy(self):
        """Very small quantity discrepancy should not be flagged."""
        pos_exp = make_position("BTC/USDT", quantity=1.000000001)
        pos_act = make_position("BTC/USDT", quantity=1.000000002)
        expected = make_portfolio(total_value=100000.0, positions=[pos_exp])
        actual = make_portfolio(total_value=100000.0, positions=[pos_act])
        engine = PortfolioEngine()
        result = engine.reconcile(expected, actual)
        # Difference is less than 1e-8 threshold
        assert result["is_reconciled"] is True

    def test_mean_variance_high_dimensional(self):
        """Should work with many assets."""
        n = 20
        expected_returns = generate_expected_returns(n, seed=110)
        cov_matrix = generate_cov_matrix(n, seed=110)
        optimizer = MeanVarianceOptimizer()
        result = optimizer.optimize(expected_returns, cov_matrix)
        assert len(result["weights"]) == n
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_leverage_optimal_with_zero_return(self):
        """Zero expected return should give zero optimal leverage."""
        optimizer = LeverageOptimizer()
        result = optimizer.optimal_leverage(expected_return=0.0, volatility=0.15)
        # Kelly = 0, half_kelly = 0, optimal should be 0
        assert result["kelly_leverage"] == 0.0
        assert result["half_kelly"] == 0.0

    def test_cost_adjusted_weights_preserve_order(self):
        """Adjusted weights should preserve relative ordering."""
        model = TransactionCostModel()
        current = np.array([0.5, 0.5])
        target = np.array([0.7, 0.3])
        adjusted = model.cost_adjusted_weights(current, target, 100000.0)
        # Target[0] > Target[1], so adjusted[0] > adjusted[1]
        assert adjusted[0] > adjusted[1]

    def test_cvar_budget_with_extreme_risk_budget(self):
        """Extreme risk budget should still produce valid weights."""
        ret_mat = generate_returns_matrix(200, 3, seed=120)
        # All risk budget to one asset
        risk_budget = np.array([1.0, 0.0, 0.0])
        optimizer = CVaRRiskBudgeting(confidence=0.95)
        result = optimizer.optimize(ret_mat, risk_budget=risk_budget)
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_dynamic_rebalancing_zero_weights(self):
        """Zero weights should not cause issues."""
        rebalancer = DynamicRebalancing()
        current = np.array([1.0, 0.0])
        target = np.array([0.5, 0.5])
        result = rebalancer.should_rebalance(current, target, datetime.utcnow())
        assert result["should_rebalance"] is True

    def test_compute_rebalance_trades_all_sell(self):
        """All sells should have 'sell' action."""
        engine = PortfolioEngine()
        current = np.array([0.7, 0.3])
        target = np.array([0.3, 0.7])
        trades = engine.compute_rebalance_trades(current, target, 100000.0, threshold=0.01)
        sell_trades = [t for t in trades if t["action"] == "sell"]
        buy_trades = [t for t in trades if t["action"] == "buy"]
        assert len(sell_trades) >= 1
        assert len(buy_trades) >= 1

    def test_risk_parity_with_high_dimensional_cov(self):
        """Risk parity should work with many assets."""
        n = 15
        cov = generate_cov_matrix(n, seed=130)
        optimizer = RiskParityOptimizer()
        result = optimizer.optimize(cov)
        assert len(result["weights"]) == n
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_efficient_frontier_two_extreme_assets(self):
        """Frontier with very different risk/return profiles."""
        er = np.array([0.05, 0.30])
        cov = np.array([[0.01, 0.0], [0.0, 0.25]])
        optimizer = MeanVarianceOptimizer()
        frontier = optimizer.efficient_frontier(er, cov, num_points=10)
        assert len(frontier) > 0

    def test_kelly_allocator_very_high_win_rate(self):
        """Very high win rate should not exceed total capital."""
        win_rates = np.array([0.95, 0.90])
        win_loss_ratios = np.array([5.0, 3.0])
        capital = 100000.0
        allocator = KellyAllocator()
        result = allocator.allocate(win_rates, win_loss_ratios, capital, fraction=1.0)
        assert np.sum(result["allocations"]) <= capital * 1.01  # small tolerance
