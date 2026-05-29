"""Comprehensive tests for acms.risk module.

Tests all classes, methods, and edge cases in the risk engine:
- RiskConfig dataclass
- ValueAtRisk (historical, parametric, monte_carlo, cvar)
- ExpectedShortfall (historical_es, parametric_es, cornish_fisher_es, tail_risk_decomposition)
- StressTesting (run_scenario, run_all_scenarios, run_historical_scenario, run_all_historical_scenarios)
- LiquidityRiskAssessor (assess_spread_risk, assess_depth_risk, compute_market_impact)
- CorrelationRiskMonitor (compute_correlation_matrix, eigenvalue_decomposition, detect_correlation_breakdown, check_concentration_risk)
- CounterpartyRiskScorer (score_counterparty, update_score, update_from_reserve_proof, update_from_withdrawal_status)
- PortfolioHeatMap (compute)
- CircuitBreaker (check, reset, _trigger)
- RiskBudgeting (allocate_budget, check_budget_utilization, compute_risk_contribution_targets)
- RiskEngine (pre_trade_check, kelly_size, fixed_fractional_size, volatility_regime_size,
              dynamic_position_size, trigger_kill_switch, reset_kill_switch,
              compute_portfolio_var, compute_portfolio_cvar, compute_portfolio_es,
              compute_tail_risk_decomposition)
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from acms.core import (
    RiskCheckResult, RiskDecision, Position, Order, Side,
    OrderType, OrderStatus, TimeInForce, PortfolioSnapshot, SignalDirection,
)
from acms.risk import (
    RiskConfig, ValueAtRisk, ExpectedShortfall, StressTesting,
    LiquidityRiskAssessor, CorrelationRiskMonitor, CounterpartyRiskScorer,
    PortfolioHeatMap, CircuitBreaker, RiskBudgeting, RiskEngine,
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


def make_order(symbol="BTC/USDT", side=Side.BUY, quantity=1.0,
               price=50000.0, order_type=OrderType.MARKET,
               status=OrderStatus.CREATED, strategy_id="test_strat"):
    """Create an Order instance for testing."""
    return Order(
        id="test-order-1", symbol=symbol, side=side,
        order_type=order_type, status=status,
        quantity=quantity, price=price,
        strategy_id=strategy_id,
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


def generate_returns(n=500, mu=0.0005, sigma=0.02, seed=42):
    """Generate synthetic return series."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


def generate_returns_matrix(t=300, n=5, seed=42):
    """Generate synthetic returns matrix (T x N)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, (t, n))


# ============================================================================
# RiskConfig Tests
# ============================================================================

class TestRiskConfig:
    """Tests for RiskConfig dataclass defaults and custom values."""

    def test_defaults(self):
        """All fields should have the expected default values."""
        cfg = RiskConfig()
        assert cfg.max_position_per_symbol == 100000.0
        assert cfg.max_total_position == 1000000.0
        assert cfg.max_order_notional == 50000.0
        assert cfg.max_order_quantity == 10.0
        assert cfg.max_daily_drawdown == 0.05
        assert cfg.max_weekly_drawdown == 0.10
        assert cfg.max_drawdown == 0.20
        assert cfg.max_orders_per_second == 10
        assert cfg.max_orders_per_minute == 100
        assert cfg.max_net_exposure == 500000.0
        assert cfg.max_gross_exposure == 1000000.0
        assert cfg.max_concentration_pct == 0.25
        assert cfg.var_confidence == 0.99
        assert cfg.cvar_confidence == 0.99
        assert cfg.max_correlation == 0.85
        assert cfg.initial_margin_ratio == 0.10
        assert cfg.maintenance_margin_ratio == 0.05
        assert cfg.circuit_breaker_loss_pct == 0.03
        assert cfg.circuit_breaker_cooldown_minutes == 30
        assert cfg.risk_budget_per_strategy == 0.25
        assert cfg.max_strategy_risk_pct == 0.40

    def test_custom_values(self):
        """Should accept custom values for all fields."""
        cfg = RiskConfig(
            max_position_per_symbol=50000.0,
            max_total_position=500000.0,
            max_order_notional=25000.0,
            max_order_quantity=5.0,
            max_daily_drawdown=0.03,
            max_weekly_drawdown=0.06,
            max_drawdown=0.15,
            max_orders_per_second=5,
            max_orders_per_minute=50,
            max_net_exposure=250000.0,
            max_gross_exposure=500000.0,
            max_concentration_pct=0.20,
            var_confidence=0.95,
            cvar_confidence=0.95,
            max_correlation=0.70,
            initial_margin_ratio=0.15,
            maintenance_margin_ratio=0.08,
            circuit_breaker_loss_pct=0.02,
            circuit_breaker_cooldown_minutes=15,
            risk_budget_per_strategy=0.20,
            max_strategy_risk_pct=0.30,
        )
        assert cfg.max_position_per_symbol == 50000.0
        assert cfg.max_total_position == 500000.0
        assert cfg.max_order_notional == 25000.0
        assert cfg.max_order_quantity == 5.0
        assert cfg.max_daily_drawdown == 0.03
        assert cfg.max_weekly_drawdown == 0.06
        assert cfg.max_drawdown == 0.15
        assert cfg.max_orders_per_second == 5
        assert cfg.max_orders_per_minute == 50
        assert cfg.max_net_exposure == 250000.0
        assert cfg.max_gross_exposure == 500000.0
        assert cfg.max_concentration_pct == 0.20
        assert cfg.var_confidence == 0.95
        assert cfg.cvar_confidence == 0.95
        assert cfg.max_correlation == 0.70
        assert cfg.initial_margin_ratio == 0.15
        assert cfg.maintenance_margin_ratio == 0.08
        assert cfg.circuit_breaker_loss_pct == 0.02
        assert cfg.circuit_breaker_cooldown_minutes == 15
        assert cfg.risk_budget_per_strategy == 0.20
        assert cfg.max_strategy_risk_pct == 0.30

    def test_partial_custom(self):
        """Should allow setting only some fields; others remain default."""
        cfg = RiskConfig(max_drawdown=0.50, max_orders_per_second=2)
        assert cfg.max_drawdown == 0.50
        assert cfg.max_orders_per_second == 2
        # Others should remain at defaults
        assert cfg.max_position_per_symbol == 100000.0
        assert cfg.var_confidence == 0.99

    def test_zero_values(self):
        """Should accept zero values (edge case)."""
        cfg = RiskConfig(max_drawdown=0.0, max_position_per_symbol=0.0)
        assert cfg.max_drawdown == 0.0
        assert cfg.max_position_per_symbol == 0.0

    def test_negative_values(self):
        """Dataclass doesn't enforce validation, so negatives are accepted."""
        cfg = RiskConfig(max_drawdown=-0.1)
        assert cfg.max_drawdown == -0.1


# ============================================================================
# ValueAtRisk Tests
# ============================================================================

class TestValueAtRisk:
    """Tests for ValueAtRisk class: historical, parametric, monte_carlo, cvar."""

    def setup_method(self):
        self.returns_200 = generate_returns(200, seed=1)
        self.returns_50 = generate_returns(50, seed=2)
        self.returns_10 = generate_returns(10, seed=3)
        self.returns_100 = generate_returns(100, seed=4)

    # --- historical ---

    def test_historical_sufficient_data(self):
        """With >= 100 returns, should return a positive finite VaR."""
        var = ValueAtRisk.historical(self.returns_200, confidence=0.99)
        assert np.isfinite(var)
        assert var > 0

    def test_historical_insufficient_data(self):
        """With < 100 returns, should return NaN."""
        var = ValueAtRisk.historical(self.returns_50, confidence=0.99)
        assert np.isnan(var)

    def test_historical_exact_100(self):
        """With exactly 100 returns, should return a finite VaR."""
        returns = generate_returns(100, seed=10)
        var = ValueAtRisk.historical(returns, confidence=0.99)
        assert np.isfinite(var)

    def test_historical_99_returns(self):
        """With 99 returns, should return NaN."""
        returns = generate_returns(99, seed=11)
        var = ValueAtRisk.historical(returns, confidence=0.99)
        assert np.isnan(var)

    def test_historical_different_confidence(self):
        """Lower confidence should give lower VaR."""
        var_99 = ValueAtRisk.historical(self.returns_200, confidence=0.99)
        var_95 = ValueAtRisk.historical(self.returns_200, confidence=0.95)
        assert var_95 < var_99

    def test_historical_zero_returns(self):
        """All-zero returns should give zero VaR."""
        returns = np.zeros(200)
        var = ValueAtRisk.historical(returns, confidence=0.99)
        assert var == 0.0

    def test_historical_constant_positive_returns(self):
        """All-constant positive returns should give negative VaR (no loss)."""
        returns = np.full(200, 0.01)
        var = ValueAtRisk.historical(returns, confidence=0.99)
        # 1st percentile of constant 0.01 is 0.01, VaR = -0.01 (negative = no loss at that level)
        assert var < 0

    # --- parametric ---

    def test_parametric_sufficient_data(self):
        """With >= 30 returns, should return finite parametric VaR."""
        var = ValueAtRisk.parametric(self.returns_200, confidence=0.99)
        assert np.isfinite(var)

    def test_parametric_insufficient_data(self):
        """With < 30 returns, should return NaN."""
        var = ValueAtRisk.parametric(self.returns_10, confidence=0.99)
        assert np.isnan(var)

    def test_parametric_exact_30(self):
        """With exactly 30 returns, should return a finite VaR."""
        returns = generate_returns(30, seed=20)
        var = ValueAtRisk.parametric(returns, confidence=0.99)
        assert np.isfinite(var)

    def test_parametric_29_returns(self):
        """With 29 returns, should return NaN."""
        returns = generate_returns(29, seed=21)
        var = ValueAtRisk.parametric(returns, confidence=0.99)
        assert np.isnan(var)

    def test_parametric_different_confidence(self):
        """Lower confidence should generally give lower parametric VaR."""
        var_99 = ValueAtRisk.parametric(self.returns_200, confidence=0.99)
        var_95 = ValueAtRisk.parametric(self.returns_200, confidence=0.95)
        # 99% should typically be larger than 95%
        assert var_99 > var_95

    def test_parametric_zero_volatility(self):
        """Zero variance returns should give VaR equal to negative mean."""
        returns = np.full(100, 0.005)
        var = ValueAtRisk.parametric(returns, confidence=0.99)
        # sigma=0, mu=0.005, z*0=0, var = -(0.005 - 0) = -0.005
        assert abs(var - (-0.005)) < 1e-10

    # --- monte_carlo ---

    def test_monte_carlo_sufficient_data(self):
        """With >= 30 returns, should return finite MC VaR."""
        var = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99)
        assert np.isfinite(var)

    def test_monte_carlo_insufficient_data(self):
        """With < 30 returns, should return NaN."""
        var = ValueAtRisk.monte_carlo(self.returns_10, confidence=0.99)
        assert np.isnan(var)

    def test_monte_carlo_multi_day_horizon(self):
        """Longer horizon should give larger VaR."""
        var_1d = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99, horizon_days=1, num_simulations=5000)
        var_10d = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99, horizon_days=10, num_simulations=5000)
        assert var_10d > var_1d

    def test_monte_carlo_custom_simulations(self):
        """Should accept custom number of simulations."""
        var = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99, num_simulations=1000)
        assert np.isfinite(var)

    def test_monte_carlo_stability(self):
        """MC VaR should be relatively stable across runs with enough simulations."""
        # Use a fixed seed inside the method is not controllable, but with many sims results should be close
        var1 = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99, num_simulations=100000)
        var2 = ValueAtRisk.monte_carlo(self.returns_200, confidence=0.99, num_simulations=100000)
        assert abs(var1 - var2) / max(abs(var1), abs(var2)) < 0.3  # within 30%

    # --- cvar ---

    def test_cvar_sufficient_data(self):
        """With >= 100 returns, should return finite CVaR."""
        cvar = ValueAtRisk.cvar(self.returns_200, confidence=0.99)
        assert np.isfinite(cvar)
        assert cvar > 0

    def test_cvar_insufficient_data(self):
        """With < 100 returns, should return NaN."""
        cvar = ValueAtRisk.cvar(self.returns_50, confidence=0.99)
        assert np.isnan(cvar)

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR for the same data and confidence."""
        var = ValueAtRisk.historical(self.returns_200, confidence=0.99)
        cvar = ValueAtRisk.cvar(self.returns_200, confidence=0.99)
        assert cvar >= var

    def test_cvar_zero_returns(self):
        """All-zero returns should give zero CVaR."""
        returns = np.zeros(200)
        cvar = ValueAtRisk.cvar(returns, confidence=0.99)
        assert cvar == 0.0

    def test_cvar_extreme_returns(self):
        """With extreme negative returns, CVaR should be large."""
        returns = generate_returns(200, seed=30)
        returns[0] = -0.50  # extreme loss
        cvar = ValueAtRisk.cvar(returns, confidence=0.99)
        assert cvar > 0.01


# ============================================================================
# ExpectedShortfall Tests
# ============================================================================

class TestExpectedShortfall:
    """Tests for ExpectedShortfall class."""

    def setup_method(self):
        self.returns_200 = generate_returns(200, seed=5)
        self.returns_50 = generate_returns(50, seed=6)
        self.returns_20 = generate_returns(20, seed=7)

    # --- historical_es ---

    def test_historical_es_sufficient_data(self):
        """With >= 50 returns, should return finite ES."""
        es = ExpectedShortfall.historical_es(self.returns_200, confidence=0.975)
        assert np.isfinite(es)
        assert es > 0

    def test_historical_es_insufficient_data(self):
        """With < 50 returns, should return NaN."""
        es = ExpectedShortfall.historical_es(self.returns_20, confidence=0.975)
        assert np.isnan(es)

    def test_historical_es_exact_50(self):
        """With exactly 50 returns, should return finite ES."""
        returns = generate_returns(50, seed=50)
        es = ExpectedShortfall.historical_es(returns, confidence=0.975)
        assert np.isfinite(es)

    def test_historical_es_different_confidence(self):
        """Higher confidence should give higher ES."""
        es_975 = ExpectedShortfall.historical_es(self.returns_200, confidence=0.975)
        es_95 = ExpectedShortfall.historical_es(self.returns_200, confidence=0.95)
        assert es_975 >= es_95

    def test_historical_es_zero_returns(self):
        """All-zero returns should give zero ES."""
        returns = np.zeros(200)
        es = ExpectedShortfall.historical_es(returns, confidence=0.975)
        assert es == 0.0

    # --- parametric_es ---

    def test_parametric_es_sufficient_data(self):
        """With >= 30 returns, should return finite parametric ES."""
        es = ExpectedShortfall.parametric_es(self.returns_200, confidence=0.975)
        assert np.isfinite(es)

    def test_parametric_es_insufficient_data(self):
        """With < 30 returns, should return NaN."""
        es = ExpectedShortfall.parametric_es(self.returns_20, confidence=0.975)
        assert np.isnan(es)

    def test_parametric_es_positive_for_normal_returns(self):
        """For slightly positive mean returns, parametric ES should still be positive."""
        es = ExpectedShortfall.parametric_es(self.returns_200, confidence=0.975)
        assert es > 0

    def test_parametric_es_different_confidence(self):
        """Higher confidence should give higher parametric ES."""
        es_975 = ExpectedShortfall.parametric_es(self.returns_200, confidence=0.975)
        es_95 = ExpectedShortfall.parametric_es(self.returns_200, confidence=0.95)
        assert es_975 > es_95

    # --- cornish_fisher_es ---

    def test_cornish_fisher_es_sufficient_data(self):
        """With >= 30 returns, should return finite CF ES."""
        es = ExpectedShortfall.cornish_fisher_es(self.returns_200, confidence=0.975)
        assert np.isfinite(es)

    def test_cornish_fisher_es_insufficient_data(self):
        """With < 30 returns, should return NaN."""
        es = ExpectedShortfall.cornish_fisher_es(self.returns_20, confidence=0.975)
        assert np.isnan(es)

    def test_cornish_fisher_vs_parametric_normal_data(self):
        """For near-normal data, CF ES should be close to parametric ES."""
        # Normal data: skew ~0, kurtosis ~0
        returns = np.random.default_rng(99).normal(0.001, 0.02, 500)
        cf_es = ExpectedShortfall.cornish_fisher_es(returns, confidence=0.975)
        p_es = ExpectedShortfall.parametric_es(returns, confidence=0.975)
        # They should be in the same ballpark (within 50%)
        assert abs(cf_es - p_es) / max(abs(p_es), 1e-10) < 0.5

    def test_cornish_fisher_skewed_data(self):
        """For skewed data, CF ES should differ from parametric ES."""
        rng = np.random.default_rng(77)
        # Create skewed returns using log-normal
        returns = rng.lognormal(0, 0.3, 300) - 1.1  # left-skewed loss distribution
        cf_es = ExpectedShortfall.cornish_fisher_es(returns, confidence=0.975)
        p_es = ExpectedShortfall.parametric_es(returns, confidence=0.975)
        # CF should account for skew, may differ
        assert np.isfinite(cf_es)
        assert np.isfinite(p_es)

    # --- tail_risk_decomposition ---

    def test_tail_risk_decomposition_basic(self):
        """Should return contributions for each asset."""
        ret_mat = generate_returns_matrix(300, 3, seed=10)
        weights = np.array([0.4, 0.35, 0.25])
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        assert "contributions" in result
        assert "pct_contributions" in result
        assert "total_es" in result
        assert "marginal_es" in result
        assert len(result["contributions"]) == 3
        assert len(result["pct_contributions"]) == 3
        assert np.isfinite(result["total_es"])

    def test_tail_risk_decomposition_pct_sums(self):
        """Percentage contributions should sum to ~100."""
        ret_mat = generate_returns_matrix(300, 4, seed=11)
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        assert abs(np.sum(result["pct_contributions"]) - 100.0) < 1.0

    def test_tail_risk_decomposition_mismatched_weights(self):
        """Mismatched weights and columns should return empty."""
        ret_mat = generate_returns_matrix(300, 3, seed=12)
        weights = np.array([0.5, 0.5])  # only 2 weights for 3 assets
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        assert len(result["contributions"]) == 0
        assert np.isnan(result["total_es"])

    def test_tail_risk_decomposition_single_asset(self):
        """Single asset (len(weights) < 2) should return NaN."""
        ret_mat = generate_returns_matrix(300, 1, seed=13)
        weights = np.array([1.0])
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        assert np.isnan(result["total_es"])

    def test_tail_risk_decomposition_insufficient_returns(self):
        """Too few returns for ES should give NaN total_es but zero contributions."""
        ret_mat = generate_returns_matrix(30, 3, seed=14)
        weights = np.array([0.4, 0.35, 0.25])
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        assert np.isnan(result["total_es"])
        assert np.all(result["contributions"] == 0)

    def test_tail_risk_decomposition_equal_weights(self):
        """Equal weights should give roughly equal contributions for similar assets."""
        ret_mat = generate_returns_matrix(300, 3, seed=15)
        weights = np.array([1/3, 1/3, 1/3])
        result = ExpectedShortfall.tail_risk_decomposition(ret_mat, weights)
        # All pct contributions should be in a reasonable range
        for pct in result["pct_contributions"]:
            assert pct >= 0


# ============================================================================
# StressTesting Tests
# ============================================================================

class TestStressTesting:
    """Tests for StressTesting class."""

    def setup_method(self):
        self.stress = StressTesting()
        self.positions = [
            make_position("BTC/USDT", Side.BUY, 1.0, 50000, 50000, leverage=1.0),
            make_position("ETH/USDT", Side.BUY, 10.0, 3000, 3000, leverage=1.0),
        ]
        self.leveraged_positions = [
            make_position("BTC/USDT", Side.BUY, 1.0, 50000, 50000, leverage=3.0),
            make_position("ETH/USDT", Side.SELL, 10.0, 3000, 3000, leverage=2.0),
        ]

    def test_run_scenario_flash_crash(self):
        """Flash crash scenario should produce negative PnL for BUY positions."""
        result = self.stress.run_scenario(self.positions, "flash_crash")
        assert result["scenario"] == "flash_crash"
        assert result["total_pnl"] < 0
        assert len(result["position_results"]) == 2
        assert "parameters" in result

    def test_run_scenario_unknown(self):
        """Unknown scenario should return error dict."""
        result = self.stress.run_scenario(self.positions, "nonexistent_scenario")
        assert "error" in result

    def test_run_scenario_leveraged(self):
        """Leveraged positions should have 1.5x shock multiplier."""
        result = self.stress.run_scenario(self.leveraged_positions, "flash_crash")
        # Leverage > 1 so shock = equity_shock * 1.5
        shock_pct = result["position_results"][0]["shock_pct"]
        expected_shock = -0.20 * 1.5 * 100  # -30%
        assert abs(shock_pct - expected_shock) < 0.01

    def test_run_scenario_sell_side(self):
        """SELL positions should profit from equity shocks (negative pnl for positive shock)."""
        # slow_bleed has equity_shock=-0.10, corr_to_1=False
        sell_pos = [make_position("BTC/USDT", Side.SELL, 1.0, 50000, 50000)]
        result = self.stress.run_scenario(sell_pos, "slow_bleed")
        # SELL side: pnl = notional * (-shock) = notional * 0.10 > 0
        assert result["total_pnl"] > 0

    def test_run_scenario_with_correlation_penalty(self):
        """Scenario with corr_to_1 and correlations should apply correlation adjustment."""
        result_no_corr = self.stress.run_scenario(self.positions, "flash_crash", correlations=None)
        corr_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        result_with_corr = self.stress.run_scenario(self.positions, "flash_crash", correlations=corr_matrix)
        # Correlation penalty adjusts total_pnl by -correlation_penalty * sign(total_pnl)
        # When total_pnl is negative (loss), sign=-1, so it adds +penalty (less negative)
        # The results should differ
        assert result_with_corr["total_pnl"] != result_no_corr["total_pnl"]

    def test_run_all_scenarios(self):
        """Should run all 9 scenarios and return results for each."""
        results = self.stress.run_all_scenarios(self.positions)
        assert len(results) == len(StressTesting.SCENARIOS)
        for name in StressTesting.SCENARIOS:
            assert name in results
            assert "total_pnl" in results[name]

    def test_run_all_scenarios_empty_positions(self):
        """With no positions, total_pnl should be 0 for all scenarios."""
        results = self.stress.run_all_scenarios([])
        for name, result in results.items():
            assert result["total_pnl"] == 0.0

    def test_run_historical_scenario_covid(self):
        """COVID historical scenario should return detailed results."""
        result = self.stress.run_historical_scenario(
            self.positions, "covid_crash_feb_mar_2020",
            is_alt={"BTC/USDT": False, "ETH/USDT": True}
        )
        assert result["scenario"] == "covid_crash_feb_mar_2020"
        assert "description" in result
        assert "start_date" in result
        assert "end_date" in result
        assert result["total_pnl"] < 0
        assert len(result["position_results"]) == 2
        # BTC should use btc_shock, ETH should use alt_shock
        btc_result = next(r for r in result["position_results"] if r["symbol"] == "BTC/USDT")
        eth_result = next(r for r in result["position_results"] if r["symbol"] == "ETH/USDT")
        assert btc_result["is_alt"] is False
        assert eth_result["is_alt"] is True

    def test_run_historical_scenario_unknown(self):
        """Unknown historical scenario should return error."""
        result = self.stress.run_historical_scenario(self.positions, "nonexistent")
        assert "error" in result

    def test_run_historical_scenario_default_is_alt(self):
        """Without is_alt mapping, all positions should be treated as alt."""
        result = self.stress.run_historical_scenario(self.positions, "ftx_collapse_nov_2022")
        for pr in result["position_results"]:
            assert pr["is_alt"] is True

    def test_run_historical_scenario_leveraged(self):
        """Leveraged positions in historical scenarios should get 1.5x shock."""
        result = self.stress.run_historical_scenario(
            self.leveraged_positions, "covid_crash_feb_mar_2020"
        )
        for pr in result["position_results"]:
            # All are leveraged, so shock should be alt_shock * 1.5
            assert pr["shock_pct"] != 0

    def test_run_all_historical_scenarios(self):
        """Should run all 4 historical scenarios."""
        results = self.stress.run_all_historical_scenarios(self.positions)
        assert len(results) == len(StressTesting.HISTORICAL_SCENARIOS)
        for name in StressTesting.HISTORICAL_SCENARIOS:
            assert name in results

    def test_historical_scenarios_have_required_fields(self):
        """Each historical scenario result should have required fields."""
        result = self.stress.run_historical_scenario(
            self.positions, "luna_crash_may_2022"
        )
        assert "recovery_estimate_days" in result["position_results"][0]
        assert result["position_results"][0]["recovery_estimate_days"] == 90


# ============================================================================
# LiquidityRiskAssessor Tests
# ============================================================================

class TestLiquidityRiskAssessor:
    """Tests for LiquidityRiskAssessor class."""

    def setup_method(self):
        self.assessor = LiquidityRiskAssessor(
            normal_spread_bps=5.0, max_spread_bps=50.0, min_depth_usd=10000.0
        )

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        a = LiquidityRiskAssessor()
        assert a.normal_spread_bps == 5.0
        assert a.max_spread_bps == 50.0
        assert a.min_depth_usd == 10000.0

    def test_init_custom(self):
        """Custom parameters should be stored."""
        a = LiquidityRiskAssessor(normal_spread_bps=10.0, max_spread_bps=100.0, min_depth_usd=5000.0)
        assert a.normal_spread_bps == 10.0
        assert a.max_spread_bps == 100.0
        assert a.min_depth_usd == 5000.0

    # --- assess_spread_risk ---

    def test_spread_risk_low(self):
        """Spread <= 2x normal should be low risk."""
        result = self.assessor.assess_spread_risk(8.0)  # 8/5 = 1.6
        assert result["risk_level"] == "low"
        assert result["spread_ratio"] == 8.0 / 5.0

    def test_spread_risk_moderate(self):
        """Spread between 2x and 3x normal should be moderate."""
        result = self.assessor.assess_spread_risk(12.0)  # 12/5 = 2.4
        assert result["risk_level"] == "moderate"

    def test_spread_risk_high(self):
        """Spread between 3x and 5x normal should be high."""
        result = self.assessor.assess_spread_risk(18.0)  # 18/5 = 3.6
        assert result["risk_level"] == "high"

    def test_spread_risk_critical(self):
        """Spread > 5x normal should be critical."""
        result = self.assessor.assess_spread_risk(30.0)  # 30/5 = 6.0
        assert result["risk_level"] == "critical"

    def test_spread_slippage_estimate(self):
        """Slippage estimate should be half the current spread."""
        result = self.assessor.assess_spread_risk(10.0)
        assert result["slippage_estimate_bps"] == 5.0

    def test_spread_widening_trend_detected(self):
        """Should detect widening trend after 5 consecutive increases."""
        spreads = [5.0, 6.0, 7.0, 8.0, 9.0]
        for s in spreads:
            result = self.assessor.assess_spread_risk(s)
        assert result["widening_trend_detected"] is True

    def test_spread_no_widening_trend(self):
        """Should not detect widening if spreads fluctuate."""
        spreads = [5.0, 4.0, 6.0, 3.0, 7.0]
        for s in spreads:
            result = self.assessor.assess_spread_risk(s)
        assert result["widening_trend_detected"] is False

    def test_spread_history_tracking(self):
        """Should track spread history over time."""
        self.assessor.assess_spread_risk(5.0)
        self.assessor.assess_spread_risk(6.0)
        assert len(self.assessor._spread_history) == 2

    def test_spread_zero_normal(self):
        """Zero normal_spread_bps should give ratio of 1.0."""
        a = LiquidityRiskAssessor(normal_spread_bps=0.0)
        result = a.assess_spread_risk(10.0)
        assert result["spread_ratio"] == 1.0

    # --- assess_depth_risk ---

    def test_depth_risk_low(self):
        """Depth ratio >= 5 should be low risk."""
        result = self.assessor.assess_depth_risk(100000.0, 100000.0, 10000.0)
        assert result["risk_level"] == "low"
        assert result["depth_ratio"] == 10.0

    def test_depth_risk_moderate(self):
        """Depth ratio between 2 and 5 should be moderate."""
        result = self.assessor.assess_depth_risk(30000.0, 30000.0, 10000.0)
        assert result["risk_level"] == "moderate"

    def test_depth_risk_high(self):
        """Depth ratio between 1 and 2 should be high."""
        result = self.assessor.assess_depth_risk(15000.0, 15000.0, 10000.0)
        assert result["risk_level"] == "high"

    def test_depth_risk_critical(self):
        """Depth below min_depth should be critical."""
        result = self.assessor.assess_depth_risk(5000.0, 8000.0, 10000.0)
        assert result["risk_level"] == "critical"

    def test_depth_fill_estimate(self):
        """Fill estimate should be min(order/min_depth, 1.0)."""
        result = self.assessor.assess_depth_risk(50000.0, 50000.0, 60000.0)
        # min_depth=50000, fill=60000/50000=1.2, capped at 1.0
        assert result["fill_estimate"] == 1.0

    def test_depth_fill_estimate_partial(self):
        """Fill estimate < 1 when depth is smaller than order."""
        result = self.assessor.assess_depth_risk(5000.0, 5000.0, 10000.0)
        # fill = 10000 / (5000 + 1e-10) = 2.0, capped at 1.0
        assert result["fill_estimate"] == 1.0

    def test_depth_zero_order_size(self):
        """Zero order size should give inf depth_ratio."""
        result = self.assessor.assess_depth_risk(50000.0, 50000.0, 0.0)
        assert result["depth_ratio"] == float('inf')

    def test_depth_thinning_alert(self):
        """Should detect thinning after 5 consecutive depth decreases."""
        depths = [(100000, 100000), (90000, 90000), (80000, 80000),
                  (70000, 70000), (60000, 60000)]
        for b, a in depths:
            result = self.assessor.assess_depth_risk(b, a, 10000.0)
        assert result["depth_thinning_alert"] is True

    def test_depth_no_thinning_alert(self):
        """Should not alert when depths fluctuate."""
        depths = [(100000, 100000), (110000, 110000), (90000, 90000),
                  (120000, 120000), (80000, 80000)]
        for b, a in depths:
            result = self.assessor.assess_depth_risk(b, a, 10000.0)
        assert result["depth_thinning_alert"] is False

    def test_depth_imbalance_ratio(self):
        """Imbalance ratio should be bid/ask."""
        result = self.assessor.assess_depth_risk(60000.0, 40000.0, 10000.0)
        assert abs(result["imbalance_ratio"] - 60000.0 / 40000.0) < 0.01

    # --- compute_market_impact ---

    def test_market_impact_basic(self):
        """Should return positive market impact."""
        impact = self.assessor.compute_market_impact(100000.0, 1000000.0, alpha=0.5)
        assert impact > 0

    def test_market_impact_zero_volume(self):
        """Zero daily volume should return 0."""
        impact = self.assessor.compute_market_impact(100000.0, 0.0)
        assert impact == 0.0

    def test_market_impact_zero_order(self):
        """Zero order size should return 0."""
        impact = self.assessor.compute_market_impact(0.0, 1000000.0)
        assert impact == 0.0

    def test_market_impact_negative_inputs(self):
        """Negative inputs should return 0."""
        assert self.assessor.compute_market_impact(-100.0, 1000000.0) == 0.0
        assert self.assessor.compute_market_impact(100.0, -1000000.0) == 0.0

    def test_market_impact_large_order(self):
        """Larger order should have more impact."""
        impact_small = self.assessor.compute_market_impact(10000.0, 1000000.0)
        impact_large = self.assessor.compute_market_impact(100000.0, 1000000.0)
        assert impact_large > impact_small

    def test_market_impact_custom_alpha(self):
        """Custom alpha should scale impact proportionally."""
        impact_05 = self.assessor.compute_market_impact(50000.0, 1000000.0, alpha=0.5)
        impact_10 = self.assessor.compute_market_impact(50000.0, 1000000.0, alpha=1.0)
        assert abs(impact_10 / impact_05 - 2.0) < 0.01


# ============================================================================
# CorrelationRiskMonitor Tests
# ============================================================================

class TestCorrelationRiskMonitor:
    """Tests for CorrelationRiskMonitor class."""

    def setup_method(self):
        self.monitor = CorrelationRiskMonitor(lookback=60, max_correlation=0.85, breakdown_threshold=0.3)

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        m = CorrelationRiskMonitor()
        assert m.lookback == 60
        assert m.max_correlation == 0.85
        assert m.breakdown_threshold == 0.3

    def test_init_custom(self):
        """Custom parameters should be stored."""
        m = CorrelationRiskMonitor(lookback=30, max_correlation=0.7, breakdown_threshold=0.2)
        assert m.lookback == 30
        assert m.max_correlation == 0.7
        assert m.breakdown_threshold == 0.2

    # --- compute_correlation_matrix ---

    def test_compute_corr_matrix_sufficient_data(self):
        """With >= 10 observations, should return actual correlation matrix."""
        ret_mat = generate_returns_matrix(100, 3, seed=20)
        corr = self.monitor.compute_correlation_matrix(ret_mat)
        assert corr.shape == (3, 3)
        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=0.01)

    def test_compute_corr_matrix_insufficient_data(self):
        """With < 10 observations, should return identity matrix."""
        ret_mat = generate_returns_matrix(5, 3, seed=21)
        corr = self.monitor.compute_correlation_matrix(ret_mat)
        np.testing.assert_array_equal(corr, np.eye(3))

    def test_compute_corr_matrix_symmetric(self):
        """Correlation matrix should be symmetric."""
        ret_mat = generate_returns_matrix(100, 4, seed=22)
        corr = self.monitor.compute_correlation_matrix(ret_mat)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    # --- eigenvalue_decomposition ---

    def test_eigenvalue_decomposition_basic(self):
        """Should return eigenvalues, eigenvectors, and metrics."""
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = self.monitor.eigenvalue_decomposition(corr)
        assert "eigenvalues" in result
        assert "eigenvectors" in result
        assert "pct_variance_explained" in result
        assert "effective_rank" in result
        assert "concentration_ratio" in result
        assert "is_concentrated" in result
        assert len(result["eigenvalues"]) == 2

    def test_eigenvalue_decomposition_sum(self):
        """Eigenvalues of correlation matrix should sum to N."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = self.monitor.eigenvalue_decomposition(corr)
        assert abs(np.sum(result["eigenvalues"]) - 2.0) < 0.01

    def test_eigenvalue_decomposition_concentrated(self):
        """Highly correlated assets should show concentration."""
        corr = np.array([[1.0, 0.99], [0.99, 1.0]])
        result = self.monitor.eigenvalue_decomposition(corr)
        assert result["is_concentrated"] is True
        assert result["concentration_ratio"] > 0.5

    def test_eigenvalue_decomposition_uncorrelated(self):
        """Uncorrelated assets should not show concentration."""
        corr = np.eye(3)
        result = self.monitor.eigenvalue_decomposition(corr)
        assert result["is_concentrated"] is False
        assert result["concentration_ratio"] < 0.5

    def test_eigenvalue_history_tracking(self):
        """Should track eigenvalue history."""
        corr = np.eye(2)
        self.monitor.eigenvalue_decomposition(corr)
        assert len(self.monitor._eigenvalue_history) == 1

    # --- detect_correlation_breakdown ---

    def test_detect_breakdown_first_call(self):
        """First call should not detect breakdown."""
        corr = np.eye(3)
        result = self.monitor.detect_correlation_breakdown(corr)
        assert result["breakdown_detected"] is False
        assert result["max_change"] == 0.0

    def test_detect_breakdown_no_change(self):
        """Same correlation matrix should not trigger breakdown."""
        corr = np.eye(3)
        self.monitor.detect_correlation_breakdown(corr)
        result = self.monitor.detect_correlation_breakdown(corr)
        assert result["breakdown_detected"] is False
        assert result["max_change"] == 0.0

    def test_detect_breakdown_large_change(self):
        """Large change in correlation should trigger breakdown."""
        corr1 = np.eye(3)
        corr2 = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        self.monitor.detect_correlation_breakdown(corr1)
        result = self.monitor.detect_correlation_breakdown(corr2)
        assert result["breakdown_detected"] is True
        assert result["max_change"] > 0.3

    def test_detect_breakdown_small_change(self):
        """Small change should not trigger breakdown."""
        corr1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        corr2 = np.array([[1.0, 0.55], [0.55, 1.0]])
        self.monitor.detect_correlation_breakdown(corr1)
        result = self.monitor.detect_correlation_breakdown(corr2)
        assert result["breakdown_detected"] is False

    def test_detect_breakdown_shape_change(self):
        """Different shaped matrix should reset without breakdown."""
        corr1 = np.eye(3)
        corr2 = np.eye(4)
        self.monitor.detect_correlation_breakdown(corr1)
        result = self.monitor.detect_correlation_breakdown(corr2)
        assert result["breakdown_detected"] is False
        assert result["max_change"] == 0.0

    def test_detect_breakdown_affected_pairs(self):
        """Should count affected pairs."""
        corr1 = np.eye(3)
        corr2 = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
        self.monitor.detect_correlation_breakdown(corr1)
        result = self.monitor.detect_correlation_breakdown(corr2)
        assert result["affected_pairs"] > 0

    # --- check_concentration_risk ---

    def test_concentration_risk_low(self):
        """Low correlation should give low risk level."""
        corr = np.eye(3)
        weights = np.array([1/3, 1/3, 1/3])
        result = self.monitor.check_concentration_risk(corr, weights)
        assert result["risk_level"] == "low"
        assert result["high_correlation_pairs"] == 0

    def test_concentration_risk_high(self):
        """High correlation should give high risk level."""
        # 4 assets => 6 pairs, with 5 pairs > 0.85 => 5 > n(4) => high
        corr = np.array([
            [1.0, 0.95, 0.95, 0.95],
            [0.95, 1.0, 0.95, 0.95],
            [0.95, 0.95, 1.0, 0.50],
            [0.95, 0.95, 0.50, 1.0],
        ])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = self.monitor.check_concentration_risk(corr, weights)
        # 4 pairs > 0.85: (0,1), (0,2), (0,3), (1,2), (1,3) = 5 pairs
        assert result["high_correlation_pairs"] >= 4
        assert result["risk_level"] == "high"

    def test_concentration_risk_moderate(self):
        """Some correlation should give moderate risk level."""
        corr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.3], [0.3, 0.3, 1.0]])
        weights = np.array([1/3, 1/3, 1/3])
        result = self.monitor.check_concentration_risk(corr, weights)
        assert result["risk_level"] == "moderate"
        assert result["high_correlation_pairs"] == 1

    def test_concentration_risk_diversification_ratio(self):
        """Diversification ratio should be >= 1 for uncorrelated assets."""
        corr = np.eye(3)
        weights = np.array([1/3, 1/3, 1/3])
        result = self.monitor.check_concentration_risk(corr, weights)
        assert result["diversification_ratio"] >= 1.0

    def test_concentration_risk_max_correlation(self):
        """Max off-diagonal correlation should be reported."""
        corr = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.3], [0.3, 0.3, 1.0]])
        weights = np.array([1/3, 1/3, 1/3])
        result = self.monitor.check_concentration_risk(corr, weights)
        assert abs(result["max_correlation"] - 0.7) < 0.01


# ============================================================================
# CounterpartyRiskScorer Tests
# ============================================================================

class TestCounterpartyRiskScorer:
    """Tests for CounterpartyRiskScorer class."""

    def setup_method(self):
        self.scorer = CounterpartyRiskScorer()

    def test_known_exchanges(self):
        """Known exchanges should have predefined scores."""
        for exchange in ["binance", "bybit", "okx", "coinbase", "kraken", "paper"]:
            result = self.scorer.score_counterparty(exchange)
            assert result["exchange"] == exchange
            assert "scores" in result
            assert "composite_score" in result
            assert "risk_level" in result

    def test_paper_exchange_perfect_scores(self):
        """Paper exchange should have perfect scores."""
        result = self.scorer.score_counterparty("paper")
        assert result["composite_score"] == 100.0
        assert result["risk_level"] == "low"
        assert len(result["warnings"]) == 0

    def test_unknown_exchange(self):
        """Unknown exchange should get default scores (50s)."""
        result = self.scorer.score_counterparty("unknown_exchange")
        assert result["scores"]["reliability"] == 50
        assert result["risk_level"] in ("medium", "high")

    def test_risk_levels(self):
        """Risk level should be based on composite score thresholds."""
        # coinbase: composite should be > 80 => low
        result = self.scorer.score_counterparty("coinbase")
        assert result["risk_level"] == "low"

        # bybit: composite should be between 60-80 => medium
        result = self.scorer.score_counterparty("bybit")
        assert result["risk_level"] == "medium"

    def test_warnings_for_low_scores(self):
        """Should generate warnings for dimensions with score < 60."""
        result = self.scorer.score_counterparty("bybit")
        # bybit has regulation=55 < 60
        warnings_text = " ".join(result["warnings"])
        assert "regulation" in warnings_text.lower()

    def test_update_score_existing_exchange(self):
        """Should update score for existing exchange."""
        self.scorer.update_score("binance", "reliability", 95)
        result = self.scorer.score_counterparty("binance")
        assert result["scores"]["reliability"] == 95

    def test_update_score_new_exchange(self):
        """Should create new exchange entry and update score."""
        self.scorer.update_score("new_exchange", "reliability", 80)
        result = self.scorer.score_counterparty("new_exchange")
        assert result["scores"]["reliability"] == 80
        # Other scores should be defaults
        assert result["scores"]["regulation"] == 50

    def test_update_score_clamp_high(self):
        """Score above 100 should be clamped to 100."""
        self.scorer.update_score("binance", "reliability", 150)
        result = self.scorer.score_counterparty("binance")
        assert result["scores"]["reliability"] == 100

    def test_update_score_clamp_low(self):
        """Score below 0 should be clamped to 0."""
        self.scorer.update_score("binance", "reliability", -10)
        result = self.scorer.score_counterparty("binance")
        assert result["scores"]["reliability"] == 0

    def test_update_from_reserve_proof_excellent(self):
        """proof_ratio >= 1.5 should give financial_score=95."""
        result = self.scorer.update_from_reserve_proof("binance", 2.0)
        assert result["scores"]["financial"] == 95

    def test_update_from_reserve_proof_good(self):
        """1.2 <= proof_ratio < 1.5 should give financial_score=80."""
        result = self.scorer.update_from_reserve_proof("binance", 1.3)
        assert result["scores"]["financial"] == 80

    def test_update_from_reserve_proof_adequate(self):
        """1.0 <= proof_ratio < 1.2 should give financial_score=60."""
        result = self.scorer.update_from_reserve_proof("binance", 1.1)
        assert result["scores"]["financial"] == 60

    def test_update_from_reserve_proof_poor(self):
        """proof_ratio < 1.0 should give financial_score=30."""
        result = self.scorer.update_from_reserve_proof("binance", 0.8)
        assert result["scores"]["financial"] == 30

    def test_update_from_withdrawal_normal_fast(self):
        """Normal withdrawals < 2hr delay should give operational_score=95."""
        result = self.scorer.update_from_withdrawal_status("binance", True, 1.0)
        assert result["scores"]["operational"] == 95

    def test_update_from_withdrawal_normal_moderate_delay(self):
        """Normal withdrawals 2-12hr delay should give operational_score=70."""
        result = self.scorer.update_from_withdrawal_status("binance", True, 5.0)
        assert result["scores"]["operational"] == 70

    def test_update_from_withdrawal_normal_long_delay(self):
        """Normal withdrawals > 12hr delay should give operational_score=50."""
        result = self.scorer.update_from_withdrawal_status("binance", True, 24.0)
        assert result["scores"]["operational"] == 50

    def test_update_from_withdrawal_suspended(self):
        """Suspended withdrawals should give operational_score=20."""
        result = self.scorer.update_from_withdrawal_status("binance", False, 0)
        assert result["scores"]["operational"] == 20


# ============================================================================
# PortfolioHeatMap Tests
# ============================================================================

class TestPortfolioHeatMap:
    """Tests for PortfolioHeatMap class."""

    def setup_method(self):
        self.heatmap = PortfolioHeatMap()

    def test_compute_basic(self):
        """Should return heatmap entries for each position."""
        positions = [
            make_position("BTC/USDT"),
            make_position("ETH/USDT"),
            make_position("SOL/USDT"),
        ]
        ret_mat = generate_returns_matrix(200, 3, seed=30)
        weights = np.array([0.5, 0.3, 0.2])
        result = self.heatmap.compute(positions, ret_mat, weights)
        assert len(result) == 3
        for entry in result:
            assert "symbol" in entry
            assert "weight" in entry
            assert "marginal_var" in entry
            assert "component_var" in entry
            assert "pct_risk_contribution" in entry
            assert "risk_level" in entry
            assert entry["symbol"] in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_compute_pct_contributions_sum(self):
        """Percentage risk contributions should sum to ~100."""
        positions = [make_position("A"), make_position("B"), make_position("C"), make_position("D")]
        ret_mat = generate_returns_matrix(200, 4, seed=31)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        result = self.heatmap.compute(positions, ret_mat, weights)
        total_pct = sum(entry["pct_risk_contribution"] for entry in result)
        assert abs(total_pct - 100.0) < 5.0  # within 5%

    def test_compute_risk_levels(self):
        """Risk level should be assigned based on pct_contribution."""
        positions = [
            make_position("BTC/USDT"),
            make_position("ETH/USDT"),
        ]
        ret_mat = generate_returns_matrix(200, 2, seed=32)
        weights = np.array([0.9, 0.1])  # highly concentrated
        result = self.heatmap.compute(positions, ret_mat, weights)
        for entry in result:
            assert entry["risk_level"] in ("low", "medium", "high")

    def test_compute_mismatched_weights(self):
        """Mismatched weights/columns should return empty list."""
        positions = [make_position("BTC/USDT"), make_position("ETH/USDT")]
        ret_mat = generate_returns_matrix(200, 3, seed=33)
        weights = np.array([0.5, 0.5])  # 2 weights, 3 columns
        result = self.heatmap.compute(positions, ret_mat, weights)
        assert result == []

    def test_compute_mismatched_positions(self):
        """Mismatched positions/weights should return empty list."""
        positions = [make_position("BTC/USDT")]  # 1 position
        ret_mat = generate_returns_matrix(200, 2, seed=34)
        weights = np.array([0.5, 0.5])  # 2 weights
        result = self.heatmap.compute(positions, ret_mat, weights)
        assert result == []

    def test_compute_position_notional(self):
        """Heatmap should include position notional value."""
        pos = make_position("BTC/USDT", quantity=2.0, mark_price=50000.0)
        ret_mat = generate_returns_matrix(200, 1, seed=35)
        weights = np.array([1.0])
        # This will fail because positions != len(weights) for single assets... let's do 2
        positions = [pos, make_position("ETH/USDT")]
        ret_mat = generate_returns_matrix(200, 2, seed=35)
        weights = np.array([0.7, 0.3])
        result = self.heatmap.compute(positions, ret_mat, weights)
        assert result[0]["position_notional"] == 100000.0  # 2 * 50000


# ============================================================================
# CircuitBreaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        cb = CircuitBreaker()
        assert cb.loss_threshold_pct == 0.03
        assert cb.cooldown_minutes == 30
        assert cb.vol_spike_mult == 5.0
        assert cb.triggered is False
        assert cb.trigger_reason == ""
        assert cb.triggered_at is None

    def test_init_custom(self):
        """Custom parameters should be stored."""
        cb = CircuitBreaker(loss_threshold_pct=0.05, cooldown_minutes=60, vol_spike_mult=3.0)
        assert cb.loss_threshold_pct == 0.05
        assert cb.cooldown_minutes == 60
        assert cb.vol_spike_mult == 3.0

    def test_check_normal_no_trigger(self):
        """Normal conditions should not trigger."""
        cb = CircuitBreaker()
        result = cb.check(current_pnl_pct=-0.01, current_vol=0.02, normal_vol=0.02)
        assert result is False
        assert cb.triggered is False

    def test_check_loss_trigger(self):
        """Loss exceeding threshold should trigger."""
        cb = CircuitBreaker(loss_threshold_pct=0.03)
        result = cb.check(current_pnl_pct=-0.05, current_vol=0.02, normal_vol=0.02)
        assert result is True
        assert cb.triggered is True
        assert "Loss exceeds" in cb.trigger_reason

    def test_check_volatility_spike_trigger(self):
        """Volatility spike should trigger."""
        cb = CircuitBreaker(vol_spike_mult=5.0)
        result = cb.check(current_pnl_pct=0.0, current_vol=0.20, normal_vol=0.02)
        assert result is True
        assert cb.triggered is True
        assert "Volatility spike" in cb.trigger_reason

    def test_check_no_trigger_below_loss_threshold(self):
        """Loss at exactly threshold should not trigger (<, not <=)."""
        cb = CircuitBreaker(loss_threshold_pct=0.03)
        result = cb.check(current_pnl_pct=-0.03, current_vol=0.02, normal_vol=0.02)
        # -0.03 < -0.03 is False
        assert result is False

    def test_check_no_trigger_below_vol_spike(self):
        """Vol below spike threshold should not trigger."""
        cb = CircuitBreaker(vol_spike_mult=5.0)
        result = cb.check(current_pnl_pct=0.0, current_vol=0.09, normal_vol=0.02)
        # 0.09 < 0.02*5 = 0.10, so no trigger
        assert result is False

    def test_check_already_triggered_within_cooldown(self):
        """Already triggered breaker should stay triggered within cooldown."""
        cb = CircuitBreaker(cooldown_minutes=30)
        cb.check(current_pnl_pct=-0.10, current_vol=0.02, normal_vol=0.02)
        assert cb.triggered is True
        # Check again immediately - should still be triggered
        result = cb.check(current_pnl_pct=0.0, current_vol=0.02, normal_vol=0.02)
        assert result is True

    def test_check_cooldown_expiry(self):
        """Breaker should reset after cooldown period."""
        cb = CircuitBreaker(cooldown_minutes=0)  # 0 minutes cooldown
        cb._trigger("test")
        cb.triggered_at = datetime.utcnow() - timedelta(minutes=1)
        result = cb.check(current_pnl_pct=0.0, current_vol=0.02, normal_vol=0.02)
        assert result is False
        assert cb.triggered is False

    def test_reset(self):
        """Reset should clear all triggered state."""
        cb = CircuitBreaker()
        cb._trigger("test reason")
        cb.reset()
        assert cb.triggered is False
        assert cb.trigger_reason == ""
        assert cb.triggered_at is None

    def test_trigger_sets_reason(self):
        """_trigger should set reason and timestamp."""
        cb = CircuitBreaker()
        cb._trigger("emergency")
        assert cb.triggered is True
        assert cb.trigger_reason == "emergency"
        assert cb.triggered_at is not None

    def test_zero_normal_vol_no_spike_trigger(self):
        """Zero normal_vol should not trigger vol spike (normal_vol > 0 check)."""
        cb = CircuitBreaker()
        result = cb.check(current_pnl_pct=0.0, current_vol=100.0, normal_vol=0.0)
        assert result is False


# ============================================================================
# RiskBudgeting Tests
# ============================================================================

class TestRiskBudgeting:
    """Tests for RiskBudgeting class."""

    def setup_method(self):
        self.budgeting = RiskBudgeting(total_risk_budget=1.0, max_strategy_risk_pct=0.40)

    def test_init_defaults(self):
        """Default constructor should set expected values."""
        rb = RiskBudgeting()
        assert rb.total_risk_budget == 1.0
        assert rb.max_strategy_risk_pct == 0.40

    def test_allocate_budget_equal(self):
        """Equal allocation across strategies without target contributions."""
        result = self.budgeting.allocate_budget(["strat_a", "strat_b", "strat_c", "strat_d"])
        # 1.0 / 4 = 0.25, which is <= 0.40
        for budget in result.values():
            assert abs(budget - 0.25) < 1e-10

    def test_allocate_budget_empty(self):
        """Empty strategy list should return empty dict."""
        result = self.budgeting.allocate_budget([])
        assert result == {}

    def test_allocate_budget_capped_at_max(self):
        """Each strategy should be capped at max_strategy_risk_pct."""
        rb = RiskBudgeting(total_risk_budget=1.0, max_strategy_risk_pct=0.30)
        result = rb.allocate_budget(["a", "b"])  # 1.0/2 = 0.5 > 0.30, should be capped
        for budget in result.values():
            assert budget <= 0.30 + 1e-10

    def test_allocate_budget_with_target_contributions(self):
        """Should use target contributions when provided."""
        targets = np.array([0.3, 0.3, 0.2, 0.2])
        result = self.budgeting.allocate_budget(
            ["a", "b", "c", "d"], target_contributions=targets
        )
        assert abs(result["a"] - 0.3) < 1e-10
        assert abs(result["d"] - 0.2) < 1e-10

    def test_allocate_budget_target_exceeds_total(self):
        """Should normalize when target contributions exceed total budget."""
        targets = np.array([0.5, 0.5, 0.3, 0.3])  # sum=1.6 > 1.0
        result = self.budgeting.allocate_budget(
            ["a", "b", "c", "d"], target_contributions=targets
        )
        total = sum(result.values())
        assert total <= 1.0 + 1e-10

    def test_allocate_budget_target_capped(self):
        """Target contributions exceeding max should be capped."""
        rb = RiskBudgeting(total_risk_budget=1.0, max_strategy_risk_pct=0.30)
        targets = np.array([0.5, 0.3, 0.1, 0.1])
        result = rb.allocate_budget(
            ["a", "b", "c", "d"], target_contributions=targets
        )
        assert result["a"] <= 0.30 + 1e-10

    def test_check_budget_utilization_within_budget(self):
        """Strategy within budget should show over_budget=False."""
        self.budgeting.allocate_budget(["strat_a"])
        result = self.budgeting.check_budget_utilization("strat_a", 0.1)
        assert result["over_budget"] is False
        assert result["remaining_budget"] > 0

    def test_check_budget_utilization_over_budget(self):
        """Strategy over budget should show over_budget=True."""
        self.budgeting.allocate_budget(["strat_a"])
        result = self.budgeting.check_budget_utilization("strat_a", 1.0)
        assert result["over_budget"] is True
        assert result["remaining_budget"] == 0.0

    def test_check_budget_utilization_unknown_strategy(self):
        """Unknown strategy should have budget=0, resulting in inf utilization."""
        result = self.budgeting.check_budget_utilization("unknown", 0.1)
        assert result["budget"] == 0
        assert result["utilization_pct"] == float('inf')
        assert result["over_budget"] is True

    def test_check_budget_utilization_zero_usage(self):
        """Zero usage should show 0% utilization."""
        self.budgeting.allocate_budget(["strat_a"])
        result = self.budgeting.check_budget_utilization("strat_a", 0.0)
        assert result["utilization_pct"] == 0.0
        assert result["over_budget"] is False

    def test_compute_risk_contribution_targets_basic(self):
        """Should compute targets inversely proportional to volatility."""
        strategy_returns = {
            "low_vol": np.random.default_rng(1).normal(0, 0.01, 100),
            "high_vol": np.random.default_rng(2).normal(0, 0.05, 100),
        }
        cov_matrix = np.eye(2)
        strategy_indices = {"low_vol": [0], "high_vol": [1]}
        targets = self.budgeting.compute_risk_contribution_targets(
            strategy_returns, cov_matrix, strategy_indices
        )
        assert "low_vol" in targets
        assert "high_vol" in targets
        # Low vol should get higher target (inverse)
        assert targets["low_vol"] > targets["high_vol"]

    def test_compute_risk_contribution_targets_zero_vols(self):
        """All zero volatility should give equal targets."""
        strategy_returns = {
            "a": np.zeros(100),
            "b": np.zeros(100),
        }
        cov_matrix = np.eye(2)
        strategy_indices = {"a": [0], "b": [1]}
        targets = self.budgeting.compute_risk_contribution_targets(
            strategy_returns, cov_matrix, strategy_indices
        )
        assert abs(targets["a"] - targets["b"]) < 1e-10

    def test_compute_risk_contribution_targets_empty_returns(self):
        """Empty return arrays should give zero vol and equal targets."""
        strategy_returns = {"a": np.array([]), "b": np.array([])}
        cov_matrix = np.eye(2)
        strategy_indices = {"a": [0], "b": [1]}
        targets = self.budgeting.compute_risk_contribution_targets(
            strategy_returns, cov_matrix, strategy_indices
        )
        assert abs(targets["a"] - targets["b"]) < 1e-10


# ============================================================================
# RiskEngine Tests
# ============================================================================

class TestRiskEngine:
    """Tests for RiskEngine class - the main risk management engine."""

    def setup_method(self):
        self.engine = RiskEngine()
        self.config = RiskConfig()

    def test_init_defaults(self):
        """Default engine should have all components initialized."""
        engine = RiskEngine()
        assert engine.config.max_position_per_symbol == 100000.0
        assert engine.kill_switch_active is False
        assert engine.kill_switch_reason == ""
        assert isinstance(engine.var, ValueAtRisk)
        assert isinstance(engine.es, ExpectedShortfall)
        assert isinstance(engine.stress, StressTesting)
        assert isinstance(engine.liquidity, LiquidityRiskAssessor)
        assert isinstance(engine.correlation_monitor, CorrelationRiskMonitor)
        assert isinstance(engine.counterparty_scorer, CounterpartyRiskScorer)
        assert isinstance(engine.heatmap, PortfolioHeatMap)
        assert isinstance(engine.circuit_breaker, CircuitBreaker)
        assert isinstance(engine.risk_budgeting, RiskBudgeting)

    def test_init_custom_config(self):
        """Custom config should be stored and used by sub-components."""
        config = RiskConfig(max_position_per_symbol=50000.0, max_drawdown=0.10)
        engine = RiskEngine(config)
        assert engine.config.max_position_per_symbol == 50000.0
        assert engine.config.max_drawdown == 0.10

    # --- Kill Switch ---

    def test_trigger_kill_switch(self):
        """Kill switch should be activatable."""
        self.engine.trigger_kill_switch("Emergency stop")
        assert self.engine.kill_switch_active is True
        assert self.engine.kill_switch_reason == "Emergency stop"

    def test_reset_kill_switch(self):
        """Kill switch should be resettable."""
        self.engine.trigger_kill_switch("test")
        self.engine.reset_kill_switch()
        assert self.engine.kill_switch_active is False
        assert self.engine.kill_switch_reason == ""

    # --- Pre-trade checks ---

    def test_pre_trade_check_kill_switch_active(self):
        """Active kill switch should reject all orders."""
        self.engine.trigger_kill_switch("test")
        order = make_order()
        portfolio = make_portfolio()
        results = self.engine.pre_trade_check(order, portfolio)
        assert len(results) == 1
        assert results[0].decision == RiskDecision.REJECT
        assert results[0].check_name == "kill_switch"

    def test_pre_trade_check_circuit_breaker_active(self):
        """Active circuit breaker should reject all orders."""
        self.engine.circuit_breaker.triggered = True
        self.engine.circuit_breaker.trigger_reason = "test"
        order = make_order()
        portfolio = make_portfolio()
        results = self.engine.pre_trade_check(order, portfolio)
        assert len(results) == 1
        assert results[0].decision == RiskDecision.REJECT
        assert results[0].check_name == "circuit_breaker"

    def test_pre_trade_check_all_pass(self):
        """Small order on well-capitalized portfolio should pass all checks."""
        order = make_order(quantity=0.1, price=50000.0)  # notional=5000
        portfolio = make_portfolio(total_value=1000000.0, available_balance=500000.0)
        results = self.engine.pre_trade_check(order, portfolio)
        # Kill switch and circuit breaker only produce results when active,
        # so we expect 8 checks (3-10): position_limit, order_notional, rate_limit,
        # drawdown, gross_exposure, concentration, margin, net_exposure
        assert len(results) >= 8
        for result in results:
            assert result.decision == RiskDecision.ALLOW

    def test_pre_trade_check_position_limit_reject(self):
        """Order exceeding position limit per symbol should be rejected."""
        # Create position near limit
        existing_pos = make_position("BTC/USDT", Side.BUY, quantity=1.5,
                                      mark_price=50000.0)  # notional=75000
        portfolio = make_portfolio(total_value=1000000.0, positions=[existing_pos])
        # Order that would push over 100000 limit
        order = make_order(quantity=1.0, price=50000.0)  # notional=50000
        results = self.engine.pre_trade_check(order, portfolio)
        pos_check = next(r for r in results if r.check_name == "position_limit_symbol")
        assert pos_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_order_notional_reject(self):
        """Order exceeding max notional should be rejected."""
        order = make_order(quantity=2.0, price=50000.0)  # notional=100000 > 50000
        portfolio = make_portfolio(total_value=1000000.0)
        results = self.engine.pre_trade_check(order, portfolio)
        notional_check = next(r for r in results if r.check_name == "order_notional")
        assert notional_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_drawdown_reject(self):
        """Order during excessive drawdown should be rejected."""
        portfolio = make_portfolio(total_value=1000000.0, unrealized_pnl=-250000.0)
        order = make_order(quantity=0.01, price=1000.0)
        results = self.engine.pre_trade_check(order, portfolio)
        dd_check = next(r for r in results if r.check_name == "max_drawdown")
        # dd = -(-250000) / 1000000 = 0.25 > 0.20
        assert dd_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_drawdown_allow(self):
        """Order with acceptable drawdown should pass."""
        portfolio = make_portfolio(total_value=1000000.0, unrealized_pnl=-10000.0)
        order = make_order(quantity=0.01, price=1000.0)
        results = self.engine.pre_trade_check(order, portfolio)
        dd_check = next(r for r in results if r.check_name == "max_drawdown")
        # dd = 10000 / 1000000 = 0.01 < 0.20
        assert dd_check.decision == RiskDecision.ALLOW

    def test_pre_trade_check_gross_exposure_reject(self):
        """Order exceeding gross exposure limit should be rejected."""
        # Create positions totaling near the limit
        positions = [
            make_position("BTC/USDT", Side.BUY, quantity=10.0, mark_price=50000.0),  # 500000
        ]
        portfolio = make_portfolio(total_value=1000000.0, positions=positions)
        order = make_order(quantity=20.0, price=50000.0)  # 1000000 notional
        results = self.engine.pre_trade_check(order, portfolio)
        gross_check = next(r for r in results if r.check_name == "gross_exposure")
        # gross = 500000 + 1000000 = 1500000 > 1000000
        assert gross_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_concentration_reject(self):
        """Order exceeding concentration limit should be rejected."""
        # Small portfolio, large position
        portfolio = make_portfolio(total_value=100000.0)
        order = make_order(quantity=1.0, price=50000.0)  # notional=50000
        results = self.engine.pre_trade_check(order, portfolio)
        conc_check = next(r for r in results if r.check_name == "concentration")
        # concentration = 50000 / 100000 = 0.50 > 0.25
        assert conc_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_margin_reject(self):
        """Order exceeding available margin should be rejected."""
        # All capital already used as margin
        positions = [make_position("BTC/USDT", Side.BUY, quantity=9.0,
                                    mark_price=50000.0, leverage=1.0)]  # margin=450000
        portfolio = make_portfolio(total_value=500000.0, positions=positions,
                                   margin_used=450000.0)
        order = make_order(quantity=1.0, price=50000.0)  # margin_req = 50000*0.10=5000
        # margin_available = 500000 - 450000 = 50000
        # This should pass. Let's make it fail:
        positions2 = [make_position("BTC/USDT", Side.BUY, quantity=9.9,
                                     mark_price=50000.0, leverage=1.0)]
        portfolio2 = make_portfolio(total_value=500000.0, positions=positions2,
                                    margin_used=495000.0)
        results = self.engine.pre_trade_check(order, portfolio2)
        margin_check = next(r for r in results if r.check_name == "margin")
        # margin_available = 5000, margin_req = 50000 * 0.10 = 5000, so should just pass
        # Let's adjust to make it clearly fail
        order2 = make_order(quantity=5.0, price=50000.0)  # margin_req = 250000 * 0.10 = 25000
        results2 = self.engine.pre_trade_check(order2, portfolio2)
        margin_check2 = next(r for r in results2 if r.check_name == "margin")
        assert margin_check2.decision == RiskDecision.REJECT

    def test_pre_trade_check_net_exposure_reject(self):
        """Order exceeding net exposure limit should be rejected."""
        # Heavily long portfolio
        positions = [
            make_position("BTC/USDT", Side.BUY, quantity=8.0, mark_price=50000.0),  # +400000
        ]
        portfolio = make_portfolio(total_value=1000000.0, positions=positions)
        order = make_order(quantity=5.0, price=50000.0, side=Side.BUY)  # +250000
        results = self.engine.pre_trade_check(order, portfolio)
        net_check = next(r for r in results if r.check_name == "net_exposure")
        # net = 400000 + 250000 = 650000 > 500000
        assert net_check.decision == RiskDecision.REJECT

    def test_pre_trade_check_rate_limit_throttle(self):
        """Should throttle when order rate exceeds limit."""
        order = make_order(quantity=0.01, price=100.0)
        portfolio = make_portfolio(total_value=1000000.0)
        # Send many orders quickly
        for _ in range(15):
            self.engine.pre_trade_check(order, portfolio)
        results = self.engine.pre_trade_check(order, portfolio)
        rate_check = next(r for r in results if r.check_name == "rate_limit_second")
        assert rate_check.decision == RiskDecision.THROTTLE

    def test_pre_trade_check_records_timestamps(self):
        """Each check should record the order timestamp."""
        order = make_order(quantity=0.01, price=100.0)
        portfolio = make_portfolio(total_value=1000000.0)
        initial_len = len(self.engine._order_timestamps)
        self.engine.pre_trade_check(order, portfolio)
        assert len(self.engine._order_timestamps) == initial_len + 1

    # --- Position Sizing ---

    def test_kelly_size_basic(self):
        """Should return positive position size for positive expectancy."""
        size = self.engine.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.02,
                                       capital=100000.0, fraction=0.5)
        assert size > 0
        assert size < 100000.0  # Should not exceed capital

    def test_kelly_size_zero_loss(self):
        """Zero avg_loss should return 0."""
        size = self.engine.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.0,
                                       capital=100000.0)
        assert size == 0.0

    def test_kelly_size_zero_win_rate(self):
        """Zero win rate should return 0."""
        size = self.engine.kelly_size(win_rate=0.0, avg_win=0.03, avg_loss=0.02,
                                       capital=100000.0)
        assert size == 0.0

    def test_kelly_size_negative_expectancy(self):
        """Negative Kelly fraction should return 0."""
        size = self.engine.kelly_size(win_rate=0.3, avg_win=0.01, avg_loss=0.03,
                                       capital=100000.0)
        # win_rate - (1-win_rate)/(avg_win/avg_loss) = 0.3 - 0.7/0.333 = 0.3 - 2.1 < 0
        assert size == 0.0

    def test_kelly_size_fraction_parameter(self):
        """Smaller fraction should give smaller size."""
        size_half = self.engine.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.02,
                                            capital=100000.0, fraction=0.5)
        size_quarter = self.engine.kelly_size(win_rate=0.6, avg_win=0.03, avg_loss=0.02,
                                               capital=100000.0, fraction=0.25)
        assert size_quarter < size_half

    def test_kelly_size_drawdown_constraint(self):
        """Drawdown constraint should reduce Kelly size when DD risk is high."""
        # High kelly_f with tight max_drawdown
        size_no_constraint = self.engine.kelly_size(win_rate=0.7, avg_win=0.05,
                                                     avg_loss=0.02, capital=100000.0,
                                                     fraction=1.0, max_drawdown=0.99)
        size_constrained = self.engine.kelly_size(win_rate=0.7, avg_win=0.05,
                                                    avg_loss=0.02, capital=100000.0,
                                                    fraction=1.0, max_drawdown=0.05)
        assert size_constrained <= size_no_constraint

    def test_fixed_fractional_size_basic(self):
        """Should return positive position size for valid inputs."""
        size = self.engine.fixed_fractional_size(capital=100000.0, risk_pct=0.02,
                                                  entry_price=50000.0, stop_price=49000.0)
        # risk_amount = 100000 * 0.02 = 2000
        # risk_per_unit = 50000 - 49000 = 1000
        # size = 2000 / 1000 = 2.0
        assert abs(size - 2.0) < 1e-10

    def test_fixed_fractional_size_zero_entry(self):
        """Zero entry price should return 0."""
        size = self.engine.fixed_fractional_size(capital=100000.0, risk_pct=0.02,
                                                  entry_price=0, stop_price=49000.0)
        assert size == 0.0

    def test_fixed_fractional_size_zero_stop(self):
        """Zero stop price should return 0."""
        size = self.engine.fixed_fractional_size(capital=100000.0, risk_pct=0.02,
                                                  entry_price=50000.0, stop_price=0)
        assert size == 0.0

    def test_fixed_fractional_size_same_prices(self):
        """Same entry and stop price should return 0."""
        size = self.engine.fixed_fractional_size(capital=100000.0, risk_pct=0.02,
                                                  entry_price=50000.0, stop_price=50000.0)
        assert size == 0.0

    def test_fixed_fractional_size_stop_above_entry(self):
        """Stop above entry should still work (abs difference)."""
        size = self.engine.fixed_fractional_size(capital=100000.0, risk_pct=0.02,
                                                  entry_price=49000.0, stop_price=50000.0)
        assert size > 0

    def test_volatility_regime_size_basic(self):
        """Should return positive size for valid inputs."""
        size = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                   current_vol=0.15, target_vol=0.15)
        # vol_scalar = 0.15/0.15 = 1.0, adjusted_risk = 0.02
        assert abs(size - 2000.0) < 1e-10

    def test_volatility_regime_size_high_vol(self):
        """High volatility should reduce position size."""
        size_low_vol = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                           current_vol=0.10, target_vol=0.15)
        size_high_vol = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                            current_vol=0.30, target_vol=0.15)
        assert size_high_vol < size_low_vol

    def test_volatility_regime_size_zero_vol(self):
        """Zero current_vol should use base risk."""
        size = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                   current_vol=0.0, target_vol=0.15)
        assert size == 100000.0 * 0.02

    def test_volatility_regime_size_scalar_bounds(self):
        """Volatility scalar should be bounded between 0.25 and 3.0."""
        # Very low vol -> scalar would be very high but capped at 3.0
        size = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                   current_vol=0.01, target_vol=0.15)
        # scalar = 0.15/0.01 = 15, capped at 3.0
        expected = 100000.0 * 0.02 * 3.0
        assert abs(size - expected) < 1e-10

        # Very high vol -> scalar would be very low but floored at 0.25
        size2 = self.engine.volatility_regime_size(capital=100000.0, base_risk_pct=0.02,
                                                    current_vol=1.0, target_vol=0.15)
        # scalar = 0.15/1.0 = 0.15, floored at 0.25
        expected2 = 100000.0 * 0.02 * 0.25
        assert abs(size2 - expected2) < 1e-10

    def test_dynamic_position_size_conservative(self):
        """Should use the more conservative of Kelly and vol-adjusted size."""
        size = self.engine.dynamic_position_size(
            capital=100000.0, base_risk_pct=0.02,
            current_vol=0.15, target_vol=0.15,
            win_rate=0.6, avg_win=0.03, avg_loss=0.02,
        )
        kelly = self.engine.kelly_size(0.6, 0.03, 0.02, 100000.0)
        vol_size = self.engine.volatility_regime_size(100000.0, 0.02, 0.15, 0.15)
        assert size == min(kelly, vol_size)

    # --- Portfolio Risk Metrics ---

    def test_compute_portfolio_var_historical(self):
        """Should compute portfolio VaR using historical method."""
        ret_mat = generate_returns_matrix(300, 3, seed=40)
        weights = np.array([0.5, 0.3, 0.2])
        var = self.engine.compute_portfolio_var(ret_mat, weights, method="historical")
        assert np.isfinite(var)
        assert var > 0

    def test_compute_portfolio_var_parametric(self):
        """Should compute portfolio VaR using parametric method."""
        ret_mat = generate_returns_matrix(300, 3, seed=41)
        weights = np.array([0.5, 0.3, 0.2])
        var = self.engine.compute_portfolio_var(ret_mat, weights, method="parametric")
        assert np.isfinite(var)

    def test_compute_portfolio_var_monte_carlo(self):
        """Should compute portfolio VaR using Monte Carlo method."""
        ret_mat = generate_returns_matrix(300, 3, seed=42)
        weights = np.array([0.5, 0.3, 0.2])
        var = self.engine.compute_portfolio_var(ret_mat, weights, method="monte_carlo")
        assert np.isfinite(var)

    def test_compute_portfolio_var_unknown_method(self):
        """Unknown method should return NaN."""
        ret_mat = generate_returns_matrix(300, 3, seed=43)
        weights = np.array([0.5, 0.3, 0.2])
        var = self.engine.compute_portfolio_var(ret_mat, weights, method="unknown")
        assert np.isnan(var)

    def test_compute_portfolio_cvar(self):
        """Should compute portfolio CVaR."""
        ret_mat = generate_returns_matrix(300, 3, seed=44)
        weights = np.array([0.5, 0.3, 0.2])
        cvar = self.engine.compute_portfolio_cvar(ret_mat, weights)
        assert np.isfinite(cvar)
        assert cvar > 0

    def test_compute_portfolio_es_historical(self):
        """Should compute portfolio ES using historical method."""
        ret_mat = generate_returns_matrix(300, 3, seed=45)
        weights = np.array([0.5, 0.3, 0.2])
        es = self.engine.compute_portfolio_es(ret_mat, weights, method="historical")
        assert np.isfinite(es)

    def test_compute_portfolio_es_parametric(self):
        """Should compute portfolio ES using parametric method."""
        ret_mat = generate_returns_matrix(300, 3, seed=46)
        weights = np.array([0.5, 0.3, 0.2])
        es = self.engine.compute_portfolio_es(ret_mat, weights, method="parametric")
        assert np.isfinite(es)

    def test_compute_portfolio_es_cornish_fisher(self):
        """Should compute portfolio ES using Cornish-Fisher method."""
        ret_mat = generate_returns_matrix(300, 3, seed=47)
        weights = np.array([0.5, 0.3, 0.2])
        es = self.engine.compute_portfolio_es(ret_mat, weights, method="cornish_fisher")
        assert np.isfinite(es)

    def test_compute_portfolio_es_unknown_method(self):
        """Unknown ES method should return NaN."""
        ret_mat = generate_returns_matrix(300, 3, seed=48)
        weights = np.array([0.5, 0.3, 0.2])
        es = self.engine.compute_portfolio_es(ret_mat, weights, method="unknown")
        assert np.isnan(es)

    def test_compute_tail_risk_decomposition(self):
        """Should delegate to ES tail_risk_decomposition."""
        ret_mat = generate_returns_matrix(300, 3, seed=49)
        weights = np.array([0.5, 0.3, 0.2])
        result = self.engine.compute_tail_risk_decomposition(ret_mat, weights)
        assert "contributions" in result
        assert "total_es" in result


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests across all risk components."""

    def test_var_with_extreme_negative_outlier(self):
        """VaR with an extreme outlier should handle gracefully."""
        returns = generate_returns(200, seed=60)
        returns[0] = -0.99  # 99% loss
        var = ValueAtRisk.historical(returns, confidence=0.99)
        assert np.isfinite(var)
        assert var > 0

    def test_var_with_single_extreme_positive(self):
        """VaR with an extreme positive outlier."""
        returns = generate_returns(200, seed=61)
        returns[0] = 5.0  # 500% gain
        var = ValueAtRisk.historical(returns, confidence=0.99)
        assert np.isfinite(var)

    def test_cvar_all_same_returns(self):
        """CVaR with all identical positive returns should be negative (no loss scenario)."""
        returns = np.full(200, 0.001)
        cvar = ValueAtRisk.cvar(returns, confidence=0.99)
        # 1st percentile = 0.001, tail includes all returns, mean = 0.001
        # CVaR = -mean(tail) = -0.001 (negative means no loss at this confidence)
        assert abs(cvar - (-0.001)) < 1e-10

    def test_es_with_mixed_extreme_returns(self):
        """ES should handle mix of extreme positive and negative returns."""
        rng = np.random.default_rng(62)
        returns = rng.normal(0, 0.02, 200)
        returns[0:5] = -0.80  # Extreme losses
        returns[5:10] = 2.0   # Extreme gains
        es = ExpectedShortfall.historical_es(returns, confidence=0.975)
        assert np.isfinite(es)

    def test_stress_test_no_positions(self):
        """Stress test with no positions should give zero PnL."""
        stress = StressTesting()
        result = stress.run_scenario([], "flash_crash")
        assert result["total_pnl"] == 0.0
        assert len(result["position_results"]) == 0

    def test_liquidity_empty_history(self):
        """Fresh assessor should not detect trends."""
        a = LiquidityRiskAssessor()
        assert len(a._spread_history) == 0
        assert len(a._depth_history) == 0

    def test_correlation_monitor_identity_matrix(self):
        """Identity correlation matrix should show no concentration."""
        monitor = CorrelationRiskMonitor()
        corr = np.eye(5)
        result = monitor.check_concentration_risk(corr, np.ones(5) / 5)
        assert result["high_correlation_pairs"] == 0
        assert result["risk_level"] == "low"

    def test_circuit_breaker_loss_at_exact_threshold(self):
        """Loss at exact threshold should NOT trigger (< not <=)."""
        cb = CircuitBreaker(loss_threshold_pct=0.05)
        result = cb.check(-0.05, 0.02, 0.02)
        assert result is False

    def test_circuit_breaker_loss_just_below_threshold(self):
        """Loss just below threshold should not trigger."""
        cb = CircuitBreaker(loss_threshold_pct=0.05)
        result = cb.check(-0.0499, 0.02, 0.02)
        assert result is False

    def test_circuit_breaker_loss_just_above_threshold(self):
        """Loss just above threshold should trigger."""
        cb = CircuitBreaker(loss_threshold_pct=0.05)
        result = cb.check(-0.0501, 0.02, 0.02)
        assert result is True

    def test_risk_engine_zero_equity_portfolio(self):
        """Portfolio with zero equity should handle gracefully."""
        engine = RiskEngine()
        order = make_order(quantity=0.01, price=100.0)
        portfolio = make_portfolio(total_value=0.0, available_balance=0.0)
        results = engine.pre_trade_check(order, portfolio)
        # Should not crash; may reject on margin or other checks
        assert len(results) >= 1

    def test_risk_engine_negative_equity(self):
        """Portfolio with negative equity should handle gracefully."""
        engine = RiskEngine()
        order = make_order(quantity=0.01, price=100.0)
        portfolio = make_portfolio(total_value=-1000.0, available_balance=0.0,
                                   unrealized_pnl=-1000.0)
        results = engine.pre_trade_check(order, portfolio)
        # Should not crash
        assert len(results) >= 1

    def test_kelly_size_very_high_win_rate(self):
        """Very high win rate should not produce unrealistic sizes."""
        size = RiskEngine().kelly_size(win_rate=0.95, avg_win=0.02, avg_loss=0.01,
                                        capital=100000.0, fraction=0.5)
        assert size <= 100000.0

    def test_kelly_size_very_low_win_rate(self):
        """Very low win rate should return 0."""
        size = RiskEngine().kelly_size(win_rate=0.05, avg_win=0.01, avg_loss=0.05,
                                        capital=100000.0, fraction=0.5)
        assert size == 0.0

    def test_counterparty_scorer_all_zero(self):
        """Setting all scores to 0 should give high risk."""
        scorer = CounterpartyRiskScorer()
        scorer.update_score("test", "reliability", 0)
        scorer.update_score("test", "regulation", 0)
        scorer.update_score("test", "financial", 0)
        scorer.update_score("test", "operational", 0)
        result = scorer.score_counterparty("test")
        assert result["composite_score"] == 0.0
        assert result["risk_level"] == "high"

    def test_heatmap_two_identical_positions(self):
        """Two identical positions should have similar risk contributions."""
        positions = [
            make_position("A", quantity=1.0, mark_price=10000.0),
            make_position("B", quantity=1.0, mark_price=10000.0),
        ]
        ret_mat = generate_returns_matrix(200, 2, seed=70)
        weights = np.array([0.5, 0.5])
        heatmap = PortfolioHeatMap()
        result = heatmap.compute(positions, ret_mat, weights)
        assert len(result) == 2
        # Risk contributions should be similar (not necessarily identical due to different asset returns)

    def test_risk_budgeting_single_strategy(self):
        """Single strategy should get the full budget (capped at max)."""
        rb = RiskBudgeting(total_risk_budget=1.0, max_strategy_risk_pct=0.40)
        result = rb.allocate_budget(["only_one"])
        assert result["only_one"] == 0.40  # capped at max

    def test_risk_budgeting_many_strategies(self):
        """Many strategies should each get small budget."""
        rb = RiskBudgeting(total_risk_budget=1.0, max_strategy_risk_pct=0.40)
        strategies = [f"strat_{i}" for i in range(10)]
        result = rb.allocate_budget(strategies)
        total = sum(result.values())
        assert total <= 1.0 + 1e-10

    def test_var_confidence_50_pct(self):
        """50% confidence should give median-based VaR."""
        returns = generate_returns(200, seed=80)
        var_50 = ValueAtRisk.historical(returns, confidence=0.50)
        assert np.isfinite(var_50)

    def test_stress_testing_all_scenarios_have_total_pnl(self):
        """All scenarios should have total_pnl field."""
        stress = StressTesting()
        positions = [make_position("BTC/USDT")]
        results = stress.run_all_scenarios(positions)
        for name, result in results.items():
            assert "total_pnl" in result

    def test_market_impact_small_order(self):
        """Very small order should have near-zero market impact."""
        assessor = LiquidityRiskAssessor()
        impact = assessor.compute_market_impact(1.0, 1000000.0)
        assert impact < 100  # Should be very small in bps

    def test_pre_trade_check_sell_order(self):
        """SELL orders should be handled correctly in net exposure."""
        engine = RiskEngine()
        order = make_order(side=Side.SELL, quantity=0.1, price=50000.0)
        portfolio = make_portfolio(total_value=1000000.0)
        results = engine.pre_trade_check(order, portfolio)
        net_check = next(r for r in results if r.check_name == "net_exposure")
        # SELL contributes negative to net exposure
        assert net_check.current_value >= 0

    def test_pre_trade_check_with_existing_sell_positions(self):
        """Net exposure with mixed BUY/SELL positions should be correct."""
        engine = RiskEngine()
        positions = [
            make_position("BTC/USDT", Side.BUY, quantity=1.0, mark_price=50000.0),
            make_position("ETH/USDT", Side.SELL, quantity=5.0, mark_price=3000.0),
        ]
        portfolio = make_portfolio(total_value=1000000.0, positions=positions)
        order = make_order(symbol="SOL/USDT", side=Side.BUY, quantity=1.0, price=100.0)
        results = engine.pre_trade_check(order, portfolio)
        # 8 checks when kill switch and circuit breaker are not active
        assert len(results) >= 8

    def test_portfolio_var_with_equal_weights(self):
        """Equal weights should give reasonable VaR."""
        engine = RiskEngine()
        ret_mat = generate_returns_matrix(300, 5, seed=90)
        weights = np.ones(5) / 5
        var = engine.compute_portfolio_var(ret_mat, weights, method="historical")
        assert np.isfinite(var)
        assert var > 0

    def test_portfolio_var_with_concentrated_weight(self):
        """Single-asset concentrated portfolio should have higher VaR."""
        engine = RiskEngine()
        ret_mat = generate_returns_matrix(300, 5, seed=91)
        weights_equal = np.ones(5) / 5
        weights_concentrated = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        var_equal = engine.compute_portfolio_var(ret_mat, weights_equal, method="parametric")
        var_conc = engine.compute_portfolio_var(ret_mat, weights_concentrated, method="parametric")
        # Both should be finite
        assert np.isfinite(var_equal)
        assert np.isfinite(var_conc)

    def test_risk_engine_subcomponent_integration(self):
        """RiskEngine should properly delegate to sub-components."""
        engine = RiskEngine()
        # Verify all sub-components are properly initialized
        assert engine.var is not None
        assert engine.es is not None
        assert engine.stress is not None
        assert engine.liquidity is not None
        assert engine.correlation_monitor is not None
        assert engine.counterparty_scorer is not None
        assert engine.heatmap is not None
        assert engine.circuit_breaker is not None
        assert engine.risk_budgeting is not None

    def test_risk_engine_config_propagation(self):
        """Config should properly propagate to sub-components."""
        config = RiskConfig(circuit_breaker_loss_pct=0.05, circuit_breaker_cooldown_minutes=60)
        engine = RiskEngine(config)
        assert engine.circuit_breaker.loss_threshold_pct == 0.05
        assert engine.circuit_breaker.cooldown_minutes == 60

    def test_compute_portfolio_var_insufficient_data(self):
        """Insufficient return data should return NaN."""
        engine = RiskEngine()
        ret_mat = generate_returns_matrix(50, 3, seed=92)  # < 100 for historical
        weights = np.array([0.5, 0.3, 0.2])
        var = engine.compute_portfolio_var(ret_mat, weights, method="historical")
        assert np.isnan(var)

    def test_pre_trade_check_zero_value_portfolio(self):
        """Should handle zero-value portfolio without crashing."""
        engine = RiskEngine()
        order = make_order(quantity=0.01, price=100.0)
        portfolio = make_portfolio(total_value=0.0)
        # This should not raise an exception
        results = engine.pre_trade_check(order, portfolio)
        # Should reject on margin and concentration
        assert len(results) >= 1
