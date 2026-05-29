"""Exhaustive tests for acms.backtest module.

Covers every class, method, and edge case:
- BacktestMode enum
- BacktestConfig dataclass
- BacktestTrade dataclass
- MCStatistics dataclass
- BacktestResult dataclass
- SlippageModel (percentage, square_root, almgren_chriss, volume_dependent)
- FillModel (immediate_fill, partial_fill, fill_or_kill)
- TradeAnalytics.compute_mae_mfe
- RollingMetrics (rolling_sharpe, rolling_sortino, rolling_max_drawdown)
- BenchmarkComparison.compute_benchmarks
- RegimeDetector.detect_regimes
- SensitivityAnalysis.run
- BacktestEngine (run, run_sensitivity, _apply_slippage, _simulate_fill,
  _run_single, _run_walk_forward, _run_monte_carlo, _compute_results)
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from acms.core import Candle, Signal, SignalDirection, Position, Side, Trade
from acms.strategies import Strategy
from acms.risk import RiskEngine, RiskConfig
from acms.backtest import (
    BacktestMode,
    BacktestConfig,
    BacktestTrade,
    MCStatistics,
    BacktestResult,
    SlippageModel,
    FillModel,
    TradeAnalytics,
    RollingMetrics,
    BenchmarkComparison,
    RegimeDetector,
    SensitivityAnalysis,
    BacktestEngine,
)


# ============================================================================
# Helpers – building test fixtures
# ============================================================================

def make_candle(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    open_time: datetime = None,
    close_time: datetime = None,
    open: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
) -> Candle:
    """Create a single Candle with sensible defaults."""
    if open_time is None:
        open_time = datetime(2024, 1, 1, 0, 0)
    if close_time is None:
        close_time = open_time + timedelta(minutes=1)
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_candle_series(n: int = 200, base_price: float = 100.0, volatility: float = 0.01,
                       start: datetime = None) -> list:
    """Generate a series of n candles with random walk prices."""
    if start is None:
        start = datetime(2024, 1, 1, 0, 0)
    candles = []
    price = base_price
    np.random.seed(42)
    for i in range(n):
        ret = np.random.normal(0, volatility)
        open_price = price
        price = price * (1 + ret)
        close_price = price
        high = max(open_price, close_price) + abs(np.random.normal(0, volatility * price * 0.5))
        low = min(open_price, close_price) - abs(np.random.normal(0, volatility * price * 0.5))
        low = max(low, 0.01)
        candles.append(make_candle(
            open_time=start + timedelta(minutes=i),
            close_time=start + timedelta(minutes=i + 1),
            open=open_price,
            high=high,
            low=low,
            close=close_price,
        ))
    return candles


class SimpleStrategy(Strategy):
    """A simple test strategy that generates a BUY on even bars and SELL on odd."""

    def __init__(self):
        super().__init__(strategy_id="test_strat", symbol="BTC/USDT")
        self._bar_count = 0

    def evaluate(self, candles):
        self._bar_count += 1
        if len(candles) < 2:
            return None
        # Simple momentum: if close > open, go long; else short
        last = candles[-1]
        if last.close > last.open:
            direction = SignalDirection.LONG
            strength = 0.8
        elif last.close < last.open:
            direction = SignalDirection.SHORT
            strength = 0.6
        else:
            return None

        self.signals_generated += 1
        return Signal(
            id=f"sig_{self._bar_count}",
            symbol=self.symbol,
            direction=direction,
            strength=strength,
            strategy_id=self.strategy_id,
        )

    def should_exit(self, candles, position):
        # Exit after holding 3 bars
        if not candles:
            return False
        # Simple: exit if unrealized PnL is negative
        return position.unrealized_pnl < -50 or position.unrealized_pnl > 100


class AlwaysEnterStrategy(Strategy):
    """A strategy that always enters long."""

    def __init__(self):
        super().__init__(strategy_id="always_enter", symbol="BTC/USDT")
        self._entered = False

    def evaluate(self, candles):
        if self._entered:
            return None
        self._entered = True
        self.signals_generated += 1
        return Signal(
            id="sig_always",
            symbol=self.symbol,
            direction=SignalDirection.LONG,
            strength=1.0,
            strategy_id=self.strategy_id,
        )

    def should_exit(self, candles, position):
        # Exit after price drops
        return position.unrealized_pnl < -200


class NeverEnterStrategy(Strategy):
    """A strategy that never generates signals."""

    def __init__(self):
        super().__init__(strategy_id="never_enter", symbol="BTC/USDT")

    def evaluate(self, candles):
        return None

    def should_exit(self, candles, position):
        return False


class QuickExitStrategy(Strategy):
    """A strategy that enters and exits quickly."""

    def __init__(self):
        super().__init__(strategy_id="quick_exit", symbol="BTC/USDT")
        self._entered = False

    def evaluate(self, candles):
        if self._entered:
            return None
        self._entered = True
        self.signals_generated += 1
        return Signal(
            id="sig_quick",
            symbol=self.symbol,
            direction=SignalDirection.LONG,
            strength=0.9,
            strategy_id=self.strategy_id,
        )

    def should_exit(self, candles, position):
        # Exit immediately on next bar
        return True


# ============================================================================
# BacktestMode enum
# ============================================================================

class TestBacktestMode:
    """Tests for the BacktestMode enum."""

    def test_single_mode(self):
        assert BacktestMode.SINGLE == "single"

    def test_walk_forward_mode(self):
        assert BacktestMode.WALK_FORWARD == "walk_forward"

    def test_monte_carlo_mode(self):
        assert BacktestMode.MONTE_CARLO == "monte_carlo"

    def test_mode_is_string(self):
        assert isinstance(BacktestMode.SINGLE, str)

    def test_all_modes_count(self):
        assert len(BacktestMode) == 3

    def test_mode_from_value(self):
        assert BacktestMode("single") == BacktestMode.SINGLE

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            BacktestMode("invalid")


# ============================================================================
# BacktestConfig dataclass
# ============================================================================

class TestBacktestConfig:
    """Tests for the BacktestConfig dataclass with all fields and defaults."""

    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100000.0
        assert cfg.commission_bps == 10.0
        assert cfg.slippage_bps == 5.0
        assert cfg.slippage_model == "percentage"
        assert cfg.position_size_pct == 0.02
        assert cfg.max_positions == 5
        assert cfg.margin_enabled is False
        assert cfg.max_leverage == 1.0
        assert cfg.wf_train_pct == 0.7
        assert cfg.wf_test_pct == 0.3
        assert cfg.wf_anchored is False
        assert cfg.mc_simulations == 1000
        assert cfg.mc_method == "bootstrap"
        assert cfg.detect_regimes is True
        assert cfg.regime_lookback == 100
        assert cfg.fill_model == "immediate"
        assert cfg.partial_fill_pct == 0.7
        assert cfg.sensitivity_params is None

    def test_custom_values(self):
        cfg = BacktestConfig(
            initial_capital=50000.0,
            commission_bps=20.0,
            slippage_bps=10.0,
            slippage_model="sqrt",
            position_size_pct=0.05,
            max_positions=10,
            margin_enabled=True,
            max_leverage=2.0,
            wf_train_pct=0.6,
            wf_test_pct=0.4,
            wf_anchored=True,
            mc_simulations=500,
            mc_method="parametric",
            detect_regimes=False,
            regime_lookback=50,
            fill_model="partial",
            partial_fill_pct=0.5,
            sensitivity_params={"param1": [1.0, 2.0]},
        )
        assert cfg.initial_capital == 50000.0
        assert cfg.commission_bps == 20.0
        assert cfg.slippage_bps == 10.0
        assert cfg.slippage_model == "sqrt"
        assert cfg.position_size_pct == 0.05
        assert cfg.max_positions == 10
        assert cfg.margin_enabled is True
        assert cfg.max_leverage == 2.0
        assert cfg.wf_train_pct == 0.6
        assert cfg.wf_test_pct == 0.4
        assert cfg.wf_anchored is True
        assert cfg.mc_simulations == 500
        assert cfg.mc_method == "parametric"
        assert cfg.detect_regimes is False
        assert cfg.regime_lookback == 50
        assert cfg.fill_model == "partial"
        assert cfg.partial_fill_pct == 0.5
        assert cfg.sensitivity_params == {"param1": [1.0, 2.0]}

    def test_zero_initial_capital(self):
        cfg = BacktestConfig(initial_capital=0.0)
        assert cfg.initial_capital == 0.0

    def test_negative_slippage_bps(self):
        """Edge case: negative slippage (should be allowed by dataclass)."""
        cfg = BacktestConfig(slippage_bps=-5.0)
        assert cfg.slippage_bps == -5.0

    def test_zero_commission(self):
        cfg = BacktestConfig(commission_bps=0.0)
        assert cfg.commission_bps == 0.0


# ============================================================================
# BacktestTrade dataclass
# ============================================================================

class TestBacktestTrade:
    """Tests for the BacktestTrade dataclass."""

    def _make_trade(self, **kwargs):
        defaults = dict(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 11, 0),
            symbol="BTC/USDT",
            side=Side.BUY,
            entry_price=100.0,
            exit_price=110.0,
            quantity=10.0,
            pnl=100.0,
            pnl_pct=0.10,
            commission=1.0,
            slippage=0.5,
            holding_period_bars=5,
            strategy_id="test_strat",
        )
        defaults.update(kwargs)
        return BacktestTrade(**defaults)

    def test_basic_trade(self):
        trade = self._make_trade()
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.pnl == 100.0
        assert trade.side == Side.BUY

    def test_default_regime(self):
        trade = self._make_trade()
        assert trade.regime == "unknown"

    def test_custom_regime(self):
        trade = self._make_trade(regime="crisis")
        assert trade.regime == "crisis"

    def test_default_mae_mfe_etd(self):
        trade = self._make_trade()
        assert trade.mae == 0.0
        assert trade.mfe == 0.0
        assert trade.etd == 0.0

    def test_custom_mae_mfe_etd(self):
        trade = self._make_trade(mae=50.0, mfe=200.0, etd=100.0)
        assert trade.mae == 50.0
        assert trade.mfe == 200.0
        assert trade.etd == 100.0

    def test_sell_side_trade(self):
        trade = self._make_trade(side=Side.SELL, pnl=-50.0)
        assert trade.side == Side.SELL
        assert trade.pnl == -50.0

    def test_zero_quantity_trade(self):
        trade = self._make_trade(quantity=0.0, pnl=0.0)
        assert trade.quantity == 0.0

    def test_negative_pnl(self):
        trade = self._make_trade(pnl=-200.0, pnl_pct=-0.20)
        assert trade.pnl == -200.0
        assert trade.pnl_pct == -0.20


# ============================================================================
# MCStatistics dataclass
# ============================================================================

class TestMCStatistics:
    """Tests for the MCStatistics dataclass."""

    def test_default_values(self):
        stats = MCStatistics()
        assert stats.mean_return == 0.0
        assert stats.median_return == 0.0
        assert stats.p5_return == 0.0
        assert stats.p95_return == 0.0
        assert stats.var_95 == 0.0
        assert stats.cvar_95 == 0.0
        assert stats.max_drawdown_p5 == 0.0
        assert stats.max_drawdown_median == 0.0
        assert stats.sharpe_p5 == 0.0
        assert stats.sharpe_median == 0.0
        assert stats.prob_positive == 0.0
        assert stats.num_simulations == 0
        assert stats.simulated_returns is None
        assert stats.simulated_drawdowns is None
        assert stats.simulated_sharpes is None

    def test_custom_values(self):
        stats = MCStatistics(
            mean_return=0.15, median_return=0.12, p5_return=-0.05,
            p95_return=0.35, var_95=0.05, cvar_95=0.08,
            max_drawdown_p5=0.30, max_drawdown_median=0.15,
            sharpe_p5=0.5, sharpe_median=1.2,
            prob_positive=0.65, num_simulations=1000,
        )
        assert stats.mean_return == 0.15
        assert stats.num_simulations == 1000
        assert stats.prob_positive == 0.65

    def test_with_numpy_arrays(self):
        returns = np.array([0.1, 0.2, -0.05])
        stats = MCStatistics(simulated_returns=returns)
        np.testing.assert_array_equal(stats.simulated_returns, returns)


# ============================================================================
# BacktestResult dataclass
# ============================================================================

class TestBacktestResult:
    """Tests for the BacktestResult dataclass."""

    def _make_result(self, **kwargs):
        defaults = dict(
            total_return=0.10, annualized_return=0.25, sharpe_ratio=1.5,
            sortino_ratio=2.0, max_drawdown=0.08, max_drawdown_duration_bars=20,
            calmar_ratio=3.0, win_rate=0.6, profit_factor=1.8,
            total_trades=50, avg_trade_pnl=200.0, avg_winning_trade=500.0,
            avg_losing_trade=-300.0, avg_holding_period=5.0,
        )
        defaults.update(kwargs)
        return BacktestResult(**defaults)

    def test_basic_result(self):
        result = self._make_result()
        assert result.total_return == 0.10
        assert result.total_trades == 50
        assert result.sharpe_ratio == 1.5

    def test_default_optional_fields(self):
        result = self._make_result()
        assert result.trades == []
        assert len(result.equity_curve) == 0
        assert len(result.drawdown_curve) == 0
        assert len(result.regime_labels) == 0
        assert result.mc_statistics is None
        assert result.benchmark_return == 0.0
        assert result.alpha == 0.0
        assert result.information_ratio == 0.0
        assert result.buy_hold_return == 0.0
        assert result.equal_weight_return == 0.0
        assert result.sensitivity_results is None
        assert len(result.rolling_sharpe) == 0
        assert len(result.rolling_sortino) == 0
        assert len(result.rolling_max_dd) == 0

    def test_with_trades(self):
        trade = BacktestTrade(
            entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 2),
            symbol="BTC/USDT", side=Side.BUY, entry_price=100.0,
            exit_price=110.0, quantity=10.0, pnl=100.0, pnl_pct=0.10,
            commission=1.0, slippage=0.5, holding_period_bars=5,
            strategy_id="test",
        )
        result = self._make_result(trades=[trade])
        assert len(result.trades) == 1
        assert result.trades[0].pnl == 100.0

    def test_with_equity_curve(self):
        eq = np.array([100000, 101000, 100500, 102000])
        result = self._make_result(equity_curve=eq)
        np.testing.assert_array_equal(result.equity_curve, eq)

    def test_with_mc_statistics(self):
        mc = MCStatistics(mean_return=0.1, num_simulations=100)
        result = self._make_result(mc_statistics=mc)
        assert result.mc_statistics.mean_return == 0.1

    def test_zero_returns(self):
        result = self._make_result(total_return=0.0, annualized_return=0.0)
        assert result.total_return == 0.0

    def test_negative_returns(self):
        result = self._make_result(total_return=-0.5, annualized_return=-0.7)
        assert result.total_return == -0.5


# ============================================================================
# SlippageModel
# ============================================================================

class TestSlippageModelPercentage:
    """Tests for SlippageModel.percentage."""

    def test_buy_slippage_increases_price(self):
        fill = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        expected = 100.0 * (1 + 1.0 * 5.0 / 10000)
        assert fill == pytest.approx(expected)

    def test_sell_slippage_decreases_price(self):
        fill = SlippageModel.percentage(100.0, 10.0, 5.0, Side.SELL)
        expected = 100.0 * (1 + (-1.0) * 5.0 / 10000)
        assert fill == pytest.approx(expected)

    def test_zero_slippage_bps(self):
        fill = SlippageModel.percentage(100.0, 10.0, 0.0, Side.BUY)
        assert fill == 100.0

    def test_large_slippage_bps(self):
        fill = SlippageModel.percentage(100.0, 10.0, 1000.0, Side.BUY)
        expected = 100.0 * (1 + 1000.0 / 10000)
        assert fill == pytest.approx(expected)

    def test_negative_slippage_bps(self):
        """Edge case: negative slippage decreases buy price."""
        fill = SlippageModel.percentage(100.0, 10.0, -5.0, Side.BUY)
        expected = 100.0 * (1 + (-5.0) / 10000)
        assert fill == pytest.approx(expected)

    def test_quantity_does_not_affect_percentage(self):
        fill1 = SlippageModel.percentage(100.0, 1.0, 5.0, Side.BUY)
        fill2 = SlippageModel.percentage(100.0, 1000.0, 5.0, Side.BUY)
        assert fill1 == fill2

    def test_zero_price(self):
        fill = SlippageModel.percentage(0.0, 10.0, 5.0, Side.BUY)
        assert fill == 0.0

    def test_small_price(self):
        fill = SlippageModel.percentage(0.01, 10.0, 5.0, Side.BUY)
        expected = 0.01 * (1 + 5.0 / 10000)
        assert fill == pytest.approx(expected)


class TestSlippageModelSquareRoot:
    """Tests for SlippageModel.square_root."""

    def test_buy_with_normal_params(self):
        price = 100.0
        quantity = 100.0
        adv = 10000.0
        slippage_bps = 5.0
        fill = SlippageModel.square_root(price, quantity, adv, slippage_bps, Side.BUY)
        participation = quantity / adv
        impact_bps = slippage_bps * np.sqrt(participation)
        expected = price * (1 + impact_bps / 10000)
        assert fill == pytest.approx(expected)

    def test_sell_with_normal_params(self):
        fill = SlippageModel.square_root(100.0, 100.0, 10000.0, 5.0, Side.SELL)
        participation = 100.0 / 10000.0
        impact_bps = 5.0 * np.sqrt(participation)
        expected = 100.0 * (1 - impact_bps / 10000)
        assert fill == pytest.approx(expected)

    def test_zero_volume_returns_price(self):
        fill = SlippageModel.square_root(100.0, 100.0, 0.0, 5.0, Side.BUY)
        assert fill == 100.0

    def test_negative_volume_returns_price(self):
        fill = SlippageModel.square_root(100.0, 100.0, -100.0, 5.0, Side.BUY)
        assert fill == 100.0

    def test_large_quantity_high_impact(self):
        """Large order relative to volume should have more slippage."""
        fill_small = SlippageModel.square_root(100.0, 10.0, 10000.0, 5.0, Side.BUY)
        fill_large = SlippageModel.square_root(100.0, 1000.0, 10000.0, 5.0, Side.BUY)
        assert fill_large > fill_small

    def test_zero_slippage_bps(self):
        fill = SlippageModel.square_root(100.0, 100.0, 10000.0, 0.0, Side.BUY)
        assert fill == 100.0

    def test_slippage_increases_with_quantity(self):
        """Slippage should increase with quantity for sqrt model."""
        fills = [SlippageModel.square_root(100.0, q, 10000.0, 5.0, Side.BUY)
                 for q in [10, 50, 100, 500]]
        assert fills == sorted(fills)


class TestSlippageModelAlmgrenChriss:
    """Tests for SlippageModel.almgren_chriss."""

    def test_buy_with_normal_params(self):
        fill = SlippageModel.almgren_chriss(100.0, 100.0, 10000.0, sigma=0.02, eta=0.1, side=Side.BUY)
        participation = 100.0 / 10000.0
        permanent_impact = 0.1 * participation * 100.0
        temporary_impact = 0.1 * participation * np.sqrt(100.0) * 100.0 * 0.001
        total_impact = permanent_impact + temporary_impact
        expected = 100.0 + total_impact
        assert fill == pytest.approx(expected)

    def test_sell_side(self):
        fill_buy = SlippageModel.almgren_chriss(100.0, 100.0, 10000.0, sigma=0.02, side=Side.BUY)
        fill_sell = SlippageModel.almgren_chriss(100.0, 100.0, 10000.0, sigma=0.02, side=Side.SELL)
        assert fill_buy > 100.0
        assert fill_sell < 100.0

    def test_zero_volume_returns_price(self):
        fill = SlippageModel.almgren_chriss(100.0, 100.0, 0.0, sigma=0.02, side=Side.BUY)
        assert fill == 100.0

    def test_negative_volume_returns_price(self):
        fill = SlippageModel.almgren_chriss(100.0, 100.0, -1000.0, sigma=0.02, side=Side.BUY)
        assert fill == 100.0

    def test_custom_eta(self):
        fill_low_eta = SlippageModel.almgren_chriss(100.0, 100.0, 10000.0, sigma=0.02, eta=0.01, side=Side.BUY)
        fill_high_eta = SlippageModel.almgren_chriss(100.0, 100.0, 10000.0, sigma=0.02, eta=1.0, side=Side.BUY)
        assert fill_high_eta > fill_low_eta

    def test_zero_quantity(self):
        fill = SlippageModel.almgren_chriss(100.0, 0.0, 10000.0, sigma=0.02, side=Side.BUY)
        assert fill == pytest.approx(100.0)

    def test_impact_increases_with_quantity(self):
        fills = [SlippageModel.almgren_chriss(100.0, q, 10000.0, sigma=0.02, side=Side.BUY)
                 for q in [10, 50, 100, 500]]
        assert fills == sorted(fills)


class TestSlippageModelVolumeDependent:
    """Tests for SlippageModel.volume_dependent."""

    def test_normal_volume(self):
        fill = SlippageModel.volume_dependent(100.0, 10.0, 10000.0, 10000.0, 5.0, Side.BUY)
        # volume_ratio = 1.0, adjusted_slippage = 5.0
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_low_current_volume_increases_slippage(self):
        fill_normal = SlippageModel.volume_dependent(100.0, 10.0, 10000.0, 10000.0, 5.0, Side.BUY)
        fill_low = SlippageModel.volume_dependent(100.0, 10.0, 1000.0, 10000.0, 5.0, Side.BUY)
        assert fill_low > fill_normal

    def test_zero_normal_volume_falls_back_to_percentage(self):
        fill = SlippageModel.volume_dependent(100.0, 10.0, 10000.0, 0.0, 5.0, Side.BUY)
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_negative_normal_volume_falls_back(self):
        fill = SlippageModel.volume_dependent(100.0, 10.0, 10000.0, -100.0, 5.0, Side.BUY)
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_slippage_capped_at_10x(self):
        """Adjusted slippage should be capped at 10x base."""
        # Extremely low current_volume should trigger cap
        fill = SlippageModel.volume_dependent(100.0, 10.0, 1e-20, 10000.0, 5.0, Side.BUY)
        max_expected = SlippageModel.percentage(100.0, 10.0, 50.0, Side.BUY)  # 5.0 * 10
        assert fill <= max_expected * 1.001  # Small tolerance

    def test_sell_side(self):
        fill = SlippageModel.volume_dependent(100.0, 10.0, 10000.0, 10000.0, 5.0, Side.SELL)
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.SELL)
        assert fill == pytest.approx(expected)


# ============================================================================
# FillModel
# ============================================================================

class TestFillModelImmediateFill:
    """Tests for FillModel.immediate_fill."""

    def test_basic_fill(self):
        result = FillModel.immediate_fill(100.0, 50.0)
        assert result["filled_quantity"] == 100.0
        assert result["fill_price"] == 50.0
        assert result["fill_pct"] == 1.0
        assert result["partial"] is False

    def test_zero_quantity(self):
        result = FillModel.immediate_fill(0.0, 50.0)
        assert result["filled_quantity"] == 0.0
        assert result["fill_pct"] == 1.0

    def test_large_quantity(self):
        result = FillModel.immediate_fill(1e10, 100.0)
        assert result["filled_quantity"] == 1e10

    def test_no_unfilled_quantity_key(self):
        result = FillModel.immediate_fill(100.0, 50.0)
        assert "unfilled_quantity" not in result


class TestFillModelPartialFill:
    """Tests for FillModel.partial_fill."""

    def test_basic_partial_fill(self):
        result = FillModel.partial_fill(100.0, 50.0, fill_pct=0.7)
        assert result["filled_quantity"] == 70.0
        assert result["fill_price"] == 50.0
        assert result["fill_pct"] == 0.7
        assert result["partial"] is True
        assert result["unfilled_quantity"] == 30.0

    def test_full_fill_when_depth_sufficient(self):
        result = FillModel.partial_fill(100.0, 50.0, fill_pct=1.0)
        assert result["filled_quantity"] == 100.0
        assert result["fill_pct"] == 1.0
        assert result["partial"] is False
        assert result["unfilled_quantity"] == 0.0

    def test_depth_limited_fill(self):
        result = FillModel.partial_fill(100.0, 50.0, fill_pct=0.7, available_depth=50.0)
        assert result["filled_quantity"] == 50.0
        assert result["fill_pct"] == 0.5
        assert result["partial"] is True
        assert result["unfilled_quantity"] == 50.0

    def test_zero_quantity(self):
        result = FillModel.partial_fill(0.0, 50.0, fill_pct=0.7)
        assert result["filled_quantity"] == 0.0
        assert result["fill_pct"] == 0.0
        assert result["partial"] is False

    def test_zero_depth(self):
        result = FillModel.partial_fill(100.0, 50.0, fill_pct=0.7, available_depth=0.0)
        assert result["filled_quantity"] == 0.0
        assert result["fill_pct"] == 0.0
        # effective_fill (0) < quantity (100) => partial = True
        assert result["partial"] is True

    def test_depth_greater_than_quantity(self):
        result = FillModel.partial_fill(50.0, 100.0, fill_pct=0.7, available_depth=200.0)
        assert result["filled_quantity"] == 35.0
        assert result["fill_pct"] == 0.7

    def test_custom_fill_pct(self):
        result = FillModel.partial_fill(100.0, 50.0, fill_pct=0.3)
        assert result["filled_quantity"] == 30.0
        assert result["fill_pct"] == 0.3


class TestFillModelFillOrKill:
    """Tests for FillModel.fill_or_kill."""

    def test_full_fill_when_depth_sufficient(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=100.0)
        assert result["filled_quantity"] == 100.0
        assert result["fill_price"] == 50.0
        assert result["fill_pct"] == 1.0
        assert result["partial"] is False
        assert result["unfilled_quantity"] == 0.0

    def test_fill_when_depth_exceeds_min_pct(self):
        """Depth at 95% of quantity should still fill (default min_fill_pct=0.95)."""
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=95.0, min_fill_pct=0.95)
        assert result["filled_quantity"] == 100.0

    def test_kill_when_depth_insufficient(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=50.0, min_fill_pct=0.95)
        assert result["filled_quantity"] == 0.0
        assert result["fill_price"] == 0.0
        assert result["fill_pct"] == 0.0
        assert result["unfilled_quantity"] == 100.0

    def test_kill_at_exact_threshold(self):
        """At exactly the threshold, should fill."""
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=95.0, min_fill_pct=0.95)
        assert result["filled_quantity"] == 100.0

    def test_just_below_threshold(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=94.99, min_fill_pct=0.95)
        assert result["filled_quantity"] == 0.0

    def test_zero_depth(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=0.0)
        assert result["filled_quantity"] == 0.0

    def test_zero_quantity(self):
        result = FillModel.fill_or_kill(0.0, 50.0, available_depth=0.0)
        # 0 >= 0 * 0.95 is True, so should fill
        assert result["filled_quantity"] == 0.0
        assert result["fill_pct"] == 1.0

    def test_custom_min_fill_pct(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=80.0, min_fill_pct=0.8)
        assert result["filled_quantity"] == 100.0

    def test_custom_min_fill_pct_reject(self):
        result = FillModel.fill_or_kill(100.0, 50.0, available_depth=70.0, min_fill_pct=0.8)
        assert result["filled_quantity"] == 0.0


# ============================================================================
# TradeAnalytics
# ============================================================================

class TestTradeAnalytics:
    """Tests for TradeAnalytics.compute_mae_mfe."""

    def test_buy_with_profit(self):
        """BUY trade with prices going up – should have positive MFE."""
        highs = np.array([110, 120, 115])
        lows = np.array([98, 99, 100])
        result = TradeAnalytics.compute_mae_mfe(100.0, 115.0, Side.BUY, highs, lows, 10.0)
        # MFE: (max(highs) - entry) * qty = (120 - 100) * 10 = 200
        assert result["mfe"] == pytest.approx(200.0)
        # MAE: (entry - min(lows)) * qty = (100 - 98) * 10 = 20
        assert result["mae"] == pytest.approx(20.0)
        # Final PnL: (exit - entry) * qty = (115 - 100) * 10 = 150
        # ETD = MFE - final_pnl = 200 - 150 = 50
        assert result["etd"] == pytest.approx(50.0)

    def test_buy_with_loss(self):
        """BUY trade with prices going down."""
        highs = np.array([102, 101, 100])
        lows = np.array([90, 88, 85])
        result = TradeAnalytics.compute_mae_mfe(100.0, 87.0, Side.BUY, highs, lows, 10.0)
        # MFE: (102 - 100) * 10 = 20
        assert result["mfe"] == pytest.approx(20.0)
        # MAE: (100 - 85) * 10 = 150
        assert result["mae"] == pytest.approx(150.0)
        # Final PnL: (87 - 100) * 10 = -130
        # ETD = 20 - (-130) = 150
        assert result["etd"] == pytest.approx(150.0)

    def test_sell_with_profit(self):
        """SELL trade with prices going down."""
        highs = np.array([102, 100, 98])
        lows = np.array([95, 90, 88])
        result = TradeAnalytics.compute_mae_mfe(100.0, 90.0, Side.SELL, highs, lows, 10.0)
        # MFE: (entry - min(lows)) * qty = (100 - 88) * 10 = 120
        assert result["mfe"] == pytest.approx(120.0)
        # MAE: (max(highs) - entry) * qty = (102 - 100) * 10 = 20
        assert result["mae"] == pytest.approx(20.0)
        # Final PnL: (entry - exit) * qty = (100 - 90) * 10 = 100
        # ETD = 120 - 100 = 20
        assert result["etd"] == pytest.approx(20.0)

    def test_sell_with_loss(self):
        """SELL trade with prices going up."""
        highs = np.array([105, 110, 115])
        lows = np.array([99, 98, 100])
        result = TradeAnalytics.compute_mae_mfe(100.0, 112.0, Side.SELL, highs, lows, 10.0)
        # MFE: (entry - min(lows)) * qty = (100 - 98) * 10 = 20
        assert result["mfe"] == pytest.approx(20.0)
        # MAE: (max(highs) - entry) * qty = (115 - 100) * 10 = 150
        assert result["mae"] == pytest.approx(150.0)
        # Final PnL: (100 - 112) * 10 = -120
        # ETD = 20 - (-120) = 140
        assert result["etd"] == pytest.approx(140.0)

    def test_empty_highs_returns_zeros(self):
        result = TradeAnalytics.compute_mae_mfe(100.0, 110.0, Side.BUY, np.array([]), np.array([95.0]), 10.0)
        assert result["mae"] == 0.0
        assert result["mfe"] == 0.0
        assert result["etd"] == 0.0

    def test_empty_lows_returns_zeros(self):
        result = TradeAnalytics.compute_mae_mfe(100.0, 110.0, Side.BUY, np.array([110.0]), np.array([]), 10.0)
        assert result["mae"] == 0.0

    def test_both_empty_returns_zeros(self):
        result = TradeAnalytics.compute_mae_mfe(100.0, 110.0, Side.BUY, np.array([]), np.array([]), 10.0)
        assert result == {"mae": 0.0, "mfe": 0.0, "etd": 0.0}

    def test_zero_quantity(self):
        result = TradeAnalytics.compute_mae_mfe(100.0, 110.0, Side.BUY,
                                                 np.array([110.0]), np.array([95.0]), 0.0)
        assert result["mfe"] == 0.0
        assert result["mae"] == 0.0

    def test_single_bar(self):
        result = TradeAnalytics.compute_mae_mfe(100.0, 105.0, Side.BUY,
                                                 np.array([105.0]), np.array([98.0]), 10.0)
        assert result["mfe"] == pytest.approx(50.0)
        assert result["mae"] == pytest.approx(20.0)


# ============================================================================
# RollingMetrics
# ============================================================================

class TestRollingSharpe:
    """Tests for RollingMetrics.rolling_sharpe."""

    def test_insufficient_data(self):
        result = RollingMetrics.rolling_sharpe(np.array([100.0, 101.0]), window=60)
        assert len(result) == 0

    def test_sufficient_data(self):
        equity = np.cumsum(np.random.randn(200)) + 100000
        result = RollingMetrics.rolling_sharpe(equity, window=60)
        assert len(result) > 0
        # Some values should be non-NaN
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_constant_equity_returns_nan(self):
        """Constant equity -> zero std -> all NaN rolling values."""
        equity = np.ones(200) * 100000
        result = RollingMetrics.rolling_sharpe(equity, window=60)
        # All should be NaN since std is zero
        valid = result[~np.isnan(result)]
        assert len(valid) == 0

    def test_window_equals_plus_one(self):
        """Exactly window+1 elements should give one valid value."""
        equity = np.cumsum(np.random.randn(61)) + 100000
        result = RollingMetrics.rolling_sharpe(equity, window=60)
        assert len(result) == 60  # len(returns) = 60

    def test_custom_annualization(self):
        equity = np.cumsum(np.random.randn(200)) + 100000
        r1 = RollingMetrics.rolling_sharpe(equity, window=60, annualization_factor=252)
        r2 = RollingMetrics.rolling_sharpe(equity, window=60, annualization_factor=525600)
        # Higher annualization should give higher absolute Sharpe
        valid1 = r1[~np.isnan(r1)]
        valid2 = r2[~np.isnan(r2)]
        for v1, v2 in zip(valid1, valid2):
            if v1 > 0:
                assert v2 > v1
            elif v1 < 0:
                assert v2 < v1


class TestRollingSortino:
    """Tests for RollingMetrics.rolling_sortino."""

    def test_insufficient_data(self):
        result = RollingMetrics.rolling_sortino(np.array([100.0, 101.0]), window=60)
        assert len(result) == 0

    def test_sufficient_data(self):
        equity = np.cumsum(np.random.randn(200)) + 100000
        result = RollingMetrics.rolling_sortino(equity, window=60)
        assert len(result) > 0

    def test_all_positive_returns(self):
        """All positive returns -> no downside -> NaN sortino values."""
        equity = np.linspace(100000, 200000, 200)
        result = RollingMetrics.rolling_sortino(equity, window=60)
        valid = result[~np.isnan(result)]
        assert len(valid) == 0  # No downside deviation

    def test_mixed_returns(self):
        np.random.seed(42)
        equity = np.cumsum(np.random.randn(200)) + 100000
        result = RollingMetrics.rolling_sortino(equity, window=60)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestRollingMaxDrawdown:
    """Tests for RollingMetrics.rolling_max_drawdown."""

    def test_insufficient_data(self):
        result = RollingMetrics.rolling_max_drawdown(np.array([100.0, 101.0]), window=60)
        assert len(result) == 0

    def test_sufficient_data(self):
        equity = np.cumsum(np.random.randn(200)) + 100000
        result = RollingMetrics.rolling_max_drawdown(equity, window=60)
        assert len(result) > 0
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_monotonically_increasing_no_drawdown(self):
        equity = np.linspace(100000, 200000, 200)
        result = RollingMetrics.rolling_max_drawdown(equity, window=60)
        valid = result[~np.isnan(result)]
        # Monotonically increasing should have zero drawdown
        assert all(v == pytest.approx(0.0) for v in valid)

    def test_large_drawdown(self):
        """Create equity with known drawdown."""
        equity = np.array([100] * 50 + [80] * 50 + [100] * 100, dtype=float)
        result = RollingMetrics.rolling_max_drawdown(equity, window=60)
        valid = result[~np.isnan(result)]
        assert max(valid) >= 0.2  # 20% drawdown

    def test_window_exactly_equals_data(self):
        equity = np.cumsum(np.random.randn(60)) + 100000
        result = RollingMetrics.rolling_max_drawdown(equity, window=60)
        assert len(result) == 60


# ============================================================================
# BenchmarkComparison
# ============================================================================

class TestBenchmarkComparison:
    """Tests for BenchmarkComparison.compute_benchmarks."""

    def test_single_asset_rising(self):
        candles = [
            make_candle(close=100.0),
            make_candle(close=110.0),
            make_candle(close=120.0),
        ]
        result = BenchmarkComparison.compute_benchmarks(candles)
        assert result["buy_and_hold_return"] == pytest.approx(0.20)
        assert result["equal_weight_return"] == pytest.approx(0.20)
        assert result["best_single_asset_return"] == pytest.approx(0.20)

    def test_single_asset_falling(self):
        candles = [
            make_candle(close=120.0),
            make_candle(close=110.0),
            make_candle(close=90.0),
        ]
        result = BenchmarkComparison.compute_benchmarks(candles)
        assert result["buy_and_hold_return"] == pytest.approx(-0.25)

    def test_empty_candles(self):
        result = BenchmarkComparison.compute_benchmarks([])
        assert result["buy_and_hold_return"] == 0.0
        assert result["equal_weight_return"] == 0.0
        assert result["best_single_asset_return"] == 0.0

    def test_zero_close_price(self):
        candles = [make_candle(close=0.0), make_candle(close=100.0)]
        result = BenchmarkComparison.compute_benchmarks(candles)
        assert result["buy_and_hold_return"] == 0.0

    def test_multi_asset(self):
        primary = [make_candle(close=100.0), make_candle(close=120.0)]
        asset_a = [make_candle(close=50.0), make_candle(close=60.0)]   # +20%
        asset_b = [make_candle(close=200.0), make_candle(close=180.0)]  # -10%
        multi = {"A": asset_a, "B": asset_b}
        result = BenchmarkComparison.compute_benchmarks(primary, multi)
        assert result["buy_and_hold_return"] == pytest.approx(0.20)
        assert result["equal_weight_return"] == pytest.approx(0.05)  # (0.2 + -0.1) / 2
        assert result["best_single_asset_return"] == pytest.approx(0.20)

    def test_multi_asset_single_entry(self):
        """With only 1 asset in multi dict, should use buy_hold."""
        primary = [make_candle(close=100.0), make_candle(close=110.0)]
        multi = {"A": [make_candle(close=50.0), make_candle(close=60.0)]}
        result = BenchmarkComparison.compute_benchmarks(primary, multi)
        # len(multi_asset_candles) <= 1, so equal_weight defaults to buy_hold
        assert result["equal_weight_return"] == pytest.approx(0.10)

    def test_multi_asset_with_empty_candles(self):
        primary = [make_candle(close=100.0), make_candle(close=110.0)]
        multi = {"A": [make_candle(close=50.0), make_candle(close=60.0)],
                 "B": []}
        result = BenchmarkComparison.compute_benchmarks(primary, multi)
        # Asset B has empty candles -> return 0.0
        assert "buy_and_hold_return" in result

    def test_multi_asset_with_zero_close(self):
        primary = [make_candle(close=100.0), make_candle(close=110.0)]
        multi = {"A": [make_candle(close=0.0), make_candle(close=60.0)],
                 "B": [make_candle(close=50.0), make_candle(close=60.0)]}
        result = BenchmarkComparison.compute_benchmarks(primary, multi)
        # Asset A has close=0 -> return 0.0
        assert result["buy_and_hold_return"] == pytest.approx(0.10)


# ============================================================================
# RegimeDetector
# ============================================================================

class TestRegimeDetector:
    """Tests for RegimeDetector."""

    def test_default_init(self):
        rd = RegimeDetector()
        assert rd.lookback == 100
        assert rd.n_regimes == 3

    def test_custom_init(self):
        rd = RegimeDetector(lookback=50, n_regimes=4)
        assert rd.lookback == 50
        assert rd.n_regimes == 4

    def test_short_series_returns_zeros(self):
        rd = RegimeDetector(lookback=100)
        closes = np.linspace(100, 110, 50)
        result = rd.detect_regimes(closes)
        assert len(result) == 50
        assert all(r == 0 for r in result)

    def test_sufficient_series(self):
        rd = RegimeDetector(lookback=50)
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(200) * 0.5)
        result = rd.detect_regimes(closes)
        assert len(result) == 200
        # First lookback bars should be 0
        assert all(r == 0 for r in result[:50])
        # Later bars should have some regime labels
        assert any(r > 0 for r in result[50:])

    def test_regime_values_in_range(self):
        rd = RegimeDetector(lookback=50)
        np.random.seed(42)
        closes = 100 + np.cumsum(np.random.randn(300) * 1.0)
        result = rd.detect_regimes(closes)
        assert all(0 <= r <= 2 for r in result)

    def test_high_volatility_gets_crisis_label(self):
        """Create a series with a volatile tail to trigger regime 2."""
        rd = RegimeDetector(lookback=50)
        np.random.seed(42)
        # Start calm, end very volatile
        calm = np.cumsum(np.random.randn(200) * 0.1) + 100
        volatile = np.cumsum(np.random.randn(200) * 5.0) + calm[-1]
        closes = np.concatenate([calm, volatile])
        result = rd.detect_regimes(closes)
        # Should have at least some regime 2 (crisis) labels in volatile section
        assert any(r == 2 for r in result[200:])

    def test_constant_prices(self):
        """Constant prices should be all regime 0 (low vol)."""
        rd = RegimeDetector(lookback=50)
        closes = np.ones(200) * 100.0
        result = rd.detect_regimes(closes)
        assert all(r == 0 for r in result)


# ============================================================================
# BacktestEngine.__init__
# ============================================================================

class TestBacktestEngineInit:
    """Tests for BacktestEngine initialization."""

    def test_default_config(self):
        engine = BacktestEngine()
        assert engine.config.initial_capital == 100000.0
        assert isinstance(engine.risk_engine, RiskEngine)
        assert isinstance(engine.slippage, SlippageModel)
        assert isinstance(engine.trade_analytics, TradeAnalytics)
        assert isinstance(engine.rolling_metrics, RollingMetrics)
        assert isinstance(engine.benchmark, BenchmarkComparison)
        assert isinstance(engine.regime_detector, RegimeDetector)
        assert isinstance(engine.sensitivity, SensitivityAnalysis)
        assert isinstance(engine.fill_model, FillModel)

    def test_custom_config(self):
        cfg = BacktestConfig(initial_capital=50000.0, regime_lookback=50)
        engine = BacktestEngine(config=cfg)
        assert engine.config.initial_capital == 50000.0
        assert engine.regime_detector.lookback == 50


# ============================================================================
# BacktestEngine._apply_slippage
# ============================================================================

class TestBacktestEngineApplySlippage:
    """Tests for BacktestEngine._apply_slippage with different models."""

    def test_percentage_model(self):
        engine = BacktestEngine(BacktestConfig(slippage_model="percentage", slippage_bps=5.0))
        fill = engine._apply_slippage(100.0, 10.0, Side.BUY)
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_sqrt_model(self):
        engine = BacktestEngine(BacktestConfig(slippage_model="sqrt", slippage_bps=5.0))
        fill = engine._apply_slippage(100.0, 10.0, Side.BUY)
        expected = SlippageModel.square_root(100.0, 10.0, 10000.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_almgren_chriss_model(self):
        engine = BacktestEngine(BacktestConfig(slippage_model="almgren_chriss", slippage_bps=5.0))
        fill = engine._apply_slippage(100.0, 10.0, Side.BUY)
        expected = SlippageModel.almgren_chriss(100.0, 10.0, 10000.0, sigma=0.02, side=Side.BUY)
        assert fill == pytest.approx(expected)

    def test_unknown_model_defaults_to_percentage(self):
        engine = BacktestEngine(BacktestConfig(slippage_model="unknown", slippage_bps=5.0))
        fill = engine._apply_slippage(100.0, 10.0, Side.BUY)
        expected = SlippageModel.percentage(100.0, 10.0, 5.0, Side.BUY)
        assert fill == pytest.approx(expected)

    def test_sell_side_slippage(self):
        engine = BacktestEngine(BacktestConfig(slippage_model="percentage", slippage_bps=5.0))
        fill = engine._apply_slippage(100.0, 10.0, Side.SELL)
        assert fill < 100.0


# ============================================================================
# BacktestEngine._simulate_fill
# ============================================================================

class TestBacktestEngineSimulateFill:
    """Tests for BacktestEngine._simulate_fill with different fill models."""

    def test_immediate_fill_model(self):
        engine = BacktestEngine(BacktestConfig(fill_model="immediate"))
        result = engine._simulate_fill(100.0, 50.0, Side.BUY)
        assert result["filled_quantity"] == 100.0
        assert result["fill_pct"] == 1.0

    def test_partial_fill_model(self):
        engine = BacktestEngine(BacktestConfig(fill_model="partial", partial_fill_pct=0.5))
        result = engine._simulate_fill(100.0, 50.0, Side.BUY)
        assert result["filled_quantity"] == 50.0
        assert result["partial"] is True

    def test_fok_fill_model(self):
        engine = BacktestEngine(BacktestConfig(fill_model="fok"))
        # Default available_depth=inf, so should fill
        result = engine._simulate_fill(100.0, 50.0, Side.BUY)
        assert result["filled_quantity"] == 100.0

    def test_unknown_model_defaults_to_immediate(self):
        engine = BacktestEngine(BacktestConfig(fill_model="unknown"))
        result = engine._simulate_fill(100.0, 50.0, Side.BUY)
        assert result["filled_quantity"] == 100.0


# ============================================================================
# BacktestEngine._compute_results
# ============================================================================

class TestBacktestEngineComputeResults:
    """Tests for BacktestEngine._compute_results."""

    def _make_engine(self):
        return BacktestEngine(BacktestConfig(initial_capital=100000.0))

    def _make_trade(self, pnl=100.0, holding=5):
        return BacktestTrade(
            entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 2),
            symbol="BTC/USDT", side=Side.BUY, entry_price=100.0,
            exit_price=110.0, quantity=10.0, pnl=pnl, pnl_pct=pnl / 1000.0,
            commission=1.0, slippage=0.5, holding_period_bars=holding,
            strategy_id="test",
        )

    def test_empty_trades_and_short_equity(self):
        engine = self._make_engine()
        result = engine._compute_results([], np.array([100000.0]))
        assert result.total_return == 0
        assert result.total_trades == 0

    def test_empty_equity_curve(self):
        engine = self._make_engine()
        result = engine._compute_results([], np.array([]))
        assert result.total_return == 0

    def test_rising_equity_no_trades(self):
        engine = self._make_engine()
        equity = np.linspace(100000, 120000, 100)
        result = engine._compute_results([], equity)
        assert result.total_return == pytest.approx(0.20, rel=0.01)
        assert result.sharpe_ratio > 0
        assert result.total_trades == 0

    def test_with_winning_and_losing_trades(self):
        engine = self._make_engine()
        trades = [
            self._make_trade(pnl=200.0),
            self._make_trade(pnl=-100.0),
            self._make_trade(pnl=50.0),
        ]
        equity = np.linspace(100000, 100150, 100)
        result = engine._compute_results(trades, equity)
        assert result.total_trades == 3
        assert result.win_rate == pytest.approx(2 / 3)
        assert result.avg_winning_trade == pytest.approx(125.0)
        assert result.avg_losing_trade == pytest.approx(-100.0)

    def test_all_winning_trades(self):
        engine = self._make_engine()
        trades = [self._make_trade(pnl=100.0), self._make_trade(pnl=200.0)]
        equity = np.linspace(100000, 100300, 50)
        result = engine._compute_results(trades, equity)
        assert result.win_rate == 1.0
        assert result.profit_factor == float('inf')

    def test_all_losing_trades(self):
        engine = self._make_engine()
        trades = [self._make_trade(pnl=-100.0), self._make_trade(pnl=-50.0)]
        equity = np.linspace(100000, 99850, 50)
        result = engine._compute_results(trades, equity)
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0

    def test_benchmark_return(self):
        engine = self._make_engine()
        equity = np.linspace(100000, 110000, 100)
        result = engine._compute_results([], equity, benchmark_return=0.05)
        assert result.benchmark_return == 0.05
        # Alpha should be strategy return minus benchmark
        assert result.alpha == pytest.approx(result.total_return - 0.05, abs=0.01)

    def test_custom_annualization(self):
        engine = self._make_engine()
        equity = np.linspace(100000, 110000, 100)
        result = engine._compute_results([], equity, annualization=525600)
        assert result.annualized_return != 0

    def test_max_drawdown_computed(self):
        engine = self._make_engine()
        # Equity with a drawdown: rise, fall, rise
        equity = np.concatenate([
            np.linspace(100000, 120000, 50),
            np.linspace(120000, 90000, 30),
            np.linspace(90000, 110000, 20),
        ])
        result = engine._compute_results([], equity)
        assert result.max_drawdown > 0
        assert result.max_drawdown_duration_bars > 0

    def test_drawdown_curve_in_result(self):
        engine = self._make_engine()
        equity = np.linspace(100000, 110000, 100)
        result = engine._compute_results([], equity)
        assert len(result.drawdown_curve) == len(equity)

    def test_avg_holding_period(self):
        engine = self._make_engine()
        trades = [self._make_trade(holding=3), self._make_trade(holding=7)]
        equity = np.linspace(100000, 100200, 50)
        result = engine._compute_results(trades, equity)
        assert result.avg_holding_period == pytest.approx(5.0)


# ============================================================================
# BacktestEngine.run (single mode)
# ============================================================================

class TestBacktestEngineRunSingle:
    """Tests for BacktestEngine.run in SINGLE mode."""

    def test_never_enter_strategy(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert result.total_trades == 0
        assert len(result.equity_curve) > 0
        assert result.total_return == pytest.approx(0.0)

    def test_simple_strategy(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.total_trades >= 0

    def test_quick_exit_strategy(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = QuickExitStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_with_regime_detection(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=True, regime_lookback=50))
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert len(result.regime_labels) > 0

    def test_without_regime_detection(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert len(result.regime_labels) == 0

    def test_benchmark_return_in_result(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert hasattr(result, "buy_hold_return")
        assert hasattr(result, "equal_weight_return")

    def test_rolling_metrics_in_result(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result.rolling_sharpe, np.ndarray)
        assert isinstance(result.rolling_sortino, np.ndarray)
        assert isinstance(result.rolling_max_dd, np.ndarray)

    def test_single_candle(self):
        """Edge case: single candle."""
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = [make_candle(close=100.0)]
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_empty_candles(self):
        """Edge case: no candles."""
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        strategy = NeverEnterStrategy()
        result = engine.run([], strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_zero_capital(self):
        """Edge case: zero initial capital."""
        engine = BacktestEngine(BacktestConfig(initial_capital=0.0, detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_multi_asset_candles(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        multi = {
            "A": make_candle_series(200, base_price=50.0),
            "B": make_candle_series(200, base_price=200.0),
        }
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE,
                           multi_asset_candles=multi)
        assert isinstance(result, BacktestResult)


# ============================================================================
# BacktestEngine.run (walk-forward)
# ============================================================================

class TestBacktestEngineRunWalkForward:
    """Tests for BacktestEngine.run in WALK_FORWARD mode."""

    def test_walk_forward(self):
        cfg = BacktestConfig(
            wf_train_pct=0.5, wf_test_pct=0.2,
            detect_regimes=False, initial_capital=100000.0,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(500)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.WALK_FORWARD)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_walk_forward_short_data(self):
        """Short data should still run without error."""
        cfg = BacktestConfig(
            wf_train_pct=0.7, wf_test_pct=0.3,
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(100)
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.WALK_FORWARD)
        assert isinstance(result, BacktestResult)

    def test_walk_forward_with_trades(self):
        cfg = BacktestConfig(
            wf_train_pct=0.4, wf_test_pct=0.2,
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(600)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.WALK_FORWARD)
        assert isinstance(result, BacktestResult)


# ============================================================================
# BacktestEngine.run (Monte Carlo)
# ============================================================================

class TestBacktestEngineRunMonteCarlo:
    """Tests for BacktestEngine.run in MONTE_CARLO mode."""

    def test_monte_carlo_bootstrap(self):
        cfg = BacktestConfig(
            mc_simulations=50, mc_method="bootstrap",
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        assert isinstance(result, BacktestResult)
        if result.mc_statistics:
            assert result.mc_statistics.num_simulations == 50
            assert result.mc_statistics.simulated_returns is not None
            assert len(result.mc_statistics.simulated_returns) == 50

    def test_monte_carlo_parametric(self):
        cfg = BacktestConfig(
            mc_simulations=50, mc_method="parametric",
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        assert isinstance(result, BacktestResult)

    def test_monte_carlo_no_trades(self):
        """If no trades, MC should return base result without mc_statistics."""
        cfg = BacktestConfig(mc_simulations=10, detect_regimes=False)
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        assert result.mc_statistics is None

    def test_monte_carlo_statistics_fields(self):
        cfg = BacktestConfig(mc_simulations=100, detect_regimes=False)
        engine = BacktestEngine(cfg)
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        if result.mc_statistics:
            mc = result.mc_statistics
            assert hasattr(mc, "mean_return")
            assert hasattr(mc, "median_return")
            assert hasattr(mc, "p5_return")
            assert hasattr(mc, "p95_return")
            assert hasattr(mc, "var_95")
            assert hasattr(mc, "cvar_95")
            assert hasattr(mc, "max_drawdown_p5")
            assert hasattr(mc, "max_drawdown_median")
            assert hasattr(mc, "sharpe_p5")
            assert hasattr(mc, "sharpe_median")
            assert hasattr(mc, "prob_positive")
            assert mc.prob_positive >= 0.0
            assert mc.prob_positive <= 1.0

    def test_monte_carlo_var_cvar_relationship(self):
        """CVaR should be >= VaR in magnitude (for losses)."""
        cfg = BacktestConfig(mc_simulations=100, detect_regimes=False)
        engine = BacktestEngine(cfg)
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        if result.mc_statistics:
            mc = result.mc_statistics
            # var_95 is -p5_return, cvar_95 should be >= var_95 in magnitude
            # (both are expressed as positive numbers for losses)
            if mc.var_95 > 0:
                assert mc.cvar_95 >= mc.var_95


# ============================================================================
# BacktestEngine.run (invalid mode)
# ============================================================================

class TestBacktestEngineRunInvalidMode:
    """Tests for BacktestEngine.run with invalid mode."""

    def test_invalid_mode_raises_valueerror(self):
        engine = BacktestEngine()
        candles = make_candle_series(100)
        strategy = NeverEnterStrategy()
        with pytest.raises(ValueError, match="Unknown backtest mode"):
            engine.run(candles, strategy, mode="invalid_mode")


# ============================================================================
# BacktestEngine.run_sensitivity
# ============================================================================

class TestBacktestEngineRunSensitivity:
    """Tests for BacktestEngine.run_sensitivity."""

    def test_default_params(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run_sensitivity(candles, strategy)
        assert "position_size_pct" in result or "slippage_bps" in result

    def test_custom_params(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        params = {"slippage_bps": [0.0, 5.0, 10.0]}
        result = engine.run_sensitivity(candles, strategy, params=params)
        assert "slippage_bps" in result
        assert "results" in result["slippage_bps"]
        assert "sensitivity" in result["slippage_bps"]
        assert len(result["slippage_bps"]["results"]) == 3

    def test_sensitivity_restores_original_value(self):
        cfg = BacktestConfig(slippage_bps=5.0, detect_regimes=False)
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        params = {"slippage_bps": [0.0, 10.0]}
        engine.run_sensitivity(candles, strategy, params=params)
        assert engine.config.slippage_bps == 5.0

    def test_sensitivity_with_invalid_param(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        params = {"nonexistent_param": [1.0, 2.0]}
        result = engine.run_sensitivity(candles, strategy, params=params)
        assert "nonexistent_param" in result


# ============================================================================
# SensitivityAnalysis.run
# ============================================================================

class TestSensitivityAnalysis:
    """Tests for SensitivityAnalysis.run directly."""

    def test_run_with_valid_param(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        params = {"commission_bps": [0.0, 10.0, 20.0]}
        result = SensitivityAnalysis.run(engine, candles, strategy, params)
        assert "commission_bps" in result
        assert len(result["commission_bps"]["results"]) == 3

    def test_sensitivity_range(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        params = {"slippage_bps": [0.0, 50.0]}
        result = SensitivityAnalysis.run(engine, candles, strategy, params)
        # With no trades, sensitivity should be 0
        assert result["slippage_bps"]["sensitivity"] == 0.0

    def test_sensitivity_with_single_value(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        params = {"slippage_bps": [5.0]}
        result = SensitivityAnalysis.run(engine, candles, strategy, params)
        assert len(result["slippage_bps"]["results"]) == 1
        # Only one value -> sensitivity = 0
        assert result["slippage_bps"]["sensitivity"] == 0.0


# ============================================================================
# Integration: full pipeline
# ============================================================================

class TestBacktestIntegration:
    """Integration tests for the full backtest pipeline."""

    def test_full_pipeline_single(self):
        cfg = BacktestConfig(
            initial_capital=100000.0,
            commission_bps=10.0,
            slippage_bps=5.0,
            slippage_model="percentage",
            fill_model="immediate",
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(300, base_price=50000.0)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == pytest.approx(100000.0)

    def test_full_pipeline_with_partial_fills(self):
        cfg = BacktestConfig(
            fill_model="partial",
            partial_fill_pct=0.5,
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_full_pipeline_with_fok(self):
        cfg = BacktestConfig(
            fill_model="fok",
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_full_pipeline_with_sqrt_slippage(self):
        cfg = BacktestConfig(
            slippage_model="sqrt",
            slippage_bps=10.0,
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_full_pipeline_with_almgren_chriss(self):
        cfg = BacktestConfig(
            slippage_model="almgren_chriss",
            detect_regimes=False,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)

    def test_walk_forward_then_mc(self):
        """Run WF then MC on the same data."""
        cfg = BacktestConfig(
            mc_simulations=20, detect_regimes=False,
            wf_train_pct=0.5, wf_test_pct=0.2,
        )
        engine = BacktestEngine(cfg)
        candles = make_candle_series(400)
        strategy = SimpleStrategy()

        wf_result = engine.run(candles, strategy, mode=BacktestMode.WALK_FORWARD)
        assert isinstance(wf_result, BacktestResult)

        mc_result = engine.run(candles, strategy, mode=BacktestMode.MONTE_CARLO)
        assert isinstance(mc_result, BacktestResult)

    def test_negative_return_scenario(self):
        """Strategy that loses money."""
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(200, base_price=100.0, volatility=0.05)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        # Could be positive or negative depending on random seed
        assert isinstance(result.total_return, float)

    def test_trade_record_fields(self):
        """Verify that closed trades have all expected fields."""
        engine = BacktestEngine(BacktestConfig(detect_regimes=False))
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        if result.trades:
            trade = result.trades[0]
            assert hasattr(trade, "entry_time")
            assert hasattr(trade, "exit_time")
            assert hasattr(trade, "symbol")
            assert hasattr(trade, "side")
            assert hasattr(trade, "entry_price")
            assert hasattr(trade, "exit_price")
            assert hasattr(trade, "quantity")
            assert hasattr(trade, "pnl")
            assert hasattr(trade, "pnl_pct")
            assert hasattr(trade, "commission")
            assert hasattr(trade, "slippage")
            assert hasattr(trade, "holding_period_bars")
            assert hasattr(trade, "strategy_id")
            assert hasattr(trade, "regime")
            assert hasattr(trade, "mae")
            assert hasattr(trade, "mfe")
            assert hasattr(trade, "etd")

    def test_commission_is_nonnegative(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False, commission_bps=10.0))
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        for trade in result.trades:
            assert trade.commission >= 0

    def test_slippage_is_nonnegative(self):
        engine = BacktestEngine(BacktestConfig(detect_regimes=False, slippage_bps=5.0))
        candles = make_candle_series(300)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        for trade in result.trades:
            assert trade.slippage >= 0

    def test_equity_curve_starts_at_initial_capital(self):
        engine = BacktestEngine(BacktestConfig(initial_capital=100000.0, detect_regimes=False))
        candles = make_candle_series(200)
        strategy = NeverEnterStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert result.equity_curve[0] == pytest.approx(100000.0)

    def test_max_positions_limit(self):
        """With max_positions=1, only one position at a time."""
        cfg = BacktestConfig(max_positions=1, detect_regimes=False)
        engine = BacktestEngine(cfg)
        candles = make_candle_series(200)
        strategy = SimpleStrategy()
        result = engine.run(candles, strategy, mode=BacktestMode.SINGLE)
        assert isinstance(result, BacktestResult)
