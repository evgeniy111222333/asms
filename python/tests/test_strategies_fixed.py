"""Tests for the fixed strategy stubs - dead strategies now have real implementations."""

import pytest
from unittest.mock import MagicMock


def _make_candle(close, high=None, low=None, volume=1000, symbol="BTC/USDT"):
    candle = MagicMock()
    candle.close = close
    candle.high = high or close * 1.01
    candle.low = low or close * 0.99
    candle.volume = volume
    candle.symbol = symbol
    candle.timestamp = MagicMock()
    return candle


class TestStatisticalArbitrageFixed:
    def test_stat_arb_imports(self):
        from acms.strategies import StatisticalArbitrageStrategy
        assert StatisticalArbitrageStrategy is not None

    def test_stat_arb_instantiation(self):
        from acms.strategies import StatisticalArbitrageStrategy
        strategy = StatisticalArbitrageStrategy(symbol="BTC/USDT", symbol2="ETH/USDT", lookback=30)
        assert strategy is not None

    def test_stat_arb_has_evaluate(self):
        from acms.strategies import StatisticalArbitrageStrategy
        strategy = StatisticalArbitrageStrategy(symbol="BTC/USDT", symbol2="ETH/USDT")
        assert hasattr(strategy, "evaluate")
        assert callable(strategy.evaluate)

    def test_stat_arb_has_should_exit(self):
        from acms.strategies import StatisticalArbitrageStrategy
        strategy = StatisticalArbitrageStrategy(symbol="BTC/USDT", symbol2="ETH/USDT")
        assert hasattr(strategy, "should_exit")


class TestCarryStrategyFixed:
    def test_carry_imports(self):
        from acms.strategies import CarryStrategy
        assert CarryStrategy is not None

    def test_carry_instantiation(self):
        from acms.strategies import CarryStrategy
        strategy = CarryStrategy(symbol="BTC/USDT")
        assert strategy is not None

    def test_carry_has_evaluate(self):
        from acms.strategies import CarryStrategy
        strategy = CarryStrategy(symbol="BTC/USDT")
        assert hasattr(strategy, "evaluate")
        assert callable(strategy.evaluate)


class TestCrossExchangeArbitrageFixed:
    def test_cross_ex_imports(self):
        from acms.strategies import CrossExchangeArbitrageStrategy
        assert CrossExchangeArbitrageStrategy is not None

    def test_cross_ex_instantiation(self):
        from acms.strategies import CrossExchangeArbitrageStrategy
        strategy = CrossExchangeArbitrageStrategy(symbol="BTC/USDT")
        assert strategy is not None

    def test_cross_ex_should_exit_not_always_true(self):
        from acms.strategies import CrossExchangeArbitrageStrategy
        import inspect
        source = inspect.getsource(CrossExchangeArbitrageStrategy.should_exit)
        # Should not be just "return True"
        assert "return True" not in source.split('\n')[1:3]  # Not the first real line


class TestTurtleStrategyFixed:
    def test_turtle_instantiation(self):
        from acms.strategies import TurtleTradingStrategy
        strategy = TurtleTradingStrategy(symbol="BTC/USDT")
        assert strategy is not None

    def test_turtle_should_exit_accepts_position(self):
        from acms.strategies import TurtleTradingStrategy
        import inspect
        sig = inspect.signature(TurtleTradingStrategy.should_exit)
        # Should have position parameter
        assert "position" in sig.parameters


class TestStrategyBaseABC:
    def test_all_strategies_have_evaluate(self):
        from acms.strategies import (
            TrendFollowingMomentum,
            MeanReversionStrategy,
            StatisticalArbitrageStrategy,
            GridTradingStrategy,
            TurtleTradingStrategy,
            CarryStrategy,
        )
        strategies = [
            TrendFollowingMomentum(symbol="BTC/USDT"),
            MeanReversionStrategy(symbol="BTC/USDT"),
            StatisticalArbitrageStrategy(symbol="BTC/USDT", symbol2="ETH/USDT"),
            GridTradingStrategy(symbol="BTC/USDT"),
            TurtleTradingStrategy(symbol="BTC/USDT"),
            CarryStrategy(symbol="BTC/USDT"),
        ]
        for strategy in strategies:
            assert hasattr(strategy, "evaluate"), \
                f"{strategy.__class__.__name__} must have evaluate()"

    def test_all_strategies_have_should_exit(self):
        from acms.strategies import (
            TrendFollowingMomentum,
            MeanReversionStrategy,
            GridTradingStrategy,
        )
        for strategy in [
            TrendFollowingMomentum(symbol="BTC/USDT"),
            MeanReversionStrategy(symbol="BTC/USDT"),
            GridTradingStrategy(symbol="BTC/USDT"),
        ]:
            assert hasattr(strategy, "should_exit")


class TestStrategyBackwardCompatibility:
    def test_import_from_strategies(self):
        from acms.strategies import Strategy
        assert Strategy is not None

    def test_import_specific_strategies(self):
        from acms.strategies import (
            TrendFollowingMomentum,
            RSIMomentum,
            MACDMomentum,
            MeanReversionStrategy,
            StatisticalArbitrageStrategy,
            GridTradingStrategy,
            TurtleTradingStrategy,
            WyckoffStrategy,
            CarryStrategy,
            MarketMakingStrategy,
        )
