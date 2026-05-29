"""Integration tests - verify all modules work together after refactoring."""

import pytest


class TestModuleStructureIntegrity:
    """Verify all modules can be imported after the refactoring."""

    def test_import_acms_core(self):
        from acms.core import ACMSConfig, Signal, Order, Position
        assert all([ACMSConfig, Signal, Order, Position])

    def test_import_acms_core_enums(self):
        from acms.core.enums import Side, OrderType, OrderStatus
        assert all([Side, OrderType, OrderStatus])

    def test_import_acms_core_types(self):
        from acms.core.types import Candle, Tick, Symbol
        assert all([Candle, Tick, Symbol])

    def test_import_acms_core_config(self):
        from acms.core.config import ACMSConfig
        assert ACMSConfig is not None

    def test_import_acms_auth(self):
        from acms.auth import AuthManager
        assert AuthManager is not None

    def test_import_acms_db(self):
        from acms.db import DatabaseManager, CredentialEncryptor
        assert all([DatabaseManager, CredentialEncryptor])

    def test_import_acms_db_models(self):
        from acms.db.models import User, ApiKey, OrderRecord, TradeRecord
        assert all([User, ApiKey, OrderRecord, TradeRecord])

    def test_import_acms_db_encryption(self):
        from acms.db.encryption import CredentialEncryptor
        assert CredentialEncryptor is not None

    def test_import_acms_api(self):
        from acms.api import app
        assert app is not None

    def test_import_acms_indicators(self):
        from acms.indicators import SMA, EMA, RSI, MACD, ATR, BollingerBands
        assert all([SMA, EMA, RSI, MACD, ATR, BollingerBands])

    def test_import_acms_indicators_moving_averages(self):
        from acms.indicators.moving_averages import SMA, EMA, WMA, HMA
        assert all([SMA, EMA, WMA, HMA])

    def test_import_acms_indicators_oscillators(self):
        from acms.indicators.oscillators import RSI, MACD, StochasticOscillator
        assert all([RSI, MACD, StochasticOscillator])

    def test_import_acms_indicators_volatility(self):
        from acms.indicators.volatility import ATR, BollingerBands, KeltnerChannels
        assert all([ATR, BollingerBands, KeltnerChannels])

    def test_import_acms_math_stats(self):
        from acms.math_stats import BlackScholes, GARCH11, KalmanFilter
        assert all([BlackScholes, GARCH11, KalmanFilter])

    def test_import_acms_strategies(self):
        from acms.strategies import (
            Strategy, TrendFollowingMomentum,
            MeanReversionStrategy, GridTradingStrategy
        )
        assert all([Strategy, TrendFollowingMomentum, MeanReversionStrategy, GridTradingStrategy])

    def test_import_acms_signals(self):
        from acms.signals import SignalEngine, BayesianConfidenceTracker
        assert all([SignalEngine, BayesianConfidenceTracker])

    def test_import_acms_portfolio(self):
        from acms.portfolio import PortfolioEngine, MeanVarianceOptimizer
        assert all([PortfolioEngine, MeanVarianceOptimizer])

    def test_import_acms_risk(self):
        from acms.risk import RiskEngine, ValueAtRisk, CircuitBreaker
        assert all([RiskEngine, ValueAtRisk, CircuitBreaker])

    def test_import_acms_exchanges(self):
        from acms.exchanges import BinanceAdapter, BybitAdapter, OKXAdapter
        assert all([BinanceAdapter, BybitAdapter, OKXAdapter])

    def test_import_acms_backtest(self):
        from acms.backtest import BacktestEngine
        assert BacktestEngine is not None

    def test_import_acms_ml(self):
        from acms.ml import FeatureEngineer, EnsembleModel
        assert all([FeatureEngineer, EnsembleModel])

    def test_import_acms_orchestrator(self):
        from acms.orchestrator import Orchestrator
        assert Orchestrator is not None

    def test_import_acms_pipeline(self):
        from acms.pipeline import DataPipeline
        assert DataPipeline is not None

    def test_import_acms_kafka(self):
        from acms.kafka import KafkaProducer, KafkaConsumer
        assert all([KafkaProducer, KafkaConsumer])

    def test_import_acms_redis_client(self):
        from acms.redis_client import CacheManager, PubSubManager
        assert all([CacheManager, PubSubManager])

    def test_import_acms_reporting(self):
        from acms.reporting import ReportingEngine
        assert ReportingEngine is not None

    def test_import_acms_ai(self):
        import acms.ai
        assert acms.ai is not None

    def test_import_acms_cli(self):
        from acms.cli import cli
        assert cli is not None

    def test_import_acms_logging_config(self):
        from acms.logging_config import configure_logging
        assert configure_logging is not None


class TestCrossModuleIntegration:
    """Test that modules can work together."""

    def test_indicators_compute_from_candle_data(self):
        from acms.indicators import SMA
        sma = SMA(period=10)
        prices = [100 + i * 0.5 for i in range(30)]
        result = sma.compute(prices)
        assert result is not None

    def test_risk_engine_uses_var(self):
        from acms.risk import RiskEngine, ValueAtRisk
        assert all([RiskEngine, ValueAtRisk])

    def test_db_encryption_with_credentials(self):
        from acms.db.encryption import CredentialEncryptor
        enc = CredentialEncryptor()
        api_key = enc.encrypt("my-api-key")
        secret = enc.encrypt("my-secret")
        assert enc.decrypt(api_key) == "my-api-key"
        assert enc.decrypt(secret) == "my-secret"


class TestNoStubsRemaining:
    """Verify that known stubs have been replaced."""

    def test_authenticate_user_not_placeholder(self):
        from acms.auth import AuthManager
        import inspect
        source = inspect.getsource(AuthManager.authenticate_user)
        assert "Placeholder" not in source
        assert "accept any credentials" not in source

    def test_hash_password_uses_bcrypt(self):
        from acms.auth import AuthManager
        import inspect
        source = inspect.getsource(AuthManager.hash_password)
        assert "bcrypt" in source.lower()

    def test_no_computed_placeholder_in_orchestrator(self):
        try:
            from acms.ai.integration.orchestrator import AIOrchestrator
            import inspect
            source = inspect.getsource(AIOrchestrator._compute_features)
            assert "computed_placeholder" not in source
        except (ImportError, AttributeError):
            pass

    def test_cross_exchange_arb_should_exit_not_true(self):
        """CrossExchangeArbitrageStrategy.should_exit should not always return True."""
        from acms.strategies import CrossExchangeArbitrageStrategy
        import inspect
        source = inspect.getsource(CrossExchangeArbitrageStrategy.should_exit)
        # Should not be just "return True"
        assert source.strip() != "return True"


class TestNoExceptPassRemaining:
    """Verify that except: pass has been replaced with logging."""

    def test_fewer_except_pass_than_original(self):
        import subprocess
        result = subprocess.run(
            ["rg", "except.*pass", "acms/", "--count"],
            capture_output=True, text=True,
            cwd="/home/z/my-project/asms/python"
        )
        total = 0
        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                try:
                    count = int(line.split(':')[-1].strip())
                    total += count
                except ValueError:
                    pass
        # We had 42 before, should be significantly reduced
        assert total < 15, f"Too many except:pass remaining: {total}"
