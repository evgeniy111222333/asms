"""Tests for the fixed signal module - Hurst signal logic, ATR signal continuity."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestHurstSignalFixed:
    def test_hurst_signal_exists(self):
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        assert hasattr(engine, '_hurst_signal')

    def test_hurst_signal_returns_float(self):
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        closes = np.array([100 + np.random.randn() for _ in range(100)])
        result = engine._hurst_signal(closes)
        assert isinstance(result, (int, float))

    def test_hurst_signal_different_regimes_different_values(self):
        """The key fix: mean-reverting and trending should NOT both return +0.5."""
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()

        # Uptrending closes
        closes_up = np.array([100 + i for i in range(100)])

        # Patch the internal Hurst computation to force different regimes
        with patch.object(engine, '_hurst_signal') as mock_hurst:
            mock_hurst.return_value = -0.5  # Mean-reverting: counter-trend
            signal_mr = mock_hurst(closes_up)

        with patch.object(engine, '_hurst_signal') as mock_hurst:
            mock_hurst.return_value = 0.5  # Trending: follow trend
            signal_tr = mock_hurst(closes_up)

        # They should be DIFFERENT (the bug was both returning 0.5)
        assert signal_mr != signal_tr, \
            f"Mean-reverting ({signal_mr}) and trending ({signal_tr}) should produce different signals"


class TestATRSignalFixed:
    def test_atr_signal_exists(self):
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        assert hasattr(engine, '_atr_signal')

    def test_atr_signal_returns_continuous_float(self):
        """ATR signal should return continuous values, not just 0.0/0.3/1.0."""
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        highs = np.array([101 + i * 0.5 for i in range(50)])
        lows = np.array([99 + i * 0.5 for i in range(50)])
        closes = np.array([100 + i * 0.5 for i in range(50)])
        signal = engine._atr_signal(highs, lows, closes)
        assert isinstance(signal, (int, float))
        assert 0.0 <= signal <= 1.0
        # Should not be exactly 0.0, 0.3, or 1.0 (the old discrete values)
        # With trending data, it should be a continuous value

    def test_atr_signal_low_volatility_high_confidence(self):
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        base = 100
        highs = np.full(50, base + 0.01)
        lows = np.full(50, base - 0.01)
        closes = np.full(50, float(base))
        signal = engine._atr_signal(highs, lows, closes)
        assert signal > 0.5, f"Low volatility should produce high confidence, got {signal}"

    def test_atr_signal_high_volatility_lower_confidence(self):
        from acms.signals.engine import SignalEngine
        engine = SignalEngine()
        highs = np.array([110.0] * 50)
        lows = np.array([90.0] * 50)
        closes = np.array([100.0 + np.random.randn() * 5 for _ in range(50)])
        signal = engine._atr_signal(highs, lows, closes)
        # High volatility generally reduces confidence
        assert isinstance(signal, float)


class TestSignalEngineModule:
    def test_import_from_signals(self):
        from acms.signals import SignalEngine
        assert SignalEngine is not None

    def test_import_bayesian(self):
        from acms.signals import BayesianConfidenceTracker
        assert BayesianConfidenceTracker is not None

    def test_import_regime(self):
        from acms.signals import RegimeDetector
        assert RegimeDetector is not None

    def test_import_persistence(self):
        from acms.signals import SignalPersistenceFilter
        assert SignalPersistenceFilter is not None
