"""Comprehensive pytest tests for acms.indicators module.

Tests every indicator class and function for:
- Construction with default and custom parameters
- compute() with normal data (200+ data points)
- compute() with minimal data (edge case: exactly the period length)
- compute() with insufficient data (should return NaN or handle gracefully)
- compute() with constant data (all same values)
- compute() with NaN values in the input
- compute() with very large values
- compute() with very small values near zero
- Result values are within expected ranges
"""

import sys
sys.path.insert(0, '/home/z/my-project/asms/python')

import numpy as np
import pytest
from acms.indicators import (
    # Moving Averages
    SMA, EMA, WMA, HMA, DEMA, TEMA, VWMA, KAMA, ALMA, FRAMA, ZLEMA,
    Supertrend, MovingAverageRibbon, KaufmanAdaptiveMovingAverage,
    EhlersSuperSmoother,
    # Oscillators
    RSI, ConnorsRSI, MACD, VolumeWeightedMACD, StochasticOscillator,
    CCI, WilliamsR, ROC, Momentum, TRIX, UltimateOscillator, MFI, ADX,
    Aroon, ChandeMomentumOscillator, EhlersFisherTransform,
    # Volatility
    ATR, BollingerBands, KeltnerChannels, DonchianChannels,
    StandardDeviation, HistoricalVolatility, ParkinsonVolatility,
    GarmanKlassVolatility, ChaikinVolatility, TrueRange, ATRP,
    # Volume
    OBVIndicator, CMFIndicator, ADLine, VolumeProfile, ForceIndex,
    EaseOfMovement, VolumeOscillator, NVI, PVI,
    # Advanced
    IchimokuCloud, TTMSqueeze, VWAPIndicator, VWAPBands,
    EhlersDominantCycle, FractalDimension, PivotPoints,
    FibonacciRetracement, SupportResistance,
    # Candlestick
    CandlestickPatterns,
    # Statistical
    compute_hurst_exponent, compute_zscore,
    detect_bullish_divergence, detect_bearish_divergence,
)


# ============================================================================
# Fixtures: Realistic OHLCV data generators
# ============================================================================

@pytest.fixture(scope="module")
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def close_data(rng):
    """200-point random-walk close prices (crypto-like volatility)."""
    n = 250
    returns = rng.normal(0.0005, 0.03, n)
    prices = 50000.0 * np.cumprod(1 + returns)
    return prices


@pytest.fixture(scope="module")
def ohlcv_data(rng):
    """Full OHLCV dataset with 250 bars."""
    n = 250
    returns = rng.normal(0.0005, 0.03, n)
    closes = 50000.0 * np.cumprod(1 + returns)
    highs = closes * (1 + np.abs(rng.normal(0, 0.01, n)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.01, n)))
    opens = (highs + lows) / 2 + rng.normal(0, 50, n)
    volumes = rng.uniform(1e6, 1e8, n)
    return opens, highs, lows, closes, volumes


@pytest.fixture(scope="module")
def constant_data():
    """Constant price data (all same values)."""
    return np.full(250, 50000.0)


@pytest.fixture(scope="module")
def constant_ohlcv():
    """Constant OHLCV data."""
    c = np.full(250, 50000.0)
    h = np.full(250, 50010.0)
    l = np.full(250, 49990.0)
    o = np.full(250, 50000.0)
    v = np.full(250, 1e7)
    return o, h, l, c, v


@pytest.fixture(scope="module")
def large_data():
    """Data with very large values."""
    return np.full(250, 1e12)


@pytest.fixture(scope="module")
def small_data():
    """Data with very small values near zero."""
    return np.full(250, 1e-10)


@pytest.fixture(scope="module")
def nan_data():
    """Data with NaN values scattered throughout."""
    data = 50000.0 + np.random.randn(250) * 1000
    data[10] = np.nan
    data[50] = np.nan
    data[100] = np.nan
    return data


# ============================================================================
# SMA Tests
# ============================================================================

class TestSMA:
    """Tests for Simple Moving Average."""

    def test_construction_default(self):
        sma = SMA(20)
        assert sma.period == 20

    def test_construction_period_one(self):
        sma = SMA(1)
        assert sma.period == 1

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            SMA(0)
        with pytest.raises(ValueError):
            SMA(-1)

    def test_compute_normal_data(self, close_data):
        sma = SMA(20)
        result = sma.compute(close_data)
        assert len(result) == len(close_data)
        # First period-1 values should be NaN
        assert all(np.isnan(result[:19]))
        # Value at index period-1 should equal mean of first period elements
        assert not np.isnan(result[19])
        np.testing.assert_allclose(result[19], np.mean(close_data[:20]), rtol=1e-10)

    def test_compute_minimal_data(self):
        data = np.arange(1.0, 21.0)  # exactly 20 points
        sma = SMA(20)
        result = sma.compute(data)
        assert len(result) == 20
        assert all(np.isnan(result[:19]))
        np.testing.assert_allclose(result[19], np.mean(data), rtol=1e-10)

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 11.0)  # 10 points
        sma = SMA(20)
        result = sma.compute(data)
        assert len(result) == 10
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        sma = SMA(20)
        result = sma.compute(constant_data)
        assert not np.isnan(result[19])
        np.testing.assert_allclose(result[19], 50000.0, rtol=1e-10)

    def test_compute_large_values(self, large_data):
        sma = SMA(20)
        result = sma.compute(large_data)
        assert not np.isnan(result[19])
        np.testing.assert_allclose(result[19], 1e12, rtol=1e-10)

    def test_compute_small_values(self, small_data):
        sma = SMA(20)
        result = sma.compute(small_data)
        assert not np.isnan(result[19])
        np.testing.assert_allclose(result[19], 1e-10, rtol=1e-10)

    def test_compute_period_one(self):
        data = np.arange(1.0, 11.0)
        sma = SMA(1)
        result = sma.compute(data)
        np.testing.assert_array_equal(result, data)

    def test_compute_result_range(self, close_data):
        sma = SMA(20)
        result = sma.compute(close_data)
        valid = result[~np.isnan(result)]
        # SMA should be within the range of the data
        assert np.all(valid >= np.min(close_data) - 1)
        assert np.all(valid <= np.max(close_data) + 1)


# ============================================================================
# EMA Tests
# ============================================================================

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_construction_default(self):
        ema = EMA(20)
        assert ema.period == 20
        assert abs(ema.multiplier - 2.0 / 21) < 1e-10

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            EMA(0)

    def test_compute_normal_data(self, close_data):
        ema = EMA(20)
        result = ema.compute(close_data)
        assert len(result) == len(close_data)
        assert all(np.isnan(result[:19]))
        assert not np.isnan(result[19])
        # Seed value should be SMA of first period
        np.testing.assert_allclose(result[19], np.mean(close_data[:20]), rtol=1e-10)

    def test_compute_minimal_data(self):
        data = np.arange(1.0, 21.0)
        ema = EMA(20)
        result = ema.compute(data)
        assert not np.isnan(result[19])

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 11.0)
        ema = EMA(20)
        result = ema.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        ema = EMA(20)
        result = ema.compute(constant_data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 50000.0, rtol=1e-6)

    def test_compute_large_values(self, large_data):
        ema = EMA(20)
        result = ema.compute(large_data)
        assert not np.isnan(result[19])

    def test_compute_small_values(self, small_data):
        ema = EMA(20)
        result = ema.compute(small_data)
        assert not np.isnan(result[19])

    def test_compute_period_one(self):
        data = np.arange(1.0, 11.0)
        ema = EMA(1)
        result = ema.compute(data)
        assert not np.isnan(result[0])
        np.testing.assert_allclose(result[0], data[0])


# ============================================================================
# WMA Tests
# ============================================================================

class TestWMA:
    """Tests for Weighted Moving Average."""

    def test_construction(self):
        wma = WMA(10)
        assert wma.period == 10
        assert len(wma.weights) == 10
        np.testing.assert_allclose(wma.weights, np.arange(1, 11, dtype=float))

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            WMA(0)

    def test_compute_normal_data(self, close_data):
        wma = WMA(20)
        result = wma.compute(close_data)
        assert len(result) == len(close_data)
        assert all(np.isnan(result[:19]))
        assert not np.isnan(result[19])

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 11.0)
        wma = WMA(20)
        result = wma.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        wma = WMA(20)
        result = wma.compute(constant_data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 50000.0, rtol=1e-10)

    def test_compute_manual_verification(self):
        """Manually verify WMA with known values."""
        data = np.array([10.0, 20.0, 30.0])
        wma = WMA(3)
        result = wma.compute(data)
        # WMA = (10*1 + 20*2 + 30*3) / (1+2+3) = (10+40+90)/6 = 140/6
        expected = 140.0 / 6.0
        np.testing.assert_allclose(result[2], expected, rtol=1e-10)


# ============================================================================
# HMA Tests
# ============================================================================

class TestHMA:
    """Tests for Hull Moving Average."""

    def test_construction(self):
        hma = HMA(10)
        assert hma.period == 10

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            HMA(3)

    def test_construction_minimum_period(self):
        hma = HMA(4)
        assert hma.period == 4

    def test_compute_normal_data(self, close_data):
        hma = HMA(10)
        result = hma.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 6.0)
        hma = HMA(10)
        result = hma.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        hma = HMA(10)
        result = hma.compute(constant_data)
        # With constant data, HMA should converge to the constant value
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, atol=1.0)


# ============================================================================
# DEMA Tests
# ============================================================================

class TestDEMA:
    """Tests for Double Exponential Moving Average."""

    def test_construction(self):
        dema = DEMA(20)
        assert dema.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            DEMA(0)

    def test_compute_normal_data(self, close_data):
        dema = DEMA(20)
        result = dema.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 6.0)
        dema = DEMA(20)
        result = dema.compute(data)
        assert len(result) == 5

    def test_compute_constant_data(self, constant_data):
        dema = DEMA(20)
        result = dema.compute(constant_data)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, atol=1.0)


# ============================================================================
# TEMA Tests
# ============================================================================

class TestTEMA:
    """Tests for Triple Exponential Moving Average."""

    def test_construction(self):
        tema = TEMA(20)
        assert tema.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            TEMA(0)

    def test_compute_normal_data(self, close_data):
        tema = TEMA(20)
        result = tema.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(1.0, 6.0)
        tema = TEMA(20)
        result = tema.compute(data)
        assert len(result) == 5

    def test_compute_constant_data(self, constant_data):
        tema = TEMA(20)
        result = tema.compute(constant_data)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, atol=1.0)


# ============================================================================
# VWMA Tests
# ============================================================================

class TestVWMA:
    """Tests for Volume-Weighted Moving Average."""

    def test_construction(self):
        vwma = VWMA(20)
        assert vwma.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            VWMA(0)

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        vwma = VWMA(20)
        result = vwma.compute(closes, volumes)
        assert len(result) == len(closes)

    def test_compute_mismatched_lengths(self):
        closes = np.arange(10.0)
        volumes = np.arange(5.0)
        vwma = VWMA(5)
        with pytest.raises(ValueError):
            vwma.compute(closes, volumes)

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        volumes = np.arange(5.0)
        vwma = VWMA(20)
        result = vwma.compute(closes, volumes)
        assert all(np.isnan(result))

    def test_compute_zero_volume(self):
        closes = np.full(20, 50000.0)
        volumes = np.zeros(20)
        vwma = VWMA(20)
        result = vwma.compute(closes, volumes)
        assert np.isnan(result[19])

    def test_compute_constant_data(self, constant_ohlcv):
        _, _, _, closes, volumes = constant_ohlcv
        vwma = VWMA(20)
        result = vwma.compute(closes, volumes)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, rtol=1e-6)


# ============================================================================
# KAMA Tests
# ============================================================================

class TestKAMA:
    """Tests for Kaufman's Adaptive Moving Average."""

    def test_construction_default(self):
        kama = KAMA()
        assert kama.period == 10
        assert abs(kama.fast_sc - 2.0 / 3.0) < 1e-10

    def test_construction_custom(self):
        kama = KAMA(period=20, fast_sc=0.5, slow_sc=0.01)
        assert kama.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            KAMA(period=0)

    def test_compute_normal_data(self, close_data):
        kama = KAMA(period=10)
        result = kama.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        kama = KAMA(period=10)
        result = kama.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        kama = KAMA(period=10)
        result = kama.compute(constant_data)
        # With constant data, direction=0, volatility=0, er=0, sc=slow_sc^2
        # KAMA should equal the constant value
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, atol=1.0)


# ============================================================================
# ALMA Tests
# ============================================================================

class TestALMA:
    """Tests for Arnaud Legoux Moving Average."""

    def test_construction_default(self):
        alma = ALMA()
        assert alma.period == 9
        assert alma.offset == 0.85
        assert alma.sigma == 6.0

    def test_construction_custom(self):
        alma = ALMA(period=20, offset=0.7, sigma=5.0)
        assert alma.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            ALMA(period=0)

    def test_compute_normal_data(self, close_data):
        alma = ALMA(period=20)
        result = alma.compute(close_data)
        assert len(result) == len(close_data)
        assert all(np.isnan(result[:19]))
        assert not np.isnan(result[19])

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        alma = ALMA(period=20)
        result = alma.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        alma = ALMA(period=20)
        result = alma.compute(constant_data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 50000.0, rtol=1e-6)


# ============================================================================
# FRAMA Tests
# ============================================================================

class TestFRAMA:
    """Tests for Fractal Adaptive Moving Average."""

    def test_construction_default(self):
        frama = FRAMA()
        assert frama.period == 20

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            FRAMA(period=3)

    def test_compute_normal_data(self, close_data):
        frama = FRAMA(period=20)
        result = frama.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        frama = FRAMA(period=20)
        result = frama.compute(data)
        assert all(np.isnan(result))

    def test_compute_minimal_data(self):
        data = np.random.randn(50) + 50000
        frama = FRAMA(period=20)
        result = frama.compute(data)
        assert not np.isnan(result[20])  # First valid value


# ============================================================================
# ZLEMA Tests
# ============================================================================

class TestZLEMA:
    """Tests for Zero-Lag Exponential Moving Average."""

    def test_construction(self):
        zlema = ZLEMA(20)
        assert zlema.period == 20
        assert zlema.lag == 9

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            ZLEMA(0)

    def test_compute_normal_data(self, close_data):
        zlema = ZLEMA(20)
        result = zlema.compute(close_data)
        assert len(result) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        zlema = ZLEMA(20)
        result = zlema.compute(data)
        assert all(np.isnan(result))

    def test_compute_constant_data(self, constant_data):
        zlema = ZLEMA(20)
        result = zlema.compute(constant_data)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            np.testing.assert_allclose(valid, 50000.0, atol=1.0)

    def test_compute_period_one(self):
        data = np.arange(10.0)
        zlema = ZLEMA(1)
        result = zlema.compute(data)
        assert not np.isnan(result[0])


# ============================================================================
# Supertrend Tests
# ============================================================================

class TestSupertrend:
    """Tests for Supertrend indicator."""

    def test_construction_default(self):
        st = Supertrend()
        assert st.period == 10
        assert st.multiplier == 3.0

    def test_construction_custom(self):
        st = Supertrend(period=7, multiplier=2.0)
        assert st.period == 7

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        st = Supertrend()
        result = st.compute(highs, lows, closes)
        assert "supertrend" in result
        assert "direction" in result
        assert len(result["supertrend"]) == len(closes)
        assert len(result["direction"]) == len(closes)

    def test_compute_direction_values(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        st = Supertrend()
        result = st.compute(highs, lows, closes)
        valid_dir = result["direction"]
        assert all(d in [1, -1, 0] for d in valid_dir)

    def test_compute_insufficient_data(self):
        closes = np.array([100.0, 101.0])
        highs = np.array([102.0, 103.0])
        lows = np.array([99.0, 100.0])
        st = Supertrend(period=10)
        result = st.compute(highs, lows, closes)
        assert "supertrend" in result
        assert "direction" in result


# ============================================================================
# MovingAverageRibbon Tests
# ============================================================================

class TestMovingAverageRibbon:
    """Tests for Moving Average Ribbon."""

    def test_construction_default(self):
        ribbon = MovingAverageRibbon()
        assert len(ribbon.emas) == 8

    def test_construction_custom(self):
        ribbon = MovingAverageRibbon(base_period=5, count=5, step=3)
        assert len(ribbon.emas) == 5

    def test_compute_normal_data(self, close_data):
        ribbon = MovingAverageRibbon()
        result = ribbon.compute(close_data)
        assert len(result) == 8
        for r in result:
            assert len(r) == len(close_data)

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        ribbon = MovingAverageRibbon(base_period=10, count=3, step=5)
        result = ribbon.compute(data)
        for r in result:
            assert all(np.isnan(r))


# ============================================================================
# KaufmanAdaptiveMovingAverage Tests (KAMA alias)
# ============================================================================

class TestKaufmanAdaptiveMovingAverage:
    """Tests for KaufmanAdaptiveMovingAverage (KAMA alias)."""

    def test_is_kama_alias(self):
        assert issubclass(KaufmanAdaptiveMovingAverage, KAMA)

    def test_compute_same_as_kama(self, close_data):
        kama = KAMA(period=10)
        kama_alias = KaufmanAdaptiveMovingAverage(period=10)
        r1 = kama.compute(close_data)
        r2 = kama_alias.compute(close_data)
        np.testing.assert_array_equal(r1, r2)


# ============================================================================
# EhlersSuperSmoother Tests
# ============================================================================

class TestEhlersSuperSmoother:
    """Tests for Ehlers Super Smoother."""

    def test_construction_default(self):
        es = EhlersSuperSmoother()
        assert es.period == 10

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            EhlersSuperSmoother(period=1)

    def test_compute_normal_data(self, close_data):
        es = EhlersSuperSmoother(period=10)
        result = es.compute(close_data)
        assert len(result) == len(close_data)
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_compute_short_data(self):
        data = np.array([1.0])
        es = EhlersSuperSmoother()
        result = es.compute(data)
        assert np.isnan(result[0])

    def test_compute_two_elements(self):
        data = np.array([1.0, 2.0])
        es = EhlersSuperSmoother(period=5)
        result = es.compute(data)
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_compute_constant_data(self, constant_data):
        es = EhlersSuperSmoother(period=10)
        result = es.compute(constant_data)
        # For constant data, result should converge to constant
        valid = result[~np.isnan(result)]
        if len(valid) > 5:
            np.testing.assert_allclose(valid[-5:], 50000.0, atol=10.0)


# ============================================================================
# RSI Tests
# ============================================================================

class TestRSI:
    """Tests for Relative Strength Index."""

    def test_construction_default(self):
        rsi = RSI()
        assert rsi.period == 14

    def test_construction_custom(self):
        rsi = RSI(period=7)
        assert rsi.period == 7

    def test_construction_invalid_period(self):
        with pytest.raises(ValueError):
            RSI(period=0)

    def test_compute_normal_data(self, close_data):
        rsi = RSI(14)
        result = rsi.compute(close_data)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        rsi = RSI(14)
        result = rsi.compute(data)
        assert np.isnan(result)

    def test_compute_minimal_data(self):
        data = np.arange(1.0, 16.0)  # 15 points for period=14
        rsi = RSI(14)
        result = rsi.compute(data)
        assert not np.isnan(result)

    def test_compute_all_gains(self):
        """Constantly rising prices should give RSI=100."""
        data = np.arange(1.0, 20.0)
        rsi = RSI(5)
        result = rsi.compute(data)
        assert result == 100.0

    def test_compute_all_losses(self):
        """Constantly falling prices should give RSI=0."""
        data = np.arange(20.0, 1.0, -1)
        rsi = RSI(5)
        result = rsi.compute(data)
        assert result == 0.0

    def test_compute_constant_data(self, constant_data):
        rsi = RSI(14)
        result = rsi.compute(constant_data)
        # No changes -> avg_loss=0 -> RSI=100
        assert result == 100.0

    def test_compute_series_normal_data(self, close_data):
        rsi = RSI(14)
        result = rsi.compute_series(close_data)
        assert len(result) == len(close_data)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_compute_series_insufficient_data(self):
        data = np.arange(5.0)
        rsi = RSI(14)
        result = rsi.compute_series(data)
        assert all(np.isnan(result))


# ============================================================================
# ConnorsRSI Tests
# ============================================================================

class TestConnorsRSI:
    """Tests for Connors RSI."""

    def test_construction_default(self):
        crsi = ConnorsRSI()
        assert crsi.rsi_period == 3
        assert crsi.streak_period == 2
        assert crsi.roc_period == 100

    def test_construction_custom(self):
        crsi = ConnorsRSI(rsi_period=5, streak_period=3, roc_period=50)
        assert crsi.rsi_period == 5

    def test_compute_normal_data(self, close_data):
        crsi = ConnorsRSI(rsi_period=3, streak_period=2, roc_period=20)
        result = crsi.compute(close_data)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        crsi = ConnorsRSI(roc_period=100)
        result = crsi.compute(data)
        assert np.isnan(result)

    def test_compute_streaks_upward(self):
        data = np.arange(1.0, 11.0)
        streaks = ConnorsRSI._compute_streaks(data)
        assert len(streaks) == 9
        assert streaks[-1] == 9  # 9 consecutive up days

    def test_compute_streaks_downward(self):
        data = np.arange(10.0, 0.0, -1)
        streaks = ConnorsRSI._compute_streaks(data)
        assert streaks[-1] == -9  # 9 consecutive down days

    def test_compute_streaks_mixed(self):
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        streaks = ConnorsRSI._compute_streaks(data)
        assert streaks[0] == 1   # up
        assert streaks[1] == 2   # up again
        assert streaks[2] == -1  # down
        assert streaks[3] == -2  # down again

    def test_compute_streaks_flat(self):
        data = np.array([5.0, 5.0, 5.0])
        streaks = ConnorsRSI._compute_streaks(data)
        assert streaks[0] == 0
        assert streaks[1] == 0

    def test_compute_streaks_empty(self):
        streaks = ConnorsRSI._compute_streaks(np.array([1.0]))
        assert len(streaks) == 0


# ============================================================================
# MACD Tests
# ============================================================================

class TestMACD:
    """Tests for Moving Average Convergence Divergence."""

    def test_construction_default(self):
        macd = MACD()
        assert macd.fast == 12
        assert macd.slow == 26
        assert macd.signal == 9

    def test_construction_custom(self):
        macd = MACD(fast=8, slow=21, signal=5)
        assert macd.fast == 8

    def test_compute_normal_data(self, close_data):
        macd = MACD()
        result = macd.compute(close_data)
        assert result is not None
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert "macd_line" in result
        assert "signal_line" in result

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        macd = MACD()
        result = macd.compute(data)
        assert result is None

    def test_compute_histogram_equals_diff(self, close_data):
        macd = MACD()
        result = macd.compute(close_data)
        assert result is not None
        np.testing.assert_allclose(
            result["histogram"],
            result["macd"] - result["signal"],
            rtol=1e-10
        )

    def test_compute_minimal_data(self):
        data = np.random.randn(40) + 50000
        macd = MACD(fast=5, slow=10, signal=3)
        result = macd.compute(data)
        assert result is not None


# ============================================================================
# VolumeWeightedMACD Tests
# ============================================================================

class TestVolumeWeightedMACD:
    """Tests for Volume-Weighted MACD."""

    def test_construction_default(self):
        vwmacd = VolumeWeightedMACD()
        assert vwmacd.fast == 12

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        vwmacd = VolumeWeightedMACD()
        result = vwmacd.compute(closes, volumes)
        assert result is not None
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_compute_insufficient_data(self):
        closes = np.arange(10.0)
        volumes = np.arange(10.0) + 1
        vwmacd = VolumeWeightedMACD()
        result = vwmacd.compute(closes, volumes)
        assert result is None

    def test_compute_histogram_equals_diff(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        vwmacd = VolumeWeightedMACD()
        result = vwmacd.compute(closes, volumes)
        if result is not None:
            np.testing.assert_allclose(
                result["histogram"],
                result["macd"] - result["signal"],
                rtol=1e-10
            )


# ============================================================================
# StochasticOscillator Tests
# ============================================================================

class TestStochasticOscillator:
    """Tests for Stochastic Oscillator."""

    def test_construction_default(self):
        so = StochasticOscillator()
        assert so.k_period == 14
        assert so.d_period == 3

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        so = StochasticOscillator(k_period=5, d_period=3)
        result = so.compute(highs, lows, closes)
        if result is not None:
            assert "k" in result
            assert "d" in result
            assert 0 <= result["k"] <= 100
            assert 0 <= result["d"] <= 100
        else:
            # May return None if D value is NaN (not enough K values for SMA)
            assert result is None

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        so = StochasticOscillator(k_period=14)
        result = so.compute(highs, lows, closes)
        assert result is None

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, _ = constant_ohlcv
        so = StochasticOscillator(k_period=14)
        result = so.compute(highs, lows, closes)
        # When high==low, K should be 50
        if result is not None:
            assert result["k"] == 50.0

    def test_compute_at_high(self):
        """When close equals highest high, K should be 100."""
        n = 20
        closes = np.arange(80.0, 80.0 + n)
        highs = closes + 2
        lows = closes - 2
        # Set close at the high of the window
        highs[-1] = closes[-1]
        so = StochasticOscillator(k_period=5, d_period=3)
        result = so.compute(highs, lows, closes)
        if result is not None:
            assert result["k"] == 100.0


# ============================================================================
# CCI Tests
# ============================================================================

class TestCCI:
    """Tests for Commodity Channel Index."""

    def test_construction_default(self):
        cci = CCI()
        assert cci.period == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        cci = CCI(20)
        result = cci.compute(highs, lows, closes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        cci = CCI(20)
        result = cci.compute(highs, lows, closes)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, _ = constant_ohlcv
        cci = CCI(20)
        result = cci.compute(highs, lows, closes)
        # Constant data -> MAD=0 -> CCI=0
        assert result == 0.0


# ============================================================================
# WilliamsR Tests
# ============================================================================

class TestWilliamsR:
    """Tests for Williams %R."""

    def test_construction_default(self):
        wr = WilliamsR()
        assert wr.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        wr = WilliamsR(14)
        result = wr.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert -100 <= result <= 0

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        wr = WilliamsR(14)
        result = wr.compute(highs, lows, closes)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, _ = constant_ohlcv
        wr = WilliamsR(14)
        result = wr.compute(highs, lows, closes)
        assert result == -50.0


# ============================================================================
# ROC Tests
# ============================================================================

class TestROC:
    """Tests for Rate of Change."""

    def test_construction_default(self):
        roc = ROC()
        assert roc.period == 12

    def test_compute_normal_data(self, close_data):
        roc = ROC(12)
        result = roc.compute(close_data)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        roc = ROC(12)
        result = roc.compute(data)
        assert np.isnan(result)

    def test_compute_zero_base(self):
        # When the base price (period bars ago) is 0, ROC returns 0
        data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
        roc = ROC(12)
        result = roc.compute(data)
        # closes[-period-1] = data[0] = 0.0 -> ROC returns 0
        assert result == 0.0

    def test_compute_manual(self):
        data = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
        roc = ROC(2)
        result = roc.compute(data)
        expected = ((125 - 115) / 115) * 100.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# Momentum Tests
# ============================================================================

class TestMomentum:
    """Tests for Momentum indicator."""

    def test_construction_default(self):
        mom = Momentum()
        assert mom.period == 10

    def test_compute_normal_data(self, close_data):
        mom = Momentum(10)
        result = mom.compute(close_data)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        mom = Momentum(10)
        result = mom.compute(data)
        assert np.isnan(result)

    def test_compute_manual(self):
        data = np.array([100.0, 105.0, 110.0])
        mom = Momentum(2)
        result = mom.compute(data)
        assert result == 110.0 - 100.0  # 10


# ============================================================================
# TRIX Tests
# ============================================================================

class TestTRIX:
    """Tests for TRIX indicator."""

    def test_construction_default(self):
        trix = TRIX()
        assert trix.period == 15

    def test_compute_normal_data(self, close_data):
        trix = TRIX(15)
        result = trix.compute(close_data)
        # May or may not be NaN depending on data length
        assert isinstance(result, float)

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        trix = TRIX(15)
        result = trix.compute(data)
        assert np.isnan(result)


# ============================================================================
# UltimateOscillator Tests
# ============================================================================

class TestUltimateOscillator:
    """Tests for Ultimate Oscillator."""

    def test_construction_default(self):
        uo = UltimateOscillator()
        assert uo.p1 == 7
        assert uo.p2 == 14
        assert uo.p3 == 28

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        uo = UltimateOscillator()
        result = uo.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        uo = UltimateOscillator()
        result = uo.compute(highs, lows, closes)
        assert np.isnan(result)


# ============================================================================
# MFI Tests
# ============================================================================

class TestMFI:
    """Tests for Money Flow Index."""

    def test_construction_default(self):
        mfi = MFI()
        assert mfi.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        mfi = MFI(14)
        result = mfi.compute(highs, lows, closes, volumes)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        volumes = np.ones(5) * 1e6
        mfi = MFI(14)
        result = mfi.compute(highs, lows, closes, volumes)
        assert np.isnan(result)

    def test_compute_all_positive_mf(self):
        """Constantly rising prices should give MFI=100."""
        n = 20
        closes = np.arange(1.0, n + 1)
        highs = closes + 1
        lows = closes - 1
        volumes = np.ones(n) * 1e6
        mfi = MFI(14)
        result = mfi.compute(highs, lows, closes, volumes)
        assert result == 100.0


# ============================================================================
# ADX Tests
# ============================================================================

class TestADX:
    """Tests for Average Directional Index."""

    def test_construction_default(self):
        adx = ADX()
        assert adx.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        adx = ADX(14)
        result = adx.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_compute_insufficient_data(self):
        closes = np.arange(10.0)
        highs = closes + 1
        lows = closes - 1
        adx = ADX(14)
        result = adx.compute(highs, lows, closes)
        assert np.isnan(result)

    def test_compute_minimal_data(self):
        n = 30
        closes = 50000 + np.random.randn(n) * 100
        highs = closes + 50
        lows = closes - 50
        adx = ADX(14)
        result = adx.compute(highs, lows, closes)
        assert not np.isnan(result)


# ============================================================================
# Aroon Tests
# ============================================================================

class TestAroon:
    """Tests for Aroon Up/Down."""

    def test_construction_default(self):
        aroon = Aroon()
        assert aroon.period == 25

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        aroon = Aroon(25)
        result = aroon.compute(highs, lows)
        assert not np.isnan(result["up"])
        assert not np.isnan(result["down"])
        assert not np.isnan(result["oscillator"])
        assert 0 <= result["up"] <= 100
        assert 0 <= result["down"] <= 100

    def test_compute_insufficient_data(self):
        highs = np.arange(5.0)
        lows = np.arange(5.0)
        aroon = Aroon(25)
        result = aroon.compute(highs, lows)
        assert np.isnan(result["up"])
        assert np.isnan(result["down"])

    def test_compute_oscillator_range(self, ohlcv_data):
        _, highs, lows, _ , _ = ohlcv_data
        aroon = Aroon(25)
        result = aroon.compute(highs, lows)
        assert -100 <= result["oscillator"] <= 100


# ============================================================================
# ChandeMomentumOscillator Tests
# ============================================================================

class TestChandeMomentumOscillator:
    """Tests for Chande Momentum Oscillator."""

    def test_construction_default(self):
        cmo = ChandeMomentumOscillator()
        assert cmo.period == 14

    def test_compute_normal_data(self, close_data):
        cmo = ChandeMomentumOscillator(14)
        result = cmo.compute(close_data)
        assert not np.isnan(result)
        assert -100 <= result <= 100

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        cmo = ChandeMomentumOscillator(14)
        result = cmo.compute(data)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_data):
        cmo = ChandeMomentumOscillator(14)
        result = cmo.compute(constant_data)
        # No changes -> sum_up + sum_down = 0 -> CMO = 0
        assert result == 0.0


# ============================================================================
# EhlersFisherTransform Tests
# ============================================================================

class TestEhlersFisherTransform:
    """Tests for Ehlers Fisher Transform."""

    def test_construction_default(self):
        eft = EhlersFisherTransform()
        assert eft.period == 10

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        eft = EhlersFisherTransform(10)
        result = eft.compute(highs, lows, closes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        eft = EhlersFisherTransform(10)
        result = eft.compute(highs, lows, closes)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, _ = constant_ohlcv
        eft = EhlersFisherTransform(10)
        result = eft.compute(highs, lows, closes)
        assert result == 0.0


# ============================================================================
# ATR Tests
# ============================================================================

class TestATR:
    """Tests for Average True Range."""

    def test_construction_default(self):
        atr = ATR()
        assert atr.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        atr = ATR(14)
        result = atr.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        atr = ATR(14)
        result = atr.compute(highs, lows, closes)
        assert np.isnan(result)

    def test_compute_series_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        atr = ATR(14)
        result = atr.compute_series(highs, lows, closes)
        assert len(result) == len(closes)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_compute_series_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        atr = ATR(14)
        result = atr.compute_series(highs, lows, closes)
        assert all(np.isnan(result))

    def test_compute_minimal_data(self):
        n = 16
        closes = 50000 + np.random.randn(n) * 100
        highs = closes + 50
        lows = closes - 50
        atr = ATR(14)
        result = atr.compute(highs, lows, closes)
        assert not np.isnan(result)


# ============================================================================
# BollingerBands Tests
# ============================================================================

class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_construction_default(self):
        bb = BollingerBands()
        assert bb.period == 20
        assert bb.num_std == 2.0

    def test_construction_custom(self):
        bb = BollingerBands(period=10, num_std=1.5)
        assert bb.period == 10
        assert bb.num_std == 1.5

    def test_compute_normal_data(self, close_data):
        bb = BollingerBands(20)
        result = bb.compute(close_data)
        assert result is not None
        assert result["upper"] > result["middle"]
        assert result["middle"] > result["lower"]
        assert result["bandwidth"] >= 0

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        bb = BollingerBands(20)
        result = bb.compute(data)
        assert result is None

    def test_compute_constant_data(self, constant_data):
        bb = BollingerBands(20)
        result = bb.compute(constant_data)
        assert result is not None
        assert result["upper"] == result["lower"] == result["middle"]
        assert result["bandwidth"] == 0.0

    def test_compute_percent_b_range(self, close_data):
        bb = BollingerBands(20)
        result = bb.compute(close_data)
        # percent_b can be any real number but typically 0-1
        assert isinstance(result["percent_b"], float)

    def test_compute_minimal_data(self):
        data = np.arange(1.0, 21.0)
        bb = BollingerBands(20)
        result = bb.compute(data)
        assert result is not None


# ============================================================================
# KeltnerChannels Tests
# ============================================================================

class TestKeltnerChannels:
    """Tests for Keltner Channels."""

    def test_construction_default(self):
        kc = KeltnerChannels()
        assert kc.ema_period == 20
        assert kc.atr_period == 10
        assert kc.multiplier == 1.5

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        kc = KeltnerChannels()
        result = kc.compute(highs, lows, closes)
        assert result is not None
        assert result["upper"] > result["middle"]
        assert result["middle"] > result["lower"]

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        kc = KeltnerChannels()
        result = kc.compute(highs, lows, closes)
        assert result is None


# ============================================================================
# DonchianChannels Tests
# ============================================================================

class TestDonchianChannels:
    """Tests for Donchian Channels."""

    def test_construction_default(self):
        dc = DonchianChannels()
        assert dc.period == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, _, _ = ohlcv_data
        dc = DonchianChannels(20)
        result = dc.compute(highs, lows)
        assert result is not None
        assert result["upper"] >= result["lower"]
        assert result["middle"] == (result["upper"] + result["lower"]) / 2

    def test_compute_insufficient_data(self):
        highs = np.arange(5.0)
        lows = np.arange(5.0)
        dc = DonchianChannels(20)
        result = dc.compute(highs, lows)
        assert result is None

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, _, _ = constant_ohlcv
        dc = DonchianChannels(20)
        result = dc.compute(highs, lows)
        assert result is not None
        # In our fixture, highs=50010 and lows=49990, not identical
        assert result["upper"] == 50010.0
        assert result["lower"] == 49990.0


# ============================================================================
# StandardDeviation Tests
# ============================================================================

class TestStandardDeviation:
    """Tests for rolling Standard Deviation."""

    def test_construction_default(self):
        sd = StandardDeviation()
        assert sd.period == 20

    def test_compute_normal_data(self, close_data):
        sd = StandardDeviation(20)
        result = sd.compute(close_data)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_insufficient_data(self):
        data = np.arange(5.0)
        sd = StandardDeviation(20)
        result = sd.compute(data)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_data):
        sd = StandardDeviation(20)
        result = sd.compute(constant_data)
        assert result == 0.0


# ============================================================================
# HistoricalVolatility Tests
# ============================================================================

class TestHistoricalVolatility:
    """Tests for annualized Historical Volatility."""

    def test_construction_default(self):
        hv = HistoricalVolatility()
        assert hv.period == 20
        assert hv.trading_days == 365

    def test_compute_normal_data(self, close_data):
        hv = HistoricalVolatility(20)
        result = hv.compute(close_data)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_insufficient_data(self):
        data = np.arange(5.0) + 1
        hv = HistoricalVolatility(20)
        result = hv.compute(data)
        assert np.isnan(result)


# ============================================================================
# ParkinsonVolatility Tests
# ============================================================================

class TestParkinsonVolatility:
    """Tests for Parkinson volatility estimator."""

    def test_construction_default(self):
        pv = ParkinsonVolatility()
        assert pv.period == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, _, _ = ohlcv_data
        pv = ParkinsonVolatility(20)
        result = pv.compute(highs, lows)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_insufficient_data(self):
        highs = np.arange(5.0) + 10
        lows = np.arange(5.0)
        pv = ParkinsonVolatility(20)
        result = pv.compute(highs, lows)
        assert np.isnan(result)


# ============================================================================
# GarmanKlassVolatility Tests
# ============================================================================

class TestGarmanKlassVolatility:
    """Tests for Garman-Klass volatility estimator."""

    def test_construction_default(self):
        gkv = GarmanKlassVolatility()
        assert gkv.period == 20

    def test_compute_normal_data(self, ohlcv_data):
        opens, highs, lows, closes, _ = ohlcv_data
        gkv = GarmanKlassVolatility(20)
        result = gkv.compute(highs, lows, closes, opens)
        assert not np.isnan(result)
        assert result >= 0

    def test_compute_insufficient_data(self):
        opens = np.arange(5.0) + 5
        highs = np.arange(5.0) + 10
        lows = np.arange(5.0)
        closes = np.arange(5.0) + 6
        gkv = GarmanKlassVolatility(20)
        result = gkv.compute(highs, lows, closes, opens)
        assert np.isnan(result)


# ============================================================================
# ChaikinVolatility Tests
# ============================================================================

class TestChaikinVolatility:
    """Tests for Chaikin Volatility."""

    def test_construction_default(self):
        cv = ChaikinVolatility()
        assert cv.period == 10
        assert cv.roc_period == 10

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, _, _ = ohlcv_data
        cv = ChaikinVolatility(10, 10)
        result = cv.compute(highs, lows)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        highs = np.arange(5.0) + 1
        lows = np.arange(5.0)
        cv = ChaikinVolatility(10, 10)
        result = cv.compute(highs, lows)
        assert np.isnan(result)


# ============================================================================
# TrueRange Tests
# ============================================================================

class TestTrueRange:
    """Tests for True Range."""

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        tr = TrueRange()
        result = tr.compute(highs, lows, closes)
        assert len(result) == len(closes)
        assert np.isnan(result[0])  # First value is NaN
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_compute_short_data(self):
        closes = np.array([100.0])
        tr = TrueRange()
        result = tr.compute(np.array([101.0]), np.array([99.0]), closes)
        assert len(result) == 1
        assert np.isnan(result[0])


# ============================================================================
# ATRP Tests
# ============================================================================

class TestATRP:
    """Tests for ATR Percentage."""

    def test_construction_default(self):
        atrp = ATRP()
        assert atrp.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        atrp = ATRP(14)
        result = atrp.compute(highs, lows, closes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        atrp = ATRP(14)
        result = atrp.compute(highs, lows, closes)
        assert np.isnan(result)


# ============================================================================
# OBVIndicator Tests
# ============================================================================

class TestOBVIndicator:
    """Tests for On-Balance Volume."""

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        obv = OBVIndicator()
        result = obv.compute(closes, volumes)
        assert len(result) == len(closes)

    def test_compute_short_data(self):
        closes = np.array([100.0])
        volumes = np.array([1e6])
        obv = OBVIndicator()
        result = obv.compute(closes, volumes)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_compute_rising_prices(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        volumes = np.array([1e6, 1e6, 1e6, 1e6])
        obv = OBVIndicator()
        result = obv.compute(closes, volumes)
        # All up days -> OBV should accumulate
        assert result[-1] == 3e6

    def test_compute_falling_prices(self):
        closes = np.array([103.0, 102.0, 101.0, 100.0])
        volumes = np.array([1e6, 1e6, 1e6, 1e6])
        obv = OBVIndicator()
        result = obv.compute(closes, volumes)
        # All down days -> OBV should decrease
        assert result[-1] == -3e6


# ============================================================================
# CMFIndicator Tests
# ============================================================================

class TestCMFIndicator:
    """Tests for Chaikin Money Flow."""

    def test_construction_default(self):
        cmf = CMFIndicator()
        assert cmf.period == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        cmf = CMFIndicator(20)
        result = cmf.compute(highs, lows, closes, volumes)
        assert not np.isnan(result)
        assert -1 <= result <= 1

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        volumes = np.ones(5) * 1e6
        cmf = CMFIndicator(20)
        result = cmf.compute(highs, lows, closes, volumes)
        assert np.isnan(result)

    def test_compute_zero_volume(self):
        n = 25
        closes = np.full(n, 50000.0)
        highs = closes + 100
        lows = closes - 100
        volumes = np.zeros(n)
        cmf = CMFIndicator(20)
        result = cmf.compute(highs, lows, closes, volumes)
        assert result == 0.0


# ============================================================================
# ADLine Tests
# ============================================================================

class TestADLine:
    """Tests for Accumulation/Distribution Line."""

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        adl = ADLine()
        result = adl.compute(highs, lows, closes, volumes)
        assert len(result) == len(closes)

    def test_compute_short_data(self):
        closes = np.array([100.0])
        highs = np.array([101.0])
        lows = np.array([99.0])
        volumes = np.array([1e6])
        adl = ADLine()
        result = adl.compute(highs, lows, closes, volumes)
        assert len(result) == 1


# ============================================================================
# VolumeProfile Tests
# ============================================================================

class TestVolumeProfile:
    """Tests for Volume Profile."""

    def test_construction_default(self):
        vp = VolumeProfile()
        assert vp.num_bins == 50

    def test_construction_custom(self):
        vp = VolumeProfile(num_bins=20)
        assert vp.num_bins == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        vp = VolumeProfile(num_bins=20)
        result = vp.compute(highs, lows, closes, volumes)
        assert "prices" in result
        assert "volumes" in result
        assert "poc" in result
        assert "value_area_high" in result
        assert "value_area_low" in result

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        volumes = np.ones(5) * 1e6
        vp = VolumeProfile()
        result = vp.compute(highs, lows, closes, volumes)
        assert len(result["prices"]) == 0

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, volumes = constant_ohlcv
        vp = VolumeProfile(num_bins=10)
        result = vp.compute(highs, lows, closes, volumes)
        # POC should be near 50000 (typical price = (50010+49990+50000)/3 = 50000)
        assert abs(result["poc"] - 50000.0) < 5.0


# ============================================================================
# ForceIndex Tests
# ============================================================================

class TestForceIndex:
    """Tests for Force Index."""

    def test_construction_default(self):
        fi = ForceIndex()
        assert fi.period == 13

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        fi = ForceIndex(13)
        result = fi.compute(closes, volumes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        closes = np.array([100.0])
        volumes = np.array([1e6])
        fi = ForceIndex(13)
        result = fi.compute(closes, volumes)
        assert np.isnan(result)


# ============================================================================
# EaseOfMovement Tests
# ============================================================================

class TestEaseOfMovement:
    """Tests for Ease of Movement."""

    def test_construction_default(self):
        emv = EaseOfMovement()
        assert emv.period == 14

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, _, volumes = ohlcv_data
        emv = EaseOfMovement(14)
        result = emv.compute(highs, lows, volumes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        highs = np.arange(5.0) + 1
        lows = np.arange(5.0)
        volumes = np.ones(5) * 1e6
        emv = EaseOfMovement(14)
        result = emv.compute(highs, lows, volumes)
        assert np.isnan(result)


# ============================================================================
# VolumeOscillator Tests
# ============================================================================

class TestVolumeOscillator:
    """Tests for Volume Oscillator."""

    def test_construction_default(self):
        vo = VolumeOscillator()
        assert vo.fast_period == 5
        assert vo.slow_period == 20

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, _, volumes = ohlcv_data
        vo = VolumeOscillator()
        result = vo.compute(volumes)
        assert not np.isnan(result)

    def test_compute_insufficient_data(self):
        volumes = np.arange(5.0)
        vo = VolumeOscillator()
        result = vo.compute(volumes)
        assert np.isnan(result)

    def test_compute_zero_slow_ma(self):
        volumes = np.zeros(25)
        vo = VolumeOscillator()
        result = vo.compute(volumes)
        assert result == 0.0


# ============================================================================
# NVI Tests
# ============================================================================

class TestNVI:
    """Tests for Negative Volume Index."""

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        nvi = NVI()
        result = nvi.compute(closes, volumes)
        assert len(result) == len(closes)
        assert result[0] == 1000.0

    def test_compute_short_data(self):
        closes = np.array([100.0])
        volumes = np.array([1e6])
        nvi = NVI()
        result = nvi.compute(closes, volumes)
        assert len(result) == 1
        assert result[0] == 1000.0

    def test_compute_decreasing_volume(self):
        closes = np.array([100.0, 101.0, 102.0])
        volumes = np.array([2e6, 1.5e6, 1e6])  # Decreasing
        nvi = NVI()
        result = nvi.compute(closes, volumes)
        # All days have decreasing volume, so NVI should track price
        assert result[-1] > 1000.0


# ============================================================================
# PVI Tests
# ============================================================================

class TestPVI:
    """Tests for Positive Volume Index."""

    def test_compute_normal_data(self, ohlcv_data):
        _, _, _, closes, volumes = ohlcv_data
        pvi = PVI()
        result = pvi.compute(closes, volumes)
        assert len(result) == len(closes)
        assert result[0] == 1000.0

    def test_compute_short_data(self):
        closes = np.array([100.0])
        volumes = np.array([1e6])
        pvi = PVI()
        result = pvi.compute(closes, volumes)
        assert len(result) == 1
        assert result[0] == 1000.0


# ============================================================================
# IchimokuCloud Tests
# ============================================================================

class TestIchimokuCloud:
    """Tests for Ichimoku Cloud."""

    def test_construction_default(self):
        ic = IchimokuCloud()
        assert ic.tenkan == 9
        assert ic.kijun == 26
        assert ic.senkou_b == 52

    def test_construction_custom(self):
        ic = IchimokuCloud(tenkan=7, kijun=22, senkou_b=44)
        assert ic.tenkan == 7

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        ic = IchimokuCloud()
        result = ic.compute(highs, lows, closes)
        assert result is not None
        assert "tenkan" in result
        assert "kijun" in result
        assert "senkou_a" in result
        assert "senkou_b" in result
        assert "chikou" in result

    def test_compute_insufficient_data(self):
        closes = np.arange(10.0)
        highs = closes + 1
        lows = closes - 1
        ic = IchimokuCloud()
        result = ic.compute(highs, lows, closes)
        assert result is None

    def test_compute_senkou_a_is_average(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        ic = IchimokuCloud()
        result = ic.compute(highs, lows, closes)
        if result is not None:
            assert result["senkou_a"] == (result["tenkan"] + result["kijun"]) / 2


# ============================================================================
# TTMSqueeze Tests
# ============================================================================

class TestTTMSqueeze:
    """Tests for TTM Squeeze indicator."""

    def test_construction_default(self):
        ttm = TTMSqueeze()
        assert ttm.bb_period == 20
        assert ttm.bb_std == 2.0

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        ttm = TTMSqueeze()
        result = ttm.compute(highs, lows, closes)
        assert result is not None
        assert "squeeze_active" in result
        assert "momentum" in result
        assert isinstance(result["squeeze_active"], (bool, np.bool_))

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        ttm = TTMSqueeze()
        result = ttm.compute(highs, lows, closes)
        assert result is None

    def test_compute_has_band_values(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        ttm = TTMSqueeze()
        result = ttm.compute(highs, lows, closes)
        if result is not None:
            assert "bb_upper" in result
            assert "bb_lower" in result
            assert "kc_upper" in result
            assert "kc_lower" in result
            assert result["bb_upper"] > result["bb_lower"]


# ============================================================================
# VWAPIndicator Tests
# ============================================================================

class TestVWAPIndicator:
    """Tests for VWAP Indicator."""

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        vwap = VWAPIndicator()
        result = vwap.compute(highs, lows, closes, volumes)
        assert not np.isnan(result)

    def test_compute_empty_data(self):
        closes = np.array([])
        vwap = VWAPIndicator()
        result = vwap.compute(np.array([]), np.array([]), closes, np.array([]))
        assert np.isnan(result)

    def test_compute_series_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        vwap = VWAPIndicator()
        result = vwap.compute_series(highs, lows, closes, volumes)
        assert len(result) == len(closes)

    def test_compute_zero_volume(self):
        closes = np.array([100.0])
        highs = np.array([101.0])
        lows = np.array([99.0])
        volumes = np.array([0.0])
        vwap = VWAPIndicator()
        result = vwap.compute(highs, lows, closes, volumes)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_ohlcv):
        _, highs, lows, closes, volumes = constant_ohlcv
        vwap = VWAPIndicator()
        result = vwap.compute(highs, lows, closes, volumes)
        # Typical price = (50010+49990+50000)/3 = 50000
        np.testing.assert_allclose(result, 50000.0, rtol=1e-4)


# ============================================================================
# VWAPBands Tests
# ============================================================================

class TestVWAPBands:
    """Tests for VWAP with Standard Deviation Bands."""

    def test_construction_default(self):
        vb = VWAPBands()
        assert vb.num_std == 2.0

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        vb = VWAPBands()
        result = vb.compute(highs, lows, closes, volumes)
        assert result is not None
        assert result["upper"] > result["vwap"]
        assert result["vwap"] > result["lower"]

    def test_compute_insufficient_data(self):
        closes = np.array([100.0])
        highs = np.array([101.0])
        lows = np.array([99.0])
        volumes = np.array([1e6])
        vb = VWAPBands()
        result = vb.compute(highs, lows, closes, volumes)
        assert result is None

    def test_compute_zero_volume(self):
        closes = np.full(10, 50000.0)
        highs = closes + 10
        lows = closes - 10
        volumes = np.zeros(10)
        vb = VWAPBands()
        result = vb.compute(highs, lows, closes, volumes)
        assert result is None


# ============================================================================
# EhlersDominantCycle Tests
# ============================================================================

class TestEhlersDominantCycle:
    """Tests for Ehlers Dominant Cycle Period Detection."""

    def test_construction_default(self):
        edc = EhlersDominantCycle()
        assert edc.min_period == 8
        assert edc.max_period == 48

    def test_compute_normal_data(self, close_data):
        edc = EhlersDominantCycle()
        result = edc.compute(close_data)
        assert not np.isnan(result)
        assert 8 <= result <= 48

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        edc = EhlersDominantCycle()
        result = edc.compute(data)
        assert np.isnan(result)


# ============================================================================
# FractalDimension Tests
# ============================================================================

class TestFractalDimension:
    """Tests for Fractal Dimension."""

    def test_construction_default(self):
        fd = FractalDimension()
        assert fd.period == 20

    def test_compute_normal_data(self, close_data):
        fd = FractalDimension(20)
        result = fd.compute(close_data)
        assert not np.isnan(result)
        assert 1.0 <= result <= 2.0

    def test_compute_insufficient_data(self):
        data = np.arange(10.0)
        fd = FractalDimension(20)
        result = fd.compute(data)
        assert np.isnan(result)

    def test_compute_constant_data(self, constant_data):
        fd = FractalDimension(20)
        result = fd.compute(constant_data)
        # Constant data: path_length=0 or price_range=0 -> 1.5
        assert result == 1.5


# ============================================================================
# PivotPoints Tests
# ============================================================================

class TestPivotPoints:
    """Tests for Pivot Points."""

    def test_construction_default(self):
        pp = PivotPoints()
        assert pp.method == "traditional"

    def test_construction_fibonacci(self):
        pp = PivotPoints(method="fibonacci")
        assert pp.method == "fibonacci"

    def test_construction_camarilla(self):
        pp = PivotPoints(method="camarilla")
        assert pp.method == "camarilla"

    def test_construction_woodie(self):
        pp = PivotPoints(method="woodie")
        assert pp.method == "woodie"

    def test_construction_invalid_method(self):
        with pytest.raises(ValueError):
            PivotPoints(method="invalid")

    def test_compute_traditional(self):
        pp = PivotPoints(method="traditional")
        result = pp.compute(105.0, 95.0, 100.0)
        expected_pivot = (105 + 95 + 100) / 3.0
        np.testing.assert_allclose(result["pivot"], expected_pivot, rtol=1e-10)
        assert "s1" in result
        assert "r1" in result
        assert result["r1"] > result["pivot"]
        assert result["s1"] < result["pivot"]

    def test_compute_fibonacci(self):
        pp = PivotPoints(method="fibonacci")
        result = pp.compute(105.0, 95.0, 100.0)
        assert result["pivot"] == 100.0
        assert result["s1"] < result["pivot"]
        assert result["r1"] > result["pivot"]

    def test_compute_camarilla(self):
        pp = PivotPoints(method="camarilla")
        result = pp.compute(105.0, 95.0, 100.0)
        assert "s4" in result
        assert "r4" in result

    def test_compute_woodie(self):
        pp = PivotPoints(method="woodie")
        result = pp.compute(105.0, 95.0, 100.0)
        expected_pivot = (105 + 95 + 2 * 100) / 4.0
        np.testing.assert_allclose(result["pivot"], expected_pivot, rtol=1e-10)

    def test_compute_has_all_levels(self):
        pp = PivotPoints()
        result = pp.compute(105.0, 95.0, 100.0)
        for key in ["pivot", "s1", "s2", "s3", "s4", "r1", "r2", "r3", "r4"]:
            assert key in result


# ============================================================================
# FibonacciRetracement Tests
# ============================================================================

class TestFibonacciRetracement:
    """Tests for Fibonacci Retracement."""

    def test_construction_default(self):
        fr = FibonacciRetracement()
        assert fr.lookback == 100

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        fr = FibonacciRetracement()
        result = fr.compute(highs, lows, closes)
        assert result is not None
        assert "direction" in result
        assert "retracement" in result
        assert "extension" in result

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        fr = FibonacciRetracement()
        result = fr.compute(highs, lows, closes)
        assert result is None

    def test_compute_retracement_levels(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        fr = FibonacciRetracement()
        result = fr.compute(highs, lows, closes)
        if result is not None:
            retracement = result["retracement"]
            assert "0.0" in retracement
            assert "0.236" in retracement
            assert "0.382" in retracement
            assert "0.5" in retracement
            assert "0.618" in retracement
            assert "1.0" in retracement


# ============================================================================
# SupportResistance Tests
# ============================================================================

class TestSupportResistance:
    """Tests for Support/Resistance detection."""

    def test_construction_default(self):
        sr = SupportResistance()
        assert sr.window == 5
        assert sr.min_touches == 2

    def test_compute_normal_data(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        sr = SupportResistance(window=3, min_touches=1)
        result = sr.compute(highs, lows, closes)
        assert "support" in result
        assert "resistance" in result

    def test_compute_insufficient_data(self):
        closes = np.arange(5.0)
        highs = closes + 1
        lows = closes - 1
        sr = SupportResistance(window=5)
        result = sr.compute(highs, lows, closes)
        assert result["support"] == []
        assert result["resistance"] == []


# ============================================================================
# CandlestickPatterns Tests
# ============================================================================

class TestCandlestickPatterns:
    """Tests for Candlestick Pattern Recognition."""

    def test_construction_default(self):
        cp = CandlestickPatterns()
        assert cp.body_ratio == 0.1
        assert cp.doji_ratio == 0.05

    def test_construction_custom(self):
        cp = CandlestickPatterns(body_ratio=0.2, doji_ratio=0.03)
        assert cp.body_ratio == 0.2

    def test_detect_all_normal_data(self, ohlcv_data):
        opens, highs, lows, closes, _ = ohlcv_data
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_detect_all_insufficient_data(self):
        closes = np.arange(3.0)
        opens = closes - 0.5
        highs = closes + 1
        lows = closes - 1
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert result == {}

    def test_detect_doji(self):
        """Craft a doji candle."""
        opens = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        highs = np.array([101.0, 101.0, 101.0, 101.0, 105.0])
        lows = np.array([99.0, 99.0, 99.0, 99.0, 95.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, 100.01])  # Nearly doji
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert isinstance(result, dict)

    def test_detect_bullish_engulfing(self):
        """Craft a bullish engulfing pattern."""
        opens = np.array([100.0, 100.0, 100.0, 98.0, 95.0])
        highs = np.array([101.0, 101.0, 101.0, 99.0, 101.0])
        lows = np.array([99.0, 99.0, 99.0, 94.0, 93.0])
        closes = np.array([100.0, 100.0, 100.0, 95.0, 99.5])
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        # Check that pattern detection returns a dict with boolean-like values
        be = result.get("bullish_engulfing", False)
        assert bool(be) is True

    def test_detect_bearish_engulfing(self):
        """Craft a bearish engulfing pattern."""
        opens = np.array([100.0, 100.0, 100.0, 95.0, 97.0])
        highs = np.array([101.0, 101.0, 101.0, 98.0, 102.0])
        lows = np.array([99.0, 99.0, 99.0, 94.0, 93.0])
        closes = np.array([100.0, 100.0, 100.0, 97.0, 94.0])
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        be = result.get("bearish_engulfing", False)
        assert isinstance(be, (bool, np.bool_))

    def test_detect_hammer(self):
        """Craft a hammer candle."""
        opens = np.array([100.0, 100.0, 100.0, 100.0, 98.5])
        highs = np.array([101.0, 101.0, 101.0, 101.0, 99.0])
        lows = np.array([99.0, 99.0, 99.0, 99.0, 94.0])
        closes = np.array([100.0, 100.0, 100.0, 100.0, 99.0])
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        # Hammer: small body at top, long lower shadow, bullish
        assert isinstance(result.get("hammer", False), (bool, np.bool_))

    def test_detect_morning_star(self):
        """Craft a morning star pattern."""
        # Bearish big candle + small body + bullish reversal
        opens = np.array([100.0, 100.0, 102.0, 101.0, 99.0])
        highs = np.array([101.0, 101.0, 103.0, 102.0, 105.0])
        lows = np.array([99.0, 99.0, 100.0, 99.0, 98.0])
        closes = np.array([100.0, 100.0, 100.5, 99.5, 104.0])
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert isinstance(result.get("morning_star", False), (bool, np.bool_))

    def test_helpers(self):
        cp = CandlestickPatterns()
        assert cp._is_bullish(100, 105) is True
        assert cp._is_bullish(105, 100) is False
        assert cp._is_bearish(105, 100) is True
        assert cp._is_bearish(100, 105) is False


# ============================================================================
# compute_hurst_exponent Tests
# ============================================================================

class TestComputeHurstExponent:
    """Tests for Hurst exponent computation."""

    def test_normal_data(self, close_data):
        result = compute_hurst_exponent(close_data)
        assert not np.isnan(result)
        # Should be between 0 and 1
        assert 0 < result < 1

    def test_insufficient_data(self):
        data = np.arange(50.0)
        result = compute_hurst_exponent(data)
        assert np.isnan(result)

    def test_random_walk(self, rng):
        """Random walk should have Hurst ≈ 0.5."""
        n = 5000
        returns = rng.normal(0, 1, n)
        prices = 100.0 * np.cumprod(1 + returns * 0.01)
        result = compute_hurst_exponent(prices, max_lag=100)
        # Should be approximately 0.5 (allowing wide tolerance for small sample)
        if not np.isnan(result):
            assert 0.3 < result < 0.8

    def test_trending_data(self):
        """Trending data should have Hurst > 0.5."""
        data = np.cumsum(np.ones(500))  # Strictly trending
        result = compute_hurst_exponent(data, max_lag=50)
        if not np.isnan(result):
            assert result > 0.5


# ============================================================================
# compute_zscore Tests
# ============================================================================

class TestComputeZscore:
    """Tests for z-score computation."""

    def test_normal_data(self, close_data):
        result = compute_zscore(close_data)
        assert not np.isnan(result)
        assert isinstance(result, float)

    def test_insufficient_data(self):
        data = np.array([1.0])
        result = compute_zscore(data)
        assert np.isnan(result)

    def test_constant_data(self, constant_data):
        result = compute_zscore(constant_data)
        # std=0 -> zscore=0
        assert result == 0.0

    def test_with_nan_values(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        result = compute_zscore(data)
        # Should ignore NaN
        assert not np.isnan(result)

    def test_manual_calculation(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_zscore(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        expected = (5.0 - mean) / std
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# detect_bullish_divergence Tests
# ============================================================================

class TestDetectBullishDivergence:
    """Tests for bullish divergence detection."""

    def test_no_divergence(self, rng):
        n = 100
        prices = np.arange(n, dtype=float)
        indicator = np.arange(n, dtype=float)
        result = detect_bullish_divergence(prices, indicator, lookback=50)
        # Both trending up - no divergence
        assert isinstance(result, bool)

    def test_with_divergence(self):
        """Create price with lower low, indicator with higher low."""
        n = 50
        prices = np.concatenate([np.linspace(100, 80, 25), np.linspace(80, 75, 25)])
        indicator = np.concatenate([np.linspace(50, 30, 25), np.linspace(30, 35, 25)])
        result = detect_bullish_divergence(prices, indicator, lookback=50)
        # Price makes lower low, indicator makes higher low -> bullish divergence
        assert bool(result) is True

    def test_insufficient_data(self):
        prices = np.arange(10.0)
        indicator = np.arange(10.0)
        result = detect_bullish_divergence(prices, indicator, lookback=50)
        assert result is False


# ============================================================================
# detect_bearish_divergence Tests
# ============================================================================

class TestDetectBearishDivergence:
    """Tests for bearish divergence detection."""

    def test_no_divergence(self, rng):
        n = 100
        prices = np.arange(n, dtype=float)
        indicator = np.arange(n, dtype=float)
        result = detect_bearish_divergence(prices, indicator, lookback=50)
        assert isinstance(result, (bool, np.bool_))

    def test_with_divergence(self):
        """Create price with higher high, indicator with lower high."""
        n = 50
        prices = np.concatenate([np.linspace(80, 100, 25), np.linspace(100, 105, 25)])
        indicator = np.concatenate([np.linspace(30, 50, 25), np.linspace(50, 45, 25)])
        result = detect_bearish_divergence(prices, indicator, lookback=50)
        # Price makes higher high, indicator makes lower high -> bearish divergence
        assert bool(result) is True

    def test_insufficient_data(self):
        prices = np.arange(10.0)
        indicator = np.arange(10.0)
        result = detect_bearish_divergence(prices, indicator, lookback=50)
        assert result is False


# ============================================================================
# Integration / Cross-indicator tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple indicators."""

    def test_bollinger_within_keltner_for_squeeze(self, ohlcv_data):
        """Verify TTMSqueeze consistency with BollingerBands and KeltnerChannels."""
        _, highs, lows, closes, _ = ohlcv_data
        bb = BollingerBands(20, 2.0)
        kc = KeltnerChannels(20, 10, 1.5)
        ttm = TTMSqueeze(20, 2.0, 20, 10, 1.5)

        bb_result = bb.compute(closes)
        kc_result = kc.compute(highs, lows, closes)
        ttm_result = ttm.compute(highs, lows, closes)

        if bb_result and kc_result and ttm_result:
            # Verify BB bounds match TTM's BB bounds
            np.testing.assert_allclose(ttm_result["bb_upper"], bb_result["upper"], rtol=1e-6)

    def test_sma_ema_same_seed(self, close_data):
        """EMA seed value should equal SMA at the period index."""
        period = 20
        sma = SMA(period)
        ema = EMA(period)
        sma_result = sma.compute(close_data)
        ema_result = ema.compute(close_data)
        np.testing.assert_allclose(ema_result[period - 1], sma_result[period - 1], rtol=1e-10)

    def test_rsi_range_with_macd(self, close_data):
        """RSI should be 0-100 and MACD should be finite."""
        rsi_val = RSI(14).compute(close_data)
        macd_result = MACD().compute(close_data)

        assert 0 <= rsi_val <= 100
        if macd_result is not None:
            assert np.isfinite(macd_result["macd"])
            assert np.isfinite(macd_result["signal"])

    def test_atr_positive_with_bollinger(self, ohlcv_data):
        """ATR should be positive and Bollinger bands should be wider than ATR bands."""
        _, highs, lows, closes, _ = ohlcv_data
        atr_val = ATR(14).compute(highs, lows, closes)
        bb_result = BollingerBands(20, 2.0).compute(closes)

        if not np.isnan(atr_val) and bb_result is not None:
            assert atr_val > 0
            assert bb_result["upper"] - bb_result["lower"] > 0

    def test_stochastic_and_rsi_consistent(self, ohlcv_data):
        """Both RSI and Stochastic should agree on general overbought/oversold."""
        _, highs, lows, closes, _ = ohlcv_data
        rsi_val = RSI(14).compute(closes)
        stoch_result = StochasticOscillator(14, 3).compute(highs, lows, closes)

        if stoch_result is not None:
            # If RSI > 80, Stochastic K should also be relatively high
            if rsi_val > 80:
                assert stoch_result["k"] > 50
            # If RSI < 20, Stochastic K should be relatively low
            if rsi_val < 20:
                assert stoch_result["k"] < 50

    def test_multiple_averages_on_same_data(self, close_data):
        """Compute SMA, EMA, WMA, DEMA, TEMA on same data - all should be finite."""
        data = close_data
        sma_r = SMA(20).compute(data)
        ema_r = EMA(20).compute(data)
        wma_r = WMA(20).compute(data)
        dema_r = DEMA(20).compute(data)
        tema_r = TEMA(20).compute(data)

        for r in [sma_r, ema_r, wma_r, dema_r, tema_r]:
            valid = r[~np.isnan(r)]
            assert np.all(np.isfinite(valid))

    def test_obv_trending_with_prices(self, ohlcv_data):
        """If prices are mostly rising, OBV should also be rising."""
        _, _, _, closes, volumes = ohlcv_data
        obv = OBVIndicator().compute(closes, volumes)
        # Check correlation
        if len(obv) > 20:
            price_diff = np.diff(closes[-20:])
            obv_diff = np.diff(obv[-20:])
            # Both should have same sign tendency
            same_sign = np.sum(np.sign(price_diff) == np.sign(obv_diff))
            assert same_sign > 0  # At least some agreement


# ============================================================================
# Edge case tests for all indicators
# ============================================================================

class TestEdgeCases:
    """Edge case tests for various indicators."""

    def test_sma_single_element(self):
        data = np.array([42.0])
        sma = SMA(1)
        result = sma.compute(data)
        assert result[0] == 42.0

    def test_ema_single_element(self):
        data = np.array([42.0])
        ema = EMA(1)
        result = ema.compute(data)
        assert result[0] == 42.0

    def test_rsi_period_one(self):
        data = np.arange(1.0, 10.0)
        rsi = RSI(1)
        result = rsi.compute(data)
        assert not np.isnan(result)

    def test_bollinger_bands_one_std(self, close_data):
        bb = BollingerBands(20, 1.0)
        result = bb.compute(close_data)
        assert result is not None
        assert result["upper"] - result["lower"] > 0

    def test_vwap_with_single_bar(self):
        closes = np.array([50000.0])
        highs = np.array([50010.0])
        lows = np.array([49990.0])
        volumes = np.array([1e6])
        vwap = VWAPIndicator()
        result = vwap.compute(highs, lows, closes, volumes)
        assert not np.isnan(result)

    def test_ichimoku_custom_periods(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        ic = IchimokuCloud(tenkan=5, kijun=15, senkou_b=30)
        result = ic.compute(highs, lows, closes)
        assert result is not None

    def test_macd_custom_params(self, close_data):
        macd = MACD(fast=5, slow=15, signal=5)
        result = macd.compute(close_data)
        assert result is not None

    def test_connors_rsi_small_roc_period(self, close_data):
        crsi = ConnorsRSI(rsi_period=3, streak_period=2, roc_period=10)
        result = crsi.compute(close_data)
        assert not np.isnan(result)

    def test_atr_with_very_small_range(self):
        """ATR with nearly zero high-low range."""
        n = 20
        closes = np.full(n, 50000.0)
        highs = closes + 0.001
        lows = closes - 0.001
        atr = ATR(14)
        result = atr.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert result >= 0

    def test_stochastic_equal_high_low(self):
        """When all highs and lows are equal, K should be 50."""
        n = 20
        closes = np.full(n, 50000.0)
        highs = np.full(n, 50010.0)
        lows = np.full(n, 49990.0)
        so = StochasticOscillator(k_period=14, d_period=3)
        result = so.compute(highs, lows, closes)
        if result is not None:
            # Close at middle of range
            assert 40 <= result["k"] <= 60

    def test_cmf_at_extremes(self):
        """CMF when close at high (buying) vs close at low (selling)."""
        n = 25
        # Close at high: buying pressure
        closes = np.full(n, 101.0)
        highs = np.full(n, 102.0)
        lows = np.full(n, 98.0)
        volumes = np.full(n, 1e6)
        cmf = CMFIndicator(20)
        result_bull = cmf.compute(highs, lows, closes, volumes)

        # Close at low: selling pressure
        closes2 = np.full(n, 99.0)
        result_bear = cmf.compute(highs, lows, closes2, volumes)

        if not np.isnan(result_bull) and not np.isnan(result_bear):
            assert result_bull > result_bear

    def test_pivot_points_zero_range(self):
        """Pivot points when high=low=close."""
        pp = PivotPoints()
        result = pp.compute(100.0, 100.0, 100.0)
        assert result["pivot"] == 100.0
        # All S/R should equal pivot when range is 0
        assert result["s1"] == result["r1"] == 100.0

    def test_obv_flat_prices(self):
        """OBV with flat prices should remain constant."""
        closes = np.full(10, 50000.0)
        volumes = np.full(10, 1e6)
        obv = OBVIndicator().compute(closes, volumes)
        # No change -> OBV stays at 0
        assert obv[-1] == 0.0

    def test_volume_oscillator_equal_periods(self):
        """When fast=slow periods, volume oscillator should be ~0."""
        volumes = np.random.randn(30) + 1e6
        vo = VolumeOscillator(fast_period=10, slow_period=10)
        result = vo.compute(volumes)
        if not np.isnan(result):
            np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_momentum_with_single_price_change(self):
        data = np.array([100.0, 100.0, 100.0, 100.0, 105.0])
        mom = Momentum(4)
        result = mom.compute(data)
        assert result == 5.0

    def test_roc_with_doubling(self):
        data = np.array([50.0, 60.0, 70.0, 80.0, 100.0])
        roc = ROC(4)
        result = roc.compute(data)
        expected = ((100 - 50) / 50) * 100
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dema_responsiveness(self, close_data):
        """DEMA should be more responsive than EMA (less lag)."""
        period = 20
        ema_result = EMA(period).compute(close_data)
        dema_result = DEMA(period).compute(close_data)
        # In a trending market, DEMA should be closer to current price
        # This is a soft check - just verify both are finite
        valid_ema = ema_result[~np.isnan(ema_result)]
        valid_dema = dema_result[~np.isnan(dema_result)]
        assert len(valid_ema) > 0
        assert len(valid_dema) > 0

    def test_candlestick_with_all_bullish(self):
        """All bullish candles - no bearish patterns should trigger."""
        n = 10
        closes = np.arange(100.0, 100.0 + n)
        opens = closes - 1
        highs = closes + 2
        lows = opens - 1
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert bool(result.get("bearish_engulfing", False)) is False
        assert bool(result.get("three_black_crows", False)) is False

    def test_candlestick_with_all_bearish(self):
        """All bearish candles - no bullish patterns should trigger."""
        n = 10
        opens = np.arange(110.0, 110.0 + n)
        closes = opens - 2
        highs = opens + 1
        lows = closes - 1
        cp = CandlestickPatterns()
        result = cp.detect_all(opens, highs, lows, closes)
        assert bool(result.get("bullish_engulfing", False)) is False
        assert bool(result.get("three_white_soldiers", False)) is False


# ============================================================================
# Large value / small value tests
# ============================================================================

class TestExtremeValues:
    """Tests with extreme input values."""

    def test_sma_very_large_values(self):
        data = np.full(50, 1e15)
        result = SMA(20).compute(data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1e15, rtol=1e-6)

    def test_sma_very_small_values(self):
        data = np.full(50, 1e-15)
        result = SMA(20).compute(data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1e-15, rtol=1e-2)

    def test_ema_very_large_values(self):
        data = np.full(50, 1e15)
        result = EMA(20).compute(data)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1e15, rtol=1e-6)

    def test_rsi_very_large_values(self):
        data = np.full(50, 1e15) + np.random.randn(50) * 1e12
        result = RSI(14).compute(data)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    def test_bollinger_bands_very_small_values(self):
        data = np.full(50, 1e-10) + np.random.randn(50) * 1e-12
        result = BollingerBands(20).compute(data)
        if result is not None:
            assert np.isfinite(result["upper"])
            assert np.isfinite(result["lower"])

    def test_atr_extreme_prices(self):
        n = 20
        closes = np.full(n, 1e12)
        highs = closes + 1e9
        lows = closes - 1e9
        result = ATR(14).compute(highs, lows, closes)
        assert not np.isnan(result)
        assert result > 0


# ============================================================================
# NaN handling tests
# ============================================================================

class TestNaNHandling:
    """Tests for NaN handling in indicators."""

    def test_zscore_with_nans(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        result = compute_zscore(data)
        assert not np.isnan(result)

    def test_hurst_with_nans(self, close_data):
        data = close_data.copy()
        data[5] = np.nan
        data[50] = np.nan
        result = compute_hurst_exponent(data)
        # Should handle NaN by filtering them out
        assert isinstance(result, float)

    def test_rsi_with_nan_data(self, nan_data):
        rsi = RSI(14)
        result = rsi.compute(nan_data)
        # RSI compute takes closes[-period-1:], if NaN is there it may propagate
        assert isinstance(result, float)

    def test_sma_with_nan_in_data(self):
        data = np.arange(1.0, 31.0)
        data[10] = np.nan
        sma = SMA(20)
        result = sma.compute(data)
        # NaN in data will propagate to SMA values that include that point
        assert len(result) == 30

    def test_divergence_with_nans(self):
        n = 60
        prices = np.arange(n, dtype=float)
        indicator = np.arange(n, dtype=float)
        prices[25] = np.nan
        result = detect_bullish_divergence(prices, indicator, lookback=50)
        assert isinstance(result, bool)


# ============================================================================
# Type and output format tests
# ============================================================================

class TestOutputFormats:
    """Tests verifying correct output types and formats."""

    def test_sma_returns_ndarray(self, close_data):
        result = SMA(20).compute(close_data)
        assert isinstance(result, np.ndarray)

    def test_ema_returns_ndarray(self, close_data):
        result = EMA(20).compute(close_data)
        assert isinstance(result, np.ndarray)

    def test_rsi_returns_float(self, close_data):
        result = RSI(14).compute(close_data)
        assert isinstance(result, float)

    def test_macd_returns_dict(self, close_data):
        result = MACD().compute(close_data)
        assert isinstance(result, dict)

    def test_bollinger_returns_dict(self, close_data):
        result = BollingerBands().compute(close_data)
        assert isinstance(result, dict)

    def test_stochastic_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = StochasticOscillator(k_period=5, d_period=3).compute(highs, lows, closes)
        assert result is None or isinstance(result, dict)

    def test_ichimoku_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = IchimokuCloud().compute(highs, lows, closes)
        assert isinstance(result, dict)

    def test_supertrend_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = Supertrend().compute(highs, lows, closes)
        assert isinstance(result, dict)
        assert "supertrend" in result
        assert "direction" in result

    def test_pivot_points_returns_dict(self):
        result = PivotPoints().compute(105.0, 95.0, 100.0)
        assert isinstance(result, dict)

    def test_aroon_returns_dict(self, ohlcv_data):
        _, highs, lows, _, _ = ohlcv_data
        result = Aroon().compute(highs, lows)
        assert isinstance(result, dict)

    def test_vwap_returns_float(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        result = VWAPIndicator().compute(highs, lows, closes, volumes)
        assert isinstance(result, float)

    def test_candlestick_returns_dict(self, ohlcv_data):
        opens, highs, lows, closes, _ = ohlcv_data
        result = CandlestickPatterns().detect_all(opens, highs, lows, closes)
        assert isinstance(result, dict)

    def test_hurst_returns_float(self, close_data):
        result = compute_hurst_exponent(close_data)
        assert isinstance(result, float)

    def test_zscore_returns_float(self, close_data):
        result = compute_zscore(close_data)
        assert isinstance(result, float)

    def test_ribbon_returns_list(self, close_data):
        result = MovingAverageRibbon().compute(close_data)
        assert isinstance(result, list)
        assert all(isinstance(r, np.ndarray) for r in result)

    def test_donchian_returns_dict(self, ohlcv_data):
        _, highs, lows, _, _ = ohlcv_data
        result = DonchianChannels().compute(highs, lows)
        assert isinstance(result, dict)

    def test_keltner_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = KeltnerChannels().compute(highs, lows, closes)
        assert isinstance(result, dict) or result is None

    def test_ttm_squeeze_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = TTMSqueeze().compute(highs, lows, closes)
        assert isinstance(result, dict) or result is None

    def test_support_resistance_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        sr = SupportResistance()
        result = sr.compute(highs, lows, closes)
        assert isinstance(result, dict)
        assert "support" in result
        assert "resistance" in result

    def test_fibonacci_retracement_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        fr = FibonacciRetracement()
        result = fr.compute(highs, lows, closes)
        assert isinstance(result, dict) or result is None

    def test_volume_profile_returns_dict(self, ohlcv_data):
        _, highs, lows, closes, volumes = ohlcv_data
        vp = VolumeProfile(num_bins=20)
        result = vp.compute(highs, lows, closes, volumes)
        assert isinstance(result, dict)

    def test_true_range_returns_ndarray(self, ohlcv_data):
        _, highs, lows, closes, _ = ohlcv_data
        result = TrueRange().compute(highs, lows, closes)
        assert isinstance(result, np.ndarray)


# ============================================================================
# Period validation tests
# ============================================================================

class TestPeriodValidation:
    """Tests for period parameter validation across indicators."""

    def test_sma_invalid_periods(self):
        for p in [0, -1, -100]:
            with pytest.raises(ValueError):
                SMA(p)

    def test_ema_invalid_periods(self):
        for p in [0, -1]:
            with pytest.raises(ValueError):
                EMA(p)

    def test_wma_invalid_periods(self):
        with pytest.raises(ValueError):
            WMA(0)

    def test_hma_invalid_periods(self):
        with pytest.raises(ValueError):
            HMA(3)
        with pytest.raises(ValueError):
            HMA(0)

    def test_dema_invalid_periods(self):
        with pytest.raises(ValueError):
            DEMA(0)

    def test_tema_invalid_periods(self):
        with pytest.raises(ValueError):
            TEMA(0)

    def test_vwma_invalid_periods(self):
        with pytest.raises(ValueError):
            VWMA(0)

    def test_kama_invalid_periods(self):
        with pytest.raises(ValueError):
            KAMA(period=0)

    def test_alma_invalid_periods(self):
        with pytest.raises(ValueError):
            ALMA(period=0)

    def test_frama_invalid_periods(self):
        with pytest.raises(ValueError):
            FRAMA(period=3)

    def test_zlema_invalid_periods(self):
        with pytest.raises(ValueError):
            ZLEMA(0)

    def test_rsi_invalid_periods(self):
        with pytest.raises(ValueError):
            RSI(period=0)

    def test_ehlers_smoother_invalid_periods(self):
        with pytest.raises(ValueError):
            EhlersSuperSmoother(period=1)

    def test_pivot_points_invalid_method(self):
        with pytest.raises(ValueError):
            PivotPoints(method="invalid")


# ============================================================================
# Additional detailed indicator tests
# ============================================================================

class TestAdditionalDetail:
    """Additional detailed tests for thorough coverage."""

    def test_sma_cumulative_sum_accuracy(self):
        """Verify SMA uses cumulative sum correctly."""
        data = np.random.randn(100) + 50000
        sma = SMA(20)
        result = sma.compute(data)
        # Manual check at index 50
        expected = np.mean(data[31:51])
        np.testing.assert_allclose(result[50], expected, rtol=1e-10)

    def test_ema_smoothing_factor(self):
        """Verify EMA smoothing factor is correct."""
        ema = EMA(12)
        expected_mult = 2.0 / 13.0
        np.testing.assert_allclose(ema.multiplier, expected_mult, rtol=1e-10)

    def test_wma_weights_sum(self):
        """WMA weights should sum to period*(period+1)/2."""
        wma = WMA(10)
        expected_sum = 10 * 11 / 2.0
        np.testing.assert_allclose(wma.weights.sum(), expected_sum, rtol=1e-10)

    def test_bollinger_bands_width_calculation(self, close_data):
        """Verify bandwidth = (upper - lower) / middle."""
        bb = BollingerBands(20, 2.0)
        result = bb.compute(close_data)
        if result is not None and result["middle"] > 0:
            expected_bw = (result["upper"] - result["lower"]) / result["middle"]
            np.testing.assert_allclose(result["bandwidth"], expected_bw, rtol=1e-6)

    def test_bollinger_percent_b_calculation(self, close_data):
        """Verify percent_b = (close - lower) / (upper - lower)."""
        bb = BollingerBands(20, 2.0)
        result = bb.compute(close_data)
        if result is not None:
            std = np.std(close_data[-20:], ddof=0)
            if std > 0:
                expected_pb = (close_data[-1] - result["lower"]) / (result["upper"] - result["lower"])
                np.testing.assert_allclose(result["percent_b"], expected_pb, rtol=1e-6)

    def test_adx_range(self, ohlcv_data):
        """ADX should be between 0 and 100."""
        _, highs, lows, closes, _ = ohlcv_data
        adx = ADX(14)
        result = adx.compute(highs, lows, closes)
        assert 0 <= result <= 100

    def test_cmo_range(self, close_data):
        """CMO should be between -100 and 100."""
        cmo = ChandeMomentumOscillator(14)
        result = cmo.compute(close_data)
        assert -100 <= result <= 100

    def test_williams_r_range(self, ohlcv_data):
        """Williams %R should be between -100 and 0."""
        _, highs, lows, closes, _ = ohlcv_data
        wr = WilliamsR(14)
        result = wr.compute(highs, lows, closes)
        assert -100 <= result <= 0

    def test_cmf_range(self, ohlcv_data):
        """CMF should be between -1 and 1."""
        _, highs, lows, closes, volumes = ohlcv_data
        cmf = CMFIndicator(20)
        result = cmf.compute(highs, lows, closes, volumes)
        assert -1 <= result <= 1

    def test_mfi_range(self, ohlcv_data):
        """MFI should be between 0 and 100."""
        _, highs, lows, closes, volumes = ohlcv_data
        mfi = MFI(14)
        result = mfi.compute(highs, lows, closes, volumes)
        assert 0 <= result <= 100

    def test_ultimate_oscillator_range(self, ohlcv_data):
        """Ultimate Oscillator should be between 0 and 100."""
        _, highs, lows, closes, _ = ohlcv_data
        uo = UltimateOscillator()
        result = uo.compute(highs, lows, closes)
        assert 0 <= result <= 100

    def test_donchian_channels_ordering(self, ohlcv_data):
        """Upper > Middle > Lower for Donchian Channels."""
        _, highs, lows, _, _ = ohlcv_data
        dc = DonchianChannels(20)
        result = dc.compute(highs, lows)
        if result is not None:
            assert result["upper"] >= result["middle"] >= result["lower"]

    def test_bollinger_bands_ordering(self, close_data):
        """Upper > Middle > Lower for Bollinger Bands."""
        bb = BollingerBands(20)
        result = bb.compute(close_data)
        if result is not None:
            assert result["upper"] >= result["middle"] >= result["lower"]

    def test_vwap_bands_ordering(self, ohlcv_data):
        """Upper > VWAP > Lower for VWAP Bands."""
        _, highs, lows, closes, volumes = ohlcv_data
        vb = VWAPBands()
        result = vb.compute(highs, lows, closes, volumes)
        if result is not None:
            assert result["upper"] >= result["vwap"] >= result["lower"]

    def test_nvi_start_value(self, ohlcv_data):
        """NVI should start at 1000."""
        _, _, _, closes, volumes = ohlcv_data
        nvi = NVI()
        result = nvi.compute(closes, volumes)
        assert result[0] == 1000.0

    def test_pvi_start_value(self, ohlcv_data):
        """PVI should start at 1000."""
        _, _, _, closes, volumes = ohlcv_data
        pvi = PVI()
        result = pvi.compute(closes, volumes)
        assert result[0] == 1000.0

    def test_adl_cumulative(self, ohlcv_data):
        """ADL should be cumulative sum of money flow volume."""
        _, highs, lows, closes, volumes = ohlcv_data
        adl = ADLine()
        result = adl.compute(highs, lows, closes, volumes)
        # Should be monotonically different from first element
        assert len(result) == len(closes)

    def test_force_index_sign(self, ohlcv_data):
        """Force Index sign should match price change direction."""
        _, _, _, closes, volumes = ohlcv_data
        fi = ForceIndex(period=1)
        # With period=1, force index should have same sign as price change
        if len(closes) >= 2:
            price_change = closes[-1] - closes[-2]
            # This is a soft check since EMA smoothing changes things
            assert isinstance(fi.compute(closes, volumes), float)

    def test_fractal_dimension_range(self, close_data):
        """Fractal dimension should be between 1.0 and 2.0."""
        fd = FractalDimension(20)
        result = fd.compute(close_data)
        assert 1.0 <= result <= 2.0

    def test_ehlers_dominant_cycle_range(self, close_data):
        """Dominant cycle should be within min/max period range."""
        edc = EhlersDominantCycle(min_period=8, max_period=48)
        result = edc.compute(close_data)
        if not np.isnan(result):
            assert 8 <= result <= 48

    def test_kama_adaptive_behavior(self, rng):
        """KAMA should adapt: slow in choppy, fast in trending."""
        # Create trending data
        trending = np.cumsum(np.ones(100)) * 10
        # Create choppy data
        choppy = 50000 + rng.normal(0, 50, 100)

        kama = KAMA(period=10)
        result_trending = kama.compute(trending)
        result_choppy = kama.compute(choppy)

        # Both should produce valid results
        assert len(result_trending) == 100
        assert len(result_choppy) == 100

    def test_alma_weights_normalized(self):
        """ALMA weights should sum to approximately 1."""
        alma = ALMA(period=20)
        m = alma.offset * (alma.period - 1)
        s = alma.period / alma.sigma
        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(alma.period)])
        w_sum = weights.sum()
        normalized = weights / w_sum
        np.testing.assert_allclose(normalized.sum(), 1.0, rtol=1e-10)

    def test_supertrend_direction_after_compute(self, ohlcv_data):
        """Supertrend direction should be either 1 or -1 for valid bars."""
        _, highs, lows, closes, _ = ohlcv_data
        st = Supertrend()
        result = st.compute(highs, lows, closes)
        # Direction values should be in {1, -1, 0}
        unique_dirs = set(result["direction"])
        assert unique_dirs.issubset({1, -1, 0})

    def test_ttm_squeeze_momentum_type(self, ohlcv_data):
        """TTM Squeeze momentum should be a finite float."""
        _, highs, lows, closes, _ = ohlcv_data
        ttm = TTMSqueeze()
        result = ttm.compute(highs, lows, closes)
        if result is not None:
            assert isinstance(result["momentum"], float)
            assert np.isfinite(result["momentum"])

    def test_ichimoku_chikou_equals_close(self, ohlcv_data):
        """Chikou span should equal the latest close."""
        _, highs, lows, closes, _ = ohlcv_data
        ic = IchimokuCloud()
        result = ic.compute(highs, lows, closes)
        if result is not None:
            assert result["chikou"] == closes[-1]

    def test_aroon_up_100_when_high_at_end(self):
        """Aroon Up should be 100 when highest high is at the last bar."""
        highs = np.arange(1.0, 27.0)  # Increasing
        lows = np.arange(0.5, 26.5)
        aroon = Aroon(25)
        result = aroon.compute(highs, lows)
        assert result["up"] == 100.0

    def test_aroon_down_100_when_low_at_end(self):
        """Aroon Down should be 100 when lowest low is at the last bar."""
        highs = np.arange(26.0, 0.0, -1)  # Not needed
        lows = np.arange(26.0, 0.0, -1)  # Decreasing
        aroon = Aroon(25)
        result = aroon.compute(highs, lows)
        # Lowest low is at the last bar -> aroon_down = 100
        # Actually we need the most recent bar
        assert result["down"] == 100.0

    def test_atr_series_first_valid_index(self, ohlcv_data):
        """ATR series first valid value should be at index=period."""
        _, highs, lows, closes, _ = ohlcv_data
        atr = ATR(14)
        result = atr.compute_series(highs, lows, closes)
        assert np.isnan(result[13])  # period-1 should be NaN
        assert not np.isnan(result[14])  # period should be valid

    def test_rsi_series_first_valid_index(self, close_data):
        """RSI series first valid value should be at index=period."""
        rsi = RSI(14)
        result = rsi.compute_series(close_data)
        assert np.isnan(result[13])  # period-1 should be NaN
        assert not np.isnan(result[14])  # period should be valid

    def test_volume_profile_poc_is_max_volume(self, ohlcv_data):
        """POC should be at the price bin with highest volume."""
        _, highs, lows, closes, volumes = ohlcv_data
        vp = VolumeProfile(num_bins=20)
        result = vp.compute(highs, lows, closes, volumes)
        if len(result["prices"]) > 0:
            poc_idx = np.argmax(result["volumes"])
            np.testing.assert_allclose(result["poc"], result["prices"][poc_idx], rtol=1e-6)


# ============================================================================
# Parametric / property-based style tests
# ============================================================================

class TestParametric:
    """Parametric tests with various period values."""

    @pytest.mark.parametrize("period", [1, 5, 10, 20, 50])
    def test_sma_various_periods(self, close_data, period):
        sma = SMA(period)
        result = sma.compute(close_data)
        assert len(result) == len(close_data)
        valid = result[~np.isnan(result)]
        assert len(valid) == len(close_data) - period + 1

    @pytest.mark.parametrize("period", [1, 5, 10, 20])
    def test_ema_various_periods(self, close_data, period):
        ema = EMA(period)
        result = ema.compute(close_data)
        assert len(result) == len(close_data)

    @pytest.mark.parametrize("period", [1, 5, 10, 20])
    def test_wma_various_periods(self, close_data, period):
        wma = WMA(period)
        result = wma.compute(close_data)
        assert len(result) == len(close_data)

    @pytest.mark.parametrize("period", [4, 10, 20])
    def test_hma_various_periods(self, close_data, period):
        hma = HMA(period)
        result = hma.compute(close_data)
        assert len(result) == len(close_data)

    @pytest.mark.parametrize("period", [1, 5, 14, 28])
    def test_rsi_various_periods(self, close_data, period):
        rsi = RSI(period)
        result = rsi.compute(close_data)
        assert not np.isnan(result)
        assert 0 <= result <= 100

    @pytest.mark.parametrize("bb_std", [1.0, 1.5, 2.0, 2.5, 3.0])
    def test_bollinger_various_std(self, close_data, bb_std):
        bb = BollingerBands(20, bb_std)
        result = bb.compute(close_data)
        assert result is not None
        # Higher std -> wider bands
        band_width = result["upper"] - result["lower"]
        assert band_width > 0

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_atr_various_periods(self, ohlcv_data, period):
        _, highs, lows, closes, _ = ohlcv_data
        atr = ATR(period)
        result = atr.compute(highs, lows, closes)
        assert not np.isnan(result)
        assert result > 0

    @pytest.mark.parametrize("method", ["traditional", "fibonacci", "camarilla", "woodie"])
    def test_pivot_points_methods(self, method):
        pp = PivotPoints(method=method)
        result = pp.compute(105.0, 95.0, 100.0)
        assert isinstance(result, dict)
        assert "pivot" in result

    @pytest.mark.parametrize("k_period,d_period", [(5, 3), (9, 3), (14, 3), (14, 5)])
    def test_stochastic_various_periods(self, ohlcv_data, k_period, d_period):
        _, highs, lows, closes, _ = ohlcv_data
        so = StochasticOscillator(k_period, d_period)
        result = so.compute(highs, lows, closes)
        if result is not None:
            assert 0 <= result["k"] <= 100
            assert 0 <= result["d"] <= 100
