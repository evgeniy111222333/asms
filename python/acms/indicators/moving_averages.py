"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict


class SMA:
    """Simple Moving Average.

    Computes the arithmetic mean of the last `period` data points.
    Uses cumulative sum for O(1) per-element computation after initialization.
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("SMA period must be >= 1")
        self.period = period

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute SMA over the data array.

        Args:
            data: 1-D array of numeric values.

        Returns:
            Array of same length; values before `period` are NaN.
        """
        if len(data) < self.period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        cumsum = np.cumsum(data, dtype=float)
        result[self.period - 1:] = (
            cumsum[self.period - 1:]
            - np.concatenate([[0.0], cumsum[:-self.period]])
        ) / self.period
        return result


class EMA:
    """Exponential Moving Average.

    Uses the standard multiplier 2/(period+1).  The seed value is the
    SMA of the first `period` elements.
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self.period = period
        self.multiplier = 2.0 / (period + 1)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute EMA over the data array."""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < self.period:
            return result
        result[self.period - 1] = np.mean(data[:self.period])
        for i in range(self.period, len(data)):
            result[i] = data[i] * self.multiplier + result[i - 1] * (1 - self.multiplier)
        return result


class WMA:
    """Weighted Moving Average.

    Weights increase linearly from 1 (oldest) to `period` (newest).
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("WMA period must be >= 1")
        self.period = period
        self.weights = np.arange(1, period + 1, dtype=float)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute WMA over the data array."""
        result = np.full_like(data, np.nan, dtype=float)
        weight_sum = self.weights.sum()
        if weight_sum == 0:
            return result
        for i in range(self.period - 1, len(data)):
            result[i] = np.dot(data[i - self.period + 1:i + 1], self.weights) / weight_sum
        return result


class HMA:
    """Hull Moving Average.

    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness.
    """

    def __init__(self, period: int):
        if period < 4:
            raise ValueError("HMA period must be >= 4")
        self.period = period
        self.wma_half = WMA(max(period // 2, 1))
        self.wma_full = WMA(period)
        self.wma_sqrt = WMA(max(int(np.sqrt(period)), 1))

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute HMA over the data array."""
        wma_half = self.wma_half.compute(data)
        wma_full = self.wma_full.compute(data)
        diff = 2 * wma_half - wma_full
        valid = ~np.isnan(diff)
        if not np.any(valid):
            return np.full_like(data, np.nan)
        return self.wma_sqrt.compute(diff)


class DEMA:
    """Double Exponential Moving Average.

    DEMA = 2*EMA - EMA(EMA)
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("DEMA period must be >= 1")
        self.period = period
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute DEMA over the data array."""
        ema1 = self.ema1.compute(data)
        valid_mask = ~np.isnan(ema1)
        ema2 = np.full_like(data, np.nan, dtype=float)
        if np.sum(valid_mask) >= self.period:
            ema2[valid_mask] = self.ema2.compute(ema1[valid_mask])
        return 2 * ema1 - ema2


class TEMA:
    """Triple Exponential Moving Average.

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("TEMA period must be >= 1")
        self.period = period
        self.ema1 = EMA(period)
        self.ema2 = EMA(period)
        self.ema3 = EMA(period)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute TEMA over the data array."""
        e1 = self.ema1.compute(data)
        valid1 = ~np.isnan(e1)
        e2 = np.full_like(data, np.nan, dtype=float)
        e3 = np.full_like(data, np.nan, dtype=float)
        if np.sum(valid1) >= self.period:
            e2[valid1] = self.ema2.compute(e1[valid1])
        valid2 = ~np.isnan(e2)
        if np.sum(valid2) >= self.period:
            e3[valid2] = self.ema3.compute(e2[valid2])
        return 3 * e1 - 3 * e2 + e3


class VWMA:
    """Volume-Weighted Moving Average.

    VWMA[i] = Sum(close*volume, n) / Sum(volume, n)
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("VWMA period must be >= 1")
        self.period = period

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute VWMA over the data arrays."""
        if len(closes) != len(volumes):
            raise ValueError("closes and volumes must have same length")
        result = np.full_like(closes, np.nan, dtype=float)
        for i in range(self.period - 1, len(closes)):
            vol_slice = volumes[i - self.period + 1:i + 1]
            close_slice = closes[i - self.period + 1:i + 1]
            total_vol = vol_slice.sum()
            if total_vol > 0:
                result[i] = np.dot(close_slice, vol_slice) / total_vol
        return result


class KAMA:
    """Kaufman's Adaptive Moving Average.

    Adjusts its speed based on market noise (efficiency ratio).
    Fast in trending markets, slow in choppy/range-bound markets.
    """

    def __init__(self, period: int = 10, fast_sc: float = 2.0 / 3.0, slow_sc: float = 2.0 / 31.0):
        if period < 1:
            raise ValueError("KAMA period must be >= 1")
        self.period = period
        self.fast_sc = fast_sc
        self.slow_sc = slow_sc

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute KAMA over the data array."""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < self.period + 1:
            return result
        result[self.period] = data[self.period]
        for i in range(self.period + 1, len(data)):
            direction = abs(data[i] - data[i - self.period])
            volatility = np.sum(np.abs(np.diff(data[i - self.period:i + 1])))
            if volatility == 0:
                er = 0.0
            else:
                er = direction / volatility
            sc = (er * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
            result[i] = result[i - 1] + sc * (data[i] - result[i - 1])
        return result


class ALMA:
    """Arnaud Legoux Moving Average.

    Uses a Gaussian distribution as the weighting function,
    providing superior smoothness and reduced lag.
    """

    def __init__(self, period: int = 9, offset: float = 0.85, sigma: float = 6.0):
        if period < 1:
            raise ValueError("ALMA period must be >= 1")
        self.period = period
        self.offset = offset
        self.sigma = sigma

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute ALMA over the data array."""
        result = np.full_like(data, np.nan, dtype=float)
        m = self.offset * (self.period - 1)
        s = self.period / self.sigma
        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(self.period)])
        w_sum = weights.sum()
        if w_sum == 0:
            return result
        weights /= w_sum
        for i in range(self.period - 1, len(data)):
            result[i] = np.dot(data[i - self.period + 1:i + 1], weights)
        return result


class FRAMA:
    """Fractal Adaptive Moving Average.

    Uses fractal dimension to adapt the smoothing constant.
    Lower dimension (smoother price) -> slower moving average.
    Higher dimension (choppier price) -> faster moving average.
    """

    def __init__(self, period: int = 20):
        if period < 4:
            raise ValueError("FRAMA period must be >= 4")
        self.period = period

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute FRAMA over the data array."""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < self.period * 2:
            return result
        half = self.period // 2
        result[self.period] = np.mean(data[:self.period])
        for i in range(self.period + 1, len(data)):
            h1 = np.max(data[i - self.period:i - half]) - np.min(data[i - self.period:i - half])
            h2 = np.max(data[i - half:i]) - np.min(data[i - half:i])
            full_range = np.max(data[i - self.period:i]) - np.min(data[i - self.period:i])
            n1 = (self.period / 2) / h1 if h1 > 0 else 1e10
            n2 = (self.period / 2) / h2 if h2 > 0 else 1e10
            n3 = self.period / full_range if full_range > 0 else 1e10
            dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2) if n3 > 0 else 0.5
            alpha = np.exp(-4.6 * (dim - 1))
            alpha = max(min(alpha, 1.0), 0.01)
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result


class ZLEMA:
    """Zero-Lag Exponential Moving Average.

    Removes lag by de-lagging the input data before applying EMA.
    """

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("ZLEMA period must be >= 1")
        self.period = period
        self.lag = (period - 1) // 2

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute ZLEMA over the data array."""
        if len(data) < self.lag + 1:
            return np.full_like(data, np.nan, dtype=float)
        adjusted = data.copy().astype(float)
        adjusted[self.lag:] = data[self.lag:] + (data[self.lag:] - data[:-self.lag]) if self.lag > 0 else data
        return EMA(self.period).compute(adjusted)


class KaufmanAdaptiveMovingAverage(KAMA):
    """Alias for KAMA for discoverability."""
    pass


class EhlersSuperSmoother:
    """Ehlers Super Smoother filter.

    A 2-pole Butterworth filter that provides superior smoothing
    with minimal lag compared to traditional moving averages.
    """

    def __init__(self, period: int = 10):
        if period < 2:
            raise ValueError("Period must be >= 2")
        self.period = period

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute Ehlers Super Smoother."""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < 2:
            return result
        a = np.exp(-1.414 * np.pi / self.period)
        b = 2 * a * np.cos(1.414 * np.pi / self.period)
        c2 = b
        c3 = -a * a
        c1 = 1 - c2 - c3
        result[0] = data[0]
        result[1] = data[1]
        for i in range(2, len(data)):
            result[i] = c1 * (data[i] + data[i - 1]) / 2 + c2 * result[i - 1] + c3 * result[i - 2]
        return result


__all__ = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'VWMA', 'KAMA', 'ALMA', 'FRAMA', 'ZLEMA', 'KaufmanAdaptiveMovingAverage', 'EhlersSuperSmoother']
