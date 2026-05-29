"""Technical Indicators Library for ACMS.

Implements 70+ technical indicators organized by category:
- Moving Averages (14+ types)
- Oscillators (16 types)
- Volatility Indicators (14 types)
- Volume Indicators (14 types)
- Candlestick Pattern Recognition (35+ patterns)
- Advanced Indicators (Connors RSI, TTM Squeeze, VWAP Bands, etc.)
- Statistical Measures (Hurst, z-score, divergence)
- Pattern Recognition (Support/Resistance, Pivot Points, Fibonacci)

All indicators handle edge cases (insufficient data, NaN, division by zero).
"""

import numpy as np
from typing import Optional, Tuple, List, Dict


# ============================================================================
# Moving Averages
# ============================================================================

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


class Supertrend:
    """Supertrend indicator.

    A trend-following overlay that uses ATR to set trailing stops.
    Direction: 1 = uptrend, -1 = downtrend.
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
        """Compute Supertrend indicator.

        Returns:
            Dict with 'supertrend' and 'direction' arrays.
        """
        atr_series = ATR(self.period).compute_series(highs, lows, closes)
        result = {"supertrend": np.full_like(closes, np.nan), "direction": np.full(len(closes), 0)}
        if np.all(np.isnan(atr_series)):
            return result
        hl2 = (highs + lows) / 2
        upper_band = hl2 + self.multiplier * atr_series
        lower_band = hl2 - self.multiplier * atr_series
        st = np.full_like(closes, np.nan)
        direction = np.ones(len(closes), dtype=int)
        for i in range(self.period, len(closes)):
            if np.isnan(atr_series[i]):
                direction[i] = direction[i - 1] if i > self.period else 1
                st[i] = st[i - 1] if i > self.period and not np.isnan(st[i - 1]) else np.nan
                continue
            if closes[i] > upper_band[i - 1]:
                direction[i] = 1
            elif closes[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
            if direction[i] == 1:
                st[i] = lower_band[i] if np.isnan(st[i - 1]) or lower_band[i] > st[i - 1] else st[i - 1]
            else:
                st[i] = upper_band[i] if np.isnan(st[i - 1]) or upper_band[i] < st[i - 1] else st[i - 1]
        result["supertrend"] = st
        result["direction"] = direction
        return result


class MovingAverageRibbon:
    """Moving Average Ribbon (multiple EMAs).

    Generates a series of EMAs at different periods to visualize
    trend strength and direction.
    """

    def __init__(self, base_period: int = 10, count: int = 8, step: int = 5):
        self.emas = [EMA(base_period + i * step) for i in range(count)]

    def compute(self, data: np.ndarray) -> List[np.ndarray]:
        """Compute all ribbon EMAs."""
        return [ema.compute(data) for ema in self.emas]


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


# ============================================================================
# Oscillators
# ============================================================================

class RSI:
    """Relative Strength Index.

    Measures the speed and magnitude of recent price changes.
    Range: 0-100. >70 overbought, <30 oversold.
    """

    def __init__(self, period: int = 14):
        if period < 1:
            raise ValueError("RSI period must be >= 1")
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest RSI value."""
        if len(closes) < self.period + 1:
            return float('nan')
        deltas = np.diff(closes[-self.period - 1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute_series(self, closes: np.ndarray) -> np.ndarray:
        """Compute full RSI series using Wilder's smoothing method."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        rsi = np.full_like(closes, np.nan, dtype=float)
        if len(closes) < self.period + 1:
            return rsi
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        for i in range(self.period, len(closes)):
            if i == self.period:
                if avg_loss == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                avg_gain = (avg_gain * (self.period - 1) + gains[i - 1]) / self.period
                avg_loss = (avg_loss * (self.period - 1) + losses[i - 1]) / self.period
                if avg_loss == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        return rsi


class ConnorsRSI:
    """Connors RSI (CRSI).

    A composite momentum oscillator combining three components:
    1. RSI of price changes
    2. UpDown Streak (percent rank of consecutive up/down days)
    3. Percent rank of current change vs historical

    CRSI = (RSI + UpDown + ROC_Rank) / 3
    """

    def __init__(self, rsi_period: int = 3, streak_period: int = 2, roc_period: int = 100):
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.roc_period = roc_period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest Connors RSI value."""
        if len(closes) < max(self.rsi_period + 1, self.streak_period + 1, self.roc_period + 1):
            return float('nan')
        rsi_val = RSI(self.rsi_period).compute(closes)
        streaks = self._compute_streaks(closes)
        if len(streaks) == 0:
            return float('nan')
        current_streak = streaks[-1]
        rank_count = np.sum(np.abs(streaks[-self.roc_period:]) <= np.abs(current_streak))
        updown_rank = (rank_count / min(len(streaks), self.roc_period)) * 100.0
        changes = np.diff(closes) / closes[:-1] * 100
        if len(changes) < 2:
            return float('nan')
        current_change = changes[-1]
        roc_rank_count = np.sum(changes[-self.roc_period:] <= current_change)
        roc_rank = (roc_rank_count / min(len(changes), self.roc_period)) * 100.0
        if np.isnan(rsi_val):
            return float('nan')
        return (rsi_val + updown_rank + roc_rank) / 3.0

    @staticmethod
    def _compute_streaks(closes: np.ndarray) -> np.ndarray:
        """Compute consecutive up/down streaks."""
        if len(closes) < 2:
            return np.array([])
        changes = np.diff(closes)
        streaks = []
        current = 0
        for ch in changes:
            if ch > 0:
                current = current + 1 if current > 0 else 1
            elif ch < 0:
                current = current - 1 if current < 0 else -1
            else:
                current = 0
            streaks.append(current)
        return np.array(streaks, dtype=float)


class MACD:
    """Moving Average Convergence Divergence.

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(signal_period) of MACD
    Histogram = MACD - Signal
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, closes: np.ndarray) -> Optional[dict]:
        """Compute MACD values."""
        if len(closes) < self.slow + self.signal:
            return None
        ema_fast = EMA(self.fast).compute(closes)
        ema_slow = EMA(self.slow).compute(closes)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        if valid.sum() < self.signal:
            return None
        signal_line = EMA(self.signal).compute(macd_line[valid])
        if np.all(np.isnan(signal_line)):
            return None
        signal_full = np.full_like(closes, np.nan)
        signal_full[valid] = signal_line
        macd_last = macd_line[-1] if not np.isnan(macd_line[-1]) else 0.0
        signal_last = signal_full[-1] if not np.isnan(signal_full[-1]) else 0.0
        hist_last = macd_last - signal_last
        return {
            "macd": macd_last,
            "signal": signal_last,
            "histogram": hist_last,
            "macd_line": macd_line,
            "signal_line": signal_full,
        }


class VolumeWeightedMACD:
    """Volume-Weighted MACD.

    Uses VWMA instead of EMA for both the fast and slow lines,
    providing volume-confirmed trend signals.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> Optional[dict]:
        """Compute Volume-Weighted MACD."""
        if len(closes) < self.slow + self.signal:
            return None
        vwma_fast = VWMA(self.fast).compute(closes, volumes)
        vwma_slow = VWMA(self.slow).compute(closes, volumes)
        macd_line = vwma_fast - vwma_slow
        valid = ~np.isnan(macd_line)
        if valid.sum() < self.signal:
            return None
        macd_valid = macd_line[valid]
        signal_line = SMA(self.signal).compute(macd_valid)
        macd_last = macd_line[-1] if not np.isnan(macd_line[-1]) else 0.0
        signal_last = signal_line[-1] if not np.isnan(signal_line[-1]) else 0.0
        return {
            "macd": macd_last,
            "signal": signal_last,
            "histogram": macd_last - signal_last,
        }


class StochasticOscillator:
    """Stochastic Oscillator (%K and %D).

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    """

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[dict]:
        """Compute Stochastic Oscillator."""
        if len(closes) < self.k_period:
            return None
        k_values = np.full_like(closes, np.nan)
        for i in range(self.k_period - 1, len(closes)):
            hh = np.max(highs[i - self.k_period + 1:i + 1])
            ll = np.min(lows[i - self.k_period + 1:i + 1])
            if hh == ll:
                k_values[i] = 50.0
            else:
                k_values[i] = ((closes[i] - ll) / (hh - ll)) * 100.0
        d_values = SMA(self.d_period).compute(k_values)
        k_val = k_values[-1]
        d_val = d_values[-1]
        if np.isnan(k_val) or np.isnan(d_val):
            return None
        return {"k": k_val, "d": d_val}


class CCI:
    """Commodity Channel Index.

    Measures deviation of typical price from its statistical mean.
    Values > +100: overbought. Values < -100: oversold.
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest CCI value."""
        if len(closes) < self.period:
            return float('nan')
        tp = (highs + lows + closes) / 3.0
        sma_tp = SMA(self.period).compute(tp)
        mad = np.full_like(tp, np.nan)
        for i in range(self.period - 1, len(tp)):
            mad[i] = np.mean(np.abs(tp[i - self.period + 1:i + 1] - sma_tp[i]))
        if mad[-1] == 0:
            return 0.0
        return (tp[-1] - sma_tp[-1]) / (0.015 * mad[-1])


class WilliamsR:
    """Williams %R.

    Momentum indicator measuring overbought/oversold levels.
    Range: -100 to 0. >-20 overbought, <-80 oversold.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest Williams %R value."""
        if len(closes) < self.period:
            return float('nan')
        hh = np.max(highs[-self.period:])
        ll = np.min(lows[-self.period:])
        if hh == ll:
            return -50.0
        return ((hh - closes[-1]) / (hh - ll)) * -100.0


class ROC:
    """Rate of Change.

    Measures percentage change between current price and price `period` bars ago.
    """

    def __init__(self, period: int = 12):
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest ROC value."""
        if len(closes) < self.period + 1:
            return float('nan')
        if closes[-self.period - 1] == 0:
            return 0.0
        return ((closes[-1] - closes[-self.period - 1]) / closes[-self.period - 1]) * 100.0


class Momentum:
    """Momentum indicator.

    Simple difference between current price and price `period` bars ago.
    """

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest Momentum value."""
        if len(closes) < self.period + 1:
            return float('nan')
        return closes[-1] - closes[-self.period - 1]


class TRIX:
    """TRIX indicator (triple-smoothed EMA rate of change)."""

    def __init__(self, period: int = 15):
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest TRIX value."""
        e1 = EMA(self.period).compute(closes)
        valid1 = ~np.isnan(e1)
        if valid1.sum() < self.period:
            return float('nan')
        e2 = EMA(self.period).compute(e1[valid1])
        valid2 = ~np.isnan(e2)
        if valid2.sum() < self.period:
            return float('nan')
        e3 = EMA(self.period).compute(e2[valid2])
        if len(e3) < 2 or np.isnan(e3[-1]) or np.isnan(e3[-2]) or e3[-2] == 0:
            return float('nan')
        return ((e3[-1] - e3[-2]) / e3[-2]) * 10000.0


class UltimateOscillator:
    """Ultimate Oscillator.

    Combines short, medium, and long-term timeframes into one oscillator.
    """

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        self.p1, self.p2, self.p3 = period1, period2, period3

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest Ultimate Oscillator value."""
        n = len(closes)
        if n < self.p3 + 1:
            return float('nan')
        bp = closes - np.minimum(lows, np.roll(closes, 1))
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
        bp[0] = 0.0
        tr[0] = 0.0
        bp_p1 = np.sum(bp[-self.p1:])
        bp_p2 = np.sum(bp[-self.p2:])
        bp_p3 = np.sum(bp[-self.p3:])
        tr_p1 = np.sum(tr[-self.p1:])
        tr_p2 = np.sum(tr[-self.p2:])
        tr_p3 = np.sum(tr[-self.p3:])
        if tr_p1 == 0 or tr_p2 == 0 or tr_p3 == 0:
            return 50.0
        avg1 = bp_p1 / tr_p1
        avg2 = bp_p2 / tr_p2
        avg3 = bp_p3 / tr_p3
        return ((4 * avg1 + 2 * avg2 + avg3) / 7) * 100.0


class MFI:
    """Money Flow Index.

    Volume-weighted RSI. >80 overbought, <20 oversold.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Compute the latest MFI value."""
        if len(closes) < self.period + 1:
            return float('nan')
        tp = (highs + lows + closes) / 3.0
        mf = tp * volumes
        pos_mf = np.where(tp > np.roll(tp, 1), mf, 0.0)
        neg_mf = np.where(tp < np.roll(tp, 1), mf, 0.0)
        pos_sum = np.sum(pos_mf[-self.period:])
        neg_sum = np.sum(neg_mf[-self.period:])
        if neg_sum == 0:
            return 100.0
        mfr = pos_sum / neg_sum
        return 100.0 - (100.0 / (1.0 + mfr))


class ADX:
    """Average Directional Index.

    Measures trend strength regardless of direction.
    <20: no trend, 20-25: weak trend, 25-50: strong trend, >50: very strong.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest ADX value."""
        if len(closes) < self.period * 2:
            return float('nan')
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        if len(tr) < self.period:
            return float('nan')
        atr = np.zeros(len(tr))
        plus_di = np.zeros(len(tr))
        minus_di = np.zeros(len(tr))
        atr[self.period - 1] = np.sum(tr[:self.period])
        plus_di[self.period - 1] = np.sum(plus_dm[:self.period])
        minus_di[self.period - 1] = np.sum(minus_dm[:self.period])
        for i in range(self.period, len(tr)):
            atr[i] = atr[i - 1] - (atr[i - 1] / self.period) + tr[i]
            plus_di[i] = plus_di[i - 1] - (plus_di[i - 1] / self.period) + plus_dm[i]
            minus_di[i] = minus_di[i - 1] - (minus_di[i - 1] / self.period) + minus_dm[i]
        plus_di_pct = np.where(atr > 0, (plus_di / atr) * 100, 0.0)
        minus_di_pct = np.where(atr > 0, (minus_di / atr) * 100, 0.0)
        di_sum = plus_di_pct + minus_di_pct
        dx = np.where(di_sum > 0, np.abs(plus_di_pct - minus_di_pct) / di_sum * 100, 0.0)
        adx = np.zeros_like(dx)
        adx[self.period * 2 - 2] = np.mean(dx[self.period - 1:self.period * 2 - 1])
        for i in range(self.period * 2 - 1, len(dx)):
            adx[i] = (adx[i - 1] * (self.period - 1) + dx[i]) / self.period
        return adx[-1] if len(adx) > 0 else float('nan')


class Aroon:
    """Aroon Up/Down indicator."""

    def __init__(self, period: int = 25):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray) -> dict:
        """Compute Aroon Up/Down values."""
        if len(highs) < self.period + 1:
            return {"up": float('nan'), "down": float('nan'), "oscillator": float('nan')}
        recent_highs = highs[-self.period - 1:]
        recent_lows = lows[-self.period - 1:]
        days_since_high = self.period - np.argmax(recent_highs)
        days_since_low = self.period - np.argmin(recent_lows)
        aroon_up = ((self.period - days_since_high) / self.period) * 100.0
        aroon_down = ((self.period - days_since_low) / self.period) * 100.0
        oscillator = aroon_up - aroon_down
        return {"up": aroon_up, "down": aroon_down, "oscillator": oscillator}


class ChandeMomentumOscillator:
    """Chande Momentum Oscillator.

    Similar to RSI but uses sum of up and down moves in denominator.
    Range: -100 to +100.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest CMO value."""
        if len(closes) < self.period + 1:
            return float('nan')
        deltas = np.diff(closes[-self.period - 1:])
        sum_up = np.sum(np.where(deltas > 0, deltas, 0.0))
        sum_down = np.sum(np.where(deltas < 0, -deltas, 0.0))
        if sum_up + sum_down == 0:
            return 0.0
        return ((sum_up - sum_down) / (sum_up + sum_down)) * 100.0


class EhlersFisherTransform:
    """Ehlers Fisher Transform.

    Converts prices into a Gaussian normal distribution,
    making turning points easier to identify.
    """

    def __init__(self, period: int = 10):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest Fisher Transform value."""
        if len(closes) < self.period:
            return float('nan')
        hl2 = (highs + lows) / 2.0
        recent = hl2[-self.period:]
        max_h = np.max(recent)
        min_l = np.min(recent)
        if max_h == min_l:
            return 0.0
        value = 2.0 * ((closes[-1] - min_l) / (max_h - min_l) - 0.5)
        value = max(-0.999, min(0.999, value))
        fisher = 0.5 * np.log((1 + value) / (1 - value))
        return float(fisher)


# ============================================================================
# Volatility Indicators
# ============================================================================

class ATR:
    """Average True Range.

    Measures market volatility by decomposing the entire range of a price.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest ATR value."""
        if len(closes) < self.period + 1:
            return float('nan')
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        if len(tr) < self.period:
            return float('nan')
        return float(np.mean(tr[-self.period:]))

    def compute_series(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Compute ATR series using Wilder's smoothing."""
        result = np.full_like(closes, np.nan, dtype=float)
        if len(closes) < self.period + 1:
            return result
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        result[self.period] = np.mean(tr[:self.period])
        for i in range(self.period + 1, len(closes)):
            idx = i - 1
            result[i] = (result[i - 1] * (self.period - 1) + tr[idx]) / self.period
        return result


class BollingerBands:
    """Bollinger Bands.

    Volatility bands placed above and below a moving average.
    Band width expands with increasing volatility, contracts with decreasing.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def compute(self, closes: np.ndarray) -> Optional[dict]:
        """Compute Bollinger Bands values."""
        if len(closes) < self.period:
            return None
        sma = np.mean(closes[-self.period:])
        std = np.std(closes[-self.period:], ddof=0)
        upper = sma + self.num_std * std
        lower = sma - self.num_std * std
        return {
            "upper": upper,
            "middle": sma,
            "lower": lower,
            "bandwidth": (2 * self.num_std * std / sma) if sma > 0 else 0.0,
            "percent_b": ((closes[-1] - lower) / (2 * self.num_std * std)) if std > 0 else 0.5,
        }


class KeltnerChannels:
    """Keltner Channels.

    Volatility-based envelopes set above and below an EMA
    using ATR as the distance measure.
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[dict]:
        """Compute Keltner Channels values."""
        if len(closes) < max(self.ema_period, self.atr_period) + 1:
            return None
        mid = EMA(self.ema_period).compute(closes)[-1]
        atr_val = ATR(self.atr_period).compute(highs, lows, closes)
        if np.isnan(mid) or np.isnan(atr_val):
            return None
        return {
            "upper": mid + self.multiplier * atr_val,
            "middle": mid,
            "lower": mid - self.multiplier * atr_val,
        }


class DonchianChannels:
    """Donchian Channels.

    Highest high and lowest low over a lookback period.
    Classic breakout system indicator.
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray) -> Optional[dict]:
        """Compute Donchian Channels values."""
        if len(highs) < self.period:
            return None
        upper = np.max(highs[-self.period:])
        lower = np.min(lows[-self.period:])
        return {"upper": upper, "middle": (upper + lower) / 2, "lower": lower}


class StandardDeviation:
    """Rolling Standard Deviation."""

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, data: np.ndarray) -> float:
        """Compute the latest rolling standard deviation."""
        if len(data) < self.period:
            return float('nan')
        return float(np.std(data[-self.period:], ddof=0))


class HistoricalVolatility:
    """Annualized Historical Volatility."""

    def __init__(self, period: int = 20, trading_days: int = 365):
        self.period = period
        self.trading_days = trading_days

    def compute(self, closes: np.ndarray) -> float:
        """Compute the latest historical volatility."""
        if len(closes) < self.period + 1:
            return float('nan')
        returns = np.diff(np.log(closes[-self.period - 1:]))
        return float(np.std(returns, ddof=1) * np.sqrt(self.trading_days) * 100)


class ParkinsonVolatility:
    """Parkinson volatility estimator.

    Uses high-low range for more efficient volatility estimation.
    """

    def __init__(self, period: int = 20, trading_days: int = 365):
        self.period = period
        self.trading_days = trading_days

    def compute(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Compute Parkinson volatility."""
        if len(highs) < self.period:
            return float('nan')
        hl_ratio = np.log(highs[-self.period:] / lows[-self.period:])
        valid = np.isfinite(hl_ratio)
        if valid.sum() < self.period // 2:
            return float('nan')
        variance = np.sum(hl_ratio[valid] ** 2) / (4 * valid.sum() * np.log(2))
        return float(np.sqrt(variance) * np.sqrt(self.trading_days) * 100)


class GarmanKlassVolatility:
    """Garman-Klass volatility estimator.

    Uses OHLC data for even more efficient volatility estimation.
    """

    def __init__(self, period: int = 20, trading_days: int = 365):
        self.period = period
        self.trading_days = trading_days

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, opens: np.ndarray) -> float:
        """Compute Garman-Klass volatility."""
        if len(highs) < self.period:
            return float('nan')
        hl = np.log(highs[-self.period:] / lows[-self.period:]) ** 2
        co = np.log(closes[-self.period:] / opens[-self.period:]) ** 2
        valid = np.isfinite(hl) & np.isfinite(co)
        if valid.sum() < self.period // 2:
            return float('nan')
        variance = np.sum(0.5 * hl[valid] - (2 * np.log(2) - 1) * co[valid]) / valid.sum()
        return float(np.sqrt(max(variance, 0)) * np.sqrt(self.trading_days) * 100)


class ChaikinVolatility:
    """Chaikin Volatility indicator.

    Measures the rate of change of the high-low spread.
    Increasing values indicate increasing volatility.
    """

    def __init__(self, period: int = 10, roc_period: int = 10):
        self.period = period
        self.roc_period = roc_period

    def compute(self, highs: np.ndarray, lows: np.ndarray) -> float:
        """Compute the latest Chaikin Volatility value."""
        if len(highs) < self.period + self.roc_period:
            return float('nan')
        hl_spread = highs[-self.period - self.roc_period:] - lows[-self.period - self.roc_period:]
        ema_spread = EMA(self.period).compute(hl_spread)
        valid = ~np.isnan(ema_spread)
        if valid.sum() < self.roc_period + 1:
            return float('nan')
        valid_vals = ema_spread[valid]
        if valid_vals[-self.roc_period] == 0:
            return 0.0
        return ((valid_vals[-1] - valid_vals[-self.roc_period]) / valid_vals[-self.roc_period]) * 100.0


class TrueRange:
    """True Range for each bar."""

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Compute True Range series."""
        if len(closes) < 2:
            return np.full_like(closes, np.nan, dtype=float)
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        return np.concatenate([[np.nan], tr])


class ATRP:
    """ATR Percentage - ATR as a percentage of closing price."""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute the latest ATR percentage."""
        atr_val = ATR(self.period).compute(highs, lows, closes)
        if np.isnan(atr_val) or closes[-1] == 0:
            return float('nan')
        return (atr_val / closes[-1]) * 100.0


# ============================================================================
# Volume Indicators
# ============================================================================

class OBVIndicator:
    """On-Balance Volume.

    Measures buying and selling pressure as a cumulative indicator.
    """

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute OBV series."""
        if len(closes) < 2:
            return np.zeros_like(closes, dtype=float)
        obv = np.zeros(len(closes), dtype=float)
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        return obv


class CMFIndicator:
    """Chaikin Money Flow.

    Measures accumulation/distribution over a period.
    Range: -1 to +1. >0 buying pressure, <0 selling pressure.
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Compute the latest CMF value."""
        if len(closes) < self.period:
            return float('nan')
        hl_diff = highs - lows
        mfv = np.where(hl_diff > 0, ((closes - lows) - (highs - closes)) / hl_diff * volumes, 0.0)
        mf_sum = np.sum(mfv[-self.period:])
        vol_sum = np.sum(volumes[-self.period:])
        if vol_sum == 0:
            return 0.0
        return mf_sum / vol_sum


class ADLine:
    """Accumulation/Distribution Line.

    Cumulative measure of money flow volume.
    """

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute A/D Line series."""
        if len(closes) < 2:
            return np.zeros_like(closes, dtype=float)
        hl_diff = highs - lows
        mfm = np.where(hl_diff > 0, ((closes - lows) - (highs - closes)) / hl_diff, 0.0)
        mfv = mfm * volumes
        return np.cumsum(mfv)


class VolumeProfile:
    """Volume Profile - volume distribution across price levels.

    Divides the price range into bins and sums volume per bin.
    """

    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                volumes: np.ndarray) -> Dict:
        """Compute Volume Profile.

        Returns:
            Dict with 'prices', 'volumes', 'poc' (point of control),
            'value_area_high', 'value_area_low'.
        """
        if len(closes) < self.num_bins:
            return {"prices": np.array([]), "volumes": np.array([]),
                    "poc": float('nan'), "value_area_high": float('nan'),
                    "value_area_low": float('nan')}
        price_min = np.min(lows)
        price_max = np.max(highs)
        if price_max == price_min:
            return {"prices": np.array([price_min]), "volumes": np.array([np.sum(volumes)]),
                    "poc": price_min, "value_area_high": price_min, "value_area_low": price_min}
        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_volumes = np.zeros(self.num_bins)
        bin_prices = (bin_edges[:-1] + bin_edges[1:]) / 2
        tp = (highs + lows + closes) / 3.0
        for i in range(len(tp)):
            idx = min(int((tp[i] - price_min) / (price_max - price_min) * self.num_bins), self.num_bins - 1)
            idx = max(0, idx)
            bin_volumes[idx] += volumes[i]
        poc_idx = np.argmax(bin_volumes)
        poc_price = bin_prices[poc_idx]
        total_vol = np.sum(bin_volumes)
        va_target = total_vol * 0.70
        va_vol = bin_volumes[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        while va_vol < va_target and (va_low_idx > 0 or va_high_idx < self.num_bins - 1):
            add_low = bin_volumes[va_low_idx - 1] if va_low_idx > 0 else 0
            add_high = bin_volumes[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else 0
            if add_low >= add_high and va_low_idx > 0:
                va_low_idx -= 1
                va_vol += bin_volumes[va_low_idx]
            elif va_high_idx < self.num_bins - 1:
                va_high_idx += 1
                va_vol += bin_volumes[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                va_vol += bin_volumes[va_low_idx]
            else:
                break
        return {
            "prices": bin_prices,
            "volumes": bin_volumes,
            "poc": poc_price,
            "value_area_high": bin_edges[va_high_idx + 1],
            "value_area_low": bin_edges[va_low_idx],
        }


class ForceIndex:
    """Force Index.

    Combines price and volume to measure buying/selling pressure.
    FI = (Close - PrevClose) * Volume
    """

    def __init__(self, period: int = 13):
        self.period = period

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Compute the latest Force Index value (smoothed with EMA)."""
        if len(closes) < 2:
            return float('nan')
        fi_raw = np.zeros(len(closes), dtype=float)
        fi_raw[0] = 0.0
        for i in range(1, len(closes)):
            fi_raw[i] = (closes[i] - closes[i - 1]) * volumes[i]
        if len(fi_raw) < self.period:
            return float('nan')
        ema_fi = EMA(self.period).compute(fi_raw)
        return ema_fi[-1] if not np.isnan(ema_fi[-1]) else float('nan')


class EaseOfMovement:
    """Ease of Movement indicator.

    Relates price movement to volume, indicating how easily price moves.
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> float:
        """Compute the latest EMV value (smoothed)."""
        if len(highs) < self.period + 1:
            return float('nan')
        dm = ((highs + lows) / 2 - np.roll((highs + lows) / 2, 1))[1:]
        br = (volumes[1:] / 1e6) / (highs[1:] - lows[1:] + 1e-10)
        emv_raw = dm / (br + 1e-10)
        if len(emv_raw) < self.period:
            return float('nan')
        return float(np.mean(emv_raw[-self.period:]))


class VolumeOscillator:
    """Volume Oscillator.

    Difference between fast and slow moving averages of volume.
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute(self, volumes: np.ndarray) -> float:
        """Compute the latest Volume Oscillator value."""
        if len(volumes) < self.slow_period:
            return float('nan')
        fast_ma = np.mean(volumes[-self.fast_period:])
        slow_ma = np.mean(volumes[-self.slow_period:])
        if slow_ma == 0:
            return 0.0
        return ((fast_ma - slow_ma) / slow_ma) * 100.0


class NVI:
    """Negative Volume Index.

    Tracks price changes on days when volume decreases from the previous day.
    Based on the theory that smart money trades on quiet days.
    """

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute NVI series."""
        if len(closes) < 2:
            return np.full_like(closes, 1000.0, dtype=float)
        nvi = np.full(len(closes), 1000.0, dtype=float)
        for i in range(1, len(closes)):
            if volumes[i] < volumes[i - 1]:
                pct_change = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0
                nvi[i] = nvi[i - 1] * (1 + pct_change)
            else:
                nvi[i] = nvi[i - 1]
        return nvi


class PVI:
    """Positive Volume Index.

    Tracks price changes on days when volume increases from the previous day.
    """

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute PVI series."""
        if len(closes) < 2:
            return np.full_like(closes, 1000.0, dtype=float)
        pvi = np.full(len(closes), 1000.0, dtype=float)
        for i in range(1, len(closes)):
            if volumes[i] > volumes[i - 1]:
                pct_change = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0
                pvi[i] = pvi[i - 1] * (1 + pct_change)
            else:
                pvi[i] = pvi[i - 1]
        return pvi


class IchimokuCloud:
    """Ichimoku Cloud indicator.

    Multi-component trend-following system:
    - Tenkan-sen (conversion line)
    - Kijun-sen (base line)
    - Senkou Span A (leading span A)
    - Senkou Span B (leading span B)
    - Chikou Span (lagging span)
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[dict]:
        """Compute Ichimoku Cloud values."""
        if len(closes) < max(self.tenkan, self.kijun, self.senkou_b):
            return None
        tenkan_val = (np.max(highs[-self.tenkan:]) + np.min(lows[-self.tenkan:])) / 2
        kijun_val = (np.max(highs[-self.kijun:]) + np.min(lows[-self.kijun:])) / 2
        senkou_a = (tenkan_val + kijun_val) / 2
        senkou_b_val = (np.max(highs[-self.senkou_b:]) + np.min(lows[-self.senkou_b:])) / 2
        chikou = closes[-1]
        return {
            "tenkan": tenkan_val,
            "kijun": kijun_val,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b_val,
            "chikou": chikou,
        }


# ============================================================================
# Advanced Indicators
# ============================================================================

class TTMSqueeze:
    """TTM Squeeze indicator.

    Detects when Bollinger Bands are inside Keltner Channels (squeeze),
    and provides momentum histogram when squeeze fires.
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_atr_period: int = 10, kc_mult: float = 1.5,
                 momentum_period: int = 20):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_period = kc_atr_period
        self.kc_mult = kc_mult
        self.momentum_period = momentum_period

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[dict]:
        """Compute TTM Squeeze indicator.

        Returns:
            Dict with 'squeeze_active' (bool), 'momentum' (float),
            'bb_upper', 'bb_lower', 'kc_upper', 'kc_lower'.
        """
        min_len = max(self.bb_period, self.kc_period, self.kc_atr_period) + 1
        if len(closes) < min_len:
            return None
        sma = np.mean(closes[-self.bb_period:])
        std = np.std(closes[-self.bb_period:], ddof=0)
        bb_upper = sma + self.bb_std * std
        bb_lower = sma - self.bb_std * std
        mid = EMA(self.kc_period).compute(closes)[-1]
        atr_val = ATR(self.kc_atr_period).compute(highs, lows, closes)
        if np.isnan(mid) or np.isnan(atr_val):
            return None
        kc_upper = mid + self.kc_mult * atr_val
        kc_lower = mid - self.kc_mult * atr_val
        squeeze_active = (bb_lower >= kc_lower) and (bb_upper <= kc_upper)
        momentum = closes[-1] - np.mean(closes[-self.momentum_period:]) if len(closes) >= self.momentum_period else 0.0
        return {
            "squeeze_active": squeeze_active,
            "momentum": momentum,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
        }


class VWAPIndicator:
    """Volume Weighted Average Price with cumulative calculation.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    Typically reset at market open each day.
    Uses (high + low + close) / 3 as typical price.
    """

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                volumes: np.ndarray) -> float:
        """Compute the latest VWAP value.

        Args:
            highs: High prices array.
            lows: Low prices array.
            closes: Close prices array.
            volumes: Volume array.

        Returns:
            The VWAP value, or NaN if insufficient data.
        """
        if len(closes) < 1 or len(volumes) < 1:
            return float('nan')
        tp = (highs + lows + closes) / 3.0
        cum_tp_vol = np.cumsum(tp * volumes)
        cum_vol = np.cumsum(volumes)
        valid = cum_vol > 0
        result = np.full_like(closes, np.nan, dtype=float)
        result[valid] = cum_tp_vol[valid] / cum_vol[valid]
        return result[-1] if len(result) > 0 else float('nan')

    def compute_series(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                       volumes: np.ndarray) -> np.ndarray:
        """Compute VWAP series."""
        if len(closes) < 1 or len(volumes) < 1:
            return np.full_like(closes, np.nan, dtype=float)
        tp = (highs + lows + closes) / 3.0
        cum_tp_vol = np.cumsum(tp * volumes)
        cum_vol = np.cumsum(volumes)
        result = np.full_like(closes, np.nan, dtype=float)
        valid = cum_vol > 0
        result[valid] = cum_tp_vol[valid] / cum_vol[valid]
        return result


class VWAPBands:
    """VWAP with Standard Deviation Bands.

    Computes VWAP along with upper and lower bands at
    configurable standard deviation multiples.
    """

    def __init__(self, num_std: float = 2.0):
        self.num_std = num_std

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                volumes: np.ndarray) -> Optional[dict]:
        """Compute VWAP with standard deviation bands.

        Returns:
            Dict with 'vwap', 'upper', 'lower', 'std_dev'.
        """
        if len(closes) < 2:
            return None
        tp = (highs + lows + closes) / 3.0
        cum_tp_vol = np.cumsum(tp * volumes)
        cum_vol = np.cumsum(volumes)
        if cum_vol[-1] == 0:
            return None
        vwap = cum_tp_vol[-1] / cum_vol[-1]
        cum_var = np.cumsum(volumes * (tp - vwap) ** 2)
        variance = cum_var[-1] / cum_vol[-1]
        std_dev = np.sqrt(max(variance, 0.0))
        return {
            "vwap": vwap,
            "upper": vwap + self.num_std * std_dev,
            "lower": vwap - self.num_std * std_dev,
            "std_dev": std_dev,
        }


class EhlersDominantCycle:
    """Ehlers Dominant Cycle Period Detection.

    Uses autocorrelation to detect the dominant cycle period
    in the price series. Useful for adapting indicator periods
    dynamically.
    """

    def __init__(self, min_period: int = 8, max_period: int = 48, avg_length: int = 3):
        self.min_period = min_period
        self.max_period = max_period
        self.avg_length = avg_length

    def compute(self, closes: np.ndarray) -> float:
        """Compute the dominant cycle period.

        Args:
            closes: Close prices array.

        Returns:
            Detected dominant cycle period, or NaN.
        """
        if len(closes) < self.max_period * 2:
            return float('nan')
        prices = closes[-self.max_period * 2:]
        n = len(prices)
        best_period = self.min_period
        best_corr = -2.0
        for period in range(self.min_period, self.max_period + 1):
            if n < period * 2:
                continue
            x = prices[:n - period]
            y = prices[period:]
            min_len = min(len(x), len(y))
            if min_len < self.avg_length + 1:
                continue
            x_seg = x[-min_len:]
            y_seg = y[-min_len:]
            std_x = np.std(x_seg)
            std_y = np.std(y_seg)
            if std_x == 0 or std_y == 0:
                continue
            corr = np.corrcoef(x_seg, y_seg)[0, 1]
            if np.isnan(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best_period = period
        return float(best_period)


class FractalDimension:
    """Fractal Dimension of a price series.

    Uses the box-counting method to estimate the fractal dimension.
    FD ≈ 1.0 for a straight line (trending), FD ≈ 1.5 for random walk,
    FD ≈ 2.0 for filling space (very choppy).
    """

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, closes: np.ndarray) -> float:
        """Compute the fractal dimension of the price series.

        Args:
            closes: Close prices array.

        Returns:
            Fractal dimension value (1.0 to 2.0), or NaN.
        """
        if len(closes) < self.period + 1:
            return float('nan')
        data = closes[-self.period:]
        n = len(data)
        path_length = np.sum(np.abs(np.diff(data)))
        max_val = np.max(data)
        min_val = np.min(data)
        price_range = max_val - min_val
        if price_range == 0 or path_length == 0:
            return 1.5
        fd = 1.0 + np.log(path_length / price_range) / np.log(n)
        return max(1.0, min(2.0, fd))


class PivotPoints:
    """Pivot Point calculations with multiple methods.

    Supports Traditional, Fibonacci, Camarilla, and Woodie methods.
    Pivot points are used to identify potential support and resistance levels.
    """

    def __init__(self, method: str = "traditional"):
        """Initialize PivotPoints.

        Args:
            method: Calculation method - 'traditional', 'fibonacci', 'camarilla', or 'woodie'.
        """
        valid_methods = ("traditional", "fibonacci", "camarilla", "woodie")
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        self.method = method

    def compute(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Compute pivot points and support/resistance levels.

        Args:
            high: Previous period high.
            low: Previous period low.
            close: Previous period close.

        Returns:
            Dict with pivot, s1-s4, r1-r4 levels.
        """
        if self.method == "traditional":
            pivot = (high + low + close) / 3.0
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            s4 = s3 - (high - low)
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            r4 = r3 + (high - low)
        elif self.method == "fibonacci":
            pivot = (high + low + close) / 3.0
            diff = high - low
            s1 = pivot - 0.382 * diff
            s2 = pivot - 0.618 * diff
            s3 = pivot - diff
            s4 = pivot - 1.382 * diff
            r1 = pivot + 0.382 * diff
            r2 = pivot + 0.618 * diff
            r3 = pivot + diff
            r4 = pivot + 1.382 * diff
        elif self.method == "camarilla":
            pivot = (high + low + close) / 3.0
            diff = high - low
            s1 = close - 1.1 / 12.0 * diff
            s2 = close - 1.1 / 6.0 * diff
            s3 = close - 1.1 / 4.0 * diff
            s4 = close - 1.1 / 2.0 * diff
            r1 = close + 1.1 / 12.0 * diff
            r2 = close + 1.1 / 6.0 * diff
            r3 = close + 1.1 / 4.0 * diff
            r4 = close + 1.1 / 2.0 * diff
        elif self.method == "woodie":
            pivot = (high + low + 2 * close) / 4.0
            diff = high - low
            s1 = 2 * pivot - high
            s2 = pivot - diff
            s3 = s1 - diff
            s4 = s2 - diff
            r1 = 2 * pivot - low
            r2 = pivot + diff
            r3 = r1 + diff
            r4 = r2 + diff
        else:
            pivot = (high + low + close) / 3.0
            s1 = s2 = s3 = s4 = r1 = r2 = r3 = r4 = pivot
        return {
            "pivot": pivot, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
        }


class FibonacciRetracement:
    """Auto Fibonacci Retracement and Extension levels.

    Detects the most recent significant swing high/low and computes
    Fibonacci retracement and extension levels automatically.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Optional[dict]:
        """Compute Fibonacci retracement and extension levels.

        Args:
            highs: High prices.
            lows: Low prices.
            closes: Close prices.

        Returns:
            Dict with 'direction', 'retracement' and 'extension' levels,
            or None if insufficient data.
        """
        if len(closes) < 10:
            return None
        lookback = min(self.lookback, len(closes))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        swing_high_idx = np.argmax(recent_highs)
        swing_low_idx = np.argmin(recent_lows)
        swing_high = recent_highs[swing_high_idx]
        swing_low = recent_lows[swing_low_idx]
        if swing_high_idx > swing_low_idx:
            direction = "up"
            high_price = swing_high
            low_price = recent_lows[swing_low_idx]
        else:
            direction = "down"
            high_price = swing_high
            low_price = swing_low
        diff = high_price - low_price
        if diff == 0:
            return None
        retracement = {
            "0.0": high_price,
            "0.236": high_price - 0.236 * diff,
            "0.382": high_price - 0.382 * diff,
            "0.5": high_price - 0.5 * diff,
            "0.618": high_price - 0.618 * diff,
            "0.786": high_price - 0.786 * diff,
            "1.0": low_price,
        }
        extension = {
            "1.272": low_price - 0.272 * diff if direction == "up" else high_price + 0.272 * diff,
            "1.618": low_price - 0.618 * diff if direction == "up" else high_price + 0.618 * diff,
            "2.0": low_price - 1.0 * diff if direction == "up" else high_price + 1.0 * diff,
            "2.618": low_price - 1.618 * diff if direction == "up" else high_price + 1.618 * diff,
        }
        return {
            "direction": direction,
            "swing_high": high_price,
            "swing_low": low_price,
            "retracement": retracement,
            "extension": extension,
        }


class SupportResistance:
    """Dynamic Support/Resistance Level Detection.

    Uses fractal pivots to identify significant support and resistance levels.
    A fractal high requires a bar with higher highs on both sides.
    A fractal low requires a bar with lower lows on both sides.
    """

    def __init__(self, window: int = 5, min_touches: int = 2, tolerance_pct: float = 0.5):
        """Initialize SupportResistance.

        Args:
            window: Number of bars on each side for fractal detection.
            min_touches: Minimum number of touches to confirm a level.
            tolerance_pct: Percentage tolerance for grouping nearby levels.
        """
        self.window = window
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct

    def compute(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, List[float]]:
        """Compute support and resistance levels.

        Args:
            highs: High prices.
            lows: Low prices.
            closes: Close prices.

        Returns:
            Dict with 'support' and 'resistance' lists of price levels.
        """
        if len(closes) < self.window * 2 + 1:
            return {"support": [], "resistance": []}
        fractal_highs: List[float] = []
        fractal_lows: List[float] = []
        for i in range(self.window, len(highs) - self.window):
            is_high = True
            is_low = True
            for j in range(1, self.window + 1):
                if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                    is_high = False
                if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                    is_low = False
            if is_high:
                fractal_highs.append(highs[i])
            if is_low:
                fractal_lows.append(lows[i])
        resistance = self._cluster_levels(fractal_highs, closes[-1])
        support = self._cluster_levels(fractal_lows, closes[-1])
        resistance.sort()
        support.sort(reverse=True)
        return {"support": support, "resistance": resistance}

    def _cluster_levels(self, levels: List[float], current_price: float) -> List[float]:
        """Cluster nearby price levels and filter by minimum touches."""
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: List[List[float]] = []
        current_cluster = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            tolerance = level * self.tolerance_pct / 100.0
            if abs(level - np.mean(current_cluster)) <= max(tolerance, current_price * self.tolerance_pct / 100.0):
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
        result = []
        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                result.append(float(np.mean(cluster)))
        return result


# ============================================================================
# Candlestick Pattern Recognition (35+ patterns)
# ============================================================================

class CandlestickPatterns:
    """Comprehensive candlestick pattern recognition.

    Detects 35+ single, double, and triple candlestick patterns.
    All methods return a dict with pattern name -> bool mapping.
    """

    def __init__(self, body_ratio: float = 0.1, doji_ratio: float = 0.05,
                 shadow_ratio: float = 0.1):
        """Initialize CandlestickPatterns.

        Args:
            body_ratio: Minimum body/range ratio for significant candles.
            doji_ratio: Maximum body/range ratio for doji candles.
            shadow_ratio: Minimum shadow/range ratio for significance.
        """
        self.body_ratio = body_ratio
        self.doji_ratio = doji_ratio
        self.shadow_ratio = shadow_ratio

    def _body(self, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute candle body sizes."""
        return np.abs(c - o)

    def _range(self, h: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Compute candle ranges."""
        return h - l

    def _upper_shadow(self, h: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute upper shadows."""
        return h - np.maximum(o, c)

    def _lower_shadow(self, l: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute lower shadows."""
        return np.minimum(o, c) - l

    def _is_bullish(self, o: float, c: float) -> bool:
        """Check if candle is bullish (close > open)."""
        return c > o

    def _is_bearish(self, o: float, c: float) -> bool:
        """Check if candle is bearish (close < open)."""
        return c < o

    def detect_all(self, opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray) -> Dict[str, bool]:
        """Detect all candlestick patterns at the current bar.

        Args:
            opens: Open prices array (at least 5 bars recommended).
            highs: High prices array.
            lows: Low prices array.
            closes: Close prices array.

        Returns:
            Dict mapping pattern name -> detected (bool).
        """
        if len(closes) < 5:
            return {}
        o, h, l, c = opens, highs, lows, closes
        results: Dict[str, bool] = {}
        # Doji variants
        results["doji"] = self._detect_doji(o, h, l, c)
        results["dragonfly_doji"] = self._detect_dragonfly_doji(o, h, l, c)
        results["gravestone_doji"] = self._detect_gravestone_doji(o, h, l, c)
        results["long_legged_doji"] = self._detect_long_legged_doji(o, h, l, c)
        # Single candle
        results["hammer"] = self._detect_hammer(o, h, l, c)
        results["inverted_hammer"] = self._detect_inverted_hammer(o, h, l, c)
        results["hanging_man"] = self._detect_hanging_man(o, h, l, c)
        results["shooting_star"] = self._detect_shooting_star(o, h, l, c)
        results["bullish_marubozu"] = self._detect_bullish_marubozu(o, h, l, c)
        results["bearish_marubozu"] = self._detect_bearish_marubozu(o, h, l, c)
        results["spinning_top"] = self._detect_spinning_top(o, h, l, c)
        # Double candle
        results["bullish_engulfing"] = self._detect_bullish_engulfing(o, h, l, c)
        results["bearish_engulfing"] = self._detect_bearish_engulfing(o, h, l, c)
        results["tweezer_top"] = self._detect_tweezer_top(o, h, l, c)
        results["tweezer_bottom"] = self._detect_tweezer_bottom(o, h, l, c)
        results["piercing_line"] = self._detect_piercing_line(o, h, l, c)
        results["dark_cloud_cover"] = self._detect_dark_cloud_cover(o, h, l, c)
        results["bullish_harami"] = self._detect_bullish_harami(o, h, l, c)
        results["bearish_harami"] = self._detect_bearish_harami(o, h, l, c)
        # Triple candle
        results["morning_star"] = self._detect_morning_star(o, h, l, c)
        results["evening_star"] = self._detect_evening_star(o, h, l, c)
        results["three_white_soldiers"] = self._detect_three_white_soldiers(o, h, l, c)
        results["three_black_crows"] = self._detect_three_black_crows(o, h, l, c)
        results["three_inside_up"] = self._detect_three_inside_up(o, h, l, c)
        results["three_inside_down"] = self._detect_three_inside_down(o, h, l, c)
        results["three_outside_up"] = self._detect_three_outside_up(o, h, l, c)
        results["three_outside_down"] = self._detect_three_outside_down(o, h, l, c)
        results["bullish_abandoned_baby"] = self._detect_bullish_abandoned_baby(o, h, l, c)
        results["bearish_abandoned_baby"] = self._detect_bearish_abandoned_baby(o, h, l, c)
        # Complex patterns
        results["rising_three_methods"] = self._detect_rising_three_methods(o, h, l, c)
        results["falling_three_methods"] = self._detect_falling_three_methods(o, h, l, c)
        results["bullish_belt_hold"] = self._detect_bullish_belt_hold(o, h, l, c)
        results["bearish_belt_hold"] = self._detect_bearish_belt_hold(o, h, l, c)
        results["bullish_counterattack"] = self._detect_bullish_counterattack(o, h, l, c)
        results["bearish_counterattack"] = self._detect_bearish_counterattack(o, h, l, c)
        results["bullish_breakaway"] = self._detect_bullish_breakaway(o, h, l, c)
        results["bearish_breakaway"] = self._detect_bearish_breakaway(o, h, l, c)
        return results

    # --- Doji Variants ---

    def _detect_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Standard doji: body is very small relative to range."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        return rng > 0 and body / rng <= self.doji_ratio

    def _detect_dragonfly_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Dragonfly doji: open=close=high, long lower shadow."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and us / rng <= self.doji_ratio and ls / rng >= 0.6

    def _detect_gravestone_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Gravestone doji: open=close=low, long upper shadow."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and ls / rng <= self.doji_ratio and us / rng >= 0.6

    def _detect_long_legged_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Long-legged doji: small body, long shadows on both sides."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and us / rng >= 0.3 and ls / rng >= 0.3

    # --- Single Candle Patterns ---

    def _detect_hammer(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Hammer: small body at top, long lower shadow (bullish reversal in downtrend)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and ls / rng >= 0.6 and us / rng <= 0.1 and self._is_bullish(o[-1], c[-1])

    def _detect_inverted_hammer(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Inverted hammer: small body at bottom, long upper shadow (bullish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and us / rng >= 0.6 and ls / rng <= 0.1

    def _detect_hanging_man(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Hanging man: same shape as hammer but in uptrend (bearish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and ls / rng >= 0.6 and us / rng <= 0.1 and self._is_bearish(o[-1], c[-1])

    def _detect_shooting_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Shooting star: small body at bottom, long upper shadow (bearish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and us / rng >= 0.6 and ls / rng <= 0.1 and self._is_bearish(o[-1], c[-1])

    def _detect_bullish_marubozu(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish marubozu: no shadows, close > open."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return self._is_bullish(o[-1], c[-1]) and us / rng <= 0.05 and ls / rng <= 0.05

    def _detect_bearish_marubozu(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish marubozu: no shadows, close < open."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return self._is_bearish(o[-1], c[-1]) and us / rng <= 0.05 and ls / rng <= 0.05

    def _detect_spinning_top(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Spinning top: small body, roughly equal shadows."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.25 and us / rng >= 0.25 and ls / rng >= 0.25

    # --- Double Candle Patterns ---

    def _detect_bullish_engulfing(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish engulfing: current bullish candle engulfs previous bearish."""
        if len(c) < 2:
            return False
        prev_bearish = self._is_bearish(o[-2], c[-2])
        curr_bullish = self._is_bullish(o[-1], c[-1])
        return prev_bearish and curr_bullish and c[-1] >= o[-2] and o[-1] <= c[-2]

    def _detect_bearish_engulfing(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish engulfing: current bearish candle engulfs previous bullish."""
        if len(c) < 2:
            return False
        prev_bullish = self._is_bullish(o[-2], c[-2])
        curr_bearish = self._is_bearish(o[-1], c[-1])
        return prev_bullish and curr_bearish and o[-1] >= c[-2] and c[-1] <= o[-2]

    def _detect_tweezer_top(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Tweezer top: two candles with same high, bearish reversal."""
        if len(c) < 2:
            return False
        same_high = abs(h[-1] - h[-2]) / max(h[-1], h[-2], 1e-10) < 0.001
        return same_high and self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])

    def _detect_tweezer_bottom(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Tweezer bottom: two candles with same low, bullish reversal."""
        if len(c) < 2:
            return False
        same_low = abs(l[-1] - l[-2]) / max(l[-1], l[-2], 1e-10) < 0.001
        return same_low and self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])

    def _detect_piercing_line(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Piercing line: bullish reversal with close above midpoint of prev bearish."""
        if len(c) < 2:
            return False
        prev_mid = (o[-2] + c[-2]) / 2.0
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and o[-1] < l[-2] and c[-1] > prev_mid and c[-1] < o[-2])

    def _detect_dark_cloud_cover(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Dark cloud cover: bearish reversal with close below midpoint of prev bullish."""
        if len(c) < 2:
            return False
        prev_mid = (o[-2] + c[-2]) / 2.0
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and o[-1] > h[-2] and c[-1] < prev_mid and c[-1] > o[-2])

    def _detect_bullish_harami(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish harami: small bullish candle inside prev large bearish."""
        if len(c) < 2:
            return False
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and o[-1] > c[-2] and c[-1] < o[-2] and o[-1] < o[-2] and c[-1] > c[-2])

    def _detect_bearish_harami(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish harami: small bearish candle inside prev large bullish."""
        if len(c) < 2:
            return False
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and c[-1] > o[-2] and o[-1] < c[-2] and c[-1] < c[-2] and o[-1] > o[-2])

    # --- Triple Candle Patterns ---

    def _detect_morning_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Morning star: bearish + small body + bullish reversal."""
        if len(c) < 3:
            return False
        b1_bearish = self._is_bearish(o[-3], c[-3])
        b2_body = self._body(o, c)[-2]
        b2_rng = self._range(h, l)[-2]
        b2_small = b2_rng > 0 and b2_body / b2_rng <= 0.33 if b2_rng > 0 else False
        b3_bullish = self._is_bullish(o[-1], c[-1])
        return b1_bearish and b2_small and b3_bullish and c[-1] > (o[-3] + c[-3]) / 2.0

    def _detect_evening_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Evening star: bullish + small body + bearish reversal."""
        if len(c) < 3:
            return False
        b1_bullish = self._is_bullish(o[-3], c[-3])
        b2_body = self._body(o, c)[-2]
        b2_rng = self._range(h, l)[-2]
        b2_small = b2_rng > 0 and b2_body / b2_rng <= 0.33 if b2_rng > 0 else False
        b3_bearish = self._is_bearish(o[-1], c[-1])
        return b1_bullish and b2_small and b3_bearish and c[-1] < (o[-3] + c[-3]) / 2.0

    def _detect_three_white_soldiers(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three white soldiers: three consecutive bullish candles, each opening within prev body."""
        if len(c) < 3:
            return False
        return (self._is_bullish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and c[-1] > c[-2] > c[-3] and o[-2] > o[-3] and o[-2] < c[-3]
                and o[-1] > o[-2] and o[-1] < c[-2])

    def _detect_three_black_crows(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three black crows: three consecutive bearish candles, each opening within prev body."""
        if len(c) < 3:
            return False
        return (self._is_bearish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and c[-1] < c[-2] < c[-3] and o[-2] < o[-3] and o[-2] > c[-3]
                and o[-1] < o[-2] and o[-1] > c[-2])

    def _detect_three_inside_up(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three inside up: bullish harami followed by bullish close."""
        if len(c) < 3:
            return False
        harami = (self._is_bearish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2])
                  and o[-2] > c[-3] and c[-2] < o[-3])
        return harami and self._is_bullish(o[-1], c[-1]) and c[-1] > c[-2]

    def _detect_three_inside_down(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three inside down: bearish harami followed by bearish close."""
        if len(c) < 3:
            return False
        harami = (self._is_bullish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2])
                  and c[-2] > o[-3] and o[-2] < c[-3])
        return harami and self._is_bearish(o[-1], c[-1]) and c[-1] < c[-2]

    def _detect_three_outside_up(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three outside up: bullish engulfing followed by higher close."""
        if len(c) < 3:
            return False
        engulfing = (self._is_bearish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2])
                     and c[-2] >= o[-3] and o[-2] <= c[-3])
        return engulfing and self._is_bullish(o[-1], c[-1]) and c[-1] > c[-2]

    def _detect_three_outside_down(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three outside down: bearish engulfing followed by lower close."""
        if len(c) < 3:
            return False
        engulfing = (self._is_bullish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2])
                     and o[-2] >= c[-3] and c[-2] <= o[-3])
        return engulfing and self._is_bearish(o[-1], c[-1]) and c[-1] < c[-2]

    def _detect_bullish_abandoned_baby(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish abandoned baby: bearish + doji gap + bullish."""
        if len(c) < 3:
            return False
        b1_bearish = self._is_bearish(o[-3], c[-3])
        b2_doji = self._range(h, l)[-2] > 0 and self._body(o, c)[-2] / self._range(h, l)[-2] <= self.doji_ratio
        b3_bullish = self._is_bullish(o[-1], c[-1])
        gap_down = l[-2] > h[-3] if h[-3] > 0 else False
        gap_up = l[-1] > h[-2] if h[-2] > 0 else False
        return b1_bearish and b2_doji and b3_bullish and gap_down and gap_up

    def _detect_bearish_abandoned_baby(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish abandoned baby: bullish + doji gap + bearish."""
        if len(c) < 3:
            return False
        b1_bullish = self._is_bullish(o[-3], c[-3])
        b2_doji = self._range(h, l)[-2] > 0 and self._body(o, c)[-2] / self._range(h, l)[-2] <= self.doji_ratio
        b3_bearish = self._is_bearish(o[-1], c[-1])
        gap_up = h[-2] < l[-3] if l[-3] > 0 else False
        gap_down = h[-1] < l[-2] if l[-2] > 0 else False
        return b1_bullish and b2_doji and b3_bearish and gap_up and gap_down

    # --- Complex Patterns ---

    def _detect_rising_three_methods(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Rising three methods: large bullish + 3 small bearish + large bullish."""
        if len(c) < 5:
            return False
        b1_bullish = self._is_bullish(o[-5], c[-5]) and self._body(o, c)[-5] / self._range(h, l)[-5] > 0.6 if self._range(h, l)[-5] > 0 else False
        small_bears = all(self._is_bearish(o[-4 + i], c[-4 + i]) for i in range(3))
        within_range = all(l[-4 + i] > l[-5] and h[-4 + i] < h[-5] for i in range(3))
        b5_bullish = self._is_bullish(o[-1], c[-1]) and c[-1] > c[-5]
        return b1_bullish and small_bears and within_range and b5_bullish

    def _detect_falling_three_methods(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Falling three methods: large bearish + 3 small bullish + large bearish."""
        if len(c) < 5:
            return False
        b1_bearish = self._is_bearish(o[-5], c[-5]) and self._body(o, c)[-5] / self._range(h, l)[-5] > 0.6 if self._range(h, l)[-5] > 0 else False
        small_bulls = all(self._is_bullish(o[-4 + i], c[-4 + i]) for i in range(3))
        within_range = all(l[-4 + i] > l[-5] and h[-4 + i] < h[-5] for i in range(3))
        b5_bearish = self._is_bearish(o[-1], c[-1]) and c[-1] < c[-5]
        return b1_bearish and small_bulls and within_range and b5_bearish

    def _detect_bullish_belt_hold(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish belt hold: opens at low, closes near high in downtrend."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        return self._is_bullish(o[-1], c[-1]) and abs(o[-1] - l[-1]) / rng < 0.01 and self._body(o, c)[-1] / rng > 0.5

    def _detect_bearish_belt_hold(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish belt hold: opens at high, closes near low in uptrend."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        return self._is_bearish(o[-1], c[-1]) and abs(o[-1] - h[-1]) / rng < 0.01 and self._body(o, c)[-1] / rng > 0.5

    def _detect_bullish_counterattack(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish counterattack: bearish candle followed by bullish closing at same level."""
        if len(c) < 2:
            return False
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and abs(c[-1] - c[-2]) / max(abs(c[-1]), abs(c[-2]), 1e-10) < 0.01)

    def _detect_bearish_counterattack(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish counterattack: bullish candle followed by bearish closing at same level."""
        if len(c) < 2:
            return False
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and abs(c[-1] - c[-2]) / max(abs(c[-1]), abs(c[-2]), 1e-10) < 0.01)

    def _detect_bullish_breakaway(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish breakaway: 5-candle bullish reversal pattern."""
        if len(c) < 5:
            return False
        long_bear = self._is_bearish(o[-5], c[-5]) and self._body(o, c)[-5] / max(self._range(h, l)[-5], 1e-10) > 0.5
        continues_down = c[-4] < c[-5] and c[-3] < c[-4]
        reversal = self._is_bullish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
        close_above = c[-1] > (o[-5] + c[-5]) / 2.0
        return long_bear and continues_down and reversal and close_above

    def _detect_bearish_breakaway(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish breakaway: 5-candle bearish reversal pattern."""
        if len(c) < 5:
            return False
        long_bull = self._is_bullish(o[-5], c[-5]) and self._body(o, c)[-5] / max(self._range(h, l)[-5], 1e-10) > 0.5
        continues_up = c[-4] > c[-5] and c[-3] > c[-4]
        reversal = self._is_bearish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
        close_below = c[-1] < (o[-5] + c[-5]) / 2.0
        return long_bull and continues_up and reversal and close_below


# ============================================================================
# Statistical Utility Functions
# ============================================================================

def compute_hurst_exponent(data: np.ndarray, max_lag: int = 50) -> float:
    """Compute the Hurst exponent using R/S analysis.

    The Hurst exponent (H) characterizes the long-range correlation:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk
    - H > 0.5: Trending (persistent)

    Args:
        data: Price or time series data.
        max_lag: Maximum lag for R/S computation.

    Returns:
        Hurst exponent value, or NaN if insufficient data.
    """
    if len(data) < 100:
        return float('nan')
    returns = np.diff(np.log(data[~np.isnan(data)]))
    if len(returns) < max_lag:
        return float('nan')
    lags = range(10, min(max_lag, len(returns) // 2))
    if len(lags) == 0:
        return float('nan')
    rs_values = []
    for lag in lags:
        segments = len(returns) // lag
        if segments < 1:
            continue
        rs_seg = []
        for i in range(segments):
            seg = returns[i * lag:(i + 1) * lag]
            if len(seg) < 2:
                continue
            mean_seg = np.mean(seg)
            cum_dev = np.cumsum(seg - mean_seg)
            r = np.max(cum_dev) - np.min(cum_dev)
            s = np.std(seg, ddof=1)
            if s > 0:
                rs_seg.append(r / s)
        if rs_seg:
            rs_values.append((np.log(lag), np.log(np.mean(rs_seg))))
    if len(rs_values) < 3:
        return float('nan')
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    if len(x) < 2:
        return float('nan')
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def compute_zscore(data: np.ndarray) -> float:
    """Compute the z-score of the latest value relative to the series.

    Args:
        data: Numeric array.

    Returns:
        Z-score of the last element, or NaN if insufficient data.
    """
    if len(data) < 2:
        return float('nan')
    valid = data[~np.isnan(data)]
    if len(valid) < 2:
        return float('nan')
    mean = np.mean(valid)
    std = np.std(valid, ddof=1)
    if std == 0:
        return 0.0
    return float((valid[-1] - mean) / std)


def detect_bullish_divergence(prices: np.ndarray, indicator: np.ndarray,
                               lookback: int = 50) -> bool:
    """Detect bullish divergence (price lower low, indicator higher low).

    Args:
        prices: Price series.
        indicator: Indicator series (same length as prices).
        lookback: Number of bars to look back.

    Returns:
        True if bullish divergence detected.
    """
    if len(prices) < lookback or len(indicator) < lookback:
        return False
    p = prices[-lookback:]
    i = indicator[-lookback:]
    valid = np.isfinite(p) & np.isfinite(i)
    if valid.sum() < 10:
        return False
    p = p[valid]
    i = i[valid]
    mid = len(p) // 2
    p_first_low = np.min(p[:mid])
    p_second_low = np.min(p[mid:])
    if p_second_low >= p_first_low:
        return False
    p_first_idx = np.argmin(p[:mid])
    p_second_idx = mid + np.argmin(p[mid:])
    return i[p_second_idx] > i[p_first_idx]


def detect_bearish_divergence(prices: np.ndarray, indicator: np.ndarray,
                               lookback: int = 50) -> bool:
    """Detect bearish divergence (price higher high, indicator lower high).

    Args:
        prices: Price series.
        indicator: Indicator series (same length as prices).
        lookback: Number of bars to look back.

    Returns:
        True if bearish divergence detected.
    """
    if len(prices) < lookback or len(indicator) < lookback:
        return False
    p = prices[-lookback:]
    i = indicator[-lookback:]
    valid = np.isfinite(p) & np.isfinite(i)
    if valid.sum() < 10:
        return False
    p = p[valid]
    i = i[valid]
    mid = len(p) // 2
    p_first_high = np.max(p[:mid])
    p_second_high = np.max(p[mid:])
    if p_second_high <= p_first_high:
        return False
    p_first_idx = np.argmax(p[:mid])
    p_second_idx = mid + np.argmax(p[mid:])
    return i[p_second_idx] < i[p_first_idx]
