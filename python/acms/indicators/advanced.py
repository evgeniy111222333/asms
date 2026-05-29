"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict

from acms.indicators.moving_averages import EMA
from acms.indicators.volatility import ATR, BollingerBands, KeltnerChannels


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


__all__ = ['TTMSqueeze', 'VWAPIndicator', 'VWAPBands', 'EhlersDominantCycle', 'FractalDimension', 'Supertrend', 'MovingAverageRibbon']
