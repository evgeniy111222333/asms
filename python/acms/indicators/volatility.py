"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict

from acms.indicators.moving_averages import EMA


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


__all__ = ['ATR', 'BollingerBands', 'KeltnerChannels', 'DonchianChannels', 'StandardDeviation', 'HistoricalVolatility', 'ParkinsonVolatility', 'GarmanKlassVolatility', 'ChaikinVolatility', 'TrueRange', 'ATRP']
