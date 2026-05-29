"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict

from acms.indicators.moving_averages import SMA, EMA


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

__all__ = ['OBVIndicator', 'CMFIndicator', 'ADLine', 'VolumeProfile', 'ForceIndex', 'EaseOfMovement', 'VolumeOscillator', 'NVI', 'PVI', 'IchimokuCloud']
