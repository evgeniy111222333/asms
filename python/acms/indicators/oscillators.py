"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict

from acms.indicators.moving_averages import SMA, EMA, VWMA


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
        if valid.sum() > 0:
            signal_full[valid] = signal_line[:valid.sum()]
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


__all__ = ['RSI', 'ConnorsRSI', 'MACD', 'VolumeWeightedMACD', 'StochasticOscillator', 'CCI', 'WilliamsR', 'ROC', 'Momentum', 'TRIX', 'UltimateOscillator', 'MFI', 'ADX', 'Aroon', 'ChandeMomentumOscillator', 'EhlersFisherTransform']
