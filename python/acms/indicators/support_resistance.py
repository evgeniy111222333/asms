"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict


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

__all__ = ['PivotPoints', 'FibonacciRetracement', 'SupportResistance']
