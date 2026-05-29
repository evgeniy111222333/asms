"""Market regime detection."""

from enum import Enum
from typing import Optional

import numpy as np

from acms.indicators import ADX, compute_hurst_exponent


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    QUIET = "quiet"
    UNKNOWN = "unknown"


class RegimeDetector:
    """Market regime detection.

    Classifies the current market into regimes:
    - TRENDING: Strong directional movement
    - MEAN_REVERTING: Range-bound, oscillating
    - VOLATILE: High volatility, unpredictable
    - QUIET: Low volatility, consolidation
    """

    def __init__(self, lookback: int = 100, adx_trend_threshold: float = 25.0,
                 vol_high_threshold: float = 0.05, vol_low_threshold: float = 0.01):
        self.lookback = lookback
        self.adx_trend_threshold = adx_trend_threshold
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

    def detect(self, closes: np.ndarray, highs: Optional[np.ndarray] = None,
               lows: Optional[np.ndarray] = None) -> MarketRegime:
        """Detect current market regime."""
        if len(closes) < 50:
            return MarketRegime.UNKNOWN
        returns = np.diff(np.log(closes[-min(self.lookback, len(closes)):]))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return MarketRegime.UNKNOWN
        vol = np.std(returns)
        hurst = compute_hurst_exponent(closes[-min(self.lookback, len(closes)):])
        adx = float('nan')
        if highs is not None and lows is not None and len(highs) >= 30:
            adx = ADX(14).compute(highs, lows, closes)
        if vol > self.vol_high_threshold:
            return MarketRegime.VOLATILE
        elif vol < self.vol_low_threshold:
            return MarketRegime.QUIET
        elif not np.isnan(adx) and adx > self.adx_trend_threshold:
            return MarketRegime.TRENDING
        elif not np.isnan(hurst) and hurst < 0.45:
            return MarketRegime.MEAN_REVERTING
        elif not np.isnan(adx) and adx > 20:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.QUIET


__all__ = [
    "MarketRegime",
    "RegimeDetector",
]
