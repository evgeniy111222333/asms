"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict


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

__all__ = ['compute_hurst_exponent', 'compute_zscore', 'detect_bullish_divergence', 'detect_bearish_divergence']
