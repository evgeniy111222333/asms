"""Price-indicator divergence detection."""

from typing import Dict

import numpy as np


class DivergenceDetector:
    """Detects multiple types of price-indicator divergences.

    Supports:
    - RSI divergence (regular and hidden)
    - MACD divergence
    - Volume divergence
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect_rsi_divergence(self, closes: np.ndarray, rsi_series: np.ndarray) -> Dict[str, bool]:
        """Detect RSI divergences (regular and hidden)."""
        result = {
            "bullish_regular": False, "bearish_regular": False,
            "bullish_hidden": False, "bearish_hidden": False,
        }
        if len(closes) < self.lookback or len(rsi_series) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        rsi = rsi_series[-self.lookback:]
        valid = np.isfinite(rsi)
        if valid.sum() < 20:
            return result
        mid = len(prices) // 2
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        p_first_low_idx = np.argmin(prices[:mid])
        p_second_low_idx = mid + np.argmin(prices[mid:])
        if p_second_low < p_first_low:
            rsi_first = rsi[p_first_low_idx]
            rsi_second = rsi[p_second_low_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second):
                if rsi_second > rsi_first:
                    result["bullish_regular"] = True
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        p_first_high_idx = np.argmax(prices[:mid])
        p_second_high_idx = mid + np.argmax(prices[mid:])
        if p_second_high > p_first_high:
            rsi_first = rsi[p_first_high_idx]
            rsi_second = rsi[p_second_high_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second):
                if rsi_second < rsi_first:
                    result["bearish_regular"] = True
        if p_second_low > p_first_low:
            rsi_first = rsi[p_first_low_idx]
            rsi_second = rsi[p_second_low_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second) and rsi_second < rsi_first:
                result["bullish_hidden"] = True
        if p_second_high < p_first_high:
            rsi_first = rsi[p_first_high_idx]
            rsi_second = rsi[p_second_high_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second) and rsi_second > rsi_first:
                result["bearish_hidden"] = True
        return result

    def detect_macd_divergence(self, closes: np.ndarray, macd_histogram: np.ndarray) -> Dict[str, bool]:
        """Detect MACD histogram divergences."""
        result = {"bullish": False, "bearish": False}
        if len(closes) < self.lookback or len(macd_histogram) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        hist = macd_histogram[-self.lookback:]
        valid = np.isfinite(hist)
        if valid.sum() < 20:
            return result
        mid = len(prices) // 2
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        if p_second_low < p_first_low:
            h_first = np.min(hist[:mid][np.isfinite(hist[:mid])]) if np.any(np.isfinite(hist[:mid])) else 0
            h_second = np.min(hist[mid:][np.isfinite(hist[mid:])]) if np.any(np.isfinite(hist[mid:])) else 0
            if h_second > h_first:
                result["bullish"] = True
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        if p_second_high > p_first_high:
            h_first = np.max(hist[:mid][np.isfinite(hist[:mid])]) if np.any(np.isfinite(hist[:mid])) else 0
            h_second = np.max(hist[mid:][np.isfinite(hist[mid:])]) if np.any(np.isfinite(hist[mid:])) else 0
            if h_second < h_first:
                result["bearish"] = True
        return result

    def detect_volume_divergence(self, closes: np.ndarray, volumes: np.ndarray) -> Dict[str, bool]:
        """Detect volume divergences."""
        result = {"bullish": False, "bearish": False}
        if len(closes) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        vols = volumes[-self.lookback:]
        mid = len(prices) // 2
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        v_first = np.mean(vols[:mid])
        v_second = np.mean(vols[mid:])
        if p_second_high > p_first_high and v_second < v_first * 0.8:
            result["bearish"] = True
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        if p_second_low < p_first_low and v_second < v_first * 0.8:
            result["bullish"] = True
        return result


__all__ = [
    "DivergenceDetector",
]
