"""Data windowing operations for feature computation."""

import numpy as np
from typing import List, Dict


class DataWindowing:
    """Sliding and expanding window operations for feature computation.

    Provides efficient window-based computations for generating
    features from time series data.
    """

    @staticmethod
    def sliding_window(data: np.ndarray, window_size: int,
                       step: int = 1) -> np.ndarray:
        """Create sliding windows over a 1D array.

        Args:
            data: Input 1D array.
            window_size: Size of each window.
            step: Step size between windows.

        Returns:
            2D array of shape (n_windows, window_size).
        """
        if len(data) < window_size:
            return np.array([])

        n_windows = (len(data) - window_size) // step + 1
        windows = np.zeros((n_windows, window_size))
        for i in range(n_windows):
            start = i * step
            windows[i] = data[start:start + window_size]
        return windows

    @staticmethod
    def expanding_window(data: np.ndarray, min_periods: int = 1) -> List[np.ndarray]:
        """Create expanding windows over a 1D array.

        Args:
            data: Input 1D array.
            min_periods: Minimum number of periods for first window.

        Returns:
            List of arrays of increasing length.
        """
        windows = []
        for i in range(min_periods, len(data) + 1):
            windows.append(data[:i])
        return windows

    @staticmethod
    def rolling_stats(data: np.ndarray, window: int) -> Dict[str, np.ndarray]:
        """Compute rolling statistics over a 1D array.

        Args:
            data: Input array.
            window: Rolling window size.

        Returns:
            Dict with rolling mean, std, min, max arrays.
        """
        if len(data) < window:
            return {"mean": np.array([]), "std": np.array([]),
                    "min": np.array([]), "max": np.array([])}

        means = np.zeros(len(data) - window + 1)
        stds = np.zeros(len(data) - window + 1)
        mins = np.zeros(len(data) - window + 1)
        maxs = np.zeros(len(data) - window + 1)

        for i in range(len(means)):
            w = data[i:i + window]
            means[i] = np.mean(w)
            stds[i] = np.std(w)
            mins[i] = np.min(w)
            maxs[i] = np.max(w)

        return {"mean": means, "std": stds, "min": mins, "max": maxs}


__all__ = ["DataWindowing"]
