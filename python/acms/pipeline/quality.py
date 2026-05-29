"""Data quality checking."""

import logging
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Performs data quality checks on market data.

    Checks include:
    - Missing data detection
    - Outlier filtering
    - Gap detection and filling
    - Duplicate detection
    """

    def __init__(self, outlier_std_threshold: float = 5.0,
                 gap_fill_method: str = "ffill"):
        self.outlier_std_threshold = outlier_std_threshold
        self.gap_fill_method = gap_fill_method

    def check_missing(self, data: Dict[str, np.ndarray]) -> Dict:
        """Detect missing values in the dataset.

        Args:
            data: Dict mapping column names to numpy arrays.

        Returns:
            Dict with missing value counts and percentages per column.
        """
        results = {}
        for col, arr in data.items():
            if arr.dtype in (np.float64, np.float32, np.int64, np.int32):
                missing_count = int(np.sum(np.isnan(arr) if arr.dtype.kind == 'f' else arr == 0))
                total = len(arr)
                results[col] = {
                    "missing_count": missing_count,
                    "missing_pct": missing_count / total * 100 if total > 0 else 0.0,
                    "total_count": total,
                }
        return results

    def detect_outliers(self, values: np.ndarray,
                         method: str = "zscore") -> np.ndarray:
        """Detect outliers in a numeric array.

        Args:
            values: Numeric array to check.
            method: Detection method ('zscore' or 'iqr').

        Returns:
            Boolean array where True indicates outlier.
        """
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool)

        valid = values[~np.isnan(values)]
        if len(valid) < 10:
            return np.zeros(len(values), dtype=bool)

        if method == "zscore":
            mean = np.mean(valid)
            std = np.std(valid)
            if std == 0:
                return np.zeros(len(values), dtype=bool)
            z_scores = np.abs((values - mean) / std)
            return z_scores > self.outlier_std_threshold

        elif method == "iqr":
            q1 = np.percentile(valid, 25)
            q3 = np.percentile(valid, 75)
            iqr = q3 - q1
            if iqr == 0:
                return np.zeros(len(values), dtype=bool)
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (values < lower) | (values > upper)

        return np.zeros(len(values), dtype=bool)

    def filter_outliers(self, data: Dict[str, np.ndarray],
                         columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Filter outliers from specified columns, replacing with NaN.

        Args:
            data: Dict mapping column names to arrays.
            columns: Columns to filter. Defaults to OHLC columns.

        Returns:
            Dict with outliers replaced by NaN.
        """
        if columns is None:
            columns = ["open", "high", "low", "close", "volume"]

        result = {}
        for col, arr in data.items():
            if col in columns and len(arr) > 10:
                outlier_mask = self.detect_outliers(arr)
                filtered = arr.copy().astype(float)
                filtered[outlier_mask] = np.nan
                result[col] = filtered
            else:
                result[col] = arr

        return result

    def detect_gaps(self, timestamps: np.ndarray,
                     expected_interval_seconds: int) -> List[Dict]:
        """Detect gaps in timestamp series.

        Args:
            timestamps: Array of timestamps (as datetime or numeric).
            expected_interval_seconds: Expected interval between timestamps.

        Returns:
            List of gap descriptions.
        """
        if len(timestamps) < 2:
            return []

        gaps = []
        for i in range(1, len(timestamps)):
            try:
                t1 = timestamps[i - 1]
                t2 = timestamps[i]
                if hasattr(t1, 'timestamp'):
                    diff = (t2 - t1).total_seconds()
                elif isinstance(t1, (int, float)):
                    diff = t2 - t1
                else:
                    continue

                if diff > expected_interval_seconds * 2:
                    gaps.append({
                        "start": str(t1),
                        "end": str(t2),
                        "gap_seconds": diff,
                        "expected_seconds": expected_interval_seconds,
                        "missing_bars": int(diff / expected_interval_seconds) - 1,
                    })
            except (TypeError, AttributeError):
                continue

        return gaps

    def fill_gaps(self, data: Dict[str, np.ndarray],
                   method: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Fill gaps (NaN values) in the data.

        Args:
            data: Dict mapping column names to arrays.
            method: Fill method ('ffill', 'bfill', 'interpolate', 'zero').

        Returns:
            Dict with gaps filled.
        """
        method = method or self.gap_fill_method
        result = {}

        for col, arr in data.items():
            if not isinstance(arr, np.ndarray):
                result[col] = arr
                continue

            arr = arr.copy().astype(float)
            nan_mask = np.isnan(arr)

            if not np.any(nan_mask):
                result[col] = arr
                continue

            if method == "ffill":
                # Forward fill
                last_valid = None
                for i in range(len(arr)):
                    if not nan_mask[i]:
                        last_valid = arr[i]
                    elif last_valid is not None:
                        arr[i] = last_valid
            elif method == "bfill":
                # Backward fill
                last_valid = None
                for i in range(len(arr) - 1, -1, -1):
                    if not nan_mask[i]:
                        last_valid = arr[i]
                    elif last_valid is not None:
                        arr[i] = last_valid
            elif method == "interpolate":
                # Linear interpolation
                valid_indices = np.where(~nan_mask)[0]
                if len(valid_indices) > 1:
                    arr[nan_mask] = np.interp(
                        np.where(nan_mask)[0],
                        valid_indices,
                        arr[valid_indices],
                    )
            elif method == "zero":
                arr[nan_mask] = 0.0

            result[col] = arr

        return result



__all__ = ["DataQualityChecker"]
