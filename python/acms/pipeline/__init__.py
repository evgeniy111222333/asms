"""Data Pipeline - Fetch, validate, transform, store pipeline.

Implements:
- DataPipeline: fetch → validate → transform → store pipeline
- Polars-based data processing (use polars for DataFrame operations)
- Parquet read/write for historical data storage
- Data download from exchanges (historical klines, trades)
- Data quality checks: missing data detection, outlier filtering, gap filling
- Resampling: convert between timeframes
- Data windowing: sliding window, expanding window for feature computation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Data pipeline configuration."""
    data_dir: str = "/data/acms"
    parquet_dir: str = "/data/acms/parquet"
    default_exchange: str = "binance"
    download_batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    quality_check_enabled: bool = True
    outlier_std_threshold: float = 5.0
    gap_fill_method: str = "ffill"


# ============================================================================
# Data Quality Checks
# ============================================================================

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


# ============================================================================
# Data Resampling
# ============================================================================

class DataResampler:
    """Resample market data between timeframes.

    Converts OHLCV data from one timeframe to another,
    properly aggregating OHLCV fields.
    """

    @staticmethod
    def resample_candles(candles: List[Dict], source_tf: str,
                          target_tf: str) -> List[Dict]:
        """Resample candle data to a different timeframe.

        Args:
            candles: List of candle dicts with OHLCV fields.
            source_tf: Source timeframe (e.g., '1m', '5m').
            target_tf: Target timeframe (e.g., '5m', '1h').

        Returns:
            List of resampled candle dicts.
        """
        if not candles:
            return []

        # Parse timeframe to minutes
        source_minutes = DataResampler._parse_timeframe(source_tf)
        target_minutes = DataResampler._parse_timeframe(target_tf)

        if source_minutes <= 0 or target_minutes <= 0:
            logger.warning("Invalid timeframe: %s -> %s", source_tf, target_tf)
            return candles

        if target_minutes <= source_minutes:
            logger.warning("Target timeframe must be larger than source")
            return candles

        ratio = target_minutes // source_minutes
        if ratio < 2:
            return candles

        resampled = []
        for i in range(0, len(candles), ratio):
            batch = candles[i:i + ratio]
            if not batch:
                break

            resampled.append({
                "open_time": batch[0].get("open_time"),
                "close_time": batch[-1].get("close_time"),
                "open": batch[0].get("open", 0),
                "high": max(c.get("high", 0) for c in batch),
                "low": min(c.get("low", 0) for c in batch),
                "close": batch[-1].get("close", 0),
                "volume": sum(c.get("volume", 0) for c in batch),
                "quote_volume": sum(c.get("quote_volume", 0) for c in batch),
                "trades": sum(c.get("trades", 0) for c in batch),
            })

        return resampled

    @staticmethod
    def _parse_timeframe(tf: str) -> int:
        """Parse timeframe string to minutes.

        Args:
            tf: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d').

        Returns:
            Number of minutes.
        """
        tf = tf.lower().strip()
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 1440
        elif tf.endswith('w'):
            return int(tf[:-1]) * 10080
        return 0


# ============================================================================
# Data Windowing
# ============================================================================

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


# ============================================================================
# Parquet Storage
# ============================================================================

class ParquetStorage:
    """Parquet file storage for historical market data.

    Provides efficient read/write operations for large
    historical datasets using Parquet format.
    """

    def __init__(self, base_dir: str = "/data/acms/parquet"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, symbol: str, timeframe: str, exchange: str = "") -> Path:
        """Get the parquet file path for a symbol/timeframe/exchange."""
        parts = [exchange, symbol.replace("/", "_"), timeframe]
        return self.base_dir / "/".join(parts) / "data.parquet"

    def write(self, data: Any, symbol: str, timeframe: str,
              exchange: str = "") -> str:
        """Write data to a Parquet file.

        Args:
            data: DataFrame or dict to write.
            symbol: Trading pair symbol.
            timeframe: Data timeframe.
            exchange: Exchange name.

        Returns:
            Path to written file.
        """
        path = self._get_path(symbol, timeframe, exchange)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import polars as pl
            if isinstance(data, dict):
                df = pl.DataFrame(data)
            elif isinstance(data, pl.DataFrame):
                df = data
            else:
                df = pl.DataFrame(data)
            df.write_parquet(str(path))
            return str(path)
        except ImportError:
            # Fallback: save as CSV
            csv_path = path.with_suffix('.csv')
            if isinstance(data, dict):
                import csv
                keys = list(data.keys())
                rows = zip(*[data[k] for k in keys])
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(keys)
                    writer.writerows(rows)
            return str(csv_path)

    def read(self, symbol: str, timeframe: str, exchange: str = "",
             columns: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> Any:
        """Read data from a Parquet file.

        Args:
            symbol: Trading pair symbol.
            timeframe: Data timeframe.
            exchange: Exchange name.
            columns: Optional list of columns to read.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Polars DataFrame or dict.
        """
        path = self._get_path(symbol, timeframe, exchange)

        try:
            import polars as pl
            if path.exists():
                df = pl.read_parquet(str(path), columns=columns)
                if start_date and "open_time" in df.columns:
                    df = df.filter(pl.col("open_time") >= start_date)
                if end_date and "open_time" in df.columns:
                    df = df.filter(pl.col("open_time") <= end_date)
                return df
            else:
                return pl.DataFrame()
        except ImportError:
            # Fallback: read CSV
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                data = {}
                import csv
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for k, v in row.items():
                            if k not in data:
                                data[k] = []
                            data[k].append(v)
                return data
            return {}

    def exists(self, symbol: str, timeframe: str, exchange: str = "") -> bool:
        """Check if data exists for a symbol/timeframe."""
        path = self._get_path(symbol, timeframe, exchange)
        return path.exists() or path.with_suffix('.csv').exists()

    def list_symbols(self, exchange: str = "") -> List[str]:
        """List all symbols with stored data."""
        exchange_dir = self.base_dir / exchange if exchange else self.base_dir
        if not exchange_dir.exists():
            return []
        symbols = []
        for d in exchange_dir.iterdir():
            if d.is_dir():
                symbols.append(d.name.replace("_", "/"))
        return sorted(symbols)


# ============================================================================
# Main Data Pipeline
# ============================================================================

class DataPipeline:
    """Complete data pipeline: fetch → validate → transform → store.

    Orchestrates the full data lifecycle from exchange download
    through quality checks to persistent storage.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.quality_checker = DataQualityChecker(
            outlier_std_threshold=self.config.outlier_std_threshold,
            gap_fill_method=self.config.gap_fill_method,
        )
        self.resampler = DataResampler()
        self.windowing = DataWindowing()
        self.storage = ParquetStorage(base_dir=self.config.parquet_dir)
        self._exchange_adapter = None

    def set_exchange(self, adapter: Any) -> None:
        """Set the exchange adapter for data fetching.

        Args:
            adapter: ExchangeAdapter instance with get_candles method.
        """
        self._exchange_adapter = adapter

    async def download_historical(self, symbol: str, timeframe: str,
                                   start_date: str, end_date: str,
                                   exchange: Optional[str] = None) -> List[Dict]:
        """Download historical candle data from exchange.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            exchange: Exchange name override.

        Returns:
            List of candle dicts.
        """
        if not self._exchange_adapter:
            logger.error("No exchange adapter configured")
            return []

        all_candles = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start

        while current < end:
            try:
                candles = await self._exchange_adapter.get_candles(
                    symbol, timeframe, limit=self.config.download_batch_size,
                )
                for c in candles:
                    all_candles.append({
                        "open_time": c.open_time.isoformat() if hasattr(c, 'open_time') else str(c.open_time),
                        "close_time": c.close_time.isoformat() if hasattr(c, 'close_time') else str(c.close_time),
                        "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                        "quote_volume": c.quote_volume if hasattr(c, 'quote_volume') else 0,
                        "trades": c.trades if hasattr(c, 'trades') else 0,
                    })

                if not candles:
                    break

                # Move forward by the time range of fetched candles
                last_time = candles[-1].close_time if candles else current
                if hasattr(last_time, '__add__'):
                    current = last_time + timedelta(minutes=1)
                else:
                    current = end

                await asyncio.sleep(0.5)  # Rate limit

            except Exception as e:
                logger.error("Download error: %s", e)
                if self.config.max_retries > 0:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                break

        return all_candles

    async def run_pipeline(self, symbol: str, timeframe: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           quality_check: bool = True,
                           store: bool = True) -> Dict:
        """Run the full data pipeline.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start_date: Optional start date for download.
            end_date: Optional end date for download.
            quality_check: Whether to run quality checks.
            store: Whether to store results.

        Returns:
            Dict with pipeline results and quality report.
        """
        result = {"symbol": symbol, "timeframe": timeframe, "status": "pending"}

        # Step 1: Fetch data
        if start_date and end_date:
            candles = await self.download_historical(symbol, timeframe, start_date, end_date)
        elif self._exchange_adapter:
            try:
                raw_candles = await self._exchange_adapter.get_candles(symbol, timeframe)
                candles = [
                    {"open": c.open, "high": c.high, "low": c.low,
                     "close": c.close, "volume": c.volume}
                    for c in raw_candles
                ]
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                return result
        else:
            result["status"] = "error"
            result["error"] = "No data source"
            return result

        result["raw_count"] = len(candles)

        if not candles:
            result["status"] = "no_data"
            return result

        # Step 2: Convert to arrays for quality checking
        data = self._candles_to_arrays(candles)

        # Step 3: Quality checks
        quality_report = {}
        if quality_check and self.config.quality_check_enabled:
            # Check missing data
            quality_report["missing"] = self.quality_checker.check_missing(data)

            # Filter outliers
            data = self.quality_checker.filter_outliers(data)

            # Fill gaps
            data = self.quality_checker.fill_gaps(data)

        result["quality_report"] = quality_report

        # Step 4: Store
        if store:
            try:
                path = self.storage.write(data, symbol, timeframe)
                result["storage_path"] = path
            except Exception as e:
                logger.warning("Storage error: %s", e)

        result["status"] = "success"
        result["final_count"] = len(candles)
        return result

    @staticmethod
    def _candles_to_arrays(candles: List[Dict]) -> Dict[str, np.ndarray]:
        """Convert list of candle dicts to columnar arrays.

        Args:
            candles: List of candle dicts.

        Returns:
            Dict mapping column names to numpy arrays.
        """
        if not candles:
            return {}

        columns = {}
        for key in ["open", "high", "low", "close", "volume"]:
            values = []
            for c in candles:
                val = c.get(key, 0)
                values.append(float(val) if val is not None else np.nan)
            columns[key] = np.array(values, dtype=np.float64)

        return columns
